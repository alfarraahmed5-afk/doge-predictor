"""Fixture generator for doge_predictor tests.

Generates seven realistic synthetic Parquet files covering all five market
regimes. Run once to populate the fixture directory::

    python tests/fixtures/doge_sample_data/generate_fixtures.py

All fixtures:
    - Have strictly monotonic UTC timestamps starting from 2022-01-01 00:00:00
    - Pass :class:`OHLCVSchema` (or :class:`FundingRateSchema`) validation
    - Have valid OHLC relationships (high >= open/close/low; low <= open/close)
    - Have ``era='training'``

Generated files (saved to the same directory as this script):
    - dogeusdt_1h_trending_bull.parquet  : 500 rows, clear uptrend
    - dogeusdt_1h_trending_bear.parquet  : 500 rows, clear downtrend
    - dogeusdt_1h_ranging.parquet        : 500 rows, tight low-ATR range
    - dogeusdt_1h_decoupled.parquet      : 500 rows, BTC corr < 0.20
    - dogeusdt_1h_mania.parquet          : 200 rows, 10x parabolic move
    - btcusdt_1h_aligned.parquet         : 500 rows, BTC aligned to DOGE times
    - funding_rates_sample.parquet       : 200 rows, 8h-cadence funding rates
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.processing.df_schemas import FundingRateSchema, OHLCVSchema, validate_df  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXTURE_DIR: Path = Path(__file__).parent
_SEED: int = 42
_START_MS: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC
_HOUR_MS: int = 3_600_000
_FUNDING_INTERVAL_MS: int = 28_800_000  # 8 h


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _make_open_times(n: int, start_ms: int = _START_MS, interval_ms: int = _HOUR_MS) -> list[int]:
    """Return a list of *n* strictly monotonic UTC epoch-ms timestamps."""
    return [start_ms + i * interval_ms for i in range(n)]


def _build_ohlcv(
    closes: np.ndarray,
    open_times: list[int],
    symbol: str,
    *,
    rng: np.random.Generator,
    base_volume: float = 5_000_000.0,
) -> pd.DataFrame:
    """Construct an OHLCV DataFrame from a close-price series.

    Open price of candle *i* equals close price of candle *i-1* (first open
    is derived from first close with a tiny random perturbation).  High and
    low are set to satisfy all OHLC invariants with a small random spread.

    Args:
        closes: 1-D array of close prices (must all be > 0).
        open_times: Monotonic UTC epoch-ms timestamps, same length as closes.
        symbol: Ticker symbol string (e.g. ``'DOGEUSDT'``).
        rng: Seeded NumPy random generator for reproducibility.
        base_volume: Baseline volume before random scaling.

    Returns:
        DataFrame with columns: ``open_time``, ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``symbol``, ``era``.
    """
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0] * (1.0 + rng.uniform(-0.001, 0.001))
    opens[1:] = closes[:-1]

    # Spread factor: [0.1%, 0.5%] of the bar body
    spread = rng.uniform(0.001, 0.005, size=n)
    body_lo = np.minimum(opens, closes)
    body_hi = np.maximum(opens, closes)
    highs = body_hi * (1.0 + spread)
    lows = body_lo * (1.0 - spread)

    # Guarantee all OHLC constraints are met
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    volumes = rng.uniform(0.5, 2.0, size=n) * base_volume

    df = pd.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "symbol": symbol,
            "era": "training",
        }
    )
    return df


# ---------------------------------------------------------------------------
# Regime-specific price series generators
# ---------------------------------------------------------------------------


def _trending_bull_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate a clear uptrending close-price series (EMA20 > EMA50 > EMA200).

    Uses a geometric Brownian motion with strong positive drift so that by
    candle 200+ the EMA alignment is unambiguous.

    Args:
        n: Number of candles.
        rng: Seeded generator.

    Returns:
        Array of close prices starting near $0.100.
    """
    drift = 0.0012        # strong positive drift per candle
    sigma = 0.008         # daily-like noise
    log_returns = rng.normal(drift, sigma, size=n)
    prices = 0.10 * np.exp(np.cumsum(log_returns))
    return np.clip(prices, 1e-6, None)


def _trending_bear_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate a clear downtrending close-price series (EMA20 < EMA50 < EMA200).

    Args:
        n: Number of candles.
        rng: Seeded generator.

    Returns:
        Array of close prices starting near $0.200.
    """
    drift = -0.0012       # strong negative drift
    sigma = 0.008
    log_returns = rng.normal(drift, sigma, size=n)
    prices = 0.20 * np.exp(np.cumsum(log_returns))
    return np.clip(prices, 1e-6, None)


def _ranging_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate a tight mean-reverting range (low ATR, low BB width).

    Uses an Ornstein-Uhlenbeck-like process around $0.090 with very low
    volatility so the resulting ATR stays well below 0.3%.

    Args:
        n: Number of candles.
        rng: Seeded generator.

    Returns:
        Array of close prices near $0.090.
    """
    mean_price = 0.090
    theta = 0.15          # mean-reversion speed
    sigma = 0.0004        # very small noise → low ATR
    prices = np.empty(n)
    prices[0] = mean_price
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mean_price - prices[i - 1]) + rng.normal(0, sigma)
    return np.clip(prices, 1e-6, None)


def _decoupled_doge_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate DOGE prices whose log-returns are uncorrelated with BTC.

    Uses pure independent noise (zero drift) — correlation with any other
    series will be near zero by construction.

    Args:
        n: Number of candles.
        rng: Seeded generator.

    Returns:
        Array of independent DOGE close prices near $0.080.
    """
    sigma = 0.010
    # Completely independent noise — zero correlation with BTC by construction
    log_returns = rng.normal(0.0, sigma, size=n)
    prices = 0.080 * np.exp(np.cumsum(log_returns))
    return np.clip(prices, 1e-6, None)


def _btc_aligned_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate BTC close prices with moderate positive drift.

    Args:
        n: Number of candles.
        rng: Seeded generator.

    Returns:
        Array of BTC close prices starting near $42,000.
    """
    drift = 0.0003
    sigma = 0.007
    log_returns = rng.normal(drift, sigma, size=n)
    prices = 42_000.0 * np.exp(np.cumsum(log_returns))
    return np.clip(prices, 1.0, None)


def _mania_prices(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Generate a parabolic 10x mania move.

    Starts near $0.050 and reaches ~$0.500 over *n* candles using
    exponentially increasing drift.

    Args:
        n: Number of candles (200 recommended).
        rng: Seeded generator.

    Returns:
        Array of close prices for a 10x mania episode.
    """
    # Drift accelerates over time to simulate mania
    drifts = np.linspace(0.005, 0.025, n)
    sigma = 0.012
    noise = rng.normal(0, sigma, size=n)
    log_returns = drifts + noise
    prices = 0.050 * np.exp(np.cumsum(log_returns))
    return np.clip(prices, 1e-6, None)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def build_dogeusdt_trending_bull(rng: np.random.Generator) -> pd.DataFrame:
    """Build 500-row trending-bull DOGE fixture."""
    n = 500
    open_times = _make_open_times(n)
    closes = _trending_bull_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "DOGEUSDT", rng=rng)


def build_dogeusdt_trending_bear(rng: np.random.Generator) -> pd.DataFrame:
    """Build 500-row trending-bear DOGE fixture."""
    n = 500
    open_times = _make_open_times(n)
    closes = _trending_bear_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "DOGEUSDT", rng=rng)


def build_dogeusdt_ranging(rng: np.random.Generator) -> pd.DataFrame:
    """Build 500-row tight-range DOGE fixture."""
    n = 500
    open_times = _make_open_times(n)
    closes = _ranging_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "DOGEUSDT", rng=rng, base_volume=2_000_000.0)


def build_dogeusdt_decoupled(rng: np.random.Generator) -> pd.DataFrame:
    """Build 500-row decoupled DOGE fixture (corr with BTC < 0.20)."""
    n = 500
    open_times = _make_open_times(n)
    closes = _decoupled_doge_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "DOGEUSDT", rng=rng)


def build_dogeusdt_mania(rng: np.random.Generator) -> pd.DataFrame:
    """Build 200-row 10x-parabolic mania DOGE fixture."""
    n = 200
    open_times = _make_open_times(n)
    closes = _mania_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "DOGEUSDT", rng=rng, base_volume=50_000_000.0)


def build_btcusdt_aligned(rng: np.random.Generator) -> pd.DataFrame:
    """Build 500-row BTC fixture aligned to the DOGE timestamp grid."""
    n = 500
    open_times = _make_open_times(n)
    closes = _btc_aligned_prices(n, rng=rng)
    return _build_ohlcv(closes, open_times, "BTCUSDT", rng=rng, base_volume=200.0)


def build_funding_rates_sample(rng: np.random.Generator) -> pd.DataFrame:
    """Build 200-row 8h-cadence funding rate fixture.

    Returns:
        DataFrame with columns: ``timestamp_ms``, ``funding_rate``, ``symbol``.
    """
    n = 200
    timestamp_ms = [_START_MS + i * _FUNDING_INTERVAL_MS for i in range(n)]
    # Funding rate ~ N(0.0001, 0.00005), clipped to valid range
    rates = rng.normal(0.0001, 0.00005, size=n)
    rates = np.clip(rates, -0.005, 0.005)
    return pd.DataFrame(
        {
            "timestamp_ms": timestamp_ms,
            "funding_rate": rates,
            "symbol": "DOGEUSDT",
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all fixture files and validate them against their schemas."""
    rng = np.random.default_rng(_SEED)

    fixtures: list[tuple[str, pd.DataFrame, str]] = [
        ("dogeusdt_1h_trending_bull.parquet", build_dogeusdt_trending_bull(rng), "ohlcv"),
        ("dogeusdt_1h_trending_bear.parquet", build_dogeusdt_trending_bear(rng), "ohlcv"),
        ("dogeusdt_1h_ranging.parquet", build_dogeusdt_ranging(rng), "ohlcv"),
        ("dogeusdt_1h_decoupled.parquet", build_dogeusdt_decoupled(rng), "ohlcv"),
        ("dogeusdt_1h_mania.parquet", build_dogeusdt_mania(rng), "ohlcv"),
        ("btcusdt_1h_aligned.parquet", build_btcusdt_aligned(rng), "ohlcv"),
        ("funding_rates_sample.parquet", build_funding_rates_sample(rng), "funding"),
    ]

    for filename, df, schema_type in fixtures:
        if schema_type == "ohlcv":
            validate_df(df, OHLCVSchema)
        else:
            validate_df(df, FundingRateSchema)

        out_path = _FIXTURE_DIR / filename
        df.to_parquet(out_path, index=False)
        print(f"  OK {filename} ({len(df)} rows) -> {out_path}")

    print(f"\nAll {len(fixtures)} fixtures generated and validated.")


if __name__ == "__main__":
    main()
