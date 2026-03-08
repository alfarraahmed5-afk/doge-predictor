#!/usr/bin/env python
"""label_regimes.py — Assign regime labels to the full post-2022 DOGEUSDT dataset.

Runs :class:`~src.regimes.classifier.DogeRegimeClassifier` over the aligned
DOGEUSDT + BTCUSDT data and writes labelled rows to:
    - ``data/regimes/regime_labels.parquet``
    - ``storage.upsert_regime_labels()`` (TimescaleDB / SQLite)

Usage::

    # Against live PostgreSQL data:
    python scripts/label_regimes.py

    # Against synthetic in-memory SQLite (CI / no live data):
    python scripts/label_regimes.py --in-memory-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_settings  # noqa: E402
from src.processing.storage import DogeStorage  # noqa: E402
from src.regimes.classifier import REGIME_LABELS, DogeRegimeClassifier  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAINING_START_MS: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC
_HOUR_MS: int = 3_600_000
_OUTPUT_DIR: Path = _REPO_ROOT / "data" / "regimes"

# In-memory test: 5 segments × 400 rows each = 2 000 rows total
_SEG_LEN: int = 400


# ---------------------------------------------------------------------------
# Synthetic data seeder (--in-memory-test)
# ---------------------------------------------------------------------------


def _build_ohlcv(
    closes: np.ndarray,
    open_times: list[int],
    symbol: str,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a close-price array.

    Args:
        closes:     Array of closing prices.
        open_times: List of UTC-ms open timestamps.
        symbol:     Trading pair symbol string.

    Returns:
        OHLCV :class:`pd.DataFrame` with era column set.
    """
    opens = np.empty(len(closes))
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    spread = 0.002
    highs = np.maximum(opens, closes) * (1.0 + spread)
    lows = np.minimum(opens, closes) * (1.0 - spread)
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.ones(len(closes)) * 1_000_000.0,
            "symbol": symbol,
            "era": "training",
            "is_interpolated": False,
            "close_time": [t + _HOUR_MS - 1 for t in open_times],
            "quote_volume": np.ones(len(closes)) * 100_000.0,
            "num_trades": np.ones(len(closes), dtype=int) * 1000,
        }
    )


def _seed_test_data(storage: DogeStorage) -> None:
    """Seed in-memory storage with 2 000 rows covering all five regimes.

    Segment layout (each 400 rows):
        0: TRENDING_BULL     — positive drift, correlated BTC
        1: RANGING_HIGH_VOL  — sawtooth period=168h, correlated BTC
        2: TRENDING_BEAR     — negative drift, correlated BTC
        3: RANGING_LOW_VOL   — OU mean-reverting, correlated BTC
        4: DECOUPLED         — zero-drift DOGE, independent BTC

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.
    """
    rng = np.random.default_rng(42)
    rng_btc_ind = np.random.default_rng(99)  # independent stream for segment 4
    n_total = _SEG_LEN * 5
    base_ms = _TRAINING_START_MS
    times = [base_ms + i * _HOUR_MS for i in range(n_total)]

    # Correlation strategy for segments 0-3:
    #   btc_log_ret = doge_log_ret + tiny_noise
    #   This gives rolling-24h log-return corr ≈ 0.98 >> 0.30 threshold.
    #   DECOUPLED will NOT fire for these segments.
    _CORR_NOISE: float = 0.0005  # tiny additive noise keeps it realistic

    # ----- Segment 0: TRENDING_BULL -----
    doge_lr0 = rng.normal(0.003, 0.005, _SEG_LEN)
    btc_lr0 = doge_lr0 + rng.normal(0, _CORR_NOISE, _SEG_LEN)
    doge0 = 0.10 * np.exp(np.cumsum(doge_lr0))
    btc0 = 30_000.0 * np.exp(np.cumsum(btc_lr0))

    # ----- Segment 1: RANGING_HIGH_VOL -----
    # Sawtooth with period=168h → 7d pct_change ≈ 0; amplitude → BB width > 0.04
    t1 = np.arange(_SEG_LEN)
    saw = 0.025 * (2 * ((t1 / 168.0) % 1.0) - 1.0)
    doge1 = np.clip(0.09 * (1.0 + saw), 1e-8, None)
    doge_lr1 = np.concatenate([[0.0], np.log(doge1[1:] / doge1[:-1])])
    btc_lr1 = doge_lr1 + rng.normal(0, _CORR_NOISE, _SEG_LEN)
    btc1 = np.clip(btc0[-1] * np.exp(np.cumsum(btc_lr1)), 1.0, None)

    # ----- Segment 2: TRENDING_BEAR -----
    doge_lr2 = rng.normal(-0.003, 0.005, _SEG_LEN)
    btc_lr2 = doge_lr2 + rng.normal(0, _CORR_NOISE, _SEG_LEN)
    doge2 = doge1[-1] * np.exp(np.cumsum(doge_lr2))
    btc2 = btc1[-1] * np.exp(np.cumsum(btc_lr2))

    # ----- Segment 3: RANGING_LOW_VOL -----
    mean_p = doge2[-1]
    theta, sig = 0.15, 0.0004
    doge3 = np.empty(_SEG_LEN)
    doge3[0] = mean_p
    for i in range(1, _SEG_LEN):
        doge3[i] = doge3[i - 1] + theta * (mean_p - doge3[i - 1]) + rng.normal(0, sig)
    doge3 = np.clip(doge3, 1e-6, None)
    doge_lr3 = np.concatenate([[0.0], np.log(doge3[1:] / doge3[:-1])])
    btc_lr3 = doge_lr3 + rng.normal(0, _CORR_NOISE * 0.5, _SEG_LEN)  # very tight
    btc3 = np.clip(btc2[-1] * np.exp(np.cumsum(btc_lr3)), 1.0, None)

    # ----- Segment 4: DECOUPLED -----
    # BTC log returns are COMPLETELY INDEPENDENT (separate RNG) → corr ≈ 0 → DECOUPLED
    doge_lr4 = rng.normal(0.0, 0.005, _SEG_LEN)
    btc_lr4 = rng_btc_ind.normal(0.0, 0.010, _SEG_LEN)
    doge4 = doge3[-1] * np.exp(np.cumsum(doge_lr4))
    btc4 = np.clip(btc3[-1] * np.exp(np.cumsum(btc_lr4)), 1.0, None)

    doge_closes = np.concatenate([doge0, doge1, doge2, doge3, doge4])
    btc_closes = np.concatenate([btc0, btc1, btc2, btc3, btc4])
    doge_closes = np.clip(doge_closes, 1e-8, None)
    btc_closes = np.clip(btc_closes, 1.0, None)

    doge_df = _build_ohlcv(doge_closes, times, "DOGEUSDT")
    btc_df = _build_ohlcv(btc_closes, times, "BTCUSDT")

    storage.upsert_ohlcv(doge_df, "DOGEUSDT", "1h")
    storage.upsert_ohlcv(btc_df, "BTCUSDT", "1h")


# ---------------------------------------------------------------------------
# Core labelling logic
# ---------------------------------------------------------------------------


def run_labelling(
    storage: DogeStorage,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Classify regimes for DOGEUSDT 1h training data and persist results.

    Args:
        storage:    :class:`~src.processing.storage.DogeStorage` instance.
        output_dir: Directory for Parquet output.  Defaults to
                    ``data/regimes/``.

    Returns:
        :class:`pd.DataFrame` of regime labels with columns:
        ``open_time``, ``symbol``, ``regime``, ``btc_corr_24h``,
        ``bb_width``, ``atr_norm``.
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    _FAR_FUTURE_MS: int = 9_999_999_999_999
    doge_df = storage.get_ohlcv("DOGEUSDT", "1h", _TRAINING_START_MS, _FAR_FUTURE_MS)
    btc_df = storage.get_ohlcv("BTCUSDT", "1h", _TRAINING_START_MS, _FAR_FUTURE_MS)

    if doge_df.empty:
        raise RuntimeError("No DOGEUSDT data found — run bootstrap first.")

    clf = DogeRegimeClassifier()
    regime_series = clf.classify(
        doge_df, btc_df=btc_df if not btc_df.empty else None
    )

    regime_df = pd.DataFrame(
        {
            "open_time": doge_df["open_time"].values,
            "symbol": "DOGEUSDT",
            "regime": regime_series.values,
        }
    )

    n_rows = storage.upsert_regime_labels(regime_df)

    out_path = output_dir / "regime_labels.parquet"
    regime_df.to_parquet(out_path, index=False)

    dist = clf.get_regime_distribution(regime_series)
    print("Regime distribution:")
    for label in REGIME_LABELS:
        pct = dist.get(label, 0.0) * 100
        print(f"  {label:<20s} {pct:5.1f}%")
    print(f"Total rows labelled: {n_rows}")
    print(f"Parquet written    : {out_path}")

    return regime_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 = success, 1 = failure.
    """
    configure_logging()

    parser = argparse.ArgumentParser(description="Assign regime labels to DOGEUSDT 1h history.")
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help="Use in-memory SQLite with synthetic data (CI mode).",
    )
    args = parser.parse_args(argv)

    try:
        if args.in_memory_test:
            engine = sa.create_engine("sqlite:///:memory:")
            storage = DogeStorage(get_settings(), engine=engine)
            storage.create_tables()
            _seed_test_data(storage)
            print("[IN-MEMORY-TEST] Synthetic data seeded.")
        else:
            storage = DogeStorage(get_settings())

        run_labelling(storage, output_dir=_REPO_ROOT / "data" / "regimes")
        print("\nlabel_regimes.py COMPLETE")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
