"""Funding rate features for the DOGE prediction pipeline.

All features are computed at time T using only data available at or before T.

Step 1 — Alignment:
    The 8h funding rate is forward-filled to 1h candles.  Each 8h rate is
    known at the START of its 8h period, so every 1h candle within that period
    can use it without lookahead.  This is NOT lookahead bias.

Step 2 — Features:
    funding_rate          forward-filled 8h rate at each 1h candle
    funding_rate_zscore   (fr_8h - roll90_mean) / roll90_std, ffilled to 1h
                          90-period window on native 8h series = ~30 days
    funding_extreme_long  (funding_rate > 0.001).astype(int)
    funding_extreme_short (funding_rate < -0.0005).astype(int)
    funding_available     1 if funding data exists for this row, else 0
                          (0 for all rows before Oct 2020 — product did not exist)

Pre-Oct-2020 rows:
    All funding features = 0.0, funding_available = 0.
    This covers the pre-Binance-DOGE-perpetual era cleanly.

Lookahead audit (verify after every modification):
    funding_rate       known at 8h period START — strictly past at T         SAFE
    funding_rate_zscore  rolling(90) on past 8h rates, then forward-filled   SAFE
    funding_extreme_long   threshold on forward-filled rate                   SAFE
    funding_extreme_short  threshold on forward-filled rate                   SAFE
    funding_available    derived from whether data exists at T                SAFE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DogeSettings, doge_settings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-10

# Binance DOGEUSDT perpetual futures launched 2020-10-01 00:00 UTC
_FUNDING_LAUNCH_MS: int = 1_601_510_400_000

_DOGE_REQUIRED: tuple[str, ...] = ("open_time",)
_FUNDING_REQUIRED: tuple[str, ...] = ("timestamp_ms", "funding_rate")

# The 5 canonical feature names produced by this module (for downstream validation)
FUNDING_FEATURE_NAMES: tuple[str, ...] = (
    "funding_rate",
    "funding_rate_zscore",
    "funding_extreme_long",
    "funding_extreme_short",
    "funding_available",
)


def compute_funding_features(
    doge_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Forward-fill 8h funding rates to 1h candles and compute derived features.

    Aligns funding_df (8h cadence) to the 1h OHLCV DataFrame by forward-filling
    each 8h rate to the 8 candles it covers.  Rows before the first available
    funding timestamp receive NaN, which is then replaced with 0.0 and flagged
    via ``funding_available = 0``.

    Lookahead bias: **none**.  The 8h funding rate is published at the *start*
    of each 8h period and is therefore known for every 1h candle within that
    period.

    Args:
        doge_df: DOGEUSDT 1h OHLCV DataFrame.  Must contain ``open_time``
            (UTC epoch ms, int), sorted ascending.
        funding_df: DOGEUSDT 8h funding rate DataFrame.  Must contain
            ``timestamp_ms`` (UTC epoch ms, int) and ``funding_rate`` (float).
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If *None*
            the module-level singleton is used.

    Returns:
        New DataFrame (``doge_df.copy()``) with five funding feature columns
        appended: ``funding_rate``, ``funding_rate_zscore``,
        ``funding_extreme_long``, ``funding_extreme_short``,
        ``funding_available``.

    Raises:
        ValueError: If any required column is missing from either input.
    """
    if cfg is None:
        cfg = doge_settings

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    missing_doge = [c for c in _DOGE_REQUIRED if c not in doge_df.columns]
    missing_funding = [c for c in _FUNDING_REQUIRED if c not in funding_df.columns]

    if missing_doge:
        raise ValueError(
            f"compute_funding_features: doge_df missing columns {missing_doge}"
        )
    if missing_funding:
        raise ValueError(
            f"compute_funding_features: funding_df missing columns {missing_funding}"
        )

    out = doge_df.copy()
    doge_times = out["open_time"].to_numpy()

    # ------------------------------------------------------------------
    # Handle empty funding_df
    # ------------------------------------------------------------------
    if len(funding_df) == 0:
        logger.warning(
            "compute_funding_features: funding_df is empty — all funding features = 0"
        )
        out["funding_rate"] = 0.0
        out["funding_rate_zscore"] = 0.0
        out["funding_extreme_long"] = np.int8(0)
        out["funding_extreme_short"] = np.int8(0)
        out["funding_available"] = np.int8(0)
        return out

    # ------------------------------------------------------------------
    # Build the native 8h funding series indexed on timestamp_ms
    # ------------------------------------------------------------------
    # Deduplicate by timestamp (keep first occurrence), then sort ascending.
    # .drop_duplicates() removes duplicate VALUES; here we deduplicate by INDEX
    # (timestamp_ms) using groupby().first() to handle any duplicate rows.
    funding_8h: pd.Series = (
        funding_df.sort_values("timestamp_ms")
        .set_index("timestamp_ms")["funding_rate"]
        .groupby(level=0)
        .first()
        .sort_index()
    )

    # ------------------------------------------------------------------
    # Compute z-score on the NATIVE 8h series (window = 90 x 8h = 30 days)
    #
    # Z-score is computed before forward-filling so the statistical
    # mean and std reflect the 8h-cadence distribution, not repeated 1h
    # values which would collapse the standard deviation toward zero.
    #
    # rolling(90).mean/std at position k uses observations [k-90+1 .. k]
    # — strictly past data.                                             SAFE
    # ------------------------------------------------------------------
    window: int = cfg.indicators.funding_zscore_window
    rolling_mean = funding_8h.rolling(window, min_periods=2).mean()
    rolling_std = funding_8h.rolling(window, min_periods=2).std()
    zscore_8h: pd.Series = (funding_8h - rolling_mean) / (rolling_std + _EPS)

    # ------------------------------------------------------------------
    # Forward-fill 8h → 1h  (strictly causal — no backward-fill)
    #
    # Union the 8h timestamps with all 1h timestamps, reindex the combined
    # series, forward-fill, then extract only the 1h rows.
    #                                                                   SAFE
    # ------------------------------------------------------------------
    all_times = pd.Index(np.union1d(funding_8h.index.to_numpy(), doge_times))

    funding_1h: pd.Series = (
        funding_8h.reindex(all_times).ffill().reindex(doge_times)
    )
    zscore_1h: pd.Series = (
        zscore_8h.reindex(all_times).ffill().reindex(doge_times)
    )

    # ------------------------------------------------------------------
    # funding_available flag
    #
    # A row is "available" if it received a forward-filled value AND
    # its open_time is on or after the funding product launch date.
    # Pre-launch 1h candles that have NaN (no data available) → 0.
    # ------------------------------------------------------------------
    has_data: np.ndarray = ~np.isnan(funding_1h.to_numpy())
    out["funding_available"] = has_data.astype(np.int8)

    n_unavailable = int((~has_data).sum())
    if n_unavailable > 0:
        logger.warning(
            "compute_funding_features: {} 1h candles have no funding data "
            "(pre-launch or gap) — all funding features = 0 for those rows",
            n_unavailable,
        )

    # ------------------------------------------------------------------
    # Fill pre-launch / gap rows with 0.0 (not NaN) per spec
    # ------------------------------------------------------------------
    funding_1h_filled = funding_1h.fillna(0.0)
    zscore_1h_filled = zscore_1h.fillna(0.0)

    out["funding_rate"] = funding_1h_filled.to_numpy()
    out["funding_rate_zscore"] = zscore_1h_filled.to_numpy()

    # ------------------------------------------------------------------
    # Extreme flags — threshold on the aligned (filled) funding rate
    #
    # NaN rows were already filled with 0.0 → flags are 0 for those rows.
    # Thresholds loaded from config; never hardcoded.                   SAFE
    # ------------------------------------------------------------------
    fr = pd.Series(out["funding_rate"].to_numpy(), index=out.index)
    out["funding_extreme_long"] = (
        fr >= cfg.funding_rate_extreme_long
    ).astype(np.int8)
    out["funding_extreme_short"] = (
        fr <= cfg.funding_rate_extreme_short
    ).astype(np.int8)

    n_added = len(out.columns) - len(doge_df.columns)
    logger.debug(
        "compute_funding_features: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
