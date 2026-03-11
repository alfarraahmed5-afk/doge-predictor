"""DOGE-specific mandatory features for the DOGE prediction pipeline.

Implements all 12 mandatory DOGE-specific features defined in CLAUDE.md Section 7.
These features capture DOGE's unique sensitivity to BTC beta, sentiment-driven
volume spikes, and retail round-number psychology.

Feature groups:
    Group 1 — BTC Rolling Correlation (3 features)
        doge_btc_corr_12h, doge_btc_corr_24h, doge_btc_corr_7d
    Group 2 — DOGE/BTC Ratio Momentum (3 features)
        dogebtc_mom_6h, dogebtc_mom_24h, dogebtc_mom_48h
    Group 3 — Volume Spike Detection (3 features)
        volume_ratio, volume_spike_flag, volume_spike_magnitude
    Group 4 — Round Number Psychology (3 features)
        nearest_round_level, distance_to_round_pct, at_round_number_flag

Lookahead audit — every feature at time T uses only data from [0 .. T]:
    doge_btc_corr_Nh    rolling(N).corr on log returns — window [T-N+1..T]    SAFE
    dogebtc_mom_Nh      log(dogebtc[T] / dogebtc[T-N]) via shift(+N)           SAFE
    volume_ratio        volume[T] / rolling(20).mean(volume[T-19..T])           SAFE
    volume_spike_flag   (volume_ratio[T] >= threshold).astype(int)              SAFE
    volume_spike_mag    volume_ratio[T].clip(upper=10) / 10.0                   SAFE
    nearest_round_level argmin(|close[T] - levels|) — single candle only       SAFE
    distance_to_round   (close[T] - nearest[T]) / nearest[T]                   SAFE
    at_round_number     |distance_to_round[T]| < 0.01                          SAFE

CRITICAL RULES (must hold after any modification):
    - BTC correlation is ALWAYS computed on LOG RETURNS, never raw prices.
      Raw price correlation gives spurious ≈ 1.0 for any two trending assets.
    - Volume normalisation uses rolling mean — NEVER raw volume.
    - All shifts are shift(+N); shift(-N) is banned (lookahead bias).
    - Round number levels are loaded from config; never hardcoded.
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

# Exact feature names as specified in CLAUDE.md Section 7
# Maps config window value (hours) → canonical column name
_CORR_WINDOW_NAMES: dict[int, str] = {
    12: "doge_btc_corr_12h",
    24: "doge_btc_corr_24h",
    168: "doge_btc_corr_7d",
}

_MOM_WINDOW_NAMES: dict[int, str] = {
    6: "dogebtc_mom_6h",
    24: "dogebtc_mom_24h",
    48: "dogebtc_mom_48h",
}

_DOGE_REQUIRED: tuple[str, ...] = ("open_time", "close", "volume")
_BTC_REQUIRED: tuple[str, ...] = ("open_time", "close")
_DOGEBTC_REQUIRED: tuple[str, ...] = ("open_time", "close")

# The 12 canonical feature names produced by this module (for downstream validation)
DOGE_FEATURE_NAMES: tuple[str, ...] = (
    "doge_btc_corr_12h",
    "doge_btc_corr_24h",
    "doge_btc_corr_7d",
    "dogebtc_mom_6h",
    "dogebtc_mom_24h",
    "dogebtc_mom_48h",
    "volume_ratio",
    "volume_spike_flag",
    "volume_spike_magnitude",
    "nearest_round_level",
    "distance_to_round_pct",
    "at_round_number_flag",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _log_ret(series: pd.Series) -> pd.Series:
    """Compute 1-period log returns: log(s[T] / s[T-1]).

    Uses shift(+1) — reads T-1, strictly past data.                        SAFE

    Args:
        series: Numeric price series with a consistent integer index.

    Returns:
        Log-return series of the same length; first row is NaN.
    """
    return np.log(series / series.shift(1))


def _align_to_doge(
    source_df: pd.DataFrame,
    column: str,
    doge_times: np.ndarray,
    out_index: pd.Index,
) -> pd.Series:
    """Align a column from *source_df* to the DOGE open_time axis.

    Performs an open_time-keyed lookup: for each DOGE timestamp the
    corresponding value from *source_df* is fetched.  Missing timestamps
    produce NaN (handled gracefully by all downstream rolling operations).

    Args:
        source_df: DataFrame that contains ``open_time`` and *column*.
        column: Name of the column to extract from *source_df*.
        doge_times: Array of DOGE ``open_time`` values (UTC epoch ms).
        out_index: The integer index of the output DataFrame to assign.

    Returns:
        Series with ``out_index`` containing values aligned to DOGE timestamps.
    """
    lookup = source_df.set_index("open_time")[column]
    aligned_values = lookup.reindex(doge_times).to_numpy()
    return pd.Series(aligned_values, index=out_index, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_doge_features(
    doge_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    dogebtc_df: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Compute all 12 mandatory DOGE-specific features and append to *doge_df*.

    All periods are read from *cfg* (config/doge_settings.yaml).  No numeric
    constant is hardcoded in this function.

    Lookahead bias: **none**.  Every column at row index T is computed using
    only rows [0 .. T].  See module-level audit comment for the full proof.

    Args:
        doge_df: DOGEUSDT OHLCV DataFrame sorted ascending by ``open_time``.
            Must contain: ``open_time`` (UTC epoch ms, int), ``close``,
            ``volume``.
        btc_df: BTCUSDT OHLCV DataFrame.  Must contain: ``open_time``,
            ``close``.  Timestamps need not be identical to *doge_df*;
            alignment is performed by ``open_time`` lookup.
        dogebtc_df: DOGEBTC OHLCV DataFrame.  Must contain: ``open_time``,
            ``close``.  Same alignment semantics as *btc_df*.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If *None*
            the module-level singleton is used.

    Returns:
        New DataFrame (``doge_df.copy()``) with all 12 feature columns
        appended.  The first ``max(correlation_windows)`` rows will contain
        NaN for correlation features (warmup period).

    Raises:
        ValueError: If any required column is missing from any input DataFrame.
    """
    if cfg is None:
        cfg = doge_settings

    # Input validation
    missing_doge = [c for c in _DOGE_REQUIRED if c not in doge_df.columns]
    missing_btc = [c for c in _BTC_REQUIRED if c not in btc_df.columns]
    missing_dogebtc = [c for c in _DOGEBTC_REQUIRED if c not in dogebtc_df.columns]

    if missing_doge:
        raise ValueError(f"compute_doge_features: doge_df missing columns {missing_doge}")
    if missing_btc:
        raise ValueError(f"compute_doge_features: btc_df missing columns {missing_btc}")
    if missing_dogebtc:
        raise ValueError(
            f"compute_doge_features: dogebtc_df missing columns {missing_dogebtc}"
        )

    out = doge_df.copy()
    doge_times = out["open_time"].to_numpy()

    # -----------------------------------------------------------------------
    # Align BTC and DOGEBTC closes to DOGE open_time axis
    # -----------------------------------------------------------------------
    btc_close = _align_to_doge(btc_df, "close", doge_times, out.index)
    dogebtc_close = _align_to_doge(dogebtc_df, "close", doge_times, out.index)

    n_btc_missing = btc_close.isna().sum()
    n_dogebtc_missing = dogebtc_close.isna().sum()
    if n_btc_missing > 0:
        logger.warning(
            "compute_doge_features: {} BTC timestamps unmatched → NaN in BTC features",
            n_btc_missing,
        )
    if n_dogebtc_missing > 0:
        logger.warning(
            "compute_doge_features: {} DOGEBTC timestamps unmatched → NaN in DOGEBTC features",
            n_dogebtc_missing,
        )

    # -----------------------------------------------------------------------
    # GROUP 1: BTC Rolling Correlations
    #
    # CRITICAL: correlation is on LOG RETURNS, not raw prices.
    # Raw-price correlation is spuriously ≈ 1.0 for any two trending assets.
    # Log-return correlation measures actual co-movement of price changes.
    #
    # rolling(N).corr(other) at T uses [T-N+1 .. T] — past data only.     SAFE
    # -----------------------------------------------------------------------
    doge_log_ret = _log_ret(out["close"])
    btc_log_ret = _log_ret(btc_close)

    for window in cfg.correlation_windows:
        col_name = _CORR_WINDOW_NAMES.get(window, f"doge_btc_corr_{window}h")
        out[col_name] = doge_log_ret.rolling(window).corr(btc_log_ret)

    # -----------------------------------------------------------------------
    # GROUP 2: DOGE/BTC Ratio Momentum
    #
    # Uses the dedicated DOGEBTC pair (DOGE denominated in BTC directly).
    # Do NOT approximate as doge_close / btc_close — they are on different
    # precision scales and the spot DOGEBTC pair captures the actual market.
    #
    # log(dogebtc[T] / dogebtc[T-N]) = log_dogebtc[T] - log_dogebtc[T-N]
    # shift(+N) pulls dogebtc_close[T-N] — strictly past data.            SAFE
    # -----------------------------------------------------------------------
    log_dogebtc = np.log(dogebtc_close.clip(lower=_EPS))

    for window in cfg.dogebtc_momentum_windows:
        col_name = _MOM_WINDOW_NAMES.get(window, f"dogebtc_mom_{window}h")
        out[col_name] = log_dogebtc - log_dogebtc.shift(window)

    # -----------------------------------------------------------------------
    # GROUP 3: Volume Spike Detection
    #
    # CRITICAL: volume is normalised by its 20-period rolling mean.
    # Raw volume is non-stationary (absolute levels change over time).
    # Normalised volume_ratio is approximately stationary around 1.0.
    #
    # min_periods=volume_rolling_window → first N-1 rows are NaN (correct).
    # volume_spike_threshold and volume_rolling_window from config.         SAFE
    # -----------------------------------------------------------------------
    vol = out["volume"]
    vol_window = cfg.volume_rolling_window
    vol_ma = vol.rolling(vol_window, min_periods=vol_window).mean()
    volume_ratio = vol / (vol_ma + _EPS)

    out["volume_ratio"] = volume_ratio
    out["volume_spike_flag"] = (volume_ratio >= cfg.volume_spike_threshold).astype(
        np.int8
    )
    # Magnitude: ratio clipped at 10 then scaled to [0, 1]
    out["volume_spike_magnitude"] = volume_ratio.clip(upper=10.0) / 10.0

    # -----------------------------------------------------------------------
    # GROUP 4: Round Number Psychology
    #
    # For each candle, find the DOGE round-number level (from config) nearest
    # to the current close, then compute signed distance and proximity flag.
    #
    # Vectorised via a (n_rows, n_levels) absolute-difference matrix.
    # All operations use close[T] only — no future data.                   SAFE
    # -----------------------------------------------------------------------
    levels = np.array(cfg.round_number_levels, dtype=np.float64)
    close_vals = out["close"].to_numpy(dtype=np.float64)

    # Shape: (n_rows, n_levels) — each row is |close[t] - level_i|
    diffs = np.abs(close_vals[:, None] - levels[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    nearest_level = levels[nearest_idx]

    distance_pct = (close_vals - nearest_level) / (nearest_level + _EPS)
    at_round = (np.abs(distance_pct) < 0.01).astype(np.int8)

    out["nearest_round_level"] = nearest_level
    out["distance_to_round_pct"] = distance_pct
    out["at_round_number_flag"] = at_round

    n_added = len(out.columns) - len(doge_df.columns)
    logger.debug(
        "compute_doge_features: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
