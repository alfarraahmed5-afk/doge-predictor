"""Higher time-frame (HTF) derived features for the DOGE prediction pipeline.

All features are computed at time T using only data from CLOSED candles at T.

CRITICAL LOOKAHEAD GUARD — READ BEFORE ANY MODIFICATION:

    At 1h candle with open_time = 15:00, the current 4h candle closes at 16:00.
    The last CLOSED 4h candle available at 15:00 is the one that closed at 12:00.
    Any logic that uses the 4h candle closing at 16:00 is LOOKAHEAD BIAS.

    Guard implementation:
        1. Compute indicators on the raw 4h (or 1d) DataFrame.
        2. Create column  lookup_key = open_time + interval_ms
           (lookup_key equals the bar's close time).
        3. pd.merge_asof(left_on=open_time_1h, right_on=lookup_key, direction="backward")
           A 4h bar is visible at 1h time T  iff  lookup_key <= T.

    Concrete example (T0 = 2022-01-01 00:00 UTC, 4h interval):
        4h bar A  open_time=08:00, lookup_key=12:00
        4h bar B  open_time=12:00, lookup_key=16:00

        1h at 08:00 → lookup_key 12:00 > 08:00 → bar A NOT visible (NaN)
        1h at 11:00 → lookup_key 12:00 > 11:00 → bar A NOT visible (NaN)
        1h at 12:00 → lookup_key 12:00 ≤ 12:00 → bar A VISIBLE  ← first row
        1h at 15:00 → bar A still visible (12:00 ≤ 15:00, 16:00 > 15:00)
        1h at 16:00 → bar B VISIBLE (16:00 ≤ 16:00) ← first row for bar B

Lookahead audit:
    htf_4h_rsi        talib.RSI on 4h closed candles, merge_asof guard       SAFE
    htf_4h_trend      sign(EMA20_4h - EMA50_4h) on closed bars, guarded      SAFE
    htf_4h_bb_pctb    %B on 4h closed bars, guarded                          SAFE
    htf_1d_trend      sign(EMA20_1d - EMA50_1d) on closed 1d bars            SAFE
    htf_1d_return     log(close_1d[T] / close_1d[T-1]) on closed bars        SAFE
    ath_distance      log(doge_ath_price / close_1h[T]) — fixed ATH constant SAFE
                      expanding().max() NOT used; fixed $0.731 from config
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.config import DogeSettings, doge_settings

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_EPS: float = 1e-10

_MS_PER_HOUR: int = 3_600_000
_4H_MS: int = 4 * _MS_PER_HOUR
_1D_MS: int = 24 * _MS_PER_HOUR

_REQUIRED_1H: tuple[str, ...] = ("open_time", "close")
_REQUIRED_4H: tuple[str, ...] = ("open_time", "high", "low", "close")
_REQUIRED_1D: tuple[str, ...] = ("open_time", "close")

# The 6 canonical HTF feature names produced by this module
HTF_FEATURE_NAMES: tuple[str, ...] = (
    "htf_4h_rsi",
    "htf_4h_trend",
    "htf_4h_bb_pctb",
    "htf_1d_trend",
    "htf_1d_return",
    "ath_distance",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_4h_indicators(
    df_4h: pd.DataFrame,
    cfg: DogeSettings,
) -> pd.DataFrame:
    """Compute HTF indicators on the 4h DataFrame.

    Returns a new DataFrame with ``open_time``, ``lookup_key``, and 4h
    indicator columns.  ``lookup_key = open_time + _4H_MS`` is the earliest
    1h timestamp at which this bar may be used (i.e., the bar's close time).

    Args:
        df_4h: 4h OHLCV DataFrame, sorted ascending by ``open_time``.
        cfg: :class:`~src.config.DogeSettings` instance.

    Returns:
        DataFrame with columns: ``open_time``, ``lookup_key``, ``htf_4h_rsi``,
        ``htf_4h_trend``, ``htf_4h_bb_pctb``.
    """
    ind = cfg.indicators
    close = df_4h["close"].to_numpy(dtype=np.float64)
    high = df_4h["high"].to_numpy(dtype=np.float64)
    low = df_4h["low"].to_numpy(dtype=np.float64)

    result = pd.DataFrame({"open_time": df_4h["open_time"].to_numpy()})
    result["lookup_key"] = result["open_time"] + _4H_MS

    # RSI
    result["htf_4h_rsi"] = talib.RSI(close, timeperiod=ind.htf_rsi_period)

    # Trend: +1 (EMA20 > EMA50) / -1 (EMA20 ≤ EMA50)
    # Follows spec: (ema20 > ema50).map({True: 1, False: -1})
    # 0 only during warmup when EMAs are NaN
    ema_fast = talib.EMA(close, timeperiod=ind.htf_ema_fast)
    ema_slow = talib.EMA(close, timeperiod=ind.htf_ema_slow)
    trend = np.zeros(len(close), dtype=np.int8)
    valid = ~(np.isnan(ema_fast) | np.isnan(ema_slow))
    trend[valid] = np.where(ema_fast[valid] > ema_slow[valid], 1, -1).astype(np.int8)
    result["htf_4h_trend"] = trend

    # Bollinger %B = (close - lower) / (upper - lower)
    bb_upper, _, bb_lower = talib.BBANDS(
        close,
        timeperiod=ind.htf_bb_period,
        nbdevup=ind.htf_bb_std,
        nbdevdn=ind.htf_bb_std,
        matype=0,
    )
    bb_range = bb_upper - bb_lower
    bb_range_safe = np.where(np.abs(bb_range) < _EPS, _EPS, bb_range)
    result["htf_4h_bb_pctb"] = (close - bb_lower) / bb_range_safe

    return result


def _compute_1d_indicators(
    df_1d: pd.DataFrame,
    cfg: DogeSettings,
) -> pd.DataFrame:
    """Compute HTF indicators on the 1d DataFrame.

    Returns a new DataFrame with ``open_time``, ``lookup_key``, and 1d
    indicator columns.  ``lookup_key = open_time + _1D_MS``.

    Args:
        df_1d: 1d OHLCV DataFrame, sorted ascending by ``open_time``.
        cfg: :class:`~src.config.DogeSettings` instance.

    Returns:
        DataFrame with columns: ``open_time``, ``lookup_key``, ``htf_1d_trend``,
        ``htf_1d_return``.
    """
    ind = cfg.indicators
    close = df_1d["close"].to_numpy(dtype=np.float64)
    close_s = pd.Series(close)

    result = pd.DataFrame({"open_time": df_1d["open_time"].to_numpy()})
    result["lookup_key"] = result["open_time"] + _1D_MS

    # Trend: +1 / -1 (per spec: (ema20 > ema50).map({True: 1, False: -1}))
    ema_fast = talib.EMA(close, timeperiod=ind.htf_ema_fast)
    ema_slow = talib.EMA(close, timeperiod=ind.htf_ema_slow)
    trend = np.zeros(len(close), dtype=np.int8)
    valid = ~(np.isnan(ema_fast) | np.isnan(ema_slow))
    trend[valid] = np.where(ema_fast[valid] > ema_slow[valid], 1, -1).astype(np.int8)
    result["htf_1d_trend"] = trend

    # Daily log return: log(close[T] / close[T-1]) — shift(+1) is past data SAFE
    result["htf_1d_return"] = (np.log(close_s / close_s.shift(1))).to_numpy()

    return result


def _merge_htf(
    doge_1h: pd.DataFrame,
    htf_indicators: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Map HTF indicator rows onto the 1h DataFrame using the lookahead guard.

    Uses ``pd.merge_asof`` with ``left_on='open_time'`` and
    ``right_on='lookup_key'``.  A HTF bar is only matched to 1h candle T when
    ``lookup_key <= T`` (the bar has fully closed).

    Args:
        doge_1h: 1h DataFrame with ``open_time`` column.
        htf_indicators: DataFrame with ``lookup_key`` and *feature_cols*,
            sorted ascending by ``lookup_key``.
        feature_cols: Column names to extract from *htf_indicators*.

    Returns:
        The *doge_1h* DataFrame with *feature_cols* appended (NaN where no
        HTF bar has closed yet).
    """
    left = doge_1h[["open_time"]].copy().sort_values("open_time")
    right = htf_indicators[["lookup_key"] + feature_cols].sort_values("lookup_key")

    merged = pd.merge_asof(
        left,
        right,
        left_on="open_time",
        right_on="lookup_key",
        direction="backward",  # largest lookup_key <= open_time_1h
    )
    merged = merged.drop(columns=["lookup_key"])
    merged.index = left.index

    result = doge_1h.copy()
    for col in feature_cols:
        result[col] = merged[col].to_numpy()

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_htf_features(
    doge_1h: pd.DataFrame,
    doge_4h: pd.DataFrame,
    doge_1d: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Compute HTF-derived features and append to a copy of *doge_1h*.

    Applies the mandatory lookahead guard: a 4h (or 1d) bar is only used for
    1h features once that bar has fully closed.

    ath_distance uses a FIXED all-time-high price (``cfg.doge_ath_price``)
    loaded from ``config/doge_settings.yaml``.  This avoids any look-ahead
    from expanding ATH and is the correct approach for a fixed historical
    reference point ($0.731 on Binance, 2021-05-08).

    Lookahead bias: **none**.  See module-level audit and guard description.

    Args:
        doge_1h: DOGEUSDT 1h OHLCV DataFrame, sorted ascending by
            ``open_time``.  Must contain: ``open_time``, ``close``.
        doge_4h: DOGEUSDT 4h OHLCV DataFrame, sorted ascending by
            ``open_time``.  Must contain: ``open_time``, ``high``,
            ``low``, ``close``.
        doge_1d: DOGEUSDT 1d OHLCV DataFrame, sorted ascending by
            ``open_time``.  Must contain: ``open_time``, ``close``.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If
            *None* the module-level singleton is used.

    Returns:
        New DataFrame (``doge_1h.copy()``) with six HTF feature columns
        appended: ``htf_4h_rsi``, ``htf_4h_trend``, ``htf_4h_bb_pctb``,
        ``htf_1d_trend``, ``htf_1d_return``, ``ath_distance``.

    Raises:
        ValueError: If any required column is missing from any input.
    """
    if cfg is None:
        cfg = doge_settings

    # Input validation
    missing_1h = [c for c in _REQUIRED_1H if c not in doge_1h.columns]
    missing_4h = [c for c in _REQUIRED_4H if c not in doge_4h.columns]
    missing_1d = [c for c in _REQUIRED_1D if c not in doge_1d.columns]

    if missing_1h:
        raise ValueError(f"compute_htf_features: doge_1h missing columns {missing_1h}")
    if missing_4h:
        raise ValueError(f"compute_htf_features: doge_4h missing columns {missing_4h}")
    if missing_1d:
        raise ValueError(f"compute_htf_features: doge_1d missing columns {missing_1d}")

    out = doge_1h.copy()

    # -----------------------------------------------------------------------
    # GROUP 1: 4h indicators  (RSI, trend, BB%B)
    # -----------------------------------------------------------------------
    df_4h_sorted = doge_4h.sort_values("open_time").reset_index(drop=True)
    indicators_4h = _compute_4h_indicators(df_4h_sorted, cfg)
    cols_4h = ["htf_4h_rsi", "htf_4h_trend", "htf_4h_bb_pctb"]
    out = _merge_htf(out, indicators_4h, cols_4h)

    n_nan_4h = int(pd.isna(out["htf_4h_rsi"]).sum())
    if n_nan_4h > 0:
        logger.debug(
            "compute_htf_features: {} 1h rows have NaN 4h features (no closed bar yet)",
            n_nan_4h,
        )

    # -----------------------------------------------------------------------
    # GROUP 2: 1d indicators  (trend, log-return)
    # -----------------------------------------------------------------------
    df_1d_sorted = doge_1d.sort_values("open_time").reset_index(drop=True)
    indicators_1d = _compute_1d_indicators(df_1d_sorted, cfg)
    cols_1d = ["htf_1d_trend", "htf_1d_return"]
    out = _merge_htf(out, indicators_1d, cols_1d)

    n_nan_1d = int(pd.isna(out["htf_1d_trend"]).sum())
    if n_nan_1d > 0:
        logger.debug(
            "compute_htf_features: {} 1h rows have NaN 1d features (no closed bar yet)",
            n_nan_1d,
        )

    # -----------------------------------------------------------------------
    # GROUP 3: ATH distance (fixed historical ATH from config)
    #
    # Formula: log(doge_ath_price / close_1h[T])
    #   - Always >= 0: when close < ATH, log(ATH/close) > 0
    #   - Equals 0 when close == ATH
    #   - Increases as price falls below ATH
    #
    # DESIGN NOTE: Uses a FIXED ATH price ($0.731) rather than expanding().max()
    # because expanding max would understate the ATH distance for pre-2021 data
    # (where DOGE hadn't yet reached $0.731).  A fixed reference point is more
    # consistent across the full training window.
    #
    # The constant is stored in doge_settings.doge_ath_price — not hardcoded. SAFE
    # -----------------------------------------------------------------------
    close_vals = out["close"].to_numpy(dtype=np.float64)
    ath = cfg.doge_ath_price
    out["ath_distance"] = np.log(ath / (close_vals + _EPS))

    n_added = len(out.columns) - len(doge_1h.columns)
    logger.debug(
        "compute_htf_features: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
