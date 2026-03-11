"""Lag, momentum, and rolling statistical features for the DOGE pipeline.

All features are computed at time T using **only** data up to and including T.
No feature uses data from T+1 or later.

Lookahead audit (verify after every modification):
    log_ret_N      log(close[T] / close[T-N]) = log(close) - log(close.shift(N))
                   shift(+N) pulls from T-N — strictly past                 SAFE
    vol_N          rolling(N).std(log_ret_1) over past N returns             SAFE
    skew_24        rolling(24).skew(log_ret_1) over past 24 returns          SAFE
    kurt_24        rolling(24).kurt(log_ret_1) over past 24 returns          SAFE
    mom_N          close[T] / close[T-N] - 1 = close / close.shift(N) - 1   SAFE
    hl_range       (high[T] - low[T]) / close[T] — single-candle only       SAFE

CRITICAL NAMING RULE:
    shift(+N)  → looks N periods INTO THE PAST   → lag feature    CORRECT
    shift(-N)  → looks N periods INTO THE FUTURE → LOOKAHEAD BIAS  BANNED
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DogeSettings, doge_settings

_REQUIRED_COLS: tuple[str, ...] = ("open_time", "high", "low", "close")


def compute_lag_features(
    df: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Compute lag, momentum, and rolling statistical features.

    All periods are read from ``cfg.indicators`` (config/doge_settings.yaml).

    Lookahead bias: **none**.  Every column at row index T is computed using
    only rows [0 .. T].  The module-level audit comment enumerates each
    feature individually.

    Args:
        df: OHLCV DataFrame sorted ascending by ``open_time``.  Must contain
            columns: ``open_time``, ``high``, ``low``, ``close``.
            ``open_time`` is UTC epoch milliseconds.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If *None*
            the module-level singleton is used.

    Returns:
        New DataFrame (``df.copy()``) with all lag/momentum columns appended.
        Rows within the warmup window will contain ``NaN`` — expected for
        long-period lags and rolling statistics.

    Raises:
        ValueError: If any required column is missing from *df*.
    """
    if cfg is None:
        cfg = doge_settings
    ind = cfg.indicators

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_lag_features: missing columns {missing}")

    out = df.copy()
    log_close = np.log(out["close"])

    # ------------------------------------------------------------------
    # 1. Log Returns  log(close[T] / close[T-N]) for each period N
    #    shift(+N) pulls close[T-N] — strictly past data                 SAFE
    # ------------------------------------------------------------------
    for n in ind.log_return_periods:
        # log(close[T]) - log(close[T-N])  ==  log(close[T] / close[T-N])
        out[f"log_ret_{n}"] = log_close - log_close.shift(n)

    # ------------------------------------------------------------------
    # 2. Rolling Volatility  std(log_ret_1) over past W candles
    #    rolling(W) uses [T-W+1 .. T] — past data only                   SAFE
    # ------------------------------------------------------------------
    log_ret_1 = out["log_ret_1"]  # already computed above
    for w in ind.rolling_vol_windows:
        out[f"vol_{w}"] = log_ret_1.rolling(w, min_periods=2).std()

    # ------------------------------------------------------------------
    # 3. Rolling Skewness  skew(log_ret_1) over past skew_period candles
    #    rolling uses only past observations                              SAFE
    # ------------------------------------------------------------------
    out["rolling_skew_24"] = log_ret_1.rolling(
        ind.rolling_skew_period, min_periods=3
    ).skew()

    # ------------------------------------------------------------------
    # 4. Rolling Kurtosis  kurt(log_ret_1) over past kurt_period candles
    #    rolling uses only past observations                              SAFE
    # ------------------------------------------------------------------
    out["rolling_kurt_24"] = log_ret_1.rolling(
        ind.rolling_kurt_period, min_periods=4
    ).kurt()

    # ------------------------------------------------------------------
    # 5. Price Momentum  close[T] / close[T-N] - 1
    #    shift(+N) pulls from the past — no lookahead                    SAFE
    # ------------------------------------------------------------------
    for n in ind.momentum_periods:
        out[f"mom_{n}"] = out["close"] / out["close"].shift(n) - 1.0

    # ------------------------------------------------------------------
    # 6. High–Low Range  (high[T] - low[T]) / close[T]
    #    Uses only the current candle's OHLC — single timestamp          SAFE
    # ------------------------------------------------------------------
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]

    n_added = len(out.columns) - len(df.columns)
    logger.debug(
        "compute_lag_features: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
