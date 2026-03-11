"""Volume-based technical indicators for the DOGE prediction pipeline.

All indicators are computed at time T using **only** data up to and including T.
No indicator uses data from T+1 or later.

Lookahead audit (verify after every modification):
    OBV            cumulative; uses close[T] vs close[T-1] + volume[T]   SAFE
    obv_ema_ratio  ewm of past OBV values only                           SAFE
    VWAP           resets at UTC midnight; cumulative within each day     SAFE
                   at T uses close/high/low/volume of candles [day-start..T]
    price_vs_vwap  (close[T] - vwap[T]) / vwap[T]                       SAFE
    volume_ma_20   rolling(20).mean() of past volumes                     SAFE
    volume_ma_ratio volume[T] / rolling_mean[T]  — both at T             SAFE
    CMF            rolling sum of money-flow volumes over past 20 candles SAFE
    cvd_approx     cumsum of per-candle delta; delta uses only T data     SAFE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.config import DogeSettings, doge_settings

_EPS: float = 1e-10
_REQUIRED_COLS: tuple[str, ...] = ("open_time", "open", "high", "low", "close", "volume")


def compute_volume_indicators(
    df: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Compute volume-based indicators and append to a copy of *df*.

    All periods are read from ``cfg.indicators`` (config/doge_settings.yaml).
    VWAP resets at UTC midnight on every calendar day.

    Lookahead bias: **none**.  Every column at row index T is computed using
    only rows [0 .. T].  See module-level audit comment for the full proof.

    Args:
        df: Raw OHLCV DataFrame sorted ascending by ``open_time``.  Must
            contain columns: ``open_time``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.  ``open_time`` is UTC epoch milliseconds.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If *None*
            the module-level singleton is used.

    Returns:
        New DataFrame (``df.copy()``) with all indicator columns appended.
        Rows within the warmup window will contain ``NaN`` for the relevant
        indicator — this is expected and correct behaviour.

    Raises:
        ValueError: If any required column is missing from *df*.
    """
    if cfg is None:
        cfg = doge_settings
    ind = cfg.indicators

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_volume_indicators: missing columns {missing}")

    out = df.copy()
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    volume = out["volume"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. On-Balance Volume (OBV)
    #    OBV[T] += volume[T]  if close[T] > close[T-1]
    #    OBV[T] -= volume[T]  if close[T] < close[T-1]
    #    Uses close[T] vs close[T-1] and volume[T] — past data only      SAFE
    # ------------------------------------------------------------------
    obv_arr = talib.OBV(close, volume)
    obv_s = pd.Series(obv_arr, index=out.index)
    out["obv"] = obv_s

    # OBV / EWM(obv, span) — ewm uses only past OBV values               SAFE
    obv_ema = obv_s.ewm(span=ind.obv_ema_span, adjust=False).mean()
    out["obv_ema_ratio"] = obv_s / (obv_ema.abs() + _EPS)

    # ------------------------------------------------------------------
    # 2. VWAP — reset at UTC midnight every calendar day
    #    At T uses: close/high/low/volume of candles [day_start .. T]     SAFE
    #    groupby(date).cumsum() resets the running sum at each midnight
    # ------------------------------------------------------------------
    dt_series = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    # Normalise to midnight UTC so all candles in the same day share the same key
    date_key = dt_series.dt.normalize()

    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0
    tp_vol = typical_price * out["volume"]

    cum_tp_vol = tp_vol.groupby(date_key).cumsum()
    cum_vol = out["volume"].groupby(date_key).cumsum()
    vwap = cum_tp_vol / (cum_vol + _EPS)
    out["vwap"] = vwap
    out["price_vs_vwap"] = (out["close"] - vwap) / (vwap.abs() + _EPS)

    # ------------------------------------------------------------------
    # 3. Volume Moving Average + Ratio
    #    rolling(20) uses past 20 candles only                            SAFE
    #    NOTE: we normalise by the rolling mean — never use raw volume
    # ------------------------------------------------------------------
    vol_s = out["volume"]
    vol_ma = vol_s.rolling(ind.volume_ma_period, min_periods=1).mean()
    out["volume_ma_20"] = vol_ma
    out["volume_ma_ratio"] = vol_s / (vol_ma + _EPS)

    # ------------------------------------------------------------------
    # 4. Chaikin Money Flow (CMF)
    #    MF multiplier = ((close - low) - (high - close)) / (high - low)
    #    CMF = sum(mf_vol, 20) / sum(volume, 20)
    #    Rolling sums over past 20 candles only                           SAFE
    # ------------------------------------------------------------------
    hl_range = (out["high"] - out["low"]).replace(0, _EPS)
    mf_mult = ((out["close"] - out["low"]) - (out["high"] - out["close"])) / hl_range
    mf_vol = mf_mult * out["volume"]
    cmf = (
        mf_vol.rolling(ind.cmf_period, min_periods=1).sum()
        / (vol_s.rolling(ind.cmf_period, min_periods=1).sum() + _EPS)
    )
    out["cmf_20"] = cmf

    # ------------------------------------------------------------------
    # 5. Cumulative Volume Delta (CVD) approximation
    #    delta[T] = (close[T] - low[T]) / (high[T] - low[T]) * volume[T]
    #    Uses only current-candle data; cumsum aggregates past values     SAFE
    # ------------------------------------------------------------------
    hl_range_np = np.where(
        (high - low) < _EPS,
        _EPS,
        high - low,
    )
    cvd_delta = (close - low) / hl_range_np * volume
    out["cvd_approx"] = pd.Series(np.cumsum(cvd_delta), index=out.index)

    n_added = len(out.columns) - len(df.columns)
    logger.debug(
        "compute_volume_indicators: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
