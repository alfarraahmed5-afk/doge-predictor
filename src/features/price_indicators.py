"""Price technical indicators for the DOGE prediction pipeline.

All indicators are computed at time T using **only** data up to and including T.
No indicator uses data from T+1 or later.

Lookahead audit (verify after every modification):
    SMA / EMA          rolling mean / ewm of past closes only               SAFE
    MACD               ewm difference of past closes only                   SAFE
    RSI                past return differences only                          SAFE
    BBANDS             rolling mean + std of past closes only               SAFE
    ATR                past high / low / close only                         SAFE
    STOCH              rolling max/min of past candles + smoothing          SAFE
    Ichimoku cloud     current_span[T] = span[T - displacement]             SAFE
                       shift(+N) pulls values from the past; max lookback
                       is senkou_b_period + displacement = 78 periods
    macd_hist_dir      sign(hist[T]) vs sign(hist[T-1])  — past only       SAFE
    stoch_crossover    k_above_d[T] vs k_above_d[T-1]   — past only        SAFE
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.config import DogeSettings, doge_settings

# Small epsilon for safe division
_EPS: float = 1e-10

# Columns that must be present in the input DataFrame
_REQUIRED_COLS: tuple[str, ...] = ("open_time", "open", "high", "low", "close", "volume")


def compute_price_indicators(
    df: pd.DataFrame,
    cfg: DogeSettings | None = None,
) -> pd.DataFrame:
    """Compute price-based technical indicators and append to a copy of *df*.

    All periods are read from ``cfg.indicators`` (config/doge_settings.yaml).
    No constant is hardcoded in this function.

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
        raise ValueError(f"compute_price_indicators: missing columns {missing}")

    out = df.copy()
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Simple Moving Averages
    #    SMA at T = mean(close[T-period+1 .. T])  — past data only      SAFE
    # ------------------------------------------------------------------
    for period in ind.sma_periods:
        out[f"sma_{period}"] = talib.SMA(close, timeperiod=period)

    # ------------------------------------------------------------------
    # 2. Exponential Moving Averages
    #    EMA at T = ewm of past closes only                              SAFE
    # ------------------------------------------------------------------
    ema_cache: dict[int, np.ndarray] = {}
    for period in ind.ema_periods:
        arr = talib.EMA(close, timeperiod=period)
        ema_cache[period] = arr
        out[f"ema_{period}"] = arr

    ema_200 = ema_cache.get(200, talib.EMA(close, timeperiod=200))
    # (close - ema_200) / ema_200 — both terms computed from past data   SAFE
    out["price_vs_ema200"] = (close - ema_200) / (np.abs(ema_200) + _EPS)

    # ------------------------------------------------------------------
    # 3. MACD  (fast_ema - slow_ema, signalperiod ewm of the difference)
    #    All ewm operations use only data up to T                        SAFE
    # ------------------------------------------------------------------
    macd_line, macd_signal_line, macd_hist = talib.MACD(
        close,
        fastperiod=ind.macd_fast,
        slowperiod=ind.macd_slow,
        signalperiod=ind.macd_signal,
    )
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal_line
    out["macd_hist"] = macd_hist

    # Direction: +1 when hist crosses from neg→pos; -1 when pos→neg; 0 otherwise
    # Uses sign(hist[T]) vs sign(hist[T-1]) — past data only             SAFE
    hist_s = pd.Series(macd_hist, index=out.index)
    sign_curr = np.sign(hist_s).fillna(0)
    sign_prev = sign_curr.shift(1).fillna(0)
    changed = sign_curr != sign_prev
    hist_dir = pd.Series(0, index=out.index, dtype=np.int8)
    hist_dir[changed & (sign_curr > 0)] = 1
    hist_dir[changed & (sign_curr < 0)] = -1
    out["macd_hist_direction"] = hist_dir

    # ------------------------------------------------------------------
    # 4. RSI
    #    RSI at T uses past return differences only                       SAFE
    # ------------------------------------------------------------------
    rsi = talib.RSI(close, timeperiod=ind.rsi_period)
    rsi_s = pd.Series(rsi, index=out.index)
    out["rsi_14"] = rsi_s
    out["rsi_overbought"] = (rsi_s > 70.0).astype(np.int8)
    out["rsi_oversold"] = (rsi_s < 30.0).astype(np.int8)

    # ------------------------------------------------------------------
    # 5. Bollinger Bands
    #    Rolling mean ± k*std of past closes only                        SAFE
    # ------------------------------------------------------------------
    bb_upper_arr, bb_mid_arr, bb_lower_arr = talib.BBANDS(
        close,
        timeperiod=ind.bb_period,
        nbdevup=ind.bb_std,
        nbdevdn=ind.bb_std,
        matype=0,  # SMA middle band
    )
    out["bb_upper"] = bb_upper_arr
    out["bb_lower"] = bb_lower_arr
    bb_width = (bb_upper_arr - bb_lower_arr) / (np.abs(bb_mid_arr) + _EPS)
    out["bb_width"] = bb_width
    out["bb_pct_b"] = (close - bb_lower_arr) / (bb_upper_arr - bb_lower_arr + _EPS)
    out["bb_squeeze_flag"] = (
        pd.Series(bb_width, index=out.index) < ind.bb_squeeze_threshold
    ).astype(np.int8)

    # ------------------------------------------------------------------
    # 6. ATR  (Average True Range)
    #    Uses high/low/close of past candles only                        SAFE
    # ------------------------------------------------------------------
    atr = talib.ATR(high, low, close, timeperiod=ind.atr_period)
    out["atr_14"] = atr
    out["atr_14_norm"] = atr / (np.abs(close) + _EPS)

    # ------------------------------------------------------------------
    # 7. Stochastic
    #    Rolling max/min over past fastk candles, then smoothed          SAFE
    # ------------------------------------------------------------------
    stoch_k, stoch_d = talib.STOCH(
        high,
        low,
        close,
        fastk_period=ind.stoch_fastk,
        slowk_period=ind.stoch_slowk,
        slowk_matype=0,
        slowd_period=ind.stoch_slowd,
        slowd_matype=0,
    )
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    # Crossover flag: +1 when K crosses above D (bullish), -1 when below
    # Compares k_above_d[T] vs k_above_d[T-1] — past data only          SAFE
    k_s = pd.Series(stoch_k, index=out.index)
    d_s = pd.Series(stoch_d, index=out.index)
    k_above = (k_s > d_s).fillna(False)
    k_above_prev = k_above.shift(1).fillna(False)
    crossover = pd.Series(0, index=out.index, dtype=np.int8)
    crossover[k_above & ~k_above_prev] = 1    # K crossed above D — bullish
    crossover[~k_above & k_above_prev] = -1   # K crossed below D — bearish
    out["stoch_crossover_flag"] = crossover

    # ------------------------------------------------------------------
    # 8. Ichimoku Cloud Position
    #
    # The cloud VISIBLE at time T consists of:
    #   Span A[T] = (Tenkan[T-disp] + Kijun[T-disp]) / 2
    #   Span B[T] = (52-period high+low)/2  calculated at T-disp
    #
    # shift(+displacement) shifts the series BACKWARD, meaning current_span[T]
    # uses the span calculated at T-displacement (strictly past data).
    # Maximum lookback: senkou_b_period + displacement = 52 + 26 = 78    SAFE
    # ------------------------------------------------------------------
    disp = ind.ichimoku_displacement
    hi_s = out["high"]
    lo_s = out["low"]

    tenkan = (
        hi_s.rolling(ind.ichimoku_tenkan).max() + lo_s.rolling(ind.ichimoku_tenkan).min()
    ) / 2
    kijun = (
        hi_s.rolling(ind.ichimoku_kijun).max() + lo_s.rolling(ind.ichimoku_kijun).min()
    ) / 2
    span_a = (tenkan + kijun) / 2
    span_b = (
        hi_s.rolling(ind.ichimoku_senkou_b).max()
        + lo_s.rolling(ind.ichimoku_senkou_b).min()
    ) / 2

    # Current cloud = spans calculated disp periods ago (no lookahead)
    current_span_a = span_a.shift(disp)
    current_span_b = span_b.shift(disp)
    cloud_top = pd.concat([current_span_a, current_span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([current_span_a, current_span_b], axis=1).min(axis=1)

    cloud_pos = pd.Series(0, index=out.index, dtype=np.int8)
    valid = cloud_top.notna() & cloud_bottom.notna()
    cloud_pos[valid & (out["close"] > cloud_top)] = 1
    cloud_pos[valid & (out["close"] < cloud_bottom)] = -1
    out["ichimoku_cloud_position"] = cloud_pos

    n_added = len(out.columns) - len(df.columns)
    logger.debug(
        "compute_price_indicators: added {} columns to {} rows",
        n_added,
        len(out),
    )
    return out
