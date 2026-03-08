"""Market regime classifier for DOGECOIN 1h candle data.

``DogeRegimeClassifier`` assigns one of five market regime labels to every
candle in a DOGEUSDT OHLCV DataFrame. All numeric thresholds are loaded from
``config/regime_config.yaml`` — none are hardcoded in this module.

Five regimes (defined in CLAUDE.md Section 6):
    ``TRENDING_BULL``     : EMA20 > EMA50 > EMA200 AND 7d return > +5%
    ``TRENDING_BEAR``     : EMA20 < EMA50 < EMA200 AND 7d return < -5%
    ``RANGING_HIGH_VOL``  : BB width > threshold
    ``RANGING_LOW_VOL``   : BB width <= threshold (safe fallback)
    ``DECOUPLED``         : BTC-DOGE 24h log-return correlation < threshold
                            **Overrides all other regimes.**

Classification priority (highest to lowest):
    DECOUPLED > TRENDING_BULL > TRENDING_BEAR > RANGING_HIGH_VOL > RANGING_LOW_VOL

Usage::

    from src.regimes.classifier import DogeRegimeClassifier

    classifier = DogeRegimeClassifier()
    regimes = classifier.classify(doge_df, btc_df)
    print(classifier.get_regime_distribution(regimes))
    regime_now = classifier.get_at(open_time_ms)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.config import RegimeConfig
from src.config import regime_config as _default_regime_config

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

REGIME_LABELS: tuple[str, ...] = (
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
)

# Required columns for classification input DataFrames
_DOGE_REQUIRED_COLS: frozenset[str] = frozenset(
    {"open_time", "open", "high", "low", "close", "volume"}
)
_BTC_REQUIRED_COLS: frozenset[str] = frozenset({"open_time", "close"})

# Rolling window for BTC-DOGE log-return correlation (candles = 24h at 1h)
_CORR_WINDOW: int = 24

# Rolling window for 7-day return (168 * 1h candles)
_ROLL7D_WINDOW: int = 168


# ---------------------------------------------------------------------------
# DogeRegimeClassifier
# ---------------------------------------------------------------------------


class DogeRegimeClassifier:
    """Classify DOGEUSDT 1h candles into one of five market regimes.

    All numeric thresholds are loaded from ``config/regime_config.yaml`` via
    the :class:`~src.config.RegimeConfig` Pydantic model — never hardcoded.

    The classification is applied to an entire DataFrame at once (vectorised).
    The last result is cached internally so that :meth:`get_at` can serve
    point-in-time lookups without re-running the full pipeline.

    Args:
        config: :class:`~src.config.RegimeConfig` instance.  Defaults to the
            module-level singleton loaded from ``config/regime_config.yaml``.

    Example::

        from src.regimes.classifier import DogeRegimeClassifier

        clf = DogeRegimeClassifier()
        regimes = clf.classify(doge_df, btc_df)
        pct = clf.get_regime_distribution(regimes)
        current = clf.get_at(1_640_995_200_000)
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self._config: RegimeConfig = (
            config if config is not None else _default_regime_config
        )
        # Cache from last classify() call: open_time_ms → regime_label
        self._regime_by_time: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Classify every candle in *df* into one of the five market regimes.

        Classification steps (applied in precedence order, lowest to highest):

        1. Compute EMA20/50/200, ATR-normalised, BB width, 7d return,
           and 24h rolling BTC-DOGE log-return correlation.
        2. Assign ``RANGING_LOW_VOL`` to all rows (safe default).
        3. Override with ``RANGING_HIGH_VOL`` where BB width > threshold.
        4. Override with ``TRENDING_BEAR`` where EMA and 7d-return align.
        5. Override with ``TRENDING_BULL`` where EMA and 7d-return align.
        6. Override with ``DECOUPLED`` where BTC-DOGE corr < threshold.
        7. Assert no NaN regime labels remain.

        Args:
            df: DOGEUSDT 1h OHLCV DataFrame.  Required columns:
                ``open_time``, ``open``, ``high``, ``low``, ``close``,
                ``volume``.
            btc_df: Optional BTCUSDT 1h OHLCV DataFrame, timestamp-aligned
                to *df*.  Required columns: ``open_time``, ``close``.
                When ``None``, ``DECOUPLED`` is never assigned.

        Returns:
            :class:`pd.Series` of regime label strings, indexed identically
            to *df*.  Every element is one of the five regime label strings.

        Raises:
            ValueError: If *df* or *btc_df* are missing required columns,
                or if any NaN / unknown regime label remains after
                classification.
        """
        self._validate_inputs(df, btc_df)

        n: int = len(df)
        close = df["close"].to_numpy(dtype=np.float64)
        high = df["high"].to_numpy(dtype=np.float64)
        low = df["low"].to_numpy(dtype=np.float64)

        thresholds = self._config.thresholds

        # --- Step 1: Compute all indicator inputs ---
        ema20: np.ndarray = talib.EMA(close, timeperiod=20)
        ema50: np.ndarray = talib.EMA(close, timeperiod=50)
        ema200: np.ndarray = talib.EMA(close, timeperiod=200)

        atr: np.ndarray = talib.ATR(high, low, close, timeperiod=14)
        # atr_norm not used in current classification logic but stored
        # for future use and for validating ATR thresholds in tests.
        atr_norm: np.ndarray = np.where(close > 0, atr / close, np.nan)

        bb_upper, _bb_mid, bb_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        # BB width = (upper - lower) / close  (as specified in CLAUDE.md)
        bb_width: np.ndarray = np.where(
            close > 0, (bb_upper - bb_lower) / close, np.nan
        )

        # 7-day return: pct_change over 168 one-hour candles
        roll7d_ret: np.ndarray = (
            pd.Series(close).pct_change(_ROLL7D_WINDOW).to_numpy()
        )

        # BTC-DOGE 24h rolling log-return correlation
        btc_corr: np.ndarray | None = self._compute_btc_corr(df, btc_df)

        # --- Step 2: Initialise all rows as RANGING_LOW_VOL ---
        regime: np.ndarray = np.full(n, "RANGING_LOW_VOL", dtype=object)

        # --- Step 3: RANGING_HIGH_VOL ---
        valid_bb: np.ndarray = ~np.isnan(bb_width)
        regime[valid_bb & (bb_width > thresholds.bb_width_low)] = "RANGING_HIGH_VOL"

        # --- Step 4: TRENDING_BEAR ---
        valid_trend: np.ndarray = (
            ~np.isnan(ema20)
            & ~np.isnan(ema50)
            & ~np.isnan(ema200)
            & ~np.isnan(roll7d_ret)
        )
        regime[
            valid_trend
            & (ema20 < ema50)
            & (ema50 < ema200)
            & (roll7d_ret < thresholds.roll7d_bear)
        ] = "TRENDING_BEAR"

        # --- Step 5: TRENDING_BULL ---
        regime[
            valid_trend
            & (ema20 > ema50)
            & (ema50 > ema200)
            & (roll7d_ret > thresholds.roll7d_bull)
        ] = "TRENDING_BULL"

        # --- Step 6: DECOUPLED (highest priority, overrides everything) ---
        if btc_corr is not None:
            valid_corr: np.ndarray = ~np.isnan(btc_corr)
            regime[
                valid_corr & (btc_corr < thresholds.btc_corr_decoupled)
            ] = "DECOUPLED"

        # --- Step 7: Validate no NaN or unknown labels remain ---
        regime_series: pd.Series = pd.Series(regime, index=df.index, dtype=str)
        self._validate_regime_series(regime_series)

        # Cache for get_at() lookup
        self._regime_by_time = dict(
            zip(df["open_time"].tolist(), regime_series.tolist())
        )

        dist = self.get_regime_distribution(regime_series)
        logger.debug("Classified {} rows. Regime distribution: {}", n, dist)

        return regime_series

    def get_regime_distribution(self, regimes: pd.Series) -> dict[str, float]:
        """Return the fractional distribution of each regime.

        Args:
            regimes: Series of regime label strings (output of
                :meth:`classify`).

        Returns:
            Dict mapping each of the five regime label strings to its
            fraction of total rows.  Values sum to 1.0.
        """
        total: int = len(regimes)
        if total == 0:
            return {label: 0.0 for label in REGIME_LABELS}
        return {
            label: float((regimes == label).sum()) / total
            for label in REGIME_LABELS
        }

    def get_at(self, timestamp_ms: int) -> str:
        """Return the regime label at a specific candle open-time.

        Requires :meth:`classify` to have been called at least once.

        Args:
            timestamp_ms: UTC epoch-ms ``open_time`` of the target candle.

        Returns:
            Regime label string.

        Raises:
            RuntimeError: If :meth:`classify` has not been called yet.
            KeyError: If *timestamp_ms* is not in the cached result.
        """
        if not self._regime_by_time:
            raise RuntimeError(
                "classify() must be called before get_at()."
            )
        try:
            return self._regime_by_time[timestamp_ms]
        except KeyError as exc:
            raise KeyError(
                f"Timestamp {timestamp_ms} not found in classified regimes."
            ) from exc

    @staticmethod
    def detect_transition(prev_regime: str, curr_regime: str) -> bool:
        """Return ``True`` if the regime changed between two consecutive candles.

        Used by :mod:`src.monitoring.regime_monitor` to detect and log
        regime transitions.

        Args:
            prev_regime: Regime label for the previous candle.
            curr_regime: Regime label for the current candle.

        Returns:
            ``True`` if the regime has changed, ``False`` if it is the same.
        """
        return prev_regime != curr_regime

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_btc_corr(
        self,
        doge_df: pd.DataFrame,
        btc_df: pd.DataFrame | None,
    ) -> np.ndarray | None:
        """Compute the rolling 24-period BTC-DOGE log-return correlation.

        **Critical**: correlation is computed on LOG RETURNS, not raw prices.
        Using raw prices produces spurious correlation for trending assets
        (Yule-Walker / spurious regression effect).

        Args:
            doge_df: DOGEUSDT OHLCV DataFrame.
            btc_df: Optional BTCUSDT OHLCV DataFrame.

        Returns:
            Numpy array of rolling correlations (NaN for the first 23 rows),
            or ``None`` if *btc_df* is ``None``.
        """
        if btc_df is None:
            return None

        doge_close = doge_df["close"].to_numpy(dtype=np.float64)
        btc_close = btc_df["close"].to_numpy(dtype=np.float64)

        # Align lengths — BTC may have slightly different row count
        min_len: int = min(len(doge_close), len(btc_close))
        doge_close = doge_close[:min_len]
        btc_close = btc_close[:min_len]

        # Log returns: log(P_t / P_{t-1})
        doge_log_ret = pd.Series(np.log(doge_close / np.roll(doge_close, 1)))
        doge_log_ret.iloc[0] = np.nan  # first value is log(P0/P_{-1}) — invalid

        btc_log_ret = pd.Series(np.log(btc_close / np.roll(btc_close, 1)))
        btc_log_ret.iloc[0] = np.nan

        corr_series = doge_log_ret.rolling(_CORR_WINDOW).corr(btc_log_ret)
        corr_arr = corr_series.to_numpy()

        # If BTC had fewer rows than DOGE, pad the tail with NaN
        if min_len < len(doge_df):
            full = np.full(len(doge_df), np.nan)
            full[:min_len] = corr_arr
            return full

        return corr_arr

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame | None,
    ) -> None:
        """Validate required columns and non-empty condition.

        Args:
            df: DOGEUSDT OHLCV DataFrame.
            btc_df: Optional BTCUSDT OHLCV DataFrame.

        Raises:
            ValueError: On missing columns or empty DataFrame.
        """
        missing = _DOGE_REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"DOGE DataFrame is missing required columns: {sorted(missing)}"
            )
        if len(df) == 0:
            raise ValueError("DOGE DataFrame is empty — cannot classify regimes.")

        if btc_df is not None:
            btc_missing = _BTC_REQUIRED_COLS - set(btc_df.columns)
            if btc_missing:
                raise ValueError(
                    f"BTC DataFrame is missing required columns: {sorted(btc_missing)}"
                )

    @staticmethod
    def _validate_regime_series(regime_series: pd.Series) -> None:
        """Assert that every row has a known, non-null regime label.

        Args:
            regime_series: Series of regime label strings.

        Raises:
            ValueError: If any NaN, empty, or unrecognised label is found.
        """
        null_count: int = int(regime_series.isna().sum())
        if null_count:
            raise ValueError(
                f"BUG: {null_count} NaN regime labels after classification."
            )

        unknown_mask = ~regime_series.isin(REGIME_LABELS)
        if unknown_mask.any():
            unknown_vals = regime_series[unknown_mask].unique().tolist()
            raise ValueError(
                f"BUG: Unrecognised regime labels found: {unknown_vals}"
            )
