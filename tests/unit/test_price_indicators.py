"""Unit tests for src/features/price_indicators.py,
src/features/volume_indicators.py, and src/features/lag_features.py.

Mandatory test: test_lag_sanity_log_ret_1 — catches lookahead in lag computation.

All synthetic OHLCV data is generated with a deterministic seed; no fixture
Parquet files are required for these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import compute_lag_features
from src.features.price_indicators import compute_price_indicators
from src.features.volume_indicators import compute_volume_indicators

# ---------------------------------------------------------------------------
# Shared OHLCV fixture helpers
# ---------------------------------------------------------------------------

_START_MS: int = 1_640_995_200_000   # 2022-01-01 00:00:00 UTC
_INTERVAL_MS: int = 3_600_000        # 1 hour


def _make_ohlcv(n: int = 400, seed: int = 42, drift: float = 0.001) -> pd.DataFrame:
    """Return a deterministic OHLCV DataFrame with *n* 1-hour candles.

    The close price follows a log-normal random walk with the given drift so
    that there are enough trending candles to exercise most indicators.
    """
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.012, n)
    close = 0.10 * np.exp(np.cumsum(log_rets))          # starts near $0.10

    noise = rng.uniform(0.002, 0.008, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0.0, 0.004, n))
    # Ensure high >= open/close and low <= open/close (OHLC sanity)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(1e6, 5e6, n)
    open_times = _START_MS + np.arange(n) * _INTERVAL_MS

    return pd.DataFrame(
        {
            "open_time": open_times.astype(np.int64),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": open_times + _INTERVAL_MS - 1,
            "quote_volume": volume * close,
            "num_trades": rng.integers(100, 1000, n),
            "is_interpolated": False,
            "era": "training",
        }
    )


def _make_flat_ohlcv(n: int = 400, seed: int = 99) -> pd.DataFrame:
    """Return a flat (ranging) OHLCV DataFrame — small random noise around $0.10."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.002, n)   # near-zero drift
    close = 0.10 * np.exp(np.cumsum(log_rets))
    noise = rng.uniform(0.001, 0.003, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0.0, 0.001, n))
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = rng.uniform(5e5, 2e6, n)
    open_times = _START_MS + np.arange(n) * _INTERVAL_MS
    return pd.DataFrame(
        {
            "open_time": open_times.astype(np.int64),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": open_times + _INTERVAL_MS - 1,
            "quote_volume": volume * close,
            "num_trades": rng.integers(50, 500, n),
            "is_interpolated": False,
            "era": "training",
        }
    )


@pytest.fixture(scope="module")
def trending_ohlcv() -> pd.DataFrame:
    """400-row trending OHLCV DataFrame (module-scoped for speed)."""
    return _make_ohlcv(n=400, seed=42, drift=0.001)


@pytest.fixture(scope="module")
def flat_ohlcv() -> pd.DataFrame:
    """400-row flat/ranging OHLCV DataFrame."""
    return _make_flat_ohlcv(n=400, seed=99)


# ===========================================================================
# MODULE 1: price_indicators
# ===========================================================================


class TestPriceIndicatorsMissingColumns:
    """Validate input checking."""

    def test_missing_high_raises(self) -> None:
        df = _make_ohlcv(50).drop(columns=["high"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_price_indicators(df)

    def test_missing_close_raises(self) -> None:
        df = _make_ohlcv(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_price_indicators(df)


class TestPriceIndicatorsSameIndex:
    """Output index must be identical to input index."""

    def test_same_index_trending(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        pd.testing.assert_index_equal(result.index, trending_ohlcv.index)

    def test_same_index_flat(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(flat_ohlcv)
        pd.testing.assert_index_equal(result.index, flat_ohlcv.index)

    def test_input_not_mutated(self, trending_ohlcv: pd.DataFrame) -> None:
        cols_before = set(trending_ohlcv.columns)
        compute_price_indicators(trending_ohlcv)
        assert set(trending_ohlcv.columns) == cols_before


class TestRSI:
    """RSI must be in [0, 100] after warmup; flags must be binary."""

    def test_rsi_range_trending(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0.0).all(), "RSI below 0 detected"
        assert (rsi <= 100.0).all(), "RSI above 100 detected"

    def test_rsi_range_flat(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(flat_ohlcv)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0.0).all()
        assert (rsi <= 100.0).all()

    def test_rsi_overbought_binary(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert set(result["rsi_overbought"].unique()).issubset({0, 1})

    def test_rsi_oversold_binary(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert set(result["rsi_oversold"].unique()).issubset({0, 1})

    def test_rsi_overbought_consistent_with_rsi(
        self, trending_ohlcv: pd.DataFrame
    ) -> None:
        result = compute_price_indicators(trending_ohlcv)
        valid = result.dropna(subset=["rsi_14"])
        flag = valid["rsi_overbought"].astype(bool)
        rsi = valid["rsi_14"]
        # Every row flagged overbought must have RSI > 70
        assert (rsi[flag] > 70.0).all()
        # Every row NOT flagged must have RSI <= 70
        assert (rsi[~flag] <= 70.0).all()


class TestBollingerBands:
    """BB structure and derived metrics."""

    def test_upper_always_above_lower(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_upper_above_lower_flat(self, flat_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(flat_ohlcv)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_bb_width_non_negative(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert (result["bb_width"].dropna() >= 0.0).all()

    def test_bb_squeeze_flag_binary(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert set(result["bb_squeeze_flag"].unique()).issubset({0, 1})

    def test_squeeze_flag_tight_price(self) -> None:
        """Very tight price series → bb_width near 0 → squeeze flag = 1."""
        n = 50
        # Build a near-constant price series (tiny noise so BB doesn't degenerate)
        rng = np.random.default_rng(7)
        close = 0.10 + rng.uniform(-1e-6, 1e-6, n)
        df = pd.DataFrame(
            {
                "open_time": _START_MS + np.arange(n) * _INTERVAL_MS,
                "open": close,
                "high": close + 1e-7,
                "low": close - 1e-7,
                "close": close,
                "volume": np.ones(n) * 1e6,
                "close_time": _START_MS + np.arange(n) * _INTERVAL_MS + _INTERVAL_MS - 1,
                "quote_volume": close * 1e6,
                "num_trades": np.ones(n, dtype=int) * 100,
                "is_interpolated": False,
                "era": "training",
            }
        )
        result = compute_price_indicators(df)
        valid = result.dropna(subset=["bb_squeeze_flag"])
        # At least some rows should be flagged as squeeze
        assert valid["bb_squeeze_flag"].sum() > 0


class TestMACDHistDirection:
    """macd_hist_direction must correctly flag sign changes."""

    def test_direction_values_in_neg1_0_1(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert set(result["macd_hist_direction"].unique()).issubset({-1, 0, 1})

    def test_crossover_up_flagged_as_1(self, trending_ohlcv: pd.DataFrame) -> None:
        """At every row where hist crosses from neg→pos, direction must be +1."""
        result = compute_price_indicators(trending_ohlcv)
        hist = result["macd_hist"]
        direction = result["macd_hist_direction"]

        sign = np.sign(hist.dropna())
        sign_prev = sign.shift(1).dropna()
        cross_up = (sign.loc[sign_prev.index] > 0) & (sign_prev < 0)
        if cross_up.any():
            assert (direction.loc[cross_up[cross_up].index] == 1).all()

    def test_crossover_down_flagged_as_neg1(self, trending_ohlcv: pd.DataFrame) -> None:
        """At every row where hist crosses from pos→neg, direction must be -1."""
        result = compute_price_indicators(trending_ohlcv)
        hist = result["macd_hist"]
        direction = result["macd_hist_direction"]

        sign = np.sign(hist.dropna())
        sign_prev = sign.shift(1).dropna()
        cross_down = (sign.loc[sign_prev.index] < 0) & (sign_prev > 0)
        if cross_down.any():
            assert (direction.loc[cross_down[cross_down].index] == -1).all()

    def test_no_direction_when_no_sign_change(
        self, trending_ohlcv: pd.DataFrame
    ) -> None:
        """Rows with no sign change must have direction == 0."""
        result = compute_price_indicators(trending_ohlcv)
        hist = result["macd_hist"]
        direction = result["macd_hist_direction"]

        sign = np.sign(hist)
        sign_prev = sign.shift(1)
        no_change = (sign == sign_prev) & sign.notna() & sign_prev.notna()
        assert (direction[no_change] == 0).all()


class TestATR:
    """ATR must be non-negative; normalised ATR must be in (0, 1) for DOGE."""

    def test_atr_non_negative(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert (result["atr_14"].dropna() >= 0.0).all()

    def test_atr_norm_positive(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert (result["atr_14_norm"].dropna() > 0.0).all()


class TestIchimoku:
    """Ichimoku cloud position must only contain {-1, 0, 1}."""

    def test_cloud_position_values(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert set(result["ichimoku_cloud_position"].unique()).issubset({-1, 0, 1})

    def test_cloud_warmup_is_zero(self, trending_ohlcv: pd.DataFrame) -> None:
        """Rows where cloud is undefined (NaN spans) must be 0 (default)."""
        result = compute_price_indicators(trending_ohlcv)
        # The first ~78 rows have undefined cloud; all must be 0, not garbage
        assert (result["ichimoku_cloud_position"].iloc[:5] == 0).all()


class TestPriceNoNaNAfterWarmup:
    """After the longest warmup window (SMA-200), core indicators must be NaN-free."""

    _WARMUP = 200  # SMA-200 lookback

    def test_sma_7_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert not result["sma_7"].iloc[self._WARMUP:].isna().any()

    def test_ema_50_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert not result["ema_50"].iloc[self._WARMUP:].isna().any()

    def test_rsi_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert not result["rsi_14"].iloc[self._WARMUP:].isna().any()

    def test_bb_upper_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert not result["bb_upper"].iloc[self._WARMUP:].isna().any()

    def test_atr_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        assert not result["atr_14"].iloc[self._WARMUP:].isna().any()

    def test_macd_line_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_price_indicators(trending_ohlcv)
        # MACD warmup = slow + signal - 2 = 26 + 9 - 2 = 33 < 200
        assert not result["macd_line"].iloc[self._WARMUP:].isna().any()


# ===========================================================================
# MODULE 2: volume_indicators
# ===========================================================================


class TestVolumeIndicatorsSameIndex:
    """Output index must match input index."""

    def test_same_index(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        pd.testing.assert_index_equal(result.index, trending_ohlcv.index)

    def test_input_not_mutated(self, trending_ohlcv: pd.DataFrame) -> None:
        cols_before = set(trending_ohlcv.columns)
        compute_volume_indicators(trending_ohlcv)
        assert set(trending_ohlcv.columns) == cols_before


class TestVolumeIndicatorsMissingColumns:
    def test_missing_volume_raises(self) -> None:
        df = _make_ohlcv(50).drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_volume_indicators(df)


class TestOBV:
    """OBV must be monotonically increasing when close always rises."""

    def test_obv_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert "obv" in result.columns

    def test_obv_monotone_on_always_rising(self) -> None:
        """When close strictly increases every candle, OBV must be non-decreasing."""
        n = 50
        close = np.linspace(0.10, 0.20, n)
        df = pd.DataFrame(
            {
                "open_time": _START_MS + np.arange(n) * _INTERVAL_MS,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.ones(n) * 1e6,
                "close_time": _START_MS + np.arange(n) * _INTERVAL_MS + _INTERVAL_MS - 1,
                "quote_volume": close * 1e6,
                "num_trades": np.ones(n, dtype=int) * 100,
                "is_interpolated": False,
                "era": "training",
            }
        )
        result = compute_volume_indicators(df)
        obv = result["obv"].to_numpy()
        assert (np.diff(obv[1:]) >= 0).all(), "OBV not non-decreasing on always-rising close"

    def test_obv_ema_ratio_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert "obv_ema_ratio" in result.columns


class TestVWAP:
    """VWAP must reset at UTC midnight and equal weighted mean of intraday candles."""

    def test_vwap_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert "vwap" in result.columns

    def test_vwap_resets_daily(self) -> None:
        """First candle of each day: VWAP == typical price of that candle."""
        # Build exactly 2 days × 3 candles = 6 rows
        n = 6
        interval = _INTERVAL_MS * 8  # 8-hour candles so 3 per day
        open_times = _START_MS + np.arange(n) * interval
        close = np.array([0.10, 0.11, 0.12, 0.13, 0.14, 0.15])
        high = close + 0.005
        low = close - 0.005
        volume = np.ones(n) * 1e6
        df = pd.DataFrame(
            {
                "open_time": open_times.astype(np.int64),
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "close_time": open_times + interval - 1,
                "quote_volume": volume * close,
                "num_trades": np.ones(n, dtype=int) * 100,
                "is_interpolated": False,
                "era": "training",
            }
        )
        result = compute_volume_indicators(df)
        # Row 0 (first candle of day 1): VWAP = typical price of candle 0
        tp0 = (high[0] + low[0] + close[0]) / 3.0
        assert abs(result["vwap"].iloc[0] - tp0) < 1e-9

        # Row 3 (first candle of day 2): VWAP = typical price of candle 3
        tp3 = (high[3] + low[3] + close[3]) / 3.0
        assert abs(result["vwap"].iloc[3] - tp3) < 1e-9

    def test_price_vs_vwap_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert "price_vs_vwap" in result.columns


class TestCMF:
    """CMF must be in [-1, 1]."""

    def test_cmf_range(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        cmf = result["cmf_20"].dropna()
        assert (cmf >= -1.0).all() and (cmf <= 1.0).all()


class TestCVD:
    """CVD approx must be monotonically increasing when close == high (full buying)."""

    def test_cvd_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert "cvd_approx" in result.columns

    def test_cvd_increases_when_close_eq_high(self) -> None:
        """When close == high every candle, delta == volume → CVD strictly increases."""
        n = 30
        price = np.linspace(0.10, 0.20, n)
        df = pd.DataFrame(
            {
                "open_time": _START_MS + np.arange(n) * _INTERVAL_MS,
                "open": price,
                "high": price,          # close == high → delta = 1 × volume
                "low": price * 0.99,
                "close": price,
                "volume": np.ones(n) * 1e6,
                "close_time": _START_MS + np.arange(n) * _INTERVAL_MS + _INTERVAL_MS - 1,
                "quote_volume": price * 1e6,
                "num_trades": np.ones(n, dtype=int) * 100,
                "is_interpolated": False,
                "era": "training",
            }
        )
        result = compute_volume_indicators(df)
        assert (np.diff(result["cvd_approx"].to_numpy()) > 0).all()


class TestVolumeNoNaN:
    """Volume-derived features should not have NaN after warmup."""

    def test_volume_ma_ratio_no_nan(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        # volume_ma_20 uses min_periods=1 so row 0 onward is valid
        assert not result["volume_ma_ratio"].isna().any()

    def test_obv_no_nan(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_volume_indicators(trending_ohlcv)
        assert not result["obv"].isna().any()


# ===========================================================================
# MODULE 3: lag_features — MANDATORY lookahead / sanity tests
# ===========================================================================


class TestLagFeaturesMissingColumns:
    def test_missing_close_raises(self) -> None:
        df = _make_ohlcv(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="missing columns"):
            compute_lag_features(df)


class TestLagFeaturesSameIndex:
    def test_same_index(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        pd.testing.assert_index_equal(result.index, trending_ohlcv.index)

    def test_input_not_mutated(self, trending_ohlcv: pd.DataFrame) -> None:
        cols_before = set(trending_ohlcv.columns)
        compute_lag_features(trending_ohlcv)
        assert set(trending_ohlcv.columns) == cols_before


class TestLogRetLagSanity:
    """MANDATORY: Verify log returns use shift(+N), not shift(-N)."""

    def test_lag_sanity_log_ret_1(self, trending_ohlcv: pd.DataFrame) -> None:
        """log_ret_1[t] must equal log(close[t] / close[t-1]).

        This test is the primary lookahead guard for lag computation.
        If shift(-1) were used instead of shift(+1), expected and actual
        would diverge — this test would catch it immediately.
        """
        result = compute_lag_features(trending_ohlcv)
        close = trending_ohlcv["close"]
        expected = np.log(close / close.shift(1))

        pd.testing.assert_series_equal(
            result["log_ret_1"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_lag_sanity_log_ret_6(self, trending_ohlcv: pd.DataFrame) -> None:
        """log_ret_6[t] must equal log(close[t] / close[t-6])."""
        result = compute_lag_features(trending_ohlcv)
        close = trending_ohlcv["close"]
        expected = np.log(close / close.shift(6))

        pd.testing.assert_series_equal(
            result["log_ret_6"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_lag_sanity_log_ret_24(self, trending_ohlcv: pd.DataFrame) -> None:
        """log_ret_24[t] must equal log(close[t] / close[t-24])."""
        result = compute_lag_features(trending_ohlcv)
        close = trending_ohlcv["close"]
        expected = np.log(close / close.shift(24))

        pd.testing.assert_series_equal(
            result["log_ret_24"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_log_ret_1_first_row_is_nan(self, trending_ohlcv: pd.DataFrame) -> None:
        """The very first log_ret_1 must be NaN (no T-1 data exists)."""
        result = compute_lag_features(trending_ohlcv)
        assert np.isnan(result["log_ret_1"].iloc[0])

    def test_log_ret_168_first_168_rows_are_nan(
        self, trending_ohlcv: pd.DataFrame
    ) -> None:
        """log_ret_168 must be NaN for the first 168 rows."""
        result = compute_lag_features(trending_ohlcv)
        assert result["log_ret_168"].iloc[:168].isna().all()
        # And the 169th row must not be NaN
        assert not np.isnan(result["log_ret_168"].iloc[168])

    def test_no_future_leakage_log_ret_1(self) -> None:
        """Explicit lookahead-bias trap.

        Construct a series where the last value is a known sentinel.
        If any future value leaks into position T, the result at T-1
        will differ from log(close[T-1] / close[T-2]).
        """
        n = 10
        close = np.arange(1.0, n + 1)   # [1, 2, 3, ..., 10]
        df = pd.DataFrame(
            {
                "open_time": _START_MS + np.arange(n) * _INTERVAL_MS,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.ones(n) * 1e6,
                "close_time": _START_MS + np.arange(n) * _INTERVAL_MS + _INTERVAL_MS - 1,
                "quote_volume": close * 1e6,
                "num_trades": np.ones(n, dtype=int) * 100,
                "is_interpolated": False,
                "era": "training",
            }
        )
        result = compute_lag_features(df)
        # At index 4: log(5/4) = log(1.25)
        expected_at_4 = np.log(5.0 / 4.0)
        assert abs(result["log_ret_1"].iloc[4] - expected_at_4) < 1e-12


class TestMomentumFeatures:
    """Momentum features: close[T]/close[T-N] - 1."""

    def test_mom_6_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert "mom_6" in result.columns

    def test_mom_values_match_formula(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        close = trending_ohlcv["close"]
        expected_mom_12 = close / close.shift(12) - 1.0
        pd.testing.assert_series_equal(
            result["mom_12"].rename(None),
            expected_mom_12.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    @pytest.mark.parametrize("n", [6, 12, 24, 48])
    def test_mom_columns_present(self, n: int, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert f"mom_{n}" in result.columns


class TestRollingStats:
    """Rolling volatility, skewness, kurtosis."""

    @pytest.mark.parametrize("w", [6, 12, 24, 48, 168])
    def test_vol_column_present(self, w: int, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert f"vol_{w}" in result.columns

    def test_vol_non_negative(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        for w in [6, 12, 24, 48, 168]:
            col = result[f"vol_{w}"].dropna()
            assert (col >= 0.0).all(), f"vol_{w} has negative values"

    def test_skew_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert "rolling_skew_24" in result.columns

    def test_kurt_present(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert "rolling_kurt_24" in result.columns


class TestHLRange:
    """High-low range must be non-negative."""

    def test_hl_range_non_negative(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert (result["hl_range"] >= 0.0).all()

    def test_hl_range_no_nan(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert not result["hl_range"].isna().any()


class TestLagNoNaNAfterWarmup:
    """After the longest lag warmup (168), core lag features must be NaN-free."""

    _WARMUP = 168

    def test_log_ret_1_no_nan_after_warmup(
        self, trending_ohlcv: pd.DataFrame
    ) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert not result["log_ret_1"].iloc[self._WARMUP:].isna().any()

    def test_mom_48_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        assert not result["mom_48"].iloc[self._WARMUP:].isna().any()

    def test_vol_168_no_nan_after_warmup(self, trending_ohlcv: pd.DataFrame) -> None:
        result = compute_lag_features(trending_ohlcv)
        # vol_168 needs 168 + 1 rows (min_periods=2 but 168-period window)
        assert not result["vol_168"].iloc[self._WARMUP + 1:].isna().any()
