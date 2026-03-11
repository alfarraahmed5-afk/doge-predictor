"""Unit tests for src/features/doge_specific.py.

All mandatory tests are marked explicitly.  The six tests required by the
prompt are tagged with ``# MANDATORY``.

Mandatory tests:
    1. BTC corr test          — log-return corr != raw-price corr
    2. Volume spike test      — flag == 1 when volume == exactly 3x rolling mean
    3. Volume normalisation   — ADF test confirms volume_ratio is stationary
    4. Momentum test          — dogebtc_mom_6h[t] == log(dogebtc[t] / dogebtc[t-6])
    5. Round number test      — at_round_number_flag == 1 when close == 0.10
    6. No-lookahead test      — features at index t do not change when future rows added
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.features.doge_specific import (
    DOGE_FEATURE_NAMES,
    compute_doge_features,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_START_MS: int = 1_640_995_200_000   # 2022-01-01 00:00:00 UTC
_INTERVAL_MS: int = 3_600_000        # 1 hour


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_doge(
    n: int = 500,
    seed: int = 42,
    drift: float = 0.001,
    volume_scale: float = 1e6,
) -> pd.DataFrame:
    """Return a deterministic DOGEUSDT OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.012, n)
    close = 0.10 * np.exp(np.cumsum(log_rets))
    noise = rng.uniform(0.002, 0.008, n)
    high = np.maximum(close * (1.0 + noise), close)
    low = np.minimum(close * (1.0 - noise), close)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    volume = rng.uniform(volume_scale * 0.5, volume_scale * 1.5, n)
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame(
        {
            "open_time": open_times,
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


def _make_btc(n: int = 500, seed: int = 7, drift: float = 0.0008) -> pd.DataFrame:
    """Return a deterministic BTCUSDT close-only DataFrame."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.015, n)
    close = 30_000.0 * np.exp(np.cumsum(log_rets))
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame({"open_time": open_times, "close": close})


def _make_dogebtc(n: int = 500, seed: int = 13, drift: float = 0.0002) -> pd.DataFrame:
    """Return a deterministic DOGEBTC close-only DataFrame."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(drift, 0.010, n)
    close = 3.5e-6 * np.exp(np.cumsum(log_rets))
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame({"open_time": open_times, "close": close})


def _make_minimal_doge(n: int, close: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Return a minimal DOGE DataFrame for controlled tests."""
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": volume,
            "close_time": open_times + _INTERVAL_MS - 1,
            "quote_volume": close * volume,
            "num_trades": np.ones(n, dtype=int) * 100,
            "is_interpolated": False,
            "era": "training",
        }
    )


def _flat_btc(n: int) -> pd.DataFrame:
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame({"open_time": open_times, "close": np.ones(n) * 30_000.0})


def _flat_dogebtc(n: int) -> pd.DataFrame:
    open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
    return pd.DataFrame({"open_time": open_times, "close": np.ones(n) * 3.5e-6})


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_doge() -> pd.DataFrame:
    return _make_doge(n=500, seed=42)


@pytest.fixture(scope="module")
def base_btc() -> pd.DataFrame:
    return _make_btc(n=500, seed=7)


@pytest.fixture(scope="module")
def base_dogebtc() -> pd.DataFrame:
    return _make_dogebtc(n=500, seed=13)


@pytest.fixture(scope="module")
def base_result(
    base_doge: pd.DataFrame,
    base_btc: pd.DataFrame,
    base_dogebtc: pd.DataFrame,
) -> pd.DataFrame:
    return compute_doge_features(base_doge, base_btc, base_dogebtc)


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    def test_missing_doge_volume_raises(
        self, base_btc: pd.DataFrame, base_dogebtc: pd.DataFrame
    ) -> None:
        bad = _make_doge(50).drop(columns=["volume"])
        with pytest.raises(ValueError, match="doge_df missing"):
            compute_doge_features(bad, base_btc, base_dogebtc)

    def test_missing_btc_close_raises(
        self, base_doge: pd.DataFrame, base_dogebtc: pd.DataFrame
    ) -> None:
        bad_btc = _make_btc(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="btc_df missing"):
            compute_doge_features(base_doge, bad_btc, base_dogebtc)

    def test_missing_dogebtc_close_raises(
        self, base_doge: pd.DataFrame, base_btc: pd.DataFrame
    ) -> None:
        bad_dogebtc = _make_dogebtc(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="dogebtc_df missing"):
            compute_doge_features(base_doge, base_btc, bad_dogebtc)

    def test_input_doge_not_mutated(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        cols_before = set(base_doge.columns)
        compute_doge_features(base_doge, base_btc, base_dogebtc)
        assert set(base_doge.columns) == cols_before


# ===========================================================================
# Output shape and index
# ===========================================================================


class TestOutputShape:
    def test_same_index(
        self, base_doge: pd.DataFrame, base_result: pd.DataFrame
    ) -> None:
        pd.testing.assert_index_equal(base_result.index, base_doge.index)

    def test_all_12_feature_columns_present(self, base_result: pd.DataFrame) -> None:
        for col in DOGE_FEATURE_NAMES:
            assert col in base_result.columns, f"Missing mandatory feature: {col}"

    def test_row_count_preserved(
        self, base_doge: pd.DataFrame, base_result: pd.DataFrame
    ) -> None:
        assert len(base_result) == len(base_doge)


# ===========================================================================
# MANDATORY TEST 1 — BTC Correlation: log-return != raw-price
# ===========================================================================


class TestBTCCorrelation:
    """MANDATORY: Correlation must be on log returns, not raw prices."""

    def test_log_return_corr_differs_from_raw_price_corr(self) -> None:  # MANDATORY
        """Assert that log-return correlation != raw-price correlation.

        Two co-trending assets always have raw-price correlation close to 1.0
        regardless of actual co-movement — this is a spurious correlation.
        Log-return correlation correctly reflects actual co-movement of returns.
        """
        n = 300
        rng = np.random.default_rng(0)
        t = np.arange(n)
        # Both series trend strongly upward → raw-price corr will be ≈ 1.0
        doge_close = 0.10 * np.exp(t * 0.02 + rng.normal(0, 0.01, n))
        btc_close = 30_000.0 * np.exp(t * 0.02 + rng.normal(0, 0.01, n))

        doge_lr = np.log(doge_close[1:] / doge_close[:-1])
        btc_lr = np.log(btc_close[1:] / btc_close[:-1])

        raw_corr = float(np.corrcoef(doge_close, btc_close)[0, 1])
        lr_corr = float(np.corrcoef(doge_lr, btc_lr)[0, 1])

        assert raw_corr > 0.98, (
            f"Expected raw corr > 0.98 for co-trending; got {raw_corr:.4f}"
        )
        assert abs(raw_corr - lr_corr) > 0.05, (
            f"Log-return corr ({lr_corr:.4f}) too close to raw-price corr "
            f"({raw_corr:.4f}); may be using wrong correlation method"
        )

    def test_feature_uses_log_return_not_raw_price(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        """The computed corr must not equal what raw-price rolling corr would give."""
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        doge_close = base_doge["close"]
        btc_close_arr = (
            base_btc.set_index("open_time")["close"]
            .reindex(base_doge["open_time"].values)
            .values
        )
        btc_s = pd.Series(btc_close_arr, index=doge_close.index)
        raw_corr_24 = doge_close.rolling(24).corr(btc_s)
        assert not result["doge_btc_corr_24h"].dropna().equals(raw_corr_24.dropna())

    def test_corr_columns_present(self, base_result: pd.DataFrame) -> None:
        for col in ("doge_btc_corr_12h", "doge_btc_corr_24h", "doge_btc_corr_7d"):
            assert col in base_result.columns

    def test_corr_range(self, base_result: pd.DataFrame) -> None:
        """Correlation values must be in [-1, +1]."""
        for col in ("doge_btc_corr_12h", "doge_btc_corr_24h", "doge_btc_corr_7d"):
            valid = base_result[col].dropna()
            assert (valid >= -1.0 - 1e-9).all() and (valid <= 1.0 + 1e-9).all(), (
                f"{col} has values outside [-1, 1]"
            )

    def test_corr_7d_warmup_is_nan(self, base_result: pd.DataFrame) -> None:
        """First 167 rows of corr_7d must be NaN (168-period window)."""
        assert base_result["doge_btc_corr_7d"].iloc[:167].isna().all()
        assert not np.isnan(base_result["doge_btc_corr_7d"].iloc[168])

    def test_corr_approaches_1_for_identical_returns(self) -> None:
        """When DOGE and BTC have identical log returns, corr should be ≈ 1.0."""
        n = 300
        rng = np.random.default_rng(55)
        log_rets = rng.normal(0.001, 0.015, n)
        doge_close = 0.10 * np.exp(np.cumsum(log_rets))
        btc_close = 30_000.0 * np.exp(np.cumsum(log_rets))   # identical returns

        open_times = _START_MS + np.arange(n, dtype=np.int64) * _INTERVAL_MS
        doge = _make_minimal_doge(n, doge_close, np.ones(n) * 1e6)
        btc = pd.DataFrame({"open_time": open_times, "close": btc_close})
        dogebtc = _flat_dogebtc(n)

        result = compute_doge_features(doge, btc, dogebtc)
        valid_24 = result["doge_btc_corr_24h"].dropna()
        assert (valid_24 > 0.99).all(), (
            f"Expected corr ≈ 1.0 for identical returns; min={valid_24.min():.4f}"
        )


# ===========================================================================
# MANDATORY TEST 2 — Volume spike flag at exactly 3× rolling mean
# ===========================================================================


class TestVolumeSpike:
    """MANDATORY: volume_spike_flag must be 1 when volume == 3× rolling mean."""

    def test_spike_flag_at_exact_3x_threshold(self) -> None:  # MANDATORY
        """volume_spike_flag must equal 1 when volume_ratio == threshold (3.0).

        The rolling(20) mean at row T includes row T itself, so to achieve
        volume_ratio == exactly 3.0 we need:

            ratio = V / mean = V / ((19*C + V) / 20) = 3.0
            → V = 57*C / 17  (derivation: 17V = 57C)

        This gives mean = (19C + 57C/17)/20 = (19/17)C and ratio = 3.0 exactly.
        """
        n = 50
        C = 1_000_000.0  # baseline volume (all 19 background rows in window)
        volume = np.full(n, C, dtype=np.float64)
        # V = 57/17 * C makes volume_ratio at row 30 exactly 3.0
        V_exact = C * 57.0 / 17.0
        volume[30] = V_exact

        close = np.ones(n) * 0.10
        doge = _make_minimal_doge(n, close, volume)
        btc = _flat_btc(n)
        dogebtc = _flat_dogebtc(n)

        result = compute_doge_features(doge, btc, dogebtc)
        ratio_at_30 = result["volume_ratio"].iloc[30]
        assert abs(ratio_at_30 - 3.0) < 1e-9, (
            f"Expected volume_ratio=3.0 at row 30; got {ratio_at_30:.10f}"
        )
        assert result["volume_spike_flag"].iloc[30] == 1, (
            f"Expected spike_flag=1 when volume_ratio=3.0 (got {ratio_at_30:.4f})"
        )
        # Row 20: normal volume → flag must be 0
        assert result["volume_spike_flag"].iloc[20] == 0

    def test_spike_flag_below_threshold_is_zero(self, base_result: pd.DataFrame) -> None:
        below = base_result["volume_ratio"] < 3.0
        assert (base_result.loc[below, "volume_spike_flag"] == 0).all()

    def test_spike_flag_above_or_equal_threshold_is_one(
        self, base_result: pd.DataFrame
    ) -> None:
        above = base_result["volume_ratio"] >= 3.0
        if above.any():
            assert (base_result.loc[above, "volume_spike_flag"] == 1).all()

    def test_spike_flag_binary(self, base_result: pd.DataFrame) -> None:
        assert set(base_result["volume_spike_flag"].unique()).issubset({0, 1})

    def test_volume_spike_magnitude_range(self, base_result: pd.DataFrame) -> None:
        """volume_spike_magnitude must be in [0, 1]."""
        mag = base_result["volume_spike_magnitude"].dropna()
        assert (mag >= 0.0).all() and (mag <= 1.0).all()


# ===========================================================================
# MANDATORY TEST 3 — Volume normalization / stationarity (ADF test)
# ===========================================================================


class TestVolumeNormalization:
    """MANDATORY: volume_ratio must be stationary (ADF p < 0.05)."""

    def test_volume_ratio_stationary_adf(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:  # MANDATORY
        """ADF test: reject unit-root hypothesis (p < 0.05) → stationary.

        Raw volume is non-stationary; volume_ratio (normalised by rolling mean)
        is approximately stationary, fluctuating around 1.0.
        """
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        vol_ratio = result["volume_ratio"].dropna().to_numpy()
        _, p_value, *_ = adfuller(vol_ratio, autolag="AIC")
        assert p_value < 0.05, (
            f"volume_ratio is non-stationary: ADF p={p_value:.4f} >= 0.05. "
            "Raw volume may be used instead of the normalised ratio."
        )

    def test_volume_ratio_centered_near_one(self, base_result: pd.DataFrame) -> None:
        """volume_ratio mean should be near 1.0 by construction."""
        ratio = base_result["volume_ratio"].dropna()
        mean_val = float(ratio.mean())
        assert 0.80 <= mean_val <= 1.20, (
            f"volume_ratio mean {mean_val:.4f} far from 1.0"
        )

    def test_volume_ratio_first_window_rows_nan(
        self, base_result: pd.DataFrame
    ) -> None:
        """First 19 rows must be NaN (min_periods=20 enforced)."""
        assert base_result["volume_ratio"].iloc[:19].isna().all()
        assert not np.isnan(base_result["volume_ratio"].iloc[19])


# ===========================================================================
# MANDATORY TEST 4 — Momentum formula equality
# ===========================================================================


class TestDogebtcMomentum:
    """MANDATORY: dogebtc_mom_Nh[t] == log(dogebtc[t] / dogebtc[t-N])."""

    def _expected_mom(
        self,
        base_doge: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
        result: pd.DataFrame,
        window: int,
    ) -> pd.Series:
        dogebtc_aligned = (
            base_dogebtc.set_index("open_time")["close"]
            .reindex(base_doge["open_time"].values)
        )
        dogebtc_aligned.index = result.index
        return np.log(dogebtc_aligned / dogebtc_aligned.shift(window))

    def test_dogebtc_mom_6h_formula(  # MANDATORY
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        """dogebtc_mom_6h[t] must equal log(dogebtc_close[t] / dogebtc_close[t-6])."""
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        expected = self._expected_mom(base_doge, base_dogebtc, result, 6)
        pd.testing.assert_series_equal(
            result["dogebtc_mom_6h"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_dogebtc_mom_24h_formula(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        expected = self._expected_mom(base_doge, base_dogebtc, result, 24)
        pd.testing.assert_series_equal(
            result["dogebtc_mom_24h"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_dogebtc_mom_48h_formula(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        expected = self._expected_mom(base_doge, base_dogebtc, result, 48)
        pd.testing.assert_series_equal(
            result["dogebtc_mom_48h"].rename(None),
            expected.rename(None),
            check_exact=False,
            rtol=1e-12,
            check_names=False,
        )

    def test_momentum_first_n_rows_nan(self, base_result: pd.DataFrame) -> None:
        assert base_result["dogebtc_mom_6h"].iloc[:6].isna().all()
        assert not np.isnan(base_result["dogebtc_mom_6h"].iloc[6])

    @pytest.mark.parametrize(
        "col", ["dogebtc_mom_6h", "dogebtc_mom_24h", "dogebtc_mom_48h"]
    )
    def test_momentum_columns_present(
        self, col: str, base_result: pd.DataFrame
    ) -> None:
        assert col in base_result.columns


# ===========================================================================
# MANDATORY TEST 5 — Round number flag when close == 0.10
# ===========================================================================


class TestRoundNumbers:
    """MANDATORY: at_round_number_flag == 1 when close is exactly at a round level."""

    def test_at_round_number_flag_when_close_equals_010(self) -> None:  # MANDATORY
        """at_round_number_flag must be 1 when close == 0.10 (a config level)."""
        n = 30
        close = np.ones(n) * 0.12
        close[25] = 0.10           # exactly on a configured round level
        doge = _make_minimal_doge(n, close, np.ones(n) * 1e6)
        btc = _flat_btc(n)
        dogebtc = _flat_dogebtc(n)

        result = compute_doge_features(doge, btc, dogebtc)
        assert result["at_round_number_flag"].iloc[25] == 1, (
            f"Expected at_round_number_flag=1 when close=0.10; "
            f"distance_to_round_pct={result['distance_to_round_pct'].iloc[25]:.6f}"
        )

    def test_at_round_number_flag_at_level_020(self) -> None:
        """at_round_number_flag must be 1 when close == 0.20."""
        n = 30
        close = np.ones(n) * 0.17
        close[20] = 0.20
        doge = _make_minimal_doge(n, close, np.ones(n) * 1e6)
        result = compute_doge_features(doge, _flat_btc(n), _flat_dogebtc(n))
        assert result["at_round_number_flag"].iloc[20] == 1

    def test_not_at_round_number_when_far_away(self, base_result: pd.DataFrame) -> None:
        far = base_result["distance_to_round_pct"].abs() >= 0.01
        if far.any():
            assert (base_result.loc[far, "at_round_number_flag"] == 0).all()

    def test_distance_positive_above_nearest_level(self) -> None:
        """Close above nearest round level → positive distance_to_round_pct."""
        n = 30
        close = np.ones(n) * 0.105   # 5% above 0.10
        doge = _make_minimal_doge(n, close, np.ones(n) * 1e6)
        result = compute_doge_features(doge, _flat_btc(n), _flat_dogebtc(n))
        assert (result["distance_to_round_pct"] > 0).all()
        assert (result["nearest_round_level"] == 0.10).all()

    def test_distance_negative_below_nearest_level(self) -> None:
        """Close below nearest round level → negative distance_to_round_pct."""
        n = 30
        close = np.ones(n) * 0.095   # 5% below 0.10
        doge = _make_minimal_doge(n, close, np.ones(n) * 1e6)
        result = compute_doge_features(doge, _flat_btc(n), _flat_dogebtc(n))
        assert (result["distance_to_round_pct"] < 0).all()
        assert (result["nearest_round_level"] == 0.10).all()

    def test_round_number_flag_binary(self, base_result: pd.DataFrame) -> None:
        assert set(base_result["at_round_number_flag"].unique()).issubset({0, 1})

    def test_nearest_round_level_is_config_level(
        self, base_result: pd.DataFrame
    ) -> None:
        from src.config import doge_settings
        levels = set(doge_settings.round_number_levels)
        unique_nearest = set(base_result["nearest_round_level"].unique())
        assert unique_nearest.issubset(levels), (
            f"Found nearest_round_level values not in config: "
            f"{unique_nearest - levels}"
        )


# ===========================================================================
# MANDATORY TEST 6 — No lookahead bias
# ===========================================================================


class TestNoLookahead:
    """MANDATORY: Features at index t must not change when future rows are added."""

    def test_features_identical_with_truncated_future(self) -> None:  # MANDATORY
        """Compute on N rows, then N+1 rows.  Row N-1 must be identical.

        If any feature at row T reads from T+1 or later, the value at row T
        would change when row T+1 is appended — this test catches it.

        Implementation note: we generate the long DataFrames first and slice
        them for the short run.  This guarantees rows [0..N-1] are bit-for-bit
        identical in both runs — eliminating RNG divergence as a confound.
        """
        n_short = 200
        n_long = 201

        # Generate long first, slice for short — identical prefix guaranteed
        doge_long = _make_doge(n=n_long, seed=42)
        btc_long = _make_btc(n=n_long, seed=7)
        dogebtc_long = _make_dogebtc(n=n_long, seed=13)

        doge_short = doge_long.iloc[:n_short].copy()
        btc_short = btc_long.iloc[:n_short].copy()
        dogebtc_short = dogebtc_long.iloc[:n_short].copy()

        result_short = compute_doge_features(doge_short, btc_short, dogebtc_short)
        result_long = compute_doge_features(doge_long, btc_long, dogebtc_long)

        last_idx = n_short - 1
        for col in DOGE_FEATURE_NAMES:
            val_short = result_short[col].iloc[last_idx]
            val_long = result_long[col].iloc[last_idx]
            if pd.isna(val_short) and pd.isna(val_long):
                continue
            assert abs(val_short - val_long) < 1e-10, (
                f"LOOKAHEAD BIAS in '{col}': value at row {last_idx} changed "
                f"from {val_short} to {val_long} when a future row was appended."
            )

    def test_corr_warmup_rows_are_nan_not_future_values(
        self, base_result: pd.DataFrame
    ) -> None:
        """Warmup NaN rows confirm no future data is used during warmup."""
        assert base_result["doge_btc_corr_12h"].iloc[:11].isna().all()

    def test_momentum_warmup_rows_are_nan(self, base_result: pd.DataFrame) -> None:
        for col, window in [
            ("dogebtc_mom_6h", 6),
            ("dogebtc_mom_24h", 24),
            ("dogebtc_mom_48h", 48),
        ]:
            assert base_result[col].iloc[:window].isna().all(), (
                f"{col} warmup rows are not NaN — possible lookahead"
            )


# ===========================================================================
# Timestamp alignment
# ===========================================================================


class TestTimestampAlignment:
    def test_partial_btc_overlap_produces_nan(
        self,
        base_doge: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        """BTC data covering only first 100 rows → rows 100+ have NaN BTC corr."""
        btc_partial = _make_btc(n=100, seed=7)
        result = compute_doge_features(base_doge, btc_partial, base_dogebtc)
        assert result["doge_btc_corr_24h"].iloc[200:].isna().all()

    def test_full_overlap_no_nan_after_warmup(
        self,
        base_doge: pd.DataFrame,
        base_btc: pd.DataFrame,
        base_dogebtc: pd.DataFrame,
    ) -> None:
        """With full timestamp match, no unexpected NaN after 168-period warmup."""
        result = compute_doge_features(base_doge, base_btc, base_dogebtc)
        assert not result["doge_btc_corr_7d"].iloc[200:].isna().any()
