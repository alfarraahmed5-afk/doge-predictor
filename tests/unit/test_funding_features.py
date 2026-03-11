"""Unit tests for src/features/funding_features.py.

Per spec requirements:
    - Test forward-fill: all 8 candles in an 8h period have the same funding_rate
    - Test funding_available == 0 for rows before 2020-10-01
    - Test z-score computed on 90-period window
    - Test extreme flag thresholds
    - Test pre-Oct-2020 rows: all funding features = 0.0, funding_available = 0

Test groups:
    TestFundingForwardFill     — forward-fill correctness (8 equal rows per 8h period)
    TestFundingAvailable       — funding_available flag + pre-Oct-2020 behaviour
    TestFundingZScore          — z-score formula and window size
    TestFundingExtremeFlags    — extreme long/short flag thresholds
    TestFundingEdgeCases       — empty funding_df, gaps, missing columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.funding_features import FUNDING_FEATURE_NAMES, compute_funding_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_1H_MS: int = 3_600_000
_8H_MS: int = 8 * _1H_MS

# 2022-01-01 00:00 UTC — training era start
_T0: int = 1_640_995_200_000

# 2020-10-01 00:00 UTC — Binance DOGE-USDT perpetuals launch
_FUNDING_LAUNCH_MS: int = 1_601_510_400_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1h(n: int, start_ms: int = _T0, seed: int = 1) -> pd.DataFrame:
    """Generate a synthetic 1h OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    times = np.array([start_ms + i * _1H_MS for i in range(n)], dtype=np.int64)
    close = 0.10 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
    return pd.DataFrame(
        {
            "open_time": times,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1e6, 5e6, n),
            "symbol": "DOGEUSDT",
            "era": "training",
        }
    )


def _make_funding(n: int, start_ms: int = _T0, seed: int = 4) -> pd.DataFrame:
    """Generate a synthetic 8h funding rate DataFrame."""
    rng = np.random.default_rng(seed)
    times = np.array([start_ms + i * _8H_MS for i in range(n)], dtype=np.int64)
    rates = rng.normal(0.0001, 0.0003, n)
    return pd.DataFrame(
        {
            "timestamp_ms": times,
            "funding_rate": rates,
            "symbol": "DOGEUSDT",
        }
    )


# ---------------------------------------------------------------------------
# TestFundingForwardFill
# ---------------------------------------------------------------------------


class TestFundingForwardFill:
    """Forward-fill correctness: all 8 1h candles in an 8h period share one rate."""

    def test_all_8_candles_have_same_rate(self):
        """SPEC: all 8 candles in an 8h period must have the same funding_rate."""
        # One 8h funding period starting at T0
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0],
                "funding_rate": [0.00042],
                "symbol": "DOGEUSDT",
            }
        )
        # 8 1h candles from T0 to T0+7h (all within the first 8h period)
        df_1h = _make_1h(8, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        rates = result["funding_rate"].to_numpy()
        assert np.all(rates == 0.00042), (
            f"All 8 candles in an 8h period must carry the same funding_rate. "
            f"Got: {rates}"
        )

    def test_rate_updates_at_next_period(self):
        """Rate changes exactly when the next 8h funding timestamp is reached."""
        rate_a = 0.00042
        rate_b = 0.00099
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0, _T0 + _8H_MS],
                "funding_rate": [rate_a, rate_b],
                "symbol": "DOGEUSDT",
            }
        )
        # 16 1h candles covering two full 8h periods
        df_1h = _make_1h(16, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        rates = result["funding_rate"].to_numpy()
        # Rows 0–7 (T0+0h to T0+7h) must carry rate_a
        assert np.all(rates[:8] == rate_a), (
            f"Rows 0–7 must carry rate_a={rate_a}. Got: {rates[:8]}"
        )
        # Rows 8–15 (T0+8h to T0+15h) must carry rate_b
        assert np.all(rates[8:16] == rate_b), (
            f"Rows 8–15 must carry rate_b={rate_b}. Got: {rates[8:16]}"
        )

    def test_no_backward_fill(self):
        """1h candles BEFORE the first funding timestamp must have funding_rate=0.0.

        The forward-fill is strictly forward; there is no backward-fill.
        Pre-funding rows are set to 0.0 and funding_available=0.
        """
        # First funding at T0+8h → rows 0–7 (T0+0h to T0+7h) have no data
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0 + _8H_MS],
                "funding_rate": [0.00050],
                "symbol": "DOGEUSDT",
            }
        )
        df_1h = _make_1h(16, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        # Rows 0–7 must be 0.0 (filled from NaN) and funding_available=0
        for i in range(8):
            assert result["funding_rate"].iloc[i] == 0.0, (
                f"Row {i} must have funding_rate=0.0 (no data, not backward-filled). "
                f"Got {result['funding_rate'].iloc[i]}"
            )
            assert result["funding_available"].iloc[i] == 0, (
                f"Row {i} must have funding_available=0. Got {result['funding_available'].iloc[i]}"
            )

        # Row 8 (T0+8h) must carry the actual rate
        assert result["funding_rate"].iloc[8] == 0.00050
        assert result["funding_available"].iloc[8] == 1

    def test_output_row_count_equals_input(self):
        """Output row count must equal the 1h DataFrame row count."""
        df_1h = _make_1h(48)
        result = compute_funding_features(df_1h, _make_funding(10))
        assert len(result) == len(df_1h)

    def test_output_is_copy(self):
        """compute_funding_features must not modify the input DataFrame in-place."""
        df_1h = _make_1h(20)
        original_cols = set(df_1h.columns)
        compute_funding_features(df_1h, _make_funding(5))
        assert set(df_1h.columns) == original_cols

    def test_all_feature_names_present(self):
        """All five canonical funding feature names must appear in output."""
        result = compute_funding_features(_make_1h(50), _make_funding(20))
        for col in FUNDING_FEATURE_NAMES:
            assert col in result.columns, f"Missing funding feature: {col}"


# ---------------------------------------------------------------------------
# TestFundingAvailable
# ---------------------------------------------------------------------------


class TestFundingAvailable:
    """funding_available flag: 0 before funding data, 1 after."""

    def test_funding_available_zero_before_launch(self):
        """SPEC: funding_available == 0 for rows before 2020-10-01.

        Before the Binance DOGE-USDT perpetual was launched (Oct 2020),
        no funding rate data exists.  These rows must have funding_available=0.
        """
        # 1h data starting well before Oct 2020
        pre_launch_start = _FUNDING_LAUNCH_MS - 10 * _1H_MS
        df_1h = _make_1h(8, start_ms=pre_launch_start)

        # Funding data only starts at the launch date
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_FUNDING_LAUNCH_MS],
                "funding_rate": [0.00010],
                "symbol": "DOGEUSDT",
            }
        )
        result = compute_funding_features(df_1h, funding)

        # The first 10 rows (before launch) must have funding_available=0
        for i in range(min(8, 10)):
            assert result["funding_available"].iloc[i] == 0, (
                f"Row {i} (pre-Oct-2020) must have funding_available=0. "
                f"Got {result['funding_available'].iloc[i]}"
            )

    def test_funding_available_one_after_data_exists(self):
        """funding_available == 1 for rows that received a forward-filled rate."""
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0],
                "funding_rate": [0.00030],
                "symbol": "DOGEUSDT",
            }
        )
        df_1h = _make_1h(8, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        # All 8 rows from T0 onward should have funding_available=1
        assert (result["funding_available"] == 1).all(), (
            f"All rows from T0 must have funding_available=1. "
            f"Got: {result['funding_available'].to_numpy()}"
        )

    def test_pre_launch_funding_rate_is_zero(self):
        """SPEC: all funding features = 0.0 for pre-launch rows."""
        pre_launch_start = _FUNDING_LAUNCH_MS - 5 * _1H_MS
        df_1h = _make_1h(4, start_ms=pre_launch_start)

        # No funding data before the window → all pre-launch rows
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_FUNDING_LAUNCH_MS + 100 * _8H_MS],  # far future
                "funding_rate": [0.00010],
                "symbol": "DOGEUSDT",
            }
        )
        result = compute_funding_features(df_1h, funding)

        assert (result["funding_rate"] == 0.0).all(), (
            "Pre-launch funding_rate must be 0.0"
        )
        assert (result["funding_rate_zscore"] == 0.0).all(), (
            "Pre-launch funding_rate_zscore must be 0.0"
        )
        assert (result["funding_available"] == 0).all(), (
            "Pre-launch funding_available must be 0"
        )
        assert (result["funding_extreme_long"] == 0).all()
        assert (result["funding_extreme_short"] == 0).all()

    def test_empty_funding_returns_all_zero(self):
        """Empty funding_df → all funding features = 0, funding_available = 0."""
        df_1h = _make_1h(10)
        funding = pd.DataFrame(columns=["timestamp_ms", "funding_rate", "symbol"])

        result = compute_funding_features(df_1h, funding)
        assert (result["funding_rate"] == 0.0).all()
        assert (result["funding_available"] == 0).all()
        assert (result["funding_extreme_long"] == 0).all()
        assert (result["funding_extreme_short"] == 0).all()


# ---------------------------------------------------------------------------
# TestFundingZScore
# ---------------------------------------------------------------------------


class TestFundingZScore:
    """Z-score is computed on the 90-period native 8h series."""

    def test_zscore_available_after_warmup(self):
        """SPEC: funding_rate_zscore uses a 90-period window on 8h funding rates.

        After 2+ 8h periods (min_periods=2) a valid z-score must be produced.
        After 90+ periods the z-score is fully warmed up.
        """
        n = 100  # 100 x 8h = 33 days
        funding = _make_funding(n, start_ms=_T0)
        df_1h = _make_1h(n * 8, start_ms=_T0)

        result = compute_funding_features(df_1h, funding)
        zscore = result["funding_rate_zscore"]

        # Rows after the warmup period must have a non-zero z-score
        post_warmup = zscore.iloc[90 * 8 :]  # after 90 * 8h = 720 1h rows
        non_zero = (post_warmup != 0.0).sum()
        assert non_zero > 0, (
            "funding_rate_zscore must be non-zero after 90-period warmup"
        )

    def test_zscore_bounded_for_normal_data(self):
        """Z-scores from random normal funding rates should be in a reasonable range."""
        n = 200
        funding = _make_funding(n, start_ms=_T0, seed=42)
        df_1h = _make_1h(n * 8, start_ms=_T0)

        result = compute_funding_features(df_1h, funding)
        zscore = result["funding_rate_zscore"].to_numpy()

        # Exclude pre-warmup zeros; check that post-warmup z-scores look normal
        post_warmup_zscore = zscore[2 * 8 :]  # skip first 2 8h periods (min_periods=2)
        non_zero = post_warmup_zscore[post_warmup_zscore != 0.0]
        if len(non_zero) > 0:
            assert np.nanmax(np.abs(non_zero)) < 20.0, (
                f"Abnormally large z-score: max abs = {np.nanmax(np.abs(non_zero)):.2f}"
            )

    def test_zscore_is_zero_when_single_funding_observation(self):
        """With only one 8h rate, z-score cannot be computed — result is 0.0."""
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0],
                "funding_rate": [0.00042],
                "symbol": "DOGEUSDT",
            }
        )
        df_1h = _make_1h(8, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        # With only 1 observation rolling(90, min_periods=2) gives NaN → filled to 0
        assert (result["funding_rate_zscore"] == 0.0).all(), (
            "Z-score must be 0.0 when there is only one 8h observation"
        )

    def test_zscore_same_for_all_8_candles_in_period(self):
        """Within each 8h period, all 8 1h candles carry the same z-score value."""
        n = 10
        funding = _make_funding(n, start_ms=_T0)
        df_1h = _make_1h(n * 8, start_ms=_T0)

        result = compute_funding_features(df_1h, funding)
        zscore = result["funding_rate_zscore"].to_numpy()

        for period in range(n):
            group = zscore[period * 8 : (period + 1) * 8]
            if not np.any(np.isnan(group)):
                assert np.all(group == group[0]), (
                    f"Z-score changed within 8h period {period}: {group}"
                )


# ---------------------------------------------------------------------------
# TestFundingExtremeFlags
# ---------------------------------------------------------------------------


class TestFundingExtremeFlags:
    """funding_extreme_long and funding_extreme_short threshold logic."""

    def test_extreme_long_flag_above_threshold(self):
        """funding_extreme_long = 1 when funding_rate >= 0.001 (crowded longs)."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [0.00150], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_long"] == 1).all(), (
            "funding_extreme_long must be 1 when rate=0.00150 > threshold 0.001"
        )

    def test_extreme_long_flag_at_threshold(self):
        """funding_extreme_long = 1 exactly at the 0.001 threshold (>= not >)."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [0.001], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_long"] == 1).all()

    def test_extreme_long_flag_below_threshold(self):
        """funding_extreme_long = 0 when funding_rate < 0.001."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [0.00050], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_long"] == 0).all()

    def test_extreme_short_flag_below_threshold(self):
        """funding_extreme_short = 1 when funding_rate <= -0.0005 (crowded shorts)."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [-0.00080], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_short"] == 1).all(), (
            "funding_extreme_short must be 1 when rate=-0.00080 < threshold -0.0005"
        )

    def test_extreme_short_flag_at_threshold(self):
        """funding_extreme_short = 1 exactly at -0.0005 threshold (<= not <)."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [-0.0005], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_short"] == 1).all()

    def test_extreme_short_flag_above_threshold(self):
        """funding_extreme_short = 0 when funding_rate > -0.0005."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [0.00010], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_short"] == 0).all()

    def test_both_flags_zero_for_normal_rate(self):
        """A normal funding rate (e.g. 0.0001) produces both flags = 0."""
        funding = pd.DataFrame(
            {"timestamp_ms": [_T0], "funding_rate": [0.0001], "symbol": "DOGEUSDT"}
        )
        result = compute_funding_features(_make_1h(8, start_ms=_T0), funding)
        assert (result["funding_extreme_long"] == 0).all()
        assert (result["funding_extreme_short"] == 0).all()


# ---------------------------------------------------------------------------
# TestFundingEdgeCases
# ---------------------------------------------------------------------------


class TestFundingEdgeCases:
    """Edge cases: missing columns, duplicates, large gaps."""

    def test_missing_doge_open_time(self):
        df = _make_1h(10).drop(columns=["open_time"])
        with pytest.raises(ValueError, match="doge_df missing columns"):
            compute_funding_features(df, _make_funding(5))

    def test_missing_funding_timestamp(self):
        df_funding = _make_funding(5).drop(columns=["timestamp_ms"])
        with pytest.raises(ValueError, match="funding_df missing columns"):
            compute_funding_features(_make_1h(10), df_funding)

    def test_missing_funding_rate_column(self):
        df_funding = _make_funding(5).drop(columns=["funding_rate"])
        with pytest.raises(ValueError, match="funding_df missing columns"):
            compute_funding_features(_make_1h(10), df_funding)

    def test_duplicate_funding_timestamps_deduplicated(self):
        """Duplicate 8h timestamps must not cause duplicate rows in output."""
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0, _T0],  # duplicate
                "funding_rate": [0.0003, 0.0005],
                "symbol": "DOGEUSDT",
            }
        )
        df_1h = _make_1h(8, start_ms=_T0)
        # Must not raise; duplicates are dropped before alignment
        result = compute_funding_features(df_1h, funding)
        assert len(result) == len(df_1h)

    def test_gap_in_funding_data_forward_fills(self):
        """A gap in funding data (e.g. 24h gap) is bridged by forward-fill."""
        # Rate at T0 and T0 + 3*8h (skip two 8h periods)
        funding = pd.DataFrame(
            {
                "timestamp_ms": [_T0, _T0 + 3 * _8H_MS],
                "funding_rate": [0.0001, 0.0009],
                "symbol": "DOGEUSDT",
            }
        )
        df_1h = _make_1h(4 * 8, start_ms=_T0)
        result = compute_funding_features(df_1h, funding)

        # Rows 0–23 (T0+0h to T0+23h) should carry 0.0001 (forward-filled)
        assert np.allclose(result["funding_rate"].iloc[:24].to_numpy(), 0.0001)
        # Rows 24–31 (T0+24h to T0+31h) should carry 0.0009
        assert np.allclose(result["funding_rate"].iloc[24:32].to_numpy(), 0.0009)
