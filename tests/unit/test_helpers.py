"""Unit tests for src/utils/helpers.py.

Tests every public function across normal paths, edge cases, and error paths.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from src.utils.helpers import (
    compute_expected_row_count,
    datetime_to_ms,
    interval_to_ms,
    ms_to_datetime,
    safe_divide,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

_JAN_1_2022_MS: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC
_JAN_1_2022_DT: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)
_HOUR_MS: int = 3_600_000


# ===========================================================================
# ms_to_datetime
# ===========================================================================


class TestMsToDatetime:
    """Tests for :func:`ms_to_datetime`."""

    def test_known_epoch(self) -> None:
        """ms=0 maps to Unix epoch 1970-01-01 00:00:00 UTC."""
        dt = ms_to_datetime(0)
        assert dt == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_known_date_2022(self) -> None:
        """Converts 2022-01-01 epoch ms to the correct datetime."""
        dt = ms_to_datetime(_JAN_1_2022_MS)
        assert dt.year == 2022
        assert dt.month == 1
        assert dt.day == 1
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_returns_utc_aware(self) -> None:
        """Result is always UTC-aware."""
        dt = ms_to_datetime(_JAN_1_2022_MS)
        assert dt.tzinfo is not None

    def test_millisecond_precision(self) -> None:
        """Sub-second precision is preserved (500ms offset)."""
        dt = ms_to_datetime(_JAN_1_2022_MS + 500)
        assert dt.microsecond == 500_000

    def test_type_error_float(self) -> None:
        """Raises TypeError when given a float."""
        with pytest.raises(TypeError, match="int"):
            ms_to_datetime(1_641_024_000_000.0)  # type: ignore[arg-type]

    def test_type_error_string(self) -> None:
        """Raises TypeError when given a string."""
        with pytest.raises(TypeError, match="int"):
            ms_to_datetime("2022-01-01")  # type: ignore[arg-type]

    def test_value_error_negative(self) -> None:
        """Raises ValueError for negative ms."""
        with pytest.raises(ValueError, match=">= 0"):
            ms_to_datetime(-1)

    def test_zero_is_valid(self) -> None:
        """ms=0 is valid (Unix epoch)."""
        dt = ms_to_datetime(0)
        assert dt.year == 1970


# ===========================================================================
# datetime_to_ms
# ===========================================================================


class TestDatetimeToMs:
    """Tests for :func:`datetime_to_ms`."""

    def test_known_date_2022(self) -> None:
        """Converts 2022-01-01 UTC to the correct epoch ms."""
        ms = datetime_to_ms(_JAN_1_2022_DT)
        assert ms == _JAN_1_2022_MS

    def test_epoch_zero(self) -> None:
        """Unix epoch returns 0 ms."""
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        assert datetime_to_ms(dt) == 0

    def test_returns_int(self) -> None:
        """Result is always an int."""
        ms = datetime_to_ms(_JAN_1_2022_DT)
        assert isinstance(ms, int)

    def test_roundtrip(self) -> None:
        """ms -> datetime -> ms is lossless for second-granular timestamps."""
        original_ms = _JAN_1_2022_MS
        dt = ms_to_datetime(original_ms)
        recovered_ms = datetime_to_ms(dt)
        assert recovered_ms == original_ms

    def test_type_error_string(self) -> None:
        """Raises TypeError when given a string."""
        with pytest.raises(TypeError, match="datetime"):
            datetime_to_ms("2022-01-01")  # type: ignore[arg-type]

    def test_type_error_int(self) -> None:
        """Raises TypeError when given an int."""
        with pytest.raises(TypeError, match="datetime"):
            datetime_to_ms(1_641_024_000_000)  # type: ignore[arg-type]

    def test_value_error_naive(self) -> None:
        """Raises ValueError for timezone-naive datetime."""
        naive_dt = datetime(2022, 1, 1)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            datetime_to_ms(naive_dt)


# ===========================================================================
# interval_to_ms
# ===========================================================================


class TestIntervalToMs:
    """Tests for :func:`interval_to_ms`."""

    @pytest.mark.parametrize(
        "interval, expected_ms",
        [
            ("1m", 60_000),
            ("3m", 180_000),
            ("5m", 300_000),
            ("15m", 900_000),
            ("30m", 1_800_000),
            ("1h", 3_600_000),
            ("2h", 7_200_000),
            ("4h", 14_400_000),
            ("6h", 21_600_000),
            ("8h", 28_800_000),
            ("12h", 43_200_000),
            ("1d", 86_400_000),
            ("3d", 259_200_000),
            ("1w", 604_800_000),
            ("1M", 2_592_000_000),
        ],
    )
    def test_all_valid_intervals(self, interval: str, expected_ms: int) -> None:
        """All recognised Binance interval strings return correct ms."""
        assert interval_to_ms(interval) == expected_ms

    def test_invalid_interval(self) -> None:
        """Raises ValueError for an unrecognised interval string."""
        with pytest.raises(ValueError, match="Unknown interval"):
            interval_to_ms("2d")

    def test_invalid_interval_mentions_valid(self) -> None:
        """Error message lists valid options."""
        with pytest.raises(ValueError, match="1h"):
            interval_to_ms("bad")

    def test_case_sensitive(self) -> None:
        """Interval strings are case-sensitive ('1H' is not '1h')."""
        with pytest.raises(ValueError):
            interval_to_ms("1H")


# ===========================================================================
# compute_expected_row_count
# ===========================================================================


class TestComputeExpectedRowCount:
    """Tests for :func:`compute_expected_row_count`."""

    def test_one_hour_one_candle(self) -> None:
        """One 1h interval = 1 candle."""
        assert compute_expected_row_count(0, _HOUR_MS, _HOUR_MS) == 1

    def test_twenty_four_hours(self) -> None:
        """24 hours / 1h interval = 24 candles."""
        assert compute_expected_row_count(0, 24 * _HOUR_MS, _HOUR_MS) == 24

    def test_exact_multiple(self) -> None:
        """Exact multiple: no ceiling needed."""
        assert compute_expected_row_count(0, 10 * _HOUR_MS, _HOUR_MS) == 10

    def test_ceiling_on_partial(self) -> None:
        """Partial interval rounds up via ceil."""
        # 1.5 hours / 1h interval -> ceil(1.5) = 2
        assert compute_expected_row_count(0, int(1.5 * _HOUR_MS), _HOUR_MS) == 2

    def test_start_equals_end_raises(self) -> None:
        """Raises ValueError when start_ms == end_ms."""
        with pytest.raises(ValueError, match="strictly less than"):
            compute_expected_row_count(100, 100, _HOUR_MS)

    def test_start_after_end_raises(self) -> None:
        """Raises ValueError when start_ms > end_ms."""
        with pytest.raises(ValueError, match="strictly less than"):
            compute_expected_row_count(200, 100, _HOUR_MS)

    def test_zero_interval_raises(self) -> None:
        """Raises ValueError when interval_ms == 0."""
        with pytest.raises(ValueError, match="interval_ms must be > 0"):
            compute_expected_row_count(0, _HOUR_MS, 0)

    def test_negative_interval_raises(self) -> None:
        """Raises ValueError when interval_ms < 0."""
        with pytest.raises(ValueError, match="interval_ms must be > 0"):
            compute_expected_row_count(0, _HOUR_MS, -1)

    def test_real_timestamps(self) -> None:
        """Works with real UTC epoch-ms timestamps."""
        # 7 days = 168 hours
        seven_days_ms = 7 * 24 * _HOUR_MS
        result = compute_expected_row_count(
            _JAN_1_2022_MS,
            _JAN_1_2022_MS + seven_days_ms,
            _HOUR_MS,
        )
        assert result == 168


# ===========================================================================
# safe_divide
# ===========================================================================


class TestSafeDivide:
    """Tests for :func:`safe_divide`."""

    def test_normal_division(self) -> None:
        """Normal case: 10 / 2 = 5.0."""
        assert safe_divide(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_fallback(self) -> None:
        """Returns fallback (0.0 by default) when denominator is zero."""
        assert safe_divide(10.0, 0.0) == 0.0

    def test_custom_fallback(self) -> None:
        """Returns custom fallback value on zero denominator."""
        assert safe_divide(10.0, 0.0, fallback=-1.0) == -1.0

    def test_nan_denominator_returns_fallback(self) -> None:
        """Returns fallback when denominator is NaN."""
        result = safe_divide(10.0, float("nan"))
        assert result == 0.0

    def test_nan_denominator_custom_fallback(self) -> None:
        """Returns custom fallback when denominator is NaN."""
        result = safe_divide(5.0, float("nan"), fallback=99.0)
        assert result == 99.0

    def test_numerator_zero(self) -> None:
        """0 / x = 0.0 for non-zero x."""
        assert safe_divide(0.0, 5.0) == 0.0

    def test_negative_values(self) -> None:
        """-10 / 2 = -5.0."""
        assert safe_divide(-10.0, 2.0) == pytest.approx(-5.0)

    def test_fractional_result(self) -> None:
        """1 / 3 is approximately 0.333..."""
        assert safe_divide(1.0, 3.0) == pytest.approx(1.0 / 3.0)

    def test_inf_numerator(self) -> None:
        """Inf / 2.0 = Inf (not affected by safe_divide)."""
        result = safe_divide(float("inf"), 2.0)
        assert math.isinf(result)

    def test_both_zero(self) -> None:
        """0 / 0 returns fallback."""
        assert safe_divide(0.0, 0.0) == 0.0
