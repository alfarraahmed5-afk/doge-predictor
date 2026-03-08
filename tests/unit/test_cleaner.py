"""Unit tests for src/processing/cleaner.py — DataCleaner."""

from __future__ import annotations

import pandas as pd
import pytest

from src.processing.cleaner import DataCleaner, RemovalRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_MS: int = 1_640_995_200_000  # 2022-01-01 00:00:00 UTC
_INTERVAL_MS: int = 3_600_000      # 1h


def _make_row(
    i: int = 0,
    open_: float = 1.0,
    high: float = 1.02,
    low: float = 0.98,
    close: float = 1.01,
    volume: float = 1_000.0,
) -> dict:
    """Return a single valid OHLCV row dict."""
    t = _BASE_MS + i * _INTERVAL_MS
    return {
        "open_time": t,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": t + _INTERVAL_MS - 1,
        "era": "training",
    }


def _make_df(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _valid_df(n: int = 5) -> pd.DataFrame:
    return _make_df(*[_make_row(i) for i in range(n)])


# ---------------------------------------------------------------------------
# RemovalRecord
# ---------------------------------------------------------------------------


class TestRemovalRecord:
    def test_frozen(self) -> None:
        r = RemovalRecord(open_time=123, symbol="DOGEUSDT", reason="high < low")
        with pytest.raises((AttributeError, TypeError)):
            r.open_time = 999  # type: ignore[misc]

    def test_fields(self) -> None:
        r = RemovalRecord(open_time=_BASE_MS, symbol="BTCUSDT", reason="close <= 0")
        assert r.open_time == _BASE_MS
        assert r.symbol == "BTCUSDT"
        assert r.reason == "close <= 0"


# ---------------------------------------------------------------------------
# DataCleaner — init and state
# ---------------------------------------------------------------------------


class TestDataCleanerInit:
    def test_empty_log_on_init(self) -> None:
        dc = DataCleaner()
        assert dc.get_removal_log() == []

    def test_clear_log_empties_log(self) -> None:
        dc = DataCleaner()
        bad = _make_df(_make_row(high=0.5, low=1.0))  # high < low
        dc.clean_ohlcv(bad, "DOGEUSDT")
        assert len(dc.get_removal_log()) == 1
        dc.clear_log()
        assert dc.get_removal_log() == []


# ---------------------------------------------------------------------------
# DataCleaner — happy path
# ---------------------------------------------------------------------------


class TestCleanOhlcvHappyPath:
    def test_all_valid_rows_unchanged(self) -> None:
        dc = DataCleaner()
        df = _valid_df(10)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 10

    def test_returns_copy_not_view(self) -> None:
        dc = DataCleaner()
        df = _valid_df(5)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        result["open"] = 999.0
        # Original must be unchanged
        assert df["open"].iloc[0] != 999.0

    def test_index_reset_to_zero_based(self) -> None:
        dc = DataCleaner()
        df = _valid_df(10)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert list(result.index) == list(range(10))

    def test_no_removal_log_when_all_valid(self) -> None:
        dc = DataCleaner()
        dc.clean_ohlcv(_valid_df(5), "DOGEUSDT")
        assert dc.get_removal_log() == []

    def test_column_order_preserved(self) -> None:
        dc = DataCleaner()
        df = _valid_df(3)
        original_cols = list(df.columns)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert list(result.columns) == original_cols


# ---------------------------------------------------------------------------
# DataCleaner — missing columns
# ---------------------------------------------------------------------------


class TestCleanOhlcvMissingColumns:
    def test_raises_valueerror_on_missing_column(self) -> None:
        dc = DataCleaner()
        df = _valid_df(3).drop(columns=["high"])
        with pytest.raises(ValueError, match="missing required columns"):
            dc.clean_ohlcv(df, "DOGEUSDT")

    def test_error_message_names_missing_columns(self) -> None:
        dc = DataCleaner()
        df = _valid_df(3).drop(columns=["high", "volume"])
        with pytest.raises(ValueError, match="high") as exc_info:
            dc.clean_ohlcv(df, "DOGEUSDT")
        assert "volume" in str(exc_info.value) or "high" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DataCleaner — individual sanity checks
# ---------------------------------------------------------------------------


class TestSanityChecks:
    def _single_bad_row(self, **kwargs: float) -> pd.DataFrame:
        return _make_df(_make_row(**kwargs))

    def test_high_lt_low_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(high=0.95, low=1.05)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        log = dc.get_removal_log()
        assert len(log) == 1
        assert log[0]["reason"] == "high < low"

    def test_high_lt_open_removed(self) -> None:
        dc = DataCleaner()
        # high=1.0, open=1.05 → high < open; low must stay valid
        df = self._single_bad_row(open_=1.05, high=1.0, low=0.98, close=0.99)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        log = dc.get_removal_log()
        assert log[0]["reason"] == "high < open"

    def test_high_lt_close_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(open_=0.99, high=1.0, low=0.98, close=1.05)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        log = dc.get_removal_log()
        assert log[0]["reason"] == "high < close"

    def test_low_gt_open_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(open_=0.95, high=1.02, low=1.0, close=1.01)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        log = dc.get_removal_log()
        assert log[0]["reason"] == "low > open"

    def test_low_gt_close_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(open_=1.01, high=1.02, low=1.0, close=0.95)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        log = dc.get_removal_log()
        assert log[0]["reason"] == "low > close"

    def test_close_zero_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(open_=0.0, high=0.0, low=0.0, close=0.0)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        assert dc.get_removal_log()[0]["reason"] == "close <= 0"

    def test_close_negative_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(open_=-1.0, high=-0.5, low=-2.0, close=-1.0)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0

    def test_negative_volume_removed(self) -> None:
        dc = DataCleaner()
        df = self._single_bad_row(volume=-1.0)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 0
        assert dc.get_removal_log()[0]["reason"] == "volume < 0"

    def test_zero_volume_is_valid(self) -> None:
        """Zero volume is permitted (illiquid but valid candle)."""
        dc = DataCleaner()
        df = self._single_bad_row(volume=0.0)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# DataCleaner — first-reason priority
# ---------------------------------------------------------------------------


class TestFirstReasonPriority:
    def test_multiple_violations_logged_once_with_first_reason(self) -> None:
        """A row failing high<low AND high<open gets logged only as 'high < low'."""
        dc = DataCleaner()
        # high=0.5, low=1.0 → fails high<low; also high<open if open>0.5
        row = _make_row(open_=0.8, high=0.5, low=1.0, close=0.6)
        df = _make_df(row)
        dc.clean_ohlcv(df, "DOGEUSDT")
        log = dc.get_removal_log()
        assert len(log) == 1
        assert log[0]["reason"] == "high < low"


# ---------------------------------------------------------------------------
# DataCleaner — partial removal
# ---------------------------------------------------------------------------


class TestPartialRemoval:
    def test_only_bad_rows_removed_good_rows_kept(self) -> None:
        dc = DataCleaner()
        rows = [
            _make_row(0),                                       # valid
            _make_row(1, high=0.5, low=1.0),                   # high < low
            _make_row(2),                                       # valid
            _make_row(3, close=0.0, open_=0.0, high=0.0, low=0.0),  # close <= 0
            _make_row(4),                                       # valid
        ]
        df = _make_df(*rows)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert len(result) == 3
        log = dc.get_removal_log()
        assert len(log) == 2

    def test_index_contiguous_after_partial_removal(self) -> None:
        dc = DataCleaner()
        rows = [_make_row(0), _make_row(1, high=0.5, low=1.0), _make_row(2)]
        df = _make_df(*rows)
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert list(result.index) == [0, 1]

    def test_removal_log_records_correct_open_time(self) -> None:
        dc = DataCleaner()
        bad_time = _BASE_MS + 1 * _INTERVAL_MS
        rows = [_make_row(0), _make_row(1, high=0.5, low=1.0)]
        df = _make_df(*rows)
        dc.clean_ohlcv(df, "DOGEUSDT")
        log = dc.get_removal_log()
        assert log[0]["open_time"] == bad_time

    def test_removal_log_records_correct_symbol(self) -> None:
        dc = DataCleaner()
        rows = [_make_row(0, high=0.5, low=1.0)]
        dc.clean_ohlcv(_make_df(*rows), "BTCUSDT")
        log = dc.get_removal_log()
        assert log[0]["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# DataCleaner — accumulation across calls
# ---------------------------------------------------------------------------


class TestAccumulationAcrossCalls:
    def test_log_accumulates_across_multiple_calls(self) -> None:
        dc = DataCleaner()
        bad = _make_df(_make_row(high=0.5, low=1.0))
        dc.clean_ohlcv(bad, "DOGEUSDT")
        dc.clean_ohlcv(bad, "BTCUSDT")
        assert len(dc.get_removal_log()) == 2

    def test_clear_log_resets_between_calls(self) -> None:
        dc = DataCleaner()
        bad = _make_df(_make_row(high=0.5, low=1.0))
        dc.clean_ohlcv(bad, "DOGEUSDT")
        dc.clear_log()
        dc.clean_ohlcv(bad, "BTCUSDT")
        log = dc.get_removal_log()
        assert len(log) == 1
        assert log[0]["symbol"] == "BTCUSDT"


# ---------------------------------------------------------------------------
# DataCleaner — get_removal_log return type
# ---------------------------------------------------------------------------


class TestGetRemovalLog:
    def test_returns_list_of_dicts(self) -> None:
        dc = DataCleaner()
        dc.clean_ohlcv(_make_df(_make_row(high=0.5, low=1.0)), "DOGEUSDT")
        log = dc.get_removal_log()
        assert isinstance(log, list)
        assert isinstance(log[0], dict)

    def test_dict_has_expected_keys(self) -> None:
        dc = DataCleaner()
        dc.clean_ohlcv(_make_df(_make_row(high=0.5, low=1.0)), "DOGEUSDT")
        log = dc.get_removal_log()
        assert set(log[0].keys()) == {"open_time", "symbol", "reason"}


# ---------------------------------------------------------------------------
# DataCleaner — empty DataFrame
# ---------------------------------------------------------------------------


class TestEmptyDataFrame:
    def test_empty_df_returns_empty(self) -> None:
        dc = DataCleaner()
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        result = dc.clean_ohlcv(df, "DOGEUSDT")
        assert result.empty

    def test_empty_df_no_log_entries(self) -> None:
        dc = DataCleaner()
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        dc.clean_ohlcv(df, "DOGEUSDT")
        assert dc.get_removal_log() == []
