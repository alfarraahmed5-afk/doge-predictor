"""Unit tests for src/processing/validator.py — DataValidator.

Each of the 9 validate_ohlcv checks is tested individually, plus tests for
validate_funding_rates and validate_feature_matrix.  No network traffic or
real database is used.

Test coverage:
    Check 1  — Missing required column returns error immediately
    Check 2  — Non-monotonic open_time raises DataValidationError (CRITICAL)
    Check 3  — Gap > 3 candles raises DataValidationError (CRITICAL)
    Check 3  — Gap <= 3 candles is a warning (not a raise)
    Check 4  — OHLCV sanity violations (high < low, close <= 0, volume < 0) logged
    Check 5  — NaN in OHLCV raises DataValidationError (CRITICAL)
    Check 5  — Inf in OHLCV raises DataValidationError (CRITICAL)
    Check 6  — Duplicate open_time logged as warning
    Check 7  — Row count deviation > 2 logged as warning
    Check 8  — Pre-2022 row with era='training' logged as warning
    Check 8  — Post-2022 row with era='context' logged as warning
    Check 9  — Stale data logged when is_live_check=True
    Check 9  — Stale check skipped when is_live_check=False
    All good — valid DataFrame returns is_valid=True, empty errors
    validate_funding_rates — NaN raises DataValidationError
    validate_funding_rates — non-8h interval logged
    validate_funding_rates — out-of-range rate logged
    validate_funding_rates — valid DataFrame passes
    validate_feature_matrix — missing column raises FeatureSchemaError
    validate_feature_matrix — NaN raises DataValidationError
    validate_feature_matrix — constant column logged as warning
    validate_feature_matrix — non-monotonic index logged as warning
    validate_feature_matrix — empty DataFrame returns is_valid=False without raise
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.ingestion.exceptions import DataValidationError
from src.processing.schemas import CandleValidationResult
from src.processing.validator import DataValidator, FeatureSchemaError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 2022-01-01 00:00:00 UTC — era boundary.
_TRAINING_START_MS: int = 1_640_995_200_000

#: 1-hour interval in milliseconds.
_INTERVAL_MS: int = 3_600_000

#: A post-2022 starting timestamp used by most tests.
_START_MS: int = _TRAINING_START_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int,
    start_ms: int = _START_MS,
    interval_ms: int = _INTERVAL_MS,
    era: str = "training",
) -> pd.DataFrame:
    """Build a valid OHLCV DataFrame with *n* consecutive candles.

    Args:
        n: Number of rows.
        start_ms: ``open_time`` of the first row (UTC epoch ms).
        interval_ms: Milliseconds per candle.
        era: Era label for all rows.

    Returns:
        DataFrame satisfying all OHLCVSchema invariants.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n):
        open_t = start_ms + i * interval_ms
        rows.append({
            "open_time": open_t,
            "open": 0.10,
            "high": 0.11,
            "low": 0.09,
            "close": 0.105,
            "volume": 1_000_000.0,
            "close_time": open_t + interval_ms - 1,
            "era": era,
        })
    return pd.DataFrame(rows)


def _make_funding(
    n: int,
    start_ms: int = _START_MS,
    interval_ms: int = 28_800_000,
    rate: float = 0.0001,
) -> pd.DataFrame:
    """Build a valid funding-rate DataFrame with *n* consecutive rows.

    Args:
        n: Number of rows.
        start_ms: ``funding_time`` of the first row (UTC epoch ms).
        interval_ms: Milliseconds per interval (default 8h = 28 800 000).
        rate: Funding rate value for all rows.

    Returns:
        DataFrame with ``funding_time`` and ``funding_rate`` columns.
    """
    return pd.DataFrame({
        "funding_time": [start_ms + i * interval_ms for i in range(n)],
        "funding_rate": [rate] * n,
    })


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def validator() -> DataValidator:
    """Return a fresh DataValidator instance.

    Returns:
        :class:`~src.processing.validator.DataValidator`.
    """
    return DataValidator()


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 1 (missing columns)
# ---------------------------------------------------------------------------


def test_check1_missing_required_column(validator: DataValidator) -> None:
    """Check 1: missing a required column returns is_valid=False immediately."""
    df = _make_ohlcv(5).drop(columns=["volume"])
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not result.is_valid
    assert any("Missing required columns" in e for e in result.errors)
    assert result.row_count == 5


def test_check1_all_required_columns_present(validator: DataValidator) -> None:
    """Check 1: all required columns present — no column error produced."""
    df = _make_ohlcv(5)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not any("Missing required columns" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 2 (monotonic)
# ---------------------------------------------------------------------------


def test_check2_non_monotonic_raises(validator: DataValidator) -> None:
    """Check 2 (CRITICAL): non-monotonic open_time raises DataValidationError."""
    df = _make_ohlcv(5)
    # Swap rows 2 and 3 to break monotonicity
    df.iloc[2], df.iloc[3] = df.iloc[3].copy(), df.iloc[2].copy()
    with pytest.raises(DataValidationError, match="monotonic"):
        validator.validate_ohlcv(df, "DOGEUSDT", "1h")


def test_check2_monotonic_passes(validator: DataValidator) -> None:
    """Check 2: monotonic open_time does not produce a monotonic error."""
    df = _make_ohlcv(10)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not any("monotonic" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 3 (interval / gaps)
# ---------------------------------------------------------------------------


def test_check3_large_gap_raises(validator: DataValidator) -> None:
    """Check 3 (CRITICAL): gap > 3 missing candles raises DataValidationError."""
    # Indices 0 and 5 → 4 missing candles in between
    times = [_START_MS + i * _INTERVAL_MS for i in [0, 1, 2, 7, 8]]
    df = pd.DataFrame({
        "open_time": times,
        "open": 0.10,
        "high": 0.11,
        "low": 0.09,
        "close": 0.105,
        "volume": 1_000_000.0,
        "close_time": [t + _INTERVAL_MS - 1 for t in times],
        "era": "training",
    })
    with pytest.raises(DataValidationError, match="gap of 4"):
        validator.validate_ohlcv(df, "DOGEUSDT", "1h")


def test_check3_small_gap_no_raise(validator: DataValidator) -> None:
    """Check 3: gap of 3 missing candles is logged but does NOT raise."""
    # Indices 0, 1, 2, 6, 7 → 3 missing candles between index 2 and 6
    times = [_START_MS + i * _INTERVAL_MS for i in [0, 1, 2, 6, 7]]
    df = pd.DataFrame({
        "open_time": times,
        "open": 0.10,
        "high": 0.11,
        "low": 0.09,
        "close": 0.105,
        "volume": 1_000_000.0,
        "close_time": [t + _INTERVAL_MS - 1 for t in times],
        "era": "training",
    })
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    # gap_count should be 1 (one gap event)
    assert result.gap_count == 1


def test_check3_no_gap_contiguous(validator: DataValidator) -> None:
    """Check 3: perfectly contiguous series produces gap_count=0."""
    df = _make_ohlcv(20)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert result.gap_count == 0


def test_check3_exactly_three_missing_no_raise(validator: DataValidator) -> None:
    """Check 3: exactly 3 missing candles is the maximum allowed — no raise."""
    # Gap of 3 missing: diff = 4 * interval_ms
    times = [_START_MS, _START_MS + 4 * _INTERVAL_MS]  # 3 candles missing
    df = pd.DataFrame({
        "open_time": times,
        "open": 0.10,
        "high": 0.11,
        "low": 0.09,
        "close": 0.105,
        "volume": 1_000_000.0,
        "close_time": [t + _INTERVAL_MS - 1 for t in times],
        "era": "training",
    })
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert result.gap_count == 1
    # No DataValidationError raised — we reached here


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 4 (OHLCV sanity)
# ---------------------------------------------------------------------------


def test_check4_high_less_than_low(validator: DataValidator) -> None:
    """Check 4: high < low violation is reported in errors (not a raise)."""
    df = _make_ohlcv(3)
    df.loc[1, "high"] = 0.08   # below low=0.09
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not result.is_valid
    assert any("high < low" in e for e in result.errors)


def test_check4_close_zero(validator: DataValidator) -> None:
    """Check 4: close <= 0 is counted and reported."""
    df = _make_ohlcv(3)
    df.loc[0, "close"] = 0.0
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert any("close <= 0" in e for e in result.errors)


def test_check4_negative_volume(validator: DataValidator) -> None:
    """Check 4: negative volume is reported."""
    df = _make_ohlcv(3)
    df.loc[2, "volume"] = -1.0
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert any("volume < 0" in e for e in result.errors)


def test_check4_valid_ohlcv_no_sanity_error(validator: DataValidator) -> None:
    """Check 4: valid OHLCV data produces no sanity errors."""
    df = _make_ohlcv(10)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    sanity_keys = ("high < low", "high < open", "high < close",
                   "low > open", "low > close", "close <= 0", "volume < 0")
    for key in sanity_keys:
        assert not any(key in e for e in result.errors), (
            f"Unexpected sanity error '{key}' in valid DataFrame"
        )


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 5 (NaN / Inf) — CRITICAL
# ---------------------------------------------------------------------------


def test_check5_nan_in_close_raises(validator: DataValidator) -> None:
    """Check 5 (CRITICAL): NaN in 'close' raises DataValidationError."""
    df = _make_ohlcv(5)
    df.loc[2, "close"] = float("nan")
    with pytest.raises(DataValidationError, match="NaN"):
        validator.validate_ohlcv(df, "DOGEUSDT", "1h")


def test_check5_inf_in_volume_raises(validator: DataValidator) -> None:
    """Check 5 (CRITICAL): Inf in 'volume' raises DataValidationError."""
    df = _make_ohlcv(5)
    df.loc[0, "volume"] = float("inf")
    with pytest.raises(DataValidationError, match="Inf"):
        validator.validate_ohlcv(df, "DOGEUSDT", "1h")


def test_check5_nan_in_high_raises(validator: DataValidator) -> None:
    """Check 5 (CRITICAL): NaN in 'high' raises DataValidationError."""
    df = _make_ohlcv(5)
    df.loc[3, "high"] = float("nan")
    with pytest.raises(DataValidationError):
        validator.validate_ohlcv(df, "DOGEUSDT", "1h")


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 6 (duplicates)
# ---------------------------------------------------------------------------


def test_check6_duplicate_open_time(validator: DataValidator) -> None:
    """Check 6: duplicate open_time is detected and reported."""
    df = _make_ohlcv(5)
    # Duplicate row 0 by adding it again
    dup = df.iloc[[0]].copy()
    df = pd.concat([df, dup], ignore_index=True).sort_values("open_time")
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert result.duplicate_count > 0
    assert any("duplicate" in e.lower() for e in result.errors)


def test_check6_no_duplicates(validator: DataValidator) -> None:
    """Check 6: no duplicate open_times — duplicate_count == 0."""
    df = _make_ohlcv(10)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert result.duplicate_count == 0


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 7 (row count)
# ---------------------------------------------------------------------------


def test_check7_row_count_deviation_warning(validator: DataValidator) -> None:
    """Check 7: large deviation between actual and expected row count is reported."""
    # 3 rows spanning 7 hours with two 2-candle gaps (≤ 3 max — no CRITICAL raise).
    # Expected rows = 7 candles; actual = 3 → deviation 4 > tolerance 2.
    times = [_START_MS, _START_MS + 3 * _INTERVAL_MS, _START_MS + 6 * _INTERVAL_MS]
    df = pd.DataFrame({
        "open_time": times,
        "open": 0.10,
        "high": 0.11,
        "low": 0.09,
        "close": 0.105,
        "volume": 1_000_000.0,
        "close_time": [t + _INTERVAL_MS - 1 for t in times],
        "era": "training",
    })
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert any("Row count deviation" in e for e in result.errors)


def test_check7_small_deviation_no_warning(validator: DataValidator) -> None:
    """Check 7: deviation <= 2 does not produce a row-count warning."""
    df = _make_ohlcv(10)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not any("Row count deviation" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 8 (era assignment)
# ---------------------------------------------------------------------------


def test_check8_pre2022_wrong_era(validator: DataValidator) -> None:
    """Check 8: pre-2022 row with era='training' produces a warning."""
    pre_start = _TRAINING_START_MS - 5 * _INTERVAL_MS
    df = _make_ohlcv(5, start_ms=pre_start, era="training")  # all wrong
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert any("pre-2022" in e for e in result.errors)


def test_check8_post2022_wrong_era(validator: DataValidator) -> None:
    """Check 8: post-2022 row with era='context' produces a warning."""
    df = _make_ohlcv(5, era="context")  # post-2022 by default (_START_MS=TRAINING_START)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert any("post-2022" in e for e in result.errors)


def test_check8_correct_era_no_warning(validator: DataValidator) -> None:
    """Check 8: correct era labels produce no era errors."""
    df = _make_ohlcv(5, era="training")
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not any("Era check" in e for e in result.errors)


def test_check8_era_column_absent_skipped(validator: DataValidator) -> None:
    """Check 8: if 'era' column is absent the check is silently skipped."""
    df = _make_ohlcv(5).drop(columns=["era"])
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert not any("Era check" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, Check 9 (stale data)
# ---------------------------------------------------------------------------


def test_check9_stale_data_live_check(validator: DataValidator) -> None:
    """Check 9: stale close_time triggers a warning in live-check mode."""
    df = _make_ohlcv(5)
    # Force close_time to be ancient (well before now - 2h)
    df["close_time"] = 1_000_000_000_000  # year 2001 — definitely stale
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h", is_live_check=True)
    assert any("Stale data" in e for e in result.errors)


def test_check9_stale_data_not_applied_when_bootstrap(validator: DataValidator) -> None:
    """Check 9: stale data check is NOT applied when is_live_check=False."""
    df = _make_ohlcv(5)
    df["close_time"] = 1_000_000_000_000  # ancient close_time
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h", is_live_check=False)
    assert not any("Stale data" in e for e in result.errors)


def test_check9_fresh_data_no_warning(validator: DataValidator) -> None:
    """Check 9: fresh close_time produces no stale warning."""
    df = _make_ohlcv(5)
    now_ms = int(time.time() * 1_000)
    df["close_time"] = now_ms - 1_000  # 1 second ago — very fresh
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h", is_live_check=True)
    assert not any("Stale data" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests — validate_ohlcv, all-good path
# ---------------------------------------------------------------------------


def test_valid_ohlcv_returns_is_valid_true(validator: DataValidator) -> None:
    """All-good: a clean valid DataFrame returns is_valid=True, errors=[]."""
    df = _make_ohlcv(100)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert result.is_valid is True
    assert result.errors == []
    assert result.row_count == 100
    assert result.gap_count == 0
    assert result.duplicate_count == 0


def test_validate_ohlcv_result_type(validator: DataValidator) -> None:
    """validate_ohlcv always returns a CandleValidationResult."""
    df = _make_ohlcv(5)
    result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
    assert isinstance(result, CandleValidationResult)


# ---------------------------------------------------------------------------
# Tests — validate_funding_rates
# ---------------------------------------------------------------------------


def test_funding_valid_passes(validator: DataValidator) -> None:
    """validate_funding_rates: valid 8h-cadence data returns is_valid=True."""
    df = _make_funding(10)
    result = validator.validate_funding_rates(df)
    assert result.is_valid is True
    assert result.errors == []


def test_funding_nan_raises(validator: DataValidator) -> None:
    """validate_funding_rates: NaN in funding_rate raises DataValidationError."""
    df = _make_funding(5)
    df.loc[2, "funding_rate"] = float("nan")
    with pytest.raises(DataValidationError, match="NaN"):
        validator.validate_funding_rates(df)


def test_funding_inf_raises(validator: DataValidator) -> None:
    """validate_funding_rates: Inf in funding_rate raises DataValidationError."""
    df = _make_funding(5)
    df.loc[0, "funding_rate"] = float("inf")
    with pytest.raises(DataValidationError, match="Inf"):
        validator.validate_funding_rates(df)


def test_funding_non_8h_interval(validator: DataValidator) -> None:
    """validate_funding_rates: non-8h gap is reported."""
    df = _make_funding(5, interval_ms=3_600_000)  # 1h instead of 8h
    result = validator.validate_funding_rates(df)
    assert not result.is_valid
    assert result.gap_count > 0


def test_funding_out_of_range_rate(validator: DataValidator) -> None:
    """validate_funding_rates: funding_rate > 0.01 is reported."""
    df = _make_funding(5)
    df.loc[1, "funding_rate"] = 0.02  # beyond +1%
    result = validator.validate_funding_rates(df)
    assert any("outside [-0.01, 0.01]" in e for e in result.errors)


def test_funding_single_row_passes(validator: DataValidator) -> None:
    """validate_funding_rates: single row (no diffs) always passes interval check."""
    df = _make_funding(1)
    result = validator.validate_funding_rates(df)
    assert result.is_valid is True


# ---------------------------------------------------------------------------
# Tests — validate_feature_matrix
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int = 10,
    cols: list[str] | None = None,
    constant_col: bool = False,
) -> pd.DataFrame:
    """Build a minimal feature DataFrame for testing.

    Args:
        n: Number of rows.
        cols: Column names. Defaults to ``["feat_a", "feat_b"]``.
        constant_col: If True, add a constant ``"const"`` column (std=0).

    Returns:
        DataFrame with numeric values and a range index.
    """
    if cols is None:
        cols = ["feat_a", "feat_b"]
    rng = np.random.default_rng(42)
    data = {c: rng.standard_normal(n) for c in cols}
    if constant_col:
        data["const"] = 1.0
    return pd.DataFrame(data)


def test_feature_matrix_valid(validator: DataValidator) -> None:
    """validate_feature_matrix: valid DataFrame with all columns passes."""
    df = _make_feature_df(cols=["feat_a", "feat_b"])
    result = validator.validate_feature_matrix(df, expected_columns=["feat_a", "feat_b"])
    assert result.is_valid is True
    assert result.errors == []


def test_feature_matrix_missing_column_raises(validator: DataValidator) -> None:
    """validate_feature_matrix: missing expected column raises FeatureSchemaError."""
    df = _make_feature_df(cols=["feat_a"])
    with pytest.raises(FeatureSchemaError, match="Missing mandatory columns"):
        validator.validate_feature_matrix(df, expected_columns=["feat_a", "feat_b"])


def test_feature_schema_error_is_data_validation_error() -> None:
    """FeatureSchemaError is a subclass of DataValidationError."""
    err = FeatureSchemaError("test")
    assert isinstance(err, DataValidationError)


def test_feature_matrix_nan_raises(validator: DataValidator) -> None:
    """validate_feature_matrix: NaN in a numeric column raises DataValidationError."""
    df = _make_feature_df(cols=["feat_a", "feat_b"])
    df.loc[3, "feat_a"] = float("nan")
    with pytest.raises(DataValidationError, match="NaN"):
        validator.validate_feature_matrix(df, expected_columns=["feat_a", "feat_b"])


def test_feature_matrix_inf_raises(validator: DataValidator) -> None:
    """validate_feature_matrix: Inf in a numeric column raises DataValidationError."""
    df = _make_feature_df(cols=["feat_a", "feat_b"])
    df.loc[0, "feat_b"] = float("inf")
    with pytest.raises(DataValidationError):
        validator.validate_feature_matrix(df, expected_columns=["feat_a", "feat_b"])


def test_feature_matrix_constant_column_warning(validator: DataValidator) -> None:
    """validate_feature_matrix: constant column (std=0) produces a warning."""
    df = _make_feature_df(cols=["feat_a", "feat_b"], constant_col=True)
    result = validator.validate_feature_matrix(
        df, expected_columns=["feat_a", "feat_b", "const"]
    )
    assert not result.is_valid
    assert any("constant" in e for e in result.errors)


def test_feature_matrix_non_monotonic_index_warning(validator: DataValidator) -> None:
    """validate_feature_matrix: non-monotonic index is reported as a warning."""
    df = _make_feature_df(cols=["feat_a", "feat_b"])
    # Shuffle index to break monotonicity
    df = df.sample(frac=1, random_state=99)
    result = validator.validate_feature_matrix(df, expected_columns=["feat_a", "feat_b"])
    assert any("monotonic" in e for e in result.errors)


def test_feature_matrix_empty_df_no_raise(validator: DataValidator) -> None:
    """validate_feature_matrix: empty DataFrame returns is_valid=False (no raise)."""
    df = pd.DataFrame(columns=["feat_a", "feat_b"])
    result = validator.validate_feature_matrix(df, expected_columns=["feat_a"])
    assert result.is_valid is False
    assert result.row_count == 0
