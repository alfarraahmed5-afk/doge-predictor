"""Unit tests for src/processing/schemas.py and src/processing/df_schemas.py.

Tests cover:
    - OHLCVRecord: rejects high < low, close <= 0
    - OHLCVSchema: rejects DataFrames with NaN, non-monotonic timestamps
    - PredictionRecord: rejects confidence_score > 1.0, invalid horizon_label
    - FundingRateRecord: rejects out-of-range funding_rate
    - FundingRateSchema: enforces 8h cadence
    - FeatureSchema: enforces UTC index and mandatory columns
    - RewardResult: basic field validation
    - CandleValidationResult: basic instantiation
    - FeatureRecord: era and regime validators
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pydantic import ValidationError

from src.processing.df_schemas import (
    FUNDING_RATE_INTERVAL_MS,
    FeatureSchema,
    FundingRateSchema,
    OHLCVSchema,
)
from src.processing.schemas import (
    MANDATORY_FEATURE_COLUMNS,
    CandleValidationResult,
    FeatureRecord,
    FundingRateRecord,
    OHLCVRecord,
    PredictionRecord,
    RewardResult,
)


# ---------------------------------------------------------------------------
# Fixtures — valid base objects
# ---------------------------------------------------------------------------


def _valid_ohlcv_kwargs() -> dict[str, Any]:
    """Return keyword arguments for a valid OHLCVRecord."""
    return {
        "open_time": 1_672_531_200_000,  # 2023-01-01 00:00 UTC in ms
        "open": 0.079,
        "high": 0.082,
        "low": 0.077,
        "close": 0.080,
        "volume": 1_500_000.0,
        "close_time": 1_672_534_800_000,  # 1h later
        "quote_volume": 120_000.0,
        "num_trades": 4_200,
        "symbol": "DOGEUSDT",
        "interval": "1h",
        "era": "training",
    }


def _valid_prediction_kwargs() -> dict[str, Any]:
    """Return keyword arguments for a valid PredictionRecord."""
    base_ts = 1_672_531_200_000
    return {
        "prediction_id": str(uuid.uuid4()),
        "created_at": base_ts,
        "open_time": base_ts,
        "symbol": "DOGEUSDT",
        "horizon_label": "SHORT",
        "horizon_candles": 4,
        "target_open_time": base_ts + 4 * 3_600_000,  # 4h later
        "price_at_prediction": 0.080,
        "predicted_direction": 1,
        "confidence_score": 0.65,
        "lstm_prob": 0.65,
        "xgb_prob": 0.60,
        "regime_label": "TRENDING_BULL",
        "model_version": "v1.0.0",
    }


def _valid_ohlcv_df(n_rows: int = 5) -> pd.DataFrame:
    """Return a valid OHLCV DataFrame with *n_rows* rows.

    Args:
        n_rows: Number of rows to generate.

    Returns:
        A valid OHLCV DataFrame.
    """
    base_ms = 1_672_531_200_000
    interval_ms = 3_600_000  # 1h
    return pd.DataFrame(
        {
            "open_time": [base_ms + i * interval_ms for i in range(n_rows)],
            "open": [0.079 + i * 0.001 for i in range(n_rows)],
            "high": [0.082 + i * 0.001 for i in range(n_rows)],
            "low": [0.077 + i * 0.001 for i in range(n_rows)],
            "close": [0.080 + i * 0.001 for i in range(n_rows)],
            "volume": [1_500_000.0 for _ in range(n_rows)],
        }
    )


def _valid_funding_df(n_rows: int = 5) -> pd.DataFrame:
    """Return a valid funding rate DataFrame.

    Args:
        n_rows: Number of rows to generate.

    Returns:
        A valid FundingRate DataFrame.
    """
    base_ms = 1_672_531_200_000
    return pd.DataFrame(
        {
            "timestamp_ms": [
                base_ms + i * FUNDING_RATE_INTERVAL_MS for i in range(n_rows)
            ],
            "funding_rate": [0.0001 * (i + 1) for i in range(n_rows)],
        }
    )


def _valid_feature_df(n_rows: int = 10) -> pd.DataFrame:
    """Return a valid feature DataFrame with a UTC DatetimeTZDtype index.

    Args:
        n_rows: Number of rows to generate.

    Returns:
        A valid feature matrix DataFrame.
    """
    base = pd.Timestamp("2023-01-01", tz="UTC")
    index = pd.date_range(base, periods=n_rows, freq="1h", tz="UTC")
    data: dict[str, Any] = {}

    for col in MANDATORY_FEATURE_COLUMNS:
        if col in {
            "volume_spike_flag",
            "funding_extreme_long",
            "funding_extreme_short",
            "at_round_number_flag",
            "htf_4h_trend",
            "htf_1d_trend",
        }:
            # Binary columns — alternate 0 and 1 to avoid constant column
            data[col] = [float(i % 2) for i in range(n_rows)]
        else:
            # Numeric columns — small non-constant values
            data[col] = [float(i) * 0.001 + 0.001 for i in range(n_rows)]

    df = pd.DataFrame(data, index=index)
    df.index.name = "open_time"
    return df


# ---------------------------------------------------------------------------
# OHLCVRecord tests
# ---------------------------------------------------------------------------


class TestOHLCVRecord:
    """Tests for the OHLCVRecord Pydantic model."""

    def test_valid_record_passes(self) -> None:
        """A record with all valid fields should instantiate without error."""
        record = OHLCVRecord(**_valid_ohlcv_kwargs())
        assert record.close == pytest.approx(0.080)
        assert record.era == "training"

    def test_rejects_high_less_than_low(self) -> None:
        """OHLCVRecord must reject records where high < low."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["high"] = 0.070   # below low=0.077
        with pytest.raises(ValidationError, match="high"):
            OHLCVRecord(**kwargs)

    def test_rejects_high_equal_to_low_is_ok(self) -> None:
        """high == low is a doji candle — it must be accepted."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["high"] = 0.080
        kwargs["low"] = 0.080
        kwargs["open"] = 0.080
        kwargs["close"] = 0.080
        record = OHLCVRecord(**kwargs)
        assert record.high == record.low

    def test_rejects_close_zero(self) -> None:
        """OHLCVRecord must reject records where close == 0."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["close"] = 0.0
        with pytest.raises(ValidationError):
            OHLCVRecord(**kwargs)

    def test_rejects_close_negative(self) -> None:
        """OHLCVRecord must reject records where close < 0."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["close"] = -0.01
        with pytest.raises(ValidationError):
            OHLCVRecord(**kwargs)

    def test_rejects_volume_negative(self) -> None:
        """OHLCVRecord must reject records where volume < 0."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["volume"] = -1.0
        with pytest.raises(ValidationError):
            OHLCVRecord(**kwargs)

    def test_rejects_close_time_before_open_time(self) -> None:
        """OHLCVRecord must reject records where close_time <= open_time."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["close_time"] = kwargs["open_time"] - 1
        with pytest.raises(ValidationError, match="close_time"):
            OHLCVRecord(**kwargs)

    def test_rejects_invalid_era(self) -> None:
        """OHLCVRecord must reject an unrecognised era value."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["era"] = "backtest"  # not in Literal
        with pytest.raises(ValidationError):
            OHLCVRecord(**kwargs)

    def test_rejects_high_below_open(self) -> None:
        """OHLCVRecord must reject high < open."""
        kwargs = _valid_ohlcv_kwargs()
        kwargs["high"] = 0.078  # below open=0.079 but above low=0.077
        kwargs["close"] = 0.078
        with pytest.raises(ValidationError, match="high"):
            OHLCVRecord(**kwargs)


# ---------------------------------------------------------------------------
# OHLCVSchema tests (Pandera DataFrame validation)
# ---------------------------------------------------------------------------


class TestOHLCVSchema:
    """Tests for the OHLCVSchema Pandera schema."""

    def test_valid_df_passes(self) -> None:
        """A properly formed OHLCV DataFrame should pass validation."""
        df = _valid_ohlcv_df()
        validated = OHLCVSchema.validate(df, lazy=True)
        assert len(validated) == 5

    def test_rejects_nan_in_close(self) -> None:
        """OHLCVSchema must reject DataFrames that contain NaN in close."""
        df = _valid_ohlcv_df()
        df.loc[df.index[2], "close"] = float("nan")
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_nan_in_volume(self) -> None:
        """OHLCVSchema must reject DataFrames that contain NaN in volume."""
        df = _valid_ohlcv_df()
        df.loc[df.index[0], "volume"] = float("nan")
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_non_monotonic_timestamps(self) -> None:
        """OHLCVSchema must reject DataFrames with non-monotonic open_time."""
        df = _valid_ohlcv_df()
        # Swap the first two rows to break monotonicity
        df.iloc[[0, 1]] = df.iloc[[1, 0]].values
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_duplicate_timestamps(self) -> None:
        """OHLCVSchema must reject DataFrames with duplicate open_time values."""
        df = _valid_ohlcv_df()
        df.iloc[1, df.columns.get_loc("open_time")] = df.iloc[
            0, df.columns.get_loc("open_time")
        ]
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_inf_in_high(self) -> None:
        """OHLCVSchema must reject DataFrames containing Inf in high."""
        df = _valid_ohlcv_df()
        df.loc[df.index[1], "high"] = float("inf")
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_high_below_close(self) -> None:
        """OHLCVSchema must reject rows where high < close."""
        df = _valid_ohlcv_df()
        df.loc[df.index[0], "high"] = 0.078  # open=0.079, close=0.080, high now < close
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)

    def test_rejects_close_zero(self) -> None:
        """OHLCVSchema must reject rows where close == 0."""
        df = _valid_ohlcv_df()
        df.loc[df.index[0], "close"] = 0.0
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            OHLCVSchema.validate(df, lazy=True)


# ---------------------------------------------------------------------------
# PredictionRecord tests
# ---------------------------------------------------------------------------


class TestPredictionRecord:
    """Tests for the PredictionRecord Pydantic model."""

    def test_valid_record_passes(self) -> None:
        """A PredictionRecord with all valid fields should instantiate."""
        record = PredictionRecord(**_valid_prediction_kwargs())
        assert record.horizon_label == "SHORT"
        assert record.predicted_direction == 1
        assert record.actual_price is None  # outcome not yet filled

    def test_rejects_confidence_score_above_one(self) -> None:
        """PredictionRecord must reject confidence_score > 1.0."""
        kwargs = _valid_prediction_kwargs()
        kwargs["confidence_score"] = 1.1
        with pytest.raises(ValidationError, match="confidence_score"):
            PredictionRecord(**kwargs)

    def test_rejects_confidence_score_below_half(self) -> None:
        """PredictionRecord must reject confidence_score < 0.5."""
        kwargs = _valid_prediction_kwargs()
        kwargs["confidence_score"] = 0.49
        with pytest.raises(ValidationError, match="confidence_score"):
            PredictionRecord(**kwargs)

    def test_rejects_invalid_horizon_label(self) -> None:
        """PredictionRecord must reject horizon_label not in the valid set."""
        kwargs = _valid_prediction_kwargs()
        kwargs["horizon_label"] = "DAILY"
        with pytest.raises(ValidationError):
            PredictionRecord(**kwargs)

    def test_rejects_invalid_horizon_label_lowercase(self) -> None:
        """PredictionRecord must reject lowercase horizon labels."""
        kwargs = _valid_prediction_kwargs()
        kwargs["horizon_label"] = "short"
        with pytest.raises(ValidationError):
            PredictionRecord(**kwargs)

    def test_rejects_invalid_predicted_direction(self) -> None:
        """PredictionRecord must reject predicted_direction not in {-1, 0, 1}."""
        kwargs = _valid_prediction_kwargs()
        kwargs["predicted_direction"] = 2
        with pytest.raises(ValidationError):
            PredictionRecord(**kwargs)

    def test_accepts_all_valid_horizons(self) -> None:
        """PredictionRecord should accept all four valid horizon labels."""
        horizon_candle_map = {
            "SHORT": 4,
            "MEDIUM": 24,
            "LONG": 168,
            "MACRO": 720,
        }
        base_ts = 1_672_531_200_000
        for label, candles in horizon_candle_map.items():
            kwargs = _valid_prediction_kwargs()
            kwargs["horizon_label"] = label
            kwargs["horizon_candles"] = candles
            kwargs["target_open_time"] = base_ts + candles * 3_600_000
            record = PredictionRecord(**kwargs)
            assert record.horizon_label == label

    def test_accepts_all_valid_directions(self) -> None:
        """PredictionRecord should accept -1, 0, and 1 as predicted_direction."""
        for direction in (-1, 0, 1):
            kwargs = _valid_prediction_kwargs()
            kwargs["predicted_direction"] = direction
            record = PredictionRecord(**kwargs)
            assert record.predicted_direction == direction

    def test_rejects_target_before_open(self) -> None:
        """PredictionRecord must reject target_open_time <= open_time."""
        kwargs = _valid_prediction_kwargs()
        kwargs["target_open_time"] = kwargs["open_time"]  # equal, not after
        with pytest.raises(ValidationError, match="target_open_time"):
            PredictionRecord(**kwargs)

    def test_rejects_invalid_regime_label(self) -> None:
        """PredictionRecord must reject an unrecognised regime_label."""
        kwargs = _valid_prediction_kwargs()
        kwargs["regime_label"] = "BULLISH"  # not a valid regime
        with pytest.raises(ValidationError, match="regime_label"):
            PredictionRecord(**kwargs)

    def test_outcome_fields_default_to_none(self) -> None:
        """Outcome fields must default to None (not yet filled by Verifier)."""
        record = PredictionRecord(**_valid_prediction_kwargs())
        assert record.actual_price is None
        assert record.actual_direction is None
        assert record.reward_score is None
        assert record.direction_correct is None
        assert record.error_pct is None
        assert record.verified_at is None


# ---------------------------------------------------------------------------
# FundingRateRecord tests
# ---------------------------------------------------------------------------


class TestFundingRateRecord:
    """Tests for the FundingRateRecord Pydantic model."""

    def test_valid_record_passes(self) -> None:
        """A FundingRateRecord with valid fields should instantiate."""
        record = FundingRateRecord(
            timestamp_ms=1_672_531_200_000,
            symbol="DOGEUSDT",
            funding_rate=0.0001,
            mark_price=0.080,
        )
        assert record.funding_rate == pytest.approx(0.0001)

    def test_rejects_funding_rate_above_max(self) -> None:
        """FundingRateRecord must reject funding_rate > 0.01."""
        with pytest.raises(ValidationError, match="funding_rate"):
            FundingRateRecord(
                timestamp_ms=1_672_531_200_000,
                symbol="DOGEUSDT",
                funding_rate=0.02,
                mark_price=0.080,
            )

    def test_rejects_funding_rate_below_min(self) -> None:
        """FundingRateRecord must reject funding_rate < -0.01."""
        with pytest.raises(ValidationError, match="funding_rate"):
            FundingRateRecord(
                timestamp_ms=1_672_531_200_000,
                symbol="DOGEUSDT",
                funding_rate=-0.02,
                mark_price=0.080,
            )

    def test_rejects_timestamp_zero(self) -> None:
        """FundingRateRecord must reject timestamp_ms == 0."""
        with pytest.raises(ValidationError):
            FundingRateRecord(
                timestamp_ms=0,
                symbol="DOGEUSDT",
                funding_rate=0.0001,
                mark_price=0.080,
            )


# ---------------------------------------------------------------------------
# FundingRateSchema tests (Pandera)
# ---------------------------------------------------------------------------


class TestFundingRateSchema:
    """Tests for the FundingRateSchema Pandera schema."""

    def test_valid_df_passes(self) -> None:
        """A properly spaced funding rate DataFrame should pass."""
        df = _valid_funding_df()
        validated = FundingRateSchema.validate(df, lazy=True)
        assert len(validated) == 5

    def test_rejects_non_8h_interval(self) -> None:
        """FundingRateSchema must reject rows not exactly 8h apart."""
        df = _valid_funding_df()
        # Corrupt the second row's timestamp
        df.iloc[1, df.columns.get_loc("timestamp_ms")] += 1_000
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            FundingRateSchema.validate(df, lazy=True)

    def test_rejects_non_monotonic(self) -> None:
        """FundingRateSchema must reject non-monotonic timestamp_ms."""
        df = _valid_funding_df(n_rows=3)
        df.iloc[[0, 1]] = df.iloc[[1, 0]].values
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            FundingRateSchema.validate(df, lazy=True)

    def test_single_row_passes(self) -> None:
        """A single-row funding rate DataFrame should pass (no interval to check)."""
        df = _valid_funding_df(n_rows=1)
        validated = FundingRateSchema.validate(df, lazy=True)
        assert len(validated) == 1


# ---------------------------------------------------------------------------
# FeatureSchema tests (Pandera)
# ---------------------------------------------------------------------------


class TestFeatureSchema:
    """Tests for the FeatureSchema Pandera schema."""

    def test_valid_df_passes(self) -> None:
        """A feature DataFrame with all mandatory columns should pass."""
        df = _valid_feature_df()
        validated = FeatureSchema.validate(df, lazy=True)
        assert len(validated) == 10

    def test_rejects_missing_mandatory_column(self) -> None:
        """FeatureSchema must reject DataFrames missing a mandatory column."""
        df = _valid_feature_df()
        df = df.drop(columns=["doge_btc_corr_24h"])
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            FeatureSchema.validate(df, lazy=True)

    def test_rejects_nan_in_feature(self) -> None:
        """FeatureSchema must reject NaN in any feature column."""
        df = _valid_feature_df()
        df.loc[df.index[3], "funding_rate"] = float("nan")
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            FeatureSchema.validate(df, lazy=True)

    def test_rejects_non_utc_index(self) -> None:
        """FeatureSchema must reject a DataFrame without a UTC-aware index."""
        df = _valid_feature_df()
        df.index = df.index.tz_localize(None)  # strip timezone
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            FeatureSchema.validate(df, lazy=True)


# ---------------------------------------------------------------------------
# CandleValidationResult tests
# ---------------------------------------------------------------------------


class TestCandleValidationResult:
    """Tests for the CandleValidationResult Pydantic model."""

    def test_valid_result_passes(self) -> None:
        """CandleValidationResult should instantiate with valid fields."""
        result = CandleValidationResult(
            is_valid=True,
            errors=[],
            row_count=1000,
            gap_count=0,
            duplicate_count=0,
        )
        assert result.is_valid is True
        assert result.errors == []

    def test_invalid_result_with_errors(self) -> None:
        """CandleValidationResult should accept is_valid=False with errors."""
        result = CandleValidationResult(
            is_valid=False,
            errors=["Gap at 2023-06-15", "Duplicate at 2023-07-01"],
            row_count=9_998,
            gap_count=1,
            duplicate_count=1,
        )
        assert result.is_valid is False
        assert len(result.errors) == 2


# ---------------------------------------------------------------------------
# FeatureRecord tests
# ---------------------------------------------------------------------------


class TestFeatureRecord:
    """Tests for the FeatureRecord Pydantic model."""

    def test_valid_record_passes(self) -> None:
        """FeatureRecord should instantiate with minimal required fields."""
        record = FeatureRecord(
            open_time=1_672_531_200_000,
            symbol="DOGEUSDT",
            era="training",
            regime="TRENDING_BULL",
        )
        assert record.doge_btc_corr_24h is None  # optional, defaults to None

    def test_rejects_invalid_era(self) -> None:
        """FeatureRecord must reject an unrecognised era value."""
        with pytest.raises(ValidationError, match="era"):
            FeatureRecord(
                open_time=1_672_531_200_000,
                symbol="DOGEUSDT",
                era="backtest",
                regime="TRENDING_BULL",
            )

    def test_rejects_invalid_regime(self) -> None:
        """FeatureRecord must reject an unrecognised regime value."""
        with pytest.raises(ValidationError, match="regime"):
            FeatureRecord(
                open_time=1_672_531_200_000,
                symbol="DOGEUSDT",
                era="training",
                regime="SIDEWAYS",
            )


# ---------------------------------------------------------------------------
# RewardResult tests
# ---------------------------------------------------------------------------


class TestRewardResult:
    """Tests for the RewardResult Pydantic model."""

    def test_valid_correct_prediction(self) -> None:
        """RewardResult for a correct prediction should instantiate."""
        result = RewardResult(
            reward_score=1.4,
            direction_score=1.0,
            magnitude_score=0.95,
            calibration_score=1.5,
            error_pct=0.02,
            direction_correct=True,
        )
        assert result.direction_correct is True
        assert result.magnitude_score == pytest.approx(0.95)

    def test_valid_incorrect_prediction(self) -> None:
        """RewardResult for an incorrect prediction should instantiate."""
        result = RewardResult(
            reward_score=-2.1,
            direction_score=-1.0,
            magnitude_score=0.70,
            calibration_score=-2.5,
            error_pct=0.08,
            direction_correct=False,
        )
        assert result.direction_correct is False
        assert result.reward_score == pytest.approx(-2.1)

    def test_rejects_negative_error_pct(self) -> None:
        """RewardResult must reject error_pct < 0."""
        with pytest.raises(ValidationError, match="error_pct"):
            RewardResult(
                reward_score=1.0,
                direction_score=1.0,
                magnitude_score=0.9,
                calibration_score=1.0,
                error_pct=-0.01,
                direction_correct=True,
            )

    def test_rejects_magnitude_above_one(self) -> None:
        """RewardResult must reject magnitude_score > 1.0."""
        with pytest.raises(ValidationError, match="magnitude_score"):
            RewardResult(
                reward_score=2.0,
                direction_score=1.0,
                magnitude_score=1.1,
                calibration_score=1.0,
                error_pct=0.0,
                direction_correct=True,
            )
