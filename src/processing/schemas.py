"""Pydantic data transfer objects for doge_predictor.

Defines the canonical record shapes for every entity that flows through the
pipeline. All inter-module data hand-offs should use these models so that type
errors are caught at the boundary rather than deep inside a computation.

Models defined:
    - OHLCVRecord           : single validated OHLCV candle
    - FundingRateRecord     : single 8h funding-rate observation
    - CandleValidationResult: summary returned by the validation pass
    - FeatureRecord         : single row of the feature matrix
    - PredictionRecord      : one row of the doge_predictions TimescaleDB table
    - RewardResult          : output of compute_reward()

Usage::

    from src.processing.schemas import OHLCVRecord, PredictionRecord
    record = OHLCVRecord(**row_dict)   # raises ValidationError on violation

Notes:
    - All timestamps are UTC epoch milliseconds (int) — never tz-naive datetimes.
    - Pydantic v2 is required (pydantic>=2.0).
"""

from __future__ import annotations

from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Module-level constants (no magic numbers in logic)
# ---------------------------------------------------------------------------

#: Minimum plausible UTC epoch-ms (2001-09-09) — sanity lower bound.
EPOCH_MS_MIN: int = 1_000_000_000_000
#: Maximum plausible UTC epoch-ms (year 2286) — sanity upper bound.
EPOCH_MS_MAX: int = 9_999_999_999_999

#: Funding rate absolute bounds (CLAUDE.md §5 FACT 5).
FUNDING_RATE_MIN: float = -0.01
FUNDING_RATE_MAX: float = 0.01

#: Prediction confidence score bounds (CLAUDE.md §10 Step 7).
CONFIDENCE_MIN: float = 0.5
CONFIDENCE_MAX: float = 1.0

#: Valid predicted direction values.
VALID_DIRECTIONS: frozenset[int] = frozenset({-1, 0, 1})

#: Valid prediction horizon labels (CLAUDE.md §11).
VALID_HORIZONS: frozenset[str] = frozenset({"SHORT", "MEDIUM", "LONG", "MACRO"})

#: Candle count per horizon label (CLAUDE.md §11 table).
HORIZON_CANDLES: dict[str, int] = {
    "SHORT": 4,
    "MEDIUM": 24,
    "LONG": 168,
    "MACRO": 720,
}

#: Valid market regime labels (CLAUDE.md §6).
VALID_REGIMES: frozenset[str] = frozenset(
    {"TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"}
)

#: Valid era labels.
VALID_ERAS: frozenset[str] = frozenset({"context", "training", "live"})

#: Mandatory DOGE feature column names (CLAUDE.md §7).
MANDATORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "doge_btc_corr_12h",
    "doge_btc_corr_24h",
    "doge_btc_corr_7d",
    "dogebtc_mom_6h",
    "dogebtc_mom_24h",
    "dogebtc_mom_48h",
    "volume_ratio",
    "volume_spike_flag",
    "funding_rate",
    "funding_rate_zscore",
    "funding_extreme_long",
    "funding_extreme_short",
    "htf_4h_rsi",
    "htf_4h_trend",
    "htf_4h_bb_pctb",
    "htf_1d_trend",
    "htf_1d_return",
    "ath_distance",
    "distance_to_round_pct",
    "at_round_number_flag",
    "nearest_round_level",
)


# ---------------------------------------------------------------------------
# OHLCVRecord
# ---------------------------------------------------------------------------


class OHLCVRecord(BaseModel):
    """Single OHLCV candle record.

    Enforces all OHLCV price-bar invariants:
    - high >= open, high >= close, high >= low
    - low <= open, low <= close
    - close > 0, volume >= 0
    - close_time > open_time

    Args:
        open_time: Candle open timestamp, UTC epoch milliseconds.
        open: Open price (USD), must be > 0.
        high: High price (USD), must be >= open and close.
        low: Low price (USD), must be <= open and close.
        close: Close price (USD), must be > 0.
        volume: Base asset volume (DOGE), must be >= 0.
        close_time: Candle close timestamp, UTC epoch milliseconds.
        quote_volume: Quote asset volume (USDT), must be >= 0.
        num_trades: Number of trades in the candle, must be >= 0.
        symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
        interval: Candle interval string (e.g. ``"1h"``).
        era: Data era — one of ``'context'``, ``'training'``, ``'live'``.
    """

    open_time: int = Field(
        ...,
        ge=EPOCH_MS_MIN,
        le=EPOCH_MS_MAX,
        description="Candle open timestamp — UTC epoch milliseconds.",
    )
    open: float = Field(..., gt=0.0, description="Open price (USD).")
    high: float = Field(..., gt=0.0, description="High price (USD).")
    low: float = Field(..., gt=0.0, description="Low price (USD).")
    close: float = Field(..., gt=0.0, description="Close price (USD).")
    volume: float = Field(..., ge=0.0, description="Base asset volume (DOGE).")
    close_time: int = Field(
        ...,
        ge=EPOCH_MS_MIN,
        le=EPOCH_MS_MAX,
        description="Candle close timestamp — UTC epoch milliseconds.",
    )
    quote_volume: float = Field(
        ..., ge=0.0, description="Quote asset volume (USDT)."
    )
    num_trades: int = Field(
        ..., ge=0, description="Number of trades in the candle."
    )
    symbol: str = Field(..., min_length=1, description="Trading pair symbol.")
    interval: str = Field(..., min_length=1, description="Candle interval string.")
    era: Literal["context", "training", "live"] = Field(
        ..., description="Data era: 'context' (pre-2022), 'training', or 'live'."
    )

    @model_validator(mode="after")
    def validate_ohlcv_invariants(self) -> "OHLCVRecord":
        """Enforce cross-field OHLCV price-bar invariants.

        Returns:
            The validated record.

        Raises:
            ValueError: If any price-bar invariant is violated.
        """
        errors: list[str] = []

        if self.high < self.low:
            errors.append(
                f"high ({self.high}) must be >= low ({self.low})"
            )
        if self.high < self.open:
            errors.append(
                f"high ({self.high}) must be >= open ({self.open})"
            )
        if self.high < self.close:
            errors.append(
                f"high ({self.high}) must be >= close ({self.close})"
            )
        if self.low > self.open:
            errors.append(
                f"low ({self.low}) must be <= open ({self.open})"
            )
        if self.low > self.close:
            errors.append(
                f"low ({self.low}) must be <= close ({self.close})"
            )
        if self.close_time <= self.open_time:
            errors.append(
                f"close_time ({self.close_time}) must be > open_time ({self.open_time})"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return self


# ---------------------------------------------------------------------------
# FundingRateRecord
# ---------------------------------------------------------------------------


class FundingRateRecord(BaseModel):
    """Single 8-hour funding rate observation.

    Args:
        timestamp_ms: Settlement timestamp, UTC epoch milliseconds.
        symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
        funding_rate: Funding rate value. Expected range: [-0.01, 0.01].
        mark_price: Mark price at settlement (USD).
    """

    timestamp_ms: int = Field(
        ...,
        gt=0,
        description="Settlement timestamp — UTC epoch milliseconds.",
    )
    symbol: str = Field(..., min_length=1, description="Trading pair symbol.")
    funding_rate: float = Field(
        ...,
        ge=FUNDING_RATE_MIN,
        le=FUNDING_RATE_MAX,
        description="Funding rate value (e.g. 0.0001 = 0.01% per 8h).",
    )
    mark_price: float = Field(
        ..., gt=0.0, description="Mark price at settlement (USD)."
    )


# ---------------------------------------------------------------------------
# CandleValidationResult
# ---------------------------------------------------------------------------


class CandleValidationResult(BaseModel):
    """Summary result produced by the candle validation pass.

    Args:
        is_valid: True if no validation errors were found.
        errors: List of human-readable error messages.
        row_count: Total number of rows inspected.
        gap_count: Number of missing candles detected in the time series.
        duplicate_count: Number of duplicate open_time values detected.
    """

    is_valid: bool = Field(..., description="True if no validation errors were found.")
    errors: list[str] = Field(
        default_factory=list, description="List of human-readable error messages."
    )
    row_count: int = Field(..., ge=0, description="Total rows inspected.")
    gap_count: int = Field(..., ge=0, description="Missing candles detected.")
    duplicate_count: int = Field(..., ge=0, description="Duplicate open_time count.")


# ---------------------------------------------------------------------------
# FeatureRecord
# ---------------------------------------------------------------------------


class FeatureRecord(BaseModel):
    """Single row of the computed feature matrix.

    Mandatory metadata fields are required; all 21 DOGE feature columns are
    Optional[float] with a default of None (populated by the feature pipeline).

    Args:
        open_time: Candle open timestamp, UTC epoch milliseconds.
        symbol: Trading pair symbol.
        era: Data era label.
        regime: Market regime label (one of the 5 defined regimes).
        doge_btc_corr_12h: Rolling 12h log-return correlation with BTC.
        doge_btc_corr_24h: Rolling 24h log-return correlation with BTC.
        doge_btc_corr_7d: Rolling 7d (168h) log-return correlation with BTC.
        dogebtc_mom_6h: log(dogebtc_close[t] / dogebtc_close[t-6]).
        dogebtc_mom_24h: log(dogebtc_close[t] / dogebtc_close[t-24]).
        dogebtc_mom_48h: log(dogebtc_close[t] / dogebtc_close[t-48]).
        volume_ratio: volume / volume.rolling(20).mean().
        volume_spike_flag: 1 if volume_ratio >= 3.0, else 0.
        funding_rate: 8h funding rate forward-filled to 1h.
        funding_rate_zscore: 90-period rolling z-score of funding_rate.
        funding_extreme_long: 1 if funding_rate > 0.001, else 0.
        funding_extreme_short: 1 if funding_rate < -0.0005, else 0.
        htf_4h_rsi: RSI computed on 4h closes, aligned to 1h.
        htf_4h_trend: Binary trend flag from 4h timeframe.
        htf_4h_bb_pctb: Bollinger %B from 4h timeframe.
        htf_1d_trend: Binary trend flag from 1d timeframe.
        htf_1d_return: Log-return of the last completed 1d candle.
        ath_distance: Distance from current close to all-time high (pct).
        distance_to_round_pct: Distance to nearest round-number level (pct).
        at_round_number_flag: 1 if price is within 0.5% of a round number.
        nearest_round_level: The nearest round psychological price level (USD).
    """

    # Metadata
    open_time: int = Field(..., ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)
    symbol: str = Field(..., min_length=1)
    era: str = Field(...)
    regime: str = Field(...)

    # DOGE-specific mandatory features (CLAUDE.md §7)
    doge_btc_corr_12h: Optional[float] = Field(default=None)
    doge_btc_corr_24h: Optional[float] = Field(default=None)
    doge_btc_corr_7d: Optional[float] = Field(default=None)
    dogebtc_mom_6h: Optional[float] = Field(default=None)
    dogebtc_mom_24h: Optional[float] = Field(default=None)
    dogebtc_mom_48h: Optional[float] = Field(default=None)
    volume_ratio: Optional[float] = Field(default=None)
    volume_spike_flag: Optional[float] = Field(default=None)
    funding_rate: Optional[float] = Field(default=None)
    funding_rate_zscore: Optional[float] = Field(default=None)
    funding_extreme_long: Optional[float] = Field(default=None)
    funding_extreme_short: Optional[float] = Field(default=None)
    htf_4h_rsi: Optional[float] = Field(default=None)
    htf_4h_trend: Optional[float] = Field(default=None)
    htf_4h_bb_pctb: Optional[float] = Field(default=None)
    htf_1d_trend: Optional[float] = Field(default=None)
    htf_1d_return: Optional[float] = Field(default=None)
    ath_distance: Optional[float] = Field(default=None)
    distance_to_round_pct: Optional[float] = Field(default=None)
    at_round_number_flag: Optional[float] = Field(default=None)
    nearest_round_level: Optional[float] = Field(default=None)

    @field_validator("era")
    @classmethod
    def validate_era(cls, v: str) -> str:
        """Validate that era is one of the recognised labels.

        Args:
            v: The era string to validate.

        Returns:
            The validated era string.

        Raises:
            ValueError: If era is not in VALID_ERAS.
        """
        if v not in VALID_ERAS:
            raise ValueError(
                f"era must be one of {sorted(VALID_ERAS)}, got '{v}'"
            )
        return v

    @field_validator("regime")
    @classmethod
    def validate_regime(cls, v: str) -> str:
        """Validate that regime is one of the five defined labels.

        Args:
            v: The regime string to validate.

        Returns:
            The validated regime string.

        Raises:
            ValueError: If regime is not in VALID_REGIMES.
        """
        if v not in VALID_REGIMES:
            raise ValueError(
                f"regime must be one of {sorted(VALID_REGIMES)}, got '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------


class PredictionRecord(BaseModel):
    """One row of the ``doge_predictions`` TimescaleDB table.

    Prediction fields are immutable after insert. Outcome fields (actual_price,
    reward_score, etc.) are written only by the Verifier and default to None.

    CLAUDE.md §11: "Prediction fields are immutable after insert. Outcome fields
    written only by Verifier."

    Args:
        prediction_id: Unique identifier for this prediction (UUID string).
        created_at: Timestamp when the prediction was logged, UTC epoch ms.
        open_time: Candle open_time at which the prediction was generated.
        symbol: Trading pair symbol.
        horizon_label: Prediction horizon — ``SHORT`` / ``MEDIUM`` / ``LONG`` / ``MACRO``.
        horizon_candles: Number of 1h candles in this horizon (4/24/168/720).
        target_open_time: open_time of the candle being predicted
            (= open_time + horizon_candles × 3_600_000 ms).
        price_at_prediction: Close price at the time prediction was generated.
        predicted_direction: Direction signal — ``-1`` (short), ``0`` (hold), ``1`` (long).
        confidence_score: Model ensemble confidence; must be in [0.5, 1.0].
        lstm_prob: Raw LSTM output probability.
        xgb_prob: Raw XGBoost output probability.
        regime_label: Market regime at prediction time.
        model_version: Version string of the model that generated this prediction.
        actual_price: Actual close price at target_open_time (filled by Verifier).
        actual_direction: Actual direction realised (filled by Verifier).
        reward_score: Final RL reward value (filled by Verifier).
        direction_correct: Whether predicted_direction matched actual (filled by Verifier).
        error_pct: Absolute percentage error vs price_at_prediction (filled by Verifier).
        verified_at: Timestamp when Verifier processed this row, UTC epoch ms.
    """

    # --- Immutable prediction fields ---
    prediction_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique prediction ID (UUID string).",
    )
    created_at: int = Field(
        ...,
        ge=EPOCH_MS_MIN,
        le=EPOCH_MS_MAX,
        description="Prediction creation timestamp — UTC epoch milliseconds.",
    )
    open_time: int = Field(
        ...,
        ge=EPOCH_MS_MIN,
        le=EPOCH_MS_MAX,
        description="Candle open_time at which prediction was generated.",
    )
    symbol: str = Field(..., min_length=1, description="Trading pair symbol.")
    horizon_label: Literal["SHORT", "MEDIUM", "LONG", "MACRO"] = Field(
        ..., description="Prediction horizon label."
    )
    horizon_candles: int = Field(
        ..., gt=0, description="Number of 1h candles in this horizon."
    )
    target_open_time: int = Field(
        ...,
        ge=EPOCH_MS_MIN,
        description="open_time of the candle being predicted (UTC epoch ms).",
    )
    price_at_prediction: float = Field(
        ..., gt=0.0, description="Close price at prediction generation time (USD)."
    )
    predicted_direction: Literal[-1, 0, 1] = Field(
        ..., description="Direction signal: -1 (short), 0 (hold), 1 (long)."
    )
    confidence_score: float = Field(
        ...,
        ge=CONFIDENCE_MIN,
        le=CONFIDENCE_MAX,
        description="Ensemble confidence score in [0.5, 1.0].",
    )
    lstm_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Raw LSTM output probability."
    )
    xgb_prob: float = Field(
        ..., ge=0.0, le=1.0, description="Raw XGBoost output probability."
    )
    regime_label: str = Field(..., description="Market regime at prediction time.")
    model_version: str = Field(
        ..., min_length=1, description="Version string of the model."
    )

    # --- Outcome fields (filled by Verifier — initially None) ---
    actual_price: Optional[float] = Field(
        default=None,
        description="Actual close price at target_open_time (USD).",
    )
    actual_direction: Optional[int] = Field(
        default=None, description="Actual direction realised (-1, 0, 1)."
    )
    reward_score: Optional[float] = Field(
        default=None, description="Final RL reward value."
    )
    direction_correct: Optional[bool] = Field(
        default=None, description="Whether predicted_direction matched actual."
    )
    error_pct: Optional[float] = Field(
        default=None, description="Absolute percentage error vs price_at_prediction."
    )
    verified_at: Optional[int] = Field(
        default=None, description="Verifier processing timestamp — UTC epoch ms."
    )

    @field_validator("regime_label")
    @classmethod
    def validate_regime_label(cls, v: str) -> str:
        """Validate that regime_label is one of the five defined labels.

        Args:
            v: The regime label string.

        Returns:
            The validated regime label.

        Raises:
            ValueError: If the label is not in VALID_REGIMES.
        """
        if v not in VALID_REGIMES:
            raise ValueError(
                f"regime_label must be one of {sorted(VALID_REGIMES)}, got '{v}'"
            )
        return v

    @field_validator("actual_direction")
    @classmethod
    def validate_actual_direction(cls, v: Optional[int]) -> Optional[int]:
        """Validate that actual_direction, if set, is -1, 0, or 1.

        Args:
            v: The actual direction value, or None.

        Returns:
            The validated direction value.

        Raises:
            ValueError: If v is set but not in {-1, 0, 1}.
        """
        if v is not None and v not in VALID_DIRECTIONS:
            raise ValueError(
                f"actual_direction must be -1, 0, or 1, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_target_after_open(self) -> "PredictionRecord":
        """Ensure target_open_time > open_time.

        Returns:
            The validated record.

        Raises:
            ValueError: If target_open_time <= open_time.
        """
        if self.target_open_time <= self.open_time:
            raise ValueError(
                f"target_open_time ({self.target_open_time}) must be > "
                f"open_time ({self.open_time})"
            )
        return self


# ---------------------------------------------------------------------------
# RewardResult
# ---------------------------------------------------------------------------


class RewardResult(BaseModel):
    """Output of the ``compute_reward()`` function.

    All component scores are present regardless of whether the prediction was
    correct; ``direction_correct`` indicates the outcome.

    Args:
        reward_score: Final combined RL reward (product of all three scores).
        direction_score: +1.0 (correct), -1.0 (wrong), or +0.1 (flat/hedge).
        magnitude_score: exp(-decay × error_pct); in (0, 1].
        calibration_score: Confidence-weighted bonus/penalty in [-3.0, +2.0].
        error_pct: Absolute percentage price error (non-negative).
        direction_correct: True if predicted direction matched actual direction.
    """

    reward_score: float = Field(..., description="Final combined RL reward.")
    direction_score: float = Field(
        ..., description="+1.0 (correct), -1.0 (wrong), +0.1 (flat)."
    )
    magnitude_score: float = Field(
        ..., ge=0.0, le=1.0, description="Magnitude accuracy score in [0, 1]."
    )
    calibration_score: float = Field(
        ..., description="Confidence-weighted calibration bonus/penalty."
    )
    error_pct: float = Field(..., ge=0.0, description="Absolute percentage price error.")
    direction_correct: bool = Field(
        ..., description="True if predicted direction matched actual."
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "EPOCH_MS_MAX",
    "EPOCH_MS_MIN",
    "FUNDING_RATE_MAX",
    "FUNDING_RATE_MIN",
    "HORIZON_CANDLES",
    "MANDATORY_FEATURE_COLUMNS",
    "VALID_DIRECTIONS",
    "VALID_ERAS",
    "VALID_HORIZONS",
    "VALID_REGIMES",
    "CandleValidationResult",
    "FeatureRecord",
    "FundingRateRecord",
    "OHLCVRecord",
    "PredictionRecord",
    "RewardResult",
]
