"""Pandera DataFrame schema contracts for doge_predictor.

Defines structural and statistical validation schemas for the three primary
DataFrame types in the pipeline. These schemas complement the Pydantic record
models in ``schemas.py``; they operate on whole DataFrames rather than
individual rows.

Schemas defined:
    - OHLCVSchema       : validates any OHLCV pandas DataFrame
    - FeatureSchema     : validates the feature matrix (UTC-indexed)
    - FundingRateSchema : validates the funding-rate DataFrame

Usage::

    from src.processing.df_schemas import OHLCVSchema, validate_df
    validated = validate_df(raw_df, OHLCVSchema)

Notes:
    - All timestamps are UTC epoch milliseconds (``int``).
    - FeatureSchema expects a DatetimeTZDtype UTC *index*, not a column.
    - FundingRateSchema enforces the exact 8h (28_800_000 ms) cadence.
    - Import ``MANDATORY_FEATURE_COLUMNS`` from here to keep the list in one
      place; it is re-exported from ``schemas.py`` for backward compat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Series

from src.processing.schemas import (
    EPOCH_MS_MAX,
    EPOCH_MS_MIN,
    MANDATORY_FEATURE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Exact milliseconds between funding-rate observations (8 hours).
FUNDING_RATE_INTERVAL_MS: int = 28_800_000

#: Funding rate absolute bounds (mirrors schemas.py).
FUNDING_RATE_MIN: float = -0.01
FUNDING_RATE_MAX: float = 0.01


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def validate_df(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    *,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate *df* against *schema* and return the validated DataFrame.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to validate against.
        lazy: If ``True`` (default), collect all errors before raising.
            Set ``False`` for fail-fast behaviour.

    Returns:
        The validated (and coerced) DataFrame.

    Raises:
        pandera.errors.SchemaErrors: If ``lazy=True`` and any check fails.
        pandera.errors.SchemaError:  If ``lazy=False`` and a check fails.
    """
    try:
        validated = schema.validate(df, lazy=lazy)
        logger.debug(
            "df_schemas.validate_df passed: {} rows, schema={}",
            len(df),
            schema.name,
        )
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "df_schemas.validate_df FAILED (schema={}): {} error(s)\n{}",
            schema.name,
            len(exc.failure_cases),
            exc.failure_cases.to_string(),
        )
        raise
    except pa.errors.SchemaError as exc:
        logger.error(
            "df_schemas.validate_df FAILED (schema={}): {}",
            schema.name,
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Schema 1 — OHLCVSchema
#
# Validates a processed OHLCV DataFrame (any symbol, any interval).
# open_time must be a column (int, UTC ms) — strictly monotonically increasing.
# close_time is included but optional; if present it must be > open_time.
# ---------------------------------------------------------------------------

OHLCVSchema = pa.DataFrameSchema(
    name="OHLCVSchema",
    columns={
        "open_time": pa.Column(
            int,
            checks=[
                pa.Check.ge(EPOCH_MS_MIN, error="open_time below minimum epoch-ms"),
                pa.Check.le(EPOCH_MS_MAX, error="open_time above maximum epoch-ms"),
            ],
            nullable=False,
            description="Candle open timestamp — UTC epoch milliseconds.",
        ),
        "open": pa.Column(
            float,
            checks=pa.Check.gt(0.0, error="open must be > 0"),
            nullable=False,
            coerce=True,
            description="Open price.",
        ),
        "high": pa.Column(
            float,
            checks=pa.Check.gt(0.0, error="high must be > 0"),
            nullable=False,
            coerce=True,
            description="High price.",
        ),
        "low": pa.Column(
            float,
            checks=pa.Check.gt(0.0, error="low must be > 0"),
            nullable=False,
            coerce=True,
            description="Low price.",
        ),
        "close": pa.Column(
            float,
            checks=pa.Check.gt(0.0, error="close must be > 0"),
            nullable=False,
            coerce=True,
            description="Close price.",
        ),
        "volume": pa.Column(
            float,
            checks=pa.Check.ge(0.0, error="volume must be >= 0"),
            nullable=False,
            coerce=True,
            description="Base asset volume.",
        ),
    },
    checks=[
        # Monotonicity — open_time strictly increasing
        pa.Check(
            lambda df: df["open_time"].is_monotonic_increasing,
            error="open_time is not strictly monotonically increasing",
        ),
        # Uniqueness
        pa.Check(
            lambda df: df["open_time"].is_unique,
            error="Duplicate open_time values detected",
        ),
        # OHLC price-bar invariants (row-level)
        pa.Check(
            lambda df: (df["high"] >= df["open"]).all(),
            error="high < open detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["high"] >= df["close"]).all(),
            error="high < close detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high < low detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["low"] <= df["open"]).all(),
            error="low > open detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["low"] <= df["close"]).all(),
            error="low > close detected in one or more rows",
        ),
        # NaN guard (all required columns)
        pa.Check(
            lambda df: not df[["open_time", "open", "high", "low", "close", "volume"]]
            .isnull()
            .any()
            .any(),
            error="NaN values found in required OHLCV columns",
        ),
        # Inf guard
        pa.Check(
            lambda df: not df.select_dtypes(include="number")
            .isin([float("inf"), float("-inf")])
            .any()
            .any(),
            error="Inf values found in OHLCV DataFrame",
        ),
    ],
    coerce=True,
    strict=False,  # allow extra columns (e.g. close_time, era, symbol)
    ordered=False,
)


# ---------------------------------------------------------------------------
# Schema 2 — FeatureSchema
#
# Validates the feature matrix that is fed into model training and inference.
# The DataFrame MUST have a DatetimeTZDtype(tz="UTC") index named "open_time".
# All 21 mandatory DOGE features must be present.
# No column may be constant (std > 0 for all numeric columns).
# ---------------------------------------------------------------------------

# Dynamically build column specs for the mandatory feature columns.
# Binary flag columns must be {0, 1}; all others are unrestricted finite floats.
_BINARY_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "volume_spike_flag",
        "funding_extreme_long",
        "funding_extreme_short",
        "at_round_number_flag",
        "htf_4h_trend",
        "htf_1d_trend",
    }
)

_feature_columns: dict[str, pa.Column] = {}

for _col in MANDATORY_FEATURE_COLUMNS:
    if _col in _BINARY_FEATURE_COLUMNS:
        _feature_columns[_col] = pa.Column(
            float,
            checks=pa.Check.isin([0.0, 1.0], error=f"{_col} must be 0 or 1"),
            nullable=False,
            coerce=True,
            description=f"Binary flag feature: {_col}.",
        )
    else:
        _feature_columns[_col] = pa.Column(
            float,
            nullable=False,
            coerce=True,
            description=f"Mandatory DOGE feature: {_col}.",
        )

FeatureSchema = pa.DataFrameSchema(
    name="FeatureSchema",
    columns=_feature_columns,
    index=pa.Index(
        dtype=pd.DatetimeTZDtype(tz="UTC"),
        checks=[
            pa.Check(
                lambda idx: idx.is_monotonic_increasing,
                error="Feature DataFrame index is not monotonically increasing",
            ),
            pa.Check(
                lambda idx: not idx.isna().any(),
                error="Feature DataFrame index contains NaT values",
            ),
        ],
        name="open_time",
        description="UTC-aware DatetimeTZDtype index (open_time).",
    ),
    checks=[
        # NaN guard
        pa.Check(
            lambda df: df.select_dtypes(include="number").isnull().sum().sum() == 0,
            error="NaN values found in feature DataFrame — run imputation first",
        ),
        # Inf guard
        pa.Check(
            lambda df: not df.select_dtypes(include="number")
            .isin([float("inf"), float("-inf")])
            .any()
            .any(),
            error="Inf values found in feature DataFrame — check feature computation",
        ),
        # No constant columns (std > 0 for all numeric columns).
        # fillna(1.0) handles edge case of a single-row DataFrame where std is NaN.
        pa.Check(
            lambda df: (
                df.select_dtypes(include="number")
                .std(ddof=1)
                .fillna(1.0)
                > 0
            ).all(),
            error="Constant column detected (std == 0) — feature is non-informative",
        ),
    ],
    # coerce=False at the schema level so the DatetimeTZDtype UTC index is
    # validated STRICTLY (tz-naive indices are rejected, not silently coerced).
    # Individual columns use coerce=True in their Column() definitions above.
    coerce=False,
    strict=False,  # allow extra feature columns beyond the mandatory set
    ordered=False,
)


# ---------------------------------------------------------------------------
# Schema 3 — FundingRateSchema
#
# Validates the funding rate time series (8h cadence).
# timestamp_ms must be strictly monotonically increasing.
# Consecutive row deltas must equal exactly 28_800_000 ms (8h).
# ---------------------------------------------------------------------------

FundingRateSchema = pa.DataFrameSchema(
    name="FundingRateSchema",
    columns={
        "timestamp_ms": pa.Column(
            int,
            checks=[
                pa.Check.ge(EPOCH_MS_MIN, error="timestamp_ms below minimum epoch-ms"),
                pa.Check.le(EPOCH_MS_MAX, error="timestamp_ms above maximum epoch-ms"),
            ],
            nullable=False,
            description="Funding settlement timestamp — UTC epoch milliseconds.",
        ),
        "funding_rate": pa.Column(
            float,
            checks=[
                pa.Check.ge(
                    FUNDING_RATE_MIN,
                    error="funding_rate below -1% is anomalous",
                ),
                pa.Check.le(
                    FUNDING_RATE_MAX,
                    error="funding_rate above +1% is anomalous",
                ),
            ],
            nullable=False,
            coerce=True,
            description="Funding rate value (e.g. 0.0001 = 0.01% per 8h).",
        ),
    },
    checks=[
        # Monotonicity
        pa.Check(
            lambda df: df["timestamp_ms"].is_monotonic_increasing,
            error="timestamp_ms is not strictly monotonically increasing",
        ),
        # Uniqueness
        pa.Check(
            lambda df: df["timestamp_ms"].is_unique,
            error="Duplicate timestamp_ms values detected",
        ),
        # Exactly 8h between consecutive rows (skip check if <= 1 row)
        pa.Check(
            lambda df: (
                len(df) <= 1
                or (df["timestamp_ms"].diff().dropna() == FUNDING_RATE_INTERVAL_MS).all()
            ),
            error=(
                f"Funding rate interval is not exactly {FUNDING_RATE_INTERVAL_MS} ms "
                f"(8h) between all consecutive rows"
            ),
        ),
    ],
    coerce=True,
    strict=False,  # allow extra columns (symbol, mark_price, etc.)
    ordered=False,
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "FUNDING_RATE_INTERVAL_MS",
    "FUNDING_RATE_MAX",
    "FUNDING_RATE_MIN",
    "FeatureSchema",
    "FundingRateSchema",
    "OHLCVSchema",
    "validate_df",
]
