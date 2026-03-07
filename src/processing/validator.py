"""Data validation module for doge_predictor.

Defines Pandera schema contracts for every DataFrame that flows through the
pipeline. ALL validation checks run here — no ad-hoc checking elsewhere.

Schemas defined:
    - OHLCVSchema         : raw Binance 1h kline rows (all symbols)
    - FundingRateSchema   : raw 8h funding rate rows
    - ProcessedOHLCVSchema: cleaned & typed OHLCV after cleaner.py
    - AlignedSchema       : multi-symbol aligned DataFrame
    - FeatureSchema       : fully-featured DataFrame fed into models

Usage::

    from src.processing.validator import OHLCVSchema, validate

    df = validate(raw_df, OHLCVSchema)   # raises SchemaError on violation
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import Series

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SchemaT = TypeVar("SchemaT", bound=pa.DataFrameSchema)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EPOCH_MS_MIN: int = 1_000_000_000_000   # 2001-09-09 — sanity lower bound
EPOCH_MS_MAX: int = 9_999_999_999_999   # year 2286 — sanity upper bound


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, *, lazy: bool = True) -> pd.DataFrame:
    """Validate *df* against *schema* and return the validated DataFrame.

    Args:
        df: DataFrame to validate.
        schema: Pandera schema to validate against.
        lazy: If True, collect all errors before raising (default True).
            Set False for fail-fast behaviour.

    Returns:
        The validated (and coerced) DataFrame.

    Raises:
        pandera.errors.SchemaErrors: If lazy=True and any check fails.
        pandera.errors.SchemaError:  If lazy=False and a check fails.
    """
    try:
        validated = schema.validate(df, lazy=lazy)
        logger.debug("Validation passed: {} rows, schema={}", len(df), schema.name)
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Validation FAILED (schema={}): {} error(s)\n{}",
            schema.name,
            len(exc.failure_cases),
            exc.failure_cases.to_string(),
        )
        raise
    except pa.errors.SchemaError as exc:
        logger.error("Validation FAILED (schema={}): {}", schema.name, exc)
        raise


# ---------------------------------------------------------------------------
# Schema 1 — Raw Binance OHLCV (klines endpoint)
#
# Binance GET /api/v3/klines response columns (after DataFrame construction):
#   open_time, open, high, low, close, volume, close_time,
#   quote_asset_volume, number_of_trades,
#   taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
# ---------------------------------------------------------------------------

RawOHLCVSchema = pa.DataFrameSchema(
    name="RawOHLCVSchema",
    columns={
        "open_time": pa.Column(
            int,
            checks=[
                pa.Check.ge(EPOCH_MS_MIN, error="open_time below minimum epoch-ms"),
                pa.Check.le(EPOCH_MS_MAX, error="open_time above maximum epoch-ms"),
            ],
            description="Candle open timestamp — UTC epoch milliseconds (int).",
        ),
        "open": pa.Column(
            float,
            checks=pa.Check.gt(0, error="open must be > 0"),
            coerce=True,
            description="Open price (USD).",
        ),
        "high": pa.Column(
            float,
            checks=pa.Check.gt(0, error="high must be > 0"),
            coerce=True,
            description="High price (USD).",
        ),
        "low": pa.Column(
            float,
            checks=pa.Check.gt(0, error="low must be > 0"),
            coerce=True,
            description="Low price (USD).",
        ),
        "close": pa.Column(
            float,
            checks=pa.Check.gt(0, error="close must be > 0"),
            coerce=True,
            description="Close price (USD).",
        ),
        "volume": pa.Column(
            float,
            checks=pa.Check.ge(0, error="volume must be >= 0"),
            coerce=True,
            description="Base asset volume (DOGE).",
        ),
        "close_time": pa.Column(
            int,
            checks=[
                pa.Check.ge(EPOCH_MS_MIN, error="close_time below minimum epoch-ms"),
                pa.Check.le(EPOCH_MS_MAX, error="close_time above maximum epoch-ms"),
            ],
            description="Candle close timestamp — UTC epoch milliseconds (int).",
        ),
        "quote_asset_volume": pa.Column(
            float,
            checks=pa.Check.ge(0),
            coerce=True,
            nullable=False,
            description="Quote asset volume (USDT).",
        ),
        "number_of_trades": pa.Column(
            int,
            checks=pa.Check.ge(0),
            coerce=True,
            description="Number of trades in candle.",
        ),
        "taker_buy_base_asset_volume": pa.Column(
            float,
            checks=pa.Check.ge(0),
            coerce=True,
            description="Taker buy base asset (DOGE) volume.",
        ),
        "taker_buy_quote_asset_volume": pa.Column(
            float,
            checks=pa.Check.ge(0),
            coerce=True,
            description="Taker buy quote asset (USDT) volume.",
        ),
    },
    checks=[
        pa.Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high < low detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["high"] >= df["open"]).all(),
            error="high < open detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["high"] >= df["close"]).all(),
            error="high < close detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["low"] <= df["open"]).all(),
            error="low > open detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["low"] <= df["close"]).all(),
            error="low > close detected in one or more rows",
        ),
        pa.Check(
            lambda df: (df["close_time"] > df["open_time"]).all(),
            error="close_time must be after open_time",
        ),
        pa.Check(
            lambda df: df["open_time"].is_unique,
            error="Duplicate open_time values detected — gaps or overlap in klines",
        ),
    ],
    coerce=True,
    strict=False,   # allow extra columns (e.g. 'ignore') to pass through
    ordered=False,
)


# ---------------------------------------------------------------------------
# Schema 2 — Raw Binance Funding Rate
#
# Binance GET /fapi/v1/fundingRate response columns:
#   symbol, fundingTime, fundingRate, markPrice
# ---------------------------------------------------------------------------

RawFundingRateSchema = pa.DataFrameSchema(
    name="RawFundingRateSchema",
    columns={
        "symbol": pa.Column(
            str,
            checks=pa.Check.isin(["DOGEUSDT"], error="Only DOGEUSDT funding rates expected"),
            description="Trading pair symbol.",
        ),
        "funding_time": pa.Column(
            int,
            checks=[
                pa.Check.ge(EPOCH_MS_MIN),
                pa.Check.le(EPOCH_MS_MAX),
            ],
            description="Funding settlement timestamp — UTC epoch milliseconds.",
        ),
        "funding_rate": pa.Column(
            float,
            checks=[
                pa.Check.ge(-0.01, error="funding_rate below -1% is anomalous"),
                pa.Check.le(0.01, error="funding_rate above +1% is anomalous"),
            ],
            coerce=True,
            description="Funding rate value (e.g. 0.0001 = 0.01% per 8h).",
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["funding_time"].is_unique,
            error="Duplicate funding_time values detected",
        ),
    ],
    coerce=True,
    strict=False,
)


# ---------------------------------------------------------------------------
# Schema 3 — Processed OHLCV (after cleaner.py)
#
# This schema enforces the cleaned, typed representation used downstream.
# Timestamps are index or column — both forms validated here.
# ---------------------------------------------------------------------------

ProcessedOHLCVSchema = pa.DataFrameSchema(
    name="ProcessedOHLCVSchema",
    columns={
        "open_time": pa.Column(
            int,
            checks=[pa.Check.ge(EPOCH_MS_MIN), pa.Check.le(EPOCH_MS_MAX)],
            description="UTC epoch milliseconds — integer.",
        ),
        "open": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "high": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "low": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "close": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "close_time": pa.Column(
            int,
            checks=[pa.Check.ge(EPOCH_MS_MIN), pa.Check.le(EPOCH_MS_MAX)],
        ),
        "quote_asset_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "number_of_trades": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        "taker_buy_base_asset_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "taker_buy_quote_asset_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        "era": pa.Column(
            str,
            checks=pa.Check.isin(["context", "training"], error="era must be 'context' or 'training'"),
            description="'context' for pre-2022, 'training' for post-2022. Never mixed in training folds.",
        ),
    },
    checks=[
        pa.Check(lambda df: (df["high"] >= df["low"]).all(), error="high < low"),
        pa.Check(lambda df: (df["high"] >= df["open"]).all(), error="high < open"),
        pa.Check(lambda df: (df["high"] >= df["close"]).all(), error="high < close"),
        pa.Check(lambda df: (df["low"] <= df["open"]).all(), error="low > open"),
        pa.Check(lambda df: (df["low"] <= df["close"]).all(), error="low > close"),
        pa.Check(lambda df: df["open_time"].is_unique, error="Duplicate open_time"),
        pa.Check(
            lambda df: df["open_time"].is_monotonic_increasing,
            error="open_time is not sorted ascending",
        ),
        pa.Check(
            lambda df: (df["close_time"] > df["open_time"]).all(),
            error="close_time must be after open_time",
        ),
        pa.Check(
            lambda df: df.isnull().sum().sum() == 0,
            error="Null values found in ProcessedOHLCV — run cleaner first",
        ),
    ],
    coerce=False,   # types must already be correct after cleaner
    strict=True,    # no unexpected columns allowed downstream
)


# ---------------------------------------------------------------------------
# Schema 4 — Aligned multi-symbol DataFrame (after aligner.py)
#
# Columns are prefixed: doge_, btc_, dogebtc_
# All timestamps must be identical across symbols.
# ---------------------------------------------------------------------------

AlignedSchema = pa.DataFrameSchema(
    name="AlignedSchema",
    columns={
        "open_time": pa.Column(int, checks=[pa.Check.ge(EPOCH_MS_MIN), pa.Check.le(EPOCH_MS_MAX)]),
        # DOGE columns
        "doge_open": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "doge_high": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "doge_low": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "doge_close": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "doge_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        # BTC columns
        "btc_open": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "btc_high": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "btc_low": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "btc_close": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "btc_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        # DOGEBTC columns
        "dogebtc_open": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "dogebtc_high": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "dogebtc_low": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "dogebtc_close": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "dogebtc_volume": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
        # Funding rate (forward-filled from 8h to 1h)
        "funding_rate": pa.Column(
            float,
            nullable=True,   # nullable at extremes of history before funding data starts
            description="8h funding rate forward-filled to 1h cadence.",
        ),
        "era": pa.Column(str, checks=pa.Check.isin(["context", "training"])),
    },
    checks=[
        pa.Check(lambda df: df["open_time"].is_unique, error="Duplicate open_time in aligned df"),
        pa.Check(
            lambda df: df["open_time"].is_monotonic_increasing,
            error="open_time not sorted ascending in aligned df",
        ),
        pa.Check(
            lambda df: (df["doge_high"] >= df["doge_low"]).all(),
            error="DOGE: high < low",
        ),
        pa.Check(
            lambda df: (df["btc_high"] >= df["btc_low"]).all(),
            error="BTC: high < low",
        ),
    ],
    coerce=False,
    strict=False,   # feature columns will be added; don't reject them here
)


# ---------------------------------------------------------------------------
# Schema 5 — Feature DataFrame (after feature pipeline, before model)
#
# Validates the mandatory DOGE-specific features and absence of NaN/Inf.
# Exact column list will be frozen to feature_columns.json after first run.
# ---------------------------------------------------------------------------

# Mandatory feature columns from CLAUDE.md Section 7
MANDATORY_FEATURE_COLUMNS: list[str] = [
    # DOGE-BTC correlation
    "doge_btc_corr_12h",
    "doge_btc_corr_24h",
    "doge_btc_corr_7d",
    # DOGEBTC momentum
    "dogebtc_mom_6h",
    "dogebtc_mom_24h",
    "dogebtc_mom_48h",
    # Volume
    "volume_ratio",
    "volume_spike_flag",
    # Funding
    "funding_rate",
    "funding_rate_zscore",
    "funding_extreme_long",
    "funding_extreme_short",
    # HTF
    "htf_4h_rsi",
    "htf_4h_trend",
    "htf_4h_bb_pctb",
    "htf_1d_trend",
    "htf_1d_return",
    # DOGE-specific
    "ath_distance",
    "distance_to_round_pct",
    "at_round_number_flag",
    "nearest_round_level",
]

# Build feature schema dynamically from mandatory column list
_mandatory_columns: dict[str, pa.Column] = {
    col: pa.Column(
        float,
        nullable=False,
        description=f"Mandatory DOGE feature: {col}",
        # Allow any finite float; NaN/Inf caught by global check below
    )
    for col in MANDATORY_FEATURE_COLUMNS
    if col not in {"volume_spike_flag", "funding_extreme_long", "funding_extreme_short",
                   "at_round_number_flag", "htf_4h_trend", "htf_1d_trend"}
}

# Binary flag columns — must be 0 or 1
_flag_columns: dict[str, pa.Column] = {
    col: pa.Column(
        int,
        checks=pa.Check.isin([0, 1], error=f"{col} must be 0 or 1"),
        nullable=False,
        coerce=True,
    )
    for col in [
        "volume_spike_flag",
        "funding_extreme_long",
        "funding_extreme_short",
        "at_round_number_flag",
        "htf_4h_trend",
        "htf_1d_trend",
    ]
}

FeatureSchema = pa.DataFrameSchema(
    name="FeatureSchema",
    columns={
        "open_time": pa.Column(
            int,
            checks=[pa.Check.ge(EPOCH_MS_MIN), pa.Check.le(EPOCH_MS_MAX)],
        ),
        "era": pa.Column(str, checks=pa.Check.isin(["context", "training"])),
        "regime_label": pa.Column(
            str,
            checks=pa.Check.isin(
                ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"],
                error="regime_label must be one of the 5 defined regimes",
            ),
            nullable=False,
        ),
        **_mandatory_columns,
        **_flag_columns,
    },
    checks=[
        pa.Check(lambda df: df["open_time"].is_unique, error="Duplicate open_time in feature df"),
        pa.Check(
            lambda df: df["open_time"].is_monotonic_increasing,
            error="open_time not sorted ascending in feature df",
        ),
        pa.Check(
            lambda df: ~df.select_dtypes("number").isin([float("inf"), float("-inf")]).any().any(),
            error="Inf values found in feature DataFrame — check feature computation",
        ),
        pa.Check(
            lambda df: df.select_dtypes("number").isnull().sum().sum() == 0,
            error="NaN values found in feature DataFrame — run forward-fill / imputation first",
        ),
        pa.Check(
            lambda df: (df["doge_btc_corr_24h"] >= -1.0).all() and (df["doge_btc_corr_24h"] <= 1.0).all(),
            error="doge_btc_corr_24h out of [-1, 1] bounds",
        ),
        pa.Check(
            lambda df: (df["volume_ratio"] >= 0).all(),
            error="volume_ratio must be >= 0",
        ),
    ],
    coerce=False,
    strict=False,   # many more feature columns exist; only mandatories enforced
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "validate",
    "RawOHLCVSchema",
    "RawFundingRateSchema",
    "ProcessedOHLCVSchema",
    "AlignedSchema",
    "FeatureSchema",
    "MANDATORY_FEATURE_COLUMNS",
]
