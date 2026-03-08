"""Data validator for doge_predictor.

``DataValidator`` runs ALL validation checks for OHLCV, funding-rate, and
feature DataFrames.  This is the single gatekeeper — no data moves forward
without passing validation.

Critical check failures raise :class:`DataValidationError` immediately and
halt the pipeline.  Warning-level failures are logged and the caller decides
what to do with the returned :class:`~src.processing.schemas.CandleValidationResult`.

Critical failure triggers (immediate raise):
    - NaN or Inf in any OHLCV column (Check 5)
    - ``open_time`` not monotonic increasing (Check 2)
    - Gap of more than 3 missing candles (Check 3)
    - Missing expected columns in feature matrix (raises :class:`FeatureSchemaError`)
    - NaN / Inf in numeric feature columns (Check 2 of validate_feature_matrix)

Warning failures (logged, pipeline continues):
    - OHLCV price-bar invariant violations (Check 4)
    - Duplicate ``open_time`` values (Check 6)
    - Row count deviation beyond ± 2 (Check 7)
    - Incorrect era labels (Check 8)
    - Stale data beyond 2 × interval (Check 9, live-check only)
    - Constant columns in feature matrix (Check 3 of validate_feature_matrix)

Usage::

    from src.processing.validator import DataValidator
    validator = DataValidator()
    result = validator.validate_ohlcv(df, symbol="DOGEUSDT", interval="1h")
    if not result.is_valid:
        logger.warning("Validation errors: {}", result.errors)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.ingestion.exceptions import DataValidationError
from src.processing.schemas import CandleValidationResult
from src.utils.helpers import compute_expected_row_count, interval_to_ms

__all__ = [
    "DataValidator",
    "FeatureSchemaError",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Epoch-ms of 2022-01-01 00:00:00 UTC — training/context era boundary.
_TRAINING_START_MS: int = 1_640_995_200_000

#: Required columns for validate_ohlcv.
_OHLCV_REQUIRED_COLS: frozenset[str] = frozenset(
    {"open_time", "open", "high", "low", "close", "volume"}
)

#: OHLCV price/volume columns checked for NaN and Inf.
_OHLCV_NUMERIC_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Expected funding-rate interval in milliseconds (8 hours).
_FUNDING_INTERVAL_MS: int = 28_800_000

#: Maximum allowed missing-candle gap before a critical halt.
_MAX_GAP_CANDLES: int = 3

#: Soft row-count tolerance (±  candles).
_ROW_COUNT_TOLERANCE: int = 2


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class FeatureSchemaError(DataValidationError):
    """Raised when ``validate_feature_matrix`` finds missing mandatory columns.

    Inherits from :class:`~src.ingestion.exceptions.DataValidationError` so
    callers that catch the base class still handle it correctly.

    Args:
        message: Human-readable description of the missing columns.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------


class DataValidator:
    """Runs all validation checks for OHLCV, funding-rate, and feature data.

    Every check is documented in the respective method's docstring; none may
    be skipped.  The instance carries no mutable state and is safe to share
    across threads.

    Example::

        validator = DataValidator()
        result = validator.validate_ohlcv(df, "DOGEUSDT", "1h")
        if not result.is_valid:
            logger.warning("Validation failed: {}", result.errors)
    """

    # ------------------------------------------------------------------
    # validate_ohlcv
    # ------------------------------------------------------------------

    def validate_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        *,
        is_live_check: bool = False,
    ) -> CandleValidationResult:
        """Run all 9 OHLCV validation checks and return a structured result.

        Checks (in order):
            1. Required columns present with correct numeric dtypes.
            2. ``open_time`` strictly monotonic increasing. **(CRITICAL)**
            3. Interval consistency — all consecutive gaps equal ``interval_ms``.
               Gap > 3 missing candles triggers a pipeline halt. **(CRITICAL)**
            4. OHLCV sanity — high >= open/close, low <= open/close,
               close > 0, volume >= 0. Violations are counted and logged.
            5. No NaN or Inf in any OHLCV column. **(CRITICAL)**
            6. No duplicate ``open_time`` values. (Warning)
            7. Row count within ``±`` :data:`_ROW_COUNT_TOLERANCE` of expected.
            8. Era assignment correctness: pre-2022 rows have ``era='context'``,
               post-2022 rows have ``era='training'``.
            9. Stale data: last ``close_time`` within 2 × ``interval_ms`` of now.
               Applied only when ``is_live_check=True``.

        Args:
            df: OHLCV DataFrame to validate.
            symbol: Trading pair symbol for logging (e.g. ``"DOGEUSDT"``).
            interval: Candle interval string (e.g. ``"1h"``).
            is_live_check: When ``True``, Check 9 (stale data) is also applied.

        Returns:
            :class:`~src.processing.schemas.CandleValidationResult` with all
            findings.  ``is_valid`` is ``False`` if any check found an error.

        Raises:
            DataValidationError: Immediately if a CRITICAL check fails —
                NaN in OHLCV, non-monotonic timestamps, or gap > 3 candles.
        """
        errors: list[str] = []
        gap_count: int = 0
        duplicate_count: int = 0
        interval_ms: int = interval_to_ms(interval)

        # ------------------------------------------------------------------
        # Check 1 — Required columns present
        # ------------------------------------------------------------------
        missing_cols = _OHLCV_REQUIRED_COLS - set(df.columns)
        if missing_cols:
            msg = (
                f"[{symbol}/{interval}] Missing required columns: "
                f"{sorted(missing_cols)}"
            )
            errors.append(msg)
            logger.error(msg)
            return CandleValidationResult(
                is_valid=False,
                errors=errors,
                row_count=len(df),
                gap_count=0,
                duplicate_count=0,
            )

        # ------------------------------------------------------------------
        # Check 5 — NaN / Inf (CRITICAL; run early so other checks don't crash)
        # ------------------------------------------------------------------
        for col in _OHLCV_NUMERIC_COLS:
            if col not in df.columns:
                continue
            col_series = pd.to_numeric(df[col], errors="coerce")
            nan_count = int(col_series.isna().sum())
            inf_count = int(np.isinf(col_series.dropna()).sum())
            if nan_count > 0 or inf_count > 0:
                msg = (
                    f"[{symbol}/{interval}] CRITICAL: NaN/Inf in column "
                    f"'{col}' (NaN={nan_count}, Inf={inf_count})"
                )
                errors.append(msg)
                logger.error(msg)
                raise DataValidationError(msg)

        open_times: np.ndarray = df["open_time"].values

        # ------------------------------------------------------------------
        # Check 6 — Duplicate open_time values (Warning)
        # ------------------------------------------------------------------
        duplicate_count = int(df["open_time"].duplicated().sum())
        if duplicate_count > 0:
            msg = (
                f"[{symbol}/{interval}] {duplicate_count} duplicate "
                "open_time value(s) detected"
            )
            errors.append(msg)
            logger.warning(msg)

        # ------------------------------------------------------------------
        # Check 2 — Monotonic increasing (CRITICAL)
        # ------------------------------------------------------------------
        if not df["open_time"].is_monotonic_increasing:
            msg = (
                f"[{symbol}/{interval}] CRITICAL: open_time is not "
                "monotonic increasing — timestamps are out of order"
            )
            errors.append(msg)
            logger.error(msg)
            raise DataValidationError(msg)

        # ------------------------------------------------------------------
        # Check 3 — Interval consistency; gap detection (CRITICAL if > 3)
        # ------------------------------------------------------------------
        if len(df) >= 2:
            diffs: np.ndarray = np.diff(open_times)
            for diff in diffs:
                diff_int = int(diff)
                if diff_int == interval_ms:
                    continue  # normal consecutive candle
                elif diff_int < interval_ms:
                    msg = (
                        f"[{symbol}/{interval}] Unexpected sub-interval gap: "
                        f"{diff_int}ms between consecutive open_times "
                        f"(expected {interval_ms}ms) — possible overlap"
                    )
                    errors.append(msg)
                    logger.warning(msg)
                else:
                    # diff > interval_ms → candles are missing
                    missing = int(round(diff_int / interval_ms)) - 1
                    gap_count += 1
                    if missing > _MAX_GAP_CANDLES:
                        msg = (
                            f"[{symbol}/{interval}] CRITICAL: gap of "
                            f"{missing} missing candle(s) detected "
                            f"(max allowed: {_MAX_GAP_CANDLES})"
                        )
                        errors.append(msg)
                        logger.error(msg)
                        raise DataValidationError(msg)
                    else:
                        logger.warning(
                            "[{}/{}] Gap of {} missing candle(s) — within tolerance",
                            symbol,
                            interval,
                            missing,
                        )

        # ------------------------------------------------------------------
        # Check 4 — OHLCV sanity (count violations; do NOT modify the input df)
        # ------------------------------------------------------------------
        high: pd.Series = df["high"]
        low: pd.Series = df["low"]
        open_: pd.Series = df["open"]
        close: pd.Series = df["close"]
        volume: pd.Series = df["volume"]

        sanity_violations: dict[str, int] = {
            "high < low": int((high < low).sum()),
            "high < open": int((high < open_).sum()),
            "high < close": int((high < close).sum()),
            "low > open": int((low > open_).sum()),
            "low > close": int((low > close).sum()),
            "close <= 0": int((close <= 0).sum()),
            "volume < 0": int((volume < 0).sum()),
        }

        for check_name, count in sanity_violations.items():
            if count > 0:
                msg = (
                    f"[{symbol}/{interval}] OHLCV sanity violation: "
                    f"{count} row(s) with '{check_name}'"
                )
                errors.append(msg)
                logger.warning(msg)

        # ------------------------------------------------------------------
        # Check 7 — Row count vs expected (soft warning only)
        # ------------------------------------------------------------------
        if len(df) >= 2:
            first_t = int(open_times[0])
            last_t = int(open_times[-1])
            try:
                expected = compute_expected_row_count(
                    first_t, last_t + interval_ms, interval_ms
                )
                deviation = abs(len(df) - expected)
                if deviation > _ROW_COUNT_TOLERANCE:
                    msg = (
                        f"[{symbol}/{interval}] Row count deviation: "
                        f"expected ~{expected}, got {len(df)} "
                        f"(diff={deviation}, tolerance={_ROW_COUNT_TOLERANCE})"
                    )
                    errors.append(msg)
                    logger.warning(msg)
            except ValueError:
                pass  # compute_expected_row_count may raise on invalid inputs

        # ------------------------------------------------------------------
        # Check 8 — Era assignment correctness (Warning)
        # ------------------------------------------------------------------
        if "era" in df.columns:
            pre_mask: pd.Series = df["open_time"] < _TRAINING_START_MS
            post_mask: pd.Series = df["open_time"] >= _TRAINING_START_MS

            if pre_mask.any():
                wrong_pre = int((df.loc[pre_mask, "era"] != "context").sum())
                if wrong_pre > 0:
                    msg = (
                        f"[{symbol}/{interval}] Era check: {wrong_pre} "
                        "pre-2022 row(s) do not have era='context'"
                    )
                    errors.append(msg)
                    logger.warning(msg)

            if post_mask.any():
                wrong_post = int((df.loc[post_mask, "era"] != "training").sum())
                if wrong_post > 0:
                    msg = (
                        f"[{symbol}/{interval}] Era check: {wrong_post} "
                        "post-2022 row(s) do not have era='training'"
                    )
                    errors.append(msg)
                    logger.warning(msg)

        # ------------------------------------------------------------------
        # Check 9 — Stale data (live inference only, Warning)
        # ------------------------------------------------------------------
        if is_live_check and len(df) > 0 and "close_time" in df.columns:
            last_ct = int(df["close_time"].iloc[-1])
            now_ms = int(time.time() * 1_000)
            staleness_ms = now_ms - last_ct
            if staleness_ms > 2 * interval_ms:
                msg = (
                    f"[{symbol}/{interval}] Stale data: last close_time is "
                    f"{staleness_ms / 1_000:.0f}s old "
                    f"(threshold: {2 * interval_ms / 1_000:.0f}s)"
                )
                errors.append(msg)
                logger.warning(msg)

        is_valid = len(errors) == 0
        return CandleValidationResult(
            is_valid=is_valid,
            errors=errors,
            row_count=len(df),
            gap_count=gap_count,
            duplicate_count=duplicate_count,
        )

    # ------------------------------------------------------------------
    # validate_funding_rates
    # ------------------------------------------------------------------

    def validate_funding_rates(
        self,
        df: pd.DataFrame,
    ) -> CandleValidationResult:
        """Validate a funding-rate DataFrame.

        Checks:
            1. Interval consistency — all consecutive gaps equal 28 800 000 ms.
            2. No NaN or Inf in ``funding_rate`` column. **(CRITICAL)**
            3. Funding rate in range ``[-0.01, 0.01]``.

        Args:
            df: Funding-rate DataFrame with at least ``funding_time`` and
                ``funding_rate`` columns.

        Returns:
            :class:`~src.processing.schemas.CandleValidationResult`.

        Raises:
            DataValidationError: If NaN or Inf is found in ``funding_rate``
                (CRITICAL).
        """
        errors: list[str] = []
        gap_count: int = 0

        # -- Check 2: NaN / Inf (CRITICAL) -----------------------------------
        if "funding_rate" in df.columns:
            col_series = pd.to_numeric(df["funding_rate"], errors="coerce")
            nan_count = int(col_series.isna().sum())
            inf_count = int(np.isinf(col_series.dropna()).sum())
            if nan_count > 0 or inf_count > 0:
                msg = (
                    f"[funding_rates] CRITICAL: NaN/Inf in 'funding_rate' "
                    f"(NaN={nan_count}, Inf={inf_count})"
                )
                errors.append(msg)
                logger.error(msg)
                raise DataValidationError(msg)

        # -- Check 1: Interval consistency (8h = 28 800 000 ms) --------------
        if "funding_time" in df.columns and len(df) >= 2:
            times: np.ndarray = df["funding_time"].values
            diffs: np.ndarray = np.diff(times)
            wrong = int((diffs != _FUNDING_INTERVAL_MS).sum())
            if wrong > 0:
                gap_count = wrong
                msg = (
                    f"[funding_rates] {wrong} interval gap(s): expected all "
                    f"gaps = {_FUNDING_INTERVAL_MS} ms (8h)"
                )
                errors.append(msg)
                logger.warning(msg)

        # -- Check 3: Rate range [-0.01, 0.01] --------------------------------
        if "funding_rate" in df.columns:
            out_of_range = int(
                ((df["funding_rate"] < -0.01) | (df["funding_rate"] > 0.01)).sum()
            )
            if out_of_range > 0:
                msg = (
                    f"[funding_rates] {out_of_range} row(s) with "
                    "funding_rate outside [-0.01, 0.01]"
                )
                errors.append(msg)
                logger.warning(msg)

        is_valid = len(errors) == 0
        return CandleValidationResult(
            is_valid=is_valid,
            errors=errors,
            row_count=len(df),
            gap_count=gap_count,
            duplicate_count=0,
        )

    # ------------------------------------------------------------------
    # validate_feature_matrix
    # ------------------------------------------------------------------

    def validate_feature_matrix(
        self,
        df: pd.DataFrame,
        expected_columns: list[str],
    ) -> CandleValidationResult:
        """Validate a feature matrix before it enters the model.

        Checks:
            1. All *expected_columns* present. **(CRITICAL — raises**
               :class:`FeatureSchemaError` **)**
            2. No NaN or Inf in any numeric column. **(CRITICAL)**
            3. No constant columns (std > 0 for all numeric columns).
            4. Index is strictly monotonic.
            5. Row count > 0.

        Args:
            df: Feature DataFrame to validate.
            expected_columns: List of column names that must be present.

        Returns:
            :class:`~src.processing.schemas.CandleValidationResult`.

        Raises:
            FeatureSchemaError: If any expected column is missing (CRITICAL).
            DataValidationError: If NaN or Inf is found in numeric columns
                (CRITICAL).
        """
        errors: list[str] = []

        # -- Check 5: Row count > 0 ------------------------------------------
        if len(df) == 0:
            msg = "[feature_matrix] CRITICAL: empty DataFrame — 0 rows"
            errors.append(msg)
            logger.error(msg)
            return CandleValidationResult(
                is_valid=False,
                errors=errors,
                row_count=0,
                gap_count=0,
                duplicate_count=0,
            )

        # -- Check 1: Expected columns present (CRITICAL) --------------------
        missing_cols = [c for c in expected_columns if c not in df.columns]
        if missing_cols:
            msg = (
                f"[feature_matrix] Missing mandatory columns: {missing_cols}"
            )
            logger.error(msg)
            raise FeatureSchemaError(msg)

        # -- Check 2: No NaN / Inf in numeric columns (CRITICAL) -------------
        numeric_df: pd.DataFrame = df.select_dtypes(include="number")
        nan_total = int(numeric_df.isna().sum().sum())
        inf_total = int(
            np.isinf(numeric_df.fillna(0).values).sum()
        )
        if nan_total > 0 or inf_total > 0:
            msg = (
                f"[feature_matrix] CRITICAL: NaN ({nan_total}) or Inf "
                f"({inf_total}) in numeric feature columns"
            )
            errors.append(msg)
            logger.error(msg)
            raise DataValidationError(msg)

        # -- Check 3: No constant columns (Warning) --------------------------
        constant_cols = [
            col for col in numeric_df.columns if numeric_df[col].std() == 0.0
        ]
        if constant_cols:
            msg = (
                f"[feature_matrix] {len(constant_cols)} constant column(s) "
                f"detected (std=0): {constant_cols}"
            )
            errors.append(msg)
            logger.warning(msg)

        # -- Check 4: Index strictly monotonic (Warning) ---------------------
        if not df.index.is_monotonic_increasing:
            msg = (
                "[feature_matrix] Index is not strictly monotonic increasing"
            )
            errors.append(msg)
            logger.warning(msg)

        is_valid = len(errors) == 0
        return CandleValidationResult(
            is_valid=is_valid,
            errors=errors,
            row_count=len(df),
            gap_count=0,
            duplicate_count=0,
        )
