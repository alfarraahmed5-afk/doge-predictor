"""Feature pipeline — orchestrates all feature computation for DOGE prediction.

This module is the single entry point for building the full feature matrix.
It provides both a functional interface (``build_feature_matrix``) and an
object-oriented interface (``FeaturePipeline``) that adds the target column,
drops warmup rows, validates, and persists outputs to disk.

Pipeline stages (in order):
    1. price_indicators.py  — SMA / EMA / MACD / RSI / BB / ATR / Stoch / Ichimoku
    2. volume_indicators.py — OBV / VWAP / CMF / CVD / volume ratios
    3. lag_features.py      — log returns / momentum / rolling stats
    4. doge_specific.py     — 12 mandatory DOGE features (BTC corr, vol spike, etc.)
    5. funding_features.py  — funding rate / z-score / extreme flags
    6. htf_features.py      — 4h RSI / trend / BB%B; 1d trend / return; ATH distance
    7. Regime label merge   — join regimes Series on open_time
    8. Regime features      — 5 one-hot + ordinal encoding from regime labels
    9. Target column        — binary next-candle direction label (shift=-1)
    10. Warmup drop         — df.dropna() removes NaN warmup rows + last row
    11. Validation          — zero NaN/Inf, mandatory features present
    12. Min-rows assertion  — verify sufficient rows remain after warmup drop

Validation (after all stages):
    - Assert zero NaN / Inf in any feature column
    - Assert all mandatory feature names from CLAUDE.md Section 7 are present
    - Assert column list matches a canonical JSON manifest (if provided)

Lookahead audit:
    Every sub-module is independently verified (see module docstrings).
    This pipeline adds no new computations; it only concatenates results.

Target column lookahead note:
    ``target = close.pct_change().shift(-1) > 0`` is INTENTIONAL lookahead
    for the TARGET ONLY.  It represents the next candle's direction and is
    used only as the supervised learning label — never as a model input.
    Every FEATURE uses shift(+N) or no shift.  This is Rule A compliant.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DogeSettings, doge_settings
from src.features.doge_specific import DOGE_FEATURE_NAMES, compute_doge_features
from src.features.funding_features import FUNDING_FEATURE_NAMES, compute_funding_features
from src.features.htf_features import HTF_FEATURE_NAMES, compute_htf_features
from src.features.lag_features import compute_lag_features
from src.features.price_indicators import compute_price_indicators
from src.features.volume_indicators import compute_volume_indicators
from src.regimes.features import REGIME_FEATURE_KEYS, get_regime_features

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# All features that MUST be present in the final matrix (from CLAUDE.md §7)
MANDATORY_FEATURE_NAMES: frozenset[str] = frozenset(
    list(DOGE_FEATURE_NAMES)
    + list(FUNDING_FEATURE_NAMES)
    + list(HTF_FEATURE_NAMES)
    + list(REGIME_FEATURE_KEYS)
)

# Minimum number of rows needed after dropping the warmup window.
# Longest indicator: EMA200 needs ~200 bars; 4h guard adds another 4h.
_MIN_ROWS_AFTER_WARMUP: int = 300

# Columns that originate from the raw OHLCV input or are pipeline meta-columns.
# These are NOT model features and are excluded from feature_columns.json.
_PASSTHROUGH_COLS: frozenset[str] = frozenset({
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "era",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_base_vol",
    "taker_buy_quote_vol",
    "is_interpolated",
    "regime_label",
    "target",
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Append a binary ``target`` column to *df*.

    The target is 1 if the **next** candle's close is higher than the current
    candle's close, otherwise 0.  The last row always receives ``NaN`` because
    there is no future candle.

    LOOKAHEAD NOTE — this is the ONLY place ``shift(-1)`` is used, and it is
    intentional: the target IS future data (the supervised label), never a
    model input feature.  Every feature uses ``shift(+N)`` only.

    Args:
        df: Feature DataFrame containing a ``close`` column.

    Returns:
        Copy of *df* with a ``target`` column appended.  Values are 0.0, 1.0,
        or ``NaN`` (last row).

    Raises:
        ValueError: If ``close`` is not present in *df*.
    """
    if "close" not in df.columns:
        raise ValueError("add_target_column: 'close' column is required")

    out = df.copy()

    # pct_change().shift(-1): next candle's % return relative to current close.
    # Last row → NaN (no future candle).
    next_ret: pd.Series = out["close"].pct_change().shift(-1)

    # Convert to 0/1 float; preserve NaN at last row for dropna() to remove.
    target_arr: np.ndarray = np.where(
        next_ret.isna(),
        np.nan,
        (next_ret > 0).astype(float),
    )
    out["target"] = target_arr
    return out


def validate_feature_matrix(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate the feature matrix for NaN, Inf, and mandatory column presence.

    This is a lightweight post-pipeline check suitable for use at inference
    time (called after every candle close).  It does NOT re-compute features.

    Args:
        df: Feature matrix returned by :func:`build_feature_matrix` or
            :meth:`FeaturePipeline.compute_all_features`.
        expected_columns: Optional explicit list of column names that must all
            be present.  Used at inference to verify the live feature set
            matches the training feature set (loaded from
            ``feature_columns_{run_id}.json``).  If *None*, only mandatory
            features from CLAUDE.md Section 7 are checked.
        strict: If *True*, raise :class:`ValueError` on any validation
            failure.  If *False* (default), return a results dict and log
            warnings.

    Returns:
        Dict with keys:

            - ``"ok"`` (bool): True iff all checks pass.
            - ``"nan_cols"`` (list[str]): Columns with at least one NaN.
            - ``"inf_cols"`` (list[str]): Columns with at least one Inf.
            - ``"missing_mandatory"`` (list[str]): Mandatory features absent.
            - ``"missing_expected"`` (list[str]): Expected columns absent.
            - ``"n_rows"`` (int): Number of rows checked.

    Raises:
        ValueError: If *strict=True* and any check fails.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    nan_cols = [c for c in numeric_cols if df[c].isna().any()]
    inf_cols = [c for c in numeric_cols if np.isinf(df[c].to_numpy()).any()]
    missing_mandatory = sorted(MANDATORY_FEATURE_NAMES - set(df.columns))

    missing_expected: list[str] = []
    if expected_columns is not None:
        missing_expected = [c for c in expected_columns if c not in df.columns]

    ok = not nan_cols and not inf_cols and not missing_mandatory and not missing_expected

    if nan_cols:
        logger.warning("validate_feature_matrix: NaN in columns {}", nan_cols)
    if inf_cols:
        logger.warning("validate_feature_matrix: Inf in columns {}", inf_cols)
    if missing_mandatory:
        logger.warning(
            "validate_feature_matrix: missing mandatory features {}",
            missing_mandatory,
        )
    if missing_expected:
        logger.warning(
            "validate_feature_matrix: missing expected columns {}",
            missing_expected,
        )

    result: dict[str, Any] = {
        "ok": ok,
        "nan_cols": nan_cols,
        "inf_cols": inf_cols,
        "missing_mandatory": missing_mandatory,
        "missing_expected": missing_expected,
        "n_rows": len(df),
    }

    if strict and not ok:
        raise ValueError(
            f"validate_feature_matrix: validation failed — {result}"
        )

    return result


# ---------------------------------------------------------------------------
# FeaturePipeline — full 12-step pipeline with persistence
# ---------------------------------------------------------------------------


def _make_run_id() -> str:
    """Generate a unique run ID as ``YYYYMMDD_HHMMSS_<8-hex>``."""
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{now}_{short_uuid}"


class FeaturePipeline:
    """End-to-end feature pipeline with target column, validation, and persistence.

    Orchestrates all 12 pipeline steps in the canonical order defined in
    CLAUDE.md Section 4.3 / 7:

        1. price_indicators → 2. volume_indicators → 3. lag_features →
        4. doge_specific → 5. funding_features → 6. htf_features →
        7. merge regime labels → 8. regime features →
        9. add target column → 10. dropna → 11. validate → 12. assert min rows

    Attributes:
        run_id: Unique identifier for this pipeline run.  Embedded in output
            file names.

    Example::

        pipe = FeaturePipeline(output_dir=Path("data/features/primary"))
        df = pipe.compute_all_features(
            doge_1h, btc_1h, dogebtc_1h, funding, doge_4h, doge_1d, regimes
        )
    """

    def __init__(
        self,
        cfg: DogeSettings | None = None,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            cfg: DOGE-specific settings.  Defaults to the module-level
                singleton loaded from ``config/doge_settings.yaml``.
            output_dir: Directory where Parquet and JSON outputs are saved.
                If *None*, outputs are not persisted to disk.
            run_id: Identifier embedded in output file names.  Auto-generated
                from timestamp + UUID if *None*.
        """
        self._cfg: DogeSettings = cfg if cfg is not None else doge_settings
        self._output_dir: Path | None = output_dir
        self.run_id: str = run_id if run_id is not None else _make_run_id()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        doge_1h: pd.DataFrame,
        btc_1h: pd.DataFrame,
        dogebtc_1h: pd.DataFrame,
        funding: pd.DataFrame,
        doge_4h: pd.DataFrame,
        doge_1d: pd.DataFrame,
        regimes: pd.Series | None = None,
        min_rows_override: int | None = None,
        for_inference: bool = False,
    ) -> pd.DataFrame:
        """Run all 12 pipeline steps and return the validated feature matrix.

        Regime labels are joined on ``open_time`` (the Series index must
        contain ``open_time`` values in UTC epoch milliseconds).  Rows with
        unknown regime labels default to ``'RANGING_LOW_VOL'``.

        The ``target`` column uses ``close.pct_change().shift(-1) > 0``
        (next-candle direction).  The last row is always NaN and is removed
        by the warmup ``dropna()`` step.

        Args:
            doge_1h: DOGEUSDT 1h OHLCV DataFrame sorted ascending.
                Must contain ``open_time``, OHLCV columns.
            btc_1h: BTCUSDT 1h OHLCV DataFrame, same timestamp grid.
            dogebtc_1h: DOGEBTC 1h OHLCV DataFrame, same timestamp grid.
            funding: DOGEUSDT 8h funding rate DataFrame with columns
                ``timestamp_ms`` and ``funding_rate``.
            doge_4h: DOGEUSDT 4h OHLCV DataFrame.
            doge_1d: DOGEUSDT 1d OHLCV DataFrame.
            regimes: Regime label Series **indexed by open_time (int ms)**.
                Maps each candle's ``open_time`` to its regime label string.
            min_rows_override: Override for the minimum post-warmup row count.
                If *None*, uses ``cfg.walk_forward.min_training_rows``.
                Pass a smaller value for test/QG runs with limited fixture data.

        Returns:
            Validated feature DataFrame with all features + ``target`` column.
            Warmup rows and the final row (NaN target) are removed.

        Raises:
            ValueError: If any required column is missing, mandatory features
                are absent, validation fails, or insufficient rows remain.
        """
        n_input = len(doge_1h)
        original_cols: set[str] = set(doge_1h.columns)

        logger.info(
            "FeaturePipeline[{}]: starting on {} 1h rows",
            self.run_id,
            n_input,
        )

        # ------------------------------------------------------------------
        # Steps 1–6: feature computation (fixed order, no changes allowed)
        # ------------------------------------------------------------------
        out = compute_price_indicators(doge_1h, cfg=self._cfg)          # Step 1
        logger.debug("step 1 (price_indicators) done — {} cols", len(out.columns))

        out = compute_volume_indicators(out, cfg=self._cfg)               # Step 2
        logger.debug("step 2 (volume_indicators) done — {} cols", len(out.columns))

        out = compute_lag_features(out, cfg=self._cfg)                    # Step 3
        logger.debug("step 3 (lag_features) done — {} cols", len(out.columns))

        out = compute_doge_features(out, btc_1h, dogebtc_1h, cfg=self._cfg)  # Step 4
        logger.debug("step 4 (doge_specific) done — {} cols", len(out.columns))

        out = compute_funding_features(out, funding, cfg=self._cfg)       # Step 5
        logger.debug("step 5 (funding_features) done — {} cols", len(out.columns))

        out = compute_htf_features(out, doge_4h, doge_1d, cfg=self._cfg) # Step 6
        logger.debug("step 6 (htf_features) done — {} cols", len(out.columns))

        # ------------------------------------------------------------------
        # Step 7: merge regime labels (join on open_time index)
        # ------------------------------------------------------------------
        if regimes is not None:
            regime_map: dict[int, str] = dict(zip(regimes.index, regimes.values))
            out["regime_label"] = (
                out["open_time"].map(regime_map).fillna("RANGING_LOW_VOL")
            )
        else:
            # No precomputed labels — default to conservative fallback.
            # Regime classification at inference (Step 3) produces the
            # *current* regime label; the per-row column is not needed
            # for inference feature columns beyond the one-hot encoding.
            out["regime_label"] = "RANGING_LOW_VOL"
        logger.debug("step 7 (regime label merge) done")

        # ------------------------------------------------------------------
        # Step 8: compute regime features — one-hot + ordinal encoding
        # ------------------------------------------------------------------
        regime_feature_rows: list[dict[str, float]] = []
        for label in out["regime_label"]:
            try:
                regime_feature_rows.append(get_regime_features(str(label)))
            except ValueError:
                regime_feature_rows.append({k: 0.0 for k in REGIME_FEATURE_KEYS})

        regime_df = pd.DataFrame(regime_feature_rows, index=out.index)
        for col in REGIME_FEATURE_KEYS:
            out[col] = regime_df[col].to_numpy()

        logger.debug("step 8 (regime_features) done — {} cols", len(out.columns))

        # ------------------------------------------------------------------
        # Step 9: add target column (intentional shift(-1) — target only)
        # ------------------------------------------------------------------
        if not for_inference:
            out = add_target_column(out)
            logger.debug("step 9 (target column) done")
        else:
            logger.debug("step 9 skipped (for_inference=True — no target column needed)")

        # ------------------------------------------------------------------
        # Step 10: drop NaN rows (warmup + last row with NaN target)
        # ------------------------------------------------------------------
        n_dropped: int = 0
        if not for_inference:
            n_before_drop = len(out)

            # Restrict dropna to feature columns only.  Passthrough OHLCV
            # columns like ``quote_volume`` and ``num_trades`` may be NULL in
            # the SQLite bootstrap (Binance does not populate those fields for
            # standard klines) and are NOT model inputs.  Dropping on ALL
            # columns would silently eliminate every training row.
            _drop_subset: list[str] = [
                c for c in out.columns
                if c not in _PASSTHROUGH_COLS
                and pd.api.types.is_numeric_dtype(out[c])
            ]
            # Always include "target" in the subset so the final NaN target
            # row is still removed (target is in _PASSTHROUGH_COLS but we
            # need to drop the last row whose target is NaN).
            if "target" in out.columns and "target" not in _drop_subset:
                _drop_subset.append("target")

            out = out.dropna(subset=_drop_subset)
            n_dropped = n_before_drop - len(out)
            logger.info(
                "FeaturePipeline[{}]: step 10 — dropped {} warmup/target-NaN rows, {} remain",
                self.run_id,
                n_dropped,
                len(out),
            )
        else:
            logger.debug("step 10 skipped (for_inference=True — keeping all rows including warmup)")

        # ------------------------------------------------------------------
        # Determine feature columns (all numeric cols added by the pipeline)
        # ------------------------------------------------------------------
        feature_cols: list[str] = [
            c for c in out.columns
            if c not in original_cols
            and c not in _PASSTHROUGH_COLS
            and pd.api.types.is_numeric_dtype(out[c])
        ]

        # ------------------------------------------------------------------
        # Step 11: validate feature matrix
        # ------------------------------------------------------------------
        if not for_inference:
            # Validate only feature columns — passthrough OHLCV columns such
            # as quote_volume and num_trades may be NULL in the SQLite bootstrap
            # (Binance does not populate them for standard klines) and are not
            # model inputs, so NaN in those columns is expected and harmless.
            cols_to_validate = feature_cols + (["target"] if "target" in out.columns else [])
            validation_result = validate_feature_matrix(
                out[cols_to_validate],
                expected_columns=feature_cols,
                strict=False,
            )
            if not validation_result["ok"]:
                raise ValueError(
                    f"FeaturePipeline[{self.run_id}]: feature matrix validation failed "
                    f"— {validation_result}"
                )
            logger.debug("step 11 (validate_feature_matrix) PASS")
        else:
            logger.debug("step 11 skipped (for_inference=True — engine validates last row only)")

        # ------------------------------------------------------------------
        # Step 12: assert minimum post-warmup row count
        # ------------------------------------------------------------------
        if not for_inference:
            min_rows = (
                min_rows_override
                if min_rows_override is not None
                else self._cfg.walk_forward.min_training_rows
            )
            if len(out) < min_rows:
                raise ValueError(
                    f"FeaturePipeline[{self.run_id}]: insufficient rows after warmup — "
                    f"{len(out)} < {min_rows}.  "
                    f"Pass min_rows_override for small test datasets."
                )

        # ------------------------------------------------------------------
        # Final log
        # ------------------------------------------------------------------
        logger.info(
            "FeaturePipeline[{}]: complete — {} input → {} output rows "
            "({} dropped), {} feature cols",
            self.run_id,
            n_input,
            len(out),
            n_dropped,
            len(feature_cols),
        )

        # ------------------------------------------------------------------
        # Persist outputs (only if output_dir is set)
        # ------------------------------------------------------------------
        if self._output_dir is not None:
            try:
                parquet_path = self._save_parquet(out)
                json_path = self._save_feature_columns_json(feature_cols)
                logger.info(
                    "FeaturePipeline[{}]: saved Parquet → {}",
                    self.run_id,
                    parquet_path,
                )
                logger.info(
                    "FeaturePipeline[{}]: saved feature columns → {}",
                    self.run_id,
                    json_path,
                )
            except OSError as exc:
                logger.error(
                    "FeaturePipeline[{}]: failed to save outputs — {}",
                    self.run_id,
                    exc,
                )

        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_parquet(self, df: pd.DataFrame) -> Path:
        """Persist *df* to ``data/features/primary/features_{run_id}.parquet``.

        Args:
            df: Feature matrix to save.

        Returns:
            Absolute path to the saved Parquet file.

        Raises:
            OSError: If the directory cannot be created or the file cannot be written.
        """
        assert self._output_dir is not None  # guarded by caller
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"features_{self.run_id}.parquet"
        df.to_parquet(path, index=False)
        return path

    def _save_feature_columns_json(self, feature_cols: list[str]) -> Path:
        """Persist the feature column list to JSON for inference-time validation.

        The JSON file is saved next to the Parquet output as
        ``feature_columns_{run_id}.json``.  At inference time, this file is
        loaded and passed to :func:`validate_feature_matrix` as
        ``expected_columns``.

        Args:
            feature_cols: Ordered list of feature column names.

        Returns:
            Absolute path to the saved JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        assert self._output_dir is not None  # guarded by caller
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"feature_columns_{self.run_id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "run_id": self.run_id,
                    "n_features": len(feature_cols),
                    "feature_columns": feature_cols,
                },
                fh,
                indent=2,
            )
        return path


# ---------------------------------------------------------------------------
# Legacy functional API — kept for backward compatibility
# ---------------------------------------------------------------------------


def build_feature_matrix(
    doge_1h: pd.DataFrame,
    btc_1h: pd.DataFrame,
    dogebtc_1h: pd.DataFrame,
    funding_df: pd.DataFrame,
    doge_4h: pd.DataFrame,
    doge_1d: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    cfg: DogeSettings | None = None,
    drop_warmup: bool = False,
    warmup_rows: int = 250,
) -> pd.DataFrame:
    """Build the full feature matrix for DOGE prediction (functional API).

    Runs all six feature sub-modules in the canonical order defined in
    ``src/features/pipeline.py``.  Optionally drops warmup rows containing
    NaN from long-period rolling indicators.

    .. note::

        For new code prefer :class:`FeaturePipeline` which also adds the
        target column, validates, and persists outputs.  This function is
        retained for backward compatibility with existing tests and scripts.

    Args:
        doge_1h: DOGEUSDT 1h OHLCV DataFrame sorted ascending by
            ``open_time``.  Must contain OHLCV columns and ``open_time``.
        btc_1h: BTCUSDT 1h OHLCV DataFrame with the same timestamp grid.
        dogebtc_1h: DOGEBTC 1h OHLCV DataFrame with the same timestamp grid.
        funding_df: DOGEUSDT 8h funding rate DataFrame with columns
            ``timestamp_ms`` and ``funding_rate``.
        doge_4h: DOGEUSDT 4h OHLCV DataFrame (need not share a timestamp
            grid with doge_1h; the HTF guard handles alignment).
        doge_1d: DOGEUSDT 1d OHLCV DataFrame.
        regime_labels: Optional :class:`pd.Series` of regime label strings
            (``"TRENDING_BULL"`` etc.) with the same positional index as
            *doge_1h*.  If *None*, regime features are set to zero.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If
            *None* the module-level singleton is used.
        drop_warmup: If *True*, drop the first *warmup_rows* rows from the
            output.  Default *False* (return all rows including NaN warmup).
        warmup_rows: Number of initial rows to drop when *drop_warmup* is
            *True*.  Default 250 (covers EMA200 warmup + HTF guard).

    Returns:
        DataFrame with all feature columns appended to a copy of *doge_1h*.
        The column order is: original OHLCV | price | volume | lag |
        doge-specific | funding | HTF | regime.

    Raises:
        ValueError: If any required input column is missing, or if mandatory
            features are absent from the output.
    """
    if cfg is None:
        cfg = doge_settings

    logger.info(
        "build_feature_matrix: starting pipeline on {} 1h rows",
        len(doge_1h),
    )

    # Stage 1 — Price indicators
    out = compute_price_indicators(doge_1h, cfg=cfg)
    logger.debug("pipeline stage 1 (price_indicators) done — {} cols", len(out.columns))

    # Stage 2 — Volume indicators
    out = compute_volume_indicators(out, cfg=cfg)
    logger.debug("pipeline stage 2 (volume_indicators) done — {} cols", len(out.columns))

    # Stage 3 — Lag / momentum / rolling stat features
    out = compute_lag_features(out, cfg=cfg)
    logger.debug("pipeline stage 3 (lag_features) done — {} cols", len(out.columns))

    # Stage 4 — DOGE-specific features
    out = compute_doge_features(out, btc_1h, dogebtc_1h, cfg=cfg)
    logger.debug("pipeline stage 4 (doge_specific) done — {} cols", len(out.columns))

    # Stage 5 — Funding rate features
    out = compute_funding_features(out, funding_df, cfg=cfg)
    logger.debug("pipeline stage 5 (funding_features) done — {} cols", len(out.columns))

    # Stage 6 — HTF features
    out = compute_htf_features(out, doge_4h, doge_1d, cfg=cfg)
    logger.debug("pipeline stage 6 (htf_features) done — {} cols", len(out.columns))

    # Stage 7 — Regime features (positional alignment)
    regime_feature_rows: list[dict[str, Any]] = []
    if regime_labels is not None:
        for label in regime_labels:
            try:
                regime_feature_rows.append(get_regime_features(str(label)))
            except ValueError:
                regime_feature_rows.append(
                    {k: 0 for k in REGIME_FEATURE_KEYS}
                )
    else:
        logger.warning("build_feature_matrix: no regime_labels provided — regime features = 0")
        regime_feature_rows = [{k: 0 for k in REGIME_FEATURE_KEYS}] * len(out)

    regime_df = pd.DataFrame(regime_feature_rows, index=out.index)
    for col in REGIME_FEATURE_KEYS:
        out[col] = regime_df[col].to_numpy()

    logger.debug("pipeline stage 7 (regime_features) done — {} cols", len(out.columns))

    # Optional warmup drop
    if drop_warmup:
        if len(out) > warmup_rows:
            out = out.iloc[warmup_rows:].copy()
            logger.info(
                "build_feature_matrix: dropped {} warmup rows, {} remain",
                warmup_rows,
                len(out),
            )
        else:
            logger.warning(
                "build_feature_matrix: drop_warmup=True but only {} rows — none dropped",
                len(out),
            )

    # Mandatory feature validation
    missing = MANDATORY_FEATURE_NAMES - set(out.columns)
    if missing:
        raise ValueError(
            f"build_feature_matrix: mandatory features missing from output: {sorted(missing)}"
        )

    logger.info(
        "build_feature_matrix: complete — {} rows x {} columns",
        len(out),
        len(out.columns),
    )
    return out
