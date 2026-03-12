"""Per-regime XGBoost training via walk-forward cross-validation.

This module trains one :class:`~src.models.xgb_model.XGBoostModel` per market
regime and optionally archives each model to MLflow.

RULE B and RULE C (CLAUDE.md §3.2) are enforced on every fold via
:class:`~src.training.scaler.FoldScaler` and
:class:`~src.training.walk_forward.WalkForwardCV` respectively.

Build order (CLAUDE.md §8):
    #. ``base_model.py``    ← done
    #. ``xgb_model.py``     ← done
    #. **``regime_trainer.py``** ← this file
    #. ``lstm_model.py``    ← done
    #. ``ensemble.py``      ← next

Usage::

    trainer = RegimeTrainer()
    models = trainer.train_per_regime(feature_df, regime_labels)
    bull_model = models.get("TRENDING_BULL")

The returned dict maps each regime name that had enough data to a fitted
:class:`~src.models.xgb_model.XGBoostModel`.  Regimes with fewer than
:const:`_MIN_REGIME_ROWS` rows are skipped with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import WalkForwardSettings, doge_settings
from src.models.xgb_model import XGBoostModel
from src.training.scaler import FoldScaler
from src.training.walk_forward import WalkForwardCV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_REGIMES: tuple[str, ...] = (
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
)

_MIN_REGIME_ROWS: int = 500   # skip regimes with fewer training rows
_TARGET_COL: str = "target"
_ERA_COL: str = "era"
_OPEN_TIME_COL: str = "open_time"

# Columns that are NOT features — excluded when identifying feature columns
_PASSTHROUGH_COLS: frozenset[str] = frozenset(
    {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "num_trades",
        "symbol",
        "era",
        "interval",
        "regime_label",
        "target",
        "is_interpolated",
    }
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegimeTrainingResult:
    """Summary of a single regime's walk-forward training run.

    Attributes:
        regime: Regime name (e.g. ``"TRENDING_BULL"``).
        n_folds: Number of walk-forward folds completed.
        fold_val_accuracies: Val accuracy from each fold.
        mean_val_accuracy: Mean across all folds.
        n_rows_used: Total regime rows used for training.
        skipped: ``True`` if the regime was skipped (insufficient data).
        skip_reason: Human-readable skip reason (empty string if not skipped).
    """

    regime: str
    n_folds: int = 0
    fold_val_accuracies: list[float] = field(default_factory=list)
    mean_val_accuracy: float = 0.0
    n_rows_used: int = 0
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# RegimeTrainer
# ---------------------------------------------------------------------------


class RegimeTrainer:
    """Train one XGBoostModel per market regime using walk-forward CV.

    Each regime model is trained on the filtered rows where
    ``regime_labels == regime``.  The final returned model for each regime
    is fitted on the **last walk-forward fold's train/val split** (most recent
    data window), which is the most relevant for live deployment.

    Args:
        walk_forward_cfg: Walk-forward settings.  Defaults to the
            ``doge_settings.walk_forward`` singleton.
        output_dir: Optional directory to save each regime model to disk.
            If *None*, models are held in memory only.
        mlflow_tracking_uri: Optional MLflow tracking URI.  If *None*, MLflow
            logging is attempted using the default URI from
            ``config/settings.yaml``; failures are logged as warnings and do
            not halt training.
    """

    def __init__(
        self,
        walk_forward_cfg: WalkForwardSettings | None = None,
        output_dir: Path | None = None,
        mlflow_tracking_uri: str | None = None,
    ) -> None:
        self._walk_forward_cfg: WalkForwardSettings = (
            walk_forward_cfg
            if walk_forward_cfg is not None
            else doge_settings.walk_forward
        )
        self._output_dir: Path | None = (
            Path(output_dir) if output_dir is not None else None
        )
        self._mlflow_uri: str | None = mlflow_tracking_uri
        # Instance-level cache: populated during train_per_regime
        self._model_cache: dict[str, XGBoostModel] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_per_regime(
        self,
        feature_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> dict[str, XGBoostModel]:
        """Train one :class:`XGBoostModel` per market regime.

        For each of the five standard regimes:

        1. Filter ``feature_df`` to rows where ``regime_labels == regime``.
        2. Skip regimes with fewer than :const:`_MIN_REGIME_ROWS` rows.
        3. Run walk-forward CV; record per-fold val accuracy.
        4. Fit the final model on the **last fold's** ``(train_df, val_df)``.
        5. Optionally archive the final model to MLflow and to disk.

        Args:
            feature_df: Full feature DataFrame with columns ``open_time``,
                ``era``, feature columns, and ``target``.  Must contain
                ``era = 'training'`` rows only for the training phase.
            regime_labels: Pandas Series, index-aligned with *feature_df*,
                containing regime strings (one of the five canonical labels).

        Returns:
            Dict mapping regime name → fitted :class:`XGBoostModel`.
            Only regimes with sufficient data and successful training are
            included; skipped regimes are absent from the dict.

        Raises:
            ValueError: If *feature_df* does not contain ``open_time``,
                ``era``, or ``target`` columns.
        """
        self._validate_feature_df(feature_df)

        feature_cols = self._get_feature_cols(feature_df)
        if not feature_cols:
            raise ValueError(
                "RegimeTrainer.train_per_regime: no feature columns found in "
                "feature_df.  Ensure the DataFrame has been built by FeaturePipeline."
            )

        logger.info(
            "RegimeTrainer: starting per-regime training on {} rows, "
            "{} feature columns, regimes: {}",
            len(feature_df),
            len(feature_cols),
            list(_ALL_REGIMES),
        )

        results: dict[str, XGBoostModel] = {}
        training_summaries: list[RegimeTrainingResult] = []

        # Align regime_labels to feature_df's index
        aligned_labels = regime_labels.reindex(feature_df.index)

        for regime in _ALL_REGIMES:
            summary = self._train_single_regime(
                regime=regime,
                feature_df=feature_df,
                regime_mask=aligned_labels == regime,
                feature_cols=feature_cols,
            )
            training_summaries.append(summary)

            if summary.skipped:
                logger.warning(
                    "RegimeTrainer: regime '{}' SKIPPED — {}",
                    regime,
                    summary.skip_reason,
                )
                continue

            logger.info(
                "RegimeTrainer: regime '{}' complete — {} folds, "
                "mean_val_acc={:.4f}, n_rows={}",
                regime,
                summary.n_folds,
                summary.mean_val_accuracy,
                summary.n_rows_used,
            )

        # Retrieve fitted models from _train_single_regime via _model_cache
        results = self._model_cache.copy()

        self._log_summary_table(training_summaries)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_single_regime(
        self,
        regime: str,
        feature_df: pd.DataFrame,
        regime_mask: pd.Series,
        feature_cols: list[str],
    ) -> RegimeTrainingResult:
        """Train XGBoostModel for one regime and cache the result.

        Args:
            regime: Regime name.
            feature_df: Full feature DataFrame.
            regime_mask: Boolean Series aligned with *feature_df*; True for
                rows belonging to this regime.
            feature_cols: Ordered list of feature column names.

        Returns:
            :class:`RegimeTrainingResult` with training metrics.
        """
        summary = RegimeTrainingResult(regime=regime)

        regime_df = feature_df[regime_mask].copy()
        n_rows = len(regime_df)
        summary.n_rows_used = n_rows

        if n_rows < _MIN_REGIME_ROWS:
            summary.skipped = True
            summary.skip_reason = (
                f"only {n_rows} rows (minimum: {_MIN_REGIME_ROWS})"
            )
            return summary

        # WalkForwardCV requires era and open_time columns — both present in
        # feature_df (validated above).  All regime rows come from the
        # training partition (era == 'training') since the feature pipeline
        # only processes training-era data.
        cv = WalkForwardCV(cfg=self._walk_forward_cfg)

        try:
            folds = cv.generate_folds(regime_df)
        except ValueError as exc:
            summary.skipped = True
            summary.skip_reason = str(exc)
            return summary

        fold_accuracies: list[float] = []
        last_train_df: pd.DataFrame | None = None
        last_val_df: pd.DataFrame | None = None

        for train_df, val_df in cv.split(regime_df):
            # RULE B — new FoldScaler per fold
            scaler = FoldScaler()
            X_train = scaler.fit_transform(train_df[feature_cols].values)
            X_val = scaler.transform(val_df[feature_cols].values)
            y_train = train_df[_TARGET_COL].values
            y_val = val_df[_TARGET_COL].values

            # Guard: skip fold if only one class in training labels
            if len(np.unique(y_train.astype(int))) < 2:
                logger.debug(
                    "RegimeTrainer: regime '{}' fold skipped — single class in y_train",
                    regime,
                )
                continue

            fold_model = XGBoostModel()
            metrics = fold_model.fit(
                X_train, y_train, X_val, y_val, feature_names=feature_cols
            )
            fold_accuracies.append(float(metrics["val_accuracy"]))

            last_train_df = train_df
            last_val_df = val_df

        if not fold_accuracies or last_train_df is None or last_val_df is None:
            summary.skipped = True
            summary.skip_reason = "no valid folds produced (all single-class or empty)"
            return summary

        summary.n_folds = len(fold_accuracies)
        summary.fold_val_accuracies = fold_accuracies
        summary.mean_val_accuracy = float(np.mean(fold_accuracies))

        # --- Final model: fit on the LAST fold's data (most recent window) ---
        final_scaler = FoldScaler()
        X_train_final = final_scaler.fit_transform(last_train_df[feature_cols].values)
        X_val_final = final_scaler.transform(last_val_df[feature_cols].values)
        y_train_final = last_train_df[_TARGET_COL].values
        y_val_final = last_val_df[_TARGET_COL].values

        final_model = XGBoostModel()
        final_model.fit(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            feature_names=feature_cols,
        )

        # Archive to disk
        if self._output_dir is not None:
            regime_dir = self._output_dir / regime
            try:
                final_model.save(regime_dir)
                logger.info(
                    "RegimeTrainer: saved regime '{}' model → {}", regime, regime_dir
                )
            except OSError as exc:
                logger.warning(
                    "RegimeTrainer: could not save regime '{}' model — {}", regime, exc
                )

        # Archive to MLflow (optional — skip gracefully if unavailable)
        self._archive_to_mlflow(regime, final_model, summary)

        # Cache the fitted model for retrieval by train_per_regime
        self._model_cache[regime] = final_model

        return summary

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return sorted list of numeric feature columns (excludes passthrough cols).

        Args:
            df: Feature DataFrame.

        Returns:
            Sorted list of feature column names.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return sorted(
            col for col in numeric_cols if col not in _PASSTHROUGH_COLS
        )

    def _validate_feature_df(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are absent.

        Args:
            df: DataFrame to check.

        Raises:
            ValueError: If ``open_time``, ``era``, or ``target`` columns are missing.
        """
        required = {_OPEN_TIME_COL, _ERA_COL, _TARGET_COL}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"RegimeTrainer: missing required columns: {sorted(missing)}.  "
                "The DataFrame must contain 'open_time', 'era', and 'target'."
            )

    def _archive_to_mlflow(
        self,
        regime: str,
        model: XGBoostModel,
        summary: RegimeTrainingResult,
    ) -> None:
        """Archive model metrics and artefacts to MLflow.

        Logs regime tag, fold count, and mean val accuracy.  Failures are
        caught and logged as warnings — MLflow unavailability must never
        halt training.

        Args:
            regime: Regime name used as an MLflow tag.
            model: Fitted :class:`XGBoostModel`.
            summary: Training summary with metrics.
        """
        try:
            import mlflow  # noqa: PLC0415

            tracking_uri = self._mlflow_uri
            if tracking_uri is None:
                from src.config import settings as _settings  # noqa: PLC0415
                tracking_uri = _settings.mlflow.tracking_uri

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("doge_predictor")

            with mlflow.start_run(run_name=f"regime_{regime}"):
                mlflow.set_tag("regime", regime)
                mlflow.log_param("n_folds", summary.n_folds)
                mlflow.log_param("n_rows_used", summary.n_rows_used)
                mlflow.log_metric("mean_val_accuracy", summary.mean_val_accuracy)
                mlflow.log_metric("best_iteration", model._best_iteration)
                for i, acc in enumerate(summary.fold_val_accuracies, start=1):
                    mlflow.log_metric("fold_val_accuracy", acc, step=i)

            logger.info(
                "RegimeTrainer: regime '{}' logged to MLflow (uri={})",
                regime,
                tracking_uri,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RegimeTrainer: MLflow logging failed for regime '{}' — {} (continuing)",
                regime,
                exc,
            )

    @staticmethod
    def _log_summary_table(summaries: list[RegimeTrainingResult]) -> None:
        """Log a formatted summary table of all regime training results.

        Args:
            summaries: List of :class:`RegimeTrainingResult` instances.
        """
        logger.info("RegimeTrainer: ===== PER-REGIME TRAINING SUMMARY =====")
        for s in summaries:
            if s.skipped:
                logger.info(
                    "  {:20s}  SKIPPED  ({})", s.regime, s.skip_reason
                )
            else:
                accs = ", ".join(f"{a:.3f}" for a in s.fold_val_accuracies)
                logger.info(
                    "  {:20s}  folds={:2d}  mean_acc={:.4f}  accs=[{}]",
                    s.regime,
                    s.n_folds,
                    s.mean_val_accuracy,
                    accs,
                )
        logger.info("RegimeTrainer: ==========================================")


__all__ = ["RegimeTrainer", "RegimeTrainingResult"]
