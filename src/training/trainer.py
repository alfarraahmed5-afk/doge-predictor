"""Full model training orchestration with MLflow integration.

:class:`ModelTrainer` is the single entry point for a complete training run.
It wires together every component built in Phase 5:

1. Walk-forward CV over ``feature_df`` (RULE C — chronological folds only).
2. Per-fold :class:`~src.training.scaler.FoldScaler` (RULE B — never fit on val).
3. Optional Optuna hyperparameter optimisation on the last fold's train+val.
4. Final :class:`~src.models.xgb_model.XGBoostModel` trained on the last fold.
5. Final :class:`~src.models.lstm_model.LSTMModel` trained on the last fold.
6. Per-regime XGBoost models via :class:`~src.training.regime_trainer.RegimeTrainer`.
7. :class:`~src.models.ensemble.EnsembleModel` meta-learner assembled from
   base-model probabilities on the last validation fold.
8. MLflow archive: hyperparameters, fold metrics, model artefacts,
   ``scaler.pkl``, ``feature_columns.json``, ``regime_config.yaml``.

Build order (CLAUDE.md §8):
    1. ``base_model.py``     ← done
    2. ``xgb_model.py``      ← done
    3. ``regime_trainer.py`` ← done
    4. ``lstm_model.py``     ← done
    5. ``ensemble.py``       ← done
    6. **``trainer.py``**    ← this file

Usage::

    trainer = ModelTrainer(output_dir=Path("models/"), run_hyperopt=False)
    result = trainer.train_full(feature_df, regime_labels)
    print(result.mean_val_accuracy)
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import WalkForwardSettings, doge_settings, settings as _global_settings
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.xgb_model import XGBoostModel
from src.training.hyperopt import HyperparameterOptimizer
from src.training.regime_trainer import RegimeTrainer
from src.training.scaler import FoldScaler
from src.training.walk_forward import WalkForwardCV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_COL: str = "target"
_ERA_COL: str = "era"
_OPEN_TIME_COL: str = "open_time"

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

_FEATURE_COLUMNS_FILENAME: str = "feature_columns.json"
_REGIME_CONFIG_FILENAME: str = "regime_config.yaml"


# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Summary of a full :class:`ModelTrainer` run.

    Attributes:
        n_folds: Number of walk-forward folds completed.
        fold_val_accuracies: XGBoost val accuracy from each fold.
        mean_val_accuracy: Mean across all fold val accuracies.
        std_val_accuracy: Standard deviation across fold val accuracies.
        best_xgb_params: Best hyperparameters found for XGBoost
            (empty dict if hyperopt was skipped).
        best_lstm_params: Best hyperparameters found for LSTM
            (empty dict if hyperopt was skipped).
        n_rows_used: Total training rows in ``feature_df``.
        seed_used: Random seed applied during training.
        mlflow_run_id: MLflow run ID for the top-level training run
            (empty string if MLflow was unavailable).
        skipped_regimes: Regime labels skipped by
            :class:`~src.training.regime_trainer.RegimeTrainer`
            (insufficient data or no valid folds).
    """

    n_folds: int = 0
    fold_val_accuracies: list[float] = field(default_factory=list)
    mean_val_accuracy: float = 0.0
    std_val_accuracy: float = 0.0
    best_xgb_params: dict[str, Any] = field(default_factory=dict)
    best_lstm_params: dict[str, Any] = field(default_factory=dict)
    n_rows_used: int = 0
    seed_used: int = 0
    mlflow_run_id: str = ""
    skipped_regimes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Orchestrate a full walk-forward training pipeline with MLflow archive.

    Steps executed by :meth:`train_full`:

    1. Validate ``feature_df`` (required columns present).
    2. Identify numeric feature columns (exclude passthrough cols).
    3. Walk-forward CV — for each fold:
       a. New :class:`~src.training.scaler.FoldScaler` (RULE B).
       b. Assert scaler not fitted on future data.
       c. Train probe :class:`~src.models.xgb_model.XGBoostModel`; record
          ``val_accuracy``.
    4. Optional Optuna hyperopt on the **last fold's** train + val
       (test set is never supplied to hyperopt).
    5. Train final XGBoost on the last fold with best params.
    6. Train final LSTM on the last fold (optionally with best LSTM params).
    7. Per-regime XGBoost via :class:`~src.training.regime_trainer.RegimeTrainer`.
    8. Assemble :class:`~src.models.ensemble.EnsembleModel` from val-fold probabilities.
    9. MLflow archive: params, metrics, artefacts, ``'stage': 'candidate'`` tag.
    10. Save models + scaler + ``feature_columns.json`` to ``output_dir`` if set.
    11. Return :class:`TrainingResult`.

    Args:
        output_dir: If given, all model artefacts are persisted here.
        mlflow_tracking_uri: MLflow tracking URI.  ``None`` reads from
            ``config/settings.yaml``.
        walk_forward_cfg: Walk-forward settings.  ``None`` uses
            ``doge_settings.walk_forward``.
        n_hyperopt_trials: Number of Optuna trials per model class.
        run_hyperopt: Set to ``False`` to skip hyperopt (useful for fast tests
            and CI runs).
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        mlflow_tracking_uri: str | None = None,
        walk_forward_cfg: WalkForwardSettings | None = None,
        n_hyperopt_trials: int = 50,
        run_hyperopt: bool = True,
    ) -> None:
        self._output_dir: Path | None = (
            Path(output_dir) if output_dir is not None else None
        )
        self._mlflow_uri: str | None = mlflow_tracking_uri
        self._walk_forward_cfg: WalkForwardSettings = (
            walk_forward_cfg
            if walk_forward_cfg is not None
            else doge_settings.walk_forward
        )
        self._n_hyperopt_trials: int = n_hyperopt_trials
        self._run_hyperopt: bool = run_hyperopt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_full(
        self,
        feature_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> TrainingResult:
        """Run the complete training pipeline.

        Args:
            feature_df: Full feature DataFrame.  Must contain ``open_time``,
                ``era``, and ``target`` columns plus numeric feature columns.
                Only rows with ``era == 'training'`` are used in training folds.
            regime_labels: Pandas Series index-aligned with *feature_df*
                containing regime label strings.

        Returns:
            :class:`TrainingResult` with all fold metrics and metadata.

        Raises:
            ValueError: If *feature_df* is missing required columns or has no
                feature columns.
        """
        self._validate_feature_df(feature_df)
        feature_cols = self._get_feature_cols(feature_df)
        if not feature_cols:
            raise ValueError(
                "ModelTrainer.train_full: no numeric feature columns found. "
                "Ensure the DataFrame has been built by FeaturePipeline."
            )

        result = TrainingResult(
            n_rows_used=len(feature_df),
            seed_used=_global_settings.project.seed,
        )

        logger.info(
            "ModelTrainer: starting full training — {} rows, {} features",
            len(feature_df),
            len(feature_cols),
        )

        # ------------------------------------------------------------------
        # Step 3 & 4 — Walk-forward CV
        # ------------------------------------------------------------------
        cv = WalkForwardCV(cfg=self._walk_forward_cfg)
        fold_accuracies: list[float] = []
        last_train_df: pd.DataFrame | None = None
        last_val_df: pd.DataFrame | None = None
        last_scaler: FoldScaler | None = None

        for train_df, val_df in cv.split(feature_df):
            scaler = FoldScaler()
            X_train = scaler.fit_transform(train_df[feature_cols].values)
            X_val = scaler.transform(val_df[feature_cols].values)
            y_train = train_df[_TARGET_COL].values
            y_val = val_df[_TARGET_COL].values

            # RULE B — assert no future data leaked into scaler
            train_end_ts = int(train_df[_OPEN_TIME_COL].max())
            scaler.assert_not_fitted_on_future(train_end_ts, train_df)

            if len(np.unique(y_train.astype(int))) < 2:
                logger.debug("ModelTrainer: fold skipped — single class in y_train")
                continue

            fold_model = XGBoostModel()
            metrics = fold_model.fit(X_train, y_train, X_val, y_val)
            fold_accuracies.append(float(metrics["val_accuracy"]))

            last_train_df = train_df
            last_val_df = val_df
            last_scaler = scaler

        if not fold_accuracies or last_train_df is None or last_val_df is None:
            logger.error("ModelTrainer: no valid folds — aborting training.")
            return result

        result.n_folds = len(fold_accuracies)
        result.fold_val_accuracies = fold_accuracies
        result.mean_val_accuracy = float(np.mean(fold_accuracies))
        result.std_val_accuracy = float(np.std(fold_accuracies))

        logger.info(
            "ModelTrainer: walk-forward complete — {} folds, "
            "mean_val_acc={:.4f} ± {:.4f}",
            result.n_folds,
            result.mean_val_accuracy,
            result.std_val_accuracy,
        )

        # Re-scale the last fold's data (final scaler for production)
        final_scaler = FoldScaler()
        X_train_final = final_scaler.fit_transform(
            last_train_df[feature_cols].values
        )
        X_val_final = final_scaler.transform(last_val_df[feature_cols].values)
        y_train_final = last_train_df[_TARGET_COL].values
        y_val_final = last_val_df[_TARGET_COL].values

        # ------------------------------------------------------------------
        # Step 5 — Hyperopt (train+val only, test never touched)
        # ------------------------------------------------------------------
        best_xgb_params: dict[str, Any] = {}
        best_lstm_params: dict[str, Any] = {}

        if self._run_hyperopt:
            optimiser = HyperparameterOptimizer(
                mlflow_tracking_uri=self._mlflow_uri,
                seed=_global_settings.project.seed,
            )

            logger.info(
                "ModelTrainer: running XGBoost hyperopt ({} trials) …",
                self._n_hyperopt_trials,
            )
            best_xgb_params = optimiser.optimize(
                XGBoostModel,
                X_train_final,
                y_train_final,
                X_val_final,
                y_val_final,
                n_trials=self._n_hyperopt_trials,
            )

            logger.info(
                "ModelTrainer: running LSTM hyperopt ({} trials) …",
                self._n_hyperopt_trials,
            )
            best_lstm_params = optimiser.optimize(
                LSTMModel,
                X_train_final,
                y_train_final,
                X_val_final,
                y_val_final,
                n_trials=self._n_hyperopt_trials,
            )

        result.best_xgb_params = best_xgb_params
        result.best_lstm_params = best_lstm_params

        # ------------------------------------------------------------------
        # Step 6 — Final XGBoost (last fold, best params)
        # ------------------------------------------------------------------
        logger.info("ModelTrainer: training final XGBoostModel …")
        final_xgb = XGBoostModel(**best_xgb_params) if best_xgb_params else XGBoostModel()
        final_xgb.fit(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            feature_names=feature_cols,
        )

        # ------------------------------------------------------------------
        # Step 7 — Final LSTM (last fold, best params)
        # ------------------------------------------------------------------
        logger.info("ModelTrainer: training final LSTMModel …")
        if best_lstm_params:
            final_lstm = LSTMModel(
                sequence_length=best_lstm_params.get("sequence_length", 60),
                dropout_lstm=best_lstm_params.get("dropout", 0.2),
                dropout_dense=best_lstm_params.get("dropout", 0.3),
                hidden_size_1=best_lstm_params.get("hidden_units", 128),
                hidden_size_2=max(best_lstm_params.get("hidden_units", 128) // 2, 32),
            )
        else:
            final_lstm = LSTMModel()

        final_lstm.fit(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val_final,
            feature_names=feature_cols,
        )

        # ------------------------------------------------------------------
        # Step 8 — Per-regime XGBoost via RegimeTrainer
        # ------------------------------------------------------------------
        logger.info("ModelTrainer: training per-regime XGBoost models …")
        regime_trainer = RegimeTrainer(
            walk_forward_cfg=self._walk_forward_cfg,
            mlflow_tracking_uri=self._mlflow_uri,
        )
        regime_models = regime_trainer.train_per_regime(feature_df, regime_labels)

        all_regimes = {
            "TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL",
            "RANGING_LOW_VOL", "DECOUPLED",
        }
        result.skipped_regimes = sorted(all_regimes - set(regime_models.keys()))
        logger.info(
            "ModelTrainer: per-regime training done — {} trained, {} skipped",
            len(regime_models),
            len(result.skipped_regimes),
        )

        # ------------------------------------------------------------------
        # Step 9 — EnsembleModel (meta-learner on val-fold probabilities)
        # ------------------------------------------------------------------
        logger.info("ModelTrainer: assembling EnsembleModel meta-learner …")
        aligned_labels = regime_labels.reindex(last_val_df.index)

        # Build regime_encoded for each val row
        from src.regimes.features import get_regime_features  # noqa: PLC0415
        regime_enc_val = np.array(
            [
                get_regime_features(str(lbl)).get("regime_encoded", 0)
                for lbl in aligned_labels.fillna("RANGING_LOW_VOL")
            ],
            dtype=np.float64,
        )

        lstm_proba_val = final_lstm.predict_proba(X_val_final)
        xgb_proba_val = final_xgb.predict_proba(X_val_final)

        meta_X_val = np.column_stack([lstm_proba_val, xgb_proba_val, regime_enc_val])
        # Mirror the meta-features on train for ensemble training
        aligned_labels_train = regime_labels.reindex(last_train_df.index)
        regime_enc_train = np.array(
            [
                get_regime_features(str(lbl)).get("regime_encoded", 0)
                for lbl in aligned_labels_train.fillna("RANGING_LOW_VOL")
            ],
            dtype=np.float64,
        )
        lstm_proba_train = final_lstm.predict_proba(X_train_final)
        xgb_proba_train = final_xgb.predict_proba(X_train_final)
        meta_X_train = np.column_stack(
            [lstm_proba_train, xgb_proba_train, regime_enc_train]
        )

        ensemble = EnsembleModel()
        ensemble.fit(meta_X_train, y_train_final, meta_X_val, y_val_final)

        # ------------------------------------------------------------------
        # Step 10 — MLflow archive
        # ------------------------------------------------------------------
        run_id = self._archive_to_mlflow(
            result=result,
            feature_cols=feature_cols,
            final_xgb=final_xgb,
            final_lstm=final_lstm,
            ensemble=ensemble,
            final_scaler=final_scaler,
        )
        result.mlflow_run_id = run_id

        # ------------------------------------------------------------------
        # Step 11 — Save models to disk
        # ------------------------------------------------------------------
        if self._output_dir is not None:
            self._save_all_artefacts(
                output_dir=self._output_dir,
                feature_cols=feature_cols,
                final_xgb=final_xgb,
                final_lstm=final_lstm,
                ensemble=ensemble,
                final_scaler=final_scaler,
                regime_models=regime_models,
            )

        # Log summary table
        self._log_summary(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Return sorted list of numeric feature columns.

        Args:
            df: Feature DataFrame.

        Returns:
            Sorted list of feature column names (excludes passthrough cols).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return sorted(col for col in numeric_cols if col not in _PASSTHROUGH_COLS)

    def _validate_feature_df(self, df: pd.DataFrame) -> None:
        """Raise ValueError if required columns are absent.

        Args:
            df: DataFrame to check.

        Raises:
            ValueError: If ``open_time``, ``era``, or ``target`` are missing.
        """
        required = {_OPEN_TIME_COL, _ERA_COL, _TARGET_COL}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"ModelTrainer: missing required columns: {sorted(missing)}. "
                "The DataFrame must contain 'open_time', 'era', and 'target'."
            )

    def _save_feature_columns_json(
        self, path: Path, feature_cols: list[str], run_id: str = ""
    ) -> None:
        """Write feature column list to ``feature_columns.json``.

        Args:
            path: Directory to write the JSON file.
            feature_cols: Ordered list of feature column names.
            run_id: Optional MLflow run ID for traceability.
        """
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
        }
        with open(path / _FEATURE_COLUMNS_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _copy_regime_config(self, dest_dir: Path) -> None:
        """Copy ``config/regime_config.yaml`` to *dest_dir*.

        Args:
            dest_dir: Destination directory.
        """
        import importlib.resources  # noqa: PLC0415

        src_path = (
            Path(__file__).resolve().parent.parent.parent
            / "config"
            / _REGIME_CONFIG_FILENAME
        )
        if src_path.exists():
            shutil.copy2(src_path, dest_dir / _REGIME_CONFIG_FILENAME)

    def _save_all_artefacts(
        self,
        output_dir: Path,
        feature_cols: list[str],
        final_xgb: XGBoostModel,
        final_lstm: LSTMModel,
        ensemble: EnsembleModel,
        final_scaler: FoldScaler,
        regime_models: dict[str, XGBoostModel],
    ) -> None:
        """Persist all trained artefacts to *output_dir*.

        Args:
            output_dir: Root output directory (created if absent).
            feature_cols: Feature column list.
            final_xgb: Fitted global XGBoostModel.
            final_lstm: Fitted LSTMModel.
            ensemble: Fitted EnsembleModel.
            final_scaler: Fitted FoldScaler (production scaler).
            regime_models: Dict of per-regime XGBoostModels.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            final_xgb.save(output_dir / "xgb_global")
        except OSError as exc:
            logger.warning("ModelTrainer: could not save XGBoost → {}", exc)

        try:
            final_lstm.save(output_dir / "lstm")
        except OSError as exc:
            logger.warning("ModelTrainer: could not save LSTM → {}", exc)

        try:
            ensemble.save(output_dir / "ensemble")
        except OSError as exc:
            logger.warning("ModelTrainer: could not save Ensemble → {}", exc)

        try:
            final_scaler.save(output_dir)
        except OSError as exc:
            logger.warning("ModelTrainer: could not save scaler → {}", exc)

        try:
            for regime, model in regime_models.items():
                model.save(output_dir / "regime_models" / regime)
        except OSError as exc:
            logger.warning("ModelTrainer: could not save regime model → {}", exc)

        try:
            self._save_feature_columns_json(output_dir, feature_cols)
        except OSError as exc:
            logger.warning("ModelTrainer: could not save feature_columns.json → {}", exc)

        try:
            self._copy_regime_config(output_dir)
        except OSError as exc:
            logger.warning("ModelTrainer: could not copy regime_config.yaml → {}", exc)

        logger.info("ModelTrainer: all artefacts saved → {}", output_dir)

    def _archive_to_mlflow(
        self,
        result: TrainingResult,
        feature_cols: list[str],
        final_xgb: XGBoostModel,
        final_lstm: LSTMModel,
        ensemble: EnsembleModel,
        final_scaler: FoldScaler,
    ) -> str:
        """Archive training metrics and artefacts to MLflow.

        Returns the MLflow run ID (empty string on failure).

        Failures are caught and logged as warnings — MLflow unavailability
        must never halt training.

        Args:
            result: Populated :class:`TrainingResult` (read-only here).
            feature_cols: Feature column list.
            final_xgb: Fitted global XGBoostModel.
            final_lstm: Fitted LSTMModel.
            ensemble: Fitted EnsembleModel.
            final_scaler: Fitted FoldScaler.

        Returns:
            MLflow run ID string, or ``""`` on failure.
        """
        try:
            import mlflow  # noqa: PLC0415

            tracking_uri = self._mlflow_uri
            if tracking_uri is None:
                from src.config import settings as _s  # noqa: PLC0415
                tracking_uri = _s.mlflow.tracking_uri

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("doge_predictor")

            with mlflow.start_run(run_name="full_training") as run:
                # Parameters
                mlflow.log_param("n_folds", result.n_folds)
                mlflow.log_param("n_rows_used", result.n_rows_used)
                mlflow.log_param("seed_used", result.seed_used)
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("run_hyperopt", self._run_hyperopt)

                if result.best_xgb_params:
                    mlflow.log_params(
                        {f"xgb_{k}": v for k, v in result.best_xgb_params.items()}
                    )
                if result.best_lstm_params:
                    mlflow.log_params(
                        {f"lstm_{k}": v for k, v in result.best_lstm_params.items()}
                    )

                # Fold metrics
                mlflow.log_metric("mean_val_accuracy", result.mean_val_accuracy)
                mlflow.log_metric("std_val_accuracy", result.std_val_accuracy)
                for i, acc in enumerate(result.fold_val_accuracies, start=1):
                    mlflow.log_metric("fold_val_accuracy", acc, step=i)

                # SHAP feature importance (optional — skip if shap unavailable)
                try:
                    import shap  # noqa: PLC0415
                    if final_xgb._is_fitted and final_xgb._booster is not None:
                        import xgboost as xgb  # noqa: PLC0415
                        explainer = shap.TreeExplainer(final_xgb._booster)
                        top_feats = final_xgb.get_top_features(n=20)
                        for feat_name, importance in top_feats:
                            mlflow.log_metric(
                                f"shap_importance_{feat_name}", importance
                            )
                except ImportError:
                    logger.debug(
                        "ModelTrainer: shap not installed — skipping SHAP logging"
                    )
                except Exception as shap_exc:  # noqa: BLE001
                    logger.debug(
                        "ModelTrainer: SHAP logging failed — {} (continuing)", shap_exc
                    )

                # Stage tag
                mlflow.set_tag("stage", "candidate")
                mlflow.set_tag(
                    "skipped_regimes", ",".join(result.skipped_regimes)
                )

                # Artefacts — save to temp dir then log
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)

                    final_xgb.save(tmp_path / "xgb_global")
                    mlflow.log_artifacts(str(tmp_path / "xgb_global"), "xgb_global")

                    final_lstm.save(tmp_path / "lstm")
                    mlflow.log_artifacts(str(tmp_path / "lstm"), "lstm")

                    ensemble.save(tmp_path / "ensemble")
                    mlflow.log_artifacts(str(tmp_path / "ensemble"), "ensemble")

                    final_scaler.save(tmp_path)
                    mlflow.log_artifact(str(tmp_path / "scaler.pkl"))

                    self._save_feature_columns_json(tmp_path, feature_cols, run.info.run_id)
                    mlflow.log_artifact(str(tmp_path / _FEATURE_COLUMNS_FILENAME))

                    self._copy_regime_config(tmp_path)
                    regime_cfg_file = tmp_path / _REGIME_CONFIG_FILENAME
                    if regime_cfg_file.exists():
                        mlflow.log_artifact(str(regime_cfg_file))

                run_id: str = run.info.run_id

            logger.info(
                "ModelTrainer: archived to MLflow (run_id={})", run_id
            )
            return run_id

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ModelTrainer: MLflow archiving failed — {} (continuing)", exc
            )
            return ""

    @staticmethod
    def _log_summary(result: TrainingResult) -> None:
        """Log a formatted summary of the training run.

        Args:
            result: :class:`TrainingResult` to summarise.
        """
        accs = ", ".join(f"{a:.3f}" for a in result.fold_val_accuracies)
        logger.info("ModelTrainer: ===== TRAINING SUMMARY =====")
        logger.info(
            "  Folds:             {}", result.n_folds
        )
        logger.info(
            "  Mean val_acc:      {:.4f} ± {:.4f}",
            result.mean_val_accuracy,
            result.std_val_accuracy,
        )
        logger.info("  Per-fold accs:     [{}]", accs)
        logger.info("  Rows used:         {}", result.n_rows_used)
        logger.info("  Seed:              {}", result.seed_used)
        logger.info("  MLflow run_id:     {}", result.mlflow_run_id or "(not logged)")
        logger.info(
            "  Skipped regimes:   {}",
            result.skipped_regimes or "none",
        )
        logger.info("ModelTrainer: =================================")


__all__ = ["ModelTrainer", "TrainingResult", "retrain_weekly"]


# ---------------------------------------------------------------------------
# retrain_weekly — standalone weekly retraining workflow
# ---------------------------------------------------------------------------


def retrain_weekly(
    storage: Any,
    mlflow_tracking_uri: str | None = None,
    output_dir: Path | None = None,
    walk_forward_cfg: "WalkForwardSettings | None" = None,
) -> bool:
    """Execute the weekly model retraining workflow.

    Steps:

    1. Archive the current production model run with tag ``'previous-production'``.
    2. Rebuild the feature matrix from the latest data in *storage*.
    3. Run the full :class:`ModelTrainer` training pipeline (hyperopt disabled
       for speed).
    4. Compare OOS accuracy of the new model vs the production model stored in
       MLflow.
    5. If the new model is better (or no production model exists): tag the new
       run as ``'candidate'`` and return ``True``.
    6. If the new model is worse: tag as ``'rejected'``, log the comparison,
       and return ``False``.

    Shadow-mode promotion (Steps 6–8 in the original design) is handled
    externally: after returning ``True`` the caller should set
    ``SHADOW_MODE=true`` and schedule :mod:`scripts.qg09_verify` to run
    after 48 h.

    Args:
        storage: Initialised :class:`~src.processing.storage.DogeStorage`
            instance (SQLite or TimescaleDB).
        mlflow_tracking_uri: MLflow tracking URI.  ``None`` reads from
            ``config/settings.yaml``.
        output_dir: Directory to save the newly trained model artefacts.
            ``None`` saves to a timestamped sub-directory of ``models/``.
        walk_forward_cfg: Walk-forward settings override.  ``None`` uses
            ``doge_settings.walk_forward`` defaults.

    Returns:
        ``True`` when the new model is tagged as ``'candidate'``;
        ``False`` when training fails or the new model is worse.
    """
    import datetime as _dt  # noqa: PLC0415

    logger.info("retrain_weekly: starting weekly retraining workflow")

    # ------------------------------------------------------------------
    # Step 1 — archive current production model as 'previous-production'
    # ------------------------------------------------------------------
    try:
        import mlflow  # noqa: PLC0415

        tracking_uri = mlflow_tracking_uri
        if tracking_uri is None:
            tracking_uri = _global_settings.mlflow.tracking_uri

        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name("doge_predictor")

        production_accuracy: float | None = None

        if exp is not None:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="tags.stage = 'production'",
                max_results=1,
            )
            if runs:
                prod_run = runs[0]
                client.set_tag(prod_run.info.run_id, "stage", "previous-production")
                production_accuracy = prod_run.data.metrics.get("mean_val_accuracy")
                logger.info(
                    "retrain_weekly: archived production run {} as 'previous-production'",
                    prod_run.info.run_id,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "retrain_weekly: MLflow step 1 failed — {} (continuing)", exc
        )
        production_accuracy = None

    # ------------------------------------------------------------------
    # Step 2 — rebuild feature matrix from latest storage data
    # ------------------------------------------------------------------
    feature_df: pd.DataFrame | None = None
    regime_labels: pd.Series | None = None

    try:
        feature_df, regime_labels = _build_feature_matrix_from_storage(storage)
    except Exception as exc:  # noqa: BLE001
        logger.error("retrain_weekly: feature matrix build failed — {}", exc)
        return False

    if feature_df is None or len(feature_df) < (
        walk_forward_cfg.min_training_rows if walk_forward_cfg else doge_settings.walk_forward.min_training_rows
    ):
        logger.error(
            "retrain_weekly: insufficient data ({} rows) for retraining",
            len(feature_df) if feature_df is not None else 0,
        )
        return False

    # ------------------------------------------------------------------
    # Step 3 — run full training pipeline (hyperopt disabled for speed)
    # ------------------------------------------------------------------
    ts_suffix = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    effective_output_dir = (
        Path(output_dir) if output_dir is not None
        else Path("models") / f"weekly_{ts_suffix}"
    )

    trainer = ModelTrainer(
        output_dir=effective_output_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        walk_forward_cfg=walk_forward_cfg,
        run_hyperopt=False,
    )

    try:
        result = trainer.train_full(feature_df, regime_labels)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        logger.error("retrain_weekly: training failed — {}", exc)
        return False

    if result.n_folds == 0:
        logger.error("retrain_weekly: training produced no valid folds — aborting")
        return False

    # ------------------------------------------------------------------
    # Step 4 — compare new model vs production
    # ------------------------------------------------------------------
    new_accuracy = result.mean_val_accuracy

    logger.info(
        "retrain_weekly: new_accuracy={:.4f}  production_accuracy={}",
        new_accuracy,
        f"{production_accuracy:.4f}" if production_accuracy is not None else "N/A",
    )

    # ------------------------------------------------------------------
    # Step 5 — tag new model as 'candidate' or 'rejected'
    # ------------------------------------------------------------------
    is_better = production_accuracy is None or new_accuracy > production_accuracy

    try:
        import mlflow as _mlflow  # noqa: PLC0415

        if result.mlflow_run_id:
            _mlflow.set_tracking_uri(
                mlflow_tracking_uri or _global_settings.mlflow.tracking_uri
            )
            _client = _mlflow.tracking.MlflowClient()
            tag_value = "candidate" if is_better else "rejected"
            _client.set_tag(result.mlflow_run_id, "stage", tag_value)
            logger.info(
                "retrain_weekly: new model run {} tagged as '{}'",
                result.mlflow_run_id,
                tag_value,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrain_weekly: MLflow tagging failed — {} (continuing)", exc)

    if is_better:
        logger.info(
            "retrain_weekly: new model ({:.4f}) is better than production ({}) "
            "— tagged as 'candidate'. Deploy shadow mode and run QG-09 after 48h.",
            new_accuracy,
            f"{production_accuracy:.4f}" if production_accuracy is not None else "none",
        )
        return True
    else:
        logger.warning(
            "retrain_weekly: new model ({:.4f}) is NOT better than production "
            "({:.4f}) — keeping current production model",
            new_accuracy,
            production_accuracy,  # type: ignore[arg-type]
        )
        return False


def _build_feature_matrix_from_storage(
    storage: Any,
    lookback_days: int = 270,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load recent OHLCV data from *storage* and build a feature matrix.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.
        lookback_days: Number of days of history to load (default: 270 ≈ 9 months).

    Returns:
        ``(feature_df, regime_labels)`` — both index-aligned.

    Raises:
        RuntimeError: If DOGEUSDT OHLCV data cannot be loaded.
    """
    import time as _time  # noqa: PLC0415

    from src.features.pipeline import FeaturePipeline  # noqa: PLC0415
    from src.regimes.classifier import DogeRegimeClassifier  # noqa: PLC0415

    end_ms = int(_time.time() * 1_000)
    start_ms = end_ms - lookback_days * 24 * 3_600_000

    doge_df = storage.get_ohlcv("DOGEUSDT", "1h", start_ms, end_ms)
    if doge_df is None or len(doge_df) == 0:
        raise RuntimeError("retrain_weekly: no DOGEUSDT data in storage")

    btc_df = storage.get_ohlcv("BTCUSDT", "1h", start_ms, end_ms)
    dogebtc_df = storage.get_ohlcv("DOGEBTC", "1h", start_ms, end_ms)
    doge_4h = storage.get_ohlcv("DOGEUSDT", "4h", start_ms, end_ms)
    doge_1d = storage.get_ohlcv("DOGEUSDT", "1d", start_ms, end_ms)
    funding_df = storage.get_funding_rates(start_ms, end_ms)

    doge_df = doge_df.copy()
    if "era" not in doge_df.columns:
        _TRAINING_START_MS = 1_640_995_200_000
        doge_df["era"] = np.where(
            doge_df["open_time"] >= _TRAINING_START_MS, "training", "context"
        )

    pipeline = FeaturePipeline()
    feature_df = pipeline.compute_all_features(
        doge_1h=doge_df,
        btc_1h=btc_df,
        dogebtc_1h=dogebtc_df,
        doge_4h=doge_4h,
        doge_1d=doge_1d,
        funding=funding_df,
        min_rows_override=100,
    )

    classifier = DogeRegimeClassifier()
    regime_series = classifier.classify(doge_df, btc_df=btc_df)
    regime_labels = regime_series.reindex(feature_df.index).fillna("RANGING_LOW_VOL")

    logger.info(
        "_build_feature_matrix_from_storage: {} rows, {} features",
        len(feature_df),
        len([c for c in feature_df.columns if c not in {
            "open_time", "close_time", "open", "high", "low", "close",
            "volume", "quote_volume", "num_trades", "symbol", "era",
            "interval", "regime_label", "target", "is_interpolated",
        }]),
    )
    return feature_df, regime_labels
