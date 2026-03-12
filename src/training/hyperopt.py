"""Optuna-based hyperparameter optimisation for XGBoost and LSTM models.

CRITICAL CONSTRAINT (CLAUDE.md §3.2 RULE C / §12 training checklist):
    Optuna trials run **only** on the supplied ``(X_train, y_train, X_val, y_val)``
    pair.  The *test set is NEVER passed into this module* — the method signature
    deliberately offers no way to supply one.

Usage::

    optimiser = HyperparameterOptimizer(seed=42)

    # Tune XGBoost
    best_xgb = optimiser.optimize(
        XGBoostModel, X_train, y_train, X_val, y_val, n_trials=50
    )
    final_xgb = XGBoostModel(**best_xgb)

    # Tune LSTM
    best_lstm = optimiser.optimize(
        LSTMModel, X_train, y_train, X_val, y_val, n_trials=20
    )

MLflow logging is attempted for every trial; failures are silently swallowed so
that a misconfigured MLflow server cannot abort an optimisation run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from loguru import logger
from optuna.samplers import TPESampler

from src.config import settings as _global_settings
from src.models.lstm_model import LSTMModel
from src.models.xgb_model import XGBoostModel

# Suppress the verbose Optuna progress bar and per-trial log lines by default.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# HyperparameterOptimizer
# ---------------------------------------------------------------------------


class HyperparameterOptimizer:
    """Optuna-based hyperparameter search for XGBoost and LSTM models.

    Each call to :meth:`optimize` creates a new Optuna study, runs *n_trials*
    trials, and returns the parameter dict that achieved the highest
    validation accuracy.  MLflow logging is attempted per trial and per run;
    any failures are caught and logged as warnings.

    Args:
        mlflow_tracking_uri: MLflow tracking URI.  ``None`` falls back to the
            value from ``config/settings.yaml``.
        mlflow_experiment: MLflow experiment name for all hyperopt runs.
        seed: Random seed passed to :class:`optuna.samplers.TPESampler` and
            to each trial model for reproducibility.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment: str = "doge_predictor_hyperopt",
        seed: int = _global_settings.project.seed,
    ) -> None:
        self._mlflow_uri: str | None = mlflow_tracking_uri
        self._mlflow_experiment: str = mlflow_experiment
        self._seed: int = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        model_class: type,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Search for the best hyperparameters for *model_class*.

        Dispatches to :meth:`_optimize_xgb` or :meth:`_optimize_lstm`
        depending on *model_class*.

        CRITICAL — TEST SET ISOLATION:
            *X_val* / *y_val* are the walk-forward **validation** fold,
            not an OOS test set.  The test set must never be passed to this
            method; the signature deliberately provides no way to supply one.

        Args:
            model_class: :class:`~src.models.xgb_model.XGBoostModel` or
                :class:`~src.models.lstm_model.LSTMModel`.
            X_train: Training feature matrix from the walk-forward fold.
            y_train: Training labels.
            X_val: Validation feature matrix from the same fold.
            y_val: Validation labels.
            n_trials: Number of Optuna trials to run.

        Returns:
            Dict of best hyperparameter values (keys depend on *model_class*).

        Raises:
            ValueError: If *model_class* is not supported.
        """
        logger.info(
            "HyperparameterOptimizer: starting {} trials for {}",
            n_trials,
            model_class.__name__,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self._seed),
        )

        if model_class is XGBoostModel:
            best_params = self._optimize_xgb(
                X_train, y_train, X_val, y_val, n_trials, study
            )
        elif model_class is LSTMModel:
            best_params = self._optimize_lstm(
                X_train, y_train, X_val, y_val, n_trials, study
            )
        else:
            raise ValueError(
                f"HyperparameterOptimizer: unsupported model_class '{model_class.__name__}'. "
                "Supported: XGBoostModel, LSTMModel."
            )

        logger.info(
            "HyperparameterOptimizer: best params for {} = {} (val_accuracy={:.4f})",
            model_class.__name__,
            best_params,
            study.best_value,
        )
        return best_params

    # ------------------------------------------------------------------
    # Private optimisation helpers
    # ------------------------------------------------------------------

    def _optimize_xgb(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int,
        study: optuna.Study,
    ) -> dict[str, Any]:
        """Run Optuna search over XGBoost hyperparameters.

        Search space:
            - ``max_depth``: int in [3, 8]
            - ``learning_rate``: float in [0.01, 0.1] (log scale)
            - ``subsample``: float in [0.6, 1.0]
            - ``colsample_bytree``: float in [0.6, 1.0]

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            n_trials: Number of Optuna trials.
            study: Pre-constructed :class:`optuna.Study`.

        Returns:
            Best XGBoost hyperparameter dict.
        """
        optimizer = self  # capture for closure

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, Any] = {
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = XGBoostModel(**params)
            try:
                metrics = model.fit(X_train, y_train, X_val, y_val)
            except Exception as exc:  # noqa: BLE001
                logger.debug("HyperparameterOptimizer XGB trial failed: {}", exc)
                return 0.0
            val_acc = float(metrics["val_accuracy"])
            optimizer._log_trial_to_mlflow(
                trial_number=trial.number,
                model_name="XGBoostModel",
                params=params,
                val_accuracy=val_acc,
            )
            return val_acc

        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _optimize_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int,
        study: optuna.Study,
    ) -> dict[str, Any]:
        """Run Optuna search over LSTM hyperparameters.

        Search space:
            - ``sequence_length``: int in [20, 80]
            - ``dropout``: float in [0.1, 0.4]
            - ``hidden_units``: int in [64, 256]

        Note:
            ``hidden_size_2`` is set to ``hidden_units // 2`` so the
            second LSTM layer is always smaller than the first.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            n_trials: Number of Optuna trials.
            study: Pre-constructed :class:`optuna.Study`.

        Returns:
            Best LSTM hyperparameter dict with keys
            ``sequence_length``, ``dropout``, ``hidden_units``.
        """
        optimizer = self  # capture for closure

        def objective(trial: optuna.Trial) -> float:
            seq_len: int = trial.suggest_int("sequence_length", 20, 80)
            dropout: float = trial.suggest_float("dropout", 0.1, 0.4)
            hidden: int = trial.suggest_int("hidden_units", 64, 256)

            model = LSTMModel(
                sequence_length=seq_len,
                dropout_lstm=dropout,
                dropout_dense=dropout,
                hidden_size_1=hidden,
                hidden_size_2=max(hidden // 2, 32),
            )
            try:
                metrics = model.fit(X_train, y_train, X_val, y_val)
            except Exception as exc:  # noqa: BLE001
                logger.debug("HyperparameterOptimizer LSTM trial failed: {}", exc)
                return 0.0
            val_acc = float(metrics["val_accuracy"])
            params = {
                "sequence_length": seq_len,
                "dropout": dropout,
                "hidden_units": hidden,
            }
            optimizer._log_trial_to_mlflow(
                trial_number=trial.number,
                model_name="LSTMModel",
                params=params,
                val_accuracy=val_acc,
            )
            return val_acc

        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def _log_trial_to_mlflow(
        self,
        trial_number: int,
        model_name: str,
        params: dict[str, Any],
        val_accuracy: float,
    ) -> None:
        """Attempt to log a single Optuna trial to MLflow.

        Failures are caught and logged as warnings — MLflow unavailability
        must never abort an optimisation run.

        Args:
            trial_number: Optuna trial index (0-based).
            model_name: Human-readable model class name (e.g. ``"XGBoostModel"``).
            params: Hyperparameter dict for this trial.
            val_accuracy: Objective metric value for this trial.
        """
        try:
            import mlflow  # noqa: PLC0415

            tracking_uri = self._mlflow_uri
            if tracking_uri is None:
                from src.config import settings as _s  # noqa: PLC0415
                tracking_uri = _s.mlflow.tracking_uri

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self._mlflow_experiment)

            with mlflow.start_run(
                run_name=f"hyperopt_{model_name}_trial_{trial_number:04d}",
                nested=True,
            ):
                mlflow.set_tag("model_class", model_name)
                mlflow.log_params(params)
                mlflow.log_metric("val_accuracy", val_accuracy)

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "HyperparameterOptimizer: MLflow trial logging failed (trial={}) — {} (continuing)",
                trial_number,
                exc,
            )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"HyperparameterOptimizer("
            f"experiment='{self._mlflow_experiment}', seed={self._seed})"
        )


__all__ = ["HyperparameterOptimizer"]
