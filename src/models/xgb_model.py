"""XGBoost binary classification model for DOGE price direction prediction.

This is the first concrete model built in Phase 5.  It must beat 53%
directional accuracy out-of-sample before the LSTM is built (CLAUDE.md §8).

Architecture:
    - Objective: ``binary:logistic``
    - n_estimators: 500 (with early stopping — actual trees may be fewer)
    - learning_rate: 0.05
    - max_depth: 5
    - subsample: 0.8
    - colsample_bytree: 0.8
    - tree_method: ``'hist'`` (fast histogram-based algorithm)
    - scale_pos_weight: computed from class ratio at fit time (class imbalance)
    - early_stopping_rounds: 20 (uses ``eval_set`` on validation fold)

All hyperparameters come from ``config/doge_settings.yaml`` via
``src.config.XGBSettings`` (or fall back to the defaults defined in this
file's constants, which are themselves sourced from CLAUDE.md §8).

Serialisation:
    Uses XGBoost's native ``save_model`` / ``load_model`` (JSON format).
    Feature names are stored alongside the model so inference can verify
    the column list matches what the model was trained on.

Lookahead audit:
    This model class has no feature-computation logic.  It receives a
    pre-built feature matrix where every column has already been verified
    as lookahead-free by the feature pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

from src.config import RegimeConfig, regime_config as _default_regime_config
from src.models.base_model import AbstractBaseModel

# ---------------------------------------------------------------------------
# Default hyperparameters (sourced from CLAUDE.md §8 — all in one place)
# ---------------------------------------------------------------------------

_DEFAULT_N_ESTIMATORS: int = 500
_DEFAULT_LEARNING_RATE: float = 0.05
_DEFAULT_MAX_DEPTH: int = 5
_DEFAULT_SUBSAMPLE: float = 0.8
_DEFAULT_COLSAMPLE_BYTREE: float = 0.8
_DEFAULT_EARLY_STOPPING: int = 20
_DEFAULT_TREE_METHOD: str = "hist"
_DEFAULT_EVAL_METRIC: str = "logloss"
_DEFAULT_OBJECTIVE: str = "binary:logistic"

# File names inside the save directory
_MODEL_FILENAME: str = "xgb_model.json"
_METADATA_FILENAME: str = "xgb_metadata.json"


# ---------------------------------------------------------------------------
# XGBoostModel
# ---------------------------------------------------------------------------


class XGBoostModel(AbstractBaseModel):
    """XGBoost binary classifier implementing the :class:`AbstractBaseModel` API.

    The model predicts the probability of the next candle closing higher than
    the current candle (P(up) = P(class=1)).

    Args:
        n_estimators: Maximum number of boosting rounds.  Early stopping may
            reduce the actual number of trees.
        learning_rate: Shrinkage applied to each tree's contribution.
        max_depth: Maximum depth of each tree.
        subsample: Fraction of rows to sample per tree (row subsampling).
        colsample_bytree: Fraction of features to sample per tree.
        early_stopping_rounds: Number of rounds without improvement on the
            validation set before training stops early.
        tree_method: XGBoost tree construction algorithm (``'hist'``
            recommended for large datasets).
        seed: Random seed for reproducibility.  Defaults to the project seed
            from ``config/settings.yaml`` (42).
        regime_cfg: Regime configuration for threshold lookups in
            :meth:`predict_signal`.  Defaults to the module-level singleton.

    Attributes:
        _booster: The fitted :class:`xgb.Booster` instance (``None`` before fit).
        _feature_names: Column names seen during fit (used for inference checks).
        _best_iteration: Best boosting round determined by early stopping.
        _scale_pos_weight: Class-ratio weight applied during the last fit.
    """

    def __init__(
        self,
        n_estimators: int = _DEFAULT_N_ESTIMATORS,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        subsample: float = _DEFAULT_SUBSAMPLE,
        colsample_bytree: float = _DEFAULT_COLSAMPLE_BYTREE,
        early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING,
        tree_method: str = _DEFAULT_TREE_METHOD,
        seed: int = 42,
        regime_cfg: RegimeConfig | None = None,
    ) -> None:
        """Initialise the XGBoost model with hyperparameters.

        Args:
            n_estimators: Maximum number of boosting rounds.
            learning_rate: Step shrinkage to prevent overfitting.
            max_depth: Maximum tree depth.
            subsample: Row subsampling ratio.
            colsample_bytree: Feature subsampling ratio per tree.
            early_stopping_rounds: Early-stopping patience.
            tree_method: XGBoost construction algorithm (``'hist'``).
            seed: Global random seed.
            regime_cfg: Regime config for :meth:`predict_signal`.
        """
        super().__init__(regime_cfg=regime_cfg)

        self._n_estimators: int = n_estimators
        self._learning_rate: float = learning_rate
        self._max_depth: int = max_depth
        self._subsample: float = subsample
        self._colsample_bytree: float = colsample_bytree
        self._early_stopping_rounds: int = early_stopping_rounds
        self._tree_method: str = tree_method
        self._seed: int = seed

        self._booster: xgb.Booster | None = None
        self._feature_names: list[str] = []
        self._best_iteration: int = 0
        self._scale_pos_weight: float = 1.0

    # ------------------------------------------------------------------
    # AbstractBaseModel implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the XGBoost model on the supplied fold data.

        Computes ``scale_pos_weight`` from the training class distribution to
        handle the typical ~50/50 imbalance in DOGE direction labels.  Uses
        ``eval_set`` for early stopping on the validation fold.

        Args:
            X_train: Training feature matrix, shape ``(n_train, n_features)``.
            y_train: Training binary labels ``{0, 1}``, shape ``(n_train,)``.
            X_val: Validation feature matrix, shape ``(n_val, n_features)``.
            y_val: Validation binary labels, shape ``(n_val,)``.
            feature_names: Optional list of feature column names.  Used for
                ``get_feature_importance()`` and saved with the model.  If
                *None*, generic names ``f0, f1, …`` are used.
            sample_weight: Optional per-sample weight array, shape
                ``(n_train,)``.  When supplied, passed to
                :class:`xgb.DMatrix` as the ``weight`` argument — used by
                the RL self-training loop to weight samples by
                ``|reward_score|``.

        Returns:
            Metrics dict with keys:

                - ``"val_accuracy"`` (float): Directional accuracy on val fold.
                - ``"best_iteration"`` (int): Best boosting round.
                - ``"scale_pos_weight"`` (float): Applied class weight.
                - ``"n_train"`` (int): Training sample count.
                - ``"n_val"`` (int): Validation sample count.

        Raises:
            ValueError: If inputs are empty or labels are not binary.
        """
        if X_train.shape[0] == 0:
            raise ValueError("XGBoostModel.fit: X_train is empty.")
        if len(np.unique(y_train.astype(int))) < 2:
            raise ValueError(
                "XGBoostModel.fit: y_train must contain both classes (0 and 1)."
            )

        # Compute scale_pos_weight from training class ratio
        n_neg: int = int((y_train == 0).sum())
        n_pos: int = int((y_train == 1).sum())
        self._scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

        logger.info(
            "XGBoostModel.fit: n_train={}, n_val={}, "
            "n_pos={}, n_neg={}, scale_pos_weight={:.4f}",
            len(X_train),
            len(X_val),
            n_pos,
            n_neg,
            self._scale_pos_weight,
        )

        # Store feature names
        if feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            n_cols = X_train.shape[1] if X_train.ndim > 1 else 1
            self._feature_names = [f"f{i}" for i in range(n_cols)]

        # Build XGBoost DMatrix objects
        dtrain = xgb.DMatrix(
            X_train,
            label=y_train,
            feature_names=self._feature_names,
            weight=sample_weight,
        )
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            feature_names=self._feature_names,
        )

        params: dict[str, Any] = {
            "objective": _DEFAULT_OBJECTIVE,
            "eval_metric": _DEFAULT_EVAL_METRIC,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "scale_pos_weight": self._scale_pos_weight,
            "tree_method": self._tree_method,
            "seed": self._seed,
            "verbosity": 0,
        }

        evals_result: dict[str, Any] = {}

        try:
            self._booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self._n_estimators,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=self._early_stopping_rounds,
                evals_result=evals_result,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError as exc:
            logger.error("XGBoostModel.fit: XGBoost training error — {}", exc)
            raise

        self._best_iteration = int(self._booster.best_iteration)
        self._is_fitted = True

        # Compute validation directional accuracy
        val_proba = self._booster.predict(
            dval, iteration_range=(0, self._best_iteration + 1)
        )
        val_acc = float(np.mean((val_proba >= 0.5).astype(int) == y_val.astype(int)))

        metrics: dict[str, Any] = {
            "val_accuracy": val_acc,
            "best_iteration": self._best_iteration,
            "scale_pos_weight": self._scale_pos_weight,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        }

        logger.info(
            "XGBoostModel.fit: complete — best_iter={}, val_accuracy={:.4f}",
            self._best_iteration,
            val_acc,
        )

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each sample.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            1-D float array of probabilities, shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._assert_fitted()

        dmatrix = xgb.DMatrix(X, feature_names=self._feature_names)
        proba: np.ndarray = self._booster.predict(  # type: ignore[union-attr]
            dmatrix,
            iteration_range=(0, self._best_iteration + 1),
        )
        return proba.astype(np.float64)

    def save(self, path: Path) -> None:
        """Serialise the model and metadata to *path/*.

        Creates *path* if it does not exist.  Writes:

        - ``xgb_model.json`` — XGBoost native JSON model
        - ``xgb_metadata.json`` — feature names, hyperparameters, best iteration

        Args:
            path: Directory to write artefacts into.

        Raises:
            RuntimeError: If the model has not been fitted.
            OSError: If the directory or files cannot be written.
        """
        self._assert_fitted()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / _MODEL_FILENAME
        try:
            self._booster.save_model(str(model_path))  # type: ignore[union-attr]
            logger.info("XGBoostModel.save: model → {}", model_path)
        except (OSError, xgb.core.XGBoostError) as exc:
            logger.error("XGBoostModel.save: failed to save model — {}", exc)
            raise

        metadata: dict[str, Any] = {
            "feature_names": self._feature_names,
            "best_iteration": self._best_iteration,
            "scale_pos_weight": self._scale_pos_weight,
            "hyperparameters": {
                "n_estimators": self._n_estimators,
                "learning_rate": self._learning_rate,
                "max_depth": self._max_depth,
                "subsample": self._subsample,
                "colsample_bytree": self._colsample_bytree,
                "early_stopping_rounds": self._early_stopping_rounds,
                "tree_method": self._tree_method,
                "seed": self._seed,
            },
        }

        meta_path = path / _METADATA_FILENAME
        try:
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logger.info("XGBoostModel.save: metadata → {}", meta_path)
        except OSError as exc:
            logger.error("XGBoostModel.save: failed to save metadata — {}", exc)
            raise

    def load(self, path: Path) -> None:
        """Deserialise a previously saved model from *path/*.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If ``xgb_model.json`` or
                ``xgb_metadata.json`` are absent.
            OSError: If any file cannot be read.
        """
        path = Path(path)

        model_path = path / _MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoostModel.load: model file not found at {model_path}"
            )

        meta_path = path / _METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"XGBoostModel.load: metadata file not found at {meta_path}"
            )

        try:
            self._booster = xgb.Booster()
            self._booster.load_model(str(model_path))
            logger.info("XGBoostModel.load: model ← {}", model_path)
        except (OSError, xgb.core.XGBoostError) as exc:
            logger.error("XGBoostModel.load: failed to load model — {}", exc)
            raise

        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                metadata: dict[str, Any] = json.load(fh)
            self._feature_names = metadata.get("feature_names", [])
            self._best_iteration = int(metadata.get("best_iteration", 0))
            self._scale_pos_weight = float(metadata.get("scale_pos_weight", 1.0))
            logger.info("XGBoostModel.load: metadata ← {}", meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("XGBoostModel.load: failed to load metadata — {}", exc)
            raise

        self._is_fitted = True

    # ------------------------------------------------------------------
    # XGBoost-specific extras
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, importance_type: str = "gain"
    ) -> dict[str, float]:
        """Return feature importance scores from the fitted booster.

        Args:
            importance_type: One of ``'weight'``, ``'gain'``, ``'cover'``,
                ``'total_gain'``, ``'total_cover'``.  Defaults to ``'gain'``
                (mean gain over all splits on a feature).

        Returns:
            Dict mapping feature name → importance score, sorted descending.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._assert_fitted()

        raw: dict[str, float] = self._booster.get_score(  # type: ignore[union-attr]
            importance_type=importance_type
        )
        # Sort descending by importance value
        return dict(sorted(raw.items(), key=lambda kv: kv[1], reverse=True))

    def get_top_features(
        self, n: int = 10, importance_type: str = "gain"
    ) -> list[tuple[str, float]]:
        """Return the top *n* features by importance.

        Args:
            n: Number of top features to return.
            importance_type: Importance type (see :meth:`get_feature_importance`).

        Returns:
            List of ``(feature_name, score)`` tuples, sorted descending.
        """
        scores = self.get_feature_importance(importance_type=importance_type)
        return list(scores.items())[:n]

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"XGBoostModel("
            f"fitted={self._is_fitted}, "
            f"n_estimators={self._n_estimators}, "
            f"best_iter={self._best_iteration})"
        )


__all__ = ["XGBoostModel"]
