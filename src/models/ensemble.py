"""Ensemble meta-learner that combines LSTM and regime-specific XGBoost outputs.

This module implements a lightweight stacked ensemble whose inputs are the
probability estimates from two base models plus the encoded market regime:

    ``[lstm_prob, xgb_prob, regime_encoded]``

A :class:`~sklearn.linear_model.LogisticRegression` meta-learner is chosen
for simplicity and interpretability: its three coefficients indicate the
relative contribution of each base-model signal and the regime context.

Build order (CLAUDE.md §8):
    1. ``base_model.py``     ← done
    2. ``xgb_model.py``      ← done
    3. ``regime_trainer.py`` ← done
    4. ``lstm_model.py``     ← done
    5. **``ensemble.py``**   ← this file
    6. ``transformer_model.py`` ← optional, only if LSTM Sharpe < 1.0

Usage::

    ensemble = EnsembleModel()
    meta_X_train = np.column_stack([lstm_proba_train, xgb_proba_train, regime_enc_train])
    meta_X_val   = np.column_stack([lstm_proba_val,   xgb_proba_val,   regime_enc_val])
    ensemble.fit(meta_X_train, y_train, meta_X_val, y_val)
    signals = ensemble.predict_signal(meta_X_test, regime_label="TRENDING_BULL")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.config import RegimeConfig, settings as _global_settings
from src.models.base_model import AbstractBaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_FILENAME: str = "ensemble_model.pkl"
_METADATA_FILENAME: str = "ensemble_metadata.json"

# Expected number of meta-learner input columns
_N_META_FEATURES: int = 3
_META_FEATURE_NAMES: tuple[str, ...] = ("lstm_prob", "xgb_prob", "regime_encoded")


# ---------------------------------------------------------------------------
# EnsembleModel
# ---------------------------------------------------------------------------


class EnsembleModel(AbstractBaseModel):
    """Logistic-regression meta-learner stacking LSTM + XGBoost predictions.

    The meta-learner receives a 3-column matrix per sample:

    * Column 0 — ``lstm_prob``: P(up) from :class:`~src.models.lstm_model.LSTMModel`.
    * Column 1 — ``xgb_prob``: P(up) from the regime-appropriate
      :class:`~src.models.xgb_model.XGBoostModel`.
    * Column 2 — ``regime_encoded``: Integer regime encoding ``[0, 4]``
      produced by :func:`~src.regimes.features.get_regime_features`.

    The meta-learner is intentionally simple (LogisticRegression) to keep the
    final decision layer interpretable.  Its three coefficients directly show
    the relative confidence weight given to each signal.

    Args:
        C: Inverse of regularisation strength (scikit-learn convention).
            Smaller values → stronger regularisation.
        max_iter: Maximum number of iterations for the LBFGS solver.
        seed: Random seed for reproducibility.
        regime_cfg: Regime configuration for
            :meth:`~src.models.base_model.AbstractBaseModel.predict_signal`.
            Defaults to the module-level singleton.

    Attributes:
        _lr: Fitted :class:`~sklearn.linear_model.LogisticRegression` instance.
        _n_train_samples: Number of training samples used in the last
            :meth:`fit` call (persisted in metadata JSON).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        seed: int = _global_settings.project.seed,
        regime_cfg: RegimeConfig | None = None,
    ) -> None:
        super().__init__(regime_cfg=regime_cfg)
        self._C: float = C
        self._max_iter: int = max_iter
        self._seed: int = seed
        self._lr: LogisticRegression | None = None
        self._n_train_samples: int = 0

    # ------------------------------------------------------------------
    # AbstractBaseModel implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Train the logistic regression meta-learner.

        Args:
            X_train: Training meta-features, shape ``(n_train, 3)`` where
                columns are ``[lstm_prob, xgb_prob, regime_encoded]``.
            y_train: Training binary labels ``{0, 1}``, shape ``(n_train,)``.
            X_val: Validation meta-features, shape ``(n_val, 3)``.
            y_val: Validation binary labels, shape ``(n_val,)``.

        Returns:
            Metrics dict with keys:

                - ``"val_accuracy"`` (float)
                - ``"train_accuracy"`` (float)
                - ``"n_train"`` (int)
                - ``"n_val"`` (int)

        Raises:
            ValueError: If ``X_train`` or ``X_val`` do not have exactly 3
                columns or if ``y_train`` is empty.
        """
        self._validate_meta_X(X_train, "X_train")
        self._validate_meta_X(X_val, "X_val")
        if len(y_train) == 0:
            raise ValueError("EnsembleModel.fit: y_train must not be empty.")

        np.random.seed(self._seed)

        self._lr = LogisticRegression(
            C=self._C,
            max_iter=self._max_iter,
            solver="lbfgs",
            random_state=self._seed,
        )
        self._lr.fit(X_train, y_train.astype(int))

        train_proba = self._lr.predict_proba(X_train)[:, 1]
        val_proba = self._lr.predict_proba(X_val)[:, 1]

        train_accuracy = float(
            np.mean((train_proba >= 0.5).astype(int) == y_train.astype(int))
        )
        val_accuracy = float(
            np.mean((val_proba >= 0.5).astype(int) == y_val.astype(int))
        )

        self._n_train_samples = int(len(y_train))
        self._is_fitted = True

        logger.info(
            "EnsembleModel fitted — train_acc={:.4f}, val_acc={:.4f}, "
            "n_train={}, n_val={}",
            train_accuracy,
            val_accuracy,
            self._n_train_samples,
            len(y_val),
        )

        return {
            "val_accuracy": val_accuracy,
            "train_accuracy": train_accuracy,
            "n_train": self._n_train_samples,
            "n_val": int(len(y_val)),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each meta-sample.

        Args:
            X: Meta-feature matrix, shape ``(n_samples, 3)``.

        Returns:
            1-D float64 array of probabilities, shape ``(n_samples,)``.
            All values are in ``[0, 1]``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If ``X`` does not have 3 columns.
        """
        self._assert_fitted()
        self._validate_meta_X(X, "X")
        assert self._lr is not None  # type guard
        return self._lr.predict_proba(X)[:, 1].astype(np.float64)

    def save(self, path: Path) -> None:
        """Serialise the fitted ensemble model to *path*.

        Writes two files:
        * ``ensemble_model.pkl`` — the fitted :class:`LogisticRegression` (joblib).
        * ``ensemble_metadata.json`` — hyperparameters and training metadata.

        Args:
            path: Directory to write artefacts into (created if absent).

        Raises:
            RuntimeError: If the model has not been fitted.
            OSError: If the directory cannot be created or files cannot be written.
        """
        self._assert_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Serialise LogisticRegression via joblib
        joblib.dump(self._lr, path / _MODEL_FILENAME)

        # Metadata JSON
        metadata: dict[str, Any] = {
            "C": self._C,
            "max_iter": self._max_iter,
            "seed": self._seed,
            "n_train_samples": self._n_train_samples,
            "meta_feature_names": list(_META_FEATURE_NAMES),
        }
        with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info("EnsembleModel saved → {}", path)

    def load(self, path: Path) -> None:
        """Deserialise a previously saved ensemble model from *path*.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If ``ensemble_model.pkl`` or
                ``ensemble_metadata.json`` are missing.
        """
        path = Path(path)
        model_file = path / _MODEL_FILENAME
        meta_file = path / _METADATA_FILENAME

        if not model_file.exists():
            raise FileNotFoundError(
                f"EnsembleModel.load: model file not found: {model_file}"
            )
        if not meta_file.exists():
            raise FileNotFoundError(
                f"EnsembleModel.load: metadata file not found: {meta_file}"
            )

        self._lr = joblib.load(model_file)

        with open(meta_file, encoding="utf-8") as fh:
            metadata = json.load(fh)

        self._C = float(metadata.get("C", self._C))
        self._max_iter = int(metadata.get("max_iter", self._max_iter))
        self._seed = int(metadata.get("seed", self._seed))
        self._n_train_samples = int(metadata.get("n_train_samples", 0))
        self._is_fitted = True

        logger.info("EnsembleModel loaded ← {}", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_meta_X(X: np.ndarray, name: str) -> None:
        """Raise ValueError if *X* does not have exactly 3 columns.

        Args:
            X: Array to validate.
            name: Argument name used in the error message.

        Raises:
            ValueError: If ``X.ndim != 2`` or ``X.shape[1] != 3``.
        """
        if X.ndim != 2 or X.shape[1] != _N_META_FEATURES:
            n_cols = X.shape[1] if X.ndim == 2 else X.ndim
            raise ValueError(
                f"EnsembleModel expects {name} with {_N_META_FEATURES} columns "
                f"[lstm_prob, xgb_prob, regime_encoded], got {n_cols} column(s)."
            )

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"EnsembleModel("
            f"C={self._C}, max_iter={self._max_iter}, "
            f"fitted={self._is_fitted})"
        )


__all__ = ["EnsembleModel"]
