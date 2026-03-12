"""Abstract base model interface for all DOGE prediction models.

Every model in ``src/models/`` **must** inherit from :class:`AbstractBaseModel`
and implement all abstract methods.  This guarantees a uniform API that the
ensemble, regime router, inference engine, and RL system can depend on.

Build order (CLAUDE.md Section 8):
    1. base_model.py  ← this file
    2. xgb_model.py
    3. regime_trainer.py
    4. lstm_model.py
    5. ensemble.py
    6. transformer_model.py  (only if LSTM Sharpe < 1.0)

Interface contract:
    - ``fit``          — train the model, return a metrics dict
    - ``predict_proba``— return P(class=1), shape (n_samples,)
    - ``save``         — serialise model + metadata to ``path/``
    - ``load``         — deserialise model from ``path/``
    - ``predict_signal``— apply regime-aware confidence threshold and return
                         BUY / SELL / HOLD string

Threshold contract:
    The confidence threshold is ALWAYS loaded from ``config/regime_config.yaml``
    via :func:`~src.config.regime_config`.  It is NEVER hardcoded.
    ``predict_signal`` delegates threshold lookup to
    :meth:`~src.config.RegimeConfig.get_confidence_threshold`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import RegimeConfig, regime_config as _default_regime_config

# ---------------------------------------------------------------------------
# Signal constants
# ---------------------------------------------------------------------------

SIGNAL_BUY: str = "BUY"
SIGNAL_SELL: str = "SELL"
SIGNAL_HOLD: str = "HOLD"

# Default regime used for threshold lookups when no regime label is supplied.
_DEFAULT_REGIME: str = "RANGING_LOW_VOL"


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class AbstractBaseModel(ABC):
    """Abstract interface that every prediction model must implement.

    Sub-classes receive the regime configuration at construction time so that
    ``predict_signal`` can apply the correct confidence threshold without any
    hardcoded values.

    Args:
        regime_cfg: Regime configuration instance.  Defaults to the
            module-level singleton loaded from ``config/regime_config.yaml``.

    Attributes:
        regime_cfg: Validated :class:`~src.config.RegimeConfig` instance.
        _is_fitted: Set to ``True`` after a successful :meth:`fit` call.
    """

    def __init__(self, regime_cfg: RegimeConfig | None = None) -> None:
        """Initialise the base model.

        Args:
            regime_cfg: Regime configuration.  If *None*, the module-level
                singleton is used.
        """
        self.regime_cfg: RegimeConfig = (
            regime_cfg if regime_cfg is not None else _default_regime_config
        )
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every sub-class
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Train the model on the supplied fold data.

        Args:
            X_train: Training feature matrix, shape ``(n_train, n_features)``.
            y_train: Training binary labels, shape ``(n_train,)``.
            X_val: Validation feature matrix, shape ``(n_val, n_features)``.
            y_val: Validation binary labels, shape ``(n_val,)``.

        Returns:
            Metrics dict with at least the key ``"val_accuracy"`` (float).
            Sub-classes may include additional keys (e.g.
            ``"best_iteration"``, ``"train_loss"``).

        Raises:
            ValueError: If input shapes are incompatible.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each sample.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            1-D float array of probabilities, shape ``(n_samples,)``.
            All values must be in ``[0, 1]``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialise the fitted model and any ancillary artefacts to *path*.

        The implementation must create *path* (and all parents) if it does
        not already exist.  At minimum it must persist enough state that
        :meth:`load` can fully reconstruct the model.

        Args:
            path: Directory to write artefacts into.

        Raises:
            RuntimeError: If the model has not been fitted.
            OSError: If the directory cannot be created or files cannot be written.
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Deserialise a previously saved model from *path*.

        After this call the model must behave identically to a freshly
        :meth:`fit`-ted and :meth:`save`-d model.  Sets ``_is_fitted = True``.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If expected artefact files are missing.
            OSError: If any artefact file cannot be read.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete method — implemented here, never overridden
    # ------------------------------------------------------------------

    def predict_signal(
        self,
        X: np.ndarray,
        regime_label: str | None = None,
    ) -> list[str]:
        """Apply the regime-aware confidence threshold and return trade signals.

        For each sample the method:

        1. Calls :meth:`predict_proba` to obtain ``P(up)`` values.
        2. Loads the confidence threshold for *regime_label* from
           :attr:`regime_cfg` (config YAML — never hardcoded).
        3. Returns ``"BUY"`` when ``P(up) >= threshold``, ``"SELL"`` when
           ``P(down) >= threshold`` (i.e. ``1 - P(up) >= threshold``),
           otherwise ``"HOLD"``.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
            regime_label: Current market regime string
                (e.g. ``"TRENDING_BULL"``).  If *None* or unrecognised, the
                default regime ``"RANGING_LOW_VOL"`` is used (highest threshold,
                most conservative fallback).

        Returns:
            List of ``"BUY"``, ``"SELL"``, or ``"HOLD"`` strings, length
            ``n_samples``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__}.predict_signal called before fit()"
            )

        # Resolve threshold — never hardcoded
        resolved_regime = regime_label if regime_label is not None else _DEFAULT_REGIME
        try:
            threshold: float = self.regime_cfg.get_confidence_threshold(resolved_regime)
        except KeyError:
            logger.warning(
                "predict_signal: unknown regime '{}' — falling back to '{}'",
                resolved_regime,
                _DEFAULT_REGIME,
            )
            threshold = self.regime_cfg.get_confidence_threshold(_DEFAULT_REGIME)

        proba: np.ndarray = self.predict_proba(X)

        signals: list[str] = []
        for p in proba:
            if p >= threshold:
                signals.append(SIGNAL_BUY)
            elif (1.0 - p) >= threshold:
                signals.append(SIGNAL_SELL)
            else:
                signals.append(SIGNAL_HOLD)

        logger.debug(
            "predict_signal: regime='{}', threshold={:.4f}, "
            "n_buy={}, n_sell={}, n_hold={}",
            resolved_regime,
            threshold,
            signals.count(SIGNAL_BUY),
            signals.count(SIGNAL_SELL),
            signals.count(SIGNAL_HOLD),
        )

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        """Raise :class:`RuntimeError` if the model has not been fitted.

        Call this at the top of :meth:`predict_proba`, :meth:`save`, etc.
        to produce a clear error message rather than a cryptic attribute error.

        Raises:
            RuntimeError: If ``_is_fitted`` is ``False``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit() first."
            )

    def directional_accuracy(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Compute the fraction of samples where the predicted direction matches.

        A correct prediction is when ``round(predict_proba(X)[i]) == y_true[i]``,
        i.e. P(up) > 0.5 matches y_true == 1, and P(up) < 0.5 matches y_true == 0.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.
            y_true: True binary labels, shape ``(n_samples,)``.

        Returns:
            Directional accuracy in ``[0, 1]``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._assert_fitted()
        proba = self.predict_proba(X)
        predicted = (proba >= 0.5).astype(int)
        return float(np.mean(predicted == y_true.astype(int)))

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"{type(self).__name__}("
            f"fitted={self._is_fitted})"
        )


__all__ = [
    "AbstractBaseModel",
    "SIGNAL_BUY",
    "SIGNAL_SELL",
    "SIGNAL_HOLD",
]
