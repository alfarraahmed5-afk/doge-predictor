"""Regime-aware routing of XGBoost inference calls.

At inference time the :class:`RegimeRouter` is given the current market regime
label and returns the appropriate pre-fitted
:class:`~src.models.xgb_model.XGBoostModel`.  If a regime-specific model is
unavailable it falls back to a global (all-regime) XGBoost model.

Build order (CLAUDE.md §8):
    5. **``regime_router.py``** ← this file (companion to ``ensemble.py``)

Usage::

    router = RegimeRouter(
        regime_models={"TRENDING_BULL": bull_model, "TRENDING_BEAR": bear_model},
        global_model=global_xgb,
    )
    xgb_model = router.route("TRENDING_BULL")  # → bull_model
    xgb_model = router.route("DECOUPLED")      # → global_xgb (fallback)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.models.xgb_model import XGBoostModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_FILENAME: str = "router_metadata.json"
_GLOBAL_SUBDIR: str = "_global"

_ALL_REGIMES: tuple[str, ...] = (
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
)


# ---------------------------------------------------------------------------
# RegimeRouter
# ---------------------------------------------------------------------------


class RegimeRouter:
    """Routes inference calls to the correct regime-specific XGBoostModel.

    The router holds a mapping from regime label to fitted
    :class:`~src.models.xgb_model.XGBoostModel`.  When the requested regime
    is not in the mapping (e.g. not enough training data to fit a
    regime-specific model), the optional *global_model* is returned instead.

    Args:
        regime_models: Dict mapping regime name to a fitted
            :class:`~src.models.xgb_model.XGBoostModel`.  May be empty if
            *global_model* is provided.
        global_model: Fallback model used when the current regime is not
            covered by *regime_models*.  If *None* and the regime is not
            found, :meth:`route` raises :class:`ValueError`.

    Raises:
        ValueError: If both *regime_models* and *global_model* are absent or
            contain unfitted models.
    """

    def __init__(
        self,
        regime_models: dict[str, XGBoostModel] | None = None,
        global_model: XGBoostModel | None = None,
    ) -> None:
        self._regime_models: dict[str, XGBoostModel] = (
            regime_models if regime_models is not None else {}
        )
        self._global_model: XGBoostModel | None = global_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, regime_label: str) -> XGBoostModel:
        """Return the XGBoostModel appropriate for the given regime.

        Args:
            regime_label: Current market regime (e.g. ``"TRENDING_BULL"``).

        Returns:
            Fitted :class:`~src.models.xgb_model.XGBoostModel`.

        Raises:
            ValueError: If the regime is not covered by *regime_models* AND
                no *global_model* is available.
        """
        model = self._regime_models.get(regime_label)
        if model is not None:
            logger.debug(
                "RegimeRouter: routing '{}' → regime-specific model", regime_label
            )
            return model

        if self._global_model is not None:
            logger.debug(
                "RegimeRouter: regime '{}' not found — using global model",
                regime_label,
            )
            return self._global_model

        raise ValueError(
            f"RegimeRouter.route: no model available for regime '{regime_label}' "
            "and no global fallback model is set."
        )

    def available_regimes(self) -> list[str]:
        """Return a sorted list of regime labels with dedicated models.

        Returns:
            Sorted list of regime label strings.
        """
        return sorted(self._regime_models.keys())

    def has_regime(self, regime_label: str) -> bool:
        """Return ``True`` if a regime-specific model exists for *regime_label*.

        Args:
            regime_label: Regime name to check.

        Returns:
            ``True`` when a regime-specific model is registered for the label.
        """
        return regime_label in self._regime_models

    def has_global_model(self) -> bool:
        """Return ``True`` if a global fallback model is registered.

        Returns:
            ``True`` when a global model is available.
        """
        return self._global_model is not None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise all registered models to subdirectories inside *path*.

        Directory layout::

            path/
              router_metadata.json
              TRENDING_BULL/
                xgb_model.json
                xgb_metadata.json
              TRENDING_BEAR/
                ...
              _global/          ← only if global model is set
                xgb_model.json
                xgb_metadata.json

        Args:
            path: Root directory for the router artefacts (created if absent).

        Raises:
            OSError: If any artefact file cannot be written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each regime model
        for regime, model in self._regime_models.items():
            model.save(path / regime)
            logger.debug("RegimeRouter: saved regime '{}' model → {}", regime, path / regime)

        # Save global fallback model
        if self._global_model is not None:
            self._global_model.save(path / _GLOBAL_SUBDIR)
            logger.debug("RegimeRouter: saved global model → {}", path / _GLOBAL_SUBDIR)

        # Write metadata
        metadata: dict[str, Any] = {
            "available_regimes": self.available_regimes(),
            "has_global": self._global_model is not None,
        }
        with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

        logger.info(
            "RegimeRouter saved — regimes={}, has_global={} → {}",
            self.available_regimes(),
            self._global_model is not None,
            path,
        )

    def load(self, path: Path) -> None:
        """Restore the router from a directory written by :meth:`save`.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If ``router_metadata.json`` is missing.
            OSError: If any model artefact cannot be read.
        """
        path = Path(path)
        meta_file = path / _METADATA_FILENAME
        if not meta_file.exists():
            raise FileNotFoundError(
                f"RegimeRouter.load: metadata not found: {meta_file}"
            )

        with open(meta_file, encoding="utf-8") as fh:
            metadata = json.load(fh)

        # Load regime-specific models
        self._regime_models = {}
        for regime in metadata.get("available_regimes", []):
            model = XGBoostModel()
            model.load(path / regime)
            self._regime_models[regime] = model
            logger.debug("RegimeRouter: loaded regime '{}' model ← {}", regime, path / regime)

        # Load global fallback model
        if metadata.get("has_global", False):
            self._global_model = XGBoostModel()
            self._global_model.load(path / _GLOBAL_SUBDIR)
            logger.debug("RegimeRouter: loaded global model ← {}", path / _GLOBAL_SUBDIR)
        else:
            self._global_model = None

        logger.info(
            "RegimeRouter loaded — regimes={}, has_global={} ← {}",
            self.available_regimes(),
            self._global_model is not None,
            path,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"RegimeRouter("
            f"regimes={self.available_regimes()}, "
            f"has_global={self._global_model is not None})"
        )


__all__ = ["RegimeRouter"]
