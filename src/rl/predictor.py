"""Multi-horizon prediction generator for the RL self-teaching system.

The :class:`MultiHorizonPredictor` sits between the inference engine and the
Prediction Store.  After the ensemble signal is produced for the *current*
candle it generates one :class:`~src.processing.schemas.PredictionRecord` per
active horizon and persists them via
:class:`~src.processing.storage.DogeStorage`.

Only horizons enabled by the current :class:`~src.rl.curriculum.CurriculumManager`
stage are emitted.  Horizons disabled by the curriculum produce no rows and
generate no DB writes.

Usage (called inside the inference pipeline after Step 12)::

    predictor = MultiHorizonPredictor(storage, curriculum_mgr)
    records = predictor.generate_and_store(
        open_time=1_700_000_000_000,
        close_price=0.102,
        predicted_direction=1,
        ensemble_prob=0.68,
        lstm_prob=0.71,
        xgb_prob=0.65,
        regime_label="TRENDING_BULL",
        model_version="v1.0",
        now_ms=1_700_000_000_000,
    )
"""

from __future__ import annotations

import time
import uuid
from typing import Final

from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg
from src.processing.schemas import HORIZON_CANDLES, PredictionRecord
from src.processing.storage import DogeStorage
from src.rl.curriculum import CurriculumManager

__all__ = ["MultiHorizonPredictor"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYMBOL: Final[str] = "DOGEUSDT"
_INTERVAL_MS: Final[int] = 3_600_000  # 1-hour candle


class MultiHorizonPredictor:
    """Generates and stores multi-horizon prediction records.

    For each horizon active in the current curriculum stage a
    :class:`~src.processing.schemas.PredictionRecord` is constructed and
    persisted.  The method is intentionally idempotent with respect to the
    storage layer — duplicate open_time/horizon combinations are silently
    ignored by the underlying upsert.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` for persistence.
        curriculum: :class:`~src.rl.curriculum.CurriculumManager` that controls
            which horizons are active.
        rl_cfg: :class:`~src.config.RLConfig` for horizon candle counts.
            Defaults to the global singleton.
    """

    def __init__(
        self,
        storage: DogeStorage,
        curriculum: CurriculumManager,
        rl_cfg: RLConfig | None = None,
    ) -> None:
        self._storage: DogeStorage = storage
        self._curriculum: CurriculumManager = curriculum
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        logger.info(
            "MultiHorizonPredictor initialised (active_horizons={})",
            curriculum.active_horizons(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_and_store(
        self,
        open_time: int,
        close_price: float,
        predicted_direction: int,
        ensemble_prob: float,
        lstm_prob: float,
        xgb_prob: float,
        regime_label: str,
        model_version: str,
        now_ms: int | None = None,
    ) -> list[PredictionRecord]:
        """Generate one PredictionRecord per active horizon and persist them.

        The ``confidence_score`` is derived from ``ensemble_prob`` as the
        absolute deviation from 0.5, projected onto ``[0.5, 1.0]``.

        ``target_open_time`` is computed as::

            open_time + horizon_candles × _INTERVAL_MS

        Args:
            open_time: UTC epoch milliseconds of the prediction candle.
            close_price: Close price at the prediction candle (USD).
            predicted_direction: Direction signal: -1 (short), 0 (hold), 1 (long).
            ensemble_prob: Ensemble probability output in [0, 1].
            lstm_prob: Raw LSTM probability output in [0, 1].
            xgb_prob: Raw XGBoost probability output in [0, 1].
            regime_label: Current market regime label.
            model_version: Model version string.
            now_ms: Current UTC epoch milliseconds — defaults to
                ``int(time.time() * 1000)`` when ``None``.

        Returns:
            List of :class:`~src.processing.schemas.PredictionRecord` objects
            that were successfully stored (one per active horizon).  Records
            that could not be stored due to validation errors are omitted.

        Raises:
            ValueError: If ``close_price <= 0`` or ``predicted_direction`` is
                not in ``{-1, 0, 1}`` or ``ensemble_prob`` is outside ``[0, 1]``.
        """
        if close_price <= 0:
            raise ValueError(f"close_price must be > 0, got {close_price}")
        if predicted_direction not in (-1, 0, 1):
            raise ValueError(
                f"predicted_direction must be -1, 0, or 1; got {predicted_direction}"
            )
        if not (0.0 <= ensemble_prob <= 1.0):
            raise ValueError(
                f"ensemble_prob must be in [0, 1]; got {ensemble_prob}"
            )

        created_at: int = now_ms if now_ms is not None else int(time.time() * 1000)

        # confidence_score ∈ [0.5, 1.0]: distance from 0.5, scaled to [0.5, 1.0]
        confidence_score: float = 0.5 + abs(ensemble_prob - 0.5)

        active_horizons = self._curriculum.active_horizons()
        stored: list[PredictionRecord] = []

        for horizon in active_horizons:
            candles = HORIZON_CANDLES.get(horizon)
            if candles is None:
                logger.warning(
                    "generate_and_store: unknown horizon '{}'; skipping", horizon
                )
                continue

            target_open_time: int = open_time + candles * _INTERVAL_MS

            try:
                record = PredictionRecord(
                    prediction_id=str(uuid.uuid4()),
                    created_at=created_at,
                    open_time=open_time,
                    symbol=_SYMBOL,
                    horizon_label=horizon,  # type: ignore[arg-type]
                    horizon_candles=candles,
                    target_open_time=target_open_time,
                    price_at_prediction=close_price,
                    predicted_direction=predicted_direction,  # type: ignore[arg-type]
                    confidence_score=confidence_score,
                    lstm_prob=lstm_prob,
                    xgb_prob=xgb_prob,
                    regime_label=regime_label,
                    model_version=model_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "generate_and_store: validation failed for horizon={}: {}",
                    horizon,
                    exc,
                )
                continue

            try:
                self._storage.insert_prediction(record)
                stored.append(record)
                logger.debug(
                    "generate_and_store: stored prediction_id={} horizon={} "
                    "target_open_time={}",
                    record.prediction_id,
                    horizon,
                    target_open_time,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "generate_and_store: insert_prediction failed for horizon={}: {}",
                    horizon,
                    exc,
                )

        logger.info(
            "MultiHorizonPredictor: stored {}/{} predictions (active_horizons={})",
            len(stored),
            len(active_horizons),
            active_horizons,
        )
        return stored

    def active_horizons(self) -> list[str]:
        """Return the currently active prediction horizons.

        Delegates to :meth:`~src.rl.curriculum.CurriculumManager.active_horizons`.

        Returns:
            List of horizon label strings.
        """
        return self._curriculum.active_horizons()
