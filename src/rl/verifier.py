"""Prediction verifier for the RL self-teaching system.

The :class:`PredictionVerifier` runs on a scheduled cadence (every hour, via
APScheduler in ``scripts/serve.py``) and closes the loop between predictions
and outcomes:

1. Queries ``doge_predictions`` for rows whose ``target_open_time`` has passed
   and whose ``verified_at`` is NULL (unverified).
2. Fetches the actual OHLCV close price at ``target_open_time`` from storage.
3. Computes ``actual_direction`` vs ``price_at_prediction``
   (CRITICAL — never vs T-1 close).
4. Calls :func:`~src.rl.reward.compute_reward` to obtain the full
   :class:`~src.processing.schemas.RewardResult`.
5. Writes the outcome columns back to ``doge_predictions`` (prediction fields
   are immutable — only outcome columns are updated).
6. Pushes the completed experience tuple to the replay buffer via
   :class:`~src.rl.replay_buffer.ReplayBuffer`.

Safety checks
-------------
* The verifier NEVER processes predictions whose ``target_open_time`` is still
  in the future.  ``DogeStorage.get_matured_unverified`` enforces this.
* Rows with missing OHLCV data are skipped with a warning (not discarded —
  they will be retried on the next hourly run).
* Interpolated candle outcomes are NOT pushed to the replay buffer.

Usage (via APScheduler in ``scripts/serve.py``)::

    verifier = PredictionVerifier(storage, rl_cfg=rl_cfg)
    # Every hour, called by scheduler:
    n_verified = verifier.run_verification()
"""

from __future__ import annotations

import time
import uuid
from typing import Final

import pandas as pd
from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg
from src.processing.schemas import PredictionRecord, RewardResult
from src.processing.storage import DogeStorage
from src.rl.replay_buffer import ReplayBuffer
from src.rl.reward import compute_reward

__all__ = ["PredictionVerifier"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTERVAL_MS: Final[int] = 3_600_000  # 1-hour candle in milliseconds
_SYMBOL: Final[str] = "DOGEUSDT"
_INTERVAL: Final[str] = "1h"


class PredictionVerifier:
    """Verifies matured predictions and records reward scores.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance used to
            read predictions, write outcomes, and push to the replay buffer.
        rl_cfg: Loaded :class:`~src.config.RLConfig`.  Defaults to the global
            singleton when not supplied (useful for production; tests should
            inject a custom instance).
        replay_buffer: Optional :class:`~src.rl.replay_buffer.ReplayBuffer`
            instance.  Created automatically if not supplied.
        skip_interpolated: When ``True`` (default), candles marked as
            ``is_interpolated=True`` in OHLCV are skipped — their prices are
            synthetic and must not pollute the reward signal.
    """

    def __init__(
        self,
        storage: DogeStorage,
        rl_cfg: RLConfig | None = None,
        replay_buffer: ReplayBuffer | None = None,
        skip_interpolated: bool = True,
    ) -> None:
        self._storage: DogeStorage = storage
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        self._replay_buffer: ReplayBuffer = replay_buffer or ReplayBuffer(storage, self._rl_cfg)
        self._skip_interpolated: bool = skip_interpolated
        logger.info("PredictionVerifier initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_verification(self, as_of_ts: int | None = None) -> int:
        """Verify all matured predictions and record rewards.

        Args:
            as_of_ts: Current timestamp as UTC epoch milliseconds.  Defaults
                to ``int(time.time() * 1000)`` when ``None``.

        Returns:
            Number of predictions successfully verified in this run.
        """
        now_ms: int = as_of_ts if as_of_ts is not None else int(time.time() * 1000)

        try:
            pending = self._storage.get_matured_unverified(now_ms)
        except Exception as exc:  # noqa: BLE001
            logger.error("run_verification: get_matured_unverified failed: {}", exc)
            return 0

        if not pending:
            logger.debug("run_verification: no matured unverified predictions")
            return 0

        logger.info("run_verification: {} matured predictions to process", len(pending))

        n_verified = 0
        for record in pending:
            try:
                ok = self._verify_single(record, now_ms)
                if ok:
                    n_verified += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "run_verification: unhandled error for prediction_id={}: {}",
                    record.prediction_id,
                    exc,
                )

        logger.info(
            "run_verification: verified {}/{} predictions",
            n_verified,
            len(pending),
        )
        return n_verified

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_single(self, record: PredictionRecord, now_ms: int) -> bool:
        """Verify one matured prediction.

        Args:
            record: The unverified prediction record.
            now_ms: Current time as UTC epoch milliseconds (for safety check).

        Returns:
            ``True`` if the record was successfully verified and written;
            ``False`` if it was skipped (OHLCV missing, interpolated, etc.).
        """
        # Safety: never process a candle that hasn't closed yet
        if record.target_open_time > now_ms - _INTERVAL_MS:
            logger.warning(
                "_verify_single: target_open_time {} is too recent (now={}); "
                "skipping prediction_id={}",
                record.target_open_time,
                now_ms,
                record.prediction_id,
            )
            return False

        # Fetch the actual candle at target_open_time
        actual_candle = self._fetch_target_candle(record)
        if actual_candle is None:
            return False  # OHLCV data not available yet — retry later

        actual_close: float = float(actual_candle["close"])
        is_interpolated: bool = bool(actual_candle.get("is_interpolated", False))

        if self._skip_interpolated and is_interpolated:
            logger.warning(
                "_verify_single: candle at target_open_time={} is interpolated; "
                "skipping prediction_id={}",
                record.target_open_time,
                record.prediction_id,
            )
            return False

        # Compute actual direction vs price_at_prediction (CRITICAL — not T-1 close)
        actual_direction: int = 1 if actual_close > record.price_at_prediction else -1

        # Compute reward
        reward_result: RewardResult = compute_reward(
            horizon=record.horizon_label,
            predicted_direction=record.predicted_direction,
            actual_direction=actual_direction,
            predicted_prob=record.confidence_score,
            price_at_prediction=record.price_at_prediction,
            actual_price=actual_close,
            regime=record.regime_label,
            rl_cfg=self._rl_cfg,
        )

        # Write outcome back to doge_predictions (immutable prediction fields untouched)
        verified_at: int = int(time.time() * 1000)
        outcome: dict = {
            "actual_price": actual_close,
            "actual_direction": actual_direction,
            "reward_score": reward_result.reward_score,
            "direction_correct": reward_result.direction_correct,
            "error_pct": reward_result.error_pct,
            "verified_at": verified_at,
        }

        try:
            updated = self._storage.update_prediction_outcome(record.prediction_id, outcome)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "_verify_single: update_prediction_outcome failed for {}: {}",
                record.prediction_id,
                exc,
            )
            return False

        if not updated:
            logger.warning(
                "_verify_single: no row updated for prediction_id={} "
                "(already verified or missing)",
                record.prediction_id,
            )
            return False

        # Push to replay buffer
        self._push_to_replay(record, reward_result, actual_close, verified_at)

        logger.debug(
            "_verify_single: prediction_id={} verified | "
            "horizon={} regime={} direction_correct={} reward={:.4f}",
            record.prediction_id,
            record.horizon_label,
            record.regime_label,
            reward_result.direction_correct,
            reward_result.reward_score,
        )
        return True

    def _fetch_target_candle(
        self, record: PredictionRecord
    ) -> dict | None:
        """Fetch the OHLCV candle at record.target_open_time.

        Args:
            record: The prediction record whose target candle we need.

        Returns:
            Dict with candle fields, or ``None`` if no matching candle found.
        """
        target_ms = record.target_open_time
        try:
            df = self._storage.get_ohlcv(
                symbol=_SYMBOL,
                interval=_INTERVAL,
                start_ms=target_ms,
                end_ms=target_ms + _INTERVAL_MS,  # exclusive end → exact candle
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_fetch_target_candle: storage error for target_open_time={}: {}",
                target_ms,
                exc,
            )
            return None

        if df.empty:
            logger.warning(
                "_fetch_target_candle: no OHLCV row for symbol={} "
                "target_open_time={} (prediction_id={})",
                _SYMBOL,
                target_ms,
                record.prediction_id,
            )
            return None

        return df.iloc[0].to_dict()

    def _push_to_replay(
        self,
        record: PredictionRecord,
        reward: RewardResult,
        actual_price: float,
        verified_at: int,
    ) -> None:
        """Push a verified experience to the replay buffer.

        Args:
            record: The verified prediction record.
            reward: The computed reward components.
            actual_price: Actual close price at the target candle.
            verified_at: Verification timestamp, UTC epoch ms.
        """
        try:
            self._replay_buffer.push(
                horizon=record.horizon_label,
                regime=record.regime_label,
                reward_score=reward.reward_score,
                model_version=record.model_version,
                predicted_price=None,  # not stored in PredictionRecord
                actual_price=actual_price,
                created_at=verified_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_push_to_replay: failed for prediction_id={}: {}",
                record.prediction_id,
                exc,
            )
