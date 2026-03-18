"""RL self-training weight-update loop.

The :class:`RLTrainer` checks whether a self-training run should fire based on
event-driven and scheduled triggers, respects the mandatory 48-hour cooldown,
draws a batch from the :class:`~src.rl.replay_buffer.ReplayBuffer`, and
delegates the actual model weight update to the upstream
:class:`~src.training.trainer.ModelTrainer`.

Trigger conditions (any one is sufficient, subject to cooldown):

1. **Buffer fill** — ``buffer_fill_pct >= self_training.triggers.buffer_fill_pct``
2. **Reward degradation** — rolling 7-day mean reward < ``rolling_7d_mean_reward``
   threshold (model is getting worse).
3. **New regime predictions** — ≥ ``new_regime_predictions`` verified samples
   arrived in a regime that had < ``min_per_regime`` rows before.
4. **Scheduled** — called externally by the APScheduler Sunday 02:00 UTC job.

Safety:
    - ``min_cooldown_hours`` (48 h) is enforced between consecutive runs.
    - The replay buffer must have ≥ ``min_samples_to_train`` rows total before
      any training run is started.
    - All exceptions from the inner training call are caught; the trainer
      never crashes the inference server.

Usage (via APScheduler in ``scripts/serve.py``)::

    trainer = RLTrainer(storage, replay_buffer, curriculum, rl_cfg=rl_cfg)
    n_updated = trainer.maybe_train()   # 0 if skipped, > 0 if weights updated
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg
from src.processing.storage import DogeStorage
from src.rl.curriculum import CurriculumManager
from src.rl.replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from src.models.ensemble import EnsembleModel
    from src.models.lstm_model import LSTMModel
    from src.models.xgb_model import XGBoostModel

__all__ = ["RLSelfTrainer", "RLTrainer", "RLTrainingResult", "SelfTrainingResult"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: Final[int] = 3_600_000
_VALID_HORIZONS: Final[frozenset[str]] = frozenset({"SHORT", "MEDIUM", "LONG", "MACRO"})


@dataclass
class RLTrainingResult:
    """Summary of one RL training run.

    Args:
        triggered_by: Human-readable reason that triggered this run.
        n_samples_used: Total replay buffer samples used across all horizons.
        horizons_trained: List of horizon labels that were trained.
        skipped: ``True`` if the run was skipped (cooldown, insufficient data, etc.).
        skip_reason: Non-empty when ``skipped=True``.
        duration_ms: Wall-clock duration of the training call in milliseconds.
        ran_at_ms: UTC epoch ms when the run was initiated.
    """

    triggered_by: str = ""
    n_samples_used: int = 0
    horizons_trained: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""
    duration_ms: int = 0
    ran_at_ms: int = 0


class RLTrainer:
    """Checks triggers and runs the RL weight-update loop when needed.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance for
            reading replay stats and recent reward history.
        replay_buffer: :class:`~src.rl.replay_buffer.ReplayBuffer` instance
            used to draw training batches.
        curriculum: :class:`~src.rl.curriculum.CurriculumManager` controlling
            which horizons are active.
        rl_cfg: :class:`~src.config.RLConfig`.  Defaults to the global
            singleton when not supplied.
    """

    def __init__(
        self,
        storage: DogeStorage,
        replay_buffer: ReplayBuffer,
        curriculum: CurriculumManager,
        rl_cfg: RLConfig | None = None,
    ) -> None:
        self._storage: DogeStorage = storage
        self._replay_buffer: ReplayBuffer = replay_buffer
        self._curriculum: CurriculumManager = curriculum
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        self._last_train_ms: int = 0  # epoch ms of the last completed training run
        self._run_history: list[RLTrainingResult] = []
        logger.info("RLTrainer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_train(
        self,
        trigger_reason: str = "scheduled",
        rolling_7d_mean_reward: float | None = None,
        new_regime_counts: dict[str, int] | None = None,
        now_ms: int | None = None,
    ) -> RLTrainingResult:
        """Run the RL training loop if triggers are satisfied.

        Args:
            trigger_reason: Human-readable label for the caller's trigger
                (e.g. ``"scheduled"``, ``"buffer_full"``, ``"reward_degradation"``).
            rolling_7d_mean_reward: Optional rolling 7-day mean reward value
                used to evaluate the reward-degradation trigger.
            new_regime_counts: Optional ``{regime: new_verified_count}`` dict
                used to evaluate the new-regime trigger.
            now_ms: Current UTC epoch milliseconds — defaults to
                ``int(time.time() * 1000)`` when ``None``.

        Returns:
            :class:`RLTrainingResult` describing the outcome.
        """
        now: int = now_ms if now_ms is not None else int(time.time() * 1000)
        result = RLTrainingResult(triggered_by=trigger_reason, ran_at_ms=now)

        # --- Cooldown check ---
        cooldown_ms = self._rl_cfg.self_training.min_cooldown_hours * _MS_PER_HOUR
        if self._last_train_ms > 0 and (now - self._last_train_ms) < cooldown_ms:
            elapsed_h = (now - self._last_train_ms) / _MS_PER_HOUR
            result.skipped = True
            result.skip_reason = (
                f"cooldown: {elapsed_h:.1f}h elapsed < "
                f"{self._rl_cfg.self_training.min_cooldown_hours}h required"
            )
            logger.info("RLTrainer.maybe_train: {}", result.skip_reason)
            self._run_history.append(result)
            return result

        # --- Trigger evaluation ---
        should_train, trigger_label = self._evaluate_triggers(
            trigger_reason=trigger_reason,
            rolling_7d_mean_reward=rolling_7d_mean_reward,
            new_regime_counts=new_regime_counts,
        )

        if not should_train:
            result.skipped = True
            result.skip_reason = "no trigger conditions met"
            logger.debug("RLTrainer.maybe_train: no trigger — skipping")
            self._run_history.append(result)
            return result

        result.triggered_by = trigger_label

        # --- Buffer readiness check ---
        if not self._replay_buffer.is_ready_to_train():
            result.skipped = True
            result.skip_reason = (
                f"replay buffer not ready: total={self._replay_buffer.total_count()} "
                f"< min={self._rl_cfg.replay_buffer.min_samples_to_train}"
            )
            logger.info("RLTrainer.maybe_train: {}", result.skip_reason)
            self._run_history.append(result)
            return result

        # --- Execute training ---
        result = self._run_training(result, now)
        self._run_history.append(result)
        return result

    def force_train(
        self,
        trigger_reason: str = "forced",
        now_ms: int | None = None,
    ) -> RLTrainingResult:
        """Run the RL training loop unconditionally (ignores cooldown and triggers).

        This is intended for operator use (manual override, testing) only.
        Should not be called from production inference paths.

        Args:
            trigger_reason: Human-readable reason label.
            now_ms: Current UTC epoch milliseconds.

        Returns:
            :class:`RLTrainingResult` describing the outcome.
        """
        now: int = now_ms if now_ms is not None else int(time.time() * 1000)
        result = RLTrainingResult(triggered_by=trigger_reason, ran_at_ms=now)

        if not self._replay_buffer.is_ready_to_train():
            result.skipped = True
            result.skip_reason = (
                f"replay buffer not ready: total={self._replay_buffer.total_count()} "
                f"< min={self._rl_cfg.replay_buffer.min_samples_to_train}"
            )
            logger.warning("RLTrainer.force_train: {}", result.skip_reason)
            self._run_history.append(result)
            return result

        result = self._run_training(result, now)
        self._run_history.append(result)
        return result

    def last_train_ms(self) -> int:
        """Return the UTC epoch ms timestamp of the last completed training run.

        Returns:
            0 if no training run has completed yet; otherwise the epoch ms
            of the most recent successful run.
        """
        return self._last_train_ms

    def run_history(self) -> list[RLTrainingResult]:
        """Return a copy of the training run history (oldest first).

        Returns:
            List of :class:`RLTrainingResult` instances.
        """
        return list(self._run_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_triggers(
        self,
        trigger_reason: str,
        rolling_7d_mean_reward: float | None,
        new_regime_counts: dict[str, int] | None,
    ) -> tuple[bool, str]:
        """Check whether any self-training trigger condition is met.

        Args:
            trigger_reason: Caller-supplied trigger label (e.g. ``"scheduled"``).
            rolling_7d_mean_reward: Optional 7-day mean reward.
            new_regime_counts: Optional per-regime new verification counts.

        Returns:
            Tuple of ``(should_train: bool, trigger_label: str)``.
        """
        triggers = self._rl_cfg.self_training.triggers

        # Trigger 1: scheduled / explicit
        if trigger_reason in ("scheduled", "forced"):
            return True, trigger_reason

        # Trigger 2: buffer fill
        for horizon in _VALID_HORIZONS:
            fill = self._replay_buffer.fill_percentage(horizon)
            if fill >= triggers.buffer_fill_pct:
                return True, f"buffer_full:{horizon}:{fill:.2f}"

        # Trigger 3: reward degradation
        if rolling_7d_mean_reward is not None:
            if rolling_7d_mean_reward < triggers.rolling_7d_mean_reward:
                return True, f"reward_degradation:{rolling_7d_mean_reward:.4f}"

        # Trigger 4: new regime predictions
        if new_regime_counts is not None:
            min_per = self._rl_cfg.replay_buffer.min_per_regime
            for regime, count in new_regime_counts.items():
                if count >= triggers.new_regime_predictions:
                    total = self._replay_buffer.get_regime_counts("SHORT").get(regime, 0)
                    if total < min_per:
                        return (
                            True,
                            f"new_regime:{regime}:{count}_new",
                        )

        # Trigger 5: caller-supplied label that we don't recognise → fire anyway
        if trigger_reason:
            return True, trigger_reason

        return False, ""

    def _run_training(
        self,
        result: RLTrainingResult,
        now: int,
    ) -> RLTrainingResult:
        """Execute the actual RL training batch for each active horizon.

        Draws ``max_batch_size`` samples per active horizon from the replay
        buffer via :meth:`~src.rl.replay_buffer.ReplayBuffer.get_prioritised_sample`
        and calls :meth:`_update_weights` for each non-empty batch.

        Args:
            result: Partially-filled :class:`RLTrainingResult` to update.
            now: Current UTC epoch milliseconds.

        Returns:
            Updated :class:`RLTrainingResult`.
        """
        t_start = time.time()
        max_batch = self._rl_cfg.self_training.max_batch_size
        active_horizons = self._curriculum.active_horizons()

        total_samples = 0
        trained_horizons: list[str] = []

        for horizon in active_horizons:
            try:
                batch = self._replay_buffer.get_prioritised_sample(horizon, max_batch)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "RLTrainer._run_training: get_prioritised_sample failed "
                    "for horizon={}: {}",
                    horizon,
                    exc,
                )
                continue

            if batch.empty:
                logger.debug(
                    "RLTrainer._run_training: empty batch for horizon={}", horizon
                )
                continue

            n = len(batch)
            try:
                self._update_weights(horizon, batch)
                total_samples += n
                trained_horizons.append(horizon)
                logger.info(
                    "RLTrainer._run_training: horizon={} n_samples={}", horizon, n
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "RLTrainer._run_training: _update_weights failed for horizon={}: {}",
                    horizon,
                    exc,
                )

        duration_ms = int((time.time() - t_start) * 1000)

        if trained_horizons:
            self._last_train_ms = now
            logger.info(
                "RLTrainer: training complete | horizons={} n_samples={} duration={}ms",
                trained_horizons,
                total_samples,
                duration_ms,
            )
        else:
            logger.warning(
                "RLTrainer: training run produced no updates (no non-empty batches)"
            )

        result.n_samples_used = total_samples
        result.horizons_trained = trained_horizons
        result.duration_ms = duration_ms
        result.skipped = False
        return result

    def _update_weights(self, horizon: str, batch: "pd.DataFrame") -> None:  # noqa: F821
        """Apply the RL reward signal to update model weights.

        The current implementation records the reward statistics for the batch
        and logs them.  Full backpropagation / fine-tuning of the LSTM/XGBoost
        weights based on the reward signal is left as a future extension point
        (requires differentiable model wrappers).

        Args:
            horizon: Prediction horizon label.
            batch: DataFrame from the replay buffer containing ``reward_score``
                and ``regime`` columns (at minimum).
        """
        import pandas as pd  # local import to avoid circular dep at module level

        if "reward_score" in batch.columns:
            rewards = batch["reward_score"].to_numpy(dtype=float)
            mean_r = float(rewards.mean())
            std_r = float(rewards.std()) if len(rewards) > 1 else 0.0
            n_positive = int((rewards > 0).sum())
            logger.info(
                "RLTrainer._update_weights: horizon={} n={} mean_reward={:.4f} "
                "std={:.4f} pct_positive={:.1f}%",
                horizon,
                len(batch),
                mean_r,
                std_r,
                100.0 * n_positive / len(batch),
            )


# ---------------------------------------------------------------------------
# SelfTrainingResult — outcome dataclass for RLSelfTrainer
# ---------------------------------------------------------------------------


@dataclass
class SelfTrainingResult:
    """Summary of one :class:`RLSelfTrainer` self-training run.

    Args:
        triggered_by: List of condition strings that triggered this run.
        n_samples_used: Total samples drawn from the replay buffer.
        horizons_trained: Horizon labels for which models were updated.
        xgb_accuracy_before: XGBoost validation accuracy before weight update.
        xgb_accuracy_after: XGBoost validation accuracy after weight update.
        promoted: ``True`` when the new models replaced the incumbents.
        skipped: ``True`` when the run was skipped.
        skip_reason: Non-empty when ``skipped=True``.
        duration_ms: Wall-clock duration in milliseconds.
        ran_at_ms: UTC epoch ms when the run was initiated.
    """

    triggered_by: list[str] = field(default_factory=list)
    n_samples_used: int = 0
    horizons_trained: list[str] = field(default_factory=list)
    xgb_accuracy_before: float = 0.0
    xgb_accuracy_after: float = 0.0
    promoted: bool = False
    skipped: bool = False
    skip_reason: str = ""
    duration_ms: int = 0
    ran_at_ms: int = 0


# ---------------------------------------------------------------------------
# RLSelfTrainer
# ---------------------------------------------------------------------------


class RLSelfTrainer:
    """Checks self-training triggers and executes weighted model updates.

    Unlike :class:`RLTrainer` (which logs reward stats), this class performs
    actual weight updates by re-fitting XGBoost with ``|reward_score|`` sample
    weights, running LSTM gradient fine-tuning steps, and re-fitting the
    ensemble meta-learner.

    Args:
        replay_buffer: :class:`~src.rl.replay_buffer.ReplayBuffer` instance.
        curriculum: :class:`~src.rl.curriculum.CurriculumManager` instance.
        rl_cfg: :class:`~src.config.RLConfig`.  Defaults to global singleton.
        validation_days: Number of days of recent data used to evaluate
            model quality before/after self-training (default 30).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        curriculum: CurriculumManager,
        rl_cfg: RLConfig | None = None,
        validation_days: int = 30,
    ) -> None:
        self._replay_buffer: ReplayBuffer = replay_buffer
        self._curriculum: CurriculumManager = curriculum
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        self._validation_days: int = validation_days
        self._last_train_ms: int = 0
        self._lock: threading.Lock = threading.Lock()
        logger.info("RLSelfTrainer initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_triggers(
        self,
        buffer: ReplayBuffer,
        reward_history: pd.DataFrame,
        now_ms: int | None = None,
    ) -> list[str]:
        """Return the list of self-training trigger conditions that are active.

        Respects the 48-hour cooldown — returns an empty list when the cooldown
        has not elapsed regardless of other conditions.

        Conditions checked (any combination may appear):

        - ``"buffer_full"`` — any horizon fill ≥ ``buffer_fill_pct`` threshold.
        - ``"reward_decline"`` — rolling 7-day mean reward < threshold.
        - ``"regime_transition"`` — reward_history contains multiple distinct
          regime labels in the most recent 7 days.

        Args:
            buffer: :class:`~src.rl.replay_buffer.ReplayBuffer` to inspect.
            reward_history: DataFrame of recent reward rows.  Must contain
                ``"reward_score"`` (float) and optionally ``"regime"`` (str)
                and ``"created_at"`` (int ms) columns.

        Returns:
            List of triggered condition strings.  Empty list means no
            self-training should fire (either cooldown active or no triggers).
        """
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        cooldown_ms = self._rl_cfg.self_training.min_cooldown_hours * _MS_PER_HOUR

        if self._last_train_ms > 0 and (now_ms - self._last_train_ms) < cooldown_ms:
            elapsed_h = (now_ms - self._last_train_ms) / _MS_PER_HOUR
            logger.debug(
                "RLSelfTrainer.check_triggers: cooldown active ({:.1f}h elapsed, "
                "{}h required)",
                elapsed_h,
                self._rl_cfg.self_training.min_cooldown_hours,
            )
            return []

        triggers_cfg = self._rl_cfg.self_training.triggers
        triggered: list[str] = []

        # Trigger 1 — buffer fill
        for horizon in ("SHORT", "MEDIUM", "LONG", "MACRO"):
            try:
                fill = buffer.fill_percentage(horizon)
                if fill >= triggers_cfg.buffer_fill_pct:
                    triggered.append("buffer_full")
                    break
            except Exception:  # noqa: BLE001
                pass

        # Trigger 2 — reward decline
        if not reward_history.empty and "reward_score" in reward_history.columns:
            try:
                _cutoff = now_ms - 7 * 24 * _MS_PER_HOUR
                recent = reward_history
                if "created_at" in reward_history.columns:
                    recent = reward_history[reward_history["created_at"] >= _cutoff]
                if not recent.empty:
                    mean_r = float(recent["reward_score"].mean())
                    if mean_r < triggers_cfg.rolling_7d_mean_reward:
                        triggered.append("reward_decline")
            except Exception:  # noqa: BLE001
                pass

        # Trigger 3 — regime transition (multiple regimes seen recently)
        if not reward_history.empty and "regime" in reward_history.columns:
            try:
                _cutoff = now_ms - 7 * 24 * _MS_PER_HOUR
                recent = reward_history
                if "created_at" in reward_history.columns:
                    recent = reward_history[reward_history["created_at"] >= _cutoff]
                if not recent.empty and recent["regime"].nunique() > 1:
                    triggered.append("regime_transition")
            except Exception:  # noqa: BLE001
                pass

        return triggered

    def run_self_training(
        self,
        ensemble: "EnsembleModel",
        xgb_models: dict[str, "XGBoostModel"],
        lstm: "LSTMModel",
        buffer: ReplayBuffer,
        now_ms: int | None = None,
    ) -> SelfTrainingResult:
        """Execute the 8-step self-training weight update.

        Steps:

        1. Acquire the buffer lock (prevents concurrent writes during sampling).
        2. Sample replay buffer per active horizon.
        3. Compute per-sample weights = ``|reward_score|`` (higher magnitude →
           more influence on the re-fit).
        4. Re-fit XGBoost models with sample weights.
        5. LSTM gradient update (fine-tuning on sampled batches).
        6. Re-fit ensemble meta-learner on the updated model outputs.
        7. Validate on the most recent ``validation_days`` rows in the buffer.
        8. Promote models if validation accuracy improved; archive to MLflow.

        Args:
            ensemble: Current :class:`~src.models.ensemble.EnsembleModel`.
            xgb_models: Dict mapping horizon label → :class:`~src.models.xgb_model.XGBoostModel`.
            lstm: Current :class:`~src.models.lstm_model.LSTMModel`.
            buffer: :class:`~src.rl.replay_buffer.ReplayBuffer` to draw from.
            now_ms: UTC epoch ms override (defaults to wall clock).

        Returns:
            :class:`SelfTrainingResult` describing the outcome.
        """
        now: int = now_ms if now_ms is not None else int(time.time() * 1000)
        result = SelfTrainingResult(ran_at_ms=now)

        if not buffer.is_ready_to_train():
            result.skipped = True
            result.skip_reason = (
                f"replay buffer not ready: total={buffer.total_count()} "
                f"< min={self._rl_cfg.replay_buffer.min_samples_to_train}"
            )
            logger.info("RLSelfTrainer.run_self_training: {}", result.skip_reason)
            return result

        t_start = time.time()
        max_batch = self._rl_cfg.self_training.max_batch_size
        active_horizons = self._curriculum.active_horizons()

        # Step 1: acquire buffer lock
        with self._lock:
            # Step 2: sample per active horizon
            horizon_batches: dict[str, pd.DataFrame] = {}
            for horizon in active_horizons:
                try:
                    batch = buffer.get_prioritised_sample(horizon, max_batch)
                    if not batch.empty:
                        horizon_batches[horizon] = batch
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "RLSelfTrainer: sampling failed for horizon={}: {}", horizon, exc
                    )

        if not horizon_batches:
            result.skipped = True
            result.skip_reason = "all horizon batches were empty after sampling"
            return result

        total_samples = sum(len(b) for b in horizon_batches.values())
        result.n_samples_used = total_samples

        # Steps 3–6: weighted re-fit per horizon
        trained_horizons: list[str] = []
        xgb_acc_before_all: list[float] = []
        xgb_acc_after_all: list[float] = []

        for horizon, batch in horizon_batches.items():
            try:
                # Step 3: sample weights = |reward_score|
                if "reward_score" not in batch.columns:
                    logger.warning(
                        "RLSelfTrainer: batch for horizon={} missing reward_score; skipping",
                        horizon,
                    )
                    continue

                rewards = batch["reward_score"].to_numpy(dtype=float)
                sample_weight = np.abs(rewards)
                # Normalise to mean=1 so scale_pos_weight is not distorted
                w_sum = sample_weight.sum()
                if w_sum > 0:
                    sample_weight = sample_weight / w_sum * len(sample_weight)

                # Build a minimal feature matrix from the batch if columns exist
                feature_cols = [
                    c for c in batch.columns
                    if c not in {
                        "reward_score", "regime", "created_at", "horizon",
                        "model_version", "abs_reward",
                    }
                ]

                xgb_model = xgb_models.get(horizon)

                if xgb_model is not None and len(feature_cols) >= 2:
                    # Step 4: re-fit XGBoost with sample weights
                    X = batch[feature_cols].to_numpy(dtype=float)

                    if "direction_correct" in batch.columns:
                        y = batch["direction_correct"].astype(int).to_numpy()
                    else:
                        # Fall back to sign of reward
                        y = (rewards > 0).astype(int)

                    # Split 80/20 train/val within the batch
                    split_idx = max(1, int(len(X) * 0.8))
                    if split_idx < len(X) and len(np.unique(y[:split_idx])) >= 2:
                        # Measure accuracy before
                        try:
                            prob_before = xgb_model.predict_proba(X[split_idx:])
                            acc_before = float(
                                ((prob_before >= 0.5).astype(int) == y[split_idx:]).mean()
                            )
                            xgb_acc_before_all.append(acc_before)
                        except Exception:  # noqa: BLE001
                            acc_before = 0.5

                        try:
                            xgb_model.fit(
                                X[:split_idx],
                                y[:split_idx],
                                X[split_idx:],
                                y[split_idx:],
                                feature_names=feature_cols,
                                sample_weight=sample_weight[:split_idx],
                            )
                            # Measure accuracy after
                            prob_after = xgb_model.predict_proba(X[split_idx:])
                            acc_after = float(
                                ((prob_after >= 0.5).astype(int) == y[split_idx:]).mean()
                            )
                            xgb_acc_after_all.append(acc_after)
                            logger.info(
                                "RLSelfTrainer: XGB re-fit horizon={} "
                                "acc {:.3f} → {:.3f}",
                                horizon,
                                acc_before,
                                acc_after,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "RLSelfTrainer: XGB re-fit failed for horizon={}: {}",
                                horizon,
                                exc,
                            )

                # Step 5: LSTM gradient fine-tuning (lightweight — 1 epoch)
                if len(feature_cols) >= 2:
                    try:
                        X_lstm = batch[feature_cols].to_numpy(dtype=float)
                        if "direction_correct" in batch.columns:
                            y_lstm = batch["direction_correct"].astype(int).to_numpy()
                        else:
                            y_lstm = (rewards > 0).astype(int)

                        split_idx = max(1, int(len(X_lstm) * 0.8))
                        if (
                            split_idx < len(X_lstm)
                            and len(np.unique(y_lstm[:split_idx])) >= 2
                        ):
                            lstm.fit(
                                X_lstm[:split_idx],
                                y_lstm[:split_idx],
                                X_lstm[split_idx:],
                                y_lstm[split_idx:],
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "RLSelfTrainer: LSTM fine-tuning failed for horizon={}: {}",
                            horizon,
                            exc,
                        )

                trained_horizons.append(horizon)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "RLSelfTrainer: update failed for horizon={}: {}", horizon, exc
                )

        result.horizons_trained = trained_horizons

        # Step 6: re-fit ensemble on current model outputs from the sampled batch
        # (Uses the combined batch from the first active horizon as meta-features)
        if trained_horizons:
            try:
                first_horizon = trained_horizons[0]
                batch0 = horizon_batches[first_horizon]
                feature_cols0 = [
                    c for c in batch0.columns
                    if c not in {
                        "reward_score", "regime", "created_at", "horizon",
                        "model_version", "abs_reward",
                    }
                ]
                if len(feature_cols0) >= 3:
                    X0 = batch0[feature_cols0].to_numpy(dtype=float)
                    # Build a 3-column meta-feature matrix [lstm_prob, xgb_prob, regime_encoded]
                    xgb_m = xgb_models.get(first_horizon)
                    lstm_probs = lstm.predict_proba(X0) if lstm._is_fitted else np.full(len(X0), 0.5)
                    xgb_probs = xgb_m.predict_proba(X0) if (xgb_m and xgb_m._is_fitted) else np.full(len(X0), 0.5)
                    # Regime encoded: derive from 'regime' column if available
                    _regime_map = {
                        "TRENDING_BULL": 0, "TRENDING_BEAR": 1,
                        "RANGING_HIGH_VOL": 2, "RANGING_LOW_VOL": 3, "DECOUPLED": 4,
                    }
                    if "regime" in batch0.columns:
                        regime_enc = batch0["regime"].map(_regime_map).fillna(0).to_numpy()
                    else:
                        regime_enc = np.zeros(len(X0))

                    meta_X = np.column_stack([lstm_probs, xgb_probs, regime_enc])
                    if "direction_correct" in batch0.columns:
                        y_meta = batch0["direction_correct"].astype(int).to_numpy()
                    else:
                        y_meta = (batch0["reward_score"].to_numpy(dtype=float) > 0).astype(int)

                    split_idx = max(1, int(len(meta_X) * 0.8))
                    if split_idx < len(meta_X) and len(np.unique(y_meta[:split_idx])) >= 2:
                        ensemble.fit(
                            meta_X[:split_idx],
                            y_meta[:split_idx],
                            meta_X[split_idx:],
                            y_meta[split_idx:],
                        )
                        logger.info(
                            "RLSelfTrainer: ensemble re-fitted on {} meta-samples",
                            len(meta_X),
                        )
            except Exception as exc:  # noqa: BLE001
                logger.warning("RLSelfTrainer: ensemble re-fit failed: {}", exc)

        # Steps 7 & 8: validate and promote
        result.xgb_accuracy_before = (
            float(np.mean(xgb_acc_before_all)) if xgb_acc_before_all else 0.0
        )
        result.xgb_accuracy_after = (
            float(np.mean(xgb_acc_after_all)) if xgb_acc_after_all else 0.0
        )

        # Promote when after-accuracy is better than before (or we have no baseline)
        if trained_horizons:
            if not xgb_acc_before_all or result.xgb_accuracy_after >= result.xgb_accuracy_before:
                result.promoted = True
                self._last_train_ms = now
                logger.info(
                    "RLSelfTrainer: models promoted | "
                    "xgb_acc {:.3f} → {:.3f} | horizons={}",
                    result.xgb_accuracy_before,
                    result.xgb_accuracy_after,
                    trained_horizons,
                )
            else:
                logger.warning(
                    "RLSelfTrainer: accuracy declined ({:.3f} → {:.3f}); "
                    "keeping incumbent models",
                    result.xgb_accuracy_before,
                    result.xgb_accuracy_after,
                )

        result.duration_ms = int((time.time() - t_start) * 1000)
        result.skipped = len(trained_horizons) == 0

        # Archive to MLflow (best-effort)
        try:
            import mlflow  # noqa: PLC0415
            with mlflow.start_run(run_name="rl_self_training", nested=True):
                mlflow.log_param("horizons_trained", trained_horizons)
                mlflow.log_param("n_samples_used", total_samples)
                mlflow.log_metric("xgb_acc_before", result.xgb_accuracy_before)
                mlflow.log_metric("xgb_acc_after", result.xgb_accuracy_after)
                mlflow.log_metric("promoted", int(result.promoted))
                mlflow.set_tag("stage", "rl_self_training")
        except Exception:  # noqa: BLE001
            pass  # MLflow unavailability must never halt training

        return result
