"""Unit tests for src/rl/rl_trainer.py — RLTrainer.

MANDATORY tests:
  1. maybe_train() skips when cooldown period has not elapsed.
  2. maybe_train() skips when replay buffer is not ready to train.
  3. maybe_train() fires when trigger conditions are met and buffer is ready.
  4. force_train() bypasses cooldown and triggers (still checks buffer).
  5. RLTrainingResult.skipped is False on a completed training run.
  6. run_history() returns a copy — mutating it does not affect internal state.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import RLConfig, _load_yaml
from src.rl.curriculum import CurriculumManager
from src.rl.rl_trainer import RLSelfTrainer, RLTrainer, RLTrainingResult, SelfTrainingResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RL_CFG: RLConfig = RLConfig(**_load_yaml("rl_config.yaml"))
_NOW_MS: int = 1_700_000_000_000
_COOLDOWN_MS: int = 48 * 3_600_000  # 48 hours in ms


def _make_curriculum(starting_stage: int = 1) -> CurriculumManager:
    return CurriculumManager(rl_cfg=_RL_CFG, starting_stage=starting_stage)


def _make_replay_buffer(
    *,
    ready: bool = True,
    fill_pct: float = 0.0,
    total_count: int = 600,
    sample: pd.DataFrame | None = None,
    regime_counts: dict | None = None,
) -> MagicMock:
    buf = MagicMock()
    buf.is_ready_to_train.return_value = ready
    buf.fill_percentage.return_value = fill_pct
    buf.total_count.return_value = total_count
    buf.get_prioritised_sample.return_value = (
        sample if sample is not None else pd.DataFrame(
            {"reward_score": [0.5, -0.2, 0.3], "regime": ["TRENDING_BULL"] * 3}
        )
    )
    buf.get_regime_counts.return_value = regime_counts or {}
    return buf


def _make_storage() -> MagicMock:
    return MagicMock()


def _make_trainer(
    *,
    ready: bool = True,
    fill_pct: float = 0.0,
    last_train_ms: int = 0,
    starting_stage: int = 1,
    sample: pd.DataFrame | None = None,
) -> RLTrainer:
    buf = _make_replay_buffer(ready=ready, fill_pct=fill_pct, sample=sample)
    curriculum = _make_curriculum(starting_stage)
    trainer = RLTrainer(
        storage=_make_storage(),
        replay_buffer=buf,
        curriculum=curriculum,
        rl_cfg=_RL_CFG,
    )
    trainer._last_train_ms = last_train_ms
    return trainer


# ---------------------------------------------------------------------------
# MANDATORY TEST 1 — Cooldown enforcement
# ---------------------------------------------------------------------------


class TestCooldown:
    """MANDATORY: maybe_train() must skip if cooldown has not elapsed."""

    def test_skips_within_cooldown(self) -> None:
        """MANDATORY: Run within 48h cooldown must skip."""
        trainer = _make_trainer(last_train_ms=_NOW_MS - (_COOLDOWN_MS - 1))
        result = trainer.maybe_train(now_ms=_NOW_MS)
        assert result.skipped is True
        assert "cooldown" in result.skip_reason

    def test_fires_after_cooldown_elapsed(self) -> None:
        """Run after full cooldown must not skip on cooldown grounds."""
        trainer = _make_trainer(last_train_ms=_NOW_MS - _COOLDOWN_MS - 1)
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        # Cooldown check passes — result depends on buffer
        assert "cooldown" not in result.skip_reason

    def test_no_previous_run_skips_cooldown_check(self) -> None:
        """last_train_ms == 0 means no prior run → cooldown check skipped."""
        trainer = _make_trainer(last_train_ms=0)
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert "cooldown" not in result.skip_reason

    def test_cooldown_reason_contains_elapsed_hours(self) -> None:
        """skip_reason message should mention elapsed time."""
        trainer = _make_trainer(last_train_ms=_NOW_MS - 10 * 3_600_000)  # 10h ago
        result = trainer.maybe_train(now_ms=_NOW_MS)
        assert result.skipped is True
        assert "10.0h" in result.skip_reason


# ---------------------------------------------------------------------------
# MANDATORY TEST 2 — Buffer not ready
# ---------------------------------------------------------------------------


class TestBufferNotReady:
    """MANDATORY: maybe_train() skips when buffer has fewer than min_samples."""

    def test_skips_when_buffer_not_ready(self) -> None:
        """MANDATORY: is_ready_to_train()=False → skip with message."""
        trainer = _make_trainer(ready=False)
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.skipped is True
        assert "not ready" in result.skip_reason

    def test_proceeds_when_buffer_is_ready(self) -> None:
        """is_ready_to_train()=True → training run executes (result.skipped=False)."""
        trainer = _make_trainer(ready=True)
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.skipped is False


# ---------------------------------------------------------------------------
# MANDATORY TEST 3 — Trigger conditions
# ---------------------------------------------------------------------------


class TestTriggerConditions:
    """MANDATORY: maybe_train() fires when a trigger condition is met."""

    def test_scheduled_trigger_fires(self) -> None:
        """MANDATORY: trigger_reason='scheduled' always fires (no trigger check)."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.skipped is False
        assert result.triggered_by == "scheduled"

    def test_buffer_fill_trigger_fires(self) -> None:
        """Buffer fill >= threshold fires training."""
        buf = _make_replay_buffer(ready=True, fill_pct=0.85)
        curriculum = _make_curriculum()
        trainer = RLTrainer(
            storage=_make_storage(),
            replay_buffer=buf,
            curriculum=curriculum,
            rl_cfg=_RL_CFG,
        )
        # Use non-scheduled trigger name so buffer_fill logic is reached
        result = trainer.maybe_train(trigger_reason="check", now_ms=_NOW_MS)
        assert result.skipped is False
        assert "buffer_full" in result.triggered_by or result.triggered_by == "check"

    def test_reward_degradation_trigger(self) -> None:
        """rolling_7d_mean_reward < 0.0 triggers training."""
        trainer = _make_trainer()
        result = trainer.maybe_train(
            trigger_reason="check",
            rolling_7d_mean_reward=-0.05,
            now_ms=_NOW_MS,
        )
        assert result.skipped is False

    def test_no_trigger_skips(self) -> None:
        """No known trigger → skips with 'no trigger' reason."""
        buf = _make_replay_buffer(ready=True, fill_pct=0.0)  # below threshold
        curriculum = _make_curriculum()
        trainer = RLTrainer(
            storage=_make_storage(),
            replay_buffer=buf,
            curriculum=curriculum,
            rl_cfg=_RL_CFG,
        )
        # Empty trigger_reason with no other conditions met
        result = trainer.maybe_train(trigger_reason="", now_ms=_NOW_MS)
        assert result.skipped is True
        assert "no trigger" in result.skip_reason


# ---------------------------------------------------------------------------
# MANDATORY TEST 4 — force_train() bypasses cooldown
# ---------------------------------------------------------------------------


class TestForceTrain:
    """MANDATORY: force_train() ignores cooldown and trigger checks."""

    def test_force_train_ignores_cooldown(self) -> None:
        """MANDATORY: force_train runs even within cooldown window."""
        trainer = _make_trainer(last_train_ms=_NOW_MS - 1)  # 1ms ago — in cooldown
        result = trainer.force_train(now_ms=_NOW_MS)
        assert result.skipped is False

    def test_force_train_skips_if_buffer_not_ready(self) -> None:
        """force_train still checks buffer readiness."""
        trainer = _make_trainer(ready=False)
        result = trainer.force_train(now_ms=_NOW_MS)
        assert result.skipped is True
        assert "not ready" in result.skip_reason

    def test_force_train_updates_last_train_ms(self) -> None:
        """force_train updates last_train_ms on success."""
        trainer = _make_trainer()
        assert trainer.last_train_ms() == 0
        trainer.force_train(now_ms=_NOW_MS)
        assert trainer.last_train_ms() == _NOW_MS


# ---------------------------------------------------------------------------
# MANDATORY TEST 5 — Completed training result fields
# ---------------------------------------------------------------------------


class TestTrainingResultFields:
    """MANDATORY: RLTrainingResult fields are correctly populated after a run."""

    def test_skipped_false_after_successful_run(self) -> None:
        """MANDATORY: result.skipped is False after a successful run."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.skipped is False

    def test_n_samples_used_is_positive(self) -> None:
        """n_samples_used > 0 when training actually ran."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.n_samples_used > 0

    def test_horizons_trained_contains_short(self) -> None:
        """Stage 1 → horizons_trained should contain 'SHORT'."""
        trainer = _make_trainer(starting_stage=1)
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert "SHORT" in result.horizons_trained

    def test_duration_ms_is_non_negative(self) -> None:
        """duration_ms must be a non-negative integer."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.duration_ms >= 0

    def test_ran_at_ms_matches_now(self) -> None:
        """ran_at_ms must equal the now_ms passed to maybe_train."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert result.ran_at_ms == _NOW_MS

    def test_triggered_by_set_correctly(self) -> None:
        """triggered_by must contain the trigger reason."""
        trainer = _make_trainer()
        result = trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert "scheduled" in result.triggered_by

    def test_last_train_ms_updated_after_run(self) -> None:
        """last_train_ms() is updated to now_ms after a completed run."""
        trainer = _make_trainer()
        trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        assert trainer.last_train_ms() == _NOW_MS


# ---------------------------------------------------------------------------
# MANDATORY TEST 6 — run_history() immutability
# ---------------------------------------------------------------------------


class TestRunHistoryImmutability:
    """MANDATORY: Mutating the returned run_history() list does not affect state."""

    def test_returned_list_is_copy(self) -> None:
        """MANDATORY: Appending to the returned list must not mutate internal state."""
        trainer = _make_trainer()
        trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        history = trainer.run_history()
        history.append(RLTrainingResult(triggered_by="injected"))
        assert len(trainer.run_history()) == 1  # internal list unchanged

    def test_run_history_grows_per_call(self) -> None:
        """Each call appends exactly one entry to run_history."""
        trainer = _make_trainer()
        trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        # Second call: use a different now_ms far enough from first to avoid cooldown
        trainer._last_train_ms = 0  # reset for test simplicity
        trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS + 1)
        assert len(trainer.run_history()) == 2

    def test_skipped_calls_also_recorded_in_history(self) -> None:
        """Skipped runs are also appended to history."""
        trainer = _make_trainer(ready=False)
        trainer.maybe_train(trigger_reason="scheduled", now_ms=_NOW_MS)
        history = trainer.run_history()
        assert len(history) == 1
        assert history[0].skipped is True


# ---------------------------------------------------------------------------
# Additional: RLTrainingResult defaults
# ---------------------------------------------------------------------------


class TestRLTrainingResultDefaults:
    def test_default_result_is_not_skipped(self) -> None:
        r = RLTrainingResult()
        assert r.skipped is False

    def test_default_result_has_empty_horizons(self) -> None:
        r = RLTrainingResult()
        assert r.horizons_trained == []

    def test_default_result_skip_reason_is_empty(self) -> None:
        r = RLTrainingResult()
        assert r.skip_reason == ""


# ---------------------------------------------------------------------------
# SelfTrainingResult defaults
# ---------------------------------------------------------------------------


class TestSelfTrainingResultDefaults:
    def test_default_not_skipped(self) -> None:
        r = SelfTrainingResult()
        assert r.skipped is False

    def test_default_not_promoted(self) -> None:
        r = SelfTrainingResult()
        assert r.promoted is False

    def test_default_empty_horizons(self) -> None:
        r = SelfTrainingResult()
        assert r.horizons_trained == []

    def test_default_empty_triggered_by(self) -> None:
        r = SelfTrainingResult()
        assert r.triggered_by == []


# ---------------------------------------------------------------------------
# RLSelfTrainer helpers
# ---------------------------------------------------------------------------


def _make_self_trainer(
    *,
    ready: bool = True,
    fill_pct: float = 0.0,
    starting_stage: int = 1,
) -> RLSelfTrainer:
    buf = MagicMock()
    buf.is_ready_to_train.return_value = ready
    buf.fill_percentage.return_value = fill_pct
    buf.total_count.return_value = 600
    buf.get_prioritised_sample.return_value = pd.DataFrame(
        {"reward_score": [0.5, -0.2, 0.3], "regime": ["TRENDING_BULL"] * 3}
    )
    buf.get_regime_counts.return_value = {}
    curriculum = CurriculumManager(rl_cfg=_RL_CFG, starting_stage=starting_stage)
    return RLSelfTrainer(
        replay_buffer=buf,
        curriculum=curriculum,
        rl_cfg=_RL_CFG,
    )


# ---------------------------------------------------------------------------
# RLSelfTrainer — check_triggers()
# ---------------------------------------------------------------------------


class TestRLSelfTrainerCheckTriggers:
    def test_cooldown_returns_empty_list(self) -> None:
        """When cooldown is active, check_triggers returns an empty list."""
        trainer = _make_self_trainer()
        # Set last_train_ms to 1ms ago (within 48h cooldown)
        trainer._last_train_ms = _NOW_MS - 1
        buf = MagicMock()
        buf.fill_percentage.return_value = 0.0
        reward_history = pd.DataFrame({"reward_score": [0.5], "created_at": [_NOW_MS]})
        result = trainer.check_triggers(buf, reward_history, now_ms=_NOW_MS)
        assert result == []

    def test_buffer_full_trigger(self) -> None:
        """Buffer fill >= threshold → 'buffer_full' in triggers."""
        trainer = _make_self_trainer()
        buf = MagicMock()
        buf.fill_percentage.return_value = 0.90  # above 0.80 threshold
        result = trainer.check_triggers(buf, pd.DataFrame(), now_ms=_NOW_MS)
        assert "buffer_full" in result

    def test_reward_decline_trigger(self) -> None:
        """Rolling 7d mean reward < 0.0 → 'reward_decline' in triggers."""
        trainer = _make_self_trainer()
        buf = MagicMock()
        buf.fill_percentage.return_value = 0.0
        reward_history = pd.DataFrame(
            {
                "reward_score": [-0.5, -0.3, -0.2],
                "created_at": [_NOW_MS - 1000, _NOW_MS - 2000, _NOW_MS - 3000],
            }
        )
        result = trainer.check_triggers(buf, reward_history, now_ms=_NOW_MS)
        assert "reward_decline" in result

    def test_regime_transition_trigger(self) -> None:
        """Multiple regimes in last 7 days → 'regime_transition' in triggers."""
        trainer = _make_self_trainer()
        buf = MagicMock()
        buf.fill_percentage.return_value = 0.0
        reward_history = pd.DataFrame(
            {
                "reward_score": [0.5, 0.3],
                "regime": ["TRENDING_BULL", "DECOUPLED"],
                "created_at": [_NOW_MS - 1000, _NOW_MS - 2000],
            }
        )
        result = trainer.check_triggers(buf, reward_history, now_ms=_NOW_MS)
        assert "regime_transition" in result

    def test_no_triggers_returns_empty_list(self) -> None:
        """No conditions met → empty list returned."""
        trainer = _make_self_trainer()
        buf = MagicMock()
        buf.fill_percentage.return_value = 0.0
        # Single regime, positive rewards
        reward_history = pd.DataFrame(
            {
                "reward_score": [0.5],
                "regime": ["TRENDING_BULL"],
                "created_at": [_NOW_MS - 1000],
            }
        )
        result = trainer.check_triggers(buf, reward_history, now_ms=_NOW_MS)
        assert "buffer_full" not in result
        assert "reward_decline" not in result


# ---------------------------------------------------------------------------
# RLSelfTrainer — run_self_training()
# ---------------------------------------------------------------------------


class TestRLSelfTrainerRunSelfTraining:
    def test_skips_when_buffer_not_ready(self) -> None:
        """run_self_training skips when buffer is not ready."""
        trainer = _make_self_trainer(ready=False)
        buf = MagicMock()
        buf.is_ready_to_train.return_value = False
        buf.total_count.return_value = 10
        ensemble = MagicMock()
        lstm = MagicMock()
        lstm._is_fitted = False
        result = trainer.run_self_training(ensemble, {}, lstm, buf, now_ms=_NOW_MS)
        assert result.skipped is True
        assert "not ready" in result.skip_reason

    def test_result_type_is_self_training_result(self) -> None:
        """run_self_training always returns a SelfTrainingResult."""
        trainer = _make_self_trainer(ready=False)
        buf = MagicMock()
        buf.is_ready_to_train.return_value = False
        buf.total_count.return_value = 10
        result = trainer.run_self_training(MagicMock(), {}, MagicMock(), buf, now_ms=_NOW_MS)
        assert isinstance(result, SelfTrainingResult)

    def test_skips_when_all_batches_empty(self) -> None:
        """run_self_training skips when all horizon batches are empty."""
        trainer = _make_self_trainer(ready=True)
        buf = MagicMock()
        buf.is_ready_to_train.return_value = True
        buf.get_prioritised_sample.return_value = pd.DataFrame()  # empty
        lstm = MagicMock()
        lstm._is_fitted = False
        result = trainer.run_self_training(MagicMock(), {}, lstm, buf, now_ms=_NOW_MS)
        assert result.skipped is True

    def test_ran_at_ms_matches_now_ms(self) -> None:
        """ran_at_ms is set to the supplied now_ms."""
        trainer = _make_self_trainer(ready=False)
        buf = MagicMock()
        buf.is_ready_to_train.return_value = False
        buf.total_count.return_value = 0
        result = trainer.run_self_training(MagicMock(), {}, MagicMock(), buf, now_ms=_NOW_MS)
        assert result.ran_at_ms == _NOW_MS
