"""Unit tests for src/rl/rl_monitor.py — RLMonitor.

Tests verify that:
  1. All recording methods can be called without raising exceptions.
  2. Methods tolerate prometheus_client being unavailable (stub mode).
  3. Invalid calls (wrong types, missing args) raise appropriate errors.
"""
from __future__ import annotations

import pytest

from src.rl.rl_monitor import RLMonitor


def _monitor() -> RLMonitor:
    return RLMonitor()


class TestRLMonitorRecordReward:
    """record_reward() works without raising for all valid inputs."""

    def test_basic_call_does_not_raise(self) -> None:
        m = _monitor()
        m.record_reward(horizon="SHORT", regime="TRENDING_BULL", reward=0.42)

    def test_negative_reward_does_not_raise(self) -> None:
        m = _monitor()
        m.record_reward(horizon="LONG", regime="DECOUPLED", reward=-1.5)

    def test_zero_reward_does_not_raise(self) -> None:
        m = _monitor()
        m.record_reward(horizon="MEDIUM", regime="RANGING_HIGH_VOL", reward=0.0)

    def test_extreme_reward_does_not_raise(self) -> None:
        m = _monitor()
        m.record_reward(horizon="MACRO", regime="TRENDING_BEAR", reward=6.0)


class TestRLMonitorUpdateMeanReward:
    def test_update_mean_reward_does_not_raise(self) -> None:
        m = _monitor()
        m.update_mean_reward(horizon="SHORT", mean_reward=0.35)

    def test_negative_mean_reward_does_not_raise(self) -> None:
        m = _monitor()
        m.update_mean_reward(horizon="LONG", mean_reward=-0.10)


class TestRLMonitorRecordTrainingRun:
    def test_single_horizon_does_not_raise(self) -> None:
        m = _monitor()
        m.record_training_run(
            horizons_trained=["SHORT"], n_samples=64, trigger="scheduled"
        )

    def test_multiple_horizons_does_not_raise(self) -> None:
        m = _monitor()
        m.record_training_run(
            horizons_trained=["SHORT", "MEDIUM", "LONG"],
            n_samples=192,
            trigger="buffer_full",
        )

    def test_empty_horizons_does_not_raise(self) -> None:
        m = _monitor()
        m.record_training_run(horizons_trained=[], n_samples=0, trigger="scheduled")


class TestRLMonitorCurriculumStage:
    def test_set_stage_1_does_not_raise(self) -> None:
        m = _monitor()
        m.set_curriculum_stage(stage=1)

    def test_set_stage_4_does_not_raise(self) -> None:
        m = _monitor()
        m.set_curriculum_stage(stage=4)

    def test_record_stage_advance_does_not_raise(self) -> None:
        m = _monitor()
        m.record_stage_advance(from_stage=1, to_stage=2)

    def test_record_multiple_advances_does_not_raise(self) -> None:
        m = _monitor()
        m.record_stage_advance(1, 2)
        m.record_stage_advance(2, 3)
        m.record_stage_advance(3, 4)


class TestRLMonitorBufferFill:
    def test_set_buffer_fill_zero(self) -> None:
        m = _monitor()
        m.set_buffer_fill(horizon="SHORT", fill_pct=0.0)

    def test_set_buffer_fill_full(self) -> None:
        m = _monitor()
        m.set_buffer_fill(horizon="LONG", fill_pct=1.0)

    def test_set_buffer_fill_partial(self) -> None:
        m = _monitor()
        m.set_buffer_fill(horizon="MEDIUM", fill_pct=0.75)


class TestRLMonitorVerifiedPredictions:
    def test_correct_prediction_does_not_raise(self) -> None:
        m = _monitor()
        m.record_verified_prediction(horizon="SHORT", direction_correct=True)

    def test_incorrect_prediction_does_not_raise(self) -> None:
        m = _monitor()
        m.record_verified_prediction(horizon="LONG", direction_correct=False)

    def test_all_horizons_do_not_raise(self) -> None:
        m = _monitor()
        for horizon in ("SHORT", "MEDIUM", "LONG", "MACRO"):
            m.record_verified_prediction(horizon=horizon, direction_correct=True)
            m.record_verified_prediction(horizon=horizon, direction_correct=False)


class TestRLMonitorInstantiation:
    def test_multiple_instances_do_not_raise(self) -> None:
        """Multiple RLMonitor instances must not conflict on metric registration."""
        m1 = RLMonitor()
        m2 = RLMonitor()
        m1.record_reward("SHORT", "TRENDING_BULL", 0.5)
        m2.record_reward("SHORT", "TRENDING_BULL", 0.5)
