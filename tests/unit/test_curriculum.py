"""Unit tests for src/rl/curriculum.py — CurriculumManager.

MANDATORY tests:
  1. Starting stage is always 1 (from rl_config.yaml).
  2. active_horizons() at stage 1 returns exactly ["SHORT"].
  3. try_advance() succeeds when all three criteria are met.
  4. try_advance() does NOT advance when any criterion is unmet.
  5. force_set_stage() bypasses criteria and sets stage immediately.
  6. Stages never regress (is_final_stage() True at stage 4, no further advance).
  7. advancement_history() is immutable — returned list copies cannot mutate state.

Additional coverage: stage_info(), StageInfo fields, invalid force_set_stage,
criteria edge cases (boundary values), history entry fields.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config import RLConfig, _load_yaml
from src.rl.curriculum import AdvancementResult, CurriculumManager, StageInfo

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_RL_CFG: RLConfig = RLConfig(**_load_yaml("rl_config.yaml"))


def _make_manager(starting_stage: int = 1) -> CurriculumManager:
    """Build a CurriculumManager from the real rl_config.yaml."""
    return CurriculumManager(rl_cfg=_RL_CFG, starting_stage=starting_stage)


# ---------------------------------------------------------------------------
# MANDATORY TEST 1 — Starting stage is 1
# ---------------------------------------------------------------------------


class TestStartingStage:
    """MANDATORY: Default starting stage must be 1."""

    def test_default_starting_stage_is_1(self) -> None:
        """MANDATORY: CurriculumManager starts at stage 1 by default."""
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        assert mgr.current_stage() == 1

    def test_starting_stage_override(self) -> None:
        """starting_stage param overrides config default."""
        mgr = _make_manager(starting_stage=3)
        assert mgr.current_stage() == 3

    def test_invalid_starting_stage_raises(self) -> None:
        """Bogus starting stage raises ValueError."""
        with pytest.raises(ValueError, match="starting_stage"):
            CurriculumManager(rl_cfg=_RL_CFG, starting_stage=99)


# ---------------------------------------------------------------------------
# MANDATORY TEST 2 — active_horizons() at stage 1
# ---------------------------------------------------------------------------


class TestActiveHorizons:
    """MANDATORY: active_horizons() returns the correct horizon list per stage."""

    def test_stage1_active_horizons_is_short_only(self) -> None:
        """MANDATORY: Stage 1 must expose exactly ['SHORT']."""
        mgr = _make_manager(starting_stage=1)
        assert mgr.active_horizons() == ["SHORT"]

    def test_stage2_active_horizons(self) -> None:
        """Stage 2 must expose ['SHORT', 'MEDIUM']."""
        mgr = _make_manager(starting_stage=2)
        assert mgr.active_horizons() == ["SHORT", "MEDIUM"]

    def test_stage3_active_horizons(self) -> None:
        """Stage 3 must expose ['SHORT', 'MEDIUM', 'LONG']."""
        mgr = _make_manager(starting_stage=3)
        assert mgr.active_horizons() == ["SHORT", "MEDIUM", "LONG"]

    def test_stage4_active_horizons(self) -> None:
        """Stage 4 (final) must expose all four horizons."""
        mgr = _make_manager(starting_stage=4)
        assert mgr.active_horizons() == ["SHORT", "MEDIUM", "LONG", "MACRO"]

    def test_active_horizons_returns_copy(self) -> None:
        """Mutating the returned list must not change internal state."""
        mgr = _make_manager(starting_stage=1)
        horizons = mgr.active_horizons()
        horizons.append("EXTRA")
        assert mgr.active_horizons() == ["SHORT"]


# ---------------------------------------------------------------------------
# MANDATORY TEST 3 — try_advance() succeeds when criteria met
# ---------------------------------------------------------------------------


class TestTryAdvanceSuccess:
    """MANDATORY: try_advance() must advance the stage when all criteria are met."""

    def test_advance_from_stage1_to_stage2(self) -> None:
        """MANDATORY: Stage 1 → 2 when accuracy>0.54 AND reward>0.30 AND days≥14."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.56,
            mean_reward=0.35,
            n_days_covered=14,
        )
        assert advanced is True
        assert mgr.current_stage() == 2

    def test_advance_from_stage2_to_stage3(self) -> None:
        """Stage 2 → 3 when accuracy>0.53 AND reward>0.20 AND days≥21."""
        mgr = _make_manager(starting_stage=2)
        advanced = mgr.try_advance(
            rolling_accuracy=0.55,
            mean_reward=0.25,
            n_days_covered=21,
        )
        assert advanced is True
        assert mgr.current_stage() == 3

    def test_advance_from_stage3_to_stage4(self) -> None:
        """Stage 3 → 4 when accuracy>0.52 AND reward>0.15 AND days≥28."""
        mgr = _make_manager(starting_stage=3)
        advanced = mgr.try_advance(
            rolling_accuracy=0.54,
            mean_reward=0.20,
            n_days_covered=28,
        )
        assert advanced is True
        assert mgr.current_stage() == 4

    def test_advance_updates_active_horizons(self) -> None:
        """After advancing stage 1→2, active_horizons() must return SHORT+MEDIUM."""
        mgr = _make_manager(starting_stage=1)
        mgr.try_advance(rolling_accuracy=0.60, mean_reward=0.40, n_days_covered=15)
        assert mgr.active_horizons() == ["SHORT", "MEDIUM"]

    def test_advance_records_history_entry(self) -> None:
        """Each advancement must append one entry to advancement_history()."""
        mgr = _make_manager(starting_stage=1)
        assert len(mgr.advancement_history()) == 0
        mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        assert len(mgr.advancement_history()) == 1

    def test_advance_history_entry_fields(self) -> None:
        """History entry must contain from_stage, to_stage, rolling_accuracy, mean_reward, n_days_covered."""
        mgr = _make_manager(starting_stage=1)
        acc, reward, days = 0.57, 0.32, 16
        mgr.try_advance(rolling_accuracy=acc, mean_reward=reward, n_days_covered=days)
        entry = mgr.advancement_history()[0]
        assert entry["from_stage"] == 1
        assert entry["to_stage"] == 2
        assert entry["rolling_accuracy"] == pytest.approx(acc)
        assert entry["mean_reward"] == pytest.approx(reward)
        assert entry["n_days_covered"] == days


# ---------------------------------------------------------------------------
# MANDATORY TEST 4 — try_advance() does NOT advance when criteria unmet
# ---------------------------------------------------------------------------


class TestTryAdvanceFailure:
    """MANDATORY: try_advance() must return False when any criterion is unmet."""

    def test_insufficient_accuracy_no_advance(self) -> None:
        """MANDATORY: accuracy < threshold → no advance."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.53,  # below 0.54 threshold
            mean_reward=0.35,
            n_days_covered=14,
        )
        assert advanced is False
        assert mgr.current_stage() == 1

    def test_insufficient_mean_reward_no_advance(self) -> None:
        """MANDATORY: mean_reward < threshold → no advance."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.56,
            mean_reward=0.29,  # below 0.30 threshold
            n_days_covered=14,
        )
        assert advanced is False
        assert mgr.current_stage() == 1

    def test_insufficient_days_no_advance(self) -> None:
        """MANDATORY: n_days_covered < min_days → no advance."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.56,
            mean_reward=0.35,
            n_days_covered=13,  # below 14-day minimum
        )
        assert advanced is False
        assert mgr.current_stage() == 1

    def test_no_advance_does_not_append_history(self) -> None:
        """Failed advancement must not append to history."""
        mgr = _make_manager(starting_stage=1)
        mgr.try_advance(rolling_accuracy=0.50, mean_reward=0.10, n_days_covered=5)
        assert len(mgr.advancement_history()) == 0

    def test_exact_boundary_accuracy_advances(self) -> None:
        """Accuracy exactly at threshold (0.54) must still advance (>=)."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.54,
            mean_reward=0.30,
            n_days_covered=14,
        )
        assert advanced is True

    def test_exact_boundary_days_advances(self) -> None:
        """n_days_covered exactly at min_days must still advance (>=)."""
        mgr = _make_manager(starting_stage=1)
        advanced = mgr.try_advance(
            rolling_accuracy=0.56,
            mean_reward=0.35,
            n_days_covered=14,  # exactly 14 — meets criterion
        )
        assert advanced is True


# ---------------------------------------------------------------------------
# MANDATORY TEST 5 — force_set_stage() bypasses criteria
# ---------------------------------------------------------------------------


class TestForceSetStage:
    """MANDATORY: force_set_stage() must immediately change the stage."""

    def test_force_to_stage_4(self) -> None:
        """MANDATORY: force_set_stage(4) sets stage to 4 instantly."""
        mgr = _make_manager(starting_stage=1)
        mgr.force_set_stage(4)
        assert mgr.current_stage() == 4

    def test_force_down_to_stage_1_from_3(self) -> None:
        """force_set_stage allows regression (emergency use only)."""
        mgr = _make_manager(starting_stage=3)
        mgr.force_set_stage(1)
        assert mgr.current_stage() == 1

    def test_force_to_same_stage_is_noop(self) -> None:
        """force_set_stage to current stage must not raise."""
        mgr = _make_manager(starting_stage=2)
        mgr.force_set_stage(2)
        assert mgr.current_stage() == 2

    def test_force_invalid_stage_raises(self) -> None:
        """force_set_stage with invalid stage must raise ValueError."""
        mgr = _make_manager(starting_stage=1)
        with pytest.raises(ValueError, match="stage="):
            mgr.force_set_stage(0)

    def test_force_out_of_range_raises(self) -> None:
        """force_set_stage(5) must raise ValueError (only stages 1–4 exist)."""
        mgr = _make_manager(starting_stage=1)
        with pytest.raises(ValueError):
            mgr.force_set_stage(5)

    def test_force_updates_active_horizons(self) -> None:
        """After force_set_stage(3), active_horizons() must reflect stage 3."""
        mgr = _make_manager(starting_stage=1)
        mgr.force_set_stage(3)
        assert mgr.active_horizons() == ["SHORT", "MEDIUM", "LONG"]


# ---------------------------------------------------------------------------
# MANDATORY TEST 6 — is_final_stage() and no advancement past stage 4
# ---------------------------------------------------------------------------


class TestFinalStage:
    """MANDATORY: Stage 4 is final; try_advance() must be a no-op."""

    def test_is_final_stage_true_at_4(self) -> None:
        """MANDATORY: is_final_stage() must be True at stage 4."""
        mgr = _make_manager(starting_stage=4)
        assert mgr.is_final_stage() is True

    def test_is_final_stage_false_at_1(self) -> None:
        """is_final_stage() must be False at stage 1."""
        mgr = _make_manager(starting_stage=1)
        assert mgr.is_final_stage() is False

    def test_try_advance_at_final_stage_returns_false(self) -> None:
        """MANDATORY: try_advance() at stage 4 must return False (no-op)."""
        mgr = _make_manager(starting_stage=4)
        advanced = mgr.try_advance(
            rolling_accuracy=0.99,
            mean_reward=10.0,
            n_days_covered=999,
        )
        assert advanced is False
        assert mgr.current_stage() == 4

    def test_advance_3_times_reaches_final(self) -> None:
        """Advancing three times from stage 1 reaches stage 4."""
        mgr = _make_manager(starting_stage=1)
        # Stage 1 → 2
        mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        # Stage 2 → 3
        mgr.try_advance(rolling_accuracy=0.55, mean_reward=0.25, n_days_covered=21)
        # Stage 3 → 4
        mgr.try_advance(rolling_accuracy=0.54, mean_reward=0.20, n_days_covered=28)
        assert mgr.is_final_stage() is True
        assert len(mgr.advancement_history()) == 3


# ---------------------------------------------------------------------------
# MANDATORY TEST 7 — advancement_history() immutability
# ---------------------------------------------------------------------------


class TestAdvancementHistoryImmutability:
    """MANDATORY: Mutating the returned advancement_history() list must not alter state."""

    def test_returned_list_is_copy(self) -> None:
        """MANDATORY: Appending to the returned list must not change internal state."""
        mgr = _make_manager(starting_stage=1)
        mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        h1 = mgr.advancement_history()
        h1.append({"injected": True})
        h2 = mgr.advancement_history()
        assert len(h2) == 1  # still 1 real entry, not 2

    def test_history_order_oldest_first(self) -> None:
        """History entries are in chronological (oldest-first) order."""
        mgr = _make_manager(starting_stage=1)
        mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        mgr.try_advance(rolling_accuracy=0.55, mean_reward=0.25, n_days_covered=21)
        history = mgr.advancement_history()
        assert history[0]["from_stage"] == 1
        assert history[1]["from_stage"] == 2


# ---------------------------------------------------------------------------
# stage_info() tests
# ---------------------------------------------------------------------------


class TestStageInfo:
    """stage_info() returns a correct StageInfo frozen dataclass."""

    def test_stage_info_stage_1(self) -> None:
        """StageInfo at stage 1 has correct fields."""
        mgr = _make_manager(starting_stage=1)
        info: StageInfo = mgr.stage_info()
        assert info.stage_number == 1
        assert info.label == "SHORT_ONLY"
        assert info.horizons == ["SHORT"]
        assert info.is_final is False

    def test_stage_info_stage_4(self) -> None:
        """StageInfo at stage 4 has is_final=True and all 4 horizons."""
        mgr = _make_manager(starting_stage=4)
        info: StageInfo = mgr.stage_info()
        assert info.stage_number == 4
        assert info.is_final is True
        assert "MACRO" in info.horizons

    def test_stage_info_is_frozen(self) -> None:
        """StageInfo must be a frozen dataclass (immutable)."""
        mgr = _make_manager(starting_stage=1)
        info: StageInfo = mgr.stage_info()
        with pytest.raises((AttributeError, TypeError)):
            info.stage_number = 99  # type: ignore[misc]

    def test_stage_info_horizons_is_list(self) -> None:
        """StageInfo.horizons must be a list of strings."""
        mgr = _make_manager(starting_stage=2)
        info: StageInfo = mgr.stage_info()
        assert isinstance(info.horizons, list)
        assert all(isinstance(h, str) for h in info.horizons)


# ---------------------------------------------------------------------------
# MANDATORY PROMPT 9.3 TESTS
# ---------------------------------------------------------------------------


def _make_verified_df(
    n: int = 100,
    accuracy: float = 0.60,
    mean_reward: float = 0.35,
) -> pd.DataFrame:
    """Build a minimal verified-predictions DataFrame.

    ``mean_reward`` is the desired overall mean of ``reward_score``; noise is
    small (std=0.02) so the sample mean reliably meets the criterion.
    ``accuracy`` controls the fraction of ``direction_correct=True`` rows.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    direction_correct = rng.random(n) < accuracy
    # reward_score has overall mean ≈ mean_reward independent of accuracy
    reward_score = mean_reward + rng.normal(0, 0.02, n)
    return pd.DataFrame({"direction_correct": direction_correct, "reward_score": reward_score})


class TestCheckAdvancement:
    """MANDATORY PROMPT 9.3: check_advancement() and advance_stage() behaviour."""

    def test_check_advancement_fails_when_only_3_of_4_criteria_met(self) -> None:
        """MANDATORY: Stage must NOT advance if only 3 out of 4 criteria are met.

        All 4 criteria must be met simultaneously: min_days, min_samples,
        min_accuracy, min_mean_reward.  Here min_samples is deliberately unmet.
        """
        mgr = _make_manager(starting_stage=1)
        # Set stage_start_ms far in the past so min_days is met
        mgr._stage_start_ms = int(
            mgr._stage_start_ms - 20 * 24 * 3_600_000  # 20 days ago
        )
        # Only 10 samples — below min_samples (50) for stage 1
        df = _make_verified_df(n=10, accuracy=0.60, mean_reward=0.40)
        result: AdvancementResult = mgr.check_advancement("SHORT", df)
        assert result.can_advance is False
        assert "min_samples" in result.failing_criteria

    def test_check_advancement_fails_when_days_below_min(self) -> None:
        """MANDATORY: Stage must NOT advance if days < min_days, even if other criteria pass."""
        mgr = _make_manager(starting_stage=1)
        # stage_start_ms = now → days_in_stage ≈ 0  (min_days = 14 for stage 1)
        df = _make_verified_df(n=200, accuracy=0.60, mean_reward=0.40)
        result: AdvancementResult = mgr.check_advancement("SHORT", df)
        assert result.can_advance is False
        assert "min_days" in result.failing_criteria

    def test_get_active_horizons_stage1_returns_short_only(self) -> None:
        """MANDATORY: get_active_horizons() must return exactly ['SHORT'] at stage 1."""
        mgr = _make_manager(starting_stage=1)
        horizons = mgr.get_active_horizons()
        assert horizons == ["SHORT"]

    def test_get_active_horizons_stage2_returns_short_and_medium(self) -> None:
        """MANDATORY: get_active_horizons() must return ['SHORT', 'MEDIUM'] at stage 2."""
        mgr = _make_manager(starting_stage=2)
        horizons = mgr.get_active_horizons()
        assert horizons == ["SHORT", "MEDIUM"]

    def test_check_max_wait_returns_true_after_exceeding_max_days(self) -> None:
        """MANDATORY: check_max_wait() must return True after max_wait_days elapsed."""
        mgr = _make_manager(starting_stage=1)
        # Stage 1 max_wait_days = 90; simulate 100 days elapsed
        past_ms = mgr._stage_start_ms - 100 * 24 * 3_600_000
        mgr._stage_start_ms = past_ms
        assert mgr.check_max_wait() is True

    def test_advance_stage_cannot_go_past_4(self) -> None:
        """MANDATORY: advance_stage() must not advance past stage 4."""
        mgr = _make_manager(starting_stage=4)
        returned = mgr.advance_stage()
        assert returned == 4
        assert mgr.get_current_stage() == 4

    def test_check_advancement_all_criteria_pass(self) -> None:
        """check_advancement() returns can_advance=True when all 4 criteria are met."""
        mgr = _make_manager(starting_stage=1)
        # 20 days elapsed (> min_days=14)
        mgr._stage_start_ms = int(mgr._stage_start_ms - 20 * 24 * 3_600_000)
        # 100 samples (> min_samples=50), accuracy=0.60 (> 0.54), reward=0.40 (> 0.30)
        df = _make_verified_df(n=100, accuracy=0.60, mean_reward=0.40)
        result: AdvancementResult = mgr.check_advancement("SHORT", df)
        assert result.can_advance is True
        assert result.failing_criteria == []

    def test_advance_stage_increments_from_1_to_2(self) -> None:
        """advance_stage() unconditionally increments stage."""
        mgr = _make_manager(starting_stage=1)
        new_stage = mgr.advance_stage()
        assert new_stage == 2
        assert mgr.get_current_stage() == 2

    def test_check_max_wait_false_when_recent(self) -> None:
        """check_max_wait() returns False immediately after stage entry."""
        mgr = _make_manager(starting_stage=1)
        # Stage was just entered (now)
        assert mgr.check_max_wait() is False

    def test_advancement_result_is_dataclass(self) -> None:
        """AdvancementResult is a plain dataclass (not frozen — mutable for easy construction)."""
        r = AdvancementResult(can_advance=True, failing_criteria=[], days_in_stage=15)
        assert r.can_advance is True
        assert r.days_in_stage == 15

    def test_get_current_stage_alias(self) -> None:
        """get_current_stage() is an alias for current_stage()."""
        mgr = _make_manager(starting_stage=3)
        assert mgr.get_current_stage() == mgr.current_stage() == 3
