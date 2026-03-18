"""Curriculum stage manager for the RL self-teaching system.

The :class:`CurriculumManager` controls which prediction horizons are active at
any point in training.  Stages advance sequentially (never regress) according
to accuracy and mean-reward thresholds defined in ``rl_config.yaml``.

Stage definitions (from CLAUDE.md Section 11)::

    Stage 1 — SHORT only      → advance when 14d accuracy > 54%  AND mean reward > 0.30
    Stage 2 — SHORT + MEDIUM  → advance when 21d accuracy > 53%
    Stage 3 — +LONG           → advance when 28d accuracy > 52%
    Stage 4 — +MACRO          → final stage, no advancement

Usage::

    manager = CurriculumManager(rl_cfg=rl_cfg)
    active_horizons = manager.active_horizons()   # ["SHORT"]

    # Check advancement using verified predictions DataFrame:
    result = manager.check_advancement("SHORT", verified_df)
    if result.can_advance:
        new_stage = manager.advance_stage()

    # After evaluating rolling performance:
    advanced = manager.try_advance(
        rolling_accuracy=0.56,
        mean_reward=0.35,
        n_days_covered=14,
    )
    if advanced:
        active_horizons = manager.active_horizons()   # ["SHORT", "MEDIUM"]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import pandas as pd
from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg

__all__ = ["AdvancementResult", "CurriculumManager", "StageInfo"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STAGE: Final[int] = 4
_MS_PER_DAY: Final[int] = 86_400_000


@dataclass
class AdvancementResult:
    """Outcome of :meth:`CurriculumManager.check_advancement`.

    Args:
        can_advance: ``True`` when ALL 4 criteria are satisfied.
        failing_criteria: Names of criteria that were not met.  Empty when
            ``can_advance=True``.
        days_in_stage: Number of full days elapsed since the current stage
            was entered.
    """

    can_advance: bool
    failing_criteria: list[str]
    days_in_stage: int


@dataclass(frozen=True)
class StageInfo:
    """Snapshot of the current curriculum stage.

    Args:
        stage_number: Current stage index (1–4).
        label: Human-readable stage label (e.g. ``"SHORT_ONLY"``).
        horizons: Active prediction horizon labels for this stage.
        is_final: ``True`` when ``stage_number == 4``.
    """

    stage_number: int
    label: str
    horizons: list[str]
    is_final: bool


class CurriculumManager:
    """Manages curriculum stage transitions.

    Args:
        rl_cfg: Loaded :class:`~src.config.RLConfig`.  Defaults to the global
            singleton.
        starting_stage: Override starting stage (mainly for testing).  Uses
            ``rl_cfg.curriculum.starting_stage`` when ``None``.
    """

    def __init__(
        self,
        rl_cfg: RLConfig | None = None,
        starting_stage: int | None = None,
    ) -> None:
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        curriculum = self._rl_cfg.curriculum

        # Validate stage keys
        if not curriculum.stages:
            raise ValueError("CurriculumManager: rl_config.yaml curriculum.stages is empty.")

        initial = starting_stage if starting_stage is not None else curriculum.starting_stage
        if initial not in curriculum.stages:
            raise ValueError(
                f"CurriculumManager: starting_stage={initial} is not in stages "
                f"({sorted(curriculum.stages.keys())})."
            )

        self._current_stage: int = initial
        self._stage_start_ms: int = int(time.time() * 1000)
        self._advance_history: list[dict] = []  # immutable snapshots of each advancement
        logger.info(
            "CurriculumManager initialised at stage={} ({})",
            self._current_stage,
            self._stage_label(self._current_stage),
        )

    # ------------------------------------------------------------------
    # Public API — read
    # ------------------------------------------------------------------

    def current_stage(self) -> int:
        """Return the current stage number (1–4).

        Returns:
            Integer stage number.
        """
        return self._current_stage

    def active_horizons(self) -> list[str]:
        """Return the list of active prediction horizon labels.

        Returns:
            e.g. ``["SHORT"]``, ``["SHORT", "MEDIUM"]``, etc.
        """
        return list(self._rl_cfg.curriculum.stages[self._current_stage].horizons)

    def is_final_stage(self) -> bool:
        """Return ``True`` when the current stage is the final one.

        Returns:
            Boolean.
        """
        return self._current_stage == _MAX_STAGE

    def stage_info(self) -> StageInfo:
        """Return a frozen snapshot of the current stage.

        Returns:
            :class:`StageInfo` instance.
        """
        stage_cfg = self._rl_cfg.curriculum.stages[self._current_stage]
        return StageInfo(
            stage_number=self._current_stage,
            label=stage_cfg.label,
            horizons=list(stage_cfg.horizons),
            is_final=self.is_final_stage(),
        )

    def advancement_history(self) -> list[dict]:
        """Return the list of past stage advancement events.

        Each entry is a dict with keys: ``from_stage``, ``to_stage``,
        ``rolling_accuracy``, ``mean_reward``, ``n_days_covered``.

        Returns:
            List of dicts (oldest first).
        """
        return list(self._advance_history)

    def get_current_stage(self) -> int:
        """Alias for :meth:`current_stage`.

        Returns:
            Integer stage number (1–4).
        """
        return self._current_stage

    def get_active_horizons(self) -> list[str]:
        """Alias for :meth:`active_horizons`.

        Returns:
            List of active horizon label strings for the current stage.
        """
        return self.active_horizons()

    def check_advancement(
        self,
        horizon: str,
        verified_predictions: pd.DataFrame,
    ) -> AdvancementResult:
        """Check whether ALL 4 advancement criteria are satisfied.

        Criteria checked simultaneously (ALL must be met):

        1. ``min_days`` — days elapsed since stage entry ≥ criterion.
        2. ``min_samples`` — row count of *verified_predictions* ≥ criterion.
        3. ``min_accuracy`` — mean of ``direction_correct`` column ≥ criterion.
        4. ``min_mean_reward`` — mean of ``reward_score`` column ≥ criterion.

        Args:
            horizon: Horizon label whose predictions to evaluate (e.g. ``"SHORT"``).
            verified_predictions: DataFrame of verified prediction rows.  Must
                contain ``"direction_correct"`` (bool) and ``"reward_score"``
                (float) columns.

        Returns:
            :class:`AdvancementResult` with ``can_advance``, ``failing_criteria``,
            and ``days_in_stage``.
        """
        now_ms: int = int(time.time() * 1000)
        days_in_stage: int = int((now_ms - self._stage_start_ms) / _MS_PER_DAY)

        if self.is_final_stage():
            return AdvancementResult(
                can_advance=False,
                failing_criteria=["already_final_stage"],
                days_in_stage=days_in_stage,
            )

        stage_cfg = self._rl_cfg.curriculum.stages[self._current_stage]
        criteria = stage_cfg.advancement_criteria

        if criteria is None:
            return AdvancementResult(
                can_advance=False,
                failing_criteria=["no_criteria_defined"],
                days_in_stage=days_in_stage,
            )

        failing: list[str] = []

        # Criterion 1: minimum days in stage
        if days_in_stage < criteria.min_days:
            failing.append("min_days")

        # Criterion 2: minimum number of verified samples
        n_samples = len(verified_predictions)
        if n_samples < criteria.min_samples:
            failing.append("min_samples")

        # Criteria 3 & 4 require data in the DataFrame
        if n_samples == 0:
            failing.append("min_accuracy")
            failing.append("min_mean_reward")
        else:
            if "direction_correct" not in verified_predictions.columns:
                failing.append("min_accuracy")
            else:
                rolling_accuracy = float(
                    verified_predictions["direction_correct"].astype(float).mean()
                )
                if rolling_accuracy < criteria.min_accuracy:
                    failing.append("min_accuracy")

            if "reward_score" not in verified_predictions.columns:
                failing.append("min_mean_reward")
            else:
                mean_reward = float(verified_predictions["reward_score"].mean())
                if mean_reward < criteria.min_mean_reward:
                    failing.append("min_mean_reward")

        return AdvancementResult(
            can_advance=len(failing) == 0,
            failing_criteria=failing,
            days_in_stage=days_in_stage,
        )

    def advance_stage(self) -> int:
        """Unconditionally increment the curriculum stage by one.

        The caller is responsible for checking :meth:`check_advancement` before
        calling this method.  Does not advance past stage 4.

        Returns:
            New stage number after increment (or current stage if already final).
        """
        if self.is_final_stage():
            logger.warning(
                "CurriculumManager.advance_stage: already at final stage ({}); "
                "no advancement.",
                _MAX_STAGE,
            )
            return self._current_stage

        from_stage = self._current_stage
        self._current_stage += 1
        self._stage_start_ms = int(time.time() * 1000)
        self._advance_history.append(
            {
                "from_stage": from_stage,
                "to_stage": self._current_stage,
                "method": "advance_stage",
            }
        )
        logger.info(
            "CurriculumManager.advance_stage: {} → {} ({})",
            from_stage,
            self._current_stage,
            self._stage_label(self._current_stage),
        )
        return self._current_stage

    def check_max_wait(self, now_ms: int | None = None) -> bool:
        """Return ``True`` if the time spent in the current stage exceeds the
        configured ``max_wait_days`` limit.

        Args:
            now_ms: Current UTC epoch milliseconds.  Defaults to wall clock
                when ``None``.

        Returns:
            ``True`` when ``days_in_stage >= stage.max_wait_days``.
        """
        ts: int = now_ms if now_ms is not None else int(time.time() * 1000)
        days_in_stage = int((ts - self._stage_start_ms) / _MS_PER_DAY)
        stage_cfg = self._rl_cfg.curriculum.stages[self._current_stage]
        max_wait = stage_cfg.max_wait_days
        exceeded = days_in_stage >= max_wait
        if exceeded:
            logger.warning(
                "CurriculumManager.check_max_wait: stage={} has been active for "
                "{} days >= max_wait_days={}",
                self._current_stage,
                days_in_stage,
                max_wait,
            )
        return exceeded

    # ------------------------------------------------------------------
    # Public API — write
    # ------------------------------------------------------------------

    def try_advance(
        self,
        rolling_accuracy: float,
        mean_reward: float,
        n_days_covered: int,
    ) -> bool:
        """Attempt to advance to the next curriculum stage.

        Checks the current stage's ``advancement_criteria``.  If all criteria
        are met the stage is incremented; otherwise the stage is unchanged.
        No-ops on the final stage.

        Args:
            rolling_accuracy: Rolling directional accuracy over the past
                ``n_days_covered`` days, as a fraction in ``[0, 1]``.
            mean_reward: Mean reward score over the same window.
            n_days_covered: Number of days covered by the rolling window.
                Must meet or exceed the stage's ``min_days`` criterion.

        Returns:
            ``True`` if the stage was advanced; ``False`` otherwise.
        """
        if self.is_final_stage():
            logger.debug("try_advance: already at final stage ({})", _MAX_STAGE)
            return False

        stage_cfg = self._rl_cfg.curriculum.stages[self._current_stage]
        criteria = stage_cfg.advancement_criteria

        if criteria is None:
            logger.warning(
                "try_advance: stage={} has no advancement_criteria; cannot advance",
                self._current_stage,
            )
            return False

        # Check all three criteria
        meets_days = n_days_covered >= criteria.min_days
        meets_accuracy = rolling_accuracy >= criteria.min_accuracy
        meets_reward = mean_reward >= criteria.min_mean_reward

        if meets_days and meets_accuracy and meets_reward:
            from_stage = self._current_stage
            self._current_stage += 1
            self._stage_start_ms = int(time.time() * 1000)
            self._advance_history.append(
                {
                    "from_stage": from_stage,
                    "to_stage": self._current_stage,
                    "rolling_accuracy": rolling_accuracy,
                    "mean_reward": mean_reward,
                    "n_days_covered": n_days_covered,
                }
            )
            logger.info(
                "CurriculumManager: advanced stage {} → {} ({}) "
                "| accuracy={:.3f} reward={:.3f} days={}",
                from_stage,
                self._current_stage,
                self._stage_label(self._current_stage),
                rolling_accuracy,
                mean_reward,
                n_days_covered,
            )
            return True

        # Log which criteria were not met
        logger.debug(
            "try_advance: stage={} criteria not met — "
            "days={}/{} accuracy={:.3f}/{:.3f} reward={:.3f}/{:.3f}",
            self._current_stage,
            n_days_covered, criteria.min_days,
            rolling_accuracy, criteria.min_accuracy,
            mean_reward, criteria.min_mean_reward,
        )
        return False

    def force_set_stage(self, stage: int) -> None:
        """Force the manager to a specific stage (testing / recovery only).

        This method bypasses advancement criteria and should not be called in
        production code.

        Args:
            stage: Target stage number (1–4).

        Raises:
            ValueError: If ``stage`` is not a valid stage number.
        """
        if stage not in self._rl_cfg.curriculum.stages:
            raise ValueError(
                f"force_set_stage: stage={stage} not in curriculum stages "
                f"({sorted(self._rl_cfg.curriculum.stages.keys())})."
            )
        logger.warning(
            "CurriculumManager.force_set_stage: {} → {} (forced)",
            self._current_stage,
            stage,
        )
        self._current_stage = stage

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stage_label(self, stage: int) -> str:
        """Return the label string for a given stage number.

        Args:
            stage: Stage number.

        Returns:
            Label string, or ``"UNKNOWN"`` if stage not found.
        """
        cfg = self._rl_cfg.curriculum.stages.get(stage)
        return cfg.label if cfg is not None else "UNKNOWN"
