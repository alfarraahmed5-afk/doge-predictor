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

from dataclasses import dataclass, field
from typing import Final

from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg

__all__ = ["CurriculumManager", "StageInfo"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STAGE: Final[int] = 4


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
