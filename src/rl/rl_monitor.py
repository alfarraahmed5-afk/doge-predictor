"""Prometheus metrics for the RL self-teaching system.

Centralises all RL-specific Prometheus metric definitions and provides
helper methods for recording training events, reward statistics, and
curriculum stage transitions.

All metric names are prefixed with ``doge_rl_`` to avoid collisions with
the inference-pipeline metrics defined in
:mod:`src.monitoring.prometheus_metrics`.

Usage::

    from src.rl.rl_monitor import RLMonitor

    monitor = RLMonitor()
    monitor.record_reward(horizon="SHORT", regime="TRENDING_BULL", reward=0.42)
    monitor.record_training_run(horizons_trained=["SHORT"], n_samples=64)
    monitor.set_curriculum_stage(stage=2)
    monitor.set_buffer_fill(horizon="SHORT", fill_pct=0.75)
"""

from __future__ import annotations

from typing import Final

from loguru import logger

__all__ = ["RLMonitor"]

# ---------------------------------------------------------------------------
# Prometheus metric definitions (with no-op stub fallback)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — RL metrics will be no-ops")


class _Stub:
    """No-op stub that absorbs any attribute access and call."""

    def __getattr__(self, name: str) -> "_Stub":
        return self

    def __call__(self, *args: object, **kwargs: object) -> "_Stub":
        return self

    def labels(self, **kwargs: object) -> "_Stub":
        return self

    def observe(self, value: float) -> None:
        pass

    def inc(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass


def _make_counter(name: str, description: str, labels: list[str]) -> object:
    if not _PROMETHEUS_AVAILABLE:
        return _Stub()
    try:
        return Counter(name, description, labels)
    except ValueError:
        from prometheus_client import REGISTRY  # type: ignore[import]
        return REGISTRY._names_to_collectors.get(name, _Stub())


def _make_gauge(name: str, description: str, labels: list[str] | None = None) -> object:
    if not _PROMETHEUS_AVAILABLE:
        return _Stub()
    try:
        return Gauge(name, description, labels or [])
    except ValueError:
        from prometheus_client import REGISTRY  # type: ignore[import]
        return REGISTRY._names_to_collectors.get(name, _Stub())


def _make_histogram(
    name: str,
    description: str,
    labels: list[str],
    buckets: tuple[float, ...] | None = None,
) -> object:
    if not _PROMETHEUS_AVAILABLE:
        return _Stub()
    kwargs: dict = {}
    if buckets is not None:
        kwargs["buckets"] = buckets
    try:
        return Histogram(name, description, labels, **kwargs)
    except ValueError:
        from prometheus_client import REGISTRY  # type: ignore[import]
        return REGISTRY._names_to_collectors.get(name, _Stub())


# ---------------------------------------------------------------------------
# Module-level metric objects
# ---------------------------------------------------------------------------

#: Total RL rewards recorded, labelled by horizon and regime.
_RL_REWARDS_TOTAL: object = _make_counter(
    "doge_rl_rewards_total",
    "Total RL reward events logged.",
    ["horizon", "regime"],
)

#: Running mean reward per horizon (updated after each verification batch).
_RL_REWARD_MEAN: object = _make_gauge(
    "doge_rl_reward_mean",
    "Rolling mean RL reward score per horizon.",
    ["horizon"],
)

#: Number of RL training runs completed.
_RL_TRAINING_RUNS_TOTAL: object = _make_counter(
    "doge_rl_training_runs_total",
    "Total number of RL training runs completed.",
    ["trigger"],
)

#: Total samples consumed across all RL training runs.
_RL_TRAINING_SAMPLES_TOTAL: object = _make_counter(
    "doge_rl_training_samples_total",
    "Total replay-buffer samples consumed by RL training.",
    ["horizon"],
)

#: Current curriculum stage (1–4).
_RL_CURRICULUM_STAGE: object = _make_gauge(
    "doge_rl_curriculum_stage",
    "Current RL curriculum stage (1=SHORT_ONLY … 4=FULL).",
)

#: Replay buffer fill percentage per horizon.
_RL_BUFFER_FILL_PCT: object = _make_gauge(
    "doge_rl_buffer_fill_pct",
    "Replay buffer fill fraction (0–1) per horizon.",
    ["horizon"],
)

#: Total verified predictions per horizon and correctness.
_RL_VERIFIED_PREDICTIONS_TOTAL: object = _make_counter(
    "doge_rl_verified_predictions_total",
    "Total predictions verified by the RL verifier.",
    ["horizon", "correct"],
)

#: Distribution of reward scores (histogram).
_RL_REWARD_HISTOGRAM: object = _make_histogram(
    "doge_rl_reward_score",
    "Distribution of individual RL reward scores.",
    ["horizon"],
    buckets=(-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 6.0),
)

#: Curriculum stage advancement counter.
_RL_STAGE_ADVANCES_TOTAL: object = _make_counter(
    "doge_rl_stage_advances_total",
    "Total RL curriculum stage advancement events.",
    ["from_stage", "to_stage"],
)


# ---------------------------------------------------------------------------
# RLMonitor
# ---------------------------------------------------------------------------


class RLMonitor:
    """Records RL-specific Prometheus metrics.

    This class is a thin wrapper around the module-level Prometheus metrics.
    All recording methods are idempotent and never raise exceptions — errors
    are logged as warnings and silently suppressed so that metric failures
    never interrupt the RL loop.

    Usage::

        monitor = RLMonitor()
        monitor.record_reward(horizon="SHORT", regime="TRENDING_BULL", reward=0.42)
    """

    def record_reward(
        self,
        horizon: str,
        regime: str,
        reward: float,
    ) -> None:
        """Record a single reward event.

        Updates the reward counter, histogram, and rolling mean gauge.

        Args:
            horizon: Prediction horizon label.
            regime: Market regime label at prediction time.
            reward: The reward score value.
        """
        try:
            _RL_REWARDS_TOTAL.labels(horizon=horizon, regime=regime).inc()  # type: ignore[union-attr]
            _RL_REWARD_HISTOGRAM.labels(horizon=horizon).observe(reward)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.record_reward: metric error: {}", exc)

    def update_mean_reward(self, horizon: str, mean_reward: float) -> None:
        """Update the rolling mean reward gauge for a horizon.

        Args:
            horizon: Prediction horizon label.
            mean_reward: Current rolling mean reward score.
        """
        try:
            _RL_REWARD_MEAN.labels(horizon=horizon).set(mean_reward)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.update_mean_reward: metric error: {}", exc)

    def record_training_run(
        self,
        horizons_trained: list[str],
        n_samples: int,
        trigger: str = "scheduled",
    ) -> None:
        """Record a completed RL training run.

        Args:
            horizons_trained: List of horizon labels that were trained in this run.
            n_samples: Total number of replay buffer samples consumed.
            trigger: Human-readable trigger reason string.
        """
        try:
            _RL_TRAINING_RUNS_TOTAL.labels(trigger=trigger).inc()  # type: ignore[union-attr]
            for horizon in horizons_trained:
                _RL_TRAINING_SAMPLES_TOTAL.labels(horizon=horizon).inc(  # type: ignore[union-attr]
                    n_samples / max(1, len(horizons_trained))
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.record_training_run: metric error: {}", exc)

    def set_curriculum_stage(self, stage: int) -> None:
        """Update the curriculum stage gauge.

        Args:
            stage: Current stage number (1–4).
        """
        try:
            _RL_CURRICULUM_STAGE.set(float(stage))  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.set_curriculum_stage: metric error: {}", exc)

    def record_stage_advance(self, from_stage: int, to_stage: int) -> None:
        """Record a curriculum stage advancement event.

        Args:
            from_stage: Previous stage number.
            to_stage: New stage number.
        """
        try:
            _RL_STAGE_ADVANCES_TOTAL.labels(  # type: ignore[union-attr]
                from_stage=str(from_stage),
                to_stage=str(to_stage),
            ).inc()
            _RL_CURRICULUM_STAGE.set(float(to_stage))  # type: ignore[union-attr]
            logger.info(
                "RLMonitor: curriculum stage {} → {}", from_stage, to_stage
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.record_stage_advance: metric error: {}", exc)

    def set_buffer_fill(self, horizon: str, fill_pct: float) -> None:
        """Update the replay buffer fill gauge for a horizon.

        Args:
            horizon: Prediction horizon label.
            fill_pct: Fill fraction in [0.0, 1.0].
        """
        try:
            _RL_BUFFER_FILL_PCT.labels(horizon=horizon).set(fill_pct)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.warning("RLMonitor.set_buffer_fill: metric error: {}", exc)

    def record_verified_prediction(
        self, horizon: str, direction_correct: bool
    ) -> None:
        """Increment the verified-predictions counter.

        Args:
            horizon: Prediction horizon label.
            direction_correct: Whether the predicted direction was correct.
        """
        try:
            _RL_VERIFIED_PREDICTIONS_TOTAL.labels(  # type: ignore[union-attr]
                horizon=horizon,
                correct=str(direction_correct).lower(),
            ).inc()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RLMonitor.record_verified_prediction: metric error: {}", exc
            )
