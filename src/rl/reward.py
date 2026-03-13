"""RL reward function for the DOGE self-teaching system.

Computes a scalar reward from the outcome of a single matured prediction.
The reward is used to update model weights via the RL self-training loop.

Formula (from CLAUDE.md Section 11)::

    reward = direction_score x magnitude_score x abs(calibration_score)
             x horizon_weight

Where:

* ``direction_score`` — +1.0 (correct) | -1.0 (wrong) | +0.1 (flat/hedge)
* ``magnitude_score`` — exp(-decay x error_pct); always in (0, 1]
* ``calibration_score`` — confidence-weighted bonus [+1.0, +2.0] for correct
  predictions, or penalty [-1.0, -3.0] for wrong ones; abs() is taken before
  multiplication so the sign comes from ``direction_score``
* ``horizon_weight`` — ``reward_weight`` when correct/flat, ``punish_weight``
  when wrong; both loaded from ``rl_config.yaml``

DECOUPLED regime reduces the ``decay_constant`` by
``RewardSettings.decoupled_decay_multiplier`` (default 0.5), which broadens
the magnitude bell-curve and reduces reward sensitivity during the highest-risk
market state.
"""

from __future__ import annotations

import math
from typing import Final

from loguru import logger

from src.config import RLConfig
from src.processing.schemas import RewardResult

__all__ = ["compute_reward"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_VALID_HORIZONS: Final[frozenset[str]] = frozenset({"SHORT", "MEDIUM", "LONG", "MACRO"})
_VALID_DIRECTIONS: Final[frozenset[int]] = frozenset({-1, 0, 1})
_DECOUPLED_REGIME: Final[str] = "DECOUPLED"

# Direction score values (named to avoid magic numbers)
_DIR_CORRECT: Final[float] = 1.0
_DIR_WRONG: Final[float] = -1.0
_DIR_FLAT: Final[float] = 0.1  # hedge / hold signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_reward(
    horizon: str,
    predicted_direction: int,
    actual_direction: int,
    predicted_prob: float,
    price_at_prediction: float,
    actual_price: float,
    regime: str,
    rl_cfg: RLConfig,
) -> RewardResult:
    """Compute the RL reward for a single matured prediction.

    Args:
        horizon: Prediction horizon label — one of ``"SHORT"``, ``"MEDIUM"``,
            ``"LONG"``, ``"MACRO"``.
        predicted_direction: Direction predicted by the model: ``+1`` (BUY /
            up), ``-1`` (SELL / down), or ``0`` (HOLD / flat).
        actual_direction: Actual market direction that materialised: ``+1``
            (price rose) or ``-1`` (price fell).
        predicted_prob: Ensemble probability produced at prediction time,
            in ``[0.0, 1.0]``.  Values near ``0.5`` indicate low confidence;
            values near ``0.0`` or ``1.0`` indicate high confidence.
        price_at_prediction: Mid-price of the candle on which the prediction
            was made (used as the denominator for ``error_pct``).
        actual_price: Closing price of the maturity candle.
        regime: Market regime label at the time of prediction (e.g.
            ``"TRENDING_BULL"``).  ``"DECOUPLED"`` triggers reduced decay.
        rl_cfg: Loaded :class:`~src.config.RLConfig` instance.

    Returns:
        :class:`~src.processing.schemas.RewardResult` with all component
        scores and the final ``reward_score``.

    Raises:
        ValueError: If ``horizon`` is not a known label, ``predicted_direction``
            or ``actual_direction`` is outside ``{-1, 0, +1}``,
            ``predicted_prob`` is outside ``[0, 1]``, or
            ``price_at_prediction`` is non-positive.
    """
    _validate_inputs(horizon, predicted_direction, actual_direction, predicted_prob, price_at_prediction)

    horizon_cfg = rl_cfg.horizons[horizon]
    reward_cfg = rl_cfg.reward

    # ------------------------------------------------------------------
    # Step 1 — direction score
    # ------------------------------------------------------------------
    is_flat = predicted_direction == 0
    is_correct = (not is_flat) and (predicted_direction == actual_direction)

    if is_flat:
        direction_score = reward_cfg.direction_flat
    elif is_correct:
        direction_score = reward_cfg.direction_correct
    else:
        direction_score = reward_cfg.direction_wrong

    # ------------------------------------------------------------------
    # Step 2 — magnitude score
    # ------------------------------------------------------------------
    error_pct = abs(actual_price - price_at_prediction) / price_at_prediction

    # DECOUPLED regime: reduce decay constant (broadens bell-curve, lowers
    # reward sensitivity in the highest-risk market state)
    decay = horizon_cfg.decay_constant
    if regime == _DECOUPLED_REGIME:
        decay = decay * reward_cfg.decoupled_decay_multiplier

    magnitude_score = math.exp(-decay * error_pct * 100.0)  # error_pct as percentage points

    # ------------------------------------------------------------------
    # Step 3 — calibration score
    # ------------------------------------------------------------------
    # Normalise predicted_prob to a confidence value in [0, 1]:
    #   prob = 0.5 → confidence = 0.0 (completely uncertain)
    #   prob = 0.0 or 1.0 → confidence = 1.0 (completely certain)
    confidence = 2.0 * abs(predicted_prob - 0.5)

    if is_correct or is_flat:
        # Linearly interpolate: low confidence → min, high confidence → max
        calibration_score = (
            reward_cfg.calibration_correct_min
            + (reward_cfg.calibration_correct_max - reward_cfg.calibration_correct_min)
            * confidence
        )
    else:
        # Wrong prediction: higher confidence → larger (more negative) penalty
        # calibration_wrong_min = -1.0, calibration_wrong_max = -3.0
        calibration_score = (
            reward_cfg.calibration_wrong_min
            + (reward_cfg.calibration_wrong_max - reward_cfg.calibration_wrong_min)
            * confidence
        )

    # ------------------------------------------------------------------
    # Step 4 — raw reward
    # ------------------------------------------------------------------
    raw_reward = direction_score * magnitude_score * abs(calibration_score)

    # ------------------------------------------------------------------
    # Step 5 — horizon weight
    # ------------------------------------------------------------------
    if is_correct or is_flat:
        horizon_weight = horizon_cfg.reward_weight
    else:
        horizon_weight = horizon_cfg.punish_weight

    reward_score = raw_reward * horizon_weight

    logger.debug(
        "compute_reward | horizon={} regime={} predicted={} actual={} "
        "prob={:.3f} error_pct={:.4f} "
        "dir={:.2f} mag={:.4f} cal={:.4f} weight={:.2f} reward={:.4f}",
        horizon,
        regime,
        predicted_direction,
        actual_direction,
        predicted_prob,
        error_pct,
        direction_score,
        magnitude_score,
        calibration_score,
        horizon_weight,
        reward_score,
    )

    return RewardResult(
        reward_score=reward_score,
        direction_score=direction_score,
        magnitude_score=magnitude_score,
        calibration_score=calibration_score,
        error_pct=error_pct,
        direction_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    horizon: str,
    predicted_direction: int,
    actual_direction: int,
    predicted_prob: float,
    price_at_prediction: float,
) -> None:
    """Validate compute_reward inputs.

    Args:
        horizon: Horizon label.
        predicted_direction: Model direction prediction.
        actual_direction: Actual market direction.
        predicted_prob: Ensemble probability.
        price_at_prediction: Price at prediction time.

    Raises:
        ValueError: On any invalid input.
    """
    if horizon not in _VALID_HORIZONS:
        raise ValueError(
            f"Unknown horizon '{horizon}'. Must be one of {sorted(_VALID_HORIZONS)}."
        )
    if predicted_direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"predicted_direction must be in {{-1, 0, +1}}, got {predicted_direction}."
        )
    if actual_direction not in (-1, 1):
        raise ValueError(
            f"actual_direction must be +1 or -1, got {actual_direction}."
        )
    if not 0.0 <= predicted_prob <= 1.0:
        raise ValueError(
            f"predicted_prob must be in [0, 1], got {predicted_prob}."
        )
    if price_at_prediction <= 0.0:
        raise ValueError(
            f"price_at_prediction must be > 0, got {price_at_prediction}."
        )
