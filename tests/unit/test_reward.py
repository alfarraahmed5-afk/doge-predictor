"""Unit tests for src/rl/reward.py — compute_reward().

All 8 MANDATORY reward scenarios are covered:
  1. Correct + low confidence  (SHORT)
  2. Correct + high confidence (SHORT)
  3. Wrong  + low confidence  (SHORT)
  4. Wrong  + high confidence (SHORT) — max punishment
  5. Correct + MEDIUM horizon
  6. Correct + LONG horizon
  7. DECOUPLED regime  (reduced decay)
  8. Flat direction    (hedge score)

Additional tests cover: validation errors, MACRO horizon weight, calibration
score bounds, and DECOUPLED vs non-DECOUPLED magnitude comparison.
"""

from __future__ import annotations

import math

import pytest

from src.config import RLConfig, _load_yaml
from src.rl.reward import compute_reward
from src.processing.schemas import RewardResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG_RAW = _load_yaml("rl_config.yaml")
_RL_CFG: RLConfig = RLConfig(**_CFG_RAW)

# Shared fixture prices — a small 2% move so error_pct is predictable
_PRICE_BASE = 0.100  # prediction price
_PRICE_UP   = 0.102  # +2% move
_PRICE_DOWN = 0.098  # -2% move


def _reward(
    horizon: str = "SHORT",
    predicted_direction: int = 1,
    actual_direction: int = 1,
    predicted_prob: float = 0.70,
    price_at_prediction: float = _PRICE_BASE,
    actual_price: float = _PRICE_UP,
    regime: str = "TRENDING_BULL",
) -> RewardResult:
    """Convenience wrapper around compute_reward with sensible defaults."""
    return compute_reward(
        horizon=horizon,
        predicted_direction=predicted_direction,
        actual_direction=actual_direction,
        predicted_prob=predicted_prob,
        price_at_prediction=price_at_prediction,
        actual_price=actual_price,
        regime=regime,
        rl_cfg=_RL_CFG,
    )


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 1 — Correct + low confidence (SHORT)
# ---------------------------------------------------------------------------


class TestScenario1CorrectLowConfidenceShort:
    """Correct direction prediction with low confidence on SHORT horizon."""

    def test_direction_correct_true(self) -> None:
        result = _reward(predicted_prob=0.55)  # close to threshold
        assert result.direction_correct is True

    def test_direction_score_positive(self) -> None:
        result = _reward(predicted_prob=0.55)
        assert result.direction_score == pytest.approx(1.0)

    def test_reward_score_positive(self) -> None:
        result = _reward(predicted_prob=0.55)
        assert result.reward_score > 0.0

    def test_calibration_score_near_min(self) -> None:
        """Low confidence → calibration close to correct_min (1.0)."""
        result = _reward(predicted_prob=0.55)
        # confidence = 2 * |0.55 - 0.5| = 0.10 → cal = 1.0 + 1.0 * 0.10 = 1.10
        assert result.calibration_score == pytest.approx(1.10, abs=1e-6)

    def test_magnitude_score_in_unit_interval(self) -> None:
        result = _reward(predicted_prob=0.55)
        assert 0.0 < result.magnitude_score <= 1.0

    def test_error_pct_matches_actual_move(self) -> None:
        result = _reward(predicted_prob=0.55)
        expected_error = abs(_PRICE_UP - _PRICE_BASE) / _PRICE_BASE
        assert result.error_pct == pytest.approx(expected_error, rel=1e-6)


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 2 — Correct + high confidence (SHORT)
# ---------------------------------------------------------------------------


class TestScenario2CorrectHighConfidenceShort:
    """Correct direction prediction with high confidence on SHORT horizon."""

    def test_direction_correct_true(self) -> None:
        result = _reward(predicted_prob=0.95)
        assert result.direction_correct is True

    def test_reward_larger_than_low_confidence(self) -> None:
        """High confidence + correct → larger reward than low confidence."""
        low_conf = _reward(predicted_prob=0.55)
        high_conf = _reward(predicted_prob=0.95)
        assert high_conf.reward_score > low_conf.reward_score

    def test_calibration_score_near_max(self) -> None:
        """High confidence → calibration close to correct_max (2.0)."""
        result = _reward(predicted_prob=0.95)
        # confidence = 2 * |0.95 - 0.5| = 0.90 → cal = 1.0 + 1.0 * 0.90 = 1.90
        assert result.calibration_score == pytest.approx(1.90, abs=1e-6)

    def test_calibration_bounded_by_correct_max(self) -> None:
        """prob=1.0 gives confidence=1.0 → calibration = correct_max = 2.0."""
        result = _reward(predicted_prob=1.0)
        assert result.calibration_score == pytest.approx(2.0, abs=1e-6)

    def test_reward_score_positive(self) -> None:
        result = _reward(predicted_prob=0.95)
        assert result.reward_score > 0.0


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 3 — Wrong + low confidence (SHORT)
# ---------------------------------------------------------------------------


class TestScenario3WrongLowConfidenceShort:
    """Wrong direction prediction with low confidence on SHORT horizon."""

    def test_direction_correct_false(self) -> None:
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.55,
            actual_price=_PRICE_DOWN,
        )
        assert result.direction_correct is False

    def test_direction_score_negative(self) -> None:
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.55,
            actual_price=_PRICE_DOWN,
        )
        assert result.direction_score == pytest.approx(-1.0)

    def test_reward_score_negative(self) -> None:
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.55,
            actual_price=_PRICE_DOWN,
        )
        assert result.reward_score < 0.0

    def test_calibration_score_near_wrong_min(self) -> None:
        """Low confidence + wrong → calibration near -1.0."""
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.55,
            actual_price=_PRICE_DOWN,
        )
        # confidence = 0.10 → cal = -1.0 + (-2.0) * 0.10 = -1.20
        assert result.calibration_score == pytest.approx(-1.20, abs=1e-6)

    def test_abs_calibration_smaller_than_high_confidence(self) -> None:
        """Low confidence wrong → smaller punishment than high confidence wrong."""
        low_wrong  = _reward(predicted_direction=1, actual_direction=-1,
                             predicted_prob=0.55, actual_price=_PRICE_DOWN)
        high_wrong = _reward(predicted_direction=1, actual_direction=-1,
                             predicted_prob=0.95, actual_price=_PRICE_DOWN)
        assert abs(low_wrong.reward_score) < abs(high_wrong.reward_score)


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 4 — Wrong + high confidence (SHORT) — max punishment
# ---------------------------------------------------------------------------


class TestScenario4WrongHighConfidenceShort:
    """Wrong direction prediction with high confidence — maximum punishment."""

    def test_reward_is_maximum_negative(self) -> None:
        """High confidence + wrong → largest negative reward vs all other SHORT wrong."""
        low_wrong = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.55,
            actual_price=_PRICE_DOWN,
        )
        high_wrong = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.95,
            actual_price=_PRICE_DOWN,
        )
        assert high_wrong.reward_score < low_wrong.reward_score

    def test_calibration_score_near_wrong_max(self) -> None:
        """prob=0.95 → confidence=0.90 → calibration = -1.0 + (-2.0)*0.90 = -2.80."""
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.95,
            actual_price=_PRICE_DOWN,
        )
        assert result.calibration_score == pytest.approx(-2.80, abs=1e-6)

    def test_calibration_worst_at_prob_zero_or_one(self) -> None:
        """prob=0.0 for a BUY that turned wrong → confidence=1.0 → cal = -3.0."""
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.0,
            actual_price=_PRICE_DOWN,
        )
        assert result.calibration_score == pytest.approx(-3.0, abs=1e-6)

    def test_reward_score_negative(self) -> None:
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.95,
            actual_price=_PRICE_DOWN,
        )
        assert result.reward_score < 0.0

    def test_short_punish_weight_applied(self) -> None:
        """SHORT punish_weight = 1.5 must be applied when wrong."""
        result = _reward(
            predicted_direction=1, actual_direction=-1, predicted_prob=0.5,
            actual_price=_PRICE_DOWN,
        )
        # direction_score=-1.0, confidence=0, cal=-1.0, abs=1.0
        # magnitude_score = exp(-0.035 * 2.0) (2% move × 100 pct-points)
        expected_magnitude = math.exp(-0.035 * 2.0)
        expected_raw = -1.0 * expected_magnitude * 1.0  # abs(cal) = 1.0
        expected_final = expected_raw * 1.5  # punish_weight for SHORT
        assert result.reward_score == pytest.approx(expected_final, rel=1e-5)


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 5 — Correct + MEDIUM horizon
# ---------------------------------------------------------------------------


class TestScenario5CorrectMediumHorizon:
    """Correct prediction on MEDIUM horizon — higher reward weight (1.5×)."""

    def test_reward_positive(self) -> None:
        result = _reward(horizon="MEDIUM", predicted_prob=0.70)
        assert result.reward_score > 0.0

    def test_reward_weight_larger_than_short(self) -> None:
        """MEDIUM reward_weight=1.5 > SHORT reward_weight=1.0."""
        short_r  = _reward(horizon="SHORT",  predicted_prob=0.70)
        medium_r = _reward(horizon="MEDIUM", predicted_prob=0.70)
        assert medium_r.reward_score > short_r.reward_score

    def test_magnitude_score_decays_slower_than_short(self) -> None:
        """MEDIUM decay_constant=0.025 < SHORT decay_constant=0.035 → higher magnitude."""
        short_r  = _reward(horizon="SHORT",  predicted_prob=0.70)
        medium_r = _reward(horizon="MEDIUM", predicted_prob=0.70)
        assert medium_r.magnitude_score > short_r.magnitude_score

    def test_direction_correct(self) -> None:
        assert _reward(horizon="MEDIUM").direction_correct is True


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 6 — Correct + LONG horizon
# ---------------------------------------------------------------------------


class TestScenario6CorrectLongHorizon:
    """Correct prediction on LONG horizon — highest reward weight (2.0×)."""

    def test_reward_positive(self) -> None:
        result = _reward(horizon="LONG", predicted_prob=0.70)
        assert result.reward_score > 0.0

    def test_reward_weight_largest(self) -> None:
        """LONG reward_weight=2.0 — correct reward > MEDIUM and SHORT."""
        short_r  = _reward(horizon="SHORT",  predicted_prob=0.70)
        medium_r = _reward(horizon="MEDIUM", predicted_prob=0.70)
        long_r   = _reward(horizon="LONG",   predicted_prob=0.70)
        assert long_r.reward_score > medium_r.reward_score
        assert long_r.reward_score > short_r.reward_score

    def test_magnitude_score_highest(self) -> None:
        """LONG decay_constant=0.015 → slowest decay → highest magnitude."""
        short_r = _reward(horizon="SHORT", predicted_prob=0.70)
        long_r  = _reward(horizon="LONG",  predicted_prob=0.70)
        assert long_r.magnitude_score > short_r.magnitude_score

    def test_direction_correct(self) -> None:
        assert _reward(horizon="LONG").direction_correct is True

    def test_punish_weight_largest_for_long_wrong(self) -> None:
        """LONG punish_weight=2.5 — wrong LONG prediction is most punished."""
        short_wrong  = _reward(horizon="SHORT",  predicted_direction=1,
                               actual_direction=-1, predicted_prob=0.70,
                               actual_price=_PRICE_DOWN)
        long_wrong   = _reward(horizon="LONG",   predicted_direction=1,
                               actual_direction=-1, predicted_prob=0.70,
                               actual_price=_PRICE_DOWN)
        assert abs(long_wrong.reward_score) > abs(short_wrong.reward_score)


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 7 — DECOUPLED regime (reduced decay)
# ---------------------------------------------------------------------------


class TestScenario7DecoupledRegime:
    """DECOUPLED regime halves decay_constant → broader magnitude bell-curve."""

    def test_magnitude_higher_than_non_decoupled(self) -> None:
        """DECOUPLED: decay multiplied by 0.5 → higher magnitude score."""
        normal    = _reward(regime="TRENDING_BULL", predicted_prob=0.70)
        decoupled = _reward(regime="DECOUPLED",     predicted_prob=0.70)
        assert decoupled.magnitude_score > normal.magnitude_score

    def test_reward_higher_correct_decoupled(self) -> None:
        """Correct prediction in DECOUPLED → higher magnitude → higher (positive) reward."""
        normal    = _reward(regime="TRENDING_BULL", predicted_prob=0.70)
        decoupled = _reward(regime="DECOUPLED",     predicted_prob=0.70)
        assert decoupled.reward_score > normal.reward_score

    def test_decoupled_magnitude_formula(self) -> None:
        """Manual calculation: decay_constant=0.035, multiplier=0.5 → decay=0.0175."""
        result = _reward(regime="DECOUPLED", predicted_prob=0.70)
        # error_pct = 2%, decay = 0.035 * 0.5 = 0.0175
        # magnitude = exp(-0.0175 * 2.0) = exp(-0.035)
        expected_magnitude = math.exp(-0.0175 * 2.0)
        assert result.magnitude_score == pytest.approx(expected_magnitude, rel=1e-6)

    def test_direction_correct_still_works(self) -> None:
        result = _reward(regime="DECOUPLED", predicted_prob=0.70)
        assert result.direction_correct is True

    def test_decoupled_punishment_larger_magnitude(self) -> None:
        """DECOUPLED halves decay → higher magnitude → larger abs(wrong reward).

        decay=0.035 → 0.0175 means the magnitude bell-curve is broader; with the
        same 2 % price error, DECOUPLED yields a higher magnitude score, so the
        wrong-prediction penalty (in absolute terms) is *larger* than in a normal
        regime.
        """
        normal_wrong    = _reward(regime="TRENDING_BULL", predicted_direction=1,
                                  actual_direction=-1, predicted_prob=0.70,
                                  actual_price=_PRICE_DOWN)
        decoupled_wrong = _reward(regime="DECOUPLED",     predicted_direction=1,
                                  actual_direction=-1, predicted_prob=0.70,
                                  actual_price=_PRICE_DOWN)
        assert abs(decoupled_wrong.reward_score) > abs(normal_wrong.reward_score)


# ---------------------------------------------------------------------------
# MANDATORY SCENARIO 8 — Flat direction (hedge score)
# ---------------------------------------------------------------------------


class TestScenario8FlatDirection:
    """Flat / hedge prediction (predicted_direction=0)."""

    def test_direction_score_is_flat_value(self) -> None:
        result = _reward(predicted_direction=0)
        assert result.direction_score == pytest.approx(0.1)

    def test_direction_correct_is_false(self) -> None:
        """A flat prediction is never marked as 'correct'."""
        result = _reward(predicted_direction=0)
        assert result.direction_correct is False

    def test_reward_score_positive(self) -> None:
        """Flat prediction → small positive reward (cautious hedge)."""
        result = _reward(predicted_direction=0)
        assert result.reward_score > 0.0

    def test_reward_smaller_than_correct_directional(self) -> None:
        """Flat hedge < correct directional reward."""
        flat_r    = _reward(predicted_direction=0, predicted_prob=0.70)
        correct_r = _reward(predicted_direction=1, predicted_prob=0.70)
        assert flat_r.reward_score < correct_r.reward_score

    def test_reward_uses_reward_weight(self) -> None:
        """Flat uses reward_weight (not punish_weight) since it's not wrong."""
        result = _reward(predicted_direction=0, predicted_prob=0.5)
        # direction_score=0.1, confidence=0, cal=1.0, abs=1.0
        # magnitude = exp(-0.035 * 2.0)
        expected_magnitude = math.exp(-0.035 * 2.0)
        expected = 0.1 * expected_magnitude * 1.0 * 1.0  # reward_weight for SHORT = 1.0
        assert result.reward_score == pytest.approx(expected, rel=1e-5)

    def test_flat_on_sell_regime(self) -> None:
        """Flat prediction when market actually went down → still positive (hedge worked)."""
        result = _reward(predicted_direction=0, actual_direction=-1,
                         actual_price=_PRICE_DOWN)
        assert result.reward_score > 0.0


# ---------------------------------------------------------------------------
# Additional tests — validation, edge cases, MACRO horizon
# ---------------------------------------------------------------------------


class TestComputeRewardValidation:
    """Input validation tests."""

    def test_unknown_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown horizon"):
            _reward(horizon="WEEKLY")

    def test_invalid_predicted_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="predicted_direction"):
            _reward(predicted_direction=2)

    def test_invalid_actual_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_direction"):
            _reward(actual_direction=0)

    def test_predicted_prob_out_of_range_high(self) -> None:
        with pytest.raises(ValueError, match="predicted_prob"):
            _reward(predicted_prob=1.1)

    def test_predicted_prob_out_of_range_low(self) -> None:
        with pytest.raises(ValueError, match="predicted_prob"):
            _reward(predicted_prob=-0.01)

    def test_zero_price_at_prediction_raises(self) -> None:
        with pytest.raises(ValueError, match="price_at_prediction"):
            _reward(price_at_prediction=0.0)

    def test_negative_price_at_prediction_raises(self) -> None:
        with pytest.raises(ValueError, match="price_at_prediction"):
            _reward(price_at_prediction=-1.0)


class TestMacroHorizon:
    """MACRO horizon has punish_weight=1.0 (not the largest)."""

    def test_reward_positive_when_correct(self) -> None:
        assert _reward(horizon="MACRO").reward_score > 0.0

    def test_punish_weight_smallest_among_horizons(self) -> None:
        """MACRO punish_weight=1.0 < SHORT punish_weight=1.5."""
        macro_wrong = _reward(horizon="MACRO", predicted_direction=1,
                              actual_direction=-1, predicted_prob=0.70,
                              actual_price=_PRICE_DOWN)
        short_wrong = _reward(horizon="SHORT", predicted_direction=1,
                              actual_direction=-1, predicted_prob=0.70,
                              actual_price=_PRICE_DOWN)
        assert abs(macro_wrong.reward_score) < abs(short_wrong.reward_score)

    def test_direction_correct_marked(self) -> None:
        assert _reward(horizon="MACRO").direction_correct is True


class TestRewardResultType:
    """Return type and field sanity."""

    def test_returns_reward_result_instance(self) -> None:
        result = _reward()
        assert isinstance(result, RewardResult)

    def test_error_pct_non_negative(self) -> None:
        assert _reward().error_pct >= 0.0

    def test_magnitude_score_at_most_one(self) -> None:
        """Even with zero price movement, magnitude ≤ 1."""
        result = _reward(actual_price=_PRICE_BASE)  # no price change
        assert result.magnitude_score <= 1.0

    def test_magnitude_score_at_zero_error_is_one(self) -> None:
        """exp(-decay × 0) = 1.0."""
        result = _reward(actual_price=_PRICE_BASE)
        assert result.magnitude_score == pytest.approx(1.0, abs=1e-9)

    def test_sell_correct(self) -> None:
        """Predicted SELL (-1) and price went down → correct."""
        result = _reward(predicted_direction=-1, actual_direction=-1,
                         predicted_prob=0.30, actual_price=_PRICE_DOWN)
        assert result.direction_correct is True
        assert result.reward_score > 0.0

    def test_sell_wrong(self) -> None:
        """Predicted SELL (-1) but price went up → wrong."""
        result = _reward(predicted_direction=-1, actual_direction=1,
                         predicted_prob=0.30, actual_price=_PRICE_UP)
        assert result.direction_correct is False
        assert result.reward_score < 0.0

    def test_full_formula_short_correct_high_conf(self) -> None:
        """End-to-end manual calculation for SHORT, correct, prob=1.0."""
        result = _reward(horizon="SHORT", predicted_direction=1,
                         actual_direction=1, predicted_prob=1.0,
                         price_at_prediction=_PRICE_BASE, actual_price=_PRICE_UP)
        # direction_score = 1.0
        # error_pct = 0.02; decay = 0.035
        # magnitude = exp(-0.035 * 2.0) = exp(-0.07)
        # confidence = 1.0; calibration = 2.0
        # raw = 1.0 * exp(-0.07) * 2.0; reward_weight = 1.0
        expected = 1.0 * math.exp(-0.07) * 2.0 * 1.0
        assert result.reward_score == pytest.approx(expected, rel=1e-6)
