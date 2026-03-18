"""Unit tests for src/rl/predictor.py — MultiHorizonPredictor.

MANDATORY tests:
  1. Only active-curriculum horizons produce PredictionRecord objects.
  2. target_open_time = open_time + horizon_candles × 3_600_000.
  3. confidence_score = 0.5 + |ensemble_prob - 0.5|  (maps to [0.5, 1.0]).
  4. Storage insert called once per active horizon.
  5. Invalid inputs raise ValueError before any storage call.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from src.config import RLConfig, _load_yaml
from src.processing.schemas import HORIZON_CANDLES
from src.rl.curriculum import CurriculumManager
from src.rl.predictor import MultiHorizonPredictor

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RL_CFG: RLConfig = RLConfig(**_load_yaml("rl_config.yaml"))
_NOW_MS: int = 1_700_000_000_000
_OPEN_TIME: int = 1_700_000_000_000
_INTERVAL_MS: int = 3_600_000


def _make_curriculum(starting_stage: int = 1) -> CurriculumManager:
    return CurriculumManager(rl_cfg=_RL_CFG, starting_stage=starting_stage)


def _make_storage() -> MagicMock:
    s = MagicMock()
    s.insert_prediction.return_value = "fake-uuid"
    return s


def _make_predictor(
    starting_stage: int = 1,
    storage: MagicMock | None = None,
) -> MultiHorizonPredictor:
    curriculum = _make_curriculum(starting_stage)
    return MultiHorizonPredictor(
        storage=storage or _make_storage(),
        curriculum=curriculum,
        rl_cfg=_RL_CFG,
    )


def _call_predictor(predictor: MultiHorizonPredictor, **overrides) -> list:
    kwargs = dict(
        open_time=_OPEN_TIME,
        close_price=0.102,
        predicted_direction=1,
        ensemble_prob=0.68,
        lstm_prob=0.70,
        xgb_prob=0.65,
        regime_label="TRENDING_BULL",
        model_version="v1.0",
        now_ms=_NOW_MS,
    )
    kwargs.update(overrides)
    return predictor.generate_and_store(**kwargs)


# ---------------------------------------------------------------------------
# MANDATORY TEST 1 — Only active horizons produce records
# ---------------------------------------------------------------------------


class TestActiveHorizonsGating:
    """MANDATORY: Only curriculum-active horizons emit PredictionRecords."""

    def test_stage1_emits_short_only(self) -> None:
        """MANDATORY: Stage 1 → exactly 1 record with horizon_label='SHORT'."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=1, storage=s)
        records = _call_predictor(pred)
        assert len(records) == 1
        assert records[0].horizon_label == "SHORT"

    def test_stage2_emits_short_and_medium(self) -> None:
        """Stage 2 → exactly 2 records: SHORT and MEDIUM."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=2, storage=s)
        records = _call_predictor(pred)
        assert len(records) == 2
        labels = {r.horizon_label for r in records}
        assert labels == {"SHORT", "MEDIUM"}

    def test_stage3_emits_three_horizons(self) -> None:
        """Stage 3 → exactly 3 records: SHORT, MEDIUM, LONG."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=3, storage=s)
        records = _call_predictor(pred)
        assert len(records) == 3
        labels = {r.horizon_label for r in records}
        assert labels == {"SHORT", "MEDIUM", "LONG"}

    def test_stage4_emits_all_four_horizons(self) -> None:
        """Stage 4 → exactly 4 records: SHORT, MEDIUM, LONG, MACRO."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=4, storage=s)
        records = _call_predictor(pred)
        assert len(records) == 4
        labels = {r.horizon_label for r in records}
        assert labels == {"SHORT", "MEDIUM", "LONG", "MACRO"}


# ---------------------------------------------------------------------------
# MANDATORY TEST 2 — target_open_time computation
# ---------------------------------------------------------------------------


class TestTargetOpenTime:
    """MANDATORY: target_open_time = open_time + horizon_candles × 3_600_000."""

    @pytest.mark.parametrize("stage, horizon", [
        (1, "SHORT"),
        (2, "MEDIUM"),
        (3, "LONG"),
        (4, "MACRO"),
    ])
    def test_target_open_time_formula(self, stage: int, horizon: str) -> None:
        """MANDATORY: target_open_time formula for all horizons."""
        pred = _make_predictor(starting_stage=stage)
        records = _call_predictor(pred, open_time=_OPEN_TIME)
        rec = next(r for r in records if r.horizon_label == horizon)
        expected_target = _OPEN_TIME + HORIZON_CANDLES[horizon] * _INTERVAL_MS
        assert rec.target_open_time == expected_target

    def test_target_open_time_short_is_4h_ahead(self) -> None:
        """SHORT target_open_time is exactly 4 hours after open_time."""
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, open_time=_OPEN_TIME)
        assert records[0].target_open_time == _OPEN_TIME + 4 * _INTERVAL_MS

    def test_target_open_time_respects_custom_open_time(self) -> None:
        """custom open_time is correctly propagated to target_open_time."""
        custom_t = 1_600_000_000_000
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, open_time=custom_t)
        assert records[0].target_open_time == custom_t + 4 * _INTERVAL_MS


# ---------------------------------------------------------------------------
# MANDATORY TEST 3 — confidence_score derivation
# ---------------------------------------------------------------------------


class TestConfidenceScore:
    """MANDATORY: confidence_score = 0.5 + |ensemble_prob - 0.5|."""

    @pytest.mark.parametrize("prob, expected_conf", [
        (0.5, 0.5),    # at neutral → minimum confidence
        (0.75, 0.75),  # 0.5 + |0.75-0.5| = 0.75
        (0.25, 0.75),  # 0.5 + |0.25-0.5| = 0.75
        (1.0, 1.0),    # maximum confidence
        (0.0, 1.0),    # maximum confidence on the sell side
        (0.68, 0.68),  # typical BUY case
    ])
    def test_confidence_score_formula(self, prob: float, expected_conf: float) -> None:
        """MANDATORY: confidence_score formula at various ensemble_prob values."""
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, ensemble_prob=prob)
        assert records[0].confidence_score == pytest.approx(expected_conf, abs=1e-9)

    def test_confidence_score_in_range(self) -> None:
        """confidence_score is always in [0.5, 1.0]."""
        import numpy as np
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.0, 1.0, 20)
        pred = _make_predictor(starting_stage=1)
        for p in probs:
            records = _call_predictor(pred, ensemble_prob=float(p))
            conf = records[0].confidence_score
            assert 0.5 <= conf <= 1.0, f"confidence_score={conf} out of [0.5, 1.0]"


# ---------------------------------------------------------------------------
# MANDATORY TEST 4 — Storage insert called once per active horizon
# ---------------------------------------------------------------------------


class TestStorageInsertCalls:
    """MANDATORY: insert_prediction is called exactly once per active horizon."""

    def test_stage1_calls_insert_once(self) -> None:
        """MANDATORY: Stage 1 → exactly 1 insert_prediction call."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=1, storage=s)
        _call_predictor(pred)
        assert s.insert_prediction.call_count == 1

    def test_stage4_calls_insert_four_times(self) -> None:
        """Stage 4 → exactly 4 insert_prediction calls."""
        s = _make_storage()
        pred = _make_predictor(starting_stage=4, storage=s)
        _call_predictor(pred)
        assert s.insert_prediction.call_count == 4

    def test_storage_failure_suppressed(self) -> None:
        """Storage errors must not raise — the failed record is excluded from results."""
        s = _make_storage()
        s.insert_prediction.side_effect = RuntimeError("DB down")
        pred = _make_predictor(starting_stage=1, storage=s)
        records = _call_predictor(pred)
        # Storage raised, so no successful inserts — returns empty list
        assert records == []

    def test_partial_storage_failure(self) -> None:
        """Only the horizons that succeed are in the returned list."""
        call_count = [0]

        def _flaky(record):
            call_count[0] += 1
            if record.horizon_label == "MEDIUM":
                raise RuntimeError("MEDIUM insert failed")

        s = _make_storage()
        s.insert_prediction.side_effect = _flaky
        pred = _make_predictor(starting_stage=2, storage=s)
        records = _call_predictor(pred)
        # SHORT succeeds, MEDIUM fails
        assert len(records) == 1
        assert records[0].horizon_label == "SHORT"


# ---------------------------------------------------------------------------
# MANDATORY TEST 5 — Invalid inputs raise ValueError
# ---------------------------------------------------------------------------


class TestInputValidation:
    """MANDATORY: Invalid inputs raise ValueError before any storage call."""

    def test_negative_close_price_raises(self) -> None:
        """MANDATORY: close_price <= 0 must raise ValueError."""
        s = _make_storage()
        pred = _make_predictor(storage=s)
        with pytest.raises(ValueError, match="close_price"):
            _call_predictor(pred, close_price=0.0)
        s.insert_prediction.assert_not_called()

    def test_invalid_direction_raises(self) -> None:
        """MANDATORY: predicted_direction not in {-1, 0, 1} must raise ValueError."""
        s = _make_storage()
        pred = _make_predictor(storage=s)
        with pytest.raises(ValueError, match="predicted_direction"):
            _call_predictor(pred, predicted_direction=2)
        s.insert_prediction.assert_not_called()

    def test_ensemble_prob_above_1_raises(self) -> None:
        """ensemble_prob > 1.0 must raise ValueError."""
        s = _make_storage()
        pred = _make_predictor(storage=s)
        with pytest.raises(ValueError, match="ensemble_prob"):
            _call_predictor(pred, ensemble_prob=1.1)
        s.insert_prediction.assert_not_called()

    def test_ensemble_prob_below_0_raises(self) -> None:
        """ensemble_prob < 0.0 must raise ValueError."""
        s = _make_storage()
        pred = _make_predictor(storage=s)
        with pytest.raises(ValueError, match="ensemble_prob"):
            _call_predictor(pred, ensemble_prob=-0.1)
        s.insert_prediction.assert_not_called()


# ---------------------------------------------------------------------------
# Additional field tests
# ---------------------------------------------------------------------------


class TestRecordFields:
    """PredictionRecord fields are correctly populated."""

    def test_symbol_is_dogeusdt(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred)
        assert records[0].symbol == "DOGEUSDT"

    def test_model_version_propagated(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, model_version="v2.5")
        assert records[0].model_version == "v2.5"

    def test_regime_label_propagated(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, regime_label="DECOUPLED")
        assert records[0].regime_label == "DECOUPLED"

    def test_predicted_direction_propagated(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, predicted_direction=-1)
        assert records[0].predicted_direction == -1

    def test_lstm_xgb_prob_propagated(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, lstm_prob=0.72, xgb_prob=0.60)
        assert records[0].lstm_prob == pytest.approx(0.72)
        assert records[0].xgb_prob == pytest.approx(0.60)

    def test_price_at_prediction_propagated(self) -> None:
        pred = _make_predictor(starting_stage=1)
        records = _call_predictor(pred, close_price=0.12345)
        assert records[0].price_at_prediction == pytest.approx(0.12345)

    def test_now_ms_defaults_to_real_time(self) -> None:
        """Omitting now_ms should use a real current timestamp (> 0)."""
        pred = _make_predictor(starting_stage=1)
        records = pred.generate_and_store(
            open_time=_OPEN_TIME,
            close_price=0.102,
            predicted_direction=1,
            ensemble_prob=0.68,
            lstm_prob=0.70,
            xgb_prob=0.65,
            regime_label="TRENDING_BULL",
            model_version="v1.0",
        )
        assert records[0].created_at > 0

    def test_prediction_id_is_unique_uuid(self) -> None:
        """Each call generates a unique prediction_id."""
        pred = _make_predictor(starting_stage=1)
        r1 = _call_predictor(pred)
        r2 = _call_predictor(pred)
        assert r1[0].prediction_id != r2[0].prediction_id

    def test_active_horizons_delegates_to_curriculum(self) -> None:
        """active_horizons() must return the curriculum's active horizons."""
        pred = _make_predictor(starting_stage=2)
        assert pred.active_horizons() == ["SHORT", "MEDIUM"]
