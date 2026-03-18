"""Unit tests for src/rl/verifier.py and src/rl/replay_buffer.py.

Critical rules checked here:
  - Verifier NEVER processes predictions before target_open_time is in the past
  - actual_direction computed vs price_at_prediction (NOT T-1 close)
  - Interpolated candles are NOT pushed to the replay buffer
  - Outcome fields are written; prediction fields remain immutable
  - Missing OHLCV data causes a skip (retry-able), not a crash
"""

from __future__ import annotations

import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import RLConfig, _load_yaml
from src.processing.schemas import PredictionRecord, RewardResult
from src.rl.replay_buffer import ReplayBuffer
from src.rl.verifier import PredictionImmutabilityError, PredictionVerifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RL_CFG: RLConfig = RLConfig(**_load_yaml("rl_config.yaml"))

_NOW_MS: int = 1_700_000_000_000  # arbitrary fixed "now"
_INTERVAL_MS: int = 3_600_000     # 1h


def _make_record(
    target_open_time: int | None = None,
    price_at_prediction: float = 0.10,
    predicted_direction: int = 1,
    horizon_label: str = "SHORT",
    regime_label: str = "TRENDING_BULL",
    is_verified: bool = False,
) -> PredictionRecord:
    """Build a minimal PredictionRecord."""
    open_time = _NOW_MS - _INTERVAL_MS * 10  # well in the past
    target = target_open_time if target_open_time is not None else open_time + _INTERVAL_MS * 4
    return PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        created_at=open_time,
        open_time=open_time,
        symbol="DOGEUSDT",
        horizon_label=horizon_label,
        horizon_candles=4,
        target_open_time=target,
        price_at_prediction=price_at_prediction,
        predicted_direction=predicted_direction,
        confidence_score=0.70,
        lstm_prob=0.70,
        xgb_prob=0.68,
        regime_label=regime_label,
        model_version="v1.0",
        verified_at=_NOW_MS if is_verified else None,
    )


def _make_candle(close: float = 0.102, is_interpolated: bool = False) -> pd.DataFrame:
    """Return a 1-row OHLCV DataFrame."""
    return pd.DataFrame([{
        "open_time": _NOW_MS - _INTERVAL_MS * 6,
        "open": 0.100,
        "high": 0.103,
        "low": 0.099,
        "close": close,
        "volume": 1000.0,
        "is_interpolated": is_interpolated,
    }])


def _mock_storage(
    pending: list[PredictionRecord] | None = None,
    candle_df: pd.DataFrame | None = None,
    update_returns: bool = True,
    prediction_by_id: PredictionRecord | None = None,
) -> MagicMock:
    """Build a storage mock with sensible defaults."""
    storage = MagicMock()
    storage.get_matured_unverified.return_value = pending or []
    storage.get_ohlcv.return_value = candle_df if candle_df is not None else _make_candle()
    storage.update_prediction_outcome.return_value = update_returns
    storage.push_replay_buffer.return_value = True
    # get_prediction_by_id defaults to returning the same record as in pending
    # (pass prediction_by_id=<altered record> to simulate tampering)
    if prediction_by_id is not None:
        storage.get_prediction_by_id.return_value = prediction_by_id
    else:
        # Return the first pending record unchanged (no tampering)
        if pending:
            storage.get_prediction_by_id.return_value = pending[0]
        else:
            storage.get_prediction_by_id.return_value = None
    return storage


def _make_verifier(storage: MagicMock | None = None) -> PredictionVerifier:
    """Build a PredictionVerifier with a mock storage and injected RLConfig."""
    s = storage or _mock_storage()
    buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
    return PredictionVerifier(storage=s, rl_cfg=_RL_CFG, replay_buffer=buf)


# ---------------------------------------------------------------------------
# Tests — run_verification high-level
# ---------------------------------------------------------------------------


class TestRunVerificationHighLevel:
    """High-level behaviour of run_verification()."""

    def test_returns_zero_when_no_pending(self) -> None:
        v = _make_verifier(_mock_storage(pending=[]))
        assert v.run_verification(as_of_ts=_NOW_MS) == 0

    def test_returns_count_of_verified(self) -> None:
        record = _make_record()
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1

    def test_returns_zero_on_storage_error(self) -> None:
        s = MagicMock()
        s.get_matured_unverified.side_effect = RuntimeError("DB down")
        v = _make_verifier(s)
        assert v.run_verification(as_of_ts=_NOW_MS) == 0

    def test_processes_multiple_records(self) -> None:
        records = [_make_record() for _ in range(5)]
        s = _mock_storage(pending=records)
        v = _make_verifier(s)
        assert v.run_verification(as_of_ts=_NOW_MS) == 5

    def test_uses_current_time_when_as_of_ts_is_none(self) -> None:
        s = _mock_storage(pending=[])
        v = _make_verifier(s)
        v.run_verification()  # no as_of_ts — must not raise
        s.get_matured_unverified.assert_called_once()
        called_ts = s.get_matured_unverified.call_args[0][0]
        # Must be a recent timestamp (within 60 seconds of now)
        assert abs(called_ts - int(time.time() * 1000)) < 60_000


# ---------------------------------------------------------------------------
# MANDATORY — actual_direction computed vs price_at_prediction
# ---------------------------------------------------------------------------


class TestActualDirectionComputation:
    """MANDATORY: direction must be computed vs price_at_prediction, not T-1."""

    def test_up_move_gives_direction_plus_one(self) -> None:
        record = _make_record(price_at_prediction=0.10)
        candle = _make_candle(close=0.102)  # +2%
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        outcome = s.update_prediction_outcome.call_args[0][1]
        assert outcome["actual_direction"] == 1

    def test_down_move_gives_direction_minus_one(self) -> None:
        record = _make_record(price_at_prediction=0.10)
        candle = _make_candle(close=0.098)  # -2%
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        outcome = s.update_prediction_outcome.call_args[0][1]
        assert outcome["actual_direction"] == -1

    def test_outcome_written_with_all_required_keys(self) -> None:
        record = _make_record()
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        outcome = s.update_prediction_outcome.call_args[0][1]
        for key in ("actual_price", "actual_direction", "reward_score",
                    "direction_correct", "error_pct", "verified_at"):
            assert key in outcome, f"Missing key: {key}"

    def test_correct_prediction_reward_positive(self) -> None:
        record = _make_record(predicted_direction=1, price_at_prediction=0.10)
        candle = _make_candle(close=0.102)  # price went up → predicted BUY = correct
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        outcome = s.update_prediction_outcome.call_args[0][1]
        assert outcome["reward_score"] > 0.0
        assert outcome["direction_correct"] is True

    def test_wrong_prediction_reward_negative(self) -> None:
        record = _make_record(predicted_direction=1, price_at_prediction=0.10)
        candle = _make_candle(close=0.098)  # price went DOWN → predicted BUY = wrong
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        outcome = s.update_prediction_outcome.call_args[0][1]
        assert outcome["reward_score"] < 0.0
        assert outcome["direction_correct"] is False


# ---------------------------------------------------------------------------
# MANDATORY — verifier does not run on future candles
# ---------------------------------------------------------------------------


class TestFutureCanleGuard:
    """MANDATORY: predictions whose target is too recent are skipped."""

    def test_target_in_near_future_is_skipped(self) -> None:
        """target_open_time < now_ms but within 1 interval of now → skip."""
        # The guard: target_open_time > now_ms - INTERVAL_MS
        near_target = _NOW_MS - _INTERVAL_MS // 2  # 30 minutes ago (candle not closed)
        record = _make_record(target_open_time=near_target)
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 0
        s.update_prediction_outcome.assert_not_called()

    def test_mature_target_is_processed(self) -> None:
        """target_open_time is well in the past → processed normally."""
        old_target = _NOW_MS - _INTERVAL_MS * 5  # 5 hours ago
        record = _make_record(target_open_time=old_target)
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1


# ---------------------------------------------------------------------------
# MANDATORY — interpolated candles are not pushed to replay buffer
# ---------------------------------------------------------------------------


class TestInterpolatedCanleSkip:
    """Interpolated candle prices must not enter the replay buffer."""

    def test_interpolated_candle_skips_verification(self) -> None:
        record = _make_record()
        candle = _make_candle(close=0.102, is_interpolated=True)
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 0
        s.update_prediction_outcome.assert_not_called()
        s.push_replay_buffer.assert_not_called()

    def test_non_interpolated_candle_is_processed(self) -> None:
        record = _make_record()
        candle = _make_candle(close=0.102, is_interpolated=False)
        s = _mock_storage(pending=[record], candle_df=candle)
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1

    def test_skip_interpolated_false_allows_interpolated(self) -> None:
        """When skip_interpolated=False, interpolated candles are processed."""
        record = _make_record()
        candle = _make_candle(close=0.102, is_interpolated=True)
        s = _mock_storage(pending=[record], candle_df=candle)
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        v = PredictionVerifier(storage=s, rl_cfg=_RL_CFG, replay_buffer=buf,
                               skip_interpolated=False)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1


# ---------------------------------------------------------------------------
# Missing OHLCV data
# ---------------------------------------------------------------------------


class TestMissingOHLCV:
    """Missing candle data causes a skip, not a crash."""

    def test_empty_candle_df_skips_gracefully(self) -> None:
        record = _make_record()
        s = _mock_storage(pending=[record], candle_df=pd.DataFrame())
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 0
        s.update_prediction_outcome.assert_not_called()

    def test_storage_error_on_ohlcv_skips(self) -> None:
        record = _make_record()
        s = _mock_storage(pending=[record])
        s.get_ohlcv.side_effect = RuntimeError("DB timeout")
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 0

    def test_update_outcome_returns_false_counts_as_skip(self) -> None:
        """If update_prediction_outcome returns False (row not found), not counted."""
        record = _make_record()
        s = _mock_storage(pending=[record], update_returns=False)
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 0


# ---------------------------------------------------------------------------
# Replay buffer is called after successful verification
# ---------------------------------------------------------------------------


class TestReplayBufferPush:
    """Replay buffer push is called exactly once per verified prediction."""

    def test_push_called_after_verification(self) -> None:
        record = _make_record()
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)
        s.push_replay_buffer.assert_called_once()

    def test_push_includes_horizon_and_regime(self) -> None:
        record = _make_record(horizon_label="MEDIUM", regime_label="TRENDING_BEAR")
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        v.run_verification(as_of_ts=_NOW_MS)

        pushed = s.push_replay_buffer.call_args[0][0]
        assert pushed["horizon_label"] == "MEDIUM"
        assert pushed["regime"] == "TRENDING_BEAR"

    def test_push_error_does_not_count_as_failure(self) -> None:
        """Push exception must not prevent the prediction from being counted as verified."""
        record = _make_record()
        s = _mock_storage(pending=[record])
        s.push_replay_buffer.side_effect = RuntimeError("buffer full")
        v = _make_verifier(s)
        # update_prediction_outcome still succeeded — count = 1
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1


# ---------------------------------------------------------------------------
# ReplayBuffer unit tests
# ---------------------------------------------------------------------------


class TestReplayBufferInit:
    """ReplayBuffer initialisation."""

    def test_initial_counts_all_zero_on_empty_storage(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        assert buf.total_count() == 0

    def test_all_horizons_present_in_counts(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        for h in ("SHORT", "MEDIUM", "LONG", "MACRO"):
            assert buf.count(h) == 0


class TestReplayBufferPushMethod:
    """ReplayBuffer.push() tests."""

    def test_push_increments_count(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        s.push_replay_buffer.return_value = True
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        buf.push(horizon="SHORT", regime="TRENDING_BULL", reward_score=1.2,
                 model_version="v1", created_at=_NOW_MS)
        assert buf.count("SHORT") == 1

    def test_invalid_horizon_raises(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        with pytest.raises(ValueError, match="Invalid horizon"):
            buf.push(horizon="WEEKLY", regime="TRENDING_BULL",
                     reward_score=1.0, model_version="v1", created_at=_NOW_MS)

    def test_invalid_regime_raises(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        with pytest.raises(ValueError, match="Invalid regime"):
            buf.push(horizon="SHORT", regime="UNKNOWN_REGIME",
                     reward_score=1.0, model_version="v1", created_at=_NOW_MS)

    def test_at_capacity_evicts_and_inserts(self) -> None:
        """At capacity: eviction fires, then insert succeeds (returns True)."""
        s = MagicMock()
        max_size = _RL_CFG.replay_buffer.max_size_per_horizon
        s.get_replay_sample.return_value = pd.DataFrame(
            [{"abs_reward": 0.5}] * max_size
        )
        s.push_replay_buffer.return_value = True
        s.get_replay_regime_counts.return_value = {"TRENDING_BULL": max_size}
        s.delete_oldest_non_protected_replay.return_value = True
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        result = buf.push(horizon="SHORT", regime="TRENDING_BULL",
                          reward_score=1.0, model_version="v1", created_at=_NOW_MS)
        # Eviction fires and insert succeeds
        assert result is True
        s.delete_oldest_non_protected_replay.assert_called_once()

    def test_fill_percentage_zero_on_empty(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        assert buf.fill_percentage("SHORT") == 0.0

    def test_is_ready_to_train_false_on_empty(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        assert buf.is_ready_to_train() is False


class TestReplayBufferSample:
    """ReplayBuffer.sample() priority weighting."""

    def _make_buf_with_pool(self, pool: pd.DataFrame) -> tuple[ReplayBuffer, MagicMock]:
        s = MagicMock()
        s.get_replay_sample.return_value = pool
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=0)
        return buf, s

    def test_sample_returns_dataframe(self) -> None:
        pool = pd.DataFrame([
            {"abs_reward": 0.5, "regime": "TRENDING_BULL", "horizon_label": "SHORT"}
            for _ in range(20)
        ])
        buf, _ = self._make_buf_with_pool(pool)
        result = buf.sample("SHORT", n=10, stratify=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 10

    def test_sample_empty_when_no_data(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        result = buf.sample("SHORT", n=10, stratify=False)
        assert result.empty

    def test_invalid_n_raises(self) -> None:
        s = MagicMock()
        s.get_replay_sample.return_value = pd.DataFrame()
        buf = ReplayBuffer(s, rl_cfg=_RL_CFG, seed=42)
        with pytest.raises(ValueError, match="n must be"):
            buf.sample("SHORT", n=0)

    def test_high_priority_rows_sampled_more_often(self) -> None:
        """High abs_reward rows should be overrepresented relative to their base rate.

        Pool: 20 high-priority (abs_reward=5.0) + 80 low-priority (abs_reward=0.1).
        Base rate = 20/100 = 20%.
        Weighted rate = (20*3)/(20*3+80*1) = 60/140 ≈ 42.9% — well above base rate.
        """
        n_high, n_low = 20, 80
        pool = pd.DataFrame({
            "abs_reward": [5.0] * n_high + [0.1] * n_low,
            "regime": ["TRENDING_BULL"] * (n_high + n_low),
            "horizon_label": ["SHORT"] * (n_high + n_low),
            "buffer_id": [str(uuid.uuid4()) for _ in range(n_high + n_low)],
        })
        buf, _ = self._make_buf_with_pool(pool)
        batch = buf.sample("SHORT", n=500, stratify=False)
        # High-priority rows (abs_reward >= 2.0) base rate = 20%;
        # with priority_oversample=3 they should appear at ~43%.
        high_prio_count = (batch["abs_reward"] >= _RL_CFG.replay_buffer.priority_threshold).sum()
        base_rate = n_high / (n_high + n_low)  # 0.20
        actual_rate = high_prio_count / len(batch)
        # Must exceed base rate by at least 5 percentage points
        assert actual_rate > base_rate + 0.05, (
            f"expected rate > {base_rate + 0.05:.2f} but got {actual_rate:.2f}"
        )


# ---------------------------------------------------------------------------
# CurriculumManager tests
# ---------------------------------------------------------------------------


class TestCurriculumManagerInit:
    """Curriculum manager initialisation."""

    def test_starts_at_stage_1_by_default(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        assert mgr.current_stage() == 1

    def test_active_horizons_stage_1(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        assert mgr.active_horizons() == ["SHORT"]

    def test_is_not_final_at_stage_1(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        assert mgr.is_final_stage() is False

    def test_custom_starting_stage(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG, starting_stage=3)
        assert mgr.current_stage() == 3


class TestCurriculumManagerAdvancement:
    """Stage advancement logic."""

    def _mgr(self, stage: int = 1) -> "CurriculumManager":
        from src.rl.curriculum import CurriculumManager
        return CurriculumManager(rl_cfg=_RL_CFG, starting_stage=stage)

    def test_advance_succeeds_when_all_criteria_met(self) -> None:
        mgr = self._mgr(1)
        advanced = mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        assert advanced is True
        assert mgr.current_stage() == 2

    def test_advance_fails_when_accuracy_below_threshold(self) -> None:
        mgr = self._mgr(1)
        advanced = mgr.try_advance(rolling_accuracy=0.50, mean_reward=0.35, n_days_covered=14)
        assert advanced is False
        assert mgr.current_stage() == 1

    def test_advance_fails_when_reward_below_threshold(self) -> None:
        mgr = self._mgr(1)
        advanced = mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.10, n_days_covered=14)
        assert advanced is False

    def test_advance_fails_when_days_below_threshold(self) -> None:
        mgr = self._mgr(1)
        advanced = mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=10)
        assert advanced is False

    def test_no_advance_at_final_stage(self) -> None:
        mgr = self._mgr(4)
        advanced = mgr.try_advance(rolling_accuracy=1.0, mean_reward=9.9, n_days_covered=999)
        assert advanced is False
        assert mgr.current_stage() == 4

    def test_stage_4_is_final(self) -> None:
        mgr = self._mgr(4)
        assert mgr.is_final_stage() is True

    def test_active_horizons_stage_2(self) -> None:
        mgr = self._mgr(2)
        assert "SHORT" in mgr.active_horizons()
        assert "MEDIUM" in mgr.active_horizons()

    def test_active_horizons_stage_4(self) -> None:
        mgr = self._mgr(4)
        horizons = mgr.active_horizons()
        for h in ("SHORT", "MEDIUM", "LONG", "MACRO"):
            assert h in horizons

    def test_advance_history_recorded(self) -> None:
        mgr = self._mgr(1)
        mgr.try_advance(rolling_accuracy=0.56, mean_reward=0.35, n_days_covered=14)
        history = mgr.advancement_history()
        assert len(history) == 1
        assert history[0]["from_stage"] == 1
        assert history[0]["to_stage"] == 2

    def test_multiple_advances_recorded(self) -> None:
        mgr = self._mgr(1)
        mgr.try_advance(rolling_accuracy=0.60, mean_reward=0.40, n_days_covered=14)
        mgr.try_advance(rolling_accuracy=0.55, mean_reward=0.25, n_days_covered=21)
        assert mgr.current_stage() == 3
        assert len(mgr.advancement_history()) == 2

    def test_stage_info_fields(self) -> None:
        mgr = self._mgr(1)
        info = mgr.stage_info()
        assert info.stage_number == 1
        assert info.label == "SHORT_ONLY"
        assert info.horizons == ["SHORT"]
        assert info.is_final is False

    def test_force_set_stage(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        mgr.force_set_stage(3)
        assert mgr.current_stage() == 3

    def test_force_set_invalid_stage_raises(self) -> None:
        from src.rl.curriculum import CurriculumManager
        mgr = CurriculumManager(rl_cfg=_RL_CFG)
        with pytest.raises(ValueError, match="force_set_stage"):
            mgr.force_set_stage(99)


# ---------------------------------------------------------------------------
# MANDATORY — immutability guard tests
# ---------------------------------------------------------------------------


class TestImmutabilityGuard:
    """MANDATORY: _assert_prediction_immutable() raises PredictionImmutabilityError
    when any immutable prediction field differs between the in-memory record and
    the stored record."""

    def _tampered(self, original: PredictionRecord, **overrides: object) -> PredictionRecord:
        """Return a copy of ``original`` with the given fields changed."""
        data = original.model_dump()
        data.update(overrides)
        return PredictionRecord.model_validate(data)

    def test_raises_when_predicted_price_modified(self) -> None:
        """MANDATORY: guard raises if price_at_prediction differs in DB."""
        record = _make_record(price_at_prediction=0.10)
        # Simulate DB storing a different price_at_prediction (tampered record)
        tampered = self._tampered(record, price_at_prediction=0.15)
        s = _mock_storage(pending=[record], prediction_by_id=tampered)
        v = _make_verifier(s)

        with pytest.raises(PredictionImmutabilityError) as exc_info:
            v.run_verification(as_of_ts=_NOW_MS)

        err = exc_info.value
        assert err.field == "price_at_prediction"
        assert err.original_value == pytest.approx(0.10)
        assert err.stored_value == pytest.approx(0.15)

    def test_raises_when_predicted_direction_modified(self) -> None:
        """Guard raises if predicted_direction differs in DB."""
        record = _make_record(predicted_direction=1)
        tampered = self._tampered(record, predicted_direction=-1)
        s = _mock_storage(pending=[record], prediction_by_id=tampered)
        v = _make_verifier(s)

        with pytest.raises(PredictionImmutabilityError) as exc_info:
            v.run_verification(as_of_ts=_NOW_MS)

        assert exc_info.value.field == "predicted_direction"

    def test_raises_when_horizon_label_modified(self) -> None:
        """Guard raises if horizon_label differs in DB."""
        record = _make_record(horizon_label="SHORT")
        tampered = self._tampered(record, horizon_label="LONG")
        s = _mock_storage(pending=[record], prediction_by_id=tampered)
        v = _make_verifier(s)

        with pytest.raises(PredictionImmutabilityError) as exc_info:
            v.run_verification(as_of_ts=_NOW_MS)

        assert exc_info.value.field == "horizon_label"

    def test_no_error_when_record_unchanged(self) -> None:
        """Guard passes when DB record matches in-memory record exactly."""
        record = _make_record()
        # Default mock returns the same record unchanged
        s = _mock_storage(pending=[record])
        v = _make_verifier(s)
        result = v.run_verification(as_of_ts=_NOW_MS)
        # Verification should succeed normally
        assert result == 1

    def test_passes_gracefully_when_db_returns_none(self) -> None:
        """Guard is skipped (not raised) when get_prediction_by_id returns None."""
        record = _make_record()
        s = _mock_storage(pending=[record], prediction_by_id=None)
        # Override so get_prediction_by_id returns None (row missing from DB)
        s.get_prediction_by_id.return_value = None
        v = _make_verifier(s)
        # Should proceed to verification (guard skipped with warning)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1

    def test_passes_gracefully_when_storage_raises(self) -> None:
        """Guard is skipped when get_prediction_by_id raises a storage error."""
        record = _make_record()
        s = _mock_storage(pending=[record])
        s.get_prediction_by_id.side_effect = RuntimeError("DB timeout")
        v = _make_verifier(s)
        # Should proceed to verification (guard skipped with warning)
        result = v.run_verification(as_of_ts=_NOW_MS)
        assert result == 1

    def test_error_attributes_populated(self) -> None:
        """PredictionImmutabilityError carries prediction_id, field, values."""
        record = _make_record(price_at_prediction=0.10)
        tampered = self._tampered(record, price_at_prediction=0.99)
        s = _mock_storage(pending=[record], prediction_by_id=tampered)
        v = _make_verifier(s)

        with pytest.raises(PredictionImmutabilityError) as exc_info:
            v.run_verification(as_of_ts=_NOW_MS)

        err = exc_info.value
        assert err.prediction_id == record.prediction_id
        assert isinstance(err.field, str)
        assert err.original_value is not None
        assert err.stored_value is not None
