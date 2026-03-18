"""Unit tests for src/rl/replay_buffer.py — ReplayBuffer.

MANDATORY tests:
  1. Regime minimum quota respected during eviction — pushing 10001 records
     with 50 DECOUPLED preserves all 50 DECOUPLED rows.
  2. Prioritised sampling over-represents |reward| > priority_threshold.
  3. Deserialised feature_vector matches the original numpy array exactly.

Additional tests cover: get_regime_counts, checkpoint, get_prioritised_sample,
push validation, and the serialize / deserialise helpers.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import numpy as np
import pandas as pd
import pytest

from src.config import RLConfig, _load_yaml
from src.rl.replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_RL_CFG: RLConfig = RLConfig(**_load_yaml("rl_config.yaml"))
_NOW_MS: int = 1_700_000_000_000


def _make_storage(
    regime_counts: dict[str, int] | None = None,
    pool: pd.DataFrame | None = None,
    delete_returns: bool = True,
) -> MagicMock:
    """Build a minimal storage mock for ReplayBuffer tests."""
    s = MagicMock()
    s.get_replay_sample.return_value = pool if pool is not None else pd.DataFrame()
    s.push_replay_buffer.return_value = True
    s.get_replay_regime_counts.return_value = regime_counts or {}
    s.delete_oldest_non_protected_replay.return_value = delete_returns
    return s


def _make_buffer(
    storage: MagicMock | None = None,
    seed: int = 42,
) -> ReplayBuffer:
    """Build a ReplayBuffer backed by a mock storage."""
    s = storage or _make_storage()
    return ReplayBuffer(s, rl_cfg=_RL_CFG, seed=seed)


def _row(
    horizon: str = "SHORT",
    regime: str = "TRENDING_BULL",
    abs_reward: float = 0.5,
    buffer_id: str | None = None,
) -> dict[str, Any]:
    return {
        "buffer_id": buffer_id or str(uuid.uuid4()),
        "horizon_label": horizon,
        "regime": regime,
        "abs_reward": abs_reward,
        "reward_score": abs_reward,
        "model_version": "v1",
        "created_at": _NOW_MS,
    }


# ---------------------------------------------------------------------------
# MANDATORY TEST 1 — Regime minimum quota respected during eviction
# ---------------------------------------------------------------------------


class TestEvictionMinPerRegimeQuota:
    """MANDATORY: pushing past max_size evicts from non-protected regimes.

    Scenario: buffer has max_size_per_horizon=10000 rows.  Of those, 50 belong
    to DECOUPLED and 9950 to TRENDING_BULL.  min_per_regime=100, so DECOUPLED
    (count=50 < 100) is protected.  On the 10001st push, delete_oldest_*
    must be called with DECOUPLED in the protected set.
    """

    def test_protected_regime_not_evicted(self) -> None:
        """MANDATORY: all 50 DECOUPLED rows are preserved after eviction."""
        max_size = _RL_CFG.replay_buffer.max_size_per_horizon  # 10000
        min_per_regime = _RL_CFG.replay_buffer.min_per_regime  # 100

        # Simulate the buffer at full capacity
        pool = pd.DataFrame([_row() for _ in range(max_size)])
        s = _make_storage(pool=pool)
        # Regime distribution: 9950 TRENDING_BULL + 50 DECOUPLED
        s.get_replay_regime_counts.return_value = {
            "TRENDING_BULL": max_size - 50,
            "DECOUPLED": 50,
        }
        buf = _make_buffer(storage=s)
        assert buf.count("SHORT") == max_size

        # Push the (max_size + 1)th record
        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=1.0,
            model_version="v1",
            created_at=_NOW_MS + 1,
        )

        # Eviction must have been triggered
        s.delete_oldest_non_protected_replay.assert_called_once()
        evict_call = s.delete_oldest_non_protected_replay.call_args
        horizon_arg = evict_call[0][0] if evict_call[0] else evict_call[1].get("horizon")
        protected_arg = evict_call[0][1] if len(evict_call[0]) > 1 else evict_call[1].get("protected_regimes")

        assert horizon_arg == "SHORT"
        # DECOUPLED (count=50) is below min_per_regime (100) → must be protected
        assert "DECOUPLED" in protected_arg, (
            f"Expected DECOUPLED in protected set; got {protected_arg}"
        )

    def test_dominant_regime_not_protected(self) -> None:
        """A regime with count > min_per_regime is NOT in the protected set."""
        max_size = _RL_CFG.replay_buffer.max_size_per_horizon
        pool = pd.DataFrame([_row() for _ in range(max_size)])
        s = _make_storage(pool=pool)
        # TRENDING_BULL has 5000 rows — well above min_per_regime=100
        s.get_replay_regime_counts.return_value = {
            "TRENDING_BULL": 5000,
            "TRENDING_BEAR": 5000,
        }
        buf = _make_buffer(storage=s)

        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=1.0,
            model_version="v1",
            created_at=_NOW_MS + 1,
        )

        evict_call = s.delete_oldest_non_protected_replay.call_args
        protected_arg = evict_call[0][1] if len(evict_call[0]) > 1 else evict_call[1].get("protected_regimes")
        # Neither regime should be protected (both above min_per_regime)
        assert "TRENDING_BULL" not in protected_arg
        assert "TRENDING_BEAR" not in protected_arg

    def test_eviction_decrements_count(self) -> None:
        """After a successful eviction + insert, net count stays at max_size."""
        max_size = _RL_CFG.replay_buffer.max_size_per_horizon
        pool = pd.DataFrame([_row() for _ in range(max_size)])
        s = _make_storage(pool=pool)
        s.get_replay_regime_counts.return_value = {"TRENDING_BULL": max_size}
        buf = _make_buffer(storage=s)
        assert buf.count("SHORT") == max_size

        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=0.5,
            model_version="v1",
            created_at=_NOW_MS + 1,
        )

        # Evict decrements by 1, insert increments by 1 → net unchanged
        assert buf.count("SHORT") == max_size

    def test_no_eviction_below_capacity(self) -> None:
        """Eviction is NOT triggered when below capacity."""
        s = _make_storage(pool=pd.DataFrame())  # empty → count = 0
        buf = _make_buffer(storage=s)

        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=1.0,
            model_version="v1",
            created_at=_NOW_MS,
        )

        s.delete_oldest_non_protected_replay.assert_not_called()


# ---------------------------------------------------------------------------
# MANDATORY TEST 2 — Prioritised sampling over-represents |reward| > threshold
# ---------------------------------------------------------------------------


class TestPrioritisedSampling:
    """MANDATORY: get_prioritised_sample gives high-reward rows 3× representation."""

    def test_high_reward_rows_overrepresented(self) -> None:
        """MANDATORY: rows with abs_reward > priority_threshold appear at > base rate.

        Pool: 20 high-priority (abs_reward=5.0) + 80 low-priority (abs_reward=0.1).
        Base rate = 20/100 = 20%.
        With 3× duplication: (20*3)/(20*3+80*1) = 60/140 ≈ 42.9%.
        Actual rate must be > base_rate + 5 percentage points.
        """
        threshold = _RL_CFG.replay_buffer.priority_threshold  # 2.0
        n_high, n_low = 20, 80
        pool = pd.DataFrame(
            [_row(abs_reward=5.0) for _ in range(n_high)]
            + [_row(abs_reward=0.1) for _ in range(n_low)]
        )
        s = _make_storage(pool=pool)
        buf = _make_buffer(storage=s, seed=0)

        batch = buf.get_prioritised_sample("SHORT", n=600)

        # Count rows that came from the high-priority pool
        high_count = (batch["abs_reward"] >= threshold).sum()
        base_rate = n_high / (n_high + n_low)  # 0.20
        actual_rate = high_count / len(batch)

        assert actual_rate > base_rate + 0.05, (
            f"Expected high-priority rate > {base_rate + 0.05:.2f}; got {actual_rate:.2f}"
        )

    def test_returns_at_most_n_rows(self) -> None:
        pool = pd.DataFrame([_row() for _ in range(50)])
        s = _make_storage(pool=pool)
        buf = _make_buffer(storage=s)
        batch = buf.get_prioritised_sample("SHORT", n=10)
        assert len(batch) <= 10

    def test_empty_pool_returns_empty_df(self) -> None:
        s = _make_storage(pool=pd.DataFrame())
        buf = _make_buffer(storage=s)
        result = buf.get_prioritised_sample("SHORT", n=10)
        assert result.empty

    def test_invalid_n_raises(self) -> None:
        buf = _make_buffer()
        with pytest.raises(ValueError, match="n must be"):
            buf.get_prioritised_sample("SHORT", n=0)

    def test_invalid_horizon_raises(self) -> None:
        buf = _make_buffer()
        with pytest.raises(ValueError):
            buf.get_prioritised_sample("DECADE", n=10)

    def test_all_low_priority_sampled_uniformly(self) -> None:
        """When no high-priority rows exist, sampling is uniformly random."""
        pool = pd.DataFrame([_row(abs_reward=0.1) for _ in range(100)])
        s = _make_storage(pool=pool)
        buf = _make_buffer(storage=s, seed=7)
        batch = buf.get_prioritised_sample("SHORT", n=50)
        assert not batch.empty
        # All abs_reward values must be low-priority
        assert (batch["abs_reward"] < _RL_CFG.replay_buffer.priority_threshold).all()


# ---------------------------------------------------------------------------
# MANDATORY TEST 3 — Feature vector round-trip serialisation
# ---------------------------------------------------------------------------


class TestFeatureVectorSerialization:
    """MANDATORY: feature_vector bytes deserialise to an identical numpy array."""

    def test_roundtrip_exact_equality(self) -> None:
        """MANDATORY: deserialise(serialise(arr)) == arr for every element."""
        rng = np.random.default_rng(99)
        original = rng.standard_normal(84).astype(np.float64)

        serialised = ReplayBuffer.serialize_feature_vector(original)
        recovered = ReplayBuffer.deserialize_feature_vector(serialised)

        assert isinstance(serialised, bytes), "serialized form must be bytes"
        assert isinstance(recovered, np.ndarray), "deserialized form must be ndarray"
        np.testing.assert_array_equal(
            original,
            recovered,
            err_msg="Deserialized feature_vector does not match original",
        )

    def test_serialized_length_is_8_bytes_per_element(self) -> None:
        """float64 = 8 bytes per element."""
        arr = np.ones(84, dtype=np.float64)
        serialised = ReplayBuffer.serialize_feature_vector(arr)
        assert len(serialised) == 84 * 8

    def test_float32_input_cast_to_float64(self) -> None:
        """Input arrays of other dtypes are cast to float64 before serialization."""
        arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        serialised = ReplayBuffer.serialize_feature_vector(arr_f32)
        recovered = ReplayBuffer.deserialize_feature_vector(serialised)
        assert recovered.dtype == np.float64
        np.testing.assert_allclose(recovered, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_push_accepts_ndarray_and_stores_as_bytes(self) -> None:
        """push() auto-serializes numpy arrays via serialize_feature_vector."""
        s = _make_storage()
        buf = _make_buffer(storage=s)
        arr = np.arange(10, dtype=np.float64)

        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=1.0,
            model_version="v1",
            created_at=_NOW_MS,
            feature_vector=arr,
        )

        pushed = s.push_replay_buffer.call_args[0][0]
        assert isinstance(pushed["feature_vector"], bytes), (
            "push() must serialise ndarray to bytes before storing"
        )
        recovered = ReplayBuffer.deserialize_feature_vector(pushed["feature_vector"])
        np.testing.assert_array_equal(arr, recovered)

    def test_get_prioritised_sample_deserializes_feature_vector(self) -> None:
        """get_prioritised_sample() deserialises feature_vector bytes in the result."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        serialised = ReplayBuffer.serialize_feature_vector(arr)

        pool = pd.DataFrame([{
            **_row(abs_reward=3.0),
            "feature_vector": serialised,
        }])
        s = _make_storage(pool=pool)
        buf = _make_buffer(storage=s)

        batch = buf.get_prioritised_sample("SHORT", n=1)
        assert not batch.empty
        fv = batch.iloc[0]["feature_vector"]
        assert isinstance(fv, np.ndarray), (
            "feature_vector in sample result must be deserialized to ndarray"
        )
        np.testing.assert_array_equal(arr, fv)

    def test_none_feature_vector_not_stored(self) -> None:
        """When feature_vector is None, the key must not appear in the pushed record."""
        s = _make_storage()
        buf = _make_buffer(storage=s)
        buf.push(
            horizon="SHORT",
            regime="TRENDING_BULL",
            reward_score=1.0,
            model_version="v1",
            created_at=_NOW_MS,
            feature_vector=None,
        )
        pushed = s.push_replay_buffer.call_args[0][0]
        assert "feature_vector" not in pushed


# ---------------------------------------------------------------------------
# get_regime_counts
# ---------------------------------------------------------------------------


class TestGetRegimeCounts:
    """get_regime_counts() returns per-regime row counts from the database."""

    def test_returns_dict_from_storage(self) -> None:
        s = _make_storage(regime_counts={"TRENDING_BULL": 42, "DECOUPLED": 8})
        buf = _make_buffer(storage=s)
        counts = buf.get_regime_counts("SHORT")
        assert counts == {"TRENDING_BULL": 42, "DECOUPLED": 8}

    def test_empty_buffer_returns_empty_dict(self) -> None:
        s = _make_storage(regime_counts={})
        buf = _make_buffer(storage=s)
        assert buf.get_regime_counts("SHORT") == {}

    def test_invalid_horizon_raises(self) -> None:
        buf = _make_buffer()
        with pytest.raises(ValueError):
            buf.get_regime_counts("WEEKLY")

    def test_storage_error_returns_empty_dict(self) -> None:
        s = _make_storage()
        s.get_replay_regime_counts.side_effect = RuntimeError("DB down")
        buf = _make_buffer(storage=s)
        result = buf.get_regime_counts("SHORT")
        assert result == {}


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """checkpoint() writes a JSON file to the given path."""

    def test_creates_json_file(self, tmp_path: Path) -> None:
        buf = _make_buffer()
        buf.checkpoint(tmp_path)
        files = list(tmp_path.glob("replay_buffer_*.json"))
        assert len(files) == 1

    def test_json_has_required_keys(self, tmp_path: Path) -> None:
        buf = _make_buffer()
        buf.checkpoint(tmp_path)
        files = list(tmp_path.glob("replay_buffer_*.json"))
        data = json.loads(files[0].read_text())
        for key in ("snapshot_ts_ms", "counts", "max_size_per_horizon",
                    "priority_threshold", "priority_oversample", "min_per_regime"):
            assert key in data, f"Missing key: {key}"

    def test_counts_match_buffer_state(self, tmp_path: Path) -> None:
        s = _make_storage()
        buf = _make_buffer(storage=s)
        # Manually set a count
        buf._counts["SHORT"] = 99
        buf.checkpoint(tmp_path)
        files = list(tmp_path.glob("replay_buffer_*.json"))
        data = json.loads(files[0].read_text())
        assert data["counts"]["SHORT"] == 99

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        buf = _make_buffer()
        buf.checkpoint(nested)
        assert nested.is_dir()
        assert any(nested.glob("replay_buffer_*.json"))
