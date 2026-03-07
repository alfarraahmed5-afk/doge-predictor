"""Unit tests for src/processing/storage.py (DogeStorage class).

All tests run against an in-memory SQLite database injected via the
``engine`` parameter of ``DogeStorage.__init__``. No PostgreSQL or
TimescaleDB is required.

Tests cover:
    - upsert_ohlcv inserts correctly and deduplicates on conflict
    - get_ohlcv respects era filter and time range
    - guard_raw_write raises PermissionError for paths inside data/raw/
    - insert_prediction → get_matured_unverified → update_prediction_outcome
      full lifecycle
    - upsert_funding_rates and get_funding_rates round-trip
    - upsert_regime_labels and get_regime_labels round-trip
    - push_replay_buffer and get_replay_sample round-trip
    - dispose() does not raise
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine

from src.config import Settings, settings as _global_settings
from src.processing.schemas import PredictionRecord
from src.processing.storage import DogeStorage, guard_raw_write


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def sqlite_engine(tmp_path: Path) -> sa.Engine:
    """Create a temporary file-based SQLite engine for one test function.

    Args:
        tmp_path: pytest-provided temporary directory.

    Yields:
        Configured SQLAlchemy ``Engine`` backed by a SQLite file.
    """
    db_file = tmp_path / "test_doge.db"
    engine = create_engine(f"sqlite:///{db_file}", echo=False)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def store(sqlite_engine: sa.Engine, tmp_path: Path) -> DogeStorage:
    """Return a ``DogeStorage`` backed by SQLite with all tables created.

    Args:
        sqlite_engine: SQLite engine fixture.
        tmp_path: pytest temporary directory (used as data_root for lock file).

    Yields:
        Fully initialised ``DogeStorage`` with schema in place.
    """
    # Build a Settings copy with tmp_path as the data root so the lock file
    # goes into the test's temp directory (not the real project data/).
    s = Settings.model_validate(_global_settings.model_dump())
    s.paths = s.paths.model_copy(
        update={
            "data_root": tmp_path,
            "raw_dir": tmp_path / "raw",
        }
    )
    storage = DogeStorage(s, engine=sqlite_engine)
    storage.create_tables()
    yield storage
    storage.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv_df(
    n: int = 5,
    symbol: str = "DOGEUSDT",
    era: str = "training",
    base_ms: int = 1_672_531_200_000,
    interval_ms: int = 3_600_000,
) -> pd.DataFrame:
    """Build a valid OHLCV DataFrame with *n* rows.

    Args:
        n: Number of rows.
        symbol: Symbol string (not a column in df; caller sets it on upsert).
        era: Era label.
        base_ms: First open_time (UTC epoch ms).
        interval_ms: Gap between consecutive open_times in ms.

    Returns:
        DataFrame ready for ``upsert_ohlcv``.
    """
    return pd.DataFrame(
        {
            "open_time": [base_ms + i * interval_ms for i in range(n)],
            "open": [0.079 + i * 0.001 for i in range(n)],
            "high": [0.082 + i * 0.001 for i in range(n)],
            "low": [0.077 + i * 0.001 for i in range(n)],
            "close": [0.080 + i * 0.001 for i in range(n)],
            "volume": [1_500_000.0 for _ in range(n)],
            "close_time": [
                base_ms + i * interval_ms + interval_ms - 1 for i in range(n)
            ],
            "quote_volume": [120_000.0 for _ in range(n)],
            "num_trades": [4_200 for _ in range(n)],
            "era": [era for _ in range(n)],
            "is_interpolated": [False for _ in range(n)],
        }
    )


def _make_prediction(
    base_ts: int = 1_672_531_200_000,
    horizon: str = "SHORT",
    candles: int = 4,
) -> PredictionRecord:
    """Build a valid PredictionRecord for testing.

    Args:
        base_ts: open_time / created_at timestamp in UTC epoch ms.
        horizon: Horizon label.
        candles: Horizon candle count.

    Returns:
        Validated ``PredictionRecord``.
    """
    return PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        created_at=base_ts,
        open_time=base_ts,
        symbol="DOGEUSDT",
        horizon_label=horizon,
        horizon_candles=candles,
        target_open_time=base_ts + candles * 3_600_000,
        price_at_prediction=0.080,
        predicted_direction=1,
        confidence_score=0.65,
        lstm_prob=0.65,
        xgb_prob=0.60,
        regime_label="TRENDING_BULL",
        model_version="v1.0.0",
    )


# ---------------------------------------------------------------------------
# upsert_ohlcv / get_ohlcv tests
# ---------------------------------------------------------------------------


class TestOHLCVStorage:
    """Tests for upsert_ohlcv and get_ohlcv."""

    def test_upsert_inserts_new_rows(self, store: DogeStorage) -> None:
        """upsert_ohlcv should insert N rows into the table."""
        df = _ohlcv_df(n=5)
        n = store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")
        # SQLite rowcount may be -1 for multi-row inserts; just verify readable
        result = store.get_ohlcv("DOGEUSDT", "1h", 0, 9_999_999_999_999)
        assert len(result) == 5

    def test_upsert_deduplicates_on_conflict(self, store: DogeStorage) -> None:
        """Inserting the same rows twice must not create duplicates."""
        df = _ohlcv_df(n=3)
        store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")
        store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")
        result = store.get_ohlcv("DOGEUSDT", "1h", 0, 9_999_999_999_999)
        assert len(result) == 3

    def test_upsert_updates_on_conflict(self, store: DogeStorage) -> None:
        """An upsert with changed values should overwrite the existing row."""
        base_ms = 1_672_531_200_000
        df_original = _ohlcv_df(n=1, base_ms=base_ms)
        store.upsert_ohlcv(df_original, symbol="DOGEUSDT", interval="1h")

        df_updated = df_original.copy()
        df_updated["close"] = 0.999  # changed close price
        store.upsert_ohlcv(df_updated, symbol="DOGEUSDT", interval="1h")

        result = store.get_ohlcv("DOGEUSDT", "1h", 0, 9_999_999_999_999)
        assert len(result) == 1
        assert float(result.iloc[0]["close"]) == pytest.approx(0.999)

    def test_get_ohlcv_respects_era_filter(self, store: DogeStorage) -> None:
        """get_ohlcv should return only rows matching the requested era."""
        base_ms = 1_672_531_200_000
        interval_ms = 3_600_000

        df_training = _ohlcv_df(n=3, era="training", base_ms=base_ms)
        df_context = _ohlcv_df(
            n=3,
            era="context",
            base_ms=base_ms + 3 * interval_ms,  # non-overlapping times
        )
        store.upsert_ohlcv(df_training, symbol="DOGEUSDT", interval="1h")
        store.upsert_ohlcv(df_context, symbol="DOGEUSDT", interval="1h")

        training_result = store.get_ohlcv(
            "DOGEUSDT", "1h", 0, 9_999_999_999_999, era="training"
        )
        context_result = store.get_ohlcv(
            "DOGEUSDT", "1h", 0, 9_999_999_999_999, era="context"
        )
        all_result = store.get_ohlcv("DOGEUSDT", "1h", 0, 9_999_999_999_999)

        assert len(training_result) == 3
        assert len(context_result) == 3
        assert len(all_result) == 6
        assert set(training_result["era"].unique()) == {"training"}
        assert set(context_result["era"].unique()) == {"context"}

    def test_get_ohlcv_respects_time_range(self, store: DogeStorage) -> None:
        """get_ohlcv should respect start_ms (inclusive) and end_ms (exclusive)."""
        base_ms = 1_672_531_200_000
        interval_ms = 3_600_000
        df = _ohlcv_df(n=10, base_ms=base_ms)
        store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")

        # Request only rows 2–5 (open_time in [base + 2h, base + 5h))
        start = base_ms + 2 * interval_ms
        end = base_ms + 5 * interval_ms
        result = store.get_ohlcv("DOGEUSDT", "1h", start, end)
        assert len(result) == 3
        assert int(result.iloc[0]["open_time"]) == start

    def test_get_ohlcv_returns_empty_for_no_match(self, store: DogeStorage) -> None:
        """get_ohlcv returns an empty DataFrame when no rows match."""
        result = store.get_ohlcv("DOGEUSDT", "1h", 0, 1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_upsert_ohlcv_invalid_interval_raises(self, store: DogeStorage) -> None:
        """upsert_ohlcv should raise ValueError for an unknown interval."""
        df = _ohlcv_df()
        with pytest.raises(ValueError, match="interval"):
            store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="5m")

    def test_upsert_multiple_symbols(self, store: DogeStorage) -> None:
        """upsert_ohlcv must handle multiple symbols in the same table."""
        df = _ohlcv_df(n=3)
        store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")
        store.upsert_ohlcv(df, symbol="BTCUSDT", interval="1h")
        doge = store.get_ohlcv("DOGEUSDT", "1h", 0, 9_999_999_999_999)
        btc = store.get_ohlcv("BTCUSDT", "1h", 0, 9_999_999_999_999)
        assert len(doge) == 3
        assert len(btc) == 3


# ---------------------------------------------------------------------------
# guard_raw_write tests
# ---------------------------------------------------------------------------


class TestGuardRawWrite:
    """Tests for the data/raw/ immutability guard."""

    def test_raw_write_raises_permission_error(self) -> None:
        """guard_raw_write must raise PermissionError for paths in data/raw/."""
        raw_path = (
            Path(__file__).parent.parent.parent
            / "data"
            / "raw"
            / "dogeusdt_1h"
            / "2023-01.parquet"
        )
        with pytest.raises(PermissionError, match="data/raw"):
            guard_raw_write(raw_path)

    def test_raw_write_class_method_raises(self, store: DogeStorage) -> None:
        """DogeStorage.guard_raw_write class alias also raises PermissionError."""
        raw_path = (
            Path(__file__).parent.parent.parent / "data" / "raw" / "test.parquet"
        )
        with pytest.raises(PermissionError):
            DogeStorage.guard_raw_write(raw_path)

    def test_safe_path_does_not_raise(self) -> None:
        """guard_raw_write must not raise for paths outside data/raw/."""
        safe_path = Path(__file__).parent.parent.parent / "data" / "processed" / "foo.parquet"
        # Should not raise
        guard_raw_write(safe_path)

    def test_processed_dir_does_not_raise(self) -> None:
        """A path inside data/processed/ must pass the guard silently."""
        processed = (
            Path(__file__).parent.parent.parent
            / "data"
            / "processed"
            / "dogeusdt_1h_clean.parquet"
        )
        guard_raw_write(processed)  # must not raise


# ---------------------------------------------------------------------------
# Funding rates tests
# ---------------------------------------------------------------------------


class TestFundingRatesStorage:
    """Tests for upsert_funding_rates and get_funding_rates."""

    def _funding_df(self, n: int = 5) -> pd.DataFrame:
        base_ms = 1_672_531_200_000
        interval_ms = 28_800_000  # 8h
        return pd.DataFrame(
            {
                "timestamp_ms": [base_ms + i * interval_ms for i in range(n)],
                "symbol": ["DOGEUSDT" for _ in range(n)],
                "funding_rate": [0.0001 * (i + 1) for i in range(n)],
                "mark_price": [0.080 + i * 0.001 for i in range(n)],
            }
        )

    def test_upsert_and_get_round_trip(self, store: DogeStorage) -> None:
        """Upserted funding rates should be fetchable in the same range."""
        df = self._funding_df(n=5)
        store.upsert_funding_rates(df)
        result = store.get_funding_rates(0, 9_999_999_999_999)
        assert len(result) == 5

    def test_deduplication(self, store: DogeStorage) -> None:
        """Inserting the same funding rates twice yields no duplicates."""
        df = self._funding_df(n=3)
        store.upsert_funding_rates(df)
        store.upsert_funding_rates(df)
        result = store.get_funding_rates(0, 9_999_999_999_999)
        assert len(result) == 3

    def test_empty_range_returns_empty_df(self, store: DogeStorage) -> None:
        """get_funding_rates returns an empty DataFrame if no rows match."""
        result = store.get_funding_rates(0, 1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Regime labels tests
# ---------------------------------------------------------------------------


class TestRegimeLabelsStorage:
    """Tests for upsert_regime_labels and get_regime_labels."""

    def _regime_df(self, n: int = 5) -> pd.DataFrame:
        base_ms = 1_672_531_200_000
        interval_ms = 3_600_000
        return pd.DataFrame(
            {
                "open_time": [base_ms + i * interval_ms for i in range(n)],
                "symbol": ["DOGEUSDT" for _ in range(n)],
                "regime": ["TRENDING_BULL" for _ in range(n)],
                "btc_corr_24h": [0.85 for _ in range(n)],
                "bb_width": [0.06 for _ in range(n)],
                "atr_norm": [0.004 for _ in range(n)],
            }
        )

    def test_upsert_and_get_round_trip(self, store: DogeStorage) -> None:
        """Regime labels should survive an upsert → get round trip."""
        df = self._regime_df(n=5)
        store.upsert_regime_labels(df)
        result = store.get_regime_labels(0, 9_999_999_999_999)
        assert len(result) == 5
        assert result.iloc[0]["regime"] == "TRENDING_BULL"

    def test_deduplication(self, store: DogeStorage) -> None:
        """Double-upserting the same regime rows must not create duplicates."""
        df = self._regime_df(n=3)
        store.upsert_regime_labels(df)
        store.upsert_regime_labels(df)
        result = store.get_regime_labels(0, 9_999_999_999_999)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Prediction lifecycle tests
# ---------------------------------------------------------------------------


class TestPredictionLifecycle:
    """Tests for insert_prediction, get_matured_unverified, update_prediction_outcome."""

    def test_full_lifecycle(self, store: DogeStorage) -> None:
        """Full cycle: insert → check not matured → mature → verify → done."""
        base_ts = 1_672_531_200_000
        record = _make_prediction(base_ts=base_ts, horizon="SHORT", candles=4)
        target_ts = record.target_open_time

        # Insert
        pred_id = store.insert_prediction(record)
        assert pred_id == record.prediction_id

        # Before target time — should NOT appear in matured list
        unverified_before = store.get_matured_unverified(base_ts - 1)
        assert len(unverified_before) == 0

        # After target time — should appear
        unverified_after = store.get_matured_unverified(target_ts)
        assert len(unverified_after) == 1
        assert unverified_after[0].prediction_id == pred_id
        assert unverified_after[0].verified_at is None

        # Update outcome
        outcome = {
            "actual_price": 0.085,
            "actual_direction": 1,
            "reward_score": 1.2,
            "direction_correct": True,
            "error_pct": 0.0625,
            "verified_at": target_ts + 3_600_000,
        }
        success = store.update_prediction_outcome(pred_id, outcome)
        assert success is True

        # Should no longer appear in matured-unverified
        unverified_final = store.get_matured_unverified(target_ts)
        assert len(unverified_final) == 0

    def test_insert_idempotent(self, store: DogeStorage) -> None:
        """Inserting the same prediction_id twice must not raise or duplicate."""
        record = _make_prediction()
        store.insert_prediction(record)
        store.insert_prediction(record)  # duplicate — ON CONFLICT DO NOTHING
        result = store.get_matured_unverified(9_999_999_999_999)
        matching = [r for r in result if r.prediction_id == record.prediction_id]
        assert len(matching) == 1

    def test_update_nonexistent_prediction_returns_false(
        self, store: DogeStorage
    ) -> None:
        """update_prediction_outcome returns False if prediction_id not found."""
        success = store.update_prediction_outcome(
            "nonexistent-uuid",
            {"actual_price": 0.080, "verified_at": 1_672_531_200_000},
        )
        assert success is False

    def test_multiple_horizons_matured_correctly(
        self, store: DogeStorage
    ) -> None:
        """Predictions across horizons mature independently."""
        base_ts = 1_672_531_200_000
        short_rec = _make_prediction(base_ts, "SHORT", 4)
        medium_rec = _make_prediction(base_ts, "MEDIUM", 24)

        store.insert_prediction(short_rec)
        store.insert_prediction(medium_rec)

        # At base_ts + 4h: SHORT is matured, MEDIUM is not
        short_target = short_rec.target_open_time
        matured = store.get_matured_unverified(short_target)
        matured_ids = {r.prediction_id for r in matured}
        assert short_rec.prediction_id in matured_ids
        assert medium_rec.prediction_id not in matured_ids


# ---------------------------------------------------------------------------
# Replay buffer tests
# ---------------------------------------------------------------------------


class TestReplayBufferStorage:
    """Tests for push_replay_buffer and get_replay_sample."""

    def _make_buffer_record(self, horizon: str = "SHORT") -> dict[str, Any]:
        """Build a minimal replay buffer record dict.

        Args:
            horizon: Horizon label for the record.

        Returns:
            Dict ready for ``push_replay_buffer``.
        """
        return {
            "buffer_id": str(uuid.uuid4()),
            "horizon_label": horizon,
            "regime": "TRENDING_BULL",
            "feature_vector": pickle.dumps([0.1, 0.2, 0.3]),
            "predicted_price": 0.080,
            "actual_price": 0.085,
            "reward_score": 1.2,
            "model_version": "v1.0.0",
            "created_at": 1_672_531_200_000,
        }

    def test_push_and_sample_round_trip(self, store: DogeStorage) -> None:
        """Pushed records should be returned by get_replay_sample."""
        for _ in range(5):
            store.push_replay_buffer(self._make_buffer_record("SHORT"))
        result = store.get_replay_sample("SHORT", 10)
        assert len(result) == 5
        assert "abs_reward" in result.columns

    def test_abs_reward_computed(self, store: DogeStorage) -> None:
        """abs_reward must equal abs(reward_score) for each pushed record."""
        record = self._make_buffer_record()
        record["reward_score"] = -1.5
        store.push_replay_buffer(record)
        result = store.get_replay_sample("SHORT", 1)
        assert len(result) == 1
        assert float(result.iloc[0]["abs_reward"]) == pytest.approx(1.5)

    def test_sample_filtered_by_horizon(self, store: DogeStorage) -> None:
        """get_replay_sample must only return rows for the requested horizon."""
        store.push_replay_buffer(self._make_buffer_record("SHORT"))
        store.push_replay_buffer(self._make_buffer_record("SHORT"))
        store.push_replay_buffer(self._make_buffer_record("MEDIUM"))

        short_sample = store.get_replay_sample("SHORT", 100)
        medium_sample = store.get_replay_sample("MEDIUM", 100)
        assert len(short_sample) == 2
        assert len(medium_sample) == 1

    def test_sample_empty_when_no_rows(self, store: DogeStorage) -> None:
        """get_replay_sample returns an empty DataFrame when buffer is empty."""
        result = store.get_replay_sample("MACRO", 10)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_push_idempotent(self, store: DogeStorage) -> None:
        """Pushing the same buffer_id twice must not duplicate the row."""
        rec = self._make_buffer_record()
        store.push_replay_buffer(rec)
        store.push_replay_buffer(rec)  # ON CONFLICT DO NOTHING
        result = store.get_replay_sample("SHORT", 100)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Dispose
# ---------------------------------------------------------------------------


class TestDispose:
    """Tests for DogeStorage.dispose."""

    def test_dispose_does_not_raise(self, store: DogeStorage) -> None:
        """dispose() must complete without raising any exception."""
        store.dispose()  # second dispose called automatically in fixture teardown
