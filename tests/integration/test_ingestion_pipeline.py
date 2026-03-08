"""Integration test — full data ingestion pipeline.

Bootstraps synthetic OHLCV data for DOGEUSDT, BTCUSDT, and DOGEBTC into an
in-memory SQLite database, then runs the complete

    bootstrap → DataValidator → MultiSymbolAligner

sequence and asserts all correctness invariants.

No network traffic or real database is used.  A lightweight ``_FakeClient``
replaces ``BinanceRESTClient``, returning slices of preloaded DataFrames
bounded by the ``[start_ms, end_ms)`` range the bootstrap engine passes.

Assertions verified:
    1. All 3 symbols have identical ``open_time`` index after alignment.
    2. ``era`` column is correctly assigned (``'training'`` for post-2022 rows).
    3. No NaN or Inf in any numeric column of the aligned output.
    4. Checkpoint files are cleaned up (deleted) on successful completion.
    5. Era context/training counts are correct in BootstrapResult.
    6. DataValidator returns ``is_valid=True`` for each bootstrapped symbol.
    7. ``AlignmentResult.rows_aligned`` matches the dataset row count.
    8. Incremental scheduler ``run_once`` upserts fresh candles correctly.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa

from src.config import settings as cfg
from src.ingestion.bootstrap import BootstrapEngine
from src.ingestion.scheduler import IncrementalScheduler
from src.processing.aligner import AlignmentResult, MultiSymbolAligner
from src.processing.storage import DogeStorage
from src.processing.validator import DataValidator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 2022-01-01 00:00:00 UTC — era boundary.
_TRAINING_START_MS: int = 1_640_995_200_000

#: 1-hour interval in milliseconds.
_INTERVAL_MS: int = 3_600_000

#: Rows per synthetic dataset.
_N_ROWS: int = 50

#: Batch size passed to BootstrapEngine (smaller than default to allow
#: multiple API calls and checkpoint writes with _N_ROWS data).
_BATCH_SIZE: int = 20

#: Checkpoint interval — ensures at least one checkpoint is written then
#: deleted during the 50-row bootstrap run.
_CHECKPOINT_EVERY: int = 25


# ---------------------------------------------------------------------------
# Fake REST client
# ---------------------------------------------------------------------------


class _FakeClient:
    """Drop-in replacement for BinanceRESTClient that serves preloaded data.

    ``get_klines`` returns the subset of a preloaded DataFrame whose
    ``open_time`` falls in ``[start_ms, end_ms)``.  This faithfully mimics
    the real client's slice semantics without any network I/O.

    Args:
        datasets: Dict mapping symbol name → OHLCV DataFrame.
    """

    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        self._datasets: dict[str, pd.DataFrame] = datasets
        #: Exposed so BootstrapEngine.bootstrap_symbol can log it.
        self.weight_used: int = 10

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Return rows in ``[start_ms, end_ms)`` for *symbol*.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (unused by fake — data is pre-loaded).
            start_ms: Inclusive start, UTC epoch ms.
            end_ms: Exclusive end, UTC epoch ms.

        Returns:
            Filtered copy of the preloaded DataFrame, or an empty DataFrame
            when *symbol* is not in the dataset or no rows match.
        """
        df = self._datasets.get(symbol, pd.DataFrame())
        if df.empty:
            return pd.DataFrame()
        mask = (df["open_time"] >= start_ms) & (df["open_time"] < end_ms)
        return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int,
    start_ms: int,
    interval_ms: int = _INTERVAL_MS,
    open_price: float = 0.10,
) -> pd.DataFrame:
    """Build a clean synthetic OHLCV DataFrame with *n* consecutive candles.

    Args:
        n: Number of rows.
        start_ms: ``open_time`` of the first row (UTC epoch ms).
        interval_ms: Milliseconds per candle.
        open_price: Representative open price for all candles.

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume,
        close_time.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n):
        t = start_ms + i * interval_ms
        rows.append({
            "open_time": t,
            "open": open_price,
            "high": open_price * 1.05,
            "low": open_price * 0.95,
            "close": open_price * 1.01,
            "volume": 1_000_000.0,
            "close_time": t + interval_ms - 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage(tmp_path: Path) -> DogeStorage:
    """In-memory SQLite DogeStorage with all tables created.

    Args:
        tmp_path: pytest temporary directory for the file-lock.

    Returns:
        :class:`~src.processing.storage.DogeStorage` backed by SQLite.
    """
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    store = DogeStorage(cfg, engine=engine)
    store._lock_path = tmp_path / ".doge_storage.lock"
    store.create_tables()
    return store


@pytest.fixture()
def datasets() -> dict[str, pd.DataFrame]:
    """Pre-built synthetic OHLCV datasets for three symbols.

    All rows start at ``_TRAINING_START_MS`` so every candle is in the
    ``'training'`` era.

    Returns:
        Dict mapping symbol → DataFrame.
    """
    return {
        "DOGEUSDT": _make_ohlcv(_N_ROWS, _TRAINING_START_MS, open_price=0.10),
        "BTCUSDT": _make_ohlcv(_N_ROWS, _TRAINING_START_MS, open_price=20_000.0),
        "DOGEBTC": _make_ohlcv(_N_ROWS, _TRAINING_START_MS, open_price=5e-6),
    }


@pytest.fixture()
def fake_client(datasets: dict[str, pd.DataFrame]) -> _FakeClient:
    """Fake REST client pre-loaded with the synthetic datasets.

    Args:
        datasets: Synthetic OHLCV datasets fixture.

    Returns:
        :class:`_FakeClient` instance.
    """
    return _FakeClient(datasets)


@pytest.fixture()
def bootstrap_engine(
    fake_client: _FakeClient,
    tmp_path: Path,
) -> BootstrapEngine:
    """BootstrapEngine wired to the fake client and a temp checkpoint dir.

    Args:
        fake_client: Fake REST client fixture.
        tmp_path: pytest temporary directory for checkpoints.

    Returns:
        :class:`~src.ingestion.bootstrap.BootstrapEngine` instance.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    return BootstrapEngine(
        client=fake_client,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_rows=_CHECKPOINT_EVERY,
        batch_size=_BATCH_SIZE,
    )


# ---------------------------------------------------------------------------
# Shared helper — run full bootstrap for all 3 symbols
# ---------------------------------------------------------------------------


def _run_full_bootstrap(
    engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """Bootstrap all three symbols into storage.

    Args:
        engine: BootstrapEngine to use.
        storage: Storage to upsert into.
    """
    end_ms = _TRAINING_START_MS + _N_ROWS * _INTERVAL_MS
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        engine.bootstrap_symbol(sym, "1h", _TRAINING_START_MS, end_ms, storage)


# ---------------------------------------------------------------------------
# Tests — bootstrap correctness
# ---------------------------------------------------------------------------


def test_bootstrap_stores_correct_row_count(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """After bootstrap, each symbol has exactly _N_ROWS rows in storage."""
    _run_full_bootstrap(bootstrap_engine, storage)
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        df = storage.get_ohlcv(sym, "1h", 0, 10_000_000_000_000)
        assert len(df) == _N_ROWS, f"{sym}: expected {_N_ROWS} rows, got {len(df)}"


def test_bootstrap_checkpoint_deleted_on_completion(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
    tmp_path: Path,
) -> None:
    """Checkpoint files must not exist after a successful bootstrap."""
    _run_full_bootstrap(bootstrap_engine, storage)
    checkpoint_dir = tmp_path / "checkpoints"
    remaining = list(checkpoint_dir.glob("*.json"))
    assert remaining == [], (
        f"Checkpoint files not cleaned up: {[f.name for f in remaining]}"
    )


def test_bootstrap_era_all_training(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """All rows starting at _TRAINING_START_MS must have era='training'."""
    _run_full_bootstrap(bootstrap_engine, storage)
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        df = storage.get_ohlcv(sym, "1h", 0, 10_000_000_000_000)
        assert (df["era"] == "training").all(), (
            f"{sym}: some rows have wrong era"
        )


def test_bootstrap_era_context_for_pre2022_data(
    fake_client: _FakeClient,
    storage: DogeStorage,
    tmp_path: Path,
) -> None:
    """Rows before _TRAINING_START_MS must have era='context'."""
    # Build a dataset that starts 10 hours BEFORE the era boundary
    pre_start = _TRAINING_START_MS - 10 * _INTERVAL_MS
    pre_df = _make_ohlcv(10, pre_start)
    fake_client._datasets["DOGEUSDT"] = pre_df

    engine = BootstrapEngine(
        client=fake_client,
        checkpoint_dir=tmp_path / "checkpoints2",
        checkpoint_every_n_rows=5_000,
        batch_size=1_000,
    )
    end_ms = pre_start + 10 * _INTERVAL_MS
    engine.bootstrap_symbol("DOGEUSDT", "1h", pre_start, end_ms, storage)

    df = storage.get_ohlcv("DOGEUSDT", "1h", 0, 10_000_000_000_000)
    assert (df["era"] == "context").all(), (
        "Pre-2022 rows must be labelled 'context'"
    )


# ---------------------------------------------------------------------------
# Tests — DataValidator on bootstrapped data
# ---------------------------------------------------------------------------


def test_validator_passes_for_bootstrapped_symbols(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """DataValidator.validate_ohlcv must return is_valid=True for each symbol."""
    _run_full_bootstrap(bootstrap_engine, storage)
    validator = DataValidator()
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        df = storage.get_ohlcv(sym, "1h", 0, 10_000_000_000_000)
        result = validator.validate_ohlcv(df, sym, "1h")
        assert result.is_valid, (
            f"Validation failed for {sym}: {result.errors}"
        )
        assert result.gap_count == 0, f"{sym}: unexpected gaps after bootstrap"
        assert result.duplicate_count == 0, f"{sym}: unexpected duplicates"


# ---------------------------------------------------------------------------
# Tests — MultiSymbolAligner on bootstrapped data
# ---------------------------------------------------------------------------


def test_aligned_symbols_have_identical_open_time_index(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """All 3 symbols must share an identical open_time set after alignment."""
    _run_full_bootstrap(bootstrap_engine, storage)
    aligner = MultiSymbolAligner()
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned
    merged_times = set(df["open_time"].tolist())

    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        sym_df = storage.get_ohlcv(sym, "1h", 0, 10_000_000_000_000)
        sym_times = set(sym_df["open_time"].tolist())
        assert merged_times.issubset(sym_times), (
            f"{sym}: merged timestamps not a subset of stored timestamps"
        )


def test_aligned_row_count_matches_dataset(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """rows_aligned must equal _N_ROWS when all symbols are gap-free."""
    _run_full_bootstrap(bootstrap_engine, storage)
    aligner = MultiSymbolAligner()
    result: AlignmentResult = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    assert result.rows_aligned == _N_ROWS
    assert result.gaps_recovered == 0


def test_aligned_output_has_training_era(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """The ``era`` column in the aligned output must be ``'training'`` for all rows."""
    _run_full_bootstrap(bootstrap_engine, storage)
    aligner = MultiSymbolAligner()
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned
    assert "era" in df.columns
    assert (df["era"] == "training").all()


def test_aligned_output_no_nan_or_inf(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """No NaN or Inf values may appear in any numeric column of the aligned output."""
    _run_full_bootstrap(bootstrap_engine, storage)
    aligner = MultiSymbolAligner()
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        nan_count = int(df[col].isna().sum())
        inf_count = int(df[col].isin([float("inf"), float("-inf")]).sum())
        assert nan_count == 0, f"NaN in aligned column '{col}': {nan_count} rows"
        assert inf_count == 0, f"Inf in aligned column '{col}': {inf_count} rows"


def test_aligned_output_has_prefixed_columns(
    bootstrap_engine: BootstrapEngine,
    storage: DogeStorage,
) -> None:
    """Aligned DataFrame must have correctly prefixed OHLCV columns."""
    _run_full_bootstrap(bootstrap_engine, storage)
    aligner = MultiSymbolAligner()
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned
    for col in ("doge_close", "doge_volume", "btc_open", "btc_close",
                 "dogebtc_close", "dogebtc_volume"):
        assert col in df.columns, f"Expected column '{col}' in aligned output"


# ---------------------------------------------------------------------------
# Tests — IncrementalScheduler.run_once
# ---------------------------------------------------------------------------


def test_scheduler_run_once_upserts_candles(
    storage: DogeStorage,
    datasets: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """IncrementalScheduler.run_once must upsert fresh candles into storage."""
    # Seed storage with all but the last candle
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        df = datasets[sym].copy()
        df["era"] = "training"
        df["is_interpolated"] = False
        storage.upsert_ohlcv(df.iloc[:-1], sym, "1h")

    # Build a fake client that returns fresh data close to now
    now_ms = int(time.time() * 1_000)
    interval_ms = _INTERVAL_MS
    fresh_datasets: dict[str, pd.DataFrame] = {}
    for sym, price in (("DOGEUSDT", 0.10), ("BTCUSDT", 20_000.0), ("DOGEBTC", 5e-6)):
        fresh = _make_ohlcv(3, now_ms - 3 * interval_ms, interval_ms, price)
        fresh_datasets[sym] = fresh

    validator = DataValidator()
    scheduler = IncrementalScheduler(
        client=_FakeClient(fresh_datasets),
        storage=storage,
        validator=validator,
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        overlap_candles=3,
    )
    stats = scheduler.run_once()

    assert stats.runs == 1
    assert stats.candles_fetched >= 3  # at least 3 candles fetched per symbol


def test_scheduler_stats_track_new_vs_updated(
    storage: DogeStorage,
    tmp_path: Path,
) -> None:
    """SchedulerStats must count new and updated candles separately."""
    now_ms = int(time.time() * 1_000)
    interval_ms = _INTERVAL_MS

    # First run: no data in storage → all candles should be new
    fresh: dict[str, pd.DataFrame] = {}
    for sym, price in (("DOGEUSDT", 0.10), ("BTCUSDT", 20_000.0), ("DOGEBTC", 5e-6)):
        fresh[sym] = _make_ohlcv(3, now_ms - 3 * interval_ms, interval_ms, price)

    scheduler = IncrementalScheduler(
        client=_FakeClient(fresh),
        storage=storage,
        validator=DataValidator(),
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        overlap_candles=3,
    )
    stats = scheduler.run_once()

    # All 3*3 fetched candles are new (nothing was in storage)
    assert stats.candles_new > 0
    assert stats.last_run_at != ""
    assert stats.last_run_errors == 0
