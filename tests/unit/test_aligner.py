"""Unit tests for src/processing/aligner.py — MultiSymbolAligner.

All tests use an in-memory SQLite storage so no database or network is needed.
OHLCV data is inserted before each test via ``upsert_ohlcv``.

Test coverage:
    - No gaps: three symbols with identical timestamps → all rows aligned
    - DOGEBTC gap of 3 candles → forward-filled, dogebtc_interpolated=True
    - DOGEBTC gap of 4 candles → AlignmentError raised
    - BTCUSDT gap > 3 candles → AlignmentError raised (non-DOGEBTC)
    - Identical open_time index assertion across all symbols
    - Common date range is the inner intersection (max-of-mins, min-of-maxes)
    - Forward-filled DOGEBTC rows have volume=0 and prices from prior candle
    - AlignmentResult fields populated correctly
    - Empty symbol list raises ValueError
    - Symbol with no data raises AlignmentError
    - _find_gap_runs static method unit-tested in isolation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa

from src.config import settings as cfg
from src.processing.aligner import AlignmentError, AlignmentResult, MultiSymbolAligner
from src.processing.storage import DogeStorage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 2022-01-01 00:00:00 UTC — era boundary.
_TRAINING_START_MS: int = 1_640_995_200_000

#: 1-hour interval in milliseconds.
_INTERVAL_MS: int = 3_600_000

#: Start point used by most tests.
_START_MS: int = _TRAINING_START_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int,
    start_ms: int = _START_MS,
    interval_ms: int = _INTERVAL_MS,
    open_price: float = 0.10,
    era: str = "training",
) -> pd.DataFrame:
    """Build a contiguous OHLCV DataFrame with *n* candles.

    Args:
        n: Number of rows.
        start_ms: ``open_time`` of the first row (UTC epoch ms).
        interval_ms: Milliseconds per candle.
        open_price: Open (and approximately close) price for all rows.
        era: Era label for all rows.

    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume,
        close_time, era.
    """
    rows: list[dict[str, Any]] = []
    for i in range(n):
        open_t = start_ms + i * interval_ms
        rows.append({
            "open_time": open_t,
            "open": open_price,
            "high": open_price * 1.05,
            "low": open_price * 0.95,
            "close": open_price * 1.01,
            "volume": 1_000_000.0,
            "close_time": open_t + interval_ms - 1,
            "era": era,
        })
    return pd.DataFrame(rows)


def _make_ohlcv_with_gap(
    n: int,
    gap_after_index: int,
    gap_size: int,
    start_ms: int = _START_MS,
) -> pd.DataFrame:
    """Build a candle DataFrame with *gap_size* missing candles after *gap_after_index*.

    Args:
        n: Total rows before the gap is inserted.
        gap_after_index: 0-based index of the last row before the gap.
        gap_size: Number of missing candles in the gap.
        start_ms: Starting ``open_time`` (UTC epoch ms).

    Returns:
        DataFrame with ``n - gap_size`` rows (gap candles excluded).
    """
    rows: list[dict[str, Any]] = []
    skip_start = gap_after_index + 1
    skip_end = gap_after_index + gap_size + 1  # exclusive
    for i in range(n):
        if skip_start <= i < skip_end:
            continue
        open_t = start_ms + i * _INTERVAL_MS
        rows.append({
            "open_time": open_t,
            "open": 0.10,
            "high": 0.11,
            "low": 0.09,
            "close": 0.105,
            "volume": 1_000_000.0,
            "close_time": open_t + _INTERVAL_MS - 1,
            "era": "training",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage(tmp_path: Path) -> DogeStorage:
    """In-memory SQLite DogeStorage with tables created.

    Args:
        tmp_path: pytest temporary directory for the filelock.

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
def aligner() -> MultiSymbolAligner:
    """Return a fresh MultiSymbolAligner.

    Returns:
        :class:`~src.processing.aligner.MultiSymbolAligner`.
    """
    return MultiSymbolAligner()


@pytest.fixture()
def full_storage(storage: DogeStorage) -> DogeStorage:
    """Storage pre-loaded with 10 candles each for DOGEUSDT, BTCUSDT, DOGEBTC.

    All three symbols share identical timestamps (no gaps).

    Args:
        storage: Base in-memory storage fixture.

    Returns:
        :class:`~src.processing.storage.DogeStorage` with data for all 3 symbols.
    """
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        df = _make_ohlcv(10)
        storage.upsert_ohlcv(df, sym, "1h")
    return storage


# ---------------------------------------------------------------------------
# Tests — _find_gap_runs (static method, isolated)
# ---------------------------------------------------------------------------


def test_find_gap_runs_no_missing() -> None:
    """_find_gap_runs: empty missing list returns empty result."""
    runs = MultiSymbolAligner._find_gap_runs([], _INTERVAL_MS)
    assert runs == []


def test_find_gap_runs_single_missing() -> None:
    """_find_gap_runs: one missing timestamp → one run of size 1."""
    missing = [_START_MS + 2 * _INTERVAL_MS]
    runs = MultiSymbolAligner._find_gap_runs(missing, _INTERVAL_MS)
    assert len(runs) == 1
    assert runs[0] == (_START_MS + 2 * _INTERVAL_MS, _START_MS + 2 * _INTERVAL_MS, 1)


def test_find_gap_runs_contiguous_run() -> None:
    """_find_gap_runs: three consecutive missing timestamps → one run of size 3."""
    missing = [_START_MS + i * _INTERVAL_MS for i in range(1, 4)]
    runs = MultiSymbolAligner._find_gap_runs(missing, _INTERVAL_MS)
    assert len(runs) == 1
    assert runs[0][2] == 3  # run_size


def test_find_gap_runs_two_separate_runs() -> None:
    """_find_gap_runs: two disjoint missing regions → two separate runs."""
    missing = [
        _START_MS + 1 * _INTERVAL_MS,
        # skip 2
        _START_MS + 5 * _INTERVAL_MS,
        _START_MS + 6 * _INTERVAL_MS,
    ]
    runs = MultiSymbolAligner._find_gap_runs(missing, _INTERVAL_MS)
    assert len(runs) == 2
    assert runs[0][2] == 1  # first run: 1 candle
    assert runs[1][2] == 2  # second run: 2 candles


# ---------------------------------------------------------------------------
# Tests — align_symbols, all-good path
# ---------------------------------------------------------------------------


def test_align_no_gaps_all_rows_present(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """No gaps: 10 rows per symbol → 10 rows in aligned output."""
    result = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    assert isinstance(result, AlignmentResult)
    assert result.rows_aligned == 10
    assert result.gaps_recovered == 0


def test_align_result_fields_populated(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """AlignmentResult contains correct common_start, common_end, rows_aligned."""
    result = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    expected_start = _START_MS
    expected_end = _START_MS + 9 * _INTERVAL_MS
    assert result.common_start == expected_start
    assert result.common_end == expected_end
    assert result.rows_aligned == 10


def test_align_produces_dataframe(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """_last_aligned is a non-empty DataFrame after a successful align."""
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    df = aligner._last_aligned
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10


def test_align_output_has_prefixed_columns(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """Aligned DataFrame columns are prefixed (doge_close, btc_open, etc.)."""
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    df = aligner._last_aligned
    assert "doge_close" in df.columns
    assert "btc_open" in df.columns
    assert "dogebtc_close" in df.columns
    assert "open_time" in df.columns


# ---------------------------------------------------------------------------
# Tests — identical open_time index across symbols
# ---------------------------------------------------------------------------


def test_align_identical_open_time_index(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """Merged DataFrame open_times must equal the intersection of all symbols."""
    result = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    df = aligner._last_aligned
    merged_times = set(df["open_time"].tolist())

    # Verify that each symbol's stored data that falls in the common range
    # equals the merged times
    for sym in ("DOGEUSDT", "BTCUSDT", "DOGEBTC"):
        sym_df = full_storage.get_ohlcv(
            sym, "1h", result.common_start, result.common_end + 1
        )
        sym_times = set(sym_df["open_time"].tolist())
        # After alignment, merged times ⊆ sym_times (inner join)
        assert merged_times.issubset(sym_times), (
            f"{sym}: merged timestamps not a subset of symbol timestamps"
        )


def test_align_open_time_is_monotonic(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """Aligned DataFrame open_time column must be strictly monotonic."""
    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=full_storage,
    )
    df = aligner._last_aligned
    assert df["open_time"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Tests — DOGEBTC forward-fill
# ---------------------------------------------------------------------------


def test_align_dogebtc_3_candle_gap_filled(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """DOGEBTC gap of exactly 3 candles is forward-filled; gaps_recovered == 3."""
    n = 10
    # DOGEUSDT and BTCUSDT: full 10 rows
    for sym in ("DOGEUSDT", "BTCUSDT"):
        storage.upsert_ohlcv(_make_ohlcv(n), sym, "1h")

    # DOGEBTC: missing candles 4, 5, 6 (0-indexed → after index 3)
    dogebtc_df = _make_ohlcv_with_gap(n, gap_after_index=3, gap_size=3)
    storage.upsert_ohlcv(dogebtc_df, "DOGEBTC", "1h")

    result = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )

    assert result.gaps_recovered == 3
    assert result.rows_aligned == 10  # all 10 rows present after fill


def test_align_dogebtc_filled_rows_marked_interpolated(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Forward-filled DOGEBTC rows have dogebtc_interpolated=True."""
    n = 8
    for sym in ("DOGEUSDT", "BTCUSDT"):
        storage.upsert_ohlcv(_make_ohlcv(n), sym, "1h")

    # DOGEBTC: 2 missing candles (indices 2, 3)
    dogebtc_df = _make_ohlcv_with_gap(n, gap_after_index=1, gap_size=2)
    storage.upsert_ohlcv(dogebtc_df, "DOGEBTC", "1h")

    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned

    assert "dogebtc_interpolated" in df.columns

    # Rows 2 and 3 (0-indexed by row position) should be marked interpolated
    interp_rows = df[df["dogebtc_interpolated"] == True]  # noqa: E712
    assert len(interp_rows) == 2


def test_align_dogebtc_filled_volume_is_zero(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Forward-filled DOGEBTC candles have volume=0 (never forward-filled)."""
    n = 6
    for sym in ("DOGEUSDT", "BTCUSDT"):
        storage.upsert_ohlcv(_make_ohlcv(n), sym, "1h")

    # DOGEBTC: missing candle at index 2
    dogebtc_df = _make_ohlcv_with_gap(n, gap_after_index=1, gap_size=1)
    storage.upsert_ohlcv(dogebtc_df, "DOGEBTC", "1h")

    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned

    interp_rows = df[df["dogebtc_interpolated"] == True]  # noqa: E712
    assert len(interp_rows) == 1
    assert float(interp_rows["dogebtc_volume"].iloc[0]) == 0.0


def test_align_dogebtc_filled_price_matches_prior_candle(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Forward-filled DOGEBTC prices match the last valid candle before the gap."""
    n = 6
    for sym in ("DOGEUSDT", "BTCUSDT"):
        storage.upsert_ohlcv(_make_ohlcv(n), sym, "1h")

    # Build DOGEBTC with known prices; gap after index 1
    dogebtc_df = _make_ohlcv_with_gap(n, gap_after_index=1, gap_size=1)
    storage.upsert_ohlcv(dogebtc_df, "DOGEBTC", "1h")

    aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    df = aligner._last_aligned

    # Row index 1 is the last valid DOGEBTC candle before the gap
    prior_close = float(df.iloc[1]["dogebtc_close"])
    interp_close = float(df[df["dogebtc_interpolated"] == True]["dogebtc_close"].iloc[0])  # noqa: E712
    assert interp_close == prior_close


# ---------------------------------------------------------------------------
# Tests — AlignmentError cases
# ---------------------------------------------------------------------------


def test_align_dogebtc_4_candle_gap_raises(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """DOGEBTC gap of 4 candles raises AlignmentError."""
    n = 10
    for sym in ("DOGEUSDT", "BTCUSDT"):
        storage.upsert_ohlcv(_make_ohlcv(n), sym, "1h")

    dogebtc_df = _make_ohlcv_with_gap(n, gap_after_index=2, gap_size=4)
    storage.upsert_ohlcv(dogebtc_df, "DOGEBTC", "1h")

    with pytest.raises(AlignmentError, match="gap of 4"):
        aligner.align_symbols(
            symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
            interval="1h",
            storage=storage,
        )


def test_align_btcusdt_gap_raises(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Non-DOGEBTC symbol gap > 3 candles also raises AlignmentError."""
    n = 10
    storage.upsert_ohlcv(_make_ohlcv(n), "DOGEUSDT", "1h")
    storage.upsert_ohlcv(_make_ohlcv(n), "DOGEBTC", "1h")

    # BTCUSDT: 5-candle gap (more than 3)
    btc_df = _make_ohlcv_with_gap(n, gap_after_index=3, gap_size=5)
    storage.upsert_ohlcv(btc_df, "BTCUSDT", "1h")

    with pytest.raises(AlignmentError, match="gap of 5"):
        aligner.align_symbols(
            symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
            interval="1h",
            storage=storage,
        )


def test_align_empty_symbol_list_raises(
    aligner: MultiSymbolAligner,
    full_storage: DogeStorage,
) -> None:
    """Empty symbols list raises ValueError immediately."""
    with pytest.raises(ValueError, match="symbols list must not be empty"):
        aligner.align_symbols(symbols=[], interval="1h", storage=full_storage)


def test_align_symbol_with_no_data_raises(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Symbol not in storage raises AlignmentError."""
    storage.upsert_ohlcv(_make_ohlcv(10), "DOGEUSDT", "1h")
    # BTCUSDT and DOGEBTC have no data

    with pytest.raises(AlignmentError, match="no data in storage"):
        aligner.align_symbols(
            symbols=["DOGEUSDT", "BTCUSDT"],
            interval="1h",
            storage=storage,
        )


# ---------------------------------------------------------------------------
# Tests — common date range
# ---------------------------------------------------------------------------


def test_align_common_date_range_is_inner_intersection(
    aligner: MultiSymbolAligner,
    storage: DogeStorage,
) -> None:
    """Common range is [max(min_dates), min(max_dates)] across all symbols."""
    # DOGEUSDT: rows 0–9 (starts at _START_MS)
    storage.upsert_ohlcv(_make_ohlcv(10), "DOGEUSDT", "1h")
    # BTCUSDT: rows 2–11 (starts 2 hours later → later min_time)
    storage.upsert_ohlcv(
        _make_ohlcv(10, start_ms=_START_MS + 2 * _INTERVAL_MS), "BTCUSDT", "1h"
    )
    # DOGEBTC: rows 0–7 (ends 2 hours earlier → earlier max_time)
    storage.upsert_ohlcv(_make_ohlcv(8), "DOGEBTC", "1h")

    result = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )

    # common_start = max(0, 2, 0) = +2h; common_end = min(9, 11, 7) = +7h
    assert result.common_start == _START_MS + 2 * _INTERVAL_MS
    assert result.common_end == _START_MS + 7 * _INTERVAL_MS
    assert result.rows_aligned == 6  # hours 2..7 inclusive
