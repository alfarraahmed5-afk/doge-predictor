#!/usr/bin/env python
"""Quality Gate 01 — Data Ingestion Verification.

Connects to storage (live PostgreSQL or in-memory SQLite with test data) and
verifies all ingestion pipeline invariants:

    Check 1  Row count >= minimum per symbol/interval
    Check 2  Timestamps are strictly monotonically increasing
    Check 3  Gap count within acceptable threshold (≤ 5 gaps per symbol)
    Check 4  Both era values present in distribution when expected
    Check 5  DataValidator returns is_valid=True for each symbol
    Check 6  All 3 symbols can be aligned by MultiSymbolAligner

Exits with code 0 when all checks PASS, code 1 if any check FAILS.

Usage::

    # Against live PostgreSQL database (requires settings.yaml configured):
    python scripts/qg01_verify.py

    # Against in-memory SQLite pre-loaded with synthetic test data (CI):
    python scripts/qg01_verify.py --in-memory-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

# ---------------------------------------------------------------------------
# Bootstrap sys.path so src/ imports work when called directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_settings  # noqa: E402
from src.processing.aligner import AlignmentError, MultiSymbolAligner  # noqa: E402
from src.processing.storage import DogeStorage  # noqa: E402
from src.processing.validator import DataValidator  # noqa: E402
from src.utils.helpers import interval_to_ms  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Symbols to check (must all be present for QG-01 to pass).
_QG01_SYMBOLS: list[str] = ["DOGEUSDT", "BTCUSDT", "DOGEBTC"]

#: Interval to check.
_QG01_INTERVAL: str = "1h"

#: Minimum acceptable row count per symbol for QG-01 to pass.
_MIN_ROW_COUNT: int = 100

#: Maximum acceptable gap count per symbol.
_MAX_GAP_COUNT: int = 5

#: UTC epoch ms: 2022-01-01 00:00:00 (era boundary).
_TRAINING_START_MS: int = 1_640_995_200_000

#: UTC epoch ms: 2019-07-01 00:00:00 (Binance DOGE listing approx).
_CONTEXT_START_MS: int = 1_561_939_200_000


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class _CheckResult:
    """Accumulates PASS/FAIL results across all QG-01 checks."""

    def __init__(self) -> None:
        self._results: list[dict[str, Any]] = []

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        """Record one check result.

        Args:
            name: Short check description.
            passed: True if the check passed.
            detail: Additional detail to print (only shown on FAIL or always).
        """
        status = "PASS" if passed else "FAIL"
        self._results.append({"name": name, "passed": passed, "detail": detail})
        marker = "  [PASS]" if passed else "  [FAIL]"
        line = f"{marker}  {name}"
        if detail and not passed:
            line += f"\n          -> {detail}"
        print(line)

    @property
    def all_passed(self) -> bool:
        """True if every recorded check passed."""
        return all(r["passed"] for r in self._results)

    @property
    def n_failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self._results if not r["passed"])


# ---------------------------------------------------------------------------
# Test-data seeder
# ---------------------------------------------------------------------------


def _seed_test_storage(storage: DogeStorage) -> None:
    """Seed *storage* with synthetic data for all three QG-01 symbols.

    Creates 200 consecutive 1h candles per symbol, split across both era
    values (100 context + 100 training).

    Args:
        storage: Empty DogeStorage instance to populate.
    """
    import random

    random.seed(42)
    interval_ms = interval_to_ms(_QG01_INTERVAL)

    # Start 200h before the era boundary so 100 rows are context, 100 training
    start_ms = _TRAINING_START_MS - 100 * interval_ms
    n = 200

    for sym in _QG01_SYMBOLS:
        base_price = {"DOGEUSDT": 0.10, "BTCUSDT": 20_000.0, "DOGEBTC": 5e-6}[sym]
        rows = []
        for i in range(n):
            t = start_ms + i * interval_ms
            era = "context" if t < _TRAINING_START_MS else "training"
            rows.append({
                "open_time": t,
                "open": base_price,
                "high": base_price * 1.02,
                "low": base_price * 0.98,
                "close": base_price * 1.01,
                "volume": 1_000_000.0,
                "close_time": t + interval_ms - 1,
                "era": era,
                "is_interpolated": False,
            })
        df = pd.DataFrame(rows)
        storage.upsert_ohlcv(df, sym, _QG01_INTERVAL)

    print(f"  -> Seeded {n} rows x {len(_QG01_SYMBOLS)} symbols into in-memory storage.\n")


# ---------------------------------------------------------------------------
# QG-01 checks
# ---------------------------------------------------------------------------


def run_qg01(storage: DogeStorage, cr: _CheckResult) -> None:
    """Execute all QG-01 checks against *storage*.

    Args:
        storage: Populated DogeStorage to verify.
        cr: :class:`_CheckResult` accumulator.
    """
    validator = DataValidator()
    all_dfs: dict[str, pd.DataFrame] = {}

    print("-" * 60)
    print("  Checks 1-5 -- Per-symbol invariants")
    print("-" * 60)

    for sym in _QG01_SYMBOLS:
        df = storage.get_ohlcv(sym, _QG01_INTERVAL, 0, 10_000_000_000_000)
        all_dfs[sym] = df

        # ── Check 1: Row count ───────────────────────────────────────────
        cr.record(
            f"[{sym}] Check 1 — Row count >= {_MIN_ROW_COUNT}",
            passed=len(df) >= _MIN_ROW_COUNT,
            detail=f"actual row count: {len(df)}",
        )

        if df.empty:
            # Cannot run further checks on empty data
            for n in range(2, 6):
                cr.record(
                    f"[{sym}] Check {n} — skipped (no data)",
                    passed=False,
                    detail="empty DataFrame — run bootstrap first",
                )
            continue

        # ── Check 2: Monotonic timestamps ────────────────────────────────
        is_mono = bool(df["open_time"].is_monotonic_increasing)
        cr.record(
            f"[{sym}] Check 2 — Timestamps monotonically increasing",
            passed=is_mono,
            detail="" if is_mono else "open_time is not monotonically increasing",
        )

        # ── Check 3: Gap count within threshold ──────────────────────────
        interval_ms = interval_to_ms(_QG01_INTERVAL)
        times = df["open_time"].tolist()
        gap_count = sum(
            1
            for a, b in zip(times, times[1:])
            if b - a > interval_ms
        )
        cr.record(
            f"[{sym}] Check 3 — Gap count <= {_MAX_GAP_COUNT}",
            passed=gap_count <= _MAX_GAP_COUNT,
            detail=f"gaps found: {gap_count}",
        )

        # ── Check 4: Era distribution ─────────────────────────────────────
        era_counts = df["era"].value_counts().to_dict() if "era" in df.columns else {}
        has_context = era_counts.get("context", 0) > 0
        has_training = era_counts.get("training", 0) > 0
        cr.record(
            f"[{sym}] Check 4 — Era 'context' present",
            passed=has_context,
            detail=f"era distribution: {era_counts}",
        )
        cr.record(
            f"[{sym}] Check 4 — Era 'training' present",
            passed=has_training,
            detail=f"era distribution: {era_counts}",
        )

        # ── Check 5: DataValidator ────────────────────────────────────────
        try:
            val_result = validator.validate_ohlcv(df, sym, _QG01_INTERVAL)
            cr.record(
                f"[{sym}] Check 5 — DataValidator is_valid=True",
                passed=val_result.is_valid,
                detail="; ".join(val_result.errors) if val_result.errors else "",
            )
        except Exception as exc:  # noqa: BLE001
            cr.record(
                f"[{sym}] Check 5 — DataValidator raised exception",
                passed=False,
                detail=str(exc),
            )

    # ── Check 6: MultiSymbolAligner ──────────────────────────────────────
    print()
    print("-" * 60)
    print("  Check 6 -- MultiSymbolAligner")
    print("-" * 60)

    aligner = MultiSymbolAligner()
    try:
        result = aligner.align_symbols(
            symbols=_QG01_SYMBOLS,
            interval=_QG01_INTERVAL,
            storage=storage,
        )
        aligned_df = aligner._last_aligned

        cr.record(
            "Check 6a — align_symbols completes without error",
            passed=True,
            detail=f"rows_aligned={result.rows_aligned} gaps_recovered={result.gaps_recovered}",
        )

        # Verify all symbols share identical open_times in aligned output
        merged_times = set(aligned_df["open_time"].tolist())
        all_identical = True
        for sym in _QG01_SYMBOLS:
            sym_times = set(all_dfs[sym]["open_time"].tolist())
            if not merged_times.issubset(sym_times):
                all_identical = False
        cr.record(
            "Check 6b — All symbols share identical open_time index",
            passed=all_identical,
            detail="" if all_identical else "open_time mismatch detected",
        )

        # No NaN or Inf in numeric columns
        numeric_cols = aligned_df.select_dtypes(include="number").columns
        nan_cols = [c for c in numeric_cols if aligned_df[c].isna().any()]
        inf_cols = [
            c for c in numeric_cols
            if aligned_df[c].isin([float("inf"), float("-inf")]).any()
        ]
        cr.record(
            "Check 6c — No NaN in aligned numeric columns",
            passed=len(nan_cols) == 0,
            detail=f"columns with NaN: {nan_cols}" if nan_cols else "",
        )
        cr.record(
            "Check 6d - No Inf in aligned numeric columns",
            passed=len(inf_cols) == 0,
            detail=f"columns with Inf: {inf_cols}" if inf_cols else "",
        )

    except AlignmentError as exc:
        cr.record(
            "Check 6 — align_symbols raised AlignmentError",
            passed=False,
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        cr.record(
            "Check 6 — align_symbols raised unexpected exception",
            passed=False,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Parse arguments, connect to storage, run QG-01 checks, return exit code.

    Returns:
        0 if all checks pass, 1 if any check fails.
    """
    parser = argparse.ArgumentParser(
        description="Quality Gate 01 — Data Ingestion Verification"
    )
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help=(
            "Seed an in-memory SQLite database with synthetic data and run "
            "all checks against it.  Use this mode in CI or before live "
            "bootstrap is complete."
        ),
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  QUALITY GATE 01 — Data Ingestion Verification")
    print("=" * 60)
    print()

    settings = get_settings()

    if args.in_memory_test:
        print("  Mode: IN-MEMORY TEST (synthetic data)")
        print()
        import tempfile

        from pathlib import Path as _Path

        engine = sa.create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = DogeStorage(settings, engine=engine)
            storage._lock_path = _Path(tmp_dir) / ".doge_storage.lock"
            storage.create_tables()
            _seed_test_storage(storage)
            cr = _CheckResult()
            run_qg01(storage, cr)
    else:
        print("  Mode: LIVE DATABASE")
        print(f"  URL:  {settings.database.url}")
        print()
        try:
            storage = DogeStorage(settings)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] Could not connect to database: {exc}")
            print()
            print("  Tip: run with --in-memory-test for CI / pre-bootstrap checks.")
            return 1
        cr = _CheckResult()
        run_qg01(storage, cr)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    if cr.all_passed:
        print("  RESULT: ALL CHECKS PASSED - QG-01 PASS")
    else:
        print(f"  RESULT: {cr.n_failed} CHECK(S) FAILED - QG-01 FAIL")
    print("=" * 60)
    print()

    return 0 if cr.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
