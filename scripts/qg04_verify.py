#!/usr/bin/env python
"""Quality Gate 04 — Regime Classification Verification.

Loads regime labels from storage and verifies all classification invariants:

    Check 1  All 5 regimes present (>= 1 row each)
    Check 2  No NaN regime_label values
    Check 3  No single regime dominates (>= 70% of rows)
    Check 4  Transition count > 0 (classification is not static)
    Check 5  Average regime duration reported per regime

Exits with code 0 (PASS) or 1 (FAIL).

Usage::

    python scripts/qg04_verify.py                  # live PostgreSQL
    python scripts/qg04_verify.py --in-memory-test  # CI / no live data
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy as sa

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.label_regimes import _seed_test_data, run_labelling  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.processing.storage import DogeStorage  # noqa: E402
from src.regimes.classifier import REGIME_LABELS  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DOMINANCE: float = 0.70   # No single regime should exceed this fraction
_MIN_TRANSITIONS: int = 1      # Classification must change at least once


# ---------------------------------------------------------------------------
# _CheckResult helper
# ---------------------------------------------------------------------------


@dataclass
class _CheckResult:
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# QG-04 checks
# ---------------------------------------------------------------------------


def _check_all_regimes_present(df: pd.DataFrame) -> _CheckResult:
    present = set(df["regime"].dropna().unique())
    missing = sorted(set(REGIME_LABELS) - present)
    passed = len(missing) == 0
    detail = "OK — all 5 regimes present" if passed else f"MISSING: {missing}"
    return _CheckResult("All 5 regimes present", passed, detail)


def _check_no_nan(df: pd.DataFrame) -> _CheckResult:
    n_nan = int(df["regime"].isna().sum())
    passed = n_nan == 0
    detail = f"OK — 0 NaN" if passed else f"FAIL — {n_nan} NaN regime rows"
    return _CheckResult("No NaN regime labels", passed, detail)


def _check_no_dominance(df: pd.DataFrame) -> _CheckResult:
    counts = df["regime"].value_counts(normalize=True)
    max_regime = counts.idxmax()
    max_frac = float(counts.max())
    passed = max_frac < _MAX_DOMINANCE
    detail = (
        f"OK — max is {max_regime} at {max_frac:.1%}"
        if passed
        else f"FAIL — {max_regime} is {max_frac:.1%} (threshold {_MAX_DOMINANCE:.0%})"
    )
    return _CheckResult("No single regime > 70%", passed, detail)


def _check_transitions(df: pd.DataFrame) -> _CheckResult:
    if len(df) < 2:
        return _CheckResult("Transition count > 0", False, "FAIL — fewer than 2 rows")
    sorted_df = df.sort_values("open_time").reset_index(drop=True)
    transitions = int((sorted_df["regime"] != sorted_df["regime"].shift()).sum()) - 1
    passed = transitions >= _MIN_TRANSITIONS
    detail = (
        f"OK — {transitions} transitions"
        if passed
        else f"FAIL — {transitions} transitions (min {_MIN_TRANSITIONS})"
    )
    return _CheckResult("Transition count > 0", passed, detail)


def _report_durations(df: pd.DataFrame) -> None:
    """Print average regime duration (candles) per regime."""
    sorted_df = df.sort_values("open_time").reset_index(drop=True)
    runs: list[dict] = []
    current = sorted_df["regime"].iloc[0]
    run_len = 1
    for r in sorted_df["regime"].iloc[1:]:
        if r == current:
            run_len += 1
        else:
            runs.append({"regime": current, "length": run_len})
            current = r
            run_len = 1
    runs.append({"regime": current, "length": run_len})

    run_df = pd.DataFrame(runs)
    print("\n  Average regime duration (candles):")
    for label in REGIME_LABELS:
        sub = run_df[run_df["regime"] == label]["length"]
        if len(sub) > 0:
            print(f"    {label:<20s} avg={sub.mean():.1f}  runs={len(sub)}")
        else:
            print(f"    {label:<20s} (not present)")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_qg04(storage: DogeStorage) -> bool:
    """Execute all QG-04 checks and print results.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.

    Returns:
        ``True`` if every check passes, ``False`` otherwise.
    """
    df = storage.get_regime_labels(0, 99_999_999_999_999)

    if df.empty:
        print("FAIL — No regime labels in storage (run label_regimes.py first).")
        return False

    checks: list[_CheckResult] = [
        _check_all_regimes_present(df),
        _check_no_nan(df),
        _check_no_dominance(df),
        _check_transitions(df),
    ]

    all_pass = True
    print(f"\n  {'Check':<35s} {'Status':<8s} Detail")
    print(f"  {'-'*35} {'-'*8} {'-'*40}")
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"  {c.name:<35s} {status:<8s} {c.detail}")
        if not c.passed:
            all_pass = False

    _report_durations(df)
    return all_pass


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 = PASS, 1 = FAIL.
    """
    configure_logging()

    parser = argparse.ArgumentParser(description="Quality Gate 04 — Regime Classification.")
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help="Seed in-memory SQLite with synthetic data before checking.",
    )
    args = parser.parse_args(argv)

    print("=" * 65)
    print("QG-04 — Regime Classification Verification")
    print("=" * 65)

    try:
        if args.in_memory_test:
            engine = sa.create_engine("sqlite:///:memory:")
            storage = DogeStorage(get_settings(), engine=engine)
            storage.create_tables()
            _seed_test_data(storage)
            print("[IN-MEMORY-TEST] Seeding synthetic data...")
            run_labelling(storage, output_dir=_REPO_ROOT / "data" / "regimes")
        else:
            storage = DogeStorage(get_settings())

        passed = run_qg04(storage)

    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print("\n" + "=" * 65)
    print(f"QG-04 RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 65)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
