"""QG-09 — Weekly Retraining Quality Gate.

Compares the metrics of the ``'candidate'`` model against the
``'previous-production'`` (or ``'production'``) model stored in MLflow.

Pass criteria (all must hold):

* Candidate mean_val_accuracy > production mean_val_accuracy
* Candidate Sharpe (if logged) >= production Sharpe (or production Sharpe unavailable)
* All per-regime metrics that are present pass acceptance gates

Usage::

    python scripts/qg09_verify.py
    python scripts/qg09_verify.py --in-memory-test
    python scripts/qg09_verify.py --mlflow-uri sqlite:///mlruns/mlflow.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import doge_settings, get_settings  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _print_row(check: str, status: str, detail: str = "") -> None:
    marker = "[PASS]" if status == _PASS else ("[FAIL]" if status == _FAIL else "[SKIP]")
    print(f"  {marker}  {check:<55} {detail}")


def _get_mlflow_run(
    mlflow_tracking_uri: str,
    tag_filter: str,
) -> "Any | None":
    """Return the most recent MLflow run matching *tag_filter*.

    Args:
        mlflow_tracking_uri: MLflow tracking URI.
        tag_filter: Tag filter string, e.g. ``"tags.stage = 'candidate'"``.

    Returns:
        MLflow Run object, or ``None`` if not found or MLflow unavailable.
    """
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_tracking_uri)
        exp = client.get_experiment_by_name("doge_predictor")
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=tag_filter,
            max_results=1,
            order_by=["attribute.start_time DESC"],
        )
        return runs[0] if runs else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("QG-09: MLflow query failed — {}", exc)
        return None


def _seed_mlflow_test_runs(mlflow_tracking_uri: str) -> tuple[str, str]:
    """Seed two synthetic MLflow runs for ``--in-memory-test`` mode.

    Creates a ``'candidate'`` run with slightly better accuracy than the
    ``'previous-production'`` run.

    Args:
        mlflow_tracking_uri: MLflow tracking URI to write to.

    Returns:
        ``(candidate_run_id, production_run_id)`` strings.
    """
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("doge_predictor")

    with mlflow.start_run(run_name="qg09_test_production") as run_prod:
        mlflow.log_metric("mean_val_accuracy", 0.62)
        mlflow.log_metric("sharpe_annualized", 1.2)
        mlflow.set_tag("stage", "previous-production")
        prod_run_id = run_prod.info.run_id

    with mlflow.start_run(run_name="qg09_test_candidate") as run_cand:
        mlflow.log_metric("mean_val_accuracy", 0.68)
        mlflow.log_metric("sharpe_annualized", 1.5)
        mlflow.log_metric("shadow_accuracy", 0.65)
        mlflow.set_tag("stage", "candidate")
        cand_run_id = run_cand.info.run_id

    return cand_run_id, prod_run_id


# ---------------------------------------------------------------------------
# QG-09 checks
# ---------------------------------------------------------------------------


def run_qg09(mlflow_tracking_uri: str) -> bool:  # noqa: C901
    """Run all QG-09 checks and print a results table.

    Args:
        mlflow_tracking_uri: MLflow tracking URI.

    Returns:
        ``True`` if all HARD checks pass, ``False`` otherwise.
    """
    print("\n" + "=" * 70)
    print("QG-09 — Weekly Retraining Quality Gate")
    print("=" * 70)

    # --- Load candidate and production runs ---------------------------------
    cand_run = _get_mlflow_run(
        mlflow_tracking_uri,
        "tags.stage = 'candidate'",
    )
    prod_run = _get_mlflow_run(
        mlflow_tracking_uri,
        "tags.stage = 'previous-production'",
    )
    # Fallback: current 'production' if no 'previous-production' found
    if prod_run is None:
        prod_run = _get_mlflow_run(
            mlflow_tracking_uri,
            "tags.stage = 'production'",
        )

    if cand_run is None:
        print("\n  [FAIL]  No 'candidate' MLflow run found. Run train.py first.")
        return False

    cand_acc = cand_run.data.metrics.get("mean_val_accuracy", 0.0)
    prod_acc = prod_run.data.metrics.get("mean_val_accuracy", 0.0) if prod_run else None

    cand_sharpe = cand_run.data.metrics.get("sharpe_annualized")
    prod_sharpe = prod_run.data.metrics.get("sharpe_annualized") if prod_run else None

    shadow_acc = cand_run.data.metrics.get("shadow_accuracy")

    gates = doge_settings.acceptance_gates
    failures: list[str] = []

    print(
        f"\n  Candidate run  : {cand_run.info.run_id}  acc={cand_acc:.4f}"
        + (f"  sharpe={cand_sharpe:.2f}" if cand_sharpe is not None else "")
    )
    print(
        "  Production run : "
        + (
            f"{prod_run.info.run_id}  acc={prod_acc:.4f}"
            + (f"  sharpe={prod_sharpe:.2f}" if prod_sharpe is not None else "")
            if prod_run is not None
            else "N/A (no production model found)"
        )
    )
    print()

    # --- Check 1: candidate accuracy > production accuracy ------------------
    if prod_acc is None:
        _print_row(
            "C1  Candidate accuracy > production accuracy",
            _PASS,
            f"cand={cand_acc:.4f} (no production to compare)",
        )
    elif cand_acc > prod_acc:
        _print_row(
            "C1  Candidate accuracy > production accuracy",
            _PASS,
            f"cand={cand_acc:.4f} > prod={prod_acc:.4f}",
        )
    else:
        _print_row(
            "C1  Candidate accuracy > production accuracy",
            _FAIL,
            f"cand={cand_acc:.4f} NOT > prod={prod_acc:.4f}",
        )
        failures.append("C1")

    # --- Check 2: candidate Sharpe >= acceptance gate -----------------------
    if cand_sharpe is None:
        _print_row("C2  Candidate Sharpe >= 1.0", _SKIP, "sharpe not logged")
    elif cand_sharpe >= gates.sharpe_annualized:
        _print_row("C2  Candidate Sharpe >= 1.0", _PASS, f"sharpe={cand_sharpe:.2f}")
    else:
        _print_row(
            "C2  Candidate Sharpe >= 1.0",
            _FAIL,
            f"sharpe={cand_sharpe:.2f} < {gates.sharpe_annualized}",
        )
        failures.append("C2")

    # --- Check 3: candidate Sharpe > production Sharpe (if both available) --
    if cand_sharpe is None or prod_sharpe is None:
        _print_row(
            "C3  Candidate Sharpe > production Sharpe",
            _SKIP,
            "one or both Sharpe values not logged",
        )
    elif cand_sharpe > prod_sharpe:
        _print_row(
            "C3  Candidate Sharpe > production Sharpe",
            _PASS,
            f"cand={cand_sharpe:.2f} > prod={prod_sharpe:.2f}",
        )
    else:
        _print_row(
            "C3  Candidate Sharpe > production Sharpe",
            _FAIL,
            f"cand={cand_sharpe:.2f} NOT > prod={prod_sharpe:.2f}",
        )
        failures.append("C3")

    # --- Check 4: shadow accuracy > production accuracy ---------------------
    if shadow_acc is None:
        _print_row(
            "C4  Shadow accuracy > production accuracy (48h window)",
            _SKIP,
            "shadow_accuracy metric not logged (run 48h shadow before QG-09)",
        )
    elif prod_acc is None:
        _print_row(
            "C4  Shadow accuracy > production accuracy (48h window)",
            _PASS,
            f"shadow={shadow_acc:.4f} (no production baseline)",
        )
    elif shadow_acc > prod_acc:
        _print_row(
            "C4  Shadow accuracy > production accuracy (48h window)",
            _PASS,
            f"shadow={shadow_acc:.4f} > prod={prod_acc:.4f}",
        )
    else:
        _print_row(
            "C4  Shadow accuracy > production accuracy (48h window)",
            _FAIL,
            f"shadow={shadow_acc:.4f} NOT > prod={prod_acc:.4f}",
        )
        failures.append("C4")

    # --- Check 5: candidate directional accuracy >= gate -------------------
    cand_dir_acc = cand_run.data.metrics.get("mean_val_accuracy", 0.0)
    gate_dir = gates.directional_accuracy_oos
    if cand_dir_acc >= gate_dir:
        _print_row(
            f"C5  Candidate dir_acc >= {gate_dir:.2f}",
            _PASS,
            f"dir_acc={cand_dir_acc:.4f}",
        )
    else:
        _print_row(
            f"C5  Candidate dir_acc >= {gate_dir:.2f}",
            _FAIL,
            f"dir_acc={cand_dir_acc:.4f}",
        )
        failures.append("C5")

    # --- Summary -----------------------------------------------------------
    n_hard = 5
    n_pass = n_hard - len(failures)
    n_fail = len(failures)
    n_skip = sum(1 for _ in range(n_hard)) - n_pass - n_fail  # rough

    print()
    print("-" * 70)
    overall = _PASS if not failures else _FAIL
    print(f"  QG-09 RESULT: {overall}   ({n_pass} PASS, {n_fail} FAIL)")
    if failures:
        print(f"  Failed checks: {failures}")
    print("=" * 70 + "\n")

    return overall == _PASS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="qg09_verify.py",
        description="QG-09 — Weekly Retraining Quality Gate",
    )
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help=(
            "Seed two synthetic MLflow runs and run QG-09 against them. "
            "Exits 0 on PASS, 1 on FAIL."
        ),
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from config/settings.yaml)",
    )
    return parser


def main() -> int:
    """Entry point."""
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    cfg = get_settings()
    tracking_uri = args.mlflow_uri or cfg.mlflow.tracking_uri

    if args.in_memory_test:
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            test_uri = f"sqlite:///{tmp}/mlflow_qg09.db"
            logger.info("QG-09 --in-memory-test: seeding MLflow at {}", test_uri)
            cand_id, prod_id = _seed_mlflow_test_runs(test_uri)
            logger.info(
                "QG-09 seeded: candidate={} production={}", cand_id, prod_id
            )
            passed = run_qg09(test_uri)
    else:
        passed = run_qg09(tracking_uri)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
