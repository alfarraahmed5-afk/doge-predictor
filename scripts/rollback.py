"""rollback.py — Rollback the inference engine to the previous production model.

Procedure:

1. Load the MLflow run tagged ``'previous-production'``.
2. Re-tag it as ``'production'``.
3. Demote the current ``'production'`` run to ``'rollback-{timestamp}'``.
4. Copy the previous-production model artefacts to the models directory.
5. Verify that the health check endpoint responds with HTTP 200.
6. Print a confirmation summary.

Usage::

    python scripts/rollback.py --models-dir models/ --health-url http://localhost:8000/health
    python scripts/rollback.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEALTH_TIMEOUT_S: int = 30   # seconds to wait for health check after rollback
_HEALTH_POLL_S: float = 2.0   # polling interval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_mlflow_client(tracking_uri: str) -> "Any":
    """Return an MLflow MlflowClient or raise ImportError.

    Args:
        tracking_uri: MLflow tracking URI.

    Returns:
        :class:`mlflow.tracking.MlflowClient` instance.

    Raises:
        ImportError: If MLflow is not installed.
    """
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def _find_run_with_tag(
    client: "Any",
    experiment_name: str,
    stage_tag: str,
) -> "Any | None":
    """Return the most recent MLflow run with ``tags.stage == stage_tag``.

    Args:
        client: MLflow MlflowClient instance.
        experiment_name: MLflow experiment name.
        stage_tag: Value of the ``stage`` tag to match.

    Returns:
        Run object, or ``None`` if not found.
    """
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.stage = '{stage_tag}'",
        max_results=1,
        order_by=["attribute.start_time DESC"],
    )
    return runs[0] if runs else None


def _download_artefacts(
    client: "Any",
    run_id: str,
    dest_dir: Path,
) -> bool:
    """Download all model artefacts from an MLflow run to *dest_dir*.

    Args:
        client: MLflow MlflowClient.
        run_id: Source run ID.
        dest_dir: Destination directory (created if absent).

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:
        import mlflow  # noqa: PLC0415

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            dst_path=str(dest_dir),
        )
        logger.info("rollback: artefacts downloaded to {}", local_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("rollback: artefact download failed — {}", exc)
        return False


def _check_health(health_url: str, timeout_s: int = _HEALTH_TIMEOUT_S) -> bool:
    """Poll *health_url* until it returns HTTP 200 or *timeout_s* elapses.

    Args:
        health_url: URL of the health check endpoint.
        timeout_s: Maximum seconds to wait.

    Returns:
        ``True`` when the endpoint returns HTTP 200 within the timeout.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    logger.info("rollback: health check PASS (HTTP 200 from {})", health_url)
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_HEALTH_POLL_S)

    logger.error(
        "rollback: health check FAIL — no HTTP 200 from {} within {}s",
        health_url,
        timeout_s,
    )
    return False


# ---------------------------------------------------------------------------
# Main rollback logic
# ---------------------------------------------------------------------------


def run_rollback(
    models_dir: Path,
    mlflow_tracking_uri: str,
    health_url: str | None,
    dry_run: bool,
) -> bool:
    """Execute the full rollback procedure.

    Args:
        models_dir: Directory where production model artefacts should land.
        mlflow_tracking_uri: MLflow tracking URI.
        health_url: Optional health check URL to verify after rollback.
        dry_run: When ``True``, log actions without executing destructive steps.

    Returns:
        ``True`` on success, ``False`` on any failure.
    """
    print("\n" + "=" * 65)
    print("  Rollback — reverting to previous production model")
    print("=" * 65)

    if dry_run:
        print("  [DRY RUN] — no changes will be made\n")

    # ------------------------------------------------------------------
    # Step 1: Find 'previous-production' and current 'production' runs
    # ------------------------------------------------------------------
    try:
        client = _get_mlflow_client(mlflow_tracking_uri)
    except ImportError:
        logger.error("rollback: MLflow is not installed")
        return False

    prev_run = _find_run_with_tag(client, "doge_predictor", "previous-production")
    curr_run = _find_run_with_tag(client, "doge_predictor", "production")

    if prev_run is None:
        print("  [FAIL]  No 'previous-production' run found in MLflow.")
        print("          Run train.py first so a production model exists.")
        return False

    prev_acc = prev_run.data.metrics.get("mean_val_accuracy", 0.0)
    curr_acc = curr_run.data.metrics.get("mean_val_accuracy", 0.0) if curr_run else None

    print(f"\n  Previous-production run: {prev_run.info.run_id}  acc={prev_acc:.4f}")
    print(
        "  Current production run : "
        + (
            f"{curr_run.info.run_id}  acc={curr_acc:.4f}"
            if curr_run is not None
            else "N/A"
        )
    )
    print()

    # ------------------------------------------------------------------
    # Step 2: Re-tag previous-production as 'production'
    # ------------------------------------------------------------------
    print("  [1/4] Re-tagging previous-production as 'production' ...")
    if not dry_run:
        try:
            client.set_tag(prev_run.info.run_id, "stage", "production")
            logger.info(
                "rollback: run {} re-tagged as 'production'",
                prev_run.info.run_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("rollback: MLflow tag update failed — {}", exc)
            return False
    print("       Done.")

    # ------------------------------------------------------------------
    # Step 3: Demote current 'production' to 'rollback-{timestamp}'
    # ------------------------------------------------------------------
    if curr_run is not None:
        rollback_tag = f"rollback-{int(time.time())}"
        print(f"  [2/4] Demoting current production to '{rollback_tag}' ...")
        if not dry_run:
            try:
                client.set_tag(curr_run.info.run_id, "stage", rollback_tag)
                logger.info(
                    "rollback: run {} demoted to '{}'",
                    curr_run.info.run_id,
                    rollback_tag,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("rollback: could not demote current run — {}", exc)
        print("       Done.")
    else:
        print("  [2/4] No current 'production' run to demote — skipping.")

    # ------------------------------------------------------------------
    # Step 4: Download previous-production artefacts to models_dir
    # ------------------------------------------------------------------
    print(f"  [3/4] Restoring model artefacts to {models_dir} ...")
    if not dry_run:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        ok = _download_artefacts(client, prev_run.info.run_id, models_dir)
        if not ok:
            logger.error("rollback: artefact restore failed")
            return False
    print("       Done.")

    # ------------------------------------------------------------------
    # Step 5: Health check
    # ------------------------------------------------------------------
    print("  [4/4] Verifying health check endpoint ...")
    if health_url and not dry_run:
        healthy = _check_health(health_url)
        if not healthy:
            print(
                f"       [WARN] Health check at {health_url} did not return 200 "
                f"within {_HEALTH_TIMEOUT_S}s."
            )
            print(
                "       The server may need restarting to pick up the restored model."
            )
        else:
            print(f"       Health check PASS: {health_url} returned HTTP 200.")
    elif health_url is None:
        print("       Health check skipped (no --health-url provided).")
    else:
        print("       Health check skipped (dry-run mode).")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("  Rollback complete.")
    print(f"    Restored run : {prev_run.info.run_id}")
    print(f"    New tag      : production")
    if curr_run:
        print(f"    Demoted run  : {curr_run.info.run_id} -> {rollback_tag}")  # type: ignore[possibly-undefined]
    print("=" * 65 + "\n")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="rollback.py",
        description="Rollback to the previous production model.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to restore model artefacts into (default: models/)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: from config/settings.yaml)",
    )
    parser.add_argument(
        "--health-url",
        type=str,
        default=None,
        help=(
            "Health check URL to verify after rollback "
            "(e.g. http://localhost:8000/health). "
            "Omit to skip the health check."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without executing destructive steps.",
    )
    return parser


def main() -> int:
    """Entry point."""
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args()

    cfg = get_settings()
    tracking_uri = args.mlflow_uri or cfg.mlflow.tracking_uri

    ok = run_rollback(
        models_dir=args.models_dir,
        mlflow_tracking_uri=tracking_uri,
        health_url=args.health_url,
        dry_run=args.dry_run,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
