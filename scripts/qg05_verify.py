"""QG-05 Full Pipeline Verification -- Phase 5, Prompt 5.4.

Runs :class:`~src.training.trainer.ModelTrainer` end-to-end on synthetic
in-memory data and verifies the complete training pipeline output.

This is a *different* script from ``qg05_xgb_sanity.py`` (Prompt 5.1).
That script tests isolated XGBoost + WalkForwardCV.  This script tests the
**full** :class:`~src.training.trainer.ModelTrainer` pipeline including LSTM,
EnsembleModel, RegimeTrainer, and artefact persistence.

Checks (HARD -- all must pass):
  1.  ``n_folds >= 3``
  2.  ``mean_val_accuracy > 53%``
  3.  No NaN in ``fold_val_accuracies``
  4.  ``seed_used == settings.project.seed > 0``
  5.  ``scaler.pkl`` saved in output_dir
  6.  ``feature_columns.json`` saved in output_dir
  7.  XGBoost model artefacts present (``xgb_global/xgb_model.json``)
  8.  LSTM model artefacts present (``lstm/lstm_model.pt``)

Checks (ADVISORY -- logged but not exit-1):
  9.  ``feature_columns.json`` has >= 10 features
  10. Per-regime model artefacts present in ``regime_models/``
  11. Ensemble model artefacts present (``ensemble/ensemble_model.pkl``)
  12. ``fold_val_accuracies`` standard deviation < 0.20 (no fold collapse)

Usage::

    .venv/Scripts/python scripts/qg05_verify.py --in-memory-test

Exit codes:
    0 -- All HARD checks pass
    1 -- Any HARD check fails
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
from loguru import logger

from src.config import WalkForwardSettings, settings as _global_settings
from src.training.trainer import ModelTrainer, TrainingResult

# ---------------------------------------------------------------------------
# Import synthetic data builder from train.py (avoids code duplication)
# ---------------------------------------------------------------------------
# We add scripts/ dir to path so we can import from train module
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from train import build_in_memory_data  # type: ignore[import]
except ImportError:
    build_in_memory_data = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# QG-05 acceptance thresholds
# ---------------------------------------------------------------------------

_QG05_ACCURACY_THRESHOLD: float = 0.53
_QG05_MIN_FOLDS: int = 3
_QG05_MIN_FEATURES: int = 10        # advisory
_QG05_MAX_FOLD_STD: float = 0.20     # advisory

# In-memory test walk-forward config (fast)
_WF_CFG_FAST = WalkForwardSettings(
    training_window_days=15,
    validation_window_days=5,
    step_size_days=5,
    min_training_rows=200,
)


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check(
    label: str,
    passed: bool,
    detail: str = "",
    *,
    hard: bool = True,
) -> bool:
    """Log a single check result and return pass/fail.

    Args:
        label: Short check name.
        passed: ``True`` when the check passes.
        detail: Optional detail string appended to the log line.
        hard: If ``True`` a failure contributes to the exit code.

    Returns:
        The *passed* argument unchanged (for chaining).
    """
    tag = "HARD" if hard else "ADVISORY"
    pfx = "PASS" if passed else "FAIL"
    msg = f"  [{pfx}] ({tag}) {label}"
    if detail:
        msg += f" — {detail}"
    if passed:
        logger.info(msg)
    elif hard:
        logger.error(msg)
    else:
        logger.warning(msg)
    return passed


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------


def _run_training(output_dir: Path) -> TrainingResult | None:
    """Run ModelTrainer on synthetic in-memory data.

    Args:
        output_dir: Directory where artefacts will be saved.

    Returns:
        :class:`~src.training.trainer.TrainingResult` on success,
        ``None`` if data generation or training raises.
    """
    if build_in_memory_data is None:
        logger.error("qg05_verify: cannot import build_in_memory_data from train.py")
        return None

    try:
        logger.info("qg05_verify: generating synthetic in-memory data …")
        feature_df, regime_labels = build_in_memory_data()
    except Exception as exc:  # noqa: BLE001
        logger.error("qg05_verify: data generation failed — {}", exc)
        return None

    try:
        logger.info("qg05_verify: running ModelTrainer (run_hyperopt=False) …")
        trainer = ModelTrainer(
            output_dir=output_dir,
            walk_forward_cfg=_WF_CFG_FAST,
            run_hyperopt=False,
        )
        result = trainer.train_full(feature_df, regime_labels)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("qg05_verify: ModelTrainer.train_full failed — {}", exc)
        return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def run_qg05(output_dir: Path) -> bool:
    """Run all QG-05 checks against a completed training artefact directory.

    Args:
        output_dir: Directory where :class:`~src.training.trainer.ModelTrainer`
            saved its artefacts.

    Returns:
        ``True`` when ALL hard checks pass.
    """
    result = _run_training(output_dir)

    hard_failures: list[str] = []

    # -----------------------------------------------------------------------
    # Check 0 — training ran without error
    # -----------------------------------------------------------------------
    if result is None:
        logger.error("  [FAIL] (HARD) Training run — ModelTrainer returned None or raised")
        return False

    logger.info("  [PASS] (HARD) Training run — ModelTrainer completed without error")

    # -----------------------------------------------------------------------
    # TrainingResult checks (HARD)
    # -----------------------------------------------------------------------
    c1 = _check(
        "n_folds >= 3",
        result.n_folds >= _QG05_MIN_FOLDS,
        f"n_folds={result.n_folds}",
    )
    if not c1:
        hard_failures.append("n_folds < 3")

    c2 = _check(
        "mean_val_accuracy > 53%",
        result.mean_val_accuracy > _QG05_ACCURACY_THRESHOLD,
        f"mean_val_accuracy={result.mean_val_accuracy:.4f}",
    )
    if not c2:
        hard_failures.append("mean_val_accuracy <= 53%")

    c3 = _check(
        "no NaN in fold_val_accuracies",
        not any(np.isnan(a) for a in result.fold_val_accuracies),
        f"n_fold_accs={len(result.fold_val_accuracies)}",
    )
    if not c3:
        hard_failures.append("NaN in fold_val_accuracies")

    seed_ok = result.seed_used > 0 and result.seed_used == _global_settings.project.seed
    c4 = _check(
        "seed_used matches settings.project.seed > 0",
        seed_ok,
        f"seed_used={result.seed_used}, settings.seed={_global_settings.project.seed}",
    )
    if not c4:
        hard_failures.append("seed_used mismatch or zero")

    # -----------------------------------------------------------------------
    # Artefact file existence checks (HARD)
    # -----------------------------------------------------------------------
    scaler_path = output_dir / "scaler.pkl"
    c5 = _check(
        "scaler.pkl saved",
        scaler_path.exists(),
        str(scaler_path),
    )
    if not c5:
        hard_failures.append("scaler.pkl missing")

    fc_path = output_dir / "feature_columns.json"
    c6 = _check(
        "feature_columns.json saved",
        fc_path.exists(),
        str(fc_path),
    )
    if not c6:
        hard_failures.append("feature_columns.json missing")

    xgb_model_path = output_dir / "xgb_global" / "xgb_model.json"
    c7 = _check(
        "XGBoost model artefact present",
        xgb_model_path.exists(),
        str(xgb_model_path),
    )
    if not c7:
        hard_failures.append("xgb_global/xgb_model.json missing")

    lstm_model_path = output_dir / "lstm" / "lstm_model.pt"
    c8 = _check(
        "LSTM model artefact present",
        lstm_model_path.exists(),
        str(lstm_model_path),
    )
    if not c8:
        hard_failures.append("lstm/lstm_model.pt missing")

    # -----------------------------------------------------------------------
    # Advisory checks
    # -----------------------------------------------------------------------
    if fc_path.exists():
        try:
            with open(fc_path, encoding="utf-8") as fh:
                fc_data = json.load(fh)
            n_feats = fc_data.get("n_features", 0)
            _check(
                f"feature_columns.json has >= {_QG05_MIN_FEATURES} features",
                n_feats >= _QG05_MIN_FEATURES,
                f"n_features={n_feats}",
                hard=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("  [ADVISORY] feature_columns.json unreadable — {}", exc)

    regime_dir = output_dir / "regime_models"
    n_regime_models = len(list(regime_dir.glob("*/xgb_model.json"))) if regime_dir.exists() else 0
    _check(
        "per-regime model artefacts present",
        n_regime_models > 0,
        f"{n_regime_models} regime model(s) found",
        hard=False,
    )

    ensemble_path = output_dir / "ensemble" / "ensemble_model.pkl"
    _check(
        "Ensemble model artefact present",
        ensemble_path.exists(),
        str(ensemble_path),
        hard=False,
    )

    if len(result.fold_val_accuracies) > 1:
        fold_std = float(np.std(result.fold_val_accuracies))
        _check(
            f"fold_val_accuracies std < {_QG05_MAX_FOLD_STD} (no fold collapse)",
            fold_std < _QG05_MAX_FOLD_STD,
            f"std={fold_std:.4f}",
            hard=False,
        )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------
    if hard_failures:
        logger.error(
            "qg05_verify: FAIL — {} hard check(s) failed: {}",
            len(hard_failures),
            hard_failures,
        )
        return False

    logger.info(
        "qg05_verify: ALL {} HARD CHECKS PASS",
        8 + 1,  # 8 artefact/metric checks + 1 training-run check
    )
    return True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="QG-05: Full ModelTrainer pipeline verification"
    )
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help="Generate synthetic data and run pipeline in-memory (CI mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save model artefacts; uses temp dir when omitted",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run QG-05 full pipeline verification.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on pass, 1 on fail.
    """
    args = _parse_args(argv)

    sep = "=" * 64
    logger.info(sep)
    logger.info("QG-05: Full ModelTrainer Pipeline Verification")
    logger.info(sep)

    if not args.in_memory_test:
        logger.warning(
            "qg05_verify: --in-memory-test flag not set; "
            "this script requires synthetic data mode to run without live data. "
            "Add --in-memory-test to run."
        )
        return 1

    use_temp = args.output_dir is None
    tmp_ctx = tempfile.TemporaryDirectory() if use_temp else None

    try:
        output_dir = Path(tmp_ctx.name) if tmp_ctx else Path(args.output_dir)
        passed = run_qg05(output_dir)
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()

    if passed:
        logger.info("QG-05 RESULT: PASS")
        logger.info(sep)
        return 0

    logger.error("QG-05 RESULT: FAIL")
    logger.info(sep)
    return 1


if __name__ == "__main__":
    sys.exit(main())
