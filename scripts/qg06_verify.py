"""QG-06 Model Loading + Inference Check -- Phase 5, Prompt 5.4.

Loads all artefacts saved by :class:`~src.training.trainer.ModelTrainer` and
runs a forward-pass inference check on synthetic test data.  This script is
the bridge between training (Phase 5) and backtesting (Phase 6): it confirms
that every saved model can be reloaded and produces valid probability outputs.

In ``--in-memory-test`` mode the script runs a fast training pass first to
populate a temporary output directory, then immediately verifies the saved
artefacts.  Alternatively you can point ``--models-dir`` at a previously
trained model directory.

Checks (HARD -- all must pass):
  1.  :class:`~src.models.xgb_model.XGBoostModel` loads from ``xgb_global/``
  2.  XGBoostModel ``predict_proba`` output shape == ``(n_samples,)``
  3.  XGBoostModel ``predict_proba`` values all in ``[0.0, 1.0]``
  4.  :class:`~src.models.lstm_model.LSTMModel` loads from ``lstm/``
  5.  LSTMModel ``predict_proba`` output shape == ``(n_samples,)``
  6.  LSTMModel ``predict_proba`` values all in ``[0.0, 1.0]``
  7.  :class:`~src.models.ensemble.EnsembleModel` loads from ``ensemble/``
  8.  EnsembleModel ``predict_proba`` on ``(n, 3)`` meta-features shape == ``(n,)``
  9.  EnsembleModel ``predict_proba`` values all in ``[0.0, 1.0]``
  10. :class:`~src.training.scaler.FoldScaler` loads from ``scaler.pkl``
  11. FoldScaler ``transform`` produces correct output shape
  12. ``predict_signal`` returns a valid signal string for all models

Checks (ADVISORY -- logged but not exit-1):
  13. :class:`~src.models.regime_router.RegimeRouter` construction from any
      ``regime_models/`` sub-directories succeeds
  14. XGBoostModel output dtype is ``float64``
  15. LSTMModel is in eval mode after ``load``

Usage::

    # Run training first, then verify inference:
    .venv/Scripts/python scripts/qg06_verify.py --in-memory-test

    # Verify an already-trained model directory:
    .venv/Scripts/python scripts/qg06_verify.py --models-dir models/

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

from src.config import WalkForwardSettings
from src.models.base_model import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.xgb_model import XGBoostModel
from src.training.scaler import FoldScaler

# ---------------------------------------------------------------------------
# Import synthetic data builder from train.py
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from train import build_in_memory_data  # type: ignore[import]
except ImportError:
    build_in_memory_data = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_TEST_SAMPLES: int = 50  # rows to run through each model in inference

_WF_CFG_FAST = WalkForwardSettings(
    training_window_days=15,
    validation_window_days=5,
    step_size_days=5,
    min_training_rows=200,
)

_VALID_SIGNALS: frozenset[str] = frozenset({SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD})


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
        The *passed* argument unchanged.
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
# Training helper
# ---------------------------------------------------------------------------


def _train_to_output_dir(output_dir: Path) -> bool:
    """Run a fast training pass and save artefacts to *output_dir*.

    Args:
        output_dir: Directory to write all model artefacts.

    Returns:
        ``True`` on success, ``False`` on error.
    """
    if build_in_memory_data is None:
        logger.error("qg06_verify: cannot import build_in_memory_data from train.py")
        return False

    try:
        from src.training.trainer import ModelTrainer  # noqa: PLC0415

        logger.info("qg06_verify: generating synthetic data and training models …")
        feature_df, regime_labels = build_in_memory_data()
        trainer = ModelTrainer(
            output_dir=output_dir,
            walk_forward_cfg=_WF_CFG_FAST,
            run_hyperopt=False,
        )
        trainer.train_full(feature_df, regime_labels)
        logger.info("qg06_verify: training complete → {}", output_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("qg06_verify: training failed — {}", exc)
        return False


# ---------------------------------------------------------------------------
# Inference verifier
# ---------------------------------------------------------------------------


def run_qg06(models_dir: Path) -> bool:
    """Run all QG-06 inference checks against *models_dir*.

    Args:
        models_dir: Root directory written by
            :class:`~src.training.trainer.ModelTrainer`.

    Returns:
        ``True`` when ALL hard checks pass.
    """
    hard_failures: list[str] = []

    # -----------------------------------------------------------------------
    # Determine feature count from feature_columns.json
    # -----------------------------------------------------------------------
    fc_path = models_dir / "feature_columns.json"
    if fc_path.exists():
        try:
            with open(fc_path, encoding="utf-8") as fh:
                fc_data = json.load(fh)
            n_features: int = int(fc_data.get("n_features", 10))
            logger.info(
                "qg06_verify: feature_columns.json — {} features", n_features
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "qg06_verify: could not read feature_columns.json — {} "
                "(falling back to n_features=10)",
                exc,
            )
            n_features = 10
    else:
        logger.warning(
            "qg06_verify: feature_columns.json not found in {} "
            "(using n_features=10 for synthetic test data)",
            models_dir,
        )
        n_features = 10

    # Synthetic test data — deterministic random, same shape as training features
    rng = np.random.default_rng(seed=999)
    X_test = rng.standard_normal((_N_TEST_SAMPLES, n_features)).astype(np.float32)

    # -----------------------------------------------------------------------
    # Check 1-3: XGBoostModel
    # -----------------------------------------------------------------------
    xgb_model = XGBoostModel()
    xgb_loaded = False
    xgb_path = models_dir / "xgb_global"

    c1_load = _check(
        "XGBoostModel loads from xgb_global/",
        _try_load(xgb_model, xgb_path),
        str(xgb_path),
    )
    if not c1_load:
        hard_failures.append("XGBoostModel load failed")
    else:
        xgb_loaded = True

    if xgb_loaded:
        try:
            proba = xgb_model.predict_proba(X_test.astype(np.float64))
            c2 = _check(
                "XGBoostModel predict_proba shape == (n_samples,)",
                proba.shape == (_N_TEST_SAMPLES,),
                f"shape={proba.shape}",
            )
            if not c2:
                hard_failures.append("XGBoostModel predict_proba wrong shape")

            c3 = _check(
                "XGBoostModel predict_proba values in [0, 1]",
                bool(np.all(proba >= 0.0) and np.all(proba <= 1.0)),
                f"min={proba.min():.4f} max={proba.max():.4f}",
            )
            if not c3:
                hard_failures.append("XGBoostModel predict_proba out of [0,1]")

            # Advisory: dtype
            _check(
                "XGBoostModel output dtype is float64",
                proba.dtype == np.float64,
                f"dtype={proba.dtype}",
                hard=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  [FAIL] (HARD) XGBoostModel predict_proba raised — {}", exc)
            hard_failures.append("XGBoostModel predict_proba raised")

    # -----------------------------------------------------------------------
    # Check 4-6: LSTMModel
    # -----------------------------------------------------------------------
    lstm_model = LSTMModel()
    lstm_loaded = False
    lstm_path = models_dir / "lstm"

    c4_load = _check(
        "LSTMModel loads from lstm/",
        _try_load(lstm_model, lstm_path),
        str(lstm_path),
    )
    if not c4_load:
        hard_failures.append("LSTMModel load failed")
    else:
        lstm_loaded = True

    if lstm_loaded:
        try:
            proba_lstm = lstm_model.predict_proba(X_test.astype(np.float64))
            c5 = _check(
                "LSTMModel predict_proba shape == (n_samples,)",
                proba_lstm.shape == (_N_TEST_SAMPLES,),
                f"shape={proba_lstm.shape}",
            )
            if not c5:
                hard_failures.append("LSTMModel predict_proba wrong shape")

            c6 = _check(
                "LSTMModel predict_proba values in [0, 1]",
                bool(np.all(proba_lstm >= 0.0) and np.all(proba_lstm <= 1.0)),
                f"min={proba_lstm.min():.4f} max={proba_lstm.max():.4f}",
            )
            if not c6:
                hard_failures.append("LSTMModel predict_proba out of [0,1]")

            # Advisory: eval mode
            try:
                _check(
                    "LSTMModel is in eval mode after load",
                    not lstm_model._network.training,  # type: ignore[attr-defined]
                    hard=False,
                )
            except AttributeError:
                pass

        except Exception as exc:  # noqa: BLE001
            logger.error("  [FAIL] (HARD) LSTMModel predict_proba raised — {}", exc)
            hard_failures.append("LSTMModel predict_proba raised")

    # -----------------------------------------------------------------------
    # Check 7-9: EnsembleModel
    # -----------------------------------------------------------------------
    ensemble_model = EnsembleModel()
    ensemble_loaded = False
    ensemble_path = models_dir / "ensemble"

    c7_load = _check(
        "EnsembleModel loads from ensemble/",
        _try_load(ensemble_model, ensemble_path),
        str(ensemble_path),
    )
    if not c7_load:
        hard_failures.append("EnsembleModel load failed")
    else:
        ensemble_loaded = True

    if ensemble_loaded:
        try:
            # Meta-features: [lstm_prob, xgb_prob, regime_encoded]
            meta_X = rng.uniform(0.0, 1.0, (_N_TEST_SAMPLES, 3))
            meta_X[:, 2] = rng.integers(0, 5, _N_TEST_SAMPLES).astype(float)
            proba_ens = ensemble_model.predict_proba(meta_X)

            c8 = _check(
                "EnsembleModel predict_proba shape == (n_samples,)",
                proba_ens.shape == (_N_TEST_SAMPLES,),
                f"shape={proba_ens.shape}",
            )
            if not c8:
                hard_failures.append("EnsembleModel predict_proba wrong shape")

            c9 = _check(
                "EnsembleModel predict_proba values in [0, 1]",
                bool(np.all(proba_ens >= 0.0) and np.all(proba_ens <= 1.0)),
                f"min={proba_ens.min():.4f} max={proba_ens.max():.4f}",
            )
            if not c9:
                hard_failures.append("EnsembleModel predict_proba out of [0,1]")

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "  [FAIL] (HARD) EnsembleModel predict_proba raised — {}", exc
            )
            hard_failures.append("EnsembleModel predict_proba raised")

    # -----------------------------------------------------------------------
    # Check 10-11: FoldScaler
    # -----------------------------------------------------------------------
    scaler = FoldScaler()
    scaler_path = models_dir / "scaler.pkl"
    scaler_loaded = False

    c10_load = _check(
        "FoldScaler loads from scaler.pkl",
        _try_load(scaler, models_dir),
        str(scaler_path),
    )
    if not c10_load:
        hard_failures.append("FoldScaler load failed")
    else:
        scaler_loaded = True

    if scaler_loaded:
        try:
            X_scaled = scaler.transform(X_test.astype(np.float64))
            c11 = _check(
                "FoldScaler transform produces correct shape",
                X_scaled.shape == X_test.shape,
                f"in={X_test.shape} out={X_scaled.shape}",
            )
            if not c11:
                hard_failures.append("FoldScaler transform wrong shape")
        except Exception as exc:  # noqa: BLE001
            logger.error("  [FAIL] (HARD) FoldScaler transform raised — {}", exc)
            hard_failures.append("FoldScaler transform raised")

    # -----------------------------------------------------------------------
    # Check 12: predict_signal for each loaded model
    # -----------------------------------------------------------------------
    _check_predict_signal(
        "XGBoostModel predict_signal",
        xgb_model if xgb_loaded else None,
        X_test.astype(np.float64),
        hard_failures,
    )
    _check_predict_signal(
        "LSTMModel predict_signal",
        lstm_model if lstm_loaded else None,
        X_test.astype(np.float64),
        hard_failures,
    )
    if ensemble_loaded:
        meta_X_check = rng.uniform(0.0, 1.0, (_N_TEST_SAMPLES, 3))
        meta_X_check[:, 2] = rng.integers(0, 5, _N_TEST_SAMPLES).astype(float)
        _check_predict_signal(
            "EnsembleModel predict_signal",
            ensemble_model,
            meta_X_check,
            hard_failures,
        )

    # -----------------------------------------------------------------------
    # Advisory: RegimeRouter
    # -----------------------------------------------------------------------
    _check_regime_router(models_dir)

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------
    if hard_failures:
        logger.error(
            "qg06_verify: FAIL — {} hard check(s) failed: {}",
            len(hard_failures),
            hard_failures,
        )
        return False

    logger.info("qg06_verify: ALL HARD CHECKS PASS")
    return True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _try_load(model: object, path: Path) -> bool:
    """Call ``model.load(path)`` and return ``True`` on success.

    Args:
        model: An object that exposes a ``load(path)`` method.
        path: Path passed to ``load``.

    Returns:
        ``True`` when ``load`` completed without raising.
    """
    try:
        model.load(path)  # type: ignore[union-attr]
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("_try_load failed for {} — {}", path, exc)
        return False


def _check_predict_signal(
    label: str,
    model: object | None,
    X: np.ndarray,
    hard_failures: list[str],
) -> None:
    """Assert that ``model.predict_signal(X)`` returns valid signal strings.

    Args:
        label: Human-readable check label.
        model: Fitted model (or ``None`` if loading failed).
        X: Feature array to pass to ``predict_signal``.
        hard_failures: List to append failure labels to.
    """
    if model is None:
        logger.warning("  [SKIP] (HARD) {} — model not loaded", label)
        return

    try:
        signals = model.predict_signal(X)  # type: ignore[union-attr]
        # predict_signal may return a single string or an array of strings
        if isinstance(signals, str):
            signals = [signals]
        else:
            signals = list(signals)

        all_valid = all(s in _VALID_SIGNALS for s in signals)
        passed = _check(
            f"{label} returns valid signal",
            all_valid,
            f"got={set(signals)}",
        )
        if not passed:
            hard_failures.append(f"{label} invalid signal")
    except Exception as exc:  # noqa: BLE001
        logger.error("  [FAIL] (HARD) {} raised — {}", label, exc)
        hard_failures.append(f"{label} raised")


def _check_regime_router(models_dir: Path) -> None:
    """Advisory: attempt to build a RegimeRouter from any saved regime models.

    Args:
        models_dir: Root directory containing ``regime_models/`` sub-directory.
    """
    regime_dir = models_dir / "regime_models"
    if not regime_dir.exists():
        logger.warning(
            "  [SKIP] (ADVISORY) RegimeRouter — regime_models/ not found in {}",
            models_dir,
        )
        return

    regime_subdirs = [d for d in regime_dir.iterdir() if d.is_dir()]
    if not regime_subdirs:
        logger.warning(
            "  [SKIP] (ADVISORY) RegimeRouter — no sub-directories in regime_models/"
        )
        return

    try:
        from src.models.regime_router import RegimeRouter  # noqa: PLC0415

        regime_models: dict[str, XGBoostModel] = {}
        for subdir in regime_subdirs:
            xgb_m = XGBoostModel()
            if _try_load(xgb_m, subdir):
                regime_models[subdir.name] = xgb_m

        if regime_models:
            router = RegimeRouter(regime_models=regime_models)
            _check(
                f"RegimeRouter constructed with {len(regime_models)} regime model(s)",
                len(router.available_regimes()) == len(regime_models),
                f"regimes={router.available_regimes()}",
                hard=False,
            )
            # Test that routing works for a known regime
            first_regime = router.available_regimes()[0]
            routed = router.route(first_regime)
            _check(
                f"RegimeRouter.route('{first_regime}') returns correct model",
                routed is regime_models[first_regime],
                hard=False,
            )
        else:
            logger.warning(
                "  [SKIP] (ADVISORY) RegimeRouter — no regime models loaded"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("  [SKIP] (ADVISORY) RegimeRouter construction failed — {}", exc)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="QG-06: Model loading + inference sanity check"
    )
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help="Run a fast training pass first, then verify inference (CI mode)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Load models from this directory instead of running training",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run QG-06 model loading + inference checks.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on pass, 1 on fail.
    """
    args = _parse_args(argv)

    sep = "=" * 64
    logger.info(sep)
    logger.info("QG-06: Model Loading + Inference Check")
    logger.info(sep)

    if args.models_dir is not None:
        # Use a pre-trained model directory
        models_dir = Path(args.models_dir)
        if not models_dir.exists():
            logger.error("qg06_verify: --models-dir '{}' does not exist", models_dir)
            return 1
        logger.info("qg06_verify: loading models from {}", models_dir)
        passed = run_qg06(models_dir)

    elif args.in_memory_test:
        # Run training, then verify
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            if not _train_to_output_dir(output_dir):
                logger.error("qg06_verify: training step failed — aborting QG-06")
                return 1
            passed = run_qg06(output_dir)

    else:
        logger.warning(
            "qg06_verify: neither --in-memory-test nor --models-dir given. "
            "Use --in-memory-test for CI or --models-dir PATH for offline check."
        )
        return 1

    if passed:
        logger.info("QG-06 RESULT: PASS")
        logger.info(sep)
        return 0

    logger.error("QG-06 RESULT: FAIL")
    logger.info(sep)
    return 1


if __name__ == "__main__":
    sys.exit(main())
