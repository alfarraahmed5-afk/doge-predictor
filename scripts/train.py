"""CLI entry point for the full DOGE prediction model training pipeline.

Runs :class:`~src.training.trainer.ModelTrainer` end-to-end on either:
  - Production data from ``data/features/primary/`` (default), or
  - Synthetic in-memory fixture data (``--in-memory-test`` flag for CI).

The ``--in-memory-test`` mode generates 2 000 rows of AR(1) OHLCV, builds a
full feature matrix via :class:`~src.features.pipeline.FeaturePipeline`, runs
:class:`~src.regimes.classifier.DogeRegimeClassifier`, then trains all models.
This is the same pattern used in ``qg05_xgb_sanity.py`` and ``qg03_verify.py``.

Usage::

    # In-memory test (CI / fixture data -- no live Binance connection needed)
    .venv/Scripts/python scripts/train.py --in-memory-test --no-hyperopt

    # Production run (requires data/features/primary/ to be populated)
    .venv/Scripts/python scripts/train.py --output-dir models/

Exit codes:
    0 -- Training succeeded and all QG-05 assertions passed
    1 -- Training failed or a QG-05 assertion failed
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.config import WalkForwardSettings
from src.features.pipeline import FeaturePipeline, _PASSTHROUGH_COLS
from src.regimes.classifier import DogeRegimeClassifier
from src.training.trainer import ModelTrainer, TrainingResult

# ---------------------------------------------------------------------------
# Synthetic-data constants (shared with qg05_xgb_sanity.py)
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_4H: int = 4 * _MS_PER_HOUR
_MS_PER_8H: int = 8 * _MS_PER_HOUR
_MS_PER_1D: int = 24 * _MS_PER_HOUR
_T0_MS: int = 1_640_995_200_000          # 2022-01-01 00:00:00 UTC

_N_1H_ROWS: int = 2_000
_N_4H_ROWS: int = 600
_N_1D_ROWS: int = 90
_N_FUNDING_ROWS: int = 300

# QG-05 acceptance threshold
_QG05_ACCURACY_THRESHOLD: float = 0.53


# ---------------------------------------------------------------------------
# Synthetic data generators (identical to qg05_xgb_sanity.py generators)
# ---------------------------------------------------------------------------


def _make_ohlcv_1h(n: int, start_ms: int, seed: int = 42) -> pd.DataFrame:
    """Generate *n* rows of 1h DOGE OHLCV with AR(1) autocorrelation=0.9."""
    rng = np.random.default_rng(seed)
    open_times = np.array([start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    lr = np.zeros(n)
    lr[0] = rng.normal(0.001, 0.008)
    for i in range(1, n):
        lr[i] = 0.9 * lr[i - 1] + rng.normal(0.0, 0.005)
    close = (0.10 * np.exp(np.cumsum(lr))).clip(0.001)
    noise = rng.uniform(0.002, 0.008, n)
    high = (close * (1.0 + noise)).clip(0.001)
    low = (close * (1.0 - noise)).clip(0.001)
    open_ = (close * (1.0 + rng.normal(0.0, 0.003, n))).clip(0.001)
    volume = rng.uniform(1e7, 5e8, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_ohlcv_4h(n: int, start_ms: int, seed: int = 43) -> pd.DataFrame:
    """Generate *n* rows of 4h DOGE OHLCV."""
    rng = np.random.default_rng(seed)
    open_times = np.array([start_ms + i * _MS_PER_4H for i in range(n)], dtype=np.int64)
    log_ret = rng.normal(0.001, 0.02, n)
    close = (0.10 * np.exp(np.cumsum(log_ret))).clip(0.001)
    noise = rng.uniform(0.005, 0.015, n)
    high = (close * (1.0 + noise)).clip(0.001)
    low = (close * (1.0 - noise)).clip(0.001)
    open_ = (close * (1.0 + rng.normal(0.0, 0.005, n))).clip(0.001)
    volume = rng.uniform(4e7, 2e9, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_ohlcv_1d(n: int, start_ms: int, seed: int = 44) -> pd.DataFrame:
    """Generate *n* rows of 1d DOGE OHLCV."""
    rng = np.random.default_rng(seed)
    open_times = np.array([start_ms + i * _MS_PER_1D for i in range(n)], dtype=np.int64)
    log_ret = rng.normal(0.003, 0.04, n)
    close = (0.10 * np.exp(np.cumsum(log_ret))).clip(0.001)
    noise = rng.uniform(0.01, 0.04, n)
    high = (close * (1.0 + noise)).clip(0.001)
    low = (close * (1.0 - noise)).clip(0.001)
    open_ = (close * (1.0 + rng.normal(0.0, 0.01, n))).clip(0.001)
    volume = rng.uniform(1e9, 1e10, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_btc_1h(
    n: int, start_ms: int, doge_close: np.ndarray, seed: int = 45
) -> pd.DataFrame:
    """Generate BTC 1h data correlated with DOGE (corr ~0.7)."""
    rng = np.random.default_rng(seed)
    open_times = np.array([start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    doge_lr = np.diff(np.log(doge_close), prepend=np.log(doge_close[0]))
    btc_lr = 0.7 * doge_lr + 0.3 * rng.normal(0.0, 0.008, n)
    close = (40_000.0 * np.exp(np.cumsum(btc_lr))).clip(1.0)
    noise = rng.uniform(0.002, 0.010, n)
    high = (close * (1.0 + noise)).clip(1.0)
    low = (close * (1.0 - noise)).clip(1.0)
    open_ = (close * (1.0 + rng.normal(0.0, 0.002, n))).clip(1.0)
    volume = rng.uniform(1e9, 1e11, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "BTCUSDT",
        "era": "training",
    })


def _make_dogebtc_1h(
    n: int,
    start_ms: int,
    doge_close: np.ndarray,
    btc_close: np.ndarray,
    seed: int = 46,
) -> pd.DataFrame:
    """Generate DOGEBTC 1h data as doge_close / btc_close."""
    rng = np.random.default_rng(seed)
    open_times = np.array([start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64)
    close = (doge_close / btc_close).clip(1e-8)
    noise = rng.uniform(0.001, 0.005, n)
    high = (close * (1.0 + noise)).clip(1e-8)
    low = (close * (1.0 - noise)).clip(1e-8)
    open_ = (close * (1.0 + rng.normal(0.0, 0.002, n))).clip(1e-8)
    volume = rng.uniform(1e6, 1e8, n)
    return pd.DataFrame({
        "open_time": open_times,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "DOGEBTC",
        "era": "training",
    })


def _make_funding(n: int, start_ms: int, seed: int = 47) -> pd.DataFrame:
    """Generate *n* rows of 8h funding rate data."""
    rng = np.random.default_rng(seed)
    timestamps = np.array([start_ms + i * _MS_PER_8H for i in range(n)], dtype=np.int64)
    funding_rate = rng.normal(0.0001, 0.0003, n).clip(-0.002, 0.003)
    return pd.DataFrame({
        "timestamp_ms": timestamps,
        "funding_rate": funding_rate,
        "symbol": "DOGEUSDT",
    })


# ---------------------------------------------------------------------------
# In-memory data pipeline
# ---------------------------------------------------------------------------


def build_in_memory_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic OHLCV + feature matrix + regime labels.

    Returns:
        Tuple of ``(feature_df, regime_labels)`` ready for
        :class:`~src.training.trainer.ModelTrainer`.
    """
    logger.info("train.py: generating synthetic in-memory data ({} 1h rows) ...", _N_1H_ROWS)

    doge_1h = _make_ohlcv_1h(_N_1H_ROWS, _T0_MS, seed=42)
    doge_4h = _make_ohlcv_4h(_N_4H_ROWS, _T0_MS, seed=43)
    doge_1d = _make_ohlcv_1d(_N_1D_ROWS, _T0_MS, seed=44)
    btc_1h = _make_btc_1h(_N_1H_ROWS, _T0_MS, doge_1h["close"].values, seed=45)
    dogebtc_1h = _make_dogebtc_1h(
        _N_1H_ROWS, _T0_MS,
        doge_1h["close"].values, btc_1h["close"].values, seed=46,
    )
    funding = _make_funding(_N_FUNDING_ROWS, _T0_MS, seed=47)

    # Classify regimes (index by open_time)
    clf = DogeRegimeClassifier()
    regimes: pd.Series = clf.classify(doge_1h, btc_df=btc_1h)
    regimes.index = doge_1h["open_time"].values
    dist = clf.get_regime_distribution(regimes)
    logger.info("  Regime distribution: {}", {k: f"{v:.2%}" for k, v in dist.items()})

    # Build full feature matrix
    pipe = FeaturePipeline(run_id="train_in_memory")
    feature_df = pipe.compute_all_features(
        doge_1h=doge_1h,
        btc_1h=btc_1h,
        dogebtc_1h=dogebtc_1h,
        funding=funding,
        doge_4h=doge_4h,
        doge_1d=doge_1d,
        regimes=regimes,
        min_rows_override=300,
    )
    logger.info(
        "  Feature matrix: {} rows x {} cols", len(feature_df), len(feature_df.columns)
    )

    # Align regime labels to feature_df index (open_time-keyed)
    regime_labels = pd.Series(
        [str(regimes.get(t, "RANGING_LOW_VOL")) for t in feature_df["open_time"]],
        index=feature_df.index,
        name="regime_label",
    )

    return feature_df, regime_labels


def load_features_from_disk(
    features_dir: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature matrix and regime labels from Parquet files.

    Args:
        features_dir: Directory containing ``features_*.parquet`` and
            optionally ``regime_labels.parquet`` files.

    Returns:
        Tuple of ``(feature_df, regime_labels)``.

    Raises:
        FileNotFoundError: If no feature Parquet files are found.
    """
    parquet_files = sorted(features_dir.glob("features_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"train.py: no features_*.parquet files found in {features_dir}"
        )

    feature_path = parquet_files[-1]
    logger.info("train.py: loading features from {}", feature_path)
    feature_df = pd.read_parquet(feature_path)

    regime_path = features_dir / "regime_labels.parquet"
    if regime_path.exists():
        regime_labels = pd.read_parquet(regime_path)["regime_label"]
        regime_labels.index = feature_df.index
    else:
        logger.warning(
            "train.py: regime_labels.parquet not found -- defaulting to RANGING_LOW_VOL"
        )
        regime_labels = pd.Series(
            ["RANGING_LOW_VOL"] * len(feature_df), index=feature_df.index
        )

    return feature_df, regime_labels


# ---------------------------------------------------------------------------
# Summary and QG-05 assertions
# ---------------------------------------------------------------------------


def _print_summary(result: TrainingResult) -> None:
    """Print a formatted training summary report.

    Args:
        result: Completed :class:`~src.training.trainer.TrainingResult`.
    """
    sep = "=" * 64
    logger.info(sep)
    logger.info("TRAINING SUMMARY REPORT")
    logger.info(sep)
    logger.info("  Walk-forward folds:  {}", result.n_folds)
    logger.info(
        "  Mean val accuracy:   {:.4f} +/- {:.4f}",
        result.mean_val_accuracy,
        result.std_val_accuracy,
    )
    per_fold = "  ".join(f"{a:.3f}" for a in result.fold_val_accuracies)
    logger.info("  Per-fold accuracies: {}", per_fold)
    logger.info("  Training rows:       {}", result.n_rows_used)
    logger.info("  Random seed:         {}", result.seed_used)
    logger.info(
        "  Best XGB params:     {}",
        result.best_xgb_params or "(hyperopt skipped)",
    )
    logger.info(
        "  Best LSTM params:    {}",
        result.best_lstm_params or "(hyperopt skipped)",
    )
    logger.info(
        "  Skipped regimes:     {}",
        result.skipped_regimes or "none",
    )
    logger.info("  MLflow run ID:       {}", result.mlflow_run_id or "(not logged)")
    logger.info(sep)


def _assert_qg05(result: TrainingResult) -> bool:
    """Assert QG-05 criteria against a :class:`~src.training.trainer.TrainingResult`.

    Checks (HARD -- all must pass):
      1. n_folds >= 3
      2. mean_val_accuracy > 53%
      3. seed_used > 0
      4. No NaN in fold_val_accuracies

    Args:
        result: Completed training result.

    Returns:
        ``True`` when all checks pass, ``False`` otherwise.
    """
    checks_passed = True

    if result.n_folds < 3:
        logger.error("QG-05 FAIL: n_folds={} < 3 minimum", result.n_folds)
        checks_passed = False
    else:
        logger.info("QG-05 PASS: n_folds={} >= 3", result.n_folds)

    if result.mean_val_accuracy <= _QG05_ACCURACY_THRESHOLD:
        logger.error(
            "QG-05 FAIL: mean_val_accuracy={:.4f} <= {:.2f} threshold",
            result.mean_val_accuracy,
            _QG05_ACCURACY_THRESHOLD,
        )
        checks_passed = False
    else:
        logger.info(
            "QG-05 PASS: mean_val_accuracy={:.4f} > {:.2f}",
            result.mean_val_accuracy,
            _QG05_ACCURACY_THRESHOLD,
        )

    if result.seed_used <= 0:
        logger.error("QG-05 FAIL: seed_used={} not positive", result.seed_used)
        checks_passed = False
    else:
        logger.info("QG-05 PASS: seed_used={}", result.seed_used)

    if any(np.isnan(a) for a in result.fold_val_accuracies):
        logger.error("QG-05 FAIL: NaN found in fold_val_accuracies")
        checks_passed = False
    else:
        logger.info(
            "QG-05 PASS: no NaN in {} fold accuracies",
            len(result.fold_val_accuracies),
        )

    return checks_passed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train DOGE prediction models (full pipeline)"
    )
    parser.add_argument(
        "--features-dir", type=Path, default=None,
        help="Directory with features_*.parquet files (default: data/features/primary/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to save model artefacts (default: temporary directory)",
    )
    parser.add_argument(
        "--in-memory-test", action="store_true",
        help="Generate synthetic data and run pipeline in-memory (CI mode)",
    )
    parser.add_argument(
        "--no-hyperopt", action="store_true",
        help="Skip Optuna hyperparameter search (faster for testing)",
    )
    parser.add_argument(
        "--n-hyperopt-trials", type=int, default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--wf-train-days", type=int, default=None,
        help="Walk-forward training window in days",
    )
    parser.add_argument(
        "--wf-val-days", type=int, default=None,
        help="Walk-forward validation window in days",
    )
    parser.add_argument(
        "--wf-step-days", type=int, default=None,
        help="Walk-forward step size in days",
    )
    parser.add_argument(
        "--min-training-rows", type=int, default=None,
        help="Minimum training rows per fold",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the full training pipeline and assert QG-05 criteria.

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 (all QG-05 checks pass) or 1 (failure).
    """
    args = _parse_args(argv)

    logger.info("=" * 64)
    logger.info("DOGE Predictor -- Full Training Pipeline")
    logger.info("=" * 64)

    # ------------------------------------------------------------------
    # Step 1 -- Load or generate data
    # ------------------------------------------------------------------
    try:
        if args.in_memory_test:
            logger.info("Mode: --in-memory-test (synthetic data)")
            feature_df, regime_labels = build_in_memory_data()
        else:
            features_dir = args.features_dir or (_ROOT / "data" / "features" / "primary")
            logger.info("Mode: production (features from {})", features_dir)
            feature_df, regime_labels = load_features_from_disk(features_dir)
    except Exception as exc:
        logger.error("train.py: data loading failed -- {}", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 2 -- Resolve walk-forward config
    # ------------------------------------------------------------------
    from src.config import doge_settings as _ds  # noqa: PLC0415

    base_wf = _ds.walk_forward
    if args.in_memory_test:
        wf_cfg = WalkForwardSettings(
            training_window_days=args.wf_train_days or 15,
            validation_window_days=args.wf_val_days or 5,
            step_size_days=args.wf_step_days or 5,
            min_training_rows=args.min_training_rows or 200,
        )
    else:
        wf_cfg = WalkForwardSettings(
            training_window_days=args.wf_train_days or base_wf.training_window_days,
            validation_window_days=args.wf_val_days or base_wf.validation_window_days,
            step_size_days=args.wf_step_days or base_wf.step_size_days,
            min_training_rows=args.min_training_rows or base_wf.min_training_rows,
        )

    logger.info(
        "Walk-forward: train={}d  val={}d  step={}d  min_rows={}",
        wf_cfg.training_window_days,
        wf_cfg.validation_window_days,
        wf_cfg.step_size_days,
        wf_cfg.min_training_rows,
    )

    # ------------------------------------------------------------------
    # Step 3 -- Run ModelTrainer
    # ------------------------------------------------------------------
    use_temp = args.output_dir is None
    tmp_ctx = tempfile.TemporaryDirectory() if use_temp else None

    try:
        output_dir = Path(tmp_ctx.name) if tmp_ctx else Path(args.output_dir)

        trainer = ModelTrainer(
            output_dir=output_dir,
            walk_forward_cfg=wf_cfg,
            run_hyperopt=not args.no_hyperopt,
            n_hyperopt_trials=args.n_hyperopt_trials,
        )
        result = trainer.train_full(feature_df, regime_labels)

    except Exception as exc:
        logger.error("train.py: ModelTrainer.train_full failed -- {}", exc)
        if tmp_ctx:
            tmp_ctx.cleanup()
        return 1

    # ------------------------------------------------------------------
    # Step 4 -- Print summary and assert QG-05
    # ------------------------------------------------------------------
    _print_summary(result)
    qg05_passed = _assert_qg05(result)

    if tmp_ctx:
        tmp_ctx.cleanup()

    if qg05_passed:
        logger.info("train.py: ALL QG-05 CHECKS PASSED -- exit 0")
        return 0
    else:
        logger.error("train.py: QG-05 FAILED -- exit 1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
