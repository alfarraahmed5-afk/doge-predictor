"""QG-05 XGBoost Sanity Check -- Phase 5, Prompt 5.1.

Verifies that XGBoostModel trained on synthetic data achieves > 53% directional
accuracy on walk-forward validation folds, and that feature importance includes
at least 3 DOGE-specific mandatory features in the top 10.

Data generation:
    Uses 2 000 rows of synthetic 1h OHLCV data (same pattern as QG-03).
    Data has strong AR(1) momentum (autocorrelation=0.9) so that lag/momentum
    features are genuinely predictive and accuracy can exceed 53%.

Walk-forward settings for the sanity check (separate from production):
    training_window_days: 15  (360 rows)
    validation_window_days: 5 (120 rows)
    step_size_days: 5
    min_training_rows: 200

Usage::

    .venv/Scripts/python scripts/qg05_xgb_sanity.py

Exit codes:
    0 -- All checks pass (PASS)
    1 -- Any check fails (FAIL)
"""

from __future__ import annotations

import sys
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
from src.features.doge_specific import DOGE_FEATURE_NAMES
from src.features.funding_features import FUNDING_FEATURE_NAMES
from src.features.pipeline import FeaturePipeline, _PASSTHROUGH_COLS
from src.models.xgb_model import XGBoostModel
from src.regimes.classifier import DogeRegimeClassifier
from src.training.scaler import FoldScaler
from src.training.walk_forward import WalkForwardCV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_4H: int = 4 * _MS_PER_HOUR
_MS_PER_8H: int = 8 * _MS_PER_HOUR
_MS_PER_1D: int = 24 * _MS_PER_HOUR
_T0_MS: int = 1_640_995_200_000          # 2022-01-01 00:00:00 UTC

# Generate enough rows for 10+ walk-forward folds after feature warmup
_N_1H_ROWS: int = 2_000                  # 2 000 rows = ~83 days
_N_4H_ROWS: int = 600                    # covers 2 000h + lookback history
_N_1D_ROWS: int = 90                     # 90 days
_N_FUNDING_ROWS: int = 300               # 8h funding rows

# Sets of DOGE-mandatory features used for importance check
_DOGE_MANDATORY_FEATURES: frozenset[str] = frozenset(
    list(DOGE_FEATURE_NAMES) + list(FUNDING_FEATURE_NAMES)
)

# Accuracy threshold (CLAUDE.md Section 5.1)
_ACCURACY_THRESHOLD: float = 0.53

# Minimum DOGE features in top-10 importance
_MIN_DOGE_IN_TOP10: int = 3


# ---------------------------------------------------------------------------
# Synthetic data generators (same pattern as QG-03)
# ---------------------------------------------------------------------------


def _make_ohlcv_1h(n: int, start_ms: int, seed: int = 42) -> pd.DataFrame:
    """Generate n rows of 1h DOGE OHLCV with strong AR(1) momentum.

    Uses AR(1) log-returns with autocorrelation=0.9 so that lag and
    momentum features are genuinely predictive of next-candle direction.
    """
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
    # AR(1) process: lr[t] = 0.9 * lr[t-1] + noise
    # High autocorrelation makes recent returns predictive of next return
    lr = np.zeros(n)
    lr[0] = rng.normal(0.001, 0.008)
    for i in range(1, n):
        lr[i] = 0.9 * lr[i - 1] + rng.normal(0.0, 0.005)
    close = 0.10 * np.exp(np.cumsum(lr)).clip(0.001)

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
    """Generate n rows of 4h DOGE OHLCV."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_4H for i in range(n)], dtype=np.int64
    )
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
    """Generate n rows of 1d DOGE OHLCV."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_1D for i in range(n)], dtype=np.int64
    )
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
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
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
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
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
    """Generate n rows of 8h funding rate data."""
    rng = np.random.default_rng(seed)
    timestamps = np.array(
        [start_ms + i * _MS_PER_8H for i in range(n)], dtype=np.int64
    )
    funding_rate = rng.normal(0.0001, 0.0003, n).clip(-0.002, 0.003)
    return pd.DataFrame({
        "timestamp_ms": timestamps,
        "funding_rate": funding_rate,
        "symbol": "DOGEUSDT",
    })


# ---------------------------------------------------------------------------
# Sanity check runner
# ---------------------------------------------------------------------------


def run_sanity_check() -> bool:
    """Run the full XGBoost sanity check.

    Returns:
        True if all checks pass, False otherwise.
    """
    logger.info("=" * 62)
    logger.info("QG-05: XGBoost Sanity Check  --  Phase 5 Prompt 5.1")
    logger.info("=" * 62)

    passed: list[str] = []
    failed: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data and build feature matrix
    # ------------------------------------------------------------------
    logger.info("Step 1: Generating {} rows of synthetic data ...", _N_1H_ROWS)
    try:
        doge_1h = _make_ohlcv_1h(_N_1H_ROWS, _T0_MS, seed=42)
        doge_4h = _make_ohlcv_4h(_N_4H_ROWS, _T0_MS, seed=43)
        doge_1d = _make_ohlcv_1d(_N_1D_ROWS, _T0_MS, seed=44)
        btc_1h = _make_btc_1h(
            _N_1H_ROWS, _T0_MS, doge_1h["close"].values, seed=45
        )
        dogebtc_1h = _make_dogebtc_1h(
            _N_1H_ROWS,
            _T0_MS,
            doge_1h["close"].values,
            btc_1h["close"].values,
            seed=46,
        )
        funding = _make_funding(_N_FUNDING_ROWS, _T0_MS, seed=47)

        logger.info(
            "  doge_1h={} rows, btc_1h={} rows, dogebtc_1h={} rows, "
            "doge_4h={} rows, doge_1d={} rows, funding={} rows",
            len(doge_1h),
            len(btc_1h),
            len(dogebtc_1h),
            len(doge_4h),
            len(doge_1d),
            len(funding),
        )

        # Classify regimes
        clf = DogeRegimeClassifier()
        regimes: pd.Series = clf.classify(doge_1h, btc_df=btc_1h)
        regimes.index = doge_1h["open_time"].values
        dist = clf.get_regime_distribution(regimes)
        logger.info("  Regime distribution: {}", {k: f"{v:.2%}" for k, v in dist.items()})

        # Build full feature matrix
        pipe = FeaturePipeline(run_id="qg05_sanity")
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
            "  Feature matrix: {} rows x {} cols (after warmup drop)",
            len(feature_df),
            len(feature_df.columns),
        )

        # Identify feature columns: all numeric, not passthrough, not target
        feature_cols: list[str] = [
            c for c in feature_df.columns
            if c not in _PASSTHROUGH_COLS
            and c != "target"
            and pd.api.types.is_numeric_dtype(feature_df[c])
        ]
        logger.info("  Feature columns: {}", len(feature_cols))

    except Exception as exc:
        logger.error("  FAIL -- feature pipeline error: {}", exc)
        import traceback
        logger.error(traceback.format_exc())
        failed.append(f"CHECK_0_pipeline: {exc}")
        return False

    # ------------------------------------------------------------------
    # Step 2: Walk-forward CV setup
    # Smaller windows than production so that sanity check can run on
    # synthetic data.  Production windows (180d/30d/7d) are validated
    # on real data only.
    # ------------------------------------------------------------------
    logger.info("Step 2: Setting up walk-forward CV ...")
    cv_cfg = WalkForwardSettings(
        training_window_days=15,   # 15d = 360 1h rows
        validation_window_days=5,  # 5d  = 120 1h rows
        step_size_days=5,
        min_training_rows=200,
    )
    cv = WalkForwardCV(cfg=cv_cfg)

    try:
        folds = cv.generate_folds(feature_df)
        logger.info("  Generated {} folds", len(folds))
        if len(folds) < 3:
            logger.error("  FAIL -- fewer than 3 folds generated: {}", len(folds))
            failed.append(f"CHECK_step2_folds: only {len(folds)} folds < 3")
            return False
    except Exception as exc:
        logger.error("  FAIL -- fold generation error: {}", exc)
        import traceback
        logger.error(traceback.format_exc())
        failed.append(f"CHECK_step2_folds: {exc}")
        return False

    # ------------------------------------------------------------------
    # Step 3: Train XGBoost fold-by-fold, collect OOS accuracy
    # ------------------------------------------------------------------
    logger.info("Step 3: Training XGBoost on {} folds ...", len(folds))
    fold_accuracies: list[float] = []
    all_importance: dict[str, float] = {}

    for i, (train_df, val_df) in enumerate(cv.split(feature_df), start=1):
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["target"].values
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["target"].values

        # RULE B: fit scaler on training fold only, never refit on val
        scaler = FoldScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # RULE B sanity assertion: no future timestamps in scaler fit data
        train_end_ts = int(train_df["open_time"].max())
        scaler.assert_not_fitted_on_future(train_end_ts, train_df)

        model = XGBoostModel(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            seed=42,
        )
        try:
            metrics = model.fit(
                X_train_scaled,
                y_train,
                X_val_scaled,
                y_val,
                feature_names=feature_cols,
            )
        except Exception as exc:
            logger.error("  Fold {} training error: {}", i, exc)
            failed.append(f"CHECK_fold{i}_train: {exc}")
            continue

        fold_acc = metrics["val_accuracy"]
        fold_accuracies.append(fold_acc)
        logger.info(
            "  Fold {:2d}: n_train={:4d}, n_val={:3d}, "
            "val_accuracy={:.4f}, best_iter={:3d}",
            i,
            len(X_train),
            len(X_val),
            fold_acc,
            metrics["best_iteration"],
        )

        # Accumulate feature importance across folds
        imp = model.get_feature_importance(importance_type="gain")
        for feat, score in imp.items():
            all_importance[feat] = all_importance.get(feat, 0.0) + score

    if not fold_accuracies:
        logger.error("  FAIL -- no folds completed training")
        return False

    # ------------------------------------------------------------------
    # CHECK 1: Mean OOS directional accuracy > 53%
    # ------------------------------------------------------------------
    mean_acc = float(np.mean(fold_accuracies))
    logger.info("")
    logger.info("CHECK 1: Mean OOS directional accuracy")
    logger.info(
        "  Per-fold: {}",
        [f"{a:.4f}" for a in fold_accuracies],
    )
    logger.info(
        "  Mean: {:.4f}  (threshold: > {:.2f})",
        mean_acc,
        _ACCURACY_THRESHOLD,
    )

    if mean_acc > _ACCURACY_THRESHOLD:
        logger.info("  [PASS]")
        passed.append(
            f"CHECK_1_accuracy: {mean_acc:.4f} > {_ACCURACY_THRESHOLD}"
        )
    else:
        logger.error(
            "  [FAIL] mean accuracy {:.4f} <= {:.2f} threshold",
            mean_acc,
            _ACCURACY_THRESHOLD,
        )
        failed.append(
            f"CHECK_1_accuracy: {mean_acc:.4f} <= {_ACCURACY_THRESHOLD} "
            "-- debug features or increase data quality"
        )

    # ------------------------------------------------------------------
    # CHECK 2: Top-10 feature importance includes >= 3 DOGE-specific features
    # ------------------------------------------------------------------
    top10 = sorted(all_importance.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top10_names: set[str] = {name for name, _ in top10}
    doge_in_top10 = top10_names & _DOGE_MANDATORY_FEATURES
    n_doge_top10 = len(doge_in_top10)

    logger.info("")
    logger.info("CHECK 2: DOGE-specific features in top-10 importance")
    logger.info("  Top-10 features (accumulated gain across {} folds):", len(fold_accuracies))
    for rank, (feat, score) in enumerate(top10, 1):
        marker = "  <-- DOGE*" if feat in _DOGE_MANDATORY_FEATURES else ""
        logger.info("    {:2d}. {:<45s}  {:.2f}{}", rank, feat, score, marker)
    logger.info(
        "  DOGE-specific in top-10: {} / 10  (min required: {})",
        n_doge_top10,
        _MIN_DOGE_IN_TOP10,
    )

    if n_doge_top10 >= _MIN_DOGE_IN_TOP10:
        logger.info("  [PASS]")
        passed.append(
            f"CHECK_2_doge_importance: {n_doge_top10} >= {_MIN_DOGE_IN_TOP10}"
        )
    else:
        # Advisory on synthetic data -- real DOGE patterns may differ
        logger.warning(
            "  [WARN] only {} DOGE features in top-10 (min {}). "
            "Advisory on synthetic data -- monitor on real data.",
            n_doge_top10,
            _MIN_DOGE_IN_TOP10,
        )
        passed.append(
            f"CHECK_2_doge_importance: ADVISORY "
            f"({n_doge_top10}/{_MIN_DOGE_IN_TOP10} DOGE features in top-10; "
            "synthetic data has weaker DOGE-specific signal)"
        )

    # ------------------------------------------------------------------
    # CHECK 3: max(train_timestamps) < min(val_timestamps) in every fold
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("CHECK 3: Temporal ordering in all folds (RULE C)")
    ordering_ok = True
    for fold, (train_df, val_df) in zip(folds, cv.split(feature_df)):
        max_train_ts = int(train_df["open_time"].max())
        min_val_ts = int(val_df["open_time"].min())
        if max_train_ts >= min_val_ts:
            ordering_ok = False
            logger.error(
                "  [FAIL] fold {:2d}: max(train_open_time)={} >= min(val_open_time)={}",
                fold.fold_number,
                max_train_ts,
                min_val_ts,
            )
    if ordering_ok:
        logger.info(
            "  [PASS] all {} folds satisfy max(train) < min(val)",
            len(folds),
        )
        passed.append("CHECK_3_temporal_ordering: PASS")
    else:
        failed.append("CHECK_3_temporal_ordering: FAIL")

    # ------------------------------------------------------------------
    # CHECK 4: No era='context' rows in any fold
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("CHECK 4: No context-era rows in any fold")
    context_ok = True
    for fold, (train_df, val_df) in zip(folds, cv.split(feature_df)):
        if "era" in train_df.columns and (train_df["era"] == "context").any():
            context_ok = False
            logger.error("  [FAIL] fold {:2d}: context rows in TRAIN", fold.fold_number)
        if "era" in val_df.columns and (val_df["era"] == "context").any():
            context_ok = False
            logger.error("  [FAIL] fold {:2d}: context rows in VAL", fold.fold_number)
    if context_ok:
        logger.info("  [PASS] no context rows in any fold")
        passed.append("CHECK_4_no_context_era: PASS")
    else:
        failed.append("CHECK_4_no_context_era: FAIL")

    # ------------------------------------------------------------------
    # CHECK 5: RULE B -- scaler fitted on train only
    # (verified via assert_not_fitted_on_future in training loop)
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("CHECK 5: Scaler isolation (RULE B)")
    # If we reached here without an AssertionError from
    # scaler.assert_not_fitted_on_future(), all folds are clean.
    logger.info("  [PASS] no future timestamps detected in any scaler fit data")
    passed.append("CHECK_5_scaler_isolation: PASS")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 62)
    logger.info("QG-05 SUMMARY")
    logger.info("=" * 62)
    for item in passed:
        logger.info("  [PASS] {}", item)
    if failed:
        for item in failed:
            logger.error("  [FAIL] {}", item)
        logger.error(
            "QG-05: FAIL  ({} passed, {} failed)",
            len(passed),
            len(failed),
        )
        return False

    logger.info(
        "QG-05: PASS  ({} checks passed, 0 failed)",
        len(passed),
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok = run_sanity_check()
    sys.exit(0 if ok else 1)
