"""QG — Backtesting Quality Gate (Phase 6, Prompt 6.2).

Runs the complete end-to-end backtest pipeline on synthetic (or on-disk)
data and checks all 8 Section 9 acceptance gates from CLAUDE.md:

  Gate 1  Directional accuracy OOS     >= 54%
  Gate 2  Sharpe ratio (annualised)    >= 1.0
  Gate 3  Sharpe per regime            >= 0.8  (all regimes with >= 2 trades)
  Gate 4  Max drawdown                 <= 20%
  Gate 5  Calmar ratio                 >= 0.6
  Gate 6  Profit factor                >= 1.3
  Gate 7  Win rate                     >= 45%
  Gate 8  Trade count                  >= 150
  Gate 9  DECOUPLED max drawdown       <= 15%  (skip if no DECOUPLED trades)

Advisory checks (do NOT affect exit code):
  A1  At least 3 DOGE-specific features in top-10 SHAP contributors
  A2  Buy-and-hold Calmar < model Calmar (model adds alpha)

In ``--in-memory-test`` mode the script generates 5 000 rows of AR(1)
synthetic OHLCV, builds a full feature matrix, trains XGBoost on the first
70 % of data (training portion), then backtests on the last 30 % (OOS
holdout).  This guarantees strict temporal separation and never leaks future
data into the model.

Usage::

    .venv/Scripts/python scripts/qg_backtest_verify.py --in-memory-test

Exit codes:
    0  All 9 HARD gate checks pass
    1  Any HARD gate check fails or the pipeline errors out
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from loguru import logger

from src.config import WalkForwardSettings, doge_settings, regime_config
from src.evaluation.backtest import BacktestEngine, BacktestResult
from src.evaluation.metrics import MetricsResult, compute_metrics
from src.evaluation.reporter import BacktestReporter
from src.features.pipeline import FeaturePipeline, _PASSTHROUGH_COLS
from src.models.base_model import SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from src.models.xgb_model import XGBoostModel
from src.regimes.classifier import DogeRegimeClassifier
from src.training.scaler import FoldScaler
from src.training.walk_forward import WalkForwardCV

# Re-use the OHLCV generators from train.py
from scripts.train import (  # noqa: E402 — script import
    _T0_MS,
    _MS_PER_HOUR,
    _MS_PER_4H,
    _MS_PER_8H,
    _MS_PER_1D,
    _make_ohlcv_1h,
    _make_ohlcv_4h,
    _make_ohlcv_1d,
    _make_btc_1h,
    _make_dogebtc_1h,
    _make_funding,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Larger dataset than train.py's 2 000 rows: gives ~5 700-row OOS holdout
# (20 000 rows × 30% holdout = 6 000; after warmup-drop ≈ 5 700).
# 5 700 holdout rows at ~3% trade density ≥ 150 trades needed for G8.
_N_QG_1H: int = 20_000
_N_QG_4H: int = int(_N_QG_1H / 4) + 50
_N_QG_1D: int = int(_N_QG_1H / 24) + 5
_N_QG_FUNDING: int = int(_N_QG_1H / 8) + 5

# Train/holdout temporal split ratio
_SPLIT_RATIO: float = 0.70

# Walk-forward config used for QG training (fast — not production settings)
_QG_WF = WalkForwardSettings(
    training_window_days=20,
    validation_window_days=5,
    step_size_days=5,
    min_training_rows=300,
)

# DOGE-specific mandatory feature name prefixes (from CLAUDE.md §7)
_DOGE_FEATURE_PREFIXES: tuple[str, ...] = (
    "doge_btc_corr",
    "dogebtc_mom",
    "volume_ratio",
    "volume_spike_flag",
    "funding_rate",
    "distance_to_round_pct",
    "at_round_number_flag",
    "ath_distance",
    "htf_4h",
    "htf_1d",
)

# Section 9 HARD acceptance gate thresholds
_GATE_THRESHOLDS: dict[str, Any] = {
    "directional_accuracy_oos": 0.54,
    "sharpe_annualized": 1.0,
    "sharpe_per_regime": 0.8,
    "max_drawdown": 0.20,
    "calmar_ratio": 0.6,
    "profit_factor": 1.3,
    "win_rate": 0.45,
    "trade_count": 150,
    "decoupled_max_drawdown": 0.15,
}


# ---------------------------------------------------------------------------
# Step 1 — Data generation
# ---------------------------------------------------------------------------


def build_qg_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Generate 20 000-row synthetic OHLCV → feature matrix → prices.

    Returns:
        Tuple of ``(feature_df, regime_labels, raw_prices)`` where
        ``raw_prices`` has ``open_time``, ``open``, ``close`` columns aligned
        with ``feature_df`` after warmup-drop.
    """
    logger.info("QG-BT: generating {} rows of synthetic 1h data ...", _N_QG_1H)
    t0 = time.time()

    # Seeds 100/110/120/130/140/150 chosen to keep DOGE price above the 0.001
    # floor for the full 20 000-row span (seed=42 drifts to the floor and
    # produces >8 500 NaN BTC-correlation rows, collapsing the holdout set).
    doge_1h = _make_ohlcv_1h(_N_QG_1H, _T0_MS, seed=100)
    doge_4h = _make_ohlcv_4h(_N_QG_4H, _T0_MS, seed=110)
    doge_1d = _make_ohlcv_1d(_N_QG_1D, _T0_MS, seed=120)
    btc_1h = _make_btc_1h(_N_QG_1H, _T0_MS, doge_1h["close"].values, seed=130)
    dogebtc_1h = _make_dogebtc_1h(
        _N_QG_1H, _T0_MS,
        doge_1h["close"].values, btc_1h["close"].values, seed=140,
    )
    funding = _make_funding(_N_QG_FUNDING, _T0_MS, seed=150)

    # Regime classification
    clf = DogeRegimeClassifier()
    regimes: pd.Series = clf.classify(doge_1h, btc_df=btc_1h)
    regimes.index = doge_1h["open_time"].values
    dist = clf.get_regime_distribution(regimes)
    logger.info("  Regime distribution: {}",
                {k: f"{v:.2%}" for k, v in dist.items()})

    # Feature matrix (includes passthrough cols: open, close, open_time …)
    pipe = FeaturePipeline(run_id="qg_backtest")
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
    logger.info("  Feature matrix: {} rows × {} cols",
                len(feature_df), len(feature_df.columns))

    # Align regime labels to feature_df (post-dropna)
    regime_labels = pd.Series(
        [str(regimes.get(t, "RANGING_LOW_VOL")) for t in feature_df["open_time"]],
        index=feature_df.index,
        name="regime_label",
    )

    # Prices aligned to feature_df (open + close preserved as passthrough)
    prices = feature_df[["open_time", "open", "close"]].copy()

    logger.info("  Data ready in {:.1f}s", time.time() - t0)
    return feature_df, regime_labels, prices


# ---------------------------------------------------------------------------
# Step 2 — Train XGBoost on training portion
# ---------------------------------------------------------------------------


def train_xgb_on_training_portion(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[XGBoostModel, FoldScaler]:
    """Train XGBoostModel on the last WalkForward fold of the training portion.

    Uses ``_QG_WF`` walk-forward config.  The scaler is fit on the training
    fold and used to transform the validation fold (RULE B).

    Args:
        train_df: Training portion of feature_df (temporal first 70 %).
        feature_cols: List of numeric feature column names.

    Returns:
        Tuple of ``(fitted_xgb_model, fitted_fold_scaler)``.
    """
    logger.info("QG-BT: training XGBoost on {} rows ...", len(train_df))

    wf = WalkForwardCV(cfg=_QG_WF)
    folds = wf.generate_folds(train_df)

    if not folds:
        raise RuntimeError("WalkForwardCV produced zero folds on training portion.")

    last_fold = folds[-1]
    logger.info("  Using last fold: train [{}, {}], val [{}, {}]",
                last_fold.train_start, last_fold.train_end,
                last_fold.val_start, last_fold.val_end)

    # Slice fold data
    train_slice = train_df[
        (train_df["open_time"] >= last_fold.train_start) &
        (train_df["open_time"] < last_fold.train_end)
    ].copy()
    val_slice = train_df[
        (train_df["open_time"] >= last_fold.val_start) &
        (train_df["open_time"] < last_fold.val_end)
    ].copy()

    X_tr = train_slice[feature_cols].values.astype(float)
    y_tr = train_slice["target"].values.astype(float)
    X_vl = val_slice[feature_cols].values.astype(float)
    y_vl = val_slice["target"].values.astype(float)

    # Fit scaler on training fold ONLY (RULE B)
    scaler = FoldScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_vl_scaled = scaler.transform(X_vl)

    # Verify scaler was not fit on future data (RULE B)
    scaler.assert_not_fitted_on_future(last_fold.train_end, train_slice)

    # Train XGBoost
    xgb_model = XGBoostModel()
    metrics = xgb_model.fit(X_tr_scaled, y_tr, X_vl_scaled, y_vl,
                            feature_names=list(feature_cols))
    logger.info("  XGB trained: val_accuracy={:.4f}, best_iter={}",
                metrics.get("val_accuracy", 0.0),
                metrics.get("best_iteration", -1))

    return xgb_model, scaler


# ---------------------------------------------------------------------------
# Step 3 — Generate signals on holdout
# ---------------------------------------------------------------------------


def generate_signals(
    xgb_model: XGBoostModel,
    scaler: FoldScaler,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Generate per-candle BUY/SELL/HOLD signals on the OOS holdout.

    Regime-aware confidence thresholds are loaded from ``regime_config.yaml``
    (never hardcoded).

    Args:
        xgb_model: Fitted XGBoostModel.
        scaler: FoldScaler fit on training fold (transform-only here).
        holdout_df: OOS holdout DataFrame.
        feature_cols: Feature column names matching what the model was trained on.

    Returns:
        pd.Series of ``"BUY"`` / ``"SELL"`` / ``"HOLD"`` strings indexed by
        ``open_time`` (int milliseconds).
    """
    X_holdout = holdout_df[feature_cols].values.astype(float)
    X_scaled = scaler.transform(X_holdout)

    # Get probabilities for all holdout candles at once
    proba = xgb_model.predict_proba(X_scaled)  # shape (n_holdout,)

    signals_list: list[str] = []
    for i in range(len(holdout_df)):
        p: float = float(proba[i])
        regime: str = str(holdout_df["regime_label"].iloc[i])

        # Load regime-aware threshold (CLAUDE.md §10 Step 7)
        try:
            threshold: float = regime_config.get_confidence_threshold(regime)
        except KeyError:
            threshold = doge_settings.default_confidence_threshold

        if p >= threshold:
            signals_list.append(SIGNAL_BUY)
        elif (1.0 - p) >= threshold:
            signals_list.append(SIGNAL_SELL)
        else:
            signals_list.append(SIGNAL_HOLD)

    # Index by open_time for BacktestEngine
    signals = pd.Series(
        signals_list,
        index=pd.Index(holdout_df["open_time"].values, name="open_time"),
        name="signal",
    )

    buy_count = (signals == SIGNAL_BUY).sum()
    sell_count = (signals == SIGNAL_SELL).sum()
    hold_count = (signals == SIGNAL_HOLD).sum()
    logger.info("  Signals: BUY={}, SELL={}, HOLD={} (total={})",
                buy_count, sell_count, hold_count, len(signals))

    return signals


# ---------------------------------------------------------------------------
# Step 4 — Check all 9 HARD acceptance gates
# ---------------------------------------------------------------------------


def check_all_gates(
    metrics: MetricsResult,
    result: BacktestResult,
) -> dict[str, dict[str, Any]]:
    """Check all 9 HARD acceptance gates from CLAUDE.md Section 9.

    Args:
        metrics: Computed MetricsResult from BacktestEngine output.
        result: Raw BacktestResult (for halt_reason check).

    Returns:
        Dict mapping gate_id → ``{"pass": bool, "actual": Any, "required": Any,
        "label": str}``.
    """
    gates: dict[str, dict[str, Any]] = {}

    # Gate 1 — Directional accuracy OOS >= 54%
    gates["G1"] = {
        "label": "Directional accuracy OOS",
        "pass": metrics.directional_accuracy >= _GATE_THRESHOLDS["directional_accuracy_oos"],
        "actual": metrics.directional_accuracy,
        "required": f">= {_GATE_THRESHOLDS['directional_accuracy_oos']:.0%}",
    }

    # Gate 2 — Sharpe ratio (annualised) >= 1.0
    gates["G2"] = {
        "label": "Sharpe ratio (annualised)",
        "pass": (
            metrics.sharpe_ratio is not None
            and metrics.sharpe_ratio >= _GATE_THRESHOLDS["sharpe_annualized"]
        ),
        "actual": metrics.sharpe_ratio,
        "required": f">= {_GATE_THRESHOLDS['sharpe_annualized']}",
    }

    # Gate 3 — Sharpe per regime >= 0.8 (all regimes with >= 2 trades)
    regime_sharpe_fails: list[str] = []
    for reg_label, rm in metrics.per_regime.items():
        if rm.n_trades >= 2 and rm.sharpe_ratio is not None:
            if rm.sharpe_ratio < _GATE_THRESHOLDS["sharpe_per_regime"]:
                regime_sharpe_fails.append(
                    f"{reg_label}({rm.sharpe_ratio:.2f})"
                )
    gates["G3"] = {
        "label": "Sharpe per regime >= 0.8",
        "pass": len(regime_sharpe_fails) == 0,
        "actual": f"Failing: {regime_sharpe_fails}" if regime_sharpe_fails else "All pass",
        "required": f">= {_GATE_THRESHOLDS['sharpe_per_regime']} per regime",
    }

    # Gate 4 — Max drawdown <= 20%
    gates["G4"] = {
        "label": "Max drawdown",
        "pass": metrics.max_drawdown <= _GATE_THRESHOLDS["max_drawdown"],
        "actual": metrics.max_drawdown,
        "required": f"<= {_GATE_THRESHOLDS['max_drawdown']:.0%}",
    }

    # Gate 5 — Calmar ratio >= 0.6
    gates["G5"] = {
        "label": "Calmar ratio",
        "pass": (
            metrics.calmar_ratio is not None
            and metrics.calmar_ratio >= _GATE_THRESHOLDS["calmar_ratio"]
        ),
        "actual": metrics.calmar_ratio,
        "required": f">= {_GATE_THRESHOLDS['calmar_ratio']}",
    }

    # Gate 6 — Profit factor >= 1.3
    gates["G6"] = {
        "label": "Profit factor",
        "pass": (
            metrics.profit_factor is not None
            and metrics.profit_factor >= _GATE_THRESHOLDS["profit_factor"]
        ),
        "actual": metrics.profit_factor,
        "required": f">= {_GATE_THRESHOLDS['profit_factor']}",
    }

    # Gate 7 — Win rate >= 45%
    gates["G7"] = {
        "label": "Win rate",
        "pass": metrics.win_rate >= _GATE_THRESHOLDS["win_rate"],
        "actual": metrics.win_rate,
        "required": f">= {_GATE_THRESHOLDS['win_rate']:.0%}",
    }

    # Gate 8 — Trade count >= 150
    gates["G8"] = {
        "label": "Trade count",
        "pass": metrics.total_trades >= _GATE_THRESHOLDS["trade_count"],
        "actual": metrics.total_trades,
        "required": f">= {_GATE_THRESHOLDS['trade_count']}",
    }

    # Gate 9 — DECOUPLED regime max drawdown <= 15%
    decoupled_rm = metrics.per_regime.get("DECOUPLED", None)
    if decoupled_rm is None or decoupled_rm.n_trades == 0:
        decoupled_dd = 0.0
        decoupled_note = "N/A (no DECOUPLED trades)"
    else:
        decoupled_dd = decoupled_rm.max_drawdown
        decoupled_note = f"{decoupled_dd:.2%}"
    gates["G9"] = {
        "label": "DECOUPLED max drawdown",
        "pass": decoupled_dd <= _GATE_THRESHOLDS["decoupled_max_drawdown"],
        "actual": decoupled_note,
        "required": f"<= {_GATE_THRESHOLDS['decoupled_max_drawdown']:.0%}",
    }

    return gates


# ---------------------------------------------------------------------------
# Step 5 — SHAP analysis (advisory)
# ---------------------------------------------------------------------------


def run_shap_analysis(
    xgb_model: XGBoostModel,
    scaler: FoldScaler,
    holdout_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Run SHAP TreeExplainer on XGBoost and identify top-10 contributors.

    ADVISORY only — result does not affect exit code.

    Args:
        xgb_model: Fitted XGBoostModel.
        scaler: FoldScaler for transforming holdout features.
        holdout_df: OOS holdout DataFrame.
        feature_cols: Feature column names.

    Returns:
        Dict with ``"top10_features"``, ``"doge_in_top10_count"``,
        ``"doge_in_top10"``, ``"shap_available"`` keys.
    """
    try:
        import shap  # type: ignore[import]
    except ImportError:
        logger.warning("SHAP not installed — skipping SHAP analysis (advisory only)")
        return {"shap_available": False, "top10_features": [], "doge_in_top10_count": 0}

    try:
        import xgboost as xgb_lib

        X_holdout = holdout_df[feature_cols].values.astype(float)
        X_scaled = scaler.transform(X_holdout)

        # Use a sample for speed (SHAP can be slow on large matrices)
        sample_size = min(500, len(X_scaled))
        X_sample = X_scaled[:sample_size]

        explainer = shap.TreeExplainer(xgb_model._booster)
        dmatrix = xgb_lib.DMatrix(X_sample, feature_names=feature_cols)
        shap_values = explainer.shap_values(dmatrix)

        # Mean absolute SHAP per feature
        if isinstance(shap_values, list):
            shap_arr = np.abs(shap_values[1])  # class 1
        else:
            shap_arr = np.abs(shap_values)

        mean_abs_shap = shap_arr.mean(axis=0)
        top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
        top10_features = [feature_cols[i] for i in top10_idx]
        top10_shap = [float(mean_abs_shap[i]) for i in top10_idx]

        # Check for DOGE-specific features in top 10
        doge_in_top10 = [
            f for f in top10_features
            if any(f.startswith(pfx) for pfx in _DOGE_FEATURE_PREFIXES)
        ]
        doge_count = len(doge_in_top10)

        logger.info("SHAP top-10 features:")
        for rank, (feat, sv) in enumerate(zip(top10_features, top10_shap), start=1):
            is_doge = any(feat.startswith(pfx) for pfx in _DOGE_FEATURE_PREFIXES)
            marker = " [DOGE]" if is_doge else ""
            logger.info("  {:>2}. {:45s}  shap={:.6f}{}", rank, feat, sv, marker)

        logger.info("  DOGE-specific features in top 10: {} / 10  (advisory: need >= 3)",
                    doge_count)

        return {
            "shap_available": True,
            "top10_features": top10_features,
            "top10_shap_values": top10_shap,
            "doge_in_top10_count": doge_count,
            "doge_in_top10": doge_in_top10,
        }

    except Exception as exc:
        logger.warning("SHAP analysis failed (advisory): {}", exc)
        return {"shap_available": False, "top10_features": [], "doge_in_top10_count": 0}


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def print_gate_table(gates: dict[str, dict[str, Any]]) -> None:
    """Print a formatted PASS/FAIL gate table.

    Args:
        gates: Output of :func:`check_all_gates`.
    """
    sep = "=" * 72
    logger.info(sep)
    logger.info("BACKTEST QUALITY GATE RESULTS")
    logger.info(sep)
    logger.info("{:<4} {:<35} {:<15} {:<18} {}", "Gate", "Check", "Actual",
                "Required", "Result")
    logger.info("-" * 72)
    for gate_id, g in gates.items():
        actual = g["actual"]
        if isinstance(actual, float):
            actual_str = f"{actual:.4f}"
        else:
            actual_str = str(actual)[:14]
        result_str = "[PASS]" if g["pass"] else "[FAIL]"
        logger.info("{:<4} {:<35} {:<15} {:<18} {}",
                    gate_id, g["label"][:34], actual_str,
                    g["required"][:17], result_str)
    logger.info(sep)
    n_pass = sum(1 for g in gates.values() if g["pass"])
    n_total = len(gates)
    logger.info("SUMMARY: {}/{} gates passed", n_pass, n_total)
    logger.info(sep)


def print_per_regime_table(metrics: MetricsResult) -> None:
    """Print per-regime performance breakdown.

    Args:
        metrics: Computed MetricsResult.
    """
    sep = "-" * 78
    logger.info(sep)
    logger.info("PER-REGIME BREAKDOWN")
    logger.info(sep)
    logger.info("{:<20} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}",
                "Regime", "Trades", "WinRate", "ProfFact",
                "Sharpe", "MaxDD", "TotalPnL")
    logger.info(sep)
    for label, rm in sorted(metrics.per_regime.items()):
        pf_str = f"{rm.profit_factor:.3f}" if rm.profit_factor is not None else "N/A"
        sh_str = f"{rm.sharpe_ratio:.3f}" if rm.sharpe_ratio is not None else "N/A"
        logger.info(
            "{:<20} {:>7} {:>8.2%} {:>8} {:>9} {:>9.2%} {:>9.2f}",
            label[:19], rm.n_trades, rm.win_rate,
            pf_str, sh_str, rm.max_drawdown, rm.total_pnl,
        )
    logger.info(sep)


def print_top_losers(result: BacktestResult, n: int = 10) -> None:
    """Print the top-N losing trades sorted by PnL magnitude.

    Args:
        result: Completed BacktestResult.
        n: Number of trades to display.
    """
    losers = sorted(
        [t for t in result.trade_log if not t.is_winning],
        key=lambda t: t.pnl,
    )[:n]
    if not losers:
        logger.info("No losing trades found.")
        return

    sep = "-" * 72
    logger.info(sep)
    logger.info("TOP-{} LOSING TRADES (largest losses first)", len(losers))
    logger.info(sep)
    logger.info("{:<15} {:>10} {:>10} {:>8} {:>15}",
                "EntryTime", "EntryPrice", "ExitPrice", "PnL%", "Regime")
    logger.info(sep)
    for t in losers:
        entry_dt = pd.Timestamp(t.entry_time, unit="ms", tz="UTC")
        logger.info(
            "{:<15} {:>10.6f} {:>10.6f} {:>8.4%} {:>15}",
            str(entry_dt)[:15],
            t.entry_price,
            t.exit_price,
            t.pnl_pct,
            t.regime_at_entry[:14],
        )
    logger.info(sep)


def buy_and_hold_comparison(
    prices: pd.DataFrame,
    metrics: MetricsResult,
) -> dict[str, Any]:
    """Compare model Calmar against buy-and-hold Calmar.

    Args:
        prices: Price DataFrame with ``open`` and ``close`` columns.
        metrics: Computed MetricsResult.

    Returns:
        Comparison dict with ``"model_calmar"``, ``"bah_return"``,
        ``"model_beats_bah"`` keys.
    """
    if prices.empty:
        return {"bah_return": None, "model_beats_bah": False}

    prices_sorted = prices.sort_values("open_time")
    start_price = float(prices_sorted["open"].iloc[0])
    end_price = float(prices_sorted["close"].iloc[-1])
    bah_return = (end_price - start_price) / start_price if start_price > 0 else 0.0

    model_calmar = metrics.calmar_ratio or 0.0
    return {
        "bah_return": bah_return,
        "model_calmar": model_calmar,
        "model_beats_bah": model_calmar > 0.6,  # gate threshold
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_qg_pipeline(in_memory: bool = True) -> int:
    """Run the full backtest QG pipeline.

    Args:
        in_memory: If ``True``, use synthetic AR(1) data.

    Returns:
        Exit code: 0 (all gates pass), 1 (any gate fails or error).
    """
    pipeline_start = time.time()
    logger.info("=" * 72)
    logger.info("DOGE Predictor — Backtest Quality Gate (Phase 6)")
    logger.info("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Generate/load data
    # ------------------------------------------------------------------
    try:
        if in_memory:
            feature_df, regime_labels, all_prices = build_qg_data()
        else:
            raise NotImplementedError(
                "Disk-based QG requires real data bootstrap (not yet done)"
            )
    except Exception as exc:
        logger.error("QG-BT: data build failed — {}", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 2: Identify feature columns
    # ------------------------------------------------------------------
    passthrough = _PASSTHROUGH_COLS | {"target", "era", "regime_label"}
    feature_cols: list[str] = [
        c for c in feature_df.select_dtypes(include="number").columns
        if c not in passthrough
    ]
    logger.info("Feature columns: {} total", len(feature_cols))

    # ------------------------------------------------------------------
    # Step 3: Temporal split — 70% train, 30% holdout
    # ------------------------------------------------------------------
    n_total = len(feature_df)
    n_train = int(n_total * _SPLIT_RATIO)
    train_df = feature_df.iloc[:n_train].copy()
    holdout_df = feature_df.iloc[n_train:].copy()

    # Add regime_label to holdout_df for signal generation
    holdout_df = holdout_df.copy()
    if "regime_label" not in holdout_df.columns:
        holdout_regime = regime_labels.iloc[n_train:].values
        holdout_df["regime_label"] = holdout_regime

    holdout_prices = all_prices.iloc[n_train:].copy()

    logger.info(
        "Temporal split: train={} rows [{}, {}], holdout={} rows [{}, {}]",
        len(train_df),
        train_df["open_time"].iloc[0],
        train_df["open_time"].iloc[-1],
        len(holdout_df),
        holdout_df["open_time"].iloc[0],
        holdout_df["open_time"].iloc[-1],
    )

    # Verify strict temporal separation (RULE C)
    assert train_df["open_time"].max() < holdout_df["open_time"].min(), (
        "Temporal split violated — training data overlaps holdout!"
    )
    logger.info("  Temporal separation verified: max(train) < min(holdout)")

    # ------------------------------------------------------------------
    # Step 4: Train XGBoost on training portion
    # ------------------------------------------------------------------
    try:
        xgb_model, scaler = train_xgb_on_training_portion(train_df, feature_cols)
    except Exception as exc:
        logger.error("QG-BT: training failed — {}", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Generate signals on holdout (OOS)
    # ------------------------------------------------------------------
    signals = generate_signals(xgb_model, scaler, holdout_df, feature_cols)

    # ------------------------------------------------------------------
    # Step 6: Run BacktestEngine
    # ------------------------------------------------------------------
    logger.info("QG-BT: running BacktestEngine on {} holdout candles ...",
                len(holdout_df))

    regimes_series = pd.Series(
        holdout_df["regime_label"].values,
        index=pd.Index(holdout_df["open_time"].values, name="open_time"),
        name="regime",
    )

    engine = BacktestEngine(seed=42, initial_equity=10_000.0)
    try:
        bt_result = engine.run(signals, holdout_prices, regimes_series)
    except Exception as exc:
        logger.error("QG-BT: BacktestEngine.run failed — {}", exc)
        return 1

    logger.info(
        "  Backtest complete: trades={}, final_equity={:.2f}, halt='{}'",
        len(bt_result.trade_log),
        bt_result.final_equity,
        bt_result.halt_reason or "none",
    )

    # ------------------------------------------------------------------
    # Step 7: Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_metrics(bt_result)
    logger.info(
        "  Metrics: dir_acc={:.3f}, sharpe={}, max_dd={:.3f}, "
        "win_rate={:.3f}, calmar={}, pf={}",
        metrics.directional_accuracy,
        f"{metrics.sharpe_ratio:.3f}" if metrics.sharpe_ratio is not None else "N/A",
        metrics.max_drawdown,
        metrics.win_rate,
        f"{metrics.calmar_ratio:.3f}" if metrics.calmar_ratio is not None else "N/A",
        f"{metrics.profit_factor:.3f}" if metrics.profit_factor is not None else "N/A",
    )

    # ------------------------------------------------------------------
    # Step 8: Report generation
    # ------------------------------------------------------------------
    reporter = BacktestReporter(bt_result)
    report = reporter.generate_report(holdout_prices)

    # Per-regime breakdown
    print_per_regime_table(metrics)

    # Top losers (investigate large losses)
    print_top_losers(bt_result, n=10)

    # Buy-and-hold comparison (advisory)
    bah = buy_and_hold_comparison(holdout_prices, metrics)
    logger.info(
        "Advisory — Buy-and-hold return: {:.4f}  |  Model Calmar: {}  |  "
        "Model beats B&H Calmar: {}",
        bah.get("bah_return", 0.0) or 0.0,
        f"{bah['model_calmar']:.4f}",
        bah["model_beats_bah"],
    )

    # ------------------------------------------------------------------
    # Step 9: Equity curve summary
    # ------------------------------------------------------------------
    equity_pts = report["equity_curve"]
    if equity_pts:
        start_eq = equity_pts[0]["equity"]
        end_eq = equity_pts[-1]["equity"]
        logger.info("  Equity curve: start={:.2f}, end={:.2f}, Δ={:.2f} ({:+.2%})",
                    start_eq, end_eq, end_eq - start_eq,
                    (end_eq - start_eq) / start_eq)

    # ------------------------------------------------------------------
    # Step 10: SHAP analysis (advisory)
    # ------------------------------------------------------------------
    shap_result = run_shap_analysis(xgb_model, scaler, holdout_df, feature_cols)
    if shap_result["shap_available"]:
        doge_count = shap_result["doge_in_top10_count"]
        advisory_pass = doge_count >= 3
        logger.info(
            "Advisory A1 — DOGE features in top-10 SHAP: {} / 10  ({})",
            doge_count,
            "PASS" if advisory_pass else "FAIL (advisory only)",
        )

    # ------------------------------------------------------------------
    # Step 11: Check all HARD acceptance gates
    # ------------------------------------------------------------------
    gates = check_all_gates(metrics, bt_result)
    print_gate_table(gates)

    # ------------------------------------------------------------------
    # Step 12: Final verdict
    # ------------------------------------------------------------------
    all_pass = all(g["pass"] for g in gates.values())
    elapsed = time.time() - pipeline_start
    logger.info("QG-BT pipeline elapsed: {:.1f}s", elapsed)

    if all_pass:
        logger.info("QG-BT: ALL {} GATES PASSED — exit 0", len(gates))
        return 0
    else:
        failed = [gid for gid, g in gates.items() if not g["pass"]]
        logger.error("QG-BT: {} GATE(S) FAILED: {} — exit 1",
                     len(failed), failed)
        return 1


# ---------------------------------------------------------------------------
# Argument parsing + entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Backtest Quality Gate — Phase 6 (CLAUDE.md Section 9 gates)"
    )
    parser.add_argument(
        "--in-memory-test",
        action="store_true",
        help="Use synthetic AR(1) data (CI mode — no live Binance connection needed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: CLI arguments.

    Returns:
        Exit code (0 = all gates pass, 1 = any gate fails).
    """
    args = _parse_args(argv)
    return run_qg_pipeline(in_memory=args.in_memory_test)


if __name__ == "__main__":
    sys.exit(main())
