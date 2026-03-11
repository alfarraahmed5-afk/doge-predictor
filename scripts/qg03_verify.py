"""Quality Gate 03 — Feature Pipeline Integrity Verification.

Checks (all must pass for PASS):
    1. Pipeline runs end-to-end without exception on synthetic fixture data
    2. Zero NaN in any feature column of the final matrix
    3. Zero Inf in any feature column of the final matrix
    4. All 12 DOGE-specific mandatory features present
    5. All 6 regime feature columns present
    6. target column present with values only 0.0 or 1.0 (no NaN after dropna)
    7. Lag sanity: log_ret_1[t] == log(close[t] / close[t-1]) for consecutive rows
    8. HTF lookahead guard: htf_4h_rsi constant within each 4h period (no intra-period change)
    9. (Advisory) Correlation: flag feature pairs with |corr| > 0.98
   10. (Advisory) Stationarity: ADF test on feature columns; flag non-stationary (p > 0.05)

Exit codes:
    0 — all checks PASS
    1 — at least one check FAILED

Usage::

    py scripts/qg03_verify.py

The script uses self-contained synthetic data and does NOT require a live
database or Binance connection.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on PYTHONPATH (so src/ imports resolve correctly)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import doge_settings
from src.features.doge_specific import DOGE_FEATURE_NAMES
from src.features.pipeline import (
    MANDATORY_FEATURE_NAMES,
    FeaturePipeline,
    validate_feature_matrix,
)
from src.regimes.features import REGIME_FEATURE_KEYS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_4H: int = 4 * _MS_PER_HOUR
_MS_PER_1D: int = 24 * _MS_PER_HOUR
_MS_PER_8H: int = 8 * _MS_PER_HOUR

# Start at 2022-01-01 00:00:00 UTC
_T0_MS: int = 1_640_995_200_000

# Generate enough 1h rows so that ~250 warmup rows are dropped and ~450 remain
_N_1H_ROWS: int = 800          # 1h candle rows
_N_4H_ROWS: int = 300          # 4h candle rows (covers 800h + history)
_N_1D_ROWS: int = 50           # 1d candle rows
_N_FUNDING_ROWS: int = 150     # 8h funding rows

_REGIME_LABELS = [
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
]

_PASS = "[PASS]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_ohlcv_1h(n: int, start_ms: int, seed: int = 42) -> pd.DataFrame:
    """Generate n rows of 1h OHLCV data starting from start_ms."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
    # Random walk for close
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 0.08 * np.exp(np.cumsum(log_ret))
    # OHLCV from close
    noise = rng.uniform(0.001, 0.005, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    volume = rng.uniform(1e7, 5e8, n)

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_.clip(0.001),
        "high": high.clip(0.001),
        "low": low.clip(0.001),
        "close": close.clip(0.001),
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_ohlcv_4h(n: int, start_ms: int, seed: int = 43) -> pd.DataFrame:
    """Generate n rows of 4h OHLCV data."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_4H for i in range(n)], dtype=np.int64
    )
    log_ret = rng.normal(0.001, 0.02, n)
    close = 0.08 * np.exp(np.cumsum(log_ret))
    noise = rng.uniform(0.005, 0.015, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0, 0.005, n))
    volume = rng.uniform(4e7, 2e9, n)

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_.clip(0.001),
        "high": high.clip(0.001),
        "low": low.clip(0.001),
        "close": close.clip(0.001),
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_ohlcv_1d(n: int, start_ms: int, seed: int = 44) -> pd.DataFrame:
    """Generate n rows of 1d OHLCV data."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_1D for i in range(n)], dtype=np.int64
    )
    log_ret = rng.normal(0.003, 0.04, n)
    close = 0.08 * np.exp(np.cumsum(log_ret))
    noise = rng.uniform(0.01, 0.04, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0, 0.01, n))
    volume = rng.uniform(1e9, 1e10, n)

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_.clip(0.001),
        "high": high.clip(0.001),
        "low": low.clip(0.001),
        "close": close.clip(0.001),
        "volume": volume,
        "symbol": "DOGEUSDT",
        "era": "training",
    })


def _make_btc_1h(n: int, start_ms: int, doge_close: np.ndarray, seed: int = 45) -> pd.DataFrame:
    """Generate n rows of BTC 1h data correlated with DOGE."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
    # Correlated returns (corr ~0.7)
    doge_lr = np.diff(np.log(doge_close), prepend=np.log(doge_close[0]))
    btc_lr = 0.7 * doge_lr + 0.3 * rng.normal(0, 0.008, n)
    close = 40000 * np.exp(np.cumsum(btc_lr))
    noise = rng.uniform(0.002, 0.01, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    volume = rng.uniform(1e9, 1e11, n)

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_.clip(1.0),
        "high": high.clip(1.0),
        "low": low.clip(1.0),
        "close": close.clip(1.0),
        "volume": volume,
        "symbol": "BTCUSDT",
        "era": "training",
    })


def _make_dogebtc_1h(
    n: int, start_ms: int, doge_close: np.ndarray, btc_close: np.ndarray, seed: int = 46
) -> pd.DataFrame:
    """Generate n rows of DOGEBTC 1h data as doge_close / btc_close."""
    rng = np.random.default_rng(seed)
    open_times = np.array(
        [start_ms + i * _MS_PER_HOUR for i in range(n)], dtype=np.int64
    )
    close = doge_close / btc_close
    noise = rng.uniform(0.001, 0.005, n)
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0, 0.002, n))
    volume = rng.uniform(1e6, 1e8, n)

    return pd.DataFrame({
        "open_time": open_times,
        "open": open_.clip(1e-8),
        "high": high.clip(1e-8),
        "low": low.clip(1e-8),
        "close": close.clip(1e-8),
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
    # Small positive funding rates (typical for DOGE)
    funding_rate = rng.normal(0.0001, 0.0003, n).clip(-0.002, 0.003)

    return pd.DataFrame({
        "timestamp_ms": timestamps,
        "funding_rate": funding_rate,
        "symbol": "DOGEUSDT",
    })


def _make_regime_series(n: int, open_times: np.ndarray, seed: int = 48) -> pd.Series:
    """Generate regime labels indexed by open_time."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(_REGIME_LABELS, size=n)
    return pd.Series(labels, index=open_times)


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print a check result and return the pass/fail flag."""
    status = _PASS if passed else _FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def _warn(label: str, detail: str = "") -> None:
    """Print an advisory warning (does not affect pass/fail)."""
    msg = f"  {_WARN}  {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Main QG-03 logic
# ---------------------------------------------------------------------------


def run_qg03() -> bool:
    """Run all Quality Gate 03 checks.

    Returns:
        True if all required checks pass, False otherwise.
    """
    print()
    print("=" * 70)
    print("  Quality Gate 03 — Feature Pipeline Integrity Verification")
    print("=" * 70)
    print()

    all_pass: list[bool] = []

    # ------------------------------------------------------------------
    # Generate synthetic data
    # ------------------------------------------------------------------
    print("Generating synthetic fixture data ...")
    t0 = time.time()

    doge_1h = _make_ohlcv_1h(_N_1H_ROWS, _T0_MS, seed=42)
    doge_4h = _make_ohlcv_4h(_N_4H_ROWS, _T0_MS - 300 * _MS_PER_4H, seed=43)
    doge_1d = _make_ohlcv_1d(_N_1D_ROWS, _T0_MS - 50 * _MS_PER_1D, seed=44)
    btc_1h = _make_btc_1h(
        _N_1H_ROWS, _T0_MS, doge_1h["close"].to_numpy(), seed=45
    )
    dogebtc_1h = _make_dogebtc_1h(
        _N_1H_ROWS, _T0_MS,
        doge_1h["close"].to_numpy(), btc_1h["close"].to_numpy(), seed=46
    )
    funding = _make_funding(_N_FUNDING_ROWS, _T0_MS, seed=47)
    regimes = _make_regime_series(
        _N_1H_ROWS, doge_1h["open_time"].to_numpy(), seed=48
    )

    print(f"  Data generated in {time.time() - t0:.1f}s")
    print(
        f"  1h: {len(doge_1h)} rows | 4h: {len(doge_4h)} rows | "
        f"1d: {len(doge_1d)} rows | funding: {len(funding)} rows"
    )
    print()
    print("Running FeaturePipeline.compute_all_features() ...")

    # ------------------------------------------------------------------
    # Check 1: pipeline runs end-to-end
    # ------------------------------------------------------------------
    df: pd.DataFrame | None = None
    pipeline_ok = False
    pipeline_error = ""

    try:
        pipe = FeaturePipeline(cfg=doge_settings, output_dir=None)
        df = pipe.compute_all_features(
            doge_1h=doge_1h,
            btc_1h=btc_1h,
            dogebtc_1h=dogebtc_1h,
            funding=funding,
            doge_4h=doge_4h,
            doge_1d=doge_1d,
            regimes=regimes,
            min_rows_override=300,  # test-data override (prod uses 3000)
        )
        pipeline_ok = True
    except Exception as exc:
        pipeline_error = str(exc)

    elapsed = time.time() - t0
    all_pass.append(
        _check(
            "Check 1: Pipeline runs end-to-end",
            pipeline_ok,
            f"{elapsed:.1f}s elapsed" if pipeline_ok else pipeline_error,
        )
    )

    if df is None:
        print()
        print("Pipeline failed — cannot run remaining checks.")
        print()
        _summarise(all_pass)
        return False

    n_rows, n_cols = df.shape
    print(f"  Output: {n_rows} rows x {n_cols} cols")
    print()
    print("-" * 70)
    print("  Required Checks")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Check 2: zero NaN in feature matrix
    # ------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    nan_cols = [c for c in numeric_cols if df[c].isna().any()]
    all_pass.append(
        _check(
            "Check 2: Zero NaN in all feature columns",
            len(nan_cols) == 0,
            f"NaN in: {nan_cols}" if nan_cols else f"{len(numeric_cols)} numeric cols checked",
        )
    )

    # ------------------------------------------------------------------
    # Check 3: zero Inf in feature matrix
    # ------------------------------------------------------------------
    inf_cols = [c for c in numeric_cols if np.isinf(df[c].to_numpy()).any()]
    all_pass.append(
        _check(
            "Check 3: Zero Inf in all feature columns",
            len(inf_cols) == 0,
            f"Inf in: {inf_cols}" if inf_cols else "OK",
        )
    )

    # ------------------------------------------------------------------
    # Check 4: all 12 mandatory DOGE-specific features present
    # ------------------------------------------------------------------
    missing_doge = [f for f in DOGE_FEATURE_NAMES if f not in df.columns]
    all_pass.append(
        _check(
            f"Check 4: All {len(DOGE_FEATURE_NAMES)} DOGE-specific mandatory features present",
            len(missing_doge) == 0,
            f"missing: {missing_doge}" if missing_doge else "OK",
        )
    )

    # ------------------------------------------------------------------
    # Check 5: all regime feature columns present
    # ------------------------------------------------------------------
    missing_regime = [f for f in REGIME_FEATURE_KEYS if f not in df.columns]
    all_pass.append(
        _check(
            f"Check 5: All {len(REGIME_FEATURE_KEYS)} regime feature columns present",
            len(missing_regime) == 0,
            f"missing: {missing_regime}" if missing_regime else "OK",
        )
    )

    # ------------------------------------------------------------------
    # Check 6: target column has only 0.0 / 1.0 (no NaN after dropna)
    # ------------------------------------------------------------------
    target_ok = False
    target_detail = ""
    if "target" in df.columns:
        target_vals = df["target"].dropna().unique()
        bad_vals = [v for v in target_vals if v not in (0.0, 1.0)]
        has_nan = df["target"].isna().any()
        target_ok = len(bad_vals) == 0 and not has_nan
        if bad_vals:
            target_detail = f"unexpected values: {bad_vals}"
        elif has_nan:
            target_detail = "NaN present after dropna — warmup drop incomplete"
        else:
            n_pos = int((df["target"] == 1.0).sum())
            n_neg = int((df["target"] == 0.0).sum())
            target_detail = f"1={n_pos} 0={n_neg} ({100*n_pos/n_rows:.1f}% positive)"
    else:
        target_detail = "'target' column missing"

    all_pass.append(
        _check("Check 6: target column values are 0.0 or 1.0 only", target_ok, target_detail)
    )

    # ------------------------------------------------------------------
    # Check 7: lag sanity — log_ret_1[t] == log(close[t] / close[t-1])
    # ------------------------------------------------------------------
    lag_ok = False
    lag_detail = ""
    if "log_ret_1" in df.columns and "close" in df.columns:
        # For rows 1..N-1 (consecutive in the filtered output), verify
        # the computed log_ret_1 matches log(close[t] / close[t-1]).
        close_arr = df["close"].to_numpy()
        lr_arr = df["log_ret_1"].to_numpy()

        # Compute expected returns from consecutive rows in the output
        expected_lr = np.log(close_arr[1:] / close_arr[:-1])
        actual_lr = lr_arr[1:]  # skip first row (no prev in filtered df)

        # Allow small tolerance for floating-point
        max_err = float(np.max(np.abs(expected_lr - actual_lr)))
        lag_ok = max_err < 1e-8
        lag_detail = (
            f"max abs error = {max_err:.2e} (threshold 1e-8)"
            if lag_ok
            else f"MISMATCH: max abs error = {max_err:.2e}"
        )
    else:
        lag_detail = "log_ret_1 or close column missing — cannot verify"

    all_pass.append(_check("Check 7: log_ret_1 lag sanity", lag_ok, lag_detail))

    # ------------------------------------------------------------------
    # Check 8: HTF lookahead guard — htf_4h_rsi constant within each 4h period
    # ------------------------------------------------------------------
    htf_ok = False
    htf_detail = ""
    if "htf_4h_rsi" in df.columns and "open_time" in df.columns:
        # Assign a 4h period bucket to each row and check RSI uniqueness per bucket
        df_tmp = df[["open_time", "htf_4h_rsi"]].copy()
        df_tmp["_4h_period"] = df_tmp["open_time"] // _MS_PER_4H
        rsi_per_period = df_tmp.groupby("_4h_period")["htf_4h_rsi"].nunique()

        # All 4h periods with 4 consecutive rows should have exactly 1 RSI value.
        # Periods at the data boundary may have fewer than 4 rows — still OK.
        bad_periods = rsi_per_period[rsi_per_period > 1]
        htf_ok = len(bad_periods) == 0
        n_periods = len(rsi_per_period)
        htf_detail = (
            f"{n_periods} 4h periods checked; {len(bad_periods)} with intra-period change"
            if not htf_ok
            else f"{n_periods} 4h periods; RSI constant within every period"
        )
    else:
        htf_detail = "htf_4h_rsi or open_time column missing"

    all_pass.append(
        _check("Check 8: HTF lookahead guard — RSI constant within each 4h period", htf_ok, htf_detail)
    )

    # ------------------------------------------------------------------
    # Advisory checks (do not affect pass/fail)
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  Advisory Checks (do not affect PASS/FAIL)")
    print("-" * 70)

    # Advisory Check 9: high correlation pairs
    _run_correlation_check(df)

    # Advisory Check 10: ADF stationarity
    _run_stationarity_check(df)

    print()
    return _summarise(all_pass)


def _run_correlation_check(df: pd.DataFrame) -> None:
    """Print feature pairs with |correlation| > 0.98."""
    try:
        feat_cols = [
            c for c in df.columns
            if c not in {"open_time", "open", "high", "low", "close",
                         "volume", "symbol", "era", "regime_label", "target"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(feat_cols) < 2:
            _warn("Check 9: Correlation", "fewer than 2 feature columns")
            return

        corr_matrix = df[feat_cols].corr().abs()
        # Get upper triangle only
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(feat_cols)):
            for j in range(i + 1, len(feat_cols)):
                c = corr_matrix.iloc[i, j]
                if c > 0.98:
                    pairs.append((feat_cols[i], feat_cols[j], float(c)))

        if pairs:
            _warn(
                f"Check 9: {len(pairs)} highly correlated pairs (|corr| > 0.98)",
                f"top pair: {pairs[0][0]} vs {pairs[0][1]} ({pairs[0][2]:.4f})",
            )
            for a, b, v in pairs[:5]:
                print(f"            {a} x {b}: {v:.4f}")
            if len(pairs) > 5:
                print(f"            ... and {len(pairs) - 5} more")
        else:
            print(f"  {_PASS}  Check 9: No feature pairs with |corr| > 0.98")

    except Exception as exc:
        _warn("Check 9: Correlation check error", str(exc))


def _run_stationarity_check(df: pd.DataFrame) -> None:
    """Run ADF test on feature columns; flag non-stationary (p > 0.05)."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        _warn("Check 10: Stationarity", "statsmodels not installed — skipping")
        return

    feat_cols = [
        c for c in df.columns
        if c not in {"open_time", "open", "high", "low", "close",
                     "volume", "symbol", "era", "regime_label", "target"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    non_stationary: list[str] = []
    errors: list[str] = []

    # Test on a subset to avoid very long runtime
    _MAX_ADF_COLS = 50
    if len(feat_cols) > _MAX_ADF_COLS:
        print(
            f"  (ADF test limited to first {_MAX_ADF_COLS} of {len(feat_cols)} feature cols)"
        )
        feat_cols = feat_cols[:_MAX_ADF_COLS]

    for col in feat_cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue
        try:
            result = adfuller(series.to_numpy(), autolag="AIC", regression="c")
            p_val = float(result[1])
            if p_val > 0.05:
                non_stationary.append(f"{col} (p={p_val:.3f})")
        except Exception as exc:
            errors.append(f"{col}: {exc}")

    if non_stationary:
        _warn(
            f"Check 10: {len(non_stationary)} non-stationary features (ADF p > 0.05)",
            f"first: {non_stationary[0]}",
        )
        for f in non_stationary[:5]:
            print(f"            {f}")
        if len(non_stationary) > 5:
            print(f"            ... and {len(non_stationary) - 5} more")
    else:
        print(
            f"  {_PASS}  Check 10: All {len(feat_cols)} tested features are stationary (ADF p <= 0.05)"
        )

    if errors:
        _warn(f"ADF errors on {len(errors)} columns", errors[0])


def _summarise(all_pass: list[bool]) -> bool:
    """Print summary and return overall result."""
    n_pass = sum(all_pass)
    n_fail = len(all_pass) - n_pass
    overall = n_fail == 0

    print("=" * 70)
    if overall:
        print(f"  QG-03 RESULT: PASS  ({n_pass}/{len(all_pass)} required checks passed)")
    else:
        print(f"  QG-03 RESULT: FAIL  ({n_fail}/{len(all_pass)} required checks FAILED)")
    print("=" * 70)
    print()
    return overall


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = run_qg03()
    sys.exit(0 if passed else 1)
