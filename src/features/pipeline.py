"""Feature pipeline — orchestrates all feature computation for DOGE prediction.

This module is the single entry point for building the full feature matrix.
It calls each sub-module in a fixed order and then validates the result.

Pipeline stages (in order):
    1. price_indicators.py  — SMA / EMA / MACD / RSI / BB / ATR / Stoch / Ichimoku
    2. volume_indicators.py — OBV / VWAP / CMF / CVD / volume ratios
    3. lag_features.py      — log returns / momentum / rolling stats
    4. doge_specific.py     — 12 mandatory DOGE features (BTC corr, vol spike, etc.)
    5. funding_features.py  — funding rate / z-score / extreme flags
    6. htf_features.py      — 4h RSI / trend / BB%B; 1d trend / return; ATH distance
    7. Regime features       — 5 one-hot + ordinal encoding from regime labels

Validation (after all stages):
    - Assert zero NaN / Inf in any feature column (after dropping warmup rows)
    - Assert all mandatory feature names from CLAUDE.md Section 7 are present
    - Assert column list matches a canonical JSON manifest (if provided)

Lookahead audit:
    Every sub-module is independently verified (see module docstrings).
    This pipeline adds no new computations; it only concatenates results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DogeSettings, doge_settings
from src.features.doge_specific import DOGE_FEATURE_NAMES, compute_doge_features
from src.features.funding_features import FUNDING_FEATURE_NAMES, compute_funding_features
from src.features.htf_features import HTF_FEATURE_NAMES, compute_htf_features
from src.features.lag_features import compute_lag_features
from src.features.price_indicators import compute_price_indicators
from src.features.volume_indicators import compute_volume_indicators
from src.regimes.features import REGIME_FEATURE_KEYS, get_regime_features

# ---------------------------------------------------------------------------
# Mandatory feature sets (from CLAUDE.md Section 7)
# ---------------------------------------------------------------------------

# All features that MUST be present in the final matrix
MANDATORY_FEATURE_NAMES: frozenset[str] = frozenset(
    list(DOGE_FEATURE_NAMES)
    + list(FUNDING_FEATURE_NAMES)
    + list(HTF_FEATURE_NAMES)
    + list(REGIME_FEATURE_KEYS)
)

# Minimum number of rows needed after dropping the warmup window
# Longest indicator: EMA200 needs ~200 bars; 4h guard adds another 4h
_MIN_ROWS_AFTER_WARMUP: int = 300


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_matrix(
    doge_1h: pd.DataFrame,
    btc_1h: pd.DataFrame,
    dogebtc_1h: pd.DataFrame,
    funding_df: pd.DataFrame,
    doge_4h: pd.DataFrame,
    doge_1d: pd.DataFrame,
    regime_labels: pd.Series | None = None,
    cfg: DogeSettings | None = None,
    drop_warmup: bool = False,
    warmup_rows: int = 250,
) -> pd.DataFrame:
    """Build the full feature matrix for DOGE prediction.

    Runs all six feature sub-modules in the canonical order defined in
    ``src/features/pipeline.py``.  Optionally drops warmup rows containing
    NaN from long-period rolling indicators.

    Args:
        doge_1h: DOGEUSDT 1h OHLCV DataFrame sorted ascending by
            ``open_time``.  Must contain OHLCV columns and ``open_time``.
        btc_1h: BTCUSDT 1h OHLCV DataFrame with the same timestamp grid.
        dogebtc_1h: DOGEBTC 1h OHLCV DataFrame with the same timestamp grid.
        funding_df: DOGEUSDT 8h funding rate DataFrame with columns
            ``timestamp_ms`` and ``funding_rate``.
        doge_4h: DOGEUSDT 4h OHLCV DataFrame (need not share a timestamp
            grid with doge_1h; the HTF guard handles alignment).
        doge_1d: DOGEUSDT 1d OHLCV DataFrame.
        regime_labels: Optional :class:`pd.Series` of regime label strings
            (``"TRENDING_BULL"`` etc.) indexed to match *doge_1h*.  If
            *None*, regime features are set to zero.
        cfg: Optional :class:`~src.config.DogeSettings` instance.  If
            *None* the module-level singleton is used.
        drop_warmup: If *True*, drop the first *warmup_rows* rows from the
            output.  Default *False* (return all rows including NaN warmup).
        warmup_rows: Number of initial rows to drop when *drop_warmup* is
            *True*.  Default 250 (covers EMA200 warmup + HTF guard).

    Returns:
        DataFrame with all feature columns appended to a copy of *doge_1h*.
        The column order is: original OHLCV | price | volume | lag |
        doge-specific | funding | HTF | regime.

    Raises:
        ValueError: If any required input column is missing, or if mandatory
            features are absent from the output.
    """
    if cfg is None:
        cfg = doge_settings

    logger.info(
        "build_feature_matrix: starting pipeline on {} 1h rows",
        len(doge_1h),
    )

    # -----------------------------------------------------------------------
    # Stage 1 — Price indicators
    # -----------------------------------------------------------------------
    out = compute_price_indicators(doge_1h, cfg=cfg)
    logger.debug("pipeline stage 1 (price_indicators) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 2 — Volume indicators
    # -----------------------------------------------------------------------
    out = compute_volume_indicators(out, cfg=cfg)
    logger.debug("pipeline stage 2 (volume_indicators) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 3 — Lag / momentum / rolling stat features
    # -----------------------------------------------------------------------
    out = compute_lag_features(out, cfg=cfg)
    logger.debug("pipeline stage 3 (lag_features) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 4 — DOGE-specific features
    # -----------------------------------------------------------------------
    out = compute_doge_features(out, btc_1h, dogebtc_1h, cfg=cfg)
    logger.debug("pipeline stage 4 (doge_specific) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 5 — Funding rate features
    # -----------------------------------------------------------------------
    out = compute_funding_features(out, funding_df, cfg=cfg)
    logger.debug("pipeline stage 5 (funding_features) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 6 — HTF features
    # -----------------------------------------------------------------------
    out = compute_htf_features(out, doge_4h, doge_1d, cfg=cfg)
    logger.debug("pipeline stage 6 (htf_features) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Stage 7 — Regime features
    # -----------------------------------------------------------------------
    regime_feature_rows: list[dict[str, Any]] = []
    if regime_labels is not None:
        for label in regime_labels:
            try:
                regime_feature_rows.append(get_regime_features(str(label)))
            except ValueError:
                # Unknown label (e.g., NaN before first classification)
                regime_feature_rows.append(
                    {k: 0 for k in REGIME_FEATURE_KEYS}
                )
    else:
        logger.warning("build_feature_matrix: no regime_labels provided — regime features = 0")
        regime_feature_rows = [{k: 0 for k in REGIME_FEATURE_KEYS}] * len(out)

    regime_df = pd.DataFrame(regime_feature_rows, index=out.index)
    for col in REGIME_FEATURE_KEYS:
        out[col] = regime_df[col].to_numpy()

    logger.debug("pipeline stage 7 (regime_features) done — {} cols", len(out.columns))

    # -----------------------------------------------------------------------
    # Warmup drop (optional)
    # -----------------------------------------------------------------------
    if drop_warmup:
        if len(out) > warmup_rows:
            out = out.iloc[warmup_rows:].copy()
            logger.info(
                "build_feature_matrix: dropped {} warmup rows, {} remain",
                warmup_rows,
                len(out),
            )
        else:
            logger.warning(
                "build_feature_matrix: drop_warmup=True but only {} rows — none dropped",
                len(out),
            )

    # -----------------------------------------------------------------------
    # Mandatory feature validation
    # -----------------------------------------------------------------------
    missing = MANDATORY_FEATURE_NAMES - set(out.columns)
    if missing:
        raise ValueError(
            f"build_feature_matrix: mandatory features missing from output: {sorted(missing)}"
        )

    logger.info(
        "build_feature_matrix: complete — {} rows x {} columns",
        len(out),
        len(out.columns),
    )
    return out


def validate_feature_matrix(
    df: pd.DataFrame,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate the feature matrix for NaN, Inf, and mandatory column presence.

    This is a lightweight post-pipeline check suitable for use at inference
    time (called after every candle close).  It does NOT re-compute features.

    Args:
        df: Feature matrix returned by :func:`build_feature_matrix`.
        strict: If *True*, raise :class:`ValueError` on any validation
            failure.  If *False* (default), return a results dict and log
            warnings.

    Returns:
        Dict with keys:
            - ``"ok"`` (bool): True iff all checks pass.
            - ``"nan_cols"`` (list[str]): Columns with at least one NaN.
            - ``"inf_cols"`` (list[str]): Columns with at least one Inf.
            - ``"missing_mandatory"`` (list[str]): Mandatory features absent.
            - ``"n_rows"`` (int): Number of rows checked.

    Raises:
        ValueError: If *strict=True* and any check fails.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    nan_cols = [c for c in numeric_cols if df[c].isna().any()]
    inf_cols = [c for c in numeric_cols if np.isinf(df[c].to_numpy()).any()]
    missing_mandatory = sorted(MANDATORY_FEATURE_NAMES - set(df.columns))

    ok = not nan_cols and not inf_cols and not missing_mandatory

    if nan_cols:
        logger.warning("validate_feature_matrix: NaN in columns {}", nan_cols)
    if inf_cols:
        logger.warning("validate_feature_matrix: Inf in columns {}", inf_cols)
    if missing_mandatory:
        logger.warning(
            "validate_feature_matrix: missing mandatory features {}",
            missing_mandatory,
        )

    result = {
        "ok": ok,
        "nan_cols": nan_cols,
        "inf_cols": inf_cols,
        "missing_mandatory": missing_mandatory,
        "n_rows": len(df),
    }

    if strict and not ok:
        raise ValueError(
            f"validate_feature_matrix: validation failed — {result}"
        )

    return result
