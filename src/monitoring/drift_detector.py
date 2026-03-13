"""Concept drift detector for doge_predictor.

Monitors feature and prediction distributions for statistical drift using
Population Stability Index (PSI) and Kolmogorov-Smirnov tests.

PSI interpretation (industry standard):
    < 0.10  — No significant change
    0.10 – 0.25 — Moderate change (WARNING)
    >= 0.25 — Significant drift (CRITICAL — investigate / retrain)

Usage::

    from src.monitoring.drift_detector import DriftDetector
    from src.monitoring.alerting import AlertManager

    detector = DriftDetector(alert_manager=AlertManager(log_dir=Path("logs")))
    detector.set_reference(training_df)           # call once after training
    report = detector.detect(recent_feature_df)   # call each hour / on schedule
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.config import DogeSettings, doge_settings
from src.monitoring.alerting import AlertManager
from src.monitoring import prometheus_metrics as _prom

# ---------------------------------------------------------------------------
# Constants — all thresholds loaded from MonitoringSettings at runtime
# ---------------------------------------------------------------------------

_EPS: float = 1e-8  # Avoid log(0) in PSI computation


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDriftResult:
    """Drift metrics for a single feature.

    Attributes:
        feature: Feature column name.
        psi: Population Stability Index (0 = no drift).
        ks_statistic: KS test statistic (0–1).
        ks_pvalue: KS test p-value (< 0.05 = significant).
        drift_level: One of ``"NONE"``, ``"WARNING"``, ``"CRITICAL"``.
    """

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drift_level: str


@dataclass(frozen=True)
class DriftReport:
    """Aggregate drift report produced by :meth:`DriftDetector.detect`.

    Attributes:
        timestamp_ms: UTC epoch milliseconds when detection was run.
        n_features_checked: Number of features evaluated.
        n_warning: Features with WARNING drift level.
        n_critical: Features with CRITICAL drift level.
        feature_results: Per-feature drift metrics.
        overall_drift_level: Worst drift level across all features.
        top_drifted_feature: Feature name with highest PSI, or ``None``.
        max_psi: Highest PSI observed, or ``0.0`` if no features checked.
        prediction_drift_psi: PSI of prediction probability distribution
            (``None`` if not supplied).
    """

    timestamp_ms: int
    n_features_checked: int
    n_warning: int
    n_critical: int
    feature_results: list[FeatureDriftResult]
    overall_drift_level: str
    top_drifted_feature: str | None
    max_psi: float
    prediction_drift_psi: float | None


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Detects concept drift in model input features and prediction probabilities.

    Uses Population Stability Index (PSI) as the primary metric and the
    Kolmogorov-Smirnov test as a secondary measure.  Thresholds are loaded
    from :class:`~src.config.MonitoringSettings` — never hardcoded.

    Args:
        alert_manager: :class:`~src.monitoring.alerting.AlertManager` instance
            used to emit WARNING / CRITICAL alerts when drift is detected.
        doge_cfg: DOGE settings (defaults to global singleton if ``None``).
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        doge_cfg: DogeSettings | None = None,
    ) -> None:
        self._alert = alert_manager
        self._cfg: DogeSettings = doge_cfg or doge_settings
        self._mon = self._cfg.monitoring

        # Reference histograms: {feature_name: (bin_edges, reference_freqs)}
        self._reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # Raw reference values for KS test
        self._reference_values: dict[str, np.ndarray] = {}
        # Reference prediction probabilities (optional)
        self._reference_proba: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(
        self,
        df: pd.DataFrame,
        proba: np.ndarray | None = None,
    ) -> None:
        """Compute and store reference distributions from training data.

        Should be called once after training completes.  Only numeric columns
        with >= ``drift_min_non_null_pct`` non-null values are stored.

        Args:
            df: Training feature DataFrame.  Non-numeric columns are skipped.
            proba: Optional 1-D array of training prediction probabilities.
        """
        n_bins = self._mon.drift_n_bins
        min_non_null = self._mon.drift_min_non_null_pct

        self._reference.clear()
        self._reference_values.clear()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        skipped = 0

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                skipped += 1
                continue
            non_null_pct = series.shape[0] / df.shape[0]
            if non_null_pct < min_non_null:
                skipped += 1
                continue
            values = series.to_numpy(dtype=float)
            # Store raw values for KS test
            self._reference_values[col] = values
            # Build histogram (frequency, not counts)
            counts, edges = np.histogram(values, bins=n_bins)
            freq = counts / (counts.sum() + _EPS)
            self._reference[col] = (edges, freq)

        if proba is not None:
            arr = np.asarray(proba, dtype=float)
            self._reference_proba = arr[np.isfinite(arr)]

        logger.info(
            "DriftDetector reference set — {} features stored, {} skipped",
            len(self._reference),
            skipped,
        )

    def detect(
        self,
        df: pd.DataFrame,
        proba: np.ndarray | None = None,
    ) -> DriftReport:
        """Detect drift by comparing ``df`` against the stored reference.

        Emits WARNING or CRITICAL alerts via :attr:`alert_manager` when
        drift is detected.  If :meth:`set_reference` has not been called,
        returns an empty report.

        Args:
            df: Recent feature DataFrame (test window).
            proba: Optional 1-D array of recent prediction probabilities.

        Returns:
            :class:`DriftReport` with per-feature and aggregate metrics.
        """
        timestamp_ms = int(time.time() * 1_000)

        if not self._reference:
            logger.warning("DriftDetector.detect() called before set_reference() — returning empty report")
            return DriftReport(
                timestamp_ms=timestamp_ms,
                n_features_checked=0,
                n_warning=0,
                n_critical=0,
                feature_results=[],
                overall_drift_level="NONE",
                top_drifted_feature=None,
                max_psi=0.0,
                prediction_drift_psi=None,
            )

        results: list[FeatureDriftResult] = []

        for col, (ref_edges, ref_freq) in self._reference.items():
            if col not in df.columns:
                continue
            test_series = df[col].dropna()
            if test_series.shape[0] == 0:
                continue
            test_values = test_series.to_numpy(dtype=float)

            psi = self._compute_psi(ref_edges, ref_freq, test_values)
            ks_stat, ks_pval = self._compute_ks(col, test_values)
            level = self._psi_to_level(psi)

            results.append(
                FeatureDriftResult(
                    feature=col,
                    psi=float(psi),
                    ks_statistic=float(ks_stat),
                    ks_pvalue=float(ks_pval),
                    drift_level=level,
                )
            )

        n_warning = sum(1 for r in results if r.drift_level == "WARNING")
        n_critical = sum(1 for r in results if r.drift_level == "CRITICAL")

        overall = "NONE"
        if n_critical > 0:
            overall = "CRITICAL"
        elif n_warning > 0:
            overall = "WARNING"

        top: FeatureDriftResult | None = max(results, key=lambda r: r.psi, default=None)
        max_psi = top.psi if top else 0.0
        top_feature = top.feature if top else None

        pred_drift_psi: float | None = None
        if proba is not None and self._reference_proba is not None:
            ref_arr = self._reference_proba
            test_arr = np.asarray(proba, dtype=float)
            test_arr = test_arr[np.isfinite(test_arr)]
            if test_arr.shape[0] > 0:
                bin_edges = np.linspace(0.0, 1.0, self._mon.drift_n_bins + 1)
                ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
                ref_freq_norm = ref_counts / (ref_counts.sum() + _EPS)
                pred_drift_psi = self._compute_psi(bin_edges, ref_freq_norm, test_arr)

        self._emit_alerts(overall, n_warning, n_critical, top_feature, max_psi, pred_drift_psi)

        report = DriftReport(
            timestamp_ms=timestamp_ms,
            n_features_checked=len(results),
            n_warning=n_warning,
            n_critical=n_critical,
            feature_results=results,
            overall_drift_level=overall,
            top_drifted_feature=top_feature,
            max_psi=max_psi,
            prediction_drift_psi=pred_drift_psi,
        )

        logger.info(
            "DriftDetector run complete — level={} features_checked={} n_warn={} n_crit={} max_psi={:.4f}",
            overall,
            len(results),
            n_warning,
            n_critical,
            max_psi,
        )
        return report

    def check_prediction_drift(self, proba: np.ndarray) -> float:
        """Compute PSI for a prediction probability array against the reference.

        Args:
            proba: 1-D array of recent prediction probabilities in [0, 1].

        Returns:
            PSI value (0.0 if no reference stored).
        """
        if self._reference_proba is None or len(self._reference_proba) == 0:
            return 0.0
        test_arr = np.asarray(proba, dtype=float)
        test_arr = test_arr[np.isfinite(test_arr)]
        if test_arr.shape[0] == 0:
            return 0.0
        bin_edges = np.linspace(0.0, 1.0, self._mon.drift_n_bins + 1)
        ref_counts, _ = np.histogram(self._reference_proba, bins=bin_edges)
        ref_freq = ref_counts / (ref_counts.sum() + _EPS)
        return float(self._compute_psi(bin_edges, ref_freq, test_arr))

    def has_reference(self) -> bool:
        """Return True if reference distributions have been set.

        Returns:
            True when :meth:`set_reference` has been called and stored at
            least one feature distribution.
        """
        return len(self._reference) > 0

    def n_reference_features(self) -> int:
        """Return the number of features with stored reference distributions.

        Returns:
            Count of features available for drift comparison.
        """
        return len(self._reference)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_psi(
        self,
        ref_edges: np.ndarray,
        ref_freq: np.ndarray,
        test_values: np.ndarray,
    ) -> float:
        """Compute PSI between reference histogram and test values.

        PSI = sum((test_freq - ref_freq) * ln(test_freq / ref_freq))

        Args:
            ref_edges: Bin edges from the reference histogram.
            ref_freq: Normalised reference bin frequencies.
            test_values: Raw test sample values.

        Returns:
            PSI value >= 0.
        """
        test_counts, _ = np.histogram(test_values, bins=ref_edges)
        test_freq = test_counts / (test_counts.sum() + _EPS)

        # Smooth zero bins to avoid log(0)
        ref_freq_s = np.where(ref_freq == 0, _EPS, ref_freq)
        test_freq_s = np.where(test_freq == 0, _EPS, test_freq)

        psi = float(np.sum((test_freq_s - ref_freq_s) * np.log(test_freq_s / ref_freq_s)))
        return max(psi, 0.0)

    def _compute_ks(
        self,
        feature: str,
        test_values: np.ndarray,
    ) -> tuple[float, float]:
        """Run two-sample KS test between reference and test distributions.

        Args:
            feature: Column name (used to look up stored reference values).
            test_values: Test sample array.

        Returns:
            Tuple of (ks_statistic, ks_pvalue).
        """
        ref_vals = self._reference_values.get(feature)
        if ref_vals is None or ref_vals.shape[0] == 0:
            return 0.0, 1.0
        try:
            result = stats.ks_2samp(ref_vals, test_values)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return 0.0, 1.0

    def _psi_to_level(self, psi: float) -> str:
        """Map a PSI value to a drift level string.

        Args:
            psi: Population Stability Index value.

        Returns:
            ``"CRITICAL"``, ``"WARNING"``, or ``"NONE"``.
        """
        if psi >= self._mon.drift_psi_critical:
            return "CRITICAL"
        if psi >= self._mon.drift_psi_warning:
            return "WARNING"
        return "NONE"

    def _emit_alerts(
        self,
        overall: str,
        n_warning: int,
        n_critical: int,
        top_feature: str | None,
        max_psi: float,
        pred_psi: float | None,
    ) -> None:
        """Emit alerts for detected drift.

        Args:
            overall: Overall drift level string.
            n_warning: Count of WARNING features.
            n_critical: Count of CRITICAL features.
            top_feature: Name of the most drifted feature.
            max_psi: Highest PSI observed.
            pred_psi: PSI of prediction probability drift (may be None).
        """
        if overall == "NONE":
            return

        payload: dict[str, Any] = {
            "n_warning": n_warning,
            "n_critical": n_critical,
            "top_feature": top_feature,
            "max_psi": round(max_psi, 4),
        }
        if pred_psi is not None:
            payload["prediction_drift_psi"] = round(pred_psi, 4)

        level = "CRITICAL" if overall == "CRITICAL" else "WARNING"
        self._alert.send_alert(
            level=level,
            message=f"Feature drift detected — {n_critical} CRITICAL, {n_warning} WARNING features; max_psi={max_psi:.4f} ({top_feature})",
            details=payload,
        )


# ---------------------------------------------------------------------------
# SimpleDriftReport — returned by detect_feature_drift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleDriftReport:
    """Lightweight drift report from mean/std comparison.

    Attributes:
        drifted_features: List of feature names where the current mean
            deviated by more than ``3 × training_std``.
        max_deviation: Largest normalised deviation observed across all
            features (``|current_mean - train_mean| / train_std``).
        alert_level: ``"NONE"``, ``"WARNING"`` (>= 1 feature drifted), or
            ``"CRITICAL"`` (>= 3 features drifted).
        n_features_checked: Number of features included in the comparison.
    """

    drifted_features: list[str]
    max_deviation: float
    alert_level: str
    n_features_checked: int


# ---------------------------------------------------------------------------
# Module-level helpers for the two new detection methods
# ---------------------------------------------------------------------------


def detect_feature_drift(
    current_features: pd.DataFrame,
    training_stats: dict[str, dict[str, float]],
) -> SimpleDriftReport:
    """Detect drift by comparing current 24h feature window to training statistics.

    For each feature present in both *current_features* and *training_stats*,
    computes::

        deviation = |current_mean - train_mean| / (train_std + eps)

    A feature is flagged as drifted when ``deviation > 3``.

    Args:
        current_features: DataFrame of current (recent) feature values.
            Only numeric columns are evaluated.
        training_stats: Dict mapping feature name →
            ``{"mean": float, "std": float}``.

    Returns:
        :class:`SimpleDriftReport` with drifted feature names, max deviation,
        and an alert level.
    """
    numeric_cols = current_features.select_dtypes(include="number").columns.tolist()
    drifted: list[str] = []
    max_dev: float = 0.0
    n_checked: int = 0

    for col in numeric_cols:
        stats = training_stats.get(col)
        if stats is None:
            continue
        train_mean = float(stats.get("mean", 0.0))
        train_std = float(stats.get("std", 0.0))
        if not (current_features[col].notna().any()):
            continue

        current_mean = float(current_features[col].mean())
        deviation = abs(current_mean - train_mean) / (train_std + 1e-8)
        if deviation > max_dev:
            max_dev = deviation
        if deviation > 3.0:
            drifted.append(col)
        n_checked += 1

    # Alert level
    if len(drifted) >= 3:
        level = "CRITICAL"
    elif len(drifted) >= 1:
        level = "WARNING"
    else:
        level = "NONE"

    logger.debug(
        "detect_feature_drift: {} features checked, {} drifted (max_dev={:.2f}), level={}",
        n_checked, len(drifted), max_dev, level,
    )
    return SimpleDriftReport(
        drifted_features=drifted,
        max_deviation=max_dev,
        alert_level=level,
        n_features_checked=n_checked,
    )


def detect_regime_drift(regime_history: "pd.Series[str]") -> bool:
    """Detect rapid regime oscillation (unstable market / classifier issue).

    Returns ``True`` if the regime changed more than 3 times within any
    6-hour rolling window of the supplied history.

    Args:
        regime_history: Pandas Series of regime label strings, indexed by
            UTC epoch milliseconds (``open_time``).  Must be sorted ascending.

    Returns:
        ``True`` when rapid oscillation is detected; ``False`` otherwise.
    """
    if regime_history.empty or len(regime_history) < 2:
        return False

    _6H_MS: int = 6 * 3_600_000
    timestamps = regime_history.index.to_numpy(dtype="int64")
    labels = regime_history.to_numpy(dtype=str)

    # Count transitions in a sliding 6-hour window
    n = len(labels)
    for i in range(n):
        window_end = timestamps[i] + _6H_MS
        transitions = 0
        for j in range(i + 1, n):
            if timestamps[j] > window_end:
                break
            if labels[j] != labels[j - 1]:
                transitions += 1
        if transitions > 3:
            logger.warning(
                "detect_regime_drift: {} transitions in 6h window ending at {}",
                transitions,
                timestamps[i] + _6H_MS,
            )
            return True

    return False
