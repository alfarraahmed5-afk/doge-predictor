"""Unit tests for src/monitoring/alerting.py, src/monitoring/drift_detector.py,
and src/monitoring/regime_monitor.py.

Run with: pytest tests/unit/test_alerting.py -v
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.monitoring.alerting import AlertManager
from src.monitoring.drift_detector import DriftDetector, DriftReport, FeatureDriftResult
from src.monitoring.regime_monitor import RegimeMonitor, RegimeMonitorStatus
from src.regimes.detector import RegimeChangeEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def log_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture()
def alert_mgr(log_dir: Path) -> AlertManager:
    return AlertManager(log_dir=log_dir)


@pytest.fixture()
def detector(alert_mgr: AlertManager) -> DriftDetector:
    return DriftDetector(alert_manager=alert_mgr)


@pytest.fixture()
def monitor(alert_mgr: AlertManager) -> RegimeMonitor:
    return RegimeMonitor(alert_manager=alert_mgr)


def _make_reference_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 1.0, n),
            "feature_b": rng.uniform(0.0, 1.0, n),
            "feature_c": rng.exponential(1.0, n),
        }
    )


def _make_event(from_r: str, to_r: str, ts: int = 0) -> RegimeChangeEvent:
    return RegimeChangeEvent(
        from_regime=from_r,
        to_regime=to_r,
        changed_at=ts,
        btc_corr=0.5,
        atr_norm=0.003,
        is_critical=(to_r == "DECOUPLED" or from_r == "DECOUPLED"),
    )


# ===========================================================================
# AlertManager tests
# ===========================================================================


class TestAlertManagerInit:
    def test_log_dir_created(self, log_dir: Path, alert_mgr: AlertManager) -> None:
        assert log_dir.exists()

    def test_alerts_log_created_on_first_send(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("INFO", "test", {})
        assert (log_dir / "alerts.log").exists()

    def test_critical_log_created_on_critical(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("CRITICAL", "critical test", {})
        assert (log_dir / "critical_alerts.log").exists()


class TestAlertManagerRouting:
    def test_info_only_in_alerts_log(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("INFO", "info msg", {"k": "v"})
        assert (log_dir / "alerts.log").exists()
        assert not (log_dir / "critical_alerts.log").exists()

    def test_warning_only_in_alerts_log(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("WARNING", "warn msg", {})
        assert (log_dir / "alerts.log").exists()
        assert not (log_dir / "critical_alerts.log").exists()

    def test_critical_in_both_logs(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("CRITICAL", "crit msg", {})
        assert (log_dir / "alerts.log").exists()
        assert (log_dir / "critical_alerts.log").exists()


class TestAlertManagerRecordFormat:
    def test_json_parseable(self, alert_mgr: AlertManager, log_dir: Path) -> None:
        alert_mgr.send_alert("INFO", "hello", {"x": 1})
        line = (log_dir / "alerts.log").read_text().strip().splitlines()[0]
        record = json.loads(line)
        assert record["level"] == "INFO"
        assert record["message"] == "hello"
        assert record["details"]["x"] == 1
        assert "timestamp_ms" in record

    def test_multiple_appended(self, alert_mgr: AlertManager, log_dir: Path) -> None:
        alert_mgr.send_alert("INFO", "first", {})
        alert_mgr.send_alert("INFO", "second", {})
        lines = (log_dir / "alerts.log").read_text().strip().splitlines()
        assert len(lines) == 2


class TestAlertManagerValidation:
    def test_invalid_level_raises(self, alert_mgr: AlertManager) -> None:
        with pytest.raises(ValueError, match="level"):
            alert_mgr.send_alert("DEBUG", "msg", {})

    def test_case_insensitive_level(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        alert_mgr.send_alert("warning", "msg", {})
        assert (log_dir / "alerts.log").exists()


class TestAlertManagerThreadSafety:
    def test_concurrent_writes_no_corruption(
        self, alert_mgr: AlertManager, log_dir: Path
    ) -> None:
        errors: list[Exception] = []

        def _send(i: int) -> None:
            try:
                alert_mgr.send_alert("INFO", f"msg-{i}", {"i": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_send, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        lines = (log_dir / "alerts.log").read_text().strip().splitlines()
        assert len(lines) == 20
        for line in lines:
            json.loads(line)  # must be valid JSON


# ===========================================================================
# DriftDetector tests
# ===========================================================================


class TestDriftDetectorSetReference:
    def test_stores_features(self, detector: DriftDetector) -> None:
        df = _make_reference_df()
        detector.set_reference(df)
        assert detector.has_reference()
        assert detector.n_reference_features() == 3

    def test_skips_non_numeric(self, detector: DriftDetector) -> None:
        df = _make_reference_df()
        df["text_col"] = "abc"
        detector.set_reference(df)
        assert detector.n_reference_features() == 3  # text_col excluded

    def test_skips_high_null_feature(self, detector: DriftDetector) -> None:
        df = _make_reference_df(n=100)
        df["sparse"] = np.where(np.arange(100) < 10, df["feature_a"], np.nan)
        detector.set_reference(df)
        assert "sparse" not in detector._reference_values

    def test_stores_proba_reference(self, detector: DriftDetector) -> None:
        df = _make_reference_df()
        proba = np.random.default_rng(0).uniform(0, 1, 500)
        detector.set_reference(df, proba=proba)
        assert detector._reference_proba is not None
        assert detector._reference_proba.shape[0] == 500


class TestDriftDetectorDetectNoReference:
    def test_empty_report_before_set_reference(self, detector: DriftDetector) -> None:
        df = _make_reference_df()
        report = detector.detect(df)
        assert report.n_features_checked == 0
        assert report.overall_drift_level == "NONE"


class TestDriftDetectorDetectNoDrift:
    def test_same_distribution_no_drift(self, detector: DriftDetector) -> None:
        # Use only normal distributions (stable PSI) with large samples
        rng = np.random.default_rng(7)
        ref_df = pd.DataFrame({"f1": rng.normal(0, 1, 2000), "f2": rng.normal(5, 2, 2000)})
        test_df = pd.DataFrame({"f1": rng.normal(0, 1, 500), "f2": rng.normal(5, 2, 500)})
        detector.set_reference(ref_df)
        report = detector.detect(test_df)
        assert report.n_features_checked == 2
        assert report.overall_drift_level == "NONE"
        assert report.max_psi < 0.10

    def test_report_fields_populated(self, detector: DriftDetector) -> None:
        df = _make_reference_df()
        detector.set_reference(df)
        report = detector.detect(df)
        assert isinstance(report, DriftReport)
        assert report.timestamp_ms > 0
        assert report.top_drifted_feature is not None
        assert len(report.feature_results) == 3


class TestDriftDetectorDetectDrift:
    def test_shifted_distribution_warning_or_critical(
        self, detector: DriftDetector
    ) -> None:
        rng = np.random.default_rng(0)
        ref_df = pd.DataFrame({"f": rng.normal(0.0, 1.0, 1000)})
        test_df = pd.DataFrame({"f": rng.normal(5.0, 1.0, 200)})
        detector.set_reference(ref_df)
        report = detector.detect(test_df)
        assert report.overall_drift_level in ("WARNING", "CRITICAL")
        assert report.max_psi >= 0.10

    def test_critical_drift_emits_alert(
        self, detector: DriftDetector, log_dir: Path
    ) -> None:
        rng = np.random.default_rng(0)
        ref_df = pd.DataFrame({"f": rng.normal(0.0, 1.0, 1000)})
        test_df = pd.DataFrame({"f": rng.normal(20.0, 1.0, 500)})
        detector.set_reference(ref_df)
        detector.detect(test_df)
        assert (log_dir / "alerts.log").exists()

    def test_feature_result_fields(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(0)
        ref_df = pd.DataFrame({"f": rng.normal(0.0, 1.0, 500)})
        test_df = pd.DataFrame({"f": rng.normal(3.0, 1.0, 200)})
        detector.set_reference(ref_df)
        report = detector.detect(test_df)
        result = report.feature_results[0]
        assert isinstance(result, FeatureDriftResult)
        assert result.feature == "f"
        assert result.psi >= 0.0
        assert 0.0 <= result.ks_statistic <= 1.0
        assert 0.0 <= result.ks_pvalue <= 1.0
        assert result.drift_level in ("NONE", "WARNING", "CRITICAL")

    def test_missing_column_in_test_skipped(self, detector: DriftDetector) -> None:
        ref_df = _make_reference_df()
        test_df = ref_df[["feature_a"]].copy()
        detector.set_reference(ref_df)
        report = detector.detect(test_df)
        assert report.n_features_checked == 1


class TestDriftDetectorPSI:
    def test_psi_near_zero_for_identical(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 1000)
        ref_counts, ref_edges = np.histogram(values, bins=10)
        ref_freq = ref_counts / ref_counts.sum()
        psi = detector._compute_psi(ref_edges, ref_freq, values)
        assert psi < 0.01

    def test_psi_large_for_shifted(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(42)
        ref_vals = rng.normal(0, 1, 1000)
        test_vals = rng.normal(10, 1, 1000)
        ref_counts, ref_edges = np.histogram(ref_vals, bins=10)
        ref_freq = ref_counts / ref_counts.sum()
        psi = detector._compute_psi(ref_edges, ref_freq, test_vals)
        assert psi >= 0.25

    def test_check_prediction_drift_no_reference(
        self, detector: DriftDetector
    ) -> None:
        proba = np.random.default_rng(0).uniform(0, 1, 100)
        assert detector.check_prediction_drift(proba) == 0.0

    def test_check_prediction_drift_same_dist(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(0)
        ref_proba = rng.uniform(0, 1, 1000)
        test_proba = rng.uniform(0, 1, 200)
        detector.set_reference(_make_reference_df(), proba=ref_proba)
        psi = detector.check_prediction_drift(test_proba)
        assert psi < 0.25


# ===========================================================================
# RegimeMonitor tests
# ===========================================================================


class TestRegimeMonitorInit:
    def test_initial_state(self, monitor: RegimeMonitor) -> None:
        status = monitor.tick(timestamp_ms=0)
        assert status.current_regime is None
        assert status.total_transitions == 0
        assert status.transition_rate_24h == 0
        assert status.current_duration_hours == 0.0


class TestRegimeMonitorOnRegimeChange:
    def test_records_transition(self, monitor: RegimeMonitor) -> None:
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", ts=1_000))
        assert monitor._total_transitions == 1
        assert monitor._current_span is not None
        assert monitor._current_span.regime == "TRENDING_BEAR"

    def test_closes_previous_span(self, monitor: RegimeMonitor) -> None:
        # First call opens the to_regime span (no prior span to close)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "RANGING_LOW_VOL", ts=1_000))
        # Second call closes RANGING_LOW_VOL and opens DECOUPLED
        monitor.on_regime_change(_make_event("RANGING_LOW_VOL", "DECOUPLED", ts=5_000))
        assert len(monitor._spans) == 1
        assert monitor._spans[0].regime == "RANGING_LOW_VOL"
        assert monitor._spans[0].end_ms == 5_000

    def test_regime_counts_incremented(self, monitor: RegimeMonitor) -> None:
        # First event opens TRENDING_BEAR span (TRENDING_BULL was never a span)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", ts=1_000))
        # Second event closes TRENDING_BEAR
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "RANGING_LOW_VOL", ts=2_000))
        assert monitor._regime_counts["TRENDING_BEAR"] == 1

    def test_transition_counter(self, monitor: RegimeMonitor) -> None:
        regimes = ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_LOW_VOL",
                   "RANGING_HIGH_VOL", "DECOUPLED"]
        for i in range(len(regimes) - 1):
            monitor.on_regime_change(
                _make_event(regimes[i], regimes[i + 1], ts=i * 1_000)
            )
        assert monitor._total_transitions == 4

    def test_status_current_regime(self, monitor: RegimeMonitor) -> None:
        ts = int(time.time() * 1_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", ts=ts))
        status = monitor.tick(timestamp_ms=ts + 3_600_000)
        assert status.current_regime == "DECOUPLED"
        assert abs(status.current_duration_hours - 1.0) < 0.01


class TestRegimeMonitorDurationAlerts:
    def test_decoupled_duration_alert(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        start_ms = 1_640_995_200_000  # 2022-01-01 00:00 UTC
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", ts=start_ms))
        monitor.tick(timestamp_ms=start_ms + 73 * 3_600_000)
        lines = (log_dir / "alerts.log").read_text().strip().splitlines()
        records = [json.loads(l) for l in lines]
        assert any("DECOUPLED" in r["message"] for r in records)
        assert any(r["level"] == "CRITICAL" for r in records)

    def test_decoupled_no_alert_before_threshold(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        # Entering DECOUPLED always fires a CRITICAL entry alert (correct behaviour).
        # This test verifies that no *duration* anomaly alert fires before 72h.
        start_ms = 1_640_995_200_000
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", ts=start_ms))
        monitor.tick(timestamp_ms=start_ms + 48 * 3_600_000)
        if (log_dir / "alerts.log").exists():
            records = [
                json.loads(line)
                for line in (log_dir / "alerts.log").read_text().strip().splitlines()
            ]
            # No duration-anomaly messages ("lasted Xh") should be present
            assert not any("lasted" in r["message"] for r in records)

    def test_generic_duration_alert(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        start_ms = 1_640_995_200_000
        monitor.on_regime_change(
            _make_event("TRENDING_BULL", "RANGING_LOW_VOL", ts=start_ms)
        )
        monitor.tick(timestamp_ms=start_ms + 241 * 3_600_000)
        assert (log_dir / "alerts.log").exists()

    def test_duration_alert_only_fires_once(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        # The entry CRITICAL and the duration CRITICAL are both written, but
        # the duration alert must fire exactly once across repeated tick() calls.
        start_ms = 1_640_995_200_000
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", ts=start_ms))
        for h in [73, 74, 75]:
            monitor.tick(timestamp_ms=start_ms + h * 3_600_000)
        lines = (log_dir / "alerts.log").read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        # Exactly one duration-anomaly alert (the "lasted Xh" message)
        duration_alerts = [r for r in records if "lasted" in r["message"]]
        assert len(duration_alerts) == 1


class TestRegimeMonitorOscillation:
    def test_oscillation_alert_on_rapid_transitions(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        ts = 0
        # 7 transitions within 1 hour window → exceeds threshold of 6
        pairs = [
            ("TRENDING_BULL", "TRENDING_BEAR"),
            ("TRENDING_BEAR", "RANGING_LOW_VOL"),
            ("RANGING_LOW_VOL", "TRENDING_BULL"),
            ("TRENDING_BULL", "RANGING_HIGH_VOL"),
            ("RANGING_HIGH_VOL", "TRENDING_BEAR"),
            ("TRENDING_BEAR", "RANGING_LOW_VOL"),
            ("RANGING_LOW_VOL", "DECOUPLED"),
        ]
        for i, (fr, to) in enumerate(pairs):
            monitor.on_regime_change(_make_event(fr, to, ts=ts + i * 100_000))
        assert (log_dir / "alerts.log").exists()

    def test_no_oscillation_under_threshold(
        self, monitor: RegimeMonitor, log_dir: Path
    ) -> None:
        ts = 0
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", ts=ts))
        monitor.on_regime_change(
            _make_event("TRENDING_BEAR", "RANGING_LOW_VOL", ts=ts + 100_000)
        )
        monitor.on_regime_change(
            _make_event("RANGING_LOW_VOL", "TRENDING_BULL", ts=ts + 200_000)
        )
        assert not (log_dir / "alerts.log").exists()

    def test_transition_rate_in_status(self, monitor: RegimeMonitor) -> None:
        now_ms = int(time.time() * 1_000)
        pairs = [
            ("TRENDING_BULL", "TRENDING_BEAR"),
            ("TRENDING_BEAR", "RANGING_LOW_VOL"),
            ("RANGING_LOW_VOL", "TRENDING_BULL"),
            ("TRENDING_BULL", "RANGING_HIGH_VOL"),
        ]
        for i, (fr, to) in enumerate(pairs):
            monitor.on_regime_change(_make_event(fr, to, ts=now_ms - (i * 60_000)))
        status = monitor.tick(timestamp_ms=now_ms)
        assert status.transition_rate_24h == 4


class TestRegimeMonitorTransitionLog:
    def test_get_transition_log(self, monitor: RegimeMonitor) -> None:
        # First event opens TRENDING_BEAR (no prior span to close)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", ts=1_000))
        # Second event closes TRENDING_BEAR span
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "DECOUPLED", ts=10_000))
        log = monitor.get_transition_log()
        assert len(log) == 1
        assert log[0]["regime"] == "TRENDING_BEAR"
        assert log[0]["end_ms"] == 10_000

    def test_get_regime_duration_hours_type(self, monitor: RegimeMonitor) -> None:
        monitor.on_regime_change(
            _make_event("TRENDING_BULL", "TRENDING_BEAR", ts=4 * 3_600_000)
        )
        hours = monitor.get_regime_duration_hours("TRENDING_BULL")
        assert isinstance(hours, float)
        assert hours >= 0.0

    def test_reset_clears_state(self, monitor: RegimeMonitor) -> None:
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", ts=1_000))
        monitor.reset()
        assert monitor._total_transitions == 0
        assert monitor._current_span is None
        assert len(monitor._spans) == 0
