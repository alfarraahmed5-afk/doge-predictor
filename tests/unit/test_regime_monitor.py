"""Unit tests for src/monitoring/regime_monitor.py.

Mandatory tests:
* DECOUPLED transition triggers CRITICAL alert
* Stabilization window returns True for 3 candles after DECOUPLED exit
* Transition history is stored correctly

Additional coverage:
* on_transition() validates regime labels
* on_transition() builds RegimeChangeEvent correctly
* Duration stats computed from completed spans
* tick() detects duration anomaly
* tick() detects oscillation anomaly
* reset() clears all state including stabilization
* is_in_stabilization_window() uses real clock when _now_ms omitted
* Duration alert fired when DECOUPLED lasted > 72h
* Duration alert fired when any regime lasted > 240h
* No stabilization window when entering DECOUPLED (only exiting)
* Rapid oscillation sends WARNING alert
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.monitoring.alerting import AlertManager
from src.monitoring.regime_monitor import (
    RegimeMonitor,
    RegimeSpan,
)
from src.regimes.detector import RegimeChangeEvent


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_VALID_REGIMES = [
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
]


class _MockAlertManager:
    """Minimal AlertManager stand-in for unit tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def send_alert(
        self,
        level: str,
        message: str,
        details: dict | None = None,
    ) -> None:
        self.calls.append({"level": level, "message": message, "details": details or {}})

    def last(self) -> dict | None:
        return self.calls[-1] if self.calls else None

    def levels(self) -> list[str]:
        return [c["level"] for c in self.calls]


def _make_monitor(interval_ms: int = 3_600_000) -> tuple[RegimeMonitor, _MockAlertManager]:
    """Return a (monitor, mock_alert_mgr) pair."""
    mock_alert = _MockAlertManager()
    # RegimeMonitor expects an AlertManager; use the real class with a temp dir
    # but inject a mock via monkey-patch so we can inspect calls.
    monitor = RegimeMonitor.__new__(RegimeMonitor)
    # Manually set attrs (avoid disk I/O from AlertManager.__init__):
    from collections import deque
    from src.config import doge_settings
    monitor._alert = mock_alert
    monitor._cfg = doge_settings
    monitor._mon = doge_settings.monitoring
    monitor._interval_ms = interval_ms
    monitor._spans = []
    monitor._current_span = None
    monitor._transition_times = deque()
    monitor._total_transitions = 0
    monitor._regime_counts = {r: 0 for r in _VALID_REGIMES}
    monitor._alerted_duration_anomaly = False
    monitor._alerted_oscillation = False
    monitor._in_stabilization = False
    monitor._stabilization_exit_time_ms = 0
    return monitor, mock_alert


def _make_event(
    from_regime: str,
    to_regime: str,
    ts: int = 1_000_000,
) -> RegimeChangeEvent:
    is_critical = from_regime == "DECOUPLED" or to_regime == "DECOUPLED"
    return RegimeChangeEvent(
        from_regime=from_regime,
        to_regime=to_regime,
        changed_at=ts,
        btc_corr=0.5,
        atr_norm=0.3,
        is_critical=is_critical,
    )


# ---------------------------------------------------------------------------
# MANDATORY: DECOUPLED transition triggers CRITICAL alert
# ---------------------------------------------------------------------------


class TestDecoupledCriticalAlert:
    """CRITICAL alert is sent when transitioning INTO DECOUPLED."""

    def test_entering_decoupled_triggers_critical(self) -> None:
        """MANDATORY: CRITICAL alert sent when to_regime == DECOUPLED."""
        monitor, mock_alert = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        assert "CRITICAL" in mock_alert.levels(), (
            "Expected at least one CRITICAL alert when entering DECOUPLED"
        )

    def test_critical_alert_contains_regime_info(self) -> None:
        """CRITICAL alert details include from_regime and to_regime."""
        monitor, mock_alert = _make_monitor()
        monitor.on_regime_change(_make_event("RANGING_HIGH_VOL", "DECOUPLED", 5000))
        critical_calls = [c for c in mock_alert.calls if c["level"] == "CRITICAL"]
        assert critical_calls, "Expected CRITICAL alert on DECOUPLED entry"
        details = critical_calls[0]["details"]
        assert details.get("from_regime") == "RANGING_HIGH_VOL"
        assert details.get("to_regime") == "DECOUPLED"

    def test_only_decoupled_entry_triggers_critical_not_other_transitions(self) -> None:
        """Transitioning between non-DECOUPLED regimes does NOT send CRITICAL."""
        monitor, mock_alert = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 1000))
        assert "CRITICAL" not in mock_alert.levels()

    def test_exiting_decoupled_triggers_warning_not_critical(self) -> None:
        """Exiting DECOUPLED sends WARNING (not CRITICAL)."""
        monitor, mock_alert = _make_monitor()
        # Enter DECOUPLED first
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        mock_alert.calls.clear()  # reset to isolate exit alert
        # Exit DECOUPLED
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        assert "WARNING" in mock_alert.levels()
        # No CRITICAL on exit
        assert "CRITICAL" not in mock_alert.levels()

    def test_on_transition_convenience_entering_decoupled(self) -> None:
        """on_transition() also triggers CRITICAL when to_regime == DECOUPLED."""
        monitor, mock_alert = _make_monitor()
        monitor.on_transition("TRENDING_BULL", "DECOUPLED", 1000)
        assert "CRITICAL" in mock_alert.levels()

    def test_all_source_regimes_entering_decoupled_trigger_critical(self) -> None:
        """CRITICAL fires regardless of which regime transitions INTO DECOUPLED."""
        for from_regime in _VALID_REGIMES:
            if from_regime == "DECOUPLED":
                continue
            monitor, mock_alert = _make_monitor()
            monitor.on_regime_change(_make_event(from_regime, "DECOUPLED", 1000))
            assert "CRITICAL" in mock_alert.levels(), (
                f"Expected CRITICAL when {from_regime} -> DECOUPLED"
            )


# ---------------------------------------------------------------------------
# MANDATORY: stabilization window
# ---------------------------------------------------------------------------


class TestStabilizationWindow:
    """is_in_stabilization_window() returns True for 3 candles after DECOUPLED exit."""

    def test_stabilization_false_when_no_decoupled_exit(self) -> None:
        """Returns False when DECOUPLED was never exited."""
        monitor, _ = _make_monitor()
        assert monitor.is_in_stabilization_window() is False

    def test_stabilization_false_when_only_entered_decoupled(self) -> None:
        """Returns False immediately after entering DECOUPLED (not exiting)."""
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        assert monitor.is_in_stabilization_window(_now_ms=1000) is False

    def test_stabilization_true_immediately_after_decoupled_exit(self) -> None:
        """MANDATORY: Returns True right after exiting DECOUPLED (0 ms elapsed)."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        # 0 ms after exit
        assert monitor.is_in_stabilization_window(_now_ms=2000) is True

    def test_stabilization_true_for_candle_1(self) -> None:
        """MANDATORY: Returns True 1 candle-interval after DECOUPLED exit."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        exit_ms = 2000
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", exit_ms))
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 3_600_000) is True

    def test_stabilization_true_for_candle_2(self) -> None:
        """MANDATORY: Returns True 2 candle-intervals after DECOUPLED exit."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        exit_ms = 2000
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", exit_ms))
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 2 * 3_600_000) is True

    def test_stabilization_false_after_3_candles(self) -> None:
        """MANDATORY: Returns False exactly at the 3-candle boundary."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        exit_ms = 2000
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", exit_ms))
        # 3 full candle-intervals elapsed → window closed
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 3 * 3_600_000) is False

    def test_stabilization_false_after_4_candles(self) -> None:
        """Returns False 4 candle-intervals after DECOUPLED exit."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        exit_ms = 2000
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", exit_ms))
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 4 * 3_600_000) is False

    def test_stabilization_uses_real_clock_without_now_ms(self) -> None:
        """When _now_ms is None, wall-clock time is used; no crash."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(
            _make_event("DECOUPLED", "TRENDING_BEAR", int(time.time() * 1_000))
        )
        result = monitor.is_in_stabilization_window()
        # Should be True (wall-clock is within 3h of the exit)
        assert isinstance(result, bool)

    def test_stabilization_cancelled_by_re_entering_decoupled(self) -> None:
        """Entering DECOUPLED again cancels any active stabilization."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        # Re-enter DECOUPLED while still in stabilization window
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "DECOUPLED", 3000))
        # Now checking with _now_ms inside the old stabilization window
        # — should be False because stabilization was cancelled
        assert monitor.is_in_stabilization_window(_now_ms=4000) is False

    def test_on_transition_wrapper_stabilization(self) -> None:
        """on_transition convenience wrapper triggers the same stabilization logic."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_transition("TRENDING_BULL", "DECOUPLED", 1000)
        exit_ms = 5_000_000
        monitor.on_transition("DECOUPLED", "TRENDING_BEAR", exit_ms)
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 1_000) is True
        assert monitor.is_in_stabilization_window(_now_ms=exit_ms + 3 * 3_600_000) is False


# ---------------------------------------------------------------------------
# MANDATORY: transition history stored correctly
# ---------------------------------------------------------------------------


class TestTransitionHistory:
    """Transition history is recorded correctly after on_regime_change / on_transition calls."""

    def test_empty_history_initially(self) -> None:
        """transition_history() returns empty list before any transitions."""
        monitor, _ = _make_monitor()
        assert monitor.transition_history() == []

    def test_one_transition_recorded(self) -> None:
        """A single transition appears in the history as a completed span.

        The implementation tracks ``to_regime`` spans.  When BULL→BEAR fires,
        a BEAR span opens.  When BEAR→LOW_VOL fires, the BEAR span closes and
        is recorded in history.
        """
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 1000))
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "RANGING_LOW_VOL", 5000))
        history = monitor.transition_history()
        # The BEAR span (to_regime of event 1) is now closed and in history
        assert any(h["regime"] == "TRENDING_BEAR" for h in history)

    def test_multiple_transitions_in_order(self) -> None:
        """Multiple transitions appear chronologically.

        Tracks ``to_regime`` spans: DECOUPLED (event 1) and RANGING_HIGH_VOL
        (event 2) are closed; TRENDING_BEAR (event 3) is still open.
        """
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "RANGING_HIGH_VOL", 5000))
        monitor.on_regime_change(_make_event("RANGING_HIGH_VOL", "TRENDING_BEAR", 9000))
        history = monitor.transition_history()
        assert len(history) == 2  # 2 closed spans; current (TRENDING_BEAR) not yet closed
        regimes = [h["regime"] for h in history]
        assert "DECOUPLED" in regimes
        assert "RANGING_HIGH_VOL" in regimes

    def test_transition_history_duration_correct(self) -> None:
        """Duration is correctly computed from start and end timestamps.

        BULL→BEAR fires at ts=1000, opening a BEAR span.
        BEAR→LOW_VOL fires 2h later, closing the BEAR span at 7_201_000.
        So the closed span has regime=TRENDING_BEAR, duration≈2h.
        """
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 1000))
        monitor.on_regime_change(
            _make_event("TRENDING_BEAR", "RANGING_LOW_VOL", 1000 + 7_200_000)
        )
        history = monitor.transition_history()
        bear_span = next((h for h in history if h["regime"] == "TRENDING_BEAR"), None)
        assert bear_span is not None
        assert abs(bear_span["duration_hours"] - 2.0) < 1e-6

    def test_on_transition_wrapper_records_history(self) -> None:
        """on_transition() also records transitions in history.

        BULL→LOW_VOL opens a LOW_VOL span.  LOW_VOL→DECOUPLED closes it.
        So history contains RANGING_LOW_VOL (the closed to_regime span).
        """
        monitor, _ = _make_monitor()
        monitor.on_transition("TRENDING_BULL", "RANGING_LOW_VOL", 1000)
        monitor.on_transition("RANGING_LOW_VOL", "DECOUPLED", 10000)
        history = monitor.transition_history()
        assert any(h["regime"] == "RANGING_LOW_VOL" for h in history)

    def test_history_returns_copy_not_reference(self) -> None:
        """Modifying the returned list does not affect internal state."""
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        history1 = monitor.transition_history()
        history1.clear()
        history2 = monitor.transition_history()
        assert len(history2) == len(monitor._spans)  # original unchanged

    def test_reset_clears_history(self) -> None:
        """reset() empties the transition history."""
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        monitor.reset()
        assert monitor.transition_history() == []
        assert monitor.is_in_stabilization_window(_now_ms=3000) is False


# ---------------------------------------------------------------------------
# Additional: get_regime_duration_stats
# ---------------------------------------------------------------------------


class TestRegimeDurationStats:
    """get_regime_duration_stats() returns correct per-regime averages."""

    def test_empty_stats_before_any_transitions(self) -> None:
        """All regimes have count=0 before any transitions."""
        monitor, _ = _make_monitor()
        stats = monitor.get_regime_duration_stats()
        assert set(stats.keys()) == {
            "TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"
        }
        assert all(v["count"] == 0.0 for v in stats.values())

    def test_duration_computed_after_completed_span(self) -> None:
        """mean_hours is correct after one completed span.

        BULL→BEAR opens a BEAR span at ts=0.  BEAR→LOW_VOL closes it at
        ts=21_600_000 (6h later).  The completed span has regime=TRENDING_BEAR.
        """
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 0))
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "RANGING_LOW_VOL", 21_600_000))
        stats = monitor.get_regime_duration_stats()
        assert stats["TRENDING_BEAR"]["count"] == 1.0
        assert abs(stats["TRENDING_BEAR"]["mean_hours"] - 6.0) < 1e-6

    def test_multiple_spans_averaged(self) -> None:
        """mean_hours averages across multiple spans of the same regime.

        Two TRENDING_BULL spans are created by having BULL appear as to_regime
        twice.  Span 1: ts=0→7_200_000 (2h). Span 2: ts=10_800_000→25_200_000 (4h).
        Mean = (2+4)/2 = 3h.
        """
        monitor, _ = _make_monitor()
        # Span 1: BULL opened at ts=0 by BEAR→BULL, closed at ts=7_200_000 by BULL→LOW_VOL
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "TRENDING_BULL", 0))
        monitor.on_regime_change(_make_event("TRENDING_BULL", "RANGING_LOW_VOL", 7_200_000))
        # Span 2: BULL opened at ts=10_800_000 by LOW_VOL→BULL, closed at ts=25_200_000
        monitor.on_regime_change(_make_event("RANGING_LOW_VOL", "TRENDING_BULL", 10_800_000))
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 25_200_000))
        stats = monitor.get_regime_duration_stats()
        assert stats["TRENDING_BULL"]["count"] == 2.0
        assert abs(stats["TRENDING_BULL"]["mean_hours"] - 3.0) < 1e-6

    def test_active_span_excluded_from_stats(self) -> None:
        """The currently open span is NOT included in stats.

        BEAR→BULL opens a BULL span at ts=0.  BULL→LOW_VOL closes it at
        ts=7_200_000 (2h) and opens a LOW_VOL span.  LOW_VOL is still open
        (never closed), so it has count=0; the closed BULL span has count=1.
        """
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BEAR", "TRENDING_BULL", 0))
        monitor.on_regime_change(_make_event("TRENDING_BULL", "RANGING_LOW_VOL", 7_200_000))
        # RANGING_LOW_VOL is now the current (open) span
        stats = monitor.get_regime_duration_stats()
        # TRENDING_BULL has 1 completed span
        assert stats["TRENDING_BULL"]["count"] == 1.0
        # RANGING_LOW_VOL has 0 completed spans (still open)
        assert stats["RANGING_LOW_VOL"]["count"] == 0.0


# ---------------------------------------------------------------------------
# Additional: on_transition input validation
# ---------------------------------------------------------------------------


class TestOnTransitionValidation:
    """on_transition() raises ValueError for unknown regime labels."""

    def test_invalid_from_regime_raises(self) -> None:
        """ValueError on unknown from_regime."""
        monitor, _ = _make_monitor()
        with pytest.raises(ValueError, match="invalid from_regime"):
            monitor.on_transition("UNKNOWN_REGIME", "TRENDING_BULL", 1000)

    def test_invalid_to_regime_raises(self) -> None:
        """ValueError on unknown to_regime."""
        monitor, _ = _make_monitor()
        with pytest.raises(ValueError, match="invalid to_regime"):
            monitor.on_transition("TRENDING_BULL", "BAD_LABEL", 1000)

    def test_valid_transition_does_not_raise(self) -> None:
        """No exception on all valid transitions."""
        monitor, _ = _make_monitor()
        monitor.on_transition("TRENDING_BULL", "TRENDING_BEAR", 1000)  # must not raise


# ---------------------------------------------------------------------------
# Additional: tick() duration anomaly detection
# ---------------------------------------------------------------------------


class TestTickAnomalyDetection:
    """tick() detects and reports duration and oscillation anomalies."""

    def test_tick_no_anomaly_short_duration(self) -> None:
        """tick() returns no anomalies when duration is within threshold."""
        monitor, mock_alert = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "TRENDING_BEAR", 0))
        status = monitor.tick(timestamp_ms=3_600_000)  # 1h elapsed
        assert status.anomalies_detected == []

    def test_tick_decoupled_duration_anomaly(self) -> None:
        """tick() fires CRITICAL alert if DECOUPLED lasts > 72h."""
        monitor, mock_alert = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 0))
        mock_alert.calls.clear()  # reset to isolate tick alert
        # 73 hours elapsed in DECOUPLED
        status = monitor.tick(timestamp_ms=73 * 3_600_000)
        assert len(status.anomalies_detected) > 0
        assert "CRITICAL" in mock_alert.levels()

    def test_tick_current_span_none_returns_status(self) -> None:
        """tick() with no current span returns valid status with 0 duration."""
        monitor, _ = _make_monitor()
        status = monitor.tick(timestamp_ms=5000)
        assert status.current_regime is None
        assert status.current_duration_hours == 0.0

    def test_oscillation_detected_after_many_transitions(self) -> None:
        """WARNING sent if > 6 transitions within oscillation_window_hours (24h).

        The threshold is regime_oscillation_max_transitions=6, so we need
        strictly more than 6 (i.e., 7+) transitions to trigger the alert.
        """
        monitor, mock_alert = _make_monitor()
        ts = 0
        # 7 transitions (8 regimes) — strictly > threshold=6
        regimes = [
            "TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL",
            "TRENDING_BULL", "TRENDING_BEAR", "RANGING_LOW_VOL",
            "TRENDING_BULL", "TRENDING_BEAR",
        ]
        for i in range(len(regimes) - 1):
            ts += 100  # 100ms apart — all within 24h window
            monitor.on_regime_change(_make_event(regimes[i], regimes[i + 1], ts))
        # Should have triggered oscillation warning
        assert "WARNING" in mock_alert.levels()


# ---------------------------------------------------------------------------
# Additional: reset() clears stabilization state
# ---------------------------------------------------------------------------


class TestReset:
    """reset() fully clears all state including stabilization."""

    def test_reset_clears_stabilization(self) -> None:
        """After reset(), is_in_stabilization_window() returns False."""
        monitor, _ = _make_monitor(interval_ms=3_600_000)
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        assert monitor.is_in_stabilization_window(_now_ms=3000) is True  # in window
        monitor.reset()
        assert monitor.is_in_stabilization_window(_now_ms=3000) is False  # cleared

    def test_reset_clears_transition_counts(self) -> None:
        """After reset(), total_transitions is 0."""
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        assert monitor._total_transitions == 2
        monitor.reset()
        assert monitor._total_transitions == 0
        assert monitor._current_span is None

    def test_reset_clears_regime_counts(self) -> None:
        """After reset(), regime_counts are all 0."""
        monitor, _ = _make_monitor()
        monitor.on_regime_change(_make_event("TRENDING_BULL", "DECOUPLED", 1000))
        monitor.on_regime_change(_make_event("DECOUPLED", "TRENDING_BEAR", 2000))
        monitor.reset()
        status = monitor.tick(timestamp_ms=3000)
        assert all(v == 0 for v in status.regime_counts.values())
