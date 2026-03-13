"""Live regime change monitor for doge_predictor.

Tracks regime durations, transition rates, and emits alerts for anomalous
regime behaviour such as:

* Unexpectedly long DECOUPLED regime duration
* Any regime persisting beyond a configurable maximum
* Rapid regime oscillation (flip-flopping) within a short window

Also manages the **3-candle post-DECOUPLED stabilisation window**: after
leaving DECOUPLED the elevated 0.72 confidence threshold remains active for
3 candle-intervals so the inference engine does not immediately revert to
lower thresholds.

All thresholds are loaded from :class:`~src.config.MonitoringSettings`
— never hardcoded.

Usage::

    from src.monitoring.regime_monitor import RegimeMonitor
    from src.monitoring.alerting import AlertManager
    from src.regimes.detector import RegimeChangeEvent

    monitor = RegimeMonitor(alert_manager=AlertManager(Path("logs")))
    monitor.on_regime_change(event)   # called by InferenceEngine on each transition
    monitor.tick(timestamp_ms)        # called each candle close — checks for duration alerts
    # Simple convenience wrapper (same as on_regime_change but no RegimeChangeEvent dep):
    monitor.on_transition("TRENDING_BULL", "DECOUPLED", timestamp_ms)
    if monitor.is_in_stabilization_window():
        threshold = 0.72  # keep elevated threshold
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.config import DogeSettings, doge_settings
from src.monitoring.alerting import AlertManager
from src.regimes.detector import RegimeChangeEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_DEFAULT_INTERVAL_MS: int = 3_600_000  # 1 h primary candle interval
_STABILIZATION_CANDLES: int = 3        # candles to wait after leaving DECOUPLED
_DECOUPLED: str = "DECOUPLED"
_ALL_REGIMES: frozenset[str] = frozenset(
    {"TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"}
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RegimeSpan:
    """A contiguous period spent in a single regime.

    Attributes:
        regime: Regime label string.
        start_ms: UTC epoch milliseconds when the regime started.
        end_ms: UTC epoch milliseconds when the regime ended (0 = ongoing).
    """

    regime: str
    start_ms: int
    end_ms: int = 0

    @property
    def duration_hours(self) -> float:
        """Duration of this span in hours (uses current time if ongoing)."""
        end = self.end_ms if self.end_ms > 0 else int(time.time() * 1_000)
        return (end - self.start_ms) / _MS_PER_HOUR


@dataclass(frozen=True)
class RegimeMonitorStatus:
    """Snapshot of the regime monitor state.

    Attributes:
        current_regime: Current active regime label.
        current_duration_hours: Hours spent in the current regime.
        total_transitions: Cumulative regime transition count.
        transition_rate_24h: Transitions in the last 24 hours.
        regime_counts: Dict mapping regime label → number of completed spans.
        anomalies_detected: List of active anomaly descriptions.
    """

    current_regime: str | None
    current_duration_hours: float
    total_transitions: int
    transition_rate_24h: int
    regime_counts: dict[str, int]
    anomalies_detected: list[str]


# ---------------------------------------------------------------------------
# RegimeMonitor
# ---------------------------------------------------------------------------


class RegimeMonitor:
    """Monitors live regime changes and alerts on anomalous behaviour.

    Maintains a history of :class:`RegimeSpan` objects, counts transitions,
    and checks for duration / oscillation anomalies on each update.

    Args:
        alert_manager: :class:`~src.monitoring.alerting.AlertManager` used
            for emitting WARNING / CRITICAL alerts.
        doge_cfg: DOGE settings (defaults to global singleton if ``None``).
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        doge_cfg: DogeSettings | None = None,
        interval_ms: int = _DEFAULT_INTERVAL_MS,
    ) -> None:
        self._alert = alert_manager
        self._cfg: DogeSettings = doge_cfg or doge_settings
        self._mon = self._cfg.monitoring
        self._interval_ms: int = interval_ms

        # History of completed + current spans
        self._spans: list[RegimeSpan] = []
        # Current open span (None before first regime assignment)
        self._current_span: RegimeSpan | None = None
        # Recent transition timestamps (ms) for oscillation detection
        self._transition_times: deque[int] = deque()
        # Cumulative transition counter
        self._total_transitions: int = 0
        # Regime completed-span counts
        self._regime_counts: dict[str, int] = {r: 0 for r in _ALL_REGIMES}
        # Track which anomalies have already been alerted (avoid spam)
        self._alerted_duration_anomaly: bool = False
        self._alerted_oscillation: bool = False

        # Stabilisation window — 3 candles after exiting DECOUPLED
        self._in_stabilization: bool = False
        self._stabilization_exit_time_ms: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_regime_change(self, event: RegimeChangeEvent) -> None:
        """Process a :class:`~src.regimes.detector.RegimeChangeEvent`.

        Closes the previous span, opens a new one, records the transition,
        checks for oscillation anomalies, and fires DECOUPLED-specific alerts.

        * Entering DECOUPLED → CRITICAL alert + cancel any active stabilisation.
        * Exiting DECOUPLED → WARNING alert + start 3-candle stabilisation window.

        Args:
            event: Immutable regime change event from :class:`RegimeChangeDetector`.
        """
        ts = event.changed_at if event.changed_at >= 0 else int(time.time() * 1_000)

        # Close the previous span
        if self._current_span is not None:
            self._current_span.end_ms = ts
            self._spans.append(self._current_span)
            self._regime_counts[self._current_span.regime] = (
                self._regime_counts.get(self._current_span.regime, 0) + 1
            )

        # Open a new span
        self._current_span = RegimeSpan(regime=event.to_regime, start_ms=ts)
        self._total_transitions += 1
        self._transition_times.append(ts)
        # Reset duration anomaly flag on regime change
        self._alerted_duration_anomaly = False

        logger.info(
            "RegimeMonitor: transition {} -> {} at ts={}",
            event.from_regime,
            event.to_regime,
            ts,
        )

        # --- DECOUPLED-specific alert routing -----------------------------------
        if event.to_regime == _DECOUPLED:
            # Entering DECOUPLED: CRITICAL alert + cancel stabilisation
            self._alert.send_alert(
                level="CRITICAL",
                message="Market entered DECOUPLED regime — BUY confidence threshold raised to 0.72",
                details={
                    "from_regime": event.from_regime,
                    "to_regime": event.to_regime,
                    "timestamp_ms": ts,
                },
            )
            logger.warning(
                "RegimeMonitor: ENTERING DECOUPLED — elevated confidence threshold active"
            )
            self._in_stabilization = False
            self._stabilization_exit_time_ms = 0

        elif event.from_regime == _DECOUPLED:
            # Exiting DECOUPLED: WARNING alert + start stabilisation window
            self._alert.send_alert(
                level="WARNING",
                message=(
                    "Market exited DECOUPLED regime — stabilisation window active for "
                    f"{_STABILIZATION_CANDLES} candles"
                ),
                details={
                    "from_regime": event.from_regime,
                    "to_regime": event.to_regime,
                    "timestamp_ms": ts,
                    "stabilization_candles": _STABILIZATION_CANDLES,
                },
            )
            logger.warning(
                "RegimeMonitor: EXITING DECOUPLED — 3-candle stabilisation window started"
            )
            self._in_stabilization = True
            self._stabilization_exit_time_ms = ts

        self._check_oscillation(ts)

    def tick(self, timestamp_ms: int | None = None) -> RegimeMonitorStatus:
        """Perform periodic anomaly checks and return current status snapshot.

        Should be called once per closed candle (hourly).

        Args:
            timestamp_ms: Current UTC epoch ms (defaults to ``time.time()``
                if ``None``).

        Returns:
            :class:`RegimeMonitorStatus` snapshot of current state.
        """
        now_ms = timestamp_ms if timestamp_ms is not None else int(time.time() * 1_000)
        anomalies: list[str] = []

        if self._current_span is not None:
            duration_h = (now_ms - self._current_span.start_ms) / _MS_PER_HOUR
            regime = self._current_span.regime

            # Check DECOUPLED-specific threshold
            if regime == "DECOUPLED" and duration_h > self._mon.regime_max_decoupled_hours:
                msg = (
                    f"DECOUPLED regime has lasted {duration_h:.1f}h "
                    f"(threshold={self._mon.regime_max_decoupled_hours}h)"
                )
                anomalies.append(msg)
                if not self._alerted_duration_anomaly:
                    self._alert.send_alert(
                        level="CRITICAL",
                        message=f"Regime anomaly: {msg}",
                        details={
                            "regime": regime,
                            "duration_hours": round(duration_h, 1),
                            "threshold_hours": self._mon.regime_max_decoupled_hours,
                        },
                    )
                    self._alerted_duration_anomaly = True

            # Check generic max duration
            elif duration_h > self._mon.regime_max_any_hours:
                msg = (
                    f"{regime} regime has lasted {duration_h:.1f}h "
                    f"(threshold={self._mon.regime_max_any_hours}h)"
                )
                anomalies.append(msg)
                if not self._alerted_duration_anomaly:
                    self._alert.send_alert(
                        level="WARNING",
                        message=f"Regime anomaly: {msg}",
                        details={
                            "regime": regime,
                            "duration_hours": round(duration_h, 1),
                            "threshold_hours": self._mon.regime_max_any_hours,
                        },
                    )
                    self._alerted_duration_anomaly = True

        transition_rate = self._count_recent_transitions(now_ms)
        if transition_rate > self._mon.regime_oscillation_max_transitions:
            anomalies.append(
                f"Rapid oscillation: {transition_rate} transitions in last "
                f"{self._mon.regime_oscillation_window_hours}h"
            )

        return RegimeMonitorStatus(
            current_regime=(
                self._current_span.regime if self._current_span else None
            ),
            current_duration_hours=(
                (now_ms - self._current_span.start_ms) / _MS_PER_HOUR
                if self._current_span
                else 0.0
            ),
            total_transitions=self._total_transitions,
            transition_rate_24h=transition_rate,
            regime_counts=dict(self._regime_counts),
            anomalies_detected=anomalies,
        )

    def on_transition(
        self,
        from_regime: str,
        to_regime: str,
        timestamp_ms: int,
    ) -> None:
        """Convenience wrapper around :meth:`on_regime_change`.

        Constructs a :class:`~src.regimes.detector.RegimeChangeEvent` from
        plain string arguments and delegates to :meth:`on_regime_change`.
        Useful when callers don't have a ``RegimeChangeEvent`` object.

        Args:
            from_regime: Regime label active before this transition.
            to_regime: New regime label.
            timestamp_ms: UTC epoch milliseconds when the transition occurred.

        Raises:
            ValueError: If either regime label is not a recognised canonical label.
        """
        if from_regime not in _ALL_REGIMES:
            raise ValueError(
                f"RegimeMonitor.on_transition: invalid from_regime={from_regime!r}. "
                f"Must be one of: {sorted(_ALL_REGIMES)}"
            )
        if to_regime not in _ALL_REGIMES:
            raise ValueError(
                f"RegimeMonitor.on_transition: invalid to_regime={to_regime!r}. "
                f"Must be one of: {sorted(_ALL_REGIMES)}"
            )

        is_critical = (from_regime == _DECOUPLED) or (to_regime == _DECOUPLED)
        event = RegimeChangeEvent(
            from_regime=from_regime,
            to_regime=to_regime,
            changed_at=timestamp_ms,
            btc_corr=0.0,
            atr_norm=0.0,
            is_critical=is_critical,
        )
        self.on_regime_change(event)

    def is_in_stabilization_window(self, _now_ms: int | None = None) -> bool:
        """Return ``True`` for 3 candle-intervals after a DECOUPLED exit.

        The inference engine calls this before every signal decision.  During
        the stabilisation window the elevated ``DECOUPLED`` confidence
        threshold (0.72) should remain active even though the regime label
        has changed.

        Args:
            _now_ms: Override for the current UTC epoch ms.  **For unit tests
                only** — production code omits this argument so that
                ``time.time()`` is used.

        Returns:
            ``True`` while fewer than ``3 × interval_ms`` milliseconds have
            elapsed since the most recent DECOUPLED exit.
        """
        if not self._in_stabilization:
            return False

        now_ms = _now_ms if _now_ms is not None else int(time.time() * 1_000)
        elapsed_ms = now_ms - self._stabilization_exit_time_ms
        window_ms = _STABILIZATION_CANDLES * self._interval_ms

        if elapsed_ms >= window_ms:
            self._in_stabilization = False
            logger.info(
                "RegimeMonitor: stabilisation window ended "
                "({:.1f}h elapsed >= {:.1f}h threshold)",
                elapsed_ms / _MS_PER_HOUR,
                window_ms / _MS_PER_HOUR,
            )
            return False

        return True

    def get_regime_duration_stats(self) -> dict[str, dict[str, float]]:
        """Return average duration statistics per regime from span history.

        Only completed spans (that have already ended) are included.  The
        currently active span is excluded until it closes.

        Returns:
            Dict mapping each of the 5 regime labels to a sub-dict with keys:

            ``mean_hours``
                Average duration (hours) across all completed spans.
            ``count``
                Number of completed spans observed.
            ``total_hours``
                Sum of all completed span durations in hours.

            Regimes with no completed spans return ``0.0`` for all keys.
        """
        durations: dict[str, list[float]] = {r: [] for r in _ALL_REGIMES}

        for span in self._spans:
            if span.end_ms > 0:
                hours = (span.end_ms - span.start_ms) / _MS_PER_HOUR
                durations[span.regime].append(hours)

        stats: dict[str, dict[str, float]] = {}
        for regime, hours_list in durations.items():
            if hours_list:
                total = sum(hours_list)
                stats[regime] = {
                    "mean_hours": total / len(hours_list),
                    "count": float(len(hours_list)),
                    "total_hours": total,
                }
            else:
                stats[regime] = {
                    "mean_hours": 0.0,
                    "count": 0.0,
                    "total_hours": 0.0,
                }

        return stats

    def transition_history(self) -> list[dict[str, Any]]:
        """Return a list of all completed regime spans (alias for :meth:`get_transition_log`).

        Returns:
            List of dicts with keys ``regime``, ``start_ms``, ``end_ms``,
            ``duration_hours``.
        """
        return self.get_transition_log()

    def get_regime_duration_hours(self, regime: str) -> float:
        """Return the total hours spent in a regime across all completed spans.

        Args:
            regime: Regime label to query.

        Returns:
            Total hours across all completed spans for the given regime.
        """
        total_ms = sum(
            (s.end_ms - s.start_ms)
            for s in self._spans
            if s.regime == regime and s.end_ms > 0
        )
        return total_ms / _MS_PER_HOUR

    def get_transition_log(self) -> list[dict[str, Any]]:
        """Return a list of completed regime spans as dicts.

        Returns:
            List of dicts with keys ``regime``, ``start_ms``, ``end_ms``,
            ``duration_hours``.
        """
        return [
            {
                "regime": s.regime,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "duration_hours": s.duration_hours,
            }
            for s in self._spans
        ]

    def reset(self) -> None:
        """Clear all internal state (useful for testing or re-initialisation).

        Resets spans, transition log, counters, anomaly flags, and the
        stabilisation window.
        """
        self._spans.clear()
        self._current_span = None
        self._transition_times.clear()
        self._total_transitions = 0
        self._regime_counts = {r: 0 for r in _ALL_REGIMES}
        self._alerted_duration_anomaly = False
        self._alerted_oscillation = False
        self._in_stabilization = False
        self._stabilization_exit_time_ms = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_recent_transitions(self, now_ms: int) -> int:
        """Count transitions within the oscillation window.

        Args:
            now_ms: Current UTC epoch milliseconds.

        Returns:
            Number of transitions in the last ``regime_oscillation_window_hours``.
        """
        cutoff = now_ms - (self._mon.regime_oscillation_window_hours * _MS_PER_HOUR)
        # Remove expired entries
        while self._transition_times and self._transition_times[0] < cutoff:
            self._transition_times.popleft()
        return len(self._transition_times)

    def _check_oscillation(self, now_ms: int) -> None:
        """Check for rapid oscillation and alert if threshold exceeded.

        Args:
            now_ms: Current UTC epoch milliseconds (time of last transition).
        """
        rate = self._count_recent_transitions(now_ms)
        threshold = self._mon.regime_oscillation_max_transitions

        if rate > threshold and not self._alerted_oscillation:
            self._alert.send_alert(
                level="WARNING",
                message=(
                    f"Regime oscillation detected — {rate} transitions in last "
                    f"{self._mon.regime_oscillation_window_hours}h "
                    f"(threshold={threshold})"
                ),
                details={
                    "transitions_in_window": rate,
                    "window_hours": self._mon.regime_oscillation_window_hours,
                    "threshold": threshold,
                    "current_regime": (
                        self._current_span.regime if self._current_span else None
                    ),
                },
            )
            self._alerted_oscillation = True
            logger.warning(
                "Regime oscillation: {} transitions in {}h window",
                rate,
                self._mon.regime_oscillation_window_hours,
            )
        elif rate <= threshold:
            # Reset flag when oscillation subsides
            self._alerted_oscillation = False
