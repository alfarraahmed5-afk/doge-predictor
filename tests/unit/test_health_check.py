"""Unit tests for src/monitoring/health_check.py — HealthStatus + HealthCheckServer.

Tests cover:
  - HealthStatus default values
  - HealthStatus.update_from_signal()
  - HealthCheckServer GET /health → 200 when healthy
  - HealthCheckServer GET /health → 503 on each degraded condition
  - Response body fields
  - degraded_reasons list on 503
  - Unknown path → 404
  - last_candle_age_seconds is None before first candle
  - start() / stop() lifecycle
"""

from __future__ import annotations

import json
import socket
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.monitoring.health_check import HealthCheckServer, HealthStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str = "/health") -> tuple[int, dict[str, Any]]:
    """Make a GET request and return (status_code, parsed_body)."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def healthy_status() -> HealthStatus:
    """HealthStatus that passes all health checks."""
    now_ms = int(time.time() * 1_000)
    status = HealthStatus(
        last_candle_close_time_ms=now_ms - 100,  # 0.1 seconds ago
        ws_connected=True,
        db_connected=True,
        model_version="v1.0",
        current_regime="TRENDING_BULL",
        interval="1h",
    )
    return status


@pytest.fixture()
def server_and_port(healthy_status: HealthStatus):
    """Start a HealthCheckServer and yield (server, port). Stop after test."""
    port = _free_port()
    srv = HealthCheckServer(healthy_status, port=port, interval="1h")
    srv.start()
    time.sleep(0.05)  # brief pause to let daemon thread bind
    yield srv, port
    srv.stop()


# ---------------------------------------------------------------------------
# TestHealthStatusDefaults
# ---------------------------------------------------------------------------


class TestHealthStatusDefaults:
    """HealthStatus default values."""

    def test_last_candle_default_zero(self) -> None:
        """Default last_candle_close_time_ms is 0."""
        assert HealthStatus().last_candle_close_time_ms == 0

    def test_ws_connected_default_false(self) -> None:
        """Default ws_connected is False (safest default)."""
        assert HealthStatus().ws_connected is False

    def test_db_connected_default_false(self) -> None:
        """Default db_connected is False (safest default)."""
        assert HealthStatus().db_connected is False

    def test_model_version_default_unknown(self) -> None:
        """Default model_version is 'unknown'."""
        assert HealthStatus().model_version == "unknown"

    def test_current_regime_default(self) -> None:
        """Default current_regime is 'RANGING_LOW_VOL'."""
        assert HealthStatus().current_regime == "RANGING_LOW_VOL"

    def test_last_signal_at_ms_default_none(self) -> None:
        """Default last_signal_at_ms is None."""
        assert HealthStatus().last_signal_at_ms is None

    def test_interval_default_1h(self) -> None:
        """Default interval is '1h'."""
        assert HealthStatus().interval == "1h"


# ---------------------------------------------------------------------------
# TestHealthStatusUpdateFromSignal
# ---------------------------------------------------------------------------


class TestHealthStatusUpdateFromSignal:
    """HealthStatus.update_from_signal() behaviour."""

    def _make_signal(self, **kwargs: Any) -> SimpleNamespace:
        defaults = {
            "timestamp_ms": 1_700_000_000_000,
            "regime": "TRENDING_BEAR",
            "model_version": "run-abc123",
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_timestamp_updated(self) -> None:
        """last_candle_close_time_ms reflects signal's timestamp_ms."""
        status = HealthStatus()
        status.update_from_signal(self._make_signal(timestamp_ms=9_000_000_000_000))
        assert status.last_candle_close_time_ms == 9_000_000_000_000

    def test_regime_updated(self) -> None:
        """current_regime is taken from the signal."""
        status = HealthStatus()
        status.update_from_signal(self._make_signal(regime="DECOUPLED"))
        assert status.current_regime == "DECOUPLED"

    def test_model_version_updated(self) -> None:
        """model_version is taken from the signal."""
        status = HealthStatus()
        status.update_from_signal(self._make_signal(model_version="v2.1"))
        assert status.model_version == "v2.1"

    def test_last_signal_at_ms_is_set(self) -> None:
        """last_signal_at_ms is set to approximately now."""
        status = HealthStatus()
        before = int(time.time() * 1_000)
        status.update_from_signal(self._make_signal())
        after = int(time.time() * 1_000)
        assert before <= status.last_signal_at_ms <= after  # type: ignore[operator]

    def test_missing_attribute_does_not_raise(self) -> None:
        """Missing attributes on the signal object use safe defaults via getattr."""
        status = HealthStatus()
        # Minimal object with no attributes at all
        status.update_from_signal(SimpleNamespace())
        assert status.last_candle_close_time_ms == 0
        assert status.current_regime == "UNKNOWN"
        assert status.model_version == "unknown"


# ---------------------------------------------------------------------------
# TestHealthCheckServerHttp
# ---------------------------------------------------------------------------


class TestHealthCheckServerHttp:
    """HTTP endpoint behaviour tests."""

    def test_healthy_returns_200(
        self,
        server_and_port: tuple[HealthCheckServer, int],
    ) -> None:
        """All conditions healthy → HTTP 200."""
        _, port = server_and_port
        code, _ = _get(port)
        assert code == 200

    def test_healthy_status_ok(
        self,
        server_and_port: tuple[HealthCheckServer, int],
    ) -> None:
        """Body status field is 'ok' when healthy."""
        _, port = server_and_port
        _, body = _get(port)
        assert body["status"] == "ok"

    def test_ws_disconnected_returns_503(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """ws_connected=False → HTTP 503."""
        healthy_status.ws_connected = False
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        try:
            code, body = _get(port)
            assert code == 503
            assert body["status"] == "degraded"
            assert any("websocket" in r for r in body.get("degraded_reasons", []))
        finally:
            srv.stop()

    def test_db_disconnected_returns_503(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """db_connected=False → HTTP 503."""
        healthy_status.db_connected = False
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        try:
            code, body = _get(port)
            assert code == 503
            assert any("db" in r for r in body.get("degraded_reasons", []))
        finally:
            srv.stop()

    def test_stale_candle_returns_503(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """last_candle_close_time_ms too old → HTTP 503."""
        # Set candle time to 5 hours ago; freshness limit for 1h interval = 7200s
        five_hours_ago = int(time.time() * 1_000) - (5 * 3_600 * 1_000)
        healthy_status.last_candle_close_time_ms = five_hours_ago
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        try:
            code, body = _get(port)
            assert code == 503
            assert any("stale" in r for r in body.get("degraded_reasons", []))
        finally:
            srv.stop()

    def test_no_candle_yet_does_not_cause_503(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """last_candle_close_time_ms == 0 (no candle yet) does NOT trigger stale check."""
        healthy_status.last_candle_close_time_ms = 0  # never received a candle
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        try:
            code, body = _get(port)
            # ws_connected and db_connected are still True from healthy_status fixture
            assert code == 200
            assert body["last_candle_age_seconds"] is None
        finally:
            srv.stop()

    def test_response_has_required_keys(
        self,
        server_and_port: tuple[HealthCheckServer, int],
    ) -> None:
        """Response body contains all 7 expected keys."""
        _, port = server_and_port
        _, body = _get(port)
        required = {
            "status",
            "last_candle_age_seconds",
            "ws_connected",
            "db_connected",
            "model_version",
            "current_regime",
            "last_signal_at",
        }
        assert required.issubset(body.keys())

    def test_model_version_in_body(
        self,
        server_and_port: tuple[HealthCheckServer, int],
        healthy_status: HealthStatus,
    ) -> None:
        """Response body reflects model_version from HealthStatus."""
        healthy_status.model_version = "run-xyz-999"
        _, port = server_and_port
        _, body = _get(port)
        assert body["model_version"] == "run-xyz-999"

    def test_current_regime_in_body(
        self,
        server_and_port: tuple[HealthCheckServer, int],
        healthy_status: HealthStatus,
    ) -> None:
        """Response body reflects current_regime from HealthStatus."""
        healthy_status.current_regime = "DECOUPLED"
        _, port = server_and_port
        _, body = _get(port)
        assert body["current_regime"] == "DECOUPLED"

    def test_last_signal_at_none_when_no_signal(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """last_signal_at is null in JSON when no signal has been emitted."""
        healthy_status.last_signal_at_ms = None
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        try:
            _, body = _get(port)
            assert body["last_signal_at"] is None
        finally:
            srv.stop()

    def test_last_candle_age_positive_when_candle_present(
        self,
        server_and_port: tuple[HealthCheckServer, int],
        healthy_status: HealthStatus,
    ) -> None:
        """last_candle_age_seconds > 0 when a candle time is set."""
        # Already set in healthy_status fixture (100 ms ago)
        _, port = server_and_port
        _, body = _get(port)
        assert body["last_candle_age_seconds"] is not None
        assert body["last_candle_age_seconds"] >= 0.0

    def test_unknown_path_returns_404(
        self,
        server_and_port: tuple[HealthCheckServer, int],
    ) -> None:
        """GET /unknown → 404."""
        _, port = server_and_port
        code, body = _get(port, "/unknown")
        assert code == 404

    def test_stop_does_not_raise(
        self,
        healthy_status: HealthStatus,
    ) -> None:
        """stop() can be called cleanly after start()."""
        port = _free_port()
        srv = HealthCheckServer(healthy_status, port=port, interval="1h")
        srv.start()
        time.sleep(0.05)
        srv.stop()  # should not raise


# ---------------------------------------------------------------------------
# TestHealthCheckServerIntervalOverride
# ---------------------------------------------------------------------------


class TestHealthCheckServerIntervalOverride:
    """Interval is overridden on HealthStatus by HealthCheckServer.__init__."""

    def test_interval_set_on_status(self) -> None:
        """Constructor writes interval to HealthStatus.interval."""
        status = HealthStatus(interval="4h")
        port = _free_port()
        srv = HealthCheckServer(status, port=port, interval="1d")
        assert status.interval == "1d"
        srv  # not started; just check attribute
