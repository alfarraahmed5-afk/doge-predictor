"""Health check HTTP server for doge_predictor.

Serves ``GET /health`` on a configurable TCP port (default 8000).

Healthy response (HTTP 200)::

    {
        "status": "ok",
        "last_candle_age_seconds": 3542.1,
        "ws_connected": true,
        "db_connected": true,
        "model_version": "v1.0",
        "current_regime": "TRENDING_BULL",
        "last_signal_at": 1741776000000
    }

Degraded response (HTTP 503) when any of these conditions are true:

    * ``last_candle_age_seconds > 2 × interval_seconds`` (stale data)
    * ``ws_connected == false``
    * ``db_connected == false``

The :class:`HealthStatus` dataclass is updated by ``scripts/serve.py``
after every successful inference cycle.

Usage::

    from src.monitoring.health_check import HealthCheckServer, HealthStatus

    status = HealthStatus()
    server = HealthCheckServer(status, port=8000, interval="1h")
    server.start()          # non-blocking — runs in a daemon thread
    # ... main inference loop ...
    server.stop()
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

from loguru import logger

from src.utils.helpers import interval_to_ms

__all__ = ["HealthStatus", "HealthCheckServer"]


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@dataclass
class HealthStatus:
    """Mutable health state shared between the inference loop and HTTP handler.

    All attribute writes happen on the inference thread; reads happen on the
    HTTP server thread.  Python's GIL makes individual attribute reads/writes
    atomic for built-in types (int, bool, str, None), so no explicit lock is
    needed here.

    Attributes:
        last_candle_close_time_ms: UTC epoch ms of the most recently processed
            closed candle.  0 means no candle has been processed yet.
        ws_connected: True when the WebSocket client reports a live connection.
        db_connected: True when the last DB health probe succeeded.
        model_version: Identifier of the loaded model artefacts.
        current_regime: Regime label from the last inference run.
        last_signal_at_ms: UTC epoch ms when the last signal was emitted.
            *None* until the first signal.
        interval: Kline interval string used to compute the freshness limit.
    """

    last_candle_close_time_ms: int = 0
    ws_connected: bool = False
    db_connected: bool = False
    model_version: str = "unknown"
    current_regime: str = "RANGING_LOW_VOL"
    last_signal_at_ms: Optional[int] = None
    interval: str = "1h"

    def update_from_signal(self, signal_event: Any) -> None:
        """Update fields from a :class:`~src.inference.signal.SignalEvent`.

        Args:
            signal_event: The SignalEvent returned by InferenceEngine.run().
        """
        self.last_candle_close_time_ms = int(getattr(signal_event, "timestamp_ms", 0))
        self.current_regime = str(getattr(signal_event, "regime", "UNKNOWN"))
        self.model_version = str(getattr(signal_event, "model_version", "unknown"))
        self.last_signal_at_ms = int(time.time() * 1_000)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


class _HealthCheckHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves ``GET /health``."""

    # Suppress default HTTPServer request logging; loguru handles it.
    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        logger.debug("[health] {} — {}", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        if self.path.rstrip("/") == "/health":
            self._handle_health()
        else:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "not found"}')

    def _handle_health(self) -> None:
        """Compute health payload and write HTTP response."""
        status: HealthStatus = self.server._health_status  # type: ignore[attr-defined]
        now_ms: int = int(time.time() * 1_000)
        interval_ms: int = interval_to_ms(status.interval)

        # ── Last candle age --------------------------------------------------
        if status.last_candle_close_time_ms > 0:
            last_candle_age_s: Optional[float] = round(
                (now_ms - status.last_candle_close_time_ms) / 1_000, 1
            )
        else:
            last_candle_age_s = None  # no candle received yet

        # ── Degraded condition checks ----------------------------------------
        degraded: bool = False
        degraded_reasons: list[str] = []

        freshness_limit_s: float = 2 * interval_ms / 1_000
        if (
            last_candle_age_s is not None
            and last_candle_age_s > freshness_limit_s
        ):
            degraded = True
            degraded_reasons.append(
                f"stale_data: last_candle_age={last_candle_age_s}s "
                f"> limit={freshness_limit_s}s"
            )

        if not status.ws_connected:
            degraded = True
            degraded_reasons.append("websocket_disconnected")

        if not status.db_connected:
            degraded = True
            degraded_reasons.append("db_disconnected")

        # ── Build response body ----------------------------------------------
        body: dict[str, Any] = {
            "status": "degraded" if degraded else "ok",
            "last_candle_age_seconds": last_candle_age_s,
            "ws_connected": status.ws_connected,
            "db_connected": status.db_connected,
            "model_version": status.model_version,
            "current_regime": status.current_regime,
            "last_signal_at": status.last_signal_at_ms,
        }
        if degraded_reasons:
            body["degraded_reasons"] = degraded_reasons

        payload: bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        http_code: int = 503 if degraded else 200

        self.send_response(http_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


# ---------------------------------------------------------------------------
# HealthCheckServer
# ---------------------------------------------------------------------------


class HealthCheckServer:
    """Threaded HTTP server that exposes ``GET /health``.

    Runs in a background daemon thread so it does not block the main
    inference loop.

    Args:
        health_status: :class:`HealthStatus` instance updated by the inference
            loop.  The same object is read by each HTTP request handler.
        port: TCP port to bind (default 8000).
        interval: Kline interval string used to compute the freshness window
            (default ``"1h"``).

    Example::

        status = HealthStatus()
        srv = HealthCheckServer(status, port=8000)
        srv.start()
        time.sleep(60)
        srv.stop()
    """

    def __init__(
        self,
        health_status: HealthStatus,
        port: int = 8000,
        interval: str = "1h",
    ) -> None:
        self._status: HealthStatus = health_status
        self._port: int = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        health_status.interval = interval

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the TCP port and start serving in a daemon background thread."""
        self._server = HTTPServer(("0.0.0.0", self._port), _HealthCheckHandler)
        # Attach the shared state so the handler can read it without a closure
        self._server._health_status = self._status  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="health-check-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("HealthCheckServer started on port {}", self._port)

    def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            logger.info("HealthCheckServer stopped.")
