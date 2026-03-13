"""Alert manager for doge_predictor monitoring.

:class:`AlertManager` routes operational alerts at three severity levels to
persistent log files.  Future integration points (Telegram, email, PagerDuty)
are clearly marked but left as stubs.

Alert levels:
    - ``INFO``     — informational event, written to ``alerts.log``
    - ``WARNING``  — degradation detected, written to ``alerts.log``
    - ``CRITICAL`` — action required, written to **both** ``alerts.log`` and
                     ``critical_alerts.log``

Usage::

    from src.monitoring.alerting import AlertManager

    mgr = AlertManager(log_dir=Path("logs"))
    mgr.send_alert("WARNING", "Stale candle detected", {"age_s": 7500})
    mgr.send_alert("CRITICAL", "WS disconnected", {"reconnect_attempts": 10})
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from loguru import logger

__all__ = ["AlertManager"]


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """Thread-safe operational alert dispatcher.

    All alerts are persisted as JSON-lines to ``alerts.log`` inside
    *log_dir*.  ``CRITICAL`` alerts are additionally written to
    ``critical_alerts.log``.

    Args:
        log_dir: Directory for alert log files.  Created if it does not exist.
            Defaults to ``logs/`` relative to the current working directory.
        alerts_log_name: Filename for the general (all-levels) log.
        critical_log_name: Filename for the CRITICAL-only log.
    """

    #: Valid alert severity levels.
    LEVEL_INFO: str = "INFO"
    LEVEL_WARNING: str = "WARNING"
    LEVEL_CRITICAL: str = "CRITICAL"

    _VALID_LEVELS: frozenset[str] = frozenset({
        LEVEL_INFO, LEVEL_WARNING, LEVEL_CRITICAL
    })

    def __init__(
        self,
        log_dir: Path | None = None,
        alerts_log_name: str = "alerts.log",
        critical_log_name: str = "critical_alerts.log",
    ) -> None:
        self._log_dir: Path = log_dir if log_dir is not None else Path("logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._alerts_log: Path = self._log_dir / alerts_log_name
        self._critical_log: Path = self._log_dir / critical_log_name

        # Reentrant lock so the same thread can call send_alert from a handler
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "AlertManager initialised (alerts={}, critical={})",
            self._alerts_log,
            self._critical_log,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_alert(
        self,
        level: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Dispatch an alert.

        Args:
            level: Severity — ``"INFO"``, ``"WARNING"``, or ``"CRITICAL"``.
                Case-insensitive.
            message: Human-readable summary (one line preferred).
            details: Optional structured context dict (serialized to JSON).

        Raises:
            ValueError: If *level* is not one of the recognised levels.
        """
        level_upper = level.upper()
        if level_upper not in self._VALID_LEVELS:
            raise ValueError(
                f"Invalid alert level {level!r}. "
                f"Valid levels: {sorted(self._VALID_LEVELS)}"
            )

        record: dict[str, Any] = {
            "timestamp_ms": int(time.time() * 1_000),
            "level": level_upper,
            "message": message,
            "details": details or {},
        }

        # --- Route to loguru at the appropriate severity --------------------
        if level_upper == self.LEVEL_INFO:
            logger.info("[ALERT] {}", message)
        elif level_upper == self.LEVEL_WARNING:
            logger.warning("[ALERT] {}", message)
        else:
            logger.error("[ALERT CRITICAL] {}", message)

        # --- Persist to file(s) ---------------------------------------------
        with self._lock:
            self._write_to_file(self._alerts_log, record)
            if level_upper == self.LEVEL_CRITICAL:
                self._write_to_file(self._critical_log, record)

        # --- Future integration stubs (replace bodies when ready) -----------
        if level_upper == self.LEVEL_CRITICAL:
            self._notify_telegram(record)
            self._notify_email(record)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_to_file(self, log_path: Path, record: dict[str, Any]) -> None:
        """Append a JSON-lines record to *log_path*.

        Args:
            log_path: Target file path.  Opened in append mode.
            record: Dict to serialise as a single JSON line.
        """
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.error("AlertManager._write_to_file failed for {}: {}", log_path, exc)

    def _notify_telegram(self, record: dict[str, Any]) -> None:
        """Send a Telegram message for CRITICAL alerts.

        Stub — replace body with real Telegram Bot API call when ready.

        Args:
            record: The alert record dict.
        """
        logger.debug(
            "AlertManager._notify_telegram stub (CRITICAL): {}", record["message"]
        )

    def _notify_email(self, record: dict[str, Any]) -> None:
        """Send an email for CRITICAL alerts.

        Stub — replace body with real SMTP / SES call when ready.

        Args:
            record: The alert record dict.
        """
        logger.debug(
            "AlertManager._notify_email stub (CRITICAL): {}", record["message"]
        )
