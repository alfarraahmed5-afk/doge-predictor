"""Centralised loguru logger configuration for doge_predictor.

All modules obtain a logger via the standard loguru import::

    from loguru import logger

This module configures the global loguru sinks. Call :func:`configure_logging`
**once** from the application entry point (``scripts/serve.py``,
``scripts/train.py``, etc.) before any log messages are emitted.

Sinks configured by :func:`configure_logging`:
    - ``logs/app.log``  — structured JSON, level >= *log_level* (default INFO)
    - ``logs/rl.log``   — structured JSON, DEBUG+, only records from ``src.rl.*``
    - ``stderr``        — human-readable coloured output, level >= *log_level*

Log rotation:
    - Max file size: 100 MB (``_ROTATION``)
    - Retention: last 10 rotated files per sink (``_RETENTION``)

Stdlib interception:
    ``logging.basicConfig`` is replaced so that third-party libraries that
    emit standard-library ``logging`` records (SQLAlchemy, MLflow, urllib3,
    filelock) are automatically routed through loguru.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger

__all__ = ["configure_logging", "get_rl_logger"]

# ---------------------------------------------------------------------------
# Module constants (derived from config/settings.yaml conventions)
# ---------------------------------------------------------------------------

_LOG_DIR: Path = Path("logs")
_APP_LOG: Path = _LOG_DIR / "app.log"
_RL_LOG: Path = _LOG_DIR / "rl.log"
_ROTATION: str = "100 MB"
_RETENTION: int = 10


# ---------------------------------------------------------------------------
# Stdlib → loguru intercept handler
# ---------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    """Route standard-library :mod:`logging` records through loguru.

    Installed via :func:`configure_logging`. Converts every
    :class:`logging.LogRecord` into a loguru log call at the matching level,
    preserving the original call site so loguru's source-location fields
    (``{name}``, ``{line}``) remain accurate.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward *record* to loguru.

        Args:
            record: Standard library log record to forward.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk the call stack to find the originating frame outside logging.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(log_level: str = "INFO") -> None:
    """Configure loguru sinks for the application.

    Must be called once at startup. Subsequent calls are safe but will add
    duplicate sinks — avoid calling more than once in production.

    Args:
        log_level: Minimum log level for ``app.log`` and ``stderr``.
            Accepts any loguru level string: ``'DEBUG'``, ``'INFO'``,
            ``'WARNING'``, ``'ERROR'``, ``'CRITICAL'``.

    Side effects:
        - Creates ``logs/`` directory if it does not exist.
        - Removes loguru's default stderr sink.
        - Adds three new sinks (stderr, app.log, rl.log).
        - Replaces the root stdlib ``logging`` handler with
          :class:`_InterceptHandler`.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Remove loguru's default stderr sink before adding custom ones.
    logger.remove()

    # --- stderr — human-readable, coloured ---
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        enqueue=True,
    )

    # --- app.log — JSON structured, all pipeline events ---
    logger.add(
        str(_APP_LOG),
        level=log_level,
        format="{message}",
        serialize=True,
        rotation=_ROTATION,
        retention=_RETENTION,
        compression="gz",
        enqueue=True,
    )

    # --- rl.log — JSON structured, RL subsystem events only ---
    logger.add(
        str(_RL_LOG),
        level="DEBUG",
        format="{message}",
        serialize=True,
        rotation=_ROTATION,
        retention=_RETENTION,
        compression="gz",
        filter=lambda record: record["name"].startswith("src.rl"),
        enqueue=True,
    )

    # --- Intercept stdlib logging ---
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for _lib in ("sqlalchemy", "mlflow", "urllib3", "filelock", "httpx"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    logger.info(
        "Logging configured — level={}, app_log={}, rl_log={}",
        log_level,
        _APP_LOG,
        _RL_LOG,
    )


def get_rl_logger() -> "logger":  # type: ignore[valid-type]
    """Return the module-level loguru logger singleton.

    RL modules should bind ``name`` so the ``rl.log`` filter activates::

        from loguru import logger
        _log = logger.bind(name="src.rl.reward")

    Returns:
        The global loguru ``logger`` object.
    """
    return logger
