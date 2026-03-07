"""Unit tests for src/utils/logger.py.

Tests that the module imports cleanly, that configure_logging() runs without
errors, and that the get_rl_logger() accessor returns the loguru logger.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from loguru import logger as loguru_logger


class TestLoggerImport:
    """The module must import cleanly and expose its public API."""

    def test_module_imports(self) -> None:
        """Importing the module raises no errors."""
        from src.utils.logger import configure_logging, get_rl_logger  # noqa: F401

    def test_public_names_exported(self) -> None:
        """__all__ contains the expected public names."""
        import src.utils.logger as logger_module
        assert "configure_logging" in logger_module.__all__
        assert "get_rl_logger" in logger_module.__all__

    def test_constants_present(self) -> None:
        """Module-level constants are defined."""
        import src.utils.logger as logger_module
        assert isinstance(logger_module._LOG_DIR, Path)
        assert isinstance(logger_module._APP_LOG, Path)
        assert isinstance(logger_module._RL_LOG, Path)
        assert isinstance(logger_module._ROTATION, str)
        assert isinstance(logger_module._RETENTION, int)


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_runs_without_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure_logging() completes without raising."""
        import src.utils.logger as logger_module
        monkeypatch.setattr(logger_module, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(logger_module, "_APP_LOG", tmp_path / "app.log")
        monkeypatch.setattr(logger_module, "_RL_LOG", tmp_path / "rl.log")
        from src.utils.logger import configure_logging
        configure_logging(log_level="WARNING")  # WARNING to reduce noise in tests

    def test_creates_log_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """configure_logging() creates the log directory if absent."""
        log_dir = tmp_path / "new_logs"
        import src.utils.logger as logger_module
        monkeypatch.setattr(logger_module, "_LOG_DIR", log_dir)
        monkeypatch.setattr(logger_module, "_APP_LOG", log_dir / "app.log")
        monkeypatch.setattr(logger_module, "_RL_LOG", log_dir / "rl.log")
        assert not log_dir.exists()
        from src.utils.logger import configure_logging
        configure_logging(log_level="WARNING")
        assert log_dir.exists()

    def test_intercept_handler_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After configure_logging(), stdlib root logger has _InterceptHandler."""
        import src.utils.logger as logger_module
        monkeypatch.setattr(logger_module, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(logger_module, "_APP_LOG", tmp_path / "app.log")
        monkeypatch.setattr(logger_module, "_RL_LOG", tmp_path / "rl.log")
        from src.utils.logger import _InterceptHandler, configure_logging
        configure_logging(log_level="WARNING")
        root_handlers = logging.root.handlers
        assert any(isinstance(h, _InterceptHandler) for h in root_handlers)


class TestGetRlLogger:
    """Tests for get_rl_logger()."""

    def test_returns_loguru_logger(self) -> None:
        """get_rl_logger() returns the loguru logger singleton."""
        from src.utils.logger import get_rl_logger
        rl_log = get_rl_logger()
        assert rl_log is loguru_logger

    def test_is_callable(self) -> None:
        """The returned logger can be called to emit a message."""
        from src.utils.logger import get_rl_logger
        rl_log = get_rl_logger()
        # Should not raise
        rl_log.debug("test rl logger message from unit test")


class TestInterceptHandler:
    """Tests for the _InterceptHandler internal class."""

    def test_is_logging_handler(self) -> None:
        """_InterceptHandler is a subclass of logging.Handler."""
        from src.utils.logger import _InterceptHandler
        assert issubclass(_InterceptHandler, logging.Handler)

    def test_emit_does_not_raise(self) -> None:
        """emit() handles a normal LogRecord without raising."""
        from src.utils.logger import _InterceptHandler
        handler = _InterceptHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        # Should not raise — routes through loguru
        handler.emit(record)
