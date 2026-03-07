"""Custom exception hierarchy for the Binance ingestion layer.

All exceptions raised by ``BinanceRESTClient``, ``BinanceFuturesClient``,
and ``BinanceWebSocketClient`` inherit from one of the four base classes
defined here.  Callers should catch the most specific exception type they
can handle and let the rest propagate.

Exception hierarchy::

    BinanceAPIError           ← base for all Binance HTTP errors
    ├── BinanceRateLimitError ← 429 / sustained 418; carries retry_after
    └── BinanceAuthError      ← 401 / 403

    DataValidationError       ← response failed Pandera schema contract
"""

from __future__ import annotations

__all__ = [
    "BinanceAPIError",
    "BinanceRateLimitError",
    "BinanceAuthError",
    "DataValidationError",
]


class BinanceAPIError(Exception):
    """Base exception for all Binance REST API errors.

    Raised when the API returns a non-2xx status code that is not more
    specifically handled by a subclass, or when the HTTP transport layer
    itself fails (e.g. timeout, connection refused) after all retry
    attempts are exhausted.

    Args:
        message: Human-readable description of the error.
        status_code: HTTP status code returned by Binance, or ``None`` for
            transport-level failures.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={str(self)!r}, "
            f"status_code={self.status_code!r})"
        )


class BinanceRateLimitError(BinanceAPIError):
    """Raised when Binance returns HTTP 429 (or 418) and all retries are spent.

    The ``retry_after`` attribute reflects the value from the
    ``Retry-After`` response header (in seconds).  When the header is
    absent, it defaults to ``60``.

    Args:
        message: Human-readable description of the error.
        retry_after: Seconds from the ``Retry-After`` header. Defaults to 60.
    """

    def __init__(
        self,
        message: str,
        retry_after: int = 60,
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after: int = retry_after

    def __repr__(self) -> str:
        return (
            f"BinanceRateLimitError("
            f"message={str(self)!r}, "
            f"retry_after={self.retry_after!r})"
        )


class BinanceAuthError(BinanceAPIError):
    """Raised for HTTP 401 (Unauthorized) or 403 (Forbidden) responses.

    These are non-retryable client errors that indicate the API key or
    signature is missing, invalid, or lacks the required permissions.

    Args:
        message: Human-readable description of the error.
        status_code: HTTP status code (401 or 403).
    """

    def __init__(
        self,
        message: str,
        status_code: int = 401,
    ) -> None:
        super().__init__(message, status_code=status_code)


class DataValidationError(Exception):
    """Raised when an API response fails Pandera schema validation.

    This indicates that the data returned by Binance does not match the
    expected structure (e.g. missing columns, wrong types, OHLC invariant
    violations, NaN/Inf values).  The original Pandera exception is
    attached as ``__cause__`` when raised via ``raise ... from exc``.

    Args:
        message: Human-readable description of which schema check failed.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
