"""Pure utility helper functions for doge_predictor.

Stateless, side-effect-free functions shared across the pipeline.
No imports from other ``src/`` modules — safe to import from anywhere.

Functions:
    ms_to_datetime: UTC epoch ms → UTC-aware datetime
    datetime_to_ms: UTC-aware datetime → UTC epoch ms
    interval_to_ms: Binance interval string → milliseconds
    compute_expected_row_count: expected candle count between two timestamps
    safe_divide: division with fallback on zero/NaN denominator
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

__all__ = [
    "ms_to_datetime",
    "datetime_to_ms",
    "interval_to_ms",
    "compute_expected_row_count",
    "safe_divide",
]

# ---------------------------------------------------------------------------
# Constants — all supported Binance kline intervals
# ---------------------------------------------------------------------------

_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def ms_to_datetime(ms: int) -> datetime:
    """Convert UTC epoch milliseconds to a UTC-aware :class:`datetime`.

    Args:
        ms: UTC epoch milliseconds. Must be a non-negative ``int``.

    Returns:
        A timezone-aware :class:`datetime` in UTC.

    Raises:
        TypeError: If *ms* is not an ``int``.
        ValueError: If *ms* is negative.

    Example::

        from src.utils.helpers import ms_to_datetime
        dt = ms_to_datetime(1_641_024_000_000)  # 2022-01-01 00:00:00 UTC
    """
    if not isinstance(ms, int):
        raise TypeError(f"ms must be int, got {type(ms).__name__}")
    if ms < 0:
        raise ValueError(f"ms must be >= 0, got {ms}")
    return datetime.fromtimestamp(ms / 1_000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert a UTC-aware :class:`datetime` to UTC epoch milliseconds.

    Args:
        dt: A timezone-aware :class:`datetime`. Must carry tzinfo (UTC or any
            other timezone — the value is normalised to UTC internally).

    Returns:
        UTC epoch milliseconds as ``int``.

    Raises:
        TypeError: If *dt* is not a :class:`datetime`.
        ValueError: If *dt* is timezone-naive.

    Example::

        from datetime import datetime, timezone
        from src.utils.helpers import datetime_to_ms
        ms = datetime_to_ms(datetime(2022, 1, 1, tzinfo=timezone.utc))
        # ms == 1_641_024_000_000
    """
    if not isinstance(dt, datetime):
        raise TypeError(f"dt must be datetime, got {type(dt).__name__}")
    if dt.tzinfo is None:
        raise ValueError(
            "dt must be timezone-aware (UTC required); got a tz-naive datetime"
        )
    return int(dt.timestamp() * 1_000)


def interval_to_ms(interval: str) -> int:
    """Convert a Binance-style interval string to milliseconds.

    Args:
        interval: A Binance kline interval string, e.g. ``'1h'``, ``'4h'``,
            ``'1d'``, ``'1w'``.

    Returns:
        Duration in milliseconds (``int``).

    Raises:
        ValueError: If *interval* is not a recognised Binance interval string.

    Example::

        from src.utils.helpers import interval_to_ms
        assert interval_to_ms('1h') == 3_600_000
        assert interval_to_ms('4h') == 14_400_000
    """
    try:
        return _INTERVAL_MS[interval]
    except KeyError:
        valid = sorted(_INTERVAL_MS.keys())
        raise ValueError(
            f"Unknown interval '{interval}'. Valid values: {valid}"
        ) from None


def compute_expected_row_count(
    start_ms: int,
    end_ms: int,
    interval_ms: int,
) -> int:
    """Compute the expected number of candles between two timestamps.

    The count is inclusive of the open candle at *start_ms* and exclusive
    of *end_ms*, matching Binance kline semantics where the last candle's
    ``open_time < end_ms``.

    Args:
        start_ms: Inclusive start timestamp (UTC epoch ms).
        end_ms: Exclusive end timestamp (UTC epoch ms). Must be > *start_ms*.
        interval_ms: Candle duration in milliseconds. Must be > 0.

    Returns:
        Expected number of candles (``int``, always >= 1).

    Raises:
        ValueError: If *start_ms* >= *end_ms* or *interval_ms* <= 0.

    Example::

        from src.utils.helpers import compute_expected_row_count, interval_to_ms
        # 1 hour of 1h candles = 1 candle
        assert compute_expected_row_count(0, 3_600_000, 3_600_000) == 1
        # 24 hours of 1h candles = 24 candles
        assert compute_expected_row_count(0, 86_400_000, 3_600_000) == 24
    """
    if start_ms >= end_ms:
        raise ValueError(
            f"start_ms ({start_ms}) must be strictly less than end_ms ({end_ms})"
        )
    if interval_ms <= 0:
        raise ValueError(f"interval_ms must be > 0, got {interval_ms}")
    return math.ceil((end_ms - start_ms) / interval_ms)


def safe_divide(
    numerator: float,
    denominator: float,
    fallback: float = 0.0,
) -> float:
    """Divide *numerator* by *denominator*, returning *fallback* on zero/NaN.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        fallback: Value returned when *denominator* is zero or NaN.
            Defaults to ``0.0``.

    Returns:
        ``numerator / denominator`` if *denominator* is finite and non-zero,
        otherwise *fallback*.

    Example::

        from src.utils.helpers import safe_divide
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, float('nan'), fallback=-1.0) == -1.0
    """
    if denominator == 0 or math.isnan(denominator):
        return fallback
    return numerator / denominator
