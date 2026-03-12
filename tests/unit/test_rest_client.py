"""Unit tests for src/ingestion/rest_client.py.

All HTTP calls are intercepted by the ``responses`` library — no real network
traffic is generated.  ``time.sleep`` is patched in tests that assert on rate-
limit or retry behaviour so the test suite runs fast.

Test coverage:
    - DataFrame returned with correct columns and row count
    - Pagination: two pages combined into a single DataFrame
    - All returned timestamps are int (never datetime)
    - Empty response → empty DataFrame, no crash
    - Invalid start/end raises ValueError
    - Malformed response triggers DataValidationError (schema validation)
    - 429 after max retries raises BinanceRateLimitError
    - 429 Retry-After header value is propagated
    - 503 followed by 200 → function succeeds (retry logic)
    - 400 → BinanceAPIError raised immediately (no retry)
    - 401 → BinanceAuthError raised immediately (no retry)
    - Weight threshold triggers time.sleep before the next request
    - get_exchange_info() is served from cache on second call (1 HTTP req)
    - get_order_book() returns a dict
    - get_recent_trades() returns a DataFrame with correct columns
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
import responses as resp_lib

from src.ingestion.exceptions import (
    BinanceAPIError,
    BinanceAuthError,
    BinanceRateLimitError,
    DataValidationError,
)
from src.ingestion.rest_client import BinanceRESTClient

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

#: Mock base URL reused in responses.add() calls.
_BASE_URL: str = "https://api.binance.com"

#: 2022-01-01 00:00:00 UTC in epoch milliseconds.
_START_MS: int = 1_640_995_200_000

#: 1-hour interval in milliseconds.
_INTERVAL_MS: int = 3_600_000

#: Klines endpoint full URL.
_KLINES_URL: str = f"{_BASE_URL}/api/v3/klines"

#: Exchange-info endpoint full URL.
_EXCHANGE_INFO_URL: str = f"{_BASE_URL}/api/v3/exchangeInfo"

#: Order-book endpoint full URL.
_DEPTH_URL: str = f"{_BASE_URL}/api/v3/depth"

#: Aggregate trades endpoint full URL.
_AGG_TRADES_URL: str = f"{_BASE_URL}/api/v3/aggTrades"

#: Default weight header value for normal responses.
_WEIGHT_HDR_NORMAL: dict[str, str] = {"X-MBX-USED-WEIGHT-1M": "10"}

#: Weight header that exceeds the 1 000 pre-emptive sleep threshold.
_WEIGHT_HDR_HIGH: dict[str, str] = {"X-MBX-USED-WEIGHT-1M": "1100"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kline_rows(
    count: int,
    start_ms: int = _START_MS,
    interval_ms: int = _INTERVAL_MS,
) -> list[list[Any]]:
    """Generate *count* synthetic Binance kline rows.

    Each row is a 12-element list matching the ``/api/v3/klines`` response
    format.  All OHLCV values pass ``OHLCVSchema`` invariants.

    Args:
        count: Number of rows to generate.
        start_ms: Open-time of the first row (UTC epoch ms).
        interval_ms: Duration of each candle in milliseconds.

    Returns:
        List of 12-element kline arrays.
    """
    rows: list[list[Any]] = []
    for i in range(count):
        open_time = start_ms + i * interval_ms
        close_time = open_time + interval_ms - 1
        rows.append([
            open_time,       # 0  open_time  (int)
            "0.10000000",    # 1  open       (str)
            "0.11000000",    # 2  high       (str) — >= open, close, low
            "0.09000000",    # 3  low        (str) — <= open, close, high
            "0.10500000",    # 4  close      (str)
            "1000000.00",    # 5  volume     (str)
            close_time,      # 6  close_time (int)
            "100000.00",     # 7  quote_volume
            100,             # 8  n_trades
            "500000.00",     # 9  taker_buy_base
            "50000.00",      # 10 taker_buy_quote
            "0",             # 11 ignore
        ])
    return rows


def _make_exchange_info() -> dict[str, Any]:
    """Return a minimal exchange-info payload."""
    return {
        "timezone": "UTC",
        "serverTime": _START_MS,
        "rateLimits": [],
        "symbols": [{"symbol": "DOGEUSDT", "status": "TRADING"}],
    }


def _make_agg_trades(count: int = 5) -> list[dict[str, Any]]:
    """Return *count* synthetic aggTrades rows."""
    rows: list[dict[str, Any]] = []
    for i in range(count):
        rows.append({
            "a": i,                         # trade_id
            "p": "0.10000000",              # price
            "q": "10000.00000000",          # qty
            "f": i * 10,                    # first trade id
            "l": i * 10 + 9,               # last trade id
            "T": _START_MS + i * 1_000,    # timestamp_ms
            "m": False,                     # is_buyer_maker
            "M": True,                      # best price match
        })
    return rows


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> BinanceRESTClient:
    """Return a fresh ``BinanceRESTClient`` with max_retries=3 for fast tests."""
    return BinanceRESTClient(
        api_key="test_key",
        api_secret="test_secret",
        max_retries=3,
    )


# ---------------------------------------------------------------------------
# Tests — get_klines: basic correctness
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_klines_returns_dataframe_with_correct_columns(
    client: BinanceRESTClient,
) -> None:
    """Single-page response → DataFrame with required columns, correct row count."""
    row_count = 100
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=_make_kline_rows(row_count),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + row_count * _INTERVAL_MS
    df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == row_count
    for col in ("open_time", "open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"


@resp_lib.activate
def test_get_klines_timestamps_are_int(
    client: BinanceRESTClient,
) -> None:
    """All open_time and close_time values in the result must be plain int."""
    rows = _make_kline_rows(10)
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=rows,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    for val in df["open_time"]:
        assert isinstance(val, int), f"open_time value {val!r} is not int"
    for val in df["close_time"]:
        assert isinstance(val, int), f"close_time value {val!r} is not int"


@resp_lib.activate
def test_get_klines_empty_response_returns_empty_dataframe(
    client: BinanceRESTClient,
) -> None:
    """Empty list from API → empty DataFrame (not an exception)."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=[],
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_get_klines_raises_on_invalid_time_range(
    client: BinanceRESTClient,
) -> None:
    """start_ms >= end_ms must raise ValueError before any HTTP call."""
    with pytest.raises(ValueError, match="start_ms"):
        client.get_klines("DOGEUSDT", "1h", _START_MS, _START_MS)

    with pytest.raises(ValueError, match="start_ms"):
        client.get_klines("DOGEUSDT", "1h", _START_MS + 1, _START_MS)


def test_get_klines_raises_on_unknown_interval(
    client: BinanceRESTClient,
) -> None:
    """Unrecognised interval string must raise ValueError from interval_to_ms."""
    with pytest.raises(ValueError, match="Unknown interval"):
        client.get_klines("DOGEUSDT", "99x", _START_MS, _START_MS + _INTERVAL_MS)


# ---------------------------------------------------------------------------
# Tests — get_klines: pagination
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_klines_pagination_two_pages(
    client: BinanceRESTClient,
) -> None:
    """First page = 1 000 rows, second page = 50 rows → 1 050 total, 2 HTTP calls."""
    page1_count = 1_000
    page2_count = 50
    page2_start = _START_MS + page1_count * _INTERVAL_MS

    page1 = _make_kline_rows(page1_count, _START_MS, _INTERVAL_MS)
    page2 = _make_kline_rows(page2_count, page2_start, _INTERVAL_MS)

    # Register two sequential responses for the same URL
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=page1,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=page2,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    # end_ms is set to exactly cover both pages.  With timestamp-based stopping
    # (no short-page heuristic), the loop advances current_start to end_ms after
    # page 2 and exits via the `current_start >= end_ms` guard — exactly 2 calls.
    end_ms = _START_MS + (page1_count + page2_count) * _INTERVAL_MS
    df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert len(df) == page1_count + page2_count
    assert len(resp_lib.calls) == 2, (
        f"Expected exactly 2 HTTP requests, got {len(resp_lib.calls)}"
    )

    # Second request's startTime must come after the first page's close_time
    import urllib.parse

    second_params = urllib.parse.parse_qs(
        urllib.parse.urlparse(resp_lib.calls[1].request.url).query
    )
    second_start = int(second_params["startTime"][0])
    first_page_last_close = int(page1[-1][6])
    assert second_start == first_page_last_close + 1


@resp_lib.activate
def test_get_klines_deduplicates_boundary_rows(
    client: BinanceRESTClient,
) -> None:
    """Duplicate open_time rows within a single response must be removed."""
    rows = _make_kline_rows(10, _START_MS, _INTERVAL_MS)
    extra_rows = _make_kline_rows(5, _START_MS + 10 * _INTERVAL_MS, _INTERVAL_MS)
    # Inject a duplicate (same open_time as row[9])
    combined = rows + [rows[-1]] + extra_rows

    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=combined,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    # 10 rows + 5 extra rows = 15 distinct open_times (0..14).
    # end_ms covers exactly row 14's close_time so the guard fires immediately.
    end_ms = _START_MS + 15 * _INTERVAL_MS
    df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    # 10 unique rows + 5 new rows = 15 (duplicate removed)
    assert df["open_time"].is_unique
    assert len(df) == 15


# ---------------------------------------------------------------------------
# Tests — get_klines: schema validation
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_klines_schema_validation_error_on_ohlc_invariant_violation(
    client: BinanceRESTClient,
) -> None:
    """A row where high < low violates OHLCVSchema → DataValidationError raised."""
    rows = _make_kline_rows(5)
    # Row index 2: set high < low to break the OHLC invariant
    rows[2][2] = "0.05000000"  # high = 0.05
    rows[2][3] = "0.09000000"  # low  = 0.09  → high(0.05) < low(0.09)

    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=rows,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 5 * _INTERVAL_MS
    with pytest.raises(DataValidationError):
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)


@resp_lib.activate
def test_get_klines_non_list_response_raises_validation_error(
    client: BinanceRESTClient,
) -> None:
    """If the API returns a JSON object instead of a list, DataValidationError is raised."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json={"code": -1000, "msg": "An unknown error occurred."},
        status=200,  # Binance occasionally wraps errors as 200
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with pytest.raises(DataValidationError):
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)


# ---------------------------------------------------------------------------
# Tests — retry logic
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_retry_on_503_succeeds(
    client: BinanceRESTClient,
) -> None:
    """503 on first attempt followed by 200 → function returns valid DataFrame."""
    rows = _make_kline_rows(5)

    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        status=503,
        body=b"Service Unavailable",
        headers=_WEIGHT_HDR_NORMAL,
    )
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=rows,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 5 * _INTERVAL_MS
    with patch("src.ingestion.rest_client.time.sleep"):
        df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert len(df) == 5
    assert len(resp_lib.calls) == 2


@resp_lib.activate
def test_retry_on_502_exhausted_raises_api_error(
    client: BinanceRESTClient,
) -> None:
    """Persistent 502 after max_retries (=3) raises BinanceAPIError."""
    for _ in range(3):  # client.max_retries == 3 from fixture
        resp_lib.add(
            resp_lib.GET,
            _KLINES_URL,
            status=502,
            body=b"Bad Gateway",
            headers=_WEIGHT_HDR_NORMAL,
        )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with patch("src.ingestion.rest_client.time.sleep"):
        with pytest.raises(BinanceAPIError) as exc_info:
            client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert exc_info.value.status_code == 502
    assert not isinstance(exc_info.value, BinanceRateLimitError)
    assert len(resp_lib.calls) == 3  # exactly max_retries attempts


@resp_lib.activate
def test_no_retry_on_400_bad_request(
    client: BinanceRESTClient,
) -> None:
    """400 response is a client error — must raise immediately without retry."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json={"code": -1121, "msg": "Invalid symbol."},
        status=400,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with pytest.raises(BinanceAPIError) as exc_info:
        client.get_klines("INVALIDSYM", "1h", _START_MS, end_ms)

    assert exc_info.value.status_code == 400
    assert len(resp_lib.calls) == 1  # no retry
    assert not isinstance(exc_info.value, (BinanceRateLimitError, BinanceAuthError))


@resp_lib.activate
def test_auth_error_401_raised_immediately(
    client: BinanceRESTClient,
) -> None:
    """401 must raise BinanceAuthError immediately without retry."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json={"code": -2014, "msg": "API-key format invalid."},
        status=401,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with pytest.raises(BinanceAuthError) as exc_info:
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert exc_info.value.status_code == 401
    assert len(resp_lib.calls) == 1  # no retry


@resp_lib.activate
def test_auth_error_403_raised_immediately(
    client: BinanceRESTClient,
) -> None:
    """403 must raise BinanceAuthError immediately without retry."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json={"code": -1003, "msg": "WAF Limit violation."},
        status=403,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with pytest.raises(BinanceAuthError) as exc_info:
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert exc_info.value.status_code == 403
    assert len(resp_lib.calls) == 1


# ---------------------------------------------------------------------------
# Tests — rate limiting (429)
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_rate_limit_error_raised_after_max_retries(
    client: BinanceRESTClient,
) -> None:
    """Persistent 429 → BinanceRateLimitError raised with correct retry_after."""
    retry_after_seconds = 5

    for _ in range(3):  # max_retries == 3
        resp_lib.add(
            resp_lib.GET,
            _KLINES_URL,
            body=b"",
            status=429,
            headers={
                "Retry-After": str(retry_after_seconds),
                "X-MBX-USED-WEIGHT-1M": "1200",
            },
        )

    end_ms = _START_MS + 10 * _INTERVAL_MS
    with patch("src.ingestion.rest_client.time.sleep"):
        with pytest.raises(BinanceRateLimitError) as exc_info:
            client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    assert isinstance(exc_info.value, BinanceRateLimitError)
    assert exc_info.value.retry_after == retry_after_seconds
    assert exc_info.value.status_code == 429
    assert len(resp_lib.calls) == 3


@resp_lib.activate
def test_rate_limit_retry_after_header_used_for_sleep(
    client: BinanceRESTClient,
) -> None:
    """On 429, the Retry-After value (not exponential backoff) drives the sleep."""
    retry_after_val = 30

    # First call → 429, second call → 200 (success after retry)
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        body=b"",
        status=429,
        headers={"Retry-After": str(retry_after_val), "X-MBX-USED-WEIGHT-1M": "1200"},
    )
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=_make_kline_rows(5),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms = _START_MS + 5 * _INTERVAL_MS

    with patch("src.ingestion.rest_client.time.sleep") as mock_sleep:
        df = client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    # At least one sleep call must have used the Retry-After value
    sleep_args = [call.args[0] for call in mock_sleep.call_args_list]
    assert retry_after_val in sleep_args, (
        f"Expected time.sleep({retry_after_val}) to be called; calls: {sleep_args}"
    )
    assert len(df) == 5
    assert len(resp_lib.calls) == 2


# ---------------------------------------------------------------------------
# Tests — weight threshold / pre-emptive sleep
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_weight_threshold_triggers_sleep_before_next_request(
    client: BinanceRESTClient,
) -> None:
    """After a response with weight=1100, the next request must be preceded by sleep."""
    # First call: returns high weight (1100 > 1000 threshold)
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=_make_kline_rows(10),
        status=200,
        headers=_WEIGHT_HDR_HIGH,
    )
    # Second call: normal weight
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=_make_kline_rows(5),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    end_ms_1 = _START_MS + 10 * _INTERVAL_MS
    end_ms_2 = _START_MS + 5 * _INTERVAL_MS

    with patch("src.ingestion.rest_client.time.sleep") as mock_sleep:
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms_1)

        # weight_used is now 1100, above the 1000 threshold
        assert client.weight_used == 1_100

        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms_2)

        # sleep must have been called (pre-emptive rate-limit avoidance)
        assert mock_sleep.called, "Expected time.sleep() to be called but it was not"

    # After sleeping, weight resets to 0 then is updated from the second response
    assert client.weight_used == 10


@resp_lib.activate
def test_weight_below_threshold_does_not_sleep(
    client: BinanceRESTClient,
) -> None:
    """When weight stays below threshold, time.sleep must NOT be called."""
    resp_lib.add(
        resp_lib.GET,
        _KLINES_URL,
        json=_make_kline_rows(5),
        status=200,
        headers={"X-MBX-USED-WEIGHT-1M": "50"},  # well below 1000
    )

    end_ms = _START_MS + 5 * _INTERVAL_MS

    with patch("src.ingestion.rest_client.time.sleep") as mock_sleep:
        client.get_klines("DOGEUSDT", "1h", _START_MS, end_ms)

    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — get_exchange_info (caching)
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_exchange_info_cached_on_second_call(
    client: BinanceRESTClient,
) -> None:
    """Second call to get_exchange_info() must be served from cache (1 HTTP req)."""
    resp_lib.add(
        resp_lib.GET,
        _EXCHANGE_INFO_URL,
        json=_make_exchange_info(),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    result1 = client.get_exchange_info()
    result2 = client.get_exchange_info()

    assert result1 == result2
    assert result1["symbols"][0]["symbol"] == "DOGEUSDT"
    assert len(resp_lib.calls) == 1, (
        f"Expected exactly 1 HTTP request for two get_exchange_info() calls; "
        f"got {len(resp_lib.calls)}"
    )


@resp_lib.activate
def test_get_exchange_info_cache_expires_after_ttl(
    client: BinanceRESTClient,
) -> None:
    """After the cache TTL is force-expired, the client must re-fetch."""
    resp_lib.add(
        resp_lib.GET,
        _EXCHANGE_INFO_URL,
        json=_make_exchange_info(),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )
    resp_lib.add(
        resp_lib.GET,
        _EXCHANGE_INFO_URL,
        json=_make_exchange_info(),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    client.get_exchange_info()
    # Force-expire the cache
    client._exchange_info_cache_ts = 0.0
    client.get_exchange_info()

    assert len(resp_lib.calls) == 2, (
        f"Expected 2 HTTP requests after cache expiry; got {len(resp_lib.calls)}"
    )


# ---------------------------------------------------------------------------
# Tests — get_order_book
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_order_book_returns_dict(
    client: BinanceRESTClient,
) -> None:
    """get_order_book() returns a plain dict with bids/asks keys."""
    mock_book = {
        "lastUpdateId": 123456,
        "bids": [["0.10000", "1000.00"]],
        "asks": [["0.10010", "800.00"]],
    }
    resp_lib.add(
        resp_lib.GET,
        _DEPTH_URL,
        json=mock_book,
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    result = client.get_order_book("DOGEUSDT", limit=5)

    assert isinstance(result, dict)
    assert "bids" in result
    assert "asks" in result
    assert "lastUpdateId" in result


# ---------------------------------------------------------------------------
# Tests — get_recent_trades
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_recent_trades_returns_dataframe_with_correct_types(
    client: BinanceRESTClient,
) -> None:
    """get_recent_trades() returns DataFrame with required columns and correct types."""
    trade_count = 5
    resp_lib.add(
        resp_lib.GET,
        _AGG_TRADES_URL,
        json=_make_agg_trades(trade_count),
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    df = client.get_recent_trades("DOGEUSDT", limit=trade_count)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == trade_count
    for col in ("trade_id", "price", "qty", "timestamp_ms", "is_buyer_maker"):
        assert col in df.columns, f"Missing column: {col}"

    # Strict type checks — timestamps must be int, prices must be float
    assert df["trade_id"].dtype.kind == "i", "trade_id must be integer dtype"
    assert df["timestamp_ms"].dtype.kind == "i", "timestamp_ms must be integer dtype"
    assert df["price"].dtype.kind == "f", "price must be float dtype"


@resp_lib.activate
def test_get_recent_trades_empty_response(
    client: BinanceRESTClient,
) -> None:
    """Empty aggTrades response → empty DataFrame, no crash."""
    resp_lib.add(
        resp_lib.GET,
        _AGG_TRADES_URL,
        json=[],
        status=200,
        headers=_WEIGHT_HDR_NORMAL,
    )

    df = client.get_recent_trades("DOGEUSDT")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests — exception class attributes
# ---------------------------------------------------------------------------


def test_binance_api_error_carries_status_code() -> None:
    """BinanceAPIError.status_code is set correctly."""
    err = BinanceAPIError("something broke", status_code=503)
    assert err.status_code == 503
    assert "something broke" in str(err)


def test_binance_rate_limit_error_default_and_custom_retry_after() -> None:
    """BinanceRateLimitError.retry_after defaults to 60 and can be overridden."""
    default_err = BinanceRateLimitError("rate limited")
    assert default_err.retry_after == 60
    assert default_err.status_code == 429

    custom_err = BinanceRateLimitError("rate limited", retry_after=30)
    assert custom_err.retry_after == 30


def test_binance_auth_error_is_subclass_of_api_error() -> None:
    """BinanceAuthError must be a subclass of BinanceAPIError."""
    err = BinanceAuthError("forbidden", status_code=403)
    assert isinstance(err, BinanceAPIError)
    assert err.status_code == 403


def test_data_validation_error_is_not_api_error() -> None:
    """DataValidationError must NOT inherit from BinanceAPIError."""
    err = DataValidationError("schema mismatch")
    assert not isinstance(err, BinanceAPIError)
    assert "schema mismatch" in str(err)
