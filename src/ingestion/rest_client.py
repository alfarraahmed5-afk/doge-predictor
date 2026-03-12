"""Binance Spot REST API client with rate limiting and exponential-backoff retry.

This module is the single point of contact between the ingestion pipeline and
the Binance HTTP API.  It is intentionally free of business logic — it fetches
data, validates the schema contract, and hands back a clean DataFrame or dict.

Key responsibilities:
    - Track the ``X-MBX-USED-WEIGHT-1M`` header on every response and
      pre-emptively sleep until the next UTC-minute boundary when the
      rolling weight approaches the 1 200 req/min Binance limit.
    - Retry transparently on server-side (5xx) and rate-limit (429/418)
      errors using exponential backoff, raising typed exceptions on
      exhaustion.
    - Paginate ``/api/v3/klines`` automatically across arbitrarily long
      date ranges (up to the full history from 2019).
    - Validate every klines response against ``OHLCVSchema`` before
      returning, surfacing schema violations as ``DataValidationError``.

Usage::

    from src.ingestion.rest_client import BinanceRESTClient

    client = BinanceRESTClient()
    df = client.get_klines("DOGEUSDT", "1h", start_ms=1_640_995_200_000,
                           end_ms=1_672_531_200_000)
"""

from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from loguru import logger

from src.ingestion.exceptions import (
    BinanceAPIError,
    BinanceAuthError,
    BinanceRateLimitError,
    DataValidationError,
)
from src.utils.helpers import compute_expected_row_count, interval_to_ms

__all__ = ["BinanceRESTClient"]

# ---------------------------------------------------------------------------
# Module-level constants — all values loaded from config in the calling layer;
# the client itself uses these defaults, which mirror doge_settings.yaml.
# ---------------------------------------------------------------------------

#: Default Binance Spot REST base URL.
_BASE_URL: str = "https://api.binance.com"

#: Pre-emptive sleep threshold (out of 1 200 weight/min limit).
_WEIGHT_THRESHOLD: int = 1_000

#: Maximum candles returned by a single klines request.
_MAX_KLINES_PER_REQUEST: int = 1_000

#: Maximum retry attempts before propagating the error.
_MAX_RETRIES: int = 5

#: Exchange-info in-process cache TTL in seconds (1 hour).
_EXCHANGE_INFO_CACHE_TTL: float = 3_600.0

#: Status codes on which the client retries with backoff.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 418, 500, 502, 503, 504})

#: Status codes that indicate a broken request — never retry.
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404})

#: Ordered column names matching the Binance klines list-of-lists response.
_KLINE_COLS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "n_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "_ignore",
]

#: Subset of kline columns kept in the returned DataFrame.
_KLINE_KEEP_COLS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
]


# ---------------------------------------------------------------------------
# BinanceRESTClient
# ---------------------------------------------------------------------------


class BinanceRESTClient:
    """Binance Spot REST API wrapper with rate limiting and retry logic.

    The client is thread-safe with respect to the rolling weight counter.
    A single instance should be reused across the ingestion pipeline rather
    than re-instantiated per request.

    Args:
        api_key: Binance API key.  Defaults to the ``BINANCE_API_KEY``
            environment variable.  Public endpoints (klines, depth, trades)
            do not require a key, but providing one raises the rate limit.
        api_secret: Binance API secret.  Defaults to
            ``BINANCE_API_SECRET``.  Currently unused (no signed endpoints
            are called by this client), but stored for future use.
        base_url: Override the REST base URL.  Useful for pointing at a
            Binance testnet or a mock server during testing.
        weight_threshold: Pre-emptive sleep when rolling weight exceeds
            this value.  Defaults to 1 000 (Binance limit is 1 200).
        max_retries: Maximum number of attempts per request.  Defaults to 5.

    Attributes:
        weight_used (int): Current rolling 1-minute weight as reported by
            the last response header.  Thread-safe via ``_weight_lock``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        weight_threshold: int = _WEIGHT_THRESHOLD,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._api_key: str = api_key or os.getenv("BINANCE_API_KEY", "")
        self._api_secret: str = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self._base_url: str = (base_url or _BASE_URL).rstrip("/")
        self._weight_threshold: int = weight_threshold
        self._max_retries: int = max_retries

        self._session: requests.Session = requests.Session()
        self._weight_lock: threading.Lock = threading.Lock()
        self.weight_used: int = 0

        # Exchange-info in-process cache
        self._exchange_info_cache: dict[str, Any] | None = None
        self._exchange_info_cache_ts: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build the HTTP headers dict for a request.

        Returns:
            Dict containing ``X-MBX-APIKEY`` if an API key is configured,
            otherwise an empty dict.
        """
        headers: dict[str, str] = {}
        if self._api_key:
            headers["X-MBX-APIKEY"] = self._api_key
        return headers

    def _update_weight(self, response_headers: Any) -> None:
        """Parse and store the rolling weight from a response header.

        Args:
            response_headers: The ``response.headers`` mapping from
                ``requests``.  If the header is missing or unparseable the
                internal counter is left unchanged.
        """
        raw: str = response_headers.get("X-MBX-USED-WEIGHT-1M", "")
        if raw:
            try:
                with self._weight_lock:
                    self.weight_used = int(raw)
            except (ValueError, TypeError):
                logger.warning(
                    "Could not parse X-MBX-USED-WEIGHT-1M header value: {!r}", raw
                )

    def _sleep_until_next_minute(self) -> float:
        """Sleep until the next UTC-minute boundary and return the duration.

        Called pre-emptively when ``weight_used >= weight_threshold`` to
        avoid triggering a 429 response.  Resets ``weight_used`` to 0
        after sleeping because the Binance rolling window will have moved.

        Returns:
            Number of seconds slept (float).
        """
        now: datetime = datetime.now(timezone.utc)
        # Seconds until the next whole UTC minute (+0.5 s safety buffer)
        seconds_to_sleep: float = (
            60.0 - now.second - now.microsecond / 1_000_000 + 0.5
        )
        seconds_to_sleep = max(1.0, seconds_to_sleep)
        logger.info(
            "Weight threshold reached (weight_used={}, threshold={}). "
            "Sleeping {:.1f}s until next UTC minute.",
            self.weight_used,
            self._weight_threshold,
            seconds_to_sleep,
        )
        time.sleep(seconds_to_sleep)
        with self._weight_lock:
            self.weight_used = 0
        return seconds_to_sleep

    def _check_weight_and_sleep(self) -> None:
        """Pre-flight weight check — sleep if at or above the threshold.

        Called once before every outgoing HTTP request.
        """
        with self._weight_lock:
            should_sleep: bool = self.weight_used >= self._weight_threshold
        if should_sleep:
            self._sleep_until_next_minute()

    # ------------------------------------------------------------------
    # Core request dispatcher
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request with rate limiting, retry, and error handling.

        Implements the following policy:
        - Pre-flight weight check before every attempt; sleep if needed.
        - Retry up to ``max_retries`` times on retryable status codes
          (429, 418, 500, 502, 503, 504) with exponential backoff.
        - For 429 responses, honour the ``Retry-After`` header instead of
          the exponential backoff.
        - Do NOT retry on client errors (400, 401, 403, 404).
        - Raise :class:`~src.ingestion.exceptions.BinanceRateLimitError`
          when 429 persists after all retries.
        - Raise :class:`~src.ingestion.exceptions.BinanceAuthError` on
          401/403.
        - Raise :class:`~src.ingestion.exceptions.BinanceAPIError` for all
          other failures.

        Args:
            method: HTTP verb (``"GET"``, ``"POST"``, …).
            endpoint: Path component starting with ``/``,
                e.g. ``"/api/v3/klines"``.
            params: Query-string parameters dict.  Values that are
                ``None`` are automatically excluded by ``requests``.

        Returns:
            Parsed JSON response body (list or dict).

        Raises:
            BinanceRateLimitError: 429 after all retries exhausted.
            BinanceAuthError: 401 or 403 response.
            BinanceAPIError: Any other non-2xx response or transport error.
        """
        url: str = self._base_url + endpoint
        headers: dict[str, str] = self._build_headers()

        last_status: int | None = None
        last_retry_after: int = 60

        for attempt in range(self._max_retries):
            # ---- pre-flight weight check ----
            self._check_weight_and_sleep()

            # ---- execute request ----
            try:
                response: requests.Response = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=30,
                )
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "Transport error on attempt {}/{} for {} {}: {}",
                    attempt + 1,
                    self._max_retries,
                    method,
                    endpoint,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    backoff: float = float(2 ** attempt)
                    logger.info("Retrying in {:.0f}s...", backoff)
                    time.sleep(backoff)
                    continue
                raise BinanceAPIError(
                    f"Request to {endpoint} failed after {self._max_retries} "
                    f"attempts: {exc}"
                ) from exc

            # ---- update rolling weight counter ----
            self._update_weight(response.headers)
            last_status = response.status_code

            logger.debug(
                "HTTP {} {} → {} | weight_used={}",
                method,
                endpoint,
                response.status_code,
                self.weight_used,
            )

            # ---- success ----
            if response.status_code == 200:
                return response.json()

            # ---- non-retryable client errors ----
            if response.status_code in _NON_RETRYABLE_STATUS_CODES:
                if response.status_code in {401, 403}:
                    raise BinanceAuthError(
                        f"Authentication/authorization error "
                        f"({response.status_code}) for {endpoint}: {response.text}",
                        status_code=response.status_code,
                    )
                raise BinanceAPIError(
                    f"Client error {response.status_code} for {endpoint}: "
                    f"{response.text}",
                    status_code=response.status_code,
                )

            # ---- retryable errors ----
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if response.status_code == 429:
                    # Honour Retry-After header from Binance
                    last_retry_after = int(
                        response.headers.get("Retry-After", 60)
                    )
                    logger.warning(
                        "Rate limit 429 on attempt {}/{} for {}. "
                        "Sleeping {}s (Retry-After).",
                        attempt + 1,
                        self._max_retries,
                        endpoint,
                        last_retry_after,
                    )
                    time.sleep(last_retry_after)
                else:
                    backoff = float(2 ** attempt)
                    logger.warning(
                        "Server error {} on attempt {}/{} for {}. "
                        "Sleeping {:.0f}s.",
                        response.status_code,
                        attempt + 1,
                        self._max_retries,
                        endpoint,
                        backoff,
                    )
                    time.sleep(backoff)

                if attempt == self._max_retries - 1:
                    # Final attempt also failed — raise typed exception
                    if last_status == 429:
                        raise BinanceRateLimitError(
                            f"Rate limit (429) persisted after "
                            f"{self._max_retries} retries for {endpoint}",
                            retry_after=last_retry_after,
                        )
                    raise BinanceAPIError(
                        f"Server error {last_status} persisted after "
                        f"{self._max_retries} retries for {endpoint}",
                        status_code=last_status,
                    )
                continue  # next attempt

            # ---- unexpected status code ----
            raise BinanceAPIError(
                f"Unexpected status {response.status_code} for {endpoint}: "
                f"{response.text}",
                status_code=response.status_code,
            )

        # Unreachable, but satisfies type checker
        raise BinanceAPIError(
            f"Request to {endpoint} failed after {self._max_retries} attempts"
        )

    # ------------------------------------------------------------------
    # DataFrame construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_klines(raw: list[list[Any]], symbol: str) -> pd.DataFrame:
        """Convert a raw Binance klines list-of-lists into a typed DataFrame.

        Args:
            raw: List of kline arrays as returned by ``/api/v3/klines``.
            symbol: Trading pair symbol (stored as a column for traceability).

        Returns:
            DataFrame with columns: open_time (int), open, high, low, close,
            volume (float), close_time (int), symbol (str).

        Raises:
            DataValidationError: If ``raw`` cannot be parsed into the
                expected structure.
        """
        try:
            df: pd.DataFrame = pd.DataFrame(raw, columns=_KLINE_COLS)
        except Exception as exc:
            raise DataValidationError(
                f"Cannot construct DataFrame from klines response: {exc}"
            ) from exc

        # Keep only the columns the pipeline needs
        df = df[_KLINE_KEEP_COLS].copy()

        # Enforce strict types — open_time and close_time must be int (UTC ms)
        df["open_time"] = df["open_time"].astype(int)
        df["close_time"] = df["close_time"].astype(int)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        df["symbol"] = symbol
        return df

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for *symbol*/*interval* in ``[start_ms, end_ms)``.

        Paginates automatically: each request fetches up to 1 000 candles.
        If the API returns exactly 1 000 rows and the last candle's
        ``close_time < end_ms``, a further request is issued using
        ``close_time + 1`` as the next ``startTime``.

        Row-count sanity check: if the actual number of rows deviates from
        the expected count (derived from the time range) by more than 2,
        a ``WARNING`` is logged.  This does not raise — sparse periods in
        DOGE history (e.g. before exchange listing) legitimately have gaps.

        All returned timestamps are ``int`` (UTC epoch milliseconds).

        Args:
            symbol: Binance trading pair, e.g. ``"DOGEUSDT"``.
            interval: Binance kline interval string, e.g. ``"1h"``, ``"4h"``.
            start_ms: Inclusive start timestamp (UTC epoch ms).
            end_ms: Exclusive end timestamp (UTC epoch ms). Must be > start_ms.

        Returns:
            DataFrame validated against ``OHLCVSchema`` with columns:
            open_time, open, high, low, close, volume, close_time, symbol.
            The DataFrame is sorted by open_time and deduplicated.

        Raises:
            ValueError: If ``start_ms >= end_ms`` or ``interval`` is unknown.
            DataValidationError: If the response fails Pandera schema validation.
            BinanceRateLimitError: If rate-limited after all retries.
            BinanceAPIError: For any other API or transport failure.
        """
        if start_ms >= end_ms:
            raise ValueError(
                f"start_ms ({start_ms}) must be strictly less than end_ms ({end_ms})"
            )

        interval_ms: int = interval_to_ms(interval)
        all_rows: list[list[Any]] = []
        current_start: int = start_ms
        page_num: int = 0

        while True:
            page_num += 1
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": _MAX_KLINES_PER_REQUEST,
            }

            logger.debug(
                "Klines page {} | symbol={} interval={} startTime={} endTime={}",
                page_num,
                symbol,
                interval,
                current_start,
                end_ms,
            )

            raw_page: Any = self._request("GET", "/api/v3/klines", params=params)

            if not isinstance(raw_page, list):
                raise DataValidationError(
                    f"Expected list from /api/v3/klines, got "
                    f"{type(raw_page).__name__}: {raw_page!r}"
                )

            if not raw_page:
                logger.debug("Empty klines page — end of data for {}", symbol)
                break

            all_rows.extend(raw_page)

            logger.info(
                "Klines page {} fetched: {} rows | symbol={} interval={} weight_used={}",
                page_num,
                len(raw_page),
                symbol,
                interval,
                self.weight_used,
            )

            # Stopping condition: reached or passed the requested end window.
            # NOTE: Do NOT stop on short pages (len < limit). Binance can return
            # fewer than `limit` rows mid-history (e.g. when the symbol listing
            # date falls inside a batch window, or when there are minor data gaps).
            # Stopping on short pages causes premature termination and truncated
            # datasets. The only safe stopping conditions are timestamp-based.
            last_close_time: int = int(raw_page[-1][6])
            if last_close_time >= end_ms:
                break

            # Advance to next page
            current_start = last_close_time + 1

            # Guard: if the next cursor has reached or passed the end boundary
            # (can happen when last_close_time = end_ms - 1ms, i.e. the batch
            # exactly fills the window), exit without making a redundant request.
            if current_start >= end_ms:
                break

        # Build and validate DataFrame
        if not all_rows:
            logger.warning(
                "No klines returned for symbol={} interval={} [{}, {})",
                symbol,
                interval,
                start_ms,
                end_ms,
            )
            return pd.DataFrame(
                columns=["open_time", "open", "high", "low", "close", "volume",
                         "close_time", "symbol"]
            )

        df: pd.DataFrame = self._parse_klines(all_rows, symbol)

        # Deduplicate and sort (page boundaries can produce duplicate rows)
        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

        # Row-count sanity check (soft — warns but does not raise)
        try:
            expected_count: int = compute_expected_row_count(
                start_ms, end_ms, interval_ms
            )
            actual_count: int = len(df)
            deviation: int = abs(actual_count - expected_count)
            if deviation > 2:
                logger.warning(
                    "Row count deviation for {} {}: expected ~{} rows, "
                    "got {} (diff={}). "
                    "Data may have gaps or the time range boundaries are imprecise.",
                    symbol,
                    interval,
                    expected_count,
                    actual_count,
                    deviation,
                )
        except ValueError:
            pass  # start_ms == end_ms edge case already caught above

        # Schema validation — import here to avoid circular imports at module level
        try:
            from src.processing.df_schemas import OHLCVSchema, validate_df  # noqa: PLC0415
            validate_df(df, OHLCVSchema)
        except Exception as exc:
            raise DataValidationError(
                f"Klines response for {symbol}/{interval} failed schema "
                f"validation: {exc}"
            ) from exc

        logger.info(
            "get_klines complete | symbol={} interval={} rows={} pages={}",
            symbol,
            interval,
            len(df),
            page_num,
        )
        return df

    def get_exchange_info(self) -> dict[str, Any]:
        """Fetch trading rules for all symbols, with a 1-hour in-process cache.

        The exchange-info endpoint is expensive (weight=10) and changes
        infrequently.  The result is cached in-process for
        ``_EXCHANGE_INFO_CACHE_TTL`` seconds (3 600 s = 1 h).

        Returns:
            Raw Binance exchange-info response dict containing ``symbols``,
            ``rateLimits``, ``timezone``, and ``serverTime`` keys.

        Raises:
            BinanceAPIError: On transport or API error.
        """
        now_ts: float = time.monotonic()
        cache_age: float = now_ts - self._exchange_info_cache_ts

        if self._exchange_info_cache is not None and cache_age < _EXCHANGE_INFO_CACHE_TTL:
            logger.debug(
                "Exchange info cache hit (age={:.0f}s < TTL={:.0f}s)",
                cache_age,
                _EXCHANGE_INFO_CACHE_TTL,
            )
            return self._exchange_info_cache

        logger.info("Fetching exchange info (cache miss or expired).")
        result: dict[str, Any] = self._request("GET", "/api/v3/exchangeInfo")

        self._exchange_info_cache = result
        self._exchange_info_cache_ts = now_ts

        logger.info(
            "Exchange info cached | symbols_count={}",
            len(result.get("symbols", [])),
        )
        return result

    def get_order_book(
        self,
        symbol: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Fetch the current order book for *symbol*.

        Args:
            symbol: Trading pair, e.g. ``"DOGEUSDT"``.
            limit: Depth of bids/asks to fetch.  Binance valid values:
                5, 10, 20, 50, 100, 500, 1000, 5000.  Defaults to 100.

        Returns:
            Dict with keys ``lastUpdateId`` (int), ``bids`` (list of
            [price_str, qty_str]), and ``asks`` (list of [price_str,
            qty_str]).

        Raises:
            BinanceAPIError: On transport or API error.
        """
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        result: dict[str, Any] = self._request("GET", "/api/v3/depth", params=params)

        logger.info(
            "Order book fetched | symbol={} depth={} weight_used={}",
            symbol,
            limit,
            self.weight_used,
        )
        return result

    def get_recent_trades(
        self,
        symbol: str,
        limit: int = 1_000,
    ) -> pd.DataFrame:
        """Fetch recent aggregate trades for *symbol*.

        Uses the ``/api/v3/aggTrades`` endpoint which de-duplicates trades
        filled at the same price, time, and direction into a single row.

        Args:
            symbol: Trading pair, e.g. ``"DOGEUSDT"``.
            limit: Maximum number of trades to return (max 1 000).

        Returns:
            DataFrame with columns: trade_id (int), price (float),
            qty (float), timestamp_ms (int), is_buyer_maker (bool).

        Raises:
            DataValidationError: If the response is malformed.
            BinanceAPIError: On transport or API error.
        """
        params: dict[str, Any] = {"symbol": symbol, "limit": limit}
        raw: Any = self._request("GET", "/api/v3/aggTrades", params=params)

        if not isinstance(raw, list):
            raise DataValidationError(
                f"Expected list from /api/v3/aggTrades, got "
                f"{type(raw).__name__}"
            )

        if not raw:
            logger.warning("aggTrades returned empty list for symbol={}", symbol)
            return pd.DataFrame(
                columns=["trade_id", "price", "qty", "timestamp_ms", "is_buyer_maker"]
            )

        try:
            df: pd.DataFrame = pd.DataFrame(raw)
            # Binance aggTrades field keys
            df = df.rename(
                columns={
                    "a": "trade_id",
                    "p": "price",
                    "q": "qty",
                    "T": "timestamp_ms",
                    "m": "is_buyer_maker",
                }
            )
            out: pd.DataFrame = df[
                ["trade_id", "price", "qty", "timestamp_ms", "is_buyer_maker"]
            ].copy()
            out["trade_id"] = out["trade_id"].astype(int)
            out["price"] = out["price"].astype(float)
            out["qty"] = out["qty"].astype(float)
            out["timestamp_ms"] = out["timestamp_ms"].astype(int)
            out["is_buyer_maker"] = out["is_buyer_maker"].astype(bool)
        except Exception as exc:
            raise DataValidationError(
                f"Cannot parse aggTrades response for {symbol}: {exc}"
            ) from exc

        logger.info(
            "aggTrades fetched | symbol={} rows={} weight_used={}",
            symbol,
            len(out),
            self.weight_used,
        )
        return out
