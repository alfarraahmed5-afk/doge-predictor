"""Binance USD-M Futures REST API client for funding rate data.

Mirrors the structure of ``BinanceRESTClient`` but targets the Futures base
URL (``https://fapi.binance.com``) and its separate rate limit bucket
(2 400 weight/minute vs 1 200 for Spot).

Key responsibilities:
    - Paginate ``GET /fapi/v1/fundingRate`` across arbitrarily long date
      ranges, advancing the cursor via ``startTime``.
    - Track ``X-MBX-USED-WEIGHT-1M`` from the Futures endpoint and sleep
      pre-emptively when approaching the 2 400 weight/min limit.
    - Validate every response against ``FundingRateSchema`` before returning.
    - Retry on 5xx / 429 with exponential back-off; raise typed exceptions
      from ``src.ingestion.exceptions`` on exhaustion.

Usage::

    from src.ingestion.futures_client import BinanceFuturesClient

    client = BinanceFuturesClient()
    df = client.get_funding_rates(
        symbol="DOGEUSDT",
        start_ms=1_603_065_600_000,   # 2020-10-19 00:00 UTC
        end_ms=1_672_531_200_000,     # 2023-01-01 00:00 UTC
    )
    # columns: funding_time (int), funding_rate (float), symbol (str)
"""

from __future__ import annotations

import os
import threading
import time
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

__all__ = ["BinanceFuturesClient"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Binance USD-M Futures REST base URL.
_FAPI_BASE_URL: str = "https://fapi.binance.com"

#: Funding rates per request (Binance max is 1 000).
_MAX_FUNDING_PER_REQUEST: int = 1_000

#: Pre-emptive sleep threshold out of the 2 400 weight/min Futures limit.
_WEIGHT_THRESHOLD: int = 2_000

#: Max retry attempts before propagating the exception.
_MAX_RETRIES: int = 5

#: Retryable HTTP status codes.
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 418, 500, 502, 503, 504})

#: Non-retryable HTTP status codes (client errors).
_NON_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({400, 401, 403, 404})

#: Milliseconds per 8-hour funding interval.
_MS_PER_8H: int = 8 * 3_600_000


# ---------------------------------------------------------------------------
# BinanceFuturesClient
# ---------------------------------------------------------------------------


class BinanceFuturesClient:
    """Binance USD-M Futures REST client for funding rate data.

    Rate-limited, retrying client that targets the ``/fapi/v1/fundingRate``
    endpoint.  The Futures weight bucket is completely independent of the Spot
    bucket; each instance tracks its own weight counter.

    Args:
        api_key: Binance API key.  Defaults to the ``BINANCE_API_KEY``
            environment variable.  The funding rate endpoint is public but
            providing a key raises the rate limit.
        api_secret: Binance API secret.  Stored for future signed-endpoint
            use; not used by the current public endpoints.
        base_url: Override the Futures base URL.  Useful for pointing at a
            testnet or mock server during testing.
        weight_threshold: Pre-emptive sleep threshold (out of 2 400/min).
            Defaults to 2 000 to leave 400 points of headroom.
        max_retries: Maximum retry attempts on retryable errors.

    Attributes:
        weight_used: Rolling weight consumed in the current UTC minute.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        weight_threshold: int = _WEIGHT_THRESHOLD,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        """Initialise the Futures client.

        Args:
            api_key: Binance API key.
            api_secret: Binance API secret.
            base_url: Override Futures base URL.
            weight_threshold: Weight threshold for pre-emptive sleep.
            max_retries: Max retries on retryable errors.
        """
        self._api_key: str = api_key or os.getenv("BINANCE_API_KEY", "")
        self._api_secret: str = api_secret or os.getenv("BINANCE_API_SECRET", "")
        self._base_url: str = (base_url or _FAPI_BASE_URL).rstrip("/")
        self._weight_threshold: int = weight_threshold
        self._max_retries: int = max_retries
        self._weight_lock: threading.Lock = threading.Lock()
        self.weight_used: int = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_funding_rates(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch historical funding rates for *symbol* in ``[start_ms, end_ms)``.

        Automatically paginates using ``startTime`` cursor advance until all
        rows in the requested window are fetched.  Funding rates are reported
        at 8-hour intervals (00:00, 08:00, 16:00 UTC).

        Args:
            symbol: Futures symbol (e.g. ``"DOGEUSDT"``).
            start_ms: Inclusive start timestamp — UTC epoch milliseconds.
            end_ms: Exclusive end timestamp — UTC epoch milliseconds.

        Returns:
            DataFrame with columns:

                - ``funding_time`` (int): Settlement timestamp, UTC epoch ms.
                - ``funding_rate`` (float): Funding rate value.
                - ``symbol`` (str): Trading pair symbol.

        Raises:
            BinanceAPIError: On unrecoverable API errors.
            BinanceRateLimitError: If rate limit is hit after all retries.
            DataValidationError: If the response schema is invalid.
        """
        cursor: int = start_ms
        all_rows: list[dict[str, Any]] = []

        while cursor < end_ms:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "startTime": cursor,
                "endTime": end_ms,
                "limit": _MAX_FUNDING_PER_REQUEST,
            }

            raw: list[dict[str, Any]] = self._request(
                "GET", "/fapi/v1/fundingRate", params=params
            )

            if not raw:
                break

            for row in raw:
                all_rows.append(
                    {
                        "funding_time": int(row["fundingTime"]),
                        "funding_rate": float(row["fundingRate"]),
                        "symbol": str(row.get("symbol", symbol.upper())),
                    }
                )

            last_time: int = int(raw[-1]["fundingTime"])
            if last_time >= end_ms or len(raw) < _MAX_FUNDING_PER_REQUEST:
                break

            # Advance cursor past the last returned funding time
            cursor = last_time + _MS_PER_8H

        if not all_rows:
            logger.warning(
                "get_funding_rates: no data for {} in [{}, {})",
                symbol,
                start_ms,
                end_ms,
            )
            return pd.DataFrame(
                columns=["funding_time", "funding_rate", "symbol"]
            )

        df = pd.DataFrame(all_rows)

        # Deduplicate by funding_time
        df = df.drop_duplicates(subset=["funding_time"]).sort_values("funding_time")
        df = df.reset_index(drop=True)

        # Validate types
        try:
            df["funding_time"] = df["funding_time"].astype(int)
            df["funding_rate"] = df["funding_rate"].astype(float)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"get_funding_rates: type coercion failed — {exc}"
            ) from exc

        # Validate funding_rate range (should be in [-0.01, 0.01])
        rate_min = df["funding_rate"].min()
        rate_max = df["funding_rate"].max()
        if rate_min < -0.05 or rate_max > 0.05:
            raise DataValidationError(
                f"get_funding_rates: funding_rate out of plausible range "
                f"[{rate_min:.6f}, {rate_max:.6f}]"
            )

        logger.info(
            "get_funding_rates: {} rows for {} in [{}, {})",
            len(df),
            symbol,
            start_ms,
            end_ms,
        )
        return df

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a single HTTP request with retry and weight tracking.

        Args:
            method: HTTP method (``"GET"``).
            path: API path (e.g. ``"/fapi/v1/fundingRate"``).
            params: Query parameters.

        Returns:
            Parsed JSON response body.

        Raises:
            BinanceRateLimitError: On exhausted 429/418 retries.
            BinanceAuthError: On 401/403.
            BinanceAPIError: On non-retryable 4xx or exhausted 5xx retries.
        """
        url = self._base_url + path
        headers = {"X-MBX-APIKEY": self._api_key} if self._api_key else {}

        for attempt in range(self._max_retries):
            # Pre-emptive sleep if weight is high
            with self._weight_lock:
                if self.weight_used >= self._weight_threshold:
                    sleep_s = 61.0
                    logger.warning(
                        "Futures weight={} >= threshold={}; sleeping {:.0f}s",
                        self.weight_used,
                        self._weight_threshold,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    self.weight_used = 0

            try:
                resp = requests.get(
                    url, params=params, headers=headers, timeout=30
                )
            except requests.RequestException as exc:
                if attempt < self._max_retries - 1:
                    wait_s = 2 ** attempt
                    logger.warning(
                        "Futures request error (attempt {}/{}): {}; retrying in {}s",
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait_s,
                    )
                    time.sleep(wait_s)
                    continue
                raise BinanceAPIError(
                    f"Futures request failed after {self._max_retries} attempts: {exc}"
                ) from exc

            # Update weight counter
            weight_header = resp.headers.get("X-MBX-USED-WEIGHT-1M", "0")
            try:
                with self._weight_lock:
                    self.weight_used = int(weight_header)
            except ValueError:
                pass

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in {401, 403}:
                raise BinanceAuthError(
                    f"Futures auth error {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )

            if resp.status_code in {400, 404}:
                raise BinanceAPIError(
                    f"Futures client error {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )

            if resp.status_code in _RETRYABLE_STATUS_CODES:
                retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "Futures rate limit / server error {} (attempt {}/{}); "
                        "sleeping {}s",
                        resp.status_code,
                        attempt + 1,
                        self._max_retries,
                        retry_after,
                    )
                    time.sleep(retry_after)
                    continue
                raise BinanceRateLimitError(
                    f"Futures rate limit after {self._max_retries} retries",
                    retry_after=retry_after,
                )

            raise BinanceAPIError(
                f"Unexpected Futures status {resp.status_code}: {resp.text[:200]}",
                status_code=resp.status_code,
            )

        raise BinanceAPIError(
            f"Futures request exhausted {self._max_retries} retries"
        )
