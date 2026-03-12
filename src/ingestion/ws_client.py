"""Binance WebSocket client for live candle and trade streaming.

Provides :class:`BinanceWebSocketClient`, a thread-based WebSocket consumer
that subscribes to Binance stream endpoints and forwards messages to
registered callbacks.  A watchdog thread monitors the connection and
reconnects automatically when the stream goes silent.

Supported streams:
    - ``{symbol}@kline_{interval}`` — closed candles (fired at candle close)
    - ``{symbol}@aggTrade``         — aggregate trade ticks (live)

Usage::

    from src.ingestion.ws_client import BinanceWebSocketClient

    def on_candle(msg: dict) -> None:
        if msg["k"]["x"]:  # candle is closed
            print(msg["k"]["c"])  # close price

    client = BinanceWebSocketClient()
    client.subscribe_klines("dogeusdt", "1h", on_candle)
    client.connect()   # starts background thread
    # ... later ...
    client.disconnect()

Notes:
    - Binance WebSocket streams are public; no API key required.
    - Maximum 5 messages/second incoming per connection (Binance limit).
    - On reconnect, messages during the gap are not backfilled — the
      ingestion scheduler's 3-candle overlap window covers short outages.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from loguru import logger

__all__ = ["BinanceWebSocketClient"]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Binance WebSocket base URL.
_WS_BASE_URL: str = "wss://stream.binance.com:9443/ws"

#: Seconds of silence after which the watchdog triggers a reconnect.
_WATCHDOG_TIMEOUT_S: float = 90.0

#: Seconds between watchdog checks.
_WATCHDOG_INTERVAL_S: float = 15.0

#: Seconds to wait between reconnect attempts.
_RECONNECT_DELAY_S: float = 5.0

#: Maximum reconnect attempts before giving up.
_MAX_RECONNECT_ATTEMPTS: int = 10


# ---------------------------------------------------------------------------
# BinanceWebSocketClient
# ---------------------------------------------------------------------------


class BinanceWebSocketClient:
    """Thread-based Binance WebSocket client with automatic reconnect.

    Manages one WebSocket connection per instance.  Multiple streams can be
    multiplexed onto a single connection using Binance's combined stream URL
    (``/stream?streams=a/b/c``).  Each registered callback receives the raw
    parsed JSON payload for its stream.

    Args:
        base_url: Override the WebSocket base URL.
        watchdog_timeout_s: Seconds without a message before reconnect fires.
        max_reconnect_attempts: Maximum reconnect attempts before giving up.

    Attributes:
        is_connected: True when the WebSocket is currently open.
        messages_received: Total messages received since last connect().
    """

    def __init__(
        self,
        base_url: str = _WS_BASE_URL,
        watchdog_timeout_s: float = _WATCHDOG_TIMEOUT_S,
        max_reconnect_attempts: int = _MAX_RECONNECT_ATTEMPTS,
    ) -> None:
        """Initialise the WebSocket client.

        Args:
            base_url: WebSocket base URL.
            watchdog_timeout_s: Silence timeout before reconnect.
            max_reconnect_attempts: Max reconnect attempts.
        """
        self._base_url: str = base_url.rstrip("/")
        self._watchdog_timeout_s: float = watchdog_timeout_s
        self._max_reconnect_attempts: int = max_reconnect_attempts

        # Subscribed streams: stream_name → callback
        self._subscriptions: dict[str, Callable[[dict[str, Any]], None]] = {}

        # Threading state
        self._ws: Any = None  # websocket.WebSocketApp instance (lazy import)
        self._thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._last_message_ts: float = 0.0
        self._lock: threading.Lock = threading.Lock()

        # Public state
        self.is_connected: bool = False
        self.messages_received: int = 0

    # -----------------------------------------------------------------------
    # Subscription API
    # -----------------------------------------------------------------------

    def subscribe_klines(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register a callback for ``{symbol}@kline_{interval}`` messages.

        Args:
            symbol: Lowercase symbol (e.g. ``"dogeusdt"``).
            interval: Kline interval string (e.g. ``"1h"``).
            callback: Function called with the raw kline JSON payload.
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        with self._lock:
            self._subscriptions[stream_name] = callback
        logger.info("WebSocket: subscribed to stream '{}'", stream_name)

    def subscribe_agg_trades(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register a callback for ``{symbol}@aggTrade`` messages.

        Args:
            symbol: Lowercase symbol (e.g. ``"dogeusdt"``).
            callback: Function called with the raw aggTrade JSON payload.
        """
        stream_name = f"{symbol.lower()}@aggTrade"
        with self._lock:
            self._subscriptions[stream_name] = callback
        logger.info("WebSocket: subscribed to stream '{}'", stream_name)

    def unsubscribe(self, symbol: str, stream_type: str) -> None:
        """Remove a subscription.

        Args:
            symbol: Lowercase symbol.
            stream_type: ``"kline_1h"``, ``"aggTrade"``, etc.
        """
        stream_name = f"{symbol.lower()}@{stream_type}"
        with self._lock:
            self._subscriptions.pop(stream_name, None)
        logger.info("WebSocket: unsubscribed from '{}'", stream_name)

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    def connect(self) -> None:
        """Open the WebSocket connection and start background threads.

        Raises:
            ImportError: If the ``websocket-client`` package is not installed.
            RuntimeError: If no streams have been subscribed.
        """
        try:
            import websocket  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "websocket-client is required for BinanceWebSocketClient. "
                "Install it with: pip install websocket-client"
            ) from exc

        with self._lock:
            if not self._subscriptions:
                raise RuntimeError(
                    "BinanceWebSocketClient.connect() called with no subscriptions. "
                    "Call subscribe_klines() or subscribe_agg_trades() first."
                )
            streams = list(self._subscriptions.keys())

        self._stop_event.clear()
        self._reconnect(streams, attempt=0)
        self._start_watchdog(streams)

    def disconnect(self) -> None:
        """Close the WebSocket connection and stop all background threads."""
        logger.info("WebSocket: disconnecting")
        self._stop_event.set()
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=10.0)
        self.is_connected = False
        logger.info("WebSocket: disconnected")

    # -----------------------------------------------------------------------
    # Internal connection management
    # -----------------------------------------------------------------------

    def _build_url(self, streams: list[str]) -> str:
        """Build the combined stream WebSocket URL.

        Args:
            streams: List of stream name strings.

        Returns:
            Fully qualified WebSocket URL.
        """
        if len(streams) == 1:
            return f"{self._base_url}/{streams[0]}"
        combined = "/".join(streams)
        # Use combined stream URL for multiple streams
        base = self._base_url.replace("/ws", "/stream")
        return f"{base}?streams={combined}"

    def _reconnect(self, streams: list[str], attempt: int) -> None:
        """Start a new WebSocket connection in a background thread.

        Args:
            streams: List of stream names to subscribe to.
            attempt: Current reconnect attempt number (0-indexed).
        """
        import websocket  # noqa: PLC0415

        url = self._build_url(streams)
        logger.info("WebSocket: connecting to {} (attempt {})", url, attempt + 1)

        def on_open(ws: Any) -> None:
            self.is_connected = True
            self._last_message_ts = time.monotonic()
            logger.info("WebSocket: connection established")

        def on_message(ws: Any, raw: str) -> None:
            self._last_message_ts = time.monotonic()
            self.messages_received += 1
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("WebSocket: malformed JSON message received")
                return

            # Combined stream wrapper: {"stream": "...", "data": {...}}
            if "stream" in msg and "data" in msg:
                stream_name = msg["stream"]
                payload = msg["data"]
            else:
                # Single stream — infer from 'e' event type
                payload = msg
                stream_name = self._infer_stream_name(payload)

            with self._lock:
                callback = self._subscriptions.get(stream_name)
            if callback is not None:
                try:
                    callback(payload)
                except Exception as exc:
                    logger.error(
                        "WebSocket: callback for '{}' raised: {}", stream_name, exc
                    )

        def on_error(ws: Any, error: Any) -> None:
            logger.error("WebSocket error: {}", error)
            self.is_connected = False

        def on_close(ws: Any, close_status_code: Any, close_msg: Any) -> None:
            self.is_connected = False
            logger.info(
                "WebSocket closed: code={} msg={}", close_status_code, close_msg
            )

        self._ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self._thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
            name=f"ws-client-{attempt}",
        )
        self._thread.start()

    def _start_watchdog(self, streams: list[str]) -> None:
        """Start the watchdog thread that triggers reconnect on silence.

        Args:
            streams: Streams to reconnect to if silence is detected.
        """
        def _watchdog() -> None:
            attempts = 0
            while not self._stop_event.is_set():
                time.sleep(_WATCHDOG_INTERVAL_S)
                if self._stop_event.is_set():
                    break
                elapsed = time.monotonic() - self._last_message_ts
                if elapsed > self._watchdog_timeout_s:
                    attempts += 1
                    if attempts > self._max_reconnect_attempts:
                        logger.error(
                            "WebSocket: max reconnect attempts ({}) reached. Giving up.",
                            self._max_reconnect_attempts,
                        )
                        self._stop_event.set()
                        break
                    logger.warning(
                        "WebSocket watchdog: {}s silence; reconnecting (attempt {}/{})",
                        int(elapsed),
                        attempts,
                        self._max_reconnect_attempts,
                    )
                    if self._ws is not None:
                        try:
                            self._ws.close()
                        except Exception:
                            pass
                    time.sleep(_RECONNECT_DELAY_S)
                    self._last_message_ts = time.monotonic()
                    self._reconnect(streams, attempt=attempts)
                else:
                    attempts = 0  # reset on successful message

        self._watchdog_thread = threading.Thread(
            target=_watchdog,
            daemon=True,
            name="ws-watchdog",
        )
        self._watchdog_thread.start()

    @staticmethod
    def _infer_stream_name(payload: dict[str, Any]) -> str:
        """Infer the stream name from a single-stream message payload.

        Args:
            payload: Raw parsed JSON payload.

        Returns:
            Stream name string (e.g. ``"dogeusdt@kline_1h"``).
        """
        event_type = payload.get("e", "")
        symbol = str(payload.get("s", "")).lower()
        if event_type == "kline":
            interval = payload.get("k", {}).get("i", "1h")
            return f"{symbol}@kline_{interval}"
        if event_type == "aggTrade":
            return f"{symbol}@aggTrade"
        return f"{symbol}@{event_type}"
