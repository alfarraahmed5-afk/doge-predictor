"""Order book features for the DOGE prediction pipeline.

This module computes per-candle order book snapshot features at inference
time.  It operates on a raw Binance order book dict (not a DataFrame) because
order book data is ephemeral and is not stored in the OHLCV Parquet files.

Usage (inference-time only):
    book = client.get_order_book()   # returns Binance depth dict
    features = compute_orderbook_features(book)

Output is a dict so it can be merged into the inference feature vector
without creating a single-row DataFrame overhead on every candle.

Lookahead audit:
    bid_ask_spread        best_ask - best_bid at snapshot time T only        SAFE
    order_book_imbalance  top-10 bid vs ask volume at snapshot time T only   SAFE
    Both are computed solely from the live order book snapshot — no past
    or future data is involved.

CRITICAL: This module is for LIVE INFERENCE only.  It must NOT be called
during feature backfilling, as historical order book data is unavailable.
"""

from __future__ import annotations

_EPS: float = 1e-10


def compute_orderbook_features(order_book: dict) -> dict[str, float]:
    """Compute order book snapshot features from a Binance depth response.

    Processes the top 10 levels of bids and asks.  Falls back to 0.0 for
    any feature that cannot be computed (e.g., empty order book).

    Args:
        order_book: Binance GET /api/v3/depth response dict with keys
            ``"bids"`` and ``"asks"``, each a list of ``[price, quantity]``
            string pairs sorted best-first (bids descending, asks ascending).

    Returns:
        Dict with keys:
            - ``"bid_ask_spread"`` (float): Relative spread
              ``(best_ask - best_bid) / mid_price``.
            - ``"order_book_imbalance"`` (float): Volume imbalance across
              top 10 levels, in ``[-1, +1]``.
              Positive → more bid pressure; Negative → more ask pressure.

    Examples:
        >>> book = {
        ...     "bids": [["0.10050", "5000"], ["0.10040", "3000"]],
        ...     "asks": [["0.10060", "2000"], ["0.10070", "4000"]],
        ... }
        >>> features = compute_orderbook_features(book)
        >>> 0.0 < features["bid_ask_spread"] < 0.01
        True
    """
    bids: list[list[str]] = order_book.get("bids", [])
    asks: list[list[str]] = order_book.get("asks", [])

    if not bids or not asks:
        return {"bid_ask_spread": 0.0, "order_book_imbalance": 0.0}

    # ------------------------------------------------------------------
    # Best bid / ask prices
    # ------------------------------------------------------------------
    best_bid: float = float(bids[0][0])
    best_ask: float = float(asks[0][0])
    mid_price: float = (best_bid + best_ask) / 2.0

    # ------------------------------------------------------------------
    # Bid-ask spread (relative)
    # bid_ask_spread = (best_ask - best_bid) / mid_price
    # ------------------------------------------------------------------
    bid_ask_spread: float = (best_ask - best_bid) / (mid_price + _EPS)

    # ------------------------------------------------------------------
    # Order book imbalance across top 10 levels
    # imbalance = (bid_vol_10 - ask_vol_10) / (bid_vol_10 + ask_vol_10)
    # ------------------------------------------------------------------
    top10_bids = bids[:10]
    top10_asks = asks[:10]

    bid_vol_10: float = sum(float(b[1]) for b in top10_bids)
    ask_vol_10: float = sum(float(a[1]) for a in top10_asks)
    total_vol: float = bid_vol_10 + ask_vol_10

    if total_vol < _EPS:
        order_book_imbalance: float = 0.0
    else:
        order_book_imbalance = (bid_vol_10 - ask_vol_10) / total_vol

    return {
        "bid_ask_spread": bid_ask_spread,
        "order_book_imbalance": order_book_imbalance,
    }
