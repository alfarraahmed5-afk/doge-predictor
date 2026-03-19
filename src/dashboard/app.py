"""FastAPI dashboard application for DOGE Predictor.

Serves the trading UI on port 8090.  Provides six REST endpoints:

    GET /                       → HTML dashboard page
    GET /api/candles            → OHLCV data (local DB for 1h/4h/1d,
                                  Binance API proxy for 1m/5m/15m/30m)
    GET /api/signals            → recent BUY/SELL predictions from DB
    GET /api/status             → latest price, signal, and regime
    GET /api/features           → current technical indicator values
    GET /api/multi_horizon      → latest prediction per time horizon

Supports both SQLite (dev / --db-path mode) and PostgreSQL / TimescaleDB
(Docker production mode) via SQLAlchemy.
"""

from __future__ import annotations

import hmac
import math
import os
import re
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger

# ---------------------------------------------------------------------------
# Optional API-key authentication
# ---------------------------------------------------------------------------

# If DASHBOARD_API_KEY is set in the environment, all /api/* endpoints
# require the caller to present it as an X-API-Key header.
# Leave the env var unset to run in open mode (backwards-compatible).
_DASHBOARD_API_KEY: str = os.environ.get("DASHBOARD_API_KEY", "")
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """FastAPI dependency that enforces X-API-Key when configured.

    Uses :func:`hmac.compare_digest` to prevent timing-based key inference.
    No-ops when ``DASHBOARD_API_KEY`` is not set.
    """
    if not _DASHBOARD_API_KEY:
        return  # open mode — no key required
    if not api_key or not hmac.compare_digest(api_key, _DASHBOARD_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject security-related HTTP response headers on every response."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "connect-src 'self' wss://stream.binance.com:9443 "
            "https://api.binance.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        return response


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(title="DOGE Predictor Dashboard", docs_url=None, redoc_url=None)

# CORS — restrict to same-origin by default; add origins via environment if needed
_CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("DASHBOARD_CORS_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS or [],  # empty = deny all cross-origin requests
    allow_methods=["GET"],
    allow_headers=["X-API-Key"],
)
app.add_middleware(_SecurityHeadersMiddleware)

_ENGINE: Optional[Any] = None          # SQLAlchemy engine
_STATIC_DIR: Path = Path(__file__).parent / "static"

# Binance public REST — no API key required for klines
_BINANCE_BASE: str = "https://api.binance.com/api/v3"
_BINANCE_TIMEOUT: int = 8  # seconds

# Map UI interval labels → DB table names (stored in local DB)
_DB_INTERVAL_MAP: dict[str, str] = {
    "1h": "ohlcv_1h",
    "4h": "ohlcv_4h",
    "1d": "ohlcv_1d",
}

# Regime confidence thresholds (mirrors regime_config.yaml)
_REGIME_THRESHOLDS: dict[str, float] = {
    "TRENDING_BULL": 0.62,
    "TRENDING_BEAR": 0.62,
    "RANGING_HIGH_VOL": 0.65,
    "RANGING_LOW_VOL": 0.70,
    "DECOUPLED": 0.72,
    "UNKNOWN": 0.65,
}

# Horizon labels → human-readable time window
_HORIZON_LABELS: dict[str, str] = {
    "SHORT": "4 hours",
    "MEDIUM": "24 hours",
    "LONG": "7 days",
    "MACRO": "30 days",
}

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_dashboard(db_path: Path) -> None:
    """Register the SQLite database path used by all endpoints.

    Args:
        db_path: Absolute path to ``doge_data.db``.
    """
    global _ENGINE
    from sqlalchemy import create_engine  # noqa: PLC0415

    _ENGINE = create_engine(f"sqlite:///{db_path}", future=True)
    logger.info("Dashboard: SQLite engine set to {}", db_path)


def init_dashboard_pg(db_url: str) -> None:
    """Register a PostgreSQL/TimescaleDB engine for all endpoints.

    Args:
        db_url: SQLAlchemy database URL, e.g.
            ``postgresql+psycopg2://user:pass@host:5432/dbname``.
    """
    global _ENGINE
    from sqlalchemy import create_engine  # noqa: PLC0415

    _ENGINE = create_engine(db_url, future=True, pool_pre_ping=True)
    # Mask password in log to avoid credential exposure in log files.
    _safe_url = re.sub(r":([^@/:]+)@", ":***@", db_url)
    logger.info("Dashboard: PostgreSQL engine set to {}", _safe_url)


def _execute(sql: str, params: tuple = ()) -> list[tuple]:
    """Run a read-only SQL query and return all rows.

    Args:
        sql: SQL statement (``?`` placeholders for SQLite,
            ``:param`` or ``%s`` for PostgreSQL — we use ``?`` style
            and replace for PostgreSQL automatically).
        params: Positional query parameters.

    Returns:
        List of row tuples, or empty list on error / uninitialised engine.
    """
    if _ENGINE is None:
        return []
    from sqlalchemy import text  # noqa: PLC0415

    # SQLAlchemy text() uses :name parameters; for positional we wrap
    # each ``?`` into a named token ``p0``, ``p1``, … for portability.
    named_sql = sql
    named_params: dict[str, Any] = {}
    idx = 0
    while "?" in named_sql:
        token = f"p{idx}"
        named_sql = named_sql.replace("?", f":{token}", 1)
        named_params[token] = params[idx]
        idx += 1

    try:
        with _ENGINE.connect() as conn:
            result = conn.execute(text(named_sql), named_params)
            return list(result.fetchall())
    except Exception as exc:
        logger.warning("Dashboard DB query failed: {}", exc)
        return []


def _table_exists(table_name: str) -> bool:
    """Return True if *table_name* exists in the connected database.

    Works for both SQLite and PostgreSQL.

    Args:
        table_name: Unquoted table name.

    Returns:
        True when the table is present.
    """
    if _ENGINE is None:
        return False
    dialect = _ENGINE.dialect.name
    if dialect == "sqlite":
        rows = _execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
    else:
        # PostgreSQL / TimescaleDB — use information_schema
        rows = _execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ?",
            (table_name,),
        )
    return bool(rows)


# ---------------------------------------------------------------------------
# Technical indicator helpers (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------


def _compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Compute RSI using Wilder's smoothing method."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
    gains = [max(0.0, d) for d in deltas[-period:]]
    losses = [max(0.0, -d) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_ema(closes: list[float], period: int) -> Optional[float]:
    """Compute exponential moving average (last value only)."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = c * k + ema * (1.0 - k)
    return ema


def _compute_log_return_corr(a: list[float], b: list[float]) -> Optional[float]:
    """Pearson correlation of log returns between two price series."""
    n = min(len(a), len(b))
    if n < 3:
        return None
    a = a[-n:]
    b = b[-n:]
    lr_a: list[float] = []
    lr_b: list[float] = []
    for i in range(1, n):
        if a[i - 1] > 0 and a[i] > 0 and b[i - 1] > 0 and b[i] > 0:
            lr_a.append(math.log(a[i] / a[i - 1]))
            lr_b.append(math.log(b[i] / b[i - 1]))
    m = len(lr_a)
    if m < 3:
        return None
    ma = sum(lr_a) / m
    mb = sum(lr_b) / m
    cov = sum((lr_a[i] - ma) * (lr_b[i] - mb) for i in range(m)) / m
    sa = (sum((x - ma) ** 2 for x in lr_a) / m) ** 0.5
    sb = (sum((x - mb) ** 2 for x in lr_b) / m) ** 0.5
    if sa < 1e-12 or sb < 1e-12:
        return None
    return max(-1.0, min(1.0, cov / (sa * sb)))


def _compute_atr_norm(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> Optional[float]:
    """Compute ATR normalised by current close price."""
    if len(closes) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hc, lc))
    atr = sum(trs[-period:]) / period
    return atr / closes[-1] if closes[-1] > 0 else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direction_to_signal(d: int | None) -> str:
    """Convert stored predicted_direction int to display string."""
    if d == 1:
        return "BUY"
    if d == -1:
        return "SELL"
    return "HOLD"


def _fetch_binance_klines(interval: str, limit: int) -> list[dict[str, Any]]:
    """Fetch OHLCV candles from Binance public REST API.

    Used for sub-hourly intervals (1m, 5m, 15m, 30m) that are not
    stored in the local database.

    Args:
        interval: Binance interval string e.g. ``"1m"``, ``"5m"``.
        limit: Number of candles to fetch (max 1000).

    Returns:
        List of candle dicts with keys: time, open, high, low, close, volume.
    """
    try:
        resp = requests.get(
            f"{_BINANCE_BASE}/klines",
            params={"symbol": "DOGEUSDT", "interval": interval, "limit": limit},
            timeout=_BINANCE_TIMEOUT,
        )
        resp.raise_for_status()
        raw: list[list] = resp.json()
        return [
            {
                "time": int(row[0]) // 1000,  # ms → seconds for TradingView
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in raw
        ]
    except Exception as exc:
        logger.warning("Dashboard: Binance kline fetch failed ({}): {}", interval, exc)
        return []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the dashboard HTML page.

    If ``DASHBOARD_API_KEY`` is configured, injects it into the page as a
    ``window._DASH_KEY`` JavaScript variable so the dashboard's fetch calls
    can include the required ``X-API-Key`` header automatically.
    """
    html_path = _STATIC_DIR / "dashboard.html"
    html = html_path.read_text(encoding="utf-8")
    if _DASHBOARD_API_KEY:
        # Inject the key so client-side fetch calls can authenticate.
        # The key is only visible to users who can already reach the page.
        inject = (
            f'<script>window._DASH_KEY="{_DASHBOARD_API_KEY}";</script>'
        )
        html = html.replace("</head>", inject + "\n</head>", 1)
    return html


_ALLOWED_INTERVALS: frozenset[str] = frozenset(
    {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
)


@app.get("/api/candles")
def get_candles(
    interval: str = Query("1h", description="Candle interval"),
    limit: int = Query(500, ge=1, le=1000, description="Number of candles"),
    _: None = Depends(_require_api_key),
) -> list[dict[str, Any]]:
    """Return OHLCV candle data for DOGEUSDT.

    For sub-hourly intervals (1m, 5m, 15m, 30m) data is fetched live from
    Binance.  For 1h, 4h, 1d the local database is used (full history).

    Args:
        interval: One of ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``4h``, ``1d``.
        limit: Max candles to return.

    Returns:
        List of ``{time, open, high, low, close, volume}`` dicts.
        ``time`` is Unix epoch seconds (required by TradingView Lightweight Charts).
    """
    if interval not in _ALLOWED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"Unsupported interval: {interval}")

    if interval not in _DB_INTERVAL_MAP:
        # Sub-hourly: proxy to Binance public API
        return _fetch_binance_klines(interval, limit)

    if _ENGINE is None:
        # DB not yet initialised — fall back to Binance for all intervals
        binance_interval = {"1h": "1h", "4h": "4h", "1d": "1d"}.get(interval, interval)
        return _fetch_binance_klines(binance_interval, limit)

    table = _DB_INTERVAL_MAP[interval]
    rows = _execute(
        f"""
        SELECT open_time, open, high, low, close, volume
        FROM   {table}
        WHERE  symbol = 'DOGEUSDT'
        ORDER  BY open_time DESC
        LIMIT  ?
        """,
        (limit,),
    )

    if not rows:
        # DB query returned nothing — try Binance as fallback
        return _fetch_binance_klines(interval, limit)

    return [
        {
            "time": int(row[0]) // 1000,
            "open": round(float(row[1]), 6),
            "high": round(float(row[2]), 6),
            "low": round(float(row[3]), 6),
            "close": round(float(row[4]), 6),
            "volume": round(float(row[5]), 2),
        }
        for row in reversed(rows)
    ]


@app.get("/api/signals")
def get_signals(
    limit: int = Query(100, ge=1, le=500, description="Number of signals"),
    _: None = Depends(_require_api_key),
) -> list[dict[str, Any]]:
    """Return recent BUY/SELL/HOLD predictions from the prediction store.

    Args:
        limit: Max rows to return.

    Returns:
        List of signal dicts, newest first.
    """
    if not _table_exists("doge_predictions"):
        return []

    rows = _execute(
        """
        SELECT open_time, predicted_direction, confidence_score,
               lstm_prob, xgb_prob, regime_label, horizon_label,
               price_at_prediction, reward_score, direction_correct
        FROM   doge_predictions
        WHERE  predicted_direction IN (-1, 0, 1)
        ORDER  BY open_time DESC
        LIMIT  ?
        """,
        (limit,),
    )

    out = []
    for (
        open_time,
        direction,
        conf,
        lstm,
        xgb,
        regime,
        horizon,
        price,
        reward,
        correct,
    ) in rows:
        out.append(
            {
                "time": int(open_time) // 1000,
                "signal": _direction_to_signal(direction),
                "confidence": round(float(conf or 0.5), 4),
                "lstm_prob": round(float(lstm or 0), 4),
                "xgb_prob": round(float(xgb or 0), 4),
                "regime": regime or "UNKNOWN",
                "horizon": horizon or "SHORT",
                "price": round(float(price or 0), 6),
                "reward": round(float(reward), 4) if reward is not None else None,
                "correct": bool(correct) if correct is not None else None,
            }
        )
    return out


@app.get("/api/status")
def get_status(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    """Return current price, 24h change, and latest prediction.

    Returns:
        Dict with keys: price, change_pct, last_candle_ms, accuracy_50, prediction.
    """
    # Latest 1h close price
    latest_rows = _execute(
        """
        SELECT open_time, close, volume
        FROM   ohlcv_1h
        WHERE  symbol = 'DOGEUSDT'
        ORDER  BY open_time DESC
        LIMIT  1
        """
    )
    latest = latest_rows[0] if latest_rows else None

    # Price 24h ago (24 candles back on 1h) — use subquery for PG/SQLite compat
    prev_rows = _execute(
        """
        SELECT close FROM ohlcv_1h
        WHERE symbol = 'DOGEUSDT'
          AND open_time < (
              SELECT open_time FROM ohlcv_1h
              WHERE symbol = 'DOGEUSDT'
              ORDER BY open_time DESC LIMIT 1
          ) - 86400000
        ORDER BY open_time DESC
        LIMIT 1
        """
    )
    prev_24h = prev_rows[0] if prev_rows else None

    # Latest prediction
    pred_row = None
    if _table_exists("doge_predictions"):
        pred_rows = _execute(
            """
            SELECT open_time, predicted_direction, confidence_score,
                   lstm_prob, xgb_prob, regime_label, horizon_label,
                   price_at_prediction
            FROM   doge_predictions
            ORDER  BY open_time DESC
            LIMIT  1
            """
        )
        pred_row = pred_rows[0] if pred_rows else None

    # Accuracy stats (last 50 verified)
    accuracy: float | None = None
    if _table_exists("doge_predictions"):
        acc_rows = _execute(
            """
            SELECT AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END)
            FROM   (
                SELECT direction_correct
                FROM   doge_predictions
                WHERE  direction_correct IS NOT NULL
                ORDER  BY open_time DESC
                LIMIT  50
            ) sub
            """
        )
        if acc_rows and acc_rows[0][0] is not None:
            accuracy = round(float(acc_rows[0][0]) * 100, 1)

    price = float(latest[1]) if latest else 0.0
    prev_price = float(prev_24h[0]) if prev_24h else price
    change_pct = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0.0

    prediction: dict[str, Any] | None = None
    if pred_row:
        (
            p_time,
            p_dir,
            p_conf,
            p_lstm,
            p_xgb,
            p_regime,
            p_horizon,
            p_price,
        ) = pred_row
        regime_key = p_regime or "UNKNOWN"
        threshold = _REGIME_THRESHOLDS.get(regime_key, 0.65)
        prediction = {
            "time": int(p_time) // 1000,
            "signal": _direction_to_signal(p_dir),
            "confidence": round(float(p_conf or 0.5), 4),
            "lstm_prob": round(float(p_lstm or 0), 4),
            "xgb_prob": round(float(p_xgb or 0), 4),
            "regime": regime_key,
            "horizon": p_horizon or "SHORT",
            "price_at_prediction": round(float(p_price or 0), 6),
            "threshold": threshold,
        }

    return {
        "price": price,
        "change_pct": round(change_pct, 2),
        "last_candle_ms": int(latest[0]) if latest else 0,
        "accuracy_50": accuracy,
        "prediction": prediction,
    }


@app.get("/api/features")
def get_features(_: None = Depends(_require_api_key)) -> dict[str, Any]:
    """Return current technical indicator values computed from the DB.

    Fetches the latest 210 1h candles for DOGEUSDT + BTCUSDT and computes
    key indicators for display on the dashboard.

    Returns:
        Dict with indicator values, or empty dict if DB unavailable.
    """
    # Fetch last 210 DOGE 1h rows (need 200 for EMA200 + buffer)
    doge_rows = _execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM   ohlcv_1h
        WHERE  symbol = 'DOGEUSDT'
        ORDER  BY open_time DESC
        LIMIT  210
        """
    )
    if not doge_rows or len(doge_rows) < 30:
        return {}

    # Reverse so oldest → newest (chronological order)
    doge_rows = list(reversed(doge_rows))
    closes  = [float(r[4]) for r in doge_rows]
    highs   = [float(r[2]) for r in doge_rows]
    lows    = [float(r[3]) for r in doge_rows]
    volumes = [float(r[5]) for r in doge_rows]

    # BTC rows for correlation
    btc_rows = _execute(
        """
        SELECT close FROM ohlcv_1h
        WHERE  symbol = 'BTCUSDT'
        ORDER  BY open_time DESC
        LIMIT  210
        """
    )
    btc_closes: list[float] = (
        [float(r[0]) for r in reversed(btc_rows)] if btc_rows else []
    )

    # Latest funding rate
    funding_rows = _execute(
        """
        SELECT funding_rate FROM funding_rates
        WHERE  symbol = 'DOGEUSDT'
        ORDER  BY timestamp_ms DESC
        LIMIT  1
        """
    )

    # ── RSI 14 ──────────────────────────────────────────────────────────────
    rsi = _compute_rsi(closes, 14)

    # ── Volume ratio (current / 20-period mean) ──────────────────────────────
    vol_ma20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    volume_ratio = (volumes[-1] / vol_ma20) if vol_ma20 and vol_ma20 > 0 else None
    volume_spike = bool(volume_ratio and volume_ratio >= 3.0)

    # ── EMA 20 / EMA 50 / EMA 200 ────────────────────────────────────────────
    ema20  = _compute_ema(closes, 20)
    ema50  = _compute_ema(closes, 50)
    ema200 = _compute_ema(closes, 200)
    price_vs_ema50_pct = (
        ((closes[-1] - ema50) / ema50 * 100) if ema50 and ema50 > 0 else None
    )

    # ── EMA trend (are EMAs stacked bullishly?) ──────────────────────────────
    ema_trend: Optional[str] = None
    if ema20 and ema50 and ema200:
        if ema20 > ema50 > ema200:
            ema_trend = "BULLISH"
        elif ema20 < ema50 < ema200:
            ema_trend = "BEARISH"
        else:
            ema_trend = "MIXED"

    # ── ATR 14 (normalised by close) ─────────────────────────────────────────
    atr_norm_pct = _compute_atr_norm(highs, lows, closes, 14)
    if atr_norm_pct is not None:
        atr_norm_pct = round(atr_norm_pct * 100, 3)

    # ── BTC-DOGE 24h log-return correlation ──────────────────────────────────
    btc_corr_24h: Optional[float] = None
    if len(btc_closes) >= 25 and len(closes) >= 25:
        btc_corr_24h = _compute_log_return_corr(closes[-25:], btc_closes[-25:])
        if btc_corr_24h is not None:
            btc_corr_24h = round(btc_corr_24h, 3)

    # ── Funding rate ─────────────────────────────────────────────────────────
    funding_rate: Optional[float] = (
        float(funding_rows[0][0]) if funding_rows else None
    )
    funding_extreme_long  = bool(funding_rate is not None and funding_rate > 0.001)
    funding_extreme_short = bool(funding_rate is not None and funding_rate < -0.0005)

    # ── Distance to nearest round number ─────────────────────────────────────
    round_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
    close_price = closes[-1]
    nearest_round: Optional[float] = None
    min_dist = float("inf")
    for lvl in round_levels:
        d = abs(close_price - lvl)
        if d < min_dist:
            min_dist = d
            nearest_round = lvl
    distance_to_round_pct = (
        (min_dist / nearest_round * 100) if nearest_round and nearest_round > 0 else None
    )
    at_round_number = bool(distance_to_round_pct is not None and distance_to_round_pct < 1.0)

    return {
        "close": round(close_price, 6),
        "rsi_14": round(rsi, 1) if rsi is not None else None,
        "rsi_overbought": bool(rsi is not None and rsi > 70),
        "rsi_oversold": bool(rsi is not None and rsi < 30),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "volume_spike": volume_spike,
        "ema20": round(ema20, 6) if ema20 else None,
        "ema50": round(ema50, 6) if ema50 else None,
        "ema200": round(ema200, 6) if ema200 else None,
        "ema_trend": ema_trend,
        "price_vs_ema50_pct": round(price_vs_ema50_pct, 2) if price_vs_ema50_pct is not None else None,
        "atr_norm_pct": atr_norm_pct,
        "btc_corr_24h": btc_corr_24h,
        "decoupled": bool(btc_corr_24h is not None and btc_corr_24h < 0.30),
        "funding_rate": round(funding_rate * 10000, 4) if funding_rate is not None else None,  # basis points
        "funding_extreme_long": funding_extreme_long,
        "funding_extreme_short": funding_extreme_short,
        "nearest_round": nearest_round,
        "distance_to_round_pct": round(distance_to_round_pct, 2) if distance_to_round_pct is not None else None,
        "at_round_number": at_round_number,
    }


@app.get("/api/multi_horizon")
def get_multi_horizon(_: None = Depends(_require_api_key)) -> list[dict[str, Any]]:
    """Return the latest prediction for each horizon (SHORT / MEDIUM / LONG / MACRO).

    The inference engine writes SHORT predictions every hour.  MEDIUM/LONG/MACRO
    predictions are generated by the MultiHorizonPredictor once the curriculum
    manager enables them.  If no prediction exists for a horizon, that entry is
    omitted from the result.

    Returns:
        List of prediction dicts, one per available horizon, ordered SHORT→MACRO.
    """
    if not _table_exists("doge_predictions"):
        return []

    horizons = ["SHORT", "MEDIUM", "LONG", "MACRO"]
    horizon_windows = {"SHORT": "4h", "MEDIUM": "24h", "LONG": "7d", "MACRO": "30d"}
    out: list[dict[str, Any]] = []

    for hz in horizons:
        rows = _execute(
            """
            SELECT open_time, predicted_direction, confidence_score,
                   lstm_prob, xgb_prob, regime_label, price_at_prediction,
                   actual_price, direction_correct
            FROM   doge_predictions
            WHERE  horizon_label = ?
            ORDER  BY open_time DESC
            LIMIT  1
            """,
            (hz,),
        )
        if not rows:
            continue
        (
            open_time,
            direction,
            conf,
            lstm,
            xgb,
            regime,
            price_at,
            actual_price,
            correct,
        ) = rows[0]

        regime_key = regime or "UNKNOWN"
        threshold  = _REGIME_THRESHOLDS.get(regime_key, 0.65)
        conf_val   = float(conf or 0.5)

        # Ensemble probability = conf (since conf = 0.5 + |prob - 0.5|,
        # and direction tells us which side, we can recover raw prob).
        # For display, use conf directly as the model's "up probability"
        # when direction == 1, else (1 - conf) for the displayed bar.
        raw_prob_up: float
        if direction == 1:
            raw_prob_up = conf_val
        elif direction == -1:
            raw_prob_up = 1.0 - conf_val
        else:
            raw_prob_up = 0.5  # HOLD — near centre

        out.append(
            {
                "horizon": hz,
                "window": horizon_windows[hz],
                "signal": _direction_to_signal(direction),
                "confidence": round(conf_val, 4),
                "prob_up": round(raw_prob_up, 4),
                "lstm_prob": round(float(lstm or 0), 4),
                "xgb_prob": round(float(xgb or 0), 4),
                "regime": regime_key,
                "threshold": threshold,
                "price_at": round(float(price_at or 0), 6),
                "actual_price": round(float(actual_price), 6) if actual_price is not None else None,
                "correct": bool(correct) if correct is not None else None,
                "time": int(open_time) // 1000,
            }
        )

    return out
