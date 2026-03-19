"""FastAPI dashboard application for DOGE Predictor.

Serves the trading UI on port 8080.  Provides four REST endpoints:

    GET /                       → HTML dashboard page
    GET /api/candles            → OHLCV data (local DB for 1h/4h/1d,
                                  Binance API proxy for 1m/5m/15m/30m)
    GET /api/signals            → recent BUY/SELL predictions from DB
    GET /api/status             → latest price, signal, and regime

Supports both SQLite (dev / --db-path mode) and PostgreSQL / TimescaleDB
(Docker production mode) via SQLAlchemy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from loguru import logger

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(title="DOGE Predictor Dashboard", docs_url=None, redoc_url=None)

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
    logger.info("Dashboard: PostgreSQL engine set to {}", db_url)


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
    """Serve the dashboard HTML page."""
    html_path = _STATIC_DIR / "dashboard.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/candles")
def get_candles(
    interval: str = Query("1h", description="Candle interval"),
    limit: int = Query(500, ge=1, le=1000, description="Number of candles"),
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
def get_status() -> dict[str, Any]:
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

    # Price 24h ago (24 candles back on 1h)
    prev_rows = _execute(
        """
        SELECT close
        FROM   ohlcv_1h
        WHERE  symbol = 'DOGEUSDT'
        ORDER  BY open_time DESC
        LIMIT  1 OFFSET 24
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
        prediction = {
            "time": int(p_time) // 1000,
            "signal": _direction_to_signal(p_dir),
            "confidence": round(float(p_conf or 0.5), 4),
            "lstm_prob": round(float(p_lstm or 0), 4),
            "xgb_prob": round(float(p_xgb or 0), 4),
            "regime": p_regime or "UNKNOWN",
            "horizon": p_horizon or "SHORT",
            "price_at_prediction": round(float(p_price or 0), 6),
        }

    return {
        "price": price,
        "change_pct": round(change_pct, 2),
        "last_candle_ms": int(latest[0]) if latest else 0,
        "accuracy_50": accuracy,
        "prediction": prediction,
    }
