"""Centralised Prometheus metric definitions for doge_predictor.

All metrics are registered once at module import time.  Import this module
wherever you need to update metrics — the same registry objects are returned.

If ``prometheus_client`` is not installed, every metric is replaced with a
no-op stub so the rest of the codebase never needs to guard against
``ImportError``.

Metrics defined here
---------------------

Inference pipeline
~~~~~~~~~~~~~~~~~~
* ``doge_inference_latency_seconds`` — Histogram, by ``signal`` label
* ``doge_inference_errors_total``    — Counter, by ``step`` label
* ``doge_signals_total``             — Counter, by ``signal`` and ``regime``
* ``doge_prediction_count_total``    — Counter, by ``horizon`` and ``regime``

Data freshness
~~~~~~~~~~~~~~
* ``doge_feature_freshness_seconds`` — Gauge (age of last closed candle)
* ``doge_last_candle_age_seconds``   — Gauge (alias used by serve.py)
* ``doge_ws_connected``              — Gauge (1 = connected, 0 = disconnected)

Feature values (live single-value gauges)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``doge_btc_corr_24h``             — Gauge
* ``doge_volume_ratio``             — Gauge
* ``doge_funding_rate_zscore``      — Gauge

Regime
~~~~~~
* ``doge_current_regime``          — Gauge with ``regime`` label; value = 1
                                     for the active regime, 0 for others.

Risk / equity
~~~~~~~~~~~~~
* ``doge_equity_drawdown_pct``     — Gauge (current drawdown as a fraction)

Usage::

    from src.monitoring.prometheus_metrics import (
        INFERENCE_LATENCY, SIGNALS_TOTAL, CURRENT_REGIME, VOLUME_RATIO,
        record_regime,
    )

    with INFERENCE_LATENCY.labels(signal="BUY").time():
        ...  # inference code

    record_regime("TRENDING_BULL")
    VOLUME_RATIO.set(2.5)
"""

from __future__ import annotations

from loguru import logger

__all__ = [
    "INFERENCE_LATENCY",
    "INFERENCE_ERRORS",
    "SIGNALS_TOTAL",
    "PREDICTION_COUNT",
    "FEATURE_FRESHNESS",
    "CANDLE_AGE",
    "WS_CONNECTED",
    "BTC_CORR_24H",
    "VOLUME_RATIO",
    "FUNDING_RATE_ZSCORE",
    "CURRENT_REGIME",
    "EQUITY_DRAWDOWN",
    "record_regime",
]

# ---------------------------------------------------------------------------
# Stub fallback so imports never fail
# ---------------------------------------------------------------------------


class _Stub:
    """No-op stub for all Prometheus metric types."""

    def labels(self, **_kw: object) -> "_Stub":
        return self

    def observe(self, *_a: object, **_kw: object) -> None:
        pass

    def inc(self, *_a: object, **_kw: object) -> None:
        pass

    def set(self, *_a: object, **_kw: object) -> None:
        pass

    def time(self) -> "_Stub":
        return self

    def __enter__(self) -> "_Stub":
        return self

    def __exit__(self, *_: object) -> None:
        pass


_STUB = _Stub()

# ---------------------------------------------------------------------------
# Metric registry — registered at module level (singleton per process)
# ---------------------------------------------------------------------------

_ALL_REGIMES = (
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
)

try:
    from prometheus_client import Counter, Gauge, Histogram

    INFERENCE_LATENCY: object = Histogram(
        "doge_inference_latency_seconds",
        "End-to-end inference latency in seconds",
        ["signal"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    INFERENCE_ERRORS: object = Counter(
        "doge_inference_errors_total",
        "Inference pipeline error count by step",
        ["step"],
    )

    SIGNALS_TOTAL: object = Counter(
        "doge_signals_total",
        "Total signals emitted by doge_predictor",
        ["signal", "regime"],
    )

    PREDICTION_COUNT: object = Counter(
        "doge_prediction_count_total",
        "Total predictions logged by horizon and regime",
        ["horizon", "regime"],
    )

    FEATURE_FRESHNESS: object = Gauge(
        "doge_feature_freshness_seconds",
        "Age of the most recently closed candle used for inference (seconds)",
    )

    CANDLE_AGE: object = Gauge(
        "doge_last_candle_age_seconds",
        "Age of the last processed candle in seconds (serve.py compat alias)",
    )

    WS_CONNECTED: object = Gauge(
        "doge_ws_connected",
        "WebSocket connection status (1=connected, 0=disconnected)",
    )

    BTC_CORR_24H: object = Gauge(
        "doge_btc_corr_24h",
        "Rolling 24h DOGE-BTC log-return correlation",
    )

    VOLUME_RATIO: object = Gauge(
        "doge_volume_ratio",
        "Current volume / rolling 20-period mean volume",
    )

    FUNDING_RATE_ZSCORE: object = Gauge(
        "doge_funding_rate_zscore",
        "90-period z-score of the DOGEUSDT 8h funding rate",
    )

    CURRENT_REGIME: object = Gauge(
        "doge_current_regime",
        "Active market regime (1 for current, 0 for others)",
        ["regime"],
    )
    # Initialise all regime labels to 0
    for _regime in _ALL_REGIMES:
        CURRENT_REGIME.labels(regime=_regime).set(0)  # type: ignore[union-attr]

    EQUITY_DRAWDOWN: object = Gauge(
        "doge_equity_drawdown_pct",
        "Current equity drawdown as a fraction (0–1)",
    )

    logger.debug("prometheus_metrics: all metrics registered successfully")

except ImportError:
    logger.warning("prometheus_client not installed — metrics stubs active")
    INFERENCE_LATENCY = _STUB
    INFERENCE_ERRORS = _STUB
    SIGNALS_TOTAL = _STUB
    PREDICTION_COUNT = _STUB
    FEATURE_FRESHNESS = _STUB
    CANDLE_AGE = _STUB
    WS_CONNECTED = _STUB
    BTC_CORR_24H = _STUB
    VOLUME_RATIO = _STUB
    FUNDING_RATE_ZSCORE = _STUB
    CURRENT_REGIME = _STUB
    EQUITY_DRAWDOWN = _STUB
except ValueError:
    # Metrics already registered (e.g. during pytest re-imports)
    logger.debug("prometheus_metrics: metrics already registered in this process")
    # Re-import from the existing registry
    from prometheus_client import REGISTRY
    def _get(name: str) -> object:
        try:
            return REGISTRY._names_to_collectors[name]  # type: ignore[attr-defined]
        except (AttributeError, KeyError):
            return _STUB

    INFERENCE_LATENCY = _get("doge_inference_latency_seconds")
    INFERENCE_ERRORS  = _get("doge_inference_errors_total")
    SIGNALS_TOTAL     = _get("doge_signals_total")
    PREDICTION_COUNT  = _get("doge_prediction_count_total")
    FEATURE_FRESHNESS = _get("doge_feature_freshness_seconds")
    CANDLE_AGE        = _get("doge_last_candle_age_seconds")
    WS_CONNECTED      = _get("doge_ws_connected")
    BTC_CORR_24H      = _get("doge_btc_corr_24h")
    VOLUME_RATIO      = _get("doge_volume_ratio")
    FUNDING_RATE_ZSCORE = _get("doge_funding_rate_zscore")
    CURRENT_REGIME    = _get("doge_current_regime")
    EQUITY_DRAWDOWN   = _get("doge_equity_drawdown_pct")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def record_regime(regime: str) -> None:
    """Set ``doge_current_regime`` gauge: 1 for the active regime, 0 for all others.

    Args:
        regime: The currently active regime label (must be one of the 5
            canonical labels).  Unknown labels are silently ignored after a
            warning.
    """
    if regime not in _ALL_REGIMES:
        logger.warning("record_regime: unknown regime '{}' — skipping metric update", regime)
        return
    for r in _ALL_REGIMES:
        try:
            CURRENT_REGIME.labels(regime=r).set(1 if r == regime else 0)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
