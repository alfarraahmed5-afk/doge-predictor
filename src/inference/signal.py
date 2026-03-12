"""Signal assembly and thresholding for the DOGE inference pipeline.

Defines :class:`SignalEvent`, the typed output of every inference run, and
:class:`RiskFilterResult`, which captures the outcome of Step 9 risk filters.

All signal thresholds are regime-aware and loaded from
``config/regime_config.yaml`` — never hardcoded here.

Usage::

    from src.inference.signal import SignalEvent, RiskFilterResult

    event = SignalEvent(
        timestamp_ms=...,
        symbol="DOGEUSDT",
        regime="TRENDING_BULL",
        signal="BUY",
        ensemble_prob=0.75,
        confidence_threshold=0.62,
        position_size_multiplier=1.0,
        risk_filters_triggered=[],
        model_version="abc123",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# RiskFilterResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskFilterResult:
    """Outcome of the Step 9 risk filter chain.

    Records which hard override rules fired and whether the BUY signal was
    suppressed, and by how much the position size should be reduced.

    Attributes:
        buy_suppressed: True if any hard override rule cancelled a BUY signal.
        position_size_multiplier: Multiplicative factor to apply to the base
            position size (1.0 = full size; 0.5 = half; 0.7 = 30% reduction).
        triggered: Names of the risk rules that fired (order of application).
    """

    buy_suppressed: bool
    position_size_multiplier: float
    triggered: list[str] = field(default_factory=list)

    # dataclasses with mutable default fields need special handling; since
    # this is frozen, we override __post_init__ to convert any list to a
    # tuple-backed immutable list alias — but for simplicity we keep list
    # and document that callers should not mutate it.


# ---------------------------------------------------------------------------
# SignalEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalEvent:
    """Complete output of a single inference run.

    Emitted by :meth:`~src.inference.engine.InferenceEngine.run` after all
    12 pipeline steps have been completed.  Consumed by the RL predictor,
    risk management layer, and any registered ``on_signal`` callbacks.

    Attributes:
        timestamp_ms: Candle open_time (UTC epoch milliseconds) for which the
            prediction was generated.
        symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
        regime: Active market regime label at prediction time (one of the five
            canonical regime strings from CLAUDE.md Section 6).
        signal: Final trade direction after risk filter application.
            One of ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        ensemble_prob: Raw ensemble model P(up) before threshold comparison.
            In ``[0.0, 1.0]``.
        confidence_threshold: Regime-adjusted threshold used for the signal
            decision (loaded from ``config/regime_config.yaml``; never
            hardcoded).
        position_size_multiplier: Net position size multiplier after all risk
            filter reductions are applied.  In ``(0.0, 1.0]``.
        risk_filters_triggered: Names of every Step 9 risk rule that fired
            during this inference run, in application order.
        model_version: MLflow run-id or model identifier string stored
            alongside the prediction record for auditability.
        lstm_prob: Raw LSTM P(up) output, logged for auditability.
        xgb_prob: Raw XGBoost P(up) output, logged for auditability.
        regime_encoded: Numeric ordinal encoding of the regime (0–4), used as
            input to the ensemble meta-learner.
        open_time: Alias for ``timestamp_ms`` (candle open_time).
        close_price: Close price at the time of prediction (for RL logging).
    """

    timestamp_ms: int
    symbol: str
    regime: str
    signal: Literal["BUY", "SELL", "HOLD"]
    ensemble_prob: float
    confidence_threshold: float
    position_size_multiplier: float
    risk_filters_triggered: list[str]
    model_version: str
    lstm_prob: float = 0.5
    xgb_prob: float = 0.5
    regime_encoded: float = 0.0
    open_time: int = 0
    close_price: float = 0.0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "SignalEvent",
    "RiskFilterResult",
]
