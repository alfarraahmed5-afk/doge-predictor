"""Real-time regime change detection for DOGEUSDT.

``RegimeChangeDetector`` wraps the stateless
:meth:`~src.regimes.classifier.DogeRegimeClassifier.detect_transition` helper
and enriches each transition with market context, risk flags, and structured
logging — ready for downstream alerting and monitoring consumers.

Usage::

    from src.regimes.detector import RegimeChangeDetector

    detector = RegimeChangeDetector()
    event = detector.detect("RANGING_LOW_VOL", "DECOUPLED",
                            btc_corr=0.15, atr_norm=0.004)
    if event:
        print(event.is_critical)   # True — involves DECOUPLED
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.regimes.classifier import REGIME_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID: frozenset[str] = frozenset(REGIME_LABELS)


# ---------------------------------------------------------------------------
# RegimeChangeEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeChangeEvent:
    """Immutable record of a single regime transition.

    Attributes:
        from_regime: Regime label before the transition.
        to_regime:   Regime label after the transition.
        changed_at:  UTC epoch-ms of the candle open_time at which the
                     transition was detected. Defaults to 0 when the
                     caller does not supply a timestamp (e.g., tests).
        btc_corr:    BTC-DOGE 24h log-return correlation at transition.
        atr_norm:    ATR as a fraction of price at transition.
        is_critical: ``True`` when ``DECOUPLED`` is the origin **or**
                     destination regime.  Critical transitions require
                     immediate attention from risk management.
    """

    from_regime: str
    to_regime: str
    changed_at: int
    btc_corr: float
    atr_norm: float
    is_critical: bool


# ---------------------------------------------------------------------------
# RegimeChangeDetector
# ---------------------------------------------------------------------------


class RegimeChangeDetector:
    """Detect and log regime transitions on every new candle.

    This class is intentionally stateless between calls; the caller is
    responsible for tracking the previous regime label.

    Args:
        None — all configuration is implicit (thresholds live in
        ``config/regime_config.yaml``; see
        :class:`~src.regimes.classifier.DogeRegimeClassifier`).
    """

    def detect(
        self,
        prev_label: str,
        curr_label: str,
        btc_corr: float,
        atr_norm: float,
        changed_at: int = 0,
    ) -> Optional[RegimeChangeEvent]:
        """Evaluate a potential regime transition and emit a structured event.

        Returns ``None`` when the regime is unchanged, or a
        :class:`RegimeChangeEvent` when a transition is detected.

        Args:
            prev_label:  Regime label from the previous candle.
            curr_label:  Regime label from the current candle.
            btc_corr:    Current BTC-DOGE 24h log-return correlation.
            atr_norm:    Current ATR / close (dimensionless).
            changed_at:  UTC epoch-ms of the transition candle. Defaults to 0.

        Returns:
            :class:`RegimeChangeEvent` if ``prev_label != curr_label``,
            otherwise ``None``.

        Raises:
            ValueError: If either *prev_label* or *curr_label* is not a
                valid regime label string.
        """
        self._validate_labels(prev_label, curr_label)

        if prev_label == curr_label:
            return None

        is_critical: bool = (
            prev_label == "DECOUPLED" or curr_label == "DECOUPLED"
        )

        event = RegimeChangeEvent(
            from_regime=prev_label,
            to_regime=curr_label,
            changed_at=changed_at,
            btc_corr=btc_corr,
            atr_norm=atr_norm,
            is_critical=is_critical,
        )

        log_fn = logger.warning if is_critical else logger.info
        log_fn(
            "Regime transition: {} -> {} | critical={} | btc_corr={:.4f} "
            "| atr_norm={:.4f} | changed_at={}",
            prev_label,
            curr_label,
            is_critical,
            btc_corr,
            atr_norm,
            changed_at,
        )

        return event

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_labels(prev_label: str, curr_label: str) -> None:
        """Raise ValueError for any unrecognised regime label.

        Args:
            prev_label: Previous regime label string.
            curr_label: Current regime label string.

        Raises:
            ValueError: If either label is not in :data:`REGIME_LABELS`.
        """
        for label in (prev_label, curr_label):
            if label not in _VALID:
                raise ValueError(
                    f"Unknown regime label {label!r}. "
                    f"Valid labels: {sorted(_VALID)}"
                )
