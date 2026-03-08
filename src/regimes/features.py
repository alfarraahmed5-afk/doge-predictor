"""Regime-derived numeric features for model input.

Converts a raw regime label string into a flat dict of numeric features
suitable for inclusion in the feature matrix.  Output includes:

- Five one-hot binary columns (one per regime).
- One ordinal ``regime_encoded`` column (integer 0–4, stored as float).

The ordinal encoding mirrors the CLAUDE.md Section 6 precedence order:
    0 → TRENDING_BULL
    1 → TRENDING_BEAR
    2 → RANGING_HIGH_VOL
    3 → RANGING_LOW_VOL
    4 → DECOUPLED

Usage::

    from src.regimes.features import get_regime_features

    features = get_regime_features("TRENDING_BULL")
    # {'regime_is_trending_bull': 1.0, 'regime_is_trending_bear': 0.0,
    #  'regime_is_ranging_high': 0.0, 'regime_is_ranging_low': 0.0,
    #  'regime_is_decoupled': 0.0, 'regime_encoded': 0.0}
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ordinal encoding — mirrors CLAUDE.md Section 6 precedence order
# ---------------------------------------------------------------------------

_REGIME_ENCODING: dict[str, int] = {
    "TRENDING_BULL": 0,
    "TRENDING_BEAR": 1,
    "RANGING_HIGH_VOL": 2,
    "RANGING_LOW_VOL": 3,
    "DECOUPLED": 4,
}

_VALID_REGIMES: frozenset[str] = frozenset(_REGIME_ENCODING.keys())

# Ordered feature key names (stable order for downstream consumers)
REGIME_FEATURE_KEYS: tuple[str, ...] = (
    "regime_is_trending_bull",
    "regime_is_trending_bear",
    "regime_is_ranging_high",
    "regime_is_ranging_low",
    "regime_is_decoupled",
    "regime_encoded",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_regime_features(regime_label: str) -> dict[str, float]:
    """Convert a regime label to a dict of numeric model-input features.

    Returns a one-hot encoding of *regime_label* across the five regime
    categories, plus an ordinal ``regime_encoded`` column.

    Args:
        regime_label: One of the five valid regime label strings:
            ``"TRENDING_BULL"``, ``"TRENDING_BEAR"``, ``"RANGING_HIGH_VOL"``,
            ``"RANGING_LOW_VOL"``, or ``"DECOUPLED"``.

    Returns:
        Dict with keys defined by :data:`REGIME_FEATURE_KEYS`:

        - ``regime_is_trending_bull``  (0.0 or 1.0)
        - ``regime_is_trending_bear``  (0.0 or 1.0)
        - ``regime_is_ranging_high``   (0.0 or 1.0)
        - ``regime_is_ranging_low``    (0.0 or 1.0)
        - ``regime_is_decoupled``      (0.0 or 1.0)
        - ``regime_encoded``           (0.0 – 4.0 ordinal)

    Raises:
        ValueError: If *regime_label* is not one of the five valid strings.

    Example::

        >>> get_regime_features("DECOUPLED")
        {'regime_is_trending_bull': 0.0, 'regime_is_trending_bear': 0.0,
         'regime_is_ranging_high': 0.0, 'regime_is_ranging_low': 0.0,
         'regime_is_decoupled': 1.0, 'regime_encoded': 4.0}
    """
    if regime_label not in _VALID_REGIMES:
        raise ValueError(
            f"Unknown regime label {regime_label!r}. "
            f"Must be one of: {sorted(_VALID_REGIMES)}"
        )

    return {
        "regime_is_trending_bull": float(regime_label == "TRENDING_BULL"),
        "regime_is_trending_bear": float(regime_label == "TRENDING_BEAR"),
        "regime_is_ranging_high": float(regime_label == "RANGING_HIGH_VOL"),
        "regime_is_ranging_low": float(regime_label == "RANGING_LOW_VOL"),
        "regime_is_decoupled": float(regime_label == "DECOUPLED"),
        "regime_encoded": float(_REGIME_ENCODING[regime_label]),
    }
