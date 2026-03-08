"""OHLCV data cleaner for doge_predictor.

Removes rows that fail OHLCV sanity checks and logs each removal for
auditability.  The cleaner **never** forward-fills any price column — callers
that need gap-filling should use :class:`~src.processing.aligner.MultiSymbolAligner`.

Usage::

    from src.processing.cleaner import DataCleaner

    cleaner = DataCleaner()
    clean_df = cleaner.clean_ohlcv(raw_df, symbol="DOGEUSDT")
    removals = cleaner.get_removal_log()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from loguru import logger

__all__ = ["DataCleaner", "RemovalRecord"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Required columns for OHLCV sanity checks.
_REQUIRED_COLS: frozenset[str] = frozenset(
    {"open_time", "open", "high", "low", "close", "volume"}
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemovalRecord:
    """Immutable record of a single row removed by the cleaner.

    Attributes:
        open_time: UTC epoch milliseconds of the removed candle.
        symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
        reason: Human-readable description of the failing sanity check.
    """

    open_time: int
    symbol: str
    reason: str


# ---------------------------------------------------------------------------
# DataCleaner
# ---------------------------------------------------------------------------


class DataCleaner:
    """Removes OHLCV rows that fail price-bar sanity checks.

    The cleaner is stateful: it accumulates a :attr:`_removal_log` of every
    row dropped across multiple :meth:`clean_ohlcv` calls.  Use
    :meth:`get_removal_log` to inspect and :meth:`clear_log` to reset it.

    Checks applied (evaluated in order; first failing check sets the reason):
        * ``high < low``   — price-bar inversion
        * ``high < open``  — open price above the candle high
        * ``high < close`` — close price above the candle high
        * ``low > open``   — open price below the candle low
        * ``low > close``  — close price below the candle low
        * ``close <= 0``   — non-positive close (degenerate candle)
        * ``volume < 0``   — negative volume (impossible)

    Complies with CLAUDE.md Section 3.1 coding standards:
        - Full type hints on all public methods.
        - Google-style docstrings.
        - No ``print()`` — uses loguru.
        - No magic numbers.
        - No mutable default arguments.
        - Returns ``df.copy()`` to avoid ``SettingWithCopyWarning``.
    """

    def __init__(self) -> None:
        """Initialise the cleaner with an empty removal log."""
        self._removal_log: list[RemovalRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Remove rows that fail OHLCV sanity checks from *df*.

        Each removed row is logged to the internal removal log with the
        first failing reason.  A row that fails multiple checks is only
        logged once (for the first check that matched it).

        The returned DataFrame is a fresh copy of the valid rows with the
        original column order preserved and a reset integer index.  The
        input DataFrame is never mutated.

        **Forward-fill is never applied** — callers that need to recover
        gaps should use :class:`~src.processing.aligner.MultiSymbolAligner`.

        Args:
            df: OHLCV DataFrame with at minimum the columns ``open_time``,
                ``open``, ``high``, ``low``, ``close``, ``volume``.
            symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``). Used for
                logging and removal records.

        Returns:
            Cleaned copy of *df* with bad rows removed and index reset to
            0-based integers.  If all rows pass the row count equals
            ``len(df)``; if all rows fail an empty DataFrame is returned.

        Raises:
            ValueError: If any required column is absent from *df*.
        """
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"clean_ohlcv({symbol!r}): missing required columns: "
                f"{sorted(missing)}"
            )

        clean = df.copy()

        # Track which rows are already flagged to record only the first reason
        bad_mask: pd.Series = pd.Series(False, index=clean.index)

        # Each tuple: (boolean Series for bad rows, reason string)
        checks: list[tuple[pd.Series, str]] = [
            (clean["high"] < clean["low"], "high < low"),
            (clean["high"] < clean["open"], "high < open"),
            (clean["high"] < clean["close"], "high < close"),
            (clean["low"] > clean["open"], "low > open"),
            (clean["low"] > clean["close"], "low > close"),
            (clean["close"] <= 0, "close <= 0"),
            (clean["volume"] < 0, "volume < 0"),
        ]

        for row_bad, reason in checks:
            # Rows that are newly flagged by this check (not yet recorded)
            newly_bad: pd.Series = row_bad & ~bad_mask
            if newly_bad.any():
                for idx in clean.index[newly_bad]:
                    self._removal_log.append(
                        RemovalRecord(
                            open_time=int(clean.at[idx, "open_time"]),
                            symbol=symbol,
                            reason=reason,
                        )
                    )
                bad_mask = bad_mask | row_bad

        n_removed = int(bad_mask.sum())
        if n_removed > 0:
            logger.warning(
                "DataCleaner: removed {}/{} row(s) from {} for OHLCV sanity failures",
                n_removed,
                len(df),
                symbol,
            )
        else:
            logger.debug(
                "DataCleaner: {} rows for {} all passed sanity checks",
                len(df),
                symbol,
            )

        return clean.loc[~bad_mask].reset_index(drop=True)

    def get_removal_log(self) -> list[dict[str, Any]]:
        """Return the accumulated removal log as a list of plain dicts.

        Returns:
            List of ``{open_time, symbol, reason}`` dicts, one per removed
            row across all :meth:`clean_ohlcv` calls since construction or
            the last :meth:`clear_log` call.  Returns an empty list when no
            rows have been removed.
        """
        return [asdict(r) for r in self._removal_log]

    def clear_log(self) -> None:
        """Clear the accumulated removal log.

        Call this at the start of each pipeline run to avoid accumulating
        stale records from previous runs.
        """
        self._removal_log.clear()
        logger.debug("DataCleaner: removal log cleared")
