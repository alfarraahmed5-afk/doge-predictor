"""Multi-symbol timestamp aligner for doge_predictor.

:class:`MultiSymbolAligner` loads OHLCV data for multiple symbols from
:class:`~src.processing.storage.DogeStorage`, finds the common date range,
and produces a single aligned DataFrame where every row corresponds to the
same ``open_time`` across all symbols.

Alignment rules:
    1. All symbols are loaded for their full stored history.
    2. The common date range is ``[max(min_open_times), min(max_open_times)]``.
    3. An inner join on ``open_time`` produces the final alignment.
    4. **DOGEBTC only**: price column gaps of ≤ 3 missing candles are
       forward-filled from the previous valid candle.  Volume is NOT
       forward-filled (set to 0).  Filled rows are tagged
       ``dogebtc_interpolated=True``.
    5. Any gap > 3 candles in **any** symbol raises :class:`AlignmentError`
       and halts the pipeline.
    6. The final merged DataFrame is asserted to have identical ``open_time``
       values across all symbol sub-sets.

The aligned DataFrame uses prefixed column names (``doge_close``, ``btc_open``,
etc.) matching :data:`~src.processing.validator.AlignedSchema`.

The aligned DataFrame is stored on the instance as ``_last_aligned`` after each
successful call to :meth:`MultiSymbolAligner.align_symbols`.

Usage::

    from src.processing.aligner import MultiSymbolAligner, AlignmentResult
    aligner = MultiSymbolAligner()
    result: AlignmentResult = aligner.align_symbols(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
        storage=storage,
    )
    aligned_df = aligner._last_aligned
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.processing.storage import DogeStorage
from src.utils.helpers import interval_to_ms

__all__ = [
    "AlignmentError",
    "AlignmentResult",
    "MultiSymbolAligner",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum number of missing candles that can be recovered by forward-fill.
_MAX_FILL_CANDLES: int = 3

#: Symbol whose price gaps are forward-filled (volume is never filled).
_FILL_SYMBOL: str = "DOGEBTC"

#: Price columns to forward-fill for DOGEBTC (volume is explicitly excluded).
_PRICE_COLS: tuple[str, ...] = ("open", "high", "low", "close")

#: Columns to include per symbol in the merged output (besides open_time/era).
_OHLCV_VALUE_COLS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Mapping from canonical symbol name → column prefix in aligned output.
_PREFIX_MAP: dict[str, str] = {
    "DOGEUSDT": "doge",
    "BTCUSDT": "btc",
    "DOGEBTC": "dogebtc",
}

#: Upper bound for storage queries (exclusive end_ms — beyond any real data).
_QUERY_END_MS: int = 10_000_000_000_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlignmentError(Exception):
    """Raised when a gap > :data:`_MAX_FILL_CANDLES` candles is detected.

    Also raised when symbols have no overlapping date range, or when a
    symbol has no data at all.

    Args:
        message: Human-readable description of the alignment failure.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentResult:
    """Immutable summary returned after a successful alignment run.

    Attributes:
        common_start: Inclusive start of the common timestamp range (UTC epoch ms).
        common_end: Inclusive end of the common timestamp range (UTC epoch ms).
        rows_aligned: Number of rows in the aligned DataFrame.
        gaps_recovered: Total DOGEBTC candle slots recovered by forward-fill.
    """

    common_start: int
    common_end: int
    rows_aligned: int
    gaps_recovered: int


# ---------------------------------------------------------------------------
# MultiSymbolAligner
# ---------------------------------------------------------------------------


class MultiSymbolAligner:
    """Aligns OHLCV data for multiple symbols to a common ``open_time`` index.

    The aligner is stateless between calls; each :meth:`align_symbols`
    invocation is independent.  After a successful call the aligned DataFrame
    is available as ``self._last_aligned``.

    See the module docstring for the full alignment specification.
    """

    def __init__(self) -> None:
        """Initialise the aligner (no arguments required)."""
        self._last_aligned: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align_symbols(
        self,
        symbols: list[str],
        interval: str,
        storage: DogeStorage,
    ) -> AlignmentResult:
        """Load, align, and return merged OHLCV data for all *symbols*.

        Steps:
            1. Load all symbols from storage (full history).
            2. Find the common date range: ``[max(min_open_times),
               min(max_open_times)]``.
            3. Trim each symbol to the common range and inner-join on
               ``open_time``; log any symbol with fewer rows than expected.
            4. DOGEBTC only: forward-fill price columns for gaps ≤ 3 candles;
               tag filled rows with ``dogebtc_interpolated=True``.
            5. For any gap > 3 candles in **any** symbol: raise
               :class:`AlignmentError`.
            6. Assert the final DataFrame has an identical ``open_time`` set
               across all symbol sub-sets.

        Args:
            symbols: List of Binance trading pair symbols (e.g. ``["DOGEUSDT",
                "BTCUSDT", "DOGEBTC"]``).  Order is preserved in the output
                column ordering.
            interval: Kline interval string (e.g. ``"1h"``).
            storage: :class:`~src.processing.storage.DogeStorage` instance
                from which OHLCV data is loaded.

        Returns:
            :class:`AlignmentResult` metadata. The aligned DataFrame is stored
            as ``self._last_aligned``.

        Raises:
            ValueError: If *symbols* is empty or *interval* is unknown.
            AlignmentError: If any symbol has no data, there is no common
                date range, or a gap > 3 candles cannot be recovered.
        """
        if not symbols:
            raise ValueError("symbols list must not be empty")

        interval_ms: int = interval_to_ms(interval)

        # ------------------------------------------------------------------
        # Step 1 — Load all symbols from storage
        # ------------------------------------------------------------------
        raw: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = storage.get_ohlcv(sym, interval, 0, _QUERY_END_MS)
            if len(df) == 0:
                raise AlignmentError(
                    f"Symbol {sym}/{interval} has no data in storage. "
                    "Run bootstrap before aligning."
                )
            raw[sym] = df
            logger.info(
                "Loaded {}/{}: {} rows (range {} – {})",
                sym,
                interval,
                len(df),
                int(df["open_time"].min()),
                int(df["open_time"].max()),
            )

        # ------------------------------------------------------------------
        # Step 2 — Common date range: [max(min), min(max)]
        # ------------------------------------------------------------------
        min_times = {sym: int(df["open_time"].min()) for sym, df in raw.items()}
        max_times = {sym: int(df["open_time"].max()) for sym, df in raw.items()}

        common_start: int = max(min_times.values())
        common_end: int = min(max_times.values())

        if common_start > common_end:
            raise AlignmentError(
                f"No common date range across {symbols}. "
                f"Latest start={common_start}, earliest end={common_end}."
            )

        logger.info(
            "Common range: {} – {} ({} expected candles)",
            common_start,
            common_end,
            (common_end - common_start) // interval_ms + 1,
        )

        # ------------------------------------------------------------------
        # Step 3 — Trim to common range; build per-symbol indexed DataFrames
        # ------------------------------------------------------------------
        trimmed: dict[str, pd.DataFrame] = {}
        for sym, df in raw.items():
            mask = (df["open_time"] >= common_start) & (
                df["open_time"] <= common_end
            )
            t = df[mask].copy().reset_index(drop=True)
            t = t.set_index("open_time")
            trimmed[sym] = t

        # Full expected index for the common range (every interval_ms step)
        expected_index: pd.RangeIndex = pd.RangeIndex(
            start=common_start,
            stop=common_end + interval_ms,  # exclusive stop
            step=interval_ms,
            name="open_time",
        )

        # Log symbols with fewer rows than expected
        expected_count = len(expected_index)
        for sym, sym_df in trimmed.items():
            if len(sym_df) < expected_count:
                logger.info(
                    "{}/{}: {} rows in common range, expected {} "
                    "(missing {} candle(s))",
                    sym,
                    interval,
                    len(sym_df),
                    expected_count,
                    expected_count - len(sym_df),
                )

        # ------------------------------------------------------------------
        # Steps 4 & 5 — Gap detection + DOGEBTC forward-fill
        # ------------------------------------------------------------------
        gaps_recovered: int = 0
        # Track which timestamps were filled for DOGEBTC so the boolean column
        # can be created reliably via index.isin() rather than per-row assignment,
        # which can produce mixed-type columns in newer pandas versions.
        filled_timestamps: set[int] = set()

        for sym in symbols:
            sym_df = trimmed[sym]
            sym_index = sym_df.index
            missing_ts: pd.Index = expected_index.difference(sym_index)

            if len(missing_ts) == 0:
                continue  # no gaps for this symbol

            gap_runs = self._find_gap_runs(missing_ts.tolist(), interval_ms)

            for run_start, run_end, run_size in gap_runs:
                if run_size > _MAX_FILL_CANDLES:
                    raise AlignmentError(
                        f"Symbol {sym}/{interval}: gap of {run_size} candles "
                        f"starting at {run_start} exceeds maximum "
                        f"({_MAX_FILL_CANDLES}). Cannot continue alignment."
                    )

                if sym == _FILL_SYMBOL:
                    # Forward-fill DOGEBTC prices; volume = 0
                    prev_ts = run_start - interval_ms
                    if prev_ts in sym_df.index:
                        prev_row = sym_df.loc[prev_ts]
                        for ts in range(
                            run_start,
                            run_end + interval_ms,
                            interval_ms,
                        ):
                            new_row: dict[str, object] = {}
                            for col in _PRICE_COLS:
                                if col in sym_df.columns:
                                    new_row[col] = float(prev_row[col])
                            if "volume" in sym_df.columns:
                                new_row["volume"] = 0.0
                            if "era" in sym_df.columns:
                                new_row["era"] = str(prev_row["era"])
                            sym_df.loc[ts] = new_row  # type: ignore[index]
                            filled_timestamps.add(ts)
                            gaps_recovered += 1
                    else:
                        logger.warning(
                            "{}/{}: no preceding row for forward-fill at {}; "
                            "gap of {} candle(s) cannot be recovered",
                            sym,
                            interval,
                            run_start,
                            run_size,
                        )
                else:
                    # Non-DOGEBTC: gap is within tolerance → inner join will
                    # exclude these timestamps; log for visibility
                    logger.warning(
                        "{}/{}: gap of {} candle(s) at {} — "
                        "excluded from inner join",
                        sym,
                        interval,
                        run_size,
                        run_start,
                    )

            # After all gap runs for DOGEBTC: sort + stamp the interpolated
            # flag column in one pass using index.isin() for correctness.
            if sym == _FILL_SYMBOL and filled_timestamps:
                sym_df = sym_df.sort_index()
                sym_df["dogebtc_interpolated"] = sym_df.index.isin(
                    filled_timestamps
                )
                trimmed[sym] = sym_df

        # ------------------------------------------------------------------
        # Inner join — build a common index from all symbols
        # ------------------------------------------------------------------
        common_index: pd.Index = expected_index.astype(int)  # type: ignore[assignment]
        for sym, sym_df in trimmed.items():
            common_index = common_index.intersection(sym_df.index)

        if len(common_index) == 0:
            raise AlignmentError(
                "Inner join produced an empty result — no timestamps common "
                "to all symbols."
            )

        # ------------------------------------------------------------------
        # Step 6 — Build merged DataFrame with prefixed columns
        # ------------------------------------------------------------------
        parts: list[pd.DataFrame] = []
        for sym in symbols:
            sym_df = trimmed[sym]
            prefix = _PREFIX_MAP.get(sym, sym.lower().replace("usdt", ""))

            # Select only the value columns that exist in this DataFrame
            cols = [c for c in _OHLCV_VALUE_COLS if c in sym_df.columns]
            sub = sym_df.loc[common_index, cols].rename(
                columns={c: f"{prefix}_{c}" for c in cols}
            )

            # DOGEBTC: carry forward the interpolated flag
            if sym == _FILL_SYMBOL:
                if "dogebtc_interpolated" in sym_df.columns:
                    sub["dogebtc_interpolated"] = (
                        sym_df.loc[common_index, "dogebtc_interpolated"].fillna(False)
                    )
                else:
                    sub["dogebtc_interpolated"] = False

            parts.append(sub)

        merged: pd.DataFrame = pd.concat(parts, axis=1)

        # Attach era from the first symbol (all should agree post-trim)
        first_sym = symbols[0]
        if "era" in trimmed[first_sym].columns:
            merged["era"] = trimmed[first_sym].loc[common_index, "era"]
        else:
            merged["era"] = "training"

        merged = merged.sort_index()
        merged.index.name = "open_time"
        merged = merged.reset_index()  # open_time → column

        # ------------------------------------------------------------------
        # Step 6 cont. — Assert identical open_time index across all symbols
        # ------------------------------------------------------------------
        merged_times = set(merged["open_time"].tolist())
        for sym in symbols:
            sym_times = set(
                trimmed[sym]
                .index[trimmed[sym].index.isin(merged["open_time"])]
                .tolist()
            )
            if sym_times != merged_times:
                raise AlignmentError(
                    f"open_time alignment assertion failed for {sym}: "
                    "symbol timestamps do not match the merged DataFrame."
                )

        logger.info(
            "Alignment complete: {} rows, common_range=[{}, {}], "
            "gaps_recovered={}",
            len(merged),
            common_start,
            common_end,
            gaps_recovered,
        )

        self._last_aligned = merged
        return AlignmentResult(
            common_start=common_start,
            common_end=common_end,
            rows_aligned=len(merged),
            gaps_recovered=gaps_recovered,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_gap_runs(
        missing_ts: list[int],
        interval_ms: int,
    ) -> list[tuple[int, int, int]]:
        """Identify contiguous runs of missing timestamps.

        Args:
            missing_ts: Sorted list of missing ``open_time`` values.
            interval_ms: Expected milliseconds between consecutive candles.

        Returns:
            List of ``(run_start, run_end, run_size)`` tuples, one per
            contiguous run.  ``run_size`` is the number of missing candles
            in that run.
        """
        if not missing_ts:
            return []

        sorted_ts = sorted(missing_ts)
        runs: list[tuple[int, int, int]] = []
        run_start = sorted_ts[0]
        prev = sorted_ts[0]

        for ts in sorted_ts[1:]:
            if ts - prev == interval_ms:
                prev = ts  # extend current run
            else:
                run_size = (prev - run_start) // interval_ms + 1
                runs.append((run_start, prev, run_size))
                run_start = ts
                prev = ts

        # Flush last run
        run_size = (prev - run_start) // interval_ms + 1
        runs.append((run_start, prev, run_size))

        return runs
