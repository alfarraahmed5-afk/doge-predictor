"""Historical data bootstrap engine for doge_predictor.

Fetches every OHLCV candle for a symbol/interval from Binance REST, validates
each batch against ``OHLCVSchema``, assigns era labels, and upserts into
``DogeStorage``.  Checkpoints every ``checkpoint_every_n_rows`` rows so that
any crash can resume from where it left off.

The engine is deliberately single-threaded; parallelism is handled by
``multi_symbol.py`` which spawns one thread per symbol.

Usage::

    from src.ingestion.bootstrap import BootstrapEngine, BootstrapResult
    from src.ingestion.rest_client import BinanceRESTClient
    from src.processing.storage import DogeStorage

    client = BinanceRESTClient()
    engine = BootstrapEngine(client, checkpoint_dir=Path("data/checkpoints"))
    result: BootstrapResult = engine.bootstrap_symbol(
        symbol="DOGEUSDT",
        interval="1h",
        start_ms=1_564_617_600_000,   # 2019-08-01 UTC
        end_ms=int(time.time() * 1000),
        storage=storage,
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

from src.ingestion.exceptions import DataValidationError
from src.ingestion.rest_client import BinanceRESTClient
from src.processing.df_schemas import OHLCVSchema, validate_df
from src.processing.storage import DogeStorage
from src.utils.helpers import compute_expected_row_count, interval_to_ms

__all__ = [
    "BootstrapEngine",
    "BootstrapResult",
    "Checkpoint",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: 2022-01-01 00:00:00 UTC — era boundary between 'context' and 'training'.
_TRAINING_START_MS: int = 1_640_995_200_000

#: Number of candles requested per API call (matches Binance limit).
_BATCH_SIZE_ROWS: int = 1_000

#: Default checkpoint interval (rows between checkpoint saves).
_DEFAULT_CHECKPOINT_EVERY: int = 5_000

#: Allowable row-count deviation before logging a warning (±2 candles).
_ROW_COUNT_TOLERANCE: int = 2


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Immutable summary returned after a completed bootstrap run.

    Attributes:
        symbol: Trading pair bootstrapped (e.g. ``"DOGEUSDT"``).
        interval: Kline interval (e.g. ``"1h"``).
        rows_fetched: Total rows upserted into storage this run
            (excludes rows already present from a prior partial run).
        rows_total: Cumulative rows in the DB after completion (includes
            rows from any prior partial run that was resumed).
        gaps_found: Number of consecutive-candle gaps detected.
        duration_seconds: Wall-clock seconds for this run.
        start_ms: Inclusive start timestamp requested (UTC epoch ms).
        end_ms: Exclusive end timestamp requested (UTC epoch ms).
        era_context_rows: Rows labelled ``era='context'`` (pre-2022).
        era_training_rows: Rows labelled ``era='training'`` (post-2022).
    """

    symbol: str
    interval: str
    rows_fetched: int
    rows_total: int
    gaps_found: int
    duration_seconds: float
    start_ms: int
    end_ms: int
    era_context_rows: int
    era_training_rows: int


@dataclass
class Checkpoint:
    """Mutable checkpoint state written to disk after each save.

    Attributes:
        symbol: Trading pair symbol.
        interval: Kline interval.
        last_open_time: ``open_time`` of the last row successfully saved.
        rows_saved: Cumulative rows saved up to this checkpoint.
        started_at: ISO-8601 UTC timestamp when the run began.
        updated_at: ISO-8601 UTC timestamp of the last checkpoint write.
    """

    symbol: str
    interval: str
    last_open_time: int
    rows_saved: int
    started_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# BootstrapEngine
# ---------------------------------------------------------------------------


class BootstrapEngine:
    """Fetches, validates, and stores historical OHLCV candles from Binance.

    Each call to :meth:`bootstrap_symbol` is fully self-contained and
    restartable: on crash it resumes from the last checkpoint.

    Args:
        client: An initialised :class:`~src.ingestion.rest_client.BinanceRESTClient`.
        checkpoint_dir: Directory where per-symbol checkpoint JSON files are
            written.  Created automatically if it does not exist.
        checkpoint_every_n_rows: Number of rows between checkpoint saves.
            Default is 5 000.  Reduce in tests.
        batch_size: Number of candles per API call (default 1 000).
    """

    #: UTC epoch ms marking the training/context era boundary.
    TRAINING_START_MS: int = _TRAINING_START_MS

    def __init__(
        self,
        client: BinanceRESTClient,
        checkpoint_dir: Path,
        checkpoint_every_n_rows: int = _DEFAULT_CHECKPOINT_EVERY,
        batch_size: int = _BATCH_SIZE_ROWS,
    ) -> None:
        self._client: BinanceRESTClient = client
        self._checkpoint_dir: Path = Path(checkpoint_dir)
        self._checkpoint_every: int = checkpoint_every_n_rows
        self._batch_size: int = batch_size
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, symbol: str, interval: str) -> Path:
        """Return the checkpoint file path for *symbol*/*interval*.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval string.

        Returns:
            Absolute :class:`pathlib.Path` to the checkpoint JSON file.
        """
        safe_symbol = symbol.replace("/", "_")
        return self._checkpoint_dir / f"{safe_symbol}_{interval}_checkpoint.json"

    def _load_checkpoint(self, symbol: str, interval: str) -> Optional[Checkpoint]:
        """Load an existing checkpoint from disk if one exists.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval string.

        Returns:
            :class:`Checkpoint` if a valid file is found, otherwise ``None``.
        """
        path = self._checkpoint_path(symbol, interval)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cp = Checkpoint(**data)
            logger.info(
                "Checkpoint loaded for {}/{}: last_open_time={} rows_saved={}",
                symbol,
                interval,
                cp.last_open_time,
                cp.rows_saved,
            )
            return cp
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Ignoring corrupt checkpoint file {}: {}. Starting fresh.",
                path,
                exc,
            )
            return None

    def _save_checkpoint(self, cp: Checkpoint) -> None:
        """Persist *cp* to disk atomically using a temp-file rename.

        Args:
            cp: Checkpoint state to save.
        """
        path = self._checkpoint_path(cp.symbol, cp.interval)
        tmp = path.with_suffix(".tmp")
        try:
            cp.updated_at = datetime.now(timezone.utc).isoformat()
            tmp.write_text(json.dumps(asdict(cp), indent=2), encoding="utf-8")
            tmp.replace(path)
            logger.debug(
                "Checkpoint saved: {}/{} last_open_time={} rows_saved={}",
                cp.symbol,
                cp.interval,
                cp.last_open_time,
                cp.rows_saved,
            )
        except OSError as exc:
            logger.error("Failed to save checkpoint {}: {}", path, exc)

    def _delete_checkpoint(self, symbol: str, interval: str) -> None:
        """Delete the checkpoint file after a successful full bootstrap.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval string.
        """
        path = self._checkpoint_path(symbol, interval)
        try:
            path.unlink(missing_ok=True)
            logger.info("Checkpoint deleted (bootstrap complete): {}", path.name)
        except OSError as exc:
            logger.warning("Could not delete checkpoint {}: {}", path, exc)

    # ------------------------------------------------------------------
    # Era assignment
    # ------------------------------------------------------------------

    def _assign_era(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add an ``era`` column to *df* based on the training-start boundary.

        Rows with ``open_time < TRAINING_START_MS`` are labelled
        ``'context'``; all others are ``'training'``.

        Args:
            df: OHLCV DataFrame with an ``open_time`` column (int, UTC ms).

        Returns:
            Copy of *df* with a new ``era`` column (str).
        """
        out = df.copy()
        out["era"] = "context"
        out.loc[out["open_time"] >= self.TRAINING_START_MS, "era"] = "training"
        return out

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _count_gaps(open_times: list[int], interval_ms: int) -> int:
        """Count the number of missing-candle gaps in *open_times*.

        A gap is defined as two consecutive open_times whose difference
        is strictly greater than *interval_ms* (i.e. at least one candle
        is missing between them).

        Args:
            open_times: Sorted list of candle open_time values (UTC epoch ms).
            interval_ms: Expected milliseconds between consecutive candles.

        Returns:
            Count of gaps found (0 means the series is contiguous).
        """
        if len(open_times) < 2:
            return 0
        sorted_times = sorted(open_times)
        gaps = sum(
            1 for a, b in zip(sorted_times, sorted_times[1:]) if b - a > interval_ms
        )
        return gaps

    # ------------------------------------------------------------------
    # Core bootstrap
    # ------------------------------------------------------------------

    def bootstrap_symbol(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        storage: DogeStorage,
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> BootstrapResult:
        """Fetch and store all candles for *symbol*/*interval* in ``[start_ms, end_ms)``.

        Resumes automatically from checkpoint if one exists.  Saves a
        checkpoint every ``checkpoint_every_n_rows`` rows.  Deletes the
        checkpoint on clean completion.

        Args:
            symbol: Binance trading pair, e.g. ``"DOGEUSDT"``.
            interval: Kline interval string, e.g. ``"1h"``.
            start_ms: Inclusive start (UTC epoch ms).  Overridden by
                checkpoint when resuming.
            end_ms: Exclusive end (UTC epoch ms).
            storage: :class:`~src.processing.storage.DogeStorage` instance
                for upserts.
            progress_callback: Optional callable receiving the row count of
                each completed batch (used for tqdm updates).

        Returns:
            :class:`BootstrapResult` with run statistics.

        Raises:
            ValueError: If ``start_ms >= end_ms`` or *interval* is unknown.
            DataValidationError: If any batch fails schema validation.
        """
        if start_ms >= end_ms:
            raise ValueError(
                f"start_ms ({start_ms}) must be < end_ms ({end_ms})"
            )

        t0: float = time.monotonic()
        interval_ms: int = interval_to_ms(interval)
        batch_window_ms: int = self._batch_size * interval_ms

        # ---- checkpoint / resume logic -----------------------------------
        cp: Optional[Checkpoint] = self._load_checkpoint(symbol, interval)
        now_iso: str = datetime.now(timezone.utc).isoformat()

        if cp is not None:
            # Resume from just after the last checkpointed candle
            resume_start: int = cp.last_open_time + interval_ms
            rows_saved_total: int = cp.rows_saved
            logger.info(
                "Resuming {}/{} from checkpoint: last_open_time={} "
                "rows_saved_so_far={}",
                symbol,
                interval,
                cp.last_open_time,
                cp.rows_saved,
            )
        else:
            resume_start = start_ms
            rows_saved_total = 0
            cp = Checkpoint(
                symbol=symbol,
                interval=interval,
                last_open_time=start_ms,
                rows_saved=0,
                started_at=now_iso,
                updated_at=now_iso,
            )

        rows_this_run: int = 0
        accumulated_since_cp: int = 0
        all_open_times: list[int] = []
        era_context: int = 0
        era_training: int = 0

        current_start: int = resume_start

        logger.info(
            "Bootstrap starting: symbol={} interval={} "
            "current_start={} end_ms={} batch_size={}",
            symbol,
            interval,
            current_start,
            end_ms,
            self._batch_size,
        )

        # ---- main fetch loop -------------------------------------------
        while current_start < end_ms:
            batch_end: int = min(current_start + batch_window_ms, end_ms)

            try:
                batch_df: pd.DataFrame = self._client.get_klines(
                    symbol, interval, current_start, batch_end
                )
            except DataValidationError:
                logger.error(
                    "Schema validation failed for {}/{} batch starting at {}. "
                    "Aborting bootstrap.",
                    symbol,
                    interval,
                    current_start,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unexpected error fetching {}/{} at {}: {}",
                    symbol,
                    interval,
                    current_start,
                    exc,
                )
                raise

            if batch_df.empty:
                logger.debug(
                    "Empty batch for {}/{} at {} — end of data.",
                    symbol,
                    interval,
                    current_start,
                )
                break

            n: int = len(batch_df)

            # ---- era assignment ----------------------------------------
            batch_df = self._assign_era(batch_df)

            # ---- mark as non-interpolated ----------------------------------
            batch_df["is_interpolated"] = False

            # ---- upsert ---------------------------------------------------
            try:
                storage.upsert_ohlcv(batch_df, symbol, interval)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "upsert_ohlcv failed for {}/{} batch at {}: {}",
                    symbol,
                    interval,
                    current_start,
                    exc,
                )
                raise

            # ---- track stats -----------------------------------------------
            all_open_times.extend(batch_df["open_time"].tolist())
            era_context += int((batch_df["era"] == "context").sum())
            era_training += int((batch_df["era"] == "training").sum())
            rows_this_run += n
            rows_saved_total += n
            accumulated_since_cp += n

            last_open_time: int = int(batch_df["open_time"].iloc[-1])
            last_close_time: int = int(batch_df["close_time"].iloc[-1])

            logger.debug(
                "{}/{} batch done: rows={} total_so_far={} weight_used={}",
                symbol,
                interval,
                n,
                rows_saved_total,
                self._client.weight_used,
            )

            # ---- checkpoint every N rows ----------------------------------
            if accumulated_since_cp >= self._checkpoint_every:
                cp.last_open_time = last_open_time
                cp.rows_saved = rows_saved_total
                self._save_checkpoint(cp)
                accumulated_since_cp = 0

            # ---- tqdm / external progress ---------------------------------
            if progress_callback is not None:
                progress_callback(n)

            # ---- advance cursor ------------------------------------------
            current_start = last_close_time + 1

            # NOTE: Do NOT break on n < batch_size here.  The outer while-loop
            # condition (current_start < end_ms) handles termination for the
            # normal continuous-data case.  Stopping early on short batches
            # causes truncation when the symbol listing date falls inside the
            # first batch window, or when minor gaps reduce a page below 1 000.
            # The only early-exit is the `batch_df.empty` check above.

        # ---- completion: delete checkpoint --------------------------------
        self._delete_checkpoint(symbol, interval)

        # ---- gap analysis -------------------------------------------------
        gaps_found: int = self._count_gaps(all_open_times, interval_ms)
        if gaps_found:
            logger.warning(
                "{}/{} bootstrap: {} gap(s) detected in {} rows fetched. "
                "Gaps are normal for pre-listing history.",
                symbol,
                interval,
                gaps_found,
                rows_this_run,
            )

        # ---- row-count sanity check (soft) --------------------------------
        try:
            expected: int = compute_expected_row_count(start_ms, end_ms, interval_ms)
            deviation: int = abs(rows_saved_total - expected)
            if deviation > _ROW_COUNT_TOLERANCE:
                logger.warning(
                    "{}/{} row count deviation: expected ~{}, got {} "
                    "(diff={}, tolerance={}). Likely gaps in source data.",
                    symbol,
                    interval,
                    expected,
                    rows_saved_total,
                    deviation,
                    _ROW_COUNT_TOLERANCE,
                )
        except ValueError:
            pass

        duration: float = time.monotonic() - t0

        logger.info(
            "Bootstrap complete: symbol={} interval={} "
            "rows_fetched={} rows_total={} gaps={} "
            "era_context={} era_training={} duration={:.1f}s",
            symbol,
            interval,
            rows_this_run,
            rows_saved_total,
            gaps_found,
            era_context,
            era_training,
            duration,
        )

        return BootstrapResult(
            symbol=symbol,
            interval=interval,
            rows_fetched=rows_this_run,
            rows_total=rows_saved_total,
            gaps_found=gaps_found,
            duration_seconds=duration,
            start_ms=start_ms,
            end_ms=end_ms,
            era_context_rows=era_context,
            era_training_rows=era_training,
        )
