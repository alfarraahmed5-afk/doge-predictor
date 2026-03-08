"""Incremental candle update scheduler for doge_predictor.

Runs an APScheduler background job every hour (at :01 past each hour) to
fetch the last few candles from Binance, validate them, and upsert into
storage.  The overlap window (default 3 candles) catches any restatements
that Binance makes to recent candles.

Usage::

    from src.ingestion.scheduler import IncrementalScheduler
    from src.ingestion.rest_client import BinanceRESTClient
    from src.processing.storage import DogeStorage
    from src.processing.validator import DataValidator

    client = BinanceRESTClient()
    storage = DogeStorage(settings)
    validator = DataValidator()

    scheduler = IncrementalScheduler(
        client=client,
        storage=storage,
        validator=validator,
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        interval="1h",
    )
    scheduler.start()   # runs in background; call scheduler.stop() to halt
    # or: scheduler.run_once()  # one-shot execution, useful for testing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from src.ingestion.rest_client import BinanceRESTClient
from src.processing.storage import DogeStorage
from src.processing.validator import DataValidator
from src.utils.helpers import interval_to_ms

__all__ = ["IncrementalScheduler", "SchedulerStats"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Era boundary — UTC epoch ms for 2022-01-01 00:00:00.
_TRAINING_START_MS: int = 1_640_995_200_000

#: Number of candles to fetch back from now on each incremental run.
_DEFAULT_OVERLAP_CANDLES: int = 3

#: APScheduler job ID for the incremental update job.
_JOB_ID: str = "incremental_candle_update"


# ---------------------------------------------------------------------------
# SchedulerStats
# ---------------------------------------------------------------------------


@dataclass
class SchedulerStats:
    """Mutable statistics accumulated across all scheduler runs.

    Attributes:
        runs: Total number of completed update cycles.
        candles_fetched: Total candles fetched from Binance across all runs.
        candles_new: Total candles inserted (not previously in storage).
        candles_updated: Total candles updated (already existed, re-upserted).
        last_run_at: ISO-8601 UTC timestamp of the last completed run.
        last_run_errors: Number of per-symbol errors in the last run.
    """

    runs: int = 0
    candles_fetched: int = 0
    candles_new: int = 0
    candles_updated: int = 0
    last_run_at: str = ""
    last_run_errors: int = 0


# ---------------------------------------------------------------------------
# IncrementalScheduler
# ---------------------------------------------------------------------------


class IncrementalScheduler:
    """Hourly incremental candle update scheduler.

    Fetches the last :attr:`overlap_candles` candles for each symbol on every
    tick.  The overlap window catches Binance restatements (corrections to
    recently published OHLCV data).  Deduplication in storage ensures no
    duplicates are written; updated candles are silently overwritten.

    Args:
        client: Initialised :class:`~src.ingestion.rest_client.BinanceRESTClient`.
        storage: :class:`~src.processing.storage.DogeStorage` for upserts.
        validator: :class:`~src.processing.validator.DataValidator` instance.
        symbols: List of Binance symbols to update (e.g. ``["DOGEUSDT",
            "BTCUSDT", "DOGEBTC"]``).
        interval: Kline interval string (e.g. ``"1h"``).
        overlap_candles: Number of candles to re-fetch on each tick to catch
            restatements.  Defaults to 3.

    Attributes:
        stats: :class:`SchedulerStats` accumulating run metrics.
    """

    def __init__(
        self,
        client: BinanceRESTClient,
        storage: DogeStorage,
        validator: DataValidator,
        symbols: list[str],
        interval: str,
        overlap_candles: int = _DEFAULT_OVERLAP_CANDLES,
    ) -> None:
        """Initialise the scheduler (does not start the background job).

        Args:
            client: Binance REST API client.
            storage: Storage layer for upserts.
            validator: DataValidator instance.
            symbols: Symbols to poll on each tick.
            interval: Kline interval (e.g. ``"1h"``).
            overlap_candles: Number of candles to re-fetch per tick.
        """
        self._client: BinanceRESTClient = client
        self._storage: DogeStorage = storage
        self._validator: DataValidator = validator
        self._symbols: list[str] = list(symbols)
        self._interval: str = interval
        self._overlap_candles: int = overlap_candles
        self._interval_ms: int = interval_to_ms(interval)

        self.stats: SchedulerStats = SchedulerStats()

        self._scheduler: BackgroundScheduler = BackgroundScheduler(
            timezone="UTC",
        )
        self._scheduler.add_job(
            func=self._run_update_cycle,
            trigger=CronTrigger(minute=1),  # fire at :01 past each hour
            id=_JOB_ID,
            name="Incremental candle update",
            replace_existing=True,
            misfire_grace_time=300,  # allow up to 5 minutes late
        )

        logger.info(
            "IncrementalScheduler initialised: symbols={} interval={} overlap={}",
            self._symbols,
            self._interval,
            self._overlap_candles,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the APScheduler background job.

        The job fires at :01 past each hour (UTC).  Call :meth:`stop` to
        shut down the scheduler cleanly.

        Raises:
            RuntimeError: If the scheduler is already running.
        """
        if self._scheduler.running:
            raise RuntimeError("IncrementalScheduler is already running.")
        self._scheduler.start()
        logger.info(
            "IncrementalScheduler started — next fire: {}",
            self._scheduler.get_job(_JOB_ID).next_run_time,
        )

    def stop(self) -> None:
        """Shut down the APScheduler background job.

        Safe to call even if the scheduler is not running.
        """
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("IncrementalScheduler stopped.")
        else:
            logger.debug("IncrementalScheduler.stop() called but scheduler was not running.")

    def run_once(self) -> SchedulerStats:
        """Execute one update cycle immediately (blocking).

        Useful for manual triggers, smoke tests, and integration tests.

        Returns:
            :class:`SchedulerStats` after this run (cumulative).
        """
        self._run_update_cycle()
        return self.stats

    # ------------------------------------------------------------------
    # Internal update cycle
    # ------------------------------------------------------------------

    def _run_update_cycle(self) -> None:
        """Fetch, validate, and upsert the latest candles for all symbols.

        For each symbol:
            1. Compute ``start_ms = now - overlap_candles * interval_ms``.
            2. Fetch candles from Binance via :meth:`BinanceRESTClient.get_klines`.
            3. Assign era labels.
            4. Validate with :class:`DataValidator` (live-check mode).
            5. Count existing rows in the overlap window (new vs updated).
            6. Upsert into storage.
            7. Log metrics.

        Errors for individual symbols are caught and logged; the cycle
        continues for the remaining symbols.
        """
        now_ms: int = int(time.time() * 1_000)
        start_ms: int = now_ms - self._overlap_candles * self._interval_ms

        run_errors: int = 0

        for symbol in self._symbols:
            try:
                batch_df = self._client.get_klines(
                    symbol, self._interval, start_ms, now_ms
                )

                if batch_df.empty:
                    logger.debug(
                        "IncrementalScheduler: no data returned for {} — skipping",
                        symbol,
                    )
                    continue

                # Assign era labels
                batch_df = batch_df.copy()
                batch_df["era"] = "training"
                batch_df.loc[
                    batch_df["open_time"] < _TRAINING_START_MS, "era"
                ] = "context"
                batch_df["is_interpolated"] = False

                # Validate (live-check mode triggers stale-data check)
                val_result = self._validator.validate_ohlcv(
                    batch_df, symbol, self._interval, is_live_check=True
                )
                validation_status: str = (
                    "PASS" if val_result.is_valid else "WARN"
                )

                # Count how many rows already exist in the overlap window
                existing_df = self._storage.get_ohlcv(
                    symbol, self._interval, start_ms, now_ms
                )
                existing_times: set[int] = (
                    set(existing_df["open_time"].tolist())
                    if not existing_df.empty
                    else set()
                )

                n_fetched: int = len(batch_df)
                n_new: int = sum(
                    1
                    for t in batch_df["open_time"]
                    if int(t) not in existing_times
                )
                n_updated: int = n_fetched - n_new

                # Upsert (deduplication is handled inside storage)
                self._storage.upsert_ohlcv(batch_df, symbol, self._interval)

                # Update cumulative stats
                self.stats.candles_fetched += n_fetched
                self.stats.candles_new += n_new
                self.stats.candles_updated += n_updated

                logger.info(
                    "IncrementalScheduler: symbol={} fetched={} new={} "
                    "updated={} validation={}",
                    symbol,
                    n_fetched,
                    n_new,
                    n_updated,
                    validation_status,
                )

            except Exception as exc:  # noqa: BLE001
                run_errors += 1
                logger.error(
                    "IncrementalScheduler: error processing {}: {}",
                    symbol,
                    exc,
                )

        self.stats.runs += 1
        self.stats.last_run_errors = run_errors
        self.stats.last_run_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "IncrementalScheduler: cycle complete — run={} errors={}",
            self.stats.runs,
            run_errors,
        )
