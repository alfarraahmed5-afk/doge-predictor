"""Multi-symbol bootstrap orchestrator for doge_predictor.

Coordinates the parallel and sequential fetching of all required symbols
and intervals, matching the execution order defined in CLAUDE.md:

    Phase 1 (parallel):   DOGEUSDT 1h  ║  BTCUSDT 1h
    Phase 2 (sequential): DOGEBTC  1h
    Phase 3 (sequential): DOGEUSDT 4h, BTCUSDT 4h, DOGEBTC 4h
    Phase 4 (sequential): DOGEUSDT 1d, BTCUSDT 1d, DOGEBTC 1d

Thread safety: ``DogeStorage`` already serialises all writes through a
``FileLock``.  Each thread operates on a separate ``BinanceRESTClient``
instance to avoid shared rate-limit state.

Usage::

    from src.ingestion.multi_symbol import MultiSymbolBootstrapper

    bootstrapper = MultiSymbolBootstrapper(
        make_client=lambda: BinanceRESTClient(),
        storage=store,
        checkpoint_dir=Path("data/checkpoints"),
    )
    report = bootstrapper.run(
        symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
        intervals=["1h"],
        start_ms=1_564_617_600_000,
        end_ms=int(time.time() * 1000),
    )
    bootstrapper.print_summary(report)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from src.ingestion.bootstrap import BootstrapEngine, BootstrapResult
from src.ingestion.rest_client import BinanceRESTClient
from src.processing.storage import DogeStorage

__all__ = ["MultiSymbolBootstrapper", "BootstrapReport"]

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Mapping from "(symbol, interval)" string key to BootstrapResult.
BootstrapReport = dict[str, BootstrapResult]

# ---------------------------------------------------------------------------
# MultiSymbolBootstrapper
# ---------------------------------------------------------------------------


class MultiSymbolBootstrapper:
    """Orchestrates parallel + sequential bootstrapping of multiple symbols.

    Each symbol/interval combination is handled by a dedicated
    :class:`~src.ingestion.bootstrap.BootstrapEngine`.  Parallel tasks
    share the same ``DogeStorage`` instance (which is filelock-protected)
    but use separate ``BinanceRESTClient`` instances (each tracking its
    own rate-limit weight counter independently).

    Args:
        make_client: Zero-argument factory that returns a new
            :class:`~src.ingestion.rest_client.BinanceRESTClient`.  Called
            once per thread to avoid shared rate-limit state.
        storage: Shared :class:`~src.processing.storage.DogeStorage` for
            all upsert operations.
        checkpoint_dir: Directory for checkpoint JSON files.
        checkpoint_every_n_rows: Rows between checkpoint saves (default 5 000).
        max_parallel_workers: Maximum threads for the parallel phase
            (default 2 — one per symbol in the parallel group).
    """

    #: Symbols that are fetched concurrently in the first phase.
    _PARALLEL_SYMBOLS: tuple[str, ...] = ("DOGEUSDT", "BTCUSDT")

    #: Symbol fetched sequentially after the parallel phase completes.
    _SEQUENTIAL_SYMBOL: str = "DOGEBTC"

    def __init__(
        self,
        make_client: Callable[[], BinanceRESTClient],
        storage: DogeStorage,
        checkpoint_dir: Path,
        checkpoint_every_n_rows: int = 5_000,
        max_parallel_workers: int = 2,
    ) -> None:
        self._make_client: Callable[[], BinanceRESTClient] = make_client
        self._storage: DogeStorage = storage
        self._checkpoint_dir: Path = Path(checkpoint_dir)
        self._checkpoint_every: int = checkpoint_every_n_rows
        self._max_workers: int = max_parallel_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_engine(self) -> BootstrapEngine:
        """Create a fresh :class:`BootstrapEngine` with its own REST client.

        Returns:
            New :class:`~src.ingestion.bootstrap.BootstrapEngine` instance.
        """
        return BootstrapEngine(
            client=self._make_client(),
            checkpoint_dir=self._checkpoint_dir,
            checkpoint_every_n_rows=self._checkpoint_every,
        )

    def _run_one(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> BootstrapResult:
        """Run bootstrap for a single symbol/interval.  Thread-safe.

        Args:
            symbol: Trading pair, e.g. ``"DOGEUSDT"``.
            interval: Kline interval, e.g. ``"1h"``.
            start_ms: Inclusive start (UTC epoch ms).
            end_ms: Exclusive end (UTC epoch ms).
            progress_callback: Optional row-count callback (for tqdm).

        Returns:
            :class:`~src.ingestion.bootstrap.BootstrapResult` for this task.
        """
        engine = self._make_engine()
        logger.info("Starting bootstrap task: {}/{}", symbol, interval)
        result = engine.bootstrap_symbol(
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            storage=self._storage,
            progress_callback=progress_callback,
        )
        logger.info(
            "Bootstrap task done: {}/{} rows_fetched={} gaps={} duration={:.1f}s",
            symbol,
            interval,
            result.rows_fetched,
            result.gaps_found,
            result.duration_seconds,
        )
        return result

    def _run_parallel(
        self,
        tasks: list[tuple[str, str, int, int]],
    ) -> BootstrapReport:
        """Execute *tasks* concurrently using a ThreadPoolExecutor.

        Args:
            tasks: List of ``(symbol, interval, start_ms, end_ms)`` tuples.

        Returns:
            ``BootstrapReport`` mapping ``"symbol/interval"`` → result.
        """
        report: BootstrapReport = {}

        if not tasks:
            return report

        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(tasks)),
            thread_name_prefix="bootstrap",
        ) as pool:
            future_to_key: dict[Future[BootstrapResult], str] = {
                pool.submit(self._run_one, sym, ivl, s, e): f"{sym}/{ivl}"
                for sym, ivl, s, e in tasks
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    report[key] = future.result()
                    logger.info("Parallel task complete: {}", key)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Parallel task FAILED for {}: {}", key, exc)
                    raise

        return report

    def _run_sequential(
        self,
        tasks: list[tuple[str, str, int, int]],
    ) -> BootstrapReport:
        """Execute *tasks* one by one in the given order.

        Args:
            tasks: List of ``(symbol, interval, start_ms, end_ms)`` tuples.

        Returns:
            ``BootstrapReport`` mapping ``"symbol/interval"`` → result.
        """
        report: BootstrapReport = {}
        for sym, ivl, s, e in tasks:
            key = f"{sym}/{ivl}"
            try:
                report[key] = self._run_one(sym, ivl, s, e)
            except Exception as exc:  # noqa: BLE001
                logger.error("Sequential task FAILED for {}: {}", key, exc)
                raise
        return report

    # ------------------------------------------------------------------
    # Public orchestration interface
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: list[str],
        intervals: list[str],
        start_ms: int,
        end_ms: int,
    ) -> BootstrapReport:
        """Orchestrate bootstrap for all *symbols* × *intervals*.

        Execution order (matching CLAUDE.md):
        1. **Parallel**: ``DOGEUSDT`` and ``BTCUSDT`` for *every* interval.
        2. **Sequential**: ``DOGEBTC`` for *every* interval.
        3. **Sequential**: Any remaining symbols for *every* interval.
        4. Intervals are processed in order (e.g. ``1h`` → ``4h`` → ``1d``).

        Note: If ``DOGEUSDT``/``BTCUSDT``/``DOGEBTC`` are not all in
        *symbols*, the method falls back to running everything sequentially.

        Args:
            symbols: Trading pairs to bootstrap,
                e.g. ``["DOGEUSDT", "BTCUSDT", "DOGEBTC"]``.
            intervals: Kline intervals to bootstrap,
                e.g. ``["1h", "4h", "1d"]``.
            start_ms: Inclusive start for all symbols (UTC epoch ms).
            end_ms: Exclusive end for all symbols (UTC epoch ms).

        Returns:
            Combined :data:`BootstrapReport` for all tasks.
        """
        wall_t0: float = time.monotonic()
        full_report: BootstrapReport = {}

        for interval in intervals:
            logger.info("=== Bootstrap interval: {} ===", interval)

            parallel_syms = [s for s in self._PARALLEL_SYMBOLS if s in symbols]
            sequential_syms_after = [
                s for s in symbols
                if s not in self._PARALLEL_SYMBOLS
            ]

            # ---- Phase A: parallel DOGEUSDT + BTCUSDT --------------------
            if parallel_syms:
                parallel_tasks = [
                    (sym, interval, start_ms, end_ms) for sym in parallel_syms
                ]
                logger.info(
                    "Phase A — parallel: {}", [t[0] for t in parallel_tasks]
                )
                full_report.update(self._run_parallel(parallel_tasks))

            # ---- Phase B: sequential remaining (DOGEBTC first, then others)
            if sequential_syms_after:
                # Put DOGEBTC first if present (CLAUDE.md ordering)
                ordered: list[str] = sorted(
                    sequential_syms_after,
                    key=lambda s: (0 if s == self._SEQUENTIAL_SYMBOL else 1, s),
                )
                seq_tasks = [(sym, interval, start_ms, end_ms) for sym in ordered]
                logger.info(
                    "Phase B — sequential: {}", [t[0] for t in seq_tasks]
                )
                full_report.update(self._run_sequential(seq_tasks))

        total_duration: float = time.monotonic() - wall_t0
        logger.info(
            "MultiSymbolBootstrapper.run() complete: {} tasks in {:.1f}s",
            len(full_report),
            total_duration,
        )
        return full_report

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def print_summary(self, report: BootstrapReport) -> None:
        """Log a formatted summary table of all bootstrap results.

        Args:
            report: The :data:`BootstrapReport` returned by :meth:`run`.
        """
        if not report:
            logger.info("No bootstrap results to summarise.")
            return

        header = (
            f"{'Task':<20} {'Rows':>8} {'Total':>8} "
            f"{'Gaps':>6} {'Ctx':>7} {'Trn':>7} {'Duration':>10}"
        )
        sep = "-" * len(header)

        lines: list[str] = [
            "",
            "Bootstrap Summary",
            sep,
            header,
            sep,
        ]

        total_rows = 0
        total_gaps = 0

        for key, r in sorted(report.items()):
            lines.append(
                f"{key:<20} {r.rows_fetched:>8,} {r.rows_total:>8,} "
                f"{r.gaps_found:>6} {r.era_context_rows:>7,} "
                f"{r.era_training_rows:>7,} {r.duration_seconds:>9.1f}s"
            )
            total_rows += r.rows_fetched
            total_gaps += r.gaps_found

        lines += [
            sep,
            f"{'TOTAL':<20} {total_rows:>8,} {'':>8} {total_gaps:>6}",
            sep,
            "",
        ]

        for line in lines:
            logger.info("{}", line)
