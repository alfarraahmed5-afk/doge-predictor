"""CLI entry point for bootstrapping historical Binance OHLCV data.

Fetches all required symbols and intervals from Binance REST and stores
them in the configured database (TimescaleDB in production, SQLite in dev).

Execution order (fixed, matches CLAUDE.md):
    1. DOGEUSDT 1h + BTCUSDT 1h  (parallel)
    2. DOGEBTC  1h               (sequential)
    3. All symbols for 4h        (sequential)
    4. All symbols for 1d        (sequential)

Usage::

    # Bootstrap with default settings (all symbols, 1h only)
    python scripts/bootstrap_doge.py

    # Restrict symbols and intervals
    python scripts/bootstrap_doge.py --symbols DOGEUSDT BTCUSDT --intervals 1h 4h

    # Override start date (ISO-8601 UTC)
    python scripts/bootstrap_doge.py --start 2021-01-01

    # Dry-run: print plan without fetching
    python scripts/bootstrap_doge.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Bootstrap src/ onto the Python path when run as a script
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import doge_settings, settings  # noqa: E402
from src.ingestion.multi_symbol import MultiSymbolBootstrapper  # noqa: E402
from src.ingestion.rest_client import BinanceRESTClient  # noqa: E402
from src.processing.storage import DogeStorage  # noqa: E402
from src.utils.helpers import datetime_to_ms  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------

_DEFAULT_SYMBOLS: list[str] = ["DOGEUSDT", "BTCUSDT", "DOGEBTC"]
_DEFAULT_INTERVALS: list[str] = ["1h"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the bootstrap CLI.

    Args:
        argv: Argument list.  ``None`` reads from ``sys.argv``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="bootstrap_doge",
        description="Bootstrap historical OHLCV data from Binance REST API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        metavar="SYMBOL",
        help=(
            "Symbols to bootstrap (space-separated). "
            f"Default: {' '.join(_DEFAULT_SYMBOLS)}"
        ),
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=_DEFAULT_INTERVALS,
        metavar="INTERVAL",
        help=(
            "Kline intervals to bootstrap (space-separated). "
            f"Default: {' '.join(_DEFAULT_INTERVALS)}"
        ),
    )
    parser.add_argument(
        "--start",
        default=doge_settings.context_start_date,
        metavar="YYYY-MM-DD",
        help=(
            "Inclusive start date (ISO-8601 UTC). "
            f"Default: {doge_settings.context_start_date}"
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Exclusive end date (ISO-8601 UTC). Default: now.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5_000,
        metavar="N",
        help="Save checkpoint every N rows. Default: 5000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without fetching any data.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date_to_ms(date_str: str) -> int:
    """Convert a ``YYYY-MM-DD`` string to UTC epoch milliseconds.

    Args:
        date_str: Date string in ``YYYY-MM-DD`` format.

    Returns:
        UTC epoch milliseconds (int).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return datetime_to_ms(dt)
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse date string {date_str!r}. Expected format: YYYY-MM-DD"
        ) from exc


def _print_plan(
    symbols: list[str],
    intervals: list[str],
    start_ms: int,
    end_ms: int,
) -> None:
    """Log the execution plan for a dry-run.

    Args:
        symbols: Symbols to bootstrap.
        intervals: Intervals to bootstrap.
        start_ms: Start timestamp (UTC epoch ms).
        end_ms: End timestamp (UTC epoch ms).
    """
    from src.utils.helpers import ms_to_datetime  # noqa: PLC0415

    start_dt = ms_to_datetime(start_ms).strftime("%Y-%m-%d %H:%M UTC")
    end_dt = ms_to_datetime(end_ms).strftime("%Y-%m-%d %H:%M UTC")

    logger.info("=== Bootstrap Execution Plan (--dry-run) ===")
    logger.info("  Symbols   : {}", symbols)
    logger.info("  Intervals : {}", intervals)
    logger.info("  Start     : {}", start_dt)
    logger.info("  End       : {}", end_dt)
    logger.info("  Tasks     : {}", len(symbols) * len(intervals))
    logger.info(
        "  Phase 1 (parallel)  : DOGEUSDT/1h + BTCUSDT/1h"
    )
    logger.info("  Phase 2 (sequential): DOGEBTC/1h")
    logger.info("  Phase 3 (sequential): 4h bootstraps")
    logger.info("  Phase 4 (sequential): 1d bootstraps")
    logger.info("============================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the bootstrap runner.

    Args:
        argv: Argument list for testing.  ``None`` reads ``sys.argv``.

    Returns:
        Exit code: ``0`` on success, non-zero on failure.
    """
    configure_logging()
    args = _parse_args(argv)

    # ---- Resolve timestamps -----------------------------------------------
    try:
        start_ms: int = _parse_date_to_ms(args.start)
    except ValueError as exc:
        logger.error("Invalid --start value: {}", exc)
        return 1

    if args.end is not None:
        try:
            end_ms: int = _parse_date_to_ms(args.end)
        except ValueError as exc:
            logger.error("Invalid --end value: {}", exc)
            return 1
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1_000)

    if start_ms >= end_ms:
        logger.error("--start must be strictly before --end.")
        return 1

    symbols: list[str] = [s.upper() for s in args.symbols]
    intervals: list[str] = args.intervals

    # ---- Dry-run ----------------------------------------------------------
    if args.dry_run:
        _print_plan(symbols, intervals, start_ms, end_ms)
        return 0

    # ---- Storage ----------------------------------------------------------
    checkpoint_dir: Path = settings.paths.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    storage = DogeStorage(settings)
    storage.create_tables()

    # ---- Progress bar (tqdm) ---------------------------------------------
    try:
        import tqdm as _tqdm_mod  # noqa: PLC0415

        pbar = _tqdm_mod.tqdm(
            total=len(symbols) * len(intervals),
            desc="Bootstrap",
            unit="task",
            ncols=100,
        )
    except ImportError:
        pbar = None  # type: ignore[assignment]

    # ---- Bootstrapper ---------------------------------------------------
    bootstrapper = MultiSymbolBootstrapper(
        make_client=lambda: BinanceRESTClient(),
        storage=storage,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n_rows=args.checkpoint_every,
    )

    logger.info(
        "Bootstrap starting: symbols={} intervals={} checkpoint_every={}",
        symbols,
        intervals,
        args.checkpoint_every,
    )

    t0 = time.monotonic()
    try:
        report = bootstrapper.run(
            symbols=symbols,
            intervals=intervals,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if pbar is not None:
            pbar.update(len(report))
    except Exception as exc:  # noqa: BLE001
        logger.error("Bootstrap FAILED with unhandled exception: {}", exc)
        if pbar is not None:
            pbar.close()
        return 1
    finally:
        if pbar is not None:
            pbar.close()

    elapsed = time.monotonic() - t0
    logger.info("All tasks completed in {:.1f}s", elapsed)
    bootstrapper.print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
