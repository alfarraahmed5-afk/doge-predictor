"""Bootstrap all Binance OHLCV + funding rate data into a local SQLite database.

This script is the SQLite fallback for environments where PostgreSQL is not
available (development, offline use).  It writes everything to
``data/doge_data.db`` and mirrors the full ``bootstrap_doge.py`` workflow.

Symbols bootstrapped:
    - DOGEUSDT  1h / 4h / 1d  (Jul 2019 – now)
    - BTCUSDT   1h / 4h / 1d  (Jul 2019 – now)
    - DOGEBTC   1h / 4h / 1d  (Jul 2019 – now)

Funding rates bootstrapped:
    - DOGEUSDT  8h  (Oct 2020 – now)

Usage::

    py scripts/run_bootstrap_sqlite.py
    py scripts/run_bootstrap_sqlite.py --dry-run
    py scripts/run_bootstrap_sqlite.py --symbols DOGEUSDT --intervals 1h
    py scripts/run_bootstrap_sqlite.py --skip-funding
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dotenv
import sqlalchemy as sa
from loguru import logger

from src.config import get_settings
from src.ingestion.bootstrap import BootstrapEngine, BootstrapResult
from src.ingestion.futures_client import BinanceFuturesClient
from src.ingestion.rest_client import BinanceRESTClient
from src.processing.storage import DogeStorage
from src.utils.helpers import interval_to_ms
from src.utils.logger import configure_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default SQLite database path (relative to project root).
_DEFAULT_DB_PATH: Path = _PROJECT_ROOT / "data" / "doge_data.db"

#: Start of DOGE context history on Binance (Jul 2019).
_CONTEXT_START_MS: int = 1_561_939_200_000  # 2019-07-01 00:00 UTC

#: Start of Binance DOGEUSDT perps (funding rates available from here).
_FUNDING_START_MS: int = 1_603_065_600_000  # 2020-10-19 00:00 UTC

#: Default symbols and intervals to bootstrap.
_DEFAULT_SYMBOLS: list[str] = ["DOGEUSDT", "BTCUSDT", "DOGEBTC"]
_DEFAULT_INTERVALS: list[str] = ["1h", "4h", "1d"]

#: Checkpoint every N rows.
_CHECKPOINT_EVERY: int = 5_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    """Return current UTC epoch milliseconds."""
    return int(time.time() * 1_000)


def _ms_to_date(ms: int) -> str:
    """Format epoch ms as YYYY-MM-DD for logging."""
    return datetime.fromtimestamp(ms / 1_000, tz=timezone.utc).strftime("%Y-%m-%d")


def _build_storage(db_path: Path, settings) -> DogeStorage:
    """Create a DogeStorage backed by SQLite at *db_path*.

    Args:
        db_path: Filesystem path for the SQLite file.
        settings: Loaded Settings instance.

    Returns:
        Ready-to-use DogeStorage with tables created.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    storage = DogeStorage(settings, engine=engine)
    storage.create_tables()
    logger.info("SQLite database ready at {}", db_path)
    return storage


def _run_ohlcv_bootstrap(
    symbols: list[str],
    intervals: list[str],
    start_ms: int,
    end_ms: int,
    storage: DogeStorage,
    checkpoint_dir: Path,
    dry_run: bool,
) -> dict[str, BootstrapResult]:
    """Bootstrap OHLCV data for all symbol/interval combos.

    Args:
        symbols: List of symbol strings (e.g. ``["DOGEUSDT", "BTCUSDT"]``).
        intervals: List of interval strings (e.g. ``["1h", "4h"]``).
        start_ms: Inclusive start timestamp (UTC epoch ms).
        end_ms: Exclusive end timestamp (UTC epoch ms).
        storage: DogeStorage instance to write into.
        checkpoint_dir: Directory for checkpoint JSON files.
        dry_run: If True, print plan but do not fetch data.

    Returns:
        Dict keyed ``"SYMBOL/interval"`` → ``BootstrapResult``.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, BootstrapResult] = {}

    for symbol in symbols:
        for interval in intervals:
            key = f"{symbol}/{interval}"

            if dry_run:
                expected_rows = (end_ms - start_ms) // interval_to_ms(interval)
                logger.info(
                    "[DRY-RUN] Would bootstrap {} {} from {} to {} (~{} rows)",
                    symbol,
                    interval,
                    _ms_to_date(start_ms),
                    _ms_to_date(end_ms),
                    expected_rows,
                )
                continue

            logger.info(
                "Bootstrapping {} {} from {} to {}",
                symbol,
                interval,
                _ms_to_date(start_ms),
                _ms_to_date(end_ms),
            )

            try:
                client = BinanceRESTClient()
                engine_obj = BootstrapEngine(
                    client=client,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_every_n_rows=_CHECKPOINT_EVERY,
                )
                result = engine_obj.bootstrap_symbol(
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    storage=storage,
                )
                results[key] = result
                logger.success(
                    "Completed {}: {} rows ({} context, {} training), {} gaps",
                    key,
                    result.rows_fetched,
                    result.era_context_rows,
                    result.era_training_rows,
                    result.gaps_found,
                )
            except Exception as exc:
                logger.error("Bootstrap FAILED for {}: {}", key, exc)

    return results


def _run_funding_bootstrap(
    symbol: str,
    start_ms: int,
    end_ms: int,
    storage: DogeStorage,
    dry_run: bool,
) -> None:
    """Bootstrap DOGEUSDT funding rates from Binance Futures.

    Args:
        symbol: Futures symbol (``"DOGEUSDT"``).
        start_ms: Inclusive start timestamp (UTC epoch ms).
        end_ms: Exclusive end timestamp (UTC epoch ms).
        storage: DogeStorage instance to write into.
        dry_run: If True, print plan but do not fetch data.
    """
    if dry_run:
        expected = (end_ms - start_ms) // (8 * 3_600_000)
        logger.info(
            "[DRY-RUN] Would bootstrap {} funding rates from {} to {} (~{} rows)",
            symbol,
            _ms_to_date(start_ms),
            _ms_to_date(end_ms),
            expected,
        )
        return

    logger.info(
        "Bootstrapping {} funding rates from {} to {}",
        symbol,
        _ms_to_date(start_ms),
        _ms_to_date(end_ms),
    )

    try:
        client = BinanceFuturesClient()
        df = client.get_funding_rates(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

        if df.empty:
            logger.warning("No funding rate data returned for {}", symbol)
            return

        # DogeStorage.upsert_funding_rates() expects column "timestamp_ms";
        # BinanceFuturesClient returns "funding_time" — rename before upsert.
        df_to_store = df.rename(columns={"funding_time": "timestamp_ms"})
        n = storage.upsert_funding_rates(df_to_store)
        logger.success(
            "Funding rates: {} rows upserted for {} ({} to {})",
            n,
            symbol,
            _ms_to_date(int(df["funding_time"].min())),
            _ms_to_date(int(df["funding_time"].max())),
        )
    except Exception as exc:
        logger.error("Funding rate bootstrap FAILED for {}: {}", symbol, exc)


def _print_summary(
    results: dict[str, BootstrapResult],
    db_path: Path,
    elapsed_s: float,
) -> None:
    """Print a formatted summary table to stdout.

    Args:
        results: Dict of BootstrapResult objects keyed by ``"SYMBOL/interval"``.
        db_path: Path to the SQLite database.
        elapsed_s: Total elapsed seconds.
    """
    logger.info("=" * 60)
    logger.info("Bootstrap Summary")
    logger.info("=" * 60)
    logger.info("Database: {}", db_path)
    logger.info("Elapsed:  {:.1f}s", elapsed_s)
    logger.info("")

    if not results:
        logger.info("No results (dry-run or all failed)")
        return

    header = f"{'Symbol/Interval':<20} {'Rows':>8} {'Context':>9} {'Training':>10} {'Gaps':>6}"
    logger.info(header)
    logger.info("-" * len(header))

    total_rows = 0
    for key, r in sorted(results.items()):
        logger.info(
            "{:<20} {:>8,} {:>9,} {:>10,} {:>6}",
            key,
            r.rows_fetched,
            r.era_context_rows,
            r.era_training_rows,
            r.gaps_found,
        )
        total_rows += r.rows_fetched

    logger.info("-" * len(header))
    logger.info("{:<20} {:>8,}", "TOTAL", total_rows)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bootstrap Binance OHLCV + funding data into SQLite"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=_DEFAULT_SYMBOLS,
        help="Symbols to bootstrap (default: DOGEUSDT BTCUSDT DOGEBTC)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=_DEFAULT_INTERVALS,
        help="Intervals to bootstrap (default: 1h 4h 1d)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: 2019-07-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--db-path",
        default=str(_DEFAULT_DB_PATH),
        help=f"SQLite database path (default: {_DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--skip-funding",
        action="store_true",
        help="Skip funding rate bootstrap",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan but do not fetch data",
    )
    return parser.parse_args()


def _parse_date_arg(date_str: str | None, default_ms: int) -> int:
    """Convert optional YYYY-MM-DD string to UTC epoch ms.

    Args:
        date_str: Date string or None.
        default_ms: Fallback value when date_str is None.

    Returns:
        UTC epoch milliseconds.
    """
    if date_str is None:
        return default_ms
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code: 0 on success, 1 on any failure.
    """
    configure_logging()
    args = _parse_args()

    # Load API keys from secrets.env
    dotenv.load_dotenv(_PROJECT_ROOT / "config" / "secrets.env")
    if not os.getenv("BINANCE_API_KEY"):
        logger.warning(
            "BINANCE_API_KEY not set — requests will use public rate limits"
        )

    settings = get_settings()

    # Parse date range
    start_ms = _parse_date_arg(args.start, _CONTEXT_START_MS)
    end_ms = _parse_date_arg(args.end, _now_ms())

    logger.info("Date range: {} to {}", _ms_to_date(start_ms), _ms_to_date(end_ms))
    logger.info("Symbols:    {}", args.symbols)
    logger.info("Intervals:  {}", args.intervals)

    db_path = Path(args.db_path)
    checkpoint_dir = _PROJECT_ROOT / "data" / "checkpoints"

    # Build storage (SQLite)
    if not args.dry_run:
        storage = _build_storage(db_path, settings)
    else:
        storage = None  # type: ignore[assignment]

    t0 = time.monotonic()

    # OHLCV bootstrap
    results = _run_ohlcv_bootstrap(
        symbols=args.symbols,
        intervals=args.intervals,
        start_ms=start_ms,
        end_ms=end_ms,
        storage=storage,
        checkpoint_dir=checkpoint_dir,
        dry_run=args.dry_run,
    )

    # Funding rate bootstrap
    if not args.skip_funding and "DOGEUSDT" in args.symbols:
        funding_start = max(start_ms, _FUNDING_START_MS)
        _run_funding_bootstrap(
            symbol="DOGEUSDT",
            start_ms=funding_start,
            end_ms=end_ms,
            storage=storage,
            dry_run=args.dry_run,
        )

    elapsed = time.monotonic() - t0
    _print_summary(results, db_path, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
