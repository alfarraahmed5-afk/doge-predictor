"""Read/write abstraction for all doge_predictor persistent storage.

This is the single module that owns every database interaction. All other
modules call into ``DogeStorage``; none of them import SQLAlchemy directly.

Architecture:
    - SQLAlchemy 2.0 Core (not ORM) for portability and performance.
    - Dialect-aware upsert: PostgreSQL and SQLite (test environment).
    - ``filelock.FileLock`` guards all write methods to prevent concurrent
      corruption when multiple processes share the same database.
    - ``data/raw/`` is permanently immutable after bootstrap. Any attempted
      write to that subtree raises ``PermissionError`` immediately.
    - Every public method wraps DB calls in ``try/except SQLAlchemyError``.

Usage::

    from src.config import settings
    from src.processing.storage import DogeStorage

    store = DogeStorage(settings)
    # In test environments, inject a SQLite engine:
    # store = DogeStorage(settings, engine=sqlite_engine)
    store.create_tables()  # creates schema (tests / first-run dev only)
    n = store.upsert_ohlcv(df, symbol="DOGEUSDT", interval="1h")

Notes:
    - ALL timestamps in the database are UTC epoch milliseconds (``int``).
    - Production uses the ``create_tables.sql`` DDL (hypertables, indexes).
      The ``create_tables()`` Python method is for test / development only.
    - ``data/raw/`` writes raise ``PermissionError`` unconditionally — call
      ``DogeStorage.guard_raw_write(path)`` before any filesystem write.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, ClassVar, Optional

import pandas as pd
import sqlalchemy as sa
from filelock import FileLock, Timeout as FileLockTimeout
from loguru import logger
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Engine,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    text,
)
from sqlalchemy.exc import SQLAlchemyError

from src.config import Settings
from src.processing.schemas import PredictionRecord

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Name of the write-lock file created under data_root.
_LOCK_FILENAME: str = ".doge_storage.lock"

#: Timeout (seconds) for acquiring the write file-lock.
_LOCK_TIMEOUT_S: int = 30

#: Column dtypes returned by get_ohlcv / get_funding_rates (for Pandas).
_OHLCV_NUMERIC_COLS: tuple[str, ...] = (
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
)

#: Mapping from interval string to table name.
_OHLCV_TABLE_NAMES: dict[str, str] = {
    "1h": "ohlcv_1h",
    "4h": "ohlcv_4h",
    "1d": "ohlcv_1d",
}

#: Path to the data/raw/ subtree computed at module import.
_RAW_DATA_DIR: Path = (Path(__file__).parent.parent.parent / "data" / "raw").resolve()


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------


def guard_raw_write(path: Path) -> None:
    """Raise ``PermissionError`` if *path* resolves into ``data/raw/``.

    ``data/raw/`` is permanently append-only after bootstrap. Any write
    attempt inside that subtree is a critical error and must be blocked
    immediately.

    Args:
        path: Filesystem path that is about to be written.

    Raises:
        PermissionError: If *path* is inside the ``data/raw/`` directory.
    """
    try:
        path.resolve().relative_to(_RAW_DATA_DIR)
        raise PermissionError(
            f"Writing to data/raw/ is forbidden after bootstrap. "
            f"data/raw/ is an immutable, append-only raw data store. "
            f"Attempted path: {path}"
        )
    except ValueError:
        pass  # path is outside data/raw/ — safe to write


# ---------------------------------------------------------------------------
# DogeStorage
# ---------------------------------------------------------------------------


class DogeStorage:
    """Read/write abstraction for all doge_predictor persistent storage.

    Wraps a SQLAlchemy connection pool and exposes typed methods for every
    table in the schema. All writes use upsert semantics and are serialised
    with a file-lock.

    Args:
        settings: Loaded ``Settings`` instance from ``src.config``.
        engine: Optional pre-built SQLAlchemy engine. When provided, it is
            used directly and no connection pool is created. This is the
            injection point for SQLite engines in unit tests.

    Attributes:
        _engine: Active SQLAlchemy ``Engine``.
        _metadata: ``MetaData`` object holding all ``Table`` definitions.
        _lock_path: Path to the write-lock file.
    """

    #: Public alias so callers can use ``DogeStorage.guard_raw_write``.
    guard_raw_write: ClassVar = staticmethod(guard_raw_write)

    #: Valid OHLCV interval → table-name mapping.
    OHLCV_TABLE_NAMES: ClassVar[dict[str, str]] = _OHLCV_TABLE_NAMES

    def __init__(
        self,
        settings: Settings,
        *,
        engine: Optional[Engine] = None,
    ) -> None:
        """Initialise the storage layer.

        Args:
            settings: Loaded ``Settings`` singleton.
            engine: Optional pre-built engine (for testing with SQLite).
                When ``None``, a PostgreSQL engine is created from
                ``settings.database``.
        """
        self._settings = settings

        if engine is not None:
            self._engine = engine
        else:
            self._engine = sa.create_engine(
                settings.database.url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
            )

        self._lock_path: Path = settings.paths.data_root / _LOCK_FILENAME
        self._metadata: MetaData = MetaData()
        self._define_tables()

        logger.info(
            "DogeStorage initialised (dialect={})",
            self._engine.dialect.name,
        )

    # -----------------------------------------------------------------------
    # Internal: table definitions
    # -----------------------------------------------------------------------

    def _define_tables(self) -> None:
        """Populate ``self._metadata`` with all ``Table`` definitions.

        These definitions are used by ``create_tables()`` (for tests / dev)
        and by all query methods. Production uses ``create_tables.sql`` to
        create the actual TimescaleDB hypertables with proper indexes.
        """
        # --- OHLCV columns (shared across 1h / 4h / 1d) ------------------
        def _ohlcv_cols() -> list[Column]:  # type: ignore[type-arg]
            return [
                Column("symbol", String(20), nullable=False),
                Column("open_time", BigInteger(), nullable=False),
                Column("open", Numeric(18, 8), nullable=False),
                Column("high", Numeric(18, 8), nullable=False),
                Column("low", Numeric(18, 8), nullable=False),
                Column("close", Numeric(18, 8), nullable=False),
                Column("volume", Numeric(18, 8), nullable=False),
                Column("close_time", BigInteger()),
                Column("quote_volume", Numeric(18, 8)),
                Column("num_trades", Integer()),
                Column("era", String(10), nullable=False),
                Column(
                    "is_interpolated",
                    Boolean(),
                    nullable=False,
                    server_default=text("0"),  # SQLite compatible (0/1)
                ),
                sa.PrimaryKeyConstraint("symbol", "open_time"),
            ]

        Table("ohlcv_1h", self._metadata, *_ohlcv_cols())
        Table("ohlcv_4h", self._metadata, *_ohlcv_cols())
        Table("ohlcv_1d", self._metadata, *_ohlcv_cols())

        # --- funding_rates ------------------------------------------------
        Table(
            "funding_rates",
            self._metadata,
            Column("timestamp_ms", BigInteger(), nullable=False, primary_key=True),
            Column("symbol", String(20), nullable=False),
            Column("funding_rate", Numeric(12, 8), nullable=False),
            Column("mark_price", Numeric(18, 8)),
        )

        # --- regime_labels ------------------------------------------------
        Table(
            "regime_labels",
            self._metadata,
            Column("open_time", BigInteger(), nullable=False),
            Column("symbol", String(20), nullable=False),
            Column("regime", String(20), nullable=False),
            Column("btc_corr_24h", Numeric(8, 4)),
            Column("bb_width", Numeric(8, 6)),
            Column("atr_norm", Numeric(8, 6)),
            sa.PrimaryKeyConstraint("symbol", "open_time"),
        )

        # --- doge_predictions ---------------------------------------------
        Table(
            "doge_predictions",
            self._metadata,
            Column("prediction_id", String(64), nullable=False, primary_key=True),
            Column("created_at", BigInteger(), nullable=False),
            Column("open_time", BigInteger(), nullable=False),
            Column("symbol", String(20), nullable=False),
            Column("horizon_label", String(10), nullable=False),
            Column("horizon_candles", Integer(), nullable=False),
            Column("target_open_time", BigInteger(), nullable=False),
            Column("price_at_prediction", Numeric(18, 8), nullable=False),
            Column("predicted_direction", SmallInteger(), nullable=False),
            Column("confidence_score", Numeric(6, 4), nullable=False),
            Column("lstm_prob", Numeric(6, 4), nullable=False),
            Column("xgb_prob", Numeric(6, 4), nullable=False),
            Column("regime_label", String(20), nullable=False),
            Column("model_version", String(64), nullable=False),
            # Outcome fields — initially NULL, filled by PredictionVerifier
            Column("actual_price", Numeric(18, 8)),
            Column("actual_direction", SmallInteger()),
            Column("reward_score", Numeric(10, 6)),
            Column("direction_correct", Boolean()),
            Column("error_pct", Numeric(10, 6)),
            Column("verified_at", BigInteger()),
        )

        # --- doge_replay_buffer -------------------------------------------
        # abs_reward is GENERATED in TimescaleDB production; in SQLite tests
        # we compute it on the Python side before insert.
        Table(
            "doge_replay_buffer",
            self._metadata,
            Column("buffer_id", String(64), nullable=False, primary_key=True),
            Column("horizon_label", String(10), nullable=False),
            Column("regime", String(20), nullable=False),
            Column("feature_vector", LargeBinary(), nullable=True),
            Column("predicted_price", Numeric(18, 8)),
            Column("actual_price", Numeric(18, 8)),
            Column("reward_score", Numeric(10, 6), nullable=False),
            Column("model_version", String(64), nullable=False),
            Column("created_at", BigInteger(), nullable=False),
            # Python-computed; GENERATED ALWAYS AS in production DDL
            Column("abs_reward", Numeric(10, 6), nullable=False),
        )

    # -----------------------------------------------------------------------
    # Schema management (tests / dev only)
    # -----------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all tables via SQLAlchemy metadata (tests / development).

        In production, run ``scripts/create_tables.sql`` instead so that
        TimescaleDB hypertables and proper indexes are created.

        Raises:
            SQLAlchemyError: If any DDL statement fails.
        """
        try:
            self._metadata.create_all(self._engine)
            logger.info("create_tables: all tables created via metadata.create_all()")
        except SQLAlchemyError as exc:
            logger.error("create_tables failed: {}", exc)
            raise

    # -----------------------------------------------------------------------
    # Internal: dialect-aware upsert
    # -----------------------------------------------------------------------

    def _upsert(
        self,
        table: Table,
        rows: list[dict[str, Any]],
        *,
        pk_cols: Optional[list[str]] = None,
    ) -> int:
        """Execute a dialect-aware upsert (INSERT … ON CONFLICT DO UPDATE).

        Args:
            table: Target SQLAlchemy ``Table`` object.
            rows: List of row dicts. Must be non-empty.
            pk_cols: Primary key column names. When ``None``, inferred from
                ``table.primary_key``.

        Returns:
            Number of rows affected (may be 0 for no-op updates, dialect-
            dependent).

        Raises:
            ValueError: If the database dialect is not supported.
            SQLAlchemyError: On any database error.
        """
        if not rows:
            return 0

        dialect = self._engine.dialect.name
        conflict_cols = pk_cols or [c.name for c in table.primary_key.columns]
        non_pk = [c.name for c in table.columns if c.name not in conflict_cols]

        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as _insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as _insert
        else:
            raise ValueError(f"Unsupported database dialect: {dialect!r}")

        stmt = _insert(table).values(rows)
        if non_pk:
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_={col: stmt.excluded[col] for col in non_pk},
            )
        else:
            # Only PK columns — use DO NOTHING (pure dedup insert)
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)

        with self._engine.begin() as conn:
            result = conn.execute(stmt)

        return result.rowcount

    # -----------------------------------------------------------------------
    # OHLCV
    # -----------------------------------------------------------------------

    def upsert_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> int:
        """Upsert OHLCV rows into the appropriate interval table.

        Rows are deduplicated on ``(symbol, open_time)``. Any row that already
        exists is updated with the incoming values (e.g. after a data repair).

        Args:
            df: DataFrame with columns matching the ``ohlcv_*`` table schema.
                Must include at minimum: ``open_time``, ``open``, ``high``,
                ``low``, ``close``, ``volume``, ``era``.
            symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
            interval: Candle interval — must be one of ``"1h"``, ``"4h"``,
                ``"1d"``.

        Returns:
            Number of rows affected by the upsert.

        Raises:
            ValueError: If *interval* is not a recognised OHLCV interval.
            SQLAlchemyError: On any database error.
        """
        if interval not in _OHLCV_TABLE_NAMES:
            raise ValueError(
                f"Unknown OHLCV interval {interval!r}. "
                f"Must be one of {sorted(_OHLCV_TABLE_NAMES)!r}."
            )

        table_name = _OHLCV_TABLE_NAMES[interval]
        table = self._metadata.tables[table_name]

        rows = df.copy().to_dict(orient="records")
        for row in rows:
            row["symbol"] = symbol

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                n = self._upsert(table, rows)
        except FileLockTimeout:
            logger.error(
                "upsert_ohlcv: timed out acquiring write lock after {}s",
                _LOCK_TIMEOUT_S,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error(
                "upsert_ohlcv failed (symbol={}, interval={}): {}",
                symbol,
                interval,
                exc,
            )
            raise

        logger.debug(
            "upsert_ohlcv: {} rows upserted (symbol={}, interval={})",
            n,
            symbol,
            interval,
        )
        return n

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        era: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV rows for a symbol within a time range.

        Args:
            symbol: Trading pair symbol (e.g. ``"DOGEUSDT"``).
            interval: Candle interval — ``"1h"``, ``"4h"``, or ``"1d"``.
            start_ms: Inclusive start timestamp, UTC epoch milliseconds.
            end_ms: Exclusive end timestamp, UTC epoch milliseconds.
            era: Optional era filter — ``"context"`` or ``"training"``.
                When ``None``, all eras are returned.

        Returns:
            DataFrame sorted ascending by ``open_time``. Empty DataFrame if
            no rows match.

        Raises:
            ValueError: If *interval* is not recognised.
            SQLAlchemyError: On any database error.
        """
        if interval not in _OHLCV_TABLE_NAMES:
            raise ValueError(
                f"Unknown OHLCV interval {interval!r}. "
                f"Must be one of {sorted(_OHLCV_TABLE_NAMES)!r}."
            )

        table = self._metadata.tables[_OHLCV_TABLE_NAMES[interval]]
        t = table

        conditions = [
            t.c.symbol == symbol,
            t.c.open_time >= start_ms,
            t.c.open_time < end_ms,
        ]
        if era is not None:
            conditions.append(t.c.era == era)

        stmt = (
            sa.select(t)
            .where(sa.and_(*conditions))
            .order_by(t.c.open_time.asc())
        )

        try:
            with self._engine.connect() as conn:
                result = conn.execute(stmt)
                rows = result.mappings().all()
        except SQLAlchemyError as exc:
            logger.error(
                "get_ohlcv failed (symbol={}, interval={}): {}",
                symbol,
                interval,
                exc,
            )
            raise

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        for col in _OHLCV_NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # -----------------------------------------------------------------------
    # Funding rates
    # -----------------------------------------------------------------------

    def upsert_funding_rates(self, df: pd.DataFrame) -> int:
        """Upsert 8h funding rate rows.

        Deduplicates on ``timestamp_ms``. Existing rows are updated on
        conflict (e.g. after a data correction from Binance).

        Args:
            df: DataFrame with columns: ``timestamp_ms``, ``symbol``,
                ``funding_rate``, and optionally ``mark_price``.

        Returns:
            Number of rows affected.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["funding_rates"]
        rows = df.copy().to_dict(orient="records")

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                n = self._upsert(table, rows, pk_cols=["timestamp_ms"])
        except FileLockTimeout:
            logger.error(
                "upsert_funding_rates: timed out acquiring write lock after {}s",
                _LOCK_TIMEOUT_S,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error("upsert_funding_rates failed: {}", exc)
            raise

        logger.debug("upsert_funding_rates: {} rows upserted", n)
        return n

    def get_funding_rates(
        self,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch funding rate rows within a time range.

        Args:
            start_ms: Inclusive start timestamp, UTC epoch milliseconds.
            end_ms: Exclusive end timestamp, UTC epoch milliseconds.

        Returns:
            DataFrame sorted ascending by ``timestamp_ms``. Empty if no rows.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["funding_rates"]
        t = table

        stmt = (
            sa.select(t)
            .where(
                sa.and_(
                    t.c.timestamp_ms >= start_ms,
                    t.c.timestamp_ms < end_ms,
                )
            )
            .order_by(t.c.timestamp_ms.asc())
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.error("get_funding_rates failed: {}", exc)
            raise

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        # Coerce numeric columns — SQLAlchemy may return Decimal objects
        for col in ("funding_rate", "mark_price", "timestamp_ms"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # -----------------------------------------------------------------------
    # Regime labels
    # -----------------------------------------------------------------------

    def upsert_regime_labels(self, df: pd.DataFrame) -> int:
        """Upsert regime label rows.

        Deduplicates on ``(symbol, open_time)``.

        Args:
            df: DataFrame with columns: ``open_time``, ``symbol``, ``regime``,
                and optionally ``btc_corr_24h``, ``bb_width``, ``atr_norm``.

        Returns:
            Number of rows affected.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["regime_labels"]
        rows = df.copy().to_dict(orient="records")

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                n = self._upsert(table, rows)
        except FileLockTimeout:
            logger.error(
                "upsert_regime_labels: timed out acquiring write lock after {}s",
                _LOCK_TIMEOUT_S,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error("upsert_regime_labels failed: {}", exc)
            raise

        logger.debug("upsert_regime_labels: {} rows upserted", n)
        return n

    def get_regime_labels(
        self,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch regime label rows within a time range.

        Args:
            start_ms: Inclusive start timestamp, UTC epoch milliseconds.
            end_ms: Exclusive end timestamp, UTC epoch milliseconds.

        Returns:
            DataFrame sorted ascending by ``open_time``. Empty if no rows.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["regime_labels"]
        t = table

        stmt = (
            sa.select(t)
            .where(
                sa.and_(
                    t.c.open_time >= start_ms,
                    t.c.open_time < end_ms,
                )
            )
            .order_by(t.c.open_time.asc())
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.error("get_regime_labels failed: {}", exc)
            raise

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # -----------------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------------

    def insert_prediction(self, record: PredictionRecord) -> str:
        """Insert a single prediction record (upsert semantics).

        Prediction fields are immutable after insert; if the same
        ``prediction_id`` already exists the row is not updated.

        Args:
            record: Validated ``PredictionRecord`` instance.

        Returns:
            The ``prediction_id`` of the inserted record.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_predictions"]
        row = record.model_dump()

        # Ensure outcome fields that are None stay NULL (not 0 / False)
        nullable_outcome_cols = (
            "actual_price",
            "actual_direction",
            "reward_score",
            "direction_correct",
            "error_pct",
            "verified_at",
        )
        for col in nullable_outcome_cols:
            if col not in row:
                row[col] = None

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                with self._engine.begin() as conn:
                    dialect = self._engine.dialect.name
                    if dialect == "postgresql":
                        from sqlalchemy.dialects.postgresql import insert as _insert
                    else:
                        from sqlalchemy.dialects.sqlite import insert as _insert  # type: ignore[no-redef]

                    stmt = _insert(table).values([row]).on_conflict_do_nothing(
                        index_elements=["prediction_id"]
                    )
                    conn.execute(stmt)
        except FileLockTimeout:
            logger.error(
                "insert_prediction: timed out acquiring write lock after {}s",
                _LOCK_TIMEOUT_S,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error(
                "insert_prediction failed (id={}): {}",
                record.prediction_id,
                exc,
            )
            raise

        logger.debug(
            "insert_prediction: {} (horizon={}, regime={})",
            record.prediction_id,
            record.horizon_label,
            record.regime_label,
        )
        return record.prediction_id

    def update_prediction_outcome(
        self,
        prediction_id: str,
        outcome: dict[str, Any],
    ) -> bool:
        """Write verified outcome fields to an existing prediction row.

        Only the Verifier should call this. Outcome fields (``actual_price``,
        ``reward_score``, ``direction_correct``, ``error_pct``,
        ``verified_at``) are set from *outcome*.

        Args:
            prediction_id: UUID string of the prediction to update.
            outcome: Dict of column-name → value for the outcome fields.
                Keys not present in the table are silently ignored.

        Returns:
            ``True`` if exactly one row was updated; ``False`` if no matching
            row was found (prediction_id not in table).

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_predictions"]
        valid_cols = {c.name for c in table.columns}
        safe_outcome = {k: v for k, v in outcome.items() if k in valid_cols}

        if not safe_outcome:
            logger.warning(
                "update_prediction_outcome: no valid columns in outcome dict "
                "for prediction_id={}",
                prediction_id,
            )
            return False

        stmt = (
            sa.update(table)
            .where(table.c.prediction_id == prediction_id)
            .values(**safe_outcome)
        )

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                with self._engine.begin() as conn:
                    result = conn.execute(stmt)
        except FileLockTimeout:
            logger.error(
                "update_prediction_outcome: timed out acquiring write lock "
                "for prediction_id={}",
                prediction_id,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error(
                "update_prediction_outcome failed (id={}): {}",
                prediction_id,
                exc,
            )
            raise

        updated = result.rowcount > 0
        logger.debug(
            "update_prediction_outcome: id={} updated={}",
            prediction_id,
            updated,
        )
        return updated

    def get_matured_unverified(self, as_of_ts: int) -> list[PredictionRecord]:
        """Return predictions whose target time has passed but are unverified.

        These are the rows that the ``PredictionVerifier`` needs to process:
        their ``target_open_time <= as_of_ts`` and ``verified_at IS NULL``.

        Args:
            as_of_ts: Current timestamp, UTC epoch milliseconds. All
                predictions with ``target_open_time <= as_of_ts`` are
                considered matured.

        Returns:
            List of ``PredictionRecord`` instances, sorted ascending by
            ``target_open_time``. Empty list if none found.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_predictions"]
        t = table

        stmt = (
            sa.select(t)
            .where(
                sa.and_(
                    t.c.target_open_time <= as_of_ts,
                    t.c.verified_at.is_(None),
                )
            )
            .order_by(t.c.target_open_time.asc())
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.error("get_matured_unverified failed: {}", exc)
            raise

        records: list[PredictionRecord] = []
        for row in rows:
            try:
                records.append(PredictionRecord.model_validate(dict(row)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "get_matured_unverified: could not parse row "
                    "(prediction_id={}): {}",
                    row.get("prediction_id", "?"),
                    exc,
                )

        return records

    # -----------------------------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------------------------

    def push_replay_buffer(self, record: dict[str, Any]) -> bool:
        """Append one experience record to the replay buffer.

        ``abs_reward`` is computed on the Python side (equal to
        ``abs(reward_score)``) so the table stays compatible with both the
        SQLite test backend (no generated columns) and the TimescaleDB
        production backend (where it is ``GENERATED ALWAYS AS``).

        Args:
            record: Dict with keys: ``buffer_id``, ``horizon_label``,
                ``regime``, ``feature_vector`` (bytes), ``reward_score``,
                ``model_version``, ``created_at``, and optionally
                ``predicted_price``, ``actual_price``.

        Returns:
            ``True`` if the row was inserted successfully.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_replay_buffer"]
        row = dict(record)

        # Compute abs_reward on the Python side
        reward = float(row.get("reward_score", 0.0))
        row["abs_reward"] = abs(reward)

        try:
            with FileLock(str(self._lock_path), timeout=_LOCK_TIMEOUT_S):
                dialect = self._engine.dialect.name
                if dialect == "postgresql":
                    from sqlalchemy.dialects.postgresql import insert as _insert
                else:
                    from sqlalchemy.dialects.sqlite import insert as _insert  # type: ignore[no-redef]

                with self._engine.begin() as conn:
                    stmt = _insert(table).values([row]).on_conflict_do_nothing(
                        index_elements=["buffer_id"]
                    )
                    conn.execute(stmt)
        except FileLockTimeout:
            logger.error(
                "push_replay_buffer: timed out acquiring write lock after {}s",
                _LOCK_TIMEOUT_S,
            )
            raise
        except SQLAlchemyError as exc:
            logger.error("push_replay_buffer failed: {}", exc)
            raise

        logger.debug(
            "push_replay_buffer: horizon={} regime={} reward={}",
            row.get("horizon_label"),
            row.get("regime"),
            reward,
        )
        return True

    def get_replay_sample(
        self,
        horizon: str,
        n: int,
    ) -> pd.DataFrame:
        """Draw a random sample from the replay buffer for a given horizon.

        Sampling is uniform random (``ORDER BY RANDOM()``), which avoids
        selection bias. Priority-weighted sampling is handled at the Python
        level by the ``ReplayBuffer`` class in ``src/rl/replay_buffer.py``.

        Args:
            horizon: Horizon label — ``"SHORT"``, ``"MEDIUM"``, ``"LONG"``,
                or ``"MACRO"``.
            n: Maximum number of rows to return.

        Returns:
            DataFrame with replay buffer columns. Empty if the buffer contains
            fewer than 1 row for the requested horizon.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_replay_buffer"]
        t = table

        stmt = (
            sa.select(t)
            .where(t.c.horizon_label == horizon)
            .order_by(sa.func.random())
            .limit(n)
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.error(
                "get_replay_sample failed (horizon={}, n={}): {}",
                horizon,
                n,
                exc,
            )
            raise

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def get_prediction_by_id(self, prediction_id: str) -> "PredictionRecord | None":
        """Fetch a single prediction record by its primary key.

        Used by the verifier's immutability guard to re-read the stored values
        and confirm no prediction field has changed since insertion.

        Args:
            prediction_id: UUID string matching ``doge_predictions.prediction_id``.

        Returns:
            :class:`~src.processing.schemas.PredictionRecord` if found, else
            ``None``.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_predictions"]
        stmt = sa.select(table).where(table.c.prediction_id == prediction_id)

        try:
            with self._engine.connect() as conn:
                row = conn.execute(stmt).mappings().first()
        except SQLAlchemyError as exc:
            logger.error("get_prediction_by_id failed (id={}): {}", prediction_id, exc)
            raise

        if row is None:
            return None
        try:
            return PredictionRecord.model_validate(dict(row))
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_prediction_by_id: parse error for {}: {}", prediction_id, exc)
            return None

    def get_replay_regime_counts(self, horizon: str) -> dict[str, int]:
        """Return per-regime record counts for the given horizon.

        Used by the replay buffer's eviction logic to identify protected regimes
        (those below the ``min_per_regime`` threshold).

        Args:
            horizon: Horizon label — ``"SHORT"``, ``"MEDIUM"``, ``"LONG"``, or
                ``"MACRO"``.

        Returns:
            Dict mapping regime label → row count. Only regimes with ≥ 1 row
            are included.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_replay_buffer"]
        stmt = (
            sa.select(table.c.regime, sa.func.count().label("cnt"))
            .where(table.c.horizon_label == horizon)
            .group_by(table.c.regime)
        )

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).mappings().all()
        except SQLAlchemyError as exc:
            logger.error(
                "get_replay_regime_counts failed (horizon={}): {}",
                horizon,
                exc,
            )
            raise

        return {str(row["regime"]): int(row["cnt"]) for row in rows}

    def delete_oldest_non_protected_replay(
        self,
        horizon: str,
        protected_regimes: set[str],
    ) -> bool:
        """Delete the oldest replay buffer row not in ``protected_regimes``.

        If ALL rows belong to protected regimes the oldest row overall is
        deleted as a last-resort fallback (buffer must not grow past capacity).

        Args:
            horizon: Horizon label to evict from.
            protected_regimes: Set of regime labels whose rows must not be
                evicted first (they are below the ``min_per_regime`` quota).

        Returns:
            ``True`` if a row was deleted; ``False`` if the buffer was empty.

        Raises:
            SQLAlchemyError: On any database error.
        """
        table = self._metadata.tables["doge_replay_buffer"]

        def _find_oldest(conn: sa.Connection, extra_where: sa.ColumnElement | None) -> str | None:
            base = sa.select(table.c.buffer_id).where(table.c.horizon_label == horizon)
            if extra_where is not None:
                base = base.where(extra_where)
            row = conn.execute(base.order_by(table.c.created_at.asc()).limit(1)).mappings().first()
            return str(row["buffer_id"]) if row else None

        try:
            with self._engine.begin() as conn:
                # First try: evict from non-protected regimes
                if protected_regimes:
                    not_protected = ~table.c.regime.in_(list(protected_regimes))
                else:
                    not_protected = None
                buffer_id = _find_oldest(conn, not_protected)

                # Fallback: all regimes are protected — evict the overall oldest
                if buffer_id is None:
                    buffer_id = _find_oldest(conn, None)

                if buffer_id is None:
                    return False

                conn.execute(sa.delete(table).where(table.c.buffer_id == buffer_id))
        except SQLAlchemyError as exc:
            logger.error(
                "delete_oldest_non_protected_replay failed (horizon={}): {}",
                horizon,
                exc,
            )
            raise

        logger.debug(
            "delete_oldest_non_protected_replay: evicted buffer_id={} (horizon={})",
            buffer_id,
            horizon,
        )
        return True

    # -----------------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------------

    def dispose(self) -> None:
        """Release all pooled DB connections.

        Call at application shutdown or between tests to prevent connection
        leaks.
        """
        self._engine.dispose()
        logger.debug("DogeStorage.dispose: connection pool released")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "DogeStorage",
    "guard_raw_write",
]
