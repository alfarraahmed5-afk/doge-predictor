"""serve.py — Production inference server entry point.

Starts the live DOGE price-prediction inference engine:

  * Binance WebSocket client streaming ``dogeusdt@kline_1h``
  * :class:`~src.inference.engine.InferenceEngine` triggered on each
    closed candle via :meth:`~src.inference.engine.InferenceEngine.run_on_closed_kline`
  * :class:`~src.monitoring.health_check.HealthCheckServer` on ``--health-port`` (default 8000)
  * Prometheus metrics HTTP endpoint on ``--metrics-port`` (default 8001)
  * APScheduler background jobs:

    - ``:01`` past every hour — :class:`~src.ingestion.scheduler.IncrementalScheduler`
      (REST top-up to keep storage current for all symbols)
    - ``:02`` past every hour — :class:`~src.rl.verifier.PredictionVerifier` (Phase 9 stub)
    - Sunday ``02:00 UTC``  — placeholder for model self-retraining

  * Graceful SIGTERM/SIGINT shutdown with 10-second drain

Usage::

    python scripts/serve.py --models-dir models/ --db-path data/doge_data.db
    python scripts/serve.py --models-dir models/  # uses TimescaleDB from settings.yaml

Environment variables (override settings.yaml):
    DOGE_DB_URL   — full SQLAlchemy database URL
    HEALTH_PORT   — health check HTTP port
    METRICS_PORT  — Prometheus metrics HTTP port
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Guard: must be run from the project root so that `src` is importable.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import doge_settings, get_settings  # noqa: E402
from src.inference.engine import InferenceEngine, StaleDataError  # noqa: E402
from src.inference.signal import SignalEvent  # noqa: E402
from src.ingestion.rest_client import BinanceRESTClient  # noqa: E402
from src.ingestion.scheduler import IncrementalScheduler  # noqa: E402
from src.ingestion.ws_client import BinanceWebSocketClient  # noqa: E402
from src.monitoring.alerting import AlertManager  # noqa: E402
from src.monitoring.health_check import HealthCheckServer, HealthStatus  # noqa: E402
from src.processing.storage import DogeStorage  # noqa: E402
from src.processing.validator import DataValidator  # noqa: E402
from src.rl.curriculum import CurriculumManager  # noqa: E402
from src.rl.predictor import MultiHorizonPredictor  # noqa: E402
from src.rl.verifier import PredictionVerifier  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# ---------------------------------------------------------------------------
# Prometheus metrics  (all registered at module level so they persist)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server as _prom_start

    _SIGNALS_TOTAL = Counter(
        "doge_signals_total",
        "Total inference signals emitted",
        ["signal", "regime"],
    )
    _INFERENCE_LATENCY = Histogram(
        "doge_inference_latency_seconds",
        "Wall-clock seconds from kline-close callback to signal emit",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    _CANDLE_AGE = Gauge(
        "doge_last_candle_age_seconds",
        "Seconds since the last successfully processed closed candle",
    )
    _WS_CONNECTED = Gauge(
        "doge_ws_connected",
        "1 when the WebSocket is connected, 0 otherwise",
    )
    _INFERENCE_ERRORS = Counter(
        "doge_inference_errors_total",
        "Total inference pipeline errors by type",
        ["error_type"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — metrics endpoint disabled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_storage(db_path: Optional[str], settings_obj: Any) -> DogeStorage:
    """Build a :class:`~src.processing.storage.DogeStorage` instance.

    Uses SQLite when *db_path* is provided (dev / single-host mode), otherwise
    falls back to the TimescaleDB URL from ``settings.yaml``.

    Args:
        db_path: Path to SQLite file, or ``None`` for PostgreSQL.
        settings_obj: Loaded :class:`~src.config.Settings` object.

    Returns:
        Initialised :class:`DogeStorage` instance.
    """
    if db_path:
        from sqlalchemy import create_engine as _create_engine

        engine = _create_engine(f"sqlite:///{db_path}", future=True)
        return DogeStorage(settings_obj, engine=engine)
    return DogeStorage(settings_obj)


def _make_kline_callback(
    symbol: str,
    engine: InferenceEngine,
    health_status: HealthStatus,
    alert_mgr: AlertManager,
    multi_horizon_predictor: Optional["MultiHorizonPredictor"] = None,
) -> Any:
    """Return a kline callback bound to *engine* for *symbol*.

    The callback is registered with
    :meth:`~src.ingestion.ws_client.BinanceWebSocketClient.subscribe_klines`.
    Only closed candles (``k.x == True``) trigger inference; open ticks are
    silently ignored.

    For non-DOGEUSDT symbols the callback is a no-op — BTCUSDT and DOGEBTC are
    maintained in storage by the :class:`~src.ingestion.scheduler.IncrementalScheduler`
    and will be read from storage by :meth:`~src.inference.engine.InferenceEngine.run_on_closed_kline`.

    Args:
        symbol: Lowercase symbol string, e.g. ``"dogeusdt"``.
        engine: Inference engine to invoke on candle close.
        health_status: Shared health state updated after each inference run.
        alert_mgr: Alert dispatcher for error notifications.
        multi_horizon_predictor: Optional :class:`~src.rl.predictor.MultiHorizonPredictor`
            that generates and stores multi-horizon RL prediction records after
            each closed candle.  When ``None`` the RL prediction step is skipped.

    Returns:
        Callable that accepts a raw kline dict from the WebSocket.
    """

    def _callback(msg: dict[str, Any]) -> None:
        k = msg.get("k", {})
        if not k.get("x", False):
            return  # candle not yet closed

        if symbol.lower() != "dogeusdt":
            # BTCUSDT / DOGEBTC: data stored by IncrementalScheduler; no inference needed
            return

        t0 = time.monotonic()
        try:
            event: Optional[SignalEvent] = engine.run_on_closed_kline(msg)
        except StaleDataError as exc:
            logger.warning("serve: StaleDataError — {}", exc)
            if _PROMETHEUS_AVAILABLE:
                _INFERENCE_ERRORS.labels(error_type="stale_data").inc()
            alert_mgr.send_alert(
                "WARNING",
                "Stale candle detected",
                {"error": str(exc)},
            )
            return
        except Exception as exc:
            logger.error("serve: unexpected inference error — {}", exc)
            if _PROMETHEUS_AVAILABLE:
                _INFERENCE_ERRORS.labels(error_type="unexpected").inc()
            alert_mgr.send_alert(
                "CRITICAL",
                "Inference pipeline error",
                {"error": str(exc)},
            )
            return

        latency = time.monotonic() - t0
        if _PROMETHEUS_AVAILABLE:
            _INFERENCE_LATENCY.observe(latency)

        if event is not None:
            health_status.update_from_signal(event)
            if _PROMETHEUS_AVAILABLE:
                _SIGNALS_TOTAL.labels(
                    signal=event.signal, regime=event.regime
                ).inc()
                _CANDLE_AGE.set(0.0)
            logger.info(
                "serve: signal={} regime={} prob={:.4f} latency={:.3f}s",
                event.signal,
                event.regime,
                event.ensemble_prob,
                latency,
            )

            # RL Step 12b — multi-horizon prediction records (Phase 9)
            if multi_horizon_predictor is not None:
                try:
                    direction = (
                        1 if event.signal == "BUY"
                        else -1 if event.signal == "SELL"
                        else 0
                    )
                    multi_horizon_predictor.generate_and_store(
                        open_time=event.open_time,
                        close_price=event.close_price,
                        predicted_direction=direction,
                        ensemble_prob=event.ensemble_prob,
                        lstm_prob=event.lstm_prob,
                        xgb_prob=event.xgb_prob,
                        regime_label=event.regime,
                        model_version=event.model_version,
                        now_ms=event.timestamp_ms,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "serve: MultiHorizonPredictor error (non-fatal): {}", exc
                    )

    return _callback


def _make_incremental_job(
    scheduler_obj: IncrementalScheduler,
    health_status: HealthStatus,
) -> Any:
    """Return a job callable for the APScheduler incremental update trigger.

    Args:
        scheduler_obj: :class:`~src.ingestion.scheduler.IncrementalScheduler`.
        health_status: Shared health state (db_connected updated on failure).

    Returns:
        Zero-argument callable suitable for APScheduler.
    """

    def _job() -> None:
        try:
            stats = scheduler_obj.run_once()
            logger.info(
                "IncrementalScheduler: +{} new, {} updated candles",
                stats.candles_new,
                stats.candles_updated,
            )
            health_status.db_connected = True
        except Exception as exc:
            logger.error("IncrementalScheduler error: {}", exc)
            health_status.db_connected = False

    return _job


def _make_verifier_job(verifier: PredictionVerifier) -> Any:
    """Return a job callable for the APScheduler PredictionVerifier trigger.

    Args:
        verifier: :class:`~src.rl.verifier.PredictionVerifier` instance.

    Returns:
        Zero-argument callable suitable for APScheduler.
    """

    def _job() -> None:
        try:
            n = verifier.run_verification()
            if n > 0:
                logger.info("PredictionVerifier: {} predictions verified", n)
        except Exception as exc:
            logger.error("PredictionVerifier error: {}", exc)

    return _job


def _make_retrain_job(storage: Any, models_dir: Path) -> Any:
    """Return the weekly retraining job callable.

    Delegates to :func:`~src.training.trainer.retrain_weekly`.  A failure
    inside ``retrain_weekly`` is caught and logged — the scheduler must not
    crash due to a single training run failure.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.
        models_dir: Directory to write the new model artefacts to.

    Returns:
        Zero-argument callable suitable for APScheduler.
    """

    def _job() -> None:
        logger.info("serve: weekly retrain job triggered")
        try:
            from src.training.trainer import retrain_weekly  # noqa: PLC0415

            success = retrain_weekly(
                storage=storage,
                output_dir=models_dir / "weekly_candidate",
            )
            if success:
                logger.info(
                    "serve: weekly retrain succeeded — candidate model ready. "
                    "Enable SHADOW_MODE and run QG-09 after 48h."
                )
            else:
                logger.warning(
                    "serve: weekly retrain did not improve over production model"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("serve: weekly retrain job failed — {}", exc)

    return _job


def _make_round_number_review_job() -> Any:
    """Return a monthly round-number config review job.

    Checks current DOGE price against configured round-number levels and
    emits a WARNING alert if the price has moved significantly above the
    highest configured level (suggesting config update is needed).

    Returns:
        Zero-argument callable suitable for APScheduler.
    """

    def _job() -> None:
        logger.info("serve: monthly round-number level review triggered")
        try:
            from src.config import doge_settings as _ds  # noqa: PLC0415

            levels = _ds.round_number_levels
            if levels:
                max_level = max(levels)
                logger.info(
                    "serve: round-number review — current levels: {} "
                    "(highest: ${:.2f})",
                    [f"${v:.2f}" for v in sorted(levels)],
                    max_level,
                )
                logger.info(
                    "serve: if DOGE price is consistently above ${:.2f}, "
                    "update config/doge_settings.yaml round_number_levels",
                    max_level,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("serve: round-number review job failed — {}", exc)

    return _job


def _make_prediction_backup_job(storage: Any) -> Any:
    """Return a daily prediction store backup job.

    Exports the last 48 hours of prediction records to a Parquet file in
    ``data/predictions/`` for offline analysis and audit.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.

    Returns:
        Zero-argument callable suitable for APScheduler.
    """

    def _job() -> None:
        logger.info("serve: daily prediction backup triggered")
        try:
            import time as _t  # noqa: PLC0415
            from pathlib import Path as _Path  # noqa: PLC0415
            import pandas as _pd  # noqa: PLC0415

            now_ms = int(_t.time() * 1_000)
            start_ms = now_ms - 48 * 3_600_000  # last 48h

            predictions = storage.get_matured_unverified(now_ms)
            if not predictions:
                logger.info("serve: prediction backup — no records to export")
                return

            rows = [p.__dict__ if hasattr(p, "__dict__") else dict(p) for p in predictions]
            df = _pd.DataFrame(rows)

            backup_dir = _Path("data") / "predictions"
            backup_dir.mkdir(parents=True, exist_ok=True)

            import datetime as _dt  # noqa: PLC0415
            ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_path = backup_dir / f"predictions_backup_{ts}.parquet"
            df.to_parquet(out_path, index=False)

            logger.info(
                "serve: prediction backup saved {} records to {}",
                len(df),
                out_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("serve: prediction backup job failed — {}", exc)

    return _job


def _probe_db(storage: DogeStorage, health_status: HealthStatus) -> None:
    """Run a lightweight DB health probe and update *health_status*.

    Opens a short-lived connection via SQLAlchemy and immediately closes it.
    Sets ``health_status.db_connected`` accordingly.

    Args:
        storage: Live :class:`~src.processing.storage.DogeStorage` instance.
        health_status: Shared health state to update.
    """
    try:
        with storage._engine.connect() as conn:  # noqa: SLF001 — internal attr; acceptable for probe
            from sqlalchemy import text as _text
            conn.execute(_text("SELECT 1"))
        health_status.db_connected = True
    except Exception as exc:
        logger.warning("DB health probe failed: {}", exc)
        health_status.db_connected = False


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="serve.py",
        description="DOGE Predictor — live inference server",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model artefacts (default: models/)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite DB file (dev mode). Omit to use TimescaleDB from settings.yaml.",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        default=int(os.environ.get("HEALTH_PORT", "8000")),
        help="Port for GET /health endpoint (default: 8000)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("METRICS_PORT", "8001")),
        help="Port for Prometheus /metrics endpoint (default: 8001)",
    )
    parser.add_argument(
        "--no-ws",
        action="store_true",
        help="Disable WebSocket client (inference only via scheduler REST top-up)",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable APScheduler background jobs",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="unknown",
        help="Model version tag stored in every PredictionRecord (default: unknown)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the inference server.

    Returns:
        Exit code — 0 on clean shutdown, 1 on fatal startup error.
    """
    configure_logging()

    parser = _build_parser()
    args = parser.parse_args()

    logger.info("serve.py starting — models_dir={} health_port={} metrics_port={}",
                args.models_dir, args.health_port, args.metrics_port)

    # ------------------------------------------------------------------
    # Shared shutdown event
    # ------------------------------------------------------------------
    shutdown_event = threading.Event()

    def _handle_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        logger.info("serve: received signal {} — initiating shutdown", signum)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # ------------------------------------------------------------------
    # Config & storage
    # ------------------------------------------------------------------
    cfg = get_settings()
    storage: Optional[DogeStorage] = None

    try:
        storage = _build_storage(args.db_path, cfg)
        logger.info("serve: storage initialised ({})",
                    f"sqlite:{args.db_path}" if args.db_path else "timescaledb")
    except Exception as exc:
        logger.error("serve: FATAL — could not connect to storage: {}", exc)
        return 1

    # ------------------------------------------------------------------
    # Shared health status
    # ------------------------------------------------------------------
    health_status = HealthStatus(interval=doge_settings.primary_interval)
    alert_mgr = AlertManager(log_dir=Path("logs"))

    # ------------------------------------------------------------------
    # Health check HTTP server (always started — even if engine fails)
    # ------------------------------------------------------------------
    health_server = HealthCheckServer(
        health_status,
        port=args.health_port,
        interval=doge_settings.primary_interval,
    )
    health_server.start()

    # ------------------------------------------------------------------
    # Prometheus metrics HTTP server
    # ------------------------------------------------------------------
    if _PROMETHEUS_AVAILABLE:
        try:
            _prom_start(args.metrics_port)
            logger.info("serve: Prometheus metrics on port {}", args.metrics_port)
        except Exception as exc:
            logger.warning("serve: could not start Prometheus metrics server: {}", exc)

    # ------------------------------------------------------------------
    # Inference engine  (non-fatal if models are not yet trained)
    # ------------------------------------------------------------------
    engine: Optional[InferenceEngine] = None
    if args.models_dir.exists():
        try:
            engine = InferenceEngine.from_artifacts(
                models_dir=args.models_dir,
                model_version=args.model_version,
                storage=storage,
            )
            logger.info("serve: InferenceEngine loaded from {}", args.models_dir)
        except Exception as exc:
            logger.warning(
                "serve: InferenceEngine could not be loaded — "
                "inference disabled until models are present. Error: {}",
                exc,
            )
            alert_mgr.send_alert(
                "WARNING",
                "InferenceEngine failed to load — inference disabled",
                {"models_dir": str(args.models_dir), "error": str(exc)},
            )
    else:
        logger.warning(
            "serve: models_dir {} not found — inference disabled",
            args.models_dir,
        )

    # ------------------------------------------------------------------
    # RL verifier + multi-horizon predictor (Phase 9)
    # ------------------------------------------------------------------
    verifier = PredictionVerifier(storage)

    try:
        from src.config import rl_config as _rl_cfg  # noqa: PLC0415
        _curriculum = CurriculumManager(rl_cfg=_rl_cfg)
        multi_horizon_predictor: Optional[MultiHorizonPredictor] = MultiHorizonPredictor(
            storage=storage,
            curriculum=_curriculum,
            rl_cfg=_rl_cfg,
        )
        logger.info(
            "serve: MultiHorizonPredictor initialised (active_horizons={})",
            _curriculum.active_horizons(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "serve: MultiHorizonPredictor could not be initialised — "
            "RL prediction records disabled. Error: {}",
            exc,
        )
        multi_horizon_predictor = None

    # ------------------------------------------------------------------
    # APScheduler
    # ------------------------------------------------------------------
    scheduler = None
    if not args.no_scheduler:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            rest_client = BinanceRESTClient(
                api_key=os.environ.get("BINANCE_API_KEY", ""),
                api_secret=os.environ.get("BINANCE_API_SECRET", ""),
            )
            validator = DataValidator()
            sched_obj = IncrementalScheduler(
                client=rest_client,
                storage=storage,
                validator=validator,
                symbols=["DOGEUSDT", "BTCUSDT", "DOGEBTC"],
                interval=doge_settings.primary_interval,
            )
            scheduler = BackgroundScheduler(timezone="UTC")

            # :01 past every hour — incremental REST top-up
            scheduler.add_job(
                _make_incremental_job(sched_obj, health_status),
                CronTrigger(minute=1),
                id="incremental_update",
                replace_existing=True,
                misfire_grace_time=300,
            )

            # :02 past every hour — RL verifier
            scheduler.add_job(
                _make_verifier_job(verifier),
                CronTrigger(minute=2),
                id="prediction_verifier",
                replace_existing=True,
                misfire_grace_time=300,
            )

            # Sunday 02:00 UTC — weekly model retraining
            scheduler.add_job(
                _make_retrain_job(storage, args.models_dir),
                CronTrigger(day_of_week="sun", hour=2, minute=0),
                id="weekly_retrain",
                replace_existing=True,
                misfire_grace_time=3600,
            )

            # First Sunday of every month 03:00 UTC — round-number level review
            scheduler.add_job(
                _make_round_number_review_job(),
                CronTrigger(day_of_week="sun", hour=3, minute=0, day="1-7"),
                id="monthly_round_number_review",
                replace_existing=True,
                misfire_grace_time=3600,
            )

            # Daily 00:05 UTC — prediction store backup to data/predictions/
            scheduler.add_job(
                _make_prediction_backup_job(storage),
                CronTrigger(hour=0, minute=5),
                id="daily_prediction_backup",
                replace_existing=True,
                misfire_grace_time=300,
            )

            scheduler.start()
            logger.info(
                "serve: APScheduler started "
                "(5 jobs: incremental, verifier, retrain, round_review, backup)"
            )
        except Exception as exc:
            logger.error("serve: APScheduler startup failed: {}", exc)
            scheduler = None

    # ------------------------------------------------------------------
    # WebSocket client
    # ------------------------------------------------------------------
    ws_client: Optional[BinanceWebSocketClient] = None
    if not args.no_ws and engine is not None:
        try:
            ws_client = BinanceWebSocketClient()

            # Primary DOGEUSDT stream — triggers inference + RL prediction on candle close
            ws_client.subscribe_klines(
                "dogeusdt",
                doge_settings.primary_interval,
                _make_kline_callback(
                    "dogeusdt", engine, health_status, alert_mgr,
                    multi_horizon_predictor=multi_horizon_predictor,
                ),
            )

            # BTCUSDT — data stored; no inference triggered from this stream
            ws_client.subscribe_klines(
                "btcusdt",
                doge_settings.primary_interval,
                _make_kline_callback("btcusdt", engine, health_status, alert_mgr),
            )

            # DOGEBTC — data stored; no inference triggered from this stream
            ws_client.subscribe_klines(
                "dogebtc",
                doge_settings.primary_interval,
                _make_kline_callback("dogebtc", engine, health_status, alert_mgr),
            )

            ws_client.connect()
            health_status.ws_connected = ws_client.is_connected
            logger.info("serve: WebSocket client connected")

            if _PROMETHEUS_AVAILABLE:
                _WS_CONNECTED.set(1.0 if ws_client.is_connected else 0.0)

        except Exception as exc:
            logger.error("serve: WebSocket startup failed: {}", exc)
            alert_mgr.send_alert(
                "CRITICAL",
                "WebSocket client failed to start",
                {"error": str(exc)},
            )
            ws_client = None
    else:
        if args.no_ws:
            logger.info("serve: WebSocket disabled via --no-ws flag")
        else:
            logger.warning("serve: WebSocket skipped — no inference engine loaded")

    # ------------------------------------------------------------------
    # Initial DB probe
    # ------------------------------------------------------------------
    _probe_db(storage, health_status)

    # ------------------------------------------------------------------
    # Main loop — runs until SIGTERM / SIGINT
    # ------------------------------------------------------------------
    logger.info("serve: entering main loop (press Ctrl+C to stop)")
    _DB_PROBE_INTERVAL_S: int = 60
    _last_db_probe: float = time.monotonic()

    while not shutdown_event.is_set():
        now = time.monotonic()

        # Periodic DB health probe
        if now - _last_db_probe >= _DB_PROBE_INTERVAL_S:
            _probe_db(storage, health_status)
            _last_db_probe = now

        # Update WS connected metric
        if ws_client is not None:
            ws_connected = ws_client.is_connected
            health_status.ws_connected = ws_connected
            if _PROMETHEUS_AVAILABLE:
                _WS_CONNECTED.set(1.0 if ws_connected else 0.0)

            if not ws_connected:
                alert_mgr.send_alert(
                    "WARNING",
                    "WebSocket disconnected",
                    {"reconnecting": True},
                )

        # Update candle age metric
        if _PROMETHEUS_AVAILABLE and health_status.last_candle_close_time_ms > 0:
            age_s = (int(time.time() * 1_000) - health_status.last_candle_close_time_ms) / 1_000
            _CANDLE_AGE.set(age_s)

        shutdown_event.wait(timeout=5.0)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    logger.info("serve: shutdown initiated — draining (10s timeout)")
    _drain_deadline = time.monotonic() + 10.0

    if ws_client is not None:
        try:
            ws_client.disconnect()
            logger.info("serve: WebSocket disconnected")
        except Exception as exc:
            logger.warning("serve: WebSocket disconnect error: {}", exc)

    if scheduler is not None:
        try:
            scheduler.shutdown(wait=False)
            logger.info("serve: APScheduler stopped")
        except Exception as exc:
            logger.warning("serve: APScheduler shutdown error: {}", exc)

    health_server.stop()

    remaining = _drain_deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)

    logger.info("serve: clean shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
