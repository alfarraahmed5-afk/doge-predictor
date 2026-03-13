"""Inference engine — live prediction pipeline for DOGE price direction.

Implements the **exact** 12-step inference pipeline from CLAUDE.md Section 10
in strict sequential order.  Each step is a discrete, testable unit.

Pipeline steps (CLAUDE.md §10):
    1.  Freshness check — last candle must be recent.
    2.  Feature computation — full pipeline on last 500 closed candles.
    3.  Regime classification — DogeRegimeClassifier on current row.
    4.  Feature validation — zero NaN/Inf, column list matches manifest.
    5.  Scaling — apply loaded scaler (NEVER refit at inference).
    6.  Base model inference — LSTM + regime XGBoost via RegimeRouter.
    7.  Load regime-adjusted confidence threshold from regime_config.yaml.
    8.  Ensemble meta-learner — [lstm_prob, xgb_prob, regime_encoded].
    9.  Risk filters (hard overrides, applied in order):
            a. funding_extreme_long == 1 → suppress BUY
            b. at_round_number_flag == 1 → reduce position by 30%
            c. btc_1h_return < btc_crash_threshold → suppress BUY
            d. RANGING_LOW_VOL regime → 50% position size
            e. DECOUPLED regime → 50% position size
    10. Signal decision — BUY / SELL / HOLD from ensemble_prob vs threshold.
    11. Log full PredictionRecord to doge_predictions table.
    12. Emit SignalEvent via registered on_signal callback.

Security / anti-lookahead guarantees:
    - Scaler is loaded from disk; never re-fitted at inference.
    - Feature columns loaded from ``feature_columns.json``; mismatch raises
      :class:`FeatureValidationError`.
    - BTC-crash check uses ``btc_1h_return`` from the closed feature row —
      computed via ``log_ret_1`` on BTC data — never forward-looking.

Usage::

    from src.inference.engine import InferenceEngine, EngineConfig

    engine = InferenceEngine.from_artifacts(models_dir=Path("models/"))
    engine.register_on_signal(my_callback)
    event = engine.run(doge_df, btc_df, dogebtc_df, funding_df, doge_4h, doge_1d)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import (
    DogeSettings,
    RegimeConfig,
    Settings,
    doge_settings as _default_doge_settings,
    regime_config as _default_regime_config,
    settings as _default_settings,
)
from src.features.pipeline import FeaturePipeline, validate_feature_matrix
from src.inference.signal import RiskFilterResult, SignalEvent
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.regime_router import RegimeRouter
from src.processing.schemas import HORIZON_CANDLES, PredictionRecord
from src.processing.storage import DogeStorage
from src.regimes.classifier import DogeRegimeClassifier
from src.regimes.features import get_regime_features
from src.training.scaler import FoldScaler

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Milliseconds per 1-hour candle interval.
_MS_PER_1H: int = 3_600_000

#: Reduced-position regime labels (CLAUDE.md §10 Step 9d/9e).
_REDUCED_POSITION_REGIMES: frozenset[str] = frozenset(
    {"RANGING_LOW_VOL", "DECOUPLED"}
)

#: BTC log-return column produced by the feature pipeline.
_BTC_LOG_RET_COL: str = "btc_log_ret_1"

#: Column names for the mandatory risk-filter features.
_FUNDING_EXTREME_LONG_COL: str = "funding_extreme_long"
_ROUND_NUMBER_FLAG_COL: str = "at_round_number_flag"

#: Default horizon used when logging SHORT predictions to the store.
_DEFAULT_HORIZON: str = "SHORT"

#: Prefix used for feature-hash computation.
_HASH_ALGORITHM: str = "sha256"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class StaleDataError(RuntimeError):
    """Raised by Step 1 when the most recent candle is too old.

    Args:
        last_close_time: close_time of the most recent candle (UTC ms).
        now_ms: Wall-clock time when the check was run (UTC ms).
        interval_ms: Expected candle interval in milliseconds.
        multiplier: Freshness multiplier from ``doge_settings.yaml``.
    """

    def __init__(
        self,
        last_close_time: int,
        now_ms: int,
        interval_ms: int,
        multiplier: int,
    ) -> None:
        age_s = (now_ms - last_close_time) / 1000.0
        limit_s = (interval_ms * multiplier) / 1000.0
        super().__init__(
            f"Stale data: last candle close_time={last_close_time} ms is "
            f"{age_s:.1f}s old; limit is {limit_s:.1f}s "
            f"({multiplier} × {interval_ms // 1000}s interval)."
        )
        self.last_close_time = last_close_time
        self.now_ms = now_ms
        self.interval_ms = interval_ms
        self.multiplier = multiplier


class FeatureValidationError(ValueError):
    """Raised by Step 4 when the live feature matrix fails validation.

    Args:
        validation_result: Dict returned by :func:`validate_feature_matrix`.
    """

    def __init__(self, validation_result: dict[str, Any]) -> None:
        issues: list[str] = []
        if validation_result.get("nan_cols"):
            issues.append(f"NaN in {validation_result['nan_cols']}")
        if validation_result.get("inf_cols"):
            issues.append(f"Inf in {validation_result['inf_cols']}")
        if validation_result.get("missing_mandatory"):
            issues.append(f"Missing mandatory: {validation_result['missing_mandatory']}")
        if validation_result.get("missing_expected"):
            issues.append(f"Missing expected: {validation_result['missing_expected']}")
        super().__init__(
            "Feature validation failed — " + "; ".join(issues)
            if issues
            else "Feature validation failed (unknown reason)."
        )
        self.validation_result = validation_result


# ---------------------------------------------------------------------------
# EngineConfig
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Runtime configuration for :class:`InferenceEngine`.

    Attributes:
        models_dir: Directory containing all serialised model artefacts.
        model_version: Model identifier string (e.g. MLflow run-id).
        symbol: Trading pair (default: ``"DOGEUSDT"``).
        interval_ms: Primary candle interval in milliseconds.
        previous_regime: Last regime label from the previous candle (used to
            detect transitions in Step 3).
        on_signal: Optional callback invoked in Step 12 with the
            :class:`~src.inference.signal.SignalEvent`.
        storage: Optional :class:`~src.processing.storage.DogeStorage`
            instance for Step 11 prediction logging.  If *None*, logging
            is skipped with a warning.
    """

    models_dir: Path
    model_version: str = "unknown"
    symbol: str = "DOGEUSDT"
    interval_ms: int = _MS_PER_1H
    previous_regime: Optional[str] = None
    on_signal: Optional[Callable[[SignalEvent], None]] = None
    storage: Optional[DogeStorage] = None


# ---------------------------------------------------------------------------
# InferenceEngine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Full 12-step live inference pipeline for DOGE price direction.

    Loads all model artefacts from ``models_dir`` at construction time.
    Each call to :meth:`run` executes the complete pipeline and returns a
    :class:`~src.inference.signal.SignalEvent`.

    Args:
        config: Engine runtime configuration.
        doge_cfg: DOGE-specific settings loaded from ``doge_settings.yaml``.
        regime_cfg: Regime configuration loaded from ``regime_config.yaml``.

    Raises:
        FileNotFoundError: If any required model artefact is missing from
            ``config.models_dir``.
    """

    def __init__(
        self,
        config: EngineConfig,
        doge_cfg: DogeSettings | None = None,
        regime_cfg: RegimeConfig | None = None,
    ) -> None:
        """Initialise the engine and load all model artefacts.

        Args:
            config: Engine runtime configuration.
            doge_cfg: DOGE settings.  Defaults to module-level singleton.
            regime_cfg: Regime config.  Defaults to module-level singleton.
        """
        self._config: EngineConfig = config
        self._doge_cfg: DogeSettings = (
            doge_cfg if doge_cfg is not None else _default_doge_settings
        )
        self._regime_cfg: RegimeConfig = (
            regime_cfg if regime_cfg is not None else _default_regime_config
        )

        # ---- Load artefacts -----------------------------------------------
        models_dir = config.models_dir

        # Feature column manifest (Step 4)
        feat_col_path = models_dir / "feature_columns.json"
        if feat_col_path.exists():
            with feat_col_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self._expected_columns: list[str] = manifest.get("feature_columns", [])
            logger.info(
                "InferenceEngine: loaded {} expected feature columns from {}",
                len(self._expected_columns),
                feat_col_path,
            )
        else:
            logger.warning(
                "InferenceEngine: feature_columns.json not found at {} — "
                "Step 4 will only check mandatory features.",
                feat_col_path,
            )
            self._expected_columns = []

        # Scaler (Step 5)
        self._scaler: FoldScaler = FoldScaler()
        scaler_path = models_dir / "scaler.pkl"
        self._scaler.load(scaler_path)
        logger.info("InferenceEngine: scaler loaded from {}", scaler_path)

        # LSTM model (Step 6)
        self._lstm: LSTMModel = LSTMModel(regime_cfg=self._regime_cfg)
        lstm_path = models_dir / "lstm"
        self._lstm.load(lstm_path)
        logger.info("InferenceEngine: LSTM loaded from {}", lstm_path)

        # RegimeRouter for XGBoost (Step 6)
        self._router: RegimeRouter = RegimeRouter(
            regime_models={}, global_model=None
        )
        router_path = models_dir / "xgb_global"
        if (models_dir / "router_metadata.json").exists():
            # Full router with per-regime models
            self._router.load(models_dir)
            logger.info("InferenceEngine: RegimeRouter loaded from {}", models_dir)
        elif router_path.exists():
            # Global XGBoost only (fallback)
            from src.models.xgb_model import XGBoostModel
            global_xgb = XGBoostModel(regime_cfg=self._regime_cfg)
            global_xgb.load(router_path)
            self._router = RegimeRouter(
                regime_models={}, global_model=global_xgb
            )
            logger.info(
                "InferenceEngine: global XGBoost loaded from {} (no router found)",
                router_path,
            )
        else:
            raise FileNotFoundError(
                f"Neither router_metadata.json nor xgb_global/ found in {models_dir}"
            )

        # Ensemble meta-learner (Step 8)
        self._ensemble: EnsembleModel = EnsembleModel(regime_cfg=self._regime_cfg)
        ensemble_path = models_dir / "ensemble"
        self._ensemble.load(ensemble_path)
        logger.info("InferenceEngine: Ensemble loaded from {}", ensemble_path)

        # Regime classifier (Step 3)
        self._classifier: DogeRegimeClassifier = DogeRegimeClassifier(
            config=self._regime_cfg
        )

        # Registered signal callbacks (Step 12)
        self._signal_callbacks: list[Callable[[SignalEvent], None]] = []
        if config.on_signal is not None:
            self._signal_callbacks.append(config.on_signal)

        logger.info(
            "InferenceEngine ready: model_version={}, symbol={}",
            config.model_version,
            config.symbol,
        )

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        models_dir: Path,
        model_version: str = "unknown",
        symbol: str = "DOGEUSDT",
        storage: Optional[DogeStorage] = None,
        doge_cfg: DogeSettings | None = None,
        regime_cfg: RegimeConfig | None = None,
    ) -> "InferenceEngine":
        """Convenience factory — build an engine from a models directory.

        Args:
            models_dir: Directory containing all model artefacts.
            model_version: Model identifier (e.g. MLflow run-id).
            symbol: Trading pair symbol.
            storage: Optional prediction store.
            doge_cfg: DOGE settings override.
            regime_cfg: Regime config override.

        Returns:
            Fully initialised :class:`InferenceEngine` instance.
        """
        config = EngineConfig(
            models_dir=models_dir,
            model_version=model_version,
            symbol=symbol,
            storage=storage,
        )
        return cls(config, doge_cfg=doge_cfg, regime_cfg=regime_cfg)

    # -----------------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------------

    def register_on_signal(
        self, callback: Callable[[SignalEvent], None]
    ) -> None:
        """Register a callback to be invoked in Step 12 with each SignalEvent.

        Multiple callbacks can be registered; they are called in registration
        order.

        Args:
            callback: Callable that accepts a :class:`SignalEvent`.
        """
        self._signal_callbacks.append(callback)
        logger.debug(
            "InferenceEngine: registered on_signal callback (total: {})",
            len(self._signal_callbacks),
        )

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------

    def run(
        self,
        doge_1h: pd.DataFrame,
        btc_1h: pd.DataFrame,
        dogebtc_1h: pd.DataFrame,
        funding: pd.DataFrame,
        doge_4h: pd.DataFrame,
        doge_1d: pd.DataFrame,
        regime_labels: pd.Series | None = None,
    ) -> SignalEvent:
        """Execute the full 12-step inference pipeline for one candle close.

        Args:
            doge_1h: Last ``inference_lookback_candles`` closed DOGE 1h candles.
                Must contain columns: ``open_time``, ``close``, ``close_time``,
                ``open``, ``high``, ``low``, ``volume``.
            btc_1h: Matching BTC 1h candles (same time window).
            dogebtc_1h: DOGE/BTC ratio 1h candles.
            funding: Funding rate DataFrame (subset covering the window).
            doge_4h: DOGE 4h candles for HTF features.
            doge_1d: DOGE 1d candles for HTF features.
            regime_labels: Optional precomputed regime Series (open_time index).
                If *None*, the classifier is run inside this call.

        Returns:
            :class:`~src.inference.signal.SignalEvent` containing the final
            signal, all intermediate probabilities, and metadata.

        Raises:
            StaleDataError: Step 1 — most recent candle is too old.
            FeatureValidationError: Step 4 — NaN/Inf or column mismatch.
            ValueError: If input DataFrames are empty or missing required cols.
        """
        # ===================================================================
        # STEP 1 — Freshness check
        # ===================================================================
        logger.debug("InferenceEngine Step 1: freshness check")
        self._step1_freshness_check(doge_1h)

        # ===================================================================
        # STEP 2 — Feature computation
        # ===================================================================
        logger.debug("InferenceEngine Step 2: feature computation")
        feature_df = self._step2_compute_features(
            doge_1h, btc_1h, dogebtc_1h, funding, doge_4h, doge_1d,
            regime_labels=regime_labels,
        )

        # ===================================================================
        # STEP 3 — Regime classification
        # ===================================================================
        logger.debug("InferenceEngine Step 3: regime classification")
        current_regime = self._step3_classify_regime(doge_1h, btc_1h)

        # ===================================================================
        # STEP 4 — Feature validation
        # ===================================================================
        logger.debug("InferenceEngine Step 4: feature validation")
        self._step4_validate_features(feature_df)

        # Extract the single current-candle feature row (last row)
        current_row: pd.Series = feature_df.iloc[-1]

        # ===================================================================
        # STEP 5 — Scaling
        # ===================================================================
        logger.debug("InferenceEngine Step 5: scaling")
        X_scaled = self._step5_scale_features(feature_df)
        # Use the last row vector for single-candle inference
        X_current: np.ndarray = X_scaled[[-1], :]

        # ===================================================================
        # STEP 6 — Base model inference
        # ===================================================================
        logger.debug("InferenceEngine Step 6: base model inference")
        lstm_prob, xgb_prob = self._step6_base_model_inference(
            X_scaled, current_regime
        )

        # ===================================================================
        # STEP 7 — Load regime-adjusted confidence threshold
        # ===================================================================
        logger.debug("InferenceEngine Step 7: load regime threshold")
        confidence_threshold = self._step7_get_threshold(current_regime)

        # ===================================================================
        # STEP 8 — Ensemble meta-learner
        # ===================================================================
        logger.debug("InferenceEngine Step 8: ensemble")
        regime_encoded = get_regime_features(current_regime)["regime_encoded"]
        ensemble_prob = self._step8_ensemble(lstm_prob, xgb_prob, regime_encoded)

        # ===================================================================
        # STEP 9 — Risk filters
        # ===================================================================
        logger.debug("InferenceEngine Step 9: risk filters")
        risk_result = self._step9_risk_filters(
            current_row=current_row,
            current_regime=current_regime,
            raw_signal="BUY" if ensemble_prob >= confidence_threshold else (
                "SELL" if (1.0 - ensemble_prob) >= confidence_threshold else "HOLD"
            ),
        )

        # ===================================================================
        # STEP 10 — Signal decision
        # ===================================================================
        logger.debug("InferenceEngine Step 10: signal decision")
        final_signal = self._step10_signal_decision(
            ensemble_prob=ensemble_prob,
            confidence_threshold=confidence_threshold,
            risk_result=risk_result,
        )

        # Build the SignalEvent (used in Step 11 and Step 12)
        open_time = int(current_row.get("open_time", doge_1h.iloc[-1]["open_time"]))
        close_price = float(current_row.get("close", doge_1h.iloc[-1]["close"]))

        event = SignalEvent(
            timestamp_ms=open_time,
            symbol=self._config.symbol,
            regime=current_regime,
            signal=final_signal,
            ensemble_prob=float(ensemble_prob),
            confidence_threshold=confidence_threshold,
            position_size_multiplier=risk_result.position_size_multiplier,
            risk_filters_triggered=list(risk_result.triggered),
            model_version=self._config.model_version,
            lstm_prob=float(lstm_prob),
            xgb_prob=float(xgb_prob),
            regime_encoded=float(regime_encoded),
            open_time=open_time,
            close_price=close_price,
        )

        # ===================================================================
        # STEP 11 — Log to prediction store
        # ===================================================================
        logger.debug("InferenceEngine Step 11: log prediction")
        self._step11_log_prediction(event, current_row)

        # ===================================================================
        # STEP 12 — Emit signal
        # ===================================================================
        logger.debug("InferenceEngine Step 12: emit signal")
        self._step12_emit_signal(event)

        logger.info(
            "Inference complete: signal={} prob={:.4f} threshold={:.4f} "
            "regime={} filters={}",
            final_signal,
            ensemble_prob,
            confidence_threshold,
            current_regime,
            risk_result.triggered,
        )

        # Update previous regime for next call
        self._config.previous_regime = current_regime

        return event

    # -----------------------------------------------------------------------
    # Private step implementations
    # -----------------------------------------------------------------------

    def _step1_freshness_check(self, doge_1h: pd.DataFrame) -> None:
        """Step 1: Assert that the most recent candle is sufficiently fresh.

        Args:
            doge_1h: DOGE 1h DataFrame; must contain a ``close_time`` column.

        Raises:
            StaleDataError: If the last ``close_time`` is more than
                ``freshness_check_multiplier × interval_ms`` ms in the past.
            ValueError: If the DataFrame is empty or missing ``close_time``.
        """
        if doge_1h.empty:
            raise ValueError("Step 1: doge_1h DataFrame is empty.")
        if "close_time" not in doge_1h.columns:
            raise ValueError("Step 1: 'close_time' column is required in doge_1h.")

        last_close_time: int = int(doge_1h["close_time"].iloc[-1])
        now_ms: int = int(time.time() * 1000)
        interval_ms: int = self._config.interval_ms
        multiplier: int = self._doge_cfg.freshness_check_multiplier
        limit_ms: int = interval_ms * multiplier

        if (now_ms - last_close_time) > limit_ms:
            raise StaleDataError(
                last_close_time=last_close_time,
                now_ms=now_ms,
                interval_ms=interval_ms,
                multiplier=multiplier,
            )

        logger.debug(
            "Step 1 OK: last close_time={} now={} age={:.1f}s limit={:.1f}s",
            last_close_time,
            now_ms,
            (now_ms - last_close_time) / 1000.0,
            limit_ms / 1000.0,
        )

    def _step2_compute_features(
        self,
        doge_1h: pd.DataFrame,
        btc_1h: pd.DataFrame,
        dogebtc_1h: pd.DataFrame,
        funding: pd.DataFrame,
        doge_4h: pd.DataFrame,
        doge_1d: pd.DataFrame,
        regime_labels: pd.Series | None,
    ) -> pd.DataFrame:
        """Step 2: Run the full FeaturePipeline on the last N closed candles.

        Args:
            doge_1h: DOGE 1h candles.
            btc_1h: BTC 1h candles.
            dogebtc_1h: DOGE/BTC 1h candles.
            funding: Funding rate DataFrame.
            doge_4h: DOGE 4h candles.
            doge_1d: DOGE 1d candles.
            regime_labels: Optional precomputed regime Series.

        Returns:
            Feature DataFrame with all mandatory features present.
        """
        pipeline = FeaturePipeline(cfg=self._doge_cfg)

        # Mark input rows as 'live' era for inference
        for df in (doge_1h, btc_1h, dogebtc_1h):
            if "era" not in df.columns:
                df = df.copy()
                df["era"] = "live"

        feature_df = pipeline.compute_all_features(
            doge_1h=doge_1h,
            btc_1h=btc_1h,
            dogebtc_1h=dogebtc_1h,
            funding=funding,
            doge_4h=doge_4h,
            doge_1d=doge_1d,
            regimes=regime_labels,
            min_rows_override=1,  # At inference: only need 1 row minimum
        )

        logger.debug(
            "Step 2: feature matrix shape = {}", feature_df.shape
        )
        return feature_df

    def _step3_classify_regime(
        self,
        doge_1h: pd.DataFrame,
        btc_1h: pd.DataFrame,
    ) -> str:
        """Step 3: Classify the current market regime.

        Args:
            doge_1h: DOGE 1h candles.
            btc_1h: BTC 1h candles.

        Returns:
            Current regime label string.
        """
        regimes: pd.Series = self._classifier.classify(
            df=doge_1h, btc_df=btc_1h
        )
        current_regime: str = str(regimes.iloc[-1])

        # Detect and log regime transitions
        if self._config.previous_regime is not None:
            changed = DogeRegimeClassifier.detect_transition(
                self._config.previous_regime, current_regime
            )
            if changed:
                logger.info(
                    "Step 3: regime transition {} -> {}",
                    self._config.previous_regime,
                    current_regime,
                )

        logger.debug("Step 3: current regime = {}", current_regime)
        return current_regime

    def _step4_validate_features(self, feature_df: pd.DataFrame) -> None:
        """Step 4: Validate feature matrix — zero NaN/Inf, correct columns.

        Args:
            feature_df: Feature DataFrame produced by Step 2.

        Raises:
            FeatureValidationError: If any validation check fails.
        """
        expected = self._expected_columns if self._expected_columns else None
        result = validate_feature_matrix(
            feature_df, expected_columns=expected, strict=False
        )
        if not result["ok"]:
            raise FeatureValidationError(result)

        logger.debug("Step 4: feature validation OK (n_rows={})", result["n_rows"])

    def _step5_scale_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Step 5: Apply the loaded scaler to the feature matrix.

        The scaler was fitted on the training fold and must NOT be re-fitted
        at inference time (RULE B).

        Args:
            feature_df: Validated feature DataFrame.

        Returns:
            Scaled feature array, shape ``(n_rows, n_features)``.
        """
        # Select only the numeric feature columns (exclude OHLCV + meta cols)
        from src.features.pipeline import _PASSTHROUGH_COLS
        feature_cols = [
            c for c in feature_df.select_dtypes(include=[np.number]).columns
            if c not in _PASSTHROUGH_COLS
        ]
        X = feature_df[feature_cols].to_numpy(dtype=np.float64)

        # Apply transform — never refit
        X_scaled: np.ndarray = self._scaler.transform(X)
        logger.debug(
            "Step 5: scaled {} rows × {} features", X_scaled.shape[0], X_scaled.shape[1]
        )
        return X_scaled

    def _step6_base_model_inference(
        self,
        X_scaled: np.ndarray,
        current_regime: str,
    ) -> tuple[float, float]:
        """Step 6: Run LSTM and regime-specific XGBoost.

        Args:
            X_scaled: Scaled feature array, shape ``(n_rows, n_features)``.
            current_regime: Current regime label for XGBoost routing.

        Returns:
            Tuple ``(lstm_prob, xgb_prob)`` for the current (last) candle.
        """
        # LSTM — predict_proba over full sequence; take the last position
        lstm_probas: np.ndarray = self._lstm.predict_proba(X_scaled)
        lstm_prob: float = float(lstm_probas[-1])

        # XGBoost via regime router — single candle row
        xgb_model = self._router.route(current_regime)
        xgb_probas: np.ndarray = xgb_model.predict_proba(X_scaled[[-1], :])
        xgb_prob: float = float(xgb_probas[0])

        logger.debug(
            "Step 6: lstm_prob={:.4f} xgb_prob={:.4f}", lstm_prob, xgb_prob
        )
        return lstm_prob, xgb_prob

    def _step7_get_threshold(self, current_regime: str) -> float:
        """Step 7: Load the regime-adjusted confidence threshold.

        The threshold is ALWAYS loaded from ``regime_config.yaml``.
        It is NEVER hardcoded anywhere in this file.

        Args:
            current_regime: Current regime label.

        Returns:
            Confidence threshold float from regime_config.yaml.
        """
        threshold: float = self._regime_cfg.get_confidence_threshold(current_regime)
        logger.debug(
            "Step 7: threshold={:.4f} for regime={}", threshold, current_regime
        )
        return threshold

    def _step8_ensemble(
        self,
        lstm_prob: float,
        xgb_prob: float,
        regime_encoded: float,
    ) -> float:
        """Step 8: Run the ensemble meta-learner.

        Args:
            lstm_prob: LSTM output probability.
            xgb_prob: XGBoost output probability.
            regime_encoded: Ordinal regime encoding (0–4).

        Returns:
            Ensemble P(up), scalar float in ``[0, 1]``.
        """
        meta_X: np.ndarray = np.array(
            [[lstm_prob, xgb_prob, regime_encoded]], dtype=np.float64
        )
        ensemble_probas: np.ndarray = self._ensemble.predict_proba(meta_X)
        ensemble_prob: float = float(ensemble_probas[0])
        logger.debug("Step 8: ensemble_prob={:.4f}", ensemble_prob)
        return ensemble_prob

    def _step9_risk_filters(
        self,
        current_row: pd.Series,
        current_regime: str,
        raw_signal: str,
    ) -> RiskFilterResult:
        """Step 9: Apply hard risk override rules in prescribed order.

        Rules applied in order (a → e):
            a. funding_extreme_long == 1 → suppress BUY (hard)
            b. at_round_number_flag == 1 → reduce position by 30%
            c. btc_1h_return < btc_crash_threshold → suppress BUY (hard)
            d. RANGING_LOW_VOL → 50% position size
            e. DECOUPLED → 50% position size

        Args:
            current_row: Feature Series for the current candle.
            current_regime: Active regime label.
            raw_signal: Pre-filter signal (``"BUY"``, ``"SELL"``, or ``"HOLD"``).

        Returns:
            :class:`RiskFilterResult` describing suppression and size changes.
        """
        buy_suppressed = False
        position_size_multiplier = 1.0
        triggered: list[str] = []

        # --- Rule a: funding_extreme_long -----------------------------------
        funding_extreme_long = float(
            current_row.get(_FUNDING_EXTREME_LONG_COL, 0.0)
        )
        if funding_extreme_long == 1.0:
            buy_suppressed = True
            triggered.append("funding_extreme_long")
            logger.warning(
                "Step 9a: BUY suppressed — funding_extreme_long == 1"
            )

        # --- Rule b: round number flag (position size reduction) ------------
        at_round_number = float(
            current_row.get(_ROUND_NUMBER_FLAG_COL, 0.0)
        )
        if at_round_number == 1.0:
            reduction = self._doge_cfg.risk.round_number_size_reduction
            position_size_multiplier *= (1.0 - reduction)
            triggered.append("at_round_number_flag")
            logger.debug(
                "Step 9b: position size reduced by {:.0%} — round number",
                reduction,
            )

        # --- Rule c: BTC crash threshold ------------------------------------
        btc_crash_threshold: float = self._doge_cfg.btc_crash_threshold

        # btc_log_ret_1 is 'log_ret_1' of BTC (produced by lag_features on
        # btc-prefixed columns or available via the pipeline's btc_ prefix).
        # The pipeline adds btc_ prefix via MultiSymbolAligner; we check both
        # possible column names.
        btc_return: float | None = None
        for col_candidate in ("btc_log_ret_1", "btc_log_ret_1h", "log_ret_1_btc"):
            if col_candidate in current_row.index:
                btc_return = float(current_row[col_candidate])
                break

        if btc_return is not None and btc_return < btc_crash_threshold:
            buy_suppressed = True
            triggered.append("btc_crash_override")
            logger.warning(
                "Step 9c: BUY suppressed — btc_1h_return={:.4f} < threshold={:.4f}",
                btc_return,
                btc_crash_threshold,
            )

        # --- Rule d: RANGING_LOW_VOL position size --------------------------
        if current_regime == "RANGING_LOW_VOL":
            regime_multiplier = self._regime_cfg.get_position_size_multiplier(
                "RANGING_LOW_VOL"
            )
            position_size_multiplier *= regime_multiplier
            triggered.append("ranging_low_vol_half_size")
            logger.debug(
                "Step 9d: position size × {:.2f} — RANGING_LOW_VOL regime",
                regime_multiplier,
            )

        # --- Rule e: DECOUPLED position size --------------------------------
        if current_regime == "DECOUPLED":
            regime_multiplier = self._regime_cfg.get_position_size_multiplier(
                "DECOUPLED"
            )
            position_size_multiplier *= regime_multiplier
            triggered.append("decoupled_half_size")
            logger.debug(
                "Step 9e: position size × {:.2f} — DECOUPLED regime",
                regime_multiplier,
            )

        return RiskFilterResult(
            buy_suppressed=buy_suppressed,
            position_size_multiplier=position_size_multiplier,
            triggered=triggered,
        )

    def _step10_signal_decision(
        self,
        ensemble_prob: float,
        confidence_threshold: float,
        risk_result: RiskFilterResult,
    ) -> str:
        """Step 10: Determine the final signal after risk filter application.

        Args:
            ensemble_prob: Ensemble P(up) from Step 8.
            confidence_threshold: Regime threshold from Step 7.
            risk_result: Risk filter outcome from Step 9.

        Returns:
            ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        """
        if ensemble_prob >= confidence_threshold:
            raw = "BUY"
        elif (1.0 - ensemble_prob) >= confidence_threshold:
            raw = "SELL"
        else:
            raw = "HOLD"

        # Apply BUY suppression (hard overrides from Step 9)
        if raw == "BUY" and risk_result.buy_suppressed:
            final = "HOLD"
            logger.info(
                "Step 10: BUY suppressed by risk filters → HOLD"
            )
        else:
            final = raw

        logger.debug(
            "Step 10: ensemble_prob={:.4f} threshold={:.4f} raw={} final={}",
            ensemble_prob,
            confidence_threshold,
            raw,
            final,
        )
        return final

    def _step11_log_prediction(
        self,
        event: SignalEvent,
        current_row: pd.Series,
    ) -> None:
        """Step 11: Write a PredictionRecord to the doge_predictions table.

        Always logs a SHORT-horizon record (4 candles ahead) representing the
        next-candle direction prediction.  Multi-horizon RL records are
        generated separately by the RL predictor module.

        Args:
            event: Completed :class:`SignalEvent` from Step 10.
            current_row: Feature Series for the current candle.
        """
        if self._config.storage is None:
            logger.warning(
                "Step 11: no storage configured — prediction not logged"
            )
            return

        # Map signal to predicted_direction
        if event.signal == "BUY":
            predicted_direction = 1
        elif event.signal == "SELL":
            predicted_direction = -1
        else:
            predicted_direction = 0

        # Compute feature hash (SHA-256 of raw feature vector bytes)
        try:
            feature_values = current_row.select_dtypes(include=[np.number]).to_numpy()
            feature_hash = hashlib.new(
                _HASH_ALGORITHM, feature_values.tobytes()
            ).hexdigest()
        except Exception:
            feature_hash = "unavailable"

        open_time: int = event.timestamp_ms
        horizon_candles: int = HORIZON_CANDLES[_DEFAULT_HORIZON]
        target_open_time: int = open_time + horizon_candles * _MS_PER_1H

        # Confidence score: max of ensemble_prob and (1 - ensemble_prob)
        confidence_score = max(event.ensemble_prob, 1.0 - event.ensemble_prob)
        confidence_score = max(0.5, min(1.0, confidence_score))

        close_price = float(
            current_row.get("close", event.close_price)
        )
        if close_price <= 0.0:
            close_price = event.close_price

        try:
            record = PredictionRecord(
                created_at=int(time.time() * 1000),
                open_time=open_time,
                symbol=self._config.symbol,
                horizon_label=_DEFAULT_HORIZON,
                horizon_candles=horizon_candles,
                target_open_time=target_open_time,
                price_at_prediction=close_price,
                predicted_direction=predicted_direction,
                confidence_score=confidence_score,
                lstm_prob=event.lstm_prob,
                xgb_prob=event.xgb_prob,
                regime_label=event.regime,
                model_version=event.model_version,
            )
            self._config.storage.insert_prediction(record)
            logger.info(
                "Step 11: prediction logged id={} direction={} feature_hash={}",
                record.prediction_id,
                predicted_direction,
                feature_hash[:12],
            )
        except Exception as exc:
            logger.error(
                "Step 11: failed to log prediction — {}", exc
            )

    def _step12_emit_signal(self, event: SignalEvent) -> None:
        """Step 12: Call all registered on_signal callbacks with the event.

        Args:
            event: Completed :class:`SignalEvent` from Steps 1–11.
        """
        for callback in self._signal_callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.error(
                    "Step 12: on_signal callback raised an exception: {}", exc
                )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _compute_feature_hash(self, feature_row: np.ndarray) -> str:
        """Compute a SHA-256 hash of the raw feature vector bytes.

        Args:
            feature_row: 1-D float64 array of feature values.

        Returns:
            Hexadecimal digest string.
        """
        return hashlib.new(
            _HASH_ALGORITHM, feature_row.tobytes()
        ).hexdigest()

    # ------------------------------------------------------------------
    # Live-serving convenience method
    # ------------------------------------------------------------------

    def run_on_closed_kline(self, kline: dict[str, Any]) -> Optional["SignalEvent"]:
        """Handle a closed kline dict from :class:`~src.ingestion.ws_client.BinanceWebSocketClient`.

        Checks whether the kline is marked as closed (``k.x == True``), fetches
        the last ``inference_lookback_candles`` rows from
        :attr:`EngineConfig.storage` for all required symbols, then delegates
        to :meth:`run`.

        This method is designed to be registered directly as a WS callback::

            ws.subscribe_klines("dogeusdt", "1h", engine.run_on_closed_kline)

        Args:
            kline: Raw kline dict from the WebSocket stream.  Must have the
                shape ``{"k": {"x": True/False, ...}}``.  Non-closed candles
                (``x == False``) are silently ignored.

        Returns:
            :class:`~src.inference.signal.SignalEvent` on success, or *None*
            if the candle was not closed, storage is unavailable, or an
            unrecoverable error occurred (error is logged but not re-raised).
        """
        k: dict[str, Any] = kline.get("k", {})
        if not k.get("x", False):
            return None  # intermediate (not-yet-closed) kline update

        if self._config.storage is None:
            logger.warning("run_on_closed_kline: no storage configured — cannot fetch data")
            return None

        now_ms: int = int(time.time() * 1_000)
        # Fetch slightly more than the lookback to absorb maintenance gaps
        _extra: int = 50
        lookback_ms_1h: int = (_extra + 500) * _MS_PER_1H
        lookback_ms_4h: int = (_extra + 500) * (4 * _MS_PER_1H)
        lookback_ms_1d: int = (_extra + 500) * (24 * _MS_PER_1H)

        try:
            doge_1h = self._config.storage.get_ohlcv(
                "DOGEUSDT", "1h", now_ms - lookback_ms_1h, now_ms + 1
            )
            btc_1h = self._config.storage.get_ohlcv(
                "BTCUSDT", "1h", now_ms - lookback_ms_1h, now_ms + 1
            )
            dogebtc_1h = self._config.storage.get_ohlcv(
                "DOGEBTC", "1h", now_ms - lookback_ms_1h, now_ms + 1
            )
            funding = self._config.storage.get_funding_rates(
                now_ms - lookback_ms_1h, now_ms + 1
            )
            doge_4h = self._config.storage.get_ohlcv(
                "DOGEUSDT", "4h", now_ms - lookback_ms_4h, now_ms + 1
            )
            doge_1d = self._config.storage.get_ohlcv(
                "DOGEUSDT", "1d", now_ms - lookback_ms_1d, now_ms + 1
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("run_on_closed_kline: storage fetch failed: {}", exc)
            return None

        try:
            return self.run(doge_1h, btc_1h, dogebtc_1h, funding, doge_4h, doge_1d)
        except Exception as exc:  # noqa: BLE001
            logger.error("run_on_closed_kline: engine.run() failed: {}", exc)
            return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

__all__ = [
    "InferenceEngine",
    "EngineConfig",
    "StaleDataError",
    "FeatureValidationError",
]
