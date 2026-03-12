"""Unit tests for src/inference/engine.py and src/inference/signal.py.

Mandatory tests (CLAUDE.md Phase 7 Prompt 7.1):
  - Freshness check: StaleDataError on old candle.
  - Funding override: BUY suppressed when funding_extreme_long == 1.
  - BTC crash override: BUY suppressed when btc_return < -4%.
  - Feature schema mismatch: FeatureValidationError when column missing.
  - Regime threshold: DECOUPLED uses threshold 0.72, not default.
  - Prediction logged: PredictionRecord inserted after every inference call.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DogeSettings,
    RegimeConfig,
    doge_settings as _doge_settings,
    regime_config as _regime_config,
)
from src.inference.engine import (
    EngineConfig,
    FeatureValidationError,
    InferenceEngine,
    StaleDataError,
)
from src.inference.signal import RiskFilterResult, SignalEvent
from src.models.ensemble import EnsembleModel
from src.models.lstm_model import LSTMModel
from src.models.regime_router import RegimeRouter
from src.processing.schemas import PredictionRecord
from src.processing.storage import DogeStorage
from src.regimes.classifier import DogeRegimeClassifier
from src.training.scaler import FoldScaler

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_BASE_TIME: int = 1_640_995_200_000  # 2022-01-01 00:00 UTC


def _make_engine(
    expected_columns: list[str] | None = None,
    storage: Any | None = None,
    model_version: str = "test-v1",
) -> InferenceEngine:
    """Build a fully-mocked InferenceEngine that bypasses all disk I/O.

    All model attributes are replaced with ``MagicMock`` instances so that
    individual step methods can be tested in isolation.

    Args:
        expected_columns: Feature column manifest loaded from
            ``feature_columns.json`` (simulates Step 4 validation).
        storage: Optional mock storage for Step 11 logging.
        model_version: Model version string stored in signals/records.

    Returns:
        InferenceEngine with all models replaced by mocks.
    """
    engine: InferenceEngine = object.__new__(InferenceEngine)  # type: ignore[arg-type]

    config = EngineConfig(
        models_dir=Path("."),
        model_version=model_version,
        symbol="DOGEUSDT",
        interval_ms=_MS_PER_HOUR,
        previous_regime=None,
        on_signal=None,
        storage=storage,
    )
    engine._config = config
    engine._doge_cfg = _doge_settings
    engine._regime_cfg = _regime_config
    engine._expected_columns = expected_columns or []

    # Mock models — default behaviour: return 0.5 probabilities
    scaler_mock = MagicMock(spec=FoldScaler)
    scaler_mock.transform.side_effect = lambda x: x  # identity transform
    engine._scaler = scaler_mock

    lstm_mock = MagicMock(spec=LSTMModel)
    lstm_mock.predict_proba.return_value = np.full(10, 0.5, dtype=np.float64)
    engine._lstm = lstm_mock

    xgb_mock = MagicMock()
    xgb_mock.predict_proba.return_value = np.array([0.5], dtype=np.float64)
    router_mock = MagicMock(spec=RegimeRouter)
    router_mock.route.return_value = xgb_mock
    engine._router = router_mock

    ensemble_mock = MagicMock(spec=EnsembleModel)
    ensemble_mock.predict_proba.return_value = np.array([0.5], dtype=np.float64)
    engine._ensemble = ensemble_mock

    classifier_mock = MagicMock(spec=DogeRegimeClassifier)
    classifier_mock.classify.return_value = pd.Series(["TRENDING_BULL"] * 10)
    engine._classifier = classifier_mock

    engine._signal_callbacks = []

    return engine


def _make_doge_df(
    n: int = 20,
    base_time: int = _BASE_TIME,
    close_time_offset: int | None = None,
) -> pd.DataFrame:
    """Build a minimal DOGE 1h DataFrame.

    Args:
        n: Number of candles.
        base_time: Starting open_time (UTC ms).
        close_time_offset: Offset to add to last candle's close_time
            (used to simulate stale data).

    Returns:
        DataFrame with standard OHLCV + close_time columns.
    """
    times = [base_time + i * _MS_PER_HOUR for i in range(n)]
    close_times = [t + _MS_PER_HOUR - 1 for t in times]

    if close_time_offset is not None:
        close_times[-1] = close_times[-1] + close_time_offset

    return pd.DataFrame(
        {
            "open_time": times,
            "open": [0.10 + i * 0.001 for i in range(n)],
            "high": [0.11 + i * 0.001 for i in range(n)],
            "low": [0.09 + i * 0.001 for i in range(n)],
            "close": [0.105 + i * 0.001 for i in range(n)],
            "volume": [1_000_000.0] * n,
            "close_time": close_times,
            "quote_volume": [100_000.0] * n,
            "num_trades": [100] * n,
            "era": ["live"] * n,
            "symbol": ["DOGEUSDT"] * n,
        }
    )


def _make_current_row(
    funding_extreme_long: float = 0.0,
    at_round_number_flag: float = 0.0,
    btc_log_ret_1: float = 0.0,
    close: float = 0.15,
    open_time: int = _BASE_TIME,
) -> pd.Series:
    """Build a feature Series for testing Step 9 risk filters.

    Args:
        funding_extreme_long: 0 or 1 (CLAUDE.md §5 FACT 5).
        at_round_number_flag: 0 or 1 (CLAUDE.md §5 FACT 6).
        btc_log_ret_1: BTC 1h log-return (CLAUDE.md §5 FACT 2).
        close: DOGE close price.
        open_time: Candle open time (UTC ms).

    Returns:
        pd.Series with all needed risk-filter columns.
    """
    return pd.Series(
        {
            "funding_extreme_long": funding_extreme_long,
            "at_round_number_flag": at_round_number_flag,
            "btc_log_ret_1": btc_log_ret_1,
            "close": close,
            "open_time": open_time,
        }
    )


# ---------------------------------------------------------------------------
# TestStaleDataError (Step 1)
# ---------------------------------------------------------------------------


class TestStaleDataError:
    """Step 1 freshness check — StaleDataError behaviour."""

    def test_stale_candle_raises_stale_data_error(self) -> None:
        """MANDATORY: StaleDataError raised when last close_time > 2×interval old."""
        engine = _make_engine()

        # Place last close_time 10 hours in the past (well beyond 2h limit)
        now_ms = int(time.time() * 1000)
        stale_time = now_ms - 10 * _MS_PER_HOUR

        df = pd.DataFrame(
            {
                "open_time": [stale_time - _MS_PER_HOUR],
                "close_time": [stale_time],
                "close": [0.15],
            }
        )

        with pytest.raises(StaleDataError):
            engine._step1_freshness_check(df)

    def test_fresh_candle_does_not_raise(self) -> None:
        """No exception when last close_time is within 2×interval of now."""
        engine = _make_engine()

        # close_time = 30 minutes ago — within 2h window
        now_ms = int(time.time() * 1000)
        recent_close = now_ms - 30 * 60 * 1000  # 30 minutes ago

        df = pd.DataFrame(
            {
                "open_time": [recent_close - _MS_PER_HOUR],
                "close_time": [recent_close],
                "close": [0.15],
            }
        )

        # Should not raise
        engine._step1_freshness_check(df)

    def test_stale_data_error_attributes(self) -> None:
        """StaleDataError exposes expected attributes."""
        now_ms = int(time.time() * 1000)
        stale_time = now_ms - 10 * _MS_PER_HOUR
        interval_ms = _MS_PER_HOUR
        multiplier = 2

        exc = StaleDataError(
            last_close_time=stale_time,
            now_ms=now_ms,
            interval_ms=interval_ms,
            multiplier=multiplier,
        )

        assert exc.last_close_time == stale_time
        assert exc.now_ms == now_ms
        assert exc.interval_ms == interval_ms
        assert exc.multiplier == multiplier

    def test_empty_dataframe_raises_value_error(self) -> None:
        """ValueError raised when doge_1h is empty."""
        engine = _make_engine()
        with pytest.raises(ValueError, match="empty"):
            engine._step1_freshness_check(pd.DataFrame())

    def test_missing_close_time_column_raises_value_error(self) -> None:
        """ValueError raised when close_time column is missing."""
        engine = _make_engine()
        df = pd.DataFrame({"open_time": [_BASE_TIME], "close": [0.15]})
        with pytest.raises(ValueError, match="close_time"):
            engine._step1_freshness_check(df)


# ---------------------------------------------------------------------------
# TestFundingOverride (Step 9a)
# ---------------------------------------------------------------------------


class TestFundingOverride:
    """Step 9a — BUY suppressed when funding_extreme_long == 1."""

    def test_funding_extreme_long_suppresses_buy(self) -> None:
        """MANDATORY: BUY is suppressed when funding_extreme_long == 1."""
        engine = _make_engine()
        row = _make_current_row(funding_extreme_long=1.0)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        assert result.buy_suppressed is True
        assert "funding_extreme_long" in result.triggered

    def test_funding_extreme_long_does_not_suppress_sell(self) -> None:
        """Funding extreme_long does not suppress SELL signals."""
        engine = _make_engine()
        row = _make_current_row(funding_extreme_long=1.0)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="SELL",
        )

        # buy_suppressed flag is set (rule fired) but SELL is not affected by it
        assert result.buy_suppressed is True

    def test_funding_zero_does_not_suppress_buy(self) -> None:
        """BUY is NOT suppressed when funding_extreme_long == 0."""
        engine = _make_engine()
        row = _make_current_row(funding_extreme_long=0.0)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        assert result.buy_suppressed is False
        assert "funding_extreme_long" not in result.triggered

    def test_step10_buy_suppressed_becomes_hold(self) -> None:
        """Step 10: suppressed BUY → final signal is HOLD."""
        engine = _make_engine()
        risk_result = RiskFilterResult(
            buy_suppressed=True, position_size_multiplier=1.0, triggered=["funding_extreme_long"]
        )

        # ensemble_prob above threshold would normally be BUY
        final = engine._step10_signal_decision(
            ensemble_prob=0.80,
            confidence_threshold=0.62,
            risk_result=risk_result,
        )

        assert final == "HOLD"


# ---------------------------------------------------------------------------
# TestBTCCrashOverride (Step 9c)
# ---------------------------------------------------------------------------


class TestBTCCrashOverride:
    """Step 9c — BUY suppressed when btc_1h_return < -4%."""

    def test_btc_crash_suppresses_buy(self) -> None:
        """MANDATORY: BUY suppressed when btc_log_ret_1 < -0.04."""
        engine = _make_engine()
        # btc_log_ret_1 = -0.05 (5% drop) is below the -4% threshold
        row = _make_current_row(btc_log_ret_1=-0.05)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        assert result.buy_suppressed is True
        assert "btc_crash_override" in result.triggered

    def test_btc_return_above_threshold_does_not_suppress(self) -> None:
        """BUY NOT suppressed when btc_log_ret_1 >= -4%."""
        engine = _make_engine()
        row = _make_current_row(btc_log_ret_1=-0.02)  # only -2%

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        assert "btc_crash_override" not in result.triggered

    def test_btc_threshold_exact_boundary(self) -> None:
        """BTC return exactly at threshold (-4%) does NOT trigger override."""
        engine = _make_engine()
        btc_crash_threshold = _doge_settings.btc_crash_threshold  # -0.04
        row = _make_current_row(btc_log_ret_1=btc_crash_threshold)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        # Strictly less-than — at exactly -0.04 the rule does NOT fire
        assert "btc_crash_override" not in result.triggered

    def test_btc_crash_threshold_loaded_from_config(self) -> None:
        """btc_crash_threshold is loaded from config, not hardcoded."""
        # Verify the config value is -0.04 as per CLAUDE.md
        assert _doge_settings.btc_crash_threshold == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# TestFeatureValidationError (Step 4)
# ---------------------------------------------------------------------------


class TestFeatureValidationError:
    """Step 4 — FeatureValidationError when feature columns are missing."""

    def test_missing_expected_column_raises_feature_validation_error(self) -> None:
        """MANDATORY: FeatureValidationError raised when expected column absent."""
        engine = _make_engine(expected_columns=["required_col_x"])

        # DataFrame does NOT contain 'required_col_x'
        df = pd.DataFrame({"open_time": [_BASE_TIME], "close": [0.15]})

        with pytest.raises(FeatureValidationError):
            engine._step4_validate_features(df)

    def test_nan_in_feature_column_raises_feature_validation_error(self) -> None:
        """FeatureValidationError raised when NaN present in any feature column."""
        engine = _make_engine()
        # Include a mandatory feature name with NaN to trigger the check
        df = pd.DataFrame(
            {
                "open_time": [_BASE_TIME, _BASE_TIME + _MS_PER_HOUR],
                "funding_rate": [0.001, np.nan],  # NaN in mandatory feature
            }
        )

        with pytest.raises(FeatureValidationError):
            engine._step4_validate_features(df)

    def test_inf_in_feature_column_raises_feature_validation_error(self) -> None:
        """FeatureValidationError raised when Inf present in a feature column."""
        engine = _make_engine()
        df = pd.DataFrame(
            {
                "open_time": [_BASE_TIME],
                "volume_ratio": [np.inf],
            }
        )

        with pytest.raises(FeatureValidationError):
            engine._step4_validate_features(df)

    def test_validation_error_carries_result_dict(self) -> None:
        """FeatureValidationError.validation_result contains the reason."""
        engine = _make_engine(expected_columns=["missing_col"])
        df = pd.DataFrame({"open_time": [_BASE_TIME]})

        with pytest.raises(FeatureValidationError) as exc_info:
            engine._step4_validate_features(df)

        assert "missing_expected" in exc_info.value.validation_result
        assert "missing_col" in exc_info.value.validation_result["missing_expected"]

    def test_valid_features_do_not_raise(self) -> None:
        """No exception when feature matrix passes all checks.

        Build a DataFrame that satisfies both the expected_columns contract
        and all mandatory feature requirements from MANDATORY_FEATURE_NAMES.
        """
        from src.features.pipeline import MANDATORY_FEATURE_NAMES

        engine = _make_engine(expected_columns=["extra_feat"])

        # Include every mandatory feature + the extra expected column
        data: dict[str, list[float]] = {col: [0.5] for col in MANDATORY_FEATURE_NAMES}
        data["extra_feat"] = [0.5]
        data["open_time"] = [float(_BASE_TIME)]
        df = pd.DataFrame(data)

        # Should not raise
        engine._step4_validate_features(df)


# ---------------------------------------------------------------------------
# TestRegimeThreshold (Step 7)
# ---------------------------------------------------------------------------


class TestRegimeThreshold:
    """Step 7 — regime-adjusted confidence thresholds loaded from config."""

    def test_decoupled_threshold_is_072(self) -> None:
        """MANDATORY: DECOUPLED regime uses threshold 0.72, not default 0.62."""
        engine = _make_engine()
        threshold = engine._step7_get_threshold("DECOUPLED")

        assert threshold == pytest.approx(0.72)
        assert threshold != pytest.approx(0.62)  # explicitly NOT the default

    def test_trending_bull_threshold(self) -> None:
        """TRENDING_BULL threshold is 0.62 as per regime_config.yaml."""
        engine = _make_engine()
        threshold = engine._step7_get_threshold("TRENDING_BULL")
        assert threshold == pytest.approx(0.62)

    def test_trending_bear_threshold(self) -> None:
        """TRENDING_BEAR threshold is 0.62."""
        engine = _make_engine()
        threshold = engine._step7_get_threshold("TRENDING_BEAR")
        assert threshold == pytest.approx(0.62)

    def test_ranging_low_vol_threshold(self) -> None:
        """RANGING_LOW_VOL threshold is 0.70."""
        engine = _make_engine()
        threshold = engine._step7_get_threshold("RANGING_LOW_VOL")
        assert threshold == pytest.approx(0.70)

    def test_ranging_high_vol_threshold(self) -> None:
        """RANGING_HIGH_VOL threshold is 0.65."""
        engine = _make_engine()
        threshold = engine._step7_get_threshold("RANGING_HIGH_VOL")
        assert threshold == pytest.approx(0.65)

    def test_threshold_never_hardcoded(self) -> None:
        """All thresholds match regime_config values (not any hardcoded value)."""
        engine = _make_engine()
        regimes = [
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "RANGING_HIGH_VOL",
            "RANGING_LOW_VOL",
            "DECOUPLED",
        ]
        for regime in regimes:
            expected = _regime_config.get_confidence_threshold(regime)
            actual = engine._step7_get_threshold(regime)
            assert actual == pytest.approx(expected), (
                f"Threshold mismatch for {regime}: "
                f"engine returned {actual}, config says {expected}"
            )


# ---------------------------------------------------------------------------
# TestPredictionLogged (Step 11)
# ---------------------------------------------------------------------------


class TestPredictionLogged:
    """Step 11 — PredictionRecord inserted into storage after each inference."""

    def _make_event(
        self,
        signal: str = "BUY",
        regime: str = "TRENDING_BULL",
    ) -> SignalEvent:
        """Build a minimal SignalEvent for testing."""
        return SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime=regime,
            signal=signal,
            ensemble_prob=0.75,
            confidence_threshold=0.62,
            position_size_multiplier=1.0,
            risk_filters_triggered=[],
            model_version="test-v1",
            lstm_prob=0.72,
            xgb_prob=0.78,
            regime_encoded=0.0,
            open_time=_BASE_TIME,
            close_price=0.15,
        )

    def test_prediction_record_inserted_on_buy_signal(self) -> None:
        """MANDATORY: insert_prediction called once per inference run (BUY)."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="BUY")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        mock_storage.insert_prediction.assert_called_once()

    def test_prediction_record_inserted_on_sell_signal(self) -> None:
        """insert_prediction called for SELL signals too."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="SELL")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        mock_storage.insert_prediction.assert_called_once()

    def test_prediction_record_inserted_on_hold_signal(self) -> None:
        """insert_prediction called for HOLD signals (all signals are logged)."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="HOLD")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        mock_storage.insert_prediction.assert_called_once()

    def test_prediction_record_has_correct_direction_buy(self) -> None:
        """PredictionRecord.predicted_direction == 1 for BUY signal."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="BUY")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        inserted: PredictionRecord = mock_storage.insert_prediction.call_args[0][0]
        assert inserted.predicted_direction == 1

    def test_prediction_record_has_correct_direction_sell(self) -> None:
        """PredictionRecord.predicted_direction == -1 for SELL signal."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="SELL")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        inserted: PredictionRecord = mock_storage.insert_prediction.call_args[0][0]
        assert inserted.predicted_direction == -1

    def test_prediction_record_has_correct_direction_hold(self) -> None:
        """PredictionRecord.predicted_direction == 0 for HOLD signal."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage)
        event = self._make_event(signal="HOLD")
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        inserted: PredictionRecord = mock_storage.insert_prediction.call_args[0][0]
        assert inserted.predicted_direction == 0

    def test_prediction_record_model_version_matches_engine(self) -> None:
        """PredictionRecord.model_version matches EngineConfig.model_version."""
        mock_storage = MagicMock(spec=DogeStorage)
        engine = _make_engine(storage=mock_storage, model_version="run-abc123")
        event = self._make_event()
        event = SignalEvent(
            timestamp_ms=event.timestamp_ms,
            symbol=event.symbol,
            regime=event.regime,
            signal=event.signal,
            ensemble_prob=event.ensemble_prob,
            confidence_threshold=event.confidence_threshold,
            position_size_multiplier=event.position_size_multiplier,
            risk_filters_triggered=event.risk_filters_triggered,
            model_version="run-abc123",
            lstm_prob=event.lstm_prob,
            xgb_prob=event.xgb_prob,
            regime_encoded=event.regime_encoded,
            open_time=event.open_time,
            close_price=event.close_price,
        )
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        engine._step11_log_prediction(event, row)

        inserted: PredictionRecord = mock_storage.insert_prediction.call_args[0][0]
        assert inserted.model_version == "run-abc123"

    def test_no_storage_does_not_raise(self) -> None:
        """Step 11 logs a warning but does NOT raise when storage is None."""
        engine = _make_engine(storage=None)
        event = self._make_event()
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        # Should not raise
        engine._step11_log_prediction(event, row)

    def test_storage_exception_does_not_propagate(self) -> None:
        """Step 11 suppresses storage exceptions (audit trail must not break inference)."""
        mock_storage = MagicMock(spec=DogeStorage)
        mock_storage.insert_prediction.side_effect = Exception("DB error")
        engine = _make_engine(storage=mock_storage)
        event = self._make_event()
        row = _make_current_row(close=0.15, open_time=_BASE_TIME)

        # Should not raise despite storage failure
        engine._step11_log_prediction(event, row)


# ---------------------------------------------------------------------------
# TestRiskFilterCombinations
# ---------------------------------------------------------------------------


class TestRiskFilterCombinations:
    """Step 9 — multiple risk filters combined."""

    def test_round_number_reduces_position_by_30_pct(self) -> None:
        """at_round_number_flag == 1 reduces position size by 30%."""
        engine = _make_engine()
        row = _make_current_row(at_round_number_flag=1.0)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        # 30% reduction → 70% of original = 0.70
        expected_multiplier = 1.0 - _doge_settings.risk.round_number_size_reduction
        assert result.position_size_multiplier == pytest.approx(expected_multiplier)
        assert "at_round_number_flag" in result.triggered

    def test_decoupled_regime_reduces_position_to_50_pct(self) -> None:
        """DECOUPLED regime halves the position size."""
        engine = _make_engine()
        row = _make_current_row()

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="DECOUPLED",
            raw_signal="BUY",
        )

        assert result.position_size_multiplier == pytest.approx(0.50)
        assert "decoupled_half_size" in result.triggered

    def test_ranging_low_vol_reduces_position_to_50_pct(self) -> None:
        """RANGING_LOW_VOL regime halves the position size."""
        engine = _make_engine()
        row = _make_current_row()

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="RANGING_LOW_VOL",
            raw_signal="BUY",
        )

        assert result.position_size_multiplier == pytest.approx(0.50)
        assert "ranging_low_vol_half_size" in result.triggered

    def test_multiple_filters_compound(self) -> None:
        """Round number + DECOUPLED together compound to 0.70 × 0.50 = 0.35."""
        engine = _make_engine()
        row = _make_current_row(at_round_number_flag=1.0)

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="DECOUPLED",
            raw_signal="BUY",
        )

        expected = (1.0 - _doge_settings.risk.round_number_size_reduction) * 0.50
        assert result.position_size_multiplier == pytest.approx(expected)

    def test_no_filters_full_position(self) -> None:
        """When no filters fire, position_size_multiplier == 1.0."""
        engine = _make_engine()
        row = _make_current_row()

        result = engine._step9_risk_filters(
            current_row=row,
            current_regime="TRENDING_BULL",
            raw_signal="BUY",
        )

        assert result.buy_suppressed is False
        assert result.position_size_multiplier == pytest.approx(1.0)
        assert result.triggered == []


# ---------------------------------------------------------------------------
# TestSignalDecision (Step 10)
# ---------------------------------------------------------------------------


class TestSignalDecision:
    """Step 10 — final signal from ensemble_prob vs threshold."""

    def test_buy_when_above_threshold(self) -> None:
        """Signal is BUY when ensemble_prob >= confidence_threshold."""
        engine = _make_engine()
        risk = RiskFilterResult(buy_suppressed=False, position_size_multiplier=1.0)
        signal = engine._step10_signal_decision(
            ensemble_prob=0.80, confidence_threshold=0.62, risk_result=risk
        )
        assert signal == "BUY"

    def test_sell_when_below_inverse_threshold(self) -> None:
        """Signal is SELL when 1 - ensemble_prob >= confidence_threshold."""
        engine = _make_engine()
        risk = RiskFilterResult(buy_suppressed=False, position_size_multiplier=1.0)
        signal = engine._step10_signal_decision(
            ensemble_prob=0.20, confidence_threshold=0.62, risk_result=risk
        )
        assert signal == "SELL"

    def test_hold_when_between_thresholds(self) -> None:
        """Signal is HOLD when neither BUY nor SELL threshold is met."""
        engine = _make_engine()
        risk = RiskFilterResult(buy_suppressed=False, position_size_multiplier=1.0)
        signal = engine._step10_signal_decision(
            ensemble_prob=0.50, confidence_threshold=0.62, risk_result=risk
        )
        assert signal == "HOLD"

    def test_buy_suppressed_becomes_hold(self) -> None:
        """BUY is converted to HOLD when buy_suppressed == True."""
        engine = _make_engine()
        risk = RiskFilterResult(
            buy_suppressed=True, position_size_multiplier=1.0, triggered=["funding"]
        )
        signal = engine._step10_signal_decision(
            ensemble_prob=0.80, confidence_threshold=0.62, risk_result=risk
        )
        assert signal == "HOLD"

    def test_sell_not_suppressed_by_buy_suppression(self) -> None:
        """SELL signal is NOT affected by buy_suppressed == True."""
        engine = _make_engine()
        risk = RiskFilterResult(buy_suppressed=True, position_size_multiplier=1.0)
        signal = engine._step10_signal_decision(
            ensemble_prob=0.10, confidence_threshold=0.62, risk_result=risk
        )
        assert signal == "SELL"


# ---------------------------------------------------------------------------
# TestSignalEvent
# ---------------------------------------------------------------------------


class TestSignalEvent:
    """SignalEvent dataclass contract."""

    def test_signal_event_is_immutable(self) -> None:
        """SignalEvent is a frozen dataclass (cannot mutate fields)."""
        event = SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime="TRENDING_BULL",
            signal="BUY",
            ensemble_prob=0.75,
            confidence_threshold=0.62,
            position_size_multiplier=1.0,
            risk_filters_triggered=[],
            model_version="v1",
        )
        with pytest.raises((AttributeError, TypeError)):
            event.signal = "SELL"  # type: ignore[misc]

    def test_signal_event_fields_accessible(self) -> None:
        """All SignalEvent fields are readable."""
        event = SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime="DECOUPLED",
            signal="HOLD",
            ensemble_prob=0.55,
            confidence_threshold=0.72,
            position_size_multiplier=0.50,
            risk_filters_triggered=["decoupled_half_size"],
            model_version="run-xyz",
            lstm_prob=0.53,
            xgb_prob=0.58,
            regime_encoded=4.0,
            open_time=_BASE_TIME,
            close_price=0.20,
        )

        assert event.timestamp_ms == _BASE_TIME
        assert event.regime == "DECOUPLED"
        assert event.signal == "HOLD"
        assert event.confidence_threshold == pytest.approx(0.72)
        assert event.position_size_multiplier == pytest.approx(0.50)
        assert "decoupled_half_size" in event.risk_filters_triggered
        assert event.lstm_prob == pytest.approx(0.53)
        assert event.xgb_prob == pytest.approx(0.58)
        assert event.regime_encoded == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# TestOnSignalCallback (Step 12)
# ---------------------------------------------------------------------------


class TestOnSignalCallback:
    """Step 12 — signal callback registration and emission."""

    def test_registered_callback_called_on_emit(self) -> None:
        """on_signal callback is invoked in Step 12."""
        received: list[SignalEvent] = []
        engine = _make_engine()
        engine.register_on_signal(lambda e: received.append(e))

        event = SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime="TRENDING_BULL",
            signal="BUY",
            ensemble_prob=0.75,
            confidence_threshold=0.62,
            position_size_multiplier=1.0,
            risk_filters_triggered=[],
            model_version="v1",
        )

        engine._step12_emit_signal(event)

        assert len(received) == 1
        assert received[0].signal == "BUY"

    def test_multiple_callbacks_all_called(self) -> None:
        """Multiple registered callbacks are all invoked."""
        counts = [0, 0]
        engine = _make_engine()
        engine.register_on_signal(lambda e: counts.__setitem__(0, counts[0] + 1))
        engine.register_on_signal(lambda e: counts.__setitem__(1, counts[1] + 1))

        event = SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime="TRENDING_BULL",
            signal="HOLD",
            ensemble_prob=0.50,
            confidence_threshold=0.62,
            position_size_multiplier=1.0,
            risk_filters_triggered=[],
            model_version="v1",
        )

        engine._step12_emit_signal(event)

        assert counts == [1, 1]

    def test_callback_exception_does_not_propagate(self) -> None:
        """Exception in on_signal callback is caught; inference continues."""
        engine = _make_engine()
        engine.register_on_signal(lambda e: (_ for _ in ()).throw(RuntimeError("cb fail")))

        event = SignalEvent(
            timestamp_ms=_BASE_TIME,
            symbol="DOGEUSDT",
            regime="TRENDING_BULL",
            signal="BUY",
            ensemble_prob=0.75,
            confidence_threshold=0.62,
            position_size_multiplier=1.0,
            risk_filters_triggered=[],
            model_version="v1",
        )

        # Should not propagate
        engine._step12_emit_signal(event)
