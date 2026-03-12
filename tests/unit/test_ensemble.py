"""Unit tests for EnsembleModel, RegimeRouter, HyperparameterOptimizer, and ModelTrainer.

Mandatory tests:
    - EnsembleModel save/load roundtrip produces identical predict_proba output.
    - predict_proba shape == (n_samples,), values in [0, 1].
    - RegimeRouter falls back to global_model for unknown regime.
    - HyperparameterOptimizer returns expected param keys for XGBoost.
    - ModelTrainer.train_full returns TrainingResult with n_folds > 0.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import WalkForwardSettings
from src.models.ensemble import EnsembleModel
from src.models.regime_router import RegimeRouter
from src.models.xgb_model import XGBoostModel
from src.training.hyperopt import HyperparameterOptimizer
from src.training.trainer import ModelTrainer, TrainingResult


# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

_TRAINING_START_MS: int = 1_640_995_200_000   # 2022-01-01 00:00 UTC
_MS_PER_HOUR: int = 3_600_000
_SEED: int = 42


def _make_meta_probs(
    n: int = 200,
    seed: int = _SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Build meta-feature matrix ``(n, 3)`` and binary labels ``(n,)``.

    Columns: [lstm_prob, xgb_prob, regime_encoded]
    """
    rng = np.random.default_rng(seed)
    lstm_prob = rng.uniform(0.2, 0.8, n)
    xgb_prob = rng.uniform(0.2, 0.8, n)
    regime_enc = rng.integers(0, 5, n).astype(float)
    X = np.column_stack([lstm_prob, xgb_prob, regime_enc])
    # Create a weakly predictable target: buy when ensemble average > 0.5
    avg = 0.5 * lstm_prob + 0.5 * xgb_prob
    y = (avg > 0.5).astype(float)
    return X, y


def _make_fitted_xgb(seed: int = _SEED) -> XGBoostModel:
    """Return a small fitted XGBoostModel for router tests."""
    rng = np.random.default_rng(seed)
    n = 300
    X_tr = rng.standard_normal((n, 5)).astype(np.float32)
    y_tr = rng.integers(0, 2, n).astype(float)
    X_v = rng.standard_normal((50, 5)).astype(np.float32)
    y_v = rng.integers(0, 2, 50).astype(float)
    model = XGBoostModel()
    model.fit(X_tr, y_tr, X_v, y_v)
    return model


def _fast_wf_cfg() -> WalkForwardSettings:
    """Return a fast walk-forward config for unit tests."""
    return WalkForwardSettings(
        training_window_days=15,
        validation_window_days=5,
        step_size_days=5,
        min_training_rows=100,
    )


def _make_feature_df(
    n_days: int = 60,
    n_features: int = 4,
    seed: int = _SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a minimal feature DataFrame + aligned regime labels.

    Returns:
        (feature_df, regime_labels) tuple ready for ModelTrainer.
    """
    n = n_days * 24  # 1h candles
    rng = np.random.default_rng(seed)

    open_times = [_TRAINING_START_MS + i * _MS_PER_HOUR for i in range(n)]

    features = {
        f"feat_{j}": rng.standard_normal(n).astype(np.float64)
        for j in range(n_features)
    }
    # Binary target derived from first feature (weakly predictable)
    close_proxy = np.cumsum(rng.standard_normal(n)) + 100
    target = (np.diff(close_proxy, prepend=close_proxy[0]) > 0).astype(float)

    df = pd.DataFrame(
        {
            "open_time": open_times,
            "era": "training",
            "target": target,
            **features,
        }
    )

    # Regime labels: cycle through the 5 regimes
    all_regimes = [
        "TRENDING_BULL",
        "TRENDING_BEAR",
        "RANGING_HIGH_VOL",
        "RANGING_LOW_VOL",
        "DECOUPLED",
    ]
    regime_list = [all_regimes[i % 5] for i in range(n)]
    regime_labels = pd.Series(regime_list, index=df.index, name="regime_label")

    return df, regime_labels


# ===========================================================================
# TestEnsembleModel
# ===========================================================================


class TestEnsembleModel:
    """Unit tests for EnsembleModel."""

    def test_fit_returns_metrics_keys(self) -> None:
        """fit() returns dict with val_accuracy, train_accuracy, n_train, n_val."""
        X, y = _make_meta_probs(200)
        n_train = 160
        model = EnsembleModel()
        metrics = model.fit(X[:n_train], y[:n_train], X[n_train:], y[n_train:])
        assert "val_accuracy" in metrics
        assert "train_accuracy" in metrics
        assert "n_train" in metrics
        assert "n_val" in metrics

    def test_val_accuracy_in_range(self) -> None:
        """val_accuracy is in [0, 1]."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        metrics = model.fit(X[:160], y[:160], X[160:], y[160:])
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_fit_sets_is_fitted(self) -> None:
        """fit() marks _is_fitted = True."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        assert not model._is_fitted
        model.fit(X[:160], y[:160], X[160:], y[160:])
        assert model._is_fitted

    def test_predict_proba_shape(self) -> None:
        """MANDATORY: predict_proba output shape == (n_samples,)."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        proba = model.predict_proba(X)
        assert proba.shape == (200,), f"Expected (200,), got {proba.shape}"

    def test_predict_proba_values_in_unit_range(self) -> None:
        """MANDATORY: all predict_proba values in [0, 1]."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        proba = model.predict_proba(X)
        assert np.all(proba >= 0.0), "predict_proba contains values < 0"
        assert np.all(proba <= 1.0), "predict_proba contains values > 1"

    def test_predict_proba_dtype_float64(self) -> None:
        """predict_proba output dtype is float64."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        proba = model.predict_proba(X)
        assert proba.dtype == np.float64

    def test_predict_proba_raises_before_fit(self) -> None:
        """predict_proba raises RuntimeError before fit."""
        model = EnsembleModel()
        X, _ = _make_meta_probs(10)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_wrong_n_cols_raises_ValueError(self) -> None:
        """ValueError raised when X has wrong number of columns."""
        X_bad = np.random.default_rng(0).random((50, 2))
        X_good, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X_good[:160], y[:160], X_good[160:], y[160:])
        with pytest.raises(ValueError, match="3 columns"):
            model.predict_proba(X_bad)

    def test_fit_wrong_n_cols_raises_ValueError(self) -> None:
        """ValueError raised when X_train has wrong number of columns during fit."""
        rng = np.random.default_rng(0)
        X_bad = rng.random((160, 5))
        y = rng.integers(0, 2, 160).astype(float)
        X_val, y_val = _make_meta_probs(40)
        model = EnsembleModel()
        with pytest.raises(ValueError, match="3 columns"):
            model.fit(X_bad, y, X_val, y_val)

    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        """save() creates ensemble_model.pkl and ensemble_metadata.json."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        model.save(tmp_path)
        assert (tmp_path / "ensemble_model.pkl").exists()
        assert (tmp_path / "ensemble_metadata.json").exists()

    def test_metadata_json_contains_expected_keys(self, tmp_path: Path) -> None:
        """ensemble_metadata.json contains C, max_iter, seed, n_train_samples."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel(C=0.5, max_iter=500, seed=7)
        model.fit(X[:160], y[:160], X[160:], y[160:])
        model.save(tmp_path)
        with open(tmp_path / "ensemble_metadata.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert meta["C"] == 0.5
        assert meta["max_iter"] == 500
        assert meta["seed"] == 7
        assert meta["n_train_samples"] == 160

    def test_save_load_roundtrip_identical_predictions(self, tmp_path: Path) -> None:
        """MANDATORY: save/load roundtrip — predict_proba identical after reload."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        proba_before = model.predict_proba(X)

        model.save(tmp_path)

        loaded = EnsembleModel()
        loaded.load(tmp_path)
        proba_after = loaded.predict_proba(X)

        np.testing.assert_array_almost_equal(proba_before, proba_after, decimal=10)

    def test_load_sets_is_fitted(self, tmp_path: Path) -> None:
        """load() sets _is_fitted = True."""
        X, y = _make_meta_probs(200)
        model = EnsembleModel()
        model.fit(X[:160], y[:160], X[160:], y[160:])
        model.save(tmp_path)

        loaded = EnsembleModel()
        assert not loaded._is_fitted
        loaded.load(tmp_path)
        assert loaded._is_fitted

    def test_load_raises_if_model_file_missing(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError if pkl is absent."""
        model = EnsembleModel()
        with pytest.raises(FileNotFoundError):
            model.load(tmp_path)

    def test_empty_y_train_raises(self) -> None:
        """fit() raises ValueError when y_train is empty."""
        X = np.empty((0, 3))
        y = np.empty(0)
        X_val, y_val = _make_meta_probs(20)
        model = EnsembleModel()
        with pytest.raises(ValueError):
            model.fit(X, y, X_val, y_val)


# ===========================================================================
# TestRegimeRouter
# ===========================================================================


class TestRegimeRouter:
    """Unit tests for RegimeRouter."""

    def test_route_returns_correct_model(self) -> None:
        """route() returns the regime-specific model when available."""
        bull = _make_fitted_xgb(seed=0)
        bear = _make_fitted_xgb(seed=1)
        router = RegimeRouter(
            regime_models={"TRENDING_BULL": bull, "TRENDING_BEAR": bear}
        )
        assert router.route("TRENDING_BULL") is bull
        assert router.route("TRENDING_BEAR") is bear

    def test_route_falls_back_to_global_model(self) -> None:
        """route() returns global_model when regime not found."""
        bull = _make_fitted_xgb(seed=0)
        global_m = _make_fitted_xgb(seed=2)
        router = RegimeRouter(
            regime_models={"TRENDING_BULL": bull},
            global_model=global_m,
        )
        result = router.route("DECOUPLED")
        assert result is global_m

    def test_route_raises_ValueError_when_no_model(self) -> None:
        """route() raises ValueError when regime not found and no global model."""
        router = RegimeRouter(regime_models={"TRENDING_BULL": _make_fitted_xgb()})
        with pytest.raises(ValueError, match="no model available"):
            router.route("DECOUPLED")

    def test_has_regime_true_false(self) -> None:
        """has_regime() returns True/False correctly."""
        router = RegimeRouter(regime_models={"TRENDING_BULL": _make_fitted_xgb()})
        assert router.has_regime("TRENDING_BULL") is True
        assert router.has_regime("DECOUPLED") is False

    def test_available_regimes_sorted(self) -> None:
        """available_regimes() returns sorted list of registered regimes."""
        router = RegimeRouter(
            regime_models={
                "TRENDING_BEAR": _make_fitted_xgb(seed=1),
                "TRENDING_BULL": _make_fitted_xgb(seed=0),
            }
        )
        regimes = router.available_regimes()
        assert regimes == sorted(regimes)
        assert set(regimes) == {"TRENDING_BULL", "TRENDING_BEAR"}

    def test_router_with_only_global_model(self) -> None:
        """Router with no regime models works if global model is set."""
        global_m = _make_fitted_xgb()
        router = RegimeRouter(global_model=global_m)
        assert router.route("TRENDING_BULL") is global_m
        assert router.route("DECOUPLED") is global_m

    def test_has_global_model(self) -> None:
        """has_global_model() returns True when a global model is registered."""
        router_with = RegimeRouter(global_model=_make_fitted_xgb())
        router_without = RegimeRouter(regime_models={})
        assert router_with.has_global_model() is True
        assert router_without.has_global_model() is False

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Save/load roundtrip — route still works after reload."""
        bull = _make_fitted_xgb(seed=0)
        global_m = _make_fitted_xgb(seed=1)
        router = RegimeRouter(
            regime_models={"TRENDING_BULL": bull},
            global_model=global_m,
        )
        router.save(tmp_path)

        loaded_router = RegimeRouter()
        loaded_router.load(tmp_path)

        assert loaded_router.has_regime("TRENDING_BULL")
        assert loaded_router.has_global_model()

        # Should route without error
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((10, 5)).astype(np.float32)
        routed = loaded_router.route("TRENDING_BULL")
        proba = routed.predict_proba(X_test)
        assert proba.shape == (10,)

    def test_save_load_metadata_file(self, tmp_path: Path) -> None:
        """save() writes router_metadata.json with correct keys."""
        router = RegimeRouter(
            regime_models={"TRENDING_BULL": _make_fitted_xgb()},
            global_model=_make_fitted_xgb(seed=1),
        )
        router.save(tmp_path)
        assert (tmp_path / "router_metadata.json").exists()
        with open(tmp_path / "router_metadata.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert "available_regimes" in meta
        assert meta["has_global"] is True

    def test_load_raises_if_metadata_missing(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError if router_metadata.json is absent."""
        router = RegimeRouter()
        with pytest.raises(FileNotFoundError):
            router.load(tmp_path)


# ===========================================================================
# TestHyperparameterOptimizer
# ===========================================================================


class TestHyperparameterOptimizer:
    """Unit tests for HyperparameterOptimizer."""

    # Shared data for XGB tests (large enough for a few folds)
    @staticmethod
    def _xgb_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        # AR(1) features for reasonable XGB performance
        n_train, n_val, n_feats = 300, 60, 8
        X_tr = rng.standard_normal((n_train, n_feats)).astype(np.float32)
        y_tr = rng.integers(0, 2, n_train).astype(float)
        X_v = rng.standard_normal((n_val, n_feats)).astype(np.float32)
        y_v = rng.integers(0, 2, n_val).astype(float)
        return X_tr, y_tr, X_v, y_v

    @staticmethod
    def _lstm_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Minimal data for fast LSTM hyperopt (1 trial, 2 epochs)."""
        rng = np.random.default_rng(seed)
        # Need n_samples > max_seq_len(80) + 1 for a valid sequence
        n_train, n_val = 120, 30
        n_feats = 4
        X_tr = rng.standard_normal((n_train, n_feats)).astype(np.float32)
        y_tr = rng.integers(0, 2, n_train).astype(float)
        X_v = rng.standard_normal((n_val, n_feats)).astype(np.float32)
        y_v = rng.integers(0, 2, n_val).astype(float)
        return X_tr, y_tr, X_v, y_v

    def test_optimize_xgb_returns_expected_keys(self) -> None:
        """optimize(XGBoostModel) returns dict with XGB param keys."""
        X_tr, y_tr, X_v, y_v = self._xgb_data()
        opt = HyperparameterOptimizer(seed=_SEED)
        best = opt.optimize(XGBoostModel, X_tr, y_tr, X_v, y_v, n_trials=2)
        assert "max_depth" in best
        assert "learning_rate" in best
        assert "subsample" in best
        assert "colsample_bytree" in best

    def test_optimize_xgb_param_ranges(self) -> None:
        """Best XGB params fall within the defined search space."""
        X_tr, y_tr, X_v, y_v = self._xgb_data()
        opt = HyperparameterOptimizer(seed=_SEED)
        best = opt.optimize(XGBoostModel, X_tr, y_tr, X_v, y_v, n_trials=3)
        assert 3 <= best["max_depth"] <= 8
        assert 0.01 <= best["learning_rate"] <= 0.1
        assert 0.6 <= best["subsample"] <= 1.0
        assert 0.6 <= best["colsample_bytree"] <= 1.0

    def test_optimize_lstm_returns_expected_keys(self) -> None:
        """optimize(LSTMModel) returns dict with LSTM param keys."""
        from src.models.lstm_model import LSTMModel

        X_tr, y_tr, X_v, y_v = self._lstm_data()
        opt = HyperparameterOptimizer(seed=_SEED)
        # 1 trial only; override max_epochs via LSTMModel default (slow otherwise)
        best = opt.optimize(LSTMModel, X_tr, y_tr, X_v, y_v, n_trials=1)
        assert "sequence_length" in best
        assert "dropout" in best
        assert "hidden_units" in best

    def test_unsupported_model_class_raises_ValueError(self) -> None:
        """ValueError raised for unsupported model_class."""
        class FakeModel:
            pass

        X_tr, y_tr, X_v, y_v = self._xgb_data()
        opt = HyperparameterOptimizer(seed=_SEED)
        with pytest.raises(ValueError, match="unsupported model_class"):
            opt.optimize(FakeModel, X_tr, y_tr, X_v, y_v, n_trials=1)

    def test_n_trials_respected(self) -> None:
        """Number of completed trials matches n_trials."""
        import optuna

        X_tr, y_tr, X_v, y_v = self._xgb_data()

        trial_count: list[int] = []

        original_optimize = optuna.Study.optimize

        def counting_optimize(self, func, n_trials=None, **kwargs):
            trial_count.append(n_trials or 0)
            return original_optimize(self, func, n_trials=n_trials, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(optuna.Study, "optimize", counting_optimize):
            opt = HyperparameterOptimizer(seed=_SEED)
            opt.optimize(XGBoostModel, X_tr, y_tr, X_v, y_v, n_trials=4)

        assert trial_count[0] == 4


# ===========================================================================
# TestModelTrainer
# ===========================================================================


class TestModelTrainer:
    """Unit tests for ModelTrainer.train_full."""

    def test_train_full_returns_training_result(self) -> None:
        """train_full() returns a TrainingResult instance."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        result = trainer.train_full(df, labels)
        assert isinstance(result, TrainingResult)

    def test_n_folds_positive(self) -> None:
        """n_folds > 0 after a successful training run."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        result = trainer.train_full(df, labels)
        assert result.n_folds > 0, f"Expected n_folds > 0, got {result.n_folds}"

    def test_mean_val_accuracy_in_range(self) -> None:
        """mean_val_accuracy is in [0, 1]."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        result = trainer.train_full(df, labels)
        assert 0.0 <= result.mean_val_accuracy <= 1.0

    def test_seed_used_matches_config(self) -> None:
        """seed_used equals the global settings seed (42)."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        result = trainer.train_full(df, labels)
        from src.config import settings
        assert result.seed_used == settings.project.seed

    def test_skipped_regimes_list_present(self) -> None:
        """skipped_regimes is a list (may be empty or non-empty)."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        result = trainer.train_full(df, labels)
        assert isinstance(result.skipped_regimes, list)

    def test_missing_required_columns_raises_ValueError(self) -> None:
        """ValueError raised when feature_df is missing 'target'."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        df_bad = df.drop(columns=["target"])
        trainer = ModelTrainer(
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        with pytest.raises(ValueError, match="missing required columns"):
            trainer.train_full(df_bad, labels)

    def test_output_dir_saves_artefacts(self, tmp_path: Path) -> None:
        """model artefacts are written to output_dir when set."""
        df, labels = _make_feature_df(n_days=60, n_features=4)
        trainer = ModelTrainer(
            output_dir=tmp_path,
            walk_forward_cfg=_fast_wf_cfg(),
            run_hyperopt=False,
        )
        trainer.train_full(df, labels)
        # At minimum: feature_columns.json and scaler.pkl should exist
        assert (tmp_path / "feature_columns.json").exists(), \
            "feature_columns.json not written"
        assert (tmp_path / "scaler.pkl").exists(), "scaler.pkl not written"

    def test_scaler_per_fold_isolation(self) -> None:
        """Each fold uses an independent FoldScaler (RULE B).

        We monkey-patch FoldScaler to count instantiations, then verify
        that at least n_folds distinct scalers were created.
        """
        from src.training import trainer as trainer_module
        import unittest.mock as mock

        instantiation_count: list[int] = [0]
        _OriginalFoldScaler = trainer_module.FoldScaler

        class CountingFoldScaler(_OriginalFoldScaler):
            def __init__(self) -> None:
                super().__init__()
                instantiation_count[0] += 1

        df, labels = _make_feature_df(n_days=60, n_features=4)

        with mock.patch.object(trainer_module, "FoldScaler", CountingFoldScaler):
            tr = ModelTrainer(
                walk_forward_cfg=_fast_wf_cfg(),
                run_hyperopt=False,
            )
            result = tr.train_full(df, labels)

        # Should have at least n_folds + 1 scalers (fold probes + final)
        assert instantiation_count[0] >= result.n_folds, (
            f"Expected >= {result.n_folds} FoldScaler instances, "
            f"got {instantiation_count[0]}"
        )
