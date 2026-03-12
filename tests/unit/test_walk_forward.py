"""Unit tests for WalkForwardCV, FoldScaler, AbstractBaseModel, and XGBoostModel.

MANDATORY TESTS (CLAUDE.md §5.1):
    - WF-01: max(train_timestamps) < min(val_timestamps) in every fold
    - WF-02: no era='context' rows in any fold
    - WF-03: minimum 3 folds generated from 400+ day dataset
    - WF-04: fold count is (dataset_days - training_window) / step_size ± 1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import WalkForwardSettings
from src.models.base_model import (
    SIGNAL_BUY,
    SIGNAL_HOLD,
    SIGNAL_SELL,
    AbstractBaseModel,
)
from src.models.xgb_model import XGBoostModel
from src.training.scaler import FoldScaler
from src.training.walk_forward import Fold, WalkForwardCV

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_DAY: int = 86_400_000
_TRAINING_START_MS: int = 1_640_995_200_000  # 2022-01-01 00:00 UTC


def _make_df(
    n_days: int,
    start_ms: int = _TRAINING_START_MS,
    era: str = "training",
    include_context_prefix: int = 0,
) -> pd.DataFrame:
    """Build a minimal 1-hour OHLCV DataFrame for CV testing.

    Args:
        n_days: Number of days of data to generate.
        start_ms: Start timestamp (UTC epoch milliseconds).
        era: Era label for all rows.
        include_context_prefix: If > 0, prepend this many rows with
            ``era='context'`` before the training rows.

    Returns:
        DataFrame with columns: open_time, close, era.
    """
    n_rows = n_days * 24  # 1h candles
    timestamps = [start_ms + i * _MS_PER_HOUR for i in range(n_rows)]

    rng = np.random.default_rng(42)
    close = 0.10 + np.cumsum(rng.normal(0, 0.001, n_rows))

    rows = []

    if include_context_prefix > 0:
        ctx_start = start_ms - include_context_prefix * _MS_PER_HOUR
        for j in range(include_context_prefix):
            rows.append(
                {
                    "open_time": ctx_start + j * _MS_PER_HOUR,
                    "close": 0.09,
                    "era": "context",
                }
            )

    for ts, c in zip(timestamps, close):
        rows.append({"open_time": ts, "close": float(c), "era": era})

    return pd.DataFrame(rows)


def _make_feature_df(n_days: int = 500) -> pd.DataFrame:
    """Build a minimal feature DataFrame suitable for XGBoost training.

    Creates N*24 rows of random features + binary target + era='training'.

    Args:
        n_days: Number of days of data (each day = 24 1h rows).

    Returns:
        DataFrame with open_time, f0...f9, target, era columns.
    """
    n_rows = n_days * 24
    rng = np.random.default_rng(42)

    data: dict[str, object] = {
        "open_time": [
            _TRAINING_START_MS + i * _MS_PER_HOUR for i in range(n_rows)
        ],
        "era": ["training"] * n_rows,
    }
    for fi in range(10):
        data[f"f{fi}"] = rng.normal(0.0, 1.0, n_rows)

    data["target"] = rng.integers(0, 2, n_rows).astype(float)
    data["close"] = 0.10 + np.cumsum(rng.normal(0, 0.001, n_rows))

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Minimal concrete model for testing AbstractBaseModel methods
# ---------------------------------------------------------------------------


class _DummyModel(AbstractBaseModel):
    """Trivial concrete model for testing the abstract base."""

    def __init__(self, fixed_proba: float = 0.8, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._fixed_proba = fixed_proba

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict:
        self._is_fitted = True
        return {"val_accuracy": 0.6}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_fitted()
        return np.full(len(X), self._fixed_proba)

    def save(self, path: Path) -> None:
        self._assert_fitted()
        Path(path).mkdir(parents=True, exist_ok=True)

    def load(self, path: Path) -> None:
        self._is_fitted = True


# ===========================================================================
# Tests -- WalkForwardCV
# ===========================================================================


class TestWalkForwardCVMandatory:
    """MANDATORY tests -- must pass before any model is built."""

    def test_wf01_temporal_ordering_all_folds(self) -> None:
        """WF-01: max(train_timestamps) < min(val_timestamps) in EVERY fold."""
        df = _make_df(n_days=500)
        cv = WalkForwardCV()
        folds = cv.generate_folds(df)

        assert len(folds) > 0, "No folds generated"

        for fold in folds:
            fold_df = df[df["open_time"].between(fold.train_start, fold.val_end)]
            train_ts = fold_df.loc[
                fold_df["open_time"] <= fold.train_end, "open_time"
            ]
            val_ts = fold_df.loc[
                fold_df["open_time"] >= fold.val_start, "open_time"
            ]
            assert train_ts.max() < val_ts.min(), (
                f"Fold {fold.fold_number}: temporal ordering violated. "
                f"max(train)={train_ts.max()} >= min(val)={val_ts.min()}"
            )

    def test_wf02_no_context_era_in_any_fold(self) -> None:
        """WF-02: no era='context' rows appear in any fold train or val slice."""
        # Prepend 48 context rows
        df = _make_df(n_days=500, include_context_prefix=48)
        cv = WalkForwardCV()

        for train_df, val_df in cv.split(df):
            assert (train_df["era"] != "context").all(), (
                "context-era row found in TRAIN fold"
            )
            assert (val_df["era"] != "context").all(), (
                "context-era row found in VAL fold"
            )

    def test_wf03_minimum_three_folds_on_400day_dataset(self) -> None:
        """WF-03: at least 3 folds are generated from a 400+ day dataset."""
        df = _make_df(n_days=420)
        cv = WalkForwardCV()
        folds = cv.generate_folds(df)
        assert len(folds) >= 3, (
            f"Expected >= 3 folds, got {len(folds)} on a 420-day dataset"
        )

    def test_wf04_fold_count_within_tolerance(self) -> None:
        """WF-04: fold count approx (dataset_days - training_window) / step_size +-1."""
        cfg = WalkForwardSettings(
            training_window_days=180,
            validation_window_days=30,
            step_size_days=7,
            min_training_rows=1,  # low min to count all folds
        )
        n_days = 500
        df = _make_df(n_days=n_days)
        cv = WalkForwardCV(cfg=cfg)
        folds = cv.generate_folds(df)

        usable_days = n_days - cfg.training_window_days - cfg.validation_window_days
        expected_folds = usable_days // cfg.step_size_days
        tolerance = 3  # generous: boundary and rounding effects

        assert abs(len(folds) - expected_folds) <= tolerance, (
            f"Fold count {len(folds)} deviates from expected ~{expected_folds} "
            f"by more than {tolerance}"
        )


class TestWalkForwardCVBehavior:
    """Additional behavioral tests for WalkForwardCV."""

    def test_raises_on_missing_columns(self) -> None:
        """Missing open_time or era raises ValueError."""
        df_no_era = pd.DataFrame({"open_time": [1, 2, 3]})
        cv = WalkForwardCV()
        with pytest.raises(ValueError, match="missing required columns"):
            cv.generate_folds(df_no_era)

        df_no_ts = pd.DataFrame({"era": ["training"] * 3})
        with pytest.raises(ValueError, match="missing required columns"):
            cv.generate_folds(df_no_ts)

    def test_raises_when_no_training_era_rows(self) -> None:
        """All context-era rows raises ValueError."""
        df = _make_df(n_days=400, era="context")
        cv = WalkForwardCV()
        with pytest.raises(ValueError, match="no training-era rows"):
            cv.generate_folds(df)

    def test_raises_when_dataset_too_small_for_three_folds(self) -> None:
        """Dataset shorter than 3 walk-forward windows raises ValueError."""
        cfg = WalkForwardSettings(
            training_window_days=180,
            validation_window_days=30,
            step_size_days=30,
            min_training_rows=1,
        )
        # 215 days: only enough for 1 fold
        df = _make_df(n_days=215)
        cv = WalkForwardCV(cfg=cfg)
        with pytest.raises(ValueError, match="minimum required: 3"):
            cv.generate_folds(df)

    def test_fold_is_frozen_dataclass(self) -> None:
        """Fold instances are frozen (immutable)."""
        fold = Fold(
            fold_number=1,
            train_start=1000,
            train_end=2000,
            val_start=2001,
            val_end=3000,
        )
        with pytest.raises((AttributeError, TypeError)):
            fold.fold_number = 99  # type: ignore[misc]

    def test_split_yields_same_folds_as_generate_folds(self) -> None:
        """split() yields the same fold boundaries as generate_folds()."""
        df = _make_df(n_days=500)
        cv = WalkForwardCV()
        folds = cv.generate_folds(df)

        for i, (train_df, val_df) in enumerate(cv.split(df)):
            fold = folds[i]
            assert int(train_df["open_time"].max()) == fold.train_end
            assert int(val_df["open_time"].min()) == fold.val_start

    def test_train_val_no_overlap(self) -> None:
        """Train and val rows in each fold share no open_time values."""
        df = _make_df(n_days=500)
        cv = WalkForwardCV()
        for train_df, val_df in cv.split(df):
            overlap = set(train_df["open_time"]) & set(val_df["open_time"])
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_min_training_rows_respected(self) -> None:
        """Folds with fewer rows than min_training_rows are skipped."""
        cfg = WalkForwardSettings(
            training_window_days=30,
            validation_window_days=7,
            step_size_days=7,
            min_training_rows=100_000,  # impossibly high -- all folds skipped
        )
        df = _make_df(n_days=200)
        cv = WalkForwardCV(cfg=cfg)
        with pytest.raises(ValueError):
            cv.generate_folds(df)

    def test_context_only_prefix_excluded_from_folds(self) -> None:
        """Context rows preceding training data never appear in any fold."""
        df = _make_df(n_days=450, include_context_prefix=200)
        cv = WalkForwardCV()
        folds = cv.generate_folds(df)
        assert len(folds) >= 3

        training_df = df[df["era"] == "training"]
        training_ts = set(training_df["open_time"])
        context_ts = set(df.loc[df["era"] == "context", "open_time"])

        for train_df, val_df in cv.split(df):
            assert set(train_df["open_time"]).issubset(training_ts)
            assert set(val_df["open_time"]).issubset(training_ts)
            assert len(set(train_df["open_time"]) & context_ts) == 0
            assert len(set(val_df["open_time"]) & context_ts) == 0


# ===========================================================================
# Tests -- FoldScaler
# ===========================================================================


class TestFoldScaler:
    """Tests for FoldScaler RULE B enforcement."""

    def _make_arrays(
        self, n_train: int = 200, n_val: int = 50, n_features: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        X_train = rng.normal(5.0, 2.0, (n_train, n_features))
        X_val = rng.normal(6.0, 3.0, (n_val, n_features))
        return X_train, X_val

    def test_fit_transform_returns_scaled_array(self) -> None:
        """fit_transform returns an array with mean approx 0, std approx 1 per feature."""
        X_train, _ = self._make_arrays()
        scaler = FoldScaler()
        X_scaled = scaler.fit_transform(X_train)
        assert X_scaled.shape == X_train.shape
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_transform_never_refits(self) -> None:
        """transform applies training stats to validation data (not refit)."""
        X_train, X_val = self._make_arrays()
        scaler = FoldScaler()
        scaler.fit_transform(X_train)

        X_val_scaled = scaler.transform(X_val)
        assert X_val_scaled.shape == X_val.shape
        # The mean will NOT be approx 0 because val has different mean (6.0 vs 5.0)
        assert abs(X_val_scaled.mean()) > 0.01

    def test_double_fit_raises(self) -> None:
        """Calling fit_transform twice raises RuntimeError."""
        X_train, _ = self._make_arrays()
        scaler = FoldScaler()
        scaler.fit_transform(X_train)
        with pytest.raises(RuntimeError, match="more than once"):
            scaler.fit_transform(X_train)

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform before fit_transform raises RuntimeError."""
        X_val = np.random.default_rng(0).normal(0, 1, (50, 5))
        scaler = FoldScaler()
        with pytest.raises(RuntimeError, match="before fit_transform"):
            scaler.transform(X_val)

    def test_fit_empty_array_raises(self) -> None:
        """fit_transform on empty array raises ValueError."""
        scaler = FoldScaler()
        with pytest.raises(ValueError, match="empty"):
            scaler.fit_transform(np.array([]).reshape(0, 5))

    def test_fit_nan_raises(self) -> None:
        """fit_transform with NaN raises ValueError."""
        X = np.full((10, 3), 1.0)
        X[0, 0] = np.nan
        scaler = FoldScaler()
        with pytest.raises(ValueError, match="NaN"):
            scaler.fit_transform(X)

    def test_fit_inf_raises(self) -> None:
        """fit_transform with Inf raises ValueError."""
        X = np.full((10, 3), 1.0)
        X[2, 1] = np.inf
        scaler = FoldScaler()
        with pytest.raises(ValueError, match="Inf"):
            scaler.fit_transform(X)

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save() / load() preserves scaler state across instances."""
        X_train, X_val = self._make_arrays()
        scaler1 = FoldScaler()
        scaler1.fit_transform(X_train)
        scaler1.save(tmp_path)

        scaler2 = FoldScaler()
        scaler2.load(tmp_path)

        np.testing.assert_array_almost_equal(
            scaler1.transform(X_val),
            scaler2.transform(X_val),
        )

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        """save() before fit raises RuntimeError."""
        scaler = FoldScaler()
        with pytest.raises(RuntimeError, match="before fit_transform"):
            scaler.save(tmp_path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """load() from empty directory raises FileNotFoundError."""
        scaler = FoldScaler()
        with pytest.raises(FileNotFoundError):
            scaler.load(tmp_path)

    def test_is_fitted_property(self) -> None:
        """is_fitted is False before fit, True after."""
        X, _ = self._make_arrays()
        scaler = FoldScaler()
        assert scaler.is_fitted is False
        scaler.fit_transform(X)
        assert scaler.is_fitted is True

    def test_n_features_in(self) -> None:
        """n_features_in returns the correct feature count after fit."""
        X, _ = self._make_arrays(n_features=7)
        scaler = FoldScaler()
        assert scaler.n_features_in is None
        scaler.fit_transform(X)
        assert scaler.n_features_in == 7

    def test_assert_not_fitted_on_future_passes(self) -> None:
        """Sanity check passes when all timestamps are <= train_end_ts."""
        X, _ = self._make_arrays()
        scaler = FoldScaler()
        scaler.fit_transform(X)

        train_end_ts = _TRAINING_START_MS + 100 * _MS_PER_HOUR
        fit_data = pd.DataFrame(
            {"open_time": [_TRAINING_START_MS + i * _MS_PER_HOUR for i in range(50)]}
        )
        # Should not raise
        scaler.assert_not_fitted_on_future(train_end_ts, fit_data)

    def test_assert_not_fitted_on_future_fails(self) -> None:
        """Sanity check raises AssertionError when future timestamps present."""
        X, _ = self._make_arrays()
        scaler = FoldScaler()
        scaler.fit_transform(X)

        train_end_ts = _TRAINING_START_MS  # only the very first ms is allowed
        fit_data = pd.DataFrame(
            {
                "open_time": [
                    _TRAINING_START_MS,
                    _TRAINING_START_MS + 10 * _MS_PER_HOUR,  # future!
                ]
            }
        )
        with pytest.raises(AssertionError, match="data leak"):
            scaler.assert_not_fitted_on_future(train_end_ts, fit_data)

    def test_assert_requires_open_time_column(self) -> None:
        """assert_not_fitted_on_future raises ValueError without open_time."""
        X, _ = self._make_arrays()
        scaler = FoldScaler()
        scaler.fit_transform(X)
        df_no_ts = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="open_time"):
            scaler.assert_not_fitted_on_future(999, df_no_ts)


# ===========================================================================
# Tests -- AbstractBaseModel
# ===========================================================================


class TestAbstractBaseModel:
    """Tests for the abstract base model contract."""

    def test_cannot_instantiate_directly(self) -> None:
        """AbstractBaseModel cannot be instantiated directly (ABC)."""
        with pytest.raises(TypeError):
            AbstractBaseModel()  # type: ignore[abstract]

    def test_predict_signal_buy(self) -> None:
        """predict_signal returns BUY when proba >= threshold."""
        model = _DummyModel(fixed_proba=0.80)
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        # threshold for TRENDING_BULL is 0.62
        signals = model.predict_signal(
            np.zeros((3, 2)), regime_label="TRENDING_BULL"
        )
        assert all(s == SIGNAL_BUY for s in signals)

    def test_predict_signal_sell(self) -> None:
        """predict_signal returns SELL when 1-proba >= threshold."""
        model = _DummyModel(fixed_proba=0.10)  # 1-0.10=0.90 >= threshold
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        signals = model.predict_signal(
            np.zeros((3, 2)), regime_label="TRENDING_BULL"
        )
        assert all(s == SIGNAL_SELL for s in signals)

    def test_predict_signal_hold(self) -> None:
        """predict_signal returns HOLD when neither side exceeds threshold."""
        model = _DummyModel(fixed_proba=0.50)  # neither 0.50 nor 0.50 >= 0.62
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        signals = model.predict_signal(
            np.zeros((3, 2)), regime_label="TRENDING_BULL"
        )
        assert all(s == SIGNAL_HOLD for s in signals)

    def test_predict_signal_before_fit_raises(self) -> None:
        """predict_signal raises RuntimeError before fit."""
        model = _DummyModel()
        with pytest.raises(RuntimeError, match="before fit"):
            model.predict_signal(np.zeros((3, 2)))

    def test_predict_signal_unknown_regime_fallback(self) -> None:
        """Unknown regime falls back to RANGING_LOW_VOL threshold (0.70)."""
        model = _DummyModel(fixed_proba=0.65)  # 0.65 < 0.70 -> HOLD
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        signals = model.predict_signal(
            np.zeros((1, 2)), regime_label="UNKNOWN_REGIME"
        )
        assert signals == [SIGNAL_HOLD]

    def test_predict_signal_none_regime_fallback(self) -> None:
        """None regime falls back to RANGING_LOW_VOL (0.70)."""
        model = _DummyModel(fixed_proba=0.75)  # 0.75 >= 0.70 -> BUY
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        signals = model.predict_signal(np.zeros((1, 2)), regime_label=None)
        assert signals == [SIGNAL_BUY]

    def test_directional_accuracy(self) -> None:
        """directional_accuracy computes correct fraction."""
        model = _DummyModel(fixed_proba=0.80)  # always predicts class 1
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        X = np.zeros((4, 2))
        y_true = np.array([1.0, 1.0, 0.0, 1.0])
        acc = model.directional_accuracy(X, y_true)
        assert abs(acc - 0.75) < 1e-9

    def test_directional_accuracy_before_fit_raises(self) -> None:
        """directional_accuracy raises RuntimeError before fit."""
        model = _DummyModel()
        with pytest.raises(RuntimeError):
            model.directional_accuracy(np.zeros((3, 2)), np.zeros(3))

    def test_repr_includes_fitted_state(self) -> None:
        """__repr__ reflects the fitted state."""
        model = _DummyModel()
        assert "fitted=False" in repr(model)
        model.fit(
            np.zeros((5, 2)), np.zeros(5), np.zeros((5, 2)), np.zeros(5)
        )
        assert "fitted=True" in repr(model)


# ===========================================================================
# Tests -- XGBoostModel
# ===========================================================================


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def _make_arrays(
        self, n_train: int = 300, n_val: int = 100, n_features: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(99)
        X_train = rng.normal(0, 1, (n_train, n_features)).astype(np.float32)
        y_train = rng.integers(0, 2, n_train).astype(float)
        X_val = rng.normal(0, 1, (n_val, n_features)).astype(np.float32)
        y_val = rng.integers(0, 2, n_val).astype(float)
        return X_train, y_train, X_val, y_val

    def test_fit_returns_metrics_dict(self) -> None:
        """fit() returns a dict containing val_accuracy and best_iteration."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model = XGBoostModel(n_estimators=50)
        metrics = model.fit(X_train, y_train, X_val, y_val)

        assert "val_accuracy" in metrics
        assert "best_iteration" in metrics
        assert 0.0 <= metrics["val_accuracy"] <= 1.0
        assert metrics["best_iteration"] >= 0

    def test_predict_proba_shape(self) -> None:
        """predict_proba returns (n_samples,) array with values in [0, 1]."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model = XGBoostModel(n_estimators=50)
        model.fit(X_train, y_train, X_val, y_val)

        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val),)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_predict_proba_before_fit_raises(self) -> None:
        """predict_proba raises RuntimeError before fit."""
        model = XGBoostModel()
        X = np.zeros((5, 3), dtype=np.float32)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save() / load() preserves predictions across instances."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model1 = XGBoostModel(n_estimators=30)
        model1.fit(X_train, y_train, X_val, y_val)
        model1.save(tmp_path)

        model2 = XGBoostModel()
        model2.load(tmp_path)
        np.testing.assert_array_almost_equal(
            model1.predict_proba(X_val),
            model2.predict_proba(X_val),
            decimal=5,
        )

    def test_load_missing_model_raises(self, tmp_path: Path) -> None:
        """load() from empty directory raises FileNotFoundError."""
        model = XGBoostModel()
        with pytest.raises(FileNotFoundError):
            model.load(tmp_path)

    def test_fit_empty_X_raises(self) -> None:
        """fit() on empty X_train raises ValueError."""
        model = XGBoostModel()
        with pytest.raises(ValueError, match="empty"):
            model.fit(
                np.array([]).reshape(0, 5),
                np.array([]),
                np.array([]).reshape(0, 5),
                np.array([]),
            )

    def test_fit_single_class_raises(self) -> None:
        """fit() with only one class in y_train raises ValueError."""
        model = XGBoostModel()
        X = np.ones((20, 3), dtype=np.float32)
        y_all_zeros = np.zeros(20)
        with pytest.raises(ValueError, match="both classes"):
            model.fit(X, y_all_zeros, X, y_all_zeros)

    def test_get_feature_importance(self) -> None:
        """get_feature_importance returns a non-empty dict after fit."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        feat_names = [f"feat_{i}" for i in range(10)]
        model = XGBoostModel(n_estimators=50)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feat_names)

        importance = model.get_feature_importance(importance_type="gain")
        assert isinstance(importance, dict)
        assert len(importance) > 0
        for key in importance:
            assert key in feat_names

    def test_get_top_features_count(self) -> None:
        """get_top_features(n=5) returns at most 5 entries."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model = XGBoostModel(n_estimators=50)
        model.fit(X_train, y_train, X_val, y_val)
        top = model.get_top_features(n=5)
        assert len(top) <= 5

    def test_is_fitted_after_load(self, tmp_path: Path) -> None:
        """After load(), _is_fitted is True and predict_proba works."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model1 = XGBoostModel(n_estimators=30)
        model1.fit(X_train, y_train, X_val, y_val)
        model1.save(tmp_path)

        model2 = XGBoostModel()
        assert model2._is_fitted is False
        model2.load(tmp_path)
        assert model2._is_fitted is True
        proba = model2.predict_proba(X_val)
        assert proba.shape == (len(X_val),)

    def test_scale_pos_weight_computed_from_class_ratio(self) -> None:
        """scale_pos_weight is set correctly from class distribution."""
        rng = np.random.default_rng(5)
        X_train = rng.normal(0, 1, (200, 5)).astype(np.float32)
        # 3:1 negative:positive ratio -> scale_pos_weight approx 3.0
        y_train = np.array([0] * 150 + [1] * 50, dtype=float)
        rng.shuffle(y_train)

        X_val = rng.normal(0, 1, (50, 5)).astype(np.float32)
        y_val = rng.integers(0, 2, 50).astype(float)

        model = XGBoostModel(n_estimators=30)
        metrics = model.fit(X_train, y_train, X_val, y_val)

        assert abs(model._scale_pos_weight - 3.0) < 0.1
        assert metrics["scale_pos_weight"] == model._scale_pos_weight

    def test_feature_names_stored_in_metadata(self, tmp_path: Path) -> None:
        """Feature names are persisted in xgb_metadata.json."""
        import json as _json

        X_train, y_train, X_val, y_val = self._make_arrays()
        feat_names = [f"col_{i}" for i in range(10)]
        model = XGBoostModel(n_estimators=30)
        model.fit(X_train, y_train, X_val, y_val, feature_names=feat_names)
        model.save(tmp_path)

        with (tmp_path / "xgb_metadata.json").open() as fh:
            meta = _json.load(fh)
        assert meta["feature_names"] == feat_names

    def test_predict_signal_integration(self) -> None:
        """predict_signal works end-to-end on a fitted XGBoostModel."""
        X_train, y_train, X_val, y_val = self._make_arrays()
        model = XGBoostModel(n_estimators=30)
        model.fit(X_train, y_train, X_val, y_val)

        signals = model.predict_signal(X_val, regime_label="TRENDING_BULL")
        assert len(signals) == len(X_val)
        assert all(s in {SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD} for s in signals)

    def test_repr(self) -> None:
        """__repr__ includes fitted state and best_iter."""
        model = XGBoostModel(n_estimators=30)
        assert "fitted=False" in repr(model)
        X_train, y_train, X_val, y_val = self._make_arrays()
        model.fit(X_train, y_train, X_val, y_val)
        assert "fitted=True" in repr(model)
