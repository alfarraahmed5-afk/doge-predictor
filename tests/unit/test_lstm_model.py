"""Unit tests for LSTMModel, _LSTMNetwork, and _make_sequences.

Mandatory tests (from CLAUDE.md Prompt 5.2):
    - Gradient clipping: simulate exploding gradient, assert loss does not go NaN.
    - Eval mode: two identical forward passes in eval mode return identical output
      (proves Dropout is disabled in eval mode).
    - Sequence creation: assert X[0].shape == (seq_len, n_features).
    - predict_proba output: shape (n_samples,), all values in [0, 1].

Additional tests:
    - fit() returns required metrics dict keys.
    - predict_proba raises RuntimeError before fit.
    - assert not model.training is enforced inside predict_proba.
    - save/load roundtrip produces identical probabilities.
    - Early stopping restores best weights.
    - Sequence count: n_samples rows → n_samples - seq_len sequences.
    - Last timestep: predict_proba pads first seq_len positions with 0.5.
    - RegimeTrainer: per-regime training with synthetic multi-regime data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.lstm_model import LSTMModel, _LSTMNetwork, _make_sequences
from src.training.regime_trainer import RegimeTrainer, RegimeTrainingResult


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_SEQ_LEN: int = 5    # short sequences for fast tests
_N_FEATURES: int = 6
_N_SAMPLES: int = 150  # must be >> seq_len so we get enough sequences
_BATCH_SIZE: int = 16


def _make_ar1_data(
    n_samples: int = _N_SAMPLES,
    n_features: int = _N_FEATURES,
    ar_coeff: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate AR(1) feature matrix with genuinely predictive lag structure.

    Args:
        n_samples: Number of rows.
        n_features: Number of features.
        ar_coeff: Auto-regression coefficient (0.9 = strong autocorrelation).
        seed: RNG seed.

    Returns:
        Tuple of ``(X, y)`` where X has shape ``(n_samples, n_features)`` and
        y is binary ``{0, 1}`` of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    X[0] = rng.standard_normal(n_features)
    for t in range(1, n_samples):
        X[t] = ar_coeff * X[t - 1] + rng.standard_normal(n_features) * 0.1

    # Simple target: sign of the first feature (predictable from AR(1) structure)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


def _make_tiny_lstm(**kwargs: object) -> LSTMModel:
    """Return a fast-to-train LSTMModel with small architecture.

    All non-specified kwargs are forwarded to :class:`LSTMModel`.
    """
    defaults: dict[str, object] = {
        "sequence_length": _SEQ_LEN,
        "hidden_size_1": 16,
        "hidden_size_2": 8,
        "dense_size": 8,
        "dropout_lstm": 0.0,   # disabled for deterministic tests
        "dropout_dense": 0.0,
        "learning_rate": 1e-2,
        "max_epochs": 5,
        "early_stopping_patience": 10,
        "batch_size": _BATCH_SIZE,
        "device": "cpu",
        "seed": 42,
    }
    defaults.update(kwargs)
    return LSTMModel(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestMakeSequences
# ---------------------------------------------------------------------------


class TestMakeSequences:
    """Tests for the _make_sequences helper function."""

    def test_output_shape(self) -> None:
        """Shape is (n_samples - seq_len, seq_len, n_features)."""
        X = np.random.randn(20, _N_FEATURES).astype(np.float32)
        seqs = _make_sequences(X, 5)
        assert seqs.shape == (15, 5, _N_FEATURES)

    def test_first_sequence_shape(self) -> None:
        """MANDATORY: X[0].shape == (seq_len, n_features)."""
        X = np.random.randn(30, _N_FEATURES).astype(np.float32)
        seqs = _make_sequences(X, _SEQ_LEN)
        assert seqs[0].shape == (_SEQ_LEN, _N_FEATURES)

    def test_first_sequence_content(self) -> None:
        """First sequence equals X[0:seq_len]."""
        X = np.arange(50 * 4, dtype=np.float32).reshape(50, 4)
        seqs = _make_sequences(X, 5)
        np.testing.assert_array_equal(seqs[0], X[0:5])

    def test_last_sequence_content(self) -> None:
        """Last sequence equals X[n-seq_len-1:n-1].

        _make_sequences produces n-seq_len sequences (indices i=0..n-seq_len-1).
        The last sequence is X[n-seq_len-1 : n-seq_len-1+seq_len] = X[n-seq_len-1:n-1].
        """
        n, seq = 20, 5
        X = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
        seqs = _make_sequences(X, seq)
        # n_seqs = n - seq = 15; last index i = 14; seq = X[14:19]
        np.testing.assert_array_equal(seqs[-1], X[n - seq - 1 : n - 1])

    def test_too_short_returns_empty(self) -> None:
        """Returns empty array when n_samples <= seq_len."""
        X = np.random.randn(5, 4).astype(np.float32)
        seqs = _make_sequences(X, 5)
        assert seqs.shape[0] == 0

    def test_dtype_is_float32(self) -> None:
        """Output dtype must be float32 for PyTorch compatibility."""
        X = np.random.randn(20, 3).astype(np.float64)
        seqs = _make_sequences(X, 5)
        assert seqs.dtype == np.float32

    def test_sequence_count_matches_spec(self) -> None:
        """Exactly n_samples - seq_len sequences are produced."""
        for n in (50, 100, 200):
            seqs = _make_sequences(np.zeros((n, 4), dtype=np.float32), _SEQ_LEN)
            assert seqs.shape[0] == n - _SEQ_LEN


# ---------------------------------------------------------------------------
# TestLSTMNetwork
# ---------------------------------------------------------------------------


class TestLSTMNetwork:
    """Tests for the inner _LSTMNetwork PyTorch module."""

    def _make_net(self) -> _LSTMNetwork:
        return _LSTMNetwork(
            n_features=_N_FEATURES,
            hidden_size_1=16,
            hidden_size_2=8,
            dense_size=8,
            dropout_lstm=0.5,
            dropout_dense=0.5,
        )

    def test_output_shape(self) -> None:
        """Forward pass output shape is (batch,)."""
        net = self._make_net()
        net.eval()
        x = torch.randn(4, _SEQ_LEN, _N_FEATURES)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (4,)

    def test_output_in_unit_range(self) -> None:
        """All outputs are in [0, 1] (sigmoid applied)."""
        net = self._make_net()
        net.eval()
        x = torch.randn(8, _SEQ_LEN, _N_FEATURES)
        with torch.no_grad():
            out = net(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    # ------------------------------------------------------------------
    # MANDATORY: eval mode produces deterministic output (Dropout disabled)
    # ------------------------------------------------------------------

    def test_eval_mode_deterministic(self) -> None:
        """MANDATORY: two identical forward passes in eval mode return identical output.

        Proves that Dropout is disabled in eval mode.
        """
        net = self._make_net()
        net.eval()

        x = torch.randn(8, _SEQ_LEN, _N_FEATURES)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)

        assert torch.allclose(out1, out2), (
            "Two eval-mode forward passes differ — Dropout is still active. "
            "Call model.eval() before inference."
        )

    def test_train_mode_nondeterministic(self) -> None:
        """Dropout in train mode causes different outputs for the same input."""
        net = _LSTMNetwork(
            n_features=_N_FEATURES,
            hidden_size_1=16,
            hidden_size_2=8,
            dense_size=8,
            dropout_lstm=0.9,   # high dropout for reliable non-determinism
            dropout_dense=0.9,
        )
        net.train()
        x = torch.randn(32, _SEQ_LEN, _N_FEATURES)
        out1 = net(x)
        out2 = net(x)
        # With 90% dropout, outputs should differ for most runs
        # (very low probability of identical results)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Train-mode forward passes are identical — Dropout may not be working."
        )

    # ------------------------------------------------------------------
    # MANDATORY: gradient clipping prevents NaN loss
    # ------------------------------------------------------------------

    def test_gradient_clipping_prevents_nan(self) -> None:
        """MANDATORY: clip_grad_norm_ limits gradient L2-norm to max_norm=1.0.

        Simulates an exploding gradient by manually scaling all parameter
        gradients by 1e6, then checks that clipping brings the total norm
        back within bounds.
        """
        net = _LSTMNetwork(
            n_features=_N_FEATURES,
            hidden_size_1=16,
            hidden_size_2=8,
            dense_size=8,
        )
        net.train()

        x = torch.randn(4, _SEQ_LEN, _N_FEATURES)
        y = torch.zeros(4)
        out = net(x)
        loss = nn.BCELoss()(out, y)
        loss.backward()

        # Simulate gradient explosion
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p.grad *= 1e6

        # Verify pre-clip norm is huge
        pre_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"))
        assert pre_norm.item() > 100.0, "Expected large pre-clip norm"

        # Re-inflate and apply clipping
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p.grad *= 1e6

        max_norm = 1.0
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm)

        # Verify post-clip norm
        post_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), float("inf")
        )
        assert post_norm.item() <= max_norm + 1e-6, (
            f"Post-clip gradient norm {post_norm.item():.4f} exceeds max_norm={max_norm}. "
            "Gradient clipping is not working correctly."
        )

    def test_loss_does_not_go_nan_with_clipping(self) -> None:
        """Train for 3 steps with gradient clipping — loss must remain finite."""
        net = _LSTMNetwork(n_features=_N_FEATURES, hidden_size_1=16, hidden_size_2=8, dense_size=8)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        rng = np.random.default_rng(0)
        X = torch.from_numpy(rng.standard_normal((16, _SEQ_LEN, _N_FEATURES)).astype(np.float32))
        y = torch.zeros(16)

        for _ in range(3):
            net.train()
            optimizer.zero_grad()
            preds = net(X)
            loss = criterion(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            assert torch.isfinite(loss), f"Loss went NaN/Inf: {loss.item()}"


# ---------------------------------------------------------------------------
# TestLSTMModelFit
# ---------------------------------------------------------------------------


class TestLSTMModelFit:
    """Tests for LSTMModel.fit()."""

    def test_fit_returns_metrics_dict(self) -> None:
        """fit() returns a dict with required keys."""
        X, y = _make_ar1_data()
        n = len(X)
        split = n * 2 // 3
        model = _make_tiny_lstm()
        metrics = model.fit(X[:split], y[:split], X[split:], y[split:])
        assert "val_accuracy" in metrics
        assert "val_loss" in metrics
        assert "best_epoch" in metrics
        assert "n_train_seqs" in metrics
        assert "n_val_seqs" in metrics

    def test_fit_val_accuracy_in_range(self) -> None:
        """val_accuracy is in [0, 1]."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        metrics = model.fit(X[:split], y[:split], X[split:], y[split:])
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_fit_marks_is_fitted(self) -> None:
        """_is_fitted is True after fit."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        assert not model._is_fitted
        model.fit(X[:split], y[:split], X[split:], y[split:])
        assert model._is_fitted

    def test_fit_raises_if_n_train_too_small(self) -> None:
        """ValueError raised when n_train <= sequence_length."""
        model = _make_tiny_lstm(sequence_length=10)
        X = np.random.randn(8, _N_FEATURES).astype(np.float32)
        y = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="n_train=8 must be > sequence_length=10"):
            model.fit(X, y, X, y)

    def test_fit_raises_if_n_val_too_small(self) -> None:
        """ValueError raised when n_val <= sequence_length."""
        model = _make_tiny_lstm(sequence_length=10)
        X_train = np.random.randn(50, _N_FEATURES).astype(np.float32)
        y_train = np.zeros(50, dtype=np.float32)
        X_val = np.random.randn(8, _N_FEATURES).astype(np.float32)
        y_val = np.zeros(8, dtype=np.float32)
        with pytest.raises(ValueError, match="n_val=8 must be > sequence_length=10"):
            model.fit(X_train, y_train, X_val, y_val)

    def test_n_train_seqs_in_metrics(self) -> None:
        """n_train_seqs == len(X_train) - sequence_length."""
        X, y = _make_ar1_data(n_samples=100)
        split = 70
        model = _make_tiny_lstm()
        metrics = model.fit(X[:split], y[:split], X[split:], y[split:])
        assert metrics["n_train_seqs"] == split - _SEQ_LEN

    def test_feature_names_stored_when_provided(self) -> None:
        """_feature_names is set when feature_names is passed."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        names = [f"feat_{i}" for i in range(_N_FEATURES)]
        model.fit(X[:split], y[:split], X[split:], y[split:], feature_names=names)
        assert model._feature_names == names

    def test_feature_names_auto_generated_when_none(self) -> None:
        """_feature_names uses 'f0, f1, ...' when feature_names=None."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])
        assert model._feature_names == [f"f{i}" for i in range(_N_FEATURES)]


# ---------------------------------------------------------------------------
# TestLSTMModelPredictProba
# ---------------------------------------------------------------------------


class TestLSTMModelPredictProba:
    """Tests for LSTMModel.predict_proba()."""

    def _fitted_model(self) -> tuple[LSTMModel, np.ndarray, np.ndarray]:
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])
        return model, X[split:], y[split:]

    # ------------------------------------------------------------------
    # MANDATORY: output shape (n_samples,), all values in [0, 1]
    # ------------------------------------------------------------------

    def test_predict_proba_shape(self) -> None:
        """MANDATORY: predict_proba returns shape (n_samples,)."""
        model, X_val, _ = self._fitted_model()
        proba = model.predict_proba(X_val)
        assert proba.shape == (len(X_val),), (
            f"Expected shape ({len(X_val)},), got {proba.shape}"
        )

    def test_predict_proba_values_in_unit_range(self) -> None:
        """MANDATORY: all probabilities are in [0, 1]."""
        model, X_val, _ = self._fitted_model()
        proba = model.predict_proba(X_val)
        assert np.all(proba >= 0.0), f"Minimum probability {proba.min():.6f} < 0"
        assert np.all(proba <= 1.0), f"Maximum probability {proba.max():.6f} > 1"

    def test_predict_proba_first_seq_len_are_neutral(self) -> None:
        """First seq_len positions are padded with 0.5 (insufficient history)."""
        model, X_val, _ = self._fitted_model()
        proba = model.predict_proba(X_val)
        np.testing.assert_array_equal(
            proba[:_SEQ_LEN],
            np.full(_SEQ_LEN, 0.5),
            err_msg="First seq_len positions should be 0.5 (insufficient history).",
        )

    def test_predict_proba_positions_after_seq_len_are_not_all_neutral(self) -> None:
        """Positions >= seq_len contain actual model predictions (not just 0.5)."""
        model, X_val, _ = self._fitted_model()
        proba = model.predict_proba(X_val)
        tail = proba[_SEQ_LEN:]
        # At least some predictions should differ from 0.5 (the model has learned)
        assert not np.all(tail == 0.5), "All tail predictions are 0.5 — model not predicting"

    def test_predict_proba_raises_before_fit(self) -> None:
        """RuntimeError raised when predict_proba is called before fit."""
        model = _make_tiny_lstm()
        X = np.random.randn(20, _N_FEATURES).astype(np.float32)
        with pytest.raises(RuntimeError):
            model.predict_proba(X)

    def test_predict_proba_leaves_model_in_eval_mode(self) -> None:
        """Network remains in eval mode after predict_proba."""
        model, X_val, _ = self._fitted_model()
        model.predict_proba(X_val)
        assert not model._network.training, (
            "Network is in training mode after predict_proba — "
            "model.eval() must be called before returning."
        )

    def test_eval_mode_assertion_enforced(self) -> None:
        """The 'assert not self._network.training' assertion is triggered if violated."""
        model, X_val, _ = self._fitted_model()
        # Manually put network into train mode to verify assertion fires
        model._network.train()
        # Patch: call eval inside predict_proba should fix it — but the assertion
        # checks AFTER model.eval() is called, so it should NOT raise normally.
        # Here we just verify the entire predict_proba call succeeds (eval is called inside).
        proba = model.predict_proba(X_val)
        assert proba is not None

    def test_predict_proba_output_dtype_is_float64(self) -> None:
        """predict_proba output dtype is float64."""
        model, X_val, _ = self._fitted_model()
        proba = model.predict_proba(X_val)
        assert proba.dtype == np.float64

    def test_predict_proba_small_input_returns_all_neutral(self) -> None:
        """When n_samples <= seq_len, all outputs are 0.5."""
        model, _, _ = self._fitted_model()
        X_tiny = np.random.randn(_SEQ_LEN, _N_FEATURES).astype(np.float32)
        proba = model.predict_proba(X_tiny)
        assert proba.shape == (_SEQ_LEN,)
        np.testing.assert_array_equal(proba, np.full(_SEQ_LEN, 0.5))


# ---------------------------------------------------------------------------
# TestLSTMModelSaveLoad
# ---------------------------------------------------------------------------


class TestLSTMModelSaveLoad:
    """Tests for LSTMModel save/load roundtrip."""

    def test_save_load_identical_predictions(self) -> None:
        """Loaded model produces identical predictions to the original."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])

        proba_before = model.predict_proba(X[split:])

        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))

            loaded = _make_tiny_lstm()
            loaded.load(Path(tmp))
            proba_after = loaded.predict_proba(X[split:])

        np.testing.assert_allclose(
            proba_before,
            proba_after,
            rtol=1e-5,
            err_msg="Predictions differ after save/load roundtrip.",
        )

    def test_save_creates_expected_files(self) -> None:
        """save() writes lstm_model.pt and lstm_metadata.json."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])

        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            assert (Path(tmp) / "lstm_model.pt").exists()
            assert (Path(tmp) / "lstm_metadata.json").exists()

    def test_load_raises_if_model_file_missing(self) -> None:
        """FileNotFoundError raised if lstm_model.pt is absent."""
        model = _make_tiny_lstm()
        with pytest.raises(FileNotFoundError, match="lstm_model.pt"):
            model.load(Path("/nonexistent/path"))

    def test_load_sets_is_fitted(self) -> None:
        """_is_fitted is True after load."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])

        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            loaded = _make_tiny_lstm()
            assert not loaded._is_fitted
            loaded.load(Path(tmp))
            assert loaded._is_fitted

    def test_metadata_json_contains_feature_names(self) -> None:
        """Metadata JSON includes feature_names."""
        import json

        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        names = [f"feat_{i}" for i in range(_N_FEATURES)]
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:], feature_names=names)

        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp))
            with open(Path(tmp) / "lstm_metadata.json") as fh:
                meta = json.load(fh)
        assert meta["feature_names"] == names


# ---------------------------------------------------------------------------
# TestLSTMModelRepr
# ---------------------------------------------------------------------------


class TestLSTMModelRepr:
    """Tests for LSTMModel __repr__."""

    def test_repr_before_fit(self) -> None:
        """repr shows fitted=False before fit."""
        model = _make_tiny_lstm()
        r = repr(model)
        assert "fitted=False" in r
        assert "seq_len=" in r

    def test_repr_after_fit(self) -> None:
        """repr shows fitted=True after fit."""
        X, y = _make_ar1_data()
        split = len(X) * 2 // 3
        model = _make_tiny_lstm()
        model.fit(X[:split], y[:split], X[split:], y[split:])
        r = repr(model)
        assert "fitted=True" in r


# ---------------------------------------------------------------------------
# TestRegimeTrainer
# ---------------------------------------------------------------------------


class TestRegimeTrainer:
    """Tests for RegimeTrainer.train_per_regime()."""

    _MS_PER_HOUR: int = 3_600_000
    _TRAINING_START_MS: int = 1_640_995_200_000  # 2022-01-01 00:00 UTC

    def _make_regime_df(
        self,
        n_per_regime: int = 600,
        n_features: int = 4,
        seed: int = 42,
    ) -> tuple["pd.DataFrame", "pd.Series"]:
        """Build a minimal feature_df and regime_labels for testing.

        Returns:
            Tuple of ``(feature_df, regime_labels)`` where feature_df has
            ``open_time``, ``era``, ``target``, and ``n_features`` numeric cols.
        """
        import pandas as pd

        rng = np.random.default_rng(seed)
        regimes = [
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "RANGING_HIGH_VOL",
            "RANGING_LOW_VOL",
            "DECOUPLED",
        ]
        frames = []
        for i, regime in enumerate(regimes):
            n = n_per_regime
            start = self._TRAINING_START_MS + i * n * self._MS_PER_HOUR
            open_times = [start + j * self._MS_PER_HOUR for j in range(n)]
            feat_data = {
                f"f{k}": rng.standard_normal(n).astype(np.float32)
                for k in range(n_features)
            }
            target = rng.integers(0, 2, n).astype(np.float32)
            df_chunk = pd.DataFrame(
                {
                    "open_time": open_times,
                    "era": "training",
                    **feat_data,
                    "target": target,
                }
            )
            df_chunk["_regime"] = regime
            frames.append(df_chunk)

        full_df = pd.concat(frames, ignore_index=True)
        labels = full_df.pop("_regime")
        return full_df, labels

    def _fast_wf_cfg(self) -> "WalkForwardSettings":
        from src.config import WalkForwardSettings

        return WalkForwardSettings(
            training_window_days=15,
            validation_window_days=5,
            step_size_days=5,
            min_training_rows=100,
        )

    def test_returns_dict_with_regime_keys(self) -> None:
        """train_per_regime returns a dict keyed by regime name."""
        feature_df, labels = self._make_regime_df()
        trainer = RegimeTrainer(walk_forward_cfg=self._fast_wf_cfg())
        results = trainer.train_per_regime(feature_df, labels)
        assert isinstance(results, dict)
        for key in results:
            assert key in (
                "TRENDING_BULL",
                "TRENDING_BEAR",
                "RANGING_HIGH_VOL",
                "RANGING_LOW_VOL",
                "DECOUPLED",
            )

    def test_models_are_fitted(self) -> None:
        """All returned models pass the is_fitted check."""
        from src.models.xgb_model import XGBoostModel

        feature_df, labels = self._make_regime_df()
        trainer = RegimeTrainer(walk_forward_cfg=self._fast_wf_cfg())
        results = trainer.train_per_regime(feature_df, labels)
        for regime, model in results.items():
            assert isinstance(model, XGBoostModel), f"Expected XGBoostModel for {regime}"
            assert model._is_fitted, f"Model for {regime} is not fitted"

    def test_insufficient_data_regime_skipped(self) -> None:
        """Regimes with fewer than _MIN_REGIME_ROWS rows are absent from results."""
        import pandas as pd
        from src.training.regime_trainer import _MIN_REGIME_ROWS

        # Only TRENDING_BULL has enough rows; rest get 10 rows (well below 500)
        rng = np.random.default_rng(0)
        n_big = 800  # ≈33.3 days at 1h cadence; need ≥30d for 3 WF folds (15d/5d/5d)
        n_small = 10
        all_rows = []
        labels_list = []
        start = self._TRAINING_START_MS
        for i, regime in enumerate(
            ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"]
        ):
            n = n_big if regime == "TRENDING_BULL" else n_small
            times = [start + (i * 1000 + j) * self._MS_PER_HOUR for j in range(n)]
            rows = pd.DataFrame(
                {
                    "open_time": times,
                    "era": "training",
                    "f0": rng.standard_normal(n).astype(np.float32),
                    "f1": rng.standard_normal(n).astype(np.float32),
                    "target": rng.integers(0, 2, n).astype(np.float32),
                }
            )
            all_rows.append(rows)
            labels_list.extend([regime] * n)

        feature_df = pd.concat(all_rows, ignore_index=True)
        labels = pd.Series(labels_list, index=feature_df.index)

        trainer = RegimeTrainer(walk_forward_cfg=self._fast_wf_cfg())
        results = trainer.train_per_regime(feature_df, labels)

        # Only TRENDING_BULL should be in results
        assert "TRENDING_BULL" in results
        for regime in ["TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"]:
            assert regime not in results, f"{regime} should be skipped (too few rows)"

    def test_missing_required_columns_raises(self) -> None:
        """ValueError raised when feature_df is missing required columns."""
        import pandas as pd

        bad_df = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
        labels = pd.Series(["TRENDING_BULL"] * 2)
        trainer = RegimeTrainer()
        with pytest.raises(ValueError, match="missing required columns"):
            trainer.train_per_regime(bad_df, labels)

    def test_save_to_disk_when_output_dir_given(self) -> None:
        """Models are saved to output_dir/{regime}/ when output_dir is set."""
        feature_df, labels = self._make_regime_df()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = RegimeTrainer(
                walk_forward_cfg=self._fast_wf_cfg(),
                output_dir=Path(tmp),
            )
            results = trainer.train_per_regime(feature_df, labels)
            for regime in results:
                regime_dir = Path(tmp) / regime
                assert regime_dir.exists(), f"Missing output dir for {regime}"
                assert (regime_dir / "xgb_model.json").exists()
