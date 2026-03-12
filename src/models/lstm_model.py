"""LSTM binary classification model for DOGE price direction prediction.

Architecture (CLAUDE.md §8):
    Input:  (batch, sequence_length=60, n_features)
    LSTM(128, return_sequences=True)  + inter-layer Dropout(0.2)
    LSTM(64,  return_sequences=False) + post-layer  Dropout(0.2)
    Linear(32) + BatchNorm1d(32) + ReLU + Dropout(0.3)
    Linear(1)  + Sigmoid → P(up)

Training requirements (all mandatory):
    - Gradient clipping: ``torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)``
      Without this, loss goes NaN on volatile DOGE data.
    - ReduceLROnPlateau: patience=5, factor=0.5, min_lr=1e-6
    - Early stopping: patience=15 on val_loss, restore best weights
    - ``model.train()`` before every training epoch
    - ``model.eval()`` before every validation pass and before any inference
    - ``assert not self._network.training`` enforced inside :meth:`predict_proba`

Sequence construction:
    For an input of ``n_samples`` rows, the k-th sequence is
    ``X[k : k + seq_len]`` and its target label is ``y[k + seq_len]``.
    This yields ``n_samples − seq_len`` labelled sequences.

    During :meth:`predict_proba` the output array has shape ``(n_samples,)``.
    Positions ``[0, seq_len)`` are padded with ``0.5`` (insufficient history).
    Position ``j ≥ seq_len`` receives the model output for sequence
    ``X[j − seq_len : j]``.

Serialisation:
    ``lstm_model.pt``      — ``torch.save`` of {state_dict, arch_params}
    ``lstm_metadata.json`` — feature names, hyperparams, best_epoch, n_features

Lookahead audit:
    This model class has no feature-computation logic.  It receives a
    pre-built, lookahead-free feature matrix from the feature pipeline.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

from src.config import (
    LSTMSettings,
    RegimeConfig,
    doge_settings as _doge_settings,
    regime_config as _default_regime_config,
    settings as _global_settings,
)
from src.models.base_model import AbstractBaseModel

# ---------------------------------------------------------------------------
# File name constants
# ---------------------------------------------------------------------------

_MODEL_FILENAME: str = "lstm_model.pt"
_METADATA_FILENAME: str = "lstm_metadata.json"

# ---------------------------------------------------------------------------
# Default hyperparameters (sourced from config — never hardcoded in business logic)
# ---------------------------------------------------------------------------

_CFG: LSTMSettings = _doge_settings.lstm  # read once at module level

_DEFAULT_SEQ_LEN: int = _CFG.sequence_length
_DEFAULT_HIDDEN1: int = _CFG.hidden_size_1
_DEFAULT_HIDDEN2: int = _CFG.hidden_size_2
_DEFAULT_DENSE: int = _CFG.dense_size
_DEFAULT_DROPOUT_LSTM: float = _CFG.dropout_lstm
_DEFAULT_DROPOUT_DENSE: float = _CFG.dropout_dense
_DEFAULT_LR: float = _CFG.learning_rate
_DEFAULT_MAX_EPOCHS: int = _CFG.max_epochs
_DEFAULT_ES_PATIENCE: int = _CFG.early_stopping_patience
_DEFAULT_LR_PATIENCE: int = _CFG.lr_scheduler_patience
_DEFAULT_LR_FACTOR: float = _CFG.lr_scheduler_factor
_DEFAULT_LR_MIN: float = _CFG.lr_scheduler_min_lr
_DEFAULT_GRAD_CLIP: float = _CFG.gradient_clip_norm
_DEFAULT_BATCH_SIZE: int = _CFG.batch_size


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Create forward-window rolling sequences from a 2-D feature array.

    For each index ``i`` in ``[0, n_samples − seq_len)``, the sequence is
    ``X[i : i + seq_len]``.  The corresponding target index is ``i + seq_len``
    (one step beyond the sequence's last feature row).

    Args:
        X: Feature array, shape ``(n_samples, n_features)``.
        seq_len: Number of consecutive rows per sequence.

    Returns:
        Float32 array of shape ``(n_samples − seq_len, seq_len, n_features)``.
        Returns an empty array of that shape if ``n_samples <= seq_len``.
    """
    n = X.shape[0]
    n_features = X.shape[1] if X.ndim > 1 else 1
    n_seqs = n - seq_len
    if n_seqs <= 0:
        return np.empty((0, seq_len, n_features), dtype=np.float32)
    seqs = np.stack([X[i : i + seq_len] for i in range(n_seqs)])
    return seqs.astype(np.float32)


def _resolve_device(device: str | None) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    Args:
        device: ``'cpu'``, ``'cuda'``, ``'auto'``, or ``None``.
            ``'auto'`` / ``None`` selects CUDA if available, else CPU.

    Returns:
        Resolved :class:`torch.device`.
    """
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Inner PyTorch Module
# ---------------------------------------------------------------------------


class _LSTMNetwork(nn.Module):
    """Inner PyTorch module implementing the LSTM architecture from CLAUDE.md §8.

    Args:
        n_features: Number of input features (width of feature matrix).
        hidden_size_1: Hidden size of the first LSTM layer.
        hidden_size_2: Hidden size of the second LSTM layer.
        dense_size: Number of units in the fully-connected layer.
        dropout_lstm: Dropout probability between LSTM layers.
        dropout_dense: Dropout probability after the dense layer.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size_1: int = _DEFAULT_HIDDEN1,
        hidden_size_2: int = _DEFAULT_HIDDEN2,
        dense_size: int = _DEFAULT_DENSE,
        dropout_lstm: float = _DEFAULT_DROPOUT_LSTM,
        dropout_dense: float = _DEFAULT_DROPOUT_DENSE,
    ) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden_size_1, batch_first=True)
        self.drop_lstm1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.drop_lstm2 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.bn = nn.BatchNorm1d(dense_size)
        self.relu = nn.ReLU()
        self.drop_dense = nn.Dropout(dropout_dense)
        self.fc2 = nn.Linear(dense_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape ``(batch, seq_len, n_features)``.

        Returns:
            P(class=1) per sample, shape ``(batch,)``.
        """
        # LSTM layer 1 — return all timesteps
        out, _ = self.lstm1(x)          # (batch, seq_len, hidden1)
        out = self.drop_lstm1(out)
        # LSTM layer 2 — take only the last timestep's hidden state
        out, _ = self.lstm2(out)        # (batch, seq_len, hidden2)
        out = out[:, -1, :]             # (batch, hidden2)
        out = self.drop_lstm2(out)
        # Fully-connected head
        out = self.fc1(out)             # (batch, dense)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop_dense(out)
        out = self.fc2(out)             # (batch, 1)
        return self.sigmoid(out).squeeze(1)  # (batch,)


# ---------------------------------------------------------------------------
# LSTMModel
# ---------------------------------------------------------------------------


class LSTMModel(AbstractBaseModel):
    """LSTM binary classifier implementing the :class:`AbstractBaseModel` API.

    Predicts P(next candle closes higher) for every sample.  All
    hyperparameters default to values from ``config/doge_settings.yaml``
    (``lstm:`` section) so that no numeric constants are hardcoded.

    Args:
        sequence_length: Number of consecutive candles per input window.
        hidden_size_1: Hidden size of the first LSTM layer.
        hidden_size_2: Hidden size of the second LSTM layer.
        dense_size: Units in the dense hidden layer.
        dropout_lstm: Dropout probability applied between LSTM layers.
        dropout_dense: Dropout probability applied after the dense layer.
        learning_rate: Adam initial learning rate.
        max_epochs: Upper bound on training epochs (early stopping may halt sooner).
        early_stopping_patience: Epochs without val_loss improvement before stopping.
        lr_scheduler_patience: Epochs without improvement before LR is reduced.
        lr_scheduler_factor: Multiplicative factor for LR reduction.
        lr_scheduler_min_lr: Floor for the learning rate scheduler.
        gradient_clip_norm: Max L2-norm for gradient clipping — **mandatory**.
        batch_size: Mini-batch size; ``shuffle=False`` is enforced (time series).
        seed: Random seed for reproducibility.
        device: ``'cpu'``, ``'cuda'``, or ``'auto'`` (default).
        regime_cfg: Regime config for :meth:`predict_signal` thresholds.

    Attributes:
        _network: The fitted :class:`_LSTMNetwork` (``None`` before fit).
        _n_features: Feature count inferred from the first :meth:`fit` call.
        _best_epoch: Epoch index with the lowest validation loss.
        _feature_names: Column names passed to :meth:`fit`.
    """

    def __init__(
        self,
        sequence_length: int = _DEFAULT_SEQ_LEN,
        hidden_size_1: int = _DEFAULT_HIDDEN1,
        hidden_size_2: int = _DEFAULT_HIDDEN2,
        dense_size: int = _DEFAULT_DENSE,
        dropout_lstm: float = _DEFAULT_DROPOUT_LSTM,
        dropout_dense: float = _DEFAULT_DROPOUT_DENSE,
        learning_rate: float = _DEFAULT_LR,
        max_epochs: int = _DEFAULT_MAX_EPOCHS,
        early_stopping_patience: int = _DEFAULT_ES_PATIENCE,
        lr_scheduler_patience: int = _DEFAULT_LR_PATIENCE,
        lr_scheduler_factor: float = _DEFAULT_LR_FACTOR,
        lr_scheduler_min_lr: float = _DEFAULT_LR_MIN,
        gradient_clip_norm: float = _DEFAULT_GRAD_CLIP,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        seed: int = _global_settings.project.seed,
        device: str | None = "auto",
        regime_cfg: RegimeConfig | None = None,
    ) -> None:
        super().__init__(regime_cfg=regime_cfg)

        self._sequence_length: int = sequence_length
        self._hidden_size_1: int = hidden_size_1
        self._hidden_size_2: int = hidden_size_2
        self._dense_size: int = dense_size
        self._dropout_lstm: float = dropout_lstm
        self._dropout_dense: float = dropout_dense
        self._learning_rate: float = learning_rate
        self._max_epochs: int = max_epochs
        self._early_stopping_patience: int = early_stopping_patience
        self._lr_scheduler_patience: int = lr_scheduler_patience
        self._lr_scheduler_factor: float = lr_scheduler_factor
        self._lr_scheduler_min_lr: float = lr_scheduler_min_lr
        self._gradient_clip_norm: float = gradient_clip_norm
        self._batch_size: int = batch_size
        self._seed: int = seed
        self._device: torch.device = _resolve_device(device)

        self._network: _LSTMNetwork | None = None
        self._n_features: int = 0
        self._best_epoch: int = 0
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # AbstractBaseModel implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Train the LSTM on the supplied walk-forward fold.

        Internally reshapes the flat ``(n_samples, n_features)`` matrices into
        ``(n_samples − seq_len, seq_len, n_features)`` sequences.  Target labels
        are aligned so that ``y_seq[i] = y[i + seq_len]`` (one step beyond the
        last feature in each window).

        Args:
            X_train: Training feature matrix, shape ``(n_train, n_features)``.
            y_train: Training binary labels ``{0, 1}``, shape ``(n_train,)``.
            X_val: Validation feature matrix, shape ``(n_val, n_features)``.
            y_val: Validation binary labels, shape ``(n_val,)``.
            feature_names: Optional column name list (used by :meth:`save`).

        Returns:
            Metrics dict with keys:

                - ``"val_accuracy"`` (float)
                - ``"val_loss"`` (float)  — best validation BCE loss
                - ``"best_epoch"`` (int)
                - ``"n_train_seqs"`` (int)
                - ``"n_val_seqs"`` (int)

        Raises:
            ValueError: If either set has fewer samples than ``sequence_length + 1``.
        """
        # --- Seeds ---
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)

        n_train, n_features = X_train.shape[0], X_train.shape[1]
        n_val = X_val.shape[0]

        if n_train <= self._sequence_length:
            raise ValueError(
                f"LSTMModel.fit: n_train={n_train} must be > sequence_length={self._sequence_length}."
            )
        if n_val <= self._sequence_length:
            raise ValueError(
                f"LSTMModel.fit: n_val={n_val} must be > sequence_length={self._sequence_length}."
            )

        # Store feature names
        self._n_features = n_features
        self._feature_names = list(feature_names) if feature_names is not None else [
            f"f{i}" for i in range(n_features)
        ]

        # --- Build sequences ---
        # X_seq[i] = X_train[i : i+seq_len], y_seq[i] = y_train[i+seq_len]
        X_train_seq = _make_sequences(X_train, self._sequence_length)   # (n_t-sl, sl, nf)
        y_train_seq = y_train[self._sequence_length :].astype(np.float32)

        X_val_seq = _make_sequences(X_val, self._sequence_length)       # (n_v-sl, sl, nf)
        y_val_seq = y_val[self._sequence_length :].astype(np.float32)

        n_train_seqs = X_train_seq.shape[0]
        n_val_seqs = X_val_seq.shape[0]

        logger.info(
            "LSTMModel.fit: n_train_seqs={}, n_val_seqs={}, n_features={}",
            n_train_seqs, n_val_seqs, n_features,
        )

        # --- Build network ---
        self._network = _LSTMNetwork(
            n_features=n_features,
            hidden_size_1=self._hidden_size_1,
            hidden_size_2=self._hidden_size_2,
            dense_size=self._dense_size,
            dropout_lstm=self._dropout_lstm,
            dropout_dense=self._dropout_dense,
        ).to(self._device)

        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self._learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self._lr_scheduler_patience,
            factor=self._lr_scheduler_factor,
            min_lr=self._lr_scheduler_min_lr,
        )
        criterion = nn.BCELoss()

        # --- DataLoaders (shuffle=False — time series ordering must be preserved) ---
        train_ds = TensorDataset(
            torch.from_numpy(X_train_seq),
            torch.from_numpy(y_train_seq),
        )
        # drop_last=True prevents a single-sample batch that breaks BatchNorm1d
        drop_last = n_train_seqs > self._batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=drop_last,
        )

        val_ds = TensorDataset(
            torch.from_numpy(X_val_seq),
            torch.from_numpy(y_val_seq),
        )
        val_loader = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False)

        # --- Training loop with early stopping & best-weight restoration ---
        best_val_loss: float = float("inf")
        best_state: dict[str, Any] = {}
        patience_counter: int = 0
        best_epoch: int = 0

        for epoch in range(self._max_epochs):
            # ---- Training phase ----
            self._network.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                preds = self._network(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()

                # MANDATORY gradient clipping — prevents NaN loss on DOGE data
                nn.utils.clip_grad_norm_(
                    self._network.parameters(), max_norm=self._gradient_clip_norm
                )
                optimizer.step()

            # ---- Validation phase ----
            self._network.eval()
            val_loss_accum: float = 0.0
            val_correct: int = 0
            val_total: int = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    preds = self._network(X_batch)
                    val_loss_accum += criterion(preds, y_batch).item() * len(y_batch)
                    predicted = (preds >= 0.5).long()
                    val_correct += (predicted == y_batch.long()).sum().item()
                    val_total += len(y_batch)

            val_loss = val_loss_accum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            scheduler.step(val_loss)

            logger.debug(
                "LSTMModel epoch {:4d}/{} — val_loss={:.6f}, val_acc={:.4f}",
                epoch + 1, self._max_epochs, val_loss, val_acc,
            )

            # --- Early stopping with best-weight restoration ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self._network.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._early_stopping_patience:
                    logger.info(
                        "LSTMModel: early stopping at epoch {} (patience={})",
                        epoch + 1, self._early_stopping_patience,
                    )
                    break

        # Restore best weights
        if best_state:
            self._network.load_state_dict(best_state)

        self._best_epoch = best_epoch
        self._is_fitted = True

        # Final val accuracy with best weights
        self._network.eval()
        final_correct: int = 0
        final_total: int = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                preds = self._network(X_batch)
                predicted = (preds >= 0.5).long()
                final_correct += (predicted == y_batch.long()).sum().item()
                final_total += len(y_batch)

        final_val_acc = final_correct / max(final_total, 1)

        metrics: dict[str, Any] = {
            "val_accuracy": final_val_acc,
            "val_loss": best_val_loss,
            "best_epoch": self._best_epoch,
            "n_train_seqs": n_train_seqs,
            "n_val_seqs": n_val_seqs,
        }

        logger.info(
            "LSTMModel.fit: complete — best_epoch={}, val_accuracy={:.4f}, val_loss={:.6f}",
            self._best_epoch, final_val_acc, best_val_loss,
        )

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each sample row in *X*.

        Samples with insufficient history (index < ``sequence_length``) are
        assigned the neutral probability ``0.5``.

        The model is switched to eval mode before inference.  An assertion
        verifies eval mode is active — violation is a critical bug.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            Float64 array of probabilities in ``[0, 1]``,
            shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._assert_fitted()

        # MANDATORY: always eval before inference
        self._network.eval()  # type: ignore[union-attr]
        assert not self._network.training, (  # type: ignore[union-attr]
            "LSTMModel.predict_proba: network must be in eval mode. "
            "This is a critical bug — call model.eval() before inference."
        )

        n_samples = X.shape[0]
        out = np.full(n_samples, 0.5, dtype=np.float64)

        if n_samples <= self._sequence_length:
            logger.debug(
                "LSTMModel.predict_proba: n_samples={} <= seq_len={} — returning all 0.5",
                n_samples, self._sequence_length,
            )
            return out

        X_seq = _make_sequences(X, self._sequence_length)  # (n-sl, sl, nf)
        X_tensor = torch.from_numpy(X_seq).to(self._device)

        preds_list: list[np.ndarray] = []
        batch_size = self._batch_size

        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = X_tensor[i : i + batch_size]
                preds_list.append(
                    self._network(batch).cpu().numpy()  # type: ignore[union-attr]
                )

        preds = np.concatenate(preds_list, axis=0).astype(np.float64)

        # Positions 0 … seq_len-1 stay at 0.5 (not enough history).
        # Position j = seq_len + k  receives preds[k].
        out[self._sequence_length :] = preds
        return out

    def save(self, path: Path) -> None:
        """Serialise the model and metadata to *path/*.

        Writes:
            - ``lstm_model.pt``      — PyTorch state dict + architecture params
            - ``lstm_metadata.json`` — feature names, hyperparams, best epoch

        Args:
            path: Directory to write artefacts into.

        Raises:
            RuntimeError: If the model has not been fitted.
            OSError: If any file cannot be written.
        """
        self._assert_fitted()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Build the checkpoint dict (arch params needed to recreate network on load)
        arch_params = {
            "n_features": self._n_features,
            "hidden_size_1": self._hidden_size_1,
            "hidden_size_2": self._hidden_size_2,
            "dense_size": self._dense_size,
            "dropout_lstm": self._dropout_lstm,
            "dropout_dense": self._dropout_dense,
        }
        checkpoint = {
            "state_dict": self._network.state_dict(),  # type: ignore[union-attr]
            "arch_params": arch_params,
        }

        model_path = path / _MODEL_FILENAME
        try:
            torch.save(checkpoint, str(model_path))
            logger.info("LSTMModel.save: model → {}", model_path)
        except (OSError, RuntimeError) as exc:
            logger.error("LSTMModel.save: failed to save model — {}", exc)
            raise

        metadata: dict[str, Any] = {
            "feature_names": self._feature_names,
            "n_features": self._n_features,
            "best_epoch": self._best_epoch,
            "hyperparameters": {
                "sequence_length": self._sequence_length,
                "hidden_size_1": self._hidden_size_1,
                "hidden_size_2": self._hidden_size_2,
                "dense_size": self._dense_size,
                "dropout_lstm": self._dropout_lstm,
                "dropout_dense": self._dropout_dense,
                "learning_rate": self._learning_rate,
                "max_epochs": self._max_epochs,
                "early_stopping_patience": self._early_stopping_patience,
                "gradient_clip_norm": self._gradient_clip_norm,
                "batch_size": self._batch_size,
            },
        }

        meta_path = path / _METADATA_FILENAME
        try:
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logger.info("LSTMModel.save: metadata → {}", meta_path)
        except OSError as exc:
            logger.error("LSTMModel.save: failed to save metadata — {}", exc)
            raise

    def load(self, path: Path) -> None:
        """Deserialise a previously saved model from *path/*.

        Recreates the :class:`_LSTMNetwork` with the saved architecture
        parameters, then loads the state dict.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If either artefact file is missing.
            OSError: If any file cannot be read.
        """
        path = Path(path)

        model_path = path / _MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"LSTMModel.load: model file not found at {model_path}"
            )

        meta_path = path / _METADATA_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"LSTMModel.load: metadata file not found at {meta_path}"
            )

        try:
            checkpoint: dict[str, Any] = torch.load(
                str(model_path), map_location=self._device, weights_only=True
            )
            arch = checkpoint["arch_params"]
            self._n_features = int(arch["n_features"])
            self._network = _LSTMNetwork(
                n_features=self._n_features,
                hidden_size_1=int(arch["hidden_size_1"]),
                hidden_size_2=int(arch["hidden_size_2"]),
                dense_size=int(arch["dense_size"]),
                dropout_lstm=float(arch["dropout_lstm"]),
                dropout_dense=float(arch["dropout_dense"]),
            ).to(self._device)
            self._network.load_state_dict(checkpoint["state_dict"])
            self._network.eval()
            logger.info("LSTMModel.load: model ← {}", model_path)
        except (OSError, RuntimeError, KeyError) as exc:
            logger.error("LSTMModel.load: failed to load model — {}", exc)
            raise

        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                metadata: dict[str, Any] = json.load(fh)
            self._feature_names = metadata.get("feature_names", [])
            self._best_epoch = int(metadata.get("best_epoch", 0))
            logger.info("LSTMModel.load: metadata ← {}", meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("LSTMModel.load: failed to load metadata — {}", exc)
            raise

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Properties and repr
    # ------------------------------------------------------------------

    @property
    def network(self) -> _LSTMNetwork | None:
        """The underlying :class:`_LSTMNetwork`, or ``None`` if not fitted."""
        return self._network

    @property
    def sequence_length(self) -> int:
        """Configured sequence length."""
        return self._sequence_length

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"LSTMModel("
            f"fitted={self._is_fitted}, "
            f"seq_len={self._sequence_length}, "
            f"best_epoch={self._best_epoch})"
        )


__all__ = ["LSTMModel", "_LSTMNetwork", "_make_sequences"]
