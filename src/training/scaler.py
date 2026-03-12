"""Per-fold feature scaler that enforces RULE B: scaler isolation.

RULE B (CLAUDE.md Section 3.2):
    StandardScaler is fitted ONLY on the training fold.
    It is used (not re-fitted) to transform validation and test folds.
    A scaler fitted on the full dataset is a **data leak** — this is a
    critical bug and :class:`FoldScaler` is designed to make it impossible.

Design decisions:
    - :meth:`fit_transform` is the ONLY method that fits the scaler; it may
      be called at most once.  Calling it a second time raises ``RuntimeError``.
    - :meth:`transform` raises ``RuntimeError`` if called before
      :meth:`fit_transform`.
    - :meth:`assert_not_fitted_on_future` provides a post-hoc sanity check
      that the data used to fit the scaler contains no future timestamps.
    - Serialisation uses ``joblib`` (same as scikit-learn recommendation).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# FoldScaler
# ---------------------------------------------------------------------------


class FoldScaler:
    """Per-fold StandardScaler with strict fit-once / transform-only semantics.

    Enforces RULE B from CLAUDE.md: the scaler is fitted exclusively on the
    training portion of a single walk-forward fold.  Subsequent calls to
    :meth:`transform` apply the already-fitted scaler without any re-fitting.

    Example::

        scaler = FoldScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled   = scaler.transform(X_val)
        scaler.save(fold_dir)

        # --- In a separate process or after reload ---
        scaler2 = FoldScaler()
        scaler2.load(fold_dir)
        X_test_scaled = scaler2.transform(X_test)

    Attributes:
        _scaler: The underlying :class:`~sklearn.preprocessing.StandardScaler`.
        _is_fitted: ``True`` after :meth:`fit_transform` has been called.
    """

    _FILENAME: str = "scaler.pkl"

    def __init__(self) -> None:
        """Initialise an unfitted FoldScaler."""
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit the scaler on *X_train* and return the scaled array.

        This is the ONLY method that fits the scaler.  May be called exactly
        once per :class:`FoldScaler` instance.

        Args:
            X_train: Training feature matrix, shape ``(n_train, n_features)``.
                Must not contain NaN or Inf values.

        Returns:
            Scaled training array, same shape as *X_train*.

        Raises:
            RuntimeError: If the scaler has already been fitted (prevents
                accidental double-fitting across folds).
            ValueError: If *X_train* is empty or contains NaN / Inf.
        """
        if self._is_fitted:
            raise RuntimeError(
                "FoldScaler.fit_transform called more than once on the same "
                "instance.  Create a new FoldScaler for each walk-forward fold "
                "to prevent data leakage across folds."
            )

        if X_train.size == 0:
            raise ValueError("FoldScaler.fit_transform: X_train is empty.")

        if np.isnan(X_train).any():
            raise ValueError(
                "FoldScaler.fit_transform: X_train contains NaN values. "
                "Drop or impute NaN before scaling."
            )

        if np.isinf(X_train).any():
            raise ValueError(
                "FoldScaler.fit_transform: X_train contains Inf values. "
                "Clip or remove Inf before scaling."
            )

        scaled: np.ndarray = self._scaler.fit_transform(X_train)
        self._is_fitted = True

        logger.debug(
            "FoldScaler.fit_transform: fitted on {} rows x {} features",
            X_train.shape[0],
            X_train.shape[1] if X_train.ndim > 1 else 1,
        )

        return scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform *X* using the already-fitted scaler.

        NEVER re-fits the scaler.  Call :meth:`fit_transform` first.

        Args:
            X: Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            Scaled array, same shape as *X*.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
            ValueError: If *X* is empty or contains NaN / Inf.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FoldScaler.transform called before fit_transform. "
                "The scaler must be fitted on the training fold first."
            )

        if X.size == 0:
            raise ValueError("FoldScaler.transform: X is empty.")

        if np.isnan(X).any():
            raise ValueError(
                "FoldScaler.transform: X contains NaN values. "
                "Drop or impute NaN before scaling."
            )

        if np.isinf(X).any():
            raise ValueError(
                "FoldScaler.transform: X contains Inf values. "
                "Clip or remove Inf before scaling."
            )

        return self._scaler.transform(X)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise the fitted scaler to *path/scaler.pkl*.

        Creates *path* (and all parents) if they do not already exist.

        Args:
            path: Directory to write the scaler pickle into.

        Raises:
            RuntimeError: If the scaler has not been fitted.
            OSError: If the directory cannot be created or the file cannot
                be written.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FoldScaler.save called before fit_transform. "
                "Fit the scaler first."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / self._FILENAME

        try:
            joblib.dump(self._scaler, out_file)
            logger.info("FoldScaler: saved scaler to {}", out_file)
        except OSError as exc:
            logger.error("FoldScaler.save failed: {}", exc)
            raise

    def load(self, path: Path) -> None:
        """Deserialise a previously saved scaler from *path/scaler.pkl*.

        After this call :meth:`transform` may be called immediately.

        Args:
            path: Directory written by a previous :meth:`save` call.

        Raises:
            FileNotFoundError: If ``scaler.pkl`` does not exist in *path*.
            OSError: If the file cannot be read.
        """
        pkl_file = Path(path) / self._FILENAME
        if not pkl_file.exists():
            raise FileNotFoundError(
                f"FoldScaler.load: scaler file not found at {pkl_file}"
            )

        try:
            self._scaler = joblib.load(pkl_file)
            self._is_fitted = True
            logger.info("FoldScaler: loaded scaler from {}", pkl_file)
        except OSError as exc:
            logger.error("FoldScaler.load failed: {}", exc)
            raise

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    def assert_not_fitted_on_future(
        self,
        train_end_ts: int,
        scaler_fit_data: pd.DataFrame,
    ) -> None:
        """Assert that *scaler_fit_data* contains no timestamps after *train_end_ts*.

        This is a post-hoc safety check to detect accidental inclusion of
        validation or test rows in the scaler's fit data.

        Args:
            train_end_ts: Maximum allowed timestamp (UTC epoch milliseconds,
                inclusive).  Typically ``max(train_df['open_time'])``.
            scaler_fit_data: DataFrame with an ``open_time`` column (int, UTC
                epoch milliseconds).  These are the rows that were used to fit
                the scaler.

        Raises:
            ValueError: If *scaler_fit_data* does not contain an ``open_time``
                column.
            AssertionError: If any row in *scaler_fit_data* has
                ``open_time > train_end_ts``.
        """
        if "open_time" not in scaler_fit_data.columns:
            raise ValueError(
                "assert_not_fitted_on_future: 'open_time' column not found in "
                "scaler_fit_data."
            )

        max_ts: int = int(scaler_fit_data["open_time"].max())
        future_count: int = int((scaler_fit_data["open_time"] > train_end_ts).sum())

        if future_count > 0:
            raise AssertionError(
                f"FoldScaler: scaler was fitted on {future_count} row(s) with "
                f"open_time > train_end_ts ({train_end_ts}).  "
                f"Max observed timestamp: {max_ts}.  "
                "This indicates a data leak — the scaler must be fitted on "
                "the training fold only."
            )

        logger.debug(
            "FoldScaler.assert_not_fitted_on_future: PASS "
            "(max_ts={}, train_end_ts={})",
            max_ts,
            train_end_ts,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """``True`` if the scaler has been fitted via :meth:`fit_transform`."""
        return self._is_fitted

    @property
    def n_features_in(self) -> int | None:
        """Number of features seen during fit, or ``None`` if not fitted."""
        if not self._is_fitted:
            return None
        return int(self._scaler.n_features_in_)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"FoldScaler(fitted={self._is_fitted})"


__all__ = ["FoldScaler"]
