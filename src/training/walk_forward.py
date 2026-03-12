"""Walk-forward cross-validation engine for time-series model training.

RULE C (CLAUDE.md Section 3.2):
    ``sklearn.model_selection.train_test_split`` is **BANNED** for this project.
    Only this module is used for all cross-validation.  Every fold is validated
    by an explicit timestamp assertion:

        ``assert max(train_timestamps) < min(val_timestamps)``

    A violation of this assertion means the fold was constructed with future
    data in the training set — a critical data-leakage bug.

Walk-forward scheme:
    The dataset is sliced into overlapping train+val windows that move forward
    in time by ``step_size_days`` at each step:

        Window k:
            train: [start + k*step, start + k*step + training_window)
            val:   [start + k*step + training_window,
                    start + k*step + training_window + validation_window)

    ``training_window_days``, ``validation_window_days``, and ``step_size_days``
    are loaded from ``config/doge_settings.yaml`` via
    :class:`~src.config.WalkForwardSettings`.

Era guard:
    Rows with ``era == 'context'`` (pre-2022 data) must NEVER appear in any
    fold.  Training and validation must only use ``era == 'training'`` rows.
    An :class:`AssertionError` is raised if this contract is violated.

Minimum folds:
    At least 3 folds are required.  If the dataset is too small to produce 3
    folds, :class:`ValueError` is raised with a diagnostic message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd
from loguru import logger

from src.config import WalkForwardSettings, doge_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_DAY: int = 86_400_000  # milliseconds per calendar day
_ERA_CONTEXT: str = "context"
_ERA_TRAINING: str = "training"
_MIN_FOLDS: int = 3  # minimum acceptable number of folds

# ---------------------------------------------------------------------------
# Fold dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """Immutable descriptor of a single walk-forward fold.

    Attributes:
        fold_number: 1-based fold index.
        train_start: Inclusive training start (UTC epoch milliseconds).
        train_end: Inclusive training end (UTC epoch milliseconds).
        val_start: Inclusive validation start (UTC epoch milliseconds).
        val_end: Inclusive validation end (UTC epoch milliseconds).
        n_train: Number of training rows.
        n_val: Number of validation rows.
    """

    fold_number: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    n_train: int = field(default=0)
    n_val: int = field(default=0)


# ---------------------------------------------------------------------------
# WalkForwardCV
# ---------------------------------------------------------------------------


class WalkForwardCV:
    """Walk-forward cross-validator for DOGE time-series data.

    Generates :class:`Fold` descriptors and provides a :meth:`split` method
    that yields ``(train_df, val_df)`` tuples with strict temporal ordering
    and era guards enforced.

    Args:
        cfg: Walk-forward settings instance.  If *None*, the module-level
            ``doge_settings.walk_forward`` singleton is used.

    Example::

        cv = WalkForwardCV()
        folds = cv.generate_folds(feature_df)
        for train_df, val_df in cv.split(feature_df):
            scaler = FoldScaler()
            X_train = scaler.fit_transform(train_df[feature_cols].values)
            X_val   = scaler.transform(val_df[feature_cols].values)
            model.fit(X_train, train_df['target'].values,
                      X_val,   val_df['target'].values)
    """

    def __init__(
        self,
        cfg: WalkForwardSettings | None = None,
    ) -> None:
        """Initialise the walk-forward CV engine.

        Args:
            cfg: Walk-forward parameters.  Defaults to
                ``doge_settings.walk_forward``.
        """
        self._cfg: WalkForwardSettings = (
            cfg if cfg is not None else doge_settings.walk_forward
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_folds(self, df: pd.DataFrame) -> list[Fold]:
        """Generate a list of :class:`Fold` objects from *df*.

        The DataFrame must contain an ``open_time`` column (UTC epoch
        milliseconds, int) and an ``era`` column.  Only rows where
        ``era == 'training'`` are eligible — context-era rows are excluded
        before fold generation.

        Args:
            df: Feature DataFrame containing ``open_time`` (int) and
                ``era`` (str) columns, sorted ascending by ``open_time``.

        Returns:
            Ordered list of :class:`Fold` objects (at least
            :const:`_MIN_FOLDS`).

        Raises:
            ValueError: If required columns are missing, if there is
                insufficient data for :const:`_MIN_FOLDS` folds, or if
                ``min_training_rows`` cannot be met by any fold.
            AssertionError: If context-era rows are detected after filtering.
        """
        self._validate_input(df)

        # Filter to training era only — context rows must never enter a fold
        training_df = df[df["era"] == _ERA_TRAINING].copy()
        if len(training_df) == 0:
            raise ValueError(
                "WalkForwardCV.generate_folds: no training-era rows found. "
                "Ensure df contains rows with era='training'."
            )

        # Confirm no context rows slipped through (belt-and-suspenders)
        assert (training_df["era"] != _ERA_CONTEXT).all(), (
            "WalkForwardCV: context-era rows detected after filtering. "
            "This is a bug in the caller."
        )

        training_df = training_df.sort_values("open_time").reset_index(drop=True)
        first_ts: int = int(training_df["open_time"].iloc[0])
        last_ts: int = int(training_df["open_time"].iloc[-1])

        train_ms: int = self._cfg.training_window_days * _MS_PER_DAY
        val_ms: int = self._cfg.validation_window_days * _MS_PER_DAY
        step_ms: int = self._cfg.step_size_days * _MS_PER_DAY

        folds: list[Fold] = []
        fold_number: int = 1
        cursor: int = first_ts

        while True:
            train_start: int = cursor
            train_end: int = cursor + train_ms - 1
            val_start: int = cursor + train_ms
            val_end: int = cursor + train_ms + val_ms - 1

            # Stop if the validation window extends beyond the dataset
            if val_end > last_ts:
                break

            # Slice the DataFrames for this fold
            train_slice = training_df[
                (training_df["open_time"] >= train_start)
                & (training_df["open_time"] <= train_end)
            ]
            val_slice = training_df[
                (training_df["open_time"] >= val_start)
                & (training_df["open_time"] <= val_end)
            ]

            n_train = len(train_slice)
            n_val = len(val_slice)

            if n_train < self._cfg.min_training_rows:
                logger.debug(
                    "WalkForwardCV: fold {} skipped — only {} train rows "
                    "(min_training_rows={})",
                    fold_number,
                    n_train,
                    self._cfg.min_training_rows,
                )
                cursor += step_ms
                fold_number += 1
                continue

            if n_val == 0:
                logger.debug(
                    "WalkForwardCV: fold {} skipped — 0 validation rows",
                    fold_number,
                )
                cursor += step_ms
                fold_number += 1
                continue

            # RULE C assertion — temporal ordering guarantee
            max_train_ts: int = int(train_slice["open_time"].max())
            min_val_ts: int = int(val_slice["open_time"].min())
            assert max_train_ts < min_val_ts, (
                f"WalkForwardCV fold {fold_number}: temporal ordering violated. "
                f"max(train open_time)={max_train_ts} >= min(val open_time)={min_val_ts}. "
                "This is a critical data-leakage bug."
            )

            # Era guard — neither slice may contain context rows
            assert (train_slice["era"] == _ERA_TRAINING).all(), (
                f"WalkForwardCV fold {fold_number}: context-era rows in TRAIN slice."
            )
            assert (val_slice["era"] == _ERA_TRAINING).all(), (
                f"WalkForwardCV fold {fold_number}: context-era rows in VAL slice."
            )

            fold = Fold(
                fold_number=fold_number,
                train_start=int(train_slice["open_time"].min()),
                train_end=max_train_ts,
                val_start=min_val_ts,
                val_end=int(val_slice["open_time"].max()),
                n_train=n_train,
                n_val=n_val,
            )
            folds.append(fold)

            logger.info(
                "WalkForwardCV: fold {:3d} — train [{} → {}] ({} rows) | "
                "val [{} → {}] ({} rows)",
                fold.fold_number,
                pd.Timestamp(fold.train_start, unit="ms", tz="UTC").strftime(
                    "%Y-%m-%d"
                ),
                pd.Timestamp(fold.train_end, unit="ms", tz="UTC").strftime(
                    "%Y-%m-%d"
                ),
                fold.n_train,
                pd.Timestamp(fold.val_start, unit="ms", tz="UTC").strftime(
                    "%Y-%m-%d"
                ),
                pd.Timestamp(fold.val_end, unit="ms", tz="UTC").strftime(
                    "%Y-%m-%d"
                ),
                fold.n_val,
            )

            cursor += step_ms
            fold_number += 1

        if len(folds) < _MIN_FOLDS:
            raise ValueError(
                f"WalkForwardCV: only {len(folds)} valid fold(s) generated "
                f"(minimum required: {_MIN_FOLDS}).  "
                f"The dataset spans {(last_ts - first_ts) / _MS_PER_DAY:.0f} days; "
                f"training_window_days={self._cfg.training_window_days}, "
                f"validation_window_days={self._cfg.validation_window_days}, "
                f"step_size_days={self._cfg.step_size_days}.  "
                "Extend the dataset or reduce the training/validation window."
            )

        logger.info(
            "WalkForwardCV: generated {} folds "
            "(training={} d, val={} d, step={} d)",
            len(folds),
            self._cfg.training_window_days,
            self._cfg.validation_window_days,
            self._cfg.step_size_days,
        )

        return folds

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield ``(train_df, val_df)`` pairs for each walk-forward fold.

        Each yielded pair has the same guarantees as :meth:`generate_folds`:
        temporal ordering is verified by assertion and context-era rows are
        excluded.

        Args:
            df: Feature DataFrame containing ``open_time`` and ``era`` columns.

        Yields:
            ``(train_df, val_df)`` tuples.  Both DataFrames are copies and
            sorted ascending by ``open_time``.

        Raises:
            ValueError: See :meth:`generate_folds`.
            AssertionError: See :meth:`generate_folds`.
        """
        folds = self.generate_folds(df)
        training_df = (
            df[df["era"] == _ERA_TRAINING]
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        for fold in folds:
            train_df = training_df[
                (training_df["open_time"] >= fold.train_start)
                & (training_df["open_time"] <= fold.train_end)
            ].copy()

            val_df = training_df[
                (training_df["open_time"] >= fold.val_start)
                & (training_df["open_time"] <= fold.val_end)
            ].copy()

            yield train_df, val_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise :class:`ValueError` if required columns are missing.

        Args:
            df: Input DataFrame to validate.

        Raises:
            ValueError: If ``open_time`` or ``era`` columns are absent.
        """
        required = {"open_time", "era"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"WalkForwardCV: missing required columns: {sorted(missing)}.  "
                "The DataFrame must contain 'open_time' (int, UTC epoch ms) "
                "and 'era' (str: 'training' | 'context')."
            )


__all__ = ["WalkForwardCV", "Fold"]
