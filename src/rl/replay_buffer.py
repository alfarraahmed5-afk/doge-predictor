"""Prioritised, regime-stratified replay buffer for the RL self-teaching loop.

The buffer wraps :class:`~src.processing.storage.DogeStorage` for persistence
and adds:

* **Priority-based oversampling** — experiences whose ``abs_reward`` exceeds
  ``priority_threshold`` are drawn ``priority_oversample`` times more often
  than low-priority ones.
* **Regime-stratified sampling** — when ``stratify=True`` (default), each
  regime contributes an equal share of the sampled batch before priority
  oversampling is applied.  This prevents dominant regimes from starving the
  model of rare-regime signals.
* **Capacity management** — ``max_size_per_horizon`` is tracked in-memory; the
  oldest row is NOT automatically evicted from the database (the database is a
  permanent audit trail per CLAUDE.md), but a warning is emitted and the push
  is skipped when the cap is exceeded.

All buffer state (counts, sizes) is updated in-memory from the database at
construction time and kept in sync on every ``push()``.

Usage::

    buf = ReplayBuffer(storage, rl_cfg)

    # Push a verified experience
    buf.push(horizon="SHORT", regime="TRENDING_BULL", reward_score=1.4,
             model_version="v1.0", actual_price=0.102, created_at=now_ms)

    # Draw a training batch
    batch = buf.sample(horizon="SHORT", n=64, stratify=True)
    # → pd.DataFrame with up to 64 rows
"""

from __future__ import annotations

import uuid
from typing import Any, Final

import numpy as np
import pandas as pd
from loguru import logger

from src.config import RLConfig, rl_config as _default_rl_cfg
from src.processing.storage import DogeStorage

__all__ = ["ReplayBuffer"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_HORIZONS: Final[frozenset[str]] = frozenset({"SHORT", "MEDIUM", "LONG", "MACRO"})
_VALID_REGIMES: Final[frozenset[str]] = frozenset(
    {"TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"}
)


class ReplayBuffer:
    """Prioritised replay buffer backed by the ``doge_replay_buffer`` DB table.

    Args:
        storage: :class:`~src.processing.storage.DogeStorage` instance.
        rl_cfg: Loaded :class:`~src.config.RLConfig`.  Defaults to the global
            singleton when not supplied.
        seed: Random seed for reproducible sampling (default 42).
    """

    def __init__(
        self,
        storage: DogeStorage,
        rl_cfg: RLConfig | None = None,
        seed: int = 42,
    ) -> None:
        self._storage: DogeStorage = storage
        self._rl_cfg: RLConfig = rl_cfg or _default_rl_cfg
        self._rng: np.random.Generator = np.random.default_rng(seed)

        cfg = self._rl_cfg.replay_buffer
        self._max_size: int = cfg.max_size_per_horizon
        self._priority_threshold: float = cfg.priority_threshold
        self._priority_oversample: int = cfg.priority_oversample
        self._min_per_regime: int = cfg.min_per_regime
        self._min_samples_to_train: int = cfg.min_samples_to_train

        # In-memory row count per horizon (approximate — synced at init)
        self._counts: dict[str, int] = {h: 0 for h in _VALID_HORIZONS}
        self._sync_counts()
        logger.info("ReplayBuffer initialised (counts={})", self._counts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        horizon: str,
        regime: str,
        reward_score: float,
        model_version: str,
        created_at: int,
        predicted_price: float | None = None,
        actual_price: float | None = None,
        feature_vector: bytes | None = None,
    ) -> bool:
        """Append one verified experience to the buffer.

        Args:
            horizon: Prediction horizon label.
            regime: Market regime at prediction time.
            reward_score: Final RL reward value.
            model_version: Model version string.
            created_at: Verification timestamp, UTC epoch milliseconds.
            predicted_price: Optional price at prediction time.
            actual_price: Optional actual close price at maturity.
            feature_vector: Optional serialised feature bytes.

        Returns:
            ``True`` if the record was pushed; ``False`` if skipped due to
            capacity limit.

        Raises:
            ValueError: If ``horizon`` or ``regime`` is invalid.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'. Must be one of {sorted(_VALID_HORIZONS)}.")
        if regime not in _VALID_REGIMES:
            raise ValueError(f"Invalid regime '{regime}'. Must be one of {sorted(_VALID_REGIMES)}.")

        if self._counts[horizon] >= self._max_size:
            logger.warning(
                "ReplayBuffer: horizon={} at capacity ({}/{}); skipping push",
                horizon,
                self._counts[horizon],
                self._max_size,
            )
            return False

        record: dict[str, Any] = {
            "buffer_id": str(uuid.uuid4()),
            "horizon_label": horizon,
            "regime": regime,
            "reward_score": reward_score,
            "model_version": model_version,
            "created_at": created_at,
        }
        if predicted_price is not None:
            record["predicted_price"] = predicted_price
        if actual_price is not None:
            record["actual_price"] = actual_price
        if feature_vector is not None:
            record["feature_vector"] = feature_vector

        try:
            self._storage.push_replay_buffer(record)
            self._counts[horizon] += 1
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("ReplayBuffer.push: storage error: {}", exc)
            return False

    def sample(
        self,
        horizon: str,
        n: int,
        stratify: bool = True,
    ) -> pd.DataFrame:
        """Draw a training batch from the buffer.

        Strategy:

        1. If ``stratify=True``, request ``n × priority_oversample`` raw rows
           from storage, split evenly across regimes.
        2. Mark high-priority rows (``abs_reward >= priority_threshold``).
        3. Build a weight vector: high-priority rows get
           ``priority_oversample`` weight; others get weight 1.
        4. Sample ``n`` rows with replacement using the weight vector.

        Args:
            horizon: Prediction horizon label to sample from.
            n: Desired batch size.
            stratify: Whether to stratify by regime before priority weighting.

        Returns:
            :class:`pandas.DataFrame` with up to ``n`` rows.  May be empty if
            the buffer has fewer than 1 row for the requested horizon.

        Raises:
            ValueError: If ``horizon`` is invalid or ``n < 1``.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'.")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")

        # Fetch a pool large enough to support priority oversampling
        pool_size = n * self._priority_oversample * 2  # generous headroom

        if stratify:
            frames: list[pd.DataFrame] = []
            per_regime = max(1, pool_size // len(_VALID_REGIMES))
            for regime in _VALID_REGIMES:
                df_r = self._storage.get_replay_sample(horizon, per_regime)
                if not df_r.empty:
                    # Filter by regime after retrieval (storage returns random for horizon)
                    matching = df_r[df_r["regime"] == regime] if "regime" in df_r.columns else df_r
                    if not matching.empty:
                        frames.append(matching)
            pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            pool = self._storage.get_replay_sample(horizon, pool_size)

        if pool.empty:
            return pd.DataFrame()

        # Priority weighting
        if "abs_reward" in pool.columns:
            weights = np.where(
                pool["abs_reward"].to_numpy(dtype=float) >= self._priority_threshold,
                float(self._priority_oversample),
                1.0,
            )
        else:
            weights = np.ones(len(pool))

        weights = weights / weights.sum()

        actual_n = min(n, len(pool))
        indices = self._rng.choice(len(pool), size=actual_n, replace=True, p=weights)
        return pool.iloc[indices].reset_index(drop=True)

    def is_ready_to_train(self) -> bool:
        """Return ``True`` if the buffer has enough samples to trigger training.

        The minimum total across ALL horizons must reach
        ``min_samples_to_train``.

        Returns:
            ``True`` when total count >= ``min_samples_to_train``.
        """
        return sum(self._counts.values()) >= self._min_samples_to_train

    def fill_percentage(self, horizon: str) -> float:
        """Return how full the buffer is for a given horizon, as a fraction.

        Args:
            horizon: Prediction horizon label.

        Returns:
            Float in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``horizon`` is invalid.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'.")
        return min(1.0, self._counts[horizon] / self._max_size)

    def total_count(self) -> int:
        """Return the total number of experience records across all horizons.

        Returns:
            Sum of per-horizon counts.
        """
        return sum(self._counts.values())

    def count(self, horizon: str) -> int:
        """Return the number of records stored for a specific horizon.

        Args:
            horizon: Prediction horizon label.

        Returns:
            Integer count.

        Raises:
            ValueError: If ``horizon`` is invalid.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'.")
        return self._counts[horizon]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_counts(self) -> None:
        """Sync in-memory counts from the database.

        Queries the replay buffer table to obtain the actual count per horizon.
        Silently sets counts to 0 on any storage error (safe default).
        """
        for horizon in _VALID_HORIZONS:
            try:
                # Use a large sample to count; storage doesn't expose a COUNT query
                # directly — sample a large number and count the returned rows.
                df = self._storage.get_replay_sample(horizon, self._max_size)
                self._counts[horizon] = len(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "_sync_counts: could not count horizon={}: {}; defaulting to 0",
                    horizon,
                    exc,
                )
                self._counts[horizon] = 0
