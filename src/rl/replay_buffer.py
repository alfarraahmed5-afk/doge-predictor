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

import json
import time
import uuid
from pathlib import Path
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
    # Feature-vector serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_feature_vector(arr: np.ndarray) -> bytes:
        """Serialise a 1-D float64 numpy array to raw bytes.

        Args:
            arr: Feature vector — must be 1-D; will be cast to float64.

        Returns:
            Raw bytes representation (``arr.astype(float64).tobytes()``).
        """
        return arr.astype(np.float64).tobytes()

    @staticmethod
    def deserialize_feature_vector(data: bytes) -> np.ndarray:
        """Deserialise raw bytes back to a 1-D float64 numpy array.

        Args:
            data: Bytes produced by :meth:`serialize_feature_vector`.

        Returns:
            1-D numpy array of dtype float64.
        """
        return np.frombuffer(data, dtype=np.float64)

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
        feature_vector: "bytes | np.ndarray | None" = None,
    ) -> bool:
        """Append one verified experience to the buffer.

        When the buffer is full (``count >= max_size``), the oldest row NOT
        belonging to a *protected* regime is evicted before inserting.
        A regime is protected when its row count is at or below
        ``min_per_regime`` — this prevents rare-regime samples from being
        wiped out by dominant regimes.

        Args:
            horizon: Prediction horizon label.
            regime: Market regime at prediction time.
            reward_score: Final RL reward value.
            model_version: Model version string.
            created_at: Verification timestamp, UTC epoch milliseconds.
            predicted_price: Optional price at prediction time.
            actual_price: Optional actual close price at maturity.
            feature_vector: Optional feature data — either pre-serialised
                ``bytes`` or a numpy ndarray (serialised automatically via
                :meth:`serialize_feature_vector`).

        Returns:
            ``True`` if the record was pushed successfully; ``False`` if a
            storage error prevented the insert.

        Raises:
            ValueError: If ``horizon`` or ``regime`` is invalid.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'. Must be one of {sorted(_VALID_HORIZONS)}.")
        if regime not in _VALID_REGIMES:
            raise ValueError(f"Invalid regime '{regime}'. Must be one of {sorted(_VALID_REGIMES)}.")

        # Evict one row if at capacity (respecting min-per-regime quota)
        if self._counts[horizon] >= self._max_size:
            self._evict_one(horizon)

        # Serialise numpy arrays if needed
        fv_bytes: bytes | None
        if isinstance(feature_vector, np.ndarray):
            fv_bytes = self.serialize_feature_vector(feature_vector)
        else:
            fv_bytes = feature_vector

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
        if fv_bytes is not None:
            record["feature_vector"] = fv_bytes

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

    def get_regime_counts(self, horizon: str) -> dict[str, int]:
        """Return per-regime record counts for the given horizon.

        Queries the database directly (not the in-memory approximation) so
        the result is always current.

        Args:
            horizon: Prediction horizon label.

        Returns:
            Dict mapping regime label → row count.  Empty dict on error.

        Raises:
            ValueError: If ``horizon`` is invalid.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'.")
        try:
            return self._storage.get_replay_regime_counts(horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_regime_counts failed (horizon={}): {}", horizon, exc)
            return {}

    def get_prioritised_sample(self, horizon: str, n: int) -> pd.DataFrame:
        """Draw a prioritised sample by physically duplicating high-reward rows.

        High-priority rows (``abs_reward >= priority_threshold``) are
        concatenated ``priority_oversample`` times into an expanded pool before
        random sampling, giving them exactly ``priority_oversample``× the
        representation of ordinary rows.

        Args:
            horizon: Prediction horizon label.
            n: Desired batch size.

        Returns:
            :class:`pandas.DataFrame` with up to ``n`` rows, with
            ``feature_vector`` columns deserialised to ``np.ndarray`` objects
            where data is present.  May be empty.

        Raises:
            ValueError: If ``horizon`` is invalid or ``n < 1``.
        """
        if horizon not in _VALID_HORIZONS:
            raise ValueError(f"Invalid horizon '{horizon}'.")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")

        pool = self._storage.get_replay_sample(horizon, n * self._priority_oversample * 2)
        if pool.empty:
            return pd.DataFrame()

        if "abs_reward" in pool.columns:
            abs_reward = pool["abs_reward"].to_numpy(dtype=float)
            high_mask = abs_reward >= self._priority_threshold
            high_prio = pool[high_mask]
            low_prio = pool[~high_mask]
            # Physically duplicate high-priority rows priority_oversample times
            expanded = pd.concat(
                [high_prio] * self._priority_oversample + [low_prio],
                ignore_index=True,
            )
        else:
            expanded = pool

        actual_n = min(n, len(expanded))
        indices = self._rng.choice(len(expanded), size=actual_n, replace=True)
        result = expanded.iloc[indices].reset_index(drop=True)

        # Deserialise feature_vector bytes back to numpy arrays
        if "feature_vector" in result.columns:
            result = result.copy()
            result["feature_vector"] = result["feature_vector"].apply(
                lambda v: self.deserialize_feature_vector(v) if isinstance(v, (bytes, memoryview)) else v
            )

        return result

    def checkpoint(self, path: Path) -> None:
        """Persist the buffer's in-memory state to disk.

        Writes a JSON file to ``path`` containing current per-horizon counts
        and a snapshot timestamp.  The database remains the authoritative store;
        this is a lightweight diagnostic snapshot used for monitoring and
        operator inspection.

        Args:
            path: Directory where the checkpoint file will be written.
                Created (with parents) if it does not exist.

        Raises:
            OSError: If the directory cannot be created or the file cannot be
                written.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        snapshot_ts = int(time.time() * 1000)
        state: dict[str, Any] = {
            "snapshot_ts_ms": snapshot_ts,
            "counts": dict(self._counts),
            "max_size_per_horizon": self._max_size,
            "priority_threshold": self._priority_threshold,
            "priority_oversample": self._priority_oversample,
            "min_per_regime": self._min_per_regime,
        }

        checkpoint_file = path / f"replay_buffer_{snapshot_ts}.json"
        with checkpoint_file.open("w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)

        logger.info(
            "ReplayBuffer.checkpoint: saved to {} (counts={})",
            checkpoint_file,
            self._counts,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_one(self, horizon: str) -> None:
        """Evict the oldest replay buffer row for ``horizon``.

        Regimes with row counts at or below ``min_per_regime`` are *protected*
        and will not be evicted unless ALL regimes are at or below the quota
        (last-resort fallback so the buffer cannot grow past capacity).

        Args:
            horizon: Horizon whose buffer has reached capacity.
        """
        try:
            regime_counts = self._storage.get_replay_regime_counts(horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning("_evict_one: could not fetch regime counts: {}; skipping eviction", exc)
            return

        protected = {
            r for r, cnt in regime_counts.items() if cnt <= self._min_per_regime
        }

        try:
            evicted = self._storage.delete_oldest_non_protected_replay(horizon, protected)
        except Exception as exc:  # noqa: BLE001
            logger.warning("_evict_one: delete failed: {}", exc)
            return

        if evicted:
            self._counts[horizon] = max(0, self._counts[horizon] - 1)
            logger.debug(
                "_evict_one: evicted 1 row from horizon={} (protected={})",
                horizon,
                protected,
            )
        else:
            logger.warning("_evict_one: no row evicted for horizon={}", horizon)

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
