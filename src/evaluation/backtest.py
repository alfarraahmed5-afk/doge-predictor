"""Backtesting engine for DOGE prediction signals.

Execution realism rules (CLAUDE.md Section 9 — non-negotiable):

* **Fill price**: Next candle's OPEN — never the signal candle's close.
* **Taker fee**: 0.10 % per leg (both entry and exit).
* **Slippage**: Random uniform [0.02 %, 0.08 %] per trade applied to fill price.
* **Position sizing**: 1 % of current equity (normal); 0.5 % in
  ``RANGING_LOW_VOL`` or ``DECOUPLED``.
* **Drawdown halt**: Stop simulation when current drawdown exceeds 25 %.

The anti-lookahead assertion checks that every entry fill time is strictly
later than the signal time.  A failure raises :class:`ExecutionLookaheadError`
immediately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import doge_settings as _default_doge_settings, DogeSettings

# ---------------------------------------------------------------------------
# Constants — all sourced from config, never hardcoded
# ---------------------------------------------------------------------------

_REDUCED_REGIME_LABELS: frozenset[str] = frozenset(
    {"RANGING_LOW_VOL", "DECOUPLED"}
)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ExecutionLookaheadError(Exception):
    """Raised when a fill timestamp is not strictly after the signal timestamp.

    This is the primary anti-lookahead guard in the backtesting engine.
    Any trade whose entry_time <= signal_time violates the execution model and
    invalidates the simulation.
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """A single completed round-trip trade.

    Attributes:
        signal_time: Timestamp (ms) of the signal candle.
        entry_time: Timestamp (ms) of the fill candle (open[t+1]).
        exit_time: Timestamp (ms) of the exit fill candle.
        entry_price: Actual fill price after slippage.
        exit_price: Actual fill price after slippage.
        position_size: Number of units held.
        pnl: Realised PnL in equity units after fees.
        pnl_pct: ``pnl / entry_cost`` as a fraction.
        regime_at_entry: Regime label at signal time.
        equity_before: Equity at trade entry (after fee deduction).
        equity_after: Equity at trade exit (after fee deduction).
        is_winning: ``True`` if ``pnl > 0``.
        duration_hours: Trade duration in candle-hours.
        entry_fee: Fee paid on entry leg.
        exit_fee: Fee paid on exit leg.
    """

    signal_time: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float
    regime_at_entry: str
    equity_before: float
    equity_after: float
    is_winning: bool
    duration_hours: float
    entry_fee: float
    exit_fee: float


@dataclass(frozen=True)
class BacktestResult:
    """Immutable container for the full backtesting output.

    Attributes:
        trade_log: All completed trades (open positions are excluded).
        equity_curve: Equity value at every processed candle (indexed by
            open_time milliseconds).
        initial_equity: Starting equity value.
        final_equity: Equity at end of simulation.
        n_signals: Total signals evaluated.
        halt_reason: Non-empty string when the simulation stopped early.
        config_snapshot: Snapshot of BacktestSettings used for this run.
    """

    trade_log: list[TradeRecord]
    equity_curve: dict[int, float]
    initial_equity: float
    final_equity: float
    n_signals: int
    halt_reason: str
    config_snapshot: dict[str, Any]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Simulates trade execution from a signal series against price data.

    All execution parameters are loaded from :class:`~src.config.DogeSettings`
    so that no numeric constants are hardcoded here.

    Args:
        doge_cfg: DOGE settings instance.  Defaults to the module-level
            singleton loaded from ``config/doge_settings.yaml``.
        seed: Random seed for slippage sampling.  Pass an explicit integer
            for reproducible results (default ``42``).
        initial_equity: Starting equity value in USD (default ``10_000.0``).
    """

    def __init__(
        self,
        doge_cfg: DogeSettings | None = None,
        seed: int = 42,
        initial_equity: float = 10_000.0,
    ) -> None:
        """Initialise the backtesting engine.

        Args:
            doge_cfg: DOGE configuration.  Uses module singleton if *None*.
            seed: Slippage RNG seed.
            initial_equity: Starting equity in USD.
        """
        self._cfg: DogeSettings = (
            doge_cfg if doge_cfg is not None else _default_doge_settings
        )
        self._seed: int = seed
        self._initial_equity: float = initial_equity

        # Convenience aliases from config
        self._taker_fee: float = self._cfg.backtest.taker_fee
        self._slip_min: float = self._cfg.backtest.slippage_min
        self._slip_max: float = self._cfg.backtest.slippage_max
        self._base_risk: float = self._cfg.backtest.position_size_pct
        self._reduced_risk: float = self._cfg.backtest.reduced_position_size_pct
        self._max_dd_halt: float = self._cfg.backtest.max_drawdown_halt

        logger.debug(
            "BacktestEngine init: fee={:.4f}, slip=[{:.4f},{:.4f}], "
            "base_risk={:.3f}, reduced_risk={:.3f}, max_dd_halt={:.2f}",
            self._taker_fee,
            self._slip_min,
            self._slip_max,
            self._base_risk,
            self._reduced_risk,
            self._max_dd_halt,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        regimes: pd.Series,
    ) -> BacktestResult:
        """Run the backtest simulation.

        Args:
            signals: Series of ``"BUY"``, ``"SELL"``, or ``"HOLD"`` strings
                indexed by ``open_time`` (int, UTC milliseconds).  Signals are
                aligned to the SIGNAL candle's ``open_time``.
            prices: DataFrame with columns ``["open_time", "open", "close"]``
                containing candle data.  ``open_time`` must be monotonically
                increasing.  The ``open`` column is used for fills.
            regimes: Series of regime label strings indexed by ``open_time``
                (int, UTC milliseconds).

        Returns:
            :class:`BacktestResult` with trade log, equity curve, and summary.

        Raises:
            ExecutionLookaheadError: If any entry fill time <= signal time.
            ValueError: If required columns are missing or inputs are empty.
        """
        self._validate_inputs(signals, prices, regimes)

        rng = random.Random(self._seed)
        np.random.seed(self._seed)

        # Build fast O(1) price lookup: open_time → open price
        price_index: dict[int, float] = dict(
            zip(prices["open_time"].astype(int), prices["open"].astype(float))
        )

        # Sorted open_time list for finding next candle
        sorted_times: list[int] = sorted(price_index.keys())
        next_open_map: dict[int, int] = {}
        for i, t in enumerate(sorted_times[:-1]):
            next_open_map[t] = sorted_times[i + 1]

        equity: float = self._initial_equity
        peak_equity: float = self._initial_equity
        equity_curve: dict[int, float] = {}
        trade_log: list[TradeRecord] = []
        halt_reason: str = ""

        # Open position state
        in_position: bool = False
        entry_price_filled: float = 0.0
        entry_time_ms: int = 0
        signal_time_ms: int = 0
        position_size_units: float = 0.0
        entry_fee: float = 0.0
        regime_at_entry: str = "RANGING_LOW_VOL"
        equity_at_entry: float = 0.0

        n_signals: int = 0

        signal_times = signals.index.tolist()

        for signal_time in signal_times:
            signal_time = int(signal_time)
            signal_val: str = str(signals.loc[signal_time])

            # Record equity at each signal candle for the curve
            equity_curve[signal_time] = equity

            n_signals += 1

            # ---- Drawdown halt check ----
            if equity > peak_equity:
                peak_equity = equity
            current_dd: float = (peak_equity - equity) / peak_equity
            if current_dd > self._max_dd_halt:
                halt_reason = (
                    f"Max drawdown halt: drawdown={current_dd:.4f} "
                    f"> threshold={self._max_dd_halt:.4f}"
                )
                logger.warning("Backtest halted: {}", halt_reason)
                break

            # ---- Determine fill candle (next candle after signal) ----
            if signal_time not in next_open_map:
                # Signal is on the last candle — no next candle to fill
                logger.debug(
                    "Signal at {} has no next candle; skipping.", signal_time
                )
                continue

            fill_time_ms: int = next_open_map[signal_time]
            fill_open_price: float = price_index[fill_time_ms]

            # ---- Anti-lookahead assertion (CRITICAL) ----
            if fill_time_ms <= signal_time:
                raise ExecutionLookaheadError(
                    f"Entry fill time {fill_time_ms} <= signal time "
                    f"{signal_time}. Lookahead bias detected — "
                    f"backtesting integrity violated."
                )

            # ---- Exit: SELL when long ----
            if signal_val == "SELL" and in_position:
                slip = rng.uniform(self._slip_min, self._slip_max)
                exit_price: float = fill_open_price * (1.0 - slip)  # sell at slightly lower
                exit_fee_amt: float = exit_price * position_size_units * self._taker_fee
                proceeds: float = (
                    exit_price * position_size_units - exit_fee_amt
                )
                pnl: float = proceeds - (entry_price_filled * position_size_units + entry_fee)
                pnl_pct: float = pnl / (entry_price_filled * position_size_units + entry_fee)
                new_equity: float = equity + pnl

                trade = TradeRecord(
                    signal_time=signal_time_ms,
                    entry_time=entry_time_ms,
                    exit_time=fill_time_ms,
                    entry_price=entry_price_filled,
                    exit_price=exit_price,
                    position_size=position_size_units,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    regime_at_entry=regime_at_entry,
                    equity_before=equity_at_entry,
                    equity_after=new_equity,
                    is_winning=pnl > 0.0,
                    duration_hours=float(fill_time_ms - entry_time_ms) / 3_600_000.0,
                    entry_fee=entry_fee,
                    exit_fee=exit_fee_amt,
                )
                trade_log.append(trade)
                logger.debug(
                    "SELL fill: regime={}, pnl={:.4f}, pnl_pct={:.4f}",
                    regime_at_entry,
                    pnl,
                    pnl_pct,
                )
                equity = new_equity
                in_position = False

            # ---- Entry: BUY when no position ----
            elif signal_val == "BUY" and not in_position:
                current_regime: str = str(
                    regimes.loc[signal_time]
                    if signal_time in regimes.index
                    else "RANGING_LOW_VOL"
                )
                risk_pct: float = (
                    self._reduced_risk
                    if current_regime in _REDUCED_REGIME_LABELS
                    else self._base_risk
                )
                trade_value: float = equity * risk_pct

                slip = rng.uniform(self._slip_min, self._slip_max)
                entry_price_slip: float = fill_open_price * (1.0 + slip)  # buy at slightly higher
                entry_fee_amt: float = trade_value * self._taker_fee
                position_size_units = trade_value / entry_price_slip

                # Deduct entry fee from equity immediately
                equity_at_entry = equity
                equity -= entry_fee_amt

                entry_price_filled = entry_price_slip
                entry_time_ms = fill_time_ms
                signal_time_ms = signal_time
                entry_fee = entry_fee_amt
                regime_at_entry = current_regime
                in_position = True

                logger.debug(
                    "BUY fill: time={}, price={:.6f}, regime={}, size={:.2f}",
                    fill_time_ms,
                    entry_price_slip,
                    current_regime,
                    position_size_units,
                )

        # Final equity curve point
        if signal_times:
            last_t = int(signal_times[-1])
            equity_curve[last_t] = equity

        config_snapshot: dict[str, Any] = {
            "taker_fee": self._taker_fee,
            "slippage_min": self._slip_min,
            "slippage_max": self._slip_max,
            "base_risk_pct": self._base_risk,
            "reduced_risk_pct": self._reduced_risk,
            "max_drawdown_halt": self._max_dd_halt,
            "initial_equity": self._initial_equity,
            "seed": self._seed,
        }

        logger.info(
            "Backtest complete: trades={}, final_equity={:.2f}, halt='{}'",
            len(trade_log),
            equity,
            halt_reason or "none",
        )

        return BacktestResult(
            trade_log=trade_log,
            equity_curve=equity_curve,
            initial_equity=self._initial_equity,
            final_equity=equity,
            n_signals=n_signals,
            halt_reason=halt_reason,
            config_snapshot=config_snapshot,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        signals: pd.Series,
        prices: pd.DataFrame,
        regimes: pd.Series,
    ) -> None:
        """Validate that inputs satisfy the engine's preconditions.

        Args:
            signals: Signal series.
            prices: Price DataFrame.
            regimes: Regime label series.

        Raises:
            ValueError: On any precondition violation.
        """
        if signals.empty:
            raise ValueError("signals Series is empty.")
        required_cols = {"open_time", "open"}
        missing = required_cols - set(prices.columns)
        if missing:
            raise ValueError(
                f"prices DataFrame is missing required columns: {missing}"
            )
        if prices.empty:
            raise ValueError("prices DataFrame is empty.")
        if not prices["open_time"].is_monotonic_increasing:
            raise ValueError(
                "prices['open_time'] is not monotonically increasing."
            )


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "TradeRecord",
    "ExecutionLookaheadError",
]
