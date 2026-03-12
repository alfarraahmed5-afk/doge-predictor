"""Backtesting metrics computation.

All metrics are computed from a :class:`~src.evaluation.backtest.BacktestResult`
and its embedded :class:`~src.evaluation.backtest.TradeRecord` list.

Annualisation assumes 8 760 hourly candles per year (365 × 24).

Per-regime metrics group trades by ``TradeRecord.regime_at_entry`` and compute
identical statistics on each subset.  Regimes with fewer than 2 trades return
``None`` for metrics that require variance (Sharpe, Calmar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from src.evaluation.backtest import BacktestResult, TradeRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOURS_PER_YEAR: int = 8_760  # 365 × 24  — per CLAUDE.md Section 9
_MIN_TRADES_FOR_RATIO: int = 2  # Minimum trades needed for ratio metrics


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsResult:
    """Full metrics suite for a backtest run.

    Attributes:
        directional_accuracy: Fraction of signals where predicted direction
            was correct (based on trade PnL sign).
        sharpe_ratio: Annualised Sharpe ratio.  ``None`` if fewer than
            :data:`_MIN_TRADES_FOR_RATIO` trades or zero std deviation.
        max_drawdown: Peak-to-trough percentage drawdown (0–1 scale).
        calmar_ratio: Annualised return / max_drawdown.  ``None`` when
            max_drawdown is zero or insufficient trades.
        win_rate: Fraction of trades with positive PnL.
        profit_factor: Gross profit / gross loss.  ``None`` if no losing
            trades exist.
        total_trades: Count of completed round-trip trades.
        avg_trade_duration_hours: Mean trade duration in hours.
        annualised_return: Geometric annualised return computed from equity
            curve.
        per_regime: Dict mapping regime label → :class:`RegimeMetrics`.
            Only regimes with at least 1 trade are included.
    """

    directional_accuracy: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    calmar_ratio: Optional[float]
    win_rate: float
    profit_factor: Optional[float]
    total_trades: int
    avg_trade_duration_hours: float
    annualised_return: float
    per_regime: dict[str, "RegimeMetrics"]


@dataclass(frozen=True)
class RegimeMetrics:
    """Per-regime metric subset.

    Attributes:
        regime: Regime label string.
        n_trades: Number of trades in this regime.
        win_rate: Fraction of winning trades.
        profit_factor: Gross profit / gross loss.  ``None`` if no losers.
        sharpe_ratio: Annualised Sharpe.  ``None`` if < 2 trades.
        max_drawdown: Worst peak-to-trough within regime trades.
        avg_trade_duration_hours: Mean trade duration in hours.
        total_pnl: Sum of all trade PnLs in this regime.
    """

    regime: str
    n_trades: int
    win_rate: float
    profit_factor: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: float
    avg_trade_duration_hours: float
    total_pnl: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_metrics(result: BacktestResult) -> MetricsResult:
    """Compute the full metrics suite from a :class:`BacktestResult`.

    Args:
        result: Completed backtest result containing trade log and equity curve.

    Returns:
        :class:`MetricsResult` with all metrics populated.
    """
    trades = result.trade_log
    n_trades = len(trades)

    logger.debug("compute_metrics: n_trades={}", n_trades)

    # ---- Directional accuracy ----
    if n_trades == 0:
        dir_acc = 0.0
    else:
        correct = sum(1 for t in trades if t.is_winning)
        dir_acc = correct / n_trades

    # ---- Win rate ----
    win_rate = dir_acc  # same definition for long-only strategy

    # ---- Profit factor ----
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0.0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0.0))
    profit_factor: Optional[float] = (
        gross_profit / gross_loss if gross_loss > 0.0 else None
    )

    # ---- Avg trade duration ----
    avg_duration = (
        sum(t.duration_hours for t in trades) / n_trades if n_trades > 0 else 0.0
    )

    # ---- Equity curve drawdown ----
    equity_values = list(result.equity_curve.values())
    max_dd = _compute_max_drawdown(equity_values)

    # ---- Annualised return ----
    ann_return = _compute_annualised_return(
        result.initial_equity,
        result.final_equity,
        len(equity_values),
    )

    # ---- Sharpe ratio ----
    sharpe: Optional[float] = None
    if n_trades >= _MIN_TRADES_FOR_RATIO:
        pnl_pcts = [t.pnl_pct for t in trades]
        sharpe = _compute_sharpe(pnl_pcts)

    # ---- Calmar ratio ----
    calmar: Optional[float] = None
    if max_dd > 0.0 and n_trades >= _MIN_TRADES_FOR_RATIO:
        calmar = ann_return / max_dd if max_dd > 0.0 else None

    # ---- Per-regime metrics ----
    regime_groups: dict[str, list[TradeRecord]] = {}
    for t in trades:
        regime_groups.setdefault(t.regime_at_entry, []).append(t)

    per_regime: dict[str, RegimeMetrics] = {}
    for regime_label, regime_trades in regime_groups.items():
        per_regime[regime_label] = _compute_regime_metrics(
            regime_label, regime_trades
        )

    metrics = MetricsResult(
        directional_accuracy=dir_acc,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=n_trades,
        avg_trade_duration_hours=avg_duration,
        annualised_return=ann_return,
        per_regime=per_regime,
    )

    logger.info(
        "Metrics: dir_acc={:.3f}, sharpe={}, max_dd={:.3f}, "
        "win_rate={:.3f}, trades={}, regimes={}",
        metrics.directional_accuracy,
        f"{metrics.sharpe_ratio:.3f}" if metrics.sharpe_ratio is not None else "N/A",
        metrics.max_drawdown,
        metrics.win_rate,
        metrics.total_trades,
        list(per_regime.keys()),
    )

    return metrics


def check_acceptance_gates(
    metrics: MetricsResult,
    gates: dict[str, float] | None = None,
) -> dict[str, bool]:
    """Check whether metrics pass the Section 9 acceptance gates.

    Args:
        metrics: Computed metrics result.
        gates: Optional dict of gate names → minimum values.  Defaults to
            the values specified in CLAUDE.md Section 9.

    Returns:
        Dict mapping gate name → ``True`` (pass) / ``False`` (fail).
    """
    if gates is None:
        gates = {
            "directional_accuracy_oos": 0.54,
            "sharpe_annualized": 1.0,
            "max_drawdown": 0.20,
            "calmar_ratio": 0.6,
            "profit_factor": 1.3,
            "win_rate": 0.45,
            "min_trade_count": 150,
        }

    results: dict[str, bool] = {}

    results["directional_accuracy_oos"] = (
        metrics.directional_accuracy >= gates["directional_accuracy_oos"]
    )
    results["sharpe_annualized"] = (
        metrics.sharpe_ratio is not None
        and metrics.sharpe_ratio >= gates["sharpe_annualized"]
    )
    results["max_drawdown"] = metrics.max_drawdown <= gates["max_drawdown"]
    results["calmar_ratio"] = (
        metrics.calmar_ratio is not None
        and metrics.calmar_ratio >= gates["calmar_ratio"]
    )
    results["profit_factor"] = (
        metrics.profit_factor is not None
        and metrics.profit_factor >= gates["profit_factor"]
    )
    results["win_rate"] = metrics.win_rate >= gates["win_rate"]
    results["min_trade_count"] = metrics.total_trades >= int(
        gates["min_trade_count"]
    )

    passed = sum(results.values())
    logger.info(
        "Acceptance gates: {}/{} passed", passed, len(results)
    )

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_max_drawdown(equity_values: list[float]) -> float:
    """Compute the peak-to-trough maximum drawdown.

    Args:
        equity_values: Equity value at each time step (chronological order).

    Returns:
        Maximum drawdown as a fraction in ``[0, 1]``.  Returns ``0.0`` for
        fewer than 2 equity points.
    """
    if len(equity_values) < 2:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for val in equity_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_annualised_return(
    initial: float,
    final: float,
    n_hourly_steps: int,
) -> float:
    """Compute the geometric annualised return.

    Args:
        initial: Starting equity.
        final: Ending equity.
        n_hourly_steps: Number of hourly candles in the simulation.

    Returns:
        Annualised return as a fraction (e.g. ``0.20`` = 20 % p.a.).
        Returns ``0.0`` when initial equity is zero or n_hourly_steps < 1.
    """
    if initial <= 0.0 or n_hourly_steps < 1:
        return 0.0
    total_return = final / initial
    years = n_hourly_steps / _HOURS_PER_YEAR
    if years <= 0.0:
        return 0.0
    try:
        ann = math.pow(total_return, 1.0 / years) - 1.0
    except (ValueError, ZeroDivisionError):
        ann = 0.0
    return ann


def _compute_sharpe(pnl_pcts: list[float]) -> Optional[float]:
    """Compute annualised Sharpe ratio from per-trade PnL percentages.

    Uses the trade PnL as the return series.  Assumes all trades are
    1-hour candle units for annualisation.

    Args:
        pnl_pcts: List of per-trade PnL percentages.

    Returns:
        Annualised Sharpe ratio, or ``None`` if std deviation is zero or
        fewer than 2 data points.
    """
    if len(pnl_pcts) < _MIN_TRADES_FOR_RATIO:
        return None
    import statistics
    mean_ret = statistics.mean(pnl_pcts)
    try:
        std_ret = statistics.stdev(pnl_pcts)
    except statistics.StatisticsError:
        return None
    if std_ret == 0.0:
        return None
    # Annualise: multiply by sqrt(8760) for hourly data
    sharpe = (mean_ret / std_ret) * math.sqrt(_HOURS_PER_YEAR)
    return sharpe


def _compute_regime_metrics(
    regime_label: str,
    trades: list[TradeRecord],
) -> RegimeMetrics:
    """Compute per-regime metrics from a subset of trades.

    Args:
        regime_label: Regime string identifier.
        trades: All trades entered while in this regime.

    Returns:
        :class:`RegimeMetrics` with per-regime statistics.
    """
    n = len(trades)
    if n == 0:
        return RegimeMetrics(
            regime=regime_label,
            n_trades=0,
            win_rate=0.0,
            profit_factor=None,
            sharpe_ratio=None,
            max_drawdown=0.0,
            avg_trade_duration_hours=0.0,
            total_pnl=0.0,
        )

    wins = sum(1 for t in trades if t.is_winning)
    win_rate = wins / n

    gross_profit = sum(t.pnl for t in trades if t.pnl > 0.0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0.0))
    profit_factor: Optional[float] = (
        gross_profit / gross_loss if gross_loss > 0.0 else None
    )

    sharpe: Optional[float] = None
    if n >= _MIN_TRADES_FOR_RATIO:
        pnl_pcts = [t.pnl_pct for t in trades]
        sharpe = _compute_sharpe(pnl_pcts)

    # Drawdown from equity_after sequence
    eq_seq = [trades[0].equity_before] + [tr.equity_after for tr in trades]
    max_dd = _compute_max_drawdown(eq_seq)

    avg_dur = sum(t.duration_hours for t in trades) / n
    total_pnl = sum(t.pnl for t in trades)

    return RegimeMetrics(
        regime=regime_label,
        n_trades=n,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        avg_trade_duration_hours=avg_dur,
        total_pnl=total_pnl,
    )


__all__ = [
    "compute_metrics",
    "check_acceptance_gates",
    "MetricsResult",
    "RegimeMetrics",
]
