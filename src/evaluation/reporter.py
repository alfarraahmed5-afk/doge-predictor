"""Backtesting report generator.

Produces a structured report dict from a :class:`~src.evaluation.backtest.BacktestResult`
including per-regime breakdowns, buy-and-hold comparison, and equity curve data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.evaluation.backtest import BacktestResult
from src.evaluation.metrics import MetricsResult, RegimeMetrics, compute_metrics

# ---------------------------------------------------------------------------
# Reporter class
# ---------------------------------------------------------------------------


class BacktestReporter:
    """Generate human-readable and machine-readable backtest reports.

    Args:
        result: Completed :class:`~src.evaluation.backtest.BacktestResult`.

    Example::

        engine = BacktestEngine()
        result = engine.run(signals, prices, regimes)
        reporter = BacktestReporter(result)
        report = reporter.generate_report(prices)
    """

    def __init__(self, result: BacktestResult) -> None:
        """Initialise the reporter.

        Args:
            result: Completed backtest result.
        """
        self._result: BacktestResult = result
        self._metrics: MetricsResult = compute_metrics(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self, prices: pd.DataFrame) -> dict[str, Any]:
        """Generate the full report dict.

        The report contains:

        * ``"summary"`` — top-level scalar metrics (Sharpe, drawdown, etc.)
        * ``"per_regime"`` — per-regime metric breakdown table (list of dicts)
        * ``"buy_and_hold"`` — comparison vs holding DOGE for the full period
        * ``"equity_curve"`` — list of ``{"time_ms": …, "equity": …}`` points
        * ``"config"`` — config snapshot used for the run
        * ``"halt_reason"`` — non-empty string if simulation stopped early

        Args:
            prices: DataFrame with ``["open_time", "open", "close"]`` used
                to compute the buy-and-hold benchmark.  Must cover the same
                period as the signals used to produce *result*.

        Returns:
            Nested report dict suitable for JSON serialisation.
        """
        logger.info("Generating backtest report")

        summary = self._build_summary()
        per_regime_table = self._build_regime_table()
        bah = self._compute_buy_and_hold(prices)
        equity_curve_points = self._build_equity_curve_points()

        report: dict[str, Any] = {
            "summary": summary,
            "per_regime": per_regime_table,
            "buy_and_hold": bah,
            "equity_curve": equity_curve_points,
            "config": self._result.config_snapshot,
            "halt_reason": self._result.halt_reason,
        }

        logger.info(
            "Report generated: {} trades, {} regimes, halt='{}'",
            self._metrics.total_trades,
            len(self._metrics.per_regime),
            self._result.halt_reason or "none",
        )

        return report

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict[str, Any]:
        """Build top-level summary metrics dict.

        Returns:
            Dict with all scalar metrics from :class:`~src.evaluation.metrics.MetricsResult`.
        """
        m = self._metrics
        return {
            "directional_accuracy": m.directional_accuracy,
            "sharpe_ratio": m.sharpe_ratio,
            "max_drawdown": m.max_drawdown,
            "calmar_ratio": m.calmar_ratio,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "total_trades": m.total_trades,
            "avg_trade_duration_hours": m.avg_trade_duration_hours,
            "annualised_return": m.annualised_return,
            "initial_equity": self._result.initial_equity,
            "final_equity": self._result.final_equity,
            "n_signals": self._result.n_signals,
        }

    def _build_regime_table(self) -> list[dict[str, Any]]:
        """Build the per-regime breakdown table.

        Returns:
            List of dicts, one per regime with all :class:`~src.evaluation.metrics.RegimeMetrics`
            fields serialised to JSON-compatible types.
        """
        rows: list[dict[str, Any]] = []
        for regime_label, rm in self._metrics.per_regime.items():
            rows.append(
                {
                    "regime": regime_label,
                    "n_trades": rm.n_trades,
                    "win_rate": rm.win_rate,
                    "profit_factor": rm.profit_factor,
                    "sharpe_ratio": rm.sharpe_ratio,
                    "max_drawdown": rm.max_drawdown,
                    "avg_trade_duration_hours": rm.avg_trade_duration_hours,
                    "total_pnl": rm.total_pnl,
                }
            )
        return rows

    def _compute_buy_and_hold(
        self, prices: pd.DataFrame
    ) -> dict[str, Any]:
        """Compute buy-and-hold benchmark for the test period.

        Assumes DOGE is purchased at the first available open price and held
        until the last available close price in *prices*.

        Args:
            prices: Price DataFrame with ``["open_time", "open", "close"]``.

        Returns:
            Dict with ``"start_price"``, ``"end_price"``, ``"return_pct"``,
            ``"annualised_return"``, ``"n_candles"`` fields.
        """
        import math

        if prices.empty or "open" not in prices.columns or "close" not in prices.columns:
            return {
                "start_price": None,
                "end_price": None,
                "return_pct": None,
                "annualised_return": None,
                "n_candles": 0,
            }

        prices_sorted = prices.sort_values("open_time")
        start_price: float = float(prices_sorted["open"].iloc[0])
        end_price: float = float(prices_sorted["close"].iloc[-1])
        n_candles: int = len(prices_sorted)

        total_return_pct: float = (end_price - start_price) / start_price if start_price > 0.0 else 0.0

        # Annualise
        years = n_candles / 8_760.0
        try:
            ann = math.pow(1.0 + total_return_pct, 1.0 / years) - 1.0 if years > 0.0 else 0.0
        except (ValueError, ZeroDivisionError):
            ann = 0.0

        return {
            "start_price": start_price,
            "end_price": end_price,
            "return_pct": total_return_pct,
            "annualised_return": ann,
            "n_candles": n_candles,
        }

    def _build_equity_curve_points(self) -> list[dict[str, Any]]:
        """Convert the equity curve dict to a list of time/equity dicts.

        Returns:
            List of ``{"time_ms": int, "equity": float}`` dicts sorted by time.
        """
        return [
            {"time_ms": t, "equity": eq}
            for t, eq in sorted(self._result.equity_curve.items())
        ]


__all__ = ["BacktestReporter"]
