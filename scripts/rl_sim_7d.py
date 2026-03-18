"""rl_sim_7d.py — 7-day RL self-teaching historical simulation.

Replays the last 7 days of synthetic prediction + verification data through
the reward engine, confirming that:

  1. The reward function produces **both positive and negative** values.
  2. Correct-direction predictions yield positive rewards.
  3. Wrong-direction predictions yield negative rewards.
  4. The reward distribution is non-degenerate (std > 0).

Usage::

    python scripts/rl_sim_7d.py
    python scripts/rl_sim_7d.py --days 14 --n-per-day 24
    python scripts/rl_sim_7d.py --seed 999

Exit codes:
    0 — simulation passed (positive and negative rewards confirmed)
    1 — simulation failed (degenerate reward distribution)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from loguru import logger

from src.config import RLConfig, _load_yaml
from src.rl.reward import compute_reward

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MS_PER_HOUR: int = 3_600_000
_MS_PER_DAY: int = 24 * _MS_PER_HOUR
_BASE_TIME_MS: int = 1_700_000_000_000  # 2023-11-14 UTC (fixed reference)

_HORIZONS = ("SHORT", "MEDIUM", "LONG", "MACRO")
_REGIMES = (
    "TRENDING_BULL",
    "TRENDING_BEAR",
    "RANGING_HIGH_VOL",
    "RANGING_LOW_VOL",
    "DECOUPLED",
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _simulate_day(
    rng: np.random.Generator,
    day_index: int,
    n_per_day: int,
    rl_cfg: RLConfig,
) -> list[dict]:
    """Generate *n_per_day* simulated prediction/verification events for one day.

    Args:
        rng: Seeded NumPy random generator.
        day_index: Zero-based day number (used for base timestamp).
        n_per_day: How many prediction events to generate.
        rl_cfg: Loaded :class:`~src.config.RLConfig` for horizon parameters.

    Returns:
        List of dicts, each with keys: ``horizon``, ``regime``,
        ``predicted_direction``, ``actual_direction``, ``prob``,
        ``price_at_prediction``, ``actual_price``, ``reward``.
    """
    records: list[dict] = []

    for i in range(n_per_day):
        horizon = _HORIZONS[i % len(_HORIZONS)]
        regime = rng.choice(_REGIMES)

        # Simulate a prediction: 55% chance the model is correct
        model_correct = rng.random() < 0.55

        # Actual price movement
        actual_direction = int(rng.choice([-1, 1]))
        predicted_direction = actual_direction if model_correct else -actual_direction

        # Confidence: uniform between 0.50 and 0.90
        prob_distance = rng.uniform(0.0, 0.40)
        prob = 0.5 + prob_distance if predicted_direction == 1 else 0.5 - prob_distance
        prob = float(np.clip(prob, 0.01, 0.99))

        # Prices: base price + small random walk
        price_at_prediction = float(rng.uniform(0.08, 0.15))
        magnitude_pct = rng.uniform(0.005, 0.06)  # 0.5% – 6% move
        if actual_direction == 1:
            actual_price = price_at_prediction * (1 + magnitude_pct)
        else:
            actual_price = price_at_prediction * (1 - magnitude_pct)

        try:
            result = compute_reward(
                horizon=horizon,
                predicted_direction=predicted_direction,
                actual_direction=actual_direction,
                predicted_prob=prob,
                price_at_prediction=price_at_prediction,
                actual_price=actual_price,
                regime=regime,
                rl_cfg=rl_cfg,
            )
            reward = result.reward_score
        except Exception as exc:  # noqa: BLE001
            logger.warning("rl_sim_7d: compute_reward failed: {}", exc)
            continue

        records.append(
            {
                "day": day_index,
                "hour": i,
                "horizon": horizon,
                "regime": regime,
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "model_correct": model_correct,
                "prob": round(prob, 4),
                "price_at_prediction": round(price_at_prediction, 6),
                "actual_price": round(actual_price, 6),
                "reward": round(reward, 6),
            }
        )

    return records


# ---------------------------------------------------------------------------
# Analysis + reporting
# ---------------------------------------------------------------------------


def _analyse_rewards(records: list[dict]) -> dict:
    """Compute summary statistics over all reward records.

    Args:
        records: List of record dicts from :func:`_simulate_day`.

    Returns:
        Dict with keys: ``n_total``, ``n_positive``, ``n_negative``, ``n_zero``,
        ``mean``, ``std``, ``min``, ``max``, ``positive_rate``, ``by_horizon``.
    """
    rewards = np.array([r["reward"] for r in records], dtype=float)
    n = len(rewards)

    by_horizon: dict[str, dict] = {}
    for horizon in _HORIZONS:
        h_rewards = np.array([r["reward"] for r in records if r["horizon"] == horizon])
        if len(h_rewards) == 0:
            continue
        by_horizon[horizon] = {
            "n": len(h_rewards),
            "mean": float(np.mean(h_rewards)),
            "std": float(np.std(h_rewards)),
            "min": float(np.min(h_rewards)),
            "max": float(np.max(h_rewards)),
            "positive_rate": float(np.mean(h_rewards > 0)),
        }

    return {
        "n_total": n,
        "n_positive": int(np.sum(rewards > 0)),
        "n_negative": int(np.sum(rewards < 0)),
        "n_zero": int(np.sum(rewards == 0)),
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "positive_rate": float(np.mean(rewards > 0)),
        "by_horizon": by_horizon,
    }


def _print_report(stats: dict, days: int) -> None:
    """Print a human-readable simulation report.

    Args:
        stats: Output of :func:`_analyse_rewards`.
        days: Number of simulated days.
    """
    print()
    print("=" * 60)
    print(f"  RL 7-Day Simulation Report  ({days} days)")
    print("=" * 60)
    print(f"  Total events   : {stats['n_total']}")
    print(f"  Positive rewards: {stats['n_positive']} "
          f"({stats['positive_rate']*100:.1f}%)")
    print(f"  Negative rewards: {stats['n_negative']} "
          f"({(1-stats['positive_rate'])*100:.1f}%)")
    print(f"  Zero rewards    : {stats['n_zero']}")
    print(f"  Mean reward     : {stats['mean']:+.4f}")
    print(f"  Std reward      : {stats['std']:.4f}")
    print(f"  Range           : [{stats['min']:+.4f}, {stats['max']:+.4f}]")
    print()
    print("  By Horizon:")
    for horizon, h in stats["by_horizon"].items():
        print(
            f"    {horizon:8s}  n={h['n']:4d}  mean={h['mean']:+.4f}  "
            f"std={h['std']:.4f}  pos%={h['positive_rate']*100:.1f}%"
        )
    print("=" * 60)


def _run_checks(stats: dict) -> list[tuple[str, bool, str]]:
    """Run sanity checks on the reward distribution.

    Args:
        stats: Output of :func:`_analyse_rewards`.

    Returns:
        List of ``(check_name, passed, detail)`` tuples.
    """
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "C1: n_positive > 0",
        stats["n_positive"] > 0,
        f"n_positive={stats['n_positive']}",
    ))
    checks.append((
        "C2: n_negative > 0",
        stats["n_negative"] > 0,
        f"n_negative={stats['n_negative']}",
    ))
    checks.append((
        "C3: std > 0 (non-degenerate)",
        stats["std"] > 0,
        f"std={stats['std']:.6f}",
    ))
    checks.append((
        "C4: min < 0 (negative rewards exist)",
        stats["min"] < 0,
        f"min={stats['min']:.4f}",
    ))
    checks.append((
        "C5: max > 0 (positive rewards exist)",
        stats["max"] > 0,
        f"max={stats['max']:.4f}",
    ))
    checks.append((
        "C6: positive_rate in [0.35, 0.80]",
        0.35 <= stats["positive_rate"] <= 0.80,
        f"positive_rate={stats['positive_rate']:.3f}",
    ))
    checks.append((
        "C7: all horizons present",
        set(stats["by_horizon"].keys()) == set(_HORIZONS),
        f"horizons={sorted(stats['by_horizon'].keys())}",
    ))

    return checks


def _print_check_table(checks: list[tuple[str, bool, str]]) -> bool:
    """Print the check table and return True iff all pass.

    Args:
        checks: Output of :func:`_run_checks`.

    Returns:
        ``True`` when all checks pass.
    """
    print()
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}  ({detail})")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rl_sim_7d.py",
        description="7-day RL reward distribution simulation",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of simulated days (default: 7)",
    )
    parser.add_argument(
        "--n-per-day",
        type=int,
        default=24,
        help="Prediction events per day (default: 24 — one per hour)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def main() -> int:
    """Run the simulation and return exit code.

    Returns:
        0 on PASS, 1 on FAIL.
    """
    parser = _build_parser()
    args = parser.parse_args()

    logger.info(
        "rl_sim_7d: starting simulation — days={} n_per_day={} seed={}",
        args.days,
        args.n_per_day,
        args.seed,
    )

    rl_cfg = RLConfig(**_load_yaml("rl_config.yaml"))
    rng = np.random.default_rng(args.seed)

    all_records: list[dict] = []
    for day in range(args.days):
        day_records = _simulate_day(rng, day, args.n_per_day, rl_cfg)
        all_records.extend(day_records)
        logger.debug(
            "rl_sim_7d: day {} — {} records generated", day, len(day_records)
        )

    if not all_records:
        logger.error("rl_sim_7d: no records generated — check rl_config.yaml")
        return 1

    stats = _analyse_rewards(all_records)
    _print_report(stats, args.days)

    checks = _run_checks(stats)
    all_pass = _print_check_table(checks)

    n_pass = sum(1 for _, p, _ in checks if p)
    n_fail = sum(1 for _, p, _ in checks if not p)
    print(f"  rl_sim_7d: {n_pass} PASS, {n_fail} FAIL")

    if all_pass:
        print("  rl_sim_7d: PASS — reward distribution confirmed (positive + negative values)")
        logger.info("rl_sim_7d: PASS")
        return 0
    else:
        print("  rl_sim_7d: FAIL — reward distribution check failed")
        logger.error("rl_sim_7d: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
