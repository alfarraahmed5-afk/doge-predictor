"""QG-08 — Shadow Mode Verification Gate.

Validates 48h+ shadow mode run before going live.

Checks (run after >= 48h of shadow mode operation):
  1. At least 100 predictions logged to doge_predictions
  2. All predictions have model_version field populated
  3. Inference latency p99 < 500ms (read from Prometheus metrics log)
  4. Directional accuracy on first 100 verified predictions > 50%
  5. No inference errors > 1% of signal count

Run:
    python scripts/qg08_verify.py --db-path data/doge_data.db
    python scripts/qg08_verify.py --in-memory-test   # synthetic validation
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_OK = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"
_results: list[tuple[str, str, str]] = []


def _record(name: str, status: str, detail: str = "") -> None:
    _results.append((name, status, detail))
    icon = {"PASS": "[+]", "FAIL": "[!]", "SKIP": "[-]"}.get(status, "[ ]")
    print(f"  {icon} {name}: {status}{(' — ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# In-memory test data seeder
# ---------------------------------------------------------------------------


def _seed_in_memory_predictions(storage: "DogeStorage") -> int:
    """Seed 120 synthetic predictions (100 verified) into a SQLite storage."""
    from src.processing.schemas import PredictionRecord

    rng = np.random.default_rng(42)
    now_ms = int(time.time() * 1000)
    _1H = 3_600_000
    regimes = ["TRENDING_BULL", "TRENDING_BEAR", "RANGING_HIGH_VOL", "RANGING_LOW_VOL", "DECOUPLED"]
    horizons = [("SHORT", 4), ("MEDIUM", 24)]

    for i in range(120):
        open_time = now_ms - _1H * (120 - i)
        h_label, h_candles = horizons[i % 2]
        target = open_time + _1H * h_candles
        price = 0.10 + rng.normal(0, 0.005)
        pred_dir = int(rng.choice([-1, 1]))
        prob = float(rng.uniform(0.55, 0.90))

        rec = PredictionRecord(
            prediction_id=str(uuid.uuid4()),
            created_at=open_time,
            open_time=open_time,
            symbol="DOGEUSDT",
            horizon_label=h_label,
            horizon_candles=h_candles,
            target_open_time=target,
            price_at_prediction=max(price, 0.001),
            predicted_direction=pred_dir,
            confidence_score=max(0.5, min(1.0, prob)),
            lstm_prob=prob,
            xgb_prob=prob * 0.98,
            regime_label=regimes[i % len(regimes)],
            model_version="v1.0-shadow",
        )
        storage.insert_prediction(rec)

    # Verify the first 100 with synthetic outcomes — 60% accuracy (predictable)
    pending = storage.get_matured_unverified(now_ms + _1H * 200)
    verified_count = 0
    for idx, p in enumerate(pending[:100]):
        # Bias: first 60 predictions are "correct" (actual matches predicted)
        if idx < 60:
            actual_dir = p.predicted_direction if p.predicted_direction != 0 else 1
        else:
            actual_dir = -p.predicted_direction if p.predicted_direction != 0 else -1
        move = 0.002 * actual_dir
        actual_price = p.price_at_prediction * (1.0 + move + rng.normal(0, 0.001))
        error_pct = abs(actual_price - p.price_at_prediction) / p.price_at_prediction
        correct = p.predicted_direction == actual_dir
        storage.update_prediction_outcome(p.prediction_id, {
            "actual_price": actual_price,
            "actual_direction": actual_dir,
            "reward_score": 1.2 if correct else -1.0,
            "direction_correct": correct,
            "error_pct": error_pct,
            "verified_at": now_ms,
        })
        verified_count += 1

    return verified_count


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check1_prediction_count(storage: "DogeStorage") -> None:
    """Assert at least 100 predictions logged."""
    print("\nCheck 1: At least 100 predictions logged")
    try:
        now_ms = int(time.time() * 1000)
        # get_matured_unverified returns only unverified; query all via a wide window
        all_pending = storage.get_matured_unverified(now_ms + 365 * 24 * 3_600_000)
        # Also count already-verified ones by checking a large window
        # Proxy: fetch unverified + already-verified by pulling a combined count
        # We use the table directly via the engine
        import sqlalchemy as sa
        table = storage._metadata.tables["doge_predictions"]
        with storage._engine.connect() as conn:
            count = conn.execute(sa.select(sa.func.count()).select_from(table)).scalar()
        n = int(count or 0)
        if n >= 100:
            _record(">=100 predictions logged", _OK, f"n={n}")
        else:
            _record(">=100 predictions logged", _FAIL, f"only {n} predictions found")
    except Exception as exc:
        _record(">=100 predictions logged", _FAIL, str(exc)[:100])


def check2_model_version_populated(storage: "DogeStorage") -> None:
    """Assert all predictions have model_version field populated."""
    print("\nCheck 2: All predictions have model_version")
    try:
        import sqlalchemy as sa
        table = storage._metadata.tables["doge_predictions"]
        with storage._engine.connect() as conn:
            null_count = conn.execute(
                sa.select(sa.func.count()).select_from(table).where(
                    sa.or_(
                        table.c.model_version.is_(None),
                        table.c.model_version == "",
                    )
                )
            ).scalar()
        n = int(null_count or 0)
        if n == 0:
            _record("all predictions have model_version", _OK)
        else:
            _record("all predictions have model_version", _FAIL,
                    f"{n} predictions missing model_version")
    except Exception as exc:
        _record("all predictions have model_version", _FAIL, str(exc)[:100])


def check3_inference_latency() -> None:
    """Assert inference latency p99 < 500ms (uses log file if available)."""
    print("\nCheck 3: Inference latency p99 < 500ms")
    logs_dir = _PROJECT_ROOT / "logs"
    app_log = logs_dir / "app.log"
    if not app_log.exists():
        _record("inference latency p99 < 500ms", _SKIP,
                "logs/app.log not found — run in shadow mode first")
        return
    try:
        latencies = []
        with app_log.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if "inference_latency_ms" in line:
                    import json
                    try:
                        entry = json.loads(line)
                        lat = float(entry.get("inference_latency_ms", 0))
                        if lat > 0:
                            latencies.append(lat)
                    except (json.JSONDecodeError, ValueError):
                        pass

        if not latencies:
            _record("inference latency p99 < 500ms", _SKIP,
                    "no latency entries found in app.log")
            return

        p99 = float(np.percentile(latencies, 99))
        if p99 < 500.0:
            _record("inference latency p99 < 500ms", _OK, f"p99={p99:.1f}ms")
        else:
            _record("inference latency p99 < 500ms", _FAIL, f"p99={p99:.1f}ms >= 500ms")
    except Exception as exc:
        _record("inference latency p99 < 500ms", _FAIL, str(exc)[:100])


def check4_directional_accuracy(storage: "DogeStorage") -> None:
    """Assert accuracy on first 100 verified predictions > 50%."""
    print("\nCheck 4: Directional accuracy on first 100 verified predictions > 50%")
    try:
        import sqlalchemy as sa
        table = storage._metadata.tables["doge_predictions"]
        with storage._engine.connect() as conn:
            rows = conn.execute(
                sa.select(table.c.direction_correct)
                .where(table.c.verified_at.is_not(None))
                .order_by(table.c.created_at.asc())
                .limit(100)
            ).fetchall()

        if len(rows) < 100:
            _record("accuracy on 100 verified > 50%", _SKIP,
                    f"only {len(rows)} verified predictions found (need 100)")
            return

        n_correct = sum(1 for (v,) in rows if v)
        accuracy = n_correct / len(rows)
        if accuracy > 0.50:
            _record("accuracy on 100 verified > 50%", _OK,
                    f"{accuracy:.1%} ({n_correct}/{len(rows)})")
        else:
            _record("accuracy on 100 verified > 50%", _FAIL,
                    f"{accuracy:.1%} <= 50%")
    except Exception as exc:
        _record("accuracy on 100 verified > 50%", _FAIL, str(exc)[:100])


def check5_low_error_rate(storage: "DogeStorage") -> None:
    """Assert no inference errors > 1% of signal count."""
    print("\nCheck 5: Inference errors < 1% of prediction count")
    logs_dir = _PROJECT_ROOT / "logs"
    app_log = logs_dir / "app.log"

    if not app_log.exists():
        _record("inference errors < 1%", _SKIP, "no log file — run shadow mode first")
        return

    try:
        import json
        errors = total = 0
        with app_log.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                    msg = str(entry.get("message", ""))
                    if "inference" in msg.lower():
                        total += 1
                    if "ERROR" in str(entry.get("level", "")) and "inference" in msg.lower():
                        errors += 1
                except (json.JSONDecodeError, ValueError):
                    pass

        if total == 0:
            _record("inference errors < 1%", _SKIP, "no inference log entries")
            return

        rate = errors / total
        if rate < 0.01:
            _record("inference errors < 1%", _OK,
                    f"error_rate={rate:.2%} ({errors}/{total})")
        else:
            _record("inference errors < 1%", _FAIL,
                    f"error_rate={rate:.2%} > 1%")
    except Exception as exc:
        _record("inference errors < 1%", _FAIL, str(exc)[:100])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="QG-08 — Shadow mode quality gate")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Path to SQLite database (default: data/doge_data.db)")
    parser.add_argument("--in-memory-test", action="store_true",
                        help="Seed synthetic data for validation without a live DB")
    args = parser.parse_args()

    print("=" * 60)
    print("QG-08 — Shadow Mode Verification Gate")
    print("=" * 60)

    from src.config import Settings, _load_yaml
    from src.processing.storage import DogeStorage

    if args.in_memory_test:
        import sqlalchemy as sa
        engine = sa.create_engine("sqlite:///:memory:", echo=False)
        s = Settings(**_load_yaml("settings.yaml"))
        storage = DogeStorage(s, engine=engine)
        storage.create_tables()
        n_verified = _seed_in_memory_predictions(storage)
        print(f"  [+] Seeded in-memory: 120 predictions, {n_verified} verified")
    else:
        db_path = args.db_path or (_PROJECT_ROOT / "data" / "doge_data.db")
        engine = __import__("sqlalchemy").create_engine(f"sqlite:///{db_path}")
        s = Settings(**_load_yaml("settings.yaml"))
        storage = DogeStorage(s, engine=engine)

    check1_prediction_count(storage)
    check2_model_version_populated(storage)
    check3_inference_latency()
    check4_directional_accuracy(storage)
    check5_low_error_rate(storage)

    # Summary
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, s, _ in _results if s == _OK)
    n_fail = sum(1 for _, s, _ in _results if s == _FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == _SKIP)
    print(f"RESULT: {n_pass} PASS  {n_fail} FAIL  {n_skip} SKIP  "
          f"(out of {len(_results)} checks)")

    hard_fail = n_fail > 0
    if hard_fail:
        print("\nQG-08: FAIL")
        return 1

    print("\nQG-08: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
