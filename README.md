# doge_predictor

DOGECOIN (DOGEUSDT) price prediction algorithm with self-teaching RL loop.

See `CLAUDE.md` for the full project specification, architecture, and build status.

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Bootstrap historical data
python scripts/run_bootstrap_sqlite.py --db-path data/doge_data.db

# 4. Label regimes
python scripts/label_regimes.py

# 5. Build features
python scripts/qg03_verify.py --in-memory-test

# 6. Train models
python scripts/train.py --in-memory-test --no-hyperopt --output-dir models/

# 7. Start inference server
python scripts/serve.py --models-dir models --db-path data/doge_data.db
```

---

## Shadow Mode Procedure

Shadow mode runs the full inference pipeline — computing features, classifying
regimes, and generating predictions — but **does not emit signals** to any
downstream consumer.  This allows safe validation before going live.

### Enabling Shadow Mode

Set the `SHADOW_MODE` environment variable to `true`:

```bash
# Direct
SHADOW_MODE=true python scripts/serve.py --models-dir models --db-path data/doge_data.db

# Docker Compose
SHADOW_MODE=true docker compose up -d
```

When `SHADOW_MODE=true`:
- The inference engine runs the full 12-step pipeline (Steps 1–11).
- Step 12 signal emission is **suppressed** — `on_signal` callbacks are **not called**.
- All predictions are logged to `doge_predictions` as normal (permanent audit trail).
- All Prometheus metrics are updated (monitoring works normally).
- The `/health` endpoint reports `shadow_mode: true` in its JSON body.

### Shadow Mode Minimum Duration

**Run shadow mode for a minimum of 48 hours before going live.**

This ensures:
- At least 100 predictions are generated for QG-08 validation.
- Model latency and accuracy can be measured on real market data.
- Any integration issues (DB connectivity, WebSocket stability) surface safely.

### Validating Shadow Mode (QG-08)

After 48h of shadow mode operation, run:

```bash
python scripts/qg08_verify.py --db-path data/doge_data.db
```

QG-08 checks:
1. ≥ 100 predictions logged to `doge_predictions`
2. All predictions have `model_version` populated
3. Inference latency p99 < 500 ms (from `logs/app.log`)
4. Directional accuracy on first 100 verified predictions > 50%
5. Inference error rate < 1%

**All 5 checks must PASS before enabling live signal emission.**

### Disabling Shadow Mode (Going Live)

Once QG-08 passes:

```bash
# Remove or unset the environment variable
unset SHADOW_MODE
# or explicitly set to false
SHADOW_MODE=false python scripts/serve.py --models-dir models --db-path data/doge_data.db
```

---

## Docker Deployment

```bash
# Build image
docker build -t doge_predictor .

# Start full stack (app + TimescaleDB + Prometheus + Grafana)
docker compose up -d

# Check health
curl http://localhost:8000/health

# View Prometheus metrics
curl http://localhost:8001/metrics

# Grafana dashboard
open http://localhost:3000  # admin / grafana_password_change_me
```

---

## Running Tests

```bash
pytest tests/ -q --cov=src --cov-fail-under=80
```

---

## Quality Gates

| Gate | Script | What it checks |
|---|---|---|
| QG-01 | `scripts/qg01_verify.py` | Data ingestion: row counts, gaps, alignment |
| QG-03 | `scripts/qg03_verify.py` | Feature pipeline: zero NaN/Inf, mandatory features |
| QG-04 | `scripts/qg04_verify.py` | Regime classification: all 5 regimes present |
| QG-05 | `scripts/qg05_verify.py` | Model training: OOS accuracy > 53%, walk-forward |
| QG-06 | `scripts/qg06_verify.py` | Model loading + inference smoke test |
| QG-BT | `scripts/qg_backtest_verify.py` | Backtesting: all 9 Section 9 gates |
| QG-07 | `scripts/qg07_verify.py` | Phase 7: pytest suite, Prometheus metrics, Grafana |
| QG-08 | `scripts/qg08_verify.py` | Shadow mode: 100 predictions, latency, accuracy |
