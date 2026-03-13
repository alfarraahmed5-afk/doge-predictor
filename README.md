# doge_predictor

DOGECOIN (DOGEUSDT) price prediction algorithm with a self-teaching reinforcement-learning loop.
Connects to Binance Spot and USD-M Futures REST/WebSocket APIs; runs a full 12-step inference
pipeline on every closed 1h candle; continuously improves via verified prediction outcomes.

> **Architecture reference**: `CLAUDE.md` — read at the start of every session.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Configuration](#2-configuration)
3. [Docker Deployment](#3-docker-deployment)
4. [Operations](#4-operations)
5. [Monitoring](#5-monitoring)
6. [Shadow Mode](#6-shadow-mode)
7. [Quality Gates](#7-quality-gates)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 or higher (3.13.x tested) |
| Docker Desktop | 4.x (for full stack) |
| Binance API key | Spot read-only is sufficient |

### Local (no Docker)

```bash
# 1. Clone and enter the repo
git clone https://github.com/alfarraahmed5-afk/doge-predictor.git
cd doge-predictor

# 2. Create virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 3. Copy the secrets template and add your Binance API keys
cp config/secrets.env.example config/secrets.env    # edit API_KEY and API_SECRET

# 4. Bootstrap historical OHLCV + funding data (takes ~3 min on first run)
python scripts/run_bootstrap_sqlite.py --db-path data/doge_data.db

# 5. Label regimes
python scripts/label_regimes.py --db-path data/doge_data.db

# 6. Train models (fast in-memory test — ~2 min)
python scripts/train.py --in-memory-test --no-hyperopt --output-dir models/

# 7. Start the inference server (SQLite mode)
python scripts/serve.py --models-dir models/ --db-path data/doge_data.db
```

The server exposes:
- `http://localhost:8000/health` — HTTP 200 when healthy, 503 when degraded
- `http://localhost:8001/metrics` — Prometheus metrics endpoint

### Run tests

```bash
pytest tests/ -q --cov=src --cov-fail-under=80
```

Expected output: **996 passed, 2 skipped, ~82% coverage**.

---

## 2. Configuration

All configuration lives in `config/`. Files are loaded **once at startup** by
`src/config.py` via Pydantic Settings models. Never hard-code values in `src/`.

### File overview

| File | Purpose |
|---|---|
| `config/settings.yaml` | Global: database, MLflow, paths, logging |
| `config/doge_settings.yaml` | DOGE-specific: symbols, intervals, risk, indicators |
| `config/regime_config.yaml` | Regime thresholds and per-regime confidence levels |
| `config/rl_config.yaml` | RL self-teaching: horizons, decay, replay buffer, curriculum |
| `config/secrets.env` | API keys — **NEVER commit this file** |

### Key settings explained

#### `config/settings.yaml`

```yaml
database:
  host: localhost      # TimescaleDB / PostgreSQL host
  port: 5432
  name: doge_predictor
  user: postgres
  password: ""         # Set via DB_PASSWORD env var in production

mlflow:
  tracking_uri: "sqlite:///mlruns/mlflow.db"   # Switch to postgresql://... for production
  experiment_name: doge_predictor
```

#### `config/doge_settings.yaml`

```yaml
# Training window — only post-2022 data enters model training folds
training_start_date: "2022-01-01"
context_start_date:  "2019-07-01"   # Full history for context features only

walk_forward:
  training_window_days:   180   # How many days per training fold
  validation_window_days:  30   # Hold-out window per fold
  step_size_days:           7   # How far each fold advances

risk:
  base_risk_pct: 0.01           # 1% of equity per trade (normal regimes)
  reduced_risk_pct: 0.005       # 0.5% in RANGING_LOW_VOL or DECOUPLED

# Hard overrides — no exceptions
btc_crash_threshold: -0.04      # Suppress all BUY if BTC drops > 4% in one hour
funding_rate_extreme_long: 0.001  # Suppress all BUY if funding > 0.001 per 8h

monitoring:
  regime_max_decoupled_hours: 72     # CRITICAL alert if DECOUPLED > 72 h straight
  regime_oscillation_max_transitions: 6  # WARNING if > 6 regime changes in 24 h
```

#### `config/regime_config.yaml`

```yaml
# Confidence threshold — signals with ensemble_prob below this are held
confidence_thresholds:
  TRENDING_BULL:    0.62
  TRENDING_BEAR:    0.62
  RANGING_HIGH_VOL: 0.65
  RANGING_LOW_VOL:  0.70
  DECOUPLED:        0.72   # Highest — most conservative
```

#### `config/rl_config.yaml`

```yaml
horizons:
  SHORT:  { candles: 4,   reward_weight: 1.0, punish_weight: 1.5 }
  MEDIUM: { candles: 24,  reward_weight: 1.5, punish_weight: 2.0 }
  LONG:   { candles: 168, reward_weight: 2.0, punish_weight: 2.5 }
  MACRO:  { candles: 720, reward_weight: 1.5, punish_weight: 1.0 }

curriculum:
  stage1_accuracy_threshold: 0.54
  stage1_mean_reward_threshold: 0.30
  stage1_min_days: 14

replay_buffer:
  capacity_per_horizon: 10000
  min_samples_to_train: 500
  priority_threshold: 0.5      # abs_reward >= this → 3x sampling weight
```

#### `config/secrets.env`

```env
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
DB_PASSWORD=your_db_password_here
```

---

## 3. Docker Deployment

### Prerequisites

- Docker Desktop 4.x installed and running
- `config/secrets.env` populated with real API keys

### Full stack

```bash
# Build and start all 4 services
docker compose up -d

# Check service health
docker compose ps
curl http://localhost:8000/health

# View logs
docker compose logs -f app

# Stop everything
docker compose down
```

### Services

| Service | Port | Purpose |
|---|---|---|
| `app` | 8000 (health), 8001 (metrics) | Inference server |
| `timescaledb` | 5432 | Time-series database (PostgreSQL + TimescaleDB) |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards (login: `admin` / `grafana_password_change_me`) |

### Environment variables (docker-compose)

```bash
# Enable shadow mode (safe validation before going live)
SHADOW_MODE=true docker compose up -d

# Override model version tag
MODEL_VERSION=v1.2 docker compose up -d
```

### First-time database setup

```bash
# Apply TimescaleDB schema (run once after timescaledb is healthy)
docker compose exec timescaledb psql -U postgres -d doge_predictor \
  -f /docker-entrypoint-initdb.d/create_tables.sql
```

---

## 4. Operations

### Trigger a manual retrain

```bash
# Full retrain with hyperparameter search (takes 10–30 min)
python scripts/train.py --output-dir models/

# Fast retrain — skip hyperopt (takes ~3 min)
python scripts/train.py --no-hyperopt --output-dir models/

# In-memory synthetic test (CI / smoke check)
python scripts/train.py --in-memory-test --no-hyperopt --output-dir models/
```

The weekly automated retrain fires every **Sunday at 02:00 UTC** via APScheduler in `serve.py`.
Advancement requires the candidate model to have:
- `mean_val_accuracy` > current production accuracy
- Sharpe ratio >= 1.0
- Per-regime Sharpe >= 0.8

### Rollback to previous model

```bash
# Dry run — shows what would happen without making changes
python scripts/rollback.py --dry-run

# Live rollback — restores previous-production model artefacts
python scripts/rollback.py --models-dir models/ --health-url http://localhost:8000/health
```

Rollback procedure:
1. Locates the MLflow run tagged `previous-production`
2. Re-tags it as `production`
3. Demotes the current production run to `rollback-{timestamp}`
4. Downloads artefacts to `--models-dir`
5. Polls `/health` until HTTP 200 (or 30 s timeout)

### Check system status

```bash
# Health endpoint (HTTP 200 = healthy, 503 = degraded)
curl http://localhost:8000/health

# Prometheus metrics (text format)
curl http://localhost:8001/metrics

# Latest predictions in SQLite
sqlite3 data/doge_data.db \
  "SELECT open_time, signal, ensemble_prob, regime FROM doge_predictions ORDER BY open_time DESC LIMIT 10;"

# View alert log
tail -f logs/alerts.log
tail -f logs/critical_alerts.log
```

### Bootstrap historical data

```bash
# Full bootstrap (all symbols, all intervals) — takes ~3 min
python scripts/run_bootstrap_sqlite.py \
  --db-path data/doge_data.db \
  --symbols DOGEUSDT BTCUSDT DOGEBTC \
  --intervals 1h 4h 1d

# Incremental update only (last 3 candles per symbol)
# Runs automatically at :01 past each UTC hour via APScheduler in serve.py

# Verify bootstrap quality
python scripts/qg01_verify.py --db-path data/doge_data.db
```

---

## 5. Monitoring

### Accessing Grafana

1. Navigate to `http://localhost:3000`
2. Login: `admin` / `grafana_password_change_me`
3. The **DOGE Predictor** dashboard is pre-provisioned at startup

### Dashboard panels

| Panel | Metric | What to watch for |
|---|---|---|
| **Inference Latency** | p50 / p95 / p99 | p99 > 500 ms → performance issue |
| **Current Regime** | Active regime label | DECOUPLED (highest risk state) → review positions |
| **WS Connected** | WebSocket status | 0 → data feed lost; inference halts |
| **Signals/hr** | BUY / SELL / HOLD count | Long period of all HOLDs → model may have degraded |
| **Prediction Count by Horizon** | SHORT / MEDIUM / LONG / MACRO | Gaps indicate RL verifier issues |
| **DOGE-BTC 24h Corr** | Rolling correlation | < 0.30 → DECOUPLED regime likely |
| **Volume Ratio** | Current vs 20-period mean | > 3.0 → volume spike flag active |
| **Funding Rate Z-Score** | Deviation from 90-period mean | Extreme values trigger hard BUY suppression |
| **Feature Freshness** | Seconds since last candle | > 7200 s (2h) → stale data alert |
| **Equity Drawdown** | Current drawdown % | > 20% → deployment should be reviewed |
| **Inference Errors/hr** | Error counter by pipeline step | Any sustained errors → check `logs/app.log` |

### Prometheus alert examples

```yaml
# In a Prometheus alerting rule file:
- alert: DogePredictorWsDisconnected
  expr: doge_ws_connected == 0
  for: 5m
  labels:
    severity: critical

- alert: DogePredictorHighLatency
  expr: histogram_quantile(0.99, doge_inference_latency_seconds_bucket) > 0.5
  for: 15m
  labels:
    severity: warning

- alert: DogePredictorDecoupledRegime
  expr: doge_current_regime{regime="DECOUPLED"} == 1
  for: 1h
  labels:
    severity: warning
```

---

## 6. Shadow Mode

Shadow mode runs the full inference pipeline but suppresses signal emission.
Use it to validate the system safely before going live.

### Enable shadow mode

```bash
# Local
SHADOW_MODE=true python scripts/serve.py --models-dir models/ --db-path data/doge_data.db

# Docker Compose
SHADOW_MODE=true docker compose up -d
```

In shadow mode:
- Steps 1–11 run normally (feature computation, regime classification, inference, logging)
- Step 12 (`on_signal` callbacks) is **suppressed** — no downstream consumers notified
- All predictions are logged to `doge_predictions` (permanent audit trail)
- `/health` responds with `"shadow_mode": true` in the JSON body

### Minimum shadow mode duration

**Run for at least 48 hours** before going live. This generates the 100+ verified
predictions needed for QG-08 and surfaces any integration issues safely.

### Validate shadow mode (QG-08)

```bash
python scripts/qg08_verify.py --db-path data/doge_data.db
```

All 5 checks must pass:
1. ≥ 100 predictions logged
2. All `model_version` fields populated
3. Inference latency p99 < 500 ms
4. Directional accuracy on first 100 verified predictions > 50%
5. Inference error rate < 1%

### Go live

```bash
SHADOW_MODE=false python scripts/serve.py --models-dir models/ --db-path data/doge_data.db
# or simply omit SHADOW_MODE — it defaults to false
```

---

## 7. Quality Gates

Run these in order before promoting a model to production.

| Gate | Script | Minimum threshold |
|---|---|---|
| **QG-01** Data ingestion | `python scripts/qg01_verify.py --in-memory-test` | 22/22 checks |
| **QG-03** Feature pipeline | `python scripts/qg03_verify.py --in-memory-test` | 8/8 required checks |
| **QG-04** Regime classification | `python scripts/qg04_verify.py --in-memory-test` | All 5 regimes present |
| **QG-05** Model training | `python scripts/qg05_verify.py --in-memory-test` | OOS accuracy > 53%, ≥ 3 folds |
| **QG-06** Model loading | `python scripts/qg06_verify.py --in-memory-test` | XGB + LSTM + Ensemble load + infer |
| **QG-BT** Backtesting | `python scripts/qg_backtest_verify.py --in-memory-test` | All 9 Section 9 acceptance gates |
| **QG-07** Deployment stack | `python scripts/qg07_verify.py` | 19/21 checks (2 SKIP if no Docker) |
| **QG-08** Shadow mode | `python scripts/qg08_verify.py --in-memory-test` | 5/5 checks |
| **QG-09** Weekly retrain | `python scripts/qg09_verify.py --in-memory-test` | 5/5 checks |

### Section 9 acceptance gates (QG-BT)

| Metric | Minimum |
|---|---|
| Directional accuracy OOS | ≥ 54% |
| Sharpe ratio (annualised) | ≥ 1.0 |
| Per-regime Sharpe | ≥ 0.8 each |
| Max drawdown | ≤ 20% |
| Calmar ratio | ≥ 0.6 |
| Profit factor | ≥ 1.3 |
| Win rate | ≥ 45% |
| Trade count | ≥ 150 |
| DECOUPLED max drawdown | ≤ 15% |

---

## 8. Troubleshooting

### Issue 1: Health endpoint returns 503 immediately after startup

**Symptom**: `curl http://localhost:8000/health` returns 503 with `"degraded_reasons": ["ws_not_connected"]`.

**Cause**: The WebSocket connection to Binance hasn't established yet. It takes 5–15 s on startup.

**Fix**: Wait 30 seconds and retry. If it persists, check:
```bash
# View app logs for WS connection errors
docker compose logs app | grep -i websocket
tail -n 50 logs/app.log | grep -i ws
```

---

### Issue 2: `StaleDataError` — inference not firing

**Symptom**: Log line `StaleDataError: last close_time is X ms old (threshold: Y ms)`.

**Cause**: The latest candle in the database is older than `freshness_check_multiplier × interval_ms`
(default: 2 × 3,600,000 ms = 7,200 s = 2 hours). This fires when:
- WebSocket lost connection and data hasn't been refreshed
- Database write failed silently
- Market was halted (rare for DOGE)

**Fix**:
```bash
# Force a manual incremental update
python -c "
from src.ingestion.scheduler import IncrementalScheduler
from src.config import get_settings
from src.processing.storage import DogeStorage
s = get_settings()
storage = DogeStorage(s)
sched = IncrementalScheduler(storage)
sched.run_once()
"
```

---

### Issue 3: `funding_extreme_long` suppressing all BUY signals

**Symptom**: All signals are `HOLD`, logs show `risk_filter: funding_extreme_long=1`.

**Cause**: This is **correct behavior** — DOGEUSDT funding rate is above `0.001 per 8h`,
indicating crowded longs. This is a hard override defined in `CLAUDE.md §5 FACT 5`.

**Not a bug**. Monitor the funding rate:
```bash
sqlite3 data/doge_data.db \
  "SELECT timestamp_ms, funding_rate FROM funding_rates ORDER BY timestamp_ms DESC LIMIT 5;"
```
When funding drops below `0.001`, BUY signals will resume automatically.

---

### Issue 4: Model training produces fewer than 3 walk-forward folds

**Symptom**: `ValueError: WalkForwardCV generated only N folds (minimum 3)`.

**Cause**: The feature DataFrame doesn't cover enough time to fill `training_window_days` (180d)
plus `validation_window_days` (30d) plus `step_size_days` (7d) × 3 folds = ~591 days minimum.

**Fix**:
```bash
# Check row count and date range in your feature parquet
python -c "
import pandas as pd, glob
files = sorted(glob.glob('data/features/primary/features_*.parquet'))
if files:
    df = pd.read_parquet(files[-1])
    print(f'Rows: {len(df)}, era=training: {(df.era==\"training\").sum()}')
    print(f'Date range: {df.open_time.min()} to {df.open_time.max()}')
"
# If data is insufficient, re-run bootstrap from 2022-01-01
python scripts/run_bootstrap_sqlite.py --db-path data/doge_data.db --start 2022-01-01
```

---

### Issue 5: `PermissionError` on Windows when running QG scripts

**Symptom**: `PermissionError: [WinError 32] The process cannot access the file because it is
being used by another process` when QG scripts clean up temporary MLflow SQLite files.

**Cause**: MLflow keeps a WAL file lock on the SQLite database even after the run ends.
This is a known Windows + SQLite concurrency issue.

**Impact**: Cosmetic only — the QG checks all passed before the error was raised.
The exit code reflects the QG result, not the cleanup failure.

**Fix**: All QG scripts now use `TemporaryDirectory(ignore_cleanup_errors=True)` (Python 3.10+).
If you see this on an older Python version, upgrade to 3.11+.

---

## Project Structure

```
doge_predictor/
├── config/                  ← All YAML config + secrets template
├── data/
│   └── doge_data.db         ← SQLite (227k OHLCV rows + 5.9k funding rows)
├── src/
│   ├── ingestion/           ← Binance REST, Futures, WebSocket, bootstrap, scheduler
│   ├── processing/          ← Cleaner, validator, aligner, storage
│   ├── regimes/             ← 5-regime classifier, change detector, feature encoder
│   ├── features/            ← All 84 feature columns + full pipeline
│   ├── models/              ← XGBoost, LSTM, Ensemble, RegimeRouter
│   ├── training/            ← Walk-forward CV, FoldScaler, Trainer, RegimeTrainer
│   ├── evaluation/          ← BacktestEngine, metrics, reporter
│   ├── inference/           ← 12-step InferenceEngine, SignalEvent
│   ├── monitoring/          ← AlertManager, HealthCheck, RegimeMonitor, DriftDetector
│   └── rl/                  ← ReplayBuffer, Verifier, CurriculumManager, compute_reward
├── scripts/                 ← Bootstrap, train, serve, rollback, QG scripts
├── tests/                   ← 996 unit + integration tests
├── grafana/                 ← Auto-provisioned Prometheus datasource + 11-panel dashboard
├── Dockerfile               ← Multi-stage Linux build with TA-Lib
├── docker-compose.yml       ← 4-service stack: app + TimescaleDB + Prometheus + Grafana
└── CLAUDE.md                ← Master specification — read before every coding session
```

---

*Last updated: 2026-03-13 — Phase 8 complete. 996 tests pass. All quality gates green.*
