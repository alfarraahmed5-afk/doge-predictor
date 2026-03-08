# CLAUDE.md — DOGE Prediction Algorithm
## Master Context File for AI Agent Coding Sessions

> **READ THIS ENTIRE FILE BEFORE WRITING ANY CODE.**
> This file is the single source of truth for every coding session.
> It defines what exists, what the standards are, and what to build next.
> Never deviate from the architecture or standards described here without explicit instruction.

---

## 1. PROJECT IDENTITY

| Field | Value |
|---|---|
| **Project Name** | `doge_predictor` |
| **Purpose** | DOGECOIN (DOGEUSDT) price prediction algorithm with self-teaching RL loop |
| **Primary Exchange** | Binance (Spot + USD-M Futures) |
| **Primary Trading Pair** | DOGEUSDT |
| **Primary Interval** | 1 hour candles |
| **Training Window** | January 1 2022 – present (post-mania normalization era) |
| **Language** | Python 3.11+ exclusively |
| **Reference Documents** | `docs/framework.docx`, `docs/devguide_v3.docx` |

---

## 2. CANONICAL DIRECTORY STRUCTURE

```
doge_predictor/
├── CLAUDE.md                          ← THIS FILE — read at every session start
├── config/
│   ├── settings.yaml                  ← global parameters
│   ├── doge_settings.yaml             ← DOGE-specific overrides
│   ├── regime_config.yaml             ← regime thresholds and routing
│   ├── rl_config.yaml                 ← RL self-teaching parameters
│   └── secrets.env                    ← API keys (NEVER commit)
├── data/
│   ├── raw/
│   │   ├── dogeusdt_1h/               ← PRIMARY: immutable, append-only
│   │   ├── dogeusdt_4h/
│   │   ├── dogeusdt_1d/
│   │   ├── dogebtc_1h/
│   │   ├── btcusdt_1h/
│   │   ├── funding_rates/
│   │   └── agg_trades/
│   ├── processed/
│   ├── features/
│   │   ├── primary/                   ← post-2022 training features (Parquet)
│   │   ├── context/                   ← pre-2022 context features only
│   │   └── live/                      ← rolling live feature buffer
│   ├── regimes/                       ← regime labels + transition log
│   ├── predictions/                   ← Prediction Store backups
│   ├── replay_buffers/                ← Replay Buffer checkpoints
│   └── checkpoints/                   ← bootstrap progress checkpoints
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── rest_client.py             ← Binance REST (rate limit + retry)
│   │   ├── futures_client.py          ← Binance Futures (funding rates)
│   │   ├── ws_client.py               ← WebSocket (live candle stream)
│   │   ├── bootstrap.py               ← historical backfill
│   │   ├── multi_symbol.py            ← parallel DOGE + BTC + DOGEBTC ingestion
│   │   └── scheduler.py               ← incremental update scheduler
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── validator.py               ← ALL validation checks run here
│   │   ├── aligner.py                 ← multi-symbol timestamp alignment
│   │   └── storage.py                 ← read/write abstraction
│   ├── regimes/
│   │   ├── __init__.py
│   │   ├── classifier.py              ← DogeRegimeClassifier (5 sub-regimes)
│   │   ├── detector.py                ← real-time regime change detection
│   │   └── features.py                ← regime-derived model features
│   ├── features/
│   │   ├── __init__.py
│   │   ├── price_indicators.py        ← SMA/EMA/MACD/RSI/BB/ATR/Stoch/Ichimoku
│   │   ├── volume_indicators.py       ← OBV/VWAP/CMF/CVD/volume ratios
│   │   ├── orderbook_features.py      ← bid-ask spread/imbalance/depth
│   │   ├── lag_features.py            ← log returns/momentum/rolling stats
│   │   ├── doge_specific.py           ← BTC corr/DOGE-BTC ratio/vol spike/round numbers
│   │   ├── funding_features.py        ← funding rate z-score/extreme flags
│   │   ├── htf_features.py            ← 4h and 1d derived features (with lookahead guard)
│   │   └── pipeline.py                ← orchestrates full feature computation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py              ← AbstractBaseModel all models must inherit
│   │   ├── lstm_model.py
│   │   ├── xgb_model.py
│   │   ├── transformer_model.py       ← optional, build last
│   │   ├── ensemble.py
│   │   └── regime_router.py           ← routes to regime-specific XGBoost weights
│   ├── training/
│   │   ├── __init__.py
│   │   ├── walk_forward.py            ← ONLY valid CV method — no sklearn split
│   │   ├── trainer.py
│   │   ├── regime_trainer.py          ← per-regime XGBoost training
│   │   ├── hyperopt.py                ← Optuna on train+val folds ONLY
│   │   └── scaler.py                  ← per-fold scaler (never fit on full dataset)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── backtest.py                ← next-candle fill, 0.1% fees, slippage
│   │   └── reporter.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py                  ← full inference pipeline with regime routing
│   │   └── signal.py                  ← signal assembly + thresholding
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py
│   │   ├── regime_monitor.py
│   │   ├── health_check.py
│   │   └── alerting.py
│   └── rl/
│       ├── __init__.py
│       ├── predictor.py               ← multi-horizon prediction generator
│       ├── verifier.py                ← matured prediction verification
│       ├── reward.py                  ← compute_reward() — core RL function
│       ├── replay_buffer.py           ← prioritised, regime-stratified buffer
│       ├── rl_trainer.py              ← self-training weight update loop
│       ├── curriculum.py              ← stage manager + advancement checks
│       └── rl_monitor.py              ← RL Prometheus metrics
├── tests/
│   ├── unit/
│   │   ├── test_rest_client.py
│   │   ├── test_validator.py
│   │   ├── test_aligner.py
│   │   ├── test_doge_features.py      ← lag sanity tests MANDATORY
│   │   ├── test_funding_features.py
│   │   ├── test_htf_features.py       ← HTF lookahead boundary tests MANDATORY
│   │   ├── test_regime_classifier.py
│   │   ├── test_walk_forward.py
│   │   ├── test_backtest.py
│   │   ├── test_reward.py             ← all 8 reward scenarios MANDATORY
│   │   ├── test_verifier.py
│   │   └── test_curriculum.py
│   ├── integration/
│   └── fixtures/
│       └── doge_sample_data/          ← realistic OHLCV fixture for all 5 regimes
├── scripts/
│   ├── bootstrap_doge.py              ← one-time historical backfill runner
│   ├── label_regimes.py               ← assigns regime labels to full history
│   ├── train.py                       ← training entry point
│   └── serve.py                       ← inference server entry point
├── mlruns/                            ← MLflow (auto-generated, do not edit)
├── notebooks/                         ← SCRATCH ONLY — never import from here
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## 3. NON-NEGOTIABLE CODING STANDARDS

These rules apply to **every file in `src/`** without exception.
Violating any of these is a bug, not a style choice.

### 3.1 Code Quality Rules

| Rule | Requirement |
|---|---|
| **Type hints** | ALL function signatures must have complete type hints. `def f(x):` is banned. |
| **Docstrings** | Every module, class, and public function needs a Google-style docstring. |
| **No magic numbers** | Every constant goes in `config/settings.yaml` or `config/doge_settings.yaml`. |
| **No print()** | Use `loguru` logger exclusively. `print()` is banned in all `src/` files. |
| **Error handling** | Every external call (API, DB, file I/O) wrapped in `try/except` with specific exception types. |
| **No mutable defaults** | Never `def f(x, data=[])`. Use `None` and assign inside body. |
| **Immutable raw data** | `data/raw/` is **append-only**. Never write to it after bootstrap. Raise `PermissionError` if attempted. |
| **Determinism** | Set seeds everywhere: `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`. Seed comes from config. |
| **No global state** | No module-level mutable variables. Pass state explicitly. |
| **DataFrame copies** | Always `df.copy()` when slicing to avoid `SettingWithCopyWarning`. |
| **Path handling** | `pathlib.Path` exclusively. No `os.path` or string concatenation for paths. |
| **Config** | Loaded once at startup via Pydantic `Settings` model. Never re-read mid-pipeline. |
| **Timestamps** | ALL timestamps are UTC epoch milliseconds (`int`). Never store tz-naive datetimes. |

### 3.2 The Three Most Critical Rules (Read Twice)

```
RULE A — NO LOOKAHEAD BIAS
  Features at time T must NEVER use data from T+1 or later.
  Target: df['target'] = df['close'].pct_change().shift(-N) > 0
  After writing any feature: explicitly verify it cannot see the future.

RULE B — SCALER ISOLATION
  StandardScaler is fitted ONLY on the training fold.
  It is then used (not re-fitted) to transform validation and test folds.
  Re-fitting at every walk-forward step is correct and mandatory.
  A scaler fitted on the full dataset is a data leak. This is a critical bug.

RULE C — NO RANDOM SPLITS ON TIME SERIES
  sklearn train_test_split is BANNED for this project.
  Only walk_forward.py is used for all CV. Assert after any split:
  max(train_timestamps) < min(val_timestamps)
```

---

## 4. DATA SOURCES & WHAT EXISTS

### 4.1 Binance Data Sources

| Source | Symbol | Interval | Date Range | Rows (est.) | Status |
|---|---|---|---|---|---|
| Spot OHLCV | DOGEUSDT | 1h | Jul 2019–now | ~42,000 | `[ ]` Not yet fetched |
| Spot OHLCV | BTCUSDT | 1h | Jul 2019–now | ~42,000 | `[ ]` Not yet fetched |
| Spot OHLCV | DOGEBTC | 1h | Jul 2019–now | ~42,000 | `[ ]` Not yet fetched |
| Spot OHLCV | DOGEUSDT | 4h | Jul 2019–now | ~10,500 | `[ ]` Not yet fetched |
| Spot OHLCV | DOGEUSDT | 1d | Jul 2019–now | ~1,750 | `[ ]` Not yet fetched |
| Futures Funding | DOGEUSDT | 8h | Oct 2020–now | ~5,400 | `[ ]` Not yet fetched |
| Agg Trades | DOGEUSDT | Tick | Last 30d | ~5M+ | `[ ]` Live only |

> **Update this table** as sources are bootstrapped. Change `[ ]` to `[x]`.

### 4.2 Training Data Rules

- **Primary training dataset**: DOGEUSDT 1h, **January 1 2022 – present only**
- Pre-2022 data is stored but **never** included in training folds
- Pre-2022 data is used only for: `ath_distance`, long-term trend context features
- Era column: `era='context'` for pre-2022, `era='training'` for post-2022
- These must be in separate Parquet partitions — **never mixed**

### 4.3 API Endpoints Quick Reference

```python
# Spot REST
GET /api/v3/klines?symbol=DOGEUSDT&interval=1h&limit=1000&startTime={ms}
GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&startTime={ms}
GET /api/v3/klines?symbol=DOGEBTC&interval=1h&limit=1000&startTime={ms}
GET /api/v3/depth?symbol=DOGEUSDT&limit=100
GET /api/v3/aggTrades?symbol=DOGEUSDT&limit=1000

# Futures REST
GET /fapi/v1/fundingRate?symbol=DOGEUSDT&limit=1000&startTime={ms}

# WebSocket
wss://stream.binance.com:9443/ws/dogeusdt@kline_1h
wss://stream.binance.com:9443/ws/dogeusdt@aggTrade
wss://stream.binance.com:9443/ws/btcusdt@kline_1h

# Rate limits
Spot REST:    1,200 weight/minute — read X-MBX-USED-WEIGHT-1M header every request
Futures REST: 2,400 weight/minute
WebSocket:    Max 5 msg/second incoming per connection
```

---

## 5. DOGE ASSET FACTS (Must Be Applied in Code)

```
FACT 1 — SENTIMENT PRIMACY
  DOGE is more sensitive to social sentiment than technical levels.
  Volume spikes > 3x rolling mean are the best available sentiment proxy.
  volume_spike_flag and volume_ratio are MANDATORY features.

FACT 2 — BTC BETA (~0.8–1.2)
  When BTC drops, DOGE drops harder.
  btc_1h_return and btc_volatility are MANDATORY model inputs.
  Emergency rule: if btc_1h_return < -4%, suppress all DOGE BUY signals.

FACT 3 — DECOUPLING
  When rolling 24h BTC-DOGE log-return correlation < 0.30 → DECOUPLED regime.
  This is the highest-risk state. Confidence threshold increases to 0.72.
  doge_btc_corr_24h is a MANDATORY feature.

FACT 4 — VOLUME LEADS PRICE
  Volume spike flag (> 3x 20-period mean) precedes directional moves within 4–12h.
  Compute on NORMALIZED volume (volume / rolling_mean), never raw volume.

FACT 5 — FUNDING RATE EXTREMES
  funding_rate > +0.001 per 8h → crowded longs → SUPPRESS ALL BUY SIGNALS (hard override)
  funding_rate < -0.0005 per 8h → crowded shorts → short squeeze potential
  funding_rate and funding_rate_zscore are MANDATORY features.

FACT 6 — ROUND NUMBER PSYCHOLOGY
  DOGE retail fixates on: $0.05, $0.10, $0.15, $0.20, $0.25, $0.30, $0.50, $1.00
  distance_to_round_pct and at_round_number_flag are MANDATORY features.
  at_round_number_flag == 1 → reduce position size by 30% on BUY signals.
```

---

## 6. THE FIVE MARKET REGIMES

The `DogeRegimeClassifier` must assign one of these labels to every candle:

| Regime | Definition | Confidence Threshold | Position Size |
|---|---|---|---|
| `TRENDING_BULL` | EMA20 > EMA50 > EMA200, 7d return > +5% | 0.62 | 100% |
| `TRENDING_BEAR` | EMA20 < EMA50 < EMA200, 7d return < -5% | 0.62 | 100% |
| `RANGING_HIGH_VOL` | BB width > 0.04, ATR > 0.5%, price between bands | 0.65 | 100% |
| `RANGING_LOW_VOL` | BB width < 0.04, ATR < 0.3% | 0.70 | **50%** |
| `DECOUPLED` | BTC-DOGE 24h corr < 0.30 — **overrides all others** | 0.72 | **50%** |

**Regime is computed BEFORE inference on every candle.**
**Confidence threshold is loaded from `regime_config.yaml` using the current regime key — never hardcoded.**

---

## 7. THE 12 MANDATORY DOGE-SPECIFIC FEATURES

These features are **non-negotiable**. A model trained without any of them is invalid.

| Feature | Module | Formula Summary |
|---|---|---|
| `doge_btc_corr_12h` | `doge_specific.py` | Rolling 12h log-return correlation with BTC |
| `doge_btc_corr_24h` | `doge_specific.py` | Rolling 24h log-return correlation with BTC |
| `doge_btc_corr_7d` | `doge_specific.py` | Rolling 168h log-return correlation with BTC |
| `dogebtc_mom_6h` | `doge_specific.py` | `log(dogebtc_close[t] / dogebtc_close[t-6])` |
| `dogebtc_mom_24h` | `doge_specific.py` | `log(dogebtc_close[t] / dogebtc_close[t-24])` |
| `dogebtc_mom_48h` | `doge_specific.py` | `log(dogebtc_close[t] / dogebtc_close[t-48])` |
| `volume_ratio` | `doge_specific.py` | `volume / volume.rolling(20).mean()` |
| `volume_spike_flag` | `doge_specific.py` | `(volume_ratio >= 3.0).astype(int)` |
| `funding_rate` | `funding_features.py` | 8h funding rate forward-filled to 1h |
| `funding_rate_zscore` | `funding_features.py` | 90-period rolling z-score of funding rate |
| `funding_extreme_long` | `funding_features.py` | `(funding_rate > 0.001).astype(int)` |
| `funding_extreme_short` | `funding_features.py` | `(funding_rate < -0.0005).astype(int)` |

**Additional mandatory features** (in `htf_features.py` and `doge_specific.py`):
`htf_4h_rsi`, `htf_4h_trend`, `htf_4h_bb_pctb`, `htf_1d_trend`, `htf_1d_return`,
`ath_distance`, `distance_to_round_pct`, `at_round_number_flag`, `nearest_round_level`

---

## 8. MODEL ARCHITECTURE SUMMARY

### Build Order (strict — do not deviate)
1. `base_model.py` — abstract interface all models must implement
2. `xgb_model.py` — build and validate first; must beat 53% directional accuracy before LSTM
3. `regime_trainer.py` — per-regime XGBoost (one model per sub-regime)
4. `lstm_model.py` — trained on all regimes; receives `regime_label_encoded` as input
5. `ensemble.py` — meta-learner on `[xgb_prob, lstm_prob, regime_encoded]`
6. `transformer_model.py` — build ONLY if LSTM Sharpe < 1.0

### BaseModel Interface (all models must implement)
```python
class BaseModel(ABC):
    def fit(self, X_train, y_train, X_val, y_val) -> dict: ...
    def predict_proba(self, X) -> np.ndarray: ...  # returns shape (n_samples,)
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

### LSTM Specification
- Input: `(batch, sequence_length=60, n_features)`
- Architecture: LSTM(128) → LSTM(64) → Dense(32) + BN + Dropout(0.3) → Sigmoid
- Optimizer: Adam(lr=1e-3) + ReduceLROnPlateau
- Gradient clipping: `max_norm=1.0` — **mandatory to prevent NaN loss**
- Always call `model.eval()` before inference, `model.train()` before training

### Walk-Forward CV Parameters (from `doge_settings.yaml`)
```yaml
training_window_days:   180
validation_window_days: 30
step_size_days:         7
min_training_rows:      3000
```

---

## 9. BACKTESTING RULES

| Rule | Specification |
|---|---|
| **Fill price** | Next candle's OPEN — never signal candle's close |
| **Taker fee** | 0.10% per leg (both entry and exit) |
| **Slippage** | Random uniform [0.02%, 0.08%] per trade |
| **Position sizing** | 1% of equity per trade; 0.5% in RANGING_LOW_VOL or DECOUPLED |
| **Drawdown halt** | Stop simulation at 25% drawdown |

### Minimum Acceptance Gates (all must pass before deployment)
| Metric | Minimum |
|---|---|
| Directional accuracy (OOS) | ≥ 54% |
| Sharpe ratio (annualized) | ≥ 1.0 (and ≥ 0.8 per regime) |
| Max drawdown | ≤ 20% |
| Calmar ratio | ≥ 0.6 |
| Profit factor | ≥ 1.3 |
| Win rate | ≥ 45% |
| Trade count | ≥ 150 |
| DECOUPLED regime max drawdown | ≤ 15% |

---

## 10. INFERENCE PIPELINE (Every Candle Close)

```
Step 1  Freshness check — last candle close_time must be within 2 × interval_ms of now
Step 2  Feature computation — full pipeline on last 500 closed candles
Step 3  Regime classification — DogeRegimeClassifier on current candle
Step 4  Feature validation — assert zero NaN/Inf, assert column list == feature_columns.json
Step 5  Scaling — load scaler.pkl saved with model (NEVER refit at inference)
Step 6  Base model inference — LSTM + regime XGBoost (log both probabilities)
Step 7  Load regime-adjusted threshold from regime_config.yaml
Step 8  Ensemble meta-learner — [lstm_prob, xgb_prob, regime_encoded]
Step 9  Risk filters:
          - funding_extreme_long == 1 → suppress BUY (hard override, no exceptions)
          - at_round_number_flag == 1 → reduce position size 30%
          - btc_1h_return < -4% → suppress BUY
          - regime == RANGING_LOW_VOL → reduce position size 50%
          - regime == DECOUPLED → threshold 0.72, position size 50%
Step 10 Signal decision: BUY / SELL / HOLD based on threshold
Step 11 Log full prediction record to doge_predictions table
Step 12 Emit signal
```

---

## 11. RL SELF-TEACHING SYSTEM

### Prediction Horizons

| Label | Candles | Target | Reward Weight | Punishment Weight |
|---|---|---|---|---|
| SHORT | 4 | T + 4h | 1.0x | 1.5x |
| MEDIUM | 24 | T + 24h | 1.5x | 2.0x |
| LONG | 168 | T + 7d | 2.0x | 2.5x |
| MACRO | 720 | T + 30d | 1.5x | 1.0x |

### Reward Function
```python
reward = direction_score × magnitude_score × abs(calibration_score)

# direction_score:   +1.0 (correct) | -1.0 (wrong) | +0.1 (flat/hedge)
# magnitude_score:   exp(-decay × error_pct)  — decay from rl_config.yaml
# calibration_score: +[1.0 to 2.0] if correct, -[1.0 to 3.0] if wrong
#                    High confidence + wrong = maximum punishment
```

### Prediction Store (TimescaleDB)
- Table: `doge_predictions`
- Prediction fields are **immutable after insert**
- Outcome fields (`actual_price`, `reward_score`, etc.) written only by Verifier
- **Never delete rows** — permanent audit trail

### Curriculum Stages
1. **Stage 1**: SHORT only → advance when 14d accuracy > 54% AND mean reward > 0.30
2. **Stage 2**: + MEDIUM → advance when 21d accuracy > 53%
3. **Stage 3**: + LONG → advance when 28d accuracy > 52%
4. **Stage 4**: + MACRO → final stage, no advancement

### Self-Training Triggers
- Scheduled: every Sunday 02:00 UTC
- Buffer 80% full
- Rolling 7d mean reward < 0.0
- 50+ new verified predictions in a new regime
- **Minimum 48h cooldown between runs**

---

## 12. KNOWN CRITICAL ERRORS (Check Every Module)

After writing any module, explicitly check for these before committing:

### Feature Engineering
- [ ] Does any feature at time T use `close[T+1]` or later? → **Lookahead bias — critical**
- [ ] Is BTC correlation computed on raw prices instead of log returns? → **Spurious correlation**
- [ ] Does the HTF (4h) feature use the current unclosed 4h candle? → **Lookahead bias**
- [ ] Is funding rate assigned to 1h candles without forward-fill? → **NaN injection**
- [ ] Is volume spike computed on raw volume (not rolling-mean-normalized)? → **Non-stationary**

### Training
- [ ] Is `train_test_split` used anywhere? → **Banned — use walk_forward only**
- [ ] Is the scaler fitted more than once per fold, or fitted on val/test data? → **Data leak**
- [ ] Are random seeds set? → **Non-reproducible results**
- [ ] Are class weights computed and applied? → **Imbalanced training**
- [ ] Is training on raw price levels instead of returns? → **Non-stationary input**

### Backtesting
- [ ] Is entry fill at signal candle's close? → **Lookahead in execution**
- [ ] Are fees (0.1% per leg) applied? → **Unrealistic results**
- [ ] Is the test set used for hyperparameter tuning? → **Invalidates all metrics**

### RL System
- [ ] Is `actual_direction` computed vs `price_at_prediction` (not T-1 close)? → **Wrong reward signal**
- [ ] Does the verifier run on candles before `close_time` is in the past? → **Lookahead**
- [ ] Are interpolated candle outcomes added to the Replay Buffer? → **Corrupt training data**
- [ ] Is the model version stored in every prediction record? → **Untraceable rewards**

---

## 13. BUILD STATUS

> **Agent: update this section at the end of every coding session.**
> Mark items complete with `[x]`. Add notes on any blockers.

### Phase 1 — Project Initialization
- [x] Directory structure created — all dirs + .gitkeep files, git init
- [x] `.gitignore` configured (secrets.env, .venv/, data/raw/, mlruns/, etc.)
- [x] Virtual environment created (Python 3.13.1 — satisfies >=3.11 requirement)
- [x] All dependencies installed from `requirements.txt` — ta-lib 0.6.8 (bundles C library), torch 2.10.0, all others installed
- [x] `ta-lib` C library confirmed at OS level — ta-lib>=0.4 bundles the C DLL on Windows; no separate OS install needed
- [x] All YAML config files created — settings.yaml, doge_settings.yaml, regime_config.yaml, rl_config.yaml, secrets.env (template only)
- [x] Pandera schema contracts written — RawOHLCVSchema, RawFundingRateSchema, ProcessedOHLCVSchema, AlignedSchema, FeatureSchema in src/processing/validator.py
- [x] Placeholder test files created — 12 unit tests + 1 integration test, all skipped; pytest collects all 12 and reports 12 skipped
- [x] `src/config.py` created — Pydantic Settings models for all 4 config files; loaded once at startup; global singletons exported
- [x] `src/processing/schemas.py` created — Pydantic v2 DTOs for all 6 pipeline record types (OHLCVRecord, FundingRateRecord, CandleValidationResult, FeatureRecord, PredictionRecord, RewardResult)
- [x] `src/processing/df_schemas.py` created — Pandera DataFrame schemas (OHLCVSchema, FeatureSchema, FundingRateSchema)
- [x] `tests/unit/test_schemas.py` created — 49 unit tests; all pass
- [x] `config/settings.yaml` fixed — removed `${...}` placeholder strings; real defaults (localhost/5432/etc.); pool_size=10, max_overflow=5, pool_timeout=30
- [x] `src/config.py` updated — `pool_timeout: int = 30` added to DatabaseSettings; defaults aligned with settings.yaml
- [x] `scripts/create_tables.sql` created — TimescaleDB DDL for all 6 tables (ohlcv_1h/4h/1d, funding_rates, regime_labels, doge_predictions, doge_replay_buffer); all hypertables, indexes, and constraints
- [x] `src/processing/storage.py` created — DogeStorage class with SQLAlchemy 2.0 Core; dialect-aware upsert (PostgreSQL/SQLite); filelock on all writes; all 11 methods; full type hints and Google docstrings
- [x] `tests/unit/test_storage.py` created — 27 unit tests using SQLite engine injection; all pass; full suite: 76 passed, 12 skipped
- [x] `src/config.py` updated — `get_settings()` accessor function added and exported
- [x] `src/utils/__init__.py` created
- [x] `src/utils/helpers.py` created — 5 pure utility functions: ms_to_datetime, datetime_to_ms, interval_to_ms, compute_expected_row_count, safe_divide
- [x] `src/utils/logger.py` created — loguru sinks (app.log JSON, rl.log JSON, stderr coloured); stdlib intercept; configure_logging() + get_rl_logger() API
- [x] `tests/unit/test_helpers.py` created — 40 tests; all pass
- [x] `tests/unit/test_logger.py` created — 10 tests; all pass
- [x] `tests/fixtures/doge_sample_data/generate_fixtures.py` created and run — 7 Parquet fixtures (trending_bull/bear, ranging, decoupled, mania, btc_aligned, funding_rates_sample)
- [x] All 7 fixture Parquet files validated against OHLCVSchema / FundingRateSchema
- [x] `tests/conftest.py` populated — session-scoped fixtures for all 7 files + all_doge_fixtures dict
- [x] **Phase 1 Quality Gate PASSED:**
  - `pytest --cov=src --cov-fail-under=80` → **84.38% coverage** (138 passed, 12 skipped)
  - `from src.config import get_settings; get_settings()` → Config OK
  - `from src.processing.storage import DogeStorage` → Storage import OK
  - `from src.processing.schemas import OHLCVRecord` → Schemas OK
  - All 7 fixture Parquet files readable, schema-valid

> **Session 1 notes (2026-03-07):**
> - Python 3.13.1 is present via `py` launcher; `.venv` created at project root
> - requirements.txt version pins loosened from CLAUDE.md originals for Python 3.13 compatibility
>   (ta-lib >=0.4, numpy >=2.0, pandas >=2.2, torch >=2.3, scipy >=1.13, pandas-ta ==0.4.71b0)
> - `src/config.py` uses `pydantic-settings` for env-var override of DB credentials
> - `secrets.env` is in `.gitignore` — template committed, real values never committed
> - Pandera schemas cover all 5 pipeline stages: raw → processed → aligned → features
> - FeatureSchema enforces all 21 mandatory DOGE features + NaN/Inf guard + regime_label enum
> - pytest run result: 12 collected, 12 skipped, coverage 0% (expected — no code exercised yet)
> - **Phase 1 is COMPLETE. Ready for Phase 2 — Data Ingestion.**

> **Session 2 notes (2026-03-07):**
> - `src/processing/schemas.py` created — 6 Pydantic v2 DTOs: OHLCVRecord, FundingRateRecord,
>   CandleValidationResult, FeatureRecord, PredictionRecord, RewardResult
> - `src/processing/df_schemas.py` created — 3 Pandera DataFrame schemas: OHLCVSchema,
>   FeatureSchema (DatetimeTZDtype UTC index, coerce=False at schema level to enforce timezone
>   strictly), FundingRateSchema (8h cadence enforcement)
> - `tests/unit/test_schemas.py` created — 49 tests; all 49 pass
> - Key decision: FeatureSchema uses `coerce=False` at the schema level (but `coerce=True` on
>   individual columns) so tz-naive indices are rejected rather than silently coerced to UTC
> - All constants extracted to module-level names (EPOCH_MS_MIN/MAX, CONFIDENCE_MIN/MAX, etc.)
> - `rl_config.yaml` was already created in Session 1 (Phase 9 RL item above updated)

> **Session 3 notes (2026-03-07):**
> - `config/settings.yaml` bug fixed — `${DB_PORT}` etc. were YAML string literals, not env-var
>   templates; changed to real defaults. Env-var override in `_build_settings()` still works.
> - `src/config.py` — added `pool_timeout: int = 30` to DatabaseSettings
> - `scripts/create_tables.sql` — idempotent TimescaleDB DDL; `abs_reward` is a GENERATED ALWAYS AS
>   STORED column in PostgreSQL; defined as regular Numeric in SQLAlchemy for cross-DB test compat
> - `src/processing/storage.py` — DogeStorage with engine injection pattern for tests; `_upsert()`
>   dynamically imports `postgresql.insert` or `sqlite.insert` based on dialect name; `abs_reward`
>   computed Python-side in `push_replay_buffer()`; `guard_raw_write()` exported at module level
> - `tests/unit/test_storage.py` — 27 tests; engine injection via `DogeStorage(s, engine=sqlite_engine)`;
>   `create_tables()` used for test schema; all 27 pass
> - Full suite result: **76 passed, 12 skipped** (12 skipped = Session 1 placeholder stubs)

> **Session 4 notes (2026-03-07) — Final Phase 1 session:**
> - `src/config.py` — `get_settings()` accessor added and exported for DI/testing patterns
> - `src/utils/helpers.py` — 5 pure utility functions; 100% test coverage
> - `src/utils/logger.py` — configure_logging() with app.log (JSON), rl.log (RL-only JSON),
>   stderr (coloured); _InterceptHandler routes stdlib logging through loguru
> - `tests/unit/test_helpers.py` — 40 tests; all pass
> - `tests/unit/test_logger.py` — 10 tests; all pass
> - Fixture generator — _START_MS constant was off by 8h (UTC offset issue); corrected to
>   1_640_995_200_000 (verified via Python: datetime(2022,1,1,tzinfo=UTC).timestamp()*1000)
> - `tests/conftest.py` — 7 session-scoped fixtures + all_doge_fixtures convenience dict
> - **Phase 1 Quality Gate: ALL CHECKS PASSED**
>   - Coverage: 84.38% (target: 80%) — 138 passed, 12 skipped
>   - get_settings() smoke: OK; DogeStorage import: OK; OHLCVRecord import: OK
>   - All 7 Parquet fixtures: readable and schema-valid
> - **Phase 1 is FULLY COMPLETE. Ready for Phase 2 — Data Ingestion.**

> **Session 5 notes (2026-03-08) — Phase 2, Prompt 2.1:**
> - New branch `feat/phase-2-ingestion` created from `feat/phase-1-scaffold`
> - `responses` library installed for HTTP mocking in unit tests
> - `src/ingestion/exceptions.py` created — 4-class hierarchy:
>   `BinanceAPIError` (base), `BinanceRateLimitError` (429, carries `retry_after`),
>   `BinanceAuthError` (401/403), `DataValidationError` (schema contract failure)
> - `src/ingestion/rest_client.py` created — `BinanceRESTClient`:
>   - `_request()`: pre-flight weight check, exponential backoff retry (2^attempt s,
>     max 5 attempts), 429 honours `Retry-After` header, non-retryable on 400/401/403/404
>   - `get_klines()`: full auto-pagination (close_time+1 as next startTime),
>     OHLCVSchema validation, deduplication by open_time, row-count sanity warning
>   - `get_exchange_info()`: 1-hour in-process TTL cache
>   - `get_order_book()`: returns raw dict
>   - `get_recent_trades()`: returns typed DataFrame via aggTrades endpoint
>   - Thread-safe weight counter via `threading.Lock`
>   - All timestamps are `int` (UTC epoch milliseconds) — never datetime objects
> - `tests/unit/test_rest_client.py` written — 27 tests, all passing:
>   - Pagination: 2-page scenario (1 000 + 50 rows) verified; second `startTime` asserted
>   - Rate limit: persistent 429 → `BinanceRateLimitError`; `Retry-After` sleep verified
>   - Retry: 503→200 succeeds; 502×3 exhausts and raises; no retry on 400/401/403
>   - Weight threshold: response with weight=1100 triggers `time.sleep` before next req
>   - Schema validation: high<low row → `DataValidationError`; non-list → `DataValidationError`
>   - Caching: `get_exchange_info()` called twice → exactly 1 HTTP request
>   - Exception attributes: `status_code`, `retry_after` on all typed exceptions
> - **Full suite result: 165 passed, 12 skipped — Coverage: 86.49% (gate: 80%) ✓**

> **Session 6 notes (2026-03-08) — Phase 2, Prompt 2.3 (Bootstrap Engine):**
> - `src/ingestion/bootstrap.py` created — `BootstrapEngine`:
>   - `BootstrapResult(frozen=True)`: symbol, interval, rows_fetched, rows_total, gaps_found,
>     duration_seconds, start_ms, end_ms, era_context_rows, era_training_rows
>   - `Checkpoint`: symbol, interval, last_open_time, rows_saved, started_at, updated_at
>   - `bootstrap_symbol()`: while-loop with cursor advance via `last_close_time + 1`;
>     breaks early if `n < batch_size` (final page); atomic checkpoint saves every N rows
>   - `_assign_era()`: `open_time >= 1_640_995_200_000` → `'training'`, else `'context'`
>   - `_count_gaps()`: counts consecutive pairs where `b − a > interval_ms`
>   - Checkpoint saved atomically via `.tmp` + `Path.replace()`; deleted on clean completion
>   - Resume: `resume_start = cp.last_open_time + interval_ms` (no double-fetch, no skip)
>   - Soft row-count sanity check via `compute_expected_row_count()` — warning only, no raise
>   - All exceptions re-raised unchanged after loguru error logging
> - `src/ingestion/multi_symbol.py` created — `MultiSymbolBootstrapper`:
>   - `run(symbols, intervals, start_ms, end_ms)` → `BootstrapReport` (dict keyed `sym/interval`)
>   - Phase A: `ThreadPoolExecutor(max_workers=2)` for DOGEUSDT + BTCUSDT concurrently
>   - Phase B: sequential DOGEBTC and any other symbols
>   - Each thread gets a fresh `BinanceRESTClient` via `make_client()` factory callable
>   - `DogeStorage` is shared; filelock in storage ensures write-safety
>   - `print_summary(report)` outputs formatted loguru table
> - `scripts/bootstrap_doge.py` updated — CLI entry point:
>   - `argparse`: `--symbols` (default: DOGEUSDT BTCUSDT DOGEBTC), `--intervals` (default: 1h),
>     `--start` (default: context_start_date from config YAML), `--end` (default: now UTC),
>     `--checkpoint-every` (default: 5 000), `--dry-run`
>   - `_parse_date_to_ms(date_str)` converts `YYYY-MM-DD` → UTC epoch ms
>   - Returns exit code 0 (success) or 1 (any failure)
> - `tests/unit/test_bootstrap.py` created — 18 tests, all passing:
>   - `_make_ohlcv_df()` helper includes `close_time` (required by bootstrap cursor logic)
>   - `sqlite_storage` fixture: in-memory SQLite, `create_tables()`, lock in `tmp_path`
>   - `mock_client` fixture: `MagicMock(spec=BinanceRESTClient)` with `weight_used=10`
>   - `engine_factory` fixture: closure over `tmp_path/checkpoints`
>   - 3 000-row full bootstrap (3 batches × 1 000), checkpoint deletion, resume from checkpoint,
>     era context/training/boundary, `_assign_era` boundary inclusive on training, all 5 gap
>     detection variants, gaps reported in result, era counts sum check
> - Key design decisions:
>   - cursor advances via `last_close_time + 1` (not `open_time + interval`): matches Binance
>     pagination semantics exactly and avoids off-by-one gaps
>   - `is_interpolated` column set to `False` on every bootstrap batch before upsert
>   - `multi_symbol.py` uses a `make_client` factory (not a shared client) so each thread
>     has independent weight counter and session
> - **Full suite result: 183 passed, 12 skipped — Coverage: 80.67% (gate: 80%) ✓**
>
> **Session 7 notes (2026-03-08) — Phase 2, Prompt 2.4 (DataValidator + MultiSymbolAligner):**
> - `src/processing/validator.py` completely replaced — old Pandera schema file removed;
>   new `DataValidator` class with 9 OHLCV checks, `validate_funding_rates`, `validate_feature_matrix`
>   - Check 1 missing columns → `is_valid=False` (no raise)
>   - Check 2 non-monotonic open_time → `DataValidationError` (CRITICAL)
>   - Check 3 gap > 3 candles → `DataValidationError` (CRITICAL); gap ≤ 3 → warning
>   - Check 4 OHLCV sanity (high<low, close≤0, volume<0) → warning
>   - Check 5 NaN/Inf → `DataValidationError` (CRITICAL) — placed early to avoid crashes
>   - Check 6 duplicates → warning
>   - Check 7 row-count deviation > 2 → warning
>   - Check 8 era assignment mismatch → warning
>   - Check 9 stale data — only triggered when `is_live_check=True`
>   - `FeatureSchemaError(DataValidationError)` raised by `validate_feature_matrix` for missing columns
>   - `_TRAINING_START_MS = 1_640_995_200_000` (2022-01-01 00:00 UTC) — era boundary
> - `src/processing/aligner.py` created — `MultiSymbolAligner`:
>   - `AlignmentError` exception; `AlignmentResult(frozen=True)` dataclass
>   - `align_symbols(symbols, interval, storage) -> AlignmentResult`
>     6-step: load → common range → trim/index → gap detection → inner join → prefixed concat
>   - DOGEBTC: forward-fills prices for gaps ≤ 3 candles; volume = 0; `dogebtc_interpolated=True`
>   - `dogebtc_interpolated` column built via `index.isin(filled_timestamps)` after all gap runs —
>     NOT via per-row assignment (which produces mixed-type columns in pandas 2.x+)
>   - `_find_gap_runs(missing_ts, interval_ms)` → `list[(run_start, run_end, run_size)]`
>   - Column prefix map: DOGEUSDT→doge_, BTCUSDT→btc_, DOGEBTC→dogebtc_
>   - `self._last_aligned` stores merged DataFrame after each successful call
> - `tests/unit/test_validator.py` — 43 tests; all passing
> - `tests/unit/test_aligner.py` — 18 tests; all passing
> - Bug fix: `test_check7_row_count_deviation_warning` — original test had a 4-candle gap
>   that caused CRITICAL raise before reaching the row-count check; fixed to use 2-candle gaps
> - Bug fix: `dogebtc_interpolated` flag — per-row `.loc` assignment produced all-False column
>   in pandas 2.x; fixed by tracking timestamps in `set[int]` and assigning via `index.isin()`
> - **Full suite result: 244 passed, 10 skipped — Coverage: 85.62% (gate: 80%) ✓**
>
> **HANDOVER — Next session: Phase 2, Prompt 2.5 — `BinanceFuturesClient`**
> - File to create: `src/ingestion/futures_client.py`
> - Purpose: fetch historical and live funding rate data from Binance USD-M Futures REST API
> - Endpoint: `GET /fapi/v1/fundingRate?symbol=DOGEUSDT&limit=1000&startTime={ms}&endTime={ms}`
> - Rate limit bucket: **Futures** (2 400 weight/minute) — separate from Spot (1 200/min)
>   Use `X-MBX-USED-WEIGHT-1M` from the **fapi** base URL (`https://fapi.binance.com`)
> - Key differences from `BinanceRESTClient`:
>   - Base URL: `https://fapi.binance.com` (not `https://api.binance.com`)
>   - Weight threshold: 2 400/min (configurable, default 1 800 to leave headroom)
>   - Funding rates come at 8-hour intervals (00:00, 08:00, 16:00 UTC)
>   - Response schema: `[{"symbol":..., "fundingTime":..., "fundingRate":...}, ...]`
> - Class: `BinanceFuturesClient(api_key, api_secret, base_url, weight_threshold, max_retries)`
>   - `get_funding_rates(symbol, start_ms, end_ms) -> pd.DataFrame`
>     columns: `funding_time (int), funding_rate (float), symbol (str)`
>     validated against `FundingRateSchema`; auto-paginates via `startTime` cursor
> - Reuse `BinanceAPIError`, `BinanceRateLimitError`, `BinanceAuthError`, `DataValidationError`
>   from `src/ingestion/exceptions.py` — do NOT introduce new exception classes
> - Test file: `tests/unit/test_futures_client.py` using `responses` library (same as REST client)
>   Minimum tests: pagination, 429 rate-limit, schema validation failure, empty response,
>   deduplication, `funding_time` int type, `funding_rate` float range check
> - After `BinanceFuturesClient`, the session after that should tackle `BinanceWebSocketClient`
>   (`src/ingestion/ws_client.py`) for live candle streaming

### Phase 2 — Data Ingestion
- [x] `BinanceRESTClient` — rate limiting, retry, weight headers
- [x] `src/ingestion/bootstrap.py` — `BootstrapEngine` with checkpointing (every N rows), atomic JSON saves, era assignment, gap detection, OHLCVSchema validation, full resume-from-checkpoint support; `BootstrapResult` + `Checkpoint` dataclasses
- [x] `src/ingestion/multi_symbol.py` — `MultiSymbolBootstrapper`; Phase A parallel (DOGEUSDT+BTCUSDT via ThreadPoolExecutor), Phase B sequential (DOGEBTC + others); per-thread fresh client via factory callable; `BootstrapReport` summary dict
- [x] `scripts/bootstrap_doge.py` — CLI entry point; argparse (--symbols, --intervals, --start, --end, --checkpoint-every, --dry-run); tqdm progress bar; `_parse_date_to_ms()` helper; delegates to `MultiSymbolBootstrapper`
- [x] `tests/unit/test_bootstrap.py` — 18 tests; all passing (3 000-row full run, checkpoint creation/deletion, resume from checkpoint, era context/training/boundary, gap detection, era counts in result)
- [x] `src/processing/validator.py` — `DataValidator` (9 OHLCV checks, funding rate validator, feature matrix validator with `FeatureSchemaError`); 43 unit tests all passing
- [x] `src/processing/aligner.py` — `MultiSymbolAligner` with DOGEBTC forward-fill, gap detection, prefixed column output, `AlignmentResult` dataclass; 18 unit tests all passing
- [ ] `BinanceFuturesClient` — funding rate endpoint
- [ ] `BinanceWebSocketClient` — reconnection + watchdog
- [ ] DOGEUSDT 1h bootstrap complete (data fetched from Binance)
- [ ] BTCUSDT 1h bootstrap complete
- [ ] DOGEBTC 1h bootstrap complete
- [ ] 4h and 1d bootstraps complete
- [ ] Funding rate bootstrap complete
- [ ] Aligner verified — all symbols on identical timestamp index
- [ ] QG-01 passed

### Phase 3 — Regime Classification
- [ ] `DogeRegimeClassifier` implemented
- [ ] Regime unit tests passing
- [ ] `label_regimes.py` run on full post-2022 dataset
- [ ] All 5 regimes present in distribution
- [ ] QG-04 passed

### Phase 4 — Feature Engineering
- [ ] Standard price indicators complete
- [ ] Standard volume indicators complete
- [ ] Lag and rolling features complete
- [ ] All 12 DOGE-specific features complete
- [ ] HTF features complete with lookahead guard
- [ ] Feature pipeline end-to-end working
- [ ] Zero NaN/Inf in feature matrix confirmed
- [ ] All feature unit tests passing
- [ ] QG-03 passed

### Phase 5 — Model Training
- [ ] `BaseModel` abstract class implemented
- [ ] `XGBoostModel` implemented and validated (> 53% directional accuracy)
- [ ] Per-regime XGBoost models trained
- [ ] `LSTMModel` implemented and validated
- [ ] `EnsembleModel` (meta-learner) implemented
- [ ] Walk-forward CV engine implemented and tested
- [ ] All models archived to MLflow with scaler + feature_columns.json
- [ ] QG-05 and QG-06 passed

### Phase 6 — Backtesting
- [ ] Backtesting engine implemented (next-open fill, fees, slippage)
- [ ] Per-regime performance reports generated
- [ ] Buy-and-hold comparison generated
- [ ] All Section 9 acceptance gates passed
- [ ] QG-06 passed

### Phase 7 — Inference & Deployment
- [ ] Inference engine implemented with all 12 steps
- [ ] All risk overrides implemented and tested
- [ ] Docker image builds and health check passes
- [ ] Shadow mode run for 48h
- [ ] Grafana dashboard configured
- [ ] Alerting configured
- [ ] QG-07 and QG-08 passed

### Phase 8 — Monitoring & Operations
- [ ] Drift detector active
- [ ] Weekly retraining scheduler configured
- [ ] Rollback procedure tested

### Phase 9 — RL Self-Teaching System
- [x] `doge_predictions` TimescaleDB table created — DDL in scripts/create_tables.sql + SQLAlchemy table def in storage.py
- [x] `doge_replay_buffer` table created — DDL in scripts/create_tables.sql + SQLAlchemy table def in storage.py
- [x] `rl_config.yaml` created — all horizons, decay constants, replay buffer, curriculum, self-training triggers
- [ ] `compute_reward()` implemented
- [ ] Reward unit tests — all 8 scenarios passing
- [ ] `PredictionVerifier` implemented
- [ ] Verifier edge case tests passing
- [ ] Replay Buffer with prioritised + regime-stratified sampling
- [ ] `CurriculumManager` implemented
- [ ] Curriculum advancement tests passing
- [ ] Multi-horizon predictor integrated into inference engine
- [ ] RL Grafana metrics live
- [ ] 7-day historical simulation completed
- [ ] Reward distribution confirmed: both positive and negative values present

---

## 14. SESSION RULES FOR CLAUDE

**At the start of every session:**
1. Read this entire file
2. Identify the current build phase from Section 13
3. Identify the specific module being built this session
4. Read the relevant section of the dev guide before writing any code
5. State which known errors from Section 12 are relevant to this module

**During every session:**
- Build one module per session — do not spread across multiple files
- Write the unit test before or alongside the implementation
- After writing any feature: check the lookahead checklist
- After writing any training code: check the training checklist
- Log what was completed at the end of the session in Section 13

**At the end of every session:**
- Update the `[ ]` checkboxes in Section 13
- Note any blockers or decisions made
- Confirm all tests pass before marking a phase item complete
- Never mark a quality gate passed unless the actual gate criteria were verified

**Never do these:**
- Write to `data/raw/` after bootstrap
- Use `train_test_split` anywhere in the codebase
- Fit a scaler on validation or test data
- Hardcode any numeric constant — use config files
- Use `print()` — use loguru logger
- Commit `secrets.env`
- Import from `notebooks/`
- Skip writing unit tests for any module in `src/`

---

## 15. DEPENDENCIES QUICK REFERENCE

```
python-binance==1.0.19
pandas==2.2.*
numpy==1.26.*
torch==2.3.*
xgboost==2.0.*
scikit-learn==1.4.*
ta-lib==0.4.*          # Requires C library: apt-get install libta-lib-dev
pandas-ta==0.3.*
mlflow==2.13.*
optuna==3.6.*
sqlalchemy==2.0.*
psycopg2-binary==2.9.*
pydantic==2.7.*
loguru==0.7.*
pandera==0.19.*
filelock==3.13.*
scipy==1.13.*
statsmodels==0.14.*
prometheus-client==0.20.*
pytest==8.*
pytest-cov==5.*
```

---

*Last updated: 2026-03-08 — v3.2 (Phase 2 Session 3 complete — DataValidator + MultiSymbolAligner)*
*Reference documents: `docs/framework.docx`, `docs/devguide_v3.docx`*
