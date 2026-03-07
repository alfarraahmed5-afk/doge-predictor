# Session Handoff — doge_predictor
> Feed this document to the AI agent at the start of the next session.
> It captures the exact state of the project after Sessions 1–4 (2026-03-07).

---

## Quick-start for next agent

```
Read CLAUDE.md in full. Then read this file. Do not write any code until both are read.
```

**Repo:** https://github.com/alfarraahmed5-afk/doge-predictor
**Branch:** `feat/phase-1-scaffold` (all sessions committed, PR ready)
**Root:** `C:\Users\fault\OneDrive\Desktop\DogePred`
**Python:** 3.13.1 via `py` launcher — use `.venv/Scripts/python` for all commands
**Git identity (local):** fault / alfarraahmed5@gmail.com

---

## Phase 1 Status: FULLY COMPLETE

The Phase 1 Quality Gate has been passed:

| Check | Result |
|---|---|
| `pytest --cov=src --cov-fail-under=80` | **84.38%** — 138 passed, 12 skipped |
| `from src.config import get_settings; get_settings()` | Config OK |
| `from src.processing.storage import DogeStorage` | Storage import OK |
| `from src.processing.schemas import OHLCVRecord` | Schemas OK |
| All 7 fixture Parquet files readable + schema-valid | PASS |

---

## Complete inventory of what exists in src/

### `src/config.py` — COMPLETE
- Pydantic Settings models for all 4 config files
- Module-level singletons: `settings`, `doge_settings`, `regime_config`, `rl_config`
- `get_settings()` accessor function (used by DI patterns and quality gate)
- Global seed applied at import time (`random`, `np`, `torch`)
- DB credentials overridable via env vars (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`)

### `src/processing/validator.py` — COMPLETE (Session 1 schemas)
5 Pandera schema contracts:
- `RawOHLCVSchema` — raw Binance kline (11 cols, OHLCV invariants)
- `RawFundingRateSchema` — Binance funding rate response
- `ProcessedOHLCVSchema` — cleaned OHLCV + `era` col, `strict=True`
- `AlignedSchema` — multi-symbol doge_/btc_/dogebtc_ + funding_rate
- `FeatureSchema` — all 21 mandatory features, NaN/Inf guard, regime_label enum
- `validate()` helper wraps pandera + loguru

### `src/processing/schemas.py` — COMPLETE (Session 2 DTOs)
6 Pydantic v2 record DTOs with validators:
- `OHLCVRecord` — OHLCV + cross-field OHLC invariants
- `FundingRateRecord` — funding rate bounded to [-0.01, 0.01]
- `CandleValidationResult` — validation result summary
- `FeatureRecord` — all 21 mandatory features as `Optional[float]`
- `PredictionRecord` — all doge_predictions fields; confidence [0.5, 1.0]; target_open_time > open_time
- `RewardResult` — RL reward decomposition

### `src/processing/df_schemas.py` — COMPLETE (Session 2 DataFrame schemas)
3 Pandera DataFrame schemas:
- `OHLCVSchema` — strict OHLC invariants, monotonic open_time, NaN/Inf guard
- `FeatureSchema` — **`coerce=False` at schema level** to enforce UTC DatetimeTZDtype index strictly
- `FundingRateSchema` — exact 28,800,000 ms (8h) cadence enforcement
- `validate_df()` helper with loguru logging

### `src/processing/storage.py` — COMPLETE (Session 3)
`DogeStorage` class — SQLAlchemy 2.0 Core (no ORM):
- Engine injection in `__init__(settings, *, engine=None)` — `engine` kwarg for test isolation
- Dialect-aware upsert: `_upsert()` dynamically imports `postgresql.insert` or `sqlite.insert`
- `FileLock` on all write methods (30s timeout)
- `guard_raw_write(path)` — raises `PermissionError` if path is inside `data/raw/`
- All 11 methods: push_ohlcv, get_ohlcv, push_funding_rates, get_funding_rates,
  push_regime_labels, get_regime_labels, push_prediction, get_matured_unverified,
  update_prediction_outcome, push_replay_buffer, get_replay_buffer_sample
- `create_tables()` for test/dev schema creation
- `dispose()` for engine cleanup

### `src/utils/__init__.py` — COMPLETE (Session 4, empty)

### `src/utils/helpers.py` — COMPLETE (Session 4)
5 pure utility functions (100% test coverage):
- `ms_to_datetime(ms: int) -> datetime` — UTC epoch ms → UTC-aware datetime
- `datetime_to_ms(dt: datetime) -> int` — UTC datetime → epoch ms
- `interval_to_ms(interval: str) -> int` — '1h' → 3_600_000, '4h' → 14_400_000, etc.
- `compute_expected_row_count(start_ms, end_ms, interval_ms) -> int`
- `safe_divide(numerator, denominator, fallback=0.0) -> float`

### `src/utils/logger.py` — COMPLETE (Session 4)
- `configure_logging(log_level='INFO')` — call ONCE at app startup
  - Creates `logs/app.log` (JSON, all events)
  - Creates `logs/rl.log` (JSON, RL events only — filter: `name.startswith('src.rl')`)
  - Adds coloured stderr sink
  - Replaces root stdlib logging handler with `_InterceptHandler`
  - Rotation: 100 MB, retention: 10 files per sink
- `get_rl_logger()` — returns loguru singleton

### All other `src/*/` `__init__.py` files — empty stubs (Sessions 1)
`ingestion`, `processing`, `regimes`, `features`, `models`, `training`,
`evaluation`, `inference`, `monitoring`, `rl`

### All implementation files beyond the above — DO NOT EXIST YET

---

## Complete inventory of what exists in tests/

### `tests/conftest.py` — COMPLETE (Session 4)
Session-scoped pytest fixtures loading 7 Parquet files:
- `doge_trending_bull`, `doge_trending_bear`, `doge_ranging`,
  `doge_decoupled`, `doge_mania` — 500-row DOGEUSDT fixtures
- `btc_aligned` — 500-row BTCUSDT aligned to DOGE timestamps
- `funding_rates_sample` — 200-row 8h funding rate fixture
- `all_doge_fixtures` — convenience dict of all 5 DOGE fixtures

### `tests/fixtures/doge_sample_data/` — COMPLETE (Session 4)
Generator script and 7 Parquet files:
- `generate_fixtures.py` — regenerate with `python tests/fixtures/doge_sample_data/generate_fixtures.py`
- `dogeusdt_1h_trending_bull.parquet` (500 rows, seed=42, EMA20 > EMA50 > EMA200)
- `dogeusdt_1h_trending_bear.parquet` (500 rows, seed=42)
- `dogeusdt_1h_ranging.parquet` (500 rows, Ornstein-Uhlenbeck near $0.090)
- `dogeusdt_1h_decoupled.parquet` (500 rows, independent noise, BTC corr ≈ 0)
- `dogeusdt_1h_mania.parquet` (200 rows, 10x exponential drift)
- `btcusdt_1h_aligned.parquet` (500 rows, ~$42,000)
- `funding_rates_sample.parquet` (200 rows, 8h intervals from 2022-01-01)

**Timestamp:** All use `_START_MS = 1_640_995_200_000` (2022-01-01 00:00:00 UTC).
**Do NOT use `1_641_024_000_000` — that is 08:00 UTC, not midnight.**

### `tests/unit/` — test files that exist and pass
| File | Tests | Status |
|---|---|---|
| `test_schemas.py` | 49 | All pass |
| `test_storage.py` | 27 | All pass (SQLite injection) |
| `test_helpers.py` | 40 | All pass |
| `test_logger.py` | 10 | All pass |
| `test_validator.py` | 1 | Skipped (placeholder) |
| `test_rest_client.py` | 1 | Skipped (placeholder) |
| `test_aligner.py` | 1 | Skipped (placeholder) |
| `test_doge_features.py` | 1 | Skipped (placeholder) |
| `test_funding_features.py` | 1 | Skipped (placeholder) |
| `test_htf_features.py` | 1 | Skipped (placeholder) |
| `test_regime_classifier.py` | 1 | Skipped (placeholder) |
| `test_walk_forward.py` | 1 | Skipped (placeholder) |
| `test_backtest.py` | 1 | Skipped (placeholder) |
| `test_reward.py` | 1 | Skipped (placeholder) |
| `test_verifier.py` | 1 | Skipped (placeholder) |
| `test_curriculum.py` | 1 | Skipped (placeholder) |

**Full suite: 138 passed, 12 skipped — 84.38% src/ coverage**

---

## Database layer

### `scripts/create_tables.sql` — COMPLETE (Session 3)
Idempotent TimescaleDB DDL — run once against a live TimescaleDB instance:
```
psql -U postgres -d doge_predictor -f scripts/create_tables.sql
```
Tables: `ohlcv_1h` (7-day chunks), `ohlcv_4h` (30-day chunks), `ohlcv_1d` (365-day chunks),
`funding_rates` (30-day chunks), `regime_labels` (plain), `doge_predictions`, `doge_replay_buffer`

**Note:** `doge_replay_buffer.abs_reward` is `GENERATED ALWAYS AS (ABS(reward_score)) STORED`
in TimescaleDB DDL, but is a regular `Numeric` column in the SQLAlchemy table definition
(computed in Python before insert). This is intentional for cross-DB test compatibility.

---

## Git state

```
Branch:   feat/phase-1-scaffold
Remote:   origin → https://github.com/alfarraahmed5-afk/doge-predictor
Latest commit: feat: Sessions 2-3 — schema contracts, DB layer, TimescaleDB DDL
```

Session 4 changes (not yet committed — commit before starting Phase 2):
- `src/config.py` — get_settings() added
- `src/utils/` — __init__.py, helpers.py, logger.py
- `tests/unit/test_helpers.py`, `tests/unit/test_logger.py`
- `tests/fixtures/doge_sample_data/generate_fixtures.py` + 7 Parquet files
- `tests/conftest.py` — populated
- `CLAUDE.md` — Section 13 updated

---

## Phase 2 Task: Data Ingestion

**Start with `src/ingestion/rest_client.py`.**

### What to build

```
src/ingestion/rest_client.py  — BinanceRESTClient
```

**Requirements:**
1. Rate-limit tracking: read `X-MBX-USED-WEIGHT-1M` header on every response; back off if > 1,100 weight/min
2. Exponential backoff retry: max 5 attempts, base delay 1s, cap 60s; retry on 429, 418, 5xx
3. `get_klines(symbol, interval, start_ms, end_ms, limit=1000)` → `list[dict]`
4. `get_depth(symbol, limit=100)` → `dict`
5. `get_agg_trades(symbol, start_ms, limit=1000)` → `list[dict]`
6. All timestamps in and out are UTC epoch milliseconds (`int`)
7. No credentials needed for public endpoints — no auth logic
8. Use `requests` (already installed); connection timeout 10s, read timeout 30s
9. Loguru logging on every request: method, endpoint, weight used, response time
10. Every method wraps HTTP calls in `try/except requests.RequestException`

**Base URL:** `https://api.binance.com`
**Rate limit endpoints (from CLAUDE.md §4.3):**
```
GET /api/v3/klines?symbol={s}&interval={i}&limit=1000&startTime={ms}&endTime={ms}
GET /api/v3/depth?symbol={s}&limit=100
GET /api/v3/aggTrades?symbol={s}&limit=1000&startTime={ms}
```

**Unit tests (`tests/unit/test_rest_client.py`):**
- Use `unittest.mock.patch` to mock `requests.Session.get` — never make real HTTP calls
- Test rate-limit header parsing
- Test retry on 429 (assert 5 attempts)
- Test exponential backoff timing (mock `time.sleep`)
- Test `get_klines` returns correctly structured data
- Test connection timeout is passed correctly
- Replace the placeholder skip with real tests

**After building rest_client.py:**
- Write `src/ingestion/futures_client.py` (funding rate endpoint) in the same session if time allows
- Do NOT start bootstrap.py until both clients exist and tests pass

### Architecture decisions already made

| Decision | Detail |
|---|---|
| Config import | `from src.config import settings, doge_settings` — never re-read YAML |
| Logging | `from loguru import logger` — no print() |
| Timestamps | All `int` UTC epoch ms — use `src.utils.helpers.interval_to_ms()` |
| Path handling | `pathlib.Path` only |
| Error handling | `try/except requests.RequestException` on every HTTP call |
| Rate limit | Read `X-MBX-USED-WEIGHT-1M` header, pause if > 1,100 |

### Quality gate for Phase 2 (QG-01)

Before declaring Phase 2 complete, the following must all pass:
- `pytest tests/unit/test_rest_client.py` — all pass, no skips
- `pytest tests/unit/test_aligner.py` — all pass, no skips
- `pytest --cov=src --cov-fail-under=80` — must remain ≥ 80%
- Manual smoke: bootstrap ~100 DOGEUSDT 1h rows into a test Parquet file and validate against `OHLCVSchema`

---

## How to run things

```bash
# Run all tests
.venv/Scripts/python -m pytest tests/unit/ --no-cov -v

# Run with coverage
.venv/Scripts/python -m pytest tests/unit/ --cov=src --cov-fail-under=80

# Smoke checks
.venv/Scripts/python -c "from src.config import get_settings; print(get_settings().project.name)"
.venv/Scripts/python -c "from src.utils.helpers import interval_to_ms; print(interval_to_ms('1h'))"
.venv/Scripts/python -c "from src.utils.logger import configure_logging; configure_logging()"

# Regenerate fixtures (if needed)
.venv/Scripts/python tests/fixtures/doge_sample_data/generate_fixtures.py
```

---

*Generated: 2026-03-07 — end of Session 4 (Final Phase 1 session)*
