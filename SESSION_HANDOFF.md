# Session Handoff — doge_predictor
> Feed this document to the AI agent at the start of the next session.
> It captures the exact state of the project after Session 1 (2026-03-07).

---

## Quick-start for next agent

```
Read CLAUDE.md in full. Then read this file. Do not write any code until both are read.
```

**Repo:** https://github.com/alfarraahmed5-afk/doge-predictor
**Branch:** `feat/phase-1-scaffold` (PR open, not yet merged)
**Root:** `C:\Users\fault\OneDrive\Desktop\DogePred`
**Python:** 3.13.1 via `py` launcher — use `.venv/Scripts/python` for all commands
**Git identity (local):** fault / alfarraahmed5@gmail.com

---

## What was completed in Session 1

### Phase 1 — Project Initialization (FULLY COMPLETE)

All Phase 1 CLAUDE.md checklist items are marked `[x]`.

| Item | File(s) | Notes |
|---|---|---|
| Directory structure | All dirs + `.gitkeep` | Canonical layout from CLAUDE.md §2 |
| `.gitignore` / `.gitattributes` | Root | LF normalization; secrets.env, .venv/, data/raw/ excluded |
| Virtual environment | `.venv/` | Python 3.13.1; NOT committed |
| All dependencies installed | `.venv/` | See version table below |
| Config YAMLs | `config/` | All 4 YAMLs + secrets.env template |
| `src/config.py` | `src/config.py` | Pydantic Settings singletons for all 4 configs |
| Pandera schema contracts | `src/processing/validator.py` | 5 schemas: RawOHLCV, RawFunding, ProcessedOHLCV, Aligned, Feature |
| Placeholder tests | `tests/unit/test_*.py` | 12 unit + 1 integration; all skip; 12/12 collected by pytest |
| `requirements.txt` | Root | Loosened pins for Python 3.13 compat (see notes) |
| `pyproject.toml` | Root | pytest + ruff + mypy config |
| Dev server config | `.claude/launch.json` | 3 configs: inference-server, mlflow-ui, jupyter-lab |
| MLflow UI | Running on port 5000 | `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` |

### Installed package versions (Python 3.13.1)

| Package | Installed | Note |
|---|---|---|
| ta-lib | 0.6.8 | Bundles C library — no OS install needed on Windows |
| torch | 2.10.0 | Supports Python 3.13 |
| pandas | 2.3.3 | pandas-ta requires >=2.3.2 |
| numpy | 2.2.6 | 2.x required for Py 3.13 wheels |
| xgboost | 3.2.0 | |
| scikit-learn | 1.8.0 | |
| pandas-ta | 0.4.71b0 | Pre-release; only version on PyPI |
| scipy | 1.17.1 | 1.14+ for Py 3.13 wheels |
| statsmodels | 0.14.6 | |
| mlflow | 3.10.1 | |
| optuna | 4.7.0 | |
| pydantic | 2.13.0b2 | Pre-release |
| pydantic-settings | 2.13.1 | |
| pandera | 0.29.0 | |
| loguru | 0.7.3 | |
| python-binance | 1.0.19 | |
| prometheus-client | 0.24.1 | |

### requirements.txt pin changes from CLAUDE.md originals

CLAUDE.md §15 pins were written for Python 3.11/3.12. The following were loosened for 3.13:

| Package | CLAUDE.md pin | Actual pin | Reason |
|---|---|---|---|
| numpy | ==1.26.* | >=2.0 | No Py 3.13 wheels for 1.26.x |
| pandas | ==2.2.* | >=2.2 | pandas-ta requires >=2.3.2 |
| torch | ==2.3.* | >=2.3 | Py 3.13 support from 2.5+ |
| ta-lib | ==0.4.* | >=0.4 | 0.6.8 bundles C lib; 0.4 won't build |
| pandas-ta | ==0.3.* | ==0.4.71b0 | Only pre-releases on PyPI |
| scipy | ==1.13.* | >=1.13 | Py 3.13 wheels from 1.14+ |
| All others | ==X.Y.* | >=X.Y | Loosened to allow compatible newer versions |

---

## Exact state of every source file

### `src/config.py` — EXISTS, COMPLETE
Pydantic Settings models for all 4 config files. Module-level singletons:
- `settings` (Settings)
- `doge_settings` (DogeSettings)
- `regime_config` (RegimeConfig)
- `rl_config` (RLConfig)

Global seed applied at import time. DB credentials overridable via env vars.
**All other modules import config from here — never re-read YAML directly.**

### `src/processing/validator.py` — EXISTS, COMPLETE
5 Pandera schema contracts with a `validate()` helper:
- `RawOHLCVSchema` — raw Binance kline response (11 columns, OHLCV invariants)
- `RawFundingRateSchema` — Binance funding rate response
- `ProcessedOHLCVSchema` — cleaned OHLCV + `era` column, strict=True
- `AlignedSchema` — multi-symbol `doge_/btc_/dogebtc_` columns + funding_rate
- `FeatureSchema` — all 21 mandatory features enforced, NaN/Inf guard, regime_label enum

### `src/processing/schemas.py` — DOES NOT EXIST YET
**This is where the next session must start.**

### `src/processing/df_schemas.py` — DOES NOT EXIST YET
**Also needed in the next session.**

### All `src/*/` subdirectory `__init__.py` files — EXIST (stubs only)
Packages: `ingestion`, `processing`, `regimes`, `features`, `models`, `training`, `evaluation`, `inference`, `monitoring`, `rl`

### All implementation files — DO NOT EXIST YET
Everything in `src/` except `config.py` and `processing/validator.py` is placeholder or empty.

---

## Task queued for next session (already planned, not started)

The following task was given but interrupted before any code was written:

```
Phase 1 continues. Today's session: define all data contracts before any data
is fetched. Contracts must exist before implementation so all modules conform to them.

Build the following in src/processing/schemas.py:
1. Pydantic models for all data transfer objects:
   OHLCVRecord:
     Fields: open_time (int, UTC ms), open, high, low, close, volume,
             close_time (int), quote_volume, num_trades (int), symbol (str),
             interval (str), era (Literal['context','training','live'])
     Validators: high >= max(open,close), low <= min(open,close),
                 high >= low, close > 0, volume >= 0, open_time < close_time
   FundingRateRecord:
     Fields: timestamp_ms (int), symbol (str), funding_rate (float),
             mark_price (float)
     Validator: timestamp_ms > 0, funding_rate between -0.01 and 0.01
   CandleValidationResult:
     Fields: is_valid (bool), errors (list[str]), row_count (int),
             gap_count (int), duplicate_count (int)
   FeatureRecord:
     Fields: open_time (int), symbol (str), era (str), regime (str),
             all feature columns as Optional[float] with default None
   PredictionRecord:
     Fields: all fields from the doge_predictions schema in CLAUDE.md Section 11
     Validators: confidence_score between 0.5 and 1.0,
                 predicted_direction in (-1, 0, 1),
                 horizon_label in ('SHORT','MEDIUM','LONG','MACRO')
   RewardResult:
     Fields: reward_score, direction_score, magnitude_score,
             calibration_score, error_pct, direction_correct (bool)

2. Pandera DataFrame schemas in src/processing/df_schemas.py:
   OHLCVSchema:
     - All required columns present with correct dtypes
     - open_time is strictly monotonically increasing
     - high >= open, high >= close (at row level)
     - low <= open, low <= close (at row level)
     - close > 0, volume >= 0
     - No NaN or Inf in any column
   FeatureSchema:
     - index is DatetimeTZDtype UTC
     - No NaN or Inf in any column
     - All mandatory DOGE features present (list from CLAUDE.md Section 7)
     - No constant columns (std > 0 for all columns)
   FundingRateSchema:
     - timestamp_ms strictly monotonic
     - interval between rows is exactly 28800000ms (8h)

3. Unit tests in tests/unit/test_schemas.py:
   - Test OHLCVRecord rejects records where high < low
   - Test OHLCVRecord rejects records where close <= 0
   - Test OHLCVSchema rejects DataFrames with NaN
   - Test OHLCVSchema rejects DataFrames with non-monotonic timestamps
   - Test PredictionRecord rejects confidence_score > 1.0
   - Test PredictionRecord rejects invalid horizon_label

Run pytest tests/unit/test_schemas.py — all tests must pass.
Update CLAUDE.md Section 13.
```

---

## Key architectural decisions made in Session 1

1. **`src/config.py` is the single config entry point.** All modules do `from src.config import doge_settings` etc. Never re-read YAML anywhere else.

2. **`src/processing/validator.py` owns the `validate()` helper.** It wraps pandera and logs via loguru. The new `df_schemas.py` will define schema objects; `validator.py` imports and uses them.

3. **`secrets.env` is a template only.** Real API keys go in `config/secrets.env` locally, never committed. `.gitignore` entry confirmed working.

4. **`data/raw/` is immutable.** Gitignored. Any write attempt after bootstrap must raise `PermissionError`.

5. **All timestamps are `int` (UTC epoch milliseconds).** No tz-naive datetimes anywhere.

6. **Version pins in `requirements.txt` are now `>=` style** (not `==`) due to Python 3.13 constraints. The installed versions above are what's actually in `.venv/`.

---

## Git state

```
Branch:   feat/phase-1-scaffold
Remote:   origin → https://github.com/alfarraahmed5-afk/doge-predictor
Commits:
  0f640f1  chore: mark Phase 1 fully complete in CLAUDE.md
  0daf566  feat: Phase 1 blockers resolved — Pandera schemas, dependency fixes
  6106060  feat: Phase 1 scaffold — complete project structure and config
```

PR open at: https://github.com/alfarraahmed5-afk/doge-predictor/compare/feat/phase-1-scaffold

---

## How to run things

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Run all tests
.venv/Scripts/python -m pytest tests/unit/ --no-cov -v

# Start MLflow UI
.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Install deps (if new machine)
.venv/Scripts/pip install -r requirements.txt --pre --only-binary :all:
.venv/Scripts/pip install -r requirements-dev.txt --pre --only-binary :all:
```

---

*Generated: 2026-03-07 — end of Session 1*
