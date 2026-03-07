-- =============================================================================
-- create_tables.sql — TimescaleDB Schema for doge_predictor
-- =============================================================================
-- Run once against a live TimescaleDB instance:
--   psql -U postgres -d doge_predictor -f scripts/create_tables.sql
--
-- Prerequisites:
--   1. TimescaleDB extension installed on the PostgreSQL server.
--   2. Database "doge_predictor" already created.
--
-- All tables are idempotent (CREATE TABLE IF NOT EXISTS, IF NOT EXISTS guards).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ---------------------------------------------------------------------------
-- Table: ohlcv_1h
-- Primary 1h OHLCV candles for DOGEUSDT, BTCUSDT, and DOGEBTC.
-- Hypertable partitioned on open_time (UTC epoch milliseconds).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ohlcv_1h (
    symbol           VARCHAR(20)    NOT NULL,
    open_time        BIGINT         NOT NULL,
    open             NUMERIC(18, 8) NOT NULL,
    high             NUMERIC(18, 8) NOT NULL,
    low              NUMERIC(18, 8) NOT NULL,
    close            NUMERIC(18, 8) NOT NULL,
    volume           NUMERIC(18, 8) NOT NULL,
    close_time       BIGINT,
    quote_volume     NUMERIC(18, 8),
    num_trades       INT,
    era              VARCHAR(10)    NOT NULL,
    is_interpolated  BOOLEAN        NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, open_time)
);

SELECT create_hypertable(
    'ohlcv_1h',
    by_range('open_time', 604800000),   -- 7-day chunks (7 * 24 * 3600 * 1000 ms)
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_1h_symbol_era
    ON ohlcv_1h (symbol, era, open_time DESC);


-- ---------------------------------------------------------------------------
-- Table: ohlcv_4h
-- 4-hour OHLCV candles for multi-timeframe analysis.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ohlcv_4h (
    symbol           VARCHAR(20)    NOT NULL,
    open_time        BIGINT         NOT NULL,
    open             NUMERIC(18, 8) NOT NULL,
    high             NUMERIC(18, 8) NOT NULL,
    low              NUMERIC(18, 8) NOT NULL,
    close            NUMERIC(18, 8) NOT NULL,
    volume           NUMERIC(18, 8) NOT NULL,
    close_time       BIGINT,
    quote_volume     NUMERIC(18, 8),
    num_trades       INT,
    era              VARCHAR(10)    NOT NULL,
    is_interpolated  BOOLEAN        NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, open_time)
);

SELECT create_hypertable(
    'ohlcv_4h',
    by_range('open_time', 2592000000),  -- 30-day chunks (30 * 24 * 3600 * 1000 ms)
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_4h_symbol_era
    ON ohlcv_4h (symbol, era, open_time DESC);


-- ---------------------------------------------------------------------------
-- Table: ohlcv_1d
-- Daily OHLCV candles for higher-timeframe trend features.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ohlcv_1d (
    symbol           VARCHAR(20)    NOT NULL,
    open_time        BIGINT         NOT NULL,
    open             NUMERIC(18, 8) NOT NULL,
    high             NUMERIC(18, 8) NOT NULL,
    low              NUMERIC(18, 8) NOT NULL,
    close            NUMERIC(18, 8) NOT NULL,
    volume           NUMERIC(18, 8) NOT NULL,
    close_time       BIGINT,
    quote_volume     NUMERIC(18, 8),
    num_trades       INT,
    era              VARCHAR(10)    NOT NULL,
    is_interpolated  BOOLEAN        NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, open_time)
);

SELECT create_hypertable(
    'ohlcv_1d',
    by_range('open_time', 31536000000), -- 365-day chunks (365 * 24 * 3600 * 1000 ms)
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_1d_symbol_era
    ON ohlcv_1d (symbol, era, open_time DESC);


-- ---------------------------------------------------------------------------
-- Table: funding_rates
-- 8-hour Binance DOGEUSDT perpetual funding rate observations.
-- Hypertable partitioned on timestamp_ms.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS funding_rates (
    timestamp_ms  BIGINT         NOT NULL,
    symbol        VARCHAR(20)    NOT NULL,
    funding_rate  NUMERIC(12, 8) NOT NULL,
    mark_price    NUMERIC(18, 8),
    PRIMARY KEY (timestamp_ms)
);

SELECT create_hypertable(
    'funding_rates',
    by_range('timestamp_ms', 2592000000),  -- 30-day chunks
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_funding_rates_symbol
    ON funding_rates (symbol, timestamp_ms DESC);


-- ---------------------------------------------------------------------------
-- Table: regime_labels
-- Per-candle market regime assignments produced by DogeRegimeClassifier.
-- Plain table — not a hypertable (queries always filter on symbol + open_time).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS regime_labels (
    open_time      BIGINT          NOT NULL,
    symbol         VARCHAR(20)     NOT NULL,
    regime         VARCHAR(20)     NOT NULL,
    btc_corr_24h   NUMERIC(8, 4),
    bb_width       NUMERIC(8, 6),
    atr_norm       NUMERIC(8, 6),
    PRIMARY KEY (symbol, open_time)
);

CREATE INDEX IF NOT EXISTS idx_regime_labels_regime
    ON regime_labels (regime, open_time DESC);


-- ---------------------------------------------------------------------------
-- Table: doge_predictions
-- Permanent audit trail of all model predictions. Rows are NEVER deleted.
-- Prediction fields are immutable after insert.
-- Outcome fields (actual_price, reward_score, …) written only by Verifier.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS doge_predictions (
    -- Immutable prediction fields
    prediction_id       VARCHAR(64)    NOT NULL,
    created_at          BIGINT         NOT NULL,
    open_time           BIGINT         NOT NULL,
    symbol              VARCHAR(20)    NOT NULL,
    horizon_label       VARCHAR(10)    NOT NULL,
    horizon_candles     INT            NOT NULL,
    target_open_time    BIGINT         NOT NULL,
    price_at_prediction NUMERIC(18, 8) NOT NULL,
    predicted_direction SMALLINT       NOT NULL,
    confidence_score    NUMERIC(6, 4)  NOT NULL,
    lstm_prob           NUMERIC(6, 4)  NOT NULL,
    xgb_prob            NUMERIC(6, 4)  NOT NULL,
    regime_label        VARCHAR(20)    NOT NULL,
    model_version       VARCHAR(64)    NOT NULL,

    -- Outcome fields (filled by PredictionVerifier, initially NULL)
    actual_price        NUMERIC(18, 8),
    actual_direction    SMALLINT,
    reward_score        NUMERIC(10, 6),
    direction_correct   BOOLEAN,
    error_pct           NUMERIC(10, 6),
    verified_at         BIGINT,

    PRIMARY KEY (prediction_id)
);

-- Verifier lookup: find matured-but-unverified predictions
CREATE INDEX IF NOT EXISTS idx_predictions_maturity
    ON doge_predictions (target_open_time, verified_at);

-- Curriculum stage queries: accuracy by horizon over time
CREATE INDEX IF NOT EXISTS idx_predictions_horizon_created
    ON doge_predictions (horizon_label, created_at DESC);

-- Regime-based performance analysis
CREATE INDEX IF NOT EXISTS idx_predictions_regime_created
    ON doge_predictions (regime_label, created_at DESC);

-- model_version auditing
CREATE INDEX IF NOT EXISTS idx_predictions_model_version
    ON doge_predictions (model_version, created_at DESC);


-- ---------------------------------------------------------------------------
-- Table: doge_replay_buffer
-- Prioritised, regime-stratified experience replay buffer for RL self-training.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS doge_replay_buffer (
    buffer_id       UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    horizon_label   VARCHAR(10)    NOT NULL,
    regime          VARCHAR(20)    NOT NULL,
    feature_vector  BYTEA          NOT NULL,
    predicted_price NUMERIC(18, 8),
    actual_price    NUMERIC(18, 8),
    reward_score    NUMERIC(10, 6) NOT NULL,
    model_version   VARCHAR(64)    NOT NULL,
    created_at      TIMESTAMPTZ    NOT NULL DEFAULT now(),

    -- Generated column: absolute reward for priority sampling
    abs_reward      NUMERIC(10, 6) GENERATED ALWAYS AS (ABS(reward_score)) STORED
);

-- Priority sampling: high-|reward| samples first
CREATE INDEX IF NOT EXISTS idx_replay_priority
    ON doge_replay_buffer (horizon_label, abs_reward DESC);

-- Regime-stratified sampling
CREATE INDEX IF NOT EXISTS idx_replay_regime
    ON doge_replay_buffer (regime, horizon_label, abs_reward DESC);

-- Chronological pruning (drop oldest when buffer is full)
CREATE INDEX IF NOT EXISTS idx_replay_created
    ON doge_replay_buffer (created_at ASC);
