-- Inicjalizacja bazy danych trading_db
-- Uruchamiane automatycznie przez docker-entrypoint-initdb.d

-- Włącz rozszerzenie TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Włącz UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Schematy per-serwis (izolacja danych)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS feature_engine;
CREATE SCHEMA IF NOT EXISTS strategy;
CREATE SCHEMA IF NOT EXISTS backtest;
CREATE SCHEMA IF NOT EXISTS ml_pipeline;
CREATE SCHEMA IF NOT EXISTS risk_mgmt;
CREATE SCHEMA IF NOT EXISTS execution;

-- ============================================================
-- Tabela OHLCV (hypertable TimescaleDB)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
    id          BIGSERIAL,
    symbol      TEXT        NOT NULL,
    interval    TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    source      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, ts)
);

-- Konwertuj na hypertable (partycjonowanie po czasie)
SELECT create_hypertable(
    'market_data.ohlcv',
    'ts',
    if_not_exists => TRUE
);

-- Indeks symbol + interval + czas (najczęstsze zapytania)
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_ts
    ON market_data.ohlcv (symbol, interval, ts DESC);

-- Kompresja danych starszych niż 7 dni
ALTER TABLE market_data.ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, interval'
);

SELECT add_compression_policy(
    'market_data.ohlcv',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ============================================================
-- Tabela sygnałów tradingowych
-- ============================================================
CREATE TABLE IF NOT EXISTS strategy.signals (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol          TEXT        NOT NULL,
    strategy_name   TEXT        NOT NULL,
    signal          TEXT        NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    confidence      DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    price           DOUBLE PRECISION NOT NULL,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    metadata        JSONB       DEFAULT '{}',
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
    ON strategy.signals (symbol, ts DESC);

-- ============================================================
-- Tabela wyników backtestów
-- ============================================================
CREATE TABLE IF NOT EXISTS backtest.results (
    id              UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    strategy_name   TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    interval        TEXT        NOT NULL,
    start_date      DATE        NOT NULL,
    end_date        DATE        NOT NULL,
    total_return    DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    sortino_ratio   DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    total_trades    INTEGER,
    parameters      JSONB       DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Tabela portfela / pozycji
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_mgmt.positions (
    id          UUID        DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    quantity    DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION,
    side        TEXT        NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    status      TEXT        NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED')),
    opened_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    closed_at   TIMESTAMPTZ
);

COMMENT ON DATABASE trading_db IS 'Trading System — mikroserwisowa baza danych';
