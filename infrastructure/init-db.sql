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
CREATE SCHEMA IF NOT EXISTS macro_data;

-- ============================================================
-- Tabela OHLCV (hypertable TimescaleDB)
-- ============================================================
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
    symbol      TEXT        NOT NULL,
    interval    TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    -- Dividend/split adjusted close. Raw OHLC is the execution price; returns
    -- are measured on this. Nullable: bars stored before 2026-07-28 lack it.
    adj_close   DOUBLE PRECISION,
    source      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Natural key: enables idempotent upserts (ON CONFLICT / merge) and dedupe.
    -- Includes ts (partition column) as required for TimescaleDB hypertable unique keys.
    PRIMARY KEY (symbol, interval, ts)
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
-- Panel fundamentów (point-in-time, P2-3)
-- ============================================================
-- `filed_at` to data, od której liczba była PUBLICZNIE znana. Odczyt as-of bierze
-- najnowszy wiersz z filed_at ŚCIŚLE wcześniejszym niż początek sesji — wiersz
-- bez filed_at jest niewidoczny dla tego odczytu (nie da się go umieścić w czasie,
-- więc nie wolno go użyć), a nie „stary".
CREATE TABLE IF NOT EXISTS market_data.fundamentals (
    symbol              TEXT NOT NULL,
    period_end          DATE NOT NULL,
    fiscal_period       TEXT NOT NULL,
    filed_at            TIMESTAMPTZ,
    revenue             DOUBLE PRECISION,
    -- Gross profitability (Novy-Marx): filerzy raportują jedno albo drugie.
    gross_profit        DOUBLE PRECISION,
    cost_of_revenue     DOUBLE PRECISION,
    net_income          DOUBLE PRECISION,
    total_assets        DOUBLE PRECISION,
    total_liabilities   DOUBLE PRECISION,
    current_assets      DOUBLE PRECISION,
    current_liabilities DOUBLE PRECISION,
    shares_outstanding  DOUBLE PRECISION,
    operating_cash_flow DOUBLE PRECISION,
    eps                 DOUBLE PRECISION,
    piotroski_f_score   INTEGER,
    source              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Klucz naturalny → idempotentny upsert przy ponownym pobraniu z EDGAR.
    PRIMARY KEY (symbol, period_end, fiscal_period)
);

-- Odczyt as-of chodzi po (symbol, filed_at DESC) — to jest ten indeks.
CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_filed
    ON market_data.fundamentals (symbol, filed_at DESC);

-- ============================================================
-- Panel makro z semantyką VINTAGE (P2-4)
-- ============================================================
-- Szereg makro ma DWIE osie czasu i obie są nośne:
--   observation_date - okres, który liczba opisuje (marzec 2015),
--   realtime_start   - dzień, od którego ta liczba BYŁA opublikowaną wartością.
-- FRED rewiduje wstecz, więc ta sama obserwacja istnieje wielokrotnie: pierwszy
-- odczyt, rewizja z kolejnego miesiąca, rewizja benchmarkowa po latach.
-- Trzymanie tylko najnowszej zamienia cechę makro w look-ahead — model dostałby
-- to, czym marzec 2015 OKAZAŁ się być, a nie to, co wtedy było wiadome.
CREATE TABLE IF NOT EXISTS macro_data.macro_observations (
    series           TEXT NOT NULL,
    observation_date DATE NOT NULL,
    -- Część klucza, NOT NULL: NULL w kluczu głównym nie jest porównywalny,
    -- więc taki wiersz duplikowałby się przy każdym backfillu.
    realtime_start   DATE NOT NULL,
    value            DOUBLE PRECISION NOT NULL,
    source           TEXT NOT NULL DEFAULT 'fred',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series, observation_date, realtime_start)
);

-- Odczyt as-of idzie po (series, realtime_start, observation_date): najpierw
-- "co mogłem zobaczyć w dniu D", potem "który okres jest w tym najnowszy".
CREATE INDEX IF NOT EXISTS idx_macro_asof
    ON macro_data.macro_observations (series, realtime_start, observation_date DESC);

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
