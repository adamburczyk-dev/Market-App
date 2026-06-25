# Trading System — Microservices Architecture

## Project overview

Production-grade algorithmic trading system. 13 independent Python microservices communicating
via NATS JetStream (events) and HTTP (request/response).

**Key docs:**
- **Project context/status/direction: this file** — see "Project status & direction" below (single source of truth I read every session)
- Full 24-week development plan: `Plan_Rozwoju_Systemu_Tradingowego_2.md` (repo root)
- Framework supplement — 12 components (risk envelope, drift/decay monitors, cost filter, regime allocator, …): `docs/framework_supplement.md`
- ML/AI integration plan (serwisy 10–13, feature tiers, model stacks): not yet a standalone doc; initial contracts live in
  `shared/trading-common` (schemas + events). Write `docs/ml_integration_plan.md` before deep work on serwisy 10–13.

## Project status & direction

> Single living context block. Read this first every session. Keep the progress log append-only.
> If a fresh analysis surfaces new bugs or improvement ideas, **propose them here and to the user** —
> do not silently proceed.

**Phase:** 1 — Foundation. **Reality check: there is a priority inversion** (Week-19+ framework work
landed before the Week-2 foundation).

**Verified ground truth** (run locally on Python 3.12 — not from memory):
- `shared/trading-common`: 126 tests green, `ruff` + `mypy --strict` clean. Contracts present:
  `OHLCVBar`, `TradingSignal`, `PortfolioMetrics`, ML/AI contracts (`CompanyProfile`,
  `FinancialStatements`, `MacroSnapshot`, `SentimentSnapshot`, `FeatureVector`), full `EventType`
  set incl. ML/AI extension, `RiskEnvelope`.
- All 9 service skeletons: `/health` `/ready` `/metrics` green.
- Framework-supplement components implemented + unit-tested but **NOT wired into FastAPI/NATS**
  (orphaned — they don't run at runtime): feature-engine calculators (`vol_regime`, `earnings_decay`,
  `cross_asset`), strategy (`decay_monitor`, `cost_filter`, `adaptive_weights`), risk-mgmt
  (`adaptive_sizing`, `regime_allocator`), ml-pipeline (`drift_detector`), backtest
  (`continuous_validation`).
- `market-data` is now **functionally implemented** (Direction #1 done): Yahoo + Alpha Vantage
  fetchers, async storage (SQLAlchemy/asyncpg, idempotent upsert), Redis cache (in-memory fallback),
  `MarketDataUpdatedEvent` publishing over **NATS JetStream** (msg-id dedup), wired through FastAPI
  lifespan. 28 tests green; verified end-to-end (fetch → store → read) incl. a lifespan smoke with
  all backends down.

**Direction (where the project should go, in order):**
1. ✅ **DONE — Foundation:** `market-data` fetch → validate → store → cache → publish event
   (NATS **JetStream**, `Nats-Msg-Id` dedup). Next refinements (deferred, non-blocking): bulk
   `ON CONFLICT` insert instead of per-row merge, a scheduled/periodic fetch job.
2. **Wire the orphaned components** into their services (API endpoints + NATS publishers/subscribers)
   so the already-tested logic runs.
3. **Build serwisy 10–13** (fundamental-data, macro-data, company-classifier, signal-aggregator)
   against the now-existing shared contracts. When `signal-aggregator` exists, move
   `adaptive_weights.py` + `cost_filter.py` there from `strategy/` (their spec home — framework_supplement B3/B4).
4. **Contracts-first** always: extend `shared/trading-common` before adding any cross-service type.

**Known issues / tech debt** (propose a fix when you touch the area):
- [P1] Orphaned components: tested but unreachable at runtime — wire incrementally (Direction #2).
- [P2] `adaptive_weights.py` / `cost_filter.py` sit in `strategy/` but belong in `signal-aggregator/` (not created yet).
- [P2] No `docs/ml_integration_plan.md`; serwisy 10–13 reference it conceptually. Initial contracts now in code — write the doc before deep ML work.
- [P2] README "Status infrastruktury (zweryfikowany)" cannot be verified without Docker (none in sandbox/CI) — treat as *expected*, not *verified*.
- [P3] `infrastructure/terraform/` is referenced in README but absent (planned).
- [env] Sandbox default `python3` is 3.11; project requires 3.12 → use `python3.12` for local installs/tests.
- [env] CI runs only on push to `main`/`develop` and PR→`main`; feature branches (`claude/*`) get no CI until a PR — verify locally before pushing.
- [env] Docker CLI + daemon are available (start `dockerd` as root if the socket is missing), but
  egress policy can block Docker Hub (seen: 403 on `production.cloudfront.docker.com`) → image
  pulls / `docker compose up` may fail. It's per-session network policy; verify containers in a
  session whose policy allows Docker Hub.

**Progress log (append-only):**
- 2026-06-25 — Full repo audit: verified tests/lint/types green on 3.12; catalogued the priority
  inversion and the orphaned framework components.
- 2026-06-25 — Consistency sprint: added 5 missing shared schemas + 7 ML/AI `EventType` values &
  their event classes (+22 tests → 126 green); replaced the dead high/low field validators with a
  `model_validator`; consolidated all project context into this CLAUDE.md section (removed
  `docs/PROJECT_STATUS.md` and `docs/git-workflow-guide.md`); fixed dangling doc references.
  Merged to `main`.
- 2026-06-25 — Direction #1 (market-data implementation): fetchers (Yahoo via yfinance, Alpha
  Vantage via aiohttp, fallback chain), `OHLCVRepository` (async, idempotent merge upsert),
  Redis cache + in-memory fallback, `NatsPublisher`/`NullPublisher`, `MarketDataService`
  orchestration, real FastAPI routes (`GET /ohlcv`, `POST /fetch`, `GET /symbols`) wired via
  lifespan with graceful degradation. Changed `init-db.sql` ohlcv PK to natural
  `(symbol, interval, ts)` to enable idempotent upserts. 27 tests green; ruff + mypy clean.
- 2026-06-25 — JetStream: `market-data` now publishes `MarketDataUpdatedEvent` via NATS **JetStream**
  (jetstream context + idempotent `ensure_stream` creating the `MARKET_DATA` stream + `Nats-Msg-Id`
  dedup header) instead of core publish. +1 test (28 green). Live container round-trip NOT run this
  session: Docker daemon is up but Docker Hub egress is policy-blocked (403) — verified via unit
  test against the nats-py JetStream API. Run the real round-trip in a Docker-Hub-allowed session.

**Next:** Direction #2 — wire an orphaned component into its service (suggest starting with
feature-engine: expose calculators via an endpoint + subscribe to `MarketDataUpdatedEvent`,
publish `FeaturesReadyEvent`).

## Architecture rules (non-negotiable)

- Every bounded context is a separate service with its own Dockerfile and `pyproject.toml`
- Inter-service communication: NATS JetStream for events, HTTP for queries
- Every service MUST expose `/health`, `/metrics` (Prometheus), and use structlog
- The only way to run the system is `docker compose up` — never bare-metal
- Helm charts must stay in sync with docker-compose definitions
- No hardcoded secrets — always `${VAR:?required}` in compose, pydantic-settings in code
- Define contracts first (Pydantic schema, event type, API endpoint), implement second

## Services

### Core (9 original)

| Service | Port | Purpose |
|---------|------|---------|
| market-data | 8001 | OHLCV fetch, validation, TimescaleDB storage |
| feature-engine | 8002 | Technical indicators, Tier 1–3 feature computation |
| strategy | 8003 | Strategy definitions, signal generation |
| backtest | 8004 | Backtesting engine, walk-forward optimization |
| ml-pipeline | 8005 | ML training, inference, model registry (MLflow) |
| risk-mgmt | 8006 | Position sizing, portfolio optimization |
| execution | 8007 | Paper/live trading, order management |
| notification | 8008 | Alerts: Telegram, email, Slack |
| dashboard | 8501 | UI: Streamlit or React |

### ML/AI Extension (4 new — initial contracts in `trading-common`; full plan TBD)

| Service | Port | Purpose | Priority |
|---------|------|---------|----------|
| fundamental-data | 8009 | SEC EDGAR (10-Q/10-K/Form4), FMP earnings revisions, Piotroski F-Score | Weeks 3–4 |
| macro-data | 8010 | FRED yield curve, credit spreads, PMI, CPI, regime detection | Weeks 3–4 |
| company-classifier | 8011 | Company profile → model stack routing | Week 5 |
| signal-aggregator | 8012 | Combines ML + rules-based + macro regime signals | Week 19 |

Infrastructure: PostgreSQL 16 + TimescaleDB, Redis 7, NATS JetStream, Prometheus + Grafana + Loki, Traefik (API Gateway)

## Service file structure

```
services/{name}/
├── src/
│   ├── main.py          # FastAPI app + lifespan
│   ├── config.py        # pydantic-settings
│   ├── api/
│   │   ├── __init__.py  # APIRouter aggregation
│   │   ├── routes.py
│   │   └── deps.py      # FastAPI dependencies
│   ├── core/            # Business logic
│   ├── events/
│   │   ├── publisher.py # NATS publish
│   │   └── subscriber.py# NATS subscribe
│   └── models/
│       ├── db.py        # SQLAlchemy ORM
│       └── schemas.py   # Re-export from trading_common
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Shared library

`shared/trading-common` — pip-installable package.
- `trading_common.schemas` — Pydantic models shared across services
  (OHLCVBar, TradingSignal, PortfolioMetrics, CompanyProfile, FinancialStatements,
  MacroSnapshot, SentimentSnapshot, FeatureVector — defined in `schemas.py`)
- `trading_common.events` — Event definitions
  (MarketDataUpdatedEvent, SignalGeneratedEvent, FundamentalsUpdatedEvent,
  MacroUpdatedEvent, SentimentUpdatedEvent, CompanyClassifiedEvent — defined in `events.py`)
- Install in each service: `pip install -e ../../shared/trading-common`

Service A NEVER imports directly from Service B. Shared types go in trading-common.

## Tech stack

- Python 3.12, FastAPI, SQLAlchemy 2.x (async), asyncpg
- NATS JetStream (`nats-py`), Redis 7
- `pyproject.toml` + hatchling — NOT `setup.py` or `requirements.txt`
- ruff (lint + format) — NOT flake8/black separately
- mypy for type checking
- pytest + pytest-asyncio + httpx for testing
- structlog (JSON in prod, ConsoleRenderer in dev)
- prometheus-client + prometheus-fastapi-instrumentator
- tenacity for retries
- pydantic-settings for configuration
- PyTorch — NOT TensorFlow (for ml-pipeline-svc)
- MLflow — model registry and experiment tracking

## Commands

```bash
# Dev environment
make up              # docker compose up -d
make down            # docker compose down
make build           # docker compose build
make test            # run all tests
make lint            # ruff check .

# Per-service
make build-market-data
cd services/market-data && pytest tests/ -v

# Kubernetes
make helm-template   # render Helm chart
make helm-install    # deploy to K8s
```

## Code conventions

- Language: conversation in Polish, all code/comments/docstrings/variables in English
- File/folder names: English, kebab-case for services
- No deprecated APIs: no `setup.py`, no `version: '3.8'` in compose, no `fillna(method='ffill')`
- Time-series data: ALWAYS time-series split, NEVER random train/test split
- If something should be an event, propose an event — don't default to synchronous HTTP
- PyTorch over TensorFlow for ML services
- ML labeling: always use Triple Barrier Method (López de Prado) — not fixed-horizon labels
- Feature ranking: always cross-sectional percentile rank, not raw values (López de Prado)

---

## Claude / Cowork Integration

> This section describes how Claude (via the Cowork desktop tool) fits into the development workflow.
> Claude has internet access, a sandboxed Linux shell (Python 3.12, pip), and can read/write files.

### What Claude can do autonomously

| Task | How | When to use |
|------|-----|-------------|
| Fetch historical OHLCV data | `yfinance` / `stooq` → CSV/Parquet | Bootstrapping training dataset before market-data-svc is ready |
| Download SEC EDGAR filings | `edgartools` + SEC EDGAR API | Seeding fundamental-data-svc test fixtures |
| Fetch FRED macro data | `fredapi` Python library | Seeding macro-data-svc test fixtures |
| Generate new service skeleton | Bash + Write tools | Scaffolding a new microservice from the standard template |
| Run backtests on historical data | Python in sandbox | Quick strategy validation before wiring into backtest-svc |
| Analyze model performance | Python + matplotlib | Reviewing ML results: feature importance, confusion matrix, Sharpe |
| Review code against checklist | Read + analysis | Pre-PR review against the review checklist below |
| Generate test fixtures | Python | Creating realistic mock data for pytest fixtures |
| Write pyproject.toml / Dockerfile | Write tool | Scaffolding new service boilerplate |

### What Claude CANNOT do

- Access live market data in real-time (15–20 min delay from public sources)
- Submit orders to any broker
- Access private APIs without keys provided in the conversation
- Run `docker compose` (no Docker daemon in sandbox)
- Access the running system at runtime

### Suggested Claude workflow per phase

**Phase 1–2 (Weeks 1–4) — Infrastructure & Data:**
```
Ask Claude to: "Fetch 3 years of daily OHLCV for [symbols] using yfinance and save to CSV"
Ask Claude to: "Generate the market-data-svc skeleton following CLAUDE.md conventions"
Ask Claude to: "Write pytest fixtures with 500 realistic OHLCVBar rows for market-data tests"
```

**Phase 3 (Weeks 5–12) — Strategies & Backtesting:**
```
Ask Claude to: "Backtest this SMA crossover strategy on AAPL 2020–2024 data"
Ask Claude to: "Calculate Sharpe, Sortino, max drawdown, and Calmar for this equity curve"
Ask Claude to: "Generate walk-forward analysis split for 3-year window, 6-month step"
```

**Phase 4 (Weeks 13–18) — ML Pipeline:**
```
Ask Claude to: "Apply Triple Barrier labeling to this OHLCV DataFrame (pt=2xATR, sl=1xATR)"
Ask Claude to: "Train XGBoost on this FeatureVector dataset with time-series cross-validation"
Ask Claude to: "Plot feature importance and SHAP values for this trained model"
Ask Claude to: "Fetch last 8 quarters of 10-Q for AAPL via EDGAR and return as FinancialStatements"
```

**Phase 5 (Weeks 22–24) — Production:**
```
Ask Claude to: "Review this Dockerfile against the multi-stage pattern in CLAUDE.md"
Ask Claude to: "Generate Prometheus alerting rules for: drawdown > 5%, order fill rate < 90%"
Ask Claude to: "Write a weekly scheduled task that fetches macro data and saves a snapshot"
```

### Data bootstrap instructions

Before ml-pipeline-svc is ready, use Claude to seed training data:

```
"Fetch daily OHLCV for [AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, XOM, SPY]
 from 2015-01-01 to today using yfinance.
 Save each symbol as a separate Parquet file in /data/raw/ohlcv/{symbol}.parquet.
 Include: open, high, low, close, volume, vwap (if available).
 Validate: no gaps > 5 trading days, no negative prices."
```

---

## Risk rules (non-negotiable)

- Every signal MUST pass through `RiskEnvelope` (trading-common) before publishing
- No order without `stop_loss` — enforce in `TradingSignal` validation
- Circuit breaker events MUST be subscribed by ALL services that generate or execute orders
- Paper trading MUST run minimum 30 days with positive Sharpe before any live capital
- Max 5% portfolio per position, max 80% total exposure — never override without human approval
- Daily loss > 5% → automatic trading halt until next day
- Drawdown > 15% → flatten all positions, require human restart
- Every strategy MUST have walk-forward OOS validation before activation
- No strategy goes live without backtested Sharpe > 0.5 on OOS data
- Position sizing is drawdown-adaptive: scales linearly from 2% risk at DD=0% to 0% at DD=15%
- Regime-aware allocation: CRISIS → max 15% equity exposure, CONTRACTION → max 35%

## Monitoring requirements (every service)

- ML models: daily drift check (PSI + rolling Sharpe), weekly full DriftReport
- Strategies: daily decay check via `StrategyDecayMonitor`, auto-probation/deactivation
- Portfolio: real-time drawdown tracking, circuit breaker armed 24/7
- Prometheus alerts:
  - `drawdown > 8%` → WARNING
  - `drawdown > 15%` → CRITICAL
  - `model drift PSI > 0.2` → WARNING
  - `strategy Sharpe < 0 (90d rolling)` → CRITICAL
  - `daily loss > 3%` → WARNING
  - `order fill rate < 90%` → WARNING
- Walk-forward revalidation: weekly (Saturday) for all active strategies
- Full design details: `docs/framework_supplement.md`

## Extended event types

Additional events beyond the original 10 (add to `EventType` enum when implementing):
- `CIRCUIT_BREAKER_TRIGGERED` = `"risk.circuit_breaker"`
- `MODEL_DRIFT_DETECTED` = `"ml.drift_detected"`
- `MODEL_RETRAINED` = `"ml.model_retrained"`
- `STRATEGY_STATUS_CHANGED` = `"strategy.status_changed"`
- `REGIME_CHANGED` = `"macro.regime_changed"`
- `FUNDAMENTALS_UPDATED` = `"fundamentals.updated"`
- `MACRO_UPDATED` = `"macro.updated"`
- `SENTIMENT_UPDATED` = `"sentiment.updated"`
- `COMPANY_CLASSIFIED` = `"company.classified"`
- `FEATURES_READY` = `"features.ready"`
- `SIGNAL_AGGREGATED` = `"signal.aggregated"`

---

## Review checklist

When modifying or creating code, verify:
- [ ] Does not cross service boundaries (no direct imports between services)
- [ ] Shared schemas live in `trading-common`, not duplicated
- [ ] No hardcoded secrets
- [ ] Has `/health` and `/metrics` endpoints
- [ ] Uses `pyproject.toml`
- [ ] Publishes/subscribes relevant NATS events
- [ ] Includes structured logging (structlog)
- [ ] Has tests (unit + at least one integration test)
- [ ] ML: time-series split used, NOT random split
- [ ] ML: features are cross-sectional rank-transformed where applicable
- [ ] New service: Dockerfile follows multi-stage pattern
- [ ] New service: added to docker-compose.yml and Helm chart
- [ ] Signal-generating code: passes signals through `RiskEnvelope.check_signal()`
- [ ] New strategy: has `StrategyDecayMonitor` integration and OOS walk-forward validation
- [ ] ML model: has `DriftDetector` integration with daily PSI check
- [ ] Trade signals: filtered through `CostAwareFilter` before execution
