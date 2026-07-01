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

**Phase:** 1 — Foundation. The earlier priority inversion is **resolved**: the foundation was built
and the framework components wired into a working **end-to-end paper-trading loop** (market-data →
feature-engine → strategy → risk-mgmt → execution → portfolio feedback) plus backtest + ml-pipeline
monitoring, notification alerting, and a dashboard BFF over the HTTP APIs. **All 9 core services are
now functionally implemented** (no skeletons left).

**Verified ground truth** (run locally on Python 3.12 — not from memory):
- `shared/trading-common`: 134 tests green, `ruff` + `mypy --strict` clean. Contracts present:
  `OHLCVBar`, `TradingSignal`, `PortfolioMetrics`, ML/AI contracts (`CompanyProfile`,
  `FinancialStatements`, `MacroSnapshot`, `SentimentSnapshot`, `FeatureVector`), full `EventType`
  set incl. ML/AI extension + `STRATEGY_REVALIDATED` (backtest→strategy), `RiskEnvelope`.
- All 9 core services functionally implemented (`/health` `/ready` `/metrics` green; no skeletons left).
- Framework-supplement components still **orphaned** (tested but not wired into FastAPI/NATS):
  feature-engine (`earnings_decay`, `cross_asset`), strategy (`adaptive_weights` only — belongs in
  signal-aggregator). (`decay_monitor`+`cost_filter` now wired into strategy;
  `adaptive_sizing`+`regime_allocator` now wired into risk-mgmt; `continuous_validation` wired into
  backtest; `drift_detector` wired into ml-pipeline; `vol_regime` is VIX/market-wide — it belongs in
  the macro/regime context, not single-symbol realized vol.)
- `market-data` is now **functionally implemented** (Direction #1 done): Yahoo + Alpha Vantage
  fetchers, async storage (SQLAlchemy/asyncpg, idempotent upsert), Redis cache (in-memory fallback),
  `MarketDataUpdatedEvent` publishing over **NATS JetStream** (msg-id dedup), wired through FastAPI
  lifespan. 28 tests green; verified end-to-end (fetch → store → read) incl. a lifespan smoke with
  all backends down.
- `feature-engine` is now **functionally implemented** (Direction #2 done): Tier-1 feature
  computation from OHLCV (numpy; raw per-symbol values), HTTP query to market-data,
  NATS **JetStream** subscriber on `market_data.updated` → compute → publish `FeaturesReadyEvent`,
  FastAPI routes (`POST /compute/{symbol}`, `GET /features/{symbol}`, `GET /features`,
  `GET /ranked`). 78 tests green; verified end-to-end on a live nats-server.
- `strategy` is now **functionally implemented** (Direction #2): JetStream subscriber on
  `features.ready` → fetch ranked+raw features from feature-engine (HTTP) → **momentum-on-ranks**
  rule → `TradingSignal` (vol-agnostic % stop) → **`RiskEnvelope`** (SL-enforcing; step-7 sizing
  treated as advisory) → **`CostAwareFilter`** → publish `SignalGeneratedEvent` (now carries
  `stop_loss`/`take_profit`). `StrategyDecayMonitor` gates output (inactive → suppress; `POST /decay`
  re-evaluates and emits `StrategyStatusChangedEvent`). 86 tests green; live-verified the full chain
  (FeaturesReady → BUY → RiskEnvelope → SignalGenerated in the SIGNALS stream).
- `risk-mgmt` is now **functionally implemented** (Direction #2): JetStream subscriber on
  `signal.generated` → **`PositionSizer`** (`adaptive_sizing` drawdown-scaled risk budget +
  `regime_allocator` exposure/sector caps + 5% position cap → size-down) → publish
  **`OrderRequestedEvent`** (new contract, risk→execution). **Circuit Breaker** armed 24/7
  (`CircuitBreaker`: YELLOW dd>8% / RED daily-loss>5% halt / BLACK dd>15% flatten) → publishes
  `CircuitBreakerTriggeredEvent` and blocks new orders when tripped. `PortfolioState`
  (updatable via `POST /portfolio`) is now **Redis-persisted** (`RedisStateRepository` snapshot on
  every update; `NullStateRepository` fallback) — on startup `restore()` reloads the snapshot and
  **re-derives** the breaker level, so a tripped halt survives a restart. Also subscribes to
  **`RegimeChangedEvent`** (`macro.regime_changed`, from macro-data) → `update_portfolio(regime)` so the
  macro regime auto-drives the RegimeAllocator exposure caps (no manual push needed). Routes `/portfolio`,
  `/circuit-breaker`, `/signal`. 97 tests green; live-verified (SignalGenerated → sized OrderRequested;
  breaker RED halts new orders; tripped breaker survives a restart via real Redis; a real
  `macro.regime_changed` event flips the regime → tightens the cap).
- `execution` is now **functionally implemented** (paper trading — **closes the loop**): JetStream
  subscriber on `order.requested` → `PaperBroker` simulates the fill → publish `OrderFilledEvent` →
  push portfolio metrics (equity/exposure/drawdown/daily-loss) back to risk-mgmt over HTTP
  (`POST /portfolio`), so fills drive sizing + the circuit breaker. `PaperBroker` (cash/positions,
  peak-equity drawdown, mark-to-fill) is now **Redis-persisted** (`RedisBrokerRepository` snapshot on
  every fill/mark; `NullBrokerRepository` fallback) — `restore()` reloads cash/positions on startup.
  Routes `/portfolio`, `/positions`, `/execute`; real `/ready`. 30 tests green; live-verified
  (OrderRequested → OrderFilled → portfolio fed back; broker cash/positions survive a restart via real Redis).
- `backtest` is now **functionally implemented** (Direction #2): wires the orphaned
  `continuous_validation` (`ContinuousWalkForward`, abstract) to a real **momentum backtest engine**
  (`core/engine.py`: numpy time-series long/flat momentum, no look-ahead, per-turn costs →
  Sharpe/maxDD/return/trades; `start_index` measures the OOS tail with IS warm-up). `EngineWalkForward`
  implements `_run_backtest` over the trailing OOS window; `BacktestService` pulls OHLCV from
  market-data (HTTP) → runs backtest/revalidation → publishes `BacktestCompletedEvent` and the new
  `StrategyRevalidatedEvent` (backtest **recommends** active/probation/deactivate; strategy **owns** the
  status). Routes `POST /run`, `POST /revalidate`; real `/ready` gates on market-data. 39 tests green;
  ruff + mypy clean; live-verified on a real `nats-server` (both events land in the `BACKTEST` stream
  and read back).
- `ml-pipeline` is now **functionally implemented** (Direction #2 — **last orphaned component**):
  wires `drift_detector` (`DriftDetector`: PSI + KS prediction-shift + rolling-Sharpe/accuracy decay)
  into the runtime. `ModelRegistry` (in-memory baseline store — placeholder for MLflow) holds each
  model's reference feature distributions + baseline Sharpe; `MLPipelineService.check_drift` computes
  per-feature PSI vs the baseline → `DriftReport` → publishes `ModelDriftDetectedEvent` only when
  actionable (drift_type feature_drift/performance_decay/accuracy_decay/prediction_shift; severity
  critical on retrain, warning on investigate). Routes `POST /models/{id}/baseline`,
  `POST /models/{id}/drift`, `GET /models`; real `/ready` (NATS). publisher + `ensure_stream(ML,
  ["ml.>"])`. 35 tests green; ruff + mypy clean; live-verified on a real `nats-server`
  (`ml.drift_detected` lands in the `ML` stream and reads back).
- `notification` is now **functionally implemented** (closes the monitoring loop — first multi-stream
  consumer): durable `EventSubscriber`s on the 4 alert-worthy events across their streams —
  `CircuitBreakerTriggeredEvent` (RISK), `OrderFilledEvent` (ORDERS), `StrategyRevalidatedEvent`
  (BACKTEST), `ModelDriftDetectedEvent` (ML). `core/alerts.py` maps each event → `Alert`
  (severity-graded); `NotificationService.dispatch` applies a min-severity gate, keeps a recent-alerts
  ring buffer, and fans out to channels with per-channel failure isolation. `core/channels.py`:
  `LogChannel` (always on), `SlackChannel`/`TelegramChannel` (HTTP, built only when configured — log-only
  otherwise). Routes `GET /channels`, `GET /alerts/recent`, `POST /test-alert`; real `/ready` (NATS);
  `ensure_stream` for all 4 source streams (start-order independent). 28 tests green; ruff + format +
  mypy clean; live-verified on a real `nats-server` (all 4 events → 4 correctly-graded alerts). Email/SMTP
  channel + a scheduler-driven digest are follow-ups.
- `dashboard` is now **functionally implemented** (last skeleton — all 9 core services done): a
  **backend-for-frontend** (FastAPI, not Streamlit — keeps `/health` `/ready` `/metrics` + structlog +
  the standard skeleton). `HttpDashboardSource` fans out read-only GETs to risk-mgmt (`/portfolio`,
  `/circuit-breaker`), execution (`/portfolio`, `/positions`), notification (`/alerts/recent`),
  ml-pipeline (`/models`); `DashboardService.overview` gathers them concurrently and is **partial-tolerant**
  (a down upstream → `sources[name]="unavailable"`, the rest still renders). Routes `GET /overview`
  (aggregated JSON) + `GET /ui` (self-contained HTML page, vanilla-JS poll, no build step); `GET /`
  redirects to the UI. real `/ready` reports per-source reachability (always 200 — the BFF tolerates
  missing upstreams). 18 tests green; ruff + format + mypy clean; **live-verified** against real
  risk-mgmt + execution (uvicorn): the real `HttpDashboardSource` aggregated their live state over HTTP
  while notification + ml-pipeline (down) showed "unavailable".
- `macro-data` (**serwis 10 — first Direction #3 service, built from scratch**): FRED macro indicators
  + rule-based market-regime detection. `core/regime.py` (`classify_regime` — severity-ordered rules on
  yield-curve inversion / BAA credit spread / PMI → the 5 `MacroRegime` values risk-mgmt's
  RegimeAllocator already consumes; tolerant of missing inputs), `core/fred_client.py` (`FredClient` —
  httpx fetch of T10Y2Y/BAA10Y/UNRATE/FEDFUNDS, disabled + None when no `FRED_API_KEY`),
  `core/service.py` (`MacroDataService.refresh` — merge FRED + manual overrides → classify → publish
  `MacroUpdatedEvent` always + `RegimeChangedEvent` only on a real transition). Routes `GET /snapshot`,
  `GET /regime`, `POST /refresh`; real `/ready` (NATS); publisher + `ensure_stream(MACRO, ["macro.>"])`.
  New service scaffold (Dockerfile, pyproject, compose port 8010, Helm values entry). 41 tests; ruff +
  format + mypy clean; live-verified on a real `nats-server` (expansion→crisis → 2×`macro.updated` +
  1×`macro.regime_changed` in the `MACRO` stream). **risk-mgmt now subscribes to `RegimeChangedEvent`**,
  so the regime auto-drives the exposure caps (macro→risk loop closed).
- `fundamental-data` (**serwis 9 — Direction #3, built from scratch**): SEC EDGAR annual fundamentals +
  (partial) Piotroski F-Score. `core/piotroski.py` (`compute_f_score` — 7 of the classic 9 signals
  computable from the `FinancialStatements` contract: 3 current-period profitability + 4 trend signals;
  current-ratio Δ and share-issuance omitted, documented, until the schema carries balance-sheet detail;
  each signal fails conservatively on missing/degenerate inputs), `core/edgar_client.py` (`EdgarClient` —
  ticker→CIK via company_tickers.json, XBRL `companyconcept` per us-gaap tag → annual `FinancialStatements`;
  disabled + [] when no `SEC_USER_AGENT`), `core/service.py` (`FundamentalDataService.refresh` from EDGAR /
  `ingest` posted statements → score → store latest-per-symbol → publish `FundamentalsUpdatedEvent`).
  Routes `GET /fundamentals[/{symbol}]`, `POST /refresh/{symbol}`, `POST /statements`; real `/ready` (NATS);
  publisher + `ensure_stream(FUNDAMENTALS, ["fundamentals.>"])`. Full scaffold (compose port 8009, Helm
  `fundamentalData` values entry). 27 tests; ruff + format + mypy clean; live-verified on a real
  `nats-server` (ingest → `fundamentals.updated` in the `FUNDAMENTALS` stream; F-score 7/7 on an
  improving firm).
- `company-classifier` (**serwis 11 — Direction #3, built from scratch**): `CompanyProfile` → investment
  style + model-stack routing (pure compute, no external API). `core/classifier.py` (`classify` — style
  scored from valuation/growth metrics: growth signals (rev/earnings growth, rich P/E, no dividend) vs
  value signals (cheap P/E & P/B, dividend); with no metrics falls back to a **sector prior**, then blend.
  `cap_tier` mega/large/mid/small/micro; `route_model_stack(style, tier)` → e.g. `growth_largecap_v1`),
  `core/service.py` (`CompanyClassifierService.classify` — enriches the profile with style + model_stack +
  `as_of`, stores latest-per-symbol, publishes `CompanyClassifiedEvent`). Routes `GET /companies[/{symbol}]`,
  `POST /classify`; real `/ready` (NATS); publisher + `ensure_stream(COMPANY, ["company.>"])`. Full scaffold
  (compose port 8011, Helm `companyClassifier`). 25 tests; ruff + format + mypy clean; live-verified on a
  real `nats-server` (classify NVDA → `company.classified` with `growth_largecap_v1` in the `COMPANY` stream).

**Direction (where the project should go, in order):**
1. ✅ **DONE — Foundation:** `market-data` fetch → validate → store → cache → publish event
   (NATS **JetStream**, `Nats-Msg-Id` dedup). Next refinements (deferred, non-blocking): bulk
   `ON CONFLICT` insert instead of per-row merge, a scheduled/periodic fetch job.
2. ✅ **DONE — Wire the orphaned components** into their services (API endpoints + NATS
   pub/sub). feature-engine, strategy, risk-mgmt, backtest, ml-pipeline all wired. (Leftover specs —
   feature-engine `earnings_decay`/`cross_asset`, strategy `adaptive_weights` — belong in later
   services, not the 7 core runtime paths; tracked under tech debt.)
3. ⏳ **IN PROGRESS — Build serwisy 10–13** (fundamental-data, macro-data, company-classifier,
   signal-aggregator) against the now-existing shared contracts. ✅ **macro-data** (10) +
   **fundamental-data** (9) + **company-classifier** (11) done. **Only signal-aggregator (12) remains.**
   When `signal-aggregator` exists, move `adaptive_weights.py` + `cost_filter.py` there from
   `strategy/` (their spec home — framework_supplement B3/B4).
4. **Contracts-first** always: extend `shared/trading-common` before adding any cross-service type.

**Known issues / tech debt** (propose a fix when you touch the area):
- [P1 ✅ done] Orphaned components wired (Direction #2 complete): feature-engine + strategy +
  risk-mgmt + backtest + ml-pipeline. Leftover specs (`earnings_decay`, `cross_asset`,
  `adaptive_weights`) belong in later services (signal-aggregator / macro), not the core runtime.
- [P1 ✅ done] `RiskEnvelope` step-7 removed — the envelope is now a pure gate; **sizing** lives in
  risk-mgmt (`PositionSizer`: drawdown-adaptive risk budget + regime cap + 5% position cap → size-down).
- [P2] `OrderRequestedEvent` (risk→execution) carries symbol/side/qty/price/SL/TP + strategy_name;
  revisit if execution needs more (e.g. order type, TIF).
- [P3 ✅ mostly done] Portfolio state (`PortfolioState` in risk-mgmt) and broker state (`PaperBroker`
  in execution) are now **Redis-persisted** (snapshot on every mutation; `restore()` on startup;
  Null*-Repository fallback when Redis is down). Both still single-instance (snapshot, not an event
  log) and the circuit-breaker auto-clears (a real system needs manual reset out of BLACK).
  feature-engine's `FeatureStore` is likewise Redis-backed (in-memory fallback) but **without**
  startup restore — features recompute from market-data, so cold-start loss is acceptable.
- [P3 ✅ done] strategy now queries risk-mgmt's **live** portfolio (`GET /portfolio`) for the
  RiskEnvelope gate, falling back to its static placeholder only when risk-mgmt is unreachable.
- [P1 ✅ done] Cross-sectional ranking: feature-engine exposes universe-level percentile ranks via
  `GET /ranked` (+ `/ranked/{symbol}`) using `cross_sectional_rank`. Raw vectors still feed the store;
  strategy/ML must consume the **ranked** vectors. (Snapshot = latest-per-symbol; align timestamps later.)
- [P2 ✅ mostly done] Robustness: subscriber has `max_deliver` + poison-`term`/transient-`nak` (D1);
  `/ready` checks deps — market-data gates on DB, feature-engine on NATS (D2); FeatureStore is
  Redis-backed with in-memory fallback via an async store interface (D3). Still open: the **push**
  consumer doesn't load-balance — use a pull / queue-group consumer for true multi-replica HA.
- [P2] `adaptive_weights.py` / `cost_filter.py` sit in `strategy/` but belong in `signal-aggregator/` (not created yet).
- [P2] No `docs/ml_integration_plan.md`; serwisy 10–13 reference it conceptually. Initial contracts now in code — write the doc before deep ML work.
- [P2] README "Status infrastruktury (zweryfikowany)" cannot be verified without Docker (none in sandbox/CI) — treat as *expected*, not *verified*.
- [P3] `infrastructure/terraform/` is referenced in README but absent (planned).
- [P2] Helm chart lags: `values.yaml` lists every service but `templates/` has a deployment only for
  `market-data`. The other 12 services (incl. macro-data / fundamental-data / company-classifier) have
  values entries but **no Deployment template** — a generic templated `Deployment`/`Service`/`HPA`
  ranging over the services map is the fix. Pre-existing; flagged since the rule requires Helm↔compose sync.
- [env] Sandbox default `python3` is 3.11; project requires 3.12 → use `python3.12` for local installs/tests.
- [env] CI runs only on push to `main`/`develop` and PR→`main`; feature branches (`claude/*`) get no CI until a PR — verify locally before pushing.
- [env] Docker CLI + daemon are available (start `dockerd` as root if the socket is missing). Under
  the **Trusted** egress policy, Docker Hub *registry* hosts are allowlisted but NOT the blob CDN
  Docker actually redirects to (`production.cloudfront.docker.com` → 403; the allowlist only has the
  Cloudflare variant `production.cloudflare.docker.com`). → `docker pull` / `docker compose up` fail
  under Trusted. Fix: edit the environment's **Network access** → **Full** (or **Custom** + add
  `production.cloudfront.docker.com`), then start a new session.
  To verify NATS/JetStream **without Docker** (Go module proxy is allowlisted):
  `GOSUMDB=off go install github.com/nats-io/nats-server/v2@v2.10.22` then run `nats-server -js`.

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
- 2026-06-25 — JetStream round-trip **verified for real** against a live `nats-server` (installed via
  `go install`, no Docker needed): the production `NatsPublisher` + `ensure_stream` created the
  `MARKET_DATA` stream, published, deduplicated a re-published `Nats-Msg-Id` (duplicate kept seq=1,
  stream count stayed 2), and a pull consumer read both messages back. Docker-based run still blocked
  by the Trusted egress (cloudfront blob host 403) — see the `[env]` note for the fix.
- 2026-06-25 — Added `scripts/verify-jetstream.py` + `make verify-jetstream` (spawns an isolated
  `nats-server -js`, runs the real publisher round-trip incl. dedup; `--url` for a running NATS).
- 2026-06-25 — Direction #2 (feature-engine wired): `compute_feature_vector` (Tier-1 numpy features +
  `vol_regime` reuse), `HttpMarketDataClient` (queries market-data over HTTP), JetStream
  `MarketDataSubscriber` on `market_data.updated` → compute → publish `FeaturesReadyEvent`,
  `FeatureStore`, FastAPI routes, lifespan with graceful degradation. +11 tests (61 green); ruff +
  mypy clean. Verified live on a local `nats-server`: published `MarketDataUpdatedEvent` → subscriber
  computed 11 features → `FeaturesReadyEvent` landed in the `FEATURES` stream.
- 2026-06-25 — Logic-review hardening (whole-system pass): (A1) `TradingSignal` now enforces
  `stop_loss` for BUY/SELL via a `model_validator`, and `RiskEnvelope` rejects orders missing
  `stop_loss` (`missing_stop_loss`, defense-in-depth) — closes the "no order without stop_loss" rule.
  (B1) Documented the intentional 5% drawdown deadband in adaptive sizing (code unchanged).
  (C1) Un-wired the VIX-calibrated `vol_regime` from per-symbol feature computation (it conflated
  implied vs realized vol); kept `realized_vol_20` as a plain feature. shared 130 + feature-engine 61
  green; ruff + mypy (incl. --strict) clean. Logged cross-sectional ranking + robustness gaps above.
- 2026-06-25 — Closed [P1] cross-sectional ranking: `core/ranking.py` (`cross_sectional_rank` —
  tie-aware average-rank percentile in [0,1], per-feature, handles missing keys),
  `FeatureStore.all_for_interval`, service `ranked_universe`/`get_ranked`, and `GET /ranked` +
  `GET /ranked/{symbol}`. +9 tests (feature-engine 70 green); ruff + mypy clean.
- 2026-06-26 — Closed the open robustness/correctness issues: Wilder RSI (C3); subscriber
  `max_deliver` + poison-`term`/transient-`nak` (D1); real `/ready` dep checks — market-data on DB,
  feature-engine on NATS (D2); Redis-backed `FeatureStore` with in-memory fallback (store interface
  made async) (D3). feature-engine 78 / market-data 30 / shared 130 green; ruff + mypy clean.
  Live-verified the async event flow on a real `nats-server` (event → compute → `FeaturesReadyEvent`).
- 2026-06-26 — Direction #2 (strategy wired): extended `SignalGeneratedEvent` with
  `stop_loss`/`take_profit` (contracts-first); built strategy — `FeaturesSubscriber` on
  `features.ready`, `HttpFeatureClient` (queries feature-engine), **momentum-on-ranks** rule,
  `StrategyService` (signal → `RiskEnvelope` → `CostAwareFilter` → publish), `StrategyHealthTracker`
  (decay gate + `StrategyStatusChangedEvent`), routes (`/status`, `/evaluate/{symbol}`, `/decay`),
  JetStream publisher, lifespan, real `/ready`. RiskEnvelope step-7 treated as advisory (logged P1).
  +20 tests (strategy 86); shared 131; ruff + mypy clean. Live-verified the chain on a real
  `nats-server` (FeaturesReady → BUY → RiskEnvelope → `SignalGeneratedEvent`).
- 2026-06-26 — RiskEnvelope step-7 fix (P1): removed the sizing rejection — the envelope is now a
  pure gate; added `OrderRequestedEvent` (risk→execution). Simplified strategy's advisory workaround.
- 2026-06-26 — Direction #2 (risk-mgmt wired): `SignalSubscriber` on `signal.generated` →
  `PositionSizer` (DrawdownAdaptiveSizer risk budget + RegimeAllocator exposure/sector caps + 5%
  position cap, real **size-down**) → publish `OrderRequestedEvent`. `CircuitBreaker` (armed 24/7,
  YELLOW/RED/BLACK on drawdown/daily-loss) publishes `CircuitBreakerTriggeredEvent` and blocks new
  orders when tripped; in-memory `PortfolioState` + routes `/portfolio`, `/circuit-breaker`, `/signal`;
  real `/ready`. +27 tests (risk-mgmt 84); ruff + mypy clean. Added risk-mgmt to docker-compose.
  Live-verified on a real `nats-server` (SignalGenerated → sized OrderRequested; RED breaker halts).
- 2026-06-26 — **Loop closed** — execution (paper trading) wired: `OrderSubscriber` on
  `order.requested` → `PaperBroker` fills → publish `OrderFilledEvent` → `HttpRiskClient` pushes
  portfolio metrics to risk-mgmt `POST /portfolio` (fills now drive sizing + circuit breaker).
  Routes `/portfolio`, `/positions`, `/execute`; real `/ready`; added to docker-compose (port 8007).
  +13 tests (execution 17); ruff + mypy clean. Live-verified on a real `nats-server`
  (OrderRequested → OrderFilled → portfolio fed back). End-to-end loop now runs:
  market-data → feature-engine → strategy → risk-mgmt → execution → portfolio feedback.
- 2026-06-26 — Loop hardening (made the risk feedback real): (1) **execution real marks** — a second
  subscriber on `market_data.updated` re-marks held positions via `HttpMarketDataClient` (latest
  close) → recomputes portfolio → pushes to risk-mgmt, so the circuit breaker reacts to **unrealized**
  market moves, not just realized fills; `EventSubscriber` generalized for both subjects. (2)
  **strategy live portfolio** — `HttpPortfolioClient` reads risk-mgmt `GET /portfolio` for the
  RiskEnvelope gate (falls back to placeholder if unreachable). +6 tests (execution 21, strategy 88);
  ruff + mypy clean. compose env wired (strategy→RISK_MGMT_URL, execution→MARKET_DATA_URL).
- 2026-06-29 — **Persistence** (state survives restarts): risk-mgmt `PortfolioState` and execution
  `PaperBroker` now snapshot to **Redis** on every mutation and `restore()` on startup, with a
  `Null*Repository` fallback when Redis is down. risk-mgmt: `core/repository.py`
  (`StateRepository`/`Null`/`Redis`), `service.restore()` re-derives the breaker level from the
  restored drawdown/daily-loss (a tripped halt survives a restart), `save()` after every
  `update_portfolio`. execution: `PaperBroker.snapshot()`/`restore()`, `core/repository.py`
  (`BrokerRepository`/`Null`/`Redis`), `service.restore()`, `save()` after every fill/mark. main.py
  for both builds a Redis client (ping → `Redis*Repository`, else `Null*`) and `aclose()`s it on
  shutdown; compose `depends_on: redis` added for both. +9 tests each (risk-mgmt 93, execution 30);
  ruff + format + mypy clean. **Live-verified against a real Redis**: tripped breaker re-derived
  after a simulated restart (risk-mgmt); broker cash/positions carried over (execution). Lifespan
  smoke confirms graceful degradation with NATS+Redis both down (Null* fallback, clean shutdown).
- 2026-06-29 — Direction #2 (**backtest** wired): contracts-first — added `STRATEGY_REVALIDATED`
  (`backtest.strategy_revalidated`) + `StrategyRevalidatedEvent` to trading-common (+3 tests, shared
  134; also typed 3 pre-existing bare-`dict` metadata/metrics fields → `dict[str, Any]` to restore
  `mypy --strict` clean). Built the backtest service around the orphaned `ContinuousWalkForward`:
  `core/engine.py` (vectorized momentum long/flat backtest — no look-ahead, entry-aligned per-turn
  costs, Sharpe/maxDD/return/trades, `start_index` for OOS-only scoring), `core/walk_forward.py`
  (`EngineWalkForward` implements `_run_backtest` on the trailing OOS window), `HttpMarketDataClient`,
  `BacktestService` (run/revalidate → publish `BacktestCompletedEvent` / `StrategyRevalidatedEvent`),
  publisher + `ensure_stream(BACKTEST, ["backtest.>"])`, routes (`POST /run`, `POST /revalidate`),
  real `/ready` (gates on market-data), lifespan. pyproject: numpy + httpx + bugbear. compose:
  MARKET_DATA_URL + depends_on nats/market-data. backtest 39 tests (was a skeleton); ruff + format +
  mypy clean. Live-verified on a real `nats-server` (both events land in the `BACKTEST` stream and
  read back; real OOS Sharpe ≈ 2.25 → "active").
- 2026-06-29 — Direction #2 (**ml-pipeline** wired — **last orphaned component; Direction #2 COMPLETE**):
  wired `drift_detector` (`DriftDetector`: PSI + KS prediction-shift + rolling-Sharpe/accuracy decay)
  into the runtime. `core/registry.py` (`ModelBaseline` + in-memory `ModelRegistry`, placeholder for
  MLflow); `core/service.py` (`MLPipelineService.register_baseline` / `check_drift` → per-feature PSI
  vs baseline → `DriftReport` → publish `ModelDriftDetectedEvent` only when actionable, mapping
  drift_type + severity); `events/publisher.py`, routes (`POST /models/{id}/baseline`,
  `POST /models/{id}/drift`, `GET /models`), real `/ready` (NATS), lifespan + `ensure_stream(ML,
  ["ml.>"])`. pyproject: bugbear immutable-calls. compose: ml-pipeline uncommented (port 8005).
  ml-pipeline 35 tests (was a skeleton); ruff + format + mypy clean; all suites green (527 total).
  Live-verified on a real `nats-server` (`ml.drift_detected` lands in the `ML` stream and reads back).
- 2026-06-29 — **notification** wired (monitoring loop closed; first multi-stream consumer): durable
  `EventSubscriber`s on `risk.circuit_breaker`, `order.filled`, `backtest.strategy_revalidated`,
  `ml.drift_detected` (each on its owning stream, `ensure_stream` so start-order independent).
  `core/alerts.py` (event → severity-graded `Alert`), `core/service.py` (`NotificationService`:
  min-severity gate, recent-alerts ring buffer, fan-out with per-channel failure isolation),
  `core/channels.py` (`LogChannel` always-on; `SlackChannel`/`TelegramChannel` HTTP, built only when
  configured), `events/subscriber.py` (reused poison-safe subscriber + `ensure_stream`). Routes
  `GET /channels`, `GET /alerts/recent`, `POST /test-alert`; real `/ready` (NATS); pyproject httpx +
  bugbear. compose: notification uncommented (port 8008, Slack/Telegram env passthrough). notification
  28 tests (was a skeleton); ruff + format + mypy clean; all suites green (555 total). Live-verified on
  a real `nats-server` (all 4 events → 4 correctly-graded alerts via the real subscribers).
- 2026-06-30 — **dashboard** wired (**last skeleton — all 9 core services now functional**): built as a
  FastAPI **backend-for-frontend** (not Streamlit, to keep the `/health` `/ready` `/metrics` + structlog
  conventions). `core/clients.py` (`HttpDashboardSource`: read-only GETs to risk-mgmt / execution /
  notification / ml-pipeline, each degrading to `None` on failure), `core/service.py`
  (`DashboardService.overview` — concurrent `asyncio.gather`, partial-tolerant, per-source status map),
  `api/ui.py` (self-contained HTML/CSS/JS page, no build step), routes `GET /overview` + `GET /ui` + root
  redirect, real `/ready` (per-source reachability, always 200). pyproject httpx + bugbear + per-file
  E501 ignore for the HTML string. compose: dashboard uncommented (8501→8000, depends_on risk-mgmt +
  execution). dashboard 18 tests (was a skeleton); ruff + format + mypy clean; all suites green (573
  total). **Live-verified** against real risk-mgmt + execution (uvicorn + a real `nats-server`): the real
  `HttpDashboardSource` aggregated their live state over HTTP (portfolio dd 0.04, AAPL 50@100) while the
  two down services correctly showed "unavailable".
- 2026-06-30 — **Direction #3 started — `macro-data` (serwis 10) built from scratch**: first new service
  (not a skeleton wiring). `core/regime.py` (`classify_regime` — severity-ordered rules on yield-curve
  inversion / BAA credit spread / PMI → the 5 `MacroRegime` values, missing-input tolerant),
  `core/fred_client.py` (`FredClient` httpx fetch of T10Y2Y/BAA10Y/UNRATE/FEDFUNDS; disabled→None with
  no `FRED_API_KEY`), `core/service.py` (`MacroDataService.refresh` — FRED + manual overrides →
  classify → publish `MacroUpdatedEvent` always + `RegimeChangedEvent` on a real transition; overrides
  are non-None-only so a None doesn't clobber a fetched value), publisher + `ensure_stream(MACRO)`,
  routes (`GET /snapshot`, `GET /regime`, `POST /refresh`), real `/ready`, full scaffold (Dockerfile,
  pyproject, observability, compose port 8010, Helm `macroData` values entry). macro-data 41 tests;
  ruff + format + mypy clean; all suites green (614 total). Live-verified on a real `nats-server`
  (expansion→crisis → 2×`macro.updated` + 1×`macro.regime_changed` in `MACRO`). Regime keys already
  match risk-mgmt's RegimeAllocator, so the output is drop-in for regime-aware exposure caps.
- 2026-07-01 — **macro→risk loop closed**: risk-mgmt now **subscribes to `RegimeChangedEvent`**
  (`macro.regime_changed`). Renamed the generic `SignalSubscriber` → `EventSubscriber` (reused for both
  `signal.generated` and `macro.regime_changed`); `service.handle_regime_changed_event` →
  `update_portfolio(regime=new_regime)` (persists; a regime change alone never trips the breaker since
  it doesn't touch drawdown/daily-loss); main.py `ensure_stream(MACRO)` + a second durable subscriber;
  config `NATS_MACRO_*`. So macro-data's regime now auto-drives the RegimeAllocator exposure caps (no
  manual `POST /portfolio`). +4 tests (risk-mgmt 97); ruff + format + mypy clean; all suites green (618
  total). Live-verified on a real `nats-server`: a published `macro.regime_changed` (expansion→crisis)
  flips risk-mgmt's regime → crisis cap 15% blocks an over-exposed BUY.
- 2026-07-01 — Direction #3 (**fundamental-data** — serwis 9, built from scratch): SEC EDGAR annual
  fundamentals + (partial) Piotroski F-Score. `core/piotroski.py` (`compute_f_score` — the 7 of 9
  classic signals computable from `FinancialStatements`; current-ratio Δ + share-issuance omitted &
  documented; conservative on missing inputs), `core/edgar_client.py` (`EdgarClient` ticker→CIK +
  XBRL `companyconcept` → annual statements; disabled without `SEC_USER_AGENT`), `core/service.py`
  (`refresh` from EDGAR / `ingest` posted statements → score → store → publish
  `FundamentalsUpdatedEvent`), routes (`GET /fundamentals[/{symbol}]`, `POST /refresh/{symbol}`,
  `POST /statements`), real `/ready`, publisher + `ensure_stream(FUNDAMENTALS)`, full scaffold
  (Dockerfile, pyproject, compose port 8009, Helm `fundamentalData`). 27 tests; ruff + format + mypy
  clean; all suites green (645 total). Live-verified on a real `nats-server` (ingest →
  `fundamentals.updated` in `FUNDAMENTALS`; F-score 7/7 on an improving firm). EDGAR live-fetch path is
  unit-tested via httpx MockTransport (SEC needs a `User-Agent` + isn't reachable from the sandbox).
- 2026-07-01 — Direction #3 (**company-classifier** — serwis 11, built from scratch): `CompanyProfile`
  → investment style + model-stack routing (pure compute, no external API). `core/classifier.py`
  (`classify` — growth vs value signal scoring from valuation/growth metrics; sector-prior fallback then
  blend; `cap_tier` + `route_model_stack(style, tier)` → `{style}_{large|small}cap_v1`), `core/service.py`
  (`classify` — enrich profile with style/model_stack/as_of, store latest-per-symbol, publish
  `CompanyClassifiedEvent`), routes (`GET /companies[/{symbol}]`, `POST /classify`), real `/ready`,
  publisher + `ensure_stream(COMPANY)`, full scaffold (compose port 8011, Helm `companyClassifier`).
  25 tests; ruff + format + mypy clean; all suites green (670 total). Live-verified on a real
  `nats-server` (classify NVDA → `company.classified` `growth_largecap_v1` in the `COMPANY` stream).

**Next:** Direction #3 has **one service left — signal-aggregator (serwis 12)**: combine ML + rules-based
+ macro-regime signals → `SignalAggregatedEvent`; when it lands, **move `adaptive_weights.py` +
`cost_filter.py` there from `strategy/`** (their spec home). That closes the full 13-service architecture.
A natural follow-up: have **feature-engine (or a Tier-2/3 layer) consume `FundamentalsUpdatedEvent` /
`CompanyClassifiedEvent`** so fundamentals + style feed ML features. Open follow-ups (non-blocking): a
generic Helm Deployment template for the 12 untemplated services; extend `FinancialStatements` with
current assets/liabilities + shares → full 9-signal Piotroski; scheduled triggers (backtest weekly
revalidation; ml-pipeline daily drift; macro/fundamentals periodic refresh); notification email/SMTP;
MLflow registry; ml-pipeline training/inference (PyTorch); `docs/ml_integration_plan.md` before deep ML
work. Deeper persistence hardening (event-log/DB, multi-instance) remains optional.

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
- Position sizing is drawdown-adaptive: full 2% risk until DD=5% (deadband), then scales linearly to 0% at DD=15%
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
