# Trading System — Microservices Architecture

## Project overview

Production-grade algorithmic trading system. 13 independent Python microservices communicating
via NATS JetStream (events) and HTTP (request/response).

**Key docs:**
- **Project context/status/direction: this file** — see "Project status & direction" below (single source of truth I read every session)
- Full 24-week development plan: `Plan_Rozwoju_Systemu_Tradingowego_2.md` (repo root)
- Framework supplement — 12 components (risk envelope, drift/decay monitors, cost filter, regime allocator, …): `docs/framework_supplement.md`
- **ML/AI integration plan: `docs/ml_integration_plan.md`** — authoritative design for the ML
  phase (cross-sectional shallow model on ranked features, triple-barrier labels, purged
  walk-forward, MLflow, `ml.signal_generated` → aggregator, drift monitoring, roadmap ML-0…ML-4).
  Read it before touching ml-pipeline.

## Project status & direction

> Single living context block. Read this first every session. Keep the progress log append-only.
> If a fresh analysis surfaces new bugs or improvement ideas, **propose them here and to the user** —
> do not silently proceed.

**Phase:** 1 — Foundation. The earlier priority inversion is **resolved**: the foundation was built
and the framework components wired into a working **end-to-end paper-trading loop** (market-data →
feature-engine → strategy → risk-mgmt → execution → portfolio feedback) plus backtest + ml-pipeline
monitoring, notification alerting, and a dashboard BFF over the HTTP APIs. **All 13 services (9 core +
4 ML/AI extension) are now functionally implemented** — no skeletons left; Direction #3 complete.

**Verified ground truth** (run locally on Python 3.12 — not from memory):
- `shared/trading-common`: 181 tests green, `ruff` + `mypy --strict` clean. Contracts present:
  `OHLCVBar`, `TradingSignal`, `PortfolioMetrics`, ML/AI contracts (`CompanyProfile`,
  `FinancialStatements`, `MacroSnapshot`, `SentimentSnapshot`, `FeatureVector`), full `EventType`
  set incl. ML/AI extension + `STRATEGY_REVALIDATED` (backtest→strategy), `RiskEnvelope`, and the shared
  **`CostAwareFilter`** (moved out of strategy — a cross-cutting gate like `RiskEnvelope`).
  `SignalAggregatedEvent` carries `sector` (R8); `StrategyStatusChangedEvent` metrics are optional
  (a revalidation-driven change has no 30d PF). `FinancialStatements` carries balance-sheet detail
  (`current_assets`/`current_liabilities`/`shares_outstanding`) for the full 9-signal Piotroski.
  **ML-2**: `MlSignalGeneratedEvent` (`ml.signal_generated`) — the per-symbol ML vote (model_id,
  signal, calibrated probability_up, horizon; deliberately NO levels — ML cannot trade alone).
  Shared utilities: `RiskEnvelope`, `CostAwareFilter`, and **`scheduler.PeriodicTask`** (in-process
  asyncio periodic jobs, exception-isolated, + `seconds_until_weekday_hour` for calendar alignment).
  **ML-0**: the pure feature/rank definitions moved here — `trading_common.features`
  (`compute_feature_vector`) + `trading_common.ranking` (`cross_sectional_rank`) — so ml-pipeline
  training reproduces feature-engine serving bit-for-bit (numpy is now a trading-common dependency).
- All 13 services functionally implemented (`/health` `/ready` `/metrics` green; no skeletons left).
- Framework-supplement components still **orphaned** (tested but not wired into FastAPI/NATS):
  feature-engine only (`earnings_decay`, `cross_asset`). (`decay_monitor`+`cost_filter` now wired into
  strategy; `adaptive_weights` moved to signal-aggregator; `cost_filter` moved to trading-common;
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
  `GET /ranked`). **Tier-2 attribute enrichment wired**: durable subscribers on
  `fundamentals.updated` (→ HTTP query back to fundamental-data → `f_score` +
  `fund_net_margin`/`fund_roa`/`fund_leverage`) and `company.classified` (→ `style_growth`/
  `style_value` encoding) fill a per-symbol `SymbolAttributeStore` (Redis-backed, in-memory
  fallback), merged into vectors at **read time** so `/features` and `/ranked` expose them
  (incl. cross-sectional f_score percentile). Attribute updates deliberately do NOT publish
  `features.ready` (no strategy re-evaluation on a fundamentals refresh — the ML tier reads the
  merged vectors). Pure feature/rank definitions now imported from **trading-common** (ML-0 —
  training/serving parity); the service keeps orchestration, store and API. 84 tests green
  (pure-function tests moved to shared); live-verified end-to-end (real uvicorn fundamental-data +
  company-classifier: ingest → f_score 7 merged; classify → growth encoding; 2-symbol universe
  ranks f_score 1.0/0.0).
- `strategy` is now **functionally implemented** (Direction #2): JetStream subscriber on
  `features.ready` → fetch ranked+raw features from feature-engine (HTTP) → **momentum-on-ranks**
  rule → `TradingSignal` (vol-agnostic % stop) → **`RiskEnvelope`** (SL-enforcing; step-7 sizing
  treated as advisory) → **`CostAwareFilter`** → publish `SignalGeneratedEvent` (now carries
  `stop_loss`/`take_profit`). `StrategyDecayMonitor` gates output (inactive → suppress; `POST /decay`
  re-evaluates and emits `StrategyStatusChangedEvent`). **R7 closed**: a second durable subscriber
  (`strategy-revalidation`) consumes `backtest.strategy_revalidated` — backtest *recommends*, strategy
  *owns*: `apply_revalidation` maps `deactivate`→`deactivated`, applies via
  `StrategyHealthTracker.apply_status`, ignores other strategies' events, poison-terms unknown
  recommendations, and publishes `StrategyStatusChangedEvent` on a real transition into the **new
  `STRATEGY` stream** (`strategy.>` — previously `strategy.status_changed` had NO stream, so a live
  publish would have failed; latent bug fixed). 56 tests green; live-verified (real
  `backtest.strategy_revalidated` → probation + event in the STRATEGY stream).
- `risk-mgmt` is now **functionally implemented** (Direction #2): JetStream subscriber on
  **`signal.aggregated`** (R1a — the signal-aggregator is the decision node; durable
  `risk-mgmt-aggregated`; the manual `POST /signal` route still accepts raw strategy signals) →
  **`PositionSizer`** (`adaptive_sizing` drawdown-scaled risk budget +
  `regime_allocator` exposure/sector caps + 5% position cap → size-down) → publish
  **`OrderRequestedEvent`** (risk→execution). BUY/SELL aggregates without price+stop_loss are
  blocked (defense-in-depth). **Circuit Breaker** armed 24/7
  (`CircuitBreaker`: YELLOW dd>8% / RED daily-loss>5% halt / BLACK dd>15% flatten) → publishes
  `CircuitBreakerTriggeredEvent` and blocks new orders when tripped. `PortfolioState`
  (updatable via `POST /portfolio`) is now **Redis-persisted** (`RedisStateRepository` snapshot on
  every update; `NullStateRepository` fallback) — on startup `restore()` reloads the snapshot and
  **re-derives** the breaker level, so a tripped halt survives a restart. Also subscribes to
  **`RegimeChangedEvent`** (`macro.regime_changed`, from macro-data) → `update_portfolio(regime)` so the
  macro regime auto-drives the RegimeAllocator exposure caps (no manual push needed). **R8 closed**:
  `process_aggregated` passes the event's `sector` into `PositionSizer.size(..., sector=...)`, so the
  regime-aware **sector caps are live** (crisis/contraction allow only defensive sectors; `sector=None`
  → gate skipped). Routes `/portfolio`, `/circuit-breaker`, `/signal`. 104 tests green; live-verified
  (SignalAggregated → sized OrderRequested; breaker RED halts new orders; tripped breaker survives a
  restart via real Redis; a real `macro.regime_changed` event flips the regime → tightens the cap;
  crisis blocks an Information-Technology BUY by sector while expansion sizes it).
- `execution` is now **functionally implemented** (paper trading — **closes the loop**): JetStream
  subscriber on `order.requested` → `PaperBroker` simulates the fill → publish `OrderFilledEvent` →
  push portfolio metrics (equity/exposure/drawdown/daily-loss) back to risk-mgmt over HTTP
  (`POST /portfolio`), so fills drive sizing + the circuit breaker. `PaperBroker` (cash/positions,
  peak-equity drawdown, mark-to-fill) is now **Redis-persisted** (`RedisBrokerRepository` snapshot on
  every fill/mark; `NullBrokerRepository` fallback) — `restore()` reloads cash/positions on startup.
  2026-07-05 review fixes wired: **R2** daily-loss baseline rolls on the first fill/mark of a new day
  (date in the snapshot; injectable clock), **R3** fills idempotent by order `event_id` (dedup set
  persisted; save-before-publish ordering), **R4** long-only (SELL = exit: capped at held qty,
  skipped when flat — matches the long/flat backtest engine), **R5** protective exits (positions carry
  SL/TP; each re-mark checks levels and paper-exits on breach, publishing a second `OrderFilledEvent`).
  Routes `/portfolio`, `/positions`, `/execute` (409 on duplicate/long-only violation); real `/ready`.
  44 tests green; live-verified (OrderRequested → OrderFilled → portfolio fed back; broker state
  survives a restart via real Redis; SL breach on re-mark exits the position).
- `backtest` is now **functionally implemented** (Direction #2): wires the orphaned
  `continuous_validation` (`ContinuousWalkForward`, abstract) to a real **momentum backtest engine**
  (`core/engine.py`: numpy time-series long/flat momentum, no look-ahead, per-turn costs →
  Sharpe/maxDD/return/trades; `start_index` measures the OOS tail with IS warm-up). `EngineWalkForward`
  implements `_run_backtest` over the trailing OOS window; `BacktestService` pulls OHLCV from
  market-data (HTTP) → runs backtest/revalidation → publishes `BacktestCompletedEvent` and the new
  `StrategyRevalidatedEvent` (backtest **recommends** active/probation/deactivate; strategy **owns** the
  status). Routes `POST /run`, `POST /revalidate`; real `/ready` gates on market-data. **Scheduled
  weekly revalidation** (Saturday 06:00 UTC via `PeriodicTask` + weekday alignment; OPT-IN
  `SCHEDULE_REVALIDATION_ENABLED` — the event drives the live strategy status (R7), so it ships
  disabled until the real activation-time OOS-Sharpe baseline is configured). 41 tests green;
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
  ["ml.>"])`. **ML-0 landed** (the dataset foundation from `docs/ml_integration_plan.md`):
  `core/labels.py` (triple-barrier labeling on the OHLC path — ±2σ₂₀·√h barriers, h=10d,
  next-bar entry, same-bar double-touch = conservative loss, truncated-untouched → unresolved),
  `core/splits.py` (purged walk-forward `Fold`s over session dates + embargo), `core/dataset.py`
  (`build_dataset`: per-session cross-section via the SHARED `trading_common.features`/`ranking` →
  full-universe rank → label → pooled matrix; level features excluded, missing attrs → neutral 0.5,
  macro one-hot appended; + `next_returns` for evaluation). **ML-1 landed** (training + registry):
  `core/model.py` (PyTorch `MlpClassifier` 2×hidden + dropout, pos_weight, early stopping with a
  min-epochs warm-up, temperature calibration), `core/evaluation.py` (tie-aware AUC, Brier, and the
  DECISION metric — cost-adjusted Sharpe of the daily-rebalanced top-quantile long-only portfolio
  with turnover costs), `core/training.py` (`run_training`: untouched holdout + purged walk-forward
  folds, per-window purged fit/val split, gate = holdout Sharpe>0.5 AND ≥2/3 recent folds AND
  Brier ≤ base rate; final model retrained on full history regardless — caller owns what a failed
  gate means), `core/model_store.py` (**MLflow sqlite backend**: params/metrics logged, artifacts =
  `state_dict` + LOAD-BEARING `metadata.json` (feature list/temperature/shape), alias-based
  promotion `production` — manual sign-off; exact load round-trip), `core/market_data_client.py`;
  service `train()` = fetch universe history → dataset → gate → MLflow version + drift baseline
  auto-registered as `{model}@v{N}`. Routes `POST /models/train` (sync, minutes — ops/scheduled),
  `POST /models/versions/{v}/promote`, `GET /models` (+registry versions). torch+mlflow deps
  (ml-pipeline only; build images with the CPU wheel index — PyPI default bundles CUDA); compose:
  `ml_mlruns` volume + MARKET_DATA_URL/MLFLOW_TRACKING_URI (Helm env mirrored, PVC = scale-up).
  **ML-2 landed** (serving): `core/serving.py` — `ServingEngine` runs ONLY the production-aliased
  model; on `features.ready` (durable `ml-pipeline-features`, interval-filtered) it pulls the
  symbol's **ranked** vector (new `HttpFeatureClient`) + the macro regime (`HttpMacroClient`,
  10-min TTL cache, degrades to the all-zeros "unknown" one-hot), assembles the input row in the
  metadata's exact feature order (missing Tier-2 attr → neutral 0.5 like training; MAJORITY of
  expected features missing → **inference refused** — schema drift, not sparsity), and publishes
  `MlSignalGeneratedEvent` only outside the dead zone (p≥0.55 BUY / p≤0.45 SELL; HOLD is silent,
  mirroring strategy semantics — stale votes TTL-expire in the aggregator). `service.promote()`
  **hot-reloads** serving (no restart to swap models). 99 tests green (gate anti-luck checks +
  serving/refusal/dead-zone/hot-reload); ruff + mypy clean; uvicorn lifespan smoke (serving
  inactive until promotion, features durable subscribed) + **live ML-2 chain verified on a real
  `nats-server`** (real trained+promoted model from a real sqlite registry: `features.ready` →
  infer → `ml.signal_generated` in the ML stream → aggregator re-aggregates 1→2 components with
  strategy levels intact).
- `notification` is now **functionally implemented** (closes the monitoring loop — first multi-stream
  consumer): durable `EventSubscriber`s on the 5 alert-worthy events across their streams —
  `CircuitBreakerTriggeredEvent` (RISK), `OrderFilledEvent` (ORDERS), `StrategyRevalidatedEvent`
  (BACKTEST), `ModelDriftDetectedEvent` (ML), `StrategyStatusChangedEvent` (STRATEGY — the *applied*
  transition, complementing the revalidation *recommendation*; demotion=warning, reactivation=info).
  `core/alerts.py` maps each event → `Alert`
  (severity-graded); `NotificationService.dispatch` applies a min-severity gate, keeps a recent-alerts
  ring buffer, and fans out to channels with per-channel failure isolation. `core/channels.py`:
  `LogChannel` (always on), `SlackChannel`/`TelegramChannel` (HTTP) and **`EmailChannel`** (SMTP via
  stdlib smtplib in a worker thread; STARTTLS + optional login; needs SMTP_HOST+EMAIL_FROM+EMAIL_TO)
  — each built only when configured, log-only otherwise. Routes `GET /channels`, `GET /alerts/recent`,
  `POST /test-alert`; real `/ready` (NATS); `ensure_stream` for all 5 source streams (start-order
  independent). 33 tests green; ruff + format + mypy clean; live-verified on a real `nats-server`
  (all 5 events → 5 correctly-graded alerts; every alert also rendered to a captured `EmailMessage`).
  A scheduler-driven digest is a follow-up.
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
  **Scheduled refresh** every 6h (`PeriodicTask`; first run at boot; runs only when `FRED_API_KEY` is
  set — transition-safe since `RegimeChangedEvent` fires only on real changes).
  New service scaffold (Dockerfile, pyproject, compose port 8010, Helm values entry). 41 tests; ruff +
  format + mypy clean; live-verified on a real `nats-server` (expansion→crisis → 2×`macro.updated` +
  1×`macro.regime_changed` in the `MACRO` stream). **risk-mgmt now subscribes to `RegimeChangedEvent`**,
  so the regime auto-drives the exposure caps (macro→risk loop closed).
- `fundamental-data` (**serwis 9 — Direction #3, built from scratch**): SEC EDGAR annual fundamentals +
  **full 9-signal Piotroski F-Score**. `core/piotroski.py` (`compute_f_score` — 3 current-period
  profitability + 6 trend signals, incl. current-ratio Δ and no-dilution enabled by the extended
  `FinancialStatements` (current assets/liabilities + shares outstanding, contracts-first);
  each signal fails conservatively on missing/degenerate inputs — legacy statements without
  balance-sheet detail cap at 7), `core/edgar_client.py` (`EdgarClient` —
  ticker→CIK via company_tickers.json, XBRL `companyconcept` per us-gaap tag → annual `FinancialStatements`;
  disabled + [] when no `SEC_USER_AGENT`), `core/service.py` (`FundamentalDataService.refresh` from EDGAR /
  `ingest` posted statements → score → store latest-per-symbol → publish `FundamentalsUpdatedEvent`).
  Routes `GET /fundamentals[/{symbol}]`, `POST /refresh/{symbol}`, `POST /statements`; real `/ready` (NATS);
  publisher + `ensure_stream(FUNDAMENTALS, ["fundamentals.>"])`. Full scaffold (compose port 8009, Helm
  `fundamental-data` services entry). Revenue has **tag fallbacks** (`Revenues` →
  `RevenueFromContractWithCustomer[Ex/In]cludingAssessedTax` → `SalesRevenueNet`), merged per period
  with earlier-tag priority — ASC-606 filers and tag-switchers both resolve; new tags: `AssetsCurrent`,
  `LiabilitiesCurrent`, `CommonStockSharesOutstanding` (+weighted-average share fallbacks).
  **Scheduled weekly universe refresh** (`refresh_universe` over `REFRESH_SYMBOLS` csv with a
  politeness pause between symbols; runs only with `SEC_USER_AGENT` + a non-empty universe). 36 tests;
  ruff + format + mypy clean; live-verified on a real `nats-server` (ingest → `fundamentals.updated`
  in the `FUNDAMENTALS` stream).
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
- `signal-aggregator` (**serwis 12 — Direction #3 finale, built from scratch**): combines multi-source
  signals (rules/strategy + ML + macro-regime) into one decision. `core/aggregator.py` (`combine` —
  signed-confidence weighted vote: +conf BUY / −conf SELL / 0 HOLD → threshold → BUY/SELL/HOLD),
  `core/adaptive_weights.py` (**moved from strategy** — `AdaptiveWeightOptimizer` EWP performance
  weighting), `core/service.py` (`SignalAggregatorService.aggregate` — optimizer weights renormalized
  over present sources → `combine` → shared **`CostAwareFilter`** gate (marginal edge → HOLD) → publish
  `SignalAggregatedEvent`; `record_outcome` adapts weights). Routes `POST /aggregate`, `POST /outcomes`,
  `GET /weights`; real `/ready` (NATS); publisher + `ensure_stream(SIGNALS, ["signal.>"])`. Full scaffold
  (compose port 8012, Helm `signalAggregator`). Also **moved `cost_filter` → trading-common** (shared gate,
  strategy now imports it from there). A **live multi-stream consumer and the decision node (R1a)**:
  durable subscribers on `signal.generated` (buffers the latest per-symbol strategy signal **with its
  price/SL/TP**, TTL-expired after `SIGNAL_TTL_SECONDS`, default 1 day — R6; the entry ages from the
  event's **emit timestamp**, so a durable replaying stream history cannot resurrect stale signals) and
  `macro.regime_changed` (`REGIME_BIAS` → market-wide directional component; **slowdown is neutral →
  contributes nothing** (R10); a transition re-aggregates every buffered symbol); each update publishes
  `signal.aggregated` carrying the order context (levels attached only when the final direction matches
  the strategy component's) **+ the symbol's `sector`** (R8 — `HttpCompanyClient` queries
  company-classifier `GET /api/v1/company-classifier/companies/{symbol}`, positive-cached, degrades to
  None), which **risk-mgmt consumes and sizes into orders** honoring the regime's sector caps.
  `POST /aggregate` is documented as ops/testing only (R9 — bypasses buffer/macro/sector enrichment).
  **The "ml" source is LIVE (ML-2, closes R11 for real)**: a third durable subscriber
  (`signal-aggregator-ml`) on `ml.signal_generated` buffers the latest per-symbol ML vote (aged from
  the emit timestamp, same TTL); an ML vote joins the strategy component in aggregation but **never
  aggregates alone** (strategy required — ML modulates strategy-led decisions). 80 tests; ruff +
  format + mypy clean; live-verified on a real `nats-server` (full chain: signal → aggregated
  BUY+levels → sized order → fill; crisis → re-aggregated HOLD → no order; sector enriched from a
  real uvicorn company-classifier over HTTP; ML vote → 2-component re-aggregation).
  **This closes the full 13-service architecture.**

**Direction (where the project should go, in order):**
1. ✅ **DONE — Foundation:** `market-data` fetch → validate → store → cache → publish event
   (NATS **JetStream**, `Nats-Msg-Id` dedup). Next refinements (deferred, non-blocking): bulk
   `ON CONFLICT` insert instead of per-row merge, a scheduled/periodic fetch job.
2. ✅ **DONE — Wire the orphaned components** into their services (API endpoints + NATS
   pub/sub). feature-engine, strategy, risk-mgmt, backtest, ml-pipeline all wired. (Leftover specs —
   feature-engine `earnings_decay`/`cross_asset`, strategy `adaptive_weights` — belong in later
   services, not the 7 core runtime paths; tracked under tech debt.)
3. ✅ **DONE — Build serwisy 10–13**: fundamental-data (9), macro-data (10), company-classifier (11),
   signal-aggregator (12) all built. `adaptive_weights.py` moved to signal-aggregator, `cost_filter.py`
   moved to trading-common (shared). **All 13 services now exist and are functional.**
4. **Contracts-first** always: extend `shared/trading-common` before adding any cross-service type.

**Known issues / tech debt** (propose a fix when you touch the area):
- [P1 ✅ done 2026-07-07] **R1 resolved as (a)** — the signal-aggregator is the **decision node**:
  `SignalAggregatedEvent` extended with price/SL/TP/strategy_name (attached only when the final
  direction matches the strategy component's); risk-mgmt's subscription switched to
  `signal.aggregated` (new durable `risk-mgmt-aggregated`; the old `risk-mgmt` durable on
  `signal.generated` is orphaned server-side — harmless, delete manually if desired). Raw
  `signal.generated` now only feeds the aggregator.
- [P1 ✅ done 2026-07-07] **R2** — `PaperBroker` day baseline is date-tagged and rolls on the first
  fill/mark of a new day (date persisted in the snapshot; injectable clock for tests).
- [P1 ✅ done 2026-07-07] **R3** — fills are idempotent by order `event_id` (persisted dedup set;
  save-before-publish so a crash replays cleanly and a publish failure dedups on redelivery).
- [P1 ✅ done 2026-07-07] **R4** — long-only: execution treats SELL as an exit (capped at held qty,
  skipped when flat). Live behavior now matches the long/flat backtest engine. Shorts, if ever wanted,
  must be modeled end-to-end (engine + sizing + broker) as a deliberate feature.
- [P1 ✅ done 2026-07-07] **R5** — protective exits: positions carry SL/TP; every re-mark checks the
  levels and paper-exits on breach (second `OrderFilledEvent`). Paper simplification: the latest BUY
  defines the position's levels (no per-lot tracking); exits use the mark price (no gap modeling).
- [P2 ✅ done 2026-07-07] **R7–R11** (2026-07-05 review, second batch): **R7** strategy subscribes
  `backtest.strategy_revalidated` (durable `strategy-revalidation`) and applies the recommendation
  (`deactivate`→`deactivated`; publishes `StrategyStatusChangedEvent` on transition) — the
  backtest→strategy loop is closed *and* the new `STRATEGY` stream (`strategy.>`) fixes the latent
  no-stream bug for `strategy.status_changed`. **R8** `SignalAggregatedEvent.sector` (contracts-first):
  aggregator enriches it from company-classifier (`HttpCompanyClient`, positive-cache, graceful None);
  risk-mgmt feeds it to `PositionSizer` → regime sector caps live. Caveat: profile sectors must use the
  RegimeAllocator's GICS-style names ("Information Technology", "Consumer Staples", …) — an unmatched
  string blocks in restrictive regimes (conservative). **R9** documented: `POST /aggregate` is
  ops/testing only (bypasses buffer + macro bias + sector enrichment; its event still reaches
  risk-mgmt). **R10** slowdown → no macro component (was HOLD 0.0, which stole weight from strategy).
  **R11** documented in config: "ml" source is pre-provisioned; live aggregation is 2-source until
  ml-pipeline emits per-symbol signals (renormalization makes the absent source free). P3s: EDGAR
  revenue **tag fallbacks** shipped (per-period merge, earlier-tag priority); durable-replay staleness
  solved by aging buffer entries from the **event emit timestamp** (durables stay `DeliverPolicy.ALL`
  for start-order independence — better than `DeliverPolicy.NEW` since TTL now guards replays); double
  cost-gating stays intentional-conservative.
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
- [P2 ✅ done] `adaptive_weights.py` moved to `signal-aggregator/`; `cost_filter.py` moved to
  `trading-common` (a shared cross-cutting gate like `RiskEnvelope`, used by both strategy and
  signal-aggregator). Neither remains in `strategy/`.
- [P2 ✅ done 2026-07-12] `docs/ml_integration_plan.md` written — the binding ML-phase design
  (see Key docs). Headline decisions: cross-sectional (pooled-universe) shallow PyTorch MLP on
  the ranked feature vectors, triple-barrier labels (2σ·√10 barriers, h=10d), purged
  walk-forward + embargo, cost-adjusted OOS-Sharpe>0.5 activation gate, MLflow local-backend
  registry, ML as a *no-levels vote* in the aggregator (cannot trade alone), daily drift +
  delayed-label outcome loop; per-style stacks and meta-labeling deliberately deferred to v2.
- [P2] README "Status infrastruktury (zweryfikowany)" cannot be verified without Docker (none in sandbox/CI) — treat as *expected*, not *verified*.
- [P3] `infrastructure/terraform/` is referenced in README but absent (planned).
- [P2 ✅ done 2026-07-07] Helm chart: `values.yaml` restructured into a **`services:` map**
  (kebab-case key = k8s name = compose name) and a **generic `templates/services.yaml`** renders
  Deployment+Service for all 13 services (probes on `/health`+`/ready`, prometheus annotations,
  common env injected: SERVICE_NAME/NATS_URL/REDIS_HOST/REDIS_PASSWORD-secret, `needsDb` → DB
  secret; per-service `env` maps mirror compose URLs). `ingress.yaml` generates the 13
  `/api/v1/{service}` routes (mirrors compose Traefik labels); dedicated market-data template
  removed; dashboard containerPort fixed 8501→8000. `values-prod.yaml` migrated; replicas >1 only
  for services **without** an event subscription (push durables don't load-balance — see the open
  robustness item; risk-mgmt/execution are single-writer Redis snapshots). Render-verified with a
  real `helm` binary (lint + template, dev & prod: 13 Deployments/Services, 13 ingress paths,
  secret refs, prod deep-merge). No HPA yet — deliberate until consumers can scale.
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
- 2026-07-01 — Direction #3 (**signal-aggregator** — serwis 12, **finale; all 13 services now built**):
  combines rules/strategy + ML + macro-regime signals into one decision. `core/aggregator.py` (`combine`
  — signed-confidence weighted vote → threshold → BUY/SELL/HOLD), `core/adaptive_weights.py`
  (**moved from strategy**), `core/service.py` (optimizer weights renormalized over present sources →
  combine → shared `CostAwareFilter` gate → publish `SignalAggregatedEvent`; `record_outcome` adapts
  weights), routes (`POST /aggregate`, `POST /outcomes`, `GET /weights`), real `/ready`, publisher +
  `ensure_stream(SIGNALS)`, full scaffold (compose port 8012, Helm `signalAggregator`). **Refactor:
  `cost_filter.py` moved strategy → trading-common** (shared gate like `RiskEnvelope`; strategy + shared
  imports updated; its 20 tests moved to shared). signal-aggregator 49 tests (incl. 22 moved
  adaptive_weights); strategy 46 (was 88, the 42 moved out); shared 154 (+20). ruff + format + mypy
  clean; all suites green (697 total). Live-verified on a real `nats-server` (consensus BUY →
  `signal.aggregated` in the `SIGNALS` stream). **Direction #3 complete — the full 13-service
  architecture is implemented.**
- 2026-07-05 — **signal-aggregator wired as a live consumer** (integration; behavior-neutral for now):
  durable `EventSubscriber`s on `signal.generated` (latest-per-symbol strategy component buffer) and
  `macro.regime_changed` (`REGIME_BIAS` expansion/recovery→BUY, slowdown→neutral, contraction/crisis→SELL
  → market-wide component; a transition re-aggregates every buffered symbol); `ensure_stream(MACRO)`;
  event-driven aggregation publishes `signal.aggregated` per update. +10 tests (signal-aggregator 59);
  ruff + format + mypy clean; all suites green (707 total). Live-verified on a real `nats-server`
  (signal.generated → aggregated BUY [1 comp]; expansion→crisis → re-aggregated HOLD [2 comps]).
  **Whole-system logic review (first Fable 5 pass)**: 5×P1 + 6×P2 + P3 findings logged above as
  **R1–R11** — headline R1: the aggregate is advisory (no consumer; event lacks price/SL/TP), so
  risk-mgmt still acts on raw strategy signals. Other P1s: R2 daily-loss never rolls over,
  R3 double-fill on redelivery, R4 live-short vs long/flat backtest mismatch, R5 SL/TP not enforced
  post-fill. Fixes awaiting user decision (recommended order: R1 decision → R2+R3 → R4+R5 → R6 TTL).

- 2026-07-07 — **Review fixes R1–R6 applied** (per user's go-ahead on the recommendation):
  **R1(a)** contracts-first: `SignalAggregatedEvent` + price/SL/TP/strategy_name (levels attached only
  when the final direction matches the strategy component's); aggregator buffers the strategy signal's
  order context; **risk-mgmt switched to `signal.aggregated`** (durable `risk-mgmt-aggregated`;
  `process_aggregated` + shared `_risk_check_and_order`; manual `POST /signal` kept). **R6** buffer TTL
  (`SIGNAL_TTL_SECONDS`, default 1 day; expired entries pruned, never resurface on regime changes).
  **R2** day-baseline rollover in `PaperBroker` (date-tagged, persisted, injectable clock). **R3**
  idempotent fills by order `event_id` (persisted dedup set; save-before-publish). **R4** long-only
  (SELL = exit, capped at held qty, skipped when flat; 409 on the manual route). **R5** protective
  exits (positions carry SL/TP; re-mark breach → paper exit + `OrderFilledEvent`). Counts: shared 155
  (+1), signal-aggregator 63 (+4), risk-mgmt 100 (+3), execution 44 (+14) → **all suites green (729)**;
  ruff + format + mypy clean. **Live full-chain verified on a real `nats-server`**:
  `signal.generated` (BUY, SL 95) → aggregated BUY+levels → sized `order.requested` (50 szt.) → fill
  @100 → mark @94 → **protective SL exit** (fill @94, flat, cash 99 700) → crisis regime →
  re-aggregated HOLD → **no new order**.

- 2026-07-07 — **Review gaps R7–R11 + P3s closed** (user: finish all gaps before new topics):
  **R7** contracts-first `StrategyStatusChangedEvent` metrics → optional; strategy consumes
  `backtest.strategy_revalidated` (renamed generic `EventSubscriber`, durable `strategy-revalidation`,
  `ensure_stream(BACKTEST)`), `StrategyHealthTracker.apply_status` + `apply_revalidation` (own-name
  filter; `deactivate`→`deactivated`; poison-term on unknown status; publishes status-changed on real
  transitions). Found & fixed a **latent bug**: `strategy.status_changed` had no JetStream stream —
  live publishes would have failed; added the `STRATEGY` stream (`strategy.>`). **R8** contracts-first
  `SignalAggregatedEvent.sector`; new `HttpCompanyClient` in signal-aggregator (queries
  company-classifier `/api/v1/company-classifier/companies/{symbol}`, positive-cache, graceful None;
  compose+Helm env `COMPANY_CLASSIFIER_URL`); risk-mgmt `_risk_check_and_order(..., sector)` →
  `PositionSizer.size(..., sector=...)` — regime sector caps now live. **R9** `POST /aggregate`
  documented ops/testing-only (+ optional `sector` in the body). **R10** `REGIME_BIAS["slowdown"] =
  None` — known-neutral regime contributes no component (no more weight-stealing HOLD). **R11**
  documented: "ml" source pre-provisioned, aggregation effectively 2-source until ml-pipeline emits.
  P3: EDGAR revenue tag fallbacks (`Revenues` → `RevenueFromContractWithCustomer[Ex/In]cludingAssessedTax`
  → `SalesRevenueNet`; per-period merge, earlier-tag priority); aggregator buffer TTL now ages from the
  **event emit timestamp** (durable replays can't resurrect stale signals; `DeliverPolicy.ALL` kept).
  Counts: shared 157 (+2), strategy 56 (+10), risk-mgmt 104 (+4), signal-aggregator 74 (+11),
  fundamental-data 30 (+3) → **all suites green (759)**; ruff + format + mypy clean. **Live-verified on
  a real `nats-server`**: (A) `backtest.strategy_revalidated` → probation + `strategy.status_changed`
  in the STRATEGY stream, foreign strategy ignored; (B) real uvicorn **company-classifier** over HTTP →
  aggregates carry `sector="Information Technology"` → **crisis blocks the BUY by sector cap**,
  slowdown re-agg stays 1-component (R10), expansion regime → sized `order.requested` (50 szt.).

- 2026-07-07 — **feature-engine Tier-2 enrichment** (first "Next" item): consumes
  `fundamentals.updated` (durable `feature-engine-fundamentals`; event announces, payload queried
  back via new `HttpFundamentalsClient` — 404→None/skip, transport error→NAK/redeliver) and
  `company.classified` (durable `feature-engine-company`; style straight from the event).
  New `core/attributes.py` (`SymbolAttributeStore`: per-symbol dict, `put` merges so the two
  handlers' disjoint keys coexist; InMemory + Redis backends) and `core/enrichment.py`
  (`fundamental_features`: f_score + net-margin/ROA/leverage, conservative on missing/zero inputs;
  `style_features`: growth (1,0) / value (0,1) / blend (0.5,0.5)). Attributes merged into vectors at
  **read time** (`get_features`/`ranked_universe`) → `/ranked` now ranks `f_score` cross-sectionally.
  Deliberate: attribute updates do NOT publish `features.ready` (no strategy re-evaluation on a
  fundamentals refresh). Renamed generic `MarketDataSubscriber`→`EventSubscriber`; `ensure_stream`
  FUNDAMENTALS+COMPANY; compose+Helm env `FUNDAMENTAL_DATA_URL`. feature-engine 93 tests (+15);
  ruff + format + mypy clean; **live-verified** (real uvicorn fundamental-data + company-classifier
  on one nats-server: POST /statements → f_score 7 merged with technicals; POST /classify → growth
  encoding; MSFT weak firm → f_score 1; ranked percentiles AAPL 1.0 / MSFT 0.0).

- 2026-07-07 — **Generic Helm chart** (top infra item closed): `values.yaml` restructured to a
  `services:` map (13 entries, kebab-case = k8s = compose names; all `enabled: true` now that every
  service is functional; env maps mirror compose inter-service URLs; market-data `needsDb`);
  new generic `templates/services.yaml` (Deployment+Service per enabled entry: health/ready probes,
  prometheus annotations, common env + secret refs, optional resources) replaces the market-data-only
  template; `ingress.yaml` now generates all 13 `/api/v1/{service}` routes (mirrors compose Traefik
  labels); dashboard containerPort fixed 8501→8000 (compose maps host 8501→container 8000);
  `values-prod.yaml` migrated to the map — replicas >1 only for non-subscribing services
  (market-data, dashboard) until pull/queue-group consumers land. **Render-verified with a real
  helm binary** (installed via Go): `helm lint` clean; dev+prod `helm template` → 13 Deployments +
  13 Services (+postgres) + Ingress with 13 paths; asserted env/secret/probe/replica invariants
  with a YAML checker. `make helm-template` target unchanged and working.

- 2026-07-07 — **Full 9-signal Piotroski** (contracts-first): `FinancialStatements` +
  `current_assets`/`current_liabilities`/`shares_outstanding` (ge=0, optional — legacy statements
  stay valid); `compute_f_score` adds `improving_current_ratio` (current-ratio Δ, degenerate
  denominator → conservative fail) and `no_dilution` (shares ≤ prior; flat counts as no issuance);
  `FScoreBreakdown.omitted` removed, `max_score` 7→9. EDGAR `TAG_MAP` += `AssetsCurrent`,
  `LiabilitiesCurrent`, `CommonStockSharesOutstanding` (fallbacks: weighted-average basic/diluted
  share tags; the candidate-merge machinery from the revenue fix reused as-is). Fixtures upgraded
  (improving firm now 9/9 with buyback, deteriorating 0/9 with dilution; legacy-shape statements cap
  at 7 — tested). Counts: shared 160 (+3), fundamental-data 33 (+3) → **all 14 suites green (780)**;
  ruff + format + mypy clean. Event path unchanged (same ingest→score→publish flow already
  live-verified), so no new NATS run needed.

- 2026-07-07 — **Scheduled triggers** (in-process, no new infra): new shared
  `trading_common.scheduler` — `PeriodicTask` (asyncio loop in the FastAPI lifespan; a failed run is
  logged and the schedule keeps ticking; clean `stop()` on shutdown; single-replica semantics
  documented — consistent with the push-consumer constraint) + `seconds_until_weekday_hour` for
  calendar alignment. Wired: **backtest** weekly Saturday-06:00-UTC walk-forward revalidation
  (OPT-IN `SCHEDULE_REVALIDATION_ENABLED` — its event drives strategy status via R7, so it needs the
  real activation-time baseline `REVALIDATION_ORIGINAL_OOS_SHARPE`); **macro-data** FRED refresh
  every 6h, first run at boot (gated on `FRED_API_KEY`; regime-transition-safe); **fundamental-data**
  weekly EDGAR `refresh_universe` over `REFRESH_SYMBOLS` (gated on `SEC_USER_AGENT` + non-empty
  universe; politeness pause between symbols). **ml-pipeline daily drift deliberately deferred**: a
  scheduled check has no live feature/prediction source until training/inference exists — lands with
  the PyTorch work (R11). Counts: shared 169 (+9), backtest 41 (+2), fundamental-data 36 (+3) →
  **all 14 suites green (794)**; ruff + format + mypy (incl. --strict on shared) clean. Verified:
  scheduler unit tests (fire/isolate/stop/align) + job-body test publishing a real
  `StrategyRevalidatedEvent` + **uvicorn lifespan smoke** on a real nats-server for all three
  services (schedulers armed/gated correctly; graceful shutdown).

- 2026-07-07 — **notification e-mail/SMTP + strategy.status_changed alerts**: new `EmailChannel`
  (stdlib `smtplib` + `EmailMessage` in `asyncio.to_thread` — no new dependency; STARTTLS + optional
  login; fresh connection per alert — human-scale volume; injectable `sender` for tests; enabled only
  when SMTP_HOST+EMAIL_FROM+EMAIL_TO are set, mirroring the Slack/Telegram gating). Fifth durable
  subscription `strategy.status_changed` (STRATEGY stream) → `from_strategy_status_changed` alert —
  the *applied* transition (R7) complementing the revalidation *recommendation*; demotion=warning,
  reactivation=info, optional-metrics-safe ("sharpe_90d n/a"). compose: SMTP_*/EMAIL_* passthrough
  env; Helm: secrets note extended. notification 33 tests (+5) → **all 14 suites green (799)**;
  ruff + format + mypy clean. **Live-verified on a real `nats-server`**: 5 events (incl. a real
  `strategy.status_changed`) → 5 correctly-graded alerts, each also rendered to a captured
  `EmailMessage` (subject `[WARNING] Strategy status: momentum_rank active → probation`).

- 2026-07-12 — **`docs/ml_integration_plan.md` written** (user delegated the ML-phase direction;
  the doc is the binding design — see Key docs). Core calls, each argued in the doc:
  **cross-sectional** pooled-universe learning on the ranked feature vectors (per-symbol
  prediction rejected at this data scale, per Gu–Kelly–Xiu); **shallow PyTorch MLP** `global_v1`
  (per-style stacks deferred until universe ≥ 200 — routing plumbing stays); **triple-barrier
  labels** (±2σ₂₀·√10 barriers, vertical h=10d, binary P(up-first)); **purged walk-forward +
  5d embargo**, decision metric = cost-adjusted OOS Sharpe of a top-quintile long-only portfolio,
  activation gate Sharpe>0.5 (holdout + 2/3 recent folds); **MLflow local-backend** registry with
  load-bearing metadata artifact; serving = `features.ready` → infer → new
  **`MlSignalGeneratedEvent`** (`ml.signal_generated`) → aggregator's third subscription
  (activates R11) as a **no-levels vote** — ML cannot trade alone, adaptive weights are the
  safety net; **daily drift schedule + delayed-label outcome loop** (resolved triple-barrier
  outcomes feed `record_outcome` + decay detection); meta-labeling, GBDT challenger, ML-derived
  levels, auto-pause → v2. Roadmap **ML-0…ML-4** (ML-0 moves pure `features`/`ranking` into
  trading-common so training reproduces serving bit-for-bit). Doc-only increment — no code.

- 2026-07-13 — **ML-0 landed** (dataset foundation per the plan): pure `features.py`/`ranking.py`
  **moved to trading-common** (`trading_common.features`/`ranking`; numpy now a shared dep;
  feature-engine imports the shared definitions — training/serving parity is structural, not
  aspirational; the 9 pure-function tests moved to shared, per the cost_filter precedent).
  ml-pipeline gains `core/labels.py` (triple barrier on the OHLC path: trailing σ₂₀ of log
  returns, barriers ±2σ√h, h=10, scan starts at the NEXT bar, same-bar double-touch resolves as
  loss, vertical resolves by net-return sign, truncated-untouched windows → unresolved/dropped;
  flat/degenerate σ → no label), `core/splits.py` (PEP-695-generic purged walk-forward folds over
  session dates; gap = horizon + embargo; degenerate gaps raise), `core/dataset.py`
  (`build_dataset`: per-session cross-section computed with the shared functions over the FULL
  feature-bearing universe — ranks match serving exactly — then labeled rows pooled;
  `EXCLUDED_FEATURES` drops price-level columns; fixed `feature_names` contract with neutral-0.5
  fill; macro one-hot; deterministic). Engineered-path tests pin every barrier case (incl. √h
  scaling and the truncated-tail asymmetry: touched → labeled, untouched → dropped). Counts:
  shared 178 (+9 moved), feature-engine 84 (−9), ml-pipeline 60 (+25) → **all 14 suites green
  (824)**; ruff + format + mypy (incl. --strict on shared) clean. Pure-compute increment — no
  event-path changes, so no NATS run.

- 2026-07-13 — **ML-1 landed** (training + registry per the plan): `core/model.py` — PyTorch
  `MlpClassifier` (2 hidden + dropout), `train_classifier` with pos_weight class balancing, early
  stopping **with a min-epochs warm-up** (found in testing: dropout-noisy val loss produced a lucky
  epoch-3 minimum that stopped training before any learning — AUC ~0.47 vs 0.78 after the fix) and
  LBFGS temperature calibration. `core/evaluation.py` — tie-aware Mann-Whitney AUC, Brier, and the
  decision metric: cost-adjusted Sharpe of the daily-rebalanced equal-weight top-quantile long-only
  portfolio (one-way turnover costing). `core/training.py` — `run_training`: untouched recent
  holdout + purged walk-forward folds (internal fit/val split also purged), gate = holdout
  Sharpe>0.5 AND ≥2/3 recent folds AND Brier ≤ base rate; final model retrained on full history,
  caller owns failed-gate semantics. `core/model_store.py` — **MLflow, sqlite backend**:
  runs+metrics logged, artifacts = `state_dict` + load-bearing `metadata.json`, **alias-based**
  promotion (`production`; stage API is deprecated), exact predict round-trip pinned by test.
  Service `train()` orchestrates fetch(HTTP)→dataset→gate→version+drift-baseline (`global_v1@vN`);
  routes `POST /models/train`, `POST /models/versions/{v}/promote`, `GET /models` extended.
  torch+mlflow deps in ml-pipeline only (pyproject notes the CPU wheel index for images); compose
  `ml_mlruns` volume + env; Helm env mirrored. **Gate sanity pinned by tests**: passes on a blatant
  trend universe, fails on driftless random walks. ml-pipeline 86 (+26) → **all 14 suites green
  (850)**; ruff + format + mypy clean; uvicorn lifespan smoke (real nats + real sqlite store) PASS.

- 2026-07-16 — **ML-2 landed** (serving per the plan — **activates R11 for real**): contracts-first
  `MlSignalGeneratedEvent` (`ml.signal_generated`, ML stream; NO levels — ML cannot trade alone).
  ml-pipeline `core/serving.py` (`ServingEngine`: production-alias model only; `features.ready`
  durable `ml-pipeline-features`, interval-filtered → ranked vector via new `HttpFeatureClient` +
  macro regime via `HttpMacroClient` with 10-min TTL cache → input row in the metadata's exact
  feature order — missing attr → neutral 0.5 like training, MAJORITY missing → **refusal** (schema
  drift ≠ sparsity) → dead-zone-silent publish at p≥0.55/p≤0.45); `service.promote()` hot-reloads
  serving. signal-aggregator: third durable `signal-aggregator-ml` buffers the latest per-symbol ML
  vote (emit-timestamp TTL) — joins strategy in aggregation, **never aggregates alone**;
  `ensure_stream(ML)`. compose+Helm: FEATURE_ENGINE_URL/MACRO_DATA_URL for ml-pipeline. Counts:
  shared 181 (+3), ml-pipeline 99 (+13), signal-aggregator 80 (+6) → **all 14 suites green (872)**;
  ruff + format + mypy clean. **Live ML-2 chain on a real `nats-server`** (real trained+promoted
  model from a real sqlite registry): `features.ready` → infer (p_up 1.0 on the engineered vector)
  → `ml.signal_generated` in ML → aggregator 1→2-component re-aggregation with strategy levels
  intact; uvicorn lifespan smoke (serving inactive until promotion; features durable subscribed).

**Next:** ML plan finishes with **ML-3** (daily drift schedule via `PeriodicTask` — the deferred
monitoring item, now that serving accumulates live vectors/predictions; delayed-label outcome
resolution → aggregator `record_outcome` + decay detection; `POST /models/{id}/pause`). Then:
real-data bootstrap + first true training run (needs market-data backfill of the configured
universe); deeper persistence (event-log/DB; pull/queue-group consumers for multi-replica HA);
notification digest (scheduler-driven, now trivial via `PeriodicTask`).

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
