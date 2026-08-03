# Trading System — Microservices Architecture

## Project overview

Production-grade algorithmic trading system. 13 independent Python microservices communicating
via NATS JetStream (events) and HTTP (request/response).

**Key docs:**
- **Project context/status/direction: this file** — see "Project status & direction" below (single source of truth I read every session)
- **Map of every document and which one wins: `docs/README.md`** — read it before adding a doc
- **DLACZEGO jest tak, a nie inaczej: `docs/decisions/`** — decyzje z dowodem, bez narracji.
  7 plików tematycznych (architektura, ryzyko, dane, etykiety, cechy, walidacja, agregacja)
  + otwarte decyzje D1–D8. **Czytaj to, zanim zmienisz coś w którymkolwiek z tych obszarów.**
- Founding architecture + 24-week build guide: `Plan_Rozwoju_Systemu_Tradingowego_2.md` (repo root).
  **Faza budowy NIE jest wykonana w całości** — na końcu jest „Aneks: status realizacji" z tabelą,
  które pozycje checklist Faz 0–5 są zbudowane, a które nie
- Framework supplement — reference for the 12 components + reference implementations of deleted
  calculators: `docs/framework_supplement.md`
- **ML architecture: `docs/ml_integration_plan.md`** — cross-sectional shallow model on ranked
  features, triple-barrier labels, purged walk-forward, MLflow, `ml.signal_generated` → aggregator,
  drift monitoring, roadmap ML-0…ML-4. **Architecture still binding; the quantitative choices
  (§4–§6: horizon, barrier width, feature set, gate) are superseded** by `core/gate.py` and the
  prediction plan. Read it before touching ml-pipeline.

## Project status & direction

> Single living context block. Read this first every session. Keep the progress log append-only.
> If a fresh analysis surfaces new bugs or improvement ideas, **propose them here and to the user** —
> do not silently proceed.

**Phase:** 1 — Foundation. The earlier priority inversion is **resolved**: the foundation was built
and the framework components wired into a working **end-to-end paper-trading loop** (market-data →
feature-engine → strategy → risk-mgmt → execution → portfolio feedback) plus backtest + ml-pipeline
monitoring, notification alerting, and a dashboard BFF over the HTTP APIs. **All 13 services (9 core +
4 ML/AI extension) are now functionally implemented** — no skeletons left; Direction #3 complete.

**Verified ground truth** (test counts measured 2026-08-03 on Python 3.12, not from memory —
**1418 testów zielonych**; `ruff` + `ruff format` + `mypy` czyste, `--strict` na shared):

| Komponent | Port | Rola | Testy |
|---|---|---|---|
| `shared/trading-common` | — | Kontrakty, wspólne obliczenia, **registry strategii** i **statystyki ryzyka** — wszystko, co musi być identyczne po obu stronach granicy serwisów | 325 |
| `market-data` | 8001 | OHLCV: pobranie (Yahoo/Alpha Vantage), walidacja, TimescaleDB, cache, harmonogram przyrostowy | 71 |
| `feature-engine` | 8002 | Wskaźniki Tier-1 + wzbogacenie Tier-2, rangi przekrojowe (`/ranked`) | 38 |
| `strategy` | 8003 | **Każda aktywna reguła** z registry → `RiskEnvelope` → `CostAwareFilter` → własny sygnał; monitor degradacji per strategia | 60 |
| `backtest` | 8004 | Ocena **reguły z registry** na historii symbolu + walk-forward, tygodniowa rewalidacja, krzywa kapitału | 58 |
| `ml-pipeline` | 8005 | Zbiór, trening, bramka G0–G5, rejestr MLflow, serwowanie, monitoring driftu, badania, **ważność cech** | 340 |
| `risk-mgmt` | 8006 | Sizing adaptacyjny, limity reżimowe i sektorowe, wyłącznik z zatrzaskiem, rejestr zleceń | 133 |
| `execution` | 8007 | Paper broker, wyjścia ochronne SL/TP, likwidacja na BLACK, feedback portfela, **historia kapitału** | 60 |
| `notification` | 8008 | 5 strumieni → alerty (log/Slack/Telegram/e-mail) | 33 |
| `fundamental-data` | 8009 | SEC EDGAR, Piotroski 9/9, **panel point-in-time** (`filed_at`) | 54 |
| `macro-data` | 8010 | FRED + detekcja reżimu → `macro.regime_changed`, **panel vintage (ALFRED)** | 54 |
| `company-classifier` | 8011 | Profil → styl inwestycyjny + routing stosu modeli | 25 |
| `signal-aggregator` | 8012 | **Węzeł decyzyjny**: każda strategia osobno + ML + makro → jedna decyzja z poziomami i sektorem | 97 |
| `dashboard` | 8501 | BFF nad HTTP pozostałych serwisów + **6 sekcji z wykresami** (kapitał, ryzyko, strategie, backtest, ML z ważnością cech, zdrowie) | 37 |
| `scripts/` | — | Bootstrap uniwersum, diagnostyka stacku, audyt zależności | 33 |

Co z tego jest **wiążące**, a nie tylko opisowe:

- **`trading-common` jest granicą.** Leży w nim wszystko, co musi dać ten sam wynik w treningu i na
  produkcji: `features` (+ `FEATURE_LOOKBACK=300` / `FULL_HISTORY=253` jako jedna stała okna),
  `ranking`, `fundamentals` (reguła as-of + wyprowadzenie czynników), `sectors`, `prices`,
  `RiskEnvelope`, `CostAwareFilter`, `sizing`, `scheduler`, `timeutil`, `constants.MAX_OHLCV_LIMIT`,
  **`risk_metrics`** (VaR/obsunięcie/korelacje — dashboard renderuje, risk-mgmt rozumuje, a dwie
  definicje „obsunięcia" w końcu poróżniłyby się co do tego, czy limit został przekroczony)
  oraz **`strategies` (registry reguł)** — backtest musi oceniać dokładnie tę regułę, którą handluje
  serwis strategii, a serwisy nie mogą się nawzajem importować.
  Duplikat tej arytmetyki po dwóch stronach granicy to rozjazd train/serve czekający na wystąpienie.
- **Serwis A nigdy nie importuje serwisu B.** Wspólne typy idą do `trading-common`.
- **Wszystkie 13 serwisów są funkcjonalne** — `/health` `/ready` `/metrics`, żadnych szkieletów.
- **Żaden komponent frameworku nie jest osierocony.** `decay_monitor` + `cost_filter` → strategy;
  `adaptive_weights` → signal-aggregator; `adaptive_sizing` + `regime_allocator` → risk-mgmt;
  `continuous_validation` → backtest; `drift_detector` → ml-pipeline. Skasowane 2026-07-25 jako
  martwy kod: `vol_regime`, `earnings_decay`, `cross_asset` — implementacje referencyjne zostają
  w `docs/framework_supplement.md` i w historii gita.

Szczegóły „jak i dlaczego" per komponent: [`docs/decisions/`](docs/decisions/); przebieg prac:
[`docs/archive/progress_log_2026-06_2026-08.md`](docs/archive/progress_log_2026-06_2026-08.md).

**Direction.** Kierunki #1–#3 (fundament, wpięcie komponentów frameworku, serwisy 10–13) są
**zamknięte** — proweniencja w [`docs/decisions/`](docs/decisions/). Obowiązuje reguła stała:
**contracts-first** — rozszerz `shared/trading-common`, zanim dodasz jakikolwiek typ przechodzący
przez granicę serwisów.

**Gdzie projekt stoi i co robimy dalej** — patrz „Next" na końcu tej sekcji.

**Known issues / tech debt** (propose a fix when you touch the area):
- [**from the archived audit backlog, 2026-07-28** — these are NOT part of the prediction track and
  would otherwise have been lost when `backlog_2026_07_27.md` was archived]
  - **FLOW-2** `CostAwareFilter`'s `base_edge_bps = 200` is a constant pulled from thin air. Units
    are consistent (verified — the audit was wrong on that), but the number should be a calibrated
    expected return, e.g. `(2p−1)·pt_mult·σ·√h`; the function already accepts
    `expected_return_bps` when given.
  - **FLOW-3 / D3** the macro regime is applied TWICE: as a directional bias in the aggregator
    (`REGIME_BIAS`) and as exposure/sector caps in risk-mgmt. The caps are its proper home; the
    aggregator bias is market timing on one global variable. Open decision.
  - **FLOW-4** portfolio state reaches risk-mgmt only over HTTP (`push_portfolio`); a failed call
    silently desynchronizes sizing from reality. An event would be the honest channel.
  - **FLOW-5** the adaptive-weight loop learns from a *modelled* outcome for the strategy source
    (ML now uses realized triple-barrier outcomes — ML-3); strategy should get the same treatment.
  - **FLOW-7** `StrategyDecayMonitor` has no probation trial period — a strategy is demoted on the
    metric with no observation window.
  - **FLOW-8** sectors are free-form strings; an unmatched name blocks in restrictive regimes
    (conservative but silent). A GICS enum + a per-symbol map is a prerequisite for the
    sector-neutral ranks planned in the prediction plan (P2-2).
  - **FLOW-9** no written criterion for moving from paper to real capital beyond "30 days positive
    Sharpe" — needs the full checklist (max drawdown observed, fill realism, breaker rehearsal).
  - [✅ done 2026-07-30] ~~**Circuit breaker does not latch**~~: BLACK now latches until a human
    clears it (`POST /circuit-breaker/reset`, refused while the breach still stands) and RED holds
    for the rest of the session instead of lifting on an intraday bounce; the latch is persisted,
    so a container restart is no longer a way to satisfy "require human restart".
  - **TS-1** `init-db.sql` włącza kompresję TimescaleDB na `market_data.ohlcv` z polityką 7 dni,
    a market-data **z założenia przepisuje historię** (naprawa po restatemencie odświeża każdy bar,
    backfill wolno powtórzyć, sonda diagnostyczna pisze). Zapis do skompresowanego chunka to inna
    ścieżka niż do świeżego i zależy od wersji rozszerzenia. Nie da się tego odtworzyć w piaskownicy
    (obraz timescale niepobieralny), więc `diagnose.py` **raportuje** teraz liczbę skompresowanych
    chunków — decyzja (wydłużyć próg kompresji albo ją wyłączyć dla tej tabeli) czeka na tę liczbę.
  - [✅ done 2026-08-02] ~~**P2-4** `macro-data` nie ma warstwy trwałości~~: panel **vintage**
    (ALFRED `realtime_start`), odczyt as-of po DWÓCH osiach czasu, `regime_by_date` ma wreszcie
    źródło. **Zostaje do zrobienia u użytkownika: `POST /backfill`** — egress do FRED jest
    zablokowany w piaskownicy, więc kod jest zweryfikowany na realnym PostgreSQL-u, ale panel jest
    pusty do czasu backfillu. Do tego czasu kolumny `macro_*` nadal wypadają jako stałe, co serwis
    **mówi wprost** w logu i w `selection.macro.days_with_regime`. D8 odblokowana.
  - **P2-5** (z zarchiwizowanego planu predykcji, opcjonalne) fractional differentiation — do
    rozważenia dopiero, gdy cechy stacjonarne się wyczerpią.
  - **D7** the backtest engine still rebalances daily while ML evaluation uses `1/h` overlapping
    tranches — the two are not comparable until the engine gets tranches too.
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
- [env] Market-data egress is blocked from the sandbox (query1/query2.finance.yahoo.com and
  stooq.com → curl 000 through the proxy, like SEC/pytorch.org before). A REAL backfill/training
  run must happen on a Docker-capable machine (`make up` → `make bootstrap-universe`); in-sandbox
  rehearsals substitute a synthetic fetcher inside a real market-data app (smoke2/md_runner pattern).
- [env] Docker CLI + daemon are available (start `dockerd` as root if the socket is missing). Under
  the **Trusted** egress policy, Docker Hub *registry* hosts are allowlisted but NOT the blob CDN
  Docker actually redirects to (`production.cloudfront.docker.com` → 403; the allowlist only has the
  Cloudflare variant `production.cloudflare.docker.com`). → `docker pull` / `docker compose up` fail
  under Trusted. Fix: edit the environment's **Network access** → **Full** (or **Custom** + add
  `production.cloudfront.docker.com`), then start a new session.
  To verify NATS/JetStream **without Docker** (Go module proxy is allowlisted):
  `GOSUMDB=off go install github.com/nats-io/nats-server/v2@v2.10.22` then run `nats-server -js`.

**Progress log.** Wpisy od 2026-06-25 do 2026-07-29 (68 pozycji, 1400 linii) są w
[`docs/archive/progress_log_2026-06_2026-08.md`](docs/archive/progress_log_2026-06_2026-08.md) —
verbatim, jako zapis proweniencji. **Destylat decyzji z tamtego okresu, bez narracji, żyje w
[`docs/decisions/`](docs/decisions/)** i to jest miejsce, do którego się zagląda, pytając „dlaczego
tak".

Skrót łuku: czerwiec — audyt repo i naprawa niespójności kontraktów; czerwiec/lipiec — wdrożenie
13 serwisów od szkieletów do działającej pętli papierowej (market-data → feature-engine → strategy
→ signal-aggregator → risk-mgmt → execution → portfolio feedback) plus backtest, ml-pipeline,
notification, dashboard; lipiec — 12 komponentów frameworku (koperta ryzyka, filtr kosztów,
monitory degradacji i driftu, alokator reżimowy, wagi adaptacyjne), serwisy 9–12, ML-0…ML-3
(zbiór, trening, serwowanie, monitoring), audyt zewnętrzny i Tier 0 (naprawa POMIARU, nie modelu),
bramka G0–G5, etapy E0–E5 planu predykcji; koniec lipca / sierpień — kampania pomiarowa na realnych
danych (414 symboli × 20 lat) i seria defektów operacyjnych znalezionych dopiero na niej.

**Ostatnie wpisy (od 2026-07-30) zostają tutaj — dalej append-only:**

- 2026-07-30 — **market-data: harmonogram przyrostowy — silnik, którego system nie miał.**
  Przegląd wszystkich 13 serwisów pokazał jedną lukę strukturalną: harmonogram miały backtest,
  macro-data, fundamental-data i ml-pipeline — czyli wszystkie **oprócz źródła**. Cały łańcuch
  zdarzeń jest zakotwiczony w `market_data.updated`, więc bez zadania cyklicznego w market-dacie
  **system nie przepracowałby samodzielnie ani jednego dnia**, a reguła „30 dni papieru z dodatnim
  Sharpe'em przed realnym kapitałem" nie ma jak zacząć się naliczać.
  **Pobieranie wznawia się od ostatniej zapisanej świecy, nie od wczoraj** (wymaganie użytkownika):
  `core/incremental.py` — `plan_fetch` liczy okno od najnowszego posiadanego baru minus zakładka,
  więc tydzień przestoju naprawia się sam zamiast zostawić trwałą dziurę; nic innego w systemie
  nigdy nie wraca do przeszłej sesji. **Najważniejsza część to jednak wykrywanie restatementu**:
  `adj_close` nie jest własnością świecy, tylko świecy **plus wszystkich późniejszych zdarzeń
  korporacyjnych**, więc po splicie dostawca przepisuje całą historię. Czysto przyrostowe
  pobieranie zostawiłoby stare bary na przedsplitowej skali — seria wyglądałaby wiarygodnie i była
  błędna dokładnie na złączeniu, a ponieważ cechy i etykiety liczą się na cenach skorygowanych
  (P0-1), szkoda trafiłaby do modelu, nie do logu. Zakładka służy właśnie temu: `adjustment_drifted`
  porównuje **współczynnik** `adj_close/close` dla dat obecnych po obu stronach i przy zmianie
  wymusza pełne odświeżenie. **Defekt złapany na żywym Postgresie**: naprawa pobierała
  `initial_history_days` (6 lat), a przy backfillu 20-letnim zostawiłoby to **14 lat na starej
  skali** — okno naprawy sięga teraz po `earliest_timestamp`, czyli po wszystko, co trzymamy.
  **Masowy upsert** zamiast `session.merge` per świeca (merge robi SELECT przed decyzją, więc
  backfill 486 symboli × 20 lat to było ~2.4 mln round-tripów): zmierzone **5000 świec w 1.12 s**.
  Do tego `POST /sync` (ten sam job ręcznie — przycisk naprawy po przestoju) i `GET /sync/status`
  (staleness per symbol, bo harmonogram, który po cichu stanął, wygląda identycznie jak taki, który
  zadziałał i nic nie znalazł). Wspólne: `seconds_until_hour` w `trading_common.scheduler` (dzienny
  odpowiednik istniejącego tygodniowego — praca ma lądować po zamknięciu sesji, a nie 24 h po
  starcie kontenera) oraz **`as_utc` przeniesione do `trading_common.timeutil`** i re-eksportowane
  z `fundamentals` — drugi serwis potrzebował tej samej reguły granicy. Ta reguła wyłapała
  **dwa błędy**: jawny `TypeError` przy porównaniu naiwnej daty z sqlite ze świadomą, i — groźniejszy
  — **cichy**: dopasowanie świec po znaczniku czasu nigdy nie trafiało, więc wykrywanie splitu
  zwracało „brak zmian" zamiast błędu. **Prod: `replicaCount` market-daty 2 → 1**, bo `PeriodicTask`
  ma semantykę jednoreplikową i każda replika pobierałaby to samo osobno.
  **Przy okazji, w Helmie: duplikat klucza `env:`** — market-data miał już taki blok niżej, więc
  YAML zachował ostatni i po cichu wyrzucił całą nową konfigurację; `helm lint` przechodził, render
  był „poprawny", efekt zerowy. Wykryte sprawdzeniem renderu, nie założeniem.
  Liczniki: shared 236 (+3), market-data 65 (+33), fundamental-data 47 → **bateria 1141**;
  ruff + format + mypy (`--strict` na shared) czyste; compose i `helm lint`/`template` (dev + prod)
  zweryfikowane. **13/13 na prawdziwym PostgreSQL-u 16** ze schematem z `init-db.sql` (nie z echa
  ORM-a): masowy upsert idempotentny, `created_at` nietknięte przez ponowny zapis, TIMESTAMPTZ
  wraca świadomy strefy, wznowienie pyta o 6 dni zamiast o całą historię, split wymusza naprawę
  sięgającą do 2012-11-20 (czyli do początku danych) i **zero barów zostaje na starej skali**,
  a jeden zepsuty symbol nie przerywa uniwersum.

- 2026-07-30 — **Kampania pomiarowa u użytkownika: cztery usterki wyszły dopiero na realnym biegu.**
  Kolejno, tak jak wychodziły. **(1) Kontrakt danych odrzucał zdrowy zbiór** (1450 sesji + 63
  purge = 1513 wobec 1512 pobranych świec) — porównanie było o jeden zbyt ostre, a strażnik
  ucięcia sam się unieważniał; przy okazji `--symbols A,B,C` z PowerShella dociera jako JEDEN token
  „A B C", więc `split_symbols` przyjmuje teraz przecinki **albo** białe znaki (separator zależny od
  pamiętania o cudzysłowach to pułapka, nie interfejs). **(2) ml-pipeline stawał się `unhealthy`
  podczas `capacity-probe`**: trasy liczące minutami trzymały pętlę zdarzeń, więc `/health` nie
  odpowiadał w budżecie 10 s × 3 — ciężka praca poszła do `asyncio.to_thread`, a healthcheck dostał
  realistyczny budżet. **(3) Zatrzask wyłącznika** (patrz „Known issues" wyżej): BLACK trzyma do
  ręcznego resetu, RED do końca sesji, jedno i drugie utrwalone — restart kontenera przestał być
  najprostszym obejściem reguły „require human restart". **(4) Dzisiejsza, najkosztowniejsza:
  wszystkie badania i trening zwróciły `HTTP 500: {}`.** Przyczyna: pułap `limit` zadeklarowany
  **dwa razy w dwóch wartościach** — ml-pipeline przyjmował `le=10_000`, market-data wydawał
  `le=5000`, a żądanie 20 lat (5040 sesji + 253 rozgrzewki = **5293**) leżało dokładnie pomiędzy,
  więc padło dopiero po zbackfillowaniu 455 symboli. Teraz jedna wspólna stała
  **`trading_common.constants.MAX_OHLCV_LIMIT`**, czytana przez route market-daty, `TrainRequest`
  ml-pipeline i oba modele żądań backtestu; skrypt bootstrapu jest świadomie stdlib-only (biegnie
  poza kontenerami), więc równość `MAX_TRAIN_LIMIT == MAX_OHLCV_LIMIT` pinuje test w
  `scripts/tests`. **Drugi defekt tego samego zdarzenia kosztował więcej niż pierwszy**: trasy
  ml-pipeline mapowały tylko `RuntimeError`/`ValueError`, więc `httpx.HTTPStatusError` uciekał jako
  **pusty** 500 (Starlette odpowiada zwykłym tekstem) i sześć różnych awarii wyglądało w raporcie
  identycznie; wspólny `_mapped_errors` daje teraz **502 z nazwą upstreamu, jego statusu i URL-a**,
  500 z nazwą typu dla nieprzewidzianych, i zachowuje 422 kontraktu danych / 503 / 400. Do tego
  `_request` w skrypcie **wyrzucał** ciało odpowiedzi, gdy nie było JSON-em — stąd dosłowne `{}`;
  teraz zachowuje tekst, a puste ciało nazywa wprost. **Znalezione przy okazji**: job `test-scripts`
  w CI instalował wyłącznie `pytest`, podczas gdy testy skryptów importują `trading_common` —
  wywracał się na **zbieraniu** testów, niewidocznie, bo CI nie biegnie na gałęziach `claude/*`.
  Liczniki: market-data 67 (+2), ml-pipeline 305 (+8), scripts 26 (+1), reszta bez zmian →
  **bateria 1200**; ruff + format + mypy (`--strict` na shared) czyste. Kontrola anty-szczęściowa:
  po cofnięciu stałej do 5000 nowe testy padają po obu stronach dokładnie tym 422.
  **Zweryfikowane na żywo (10/10)**: realna market-data (własny lifespan, silnik sqlite; route'y,
  repozytorium, cache produkcyjne) na uvicornie + **realny `HttpMarketDataClient` ml-pipeline**
  wyciąga 5293 świece z URL-a z tracebacku użytkownika, a `MAX_OHLCV_LIMIT + 1` nadal dostaje 422;
  realny ml-pipeline (na realnym `nats-server`) nad upstreamem odpowiadającym 422 zwraca
  `502: upstream 422 from http://…/ohlcv/AMZN?interval=1d&limit=5293` — czyli to, co powinno było
  stać w raporcie zamiast `HTTP 500: {}`.

- 2026-07-31 — **Przegląd `diagnose.py` na zgłoszenie „nie jesteśmy zabezpieczeni przed ponownym
  wrzuceniem tych samych danych — sypie 500" — i dwa realne defekty znalezione po drodze.**
  **(1) Regresja z masowego upsertu**: partia zawierająca ten sam `(symbol, interval, ts)` dwa razy
  wywracała **cały** zapis symbolu — Postgres odmawia `ON CONFLICT DO UPDATE`, którego własna lista
  VALUES nazywa ten sam klucz dwukrotnie („cannot affect row a second time"), a poprzedni
  `session.merge` per bar po prostu stosował drugi wiersz. `save_bars` **deduplikuje** (wygrywa
  ostatnie wystąpienie, klucz po INSTANCIE — naiwny i świadomy znacznik tej samej chwili to dwie
  różne wartości Pythona i jeden wiersz w TIMESTAMPTZ, czyli dokładnie para, którą naiwna
  deduplikacja by przepuściła) i loguje, ile wierszy się powtórzyło; zwracana liczba to teraz
  faktycznie zapisane bary. **Zweryfikowane na prawdziwym Postgresie w obie strony**: przed
  poprawką `CardinalityViolationError`, po — 4/4, a ponowny zapis **identycznego** okna był
  idempotentny już wcześniej (czyli sam powtórzony fetch NIE jest tym 500).
  **(2) Kontrola pokrycia obcinała własne wejście**: `validate_coverage` miała zaszyte `limit=5000`
  mimo komentarza, że czyta CAŁĄ historię po to, by porównanie z `train_limit` było prawdziwe.
  W raporcie użytkownika **346 z 455 symboli** ma dokładnie 5000 sesji i `first` spóźnione o sześć
  tygodni względem zamówionego zakresu, a strażnik `stored_max > train_limit` **nie mógł zadziałać**,
  bo `stored_max` był z definicji poniżej `train_limit`. Limit bierze się teraz ze wspólnego pułapu,
  a trafienie w sufit odczytu jest **nazwane** w nocie („session count is a lower bound") — liczba,
  która pochodzi z zapytania, nie ma udawać pomiaru. Test, który to pinował, pinował **defekt**
  (`assert "limit=5000"`), więc został przepisany na własność, nie na literał.
  **(3) Powody awarii wracają do raportu**: `backfill` zbiera `symbol → dlaczego`, konsola grupuje
  awarie po przyczynie. Dotąd artefakt miał samą listę nazw, więc „41 symboli padło" było nie do
  odróżnienia od „41 tickerów już nie istnieje" — na liście, która **celowo** zawiera spółki wycofane.
  **`diagnose.py`**: nowa sekcja **TIMESCALEDB** (hypertabela, liczba chunków, ile SKOMPRESOWANYCH,
  zadania) — to jedyna własność tej bazy, której nie odtworzy żaden test na zwykłym Postgresie
  (patrz TS-1); sonda fetchu robi teraz **dwa** przebiegi tym samym oknem i wprost orzeka, czy zapis
  jest idempotentny (jednorazowa sonda nie odróżnia „działa" od „działa raz"); `psql` przy błędzie
  pokazuje linię `ERROR:`, a nie ostatnią (czyli samotny daszek `^`). **Błąd we własnej pierwszej
  wersji tej sekcji, złapany uruchomieniem**: `SELECT CASE WHEN to_regclass('timescaledb_information.…')
  IS NULL THEN 'n/d' ELSE (…) END` **nie chroni** — Postgres rozwiązuje nazwy relacji przy
  PARSOWANIU, więc na bazie bez rozszerzenia leciały trzy „relation does not exist"; teraz bramką
  jest osobne zapytanie o `pg_extension`. **Sprawdzone i NIE zmienione** (weryfikacja zamiast
  założenia): `docker compose ps --all --format json` w Compose **v5.1.1** to nadal **JSONL**, nie
  tablica — parser był poprawny (potwierdzone na dwóch realnych kontenerach z lokalnie zbudowanego
  obrazu); porty wszystkich 13 serwisów w `SERVICES` zgadzają się z compose.
  Liczniki: market-data 69 (+2), scripts 29 (+3) → **bateria 1205**; ruff + format + mypy czyste.
  Kontrola anty-szczęściowa: wszystkie 3 nowe testy skryptów padają na kodzie sprzed poprawki.

- 2026-07-31 — **Traceback był wyrzucany przez logger we WSZYSTKICH 13 serwisach.**
  Użytkownik wkleił linię, która brzmi dokładnie jak diagnoza i nie jest nią:
  `{"symbol": "AAPL", "exc_info": true, "event": "Fetch-and-store failed", "level": "error"}`.
  `JSONRenderer` **nie formatuje** `exc_info` — serializuje go jako goły boolean i traceback
  przepada; `structlog.processors.format_exc_info` nigdy nie było w łańcuchu procesorów, a
  `app.debug` pod compose jest False, więc dotyczyło to wyłącznie ścieżki produkcyjnej. Każde
  `logger.exception` w systemie od zawsze mówiło „coś padło" bez możliwości zapytania „co".
  Odtworzone w izolacji (ta sama linia bez poprawki, pełny traceback w polu `exception` po niej),
  poprawione w 13 identycznych kopiach `observability.py`, przypięte testem po obu stronach
  (JSON niesie traceback; ConsoleRenderer w dev nadal renderuje po swojemu, ładniej).
  **`diagnose.py`**: sekcja z logami oznacza teraz trafienia `>>`, a linie bez znacznika opisuje
  wprost jako KONTEKST — bez tego sąsiednie `200 OK` czytały się jak awarie (pytanie użytkownika).
  Obcięcie dopasowanej linii podniesione 200 → 2000 znaków, bo traceback jedzie teraz w JEDNEJ
  linii JSON-a i stare obcięcie wyrzucałoby dokładnie to, co właśnie naprawiliśmy. Predykat łapie
  też `"exception"`. Liczniki: market-data 71 (+2) → **bateria 1207**; ruff + format + mypy czyste.

- 2026-07-31 — **Badania przeszły na 414 symbolach × 20 lat; trening i alpha-decay padły na
  timeoucie KLIENTA.** Wyniki w następnym wpisie; tu poprawka, bez której kolejny bieg straci to
  samo. Read timeout to zdanie o gnieździe, nie o pracy: **uvicorn nie anuluje endpointu, gdy
  klient się rozłączy** (zmierzone osobną sondą, nie założone), więc trening biegł dalej, zapisał
  się do MLflow — a raport, istniejący wyłącznie w odpowiedzi HTTP, przepadł. Teraz ml-pipeline
  **zapamiętuje ostatni ukończony bieg** każdej długiej operacji (`record_run` wołane w trasie po
  awaicie) i wydaje go przez `GET /runs` + `GET /runs/{operation}`; skrypt bootstrapu po timeoucie
  **odpytuje** ten endpoint zamiast się poddawać. Pułapka, przed którą broni `previous`: kontener
  może trzymać WCZEŚNIEJSZY raport tej samej operacji, a zwrócenie go byłoby gorsze niż timeout —
  wyglądałoby jak świeża odpowiedź; akceptowany jest tylko raport, którego przed wywołaniem nie
  było. Nieudany bieg **nie jest** zapisywany (polling nie może wziąć błędu za wynik), a 404 znaczy
  „jeszcze nie skończył", nigdy „skończył i jest pusty". `--train-timeout` 1800 → 5400 s, plus do
  60 min dobijania. Liczniki: ml-pipeline 309 (+4), scripts 32 (+3) → **bateria 1214**; ruff +
  format + mypy czyste.

- 2026-08-01 — **Trening #3 (414 symboli × 20 lat) i sonda pojemności: model ZAPADŁ SIĘ, a kontrola
  sondy była za słaba.** Bieg przeszedł w całości (1 868 128 próbek, 4778 sesji, 61 foldów, wersja 3
  w MLflow) i **bramka odrzuciła model na 5 z 6 warunków**, zaczynając od G0: `pred_std = 0.0`,
  `auc_train = 0.5000`, IC dokładnie 0.0000 — sieć wypuszczała **jedną liczbę dla każdego wiersza**.
  Sygnatura skali: ta sama konfiguracja w sondzie (60 tys. wierszy) daje `auc_train` 0.584, a na
  1,78 mln wierszy zapada się do stałej; foldy (krótsze okna) mają `auc_train` śr. 0.521 i
  `pred_std` ~0.013, przy czym **dwa foldy też są dokładnie 0.0**.
  **Eksperyment obalił obie moje hipotezy mechanizmu**: na syntetycznym panelu ze słabym sygnałem
  ani rosnąca liczba kroków (1k → 63k), ani weight decay, ani obniżone lr nie powodują zapadnięcia
  (`pred_std` stabilnie ~0.029). Czyli to nie jest sam optymalizator.
  **To skierowało mnie na samą sondę — i tam był defekt.** Kontrola permutowała etykiety
  **globalnie**, co niszczy naraz dwie rzeczy: sparowanie nazwy z etykietą (pytanie sondy) ORAZ to,
  że etykiety jednej sesji w większości się zgadzają (triple barrier h=10 to w dużej mierze efekt
  DATY — zmierzona korelacja par na realnym panelu to 0.358). Cechy są trwałe, więc duży model uczy
  się „ta konfiguracja to ta data, a ta data rosła" **bez żadnej przewagi przekrojowej**, a kontrola
  po globalnym przetasowaniu tego nie potrafi. **Zmierzone**: na panelu, w którym etykieta jest
  czystym efektem daty i przekrojowej przewidywalności NIE MA z konstrukcji, stara sonda dała gap
  **+0.066 przy progu 0.05, pełną separację i werdykt „learnable structure EXISTS"**. Po zmianie na
  permutację **wewnątrz sesji** (zachowuje dzienną stopę pozytywów, oddaje tylko sparowanie nazw)
  ten sam panel daje +0.0038 i „NO learnable structure". Przy okazji: kontrola, która nie znajdzie
  swoich sesji, **rzuca teraz błąd** — cicho niepermutująca kontrola dawałaby zawsze „brak
  struktury", czyli pomyłkę w stronę, której nikt nie zauważa.
  **Konsekwencja dla wcześniejszego wpisu**: werdykt „struktura ISTNIEJE" (gap +0.201) pochodzi ze
  starej, zawyżającej kontroli i **wymaga powtórzenia**; +0.201 jest wyraźnie ponad zmierzoną
  podłogą artefaktu +0.066, więc nie jest to czysty artefakt, ale liczba nie jest wiarygodna.
  **Alpha decay (P5-4) — pierwszy pełny wynik**: próg z permutowanego nulla |t| = 4.03; cztery cechy
  go przechodzą. `amihud_20` IC **+0.0619** (t 6.26) i `dollar_volume_20` −0.0472 (t −5.22) — obie
  rosną monotonicznie z horyzontem, szczyt na 63 sesjach, i są **niewrażliwe na opóźnienie wejścia**
  (99.8% IC po 5 dniach): to premia za niepłynność/rozmiar, czyli premia za ryzyko, nie sygnał
  czasowy. `return_1d` IC −0.0177 (t −6.68) szczytuje na h=1 i **traci 72% IC po jednym dniu
  opóźnienia** — realna alfa krótkoterminowa, której nasza architektura (świece dzienne, cechy po
  zamknięciu, zlecenie następnej sesji) z definicji nie zdąży zebrać. `return_5d` (t −4.60) pośrodku.
  Werdykt studium: IC szczytuje średnio na **34 sesjach**, więc horyzont 10 zamyka etykietę w środku
  ruchu. Zgadza się to z target study, które wskazało horyzont **63**.
  ml-pipeline 312 (+2); ruff + format + mypy czyste.

- 2026-08-01 — **Panel fundamentów mógł trzymać najwyżej DWA lata na spółkę — pytanie użytkownika
  („czy nie powinniśmy mieć danych finansowych przed strojeniem modelu?") trafiło w realną dziurę.**
  P2-3 zbudowało poprawny panel point-in-time (`filed_at`, odczyt as-of, `GET /panel`, złączenie w
  `build_dataset`), ale **jedyną ścieżką, która cokolwiek do niego zapisywała, był `refresh`** —
  a ten woła `latest_statements(count=2)`. Odpowiedź na „co wiemy teraz" (pytanie serwowania) była
  jednocześnie całą historią, jaką miało dostać uczenie. Złączenie po 20 latach dawało więc
  neutralne 0.5 na prawie każdej sesji, czyli rodzinę cech obecną wyłącznie z nazwy. Dane były na
  wyciągnięcie ręki: `EdgarClient` **już pobiera wszystkie roczne okresy** i dopiero na końcu tnie
  `periods[:count]`. Nowe `refresh_history` zapisuje **cały** panel (każdy okres punktowany
  względem SWOJEGO poprzednika — F-Score porównuje kolejne lata, więc ocenianie 2012 względem 2025
  dałoby liczbę wyglądającą jak wynik i nic nieznaczącą), publikuje **jedno** zdarzenie zamiast
  dwudziestu (`fundamentals.updated` mówi, że zmieniła się bieżąca wiedza — odtwarzanie 20 lat
  budziłoby każdego konsumenta 20 razy na spółkę), route `POST /backfill/{symbol}?periods=N`.
  **Druga dziura, obok**: `SCHEDULE_REFRESH_ENABLED` / `REFRESH_SYMBOLS` / `REFRESH_SYMBOL_PAUSE_S`
  **nigdy nie były przekazane w compose ani w Helmie**, więc tygodniowy harmonogram odświeżania
  istniał w kodzie i był nieosiągalny w działającym systemie — panel po jednorazowym wypełnieniu
  i tak by się zestarzał. Dodane po obu stronach (render sprawdzony: `helm template` i
  `docker compose config`). Skrypt bootstrapu: `--fundamentals-backfill` (+ `--fundamentals-pause`,
  `--fundamental-data-url`) przechodzi po uniwersum, raportuje liczbę okresów per symbol i grupuje
  awarie po przyczynie. Ograniczenie warte zapamiętania: EDGAR czytamy **tylko rocznie (10-K)**,
  więc 20 lat to ~20 obserwacji na spółkę — rodzina rankuje przekrój wolno i z definicji nie może
  nic znaczyć przy horyzoncie 10 sesji, a może przy 63. fundamental-data 54 testy (+7); ruff +
  format + mypy czyste.

- 2026-08-02 — **Powrót do specyfikacji z badań: audyt luk + rodzina fundamentalna z §5.**
  Użytkownik słusznie zauważył, że projekt zawęził się do diagnostyki modelu na najwęższym możliwym
  wejściu, podczas gdy duża część udokumentowanej specyfikacji jest niezbudowana. **Audyt luk
  względem własnych dokumentów** (zweryfikowany kodem, nie pamięcią): (a) `Plan_Rozwoju` Faza 1
  wymienia z nazwy ~20 rodzin wskaźników i checklistę „30+" — mamy z nich SMA 10/20/50, RSI i ROC;
  brak EMA, MACD, ADX, Aroon, Stochastic, CCI, Williams %R, Bollinger, ATR, Keltner, OBV, VWAP,
  A/D, MFI, formacji świecowych (TA-Lib nie jest nawet zależnością); (b) Faza 3 mówi „100+ features"
  — model dostaje 15; (c) `SentimentSnapshot` + `SENTIMENT_UPDATED` istnieją w kontraktach, ale
  **żaden serwis ich nie produkuje**; (d) Faza 2 wymienia 5 strategii + registry — jest jedna
  (`momentum.py`), registry nie ma; (e) macro-data nie ma magazynu historii, więc `regime_by_date`
  nie ma czym wypełnić i 5 kolumn `macro_*` wypada jako stałe w każdym treningu (P2-4 nietknięte);
  (f) dashboard ma ~1,5 z 6 sekcji ze spec i **zero wykresów**; (g) Faza 3 wymaga raportu feature
  importance — nie ma go.
  **Zrobione (wybór użytkownika): rodzina fundamentalna z §5 planu predykcji.** Contracts-first:
  `FinancialStatements` += `gross_profit`, `cost_of_revenue` (filerzy raportują jedno ALBO drugie,
  a pokrycie decyduje, czy to w ogóle jest czynnik) + kolumny w ORM, `init-db.sql` i **idempotentny
  `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`** w starcie serwisu — dokładnie ta sama pułapka, którą
  zastawił `adj_close`: `create_all` tworzy brakujące TABELE, nigdy brakujących KOLUMN. EDGAR:
  nowe tagi `GrossProfit` + fallbacki `CostOfRevenue`/`CostOfGoodsAndServicesSold`/…
  `fundamental_features` przyjmuje teraz `prior` i `price` i liczy **pięć nowych cech**:
  `fund_gross_profitability` (Novy-Marx 2013), `fund_accruals` (Sloan 1996, **ze znakiem** — anomalia
  polega na tym, że WYSOKIE accruals zapowiadają NISKIE zwroty), `fund_asset_growth`
  (Cooper–Gulen–Schill 2008), `fund_book_to_market` i `fund_earnings_yield` (Fama–French).
  Dwie decyzje warte zapamiętania: **cena musi być SUROWA** — liczba akcji pochodzi ze sprawozdania,
  więc pomnożenie jej przez cenę skorygowaną wstecz liczy kapitalizację, która nigdy nie istniała
  (późniejszy split 2:1 zmniejsza ją o połowę); oraz **poprzednie sprawozdanie musi przejść TEN SAM
  odcięcie czasowe** (`prior_available_before`) — wybór po samym okresie fiskalnym sięgnąłby po
  filing jeszcze nieopublikowany, gdy restatement albo spóźniony filer odwracają kolejność
  publikacji względem okresu. Bez `prior`/`price` funkcja zwraca dokładnie dawne cztery cechy, więc
  ścieżka serwowania działa bez zmian. **Adopcja do modelu czeka na pomiar** (reguła etapu E2:
  rodzina wchodzi, gdy tabela IC to potwierdzi) — a włączenie w treningu wymaga włączenia
  w serwowaniu w tym samym kroku. Liczniki: shared 244 (+8), ml-pipeline 315 (+3),
  fundamental-data 54 → **bateria ~1250**; ruff + format + mypy (`--strict` na shared) czyste.
  **Zweryfikowane na prawdziwym PostgreSQL-u (12/12)**: tabela utworzona BEZ nowych kolumn dostaje
  je migracją ze startu (dwukrotne uruchomienie bez błędu), stary wiersz przeżywa, obie kolumny
  robią round-trip przez realny store, pięć czynników liczy się as-of z dokładnością do 1e-9,
  a sesja sprzed drugiego filingu **nie wymyśla** wzrostu aktywów.

- 2026-08-02 — **Faza 2: pięć strategii, registry i trzy defekty, które przy JEDNEJ regule były
  nieodróżnialne od poprawnego działania.** Cała warstwa decyzyjna jest zbudowana dla WIELU źródeł
  regułowych; przy jednym mechanizmy nie były „brakujące", tylko **bezczynne**, i dlatego przetrwały.
  **(1) Registry mieszka w `trading-common`, nie w serwisie** — bo backtest musi oceniać tę samą
  regułę, którą handluje serwis, a serwisy nie mogą się importować. Protokół deklaruje
  `required_features` (wektor własny) **osobno od** `required_ranks` (percentyle przekrojowe), i ten
  podział nie jest kosmetyczny: decyduje, gdzie regułę **da się w ogóle ocenić**. Rejestracja odmawia
  reguły czytającej cechę, której nikt nie liczy — zamiast `KeyError` na pierwszym symbolu pierwszej
  sesji. **(2) Bufor agregatora kluczowany PARĄ (symbol, strategia)**: dotąd druga strategia dla
  AAPL **nadpisywała** pierwszą, a zwycięzcą była ta dostarczona ostatnia; wszystkie sygnały wchodziły
  jako jedno źródło `"strategy"`, więc `AdaptiveWeightOptimizer` — generyczny po nazwach źródeł — nie
  miał czego rozróżniać. Teraz źródło to `strategy:{nazwa}` brane z registry, wygasanie działa **per
  wpis**, a lista komponentów jest sortowana po nazwie (`components_present` nie może zależeć od
  kolejności dostarczenia). **Poziomy wybierane są PO głosowaniu**, spośród wpisów zgodnych
  z końcowym kierunkiem, najwyższą pewnością, remis po nazwie strategii — bez determinizmu ten sam
  zestaw wejść dawałby różne zlecenia. **(3) Backtest w ogóle nie czytał `strategy_name`**: odpalał
  zaszytą regułę momentum na cenach i stemplował wynik przekazaną nazwą, więc „rewalidacja
  momentum_rank" oceniała kod, który nigdy nie działał na produkcji — a dowolna nazwa dawała te same
  liczby. Silnik ocenia teraz regułę z registry na tym samym oknie cech co serwowanie
  (`FEATURE_LOOKBACK`), a **`momentum_rank` dostaje jawny błąd 422** („wymaga backtestu uniwersum")
  zamiast proxy na cenie — to było dokładnie dotychczasowe zachowanie, tylko nienazwane. Przy okazji
  wyszło, że `run_backtest` liczył na cenach **surowych**, a `revalidate` na **skorygowanych**: dwie
  ścieżki tego samego serwisu mierzyły różne aktywa. Domyślna `REVALIDATION_STRATEGY` zmieniona
  z `momentum_rank` na `sma_ema_crossover`, bo tamta z definicji nie przejdzie.
  **Reguły:** `sma_ema_crossover`, `rsi_bollinger_reversion` (zamyka **D5** — to przeciwny zakład do
  momentum, więc osobna reguła, nie doklejka), `macd_confirmation`, `donchian_breakout`. **Nazwa
  `macd_divergence` z planu została świadomie zmieniona**: dywergencja porównuje swingi ceny
  i oscylatora w CZASIE, a reguła widzi JEDEN wektor cech bez historii — wysłanie reguły potwierdzenia
  pod nazwą dywergencji unieważniłoby każdy późniejszy raport o niej. **Pair trading odłożony** (druga
  seria — ta sama blokada co `beta_60`). **Stop skalowany zmiennością (P5-3 dla ścieżki regułowej)**:
  reguła zwraca szerokość w **wielokrotnościach ATR** (rewersja 1.5, trend 2.0, wybicie 2.5), bo wie,
  ile miejsca potrzebuje jej własny pomysł, a nie zna ceny wykonania; przeliczenie idzie przez
  `atr_pct_14` — **bezwymiarowy**, bo ATR liczy się na skali skorygowanej, a zlecenie na surowej.
  **Wskaźniki** (EMA 12/26, MACD+histogram, Bollinger+%B+szerokość, Donchian(20), ATR(14)) trafiają do
  `RULE_ONLY_FEATURES` w `trading-common`, a ml-pipeline **importuje** ten zbiór do
  `EXCLUDED_FEATURES` — wskaźnik dodany po jednej stronie nie może wejść do kontraktu cech modelu
  przez zapomnienie po drugiej. Donchian liczony jest z **POPRZEDNICH** 20 barów: kanał zawierający
  dzisiejszy bar nie może zostać przebity, więc reguła byłaby cicha na zawsze, nigdy nie zgłaszając
  błędu. Liczniki: shared 289 (+45), strategy 60 (+4), signal-aggregator 97 (+11), backtest 56 (+15)
  → **bateria 1311**; ruff + format + mypy (`--strict` na shared) czyste, `check-dependencies` OK.
  **Kontrola anty-szczęściowa**: po przywróceniu dwóch defektów agregatora (klucz = symbol, źródło =
  `"strategy"`) **5 z 11** nowych testów pada — dwa z nich przechodziły w pierwszej wersji i zostały
  wzmocnione (zwycięzca poziomów przychodzi teraz PIERWSZY, bo przy nadpisywaniu ostatni i tak wygrywał).
  **Zweryfikowane na żywo (15/15)** na realnym `nats-server` + dwóch realnych serwisach na uvicornie
  (podmieniony wyłącznie feature-engine, prawdziwym serwerem HTTP — egress jest zablokowany): jedno
  `features.ready` → **4 zdarzenia `signal.generated`** (BUY/BUY/BUY/SELL — reguły faktycznie się nie
  zgadzają), **jedna** decyzja z `components_present` = 4 nazwane źródła i **zero** `"strategy"`,
  wagi adaptacyjne rozjeżdżają się **0.667 vs 0.056** ze wspólnego startu 0.143, rewalidacja jednej
  strategii zmienia status **tylko** jej, a wyłączona reguła przestaje emitować, gdy reszta emituje dalej.

- 2026-08-02 — **Dashboard: 6 sekcji ze spec — i szereg czasowy, którego NIKT nie trzymał.**
  Sekcje 1, 2 i 4 planu (krzywa kapitału, VaR, wykresy backtestu) padały wszystkie na tym samym:
  broker przeliczał equity przy każdym fillu i marku **i wyrzucał je**, risk-mgmt trzymał snapshot,
  a backtest liczył tablicę `equity` w `score_positions` i jej nie zwracał. Można było narysować
  te wykresy tylko fałszywie, więc najpierw powstały dane. **Punkt na SESJĘ, nie na mutację**:
  zapisywanie każdej zmiany uzależniłoby okno historii od tego, ile symboli akurat tego dnia
  odświeżono (pracowity dzień wypchnąłby spokojny tydzień z „ośmiu lat"), a system i tak handluje
  na barach dziennych; punkt dnia jest nadpisywany, więc niesie najnowszą wartość. Utrwalony
  w snapshotcie, przywracany, tolerancyjny na snapshot sprzed zmiany układu.
  **`trading_common.risk_metrics`** — VaR **historyczny, nie parametryczny** (rozkład normalny na
  zwrotach dziennych zaniża dokładnie ten ogon, który liczba ma opisywać, a mamy realną ścieżkę),
  CVaR, seria obsunięcia, korelacje. Każda funkcja **odmawia przy zbyt małej próbie** — VaR z 12
  obserwacji to nie konserwatywny VaR, tylko liczba bez rozkładu z próby, a wykres narysowałby ją
  bez mrugnięcia. **Defekt złapany własnym testem**: `(1.0 - 0.95) * 100` to w binarnym floacie
  `5.000000000000004`, więc `ceil` dawał 6 i kwantyl lądował **za** ogonem — na kanonicznej wartości
  95%, akurat. Efekt: VaR 0.0 na serii, która straciła w pięciu dniach. Test przypina ten przypadek.
  **Trzy defekty znalezione dopiero na biegu na żywo**, nie w testach: (1) `resp.json()` na
  nie-JSON-owym ciele rzuca `JSONDecodeError`, który **nie jest** `httpx.HTTPError`, więc uciekał
  z obsługi i zamieniał 500 upstreamu w 500 dashboardu — ta sama klasa, co dawne `HTTP 500: {}`;
  (2) backtest przy nieosiągalnej market-dacie oddawał goły tekstowy 500 (mapowanie `httpx` dodane,
  502 z nazwą upstreamu); (3) reguła przekrojowa była odrzucana **po** pobraniu danych, więc przy
  padniętej market-dacie zwracała zły błąd — `ensure_single_symbol_evaluable` idzie teraz przed
  fetchem. Do tego pusta siatka korelacji przy niedostępnej market-dacie wyglądała identycznie jak
  brak pozycji: sekcja raportuje teraz `held_symbols` obok `correlated_symbols`.
  **UI**: wykresy to **inline SVG** liczone w kilkunastu liniach vanilla JS — kontener nie ma
  dostępu do CDN-u ani bundlera w toolchainie, więc biblioteka oznaczałaby ręczne wendorowanie
  i pinowanie wersji dla czterech typów wykresu. Odświeżana jest **tylko widoczna sekcja** (sonda
  zdrowia i siatka korelacji kosztują realną pracę u sąsiadów). Backtest jest **na żądanie**, a jego
  status przechodzi na wylot: 404 i 422 to odpowiedzi, nie awarie. Feature importance świadomie
  **nie ma wykresu** — pusty wykres sugerowałby, że model nie ma ważnych cech, a nie że nikt tego
  nie zmierzył. Liczniki: shared 308 (+19), execution 60 (+12), backtest 58 (+2), dashboard 34 (+16)
  → **bateria 1352**; ruff + format + mypy czyste, `check-dependencies` OK, compose i Helm
  zsynchronizowane (render + `helm lint` sprawdzone). **Zweryfikowane na żywo (19/19)** na pięciu
  realnych serwisach na uvicornie: transakcje przez prawdziwy `POST /execute` → krzywa w sekcji 1,
  VaR **odmawia** przy jednej sesji zamiast zmyślać, wszystkie 5 reguł z wagami w sekcji 3,
  `momentum_rank` odrzucony 422 zamiast fałszywego wykresu, 12 serwisów odpytanych z pomiarem
  opóźnienia. Przy okazji harness: `ss -lntp` nie pokazuje PID-ów bez uprawnień, więc skrypt
  zatrzymujący po porcie **po cichu nic nie ubijał** i stare procesy trzymały porty — zabijanie idzie
  teraz po linii poleceń.

- 2026-08-02 — **P2-4: historia makro — i dlaczego zwykły backfill byłby gorszy niż jego brak.**
  `build_dataset` od zawsze przyjmował `regime_by_date` i **nikt nigdy tego parametru nie
  przekazał**, więc 5 kolumn `macro_*` było w każdym treningu stałym zerem i wypadało przez filtr
  wariancji. Rodzina istniała z nazwy. **Sednem nie było jednak dodanie tabeli, tylko VINTAGE:**
  FRED rewiduje szeregi wstecz, więc zapytanie dziś o marzec 2015 zwraca wartość po rewizjach —
  liczbę, której wtedy nikt nie mógł znać. Backfill „ostatnich wartości" wyglądałby kompletnie
  i byłby błędny dokładnie tam, gdzie najtrudniej to zauważyć: wiarygodne liczby, żadna niedostępna
  w swoim czasie. ALFRED (`realtime_start`/`realtime_end`) zwraca każdą obserwację razem z oknem,
  w którym BYŁA opublikowaną wartością. Panel kluczowany trójką `(series, observation_date,
  realtime_start)` — szereg makro ma **dwie osie czasu** i obie są nośne. **Odczyt as-of jest
  dwuwymiarowy i pomylenie osi to dwa różne błędy**: bez `realtime_start` model dostaje rewizje,
  które jeszcze nie istniały; bez `observation_date` dostaje to, co ostatnio zrewidowano, zamiast
  najnowszego okresu. Żaden się nie wywala. **Opóźnienie publikacji wychodzi za darmo** — marzec jest
  publikowany w kwietniu, a `realtime_start` to koduje, więc nie ma osobnego parametru „lag".
  **Wiersz bez vintage jest NIEWIDOCZNY dla historii**, nie „stary" (ta sama reguła co `filed_at`),
  a wartownik siedzi w dalekiej **przyszłości**: zakodowanie „nieznane" jako stara data zrobiłoby
  odwrotność i uczyniłoby każdy niedatowany wiersz najwcześniejszą rzeczą, jaką wiedzieliśmy.
  **Reżim jest wyprowadzany przy odczycie, nie zapisywany** — utrwalenie etykiety zamroziłoby jedną
  wersję `classify_regime` w danych. **Dzień, którego nie da się sklasyfikować, jest NIEOBECNY**,
  nie wypełniony domyślnym „expansion". **Złapane po drodze:** (1) `regime_by_date` było kluczowane
  `datetime`, a sesje to znaczniki ze strefą — dopasowanie po dokładnym instancie **nigdy by nie
  trafiło**, i to po cichu; działało wyłącznie w testach, bo tylko one konstruowały te same
  instanty. Klucz to teraz `date`. (2) Pierwsza wersja `regime_history` robiła zapytanie na dzień —
  7300 round-tripów na 20 lat; jedno przejście po panelu. (3) Brakujący import `text` w starcie
  macro-daty wywaliłby serwis na Postgresie (ruff, nie test). Liczniki: macro-data 54 (+13),
  ml-pipeline 321 (+6), shared 308 → **bateria 1378**; ruff + format + mypy czyste,
  `check-dependencies` OK, compose i Helm zsynchronizowane (`needsDb`, render sprawdzony).
  **Kontrola anty-szczęściowa**: po usunięciu samego filtru `realtime_start <= day` pada **4 z 13**
  testów vintage — dokładnie te o wycieku rewizji. **Zweryfikowane na prawdziwym PostgreSQL-u
  (13/13)** ze schematem z `init-db.sql`, nie z echa ORM-a: klucz główny to faktycznie trójka
  z vintage, rewizja z 2016 nie wycieka do zapytania o 2015, powtórzony wiersz w jednej partii nie
  wywraca zapisu (`ON CONFLICT` nie znosi dwóch tych samych kluczy w jednym VALUES), a lipcowe
  pogorszenie zmienia reżim `slowdown → crisis` dopiero od lipca.
  **Do zrobienia u użytkownika:** `POST /api/v1/macro-data/backfill` — egress do FRED jest
  w piaskownicy zablokowany, więc panel jest pusty do czasu backfillu, a `GET /coverage` mówi wprost,
  ile wierszy jest i ile z nich jest niedatowanych.

- 2026-08-02 — **Wskaźniki z checklisty Fazy 1 — i defekt, przez który kandydat NIGDY nie mógł
  zapracować na wejście.** Reguła etapu E2 brzmi „rodzina wchodzi do modelu, gdy tabela IC to
  potwierdzi". Okazała się **niewykonalna**: `alpha_decay` woła `build_dataset`, ten stosuje
  `EXCLUDED_FEATURES`, a nowe wskaźniki są właśnie tam — czyli **jedyna ścieżka, która mogła zmierzyć
  kandydata, sama go ukrywała**. Rozbite na dwa zbiory, bo to dwa różne powody:
  `INADMISSIBLE_FEATURES` (poziomy cenowe, duplikat `momentum_20`) — **żaden pomiar tego nie zmieni**,
  ranga poziomu to proxy na cenę akcji; oraz `CANDIDATE_FEATURES` — policzone, jeszcze nieprzyjęte,
  wpuszczane przez `build_dataset(include_candidates=True)` i `POST /models/alpha-decay`
  z `include_candidates`. **Trening nigdy nie ustawia tej flagi**, więc rodzina nie wejdzie do
  kontraktu mimochodem, a pomiar nic nie kosztuje w rozliczeniu wielokrotnego testowania (studium
  jest model-free). **Rodziny:** Stochastic %K/%D, CCI(20), ADX(14)+±DI, Aroon(25), MFI(14),
  OBV i A/D **jako nachylenie** (skumulowana suma rankuje to, jak długo spółka jest notowana, a nie
  sygnał — normalizowane średnim wolumenem), VWAP **jako stosunek**, Keltner **jako pozycja**.
  **Świadomie pominięte:** Williams %R (to `%K - 100`, transformacja liniowa → **identyczna ranga
  przekrojowa**; dodanie odtworzyłoby duplikat, który T0-7 usunął) i formacje świecowe (TA-Lib nie
  jest zależnością, kilkadziesiąt formacji to duża powierzchnia przy znikomym dowodzie przekrojowym).
  **Dwa realne błędy złapane testami, nie przeglądem:** MFI potrzebuje **21** barów, nie 20 (każdy
  z 20 przepływów jest klasyfikowany ruchem względem POPRZEDNIEJ ceny typowej), a w pętli %D indeks
  `end - 14` schodził poniżej zera — w Pythonie to wycinek liczony od końca, więc „za mało historii"
  zamieniało się w **pusty wycinek i `ValueError` trzy funkcje dalej**; wywróciło 17 testów backtestu,
  bo to on liczy cechy na krótkich oknach. Oba przypięte, plus test przechodzący każdą długość serii
  od 1 do 39. Liczniki: shared 325 (+17), ml-pipeline 325 (+4) → **bateria 1400**; ruff + format +
  mypy (`--strict` na shared) czyste, `check-dependencies` OK. **Kontrola anty-szczęściowa**: po
  zlaniu obu zbiorów z powrotem w jeden pada test „kandydaci są wpuszczani do pomiaru".

- 2026-08-03 — **Raport feature importance (Faza 3): co model UŻYWA, a nie co koreluje — i rodzina,
  bez której wniosek byłby odwrotny.** `per_feature_ic` mierzy dowód **marginalny** (czy kolumna
  przewiduje sama z siebie) i z zasady nie odpowiada na pytanie, które podejmuje decyzje o kontrakcie
  cech: czy model **potrzebuje** tej kolumny, mając pozostałe czternaście. `core/importance.py`
  mierzy to drugie, permutacyjnie, i cztery decyzje robią różnicę między liczbą a jej obrazkiem.
  **(1) Poza próbą, na holdoucie** — permutacja w oknie, na którym model był fitowany, raportuje to,
  co zapamiętał. **(2) Permutacja WEWNĄTRZ sesji** — ta sama lekcja co przy sondzie pojemności,
  w innym pytaniu: globalne tasowanie przenosi wartości MIĘDZY datami, więc niszczy nie tylko
  sparowanie nazwy z wartością, ale i rozkład brzegowy cechy w sesji, a miarą jest przekrojowe IC
  per sesja. **Zmierzone**: cecha stała w obrębie sesji (kształt KAŻDEGO one-hota `macro_*`) dostaje
  wewnątrzsesyjnie dokładnie `ΔIC 0.00000, t 0.00`, a globalnie **`ΔIC +0.236, t +3.55`** — połowa
  ważności prawdziwego sygnału, wymyślona w całości. **(3) Spadek IC PAROWANY sesja po sesji** —
  nieparowane porównanie ginie w zmienności samego IC; parowanie ją usuwa, bo oba człony widziały ten
  sam dzień. Próg to skorygowana Šidákiem wartość dla liczby testowanych cech, żeby największe
  z piętnastu losowań szumu nie zostało odczytane jako ulubione wejście modelu. **(4) Także dla
  RODZIN** — permutacja dzieli zasługę między skorelowane kolumny, więc dwa bliźniaki wyglądają
  każdy na nieważny; rodzina to zresztą jednostka, w której ten projekt podejmuje decyzje.
  **Bieg na żywo pokazał dokładnie ten mechanizm i bez rodzin dałby odwrotny wniosek**: na 20 nazwach
  × 1047 sesji **żadna pojedyncza cecha nie przeszła progu** (najwyżej `amihud_20`, t +1.63),
  a rodzina **`liquidity` przeszła z t +4.18 i ΔIC +0.094** przy bazowym IC modelu 0.124 — trzy
  czwarte całej mocy rankującej; `amihud_20` i `dollar_volume_20` są oznaczone jako swoje bliźniaki.
  **Podłoga jest MIERZONA**: `POST /models/feature-importance` fituje własny model z **posadzoną
  kolumną czystego szumu** (prawidłowa ranga przekrojowa, zero informacji z konstrukcji) i mówi, ile
  ta kolumna zdobyła — na żywo `t = +0.16`. Studium jest **wyłącznie diagnostyczne**: ma kolumnę,
  której serwowanie nie umie wytworzyć, więc nic z niego nie jest rejestrowane; bieg treningowy mierzy
  model PRODUKCYJNY i posadzonej kolumny nie ma. `include_candidates` daje warunkową połowę etapu E2:
  alpha-decay mówi, czy rodzina przewiduje sama, to — czy model by jej UŻYŁ. **Defekt złapany testem,
  nie przeglądem**: model ignorujący kolumnę DOKŁADNIE (drzewo, które nigdy na niej nie dzieli) daje
  identyczne predykcje, a powtórzenia są sumowane i dzielone — co w binarnym floacie nie jest
  identycznością; spadek `1.7e-17` przy błędzie `5.2e-18` dał kolumnie nieczytanej przez model
  **t = +3.23** i ogłoszenie jej jako istotnej. **Przy okazji, w dashboardzie**: `GET /runs` zwraca
  LISTĘ `{operation, completed_at}`, a sekcja ML czytała ją jak mapę, więc w kolumnie „Operation"
  stały pozycje tablicy (`0`, `1`); fixture testowy powielał ten sam błędny kształt. Sekcja ML rysuje
  teraz tabelę ważności (słupki ze ZNAKIEM — permutacja cechy, na której model opiera się odwrotnie,
  IC **poprawia**, a wykres modułów by to ukrył), nazywa **źródło** tabeli (bieg treningowy = model
  serwowany vs studium = model diagnostyczny z posadzoną kolumną) i rozróżnia trzy rodzaje braku:
  „nikt nie zmierzył", „pomiar wyłączony", „ml-pipeline nie odpowiada". Liczniki: ml-pipeline 340
  (+15), dashboard 37 (+3) → **bateria 1418**; ruff + format + mypy czyste, `check-dependencies` OK.
  **Kontrola anty-szczęściowa**: po zamianie permutacji na globalną i usunięciu progu na pył padają
  dokładnie 2 nowe testy — ten o cesze stałej w sesji i ten o kolumnie kontrolnej.
  **Zweryfikowane na żywo (15/15)** na realnej market-dacie (własny lifespan, sqlite, 26 000 świec
  dla 20 symboli) + realnym ml-pipeline i realnym dashboardzie na uvicornie: studium 200 w 33 s,
  trening 200 z wypełnionym `gate.importance`, wersja 1 w MLflow, indeks `/runs` z obiema operacjami,
  a `measured_at` w dashboardzie zgadza się co do znacznika z `completed_at` biegu treningowego.

**Next (2026-08-03): tor predykcji zablokowany na pomiarze — pracujemy poza nim.**

Etapy **strategii + registry**, **dashboardu**, **historii makro (P2-4)**, **wskaźników** i **raportu
feature importance** są zamknięte (wpisy z 2026-08-02 i 2026-08-03 na końcu logu). Kod toru predykcji
(E0–E5) jest skończony i **nie ma tam sensownego następnego zadania programistycznego**: każda pozostała decyzja jest bramkowana liczbami, których nie mamy (t-stat
IC ≥ 2). Trening #3 na 414 symbolach × 20 lat skończył się zapadnięciem modelu do stałej i
odrzuceniem przez bramkę na 5 z 6 warunków, a sonda pojemności wymaga powtórzenia po naprawie jej
kontroli (permutacja wewnątrz sesji). To czeka na bieg u użytkownika.

**Audyt luk 2026-08-02 pokazał, że projekt zawęził się do jednego wątku**, podczas gdy duża część
udokumentowanej specyfikacji jest niezbudowana. Otwarte fronty, w kolejności do podjęcia:

| Front | Stan | Źródło wymagania |
|---|---|---|
| ~~Strategie + registry~~ | ✅ **zamknięte 2026-08-02** — 5 reguł w registry, agregator kluczowany parą, backtest ocenia regułę z registry | `Plan_Rozwoju` Faza 2 |
| ~~Dashboard / frontend~~ | ✅ **zamknięte 2026-08-02** — 6 sekcji, wykresy SVG, historia kapitału w execution | `Plan_Rozwoju` Faza 4, Tydzień 21 |
| ~~Historia makro (P2-4)~~ | ✅ **zamknięte 2026-08-02** — panel vintage z ALFRED, odczyt as-of po dwóch osiach, `regime_by_date` wreszcie ma źródło. **Backfill u użytkownika** (egress do FRED zablokowany w piaskownicy) | plan predykcji §13 (T2-2) |
| ~~Wskaźniki techniczne~~ | ✅ **zamknięte 2026-08-02** — ~17 z ~20 rodzin, a co ważniejsze: kandydat jest **mierzalny bez bycia zaadoptowanym** (`include_candidates`) | `Plan_Rozwoju` Faza 1, Tydzień 3 |
| ~~Raport feature importance~~ | ✅ **zamknięte 2026-08-03** — permutacja wewnątrz sesji na holdoucie, spadek IC parowany per sesja, próg Šidáka, wiersze per RODZINA i mierzona podłoga szumu; sekcja ML dashboardu rysuje tabelę i nazywa jej źródło | `Plan_Rozwoju` Faza 3, checklist |
| **Sentyment / FinBERT** ← **BIEŻĄCY ETAP** | kontrakt i event istnieją, **producenta nie ma** | `Plan_Rozwoju` Faza 3, Tydzień 17–18 |

Aneks statusu z mapowaniem checklist Faz 1–5 na kod: `Plan_Rozwoju_Systemu_Tradingowego_2.md`,
sekcja „Aneks: status realizacji". Mapa dokumentów i zasada „który plik wygrywa": `docs/README.md`.
Proweniencja decyzji: `docs/decisions/`.

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

Claude pracuje w piaskownicy z Pythonem 3.12, dostępem do sieci przez proxy i powłoką. Ograniczenia,
które realnie kształtują pracę (reszta jest w `[env]` w „Known issues"):

- **brak dostępu do działającego systemu użytkownika** — każdy realny backfill, trening i bieg
  diagnostyczny wykonuje użytkownik u siebie, a Claude dostaje raport JSON z `reports/`;
- **egress do dostawców danych rynkowych jest zablokowany** (Yahoo, Stooq, SEC, Docker Hub blob CDN),
  więc weryfikacja „na żywo" oznacza tu: realny `nats-server` i realny PostgreSQL uruchomione lokalnie,
  realne serwisy na uvicornie, a podmieniony wyłącznie fetcher (wzorzec `smoke2`/`md_runner`);
- **żadnych zleceń do brokera** — cała ścieżka egzekucji jest papierowa.

Bootstrap danych i kampania pomiarowa mają swoje narzędzie: `scripts/bootstrap-universe.py`
(+ `scripts/diagnose.py` do diagnostyki stacku). Opis użycia jest w `README.md`.

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

### Instrukcje dla człowieka podlegają tej samej weryfikacji co kod

Komenda albo nazwa podana użytkownikowi jest **twierdzeniem o repozytorium**, nie prozą. Kod bywał
przetestowany, a instrukcja obok niego pisana z pamięci — i to ona zawodziła. Zanim wyślesz:

- [ ] Każdy literał sprawdzony w źródle, nie z pamięci: nazwy wolumenów i projektu Compose
      (`name:` w `infrastructure/docker-compose.yml` — projekt to **`trading-system`**, NIE nazwa
      katalogu), serwisów, ścieżek, zmiennych środowiskowych
- [ ] Każda flaga potwierdzona przez `--help`, nie z pamięci
- [ ] Runbook przejrzany krok po kroku względem `Makefile` / `make.ps1` — czy nic nie wypadło
      (np. `build` przed `up`)
- [ ] Składnia powłoki dobrana do platformy użytkownika. Windows PowerShell 5.1 **nie zna** `||`
      ani `&&`, a `2>/dev/null` to `2>$null`; `make.ps1` przyjmuje flagi wprost, bez `ARGS="…"`
- [ ] Najgroźniejsza klasa błędu: instrukcja, która **nie wywala się głośno, tylko po cichu nic nie
      robi** (zły `docker volume rm` → „no such volume" → pułapka wygląda na rozbrojoną). Przy
      komendach czyszczących pokaż, jak sprawdzić, że zadziałały
