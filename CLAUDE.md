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

**Verified ground truth** (test counts measured 2026-08-02 on Python 3.12, not from memory —
**1236 testów zielonych**; `ruff` + `ruff format` + `mypy` czyste, `--strict` na shared):

| Komponent | Port | Rola | Testy |
|---|---|---|---|
| `shared/trading-common` | — | Kontrakty i **wszystko, co musi być identyczne po obu stronach granicy serwisów** | 244 |
| `market-data` | 8001 | OHLCV: pobranie (Yahoo/Alpha Vantage), walidacja, TimescaleDB, cache, harmonogram przyrostowy | 71 |
| `feature-engine` | 8002 | Wskaźniki Tier-1 + wzbogacenie Tier-2, rangi przekrojowe (`/ranked`) | 38 |
| `strategy` | 8003 | Reguła momentum-on-ranks → `RiskEnvelope` → `CostAwareFilter` → sygnał; monitor degradacji | 56 |
| `backtest` | 8004 | Silnik wektorowy long/flat + walk-forward, tygodniowa rewalidacja | 41 |
| `ml-pipeline` | 8005 | Zbiór, trening, bramka G0–G5, rejestr MLflow, serwowanie, monitoring driftu, badania | 315 |
| `risk-mgmt` | 8006 | Sizing adaptacyjny, limity reżimowe i sektorowe, wyłącznik z zatrzaskiem, rejestr zleceń | 133 |
| `execution` | 8007 | Paper broker, wyjścia ochronne SL/TP, likwidacja na BLACK, feedback portfela | 48 |
| `notification` | 8008 | 5 strumieni → alerty (log/Slack/Telegram/e-mail) | 33 |
| `fundamental-data` | 8009 | SEC EDGAR, Piotroski 9/9, **panel point-in-time** (`filed_at`) | 54 |
| `macro-data` | 8010 | FRED + detekcja reżimu → `macro.regime_changed` | 41 |
| `company-classifier` | 8011 | Profil → styl inwestycyjny + routing stosu modeli | 25 |
| `signal-aggregator` | 8012 | **Węzeł decyzyjny**: strategia + ML + makro → jedna decyzja z poziomami i sektorem | 86 |
| `dashboard` | 8501 | BFF nad HTTP pozostałych serwisów + prosta strona | 18 |
| `scripts/` | — | Bootstrap uniwersum, diagnostyka stacku, audyt zależności | 33 |

Co z tego jest **wiążące**, a nie tylko opisowe:

- **`trading-common` jest granicą.** Leży w nim wszystko, co musi dać ten sam wynik w treningu i na
  produkcji: `features` (+ `FEATURE_LOOKBACK=300` / `FULL_HISTORY=253` jako jedna stała okna),
  `ranking`, `fundamentals` (reguła as-of + wyprowadzenie czynników), `sectors`, `prices`,
  `RiskEnvelope`, `CostAwareFilter`, `sizing`, `scheduler`, `timeutil`, `constants.MAX_OHLCV_LIMIT`.
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
  - **P2-4** (z zarchiwizowanego planu predykcji) `macro-data` **nie ma żadnej warstwy trwałości** —
    tylko bieżący snapshot w pamięci. Dlatego `regime_by_date` nie ma czym wypełnić i 5 kolumn
    `macro_*` wypada jako stałe z KAŻDEGO treningu. Wymaga też danych **vintage** (ALFRED): FRED
    rewiduje szeregi wstecz, więc makro jako cecha bez vintage to look-ahead na rewizjach.
    Odblokowuje decyzję D8 (reżim jako cecha czy jako warunkowanie).
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

**Next (2026-08-02): tor predykcji zablokowany na pomiarze — pracujemy poza nim.**

Kod toru predykcji (E0–E5) jest skończony i **nie ma tam sensownego następnego zadania
programistycznego**: każda pozostała decyzja jest bramkowana liczbami, których nie mamy (t-stat
IC ≥ 2). Trening #3 na 414 symbolach × 20 lat skończył się zapadnięciem modelu do stałej i
odrzuceniem przez bramkę na 5 z 6 warunków, a sonda pojemności wymaga powtórzenia po naprawie jej
kontroli (permutacja wewnątrz sesji). To czeka na bieg u użytkownika.

**Audyt luk 2026-08-02 pokazał, że projekt zawęził się do jednego wątku**, podczas gdy duża część
udokumentowanej specyfikacji jest niezbudowana. Otwarte fronty, w kolejności do podjęcia:

| Front | Stan | Źródło wymagania |
|---|---|---|
| **Strategie + registry** ← **BIEŻĄCY ETAP** | 1 reguła zamiast 5, brak registry; agregator kluczuje bufor samym symbolem, więc druga strategia nadpisuje pierwszą | `Plan_Rozwoju` Faza 2 |
| Dashboard / frontend | ~1,5 z 6 sekcji, zero wykresów | `Plan_Rozwoju` Faza 4, Tydzień 21 |
| Historia makro (P2-4) | macro-data nie ma warstwy trwałości → 5 kolumn `macro_*` wypada z każdego treningu | plan predykcji §13 (T2-2) |
| Wskaźniki techniczne | ~5 z ~20 rodzin z checklisty „30+" | `Plan_Rozwoju` Faza 1, Tydzień 3 |
| Sentyment / FinBERT | kontrakt i event istnieją, **producenta nie ma** | `Plan_Rozwoju` Faza 3, Tydzień 17–18 |
| Raport feature importance | brak | `Plan_Rozwoju` Faza 3, checklist |

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
