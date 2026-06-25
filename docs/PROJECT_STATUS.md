# Project Status & Context — Trading System

> Żywy dokument kontekstu. Czytaj go na początku każdej sesji, by wiedzieć **na jakim etapie
> jest projekt** i **jakie błędy/luki są znane**. Sekcja „Dziennik postępu" jest *append-only* —
> dopisuj nowy wpis na końcu, nie nadpisuj poprzednich.
>
> Generowany ręcznie podczas audytów repo. Ostatni pełny audyt: **2026-06-25**.

---

## 1. TL;DR — stan rzeczywisty

**Deklarowany etap (CLAUDE.md):** Faza 1, Tydzień 2, fokus: pełna implementacja `market-data-svc`.

**Etap rzeczywisty:** Występuje **inwersja priorytetów**:
- Fundament (`market-data`) jest nadal **szkieletem** — endpointy zwracają `501 Not Implemented`
  (`services/market-data/src/api/routes.py:22`).
- Jednocześnie zaimplementowano **12 zaawansowanych komponentów** z `docs/framework_supplement.md`
  (harmonogram Tydzień 19+), z bogatym pokryciem testami (~335 funkcji testowych w 23 plikach).
- **Problem:** te komponenty są **osierocone** — żaden nie jest wpięty do aplikacji FastAPI ani do
  NATS. `main.py`/`routes.py` każdego serwisu importują wyłącznie `observability`. Logika istnieje
  i jest przetestowana jednostkowo, ale **nie działa w runtime** (brak endpointów, brak subskrypcji
  zdarzeń, brak publikowania zdarzeń).

Innymi słowy: mamy dobrze przetestowaną *bibliotekę* algorytmów ryzyka/ML zamkniętą wewnątrz
serwisów, które jako serwisy nie robią jeszcze nic poza `/health`.

---

## 2. Zweryfikowane fakty (ground truth z sesji 2026-06-25)

Uruchomione lokalnie na **Python 3.12.3** (venv), nie z pamięci:

| Sprawdzenie | Wynik |
|---|---|
| `pytest shared/trading-common` | **104 passed** |
| `pytest services/strategy` | **66 passed** |
| `pytest services/risk-mgmt` | **57 passed** |
| `pytest services/market-data` | **7 passed** (tylko health/skeleton) |
| `ruff check .` (całe repo) | **All checks passed** |
| `mypy shared/trading-common/src` | **Success, no issues** |

Pozostałe serwisy (feature-engine, ml-pipeline, backtest, execution, notification, dashboard)
nie były odpalane w tej sesji, ale mają analogiczną strukturę i przeszły lint/mypy w całym repo.

---

## 3. Macierz komponentów — co istnieje, co jest wpięte

| Serwis | Szkielet (health/ready/metrics) | Komponenty core | Wpięte do API / NATS? |
|---|:---:|---|:---:|
| market-data | ✓ | — (routes zwracają 501) | n/d |
| feature-engine | ✓ | `vol_regime`, `earnings_decay`, `cross_asset` | ✗ osierocone |
| strategy | ✓ | `decay_monitor`, `cost_filter`, `adaptive_weights` | ✗ osierocone (routes = „skeleton") |
| backtest | ✓ | `continuous_validation` | ✗ osierocone |
| ml-pipeline | ✓ | `monitoring/drift_detector` | ✗ osierocone |
| risk-mgmt | ✓ | `adaptive_sizing`, `regime_allocator` | ✗ osierocone |
| execution | ✓ | — | n/d |
| notification | ✓ | — | n/d |
| dashboard | ✓ | — | n/d |
| **shared/trading-common** | n/d | `risk_envelope` (A1), schemas, events, utils, constants | ✓ (poprawna lokalizacja) |

Infrastruktura: `docker-compose.yml` uruchamia infrastrukturę (postgres/redis/nats/prometheus/
grafana/loki/traefik) + 4 serwisy aplikacyjne (market-data, feature-engine, strategy, backtest).
Reszta serwisów jest zakomentowana (rollout fazowy) — to celowe.

---

## 4. Znane luki i potencjalne błędy (priorytetyzowane)

### P0 — blokujące spójność / mylące

1. **`docs/ml_integration_plan.md` NIE ISTNIEJE.** CLAUDE.md odwołuje się do niego **4 razy** jako
   źródła prawdy dla schematów (Fragment 1), zdarzeń (Fragment 2) oraz definicji 4 serwisów ML/AI.
   Brak tego pliku blokuje świadome dodawanie kontraktów („contracts first" to twarda reguła).
   → Trzeba go odtworzyć/dostarczyć albo zdjąć odwołania z CLAUDE.md.

2. **Inwersja priorytetów + komponenty osierocone.** Zrobiono pracę z Tygodnia 19+ przed
   fundamentem z Tygodnia 2. Komponenty nie są wpięte do runtime (patrz §3). Ryzyko: fałszywe
   poczucie postępu — testy zielone, ale system jako całość nie przetwarza ani jednego bara OHLCV.

### P1 — rozjazd kontraktów (trading-common vs CLAUDE.md)

3. **Brak 5 schematów** w `trading_common/schemas.py`, które CLAUDE.md deklaruje jako współdzielone:
   `CompanyProfile`, `FinancialStatements`, `MacroSnapshot`, `SentimentSnapshot`, `FeatureVector`.

4. **Brak 7 rozszerzonych typów zdarzeń** w `EventType` (`trading_common/events.py:13`):
   `REGIME_CHANGED`, `FUNDAMENTALS_UPDATED`, `MACRO_UPDATED`, `SENTIMENT_UPDATED`,
   `COMPANY_CLASSIFIED`, `FEATURES_READY`, `SIGNAL_AGGREGATED`.
   (Część rozszerzeń *jest* już dodana: `CIRCUIT_BREAKER_TRIGGERED`, `MODEL_DRIFT_DETECTED`,
   `MODEL_RETRAINED`, `STRATEGY_STATUS_CHANGED` — więc luka jest częściowa.)

5. **Zła lokalizacja komponentów B3/B4.** `adaptive_weights.py` i `cost_filter.py` leżą w
   `services/strategy/`, a wg `framework_supplement.md` (B3/B4) ich docelowy dom to
   `services/signal-aggregator/` — który jeszcze nie istnieje. Do przeniesienia przy tworzeniu
   signal-aggregator.

### P2 — nieaktualna dokumentacja / braki strukturalne

6. **README §Testowanie nieaktualne:** deklaruje „RAZEM 80" testów; faktycznie ~335 funkcji
   testowych. Tabela do aktualizacji.

7. **`infrastructure/terraform/`** — README i drzewo plików o nim wspominają, katalog nie istnieje.

8. **Ścieżka planu rozwoju:** CLAUDE.md wskazuje `docs/Plan_Rozwoju_Systemu_Tradingowego_2.md`,
   plik faktycznie leży w **roocie** repo. Ujednolicić.

9. **4 serwisy ML/AI** (`fundamental-data`, `macro-data`, `company-classifier`,
   `signal-aggregator`) nie istnieją — planowane na Tydzień 3–19, więc to oczekiwane, ale warto
   mieć świadomość przy czytaniu CLAUDE.md.

10. **README „Status infrastruktury (zweryfikowany) ✓ healthy"** — niemożliwy do weryfikacji w tym
    środowisku (brak Dockera w sandboxie). Deklaracja może być myląca; warto oznaczyć jako
    „oczekiwane", nie „zweryfikowane".

### P3 — subtelne / jakość kodu / środowisko

11. **Martwy walidator** w `schemas.py:32-38`. W Pydantic v2 pola są walidowane w kolejności
    definicji; przy walidacji `high` pole `low` nie ma jeszcze wpisu w `info.data`, więc warunek
    `if "low" in data` nigdy nie jest spełniony → `high_gte_low` to dead code. Cross-walidacja
    *działa* poprawnie, ale tylko przez `low_lte_high` (linie 40-46). Do uproszczenia (jeden
    walidator po `low`, albo `model_validator(mode="after")`). Nie powoduje błędu funkcjonalnego.

12. **Sandbox vs projekt — wersja Pythona.** Domyślny `python3` w sandboxie to **3.11.15**, a
    `requires-python = ">=3.12"`. `pip install -e` odrzuca instalację na 3.11. Lokalne testy
    wymagają jawnie `python3.12` (dostępny: 3.12.3; jest też 3.13.12).

13. **CI nie odpala się na branchach roboczych.** `.github/workflows/ci.yml` triggeruje na
    `push: [main, develop]` i `pull_request -> main`. Push na `claude/*` (bieżący branch) **nie**
    uruchamia CI aż do otwarcia PR. Świadomie weryfikować lokalnie przed PR.

---

## 5. Rekomendacja — od czego zacząć

Dwie ścieżki, decyzja należy do właściciela:

- **A. Trzymać się CLAUDE.md (fundament):** dokończyć `market-data` — `YahooFetcher` +
  `AlphaVantageFetcher`, async storage (TimescaleDB), Redis cache, publikacja
  `MarketDataUpdatedEvent`. Odblokowuje cały pipeline danych. To oficjalny „Next" z CLAUDE.md.

- **B. Sprint spójności (najpierw kontrakty):** zaadresować P0–P1 — odtworzyć/zdjąć referencję do
  `ml_integration_plan.md`, uzupełnić brakujące schematy i `EventType`, wpiąć istniejące komponenty
  do endpointów/subskrypcji, zaktualizować README/CLAUDE.

**Sugestia:** krótki **sprint spójności (B, zakres P0–P1)** *przed* rozwojem market-data, bo:
(1) „contracts first" to twarda reguła projektu, (2) brak `ml_integration_plan.md` blokuje świadome
dodawanie schematów, (3) wpięcie istniejących komponentów daje natychmiastową wartość z już
napisanego, przetestowanego kodu. Potem ścieżka A (market-data) na zdrowych fundamentach.

Minimalny, bezpieczny pierwszy krok (jeśli trzeba „coś ruszyć" bez decyzji strategicznej):
uzupełnić `EventType` o 7 brakujących wartości + dodać 5 brakujących schematów jako kontrakty
(bez logiki) — to czysto addytywne, nie psuje niczego i odblokowuje resztę.

---

## 6. Dziennik postępu (append-only)

### 2026-06-25 — Pełny audyt repozytorium (sesja startowa)
- Przeprowadzono pełną analizę struktury repo (9 serwisów + shared + infra + docs).
- Zweryfikowano testy lokalnie na py3.12: shared 104 ✓, strategy 66 ✓, risk-mgmt 57 ✓,
  market-data 7 ✓. `ruff` i `mypy` (shared) czyste.
- Zidentyfikowano 13 kwestii (P0–P3), w tym brak `docs/ml_integration_plan.md`, osierocone
  komponenty framework_supplement i inwersję priorytetów względem harmonogramu.
- Utworzono ten dokument (`docs/PROJECT_STATUS.md`).
- **Stan:** audyt zakończony, brak zmian w kodzie produkcyjnym. Czeka decyzja: ścieżka A vs B (§5).
