# Trading System — brief do przeglądu zewnętrznego

> **Do czego służy ten plik.** Jest samowystarczalnym opisem systemu: architektury, ścieżki
> decyzyjnej, wszystkich progów i parametrów oraz wyników pierwszego prawdziwego treningu.
> Czytelnik **nie ma dostępu do repozytorium** — wszystko, co potrzebne do oceny logiki,
> zaplanowania kolejnych kroków i wskazania luk, jest tutaj.
>
> Stan na 2026-07-26. System działa (paper trading), 13 serwisów, 849 testów zielonych.
> Pierwszy trening ML na realnych danych **nie przeszedł bramki aktywacyjnej** — szczegóły
> w sekcji 9. To jest główny punkt do dyskusji.

---

## 1. Czym to jest i jaki jest cel

Algorytmiczny system tradingowy zbudowany jako 13 niezależnych mikroserwisów Pythona
(FastAPI), komunikujących się zdarzeniami przez NATS JetStream i zapytaniami HTTP.
Cel: **systematyczne, przekrojowe strategie na akcjach amerykańskich**, na razie wyłącznie
paper trading. Kapitał realny wymaga 30 dni papierowego handlu z dodatnim Sharpe'em.

Filozofia projektu, istotna przy ocenie:

- **Ryzyko jest bramą, nie sugestią.** Każdy sygnał przechodzi twarde reguły; nie da się ich
  ominąć „bo model jest pewny".
- **Uczciwość ważniejsza niż wynik.** Bramka aktywacji ma odrzucać modele; nieudany trening
  jest wynikiem, nie awarią. Metryki są dobrane tak, żeby odróżnić przewagę od farta.
- **Kontrakty najpierw.** Typy współdzielone żyją w bibliotece `trading-common`; serwis A nigdy
  nie importuje z serwisu B.

---

## 2. Serwisy (13)

### Rdzeń

| Serwis | Rola |
|---|---|
| `market-data` | Pobieranie OHLCV (Yahoo → Alpha Vantage), walidacja, TimescaleDB, cache Redis, publikacja zdarzeń |
| `feature-engine` | Cechy techniczne + wzbogacenie fundamentami/stylem, **rangi przekrojowe** |
| `strategy` | Reguła momentum na rangach → sygnał, brama ryzyka, filtr kosztów, monitor degradacji |
| `backtest` | Silnik long/flat bez podglądania przyszłości, walk-forward, cotygodniowa rewalidacja |
| `ml-pipeline` | Trening (PyTorch), rejestr MLflow, serwowanie głosu ML, dzienny monitoring driftu |
| `risk-mgmt` | Wielkość pozycji, limity reżimowe/sektorowe, wyłącznik bezpieczeństwa, stan portfela |
| `execution` | Paper trading: wypełnienia, ochronne wyjścia SL/TP, idempotencja |
| `notification` | Alerty (log, Slack, Telegram, e-mail) z 5 strumieni zdarzeń |
| `dashboard` | Backend-for-frontend, agreguje odczyty z 4 serwisów |

### Rozszerzenie ML/AI

| Serwis | Rola |
|---|---|
| `fundamental-data` | SEC EDGAR, sprawozdania roczne, 9-sygnałowy F-Score Piotroskiego |
| `macro-data` | FRED (krzywa, spready, PMI), regułowa detekcja reżimu rynkowego |
| `company-classifier` | Profil spółki → styl inwestycyjny (growth/value/blend) + routing stosu modeli |
| `signal-aggregator` | **Węzeł decyzyjny**: łączy sygnał reguł + głos ML + reżim makro w jedną decyzję |

Infrastruktura: PostgreSQL 16 + TimescaleDB, Redis 7, NATS JetStream, Prometheus/Grafana/Loki,
Traefik. Uruchamiane wyłącznie przez `docker compose`.

---

## 3. Ścieżka decyzyjna (przepływ zdarzeń)

```
market-data ──market_data.updated──▶ feature-engine ──features.ready──┬──▶ strategy
                                            ▲                         │        │
                              fundamentals.updated                    │  signal.generated
                              company.classified                      ▼        │
                                            │                    ml-pipeline    │
                                   fundamental-data          ml.signal_generated│
                                   company-classifier                 │        │
                                                                      ▼        ▼
   macro-data ──macro.regime_changed────────────────────────▶ signal-aggregator
        │                                                              │
        │                                                     signal.aggregated
        │                                                              ▼
        └──────────────macro.regime_changed───────────────────▶    risk-mgmt
                                                                       │
                                                                order.requested
                                                                       ▼
                                                                   execution
                                                                       │
                                                        order.filled ──┤
                                                                       ▼
                                       (HTTP: metryki portfela z powrotem do risk-mgmt)

backtest ──backtest.strategy_revalidated──▶ strategy
notification ◀── 5 strumieni alertów        dashboard ── (HTTP, tylko odczyt)
```

**Kluczowe własności tej ścieżki:**

1. **Agregator jest węzłem decyzyjnym.** Sygnał ze `strategy` i głos ML to *komponenty*, nie
   decyzje. Dopiero `signal.aggregated` trafia do risk-mgmt.
2. **ML nigdy nie handluje samodzielnie.** `ml.signal_generated` celowo **nie zawiera poziomów
   SL/TP** i nie jest agregowany bez komponentu strategii. Modeluje modulację decyzji.
3. **Sprzężenie zwrotne portfela.** Każde wypełnienie wraca po HTTP do risk-mgmt, więc wielkość
   kolejnych pozycji i wyłącznik bezpieczeństwa reagują na realny stan portfela.
4. **Reżim makro steruje ekspozycją automatycznie** — `macro.regime_changed` idzie równolegle do
   agregatora (kierunkowy bias) i do risk-mgmt (limity ekspozycji i sektorowe).

### Strumienie JetStream

`MARKET_DATA`, `FEATURES`, `SIGNALS`, `ORDERS`, `RISK`, `ML`, `BACKTEST`, `STRATEGY`, `MACRO`,
`FUNDAMENTALS`, `COMPANY` — 11 strumieni, subskrypcje durable z `max_deliver`; komunikat trwale
błędny jest terminowany (`term`), przejściowy → `nak` i ponowne dostarczenie.

---

## 4. Cechy i ich przetwarzanie

### Cechy techniczne (liczone z OHLCV, wspólny kod dla treningu i serwowania)

`close`, `return_1d`, `return_5d`, `return_20d`, `momentum_20` (= return_20d), `sma_10`,
`sma_20`, `sma_50`, `price_to_sma50`, `rsi_14` (Wilder), `realized_vol_20`, `volume_ratio`.

**Ważne:** definicje cech i transformacja rang mieszkają w bibliotece współdzielonej
(`trading_common.features` / `trading_common.ranking`), więc ml-pipeline liczy cechy do treningu
**tym samym kodem**, którym feature-engine liczy je na produkcji. Zgodność trening/serwowanie
jest strukturalna, nie deklaratywna.

### Ranga przekrojowa

Wszystkie cechy są przekształcane na **percentylową rangę przekrojową w [0,1]** w obrębie
uniwersum danej sesji (tie-aware, średnia ranga przy remisach). Model nigdy nie widzi wartości
surowych — to świadoma decyzja (López de Prado): rangi są odporne na zmiany reżimu zmienności.

### Cechy Tier-2 (wzbogacenie)

feature-engine dokleja przy odczycie: `f_score` (Piotroski 0–9), `fund_net_margin`, `fund_roa`,
`fund_leverage`, `style_growth`, `style_value`. **Uwaga — istotna luka:** te cechy trafiają do
serwowania, ale **nie do zbioru treningowego** (patrz sekcja 10, luka L1).

### Cechy makro

Do zbioru dołączany jest one-hot reżimu makro: `macro_expansion`, `macro_recovery`,
`macro_slowdown`, `macro_contraction`, `macro_crisis`. **Obecnie są to stałe zera**, bo trening
nie dostaje historii reżimów (luka L2).

### Cechy wykluczone z modelu

`close`, `sma_10`, `sma_20`, `sma_50` — ich ranga przekrojowa jest proxy poziomu ceny, a nie
sygnału. Wykluczone celowo.

---

## 5. Reguła strategii (nie-ML)

```
BUY  ← ranga momentum ≥ 0.80  ORAZ  RSI < 70   (czoło uniwersum, jeszcze nie wykupione)
SELL ← ranga momentum ≤ 0.20  ORAZ  RSI > 30   (ogon uniwersum, jeszcze nie wyprzedane)
HOLD ← w pozostałych przypadkach
```
Pewność sygnału = ranga momentum (dla BUY) lub 1 − ranga (dla SELL). Stop-loss ustawiany
procentowo (niezależnie od zmienności — patrz luka L5).

**Monitor degradacji strategii** (`StrategyDecayMonitor`), sprawdzany dziennie:
- ACTIVE: Sharpe ≥ 0.5, Profit Factor ≥ 1.2, Win Rate ≥ 0.4
- DEACTIVATED: Sharpe < 0 **lub** PF < 0.8 **lub** ponad 30 dni na probacji
- PROBATION: wszystko pomiędzy

Backtest co tydzień (sobota 06:00 UTC) robi walk-forward i **rekomenduje** zmianę statusu
(`backtest.strategy_revalidated`); `strategy` jest właścicielem statusu i stosuje rekomendację.

---

## 6. Warstwa ML — projekt

Pełny projekt: cross-sectional (pula całego uniwersum), płytka sieć, etykiety triple-barrier,
walidacja purged walk-forward. Poniżej wszystkie parametry.

### Etykiety — Triple Barrier (López de Prado)

| Parametr | Wartość | Znaczenie |
|---|---|---|
| `sigma_window` | 20 sesji | okno do estymacji dziennej zmienności |
| `pt_mult` / `sl_mult` | 2.0 / 2.0 | bariery na ±2σ·√h wokół ceny wejścia |
| `horizon` | 10 sesji | bariera pionowa |

Zasady: wejście na **następnej** świecy po sesji cechowej; dotknięcie obu barier w tej samej
świecy = konserwatywnie strata; bariera pionowa rozstrzyga znakiem zwrotu netto; okno ucięte
przez koniec historii bez dotknięcia bariery → **brak etykiety** (wiersz odrzucony, nie zgadywany).

Etykieta binarna: 1 = pierwsza dotknięta bariera górna.

### Budowa zbioru

| Parametr | Wartość |
|---|---|
| `min_history` | 60 sesji zanim symbol wejdzie do przekroju |
| `lookback` | 250 sesji podawanych do liczenia cech (jak w serwowaniu) |
| `min_universe` | 2 symbole — sesja z mniejszym przekrojem jest pomijana |

Dla każdej sesji: policz cechy dla **całego** uniwersum mającego historię → ranguj przekrojowo →
etykietuj → wrzuć do wspólnej puli. Brakująca cecha Tier-2 → neutralne 0.5.

### Podział czasowy — purged walk-forward

| Parametr | Wartość |
|---|---|
| `train_size` | 756 sesji (~3 lata) |
| `test_size` | 63 sesje (~3 miesiące) |
| `holdout_size` | 126 sesji (~6 miesięcy, **nietykane** przy selekcji) |
| `val_size` | 63 sesje (ogon okna treningowego, na early stopping i kalibrację) |
| `embargo` | 5 sesji |
| przerwa czyszcząca | horizon + embargo = 15 sesji |

Nigdy podział losowy. Przerwa między train a test usuwa wyciek z nakładających się etykiet
(etykieta sięga 10 sesji w przód).

### Model

PyTorch MLP: warstwy ukryte **(32, 16)**, dropout 0.3, lr 3e-3, weight decay 1e-4,
batch 256, max 200 epok, **min 30 epok przed early stoppingiem** (dropout czyni wczesną stratę
walidacyjną hałaśliwą — bez tego trening zatrzymywał się na szczęśliwym minimum w 3. epoce),
patience 15, ważenie klas przez `pos_weight`, kalibracja temperaturowa (LBFGS) na zbiorze
walidacyjnym.

### Metryka decyzyjna i bramka aktywacji

Metryką **nie jest** accuracy ani AUC, tylko **Sharpe portfela po kosztach**:
codziennie rebalansowany, równoważony, **long-only top-kwintyl** (quantile 0.2) po
predykcji, koszty 5 bps od obrotu jednostronnego.

Bramka przepuszcza model tylko gdy **wszystkie** warunki są spełnione:
1. Sharpe na holdoucie > **0.5**
2. Sharpe > 0.5 na **co najmniej 2 z 3 ostatnich foldów** (warunek anty-fartowy)
3. Brier nie gorszy niż wskaźnik bazowy + 0.01 (sensowna kalibracja)

Model jest zapisywany do MLflow **niezależnie od wyniku bramki** (nieudany bieg to też wynik).
**Promocja jest ręczna** — alias `production` w rejestrze; serwowanie przeładowuje się na gorąco.

### Diagnostyka odróżniająca przewagę od farta

Sharpe na 63-sesyjnym foldzie jest hałaśliwy, więc raport podaje dodatkowo:

- **`lift`** = trafność wybranego top-kwintyla − trafność bazowa. Ile realnie daje *selekcja*.
  **Wysoki Sharpe przy lifcie ≈ 0 to fart, nie sygnał.**
- **`pred_std`, p10, p90** — rozrzut predykcji. Bliski zeru = model zapadnięty, zwraca stałą.
- **`base_rate`** — jaki procent spółek rósł w danym oknie (reżim rynkowy).

### Serwowanie

`features.ready` → pobierz rangowany wektor + reżim makro → złóż wiersz **w dokładnej kolejności
cech z metadanych modelu** → predykcja. Progi: p ≥ **0.55** → BUY, p ≤ **0.45** → SELL,
strefa martwa pomiędzy jest **cicha** (brak zdarzenia). Brak **większości** oczekiwanych cech →
**odmowa predykcji** (dryf schematu to nie to samo co rzadkie dane).

### Pętla uczenia się z realnych wyników

Każda predykcja (także HOLD) trafia do rolling logu. Dojrzały głos (po ~10 sesjach) jest
rozliczany **tą samą regułą triple-barrier co trening**, na świeżej historii. Zrealizowany zwrot
ze znakiem kierunku:
1. leci do agregatora (`POST /outcomes`) i **przesuwa wagi adaptacyjne źródła „ml"**,
2. zasila kroczące Sharpe/accuracy do wykrywania degradacji.

Głos nierozstrzygalny po 42 dniach jest porzucany (bez fabrykowania wyniku).

### Monitoring driftu (codziennie, 24 h, pierwszy bieg 1 h po starcie)

| Sygnał | Próg |
|---|---|
| PSI cechy | > 0.20 → dryf cech |
| Test KS na predykcjach | p < 0.01 → przesunięcie rozkładu predykcji |
| Spadek kroczącego Sharpe'a | < −30% względem bazy → degradacja wydajności |
| Krocząca trafność | < 0.48 → gorzej niż losowo |

Przy mniej niż 10 rozstrzygniętych wynikach wejścia wydajnościowe są **neutralne**, a raport
jawnie ustawia `performance_measured=false` (żadnych zmyślonych metryk).

---

## 7. Agregacja sygnałów

Trzy źródła: `strategy` (reguły), `ml` (model), `macro` (reżim).

- Głosowanie ważone **znakiem pewności**: +conf dla BUY, −conf dla SELL, 0 dla HOLD.
- Wagi **adaptacyjne** (EWP na podstawie kroczącego information ratio), floor 0.05, cap 0.60,
  renormalizowane po obecnych źródłach — brak źródła jest „darmowy".
- Próg decyzji: |wynik ważony| ≥ **0.2** → BUY/SELL, inaczej HOLD.
- Bias makro: expansion/recovery → BUY, contraction/crisis → SELL, **slowdown → brak komponentu**
  (znany neutralny reżim nie może zabierać wagi strategii).
- Bufor sygnałów per symbol z **TTL 1 dnia**, liczonym od znacznika emisji zdarzenia (żeby
  odtworzenie strumienia nie wskrzesiło starych sygnałów).
- Wynik przechodzi przez **filtr kosztów**: koszt = spread 5 bps + poślizg 5 bps + wpływ 2 bps,
  liczony na obie nogi; wymagana przewaga ≥ **2× koszt**, inaczej HOLD.
- ML **nigdy nie agreguje samotnie** — bez komponentu strategii nie ma decyzji.

---

## 8. Zarządzanie ryzykiem (twarde, nienegocjowalne)

### Brama `RiskEnvelope` (każdy sygnał)

| Limit | Wartość |
|---|---|
| Maks. pozycja | 5% portfela |
| Maks. łączna ekspozycja | 80% |
| Maks. strata na transakcję | 2% |
| Maks. strata dzienna | 5% |
| Maks. obsunięcie | 15% |
| Maks. skorelowanych pozycji | 3 |
| Min. pewność sygnału | 0.55 |

**Zlecenie bez `stop_loss` jest niemożliwe** — wymuszone walidacją kontraktu `TradingSignal`
i powtórnie przez `RiskEnvelope`.

### Wielkość pozycji zależna od obsunięcia

Bazowe ryzyko 2% na transakcję, **pełne do obsunięcia 5%** (martwa strefa), potem liniowo
do zera przy obsunięciu 15%. Cap 5% wartości portfela na pozycję.

### Limity reżimowe

| Reżim | Maks. ekspozycja | Dozwolone sektory |
|---|---|---|
| expansion | 90% | wszystkie |
| recovery | 80% | wszystkie |
| slowdown | 60% | defensywne (Health Care, Consumer Staples, Utilities…) |
| contraction | 35% | defensywne |
| crisis | 15% | defensywne |

Sektor spółki pochodzi z company-classifier. **Uwaga:** nazwy sektorów muszą być w konwencji
GICS („Information Technology", „Consumer Staples") — nierozpoznany napis blokuje w reżimach
restrykcyjnych (konserwatywnie).

### Wyłącznik bezpieczeństwa (uzbrojony 24/7)

| Poziom | Warunek | Akcja |
|---|---|---|
| YELLOW | obsunięcie > 8% | ogranicz ryzyko |
| RED | strata dzienna > 5% | wstrzymaj nowe zlecenia do jutra |
| BLACK | obsunięcie > 15% | zamknij wszystkie pozycje |

Stan przeżywa restart (odtwarzany z Redisa). **Znane uproszczenie:** poziomy same się kasują,
gdy warunki się poprawią — realny system wymagałby ręcznego resetu z BLACK.

### Wykonanie (paper)

Long-only (SELL = wyjście, ograniczone do posiadanej ilości, pomijane gdy brak pozycji — zgodnie
z silnikiem backtestu long/flat). Pozycje niosą SL/TP; każde przeszacowanie ceny sprawdza poziomy
i wychodzi przy przebiciu. Wypełnienia idempotentne po `event_id` zlecenia.

---

## 9. Wynik pierwszego prawdziwego treningu (2026-07-26) — **do oceny**

### Dane

34 duże spółki amerykańskie z rozrzutem po sektorach GICS, 1505 sesji każda (2020-07 → 2026-07).
Zbiór: **48 827 próbek**, 1438 sesji, mediana 34 spółek na sesję, 13 cech,
`positive_rate` 0.552.

### Werdykt bramki: **ODRZUCONY**

Powód formalny: Sharpe na holdoucie **−1.07** (próg 0.5).

| Metryka | Holdout | Interpretacja |
|---|---|---|
| Sharpe | −1.07 | portfel tracił |
| AUC | 0.483 | poniżej 0.5 — gorzej niż losowo |
| lift | **+0.0009** | selekcja nie daje przewagi |
| pred_std | 0.0073 (p10 0.475, p90 0.494) | model zwraca praktycznie stałą |

### Foldy — i dlaczego wysokie Sharpe'y są mylące

| Fold | Sharpe | AUC | lift | base_rate | wniosek |
|---|---|---|---|---|---|
| fold_0 | **3.85** | 0.506 | **−0.013** | 0.677 | rynek, nie model |
| fold_1 | 0.23 | 0.504 | +0.021 | 0.567 | — |
| fold_2 | 1.81 | 0.499 | −0.007 | 0.610 | rynek |
| fold_3 | 1.72 | 0.428 | **−0.054** | 0.592 | rynek (selekcja szkodziła) |
| fold_4 | 1.10 | 0.518 | +0.026 | 0.519 | słaby sygnał? |
| fold_5 | −1.76 | 0.471 | −0.005 | 0.450 | — |
| fold_6 | 3.65 | 0.563 | +0.066 | 0.626 | najlepszy fold |
| fold_7 | 2.57 | 0.541 | +0.028 | 0.562 | — |

**Średnia AUC foldów: 0.504. Średni lift: +0.008 przy odchyleniu 0.033, znaki zmienne (−+−−+−++).**

Wniosek: model **nie ma przewagi przekrojowej**. Foldy z wysokim Sharpe'em wypadły dobrze,
bo w tych oknach rosła większość spółek (`base_rate` do 0.677) — dowolny portfel long zarabiał,
a wybór modelu bywał *gorszy* od średniej (ujemny lift). Gdyby raport zawierał sam Sharpe,
wyglądałby jak „6 z 8 foldów dodatnich", czyli fałszywy sukces. Warunek „≥2 z 3 ostatnich
foldów" **przeszedł**; model obalił dopiero nietknięty holdout.

### Otwarte pytanie do rozstrzygnięcia

Czy `pred_std` ≈ 0.007 oznacza **brak sygnału w danych**, czy **niedouczenie modelu**?
Rozstrzygnięcie: sprawdzić AUC **na zbiorze treningowym**. Jeśli model nie potrafi dopasować
nawet danych treningowych → problem optymalizacji (pojemność, epoki, lr). Jeśli dopasowuje
train, a nie generalizuje → w tych cechach po prostu nie ma przewagi. **Ta diagnostyka nie jest
jeszcze zaimplementowana** i jest naturalnym pierwszym krokiem.

---

## 10. Znane luki i dług techniczny

| # | Luka | Waga | Opis |
|---|---|---|---|
| **L1** | Tier-2 nie trafia do treningu | wysoka | Zbiór budowany jest wyłącznie z OHLCV. F-Score, marże i styl są dostępne przy serwowaniu, ale model nigdy się na nich nie uczył — więc ich nie używa. |
| **L2** | Cechy makro są stałymi zerami | wysoka | Trening nie dostaje historii reżimów, więc 5 z 13 cech nie niesie informacji. |
| **L3** | Uniwersum 34 spółki | wysoka | Ranga przekrojowa z 34 nazw jest zgrubna, top-kwintyl to 7 pozycji. Plan zakłada stosy per styl dopiero przy ≥200 spółek. |
| **L4** | Brak diagnostyki train-vs-val | średnia | Nie wiadomo, czy model niedouczony, czy brak sygnału (patrz sekcja 9). |
| **L5** | Stop-loss procentowy, nie zmiennościowy | średnia | Strategia stawia SL jako stały procent, ignorując zmienność spółki — niespójne z etykietami ML, które są skalowane przez σ. |
| **L6** | Konsumenci push nie skalują się | średnia | Subskrypcje push nie rozkładają obciążenia; wiele replik wymaga konsumentów pull/queue-group. Dziś repliki >1 mają tylko serwisy bez subskrypcji. |
| **L7** | Stan jako snapshot, nie log zdarzeń | średnia | Portfel i broker to migawki w Redisie. Brak odtwarzalnej historii zdarzeń. |
| **L8** | Wyłącznik sam się kasuje | średnia | Wyjście z BLACK powinno wymagać decyzji człowieka. |
| **L9** | Podwójne bramkowanie kosztami | niska | Filtr kosztów działa i w strategii, i w agregatorze — świadomie konserwatywne, ale warto zweryfikować. |
| **L10** | Brak meta-labelingu i modelu konkurencyjnego | niska | Świadomie odłożone do v2 (meta-labeling, GBDT jako challenger, stosy per styl). |

---

## 11. Historia błędów wartych uwagi przy przeglądzie

Kilka defektów znalezionych przy pierwszym realnym uruchomieniu — pokazują, jakiego typu pułapki
ten system generuje:

1. **Cache w market-data zwracał mniej danych, niż zażądano.** Klucz `(symbol, interval)` bez
   `limit`, ale zapis już przycięty do `limit`. feature-engine czytał 250 świec, trening prosił
   o 2000 i dostawał 250 — model widział 183 sesje zamiast 1438. Naprawione: wpis cache'a
   odpowiada tylko na zapytania, które faktycznie pokrywa.
2. **Rejestr MLflow po cichu nie działał.** Katalog wolumenu powstawał jako root, kontener działa
   jako użytkownik nieuprzywilejowany → trening kończył się sukcesem i **niczego nie zapisywał**.
3. **`httpx` zadeklarowany tylko w zależnościach deweloperskich**, choć importowany w runtime —
   testy zielone, kontener nie wstawał. Dodany strażnik w CI porównujący importy z deklaracjami.
4. **Healthcheck Postgresa nie uwierzytelniał** (`pg_isready`), więc niezgodne hasło dawało
   kontener „healthy" i padające zapisy.

Wspólny mianownik: **zielone testy nie mówiły nic o tym, czy system da się uruchomić.** Wiele
z tych pułapek jest niewidocznych bez realnego środowiska.

---

## 12. Czego oczekuję od przeglądu

1. **Ocena logiki ML.** Czy projekt (rangi przekrojowe, triple barrier h=10, purged walk-forward,
   Sharpe top-kwintyla jako metryka decyzyjna) jest spójny? Gdzie są błędy metodologiczne?
2. **Priorytetyzacja kolejnych kroków.** Mam trzy kandydatury: poszerzyć uniwersum (L3),
   wprowadzić Tier-2 do treningu (L1), podać historię makro (L2). Co da najwięcej i w jakiej
   kolejności? Czy jest coś ważniejszego, czego nie widzę?
3. **Luki w logice biznesowej**, nie tylko ML: czy ścieżka sygnał → agregacja → ryzyko →
   wykonanie ma dziury? Czy reguły ryzyka są spójne (np. L5 — SL procentowy vs etykiety σ)?
4. **Czy bramka aktywacji jest dobrze skonstruowana?** Czy warunek „≥2 z 3 ostatnich foldów"
   plus holdout wystarcza? Czy `lift` powinien być **twardym** warunkiem bramki, a nie tylko
   diagnostyką? (Ten bieg przeszedł warunek foldowy mimo lifta ≈ 0.)
5. **Realizm horyzontu.** Czy 10 sesji przy dziennych danych i 13 cechach technicznych to
   sensowny cel, czy z góry przegrana walka o stosunek sygnału do szumu?

---

## Załącznik: parametry w jednym miejscu

```
ETYKIETY:      sigma_window=20, pt=sl=2.0σ√h, horizon=10 sesji
ZBIÓR:         min_history=60, lookback=250, min_universe=2, brak cechy → 0.5
PODZIAŁ:       train=756, test=63, holdout=126, val=63, embargo=5, przerwa=15
MODEL:         MLP (32,16), dropout 0.3, lr 3e-3, wd 1e-4, batch 256,
               max 200 epok, min 30 epok, patience 15, kalibracja temperaturowa
PORTFEL OCENY: top-kwintyl (0.2), long-only, równoważony, rebalans dzienny, 5 bps
BRAMKA:        holdout Sharpe > 0.5 AND ≥2/3 ostatnich foldów AND Brier ≤ bazowy + 0.01
SERWOWANIE:    BUY p≥0.55, SELL p≤0.45, strefa martwa cicha,
               >50% brakujących cech → odmowa predykcji
DRYF:          PSI>0.20, KS p<0.01, spadek Sharpe'a <−30%, trafność <0.48
               min. 10 wyników do pomiaru wydajności, porzucenie głosu po 42 dniach
AGREGACJA:     próg 0.2, wagi EWP [0.05, 0.60], TTL bufora 1 dzień,
               koszty 5+5+2 bps × 2 nogi, wymagana przewaga ≥ 2× koszt
STRATEGIA:     BUY ranga≥0.80 & RSI<70, SELL ranga≤0.20 & RSI>30
DEGRADACJA:    ACTIVE Sharpe≥0.5, PF≥1.2, WR≥0.4; DEACTIVATE Sharpe<0 lub PF<0.8
               lub >30 dni probacji
RYZYKO:        pozycja 5%, ekspozycja 80%, strata/transakcja 2%, dzienna 5%,
               obsunięcie 15%, min. pewność 0.55, maks. 3 skorelowane
SIZING:        2% ryzyka do DD 5%, liniowo do 0% przy DD 15%
REŻIMY:        expansion 90% / recovery 80% / slowdown 60% / contraction 35% / crisis 15%
WYŁĄCZNIK:     YELLOW dd>8%, RED strata dzienna>5%, BLACK dd>15%
HARMONOGRAM:   backtest sobota 06:00 UTC, makro co 6 h, fundamenty tygodniowo,
               monitoring ML co 24 h (pierwszy bieg 1 h po starcie)
```
