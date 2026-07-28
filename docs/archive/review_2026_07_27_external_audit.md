# REVIEW-2026-07-27 — Audyt systemu tradingowego i backlog naprawczy

> ## 🗄️ ARCHIWUM — audyt zewnętrzny, dokument historyczny
>
> Wejściem był [`project_brief_for_review.md`](project_brief_for_review.md) (stan 2026-07-26).
> **Każde twierdzenie tego audytu zostało zweryfikowane na kodzie i na danych z realnego biegu**
> — wyniki weryfikacji (w tym cztery korekty i dwa znaleziska własne) są w
> [`backlog_2026_07_27.md`](backlog_2026_07_27.md) §1–2 oraz w logu postępu w
> [`../../CLAUDE.md`](../../CLAUDE.md). Nie traktuj zaleceń stąd jako obowiązujących bez
> sprawdzenia tamtej weryfikacji: część została przyjęta ze zmianami, część odrzucona.

> **Wejście:** `project_brief_for_review.md` (stan 2026-07-26, 13 serwisów, 849 testów, pierwszy trening ML odrzucony przez bramkę).
> **Zakres:** logika ML, logika przepływu zdarzeń, ocena parametrów, priorytetyzacja, roadmapa.
> **Format:** dokument roboczy dla Claude Code. Każde zadanie ma ID, serwis, ścieżki, kryteria akceptacji.

---

## 0. Jak korzystać z tego pliku (instrukcja dla Claude Code)

1. Zadania wykonuj **w kolejności Tier 0 → Tier 1 → Tier 2 → Tier 3**. Wewnątrz tieru kolejność ID.
2. Każde zadanie ma sekcję `Kryteria akceptacji` — to są **testy do napisania**, nie opis. Zadanie jest zrobione, gdy testy przechodzą.
3. Zadania oznaczone `DECYZJA:` wymagają potwierdzenia człowieka przed implementacją — zapytaj, nie zgaduj.
4. Nie łącz zadań z różnych tierów w jednym PR. Jeden task ID = jeden commit/PR.
5. Nie uruchamiaj ponownego treningu przed zamknięciem **całego Tier 0** — poprzedni wynik jest niediagnostyczny.
6. Ścieżki plików w tym dokumencie są **hipotezami** wynikającymi z briefu. Zweryfikuj rzeczywistą lokalizację przed edycją; jeśli się różni, popraw ten dokument.

---

## 1. Streszczenie — 7 ustaleń krytycznych

| # | Ustalenie | Konsekwencja |
|---|---|---|
| **F1** | **Bramka aktywacji nie ma mocy statystycznej.** Sharpe na 63 sesjach ma błąd standardowy ≈ 2.0, na 126 sesjach ≈ 1.4. Próg 0.5 leży wewnątrz szumu. | Cała tabela foldów (3.85, 1.81, −1.76…) to szum. Odrzucenie modelu było *przypadkowo* trafne, ale bramka nie potrafiłaby też przyjąć dobrego modelu. **Sharpe nie może być metryką decyzyjną przy tej długości okna.** |
| **F2** | **Triple barrier zdegenerował się do fixed-horizon.** Bariery na `2.0·σ·√h` to dokładnie ±2 odchylenia standardowe rozkładu 10-dniowego zwrotu. Prawdopodobieństwo dotknięcia którejkolwiek ≈ 9–15%. | **~85–90% etykiet rozstrzyga bariera pionowa**, czyli znak zwrotu 10-dniowego. Cała przewaga metody López de Prado (ścieżka, heteroskedastyczność) jest nieaktywna. |
| **F3** | **Horyzont etykiety (10 sesji) ≠ horyzont portfela oceny (1 sesja, rebalans dzienny).** | Model uczy się jednego obiektu, jest oceniany na innym. Dodatkowo: pełny obrót dzienny przy 5 bps to **~11–13% kosztów rocznie ≈ 0.5–0.6 jednostki Sharpe'a**. Znaczna część holdoutowego −1.07 to koszt obrotu zapadniętego modelu, nie dowód na ujemną przewagę. |
| **F4** | **Próg serwowania (`p ≥ 0.55`) nie odpowiada metryce oceny (top-kwintyl).** Przy `base_rate` 0.552 próg 0.55 to „powyżej mediany rynku" — przepuszcza ~połowę uniwersum. | Trzeci już przypadek rozjazdu trening/serwowanie. Model uczony i oceniany **przekrojowo** jest serwowany **absolutnie**. |
| **F5** | **Cechy makro są stałymi zerami w treningu, ale niezerowymi jedynkami w serwowaniu.** | To nie jest „zmarnowana pojemność" (jak w opisie L2) — to **żywy błąd poprawności**. Model dostaje na produkcji wejście spoza rozkładu treningowego. Priorytet krytyczny, naprawa 15 minut (usuń kolumny z listy cech). |
| **F6** | **`momentum_20` to główny sygnał strategii i jest po złej stronie znanej anomalii.** 1-miesięczny zwrot przekrojowy to klasyczny **short-term reversal** (Jegadeesh 1990). Kanoniczne momentum 12-1 *pomija* ostatni miesiąc dokładnie po to, by go uniknąć. Reguła `BUY ← ranga momentum_20 ≥ 0.80` kupuje więc odwrotność. | Wyjaśnia ujemny `lift` w foldach 0, 2, 3. Zestaw cech nie zawiera **żadnej** cechy o horyzoncie > 20 sesji. |
| **F7** | **Uniwersum 34 nazw jest wiążącym ograniczeniem, nie niedogodnością.** Prawo fundamentalne Grinolda: `IR ≈ IC·√BR`. Błąd standardowy IC przy 34 nazwach ≈ 0.17 na przekrój; przy 500 ≈ 0.045. Wykrywalność sygnału rośnie ~4×. | Rozszerzenie uniwersum daje **więcej niż wszystkie pozostałe zmiany razem** i kosztuje tylko czas pobrania danych. |

**Odpowiedź na pytanie „brak sygnału czy niedouczenie":** prawie na pewno **ani jedno, ani drugie w czystej postaci**. `pred_std = 0.0073` przy zastosowanej kalibracji temperaturowej jest w dużej mierze **artefaktem kalibracji**: gdy walidacyjne AUC ≈ 0.5, optymalna temperatura rośnie i spłaszcza predykcje do `base_rate`. Diagnostyka musi rozdzielić trzy rzeczy: AUC treningowe, `pred_std` **przed** kalibracją i wartość temperatury `T`. Zadanie **T0-3**.

---

## 2. Diagnoza wyniku treningu (2026-07-26) — rozbiór liczbowy

### 2.1 Efektywna wielkość próby

```
Nominalnie:  48 827 próbek
Sesje:       1438
Nakładanie:  horizon = 10 → ~144 niezależnych punktów w czasie
Przekrój:    34 duże spółki US, średnia korelacja par ~0.4–0.6
             → efektywna liczba niezależnych nazw ≈ 5–8
Efektywne N ≈ 144 × 6 ≈ 850 obserwacji
Cechy:       13 nominalnie
```

Własne źródło projektu (`Kompletny przewodnik po modelach AI/ML…`) podaje wymóg **50–200 obserwacji na cechę** w finansach. Przy efektywnym N ≈ 850 i 13 cechach jesteś na **~65 obs/cechę** — dokładnie na dolnej granicy. Po rozszerzeniu uniwersum do 500 nazw: ~440 obs/cechę.

### 2.2 Ile cech naprawdę niesie informację

| Cecha | Status |
|---|---|
| `momentum_20` | **duplikat** `return_20d` (brief §4 explicite: „= return_20d") → kolinearność doskonała |
| `close`, `sma_10`, `sma_20`, `sma_50` | wykluczone (słusznie) |
| `macro_*` × 5 | stałe zera → **zerowa wariancja** |
| `return_20d`, `price_to_sma50` | silnie skorelowane po rangowaniu (oba ≈ momentum 1–2 mies.) |
| **Realnie niezależne** | `return_1d`, `return_5d`, `return_20d`, `rsi_14`, `realized_vol_20`, `volume_ratio` — **6 cech, wszystkie krótkoterminowe techniczne** |

Nie ma tu ani jednej cechy z listy Tier-1 z własnych materiałów projektu (momentum 12-1, EV/EBITDA, GP/Assets, F-Score, revision breadth). Model dostał najsłabszy możliwy zestaw.

> Uwaga do `rsi_14`: materiał źródłowy projektu wprost stwierdza, że RSI **nie ma istotnej statystycznie mocy predykcyjnej** po korekcie na multiple testing (badanie na 15 mln+ kombinacji parametrów). Trzymanie RSI jako filtru higienicznego jest do obrony; oczekiwanie po nim alfy — nie.

### 2.3 Rozkład etykiet — dowód na F2

```
bariera = pt_mult · σ_daily · √h = 2.0 · σ · √10 = 6.325·σ_daily
SD(zwrot 10-dniowy) = σ·√10 = 3.162·σ_daily
bariera / SD_terminalne = 2.0

Zasada odbicia (BM bez dryfu):
P(max W_t ≥ a) = 2·P(W_T ≥ a) = 2·Φ̄(2.0) = 0.0455
P(dotknięcie którejkolwiek) ≈ 0.09   (z dryfem byka: ~0.12–0.15)

⇒ ~85–90% etykiet rozstrzyga bariera PIONOWA
```

**To nie jest triple barrier, to fixed-horizon w przebraniu.** Docelowo `pt_mult = sl_mult ≈ 1.0–1.2` daje ~50–60% rozstrzygnięć na barierach poziomych. Zadanie **T2-4**.

### 2.4 Skąd naprawdę wziął się holdout Sharpe −1.07

```
pred_std = 0.0073, p10 = 0.475, p90 = 0.494
⇒ ranking top-kwintyla jest w praktyce losowy każdego dnia
⇒ obrót dzienny bliski 100%
⇒ koszt ≈ 100% × 5 bps × 252 sesji ≈ 12.6% rocznie
⇒ zmienność portfela 7 dużych spółek EW ≈ 20% rocznie
⇒ sam dryf kosztowy ≈ 0.6 jednostki Sharpe'a
```

Czyli **ponad połowa** ujemnego wyniku to koszt tarcia zapadniętego modelu, a nie ujemna selekcja. Raport **musi** podawać Sharpe brutto i netto oraz obrót. Zadanie **T0-5**.

### 2.5 Dlaczego tabela foldów jest nieczytelna

```
SE(Sharpe zannualizowany) ≈ √(252 / N_dni)
  N = 63  (fold)    → SE ≈ 2.00
  N = 126 (holdout) → SE ≈ 1.41
```

Fold_0 = 3.85 i fold_5 = −1.76 są **nieodróżnialne** od siebie i od zera na poziomie 1σ. Warunek „≥2 z 3 ostatnich foldów > 0.5" ma moc statystyczną bliską rzutowi monetą. Natomiast:

```
SE(IC na przekroju) ≈ 1/√(N_nazw − 1)
  34 nazwy  → 0.175 ;  uśrednione po 144 niezależnych oknach → SE(mean IC) ≈ 0.015
  500 nazw  → 0.045 ;  → SE(mean IC) ≈ 0.004
```

**IC/ICIR jest metryką o rząd wielkości mocniejszą niż Sharpe** przy tej długości próby. To musi być podstawa bramki. Zadanie **T1-3**.

---

## 3. Backlog

### Legenda wagi
`KRYT` — blokuje kolejny trening · `WYS` — istotny wpływ na wynik · `ŚRE` — dług, do zaplanowania · `NIS` — kosmetyka

---

## TIER 0 — przed jakimkolwiek kolejnym treningiem

### T0-1 · Kontrakt danych treningowych (`assert_training_data_contract`) · KRYT
**Serwis:** `ml-pipeline` · **Pliki:** `services/ml-pipeline/src/core/dataset/contract.py` (nowy), wywołanie w `dataset_builder.py`

**Problem.** Dwa najgorsze defekty z historii (§11 briefu: obcięty cache 250 vs 2000 świec; §4: stałe zera w cechach makro) zostałyby wykryte przez **jedno** asercyjne przejście po zbiorze. Zielone testy jednostkowe nie mówiły nic o kształcie danych.

**Zmiana.** Przed każdym treningiem uruchom twardy kontrakt; wynik zaloguj do MLflow jako artefakt `data_contract.json`.

```python
# Kontrakt (pseudokod — implementuj jako dataclass + walidator)
class TrainingDataContract:
    min_sessions: int = 1000
    min_symbols_per_session: int = 20      # patrz T0-6
    max_missing_rate_per_feature: float = 0.10
    min_feature_variance: float = 1e-6      # ZABIJA stałe kolumny
    expected_session_count_tolerance: float = 0.02   # żądane vs otrzymane
    label_resolution_report: bool = True    # udział barier: upper/lower/vertical
```

**Kryteria akceptacji**
- [ ] `test_contract_rejects_constant_feature` — kolumna stałych zer podnosi `TrainingDataContractError`.
- [ ] `test_contract_rejects_truncated_history` — żądanie 2000 sesji, zwrot 250 → błąd, nie ciche przejście.
- [ ] `test_contract_rejects_thin_cross_section` — sesja z < `min_symbols_per_session` odrzucona z raportem.
- [ ] Trening kończy się `exit != 0` przy naruszeniu kontraktu; artefakt `data_contract.json` obecny w MLflow również przy porażce.
- [ ] Raport zawiera rozkład rozstrzygnięć etykiet (`upper` / `lower` / `vertical` / `unlabeled`).

---

### T0-2 · Usuń stałe cechy makro z listy cech modelu · KRYT
**Serwis:** `ml-pipeline` + `trading-common` · **Pliki:** `shared/trading-common/src/trading_common/features/registry.py`, `services/ml-pipeline/src/core/dataset/*`

**Problem (F5).** `macro_expansion|recovery|slowdown|contraction|crisis` są zerami w treningu i jedynkami w serwowaniu. To nie jest luka — to rozjazd trening/serwowanie, który podaje modelowi wejście spoza rozkładu treningowego. Dopóki nie ma historii reżimów (T2-2), kolumny **muszą zniknąć z listy cech**, a nie być zerowane.

**Zmiana.** Lista cech modelu budowana dynamicznie; cecha bez wariancji w zbiorze treningowym jest usuwana z `model_metadata.feature_order` i logowana jako `dropped_zero_variance`.

**Kryteria akceptacji**
- [ ] `test_zero_variance_features_excluded_from_metadata`
- [ ] Serwowanie: jeśli przychodząca cecha nie jest w `feature_order`, jest ignorowana (już tak jest) **i liczona metryką** `ml_serving_unused_feature_total{feature=...}`.
- [ ] Serwowanie: jeśli cecha z `feature_order` jest nieobecna → jak dotąd (imputacja/odmowa), bez zmian.
- [ ] Nowy model po treningu ma `n_features = 8`, nie 13 (7 technicznych po usunięciu duplikatu + `rsi_14`… policz faktycznie i zapisz w metadanych).

---

### T0-3 · Diagnostyka „niedouczenie vs brak sygnału" · KRYT
**Serwis:** `ml-pipeline` · **Pliki:** `services/ml-pipeline/src/core/training/report.py`

**Problem.** Otwarte pytanie z §9 briefu. Sam AUC treningowy nie wystarczy — kalibracja temperaturowa maskuje obraz.

**Zmiana.** Raport treningowy raportuje obowiązkowo:

| Pole | Znaczenie |
|---|---|
| `auc_train`, `auc_val`, `auc_test`, `auc_holdout` | rozdziel dopasowanie od generalizacji |
| `pred_std_pre_calibration`, `pred_std_post_calibration` | rozdziel zapadnięcie modelu od spłaszczenia przez kalibrację |
| `calibration_temperature` | `T ≫ 1` ⇒ model bez sygnału, kalibracja poprawnie go wygasza |
| `loss_train_final`, `loss_val_final`, `epochs_run`, `early_stop_reason` | czy trening w ogóle się nauczył |
| `n_effective_samples` | `n_sessions / horizon × n_symbols_effective` |

**Reguła interpretacyjna (wpisz do docstringa):**
```
auc_train ≈ 0.5  → problem optymalizacji (pojemność / lr / epoki / skalowanie)
auc_train > 0.6 AND auc_val ≈ 0.5  → przeuczenie, brak sygnału w cechach
auc_train ≈ auc_val ≈ 0.5 AND T >> 1  → brak sygnału, model uczciwie się poddał
```

**Kryteria akceptacji**
- [ ] Wszystkie pola obecne w `TrainingReport` (Pydantic) i w MLflow.
- [ ] `test_report_contains_diagnostic_fields`
- [ ] Uruchomiony na danych z 2026-07-26 (odtworzenie) — wpisz wynik w `docs/ml-runs/2026-07-26-rerun.md`.

---

### T0-4 · Zgodność horyzontu oceny z horyzontem etykiety · KRYT
**Serwis:** `ml-pipeline` (+ `backtest` dla spójności) · **Pliki:** `services/ml-pipeline/src/core/evaluation/portfolio.py`

**Problem (F3).** Etykieta ma horyzont 10 sesji, portfel oceny rebalansuje się codziennie. To dwa różne obiekty, a różnicę płacisz w kosztach (~12.6%/rok).

**Zmiana.** Portfel oceny jako **nakładające się transze** (Jegadeesh–Titman): każdego dnia rebalansujemy `1/h` kapitału, pozycja żyje `h` sesji. Obrót spada ~10×.

```
dla h = 10:
  10 równoległych transz po 10% kapitału
  dzień t: transza (t mod 10) jest odnawiana na podstawie predykcji z dnia t
  obrót dzienny ≈ 10% zamiast ~100%
  koszt roczny ≈ 1.3% zamiast ~12.6%
```

**Kryteria akceptacji**
- [ ] `test_overlapping_portfolio_turnover` — obrót dzienny ≤ `1/h + tolerancja`.
- [ ] Raport podaje `turnover_daily_mean`, `sharpe_gross`, `sharpe_net`, `cost_drag_annualized`.
- [ ] `DECYZJA:` czy `backtest` (silnik long/flat) ma używać tej samej konwencji — porównywalność wymaga tak.

---

### T0-5 · Metryki relatywne i obrót w raporcie · KRYT
**Serwis:** `ml-pipeline` · **Pliki:** `services/ml-pipeline/src/core/evaluation/metrics.py`

**Problem.** Raport nie zawiera **żadnego benchmarku**. Sharpe 3.85 przy `base_rate` 0.677 to rynek, co sam zauważyłeś — ale system tego nie mierzy, mierzy to człowiek patrząc na `base_rate`.

**Zmiana.** Dla każdego foldu i holdoutu policz:

| Metryka | Definicja |
|---|---|
| `sharpe_ew_universe` | równoważony portfel **całego** uniwersum = benchmark |
| `sharpe_active` | Sharpe szeregu (portfel − benchmark) |
| `sharpe_long_short` | top-kwintyl − dolny-kwintyl (odporny na `base_rate` z definicji) |
| `ic_mean`, `ic_std`, `icir` | Spearman(pred, forward_return_h) po przekrojach |
| `turnover_daily_mean` | patrz T0-4 |
| `sharpe_gross`, `sharpe_net` | przed/po kosztach |
| `baseline_momentum_rank_ic` | IC surowej rangi `return_20d` jako predyktora |
| `baseline_logreg_ic` | IC regresji logistycznej na tych samych cechach |

**Zasada, którą utrwal w kodzie:** *model, który nie bije surowej rangi jednej cechy, nie zasługuje na warstwę ML.*

**Kryteria akceptacji**
- [ ] `test_metrics_include_baselines`
- [ ] `test_long_short_sharpe_insensitive_to_base_rate` — syntetyczny scenariusz „wszystko rośnie" daje `sharpe_long_short ≈ 0`, a `sharpe` (long-only) wysokie.
- [ ] Raport odrzuca się jako niekompletny bez `icir` i `turnover`.

---

### T0-6 · `min_universe`: 2 → 20 · WYS
**Serwis:** `ml-pipeline` · **Pliki:** konfiguracja budowy zbioru

**Problem.** Przy 2 symbolach ranga przekrojowa to zbiór `{0, 1}` — czysty szum trafiający do wspólnej puli treningowej. Do sensownego kwintyla potrzeba ≥ 4 nazw na kwintyl, czyli ≥ 20 nazw.

**Kryteria akceptacji**
- [ ] `min_universe = 20` (parametr, nie stała).
- [ ] Kontrakt T0-1 raportuje liczbę sesji pominiętych z tego powodu.
- [ ] `test_thin_cross_section_skipped`

---

### T0-7 · Usuń duplikat `momentum_20` · NIS
**Serwis:** `trading-common` · **Pliki:** `shared/trading-common/src/trading_common/features/`

`momentum_20` jest identyczne z `return_20d`. Zostaw jedną nazwę (proponuję `return_20d`, `momentum_20` jako alias deprecated na jeden cykl). Strategia używa `momentum_20` — patrz T1-4, i tak zmienia definicję.

**Kryteria akceptacji**
- [ ] `test_no_duplicate_features_in_registry` — porównanie kolumnami, nie nazwami.

---

## TIER 1 — największa oczekiwana wartość (tygodnie 1–3)

### T1-1 · Rozszerzenie uniwersum do 200–500 nazw, point-in-time · WYS · **NAJWAŻNIEJSZE**
**Serwis:** `market-data` + `company-classifier` · **Pliki:** `services/market-data/src/core/universe/`

**Uzasadnienie (F7).** Prawo fundamentalne: `IR ≈ IC·√BR`. Przy `IC = 0.03`:
```
N = 34,  h = 10 → BR ≈ 34 × 25 = 850     → IR_nominalne ≈ 0.88 (realnie ~0.3 po haircutach)
N = 500, h = 10 → BR ≈ 500 × 25 = 12600  → IR_nominalne ≈ 3.4 (realnie ~1.0–1.5)
```
Mnożnik na samą wykrywalność sygnału: `√(500/34) ≈ 3.8×`. Żadna inna zmiana w tym backlogu nie daje takiego przyrostu, a koszt to czas pobrania danych z yfinance.

**⚠️ PUŁAPKA — survivorship bias.** Jeśli weźmiesz **dzisiejszy** skład S&P 500 i pobierzesz 6 lat historii, dostaniesz zbiór samych ocalałych. Efekt: zawyżony `base_rate`, zawyżony backtest, model uczy się „kupuj to, co przetrwało". To jest błąd, który sam z siebie potrafi wygenerować fałszywą przewagę.

**Zmiana.**
1. Tabela `universe_membership(symbol, valid_from, valid_to, index_name)` w TimescaleDB.
2. Źródło składu point-in-time: historia zmian S&P 500 (publicznie dostępna lista dodań/usunięć) → rekonstrukcja składu na każdą datę. Alternatywnie: własna lista + jawne udokumentowanie biasu.
3. Budowa zbioru pyta o skład **na datę sesji**, nie o skład dzisiejszy.
4. Delistingi: zwroty do dnia delistingu, potem symbol wypada z przekroju (nie jest usuwany wstecz).

**Kryteria akceptacji**
- [ ] `test_universe_is_point_in_time` — zapytanie o 2021-03-01 nie zwraca spółki dodanej do indeksu w 2024.
- [ ] `test_delisted_symbol_present_before_removal_absent_after`
- [ ] Raport treningowy loguje `universe_source`, `n_symbols_median`, `n_symbols_min/max`, `survivorship_bias_controlled: bool`.
- [ ] `DECYZJA:` docelowy rozmiar uniwersum (rekomendacja: **S&P 500 point-in-time**; minimum akceptowalne: 200 nazw z rozrzutem po 11 sektorach GICS).

---

### T1-2 · Rozszerzenie historii do 2005+ · WYS
**Serwis:** `market-data`

**Problem.** Okno 2020-07 → 2026-07 zaczyna się tuż po krachu COVID i zawiera głównie hossę (`base_rate` do 0.677). Okno treningowe 756 sesji = 3 lata, więc model widzi najwyżej jedno przejście reżimu na fold. Nie ma w tych danych ani jednego pełnego cyklu kredytowego.

**Zmiana.** Backfill do **2005-01** (yfinance daje to za darmo). Zysk: 2008, 2011, 2015-16, 2018, 2020 — pięć reżimów stresowych zamiast jednego.

**Kryteria akceptacji**
- [ ] ≥ 5000 sesji na symbol dla nazw notowanych przez cały okres.
- [ ] Walidacja jakości: brak luk > 5 sesji poza świętami, brak zwrotów |r| > 50% bez odpowiadającego splitu/dywidendy.
- [ ] Ceny **skorygowane** o splity i dywidendy — `test_adjusted_close_consistency` na znanym splicie (np. AAPL 2020-08-31 4:1).
- [ ] Kontrakt T0-1 zaktualizowany: `min_sessions = 3000`.

---

### T1-3 · Nowa bramka aktywacji · WYS
**Serwis:** `ml-pipeline` · **Pliki:** `services/ml-pipeline/src/core/gating/activation_gate.py`

**Problem (F1).** Obecna bramka: `holdout Sharpe > 0.5 AND ≥2/3 ostatnich foldów AND Brier ≤ baseline + 0.01`. Warunek foldowy przeszedł mimo `lift ≈ 0` — sam to zauważyłeś. Przyczyna jest głębsza niż „lift powinien być twardy": **Sharpe na 63–126 sesjach nie ma mocy statystycznej** (SE ≈ 1.4–2.0).

**Nowa specyfikacja bramki.** Wszystkie warunki muszą być spełnione, kolejność = kolejność sprawdzania (fail-fast):

```
G0  SANITY
    pred_std_pre_calibration      > 0.02
    calibration_temperature       < 5.0
    n_effective_samples / n_features > 50
    data_contract.passed          == True

G1  SYGNAŁ (przekrojowy, wysoka moc)
    icir_folds                    > 0.30          # mean(IC)/std(IC) po foldach
    ic_mean_holdout               > 0.015
    liczba foldów z IC > 0        >= 7 z 8        # test dwumianowy p < 0.05

G2  PRZEWAGA NAD BASELINE (relatywna, nie absolutna)
    ic_mean_model                 > ic_baseline_momentum_rank + 0.005
    ic_mean_model                 > ic_baseline_logreg
    sharpe_long_short_holdout     > 0.0
    sharpe_active_holdout         > 0.0           # vs equal-weight universe

G3  EKONOMIA
    sharpe_net_holdout            > 0.5
    sharpe_net / sharpe_gross     > 0.5           # koszt nie zjada > połowy
    turnover_daily_mean           < 0.25

G4  KALIBRACJA
    brier                         <= brier_baseline + 0.01

G5  WIELOKROTNE TESTOWANIE
    deflated_sharpe_ratio(n_trials) > 0.0         # Bailey & López de Prado
    n_trials pobierane z licznika w MLflow
```

**Uwagi projektowe.**
- Sharpe **zostaje**, ale jako warunek ekonomiczny w G3, nie jako główny dowód na sygnał. Dowodem jest IC/ICIR (G1) i przewaga nad baseline (G2).
- `lift` przestaje być potrzebny jako osobny warunek — `IC` to jego ciągła, mocniejsza wersja korzystająca z całego przekroju, a nie tylko z kwintyla.
- **DSR wymaga licznika prób.** Bez niego po 20 podejściach znajdziesz fałszywą strategię przy 5% istotności (to dosłownie liczba z Twoich materiałów źródłowych). Zaimplementuj `n_trials` jako trwały licznik w MLflow, inkrementowany przy każdym uruchomieniu treningu tej rodziny modeli — **także przy porażce**.

**Kryteria akceptacji**
- [ ] `test_gate_rejects_collapsed_model` — `pred_std = 0.007` odrzucone na G0, bez liczenia Sharpe'a.
- [ ] `test_gate_rejects_bull_market_luck` — syntetyczny scenariusz: wszystkie akcje rosną, model losowy → wysokie `sharpe`, ale `sharpe_long_short ≈ 0` → odrzucony na G2.
- [ ] `test_gate_rejects_model_not_beating_single_feature_baseline`
- [ ] `test_gate_accepts_synthetic_signal` — wstrzyknięty sztuczny sygnał o IC ≈ 0.05 przechodzi wszystkie bramki. **Bez tego testu nie wiesz, czy bramka jest w ogóle przechodzalna.**
- [ ] Raport bramki podaje, na którym warunku i z jaką wartością nastąpiło odrzucenie.

---

### T1-4 · Cechy o horyzoncie długim + poprawna definicja momentum · WYS
**Serwis:** `trading-common` (definicje) + `feature-engine` + `ml-pipeline`

**Problem (F6).** Najdłuższa cecha to 20 sesji. Kanoniczne, najlepiej udokumentowane momentum to **12-1 miesięcy**, które *pomija* ostatni miesiąc, bo ostatni miesiąc niesie odwrotny efekt (short-term reversal).

**Dodaj do `trading_common.features`:**

| Cecha | Definicja | Uzasadnienie |
|---|---|---|
| `momentum_12_1` | `close[t-21]/close[t-252] - 1` | Jegadeesh & Titman — najsilniejsza anomalia |
| `momentum_6_1` | `close[t-21]/close[t-126] - 1` | krótszy wariant, częściowo niezależny |
| `momentum_12_1_vol_scaled` | `momentum_12_1 / realized_vol_60` | Barroso & Santa-Clara — redukuje crash risk momentum |
| `reversal_1m` | `return_20d` **z oczekiwanym znakiem ujemnym** | osobna cecha, nie mieszana z momentum |
| `dist_52w_high` | `close / max(close, 252) - 1` | proxy momentum odporny na outliery |
| `vol_ratio_20_60` | `realized_vol_20 / realized_vol_60` | reżim zmienności per spółka |
| `beta_60` | rolling beta do SPY, 60 sesji | umożliwia neutralizację (T2-5) |

**Zmiana w regule strategii (§5 briefu).**
```
PRZED:  BUY ← ranga(momentum_20) ≥ 0.80 AND rsi_14 < 70
PO:     BUY ← ranga(momentum_12_1_vol_scaled) ≥ 0.80 AND ranga(reversal_1m) ≤ 0.80
                                                      # nie kupuj tego, co właśnie odjechało
```
`DECYZJA:` czy zostawić filtr RSI. Rekomendacja: zostaw jako filtr higieniczny, ale **usuń z opisu jako źródło alfy** i dodaj do backlogu A/B test „z RSI vs bez".

**Kryteria akceptacji**
- [ ] Definicje w `trading-common`, wspólny kod trening/serwowanie (jak dotąd — to działa dobrze).
- [ ] `test_momentum_12_1_skips_recent_month` — na syntetycznej serii z odjazdem w ostatnim miesiącu cecha nie reaguje.
- [ ] `test_feature_correlation_matrix` — raport z macierzą korelacji Spearmana między rangami cech; ostrzeżenie przy |ρ| > 0.9.
- [ ] Wymóg `min_history` podniesiony do `252 + 21 = 273` sesji dla nazw z `momentum_12_1` (albo imputacja `0.5` — **jawnie zdecyduj i udokumentuj**).

---

### T1-5 · Przeciek point-in-time w danych fundamentalnych · **KRYT (weryfikacja)**
**Serwis:** `fundamental-data`

**Pytanie do rozstrzygnięcia natychmiast:** F-Score i dane ze sprawozdań są przypisywane do daty **końca okresu sprawozdawczego** czy do **daty publikacji (filing date)**?

Jeśli do końca okresu — masz **look-ahead bias rzędu 45–90 dni** i każdy backtest z Tier-2 będzie zawyżony. To najczęstszy i najbardziej niszczący błąd w systemach fundamentalnych.

**Zmiana (jeśli potrzebna).**
- `financial_statements` przechowuje **oba** znaczniki: `period_end` i `filed_at` (EDGAR podaje `acceptanceDateTime`).
- Zapytanie „fundamenty na dzień D" zwraca najnowszy raport z `filed_at <= D`, nigdy `period_end <= D`.
- Restatementy: przechowuj wersjonowanie (`accession_number`), zapytanie point-in-time zwraca wersję **znaną w dniu D**, nie skorygowaną później.

**Kryteria akceptacji**
- [ ] `test_fundamentals_as_of_uses_filing_date` — raport za Q4 z `period_end=2024-12-31`, `filed_at=2025-02-20` nie jest widoczny dla zapytania o 2025-01-15.
- [ ] `test_restatement_not_visible_before_publication`
- [ ] Kolumna `filed_at` NOT NULL, indeks na `(symbol, filed_at)`.

---

## TIER 2 — pogłębienie (tygodnie 3–6)

### T2-1 · Tier-2 (fundamenty) do zbioru treningowego · WYS
**Zależność:** T1-5 musi być zamknięte. Bez point-in-time to zadanie **pogorszy** system, dając fałszywą przewagę.

Cechy z listy Tier-1 własnych materiałów projektu, w kolejności udokumentowanej siły:
1. `ev_ebitda` (ranga przekrojowa, odwrócona — niskie = dobre)
2. `gross_profits_to_assets` (Novy-Marx & Medhat — subsumuje jakość)
3. `f_score` (Piotroski, już liczony)
4. `shareholder_yield`
5. `accruals_ratio` (jakość zysków, filtr negatywny)

**Kryteria akceptacji**
- [ ] Te same cechy w treningu i serwowaniu (wspólny kod `trading_common`).
- [ ] `test_no_tier2_feature_in_serving_absent_from_training` — asercja symetrii, **trwały strażnik przeciw powtórce L1/L2**.
- [ ] Raport pokrycia: `%` symbolo-sesji z niepustym `f_score` itd. Poniżej 70% pokrycia — cecha nie wchodzi.

---

### T2-2 · Historia reżimów makro · ŚRE
**Serwis:** `macro-data` → `ml-pipeline`

Backfill FRED od 2005, przeliczenie reguły reżimu **wstecz** i zapis szeregu `macro_regime(date)` do TimescaleDB. Dopiero wtedy cechy makro wracają do listy (odwrócenie T0-2).

**⚠️ Uwaga.** FRED publikuje z opóźnieniem i **rewiduje** dane (PMI, zatrudnienie). Reżim liczony z dzisiejszej wersji szeregu to look-ahead. Użyj serii ALFRED (vintage) albo zastosuj stałe opóźnienie publikacyjne (np. +30 dni dla danych miesięcznych) i to udokumentuj.

**Kryteria akceptacji**
- [ ] `test_macro_regime_uses_vintage_or_lag`
- [ ] Rozkład reżimów w historii 2005+ — żaden reżim < 3% obserwacji (inaczej one-hot jest bezużyteczny).
- [ ] `DECYZJA:` reżim jako cecha wejściowa modelu **czy** jako zmienna warunkująca (osobne modele per reżim / interakcje). Rekomendacja na start: cecha wejściowa + interakcja z `momentum_12_1`.

---

### T2-3 · LightGBM jako challenger · ŚRE
Własne materiały projektu są tu jednoznaczne: gradient boosting > sieci na tabelarycznych danych finansowych (Grinsztajn et al., NeurIPS 2022), a MLP (32,16) na 6 realnych cechach i efektywnym N ≈ 850 to wybór trudny do obrony.

**Zmiana.** `model_type` jako parametr; `lightgbm` i `logreg` obok `mlp`. Wszystkie trzy trenowane w tym samym biegu, raport porównawczy, bramka stosowana do najlepszego wg `icir` na foldach (nie na holdoucie!).

**Kryteria akceptacji**
- [ ] `test_model_registry_supports_multiple_types`
- [ ] Wybór championa **nigdy** nie dotyka holdoutu — asercja w kodzie, nie tylko w dokumentacji.
- [ ] MLflow: jeden `run` nadrzędny, trzy `nested runs`.

---

### T2-4 · Rekalibracja barier triple-barrier · ŚRE
**Zależność:** raport rozstrzygnięć etykiet z T0-1.

```
pt_mult = sl_mult: 2.0 → 1.0   (docelowo ~50–60% rozstrzygnięć na barierach poziomych)
horizon: DECYZJA 10 vs 21 sesji
```
Argument za `h = 21`: zgodność z horyzontem czynników fundamentalnych (1–3 mies. wg własnych materiałów), niższy obrót, mniejszy udział kosztów. Argument za `h = 10`: więcej niezależnych okien. Rekomendacja: **przetestuj oba jako osobne modele**, ale traktuj to jako 2 próby w liczniku DSR.

**Kryteria akceptacji**
- [ ] Raport `label_resolution` pokazuje `upper + lower ≥ 0.45` po zmianie.
- [ ] `test_barrier_resolution_distribution` na syntetycznym GBM o znanej σ.

---

### T2-5 · Neutralizacja rang względem sektora i bety · ŚRE
Ranga przekrojowa po całym uniwersum miesza sygnał selekcji z ekspozycją sektorową. Przy 500 nazwach to zaczyna dominować.

**Zmiana.** Opcjonalny tryb `rank_within_sector` + demeaning względem `beta_60`. Sektor już masz z `company-classifier`.

**Kryteria akceptacji**
- [ ] `test_sector_neutral_ranks_sum_to_uniform_within_sector`
- [ ] Raport: IC przed i po neutralizacji — jeśli IC spada do zera po neutralizacji, sygnał był ekspozycją sektorową, nie selekcją. **To jest kluczowa diagnostyka.**

---

### T2-6 · Meta-labeling — rekomendowana zmiana architektury warstwy ML · WYS
**Serwis:** `ml-pipeline` + `signal-aggregator` · **Podnoszę z L10 (niska) do priorytetu architektonicznego.**

**Uzasadnienie.** Obecny projekt ma ML i strategię jako **równoległych głosujących** z adaptacyjnymi wagami i bezwymiarowym progiem 0.2 porównywanym z kosztem w bps (patrz FLOW-2 — to niespójność jednostek). Meta-labeling rozwiązuje to strukturalnie:

```
OBECNIE:
  strategy  → sygnał (conf = ranga)      ┐
  ml        → sygnał (conf = p)          ├→ głosowanie ważone → próg 0.2 → decyzja
  macro     → sygnał (conf = ?)          ┘

META-LABELING:
  strategy  → STRONA zakładu (long/flat)  [model pierwotny, optymalizuj recall]
  ml        → CZY WCHODZIĆ + JAK DUŻO     [meta-model: P(ten konkretny sygnał zarobi)]
                                           [optymalizuj precision]
  macro     → wyłącznie limity ekspozycji w risk-mgmt (patrz FLOW-3)
```

**Co to naprawia jednocześnie:**
- Invariant „ML nigdy nie handluje samodzielnie" staje się **strukturalny**, nie regulaminowy.
- Wyjście ML ma bezpośrednią interpretację ekonomiczną → filtr kosztów przestaje mieszać jednostki (FLOW-2).
- Zbiór treningowy meta-modelu jest mniejszy, ale **znacznie mniej zaszumiony** (uczy się tylko na zdarzeniach, gdzie strategia coś powiedziała).
- Znika problem wyścigu strategia/ML w agregatorze (FLOW-1) — zależność jest sekwencyjna z definicji.
- `p` meta-modelu naturalnie mapuje się na wielkość pozycji.

**Kryteria akceptacji**
- [ ] `DECYZJA:` akceptacja zmiany architektury — to zmiana kontraktu `signal.aggregated`, wymaga ADR.
- [ ] ADR w `docs/adr/00XX-meta-labeling.md`.
- [ ] Etykieta meta-modelu: „czy sygnał strategii z dnia t zakończył się dotknięciem bariery górnej" — **ta sama reguła triple-barrier co dotąd**.

---

## TIER 3 — dojrzałość (tygodnie 6–12)

| ID | Zadanie | Waga | Notatka |
|---|---|---|---|
| T3-1 | **CPCV + PBO + DSR** zamiast pojedynczej ścieżki walk-forward | ŚRE | Złoty standard wg własnych materiałów. Rób **po** T1, nie przed — CPCV na modelu bez sygnału da tylko ładniejszy rozkład zera. `skfolio.CombinatorialPurgedCV`. |
| T3-2 | **σ-skalowany SL/TP** (L5) | WYS | Patrz FLOW-5 — to nie tylko niespójność, to zepsuta pętla uczenia. |
| T3-3 | **Zatrzaskowy wyłącznik** (L8) + wyjątek dla zleceń likwidacyjnych | WYS | Patrz FLOW-6. |
| T3-4 | **Event sourcing stanu portfela** (L7) | ŚRE | Patrz FLOW-4. |
| T3-5 | **Konsumenci pull / queue group** (L6) | ŚRE | Blokuje skalowanie replik; dziś nieodczuwalne. |
| T3-6 | **Fractional differentiation** cech cenowych | NIS | `d < 0.2` zwykle wystarcza (ADF). Realna wartość dopiero po T1-4. |
| T3-7 | **Sentyment FinBERT** | ŚRE | Wg materiałów ~29% atrybucji SHAP. Ale **dopiero po** przejściu bramki przez model na cenach+fundamentach. |
| T3-8 | **`llm-svc`** | NIS | `DECYZJA:` **rekomenduję odłożyć.** Dopóki warstwa ML nie przeszła bramki, LLM nie ma czego wzmacniać. Wyjątek: LLM jako narzędzie deweloperskie (poza ścieżką decyzyjną) — to bez zastrzeżeń. |

---

## 4. Luki w logice przepływu (nie-ML)

### FLOW-1 · Wyścig strategia/ML w agregatorze · WYS
**Serwis:** `signal-aggregator`

`features.ready` rozgałęzia się równolegle do `strategy` i `ml-pipeline`. Obie ścieżki produkują komponenty asynchronicznie. Regulamin mówi: „ML nigdy nie agreguje samotnie". **Nie ma reguły symetrycznej: „strategy nigdy nie agreguje bez oczekiwania na ML".**

Konsekwencja: jeśli `strategy` jest szybsza (a jest — reguła vs inferencja), agregator może wyemitować `signal.aggregated` zanim głos ML dotrze, a renormalizacja wag („brak źródła jest darmowy") sprawi, że **waga strategii cicho skoczy do 1.0**. Głos ML nigdy nie zmodyfikuje decyzji, a nikt tego nie zauważy, bo system nie mierzy składu komponentów.

**Zmiana.** Join po korelacji zamiast bufora czasowego:
```
features.ready         → niesie feature_set_id
signal.generated       → echo feature_set_id
ml.signal_generated    → echo feature_set_id
signal-aggregator      → join po feature_set_id z timeoutem Δ (konfigurowalny, start 5 s)
                       → emituj DOKŁADNIE RAZ na (symbol, session)
                       → signal.aggregated niesie components_present: list[str]
```

**Kryteria akceptacji**
- [ ] `test_aggregator_waits_for_expected_components_until_timeout`
- [ ] `test_aggregator_emits_exactly_once_per_symbol_session`
- [ ] Metryka `aggregator_components_present_total{component=...}` + `aggregator_join_timeout_total`.
- [ ] Dashboard: histogram liczby komponentów na decyzję. **Jeśli ML uczestniczy w < 90% decyzji, coś jest zepsute.**

---

### FLOW-2 · Filtr kosztów porównuje wielkości o różnych jednostkach · WYS
**Serwis:** `signal-aggregator` (i `strategy`)

Obecnie: wynik ważonego głosowania ∈ [−1, 1] jest porównywany z `2 × (5+5+2 bps)`. **To nie są te same jednostki.** Bezwymiarowa pewność nie jest oczekiwanym zwrotem. Filtr kosztów jest więc albo zawsze przepuszczający, albo blokuje przypadkowo — w zależności od skali wag.

To jest realna przyczyna, dla której L9 („podwójne bramkowanie kosztami") wygląda na nieszkodliwą redundancję: **oba filtry są w praktyce nieaktywne**.

**Zmiana.** Jawne odwzorowanie pewność → oczekiwany zwrot, korzystające z tego, że prawdopodobieństwa są skalibrowane:

```python
# Etykieta = dotknięcie bariery ±pt_mult·σ·√h.
# Dla skalibrowanego p oczekiwany zwrot na horyzoncie h:
expected_return_h = (2 * p - 1) * pt_mult * sigma_daily * sqrt(horizon)

# Filtr kosztów staje się poprawny wymiarowo:
required_edge = 2 * (spread_bps + slippage_bps + impact_bps) / 10_000
take_trade = expected_return_h > required_edge
```

Sprawdzenie na obecnych parametrach: `σ_daily ≈ 1.5%`, `pt_mult = 2`, `h = 10` → bariera ≈ 9.5%. Przy `p = 0.55`: `expected_return ≈ 0.95%` vs wymagane `0.24%`. Filtr przepuszcza z ogromnym zapasem — czyli **dziś nie filtruje nic**.

**Dla komponentu `strategy`** pewność = ranga momentum, która **nie ma żadnej interpretacji zwrotowej**. Albo wyestymuj mapowanie ranga → oczekiwany zwrot z backtestu (tabela kwintyl → historyczny średni zwrot h-dniowy), albo — lepiej — przejdź na meta-labeling (T2-6), gdzie problem znika.

**Kryteria akceptacji**
- [ ] `test_cost_filter_units` — sygnał o `expected_return` poniżej kosztu jest odrzucany; test na konkretnych liczbach, nie na mockach.
- [ ] Metryka `cost_filter_rejected_total` — jeśli przez tydzień wynosi 0, filtr jest martwy.
- [ ] Usuń **jedną** z dwóch warstw filtrowania (L9) — po naprawie jednostek duplikacja przestaje być „konserwatywna", staje się podwójnym liczeniem.

---

### FLOW-3 · Makro liczone podwójnie · WYS
**Serwisy:** `signal-aggregator` + `risk-mgmt`

Reżim makro to **jedna zmienna globalna**, identyczna dla wszystkich symboli danego dnia. Wchodzi:
1. do agregatora jako **komponent głosujący** z wagą do 0.60 (expansion → BUY na wszystkim, contraction → SELL na wszystkim),
2. do `risk-mgmt` jako **limit ekspozycji** (90% / 80% / 60% / 35% / 15%).

To jest to samo przekonanie zastosowane dwa razy. Gorzej: punkt (1) zamienia system **przekrojowy** (selekcja względna, z założenia w dużym stopniu rynkowo-neutralna) w **market-timing** — a adaptacyjne wagi w hossie chętnie podkręcą wagę makro, bo „miało rację". To klasyczna pułapka: nauka market-timingu z 6 lat danych zdominowanych przez jeden reżim.

**Zmiana (rekomendacja).** Usuń makro z głosowania w agregatorze. Zostaw wyłącznie jako sterowanie ekspozycją w `risk-mgmt`, gdzie już jest i gdzie jest właściwe miejsce.

Wariant miękki, jeśli chcesz zachować bias: obniż `cap` wagi makro z 0.60 do **0.15** i wyłącz dla niego adaptację wag.

**Kryteria akceptacji**
- [ ] `DECYZJA:` usunięcie vs ograniczenie.
- [ ] Backtest porównawczy: ta sama strategia z makro w agregatorze i bez. Różnica w Sharpe **i w ekspozycji na SPY** (beta portfela) — to jest metryka, która pokaże, czy makro robi selekcję czy timing.
- [ ] `test_macro_component_absent_from_aggregation` (jeśli wariant twardy).

---

### FLOW-4 · Stan portfela wraca tylko po HTTP · WYS
**Serwisy:** `execution` → `risk-mgmt`

Brief traktuje to jako mocną stronę („sprzężenie zwrotne portfela"). To też **synchroniczne sprzężenie w ścieżce krytycznej i jedyne źródło stanu**. Jeśli wywołanie HTTP padnie, widok portfela w `risk-mgmt` cicho rozjeżdża się z prawdą, a przy L7 (migawki w Redisie) nie ma jak tego uzgodnić. Skutek: błędny sizing i **ślepy wyłącznik bezpieczeństwa**.

Jednocześnie `order.filled` **już jest** na JetStreamie (widać na diagramie w §3).

**Zmiana.**
- `risk-mgmt` subskrybuje `order.filled` i buduje stan z **logu zdarzeń** (źródło prawdy).
- HTTP zostaje jako **okresowa rekoncyliacja** (co N minut), nie jako ścieżka podstawowa.
- Rozjazd > tolerancja → alert `RECONCILIATION_MISMATCH` do `notification` i przejście wyłącznika w YELLOW (nie handlujemy na niepewnym stanie).

To zamyka L7 dla najważniejszego kawałka stanu, bez przepisywania całego systemu na event sourcing.

**Kryteria akceptacji**
- [ ] `test_portfolio_state_rebuilt_from_event_log`
- [ ] `test_reconciliation_mismatch_raises_alert_and_degrades_circuit_breaker`
- [ ] Metryka `portfolio_reconciliation_drift` (wartość bezwzględna różnicy).

---

### FLOW-5 · Pętla uczenia rozlicza wynik hipotetyczny, nie zrealizowany · WYS
**Serwisy:** `ml-pipeline` + `signal-aggregator` + `execution`

Brief §6: dojrzały głos jest rozliczany „tą samą regułą triple-barrier co trening, na świeżej historii". Ale **rzeczywista pozycja była zarządzana procentowym stop-lossem** (L5), a nie barierą ±2σ√h. Więc:

- wynik, który przesuwa **wagi adaptacyjne** w agregatorze, to wynik **hipotetyczny**,
- P&L, który faktycznie zobaczył portfel, jest inny,
- przy niskiej zmienności procentowy SL jest **luźniejszy** niż bariera, przy wysokiej — **ciaśniejszy**; błąd jest więc systematyczny i skorelowany ze zmiennością.

System uczy się z wyników, których nie osiągnął.

**Zmiana.**
1. SL/TP σ-skalowane, z **tymi samymi mnożnikami co etykiety** (`pt_mult·σ·√h`) — to zamyka L5 i jednocześnie uspójnia pętlę.
2. Rozliczaj **oba** wyniki i loguj oba: `outcome_hypothetical` (bariera) i `outcome_realized` (faktyczne wypełnienia). Wagi adaptacyjne karm `outcome_realized`.
3. Metryka rozjazdu `outcome_divergence` — jeśli rośnie, model wykonania nie odpowiada modelowi treningowemu.

**Kryteria akceptacji**
- [ ] `test_stop_loss_scales_with_volatility`
- [ ] `test_adaptive_weights_use_realized_outcome`
- [ ] Kontrakt `TradingSignal` niesie `sigma_daily` użytą do wyliczenia poziomów (audytowalność).

---

### FLOW-6 · Kolejność wyłącznika: BLACK vs RED · WYS
**Serwis:** `risk-mgmt` + `execution`

RED = „wstrzymaj nowe zlecenia do jutra". BLACK = „zamknij wszystkie pozycje". Zamknięcie pozycji **jest zleceniem**. Scenariusz: RED zapala się przy stracie dziennej > 5%, sytuacja pogarsza się, BLACK zapala się przy DD > 15% — i zlecenia likwidacyjne trafiają w blokadę RED.

To błąd, który ujawnia się wyłącznie w najgorszym dniu roku.

**Zmiana.**
- `OrderRequest` niesie `intent: NEW | REDUCE | LIQUIDATE`.
- Blokady RED/YELLOW dotyczą wyłącznie `intent == NEW`.
- `LIQUIDATE` przechodzi zawsze, także przy BLACK.
- Zatrzask (L8): wyjście z BLACK wyłącznie przez jawną akcję człowieka (`POST /circuit-breaker/reset` z powodem, logowane).

**Kryteria akceptacji**
- [ ] `test_liquidation_orders_bypass_red_halt`
- [ ] `test_black_state_persists_after_conditions_improve`
- [ ] `test_black_reset_requires_explicit_call`

---

### FLOW-7 · Monitor degradacji strategii nie ma próby · ŚRE
**Serwis:** `strategy`

`DEACTIVATED: Sharpe < 0 lub PF < 0.8 lub > 30 dni probacji`. Przy limicie pozycji 5% i ekspozycji 80% masz maks. ~16 pozycji; przez 30 dni zamkniesz może 10–30 transakcji. **Sharpe z 30 transakcji to szum** (ten sam problem co F1, w innym miejscu systemu). System będzie dezaktywował strategie losowo.

**Zmiana.**
```
DEACTIVATE dozwolone tylko gdy:
    n_closed_trades >= 50  AND  n_sessions_observed >= 120
W przeciwnym razie maksymalny werdykt to PROBATION (bez limitu 30 dni,
   albo z limitem liczonym w transakcjach, nie w dniach).
```

**Kryteria akceptacji**
- [ ] `test_decay_monitor_requires_minimum_sample_before_deactivation`
- [ ] Raport monitora podaje `n_trades` i przedział ufności Sharpe'a, nie punktową wartość.

---

### FLOW-8 · Sektory GICS jako łańcuchy znaków · NIS
**Serwis:** `company-classifier` → `trading-common`

Nierozpoznany napis blokuje w restrykcyjnych reżimach — konserwatywnie, ale cicho. Literówka w źródle danych = cicha blokada handlu na spółce.

**Zmiana.** `GicsSector` jako `StrEnum` w `trading-common`, walidacja na granicy `company-classifier`, metryka `classifier_unknown_sector_total{raw_value=...}` + alert przy > 0.

---

### FLOW-9 · Kryterium przejścia na kapitał realny · WYS
Brief §1: „30 dni papierowego handlu z dodatnim Sharpe'em". `SE(Sharpe)` przy 30 sesjach ≈ 2.9. To kryterium **nie niesie informacji**.

**Rekomendowane kryterium:**
```
min. 6 miesięcy paper tradingu (126 sesji)
AND >= 100 zamkniętych transakcji
AND Sharpe netto > 0.5
AND tracking error paper vs backtest < 25% (inaczej model wykonania jest zły)
AND zero incydentów rozjazdu rekoncyliacji (FLOW-4)
AND wyłącznik przetestowany w trybie chaos (wymuszony BLACK + reset)
```
Dodatkowo: paper trading z wypełnieniami po 5 bps **nie testuje tego, co zabija systemy** (poślizg, wypełnienia częściowe, zawieszenia notowań, luki otwarcia). Rozważ okres „shadow": zlecenia liczone realnie, porównywane z faktycznymi cenami następnego otwarcia.

---

## 5. Ocena parametrów — tabela zmian

| Parametr | Obecnie | Rekomendacja | Uzasadnienie |
|---|---|---|---|
| `pt_mult` / `sl_mult` | 2.0 / 2.0 | **1.0 / 1.0** | ~85–90% etykiet rozstrzyga bariera pionowa; triple barrier nieaktywny (F2) |
| `horizon` | 10 | 10 lub **21** — zbadaj oba | zgodność z horyzontem czynników fundamentalnych, niższy obrót |
| `min_universe` | 2 | **20** | ranga z 2 nazw to szum |
| `min_history` | 60 | **273** (dla `momentum_12_1`) | cecha 12-1 potrzebuje 252+21 sesji |
| rozmiar uniwersum | 34 | **200–500 point-in-time** | `IR ≈ IC·√BR`; mnożnik wykrywalności ~3.8× (F7) |
| historia | 2020-07+ | **2005-01+** | jeden reżim → pięć; darmowe |
| rebalans portfela oceny | dzienny | **nakładające się transze 1/h** | koszt 12.6%/rok → 1.3%/rok (F3) |
| próg serwowania | `p ≥ 0.55` abs. | **top-kwintyl przekrojowy** + `p > base_rate + margines` | zgodność z metryką oceny (F4) |
| metryka bramki | Sharpe | **ICIR + IC + relatywne** | SE(Sharpe) ≈ 1.4–2.0 przy tych oknach (F1) |
| `holdout_size` | 126 | 252 jeśli historia pozwoli | 6 mies. to za mało na jakąkolwiek konkluzję |
| architektura modelu | MLP (32,16) | **+ LightGBM + logreg jako obowiązkowe baseline'y** | GBDT > NN na tabelarycznych danych finansowych |
| `pos_weight` | włączony | **wyłącz** przy `base_rate` 0.55 | 55/45 nie wymaga ważenia; zaburza kalibrację |
| dropout | 0.3 | 0.1–0.2 przy 6–8 cechach | 0.3 na warstwie 32 jednostek to bardzo dużo |
| cap wagi `macro` w agregatorze | 0.60 | **usuń komponent** lub 0.15 | podwójne liczenie + ukryty market-timing (FLOW-3) |
| SL | procentowy | **σ-skalowany, `pt_mult·σ·√h`** | spójność z etykietami i pętlą uczenia (FLOW-5) |
| kryterium go-live | 30 dni | **126 sesji + 100 transakcji + TE** | 30 dni nie niesie informacji (FLOW-9) |
| `DEACTIVATE` w monitorze | dowolna próba | **≥ 50 transakcji i ≥ 120 sesji** | losowa dezaktywacja (FLOW-7) |
| licznik prób do DSR | brak | **trwały licznik w MLflow** | ~20 backtestów wystarcza na fałszywe odkrycie |

---

## 6. Metryki Prometheus do dodania

```
# ml-pipeline
ml_training_data_contract_violations_total{rule}
ml_feature_zero_variance_total{feature}
ml_label_resolution_ratio{barrier="upper|lower|vertical|none"}
ml_prediction_std                       # gauge, per model version
ml_calibration_temperature              # gauge
ml_ic_rolling{window="20|60"}
ml_serving_imputed_feature_ratio        # histogram
ml_serving_unused_feature_total{feature}

# signal-aggregator
aggregator_components_present_total{component}
aggregator_join_timeout_total
aggregator_decisions_total{n_components}
cost_filter_rejected_total{source}
aggregator_source_weight{source}        # gauge — czy wagi nie uciekają do cap/floor

# risk-mgmt
portfolio_reconciliation_drift          # gauge
circuit_breaker_level                   # gauge 0..3
circuit_breaker_transitions_total{from,to}
risk_gate_rejections_total{rule}

# execution
order_intent_total{intent="NEW|REDUCE|LIQUIDATE"}
fill_slippage_bps                       # histogram
outcome_divergence                      # histogram: hipotetyczny vs zrealizowany

# company-classifier
classifier_unknown_sector_total{raw_value}
```

**Panel „czy system w ogóle myśli"** (jeden dashboard, 6 wykresów): rozkład komponentów na decyzję, `ml_prediction_std` w czasie, `ic_rolling`, wagi źródeł, `cost_filter_rejected_total`, `portfolio_reconciliation_drift`. Jeśli którykolwiek jest płaski przez tydzień — jakaś część systemu jest martwa i tego nie widać w testach.

---

## 7. Roadmapa 12 tygodni

| Tydzień | Zakres | Kamień milowy |
|---|---|---|
| **1** | Tier 0 w całości (T0-1…T0-7) | Kontrakt danych zielony; diagnostyka odpowiada na pytanie „niedouczenie czy brak sygnału"; **rerun treningu na starych danych z pełnym raportem** |
| **2–3** | T1-1, T1-2 (uniwersum + historia) | ≥ 200 nazw point-in-time, ≥ 3000 sesji, survivorship udokumentowany |
| **3** | T1-3, T1-4 (bramka + cechy) | Bramka z testem „przechodzalności"; `momentum_12_1` w rejestrze cech |
| **4** | **Trening #2** — cechy cenowe, szerokie uniwersum | Decyzja: czy w samych cenach jest IC > 0.015. Jeśli nie — nie dokładaj fundamentów, tylko zbadaj dlaczego |
| **5** | T1-5 (point-in-time fundamentów) — **weryfikacja przed użyciem** | Test `filed_at` zielony |
| **6–7** | T2-1, T2-3 (Tier-2 + LightGBM challenger) | Porównanie 3 modeli, IC per cecha |
| **7** | **Trening #3** — pełny zestaw czynników Tier-1 | Decyzja bramki |
| **8** | FLOW-1, FLOW-2, FLOW-3 (agregator) | Join po `feature_set_id`; jednostki filtru kosztów; decyzja o makro |
| **9** | FLOW-4, FLOW-5, FLOW-6 (ryzyko/wykonanie) | Stan z logu zdarzeń; σ-skalowane SL; zatrzask wyłącznika |
| **10** | T2-2, T2-4, T2-5 (makro, bariery, neutralizacja) | Reżimy z vintage; ~50% etykiet na barierach poziomych |
| **11** | T2-6 (meta-labeling) — jeśli ADR zaakceptowany | Nowa architektura warstwy ML |
| **12** | T3-1 (CPCV + PBO + DSR), FLOW-7, FLOW-9 | Rozkład Sharpe'a zamiast punktu; nowe kryterium go-live |

**Bramka między tygodniami 4 a 5.** Jeśli po rozszerzeniu uniwersum, historii i dodaniu momentum 12-1 model nadal ma `IC ≈ 0`, to **nie dokładaj kolejnych warstw danych**. To sygnał, że problem jest w konstrukcji zadania (horyzont, etykieta, uniwersum dużych spółek US = najbardziej efektywny segment rynku), a nie w liczbie cech. Rozważ wtedy zmianę segmentu (mid-cap ma więcej alfy wg materiałów źródłowych) albo horyzontu.

---

## 8. Decyzje do podjęcia przez człowieka (kandydaci na ADR)

| # | Decyzja | Rekomendacja |
|---|---|---|
| D1 | Docelowy rozmiar i źródło uniwersum | S&P 500 point-in-time; minimum 200 nazw |
| D2 | `horizon` 10 vs 21 sesji | zbadaj oba, licz jako 2 próby w DSR |
| D3 | Makro: usunąć z agregatora czy ograniczyć cap do 0.15 | usunąć — należy do warstwy ryzyka |
| D4 | Meta-labeling zamiast równoległego głosowania | tak, po treningu #3 |
| D5 | Filtr RSI w regule strategii | zostaw jako higiena, przestań traktować jako alfę, dodaj A/B |
| D6 | `llm-svc` teraz czy po przejściu bramki | **po** — dziś nie ma czego wzmacniać |
| D7 | Czy `backtest` przechodzi na nakładające się transze | tak, inaczej backtest i ocena ML są nieporównywalne |
| D8 | Reżim makro jako cecha czy jako warunkowanie modelu | cecha + interakcja z momentum na start |

---

## 9. Co w tym systemie jest zrobione dobrze (nie psuj)

Żeby backlog nie zniekształcił obrazu — kilka rzeczy jest zrobionych na poziomie, którego nie ma w większości podobnych projektów:

- **Wspólny kod cech dla treningu i serwowania w `trading-common`.** Zgodność strukturalna zamiast deklaratywnej. To eliminuje całą klasę błędów; L1/L2 to nie porażka tej decyzji, tylko dowód, że warto ją rozszerzyć na cechy Tier-2 i makro.
- **`lift`, `pred_std`, `base_rate` w raporcie.** Bez `base_rate` uwierzyłbyś w „6 z 8 foldów dodatnich". Ten jeden pomysł uratował projekt przed wdrożeniem szumu.
- **Model zapisywany do MLflow także przy porażce, promocja ręczna.** Właściwa asymetria.
- **Odmowa predykcji przy dryfie schematu, odrzucanie niezaetykietowanych wierszy, porzucanie nierozstrzygalnego głosu po 42 dniach.** Konsekwentna odmowa fabrykowania danych.
- **`performance_measured=false` przy < 10 wynikach.** Rzadko spotykana uczciwość w raportowaniu.
- **Strażnik CI porównujący importy z deklaracjami zależności** — dokładnie właściwa reakcja na incydent z `httpx`.
- **Zlecenie bez `stop_loss` jest niemożliwe na poziomie kontraktu.** Wymuszenie w typie, nie w regulaminie.

Wniosek z §11 briefu — „zielone testy nie mówiły nic o tym, czy system da się uruchomić" — jest trafny i uogólnia się dalej: **zielone testy nie mówią też nic o tym, czy dane mają sens**. Zadanie T0-1 jest bezpośrednią odpowiedzią na tę lekcję.

---

## 10. Jednozdaniowa konkluzja

Infrastruktura, kontrakty i dyscyplina raportowania są mocne; **warstwa ML dostała najsłabszy możliwy zestaw wejść (6 realnych cech krótkoterminowych, 34 nazwy, 6 lat jednego reżimu) i jest oceniana metryką bez mocy statystycznej** — dlatego pierwszy trening nie tyle „pokazał brak sygnału", co **nie mógł niczego pokazać**; napraw najpierw pomiar (Tier 0) i szerokość (Tier 1), zanim wyciągniesz wniosek o sygnale.
