# Plan poprawy przewidywania — 2026-07-28

> **Cel:** doprowadzić do stanu, w którym jesteśmy w stanie **zmierzyć** przewagę predykcyjną, a
> potem ją **mieć**. W tej kolejności — bo dokładanie cech do układu, w którym nic nie da się
> odróżnić od zera, produkuje wyniki niefalsyfikowalne.
>
> Wszystkie liczby „stan wyjściowy" poniżej są **zweryfikowane w kodzie lub zmierzone**
> w biegu #2 / próbach generalnych, nie przywołane z pamięci. Tezy z literatury są oznaczone
> jako literatura — nie odtwarzałem ich na naszych danych.

---

## 1. Stan wyjściowy — co model faktycznie dostaje

Zweryfikowane odczytem kodu (`trading_common/features.py`, `ml-pipeline/src/core/dataset.py`,
`market-data/src/core/fetchers/yahoo.py`, `models/db.py`, `infrastructure/init-db.sql`):

| Fakt | Wartość | Konsekwencja |
|---|---|---|
| Cechy wejściowe modelu (P2-1 podniosło do 15 — patrz §5) | **7**: `return_1d`, `return_5d`, `return_20d`, `price_to_sma50`, `rsi_14`, `realized_vol_20`, `volume_ratio` | Cztery pierwsze mierzą **to samo** — niedawny kierunek ceny. Realnie ~2–3 niezależne wymiary informacji, nie 7 |
| Ceny | `auto_adjust=False`, zapisywany wyłącznie surowy `Close`; **`Adj Close` odrzucany**, brak kolumny `adj_close` w kontrakcie, ORM i schemacie | **Dywidendy nie są uwzględnione.** Spółki dywidendowe (PG, KO, XOM, JNJ, CVX, PFE…) mają systematycznie zaniżony zwrot względem niepłacących (NVDA, AMZN, GOOGL, TSLA) — o rząd wielkości porównywalny z poszukiwaną alfą |
| Etykiety | triple barrier `pt=sl=2.0σ`, `h=10`; **90.9% rozstrzyga bariera pionowa** (zmierzone przez pełny pipeline) | Etykieta jest de facto znakiem zwrotu 10-dniowego. Bariery nie pracują |
| Ważenie próbek | brak — każdy wiersz waży tyle samo | Etykieta h=10 przy próbkowaniu dziennym nakłada się z ~9 sąsiednimi. Model widzi 48 827 wierszy, informacji ma **523** |
| Efektywna próba | **523** z 48 827; **3.64** niezależne nazwy z 34; średnia korelacja par 0.253 | To jest podstawa wszystkich testów istotności |
| Okno cech (naprawione 2026-07-28 → 300 z jednej wspólnej stałej) | `lookback=250` w treningu **i** `limit=250` w feature-engine | **Momentum 12-1 i odległość od maksimum 52-tyg. są dziś nieobliczalne** (potrzebują ≥252 sesji). Podniesienie musi nastąpić po OBU stronach, inaczej odtworzymy rozjazd trening/serwowanie |
| Fundamenty | `fundamental-data` liczy pełny 9-sygnałowy F-Score, `feature-engine` wstrzykuje go do `/ranked` | **Trening ich nie widzi** — `build_dataset` liczy cechy z samych świec. Serwowanie składa wiersz w kolejności z metadanych, więc dodatkowe klucze są ignorowane (to nie błąd, ale zmarnowana infrastruktura) |
| Historia fundamentów | `fundamental-data` trzyma **latest-per-symbol w pamięci**; `filed_at` istnieje w kontrakcie, **nikt go nie wypełnia** | Panelu point-in-time nie ma. Bez niego fundamenty w treningu = look-ahead |

### Próg wykrywalności — liczba, która porządkuje cały plan

Zmierzone `ic_std` ≈ 0.26–0.28 na sesję (przy 34 nazwach). Błąd standardowy średniego IC:

```
SE(IC) = ic_std / √T
  holdout   T = 125  →  SE = 0.025  →  wykrywalne przy t=2:  IC ≥ 0.050
  cała OOS  T = 630  →  SE = 0.011  →  wykrywalne przy t=2:  IC ≥ 0.022
```

**Realistyczne IC dla cech cenowych na dużych spółkach to 0.01–0.03** (literatura). Czyli
oczekiwany efekt leży **pod naszym progiem wykrywalności**. Przy 34 nazwach nie odróżnimy
„nie ma przewagi" od „jest przewaga, ale jej nie widzimy" — i żadna liczba nowych cech tego nie
zmieni, bo próg zależy od szerokości przekroju, nie od modelu.

Przy ~300 nazwach szum IC na sesję spada z grubsza jak `1/√N` → próg wykrywalności ≈ **0.007**
(założenie: inflacja szumu przez wspólny czynnik, zmierzona dziś na 1.6×, pozostaje podobna).

### Prawo Grinolda — i pułapka, w którą sam wpadłem licząc to pierwszy raz

`IR ≈ IC · √BR`. Kusi, żeby powiedzieć „34 nazwy → 300 nazw, więc szerokość rośnie 9×". **To jest
nieprawda dla portfela long-only** i wynika to wprost z naszej własnej zmierzonej liczby.
Efektywna liczba niezależnych nazw to `N / (1 + (N−1)·ρ)`, przy średniej korelacji par ρ = **0.253**:

```
N =  34  →  N_eff = 3.64   (zmierzone)
N = 100  →  N_eff = 3.84
N = 300  →  N_eff = 3.91
N → ∞    →  N_eff → 1/ρ = 3.95      ← nasycenie
```

**Dokładanie nazw do książki long-only nie dodaje szerokości.** Przy ρ ≈ 0.25 nasz portfel jest
efektywnie zakładem na ~4 aktywa niezależnie od tego, ile spółek trzyma — bo dominuje w nim
wspólny czynnik rynkowy. To jest dokładnie ta sama rzecz, którą bramka wyłapała jako „Sharpe 0.79
przy benchmarku 1.36": **long-only mierzy głównie rynek**.

Szerokość rośnie dopiero, gdy usuniemy wspólny czynnik — czyli w książce **względem benchmarku
albo long-short**, gdzie liczy się korelacja *reszt* (dla dużych spółek typowo 0.05–0.15):

```
ρ_resid 0.10, N = 300  →  N_eff ≈ 9.7   →  BR ≈ 245/rok (h=10)  →  IR ≈ 0.31 przy IC 0.02
ρ_resid 0.10, N = 300, h = 21                →  BR ≈ 117/rok    →  IR ≈ 0.22 przy IC 0.02
                                                                  →  IR ≈ 0.32 przy IC 0.03
```

Trzy wnioski, które **zmieniają plan**:

1. Uniwersum rozszerzamy **przede wszystkim dla mocy pomiarowej** (próg IC 0.022 → 0.007), a nie
   dla szerokości portfela long-only — ta się nasyca.
2. Jeżeli chcemy, żeby szerokość w ogóle rosła, **oceniana i docelowo handlowana książka musi być
   względna** (active / long-short), nie long-only. `sharpe_active` jest już warunkiem **G3**
   (musi być > 0), ale **metryką, po której bramka ocenia wynik, wciąż jest Sharpe książki
   long-only** — a ta, jak wyżej, mierzy głównie rynek. Nowe zadanie: **P3-4 — przenieść metrykę
   decyzyjną na książkę względną** (long-short albo active), z long-only jako kontekstem.
3. Nawet w optymistycznym wariancie (300 nazw, IC 0.03, książka względna) IR ≈ 0.32 — **poniżej
   reguły „Sharpe > 0.5"**. Reguła nie jest zła; ona po prostu mówi, że przewaga rzędu IC 0.02–0.03
   na dziennym/dwutygodniowym horyzoncie w akcjach dużych spółek **nie wystarcza**, i trzeba albo
   silniejszego sygnału (fundamenty, dłuższy horyzont, dane alternatywne), albo pogodzić się
   z tym, że warstwa ML zostaje modulatorem reguły, a nie samodzielnym źródłem alfy.

---

## 2. Zasada porządkująca

1. **Najpierw to, co uniemożliwia pomiar** (dane, etykiety, ważenie).
2. **Potem to, co czyni cel wyuczalnym** (horyzont, kalibracja barier).
3. **Potem wejścia** (cechy — tu jest alfa).
4. **Potem szerokość** (uniwersum, historia — tu jest moc statystyczna).
5. **Potem klasa modelu** (drzewa, ensembling, CPCV).
6. **Na końcu wejście ze zleceniem** (meta-labeling, koszty, sizing).

Etapy 3 i 4 można prowadzić równolegle; 6 ma sens dopiero, gdy 3–5 dadzą sygnał różny od zera.

---

## 3. Etap 0 — poprawność danych (blokujący)

| ID | Zadanie | Dlaczego to jest pierwsze |
|---|---|---|
| **P0-1** | **Ceny skorygowane o dywidendy.** Contracts-first: `adj_close` w `OHLCVBar`, kolumna w schemacie i ORM, `auto_adjust=True` (albo zapis `Adj Close`); cechy i etykiety liczone na serii skorygowanej, surowy `close` zostaje dla egzekucji | Bez tego ranking przekrojowy ma wbudowany, systematyczny błąd skorelowany z osią value/quality — dokładnie tam, gdzie szukamy sygnału |
| **P0-2** | **Weryfikacja splitów** — jedno zapytanie: czy w dniu splitu (AAPL 2020-08-31 4:1, NVDA 2024-06-10 10:1, AMZN 2022-06-06 20:1) w bazie widnieje zwrot ≈ −75%/−90%/−95% | Yahoo koryguje OHLC o splity, ale **tego nie zakładam** — sprawdzenie kosztuje minutę, a nieskorygowany split to −75% „zwrotu" przebijające dolną barierę |
| **P0-3** | **Ważenie nakładających się etykiet** — wagi z średniej unikalności (López de Prado, AFML rozdz. 4) albo próbkowanie co `h` sesji | Dziś model dostaje 48 827 wierszy jako niezależne, gdy informacji jest 523. To zawyża pewność siebie treningu i zaniża wagę rzadkich reżimów |
| **P0-4** | `label_resolution` i efektywna próba **per fold** w raporcie (dziś tylko globalnie) | Fold z 95% barierą pionową i fold z 55% to dwa różne zadania — dziś wyglądają identycznie |

**Bramka E0:** raport pokazuje udział barier poziomych, wagi unikalności i ESS per fold. Bez tego
nie zaczynamy etapu 1.

---

## 4. Etap 1 — co właściwie przewidujemy

| ID | Zadanie | Uwaga |
|---|---|---|
| **P1-1** | **Kalibracja barier** do ~50–60% rozstrzygnięć poziomych. Zmierzone: `pt=sl=1.0σ` → 46.3% pionowa | Przy 2.0σ etykieta nie niesie informacji o *ścieżce*, tylko o znaku — cała maszyneria triple barrier jest wtedy dekoracją |
| **P1-2** | **Eksperyment horyzontu: 10 vs 21 vs 63 sesje**, oceniany t-statem IC na foldach | Literatura: przekrojowa alfa w akcjach jest najsilniejsza w horyzoncie 1–12 miesięcy; dzienny jest zdominowany przez mikrostrukturę i kosztuje obrotem. To najtańsza dźwignia w obrębie danych, które mamy |
| **P1-3** | **Etykieta zwrotu nadwyżkowego** — barierę liczyć na zwrocie względem mediany przekroju, nie absolutnym | Model przekrojowy ma rankować *względem rynku*; etykieta absolutna każe mu przewidywać betę, czyli to, na czym bramka już raz go przyłapała |

**Decyzja D-horyzont:** wybieramy horyzont **po zmierzonym t-stacie IC**, nie z preferencji.
Jeśli 63 wygra, zmienia się też definicja transz i częstotliwość rebalansu.

---

## 5. Etap 2 — cechy (największa dźwignia w obrębie posiadanych danych)

Rodziny cech z udokumentowaną przewagą przekrojową, których **nie mamy** (literatura w nawiasach):

| Grupa | Cechy | Źródło anomalii |
|---|---|---|
| Momentum | `momentum_12_1` (12 mies. bez ostatniego), `momentum_6_1`, `dist_52w_high` | Jegadeesh–Titman 1993; George–Hwang 2004 |
| Rewersja | `reversal_1m` (osobno od momentum!), `MAX` — maks. dzienny zwrot z miesiąca | Lehmann 1990; Bali i in. 2011 |
| Ryzyko | `beta_60`, `idio_vol_60` (reszta z regresji na indeks), skośność, `downside_vol` | Ang i in. 2006; Frazzini–Pedersen 2014 |
| Płynność | Amihud (|zwrot|/obrót), `dollar_volume`, `turnover` | Amihud 2002 |
| Kontekst | **rangi neutralizowane sektorowo** (ranga wewnątrz sektora GICS) | Sektor to dziś ukryty zakład — przy 34 nazwach z 11 sektorów ranking globalny jest w dużej mierze rankingiem sektorów |
| Fundamenty (P2-3) | F-Score (mamy!), gross profitability, B/M, E/P, wzrost aktywów, accruals | Novy-Marx 2013; Fama–French 2015 |

**P2-1 ✅ zrobione 2026-07-28** — rodzina cenowa (tania, bez nowych źródeł). Wdrożone 8 cech:
`momentum_12_1`, `momentum_6_1`, `dist_52w_high`, `max_ret_1m` (MAX), `downside_vol_20`, `skew_60`,
`amihud_20`, `dollar_volume_20` → wejście modelu 7 → 15 kolumn. Trzy decyzje warte zapamiętania:

- **`reversal_1m` NIE powstał** — to dosłownie `return_20d`, który już mamy. Osobna nazwa dla tej
  samej kolumny odtworzyłaby duplikat `momentum_20`, który T0-7 właśnie usunął z wejścia modelu.
  Wymóg planu „rewersja osobno od momentum" spełnia rozdzielenie w drugą stronę: `momentum_12_1`
  **pomija** ostatni miesiąc, więc obie cechy mierzą teraz rozłączne okna.
- **`beta_60` i `idio_vol_60` odłożone** — wymagają serii rynkowej, a `compute_feature_vector`
  z definicji widzi jeden symbol. Dodanie ich zmienia kontrakt serwowania (feature-engine liczy
  wektor per symbol na `features.ready`), więc to osobne zadanie, nie efekt uboczny tego.
- **Okno podniesione po obu stronach naraz i strukturalnie**: `FEATURE_LOOKBACK = 300`
  i `FULL_HISTORY = 253` żyją teraz w `trading_common.features`, a `DatasetParams` (trening)
  i `Settings.FEATURE_LOOKBACK` (serwowanie) czytają **tę samą stałą**. Dwa niezależne domyślne
  ustawienia to rozjazd czekający na wystąpienie; jedna stała nim nie jest. `min_history` wynika
  z wymagania NAJWOLNIEJSZEJ cechy, więc porusza się razem ze zbiorem cech.

Przy okazji wzmocniona **G2**: komparatorem jest teraz **najlepsza z WSZYSTKICH** surowych cech
(dotąd jedna zadeklarowana, `return_20d`). Wybór maksimum na tym samym oknie jest obciążony w górę,
czyli bramka robi się trudniejsza z każdą dołożoną cechą — to właściwy kierunek błędu.

**P2-2** neutralizacja sektorowa — wymaga mapy sektorów per symbol (statyczna, tania;
`company-classifier` już operuje na `sector`).

**P2-3** **fundamenty point-in-time** — najdroższe i najbardziej wartościowe zadanie w tym etapie:
wypełnić `filed_at` z EDGAR (`acceptanceDateTime`), zapisać **panel historyczny** w Postgresie
(dziś jest tylko latest-per-symbol w pamięci), dołączać as-of daty sesji. Bez `filed_at` to jest
look-ahead, nie cecha.

**Bramka E2:** każda nowa rodzina cech oceniana **osobno** po t-stacie IC na foldach, zanim wejdzie
do zbioru. Cecha, która nie podnosi t-statu, jest wymiarem do przeuczenia, nie informacją.

Narzędzie pomiaru gotowe (P2-1): `evaluation.per_feature_ic` liczy **samodzielne IC i jego t-stat
dla KAŻDEJ surowej cechy**, model-free (nie zużywa `n_trials`), a raport bootstrapu drukuje tabelę
posortowaną po |t| z gwiazdką przy |t| ≥ 2. Czytamy **t, nie poziom IC**: przy 34 nazwach IC 0.02
i −0.02 to ta sama obserwacja widziana dwa razy, dopóki liczba przekrojów ich nie rozróżni.

---

## 6. Etap 3 — uniwersum i historia (moc statystyczna)

| ID | Zadanie | Efekt liczbowy |
|---|---|---|
| **P3-1** | Uniwersum **200–500 nazw point-in-time** (członkostwo w indeksie na datę — survivorship bias!) | Próg wykrywalności IC 0.022 → ~0.007; BR 92 → ~360 |
| **P3-2** | Historia **od 2005** | Obejmuje 2008, 2011, 2015–16, 2018, 2020, 2022 — dziś mamy sześć lat jednego reżimu (hossa mega-capów) |
| **P3-3** | Kontrola jakości: braki, halty, spółki z krótką historią, granice sesji | Rzadkie brzegi to miejsce, gdzie `min_universe=20` zacznie realnie pracować |
| **P3-4** | **Metryka decyzyjna = książka względna** (long-short lub active), long-only jako kontekst | Wynika wprost z nasycenia szerokości przy ρ=0.253: w long-only dokładanie nazw nie dodaje zakładów, a Sharpe mierzy betę. Bez tego rozszerzenie uniwersum poprawi tylko pomiar, nie wynik |

**Uwaga o survivorship:** dzisiejsze 34 nazwy to lista zwycięzców wybrana *ex post*. Model uczony
na takim uniwersum uczy się, że „duże spółki technologiczne rosną" — i to jest jedyna rzecz, którą
w tych danych naprawdę widać.

---

## 7. Etap 4 — klasa modelu

| ID | Zadanie | Uzasadnienie |
|---|---|---|
| **P4-1** | **Challenger GBDT** (LightGBM/XGBoost) obok MLP | Przy tej wielkości próby i tabelarycznych cechach drzewa zwykle wygrywają; sieci wygrywają dopiero przy bardzo szerokim przekroju i wielu charakterystykach (Gu–Kelly–Xiu). Nasza sonda pojemności mierzy to samo pytanie od strony MLP |
| **P4-2** | **Ensembling po ziarnach** (5–10 modeli, uśrednione predykcje) | Zmienność fold-do-foldu w biegu #2 jest częściowo szumem inicjalizacji; uśrednianie to najtańsza redukcja wariancji |
| **P4-3** | Wagi próbek (z P0-3) + neutralizacja sektorowa w funkcji straty | Model przestaje wygrywać na sektorze i na skorelowanych, nakładających się wierszach |
| **P4-4** | **CPCV** — kombinatoryczna walidacja purged (AFML rozdz. 12) | Daje *rozkład* ścieżek OOS zamiast jednej; przy naszej próbie pojedyncza ścieżka walk-forward jest jednym losowaniem |

**Uwaga o `n_trials`:** każda z tych prób podnosi liczbę spojrzeń na te same dane. Sweep już
wypuszcza `n_trials` do bramki (G5); przy CPCV i ensemblingu trzeba to prowadzić dalej —
inaczej deflated Sharpe kłamie na naszą korzyść.

---

## 8. Etap 5 — wejście ze zleceniem

To jest osobne zadanie od rankingu i osobny model. Kanoniczna forma to **meta-labeling**
(López de Prado, AFML rozdz. 3):

```
model podstawowy  →  KTÓRE spółki i w którą stronę        (ranking przekrojowy)
model wtórny      →  CZY wchodzić w ten konkretny sygnał   (P(zysk po kosztach))
                  →  ILE                                   (wielkość z prawdopodobieństwa)
```

Model wtórny uczy się **wyłącznie na sygnałach modelu podstawowego**, a jego etykieta to
„czy ta transakcja zarobiła po kosztach". Cechy modelu wtórnego są inne niż alfa:

- siła i pewność sygnału podstawowego (kalibrowane prawdopodobieństwo, dyspersja przekroju),
- reżim zmienności i szerokość rynku,
- płynność nazwy i szacowany koszt wejścia (spread + impact),
- odległość do stopu w jednostkach σ (nasze SL są dziś procentowe — do zmiany na vol-skalowane),
- dni do publikacji wyników (fundamental-data ma daty),
- świeżość sygnału (ile sesji od zmiany rangi).

| ID | Zadanie |
|---|---|
| **P5-1** | Meta-model: etykieta = „zysk po kosztach", trening tylko na wygenerowanych sygnałach, wyjście = prawdopodobieństwo → veto lub mnożnik wielkości |
| **P5-2** | **Realistyczny model kosztów**: pół spreadu + impact ~ √(udział w wolumenie) zamiast płaskich 5 bps; per nazwa, skalowany zmiennością |
| **P5-3** | Sizing z kalibrowanego prawdopodobieństwa (ułamkowy Kelly ograniczony istniejącą kopertą ryzyka) zamiast stałych 2% ryzyka |
| **P5-4** | Profil zaniku alfy → z niego wynika okres trzymania i pilność wejścia (transze z T0-4 są pierwszym przybliżeniem) |

**Warunek wejścia w etap 5:** model podstawowy ma IC istotnie > 0 (G1 bramki). Meta-labeling
na sygnale bez przewagi tylko filtruje szum — precyzyjniej, ale nadal szum.

---

## 9. Bramki decyzyjne (kiedy przerywamy)

| Po etapie | Warunek kontynuacji | Co robimy, gdy niespełniony |
|---|---|---|
| E0+E1 | udział barier poziomych 40–70%, ESS raportowana per fold | wracamy do definicji etykiety — nie do modelu |
| E2 (cechy cenowe) | t-stat IC rośnie względem 7 cech bazowych **na foldach** | cechy cenowe wyczerpane na tym uniwersum → E3 przed dalszymi cechami |
| E3 (uniwersum) | próg wykrywalności IC < 0.01 osiągnięty | nie ma sensu mierzyć dalej — problem jest w danych, nie w modelu |
| E4 (model) | najlepszy kandydat ma t-stat IC ≥ 2 na foldach | **uczciwa konkluzja: na tych danych i tym horyzoncie nie ma przewagi**; sensowne kierunki to inne dane (intraday, alternatywne) albo inna klasa strategii |
| E5 | bramka G0–G5 przechodzi na holdoucie | promocja tylko do papieru + 30 dni dodatniego Sharpe'a przed kapitałem |

---

## 10. Uczciwe oczekiwania

- **Cechy cenowe na 34 mega-capach w horyzoncie 10 dni: spodziewane IC ≈ 0.00–0.02.** To leży pod
  naszym progiem wykrywalności (0.022). Innymi słowy — nawet gdyby przewaga była, dziś jej **nie
  zobaczymy**. Dlatego etap 3 nie jest opcjonalny.
- Cały zestaw z literatury (Gu–Kelly–Xiu: ~30 000 spółek, 94 charakterystyki, horyzont miesięczny)
  daje Sharpe rzędu 1.2–1.5 na spreadzie decyli **przy uniwersum trzy rzędy wielkości szerszym niż
  nasze**. Skalowanie w dół nie jest liniowe: pierwsze, co znika, to możliwość odróżnienia wyniku
  od zera.
- **Możliwe zakończenie: „nie ma przewagi".** System jest zbudowany tak, żeby móc to powiedzieć —
  bramka G0–G5, DSR, kontrola na przetasowanych etykietach. To jest cecha, nie porażka.

---

## 11. Czego świadomie NIE robimy teraz

- **Nie dokładamy warstw do sieci.** Diagnoza biegu #2 to `auc_train ≈ 0.52` — model nie dopasowuje
  nawet danych treningowych. „Więcej warstw ML" w sensie użytecznym oznacza **drugi etap decyzyjny**
  (meta-labeling), a nie głębszą sieć.
- **Nie robimy kolejnych sweepów na obecnym układzie** — każda próba podnosi `n_trials`, a moc
  statystyczna zostaje bez zmian.
- **Nie promujemy niczego na becie.** Zamknięte przez G3 (active Sharpe) i G5 (DSR).
- **Nie dodajemy sentymentu / LLM / newsów** przed zamknięciem etapów 0–3. To najdroższe źródło
  o najgorszym stosunku sygnału do pracy przy naszej obecnej mocy pomiarowej.

---

## 12. Kolejność wykonania i gdzie co biegnie

| # | Zadania | Nakład | Gdzie |
|---|---|---|---|
| 1 | P0-1 ceny skorygowane, P0-2 kontrola splitów | ~0.5 dnia | kod u mnie; **ponowny backfill u Ciebie** (ceny w bazie są dziś niepełne) |
| 2 | P0-3 wagi unikalności, P0-4 raport per fold | ~0.5 dnia | u mnie, weryfikacja na syntetyku |
| 3 | P1-1 kalibracja barier, P1-3 etykieta nadwyżkowa | ~0.5 dnia | u mnie |
| 4 | P1-2 eksperyment horyzontu (10/21/63) | 1 bieg | **u Ciebie** — to pomiar na realnych danych |
| 5 | P2-1 rodzina cech cenowych + `lookback` po obu stronach, P2-2 neutralizacja sektorowa | ~1 dzień | u mnie |
| 6 | Pomiar: czy nowe cechy podnoszą t-stat IC | 1 bieg | **u Ciebie** |
| 7 | P3-1/P3-2 uniwersum point-in-time + historia od 2005, P3-4 metryka względna | ~1–2 dni + długi backfill | kod u mnie, **backfill i trening u Ciebie** (300 symboli × 20 lat to godziny pobierania) |
| 8 | P4-1 GBDT, P4-2 ensembling, P4-4 CPCV | ~1–2 dni | u mnie |
| 9 | P2-3 fundamenty point-in-time (EDGAR `filed_at` + panel historyczny) | ~2 dni | u mnie; pobranie EDGAR **u Ciebie** (sandbox nie ma egressu do SEC) |
| 10 | Etap 5 — meta-labeling, koszty, sizing | ~2 dni | u mnie, po przejściu bramki E4 |

**Decyzje, które muszę usłyszeć od Ciebie przed startem:**

1. **Źródło uniwersum point-in-time** — historyczne składy S&P 500 nie są darmowe w czystej formie.
   Warianty: (a) zrekonstruować przybliżenie z dostępnych list + dat wejścia/wyjścia z Wikipedii,
   (b) użyć uniwersum „wszystko, co miało ≥ X mln USD obrotu dziennego w danym roku" liczonego
   z naszych danych, (c) kupić dane. Rekomendacja: **(b)** — jest odtwarzalne, nie ma survivorship
   z definicji i nie wymaga zewnętrznego źródła.
2. **Horyzont** — czy zgadzasz się, żeby wybrał go pomiar (P1-2), nawet jeśli wyjdzie 63 sesje
   i tym samym rebalans stanie się miesięczny.
3. **Książka względna** (P3-4) — czy docelowo chcemy handlować long-short/względem benchmarku,
   czy zostajemy przy long-only. To zmienia i metrykę, i to, czy szerokość w ogóle rośnie.

---

## 13. Co ten plan dziedziczy po `backlog_2026_07_27.md`

Backlog po audycie zewnętrznym zostaje **zarchiwizowany** (`docs/archive/`) — ten plan jest jego
następcą dla toru predykcji. Żeby archiwizacja niczego nie zgubiła, poniżej pełne mapowanie
niezamkniętych pozycji:

| Backlog | Tutaj | Zmiana względem backlogu |
|---|---|---|
| T1-1 uniwersum 200–500 point-in-time | **P3-1** | uzasadnienie zmienione: **moc pomiarowa**, nie szerokość (ta się nasyca — §1) |
| T1-2 historia od 2005 | **P3-2** | bez zmian |
| T1-4 `momentum_12_1`, `momentum_6_1`, vol-scaled, `dist_52w_high`, `beta_60` | **P2-1** | rozszerzone o rewersję 1M osobno od momentum, MAX, Amihud, dollar volume; + twardy warunek `lookback` po obu stronach |
| T1-5 `filed_at` point-in-time | **P2-3** | rozszerzone: potrzebny **panel historyczny**, nie tylko pole (dziś latest-per-symbol w pamięci) |
| T2-1 Tier-2 do treningu | **P2-3** | to samo zadanie — fundamenty wchodzą razem z panelem |
| T2-2 historia makro z vintage/lag | **P2-4** (nowe) | makro jako cecha wymaga danych vintage, inaczej to look-ahead na rewizjach |
| T2-3 LightGBM + logreg jako baseline | **P4-1** | + **regresja logistyczna jako obowiązkowy baseline liniowy** — model, który nie bije liniowego, nie zasługuje na sieć |
| T2-4 rekalibracja barier 2.0 → 1.0 | **P1-1** | poparte pomiarem (46.3% pionowa przy 1.0σ) |
| T2-5 neutralizacja sektorowa | **P2-2** | + zależność od FLOW-8 (sektory jako wolne łańcuchy znaków) |
| T2-6 meta-labeling | **P5-1** | warunek wejścia: G1 bramki spełnione |
| T3 CPCV + PBO | **P4-4** | DSR już zrobione (G5); dochodzi PBO (probability of backtest overfitting) |
| T3 σ-skalowany stop-loss | **P5-3** | dziś SL są procentowe |
| T3 fractional differentiation | **P2-5** (opcjonalne) | do rozważenia dopiero, gdy cechy stacjonarne wyczerpią się |

**Pozycje z backlogu, które NIE należą do toru predykcji**, przeniesione do „Known issues /
tech debt" w `CLAUDE.md`: FLOW-2 (arbitralne 200 bps w filtrze kosztów), FLOW-3/D3 (makro liczone
podwójnie), FLOW-4 (stan portfela tylko po HTTP), FLOW-5 (rozliczanie wyniku strategii),
FLOW-7 (brak okresu próbnego w monitorze degradacji), FLOW-8 (sektory jako łańcuchy znaków),
FLOW-9 (kryterium przejścia na realny kapitał), zatrzask wyłącznika, event sourcing,
konsumenci pull.

### Decyzje otwarte, przeniesione z backlogu (D1–D8)

| # | Decyzja | Status |
|---|---|---|
| D1 | Rozmiar/źródło uniwersum | **otwarta** → §12 pkt 1; rekomendacja zmieniona na rekonstrukcję po obrocie (odtwarzalna, bez survivorship) |
| D2 | Horyzont 10 vs 21 | **otwarta** → rozszerzona do 10/21/63, rozstrzyga pomiar (P1-2) |
| D3 | Makro: usunąć z agregatora | **otwarta** — nie zrobione; `REGIME_BIAS` nadal działa. Przeniesione do tech debt |
| D4 | Meta-labeling zamiast równoległego głosowania | kierunek przyjęty → **P5-1**, po bramce E4 |
| D5 | Filtr RSI | **otwarta** → wchodzi jako zwykły kandydat do oceny per cecha w **E2** |
| D6 | `llm-svc` | **rozstrzygnięta: odłożone** (§11) |
| D7 | Backtest na nakładających się transzach | **otwarta** — silnik backtestu nadal dzienny, więc backtest i ocena ML są nieporównywalne |
| D8 | Reżim jako cecha czy warunkowanie | **otwarta** → wraca razem z P2-4 (makro vintage) |
