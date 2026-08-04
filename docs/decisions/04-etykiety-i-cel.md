# 04 — Etykiety i cel

## Triple barrier zamiast zwrotu o stałym horyzoncie

**Kiedy:** 2026-07-12 (`ml_integration_plan.md` §4), za López de Prado
**Dlaczego:** zwrot o stałym horyzoncie ignoruje **ścieżkę**. +1% po 10 dniach, które po drodze
zaliczyły −8%, nie jest wygraną — pozycja z realnym stopem już by nie istniała. Bariery ±σ·√h
skalują cel zmiennością nazwy, więc spółka spokojna i spółka rozchwiana dostają porównywalne zadanie.
**Reguły rozstrzygania:** skan zaczyna się od **następnej** świecy (sygnał liczony po zamknięciu),
dotknięcie obu barier w tej samej sesji rozstrzyga się **konserwatywnie jako strata**, a okno
obcięte końcem danych bez dotknięcia → brak etykiety (nie zgadujemy).

## Szerokość barier 2.0σ → **1.0σ**

**Kiedy:** 2026-07-28 (P1-1), potwierdzone pomiarem na realnych danych 2026-07-31
**Dlaczego:** przy 2.0σ prawie nic nie dociera do barier poziomych — etykieta degeneruje się do
znaku zwrotu w dniu wygaśnięcia, czyli do tego samego zwrotu o stałym horyzoncie, którego triple
barrier miał uniknąć.
**Dowód (414 symboli × 20 lat, skan szerokości):**

| pt_mult | rozstrzygnięć poziomych |
|---|---|
| 0.5σ | 94.5% |
| 0.75σ | 78.9% |
| **1.0σ** | **60.2%** ← pasmo docelowe 40–70% |
| 1.5σ | 31.7% |
| 2.0σ | **16.5%** |

**Zastąpiła:** 2.0σ z pierwotnego planu ML. Ta wartość dawała 90.9% rozstrzygnięć na barierze
pionowej w pełnym pipelinie — audyt zewnętrzny (F2) miał rację.
**Metodologicznie ważne:** szerokość barier to **własność etykiety** — mierzy się ją skanując realne
ścieżki cen, **bez dopasowywania modelu**, więc nie kosztuje ani jednej próby w rozliczeniu
wielokrotnego testowania.

## Horyzont — wybiera POMIAR, nie preferencja (D2)

**Status: rozstrzygnięty pomiarem, czeka na wdrożenie.**
**Kiedy:** prior 21 sesji (2026-07-28), pomiar 2026-07-31/08-01
**Dlaczego pomiar, a nie wybór:** horyzont kusi, żeby dopasować model do każdego wariantu
i zatrzymać zwycięzcę — klasyczna fabryka biasu selekcji. Zamiast tego oceniany jest statystyką
**model-free**: IC surowych rang cech względem każdej kandydującej etykiety. Cel, którego żadna
surowa cecha nie rankuje lepiej niż losowo, nie zostanie zrankowany przez model zbudowany na tych
cechach — i dowiadujemy się tego za darmo.

**Wynik (414 symboli):** najlepszy kandydat to **horyzont 63, etykieta nadwyżkowa** (mean |IC|
0.0274). Drugi — horyzont 63 absolutny z `momentum_12_1` IC **+0.052**, czyli klasyczny efekt
momentum 12-1 na horyzoncie kwartalnym.
**Potwierdzenie z drugiej strony:** studium zaniku alfy pokazało, że IC wiarygodnych cech szczytuje
średnio na **34 sesjach**, a dwie najsilniejsze (`amihud_20`, `dollar_volume_20`) rosną monotonicznie
aż do 63. **Obecny horyzont 10 zamyka etykietę w środku ruchu.**
**Zastrzeżenie — NAPRAWIONE 2026-08-03:** zwycięzca opierał się na `close`, czyli cesze poziomu
wykluczonej z wejścia modelu. Przyczyną był defekt, nie interpretacja: `_score_one_target`
iterował po **wszystkich** cechach, bez filtru wykluczeń, więc `mean |IC| 0.0274` policzono nad
zbiorem kolumn **szerszym niż wejście modelu**. Studium stosuje teraz tę samą regułę dwóch zbiorów
co `build_dataset` (`INADMISSIBLE` + `CANDIDATE`), a raport nazywa własny `feature_scope`.
Kandydaci są mierzalni przez `include_candidates=True`, jak w regule E2.
**Czego to nie zmieniło:** ranking. Na niezależnym panelu syntetycznym (30 symboli × 700 sesji,
mieszane daty debiutu, realny serwis) po wykluczeniu poziomów **dalej wygrywa horyzont 63
z etykietą nadwyżkową**, a porządek 63 ≻ 21 ≻ 10 i nadwyżkowa ≻ absolutna zachowuje się na każdym
poziomie. Zwycięskie cechy to `price_to_sma50` i `dist_52w_high` — **bezwymiarowe**, nie poziomy.
**Do potwierdzenia na realnych 414 symbolach** — panel syntetyczny pokazuje, że maszyneria mierzy
to, co trzeba, a nie że liczba jest ta sama.

### Wdrożenie (2026-08-03/04): co się okazało po drodze

**Horyzont był zadeklarowany CZTERY razy** (`LabelParams.horizon`, `TrainingParams.horizon`,
`Settings.LABEL_HORIZON_DAYS`, `MetaParams.horizon`) i nic ich nie porównywało — `TrainRequest`
nie wystawia horyzontu, więc `TrainingParams` sięgał po własną wartość domyślną, a zbiór danych po
etykietową. Groźny kierunek rozjazdu jest **cichy**: horyzont etykiety WIĘKSZY niż horyzont purge'u
wpuszcza okno etykiety do każdego bloku testowego i **poprawia** metryki. Teraz jedna stała
`LABEL_HORIZON`, przypięta testem zgodności.

**Etykieta nadwyżkowa nie była ścieżką, tylko przyrządem.** `build_dataset` i `OutcomeResolver`
wołały `triple_barrier_label` **bezwarunkowo**, więc `excess=True` nie zmieniało w treningu nic —
wdrożenie D2 przez samo przestawienie flagi byłoby zmianą bez skutku.

**Pomiar, który rozstrzygnął D2, mieszał dwa rodzaje etykiet.** Dawny `_market_path` brał
`max(lengths)` i zostawiał tylko serie pełnej długości, a każdą etykietę osłaniał warunkiem
`len(market) == n`. Przy niejednorodnych datach debiutu to benchmark **ocalałych**, a każda krótsza
spółka wypadała na etykietę **absolutną** wewnątrz kandydata raportowanego jako nadwyżkowy.
Nowy `core/market_path.py` kluczuje benchmark **datą sesji** i liczy go z **mediany dziennych
log-zwrotów** (indeks rebalansowany, uczciwy wobec przeżywalności — spółka wnosi wkład tylko
w dni, w których notowana).

**Koszt decyzji, wprost:** przy h=63 `n_effective_samples` spada z ~1873 do **~294**, przy własnej
regule modułu „poniżej ~50 obserwacji na cechę dane finansowe nie niosą wnioskowania" (×15 cech =
750). To jest cena horyzontu kwartalnego i ma być widoczna **przed** biegiem, nie po nim.

**Szerokość barier przy h=63:** na panelu syntetycznym `pt_mult = 1.0` daje `horizontal_share`
**0.5957**, czyli wewnątrz pasma 40–70%. **Pułapka przy czytaniu raportu:** blok `calibration`
z najwyższego poziomu jest kalibrowany na **bieżącym** horyzoncie (potwierdzone na żywo: `horizon:
10`), więc szerokość zwycięzcy czyta się z `targets.candidates[]`, nie stamtąd.

## Etykieta nadwyżkowa — zbudowana, domyślnie wyłączona (P1-3)

**Kiedy:** 2026-07-28
**Dlaczego istnieje:** w spadającym rynku spółka, która spada mniej, jest **wygrana** dla etykiety
nadwyżkowej i **przegrana** dla absolutnej. Przy książce ocenianej względem uniwersum (patrz
`06-walidacja-i-bramka.md`, P3-4) to drugie jest niespójne z tym, co mierzymy.
**Dlaczego wyłączona:** włączy ją pomiar, nie preferencja.
**Świadomy koszt:** skanowanie close-to-close, bo dla syntetycznej nogi rynkowej nie ma ścieżki
śróddziennej — bariera dotknięta i odwrócona w ciągu sesji umyka.
**Stan 2026-08-04:** ścieżka **zbudowana i przetestowana**, flaga wciąż `False`. Pomiar wskazał
etykietę nadwyżkową, ale przestawienie domyślnej czeka na skalibrowane `pt_mult` przy h=63
z **realnego** panelu — patrz „Wdrożenie" wyżej.
**Co ZOSTAJE absolutne, świadomie:** `next_returns` (P&L książki — książka jest long-only na
gotówce, a `relative_metrics` już odejmuje uniwersum, żeby dać `sharpe_active`; odjęcie benchmarku
drugi raz po cichu zmieniłoby to, co czyta warunek ekonomiczny bramki) oraz `signed_return`
w resolverze (to są pieniądze; spółka, która spadła 3% przy rynku −5%, nikomu nie zapłaciła 2%).
`label`/`correct` liczone są **regułą treningową** — inaczej monitor driftu ocenia model względem
pytania, którego mu nie zadano.

## Nakładanie etykiet obsługujemy purgingiem, nie odrzucaniem próbek

**Kiedy:** `ml_integration_plan.md` §4/§6
**Dlaczego:** przy tej wielkości danych odrzucanie 9 z 10 wierszy byłoby droższe niż problem, który
rozwiązuje. Nakładanie jest adresowane trzykrotnie: purge + embargo w podziałach, wagi unikalności
w stracie, transze `1/h` w ocenie portfela.
