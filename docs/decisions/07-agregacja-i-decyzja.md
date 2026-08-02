# 07 — Agregacja i decyzja

## Węzłem decyzyjnym jest `signal-aggregator` (R1a)

**Kiedy:** 2026-07-07
**Dlaczego:** przed tą decyzją agregat był **doradczy** — nie miał konsumenta, a risk-mgmt działał
na surowych sygnałach strategii. Czyli komponent zbudowany po to, żeby łączyć źródła, nie wpływał
na żadne zlecenie.
**Rozstrzygnięcie:** `SignalAggregatedEvent` niesie kontekst zlecenia (cena, SL, TP, `strategy_name`,
`sector`), a risk-mgmt subskrybuje `signal.aggregated`. Surowy `signal.generated` karmi wyłącznie
agregator.
**Warunek poprawności:** poziomy są dołączane **tylko wtedy, gdy końcowy kierunek zgadza się
z kierunkiem komponentu, z którego pochodzą** — inaczej zlecenie SELL dostałoby stop policzony dla
BUY.

## ML jest głosem BEZ poziomów i nie handluje samo

**Kiedy:** 2026-07-16 (ML-2)
**Dlaczego:** model przewiduje kierunek, a nie punkt wyjścia. Gdyby dostarczał SL/TP, byłyby to
liczby wymyślone poza jego zadaniem.
**Twarda reguła:** głos ML **nigdy nie agreguje sam** — wymagany jest komponent strategii. ML
moduluje decyzję prowadzoną przez regułę; wagi adaptacyjne są siatką bezpieczeństwa.
**Cisza jest celowa:** w martwej strefie (0.45 < p < 0.55) ML nie publikuje nic, tak samo jak
strategia nie publikuje HOLD. Nieaktualne głosy wygasają po TTL.

## Komponenty są SCALANE w oknie, nie rozstrzygane po przybyciu (N2)

**Kiedy:** 2026-07-27
**Dlaczego:** `features.ready` rozchodzi się równolegle do strategii i ml-pipeline, a ścieżka
regułowa (porównanie) **zawsze wygrywa z inferencją**. Agregator publikował więc decyzję
samą-strategią, a chwilę później decyzję z ML — jedna decyzja rodziła się dwa razy.
**Rozwiązanie:** `JOIN_WINDOW_SECONDS` (5 s) scala komponenty; `schedule_decision` odracza,
`drain_pending` opróżnia (także przy zamknięciu).
**Czym okno NIE jest:** to **koalescer, nie blokada raz-na-sesję**. Zmiana reżimu makro albo świeży
sygnał strategii legalnie re-decydują — pierwsza implementacja („emisja dokładnie raz na
symbol/sesję") wywróciła 6 testów i **one miały rację**.
**Anty-podwójne-zlecenie to rejestr risk-mgmt**, nie okno — patrz `02-ryzyko.md`.

## `components_present` — bo cisza jest niewidoczna w `confidence`

**Kiedy:** 2026-07-27
**Dlaczego:** przy nieobecnym źródle wagi się renormalizują, więc brak głosu ML nie zmienia
`confidence` w żaden rozpoznawalny sposób. Sam `components_count` też nie odpowiada na pytanie
„czy ML w ogóle dociera". Zdarzenie niesie **nazwy** źródeł, które faktycznie weszły do decyzji.

## Makro: reżim „slowdown" nie wnosi komponentu (R10)

**Kiedy:** 2026-07-07
**Dlaczego:** wcześniej wnosił HOLD z pewnością 0.0, co przy renormalizacji **kradło wagę** komponentowi
strategii. Reżim znany jako neutralny ma nie głosować, a nie głosować „nie wiem".

## Makro liczone jest DWA RAZY — otwarte (D3/FLOW-3)

**Status: otwarta.** Reżim wchodzi jako kierunkowe nastawienie w agregatorze (`REGIME_BIAS`)
**i** jako limity ekspozycji/sektorów w risk-mgmt. Limity są jego właściwym miejscem; nastawienie
w agregatorze to market timing na jednej zmiennej globalnej. Decyzja czeka.

## Wagi adaptacyjne uczą się z REALIZOWANYCH wyników

**Kiedy:** ML-3 (2026-07-16) dla źródła „ml"
**Dlaczego:** wcześniej pętla uczyła się z wyniku **modelowanego**. Dla ML zamknięte: dojrzały głos
jest odtwarzany na świeżej historii **tą samą regułą triple barrier co trening**, a zwrot ze znakiem
kierunku trafia do `POST /outcomes`.
**Dowód:** na żywo 3 dojrzałe głosy BUY (średnio +7.6%) przesunęły wagę „ml" 0.33 → 0.86.
**Otwarte (FLOW-5):** źródło „strategy" nadal rozlicza się wynikiem modelowanym — powinno dostać
to samo traktowanie.

## Bufor jest kluczowany PARĄ (symbol, strategia)

**Kiedy:** 2026-08-02 (etap strategii + registry)
**Co było:** `dict[str, BufferedSignal]` kluczowany samym symbolem — druga strategia dla tego samego
symbolu **nadpisywała** pierwszą, a zwycięzcą była ta, którą NATS dostarczył ostatnią. Wszystkie
sygnały wchodziły też jako jedno źródło `"strategy"`, więc `AdaptiveWeightOptimizer` — generyczny po
nazwach źródeł — nie miał czego rozróżniać. Przy jednej regule oba defekty były **nieodróżnialne od
poprawnego działania** i dokładnie dlatego przetrwały.
**Rozstrzygnięcie:** `dict[str, dict[str, BufferedSignal]]`, źródło `strategy:{nazwa}` brane
z registry (`strategy_names()`), wygasanie **per wpis** (jedna reguła, która zamilkła, nie wycofuje
świeżego sygnału innej), a lista komponentów sortowana po nazwie — `components_present` nie może
zależeć od kolejności dostarczenia.
**Wybór poziomów przy wielu strategiach:** poziomy wybierane są **po** głosowaniu, spośród wpisów
**zgodnych z końcowym kierunkiem** (stop policzony dla BUY jest bez sensu na SELL); wygrywa
najwyższa `confidence`, remis rozstrzyga **nazwa strategii**. Bez deterministycznego remisu ten sam
zestaw wejść dawałby różne zlecenia.
**Dowód:** 5 testów regresyjnych pada na kodzie sprzed poprawki; na żywym `nats-server` jedno
`features.ready` daje 4 zdarzenia `signal.generated` (BUY/BUY/BUY/SELL), jedną decyzję
z `components_present` = 4 nazwane źródła, a wagi adaptacyjne rozjeżdżają się 0.667 vs 0.056 ze
wspólnego startu 0.143.
