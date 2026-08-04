# 03 — Dane

## Zwroty liczymy na cenie SKORYGOWANEJ, egzekucję na SUROWEJ

**Kiedy:** 2026-07-28 (P0-1)
**Dlaczego:** `adj_close` uwzględnia dywidendy i splity — bez niego zwrot spółki dywidendowej jest
systematycznie zaniżony, a split wygląda jak −50% w jednej sesji. Ale zlecenie płaci cenę **surową**,
więc obie serie muszą istnieć obok siebie.
**Zastąpiła:** `auto_adjust=False` + odrzucanie `Adj Close` — czyli przez pierwsze tygodnie
dywidendy nie istniały w żadnym obliczeniu.
**Dowód:** na parze świec z luką dywidendową zwrot surowy pokazuje −2%, skorygowany 0%.
**Konsekwencja, która zaskakuje:** współczynnik korekty stosuje się do **całej świecy** (OHLC), bo
bariery triple-barrier porównują się z high/low — mieszanie skal fałszowałoby dotknięcia.

**Gdzie to gryzie:** kapitalizacja rynkowa liczy się z ceny **surowej**. Liczba akcji pochodzi ze
sprawozdania (stan na datę filingu), więc pomnożenie jej przez cenę korygowaną wstecz daje
kapitalizację, która nigdy nie istniała — późniejszy split 2:1 zmniejsza ją o połowę.

## Point-in-time albo wcale

**Kiedy:** 2026-07-29 (P2-3), rozszerzone 2026-08-01
**Dlaczego:** doklejenie dzisiejszego F-score'u do sesji z 2022 uczy model faktów opublikowanych
dwa lata później. Ta awaria **nie zgłasza się jako błąd — poprawia backtest**.
**Reguły, każda dobrana tak, żeby błąd mógł iść tylko w bezpieczną stronę:**

- pojedyncza wartość: wygrywa **najwcześniejsze** zgłoszenie (ten sam okres wraca w korektach
  i w kolumnie porównawczej następnego 10-K; liczy się, kiedy rynek dostał liczbę pierwszy raz);
- całe sprawozdanie: **najpóźniejsza** z dat jego pól (użyteczne, gdy każde pole jest publiczne —
  datowanie po najwcześniejszym twierdziłoby, że znamy liczby jeszcze nieogłoszone);
- brak daty → sprawozdanie **niewidoczne** dla odczytu as-of. **Niedatowane ≠ stare**: gdyby mogło
  wygrać złączenie, byłoby „znane" dla całej historii;
- odcięcie dla sesji D to **północ UTC tego dnia** — raport złożony tego samego dnia nie wchodzi,
  bo filingi wychodzą po sesji.

**Dowód:** ranga `f_score` jednego symbolu przeskakuje **dokładnie na dacie jego drugiego
zgłoszenia** (0.000 przed, 1.000 po), a kontrola anty-szczęściowa pokazuje, że wersja bez
point-in-time stawia go na szczycie od pierwszego dnia.

**Rozszerzenie 2026-08-01:** poprzednie sprawozdanie (potrzebne do wzrostu aktywów) musi przejść
**to samo odcięcie**. Wybór po samym okresie fiskalnym sięgnąłby po filing jeszcze nieopublikowany,
gdy restatement albo spóźniony filer odwracają kolejność publikacji względem okresu.

## Panel fundamentów musi trzymać historię, nie tylko teraźniejszość

**Kiedy:** 2026-08-01
**Dlaczego:** `refresh` woła `latest_statements(count=2)` — to poprawna odpowiedź na „co wiemy
teraz" (pytanie serwowania), ale była **jedyną ścieżką zapisu do panelu**. Uczenie dostawało więc
najwyżej dwa lata na spółkę, a złączenie po 20 latach dawało neutralne 0.5 na prawie każdej sesji.
**Dowód:** `dataset.fundamental_coverage` — liczba, która odróżnia „cecha słaba" od „cechy, której
tam nie było".
**Ograniczenie:** EDGAR czytamy tylko rocznie (10-K), więc 20 lat to ~20 obserwacji na spółkę.
Rodzina rankuje przekrój wolno i z definicji nie może nic znaczyć przy horyzoncie 10 sesji —
warunek, który D2 rozstrzyga na jej korzyść: przy horyzoncie kwartalnym (63) roczne
sprawozdanie ma szansę cokolwiek zrankować. Nadal wchodzi dopiero po pomiarze IC.

## Uniwersum point-in-time rekonstruowane po obrocie (D1)

**Kiedy:** 2026-07-28
**Dlaczego:** alternatywą był zewnętrzny dostawca składu indeksu — płatny i nieodtwarzalny.
Rekonstrukcja z naszych danych (mediana obrotu dolarowego w oknie trailing, top-N, rebalans
kwartalny) jest odtwarzalna i z definicji nie zawiera survivorship — **selekcja nie patrzy na
zwroty**, więc nie może przypadkiem stać się strategią.
**Świadoma decyzja:** przed pierwszym rebalansem uniwersum jest **puste, nie „wszyscy"** — domyślne
„wszyscy" po cichu przywracałoby listę ocalałych na tych właśnie sesjach.
**Czego to NIE naprawia:** jeśli podana lista tickerów to lista ocalałych, żadna reguła selekcji nie
odzyska nazw, których nie ma. `survivorship_report` **mierzy** wejścia i wyjścia i nazywa listę
ocalałych listą ocalałych. Stan na 2026-08-01: `names_ending_early: 0` — uniwersum nadal nie zawiera
ani jednego wyjścia.

## Wagi unikalności — nakładające się etykiety nie liczą się w pełni

**Kiedy:** 2026-07-28 (P0-3, AFML rozdz. 4)
**Dlaczego:** etykieta h=10 przy próbkowaniu dziennym dzieli okno z ~9 sąsiadami, więc strata
liczyła jeden epizod rynkowy dziesięć razy.
**Dowód skali problemu:** efektywna wielkość próby na realnym panelu to **1328 z 1 868 128 wierszy**
(korelacja par 0.358).
**Subtelność:** normalizacja po **sumie wag**, nie po liczbie wierszy — inaczej osłabienie wag
byłoby przebraną zmianą kroku uczenia.

## Kontrakt danych — twarde asercje na kształcie, nie na wyniku

**Kiedy:** 2026-07-27 (T0-1)
**Dlaczego:** model wytrenowany na uciętej historii albo na cienkim przekroju wygląda jak
wytrenowany i nie znaczy nic. Naruszenie → wyjątek i HTTP 422 z pełnym raportem.
**Świadoma decyzja:** porównujemy z **faktycznie otrzymanymi świecami**, nie z `limit` żądania —
inaczej byłby fałszywy alarm zawsze, gdy baza ma mniej historii.

## Pułap `limit` jest JEDNĄ stałą

**Kiedy:** 2026-07-30
**Dlaczego:** zadeklarowany dwa razy w dwóch wartościach (ml-pipeline `le=10_000`, market-data
`le=5000`) objawił się dopiero jako 422 na najdłuższym biegu — po zbackfillowaniu 455 symboli.
Żądanie 20 lat (5040 sesji + 253 rozgrzewki = 5293) leżało dokładnie pomiędzy.
**Teraz:** `trading_common.constants.MAX_OHLCV_LIMIT`, czytany przez wszystkich.
**Powtórka tego samego błędu piętro wyżej:** kontrola pokrycia miała zaszyte `limit=5000`, więc
346 z 455 symboli raportowało dokładnie 5000 sesji, a strażnik `stored_max > train_limit` **nie mógł
zadziałać** — obcinał własne wejście.

## Zapis jest idempotentny i odporny na powtórzone bary

**Kiedy:** 2026-07-31
**Dlaczego:** Postgres odmawia `ON CONFLICT DO UPDATE`, którego własna lista VALUES nazywa ten sam
klucz dwukrotnie. Dostawca zwracający zdublowany bar wywracał **cały** zapis symbolu — i każdą próbę
ponowienia, bo payload jest deterministyczny.
**Dowód:** przed poprawką `CardinalityViolationError` na prawdziwym Postgresie, po — 4/4.
**Klucz deduplikacji to INSTANT**, nie obiekt daty: naiwny i świadomy znacznik tej samej chwili to
dwie różne wartości Pythona i jeden wiersz w TIMESTAMPTZ.

## Pobieranie przyrostowe wznawia się od ostatniej świecy, nie od wczoraj

**Kiedy:** 2026-07-30
**Dlaczego:** tydzień przestoju naprawia się sam, zamiast zostawić trwałą dziurę.
**Najważniejsza część to wykrywanie restatementu:** `adj_close` nie jest własnością świecy, tylko
świecy **plus wszystkich późniejszych zdarzeń korporacyjnych**, więc po splicie dostawca przepisuje
całą historię. Czysto przyrostowe pobieranie zostawiłoby stare bary na przedsplitowej skali —
seria wyglądałaby wiarygodnie i była błędna dokładnie na złączeniu. Zakładka służy porównaniu
**współczynnika** `adj_close/close`; przy zmianie wymuszane jest pełne odświeżenie **do
`earliest_timestamp`**, nie do domyślnego okna.

## Historia makro jest VINTAGE albo jej nie ma (P2-4)

**Kiedy:** 2026-08-02
**Stan przed:** macro-data trzymała jeden snapshot w pamięci. `build_dataset` od zawsze przyjmował
`regime_by_date` i **nikt nigdy tego parametru nie przekazał**, więc pięć kolumn `macro_*` było
w każdym treningu stałym zerem i wypadało przez filtr zerowej wariancji. Rodzina istniała z nazwy.
**Dlaczego vintage, a nie zwykły backfill:** FRED **rewiduje szeregi wstecz**. Zapytanie dziś o marzec
2015 zwraca wartość po rewizjach — liczbę, której wtedy nikt nie mógł znać. Backfill „ostatnich
wartości" wyglądałby kompletnie i byłby błędny dokładnie tam, gdzie najtrudniej to zauważyć.
**Rozstrzygnięcie:** ALFRED (`realtime_start`/`realtime_end` na tym samym endpoincie) zwraca każdą
obserwację razem z oknem, w którym BYŁA opublikowaną wartością. Panel jest kluczowany trójką
`(series, observation_date, realtime_start)` — szereg makro ma **dwie osie czasu** i obie są nośne:
okres, który liczba opisuje, oraz dzień, od którego ta liczba była znana.
**Odczyt as-of jest dwuwymiarowy.** Pominięcie `realtime_start` karmi model rewizjami, które jeszcze
nie istniały; pominięcie `observation_date` zwraca to, co ostatnio zrewidowano, zamiast najnowszego
okresu. To dwa różne błędy i żaden się nie wywala.
**Opóźnienie publikacji wychodzi za darmo:** marzec jest publikowany w kwietniu, a `realtime_start`
to koduje — nie ma osobnego parametru „lag" do przestrojenia.
**Wiersz bez `realtime_start` jest NIEWIDOCZNY dla odczytów historycznych**, nie „stary". Ta sama
reguła co `filed_at` w fundamentach: faktu, którego nie umiemy zadatować, nie wolno użyć
point-in-time. Wartownik jest w dalekiej **przyszłości** — zakodowanie „nieznane" jako bardzo stara
data zrobiłoby dokładnie odwrotność i uczyniłoby każdy niedatowany wiersz najwcześniejszą rzeczą,
jaką wiedzieliśmy.
**Reżim jest WYPROWADZANY przy odczycie, nie zapisywany.** Utrwalenie etykiety zamroziłoby jedną
wersję `classify_regime` w danych: po zmianie progu historia dalej twierdziłaby swoje, bez śladu,
że policzono ją innymi regułami.
**Dzień, którego nie da się sklasyfikować, jest NIEOBECNY** w odpowiedzi, nie wypełniony domyślnym
„expansion" — `_regime_one_hot` zamienia brak w same zera, a wymyślona wartość byłaby faktem
zmyślonym tam, gdzie prawdą jest brak.
**Zweryfikowane na prawdziwym PostgreSQL-u (13/13)** ze schematem z `init-db.sql`: rewizja z 2016
nie wycieka do zapytania o 2015, klucz główny to faktycznie trójka z vintage, powtórzony wiersz
w jednej partii nie wywraca zapisu, a lipcowe pogorszenie zmienia reżim dopiero od lipca.
**Wciąż otwarte: D8** — czy reżim ma być cechą modelu, czy warunkowaniem (osobny model per reżim).
Teraz jest przynajmniej mierzalny.
