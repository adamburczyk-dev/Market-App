# 01 — Architektura

## Podział na 13 mikroserwisów zamiast monolitu

**Kiedy:** 2026-06 (dokument założycielski `Plan_Rozwoju_Systemu_Tradingowego_2.md`)
**Dlaczego:** granice kontekstów w tradingu są naturalnie rozłączne (dane rynkowe, cechy, strategie,
ryzyko, egzekucja) i mają różne profile obciążenia — trening ML zjada minuty CPU, egzekucja musi
odpowiadać natychmiast. W monolicie długi backtest blokuje ścieżkę zleceń.
**Dowód:** zaobserwowane wprost — trasy ml-pipeline liczące minutami zablokowały pętlę zdarzeń
i `/health` przestał odpowiadać w budżecie healthchecku (2026-07-30). W osobnym serwisie kosztowało
to tylko ten serwis.

## Serwis A nigdy nie importuje serwisu B

**Kiedy:** reguła od początku, egzekwowana w „Architecture rules"
**Dlaczego:** import między serwisami zamienia je z powrotem w monolit z osobnymi Dockerfile'ami —
wdrożenie jednego wymusza wdrożenie drugiego, a testy przestają być niezależne.
**Konsekwencja:** wszystko, co współdzielone, ląduje w `trading-common`.

## `trading-common` jest granicą, nie workiem na narzędzia

**Kiedy:** ML-0 (2026-07-13), rozszerzane przy każdym kolejnym etapie
**Dlaczego:** kod, który **musi dać ten sam wynik w treningu i na produkcji**, nie może istnieć
w dwóch kopiach po dwóch stronach granicy serwisów. Duplikat tej arytmetyki to rozjazd
train/serve czekający na wystąpienie.
**Dowód:** zanim `features`/`ranking` trafiły do shared, feature-engine i ml-pipeline liczyły cechy
osobno; okno historii miało dwie niezależne wartości domyślne (250 vs 250), które przy zmianie
rodziny cech rozjechałyby się bez żadnego sygnału. Po przeniesieniu `FEATURE_LOOKBACK` i
`FULL_HISTORY` są **jedną stałą**, a test wymusza `FEATURE_LOOKBACK >= FULL_HISTORY`.

Co jest w `trading-common` i dlaczego akurat to:

| Moduł | Powód |
|---|---|
| `features`, `ranking` | trening musi odtworzyć serwowanie co do bitu |
| `fundamentals` | reguła as-of i wyprowadzenie czynników — ta sama po obu stronach |
| `prices` | jedna definicja tego, czym jest zwrot (skorygowany vs surowy) |
| `sectors` | normalizacja GICS używana i przez ranking, i przez limity ryzyka |
| `RiskEnvelope`, `CostAwareFilter`, `sizing` | bramki przekrojowe — każdy producent sygnału przez nie przechodzi |
| `scheduler`, `timeutil`, `constants` | wspólne reguły czasu i pułapy, które muszą się zgadzać między serwisami |

## Zdarzenia przez NATS JetStream, zapytania przez HTTP

**Kiedy:** dokument założycielski
**Dlaczego:** „coś się stało" to rozgłoszenie do nieznanej liczby odbiorców i musi przetrwać restart
odbiorcy → trwały strumień. „Podaj mi historię AAPL" ma jednego adresata i odpowiedź → HTTP.
**Dowód, że trwałość jest potrzebna:** durable consumer odtwarzający historię strumienia potrafił
wskrzesić nieaktualne sygnały — dlatego wpisy w buforze agregatora starzeją się od **znacznika emisji**
zdarzenia, nie od czasu odbioru.

## Contracts-first

**Kiedy:** reguła stała
**Dlaczego:** typ przechodzący przez granicę serwisów dodany „przy okazji" implementacji istnieje
w dwóch niezgodnych wersjach do pierwszego wdrożenia.
**Praktyka:** najpierw schemat/zdarzenie w `trading-common`, potem producent, potem konsument.
Każde rozszerzenie kontraktu jest wstecznie zgodne (nowe pola opcjonalne), bo stare wiersze w bazie
i stare zdarzenia w strumieniu muszą pozostać ważne.

## Migracje kolumn robimy jawnie, bo `create_all` ich nie robi

**Kiedy:** 2026-07-28 (`adj_close`), powtórzone 2026-08-02 (`gross_profit`/`cost_of_revenue`)
**Dlaczego:** `Base.metadata.create_all` tworzy brakujące **tabele**, nigdy brakujących **kolumn**.
Baza założona przed dodaniem pola po cichu zapisywałaby dalej bez niego.
**Dowód:** zweryfikowane na prawdziwym PostgreSQL-u — tabela utworzona bez kolumny dostaje ją
idempotentnym `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` przy starcie serwisu, dwukrotne
uruchomienie nie rzuca błędu, a istniejący wiersz przeżywa z `NULL`.

## Helm musi być zgodny z compose

**Kiedy:** 2026-07-07
**Dlaczego:** dwie definicje tego samego wdrożenia rozjeżdżają się w tygodniach.
**Dowód, że to nie jest teoria:** duplikat klucza `env:` w wartościach Helma sprawił, że YAML
zachował ostatni blok i po cichu wyrzucił całą konfigurację — `helm lint` przechodził, render
wyglądał poprawnie, efekt był zerowy. Wykryte sprawdzeniem renderu, nie założeniem.
Druga: `ports: !reset` w nakładce produkcyjnej **kasuje** wartość zamiast ją podmienić — brama nie
publikowałaby żadnego portu. Compose **scala** listy, więc `ports: []` też jest no-opem.
