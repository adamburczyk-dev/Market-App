# 06 — Walidacja i bramka aktywacyjna

## Purged walk-forward + embargo, nigdy losowy podział

**Kiedy:** od początku (reguła w `CLAUDE.md`), za López de Prado
**Dlaczego:** losowy podział na danych czasowych z nakładającymi się etykietami wycieka przyszłość
do treningu przez samo sąsiedztwo wierszy. Purge usuwa wiersze, których okno etykiety wchodzi
w okno testowe; embargo dokłada bufor po nim.

## Metryka decyzyjna to Sharpe **ACTIVE** (P3-4)

**Kiedy:** 2026-07-29
**Dlaczego:** portfel long-only w rosnącym rynku osiąga wysoki Sharpe **na samej becie**.
**Dowód, i to dwukrotny:** fold_0 biegu #1 miał Sharpe 3.85 przy **ujemnym** lifcie −0.013 (dwie
trzecie spółek rosło, więc dowolny portfel long zarabiał). Bieg #2 **przeszedł starą bramkę**
z absolutnym Sharpe 0.79 przy AUC 0.4865 i active **−1.06** — uniwersum equal-weight robiło 1.36.
**Próg dla active to 0.0, nie 0.5** — to różnica dwóch skorelowanych serii i jest znacznie trudniejsza
do zdobycia niż liczba absolutna.
**Sharpe absolutny zostaje jako warunek wtórny**, bo reguła projektu „nic poniżej OOS 0.5 na żywo"
dotyczy tego, czym się realnie handluje.

## Szerokość long-only nasyca się — stąd książka względna

**Kiedy:** 2026-07-28 (D3)
**Dowód:** korelacja par ρ = 0.253 → efektywna liczba zakładów `1/ρ` ≈ **3.95 nazw**. Dokładanie
spółek do książki long-only **nie dodaje zakładów** — rośnie tylko moc pomiarowa.
**Świadome ograniczenie:** handel zostaje long-only. Shorty wymagają modelowania end-to-end (silnik,
sizing, broker) i to osobna decyzja, nie efekt uboczny zmiany metryki.

## Bramka G0–G5 zamiast progu na Sharpe

**Kiedy:** 2026-07-27 (T1-3)
**Dlaczego:** stara bramka (Sharpe > 0.5 + 2/3 foldów + Brier) **przepuściła** model bez śladu
sygnału. Każdy z sześciu warunków zamyka inny sposób, w jaki model bez przewagi może wyglądać dobrze:

| # | Warunek | Co zamyka |
|---|---|---|
| G0 | sanity: `best_epoch > 1`, predykcje niestałe, dość okien | w biegu #2 dwa foldy oceniały wagi **sprzed nauki**; w biegu #3 model wypuszczał jedną liczbę dla każdego wiersza |
| G1 | **t-stat średniego IC ≥ 2** | na 63 sesjach SE Sharpe'a to ~2.0; średnie IC ma o rząd wielkości większą moc, bo liczy każdą nazwę w każdym przekroju |
| G2 | przewaga nad rangą **najlepszej** surowej cechy, **ze znakiem** | model, który nie bije jednej cechy, nie zasługuje na warstwę ML. Maksimum wybierane na tym samym oknie jest obciążone w górę → bramka trudnieje z każdą dołożoną cechą, i to jest właściwy kierunek błędu |
| G3 | ekonomia: active > 0 **i** absolutny > 0.5 **i** lift > 0 **i** 2/3 ostatnich foldów | patrz wyżej — beta udająca alfę |
| G4 | kalibracja: Brier vs base rate **okna**, porównanie **parami** z błędem standardowym | stara wersja porównywała z base rate całego zbioru i dawała 0.01 luzu, więc nie mogła oblać niczego realnego |
| G5 | **deflated Sharpe** (Bailey–López de Prado) | wielokrotne testowanie: `n_trials` wchodzi do wzoru |

**Trzy decyzje warte zapamiętania:**
1. DSR liczony na **sklejonej krzywej OOS** (foldy + holdout) — 126 sesji holdoutu nie ustala
   Sharpe'a przy żadnej sensownej ufności (SE ≈ 1.4 rocznie).
2. Próg DSR **0.90, nie 0.95** — świadomie: ta bramka rządzi promocją do **papierowego** głosu,
   a między nią a pieniędzmi stoi osobna reguła „30 dni papieru".
3. `n_trials` to **wejście uczciwościowe**, nie pokrętło — zaniżenie go czyni G5 optymistycznym.
   Dlatego sweep konfiguracji **raportuje liczbę prób** jako `suggested_n_trials`.

**Test przechodzalności jest obowiązkowy.** Bez niego bramka jest nieodróżnialna od trwałego „nie":
syntetyczne uniwersum, w którym przewaga jest **interakcją** cech (żadna pojedyncza jej nie łapie,
więc G2 jest sprawdzalne) przechodzi wszystkie 6 warunków end-to-end.

## Diagnostyka, bez której bramka jest ślepa

**Kiedy:** 2026-07-25 / 2026-07-27 (T0-3, T0-5)
- **lift** — trafność wybranej kwantyli minus base rate. Lift ≈ 0 przy wysokim Sharpe = szczęście,
  nie sygnał. Bez tego raport biegu #1 czytałby się jako „6 z 8 foldów dodatnich".
- **`pred_std`** przed i po kalibracji — `pred_std ≈ 0` to zapadnięty model. Sygnatura zmierzona:
  uczące się uniwersum 0.427, czysty szum 0.027, realny bieg **0.0073**.
- **`auc_train`** — odróżnia „nie umie się nauczyć" od „nie ma czego".
- **efektywna wielkość próby** — oś czasu przez horyzont, przekrój przez korelację par.
- **IC/ICIR, benchmark equal-weight, `sharpe_active`, obrót i koszt** — test-pułapka odtwarza
  fold_0: szum w hossie daje long-only Sharpe 3.96, active **−1.40**, IC −0.0002.

## Sonda pojemności — i jej kontrola

**Kiedy:** 2026-07-27, kontrola naprawiona 2026-08-01
**Po co:** odróżnia „problem optymalizacji" od „nie ma czego się uczyć". Duży, nieregularyzowany
model na tych samych wierszach, na których mierzony jest train AUC.
**Sednem jest kontrola na przetasowanych etykietach** — sieć o dużej pojemności zapamiętuje losowe
etykiety, więc samo wysokie train AUC niczego nie dowodzi.
**Defekt kontroli i jego naprawa:** permutacja **globalna** niszczy naraz sparowanie nazwy
z etykietą (pytanie sondy) ORAZ to, że etykiety jednej sesji w większości się zgadzają (triple
barrier h=10 to w dużej mierze efekt **daty**; korelacja par 0.358). Cechy są trwałe, więc duży
model uczy się „ta konfiguracja to ta data, a ta data rosła" **bez żadnej przewagi przekrojowej**.
**Dowód:** na panelu, w którym etykieta jest czystym efektem daty i przekrojowej przewidywalności
NIE MA z konstrukcji, stara sonda dała gap **+0.066 przy progu 0.05, pełną separację i werdykt
„learnable structure EXISTS"**. Po zmianie na permutację **wewnątrz sesji** ten sam panel daje
+0.0038 i „NO learnable structure".
**Konsekwencja:** werdykt „+0.201, struktura istnieje" z 2026-08-01 pochodzi ze starej kontroli
i **wymaga powtórzenia**.

## CPCV — rozrzut zamiast jednej liczby

**Kiedy:** 2026-07-29 (P4-4)
**Dlaczego:** walk-forward daje **jedną** ścieżkę OOS, a każda liczba w bramce jest jednym losowaniem
z jej rozkładu — foldy biegu #2 szły od Sharpe −1.61 do +4.54 na tych samych danych.
**Najważniejsza liczba to `share_positive`**: strategia dodatnia na 3 z 5 sposobów pocięcia tych
samych danych **nie została pokazana**, cokolwiek mówi średnia.
**Błąd popełniony przy budowie:** pierwsza wersja składała ścieżki z **rozłącznych** podziałów
(1-faktoryzacja grafu pełnego — nie istnieje dla większości (N,k)); zachłanne szukanie po cichu
zwracało jedną zdegenerowaną ścieżkę. Poprawnie: j-ty podział testujący grupę g dostarcza predykcji
dla g w ścieżce j — jeden podział służy wielu ścieżkom.

## Transze `1/h` zamiast dziennego rebalansu

**Kiedy:** 2026-07-27 (T0-4, Jegadeesh–Titman)
**Dlaczego:** oceniany obiekt musi być tym samym, co trenowany — etykieta ma horyzont h, więc
pozycja trzymana jest h sesji.
**Zmierzone:** obrót 80% → 8%, koszt 10.0%/rok → 1.0%/rok. **Uzasadnienie to zgodność horyzontów,
NIE odzysk kosztów** — dryf kosztowy realnego biegu to 0.14 jedn. Sharpe'a, nie 0.6.
**Otwarte (D7):** silnik backtestu nadal rebalansuje dziennie, więc backtest i ocena ML są
nieporównywalne.

---

## Decyzje otwarte odziedziczone po planie predykcji

Plan `plan_2026_07_28_prediction.md` został zarchiwizowany 2026-08-02 jako kodowo skończony (E0–E5).
Jego niezamknięte decyzje żyją tutaj, żeby nie zniknęły razem z dokumentem.

| # | Decyzja | Status | Co ją odblokuje |
|---|---|---|---|
| **D1** | Źródło uniwersum point-in-time | rekomendacja przyjęta (rekonstrukcja po obrocie, `03-dane.md`), ale **lista kandydatów nadal jest listą ocalałych** | dostawca składu indeksu albo dane spółek wycofanych; `survivorship_report` mierzy, ile brakuje |
| **D2** | Horyzont 10 / 21 / 63 | **rozstrzygnięty pomiarem na 63** (`04-etykiety-i-cel.md`), niewdrożony | zmiana `LabelParams.horizon` + ponowna ocena transz i rebalansu |
| **D3** | Makro liczone dwa razy (nastawienie w agregatorze + limity w risk-mgmt) | **otwarta** — patrz `07-agregacja-i-decyzja.md` | decyzja człowieka; rekomendacja: limity zostają, nastawienie znika |
| **D5** | Filtr RSI jako osobna reguła czy doklejka do momentum | **zamknięta 2026-08-02: osobna reguła** (`rsi_bollinger_reversion`). Są to przeciwne zakłady — momentum kupuje szczyt przekroju, rewersja kupuje wyprzedaną nazwę — więc sklejenie ich uśredniłoby dokładnie tę niezgodę, którą agregator ma ważyć. Zmierzalne: reguły są teraz osobnymi źródłami wag | — |
| **D7** | Backtest na nakładających się transzach | **otwarta** — silnik nadal rebalansuje dziennie, więc backtest i ocena ML mierzą różne obiekty | przepisanie silnika na przekrojowy z transzami `1/h` |
| **D8** | Reżim jako cecha czy jako warunkowanie modelu | **otwarta** → wraca razem z P2-4 | historia makro vintage |

Decyzje **rozstrzygnięte** przez ten plan i opisane w pozostałych plikach tego folderu: D4
(meta-labeling zamiast równoległego głosowania → `05-cechy.md` / P5-1), D6 (`llm-svc` — odłożone).
