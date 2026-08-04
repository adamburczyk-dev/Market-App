# 05 — Cechy

## Model dostaje RANGI PRZEKROJOWE, nie wartości surowe

**Kiedy:** 2026-06-25, za López de Prado
**Dlaczego:** poziom RSI 70 znaczy co innego w hossie i w bessie, a poziom ceny nie znaczy nic.
Percentyl w przekroju danej sesji jest porównywalny w czasie i odporny na zmianę reżimu.
**Konsekwencja projektowa:** ranking wymaga uniwersum — stąd `min_universe` i pojęcie „sesji zbyt
cienkiej, żeby rankować". Percentyl po 3 wartościach to zbiór {0, 0.5, 1}.
**Wyjątek:** cechy poziomu (`close`, `sma_*`) są liczone, ale **wykluczone z wejścia modelu** —
ich ranga przekrojowa jest proxy na cenę akcji, nie na sygnał.

## Rodzina cenowa (P2-1)

**Kiedy:** 2026-07-28. Wejście modelu 7 → 15 kolumn.
**Dlaczego akurat te:** każda ma udokumentowaną anomalię przekrojową, nie „bo TA-Lib je ma".

| Cecha | Źródło |
|---|---|
| `momentum_12_1`, `momentum_6_1` (**pomijają ostatni miesiąc**) | Jegadeesh–Titman 1993 |
| `dist_52w_high` | George–Hwang 2004 |
| `max_ret_1m` | Bali i in. 2011 — „loteryjność", rankuje przeciwnie do średniej |
| `downside_vol_20` | semiodchylenie: seria, która tylko rośnie, jest zmienna dla `realized_vol_20` i bezryzykowna dla tej cechy |
| `skew_60` | — |
| `amihud_20`, `dollar_volume_20` | Amihud 2002 |

**`reversal_1m` świadomie NIE powstał** — to dosłownie `return_20d`, który już mamy. Osobna nazwa
dla tej samej kolumny odtworzyłaby duplikat `momentum_20`, który T0-7 właśnie usunął. Wymóg planu
„rewersja osobno od momentum" realizuje **pominięcie ostatniego miesiąca** po stronie momentum —
okna są teraz rozłączne.

**`beta_60` i `idio_vol_60` odłożone z powodu strukturalnego:** wymagają serii rynkowej, a
`compute_feature_vector` z definicji widzi jeden symbol. Zmiana kontraktu serwowania, nie efekt
uboczny. **Ta sama blokada dotyczy pair tradingu.**

## Duplikaty wychodzą z wejścia modelu

**Kiedy:** 2026-07-27 (T0-7)
**Dlaczego:** `momentum_20` był dosłownym duplikatem `return_20d` — doskonała współliniowość daje tę
samą informację dwa razy i sztucznie zawyża jej wpływ.
**Jak wykryte:** test porównuje kolumny **po wartościach**, nie po nazwach, i wymaga 25-symbolowego
przekroju — przy trzech nazwach rangi się sklejają i wszystko wygląda na duplikat.

## Kolumny o zerowej wariancji WYPADAJĄ z kontraktu cech

**Kiedy:** 2026-07-27 (T0-2)
**Dlaczego:** kolumny `macro_*` były zerowe w treningu (brak historii reżimów) i jedynkowe przy
serwowaniu — **żywy rozjazd train/serve**, model spotykałby wzorzec wejścia, którego nie widział.
Dopóki historia makro nie istnieje, muszą **opuścić kontrakt**, a nie być cicho zerowane.
**Otwarte:** P2-4 (historia makro vintage) — dopóki nie powstanie, 5 kolumn wypada z każdego treningu.

## Neutralizacja sektorowa — zbudowana, mierzalna, domyślnie wyłączona (P2-2)

**Kiedy:** 2026-07-28, pomiar 2026-07-31
**Dlaczego odjęcie mediany sektora, a nie rangowanie wewnątrz sektora:** przy 34 nazwach na
11 sektorów to ~3 nazwy na grupę. Odjęcie mediany zachowuje rozdzielczość całego przekroju.
Sektory poniżej `MIN_SECTOR_SIZE=4` idą do jednej grupy resztowej (nie zostają nietknięte — inaczej
część nazw byłaby na innej skali).
**Pomiar na 414 nazwach:** wszystkie 11 sektorów powyżej progu,
`share_neutralized_against_peers = 1.0`, zero nazw w grupie resztowej. Poprawa dla 10 z 15 cech,
średnie |t| **2.88 → 3.24**.
**Pułapka interpretacyjna, o mało nie wysłana w odwrotną stronę:** spadek |t| po neutralizacji nie
jest werdyktem o transformacji — to informacja o **danych** („ranking był w dużej mierze rankingiem
sektorów"). Raport pyta o to, **co przeżywa** (`strong_neutral`), bo tylko ta resztka jest czymś,
co model przekrojowy może uznać za swoje.
**Warunek adopcji:** włączenie w treningu wymaga włączenia w serwowaniu **w tym samym kroku**,
inaczej odtworzymy rozjazd, przed którym broni wspólny moduł.

## Rodzina fundamentalna (§5 planu predykcji)

**Kiedy:** 2026-08-02
**Dlaczego:** wszystkie 15 cech modelu to cena i wolumen — jedna rodzina. Studium zaniku alfy
pokazało, że jej wiarygodna zawartość to rewersja 1-dniowa (nasza architektura jej nie zdąży
wykonać — 28% IC po jednym dniu) i premia za niepłynność (premia za ryzyko, nie sygnał).
Wartość, jakość i rentowność to kanoniczne predyktory **przekrojowe**, których z ceny nie da się
wyrazić.

| Cecha | Źródło |
|---|---|
| `fund_gross_profitability` | Novy-Marx 2013 — im dalej w dół rachunku wyników, tym więcej wyborów księgowych |
| `fund_accruals` (**ze znakiem**) | Sloan 1996 — anomalia polega na tym, że WYSOKIE accruals zapowiadają NISKIE zwroty |
| `fund_asset_growth` | Cooper–Gulen–Schill 2008 |
| `fund_book_to_market`, `fund_earnings_yield` | Fama–French |

**Ograniczenie:** EDGAR rocznie (10-K) → ~20 obserwacji na spółkę przez 20 lat. Rodzina rankuje
przekrój wolno; przy horyzoncie 10 sesji nie może nic znaczyć, przy 63 może — a D2 wskazał
63, więc warunek rozstrzyga się na korzyść tej rodziny (pomiar IC wciąż obowiązuje).
**Status: zbudowana, NIE w modelu** — wchodzi, gdy tabela IC per cecha to potwierdzi.

## Blok klasycznej analizy technicznej — dla REGUŁ, nie dla modelu

**Kiedy:** 2026-08-02 (etap strategii)
**Co doszło:** EMA 12/26, MACD(12,26,9) + linia sygnału + histogram, Bollinger(20,2) + %B
i szerokość, Donchian(20) + pozycja w kanale, ATR(14) + `atr_pct_14`.
**Dlaczego akurat tyle:** każdy z nich ma **konsumenta** — konkretną regułę z registry. Reszta
checklisty „30+" zostaje niezbudowana, bo wskaźnik bez konsumenta to kolumna do policzenia
i utrzymania, która niczego nie rozstrzyga.
**Twarda granica:** wszystkie te nazwy są w `RULE_ONLY_FEATURES` w `trading_common.features`,
a ml-pipeline **importuje** ten zbiór do `EXCLUDED_FEATURES`. Zbiór mieszka przy kodzie, który je
produkuje, więc wskaźnik dodany po jednej stronie nie może wejść do kontraktu cech modelu przez
zapomnienie po drugiej. Ranga przekrojowa EMA to i tak proxy na cenę akcji.
**Dwie decyzje warte zapamiętania:**
- **Donchian liczony z POPRZEDNICH 20 barów.** Kanał zawierający dzisiejszy bar nie może zostać
  przebity (jego maksimum jest z definicji ≥ dzisiejszego zamknięcia), więc reguła wybicia byłaby
  cicha na zawsze i **nigdy nie zgłosiłaby błędu**.
- **`atr_pct_14` obok `atr_14`.** Wskaźniki liczą się na cenach **skorygowanych** (jedna skala, więc
  wolno je porównywać między sobą), a zlecenie wychodzi po cenie **surowej**. Reguły konsumują
  wyłącznie postaci bezwymiarowe (`atr_pct_14`, `bb_pct_b`, `donchian_pos_20`, znak `macd_hist`),
  więc wynik wolno przyłożyć do ceny wykonania bez mieszania skal.

## Reguła etapu E2: rodzina wchodzi do modelu po POMIARZE

**Kiedy:** 2026-07-28, stosowane konsekwentnie od tego czasu
**Dlaczego:** dokładanie kolumn „bo są" rozcieńcza sygnał i podnosi liczbę prób w rozliczeniu
wielokrotnego testowania. Pomiar jest **model-free** (IC surowych rang + t-stat), więc nic nie
kosztuje w rozliczeniu bramki.
**Praktyczna konsekwencja:** nowe wskaźniki liczone dla **strategii regułowych** trafiają do
`EXCLUDED_FEATURES` — kontrakt cech modelu nie może zmienić się mimochodem.

## Kandydat musi być MIERZALNY, nie będąc zaadoptowanym

**Kiedy:** 2026-08-02 (etap wskaźników)
**Defekt strukturalny:** `EXCLUDED_FEATURES` był jednym zbiorem, a `alpha_decay` woła
`build_dataset`, który go stosuje. Czyli **jedyna ścieżka, która mogła zmierzyć nowy wskaźnik, sama
go ukrywała** — reguła etapu E2 („rodzina wchodzi, gdy tabela IC to potwierdzi") była niewykonalna,
bo tabela IC nigdy nie mogła zobaczyć kandydata.
**Rozstrzygnięcie — dwa zbiory, bo to dwa różne powody:**
- `INADMISSIBLE_FEATURES` — poziomy cenowe i duplikat `momentum_20`. **Żaden pomiar tego nie zmieni**:
  ranga przekrojowa poziomu to proxy na cenę akcji. Wykluczone na obu ścieżkach.
- `CANDIDATE_FEATURES` (= `RULE_ONLY_FEATURES`) — policzone, jeszcze nieprzyjęte.
  `build_dataset(include_candidates=True)` je wpuszcza, a **trening nigdy nie ustawia tej flagi**.
**Pomiar nic nie kosztuje w rozliczeniu wielokrotnego testowania** — studium jest model-free (IC
surowych rang + t-stat), więc nie zużywa prób z bramki.

## Rodziny z checklisty Fazy 1 — w postaci, która ma sens po rangowaniu

**Kiedy:** 2026-08-02
**Kryterium doboru:** nie „bo checklista wymienia", tylko: (a) niewspółliniowe z tym, co już jest,
(b) liczalne z samego OHLCV, (c) **ranga przekrojowa musi coś znaczyć**. Ostatni punkt przeformułował
połowę listy.

| Rodzina | Postać i dlaczego taka |
|---|---|
| Stochastic %K/%D(14,3) | pozycja zamknięcia w zakresie high-low; różna od RSI, które widzi tylko zamknięcia |
| CCI(20) | cena typowa vs własna średnia, skalowana **średnim odchyleniem bezwzględnym** (definicja Lamberta — użycie odchylenia standardowego przeskalowałoby konwencję ±100) |
| ADX(14) + ±DI | siła trendu **bez kierunku** — dlatego reguła wybicia i reguła rewersji chcą przeciwnych odczytów tej samej liczby |
| Aroon(25) | czas od ekstremum — jedyna rodzina mówiąca coś, czego nie mówią rodziny cenowe |
| MFI(14) | RSI ważony pieniądzem; odróżnia wzrost, za którym poszedł kapitał, od takiego, za którym nie |
| OBV / A/D | **nachylenie, nie poziom**: skumulowana suma rankuje to, jak długo spółka jest notowana, a nie sygnał. Normalizowane średnim wolumenem, żeby porównywały się mega-cap i small-cap |
| VWAP(20) | **stosunek** `close/VWAP` — sam VWAP to cena |
| Keltner(20) | tylko **pozycja** w kanale; pasma to poziomy cenowe |

**Świadomie POMINIĘTE, z powodem:**
- **Williams %R** — to `%K - 100`, transformacja liniowa, więc jego ranga przekrojowa jest
  **identyczna** ze Stochastic %K. Dodanie go odtworzyłoby duplikat `momentum_20`, który T0-7 usunął.
- **Formacje świecowe** — TA-Lib nie jest zależnością, a ręczne wdrożenie kilkudziesięciu formacji
  to duża powierzchnia przy znikomym udokumentowanym dowodzie przekrojowym.

**Złapane testem, nie przeglądem:** MFI potrzebuje **21** barów, nie 20 (każdy z 20 przepływów jest
klasyfikowany ruchem względem POPRZEDNIEJ ceny typowej), a w pętli %D indeks `end - 14` schodził
poniżej zera — w Pythonie to wycinek liczony od końca, więc „za mało historii" zamieniało się
w pusty wycinek i `ValueError` trzy funkcje dalej. Oba przypinają teraz testy.

## Ważność cech: co model UŻYWA, a nie co koreluje (Faza 3)

`per_feature_ic` (instrument etapu E2) mierzy dowód **marginalny**: czy dana kolumna przewiduje
sama z siebie. Nie odpowiada na pytanie, które podejmuje decyzje o kontrakcie cech — czy model
**potrzebuje** tej kolumny, mając pozostałe czternaście. Te dwie rzeczy rozjeżdżają się rutynowo:
cecha o silnym IC bywa redundantna, a cecha bez własnego IC może nieść interakcję, na której model
faktycznie handluje. `core/importance.py` mierzy to drugie — permutacyjnie.

Cztery decyzje robią różnicę między liczbą a jej obrazkiem:

**Mierzone POZA próbą, na holdoucie.** Permutacja kolumny w oknie, na którym model był FITOWANY,
raportuje to, co zapamiętał. Holdout — ostatnie `holdout_size` sesji, nietknięte przy selekcji — to
jedyne okno, gdzie spadek znaczy „tyle tej cechy niesie zdolność predykcyjną".

**Permutacja WEWNĄTRZ sesji** — lekcja z sondy pojemności ([06](06-walidacja-i-bramka.md)) w innym
pytaniu. Globalne tasowanie kolumny przenosi wartości MIĘDZY datami, więc niszczy dwie rzeczy naraz:
które NAZWISKO miało którą wartość ORAZ rozkład brzegowy cechy w danej sesji — a cechy są trwałe,
więc ten rozkład dryfuje z reżimem. Miarą jest przekrojowe IC per sesja, więc sama zmiana rozkładu
zarejestrowałaby się jako ważność nawet dla kolumny, której model nie czyta.
**Zmierzone, nie założone**: cecha STAŁA w obrębie sesji (kształt każdego one-hota `macro_*`) dostaje
przy permutacji wewnątrzsesyjnej dokładnie `ΔIC 0.00000, t 0.00`, a przy tasowaniu globalnym
**`ΔIC +0.236, t +3.55`** — połowa ważności prawdziwego sygnału na tym samym panelu, wymyślona
w całości. Rodzina `macro_*` weszłaby do raportu jako istotna.

**Punktowane spadkiem IC, PAROWANYM sesja po sesji.** IC to wielkość, na której zbudowane są rangowe
warunki bramki, więc ważność jest w tej samej jednostce co decyzja, którą informuje. Ważniejsze jest
parowanie: nieparowane porównanie dwóch średnich po 126 sesjach ginie w zmienności samego IC,
a różnica per sesja tę zmienność usuwa, bo oba człony widziały ten sam dzień. Błąd standardowy
różnicy parowanej zamienia spadek w dowód, a **próg skorygowany Šidákiem na liczbę testowanych cech**
nie pozwala odczytać największego z piętnastu losowań szumu jako ulubionego wejścia modelu.

**Raportowane także dla RODZIN.** Permutacja dzieli zasługę między skorelowane cechy: dwa
bliskie duplikaty wyglądają każdy na nieważny, bo model odzyskuje sygnał przez bliźniaka. To nie jest
wada do zaklinania, tylko powód, żeby permutować całą rodzinę naraz — a rodzina jest jednostką,
w której ten projekt podejmuje decyzje. Każdy wiersz niesie dodatkowo najsilniejszą korelację z cechą
**spoza** swojej grupy, więc zero obok bliźniaka 0.95 czyta się jako redundancja, a nie jako brak
znaczenia.

**Bieg na żywo pokazał dokładnie ten mechanizm.** Na uniwersum 20 nazw × 1047 sesji **żadna
pojedyncza cecha nie przeszła progu** (najwyżej `amihud_20`, t +1.63), a **rodzina `liquidity`
przeszła z t +4.18 i ΔIC +0.094** przy bazowym IC modelu 0.124 — czyli trzy czwarte całej mocy
rankującej. Raport wyłącznie per-cecha orzekłby „model nie zależy od niczego".

**Podłoga jest MIERZONA, nie założona.** Trasa `POST /models/feature-importance` fituje własny model
z **posadzoną kolumną czystego szumu** (prawidłowa ranga przekrojowa, zero informacji z konstrukcji)
i raportuje, ile ta kolumna zdobyła: na biegu na żywo `t = +0.16`. Korekta Šidáka zakłada
niezależność testów i obojętność modelu na bezużyteczne wejście — model, który przyczepił się do
szumu, mówi to tutaj, a nie w przypisie. Studium jest **wyłącznie diagnostyczne**: dopasowany model
ma kolumnę, której serwowanie nie umie wytworzyć, więc nic z niego nie jest rejestrowane ani
promowalne. Bieg treningowy mierzy model PRODUKCYJNY i posadzonej kolumny nie ma (`noise_control:
null`).

**Defekt złapany testem, nie przeglądem.** Model, który ignoruje kolumnę **dokładnie** (drzewo, które
nigdy na niej nie dzieli; waga zero), daje identyczne predykcje przed i po permutacji — a powtórzenia
są sumowane i dzielone, co w binarnym floacie nie jest identycznością. Zmierzony spadek wyszedł
`1.7e-17` przy błędzie standardowym `5.2e-18`, więc kolumna, której model dowodnie nie czyta,
zdobyła **t = +3.23** i została ogłoszona jako przekraczająca próg istotności. Pył podzielony przez
pył to iloraz, nie dowód — `IC_DROP_EPSILON` zeruje spadki poniżej 1e-9.
