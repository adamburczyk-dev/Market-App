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
przekrój wolno; przy horyzoncie 10 sesji nie może nic znaczyć, przy 63 może.
**Status: zbudowana, NIE w modelu** — wchodzi, gdy tabela IC per cecha to potwierdzi.

## Reguła etapu E2: rodzina wchodzi do modelu po POMIARZE

**Kiedy:** 2026-07-28, stosowane konsekwentnie od tego czasu
**Dlaczego:** dokładanie kolumn „bo są" rozcieńcza sygnał i podnosi liczbę prób w rozliczeniu
wielokrotnego testowania. Pomiar jest **model-free** (IC surowych rang + t-stat), więc nic nie
kosztuje w rozliczeniu bramki.
**Praktyczna konsekwencja:** nowe wskaźniki liczone dla **strategii regułowych** trafiają do
`EXCLUDED_FEATURES` — kontrakt cech modelu nie może zmienić się mimochodem.
