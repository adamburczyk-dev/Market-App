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
**Zastrzeżenie:** zwycięzca opiera się na `close` — cesze **poziomu, wykluczonej z wejścia modelu** —
więc akurat tej przewagi model nie zobaczy.

## Etykieta nadwyżkowa — zbudowana, domyślnie wyłączona (P1-3)

**Kiedy:** 2026-07-28
**Dlaczego istnieje:** w spadającym rynku spółka, która spada mniej, jest **wygrana** dla etykiety
nadwyżkowej i **przegrana** dla absolutnej. Przy książce ocenianej względem uniwersum (patrz
`06-walidacja-i-bramka.md`, P3-4) to drugie jest niespójne z tym, co mierzymy.
**Dlaczego wyłączona:** włączy ją pomiar, nie preferencja.
**Świadomy koszt:** skanowanie close-to-close, bo dla syntetycznej nogi rynkowej nie ma ścieżki
śróddziennej — bariera dotknięta i odwrócona w ciągu sesji umyka.

## Nakładanie etykiet obsługujemy purgingiem, nie odrzucaniem próbek

**Kiedy:** `ml_integration_plan.md` §4/§6
**Dlaczego:** przy tej wielkości danych odrzucanie 9 z 10 wierszy byłoby droższe niż problem, który
rozwiązuje. Nakładanie jest adresowane trzykrotnie: purge + embargo w podziałach, wagi unikalności
w stracie, transze `1/h` w ocenie portfela.
