# 02 — Ryzyko

Reguły nienegocjowalne żyją w `CLAUDE.md` („Risk rules"). Tutaj jest **dlaczego** każda z nich ma
taki kształt i co się stało, gdy któraś była tylko deklaracją.

## Dwie warstwy: `RiskEnvelope` (bramka) i risk-mgmt (sizing)

**Kiedy:** 2026-06-26
**Dlaczego:** koperta jest tania, bezstanowa i przechodzi przez nią **każdy** producent sygnału —
dlatego siedzi w `trading-common`. Sizing wymaga stanu portfela (obsunięcie, ekspozycja, reżim),
więc należy do serwisu, który ten stan trzyma.
**Zastąpiła:** wcześniej koperta miała krok 7 („sizing"), który odrzucał sygnały. To mieszało bramkę
z decyzją o wielkości — usunięte; koperta jest **czystą bramką**, sizing jest w `PositionSizer`.

## Żadnego zlecenia bez `stop_loss`

**Kiedy:** 2026-06-25
**Dlaczego:** pozycja bez zdefiniowanego wyjścia nie ma ograniczonej straty, a wszystkie limity
portfelowe liczą się od odległości do stopu.
**Egzekwowane w dwóch miejscach naraz:** `model_validator` w `TradingSignal` (kontrakt) i osobne
sprawdzenie w `RiskEnvelope` (`missing_stop_loss`). Obrona w głąb jest tu celowa — kontrakt można
obejść, konstruując zdarzenie ręcznie.

## Wyłącznik **zatrzaskuje się**

**Kiedy:** 2026-07-30
**Dlaczego:** reguła brzmi „drawdown > 15% → zamknij pozycje, **wymagaj restartu przez człowieka**".
Wyłącznik, który raportuje tylko bieżącą metrykę, spełnia to pozornie: po odbiciu rynku sam się
kasuje i system wznawia handel po katastrofalnej stracie, bez niczyjej decyzji.
**Dowód:** test odtwarzający sekwencję — 16% obsunięcia trafia BLACK, rynek odbija do 5%, stary
wyłącznik przepuszczał nowe zlecenia. Analogicznie RED („dzienna strata > 5% → halt do jutra")
znikał przy śróddziennym odbiciu, choć reguła mówi o **czasie**, nie o metryce.
**Zatrzask jest utrwalony**, bo inaczej restart kontenera byłby najprostszym obejściem wymogu
„require human restart" — i nikt by tego nie zauważył. Wyjście z BLACK: `POST /circuit-breaker/reset`,
odmawiany, dopóki naruszenie trwa (reset **potwierdza** powrót, nie wykonuje go).

## Halt blokuje tylko `intent=NEW`

**Kiedy:** 2026-07-27 (N1)
**Dlaczego:** zamknięcie pozycji też jest zleceniem. Halt, który je blokuje, jest dokładnym
odwróceniem sensu bezpiecznika — i objawiłby się wyłącznie w najgorszym dniu roku.
**Dowód:** `CircuitBreakerTriggeredEvent(action_taken="flatten_all")` konsumowała wcześniej **tylko**
`notification` — czyli reguła „DD > 15% → flatten all" była alertem, nie akcją. Żadna pozycja nie
była zamykana. Zweryfikowane na żywo po naprawie: 2 pozycje → BLACK → książka pusta, 2 zlecenia
likwidacyjne w strumieniu ORDERS.

## Sizing adaptacyjny do obsunięcia, z martwą strefą

**Kiedy:** framework supplement A3
**Dlaczego:** pełne ryzyko przy małym obsunięciu jest normalną zmiennością, nie sygnałem — liniowe
skalowanie od zera karałoby za szum. Stąd pełne 2% ryzyka do DD=5%, potem liniowo do 0% przy DD=15%.

## Limity sektorowe działają na znormalizowanych nazwach GICS

**Kiedy:** 2026-07-28 (P2-2, zamyka FLOW-8)
**Dlaczego:** `is_sector_allowed` porównywał tekst **dosłownie**, więc profil mówiący „Healthcare"
zamiast „Health Care" nie trafiał na listę dozwolonych i BUY był **po cichu odrzucany**. Różnica
w zapisie danych działała jak decyzja ryzyka — fail-closed, czyli najtrudniej zauważalnie.
**Dowód:** zweryfikowane na żywo — w reżimie kryzysu „Healthcare" i „Consumer Defensive" dostają
zlecenia (przed poprawką obie były odrzucane), a „Information Technology" i nieznany „Crypto" nadal nie.
**Świadome ograniczenie:** ciąg, który nie normalizuje się do żadnego sektora, **nadal jest
odrzucany**. Normalizacja nie może stać się pobłażliwością — ta bramka jest z założenia konserwatywna.

## Stan portfela i wyłącznika przeżywa restart

**Kiedy:** 2026-06-29
**Dlaczego:** snapshot w Redisie po każdej mutacji + `restore()` na starcie. Bez tego restart
kasował obsunięcie i historię zleceń.
**Znane ograniczenie:** to snapshot, nie log zdarzeń — pojedynczy pisarz. Event sourcing jest
w tech debt.

## Idempotencja zleceń: `OrderLedger` per (symbol, strona, sesja)

**Kiedy:** 2026-07-27 (N2)
**Dlaczego:** agregator publikuje decyzję za każdym razem, gdy dojdzie nowy komponent. Bez rejestru
głos ML dokładał **drugie zlecenie i podwajał pozycję**.
**Dowód:** kontrola anty-szczęściowa — z wyłączonym rejestrem ta sama sekwencja zdarzeń daje
2 zlecenia; z rejestrem 1.
**Subtelność:** rejestrowane jest **tylko faktycznie wystawione** zlecenie. Odrzucone przez halt,
sizing albo limit sektorowy nie zostawiło ekspozycji, więc wolno spróbować ponownie. Sesja liczona
ze znacznika **emisji** zdarzenia, żeby redelivery durable'a nie otworzyło pozycji drugi raz.

## Przejście na realny kapitał — kryterium niedomknięte

**Status: otwarta (FLOW-9).** Jest reguła „30 dni papieru z dodatnim Sharpe'em", ale brak pełnej
listy: maksymalne zaobserwowane obsunięcie, realizm wypełnień, próba generalna wyłącznika.
Do napisania, zanim ktokolwiek pomyśli o pieniądzach.
