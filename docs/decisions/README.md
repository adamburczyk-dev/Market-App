# Decyzje projektowe — skąd się wzięły

Ten folder odpowiada na jedno pytanie: **dlaczego jest tak, a nie inaczej**. Bez narracji, bez
opisu przebiegu prac — te są w [`../archive/progress_log_2026-06_2026-08.md`](../archive/progress_log_2026-06_2026-08.md).

Każda pozycja ma ten sam kształt:

> **Decyzja** — co ustalono
> **Kiedy** — data
> **Dlaczego** — argument, nie preferencja
> **Dowód** — pomiar, test albo cytat z literatury, na którym decyzja stoi
> **Zastąpiła** — co obowiązywało wcześniej (jeśli coś obowiązywało)

Decyzja **otwarta** jest oznaczona jawnie i mówi, co ją odblokuje. Decyzja podjęta na podstawie
pomiaru podaje liczbę — jeśli liczby nie ma, to jest preferencja i tak ma być napisane.

| Plik | Zakres |
|---|---|
| [`01-architektura.md`](01-architektura.md) | Podział na serwisy, granice, komunikacja, contracts-first |
| [`02-ryzyko.md`](02-ryzyko.md) | Koperta ryzyka, wyłącznik, sizing, limity reżimowe i sektorowe |
| [`03-dane.md`](03-dane.md) | Ceny skorygowane, point-in-time, uniwersum, pułapy i kontrakty danych |
| [`04-etykiety-i-cel.md`](04-etykiety-i-cel.md) | Triple barrier, szerokość barier, horyzont, etykieta nadwyżkowa |
| [`05-cechy.md`](05-cechy.md) | Rangi przekrojowe, rodziny cech, neutralizacja sektorowa, co weszło i co nie |
| [`06-walidacja-i-bramka.md`](06-walidacja-i-bramka.md) | Purged walk-forward, G0–G5, DSR, CPCV, metryka decyzyjna |
| [`07-agregacja-i-decyzja.md`](07-agregacja-i-decyzja.md) | Kto podejmuje decyzję, scalanie komponentów, idempotencja zleceń |

**Reguła:** jeśli podejmujesz decyzję, która zmienia którąkolwiek z tych pozycji, dopisz ją tutaj
**razem z dowodem**. Decyzja bez dowodu wygląda po miesiącu identycznie jak przypadek.
