# Mapa dokumentacji

Jeden plik = jedna rola. Jeśli dwa dokumenty mówią co innego, obowiązuje ten wyżej w tabeli.

## Żywe — czytaj i aktualizuj

| Dokument | Rola | Kiedy do niego zaglądać |
|---|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | **Jedyne źródło prawdy o stanie projektu**: co jest zrobione, znane długi techniczne, bieżący etap, log ostatnich prac | Zawsze na początku. Jeśli cokolwiek tutaj przeczy CLAUDE.md, wygrywa CLAUDE.md |
| [`decisions/`](decisions/) | **Dlaczego jest tak, a nie inaczej** — decyzje z dowodem, bez narracji. 7 plików tematycznych + otwarte decyzje D1–D8 | Zanim zmienisz cokolwiek w architekturze, ryzyku, danych, etykietach, cechach, walidacji albo agregacji |
| [`../README.md`](../README.md) | Jak system działa i jak go uruchomić: architektura, zdarzenia, bramka aktywacji, bootstrap i kampania pomiarowa, reguły ryzyka | Uruchamianie, onboarding, opis dla kogoś z zewnątrz |
| [`../Plan_Rozwoju_Systemu_Tradingowego_2.md`](../Plan_Rozwoju_Systemu_Tradingowego_2.md) | **Dokument założycielski + aneks statusu**. Dekompozycja na 13 serwisów, wzorce infrastruktury, checklisty Faz 0–5 — a na końcu **tabela, które pozycje są zbudowane, a które nie** | Gdy pytasz „co jeszcze mieliśmy zrobić". **Faza budowy NIE jest wykonana w całości** — patrz aneks |
| [`ml_integration_plan.md`](ml_integration_plan.md) | **Architektura fazy ML** — dlaczego przekrojowo, dlaczego triple barrier, dlaczego ML jest głosem bez poziomów, jak wygląda rejestr i serwowanie | Gdy pytanie brzmi „jak to jest zbudowane i dlaczego". Wybory ilościowe (§4–§6) są **zastąpione** przez `decisions/04` i `decisions/06` |
| [`framework_supplement.md`](framework_supplement.md) | **Referencja 12 komponentów frameworku** (koperta ryzyka, filtr kosztów, monitory degradacji i driftu, alokator reżimowy…). Wszystkie wdrożone; zawiera też implementacje **usuniętego** kodu (`vol_regime`, `earnings_decay`, `cross_asset`) | Gdy potrzebujesz specyfikacji komponentu albo chcesz przywrócić skasowany kalkulator |

## Archiwum — nie aktualizujemy, trzymamy dla proweniencji

W [`archive/`](archive/). Zamrożone w chwili, w której powstały; ich liczby i wnioski są historyczne.

| Dokument | Czym było | Dlaczego zamrożone |
|---|---|---|
| `archive/progress_log_2026-06_2026-08.md` | Append-only log postępu, 68 wpisów | Zajmował 66% `CLAUDE.md`, który ładuje się do kontekstu każdej sesji. Destylat decyzji jest w `decisions/` |
| `archive/plan_2026_07_28_prediction.md` | Lista robocza toru predykcji, etapy E0–E5 | Kodowo skończona. Niezamknięte pozycje przeniesione (P2-4/P2-5 → „Known issues", D1–D8 → `decisions/06`, kampania pomiarowa → `README.md`) |
| `archive/backlog_2026_07_27.md` | Lista robocza po audycie (Tier 0 → Tier 3) | Zamknięta; pozycje przeniesione do planu predykcji, a stamtąd dalej |
| `archive/review_2026_07_27_external_audit.md` | Audyt zewnętrzny (F1–F7, FLOW-1…9, T0–T3) | Każde twierdzenie zweryfikowane na kodzie/danych — część potwierdzona, część odrzucona |
| `archive/project_brief_for_review.md` | Opis systemu (stan 2026-07-26) dla zewnętrznego recenzenta | Spełnił zadanie. **Do kolejnego przeglądu trzeba wygenerować NOWY brief**, nie odświeżać tego |

## Zasady

1. **Nie tworzymy drugiej listy roboczej.** Bieżący etap i otwarte fronty są w „Next" w `CLAUDE.md`.
   Rzeczy poza bieżącym etapem idą do „Known issues / tech debt" tamże.
2. **Plan, który przestał obowiązywać, ląduje w `archive/` z nagłówkiem**, a jego niezamknięte
   pozycje muszą wcześniej zostać przeniesione — z zachowaniem oryginalnych ID.
3. **Dokument opisujący stan (brief, raport z biegu) nigdy nie jest aktualizowany w miejscu** —
   generujemy nowy. Inaczej traci się możliwość powiedzenia „tak wyglądało wtedy".
4. **Decyzja bez dowodu to preferencja** i tak ma być opisana. Jeśli stoi za nią pomiar, w
   `decisions/` musi być liczba.
5. Wyniki treningów żyją w `reports/` jako pliki JSON — to artefakty, nie dokumentacja.
