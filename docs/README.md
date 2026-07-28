# Mapa dokumentacji

Jeden plik = jedna rola. Jeśli dwa dokumenty mówią co innego, obowiązuje ten wyżej w tabeli.

## Żywe — czytaj i aktualizuj

| Dokument | Rola | Kiedy do niego zaglądać |
|---|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | **Jedyne źródło prawdy o stanie projektu**: co jest zrobione, log postępu (append-only), znane długi techniczne, bieżący kierunek | Zawsze na początku. Jeśli cokolwiek tutaj przeczy CLAUDE.md, wygrywa CLAUDE.md |
| [`../README.md`](../README.md) | Jak system działa i jak go uruchomić: architektura, zdarzenia, bramka aktywacji, bootstrap, reguły ryzyka | Uruchamianie, onboarding, opis dla kogoś z zewnątrz |
| [`plan_2026_07_28_prediction.md`](plan_2026_07_28_prediction.md) | **Bieżący plan roboczy toru predykcji** — etapy E0–E5 z bramkami decyzyjnymi. Następca `backlog_2026_07_27.md` | Planowanie kolejnego kroku w ML/danych. Tu są otwarte decyzje D1–D8 |
| [`ml_integration_plan.md`](ml_integration_plan.md) | **Architektura fazy ML** — dlaczego przekrojowo, dlaczego triple barrier, dlaczego ML jest głosem bez poziomów, jak wygląda rejestr i serwowanie. Zaimplementowane (ML-0…ML-3) | Gdy pytanie brzmi „jak to jest zbudowane i dlaczego". Sekcja bramki (§6) jest **nieaktualna** — obowiązuje `ml-pipeline/src/core/gate.py` |
| [`framework_supplement.md`](framework_supplement.md) | **Referencja 12 komponentów frameworku** (koperta ryzyka, filtr kosztów, monitory degradacji i driftu, alokator reżimowy…). Wszystkie wdrożone; dokument zawiera też referencyjne implementacje **usuniętego** kodu (`vol_regime`, `earnings_decay`, `cross_asset`) | Gdy potrzebujesz specyfikacji komponentu albo chcesz przywrócić skasowany kalkulator |
| [`../Plan_Rozwoju_Systemu_Tradingowego_2.md`](../Plan_Rozwoju_Systemu_Tradingowego_2.md) | **Dokument założycielski** — dekompozycja na 13 serwisów, wzorce Dockerfile/compose, harmonogram 24 tygodni | Kontekst architektoniczny i uzasadnienie podziału na serwisy. Faza budowy jest **wykonana**; harmonogram kalendarzowy zastąpiony planem badawczym powyżej |

## Archiwum — nie aktualizujemy, trzymamy dla proweniencji

W [`archive/`](archive/). Zamrożone w chwili, w której powstały; ich liczby i wnioski są historyczne.

| Dokument | Czym było | Dlaczego zamrożone |
|---|---|---|
| `archive/project_brief_for_review.md` | Samowystarczalny opis systemu (stan 2026-07-26) przygotowany dla zewnętrznego recenzenta | Spełnił zadanie — na jego podstawie powstał audyt. Zawiera nieaktualne progi i wyniki (m.in. starą bramkę aktywacji). **Do kolejnego przeglądu zewnętrznego trzeba wygenerować nowy brief**, nie odświeżać tego |
| `archive/review_2026_07_27_external_audit.md` | Audyt zewnętrzny (F1–F7, FLOW-1…9, T0–T3) | Wejście do backlogu. Każde twierdzenie zostało zweryfikowane na kodzie/danych — wyniki weryfikacji w backlogu i w logu CLAUDE.md |
| `archive/backlog_2026_07_27.md` | Lista robocza po audycie (Tier 0 → Tier 3, decyzje D1–D8) | Tier 0, N1, N2, T1-3 i sonda pojemności **zamknięte**; reszta przeniesiona do `plan_2026_07_28_prediction.md` §13 (mapowanie ID 1:1) i do „Known issues" w CLAUDE.md. Zostawiony jako zapis, jak i dlaczego podjęto tamte decyzje |

## Zasady

1. **Nie tworzymy drugiej listy roboczej.** Jest jedna: `plan_2026_07_28_prediction.md`.
   Rzeczy spoza toru predykcji (infrastruktura, przepływ zdarzeń, ryzyko) idą do „Known issues /
   tech debt" w CLAUDE.md.
2. **Plan, który przestał obowiązywać, ląduje w `archive/` z nagłówkiem**, a jego niezamknięte
   pozycje muszą wcześniej zostać przeniesione — z zachowaniem oryginalnych ID.
3. **Dokument opisujący stan (brief, raport z biegu) nigdy nie jest aktualizowany w miejscu** —
   generujemy nowy. Inaczej traci się możliwość powiedzenia „tak wyglądało wtedy".
4. Wyniki treningów żyją w `reports/` jako pliki JSON — to artefakty, nie dokumentacja.
