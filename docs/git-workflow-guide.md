# Git Workflow — Przewodnik po plikach konfiguracyjnych

> Notatka edukacyjna: opis każdego pliku związanego z gitem w tym projekcie.
> Aktualizuj gdy dodajesz nowe hooki lub zmieniasz strategię gałęzi.

---

## 1. `.gitignore` — "czego git ma nie widzieć"

Git śledzi każdą zmianę w projekcie. `.gitignore` to lista rzeczy, których celowo **nie** chcesz wersjonować.

### Co ignorujemy i dlaczego

| Wzorzec | Powód |
|---------|-------|
| `.env` | Zawiera hasła i API keys — **NIGDY** nie może trafić do repo |
| `__pycache__/`, `*.pyc` | Skompilowany bytecode Pythona — generowany lokalnie, różni się między maszynami |
| `.venv/`, `venv/` | Folder z bibliotekami — każdy developer odtwarza go przez `pip install` |
| `*.egg-info/`, `dist/`, `build/` | Artefakty budowania pakietu — generowane na żądanie |
| `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/` | Cache narzędzi developerskich — tymczasowe, lokalne |
| `.coverage`, `coverage.xml`, `htmlcov/` | Raporty z testów — generujesz lokalnie, nie commitujesz |
| `.idea/`, `.vscode/` | Ustawienia edytora — każdy developer ma inne preferencje |
| `.DS_Store`, `Thumbs.db` | "Śmieci" systemu operacyjnego (macOS/Windows) |
| `mlruns/`, `mlartifacts/` | Eksperymenty MLflow — mogą ważyć gigabajty |
| `data/`, `*.parquet`, `*.pkl`, `*.joblib` | Surowe dane i modele ML — za duże na git |
| `infrastructure/terraform/*.tfstate` | Stan Terraform — zawiera wrażliwe dane infrastruktury |

**Reguła ogólna**: jeśli coś jest *generowane automatycznie*, *tajne* lub *lokalne dla maszyny* → trafia do `.gitignore`.

### Czego NIE ignorujemy (a mogłoby się wydawać, że tak)

- `.env.example` — szablon z placeholderami (bez prawdziwych haseł) — **commitujemy**
- `pyproject.toml`, `requirements*.txt` — specyfikacja zależności — **commitujemy**

---

## 2. `.gitattributes` — "jak git ma traktować pliki"

Rozwiązuje problem **końców linii (line endings)** — krytyczny na Windows.

### Problem

| System | Znak końca linii | Reprezentacja |
|--------|-----------------|---------------|
| Linux / macOS | LF | `0x0A` |
| Windows | CRLF | `0x0D 0x0A` |

Gdy Windows-owy developer commituje plik, git widzi "zmiany" we wszystkich liniach (bo zmienił się bajt końca linii), choć treść jest identyczna. To zatruwa `git diff` i `git blame`.

### Konfiguracja w projekcie

```gitattributes
* text=auto eol=lf
```

Ta jedna linia robi trzy rzeczy automatycznie:
1. **Repo zawsze przechowuje LF** — niezależnie od systemu operacyjnego
2. **Checkout na Windows** — konwertuje LF → CRLF (dla komfortu edycji)
3. **Commit z Windows** — konwertuje CRLF → LF (przed zapisem do repo)

```gitattributes
*.sh text eol=lf
*.yml text eol=lf
*.py text eol=lf
```

Te pliki **zawsze mają LF**, nawet po checkout na Windows. Dlaczego? Skrypty shell uruchamiane w kontenerach Docker (Linux) muszą mieć LF — CRLF powoduje błąd:
```
bad interpreter: /usr/bin/env^M: No such file or directory
```

```gitattributes
*.png binary
*.jpg binary
*.gz binary
```

Pliki binarne — git nie dotyka końców linii (konwersja uszkodzi plik).

---

## 3. `.pre-commit-config.yaml` — "strażnik przed commitowaniem"

`pre-commit` uruchamia zestaw sprawdzeń **automatycznie zanim tworzysz commit**. Jeśli cokolwiek zawiedzie — commit jest blokowany.

### Jak działa technicznie

```
git commit  →  git uruchamia .git/hooks/pre-commit  →  pre-commit uruchamia hooki  →  commit (jeśli OK)
```

Narzędzie `pre-commit` instaluje swój skrypt w `.git/hooks/pre-commit`. Zarządza własnym wirtualnym środowiskiem z narzędziami (ruff, itp.) — niezależnie od tego co masz zainstalowane globalnie.

### Hooki w projekcie

#### Blok 1: `pre-commit-hooks` (standardowe czystości)

| Hook | Co robi | Przykład problemu |
|------|---------|------------------|
| `trailing-whitespace` | Usuwa spacje na końcu linii | `def foo():   ` |
| `end-of-file-fixer` | Zapewnia pustą linię na końcu pliku | Standard POSIX; brak powoduje ostrzeżenia w edytorach |
| `check-yaml` | Waliduje składnię YAML | Zapomniałeś zamknąć cudzysłów |
| `check-toml` | Waliduje składnię TOML | Błąd w `pyproject.toml` |
| `check-merge-conflict` | Blokuje jeśli są znaczniki konfliktu | `<<<<<<< HEAD` w pliku |
| `check-added-large-files` | Blokuje pliki > 500 KB | Przypadkowo dodałeś plik danych |
| `no-commit-to-branch` | Blokuje bezpośredni commit do `main` | Pracuj na `develop`! |

#### Blok 2: `ruff-pre-commit` (jakość kodu Python)

| Hook | Co robi |
|------|---------|
| `ruff --fix` | Lintuje Python — sprawdza styl, wykrywa błędy, **auto-naprawia** co może |
| `ruff-format` | Formatuje Python — wcięcia, długość linii, cudzysłowy |

### Dlaczego `python -m pre_commit run ruff` zamiast `ruff` bezpośrednio?

`pre-commit` instaluje ruff w swoim własnym cache (`~/.cache/pre-commit/`), nie globalnie. Wywołanie `ruff` bezpośrednio szukałoby ruff w PATH (gdzie go może nie być). Wywołanie przez `python -m pre_commit run ruff` zawsze działa.

### Jak używać ręcznie

```bash
# Uruchom wszystkie hooki na wszystkich plikach
python -m pre_commit run --all-files

# Uruchom tylko ruff na wszystkich plikach
python -m pre_commit run ruff --all-files

# Uruchom tylko na staged plikach (tak jak przy commit)
python -m pre_commit run

# Zainstaluj/zaktualizuj hooki w .git/hooks/
python -m pre_commit install
```

### Kiedy użyć `--no-verify`

Tylko w wyjątkowych sytuacjach (np. inicjalny commit na `main` podczas setup):
```bash
git commit --no-verify -m "chore: initial commit"
```
**Nie używaj na co dzień** — hooki są po to żeby pomagać, nie utrudniać.

---

## 4. Strategia gałęzi (Branch Strategy)

```
main        ← prod-ready; chroniona przez hook; tylko przez PR
  │
  └── develop  ← codzienna praca; tutaj commitujesz
        │
        └── feat/nazwa-feature  ← opcjonalnie dla większych zmian
```

### Zasady

1. **Nigdy nie commituj bezpośrednio do `main`** — hook `no-commit-to-branch` to technicznie egzekwuje
2. **`main` = zawsze działa** — można wdrożyć na produkcję w każdej chwili
3. **`develop` = w trakcie pracy** — może być niestabilny między commitami
4. **Zmiana trafia do `main` przez Pull Request** z `develop` (po code review / CI)

### Typowy workflow

```bash
# Codzienna praca
git switch develop
# ... edytuj pliki ...
git add services/market-data/src/api/routes.py
git commit -m "feat(market-data): add OHLCV ingestion endpoint"
git push origin develop

# Gdy feature jest gotowy → PR na GitHubie: develop → main
```

### Tworzenie nowej gałęzi dla większej funkcji

```bash
git switch develop
git switch -c feat/market-data-websocket
# ... praca ...
git push origin feat/market-data-websocket
# → PR: feat/market-data-websocket → develop
```

---

## 5. Conventional Commits — format wiadomości

Format: **`typ(zakres): opis w trybie rozkazującym`**

### Typy

| Typ | Kiedy używać | Przykład |
|-----|-------------|---------|
| `feat` | Nowa funkcjonalność | `feat(market-data): add WebSocket endpoint` |
| `fix` | Naprawa błędu | `fix(events): replace deprecated datetime.utcnow` |
| `chore` | Utrzymanie, konfiguracja | `chore(deps): upgrade pydantic to 2.6` |
| `docs` | Tylko dokumentacja | `docs(readme): update infrastructure status` |
| `test` | Dodanie/zmiana testów | `test(market-data): add integration tests` |
| `refactor` | Restrukturyzacja bez zmiany zachowania | `refactor(config): extract settings to separate module` |
| `perf` | Optymalizacja wydajności | `perf(feature-engine): cache indicator calculations` |

### Zasady dobrego opisu

- Tryb rozkazujący: `add`, `fix`, `update`, `remove` — **nie** `added`, `fixes`, `updated`
- Max ~72 znaki w pierwszej linii
- Zakres w nawiasie: nazwa serwisu lub komponentu (`market-data`, `infra`, `events`)
- Opis: co zostało zrobione, **nie** jak

### Przykłady z naszego projektu

```
chore(infra): init project scaffold with 9 microservices
fix(events): replace deprecated datetime.utcnow with timezone-aware call
test(market-data): add conftest env setup before src imports
chore(git): add .gitattributes for LF normalization on Windows
```

---

## 6. `.github/workflows/` — automatyzacja po stronie GitHuba

Folder `.github/workflows/` zawiera pliki YAML definiujące **GitHub Actions** — automatyczne zadania uruchamiane przez GitHub w odpowiedzi na zdarzenia (push, PR, ręczne wyzwolenie).

Analogia: pre-commit to strażnik na Twojej lokalnej maszynie. GitHub Actions to strażnik po stronie serwera — sprawdza wszystko jeszcze raz, niezależnie od Ciebie.

### Trzy pliki = trzy etapy pipeline'u

```
push/PR  →  ci.yml (testy)  →  build-images.yml (obrazy Docker)  →  deploy.yml (Kubernetes)
```

---

### `ci.yml` — Continuous Integration (testy i linting)

**Kiedy się uruchamia:**
- Każdy push do `main` lub `develop`
- Każdy Pull Request skierowany do `main`

**Co robi — krok po kroku:**

#### Job 1: `detect-changes` — inteligentne wykrywanie zmian

Zamiast testować wszystkie 9 serwisów przy każdym commicie, CI najpierw sprawdza **które pliki się zmieniły**. Używa akcji `dorny/paths-filter`.

```yaml
filters:
  market-data:
    - 'services/market-data/**'
    - 'shared/**'         # ← jeśli shared się zmienił, testuj też market-data
```

Jeśli zmieniłeś tylko `services/market-data/` → uruchomią się testy tylko dla `market-data`. Jeśli zmieniłeś `shared/` → uruchomią się testy **wszystkich** serwisów (bo wszystkie zależą od shared).

#### Job 2: `test-shared` — testy biblioteki współdzielonej

Zawsze uruchamia się (niezależnie od wykrytych zmian):
1. Instaluje `shared/trading-common[dev]`
2. Lintuje ruffem
3. Sprawdza typy mypem
4. Uruchamia pytest z coverage
5. Wysyła raport do Codecov (śledzenie pokrycia testami w czasie)

#### Job 3: `test-services` — testy serwisów w matrix

Klucz: `strategy.matrix`. GitHub uruchamia **równoległe** jody dla każdego serwisu z listy — 9 serwisów = 9 równoległych procesów. Każdy:
1. Instaluje `shared/trading-common` (bez dev deps — szybciej)
2. Instaluje zależności danego serwisu
3. Lintuje (`ruff check`)
4. Sprawdza typy (`mypy`)
5. Uruchamia testy z coverage
6. Wysyła raport do Codecov z flagą nazwy serwisu

`fail-fast: false` — jeśli testy jednego serwisu padną, pozostałe **kontynuują** (widzisz wszystkie błędy naraz, nie tylko pierwszy).

---

### `build-images.yml` — Build & Push Docker Images

**Kiedy się uruchamia:**
- Push do `main` (produkcyjne obrazy)
- `workflow_dispatch` — ręczne wyzwolenie z poziomu GitHuba (opcjonalnie dla wybranego serwisu)

**Co robi:**

Buduje obrazy Docker dla wszystkich 9 serwisów równolegle (ta sama technika `matrix`) i pushuje je do **GitHub Container Registry** (`ghcr.io`).

#### Tagowanie obrazów

```yaml
tags: |
  type=sha                                          # ghcr.io/.../market-data:abc1234
  type=ref,event=branch                             # ghcr.io/.../market-data:main
  type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}  # :latest tylko z main
```

Każdy obraz dostaje trzy tagi:
- **SHA commitu** — unikalny, niezmienialny; zawsze wiesz dokładnie co jest wdrożone
- **Nazwa gałęzi** — `main` lub `develop`; aktualizowany przy każdym push
- **`latest`** — tylko dla `main`; konwencja "najnowsza stabilna wersja"

#### Cache warstw Dockera

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

GitHub Actions cache — warstwy Docker są cache'owane między runami. Jeśli `requirements.txt` się nie zmienił, warstwa z `pip install` jest pobrana z cache zamiast budowania od nowa. Dramatycznie skraca czas buildu.

#### Uprawnienia

```yaml
permissions:
  contents: read
  packages: write    # ← wymagane do pushowania do ghcr.io
```

Minimalny zakres uprawnień — bezpieczeństwo przez zasadę least privilege.

---

### `deploy.yml` — Deploy to Kubernetes

**Kiedy się uruchamia:**
- Automatycznie po **sukcesie** `build-images.yml` (zdarzenie `workflow_run`)
- Ręcznie przez `workflow_dispatch` z wyborem środowiska: `staging` lub `production`

**Co robi:**

```yaml
if: ${{ github.event.workflow_run.conclusion == 'success' || github.event_name == 'workflow_dispatch' }}
```

Warunek: uruchom deploy tylko jeśli build się powiódł (albo uruchamiasz ręcznie). Zabezpieczenie przed deployem zepsutych obrazów.

#### Kroki deploymentu

1. **`azure/setup-helm@v4`** — instaluje Helm na runnerze GitHuba
2. **`azure/k8s-set-context@v4`** — konfiguruje `kubectl` używając `KUBECONFIG` (sekret w GitHubie — nigdy w kodzie!)
3. **`helm upgrade --install`** — deploy lub aktualizacja całego systemu przez Helm chart

#### Staging vs Production

```bash
VALUES_FILE=./infrastructure/helm/values.yaml
if [ "${{ inputs.environment }}" = "production" ]; then
  VALUES_FILE="$VALUES_FILE -f ./infrastructure/helm/values-prod.yaml"
fi
```

Staging = domyślne wartości. Production = nadpisuje wartości plikiem `values-prod.yaml` (np. większe repliki, inne limity zasobów).

#### Weryfikacja po deploy

```bash
kubectl rollout status deployment/market-data -n trading-system
kubectl rollout status deployment/feature-engine -n trading-system
```

Czeka aż Kubernetes potwierdzi że nowe pody uruchomiły się poprawnie. Jeśli rollout się nie powiedzie w ciągu 5 minut (`--timeout 5m`) — helm automatycznie robi rollback.

---

### Sekrety GitHub Actions

Workflow używa trzech sekretów skonfigurowanych w Settings → Secrets repozytorium:

| Sekret | Używany przez | Do czego |
|--------|--------------|---------|
| `GITHUB_TOKEN` | `build-images.yml` | Automatyczny token GitHuba — push do ghcr.io |
| `KUBECONFIG` | `deploy.yml` | Plik konfiguracyjny kubectl do klastra K8s |
| *(Codecov token)* | `ci.yml` | Opcjonalny token do raportu coverage |

`GITHUB_TOKEN` jest tworzony automatycznie przez GitHub dla każdego runu — nie musisz go konfigurować.

---

## 7. Pełny obraz: jak wszystko się łączy

```
Twój komputer                    GitHub
─────────────────────────────    ─────────────────────────────────────────────
git commit
  └── pre-commit hooks
        ├── trailing-whitespace
        ├── check-yaml
        ├── ruff lint+format
        └── no-commit-to-main

git push origin develop     →    CI (ci.yml) uruchamia się automatycznie
                                   ├── detect-changes: które serwisy?
                                   ├── test-shared: testy shared lib
                                   └── test-services: testy (matrix, równolegle)

PR: develop → main          →    CI uruchamia się ponownie dla PR
  (merge po zielonym CI)

push do main                →    build-images.yml
                                   └── buduje + pushuje 9 obrazów Docker do ghcr.io

build sukces                →    deploy.yml
                                   ├── helm upgrade → staging
                                   └── kubectl rollout status (weryfikacja)
```

---

## 8. Podsumowanie: po co to wszystko?

| Plik / Mechanizm | Chroni przed / Zapewnia |
|-----------------|------------------------|
| `.gitignore` | Przypadkowym commitowaniem sekretów i generowanych plików |
| `.gitattributes` | Bałaganem z line endings na Windows; błędami w Dockerze |
| `.pre-commit-config.yaml` | Złym kodem i literówkami trafiającymi do repo (lokalnie) |
| Branch strategy | Zepsuciem stabilnej wersji projektu |
| Conventional Commits | Czytelną historią git (wiesz co i dlaczego zmieniono) |
| `ci.yml` | Kodem który nie przechodzi testów trafiającym do `main` |
| `build-images.yml` | Zawsze aktualne obrazy Docker w rejestrze po każdym merge |
| `deploy.yml` | Automatycznym, powtarzalnym deploymentem na Kubernetes |

Razem tworzą **sieć bezpieczeństwa**: możesz pracować szybko, bo narzędzia pilnują jakości automatycznie — najpierw lokalnie (pre-commit), potem na serwerze (GitHub Actions).
