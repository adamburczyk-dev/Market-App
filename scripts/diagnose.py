#!/usr/bin/env python3
"""One-shot diagnostics for a running stack - output is meant to be pasted.

Collects, in a single pass: container health, per-service /health and /ready,
the database schema state, Redis and NATS reachability, a live fetch probe
with its real error message, and the most recent traceback from market-data.

Standard library only, so it runs anywhere Python 3 does:

    python scripts/diagnose.py

Secrets are never printed: .env is read only to authenticate the Redis probe.
Output is deliberately ASCII-only so it survives every Windows code page.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "infrastructure" / "docker-compose.yml"
ENV_FILE = REPO / ".env"

# name -> host port (as published by docker-compose.yml)
SERVICES = {
    "market-data": 8001,
    "feature-engine": 8002,
    "strategy": 8003,
    "backtest": 8004,
    "ml-pipeline": 8005,
    "risk-mgmt": 8006,
    "execution": 8007,
    "notification": 8008,
    "fundamental-data": 8009,
    "macro-data": 8010,
    "company-classifier": 8011,
    "signal-aggregator": 8012,
    "dashboard": 8501,
}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def run(args: list[str], timeout: float = 60.0) -> tuple[int, str]:
    """Run a command, never raise; returns (exit code, combined output)."""
    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, f"nie znaleziono polecenia: {args[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"przekroczono limit czasu: {' '.join(args)}"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash
        return 1, f"{type(exc).__name__}: {exc}"


def compose(*args: str, timeout: float = 60.0) -> tuple[int, str]:
    return run(
        ["docker", "compose", "-f", str(COMPOSE), "--env-file", str(ENV_FILE), *args],
        timeout=timeout,
    )


def http(method: str, url: str, timeout: float = 10.0) -> tuple[int, str]:
    """Returns (status, body). Status 0 means the connection itself failed."""
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def env_value(key: str) -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith(f"{key}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip()
    return None


def check_environment() -> None:
    section("SRODOWISKO")
    print(f"python      : {sys.version.split()[0]} ({sys.platform})")
    print(f"repo        : {REPO}")
    print(f".env obecny : {ENV_FILE.exists()}")
    for key in ("DB_PASSWORD", "REDIS_PASSWORD"):
        value = env_value(key)
        # nigdy nie drukujemy wartosci - tylko czy jest ustawiona
        print(f"{key:<12}: {'ustawione' if value else 'BRAK'}")
    code, out = run(["docker", "version", "--format", "{{.Server.Version}}"])
    print(
        f"docker      : {out.strip() if code == 0 else 'NIEDOSTEPNY -> ' + out.strip()[:120]}"
    )


def check_containers() -> None:
    section("KONTENERY")
    # --all, bo bez tego kontener, ktory sie wywrocil, po prostu ZNIKA z listy
    # i wyglada jak nieistniejacy zamiast jak awaria.
    code, out = compose("ps", "--all", "--format", "json")
    if code != 0:
        print(out.strip()[:800])
        return
    rows = []
    for line in out.strip().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        print("brak kontenerow (czy stack jest podniesiony?)")
        return
    seen = set()
    for row in sorted(rows, key=lambda r: r.get("Service", "")):
        name = row.get("Service", "?")
        seen.add(name)
        state = row.get("State", "?")
        health = row.get("Health") or "-"
        exit_code = row.get("ExitCode")
        flag = "OK " if state == "running" and health in ("healthy", "-") else "!! "
        suffix = (
            f" exit={exit_code}" if state != "running" and exit_code is not None else ""
        )
        print(f"  {flag}{name:<20} {state:<10} health={health}{suffix}")
    missing = sorted(set(SERVICES) - seen)
    if missing:
        print(f"  !! BRAK KONTENERA w ogole: {', '.join(missing)}")
        print(
            "     (nie zbudowany albo usuniety - sprawdz: docker compose logs <serwis>)"
        )


def check_services() -> None:
    section("SERWISY: /health i /ready")
    for name, port in SERVICES.items():
        status, body = http("GET", f"http://localhost:{port}/health", timeout=5)
        if status == 0:
            print(f"  !! {name:<20} nieosiagalny ({body[:60]})")
            continue
        ready_status, ready_body = http(
            "GET", f"http://localhost:{port}/ready", timeout=5
        )
        detail = ""
        try:
            checks = json.loads(ready_body).get("checks")
            if checks:
                detail = " " + json.dumps(checks, separators=(",", ":"))
        except (json.JSONDecodeError, AttributeError):
            detail = " " + ready_body[:60].replace("\n", " ")
        flag = "OK " if ready_status == 200 else "!! "
        print(f"  {flag}{name:<20} health={status} ready={ready_status}{detail}")


def psql(sql: str, timeout: float = 30.0) -> tuple[int, str]:
    user = env_value("DB_USER") or "trader"
    db = env_value("DB_NAME") or "trading_db"
    code, out = compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        user,
        "-d",
        db,
        "-tAc",
        sql,
        timeout=timeout,
    )
    lines = out.strip().splitlines()
    if not lines:
        return code, ""
    if code != 0:
        # psql prints "ERROR: ..." first and a caret marker last; -tAc puts the
        # VALUE last on success. Taking the last line either way turned every
        # failed query into a lone "^" pointing at nothing.
        return code, next((ln for ln in lines if "ERROR" in ln), lines[0])
    return code, lines[-1]


def report_queries(queries: dict[str, str]) -> None:
    for label, sql in queries.items():
        code, value = psql(sql)
        print(f"  {'OK ' if code == 0 else '!! '}{label:<18}: {value[:200]}")


def check_database() -> None:
    section("BAZA DANYCH")
    report_queries(
        {
            "rozszerzenia": "SELECT string_agg(extname, ', ') FROM pg_extension",
            "schematy": (
                "SELECT string_agg(nspname, ', ') FROM pg_namespace "
                "WHERE nspname NOT LIKE 'pg_%' AND nspname <> 'information_schema'"
            ),
            "tabela ohlcv": (
                "SELECT count(*)::text FROM information_schema.tables "
                "WHERE table_schema='market_data' AND table_name='ohlcv'"
            ),
            "wierszy w ohlcv": "SELECT count(*)::text FROM market_data.ohlcv",
            "symboli w ohlcv": "SELECT count(DISTINCT symbol)::text FROM market_data.ohlcv",
            "zakres ohlcv": (
                "SELECT coalesce(min(ts)::date::text || ' .. ' || max(ts)::date::text, 'pusto') "
                "FROM market_data.ohlcv"
            ),
        }
    )


def check_timescale() -> None:
    """Stan hypertabeli i KOMPRESJI.

    To jedyna wlasciwosc tej bazy, ktorej nie odtworzy zaden test na zwyklym
    Postgresie: init-db.sql wlacza kompresje z polityka 7 dni, a market-data z
    zalozenia PRZEPISUJE historie (naprawa po splicie odswieza kazdy bar, backfill
    da sie powtorzyc, sonda ponizej pisze). Zapis do skompresowanego chunka to
    inna sciezka niz zapis do swiezego, wiec raport musi pokazywac, ile chunkow
    jest skompresowanych - inaczej awaria zapisu na starym oknie nie ma w tym
    raporcie zadnego wytlumaczenia.
    """
    section("TIMESCALEDB (hypertabela i kompresja)")
    # Gate na obecnosc rozszerzenia, a NIE `CASE WHEN to_regclass(...)` w jednym
    # zapytaniu: Postgres rozwiazuje nazwy relacji przy PARSOWANIU, wiec galaz
    # else-owa i tak wywala sie bledem katalogu na bazie bez timescaledb.
    # (Sprawdzone na prawdziwym Postgresie - pierwsza wersja tej sekcji dawala
    # trzy razy "relation ... does not exist" zamiast czytelnego "n/d".)
    code, present = psql(
        "SELECT count(*)::text FROM pg_extension WHERE extname='timescaledb'"
    )
    if code != 0:
        print(f"  !! nie udalo sie sprawdzic rozszerzenia: {present[:200]}")
        return
    if present.strip() != "1":
        print(
            "  -  rozszerzenie timescaledb nieobecne - tabela ohlcv jest zwykla tabela"
        )
        return
    report_queries(
        {
            "hypertabela": (
                "SELECT count(*)::text FROM timescaledb_information.hypertables "
                "WHERE hypertable_name='ohlcv'"
            ),
            "chunkow": (
                "SELECT count(*)::text FROM timescaledb_information.chunks "
                "WHERE hypertable_name='ohlcv'"
            ),
            # Osobnym zapytaniem: nazwa kolumny zalezy od wersji rozszerzenia,
            # a jesli ta jedna linia padnie, liczba chunkow ma sie i tak pokazac.
            "skompresowanych": (
                "SELECT count(*)::text FROM timescaledb_information.chunks "
                "WHERE hypertable_name='ohlcv' AND is_compressed"
            ),
            "zadania": (
                "SELECT coalesce(string_agg(proc_name || ' co ' || "
                "coalesce(schedule_interval::text,'?'), ', '), 'brak') "
                "FROM timescaledb_information.jobs WHERE hypertable_name='ohlcv'"
            ),
        }
    )
    print(
        "     (skompresowany chunk to inna sciezka zapisu niz swiezy - jesli zapis\n"
        "      do starego okna zawodzi, ta liczba jest pierwszym podejrzanym)"
    )


def check_redis_nats() -> None:
    section("REDIS i NATS")
    password = env_value("REDIS_PASSWORD")
    args = ["exec", "-T", "redis", "redis-cli"]
    if password:
        args += ["-a", password]  # przekazane, nigdy nie drukowane
    _code, out = compose(*args, "ping", timeout=30)
    answer = "PONG" if "PONG" in out else out.strip()[:120]
    print(f"  {'OK ' if 'PONG' in out else '!! '}redis ping        : {answer}")

    status, body = http("GET", "http://localhost:8222/jsz?streams=1", timeout=5)
    if status != 200:
        print(f"  !! nats monitoring   : nieosiagalny ({body[:80]})")
        return
    try:
        data = json.loads(body)
        streams = data.get("streams", 0)
        messages = data.get("messages", 0)
        names = [
            s.get("name")
            for account in data.get("account_details", [])
            for s in account.get("stream_detail", [])
        ]
        print(f"  OK nats jetstream    : strumieni={streams} wiadomosci={messages}")
        if names:
            print(
                f"     strumienie        : {', '.join(sorted(n for n in names if n))}"
            )
    except (json.JSONDecodeError, AttributeError) as exc:
        print(f"  !! nats monitoring   : nieczytelna odpowiedz ({exc})")


def probe_fetch() -> None:
    """Sonda ZAPISU - i celowo dwa razy pod rzad, tym samym oknem.

    Pierwsze wywolanie sprawdza cala sciezke pobrania (siec, walidacja, zapis,
    cache, event). Drugie sprawdza cos innego i wazniejszego: czy ponowne
    wrzucenie TYCH SAMYCH danych przechodzi. Ma przechodzic - klucz naturalny
    (symbol, interval, ts) jest po to, zeby zapis byl idempotentny, a caly
    harmonogram przyrostowy swiadomie pobiera zakladke na juz posiadane bary.
    Jednorazowa sonda nie odroznia "dziala" od "dziala tylko raz".
    """
    section("PROBNY FETCH (market-data) - UWAGA: to zapisuje do bazy")
    url = (
        "http://localhost:8001/api/v1/market-data/fetch/AAPL"
        "?interval=1d&start_date=2024-01-02&end_date=2024-01-10"
    )
    print("  okno: AAPL 1d 2024-01-02..2024-01-10 (te same bary, dwa razy)")
    results = []
    for attempt in (1, 2):
        status, body = http("POST", url, timeout=120)
        results.append(status)
        print(f"  proba {attempt}: POST /fetch/AAPL -> HTTP {status}")
        try:
            parsed = json.loads(body)
            print(f"    odpowiedz: {json.dumps(parsed, ensure_ascii=True)[:600]}")
        except json.JSONDecodeError:
            print(f"    odpowiedz (nie-JSON): {body[:600]!r}")
    if results[0] == 200 and results[1] != 200:
        print(
            "  !! ZAPIS NIE JEST IDEMPOTENTNY: pierwszy przeszedl, drugi nie. "
            "Kazde ponowne pobranie tego okna (zakladka harmonogramu, powtorzony "
            "backfill, naprawa po splicie) bedzie sie tak konczyc."
        )
    elif results == [200, 200]:
        print("  OK ponowny zapis tych samych danych przechodzi (idempotentny)")


def looks_like_error(line: str) -> bool:
    low = line.lower()
    return "traceback" in low or "error" in low or '"exception"' in low


def recent_errors() -> None:
    section("OSTATNIE BLEDY W LOGACH market-data")
    code, out = compose("logs", "--tail", "400", "market-data", timeout=60)
    if code != 0:
        print(out.strip()[:400])
        return
    lines = out.splitlines()
    hits = [i for i, line in enumerate(lines) if looks_like_error(line)]
    if not hits:
        print("  brak linii z bledami w ostatnich 400 liniach")
        return
    # Kontekst wokol trafienia jest po to, zeby bylo widac, co dzialo sie tuz
    # przed bledem - ale bez oznaczenia wygladal jak lista bledow, wiec zwykle
    # "200 OK" czytalo sie jako awarie. Trafienia sa teraz oznaczone.
    print(f"  znaleziono {len(hits)} linii z bledem w ostatnich {len(lines)}")
    print("  ('>>' = dopasowana linia; linie bez znacznika to KONTEKST, nie bledy)")
    hitset = set(hits)
    start = max(0, hits[-1] - 3)
    for i in range(start, min(len(lines), start + 30)):
        marker = ">>" if i in hitset else "  "
        # Traceback jedzie teraz w JEDNEJ linii jako pole "exception" (JSON),
        # wiec obciecie do 200 znakow wyrzucalo dokladnie to, po co tu jestesmy.
        print(f"  {marker} {lines[i][: 2000 if i in hitset else 200]}")


def main() -> int:
    print("DIAGNOSTYKA STACKU - wklej cala ta sekcje w rozmowie")
    for step in (
        check_environment,
        check_containers,
        check_services,
        check_database,
        check_timescale,
        check_redis_nats,
        probe_fetch,
        recent_errors,
    ):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - jedna sekcja nie moze zabic raportu
            print(f"\n!! sekcja {step.__name__} przerwana: {type(exc).__name__}: {exc}")
    print("\n=== KONIEC ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
