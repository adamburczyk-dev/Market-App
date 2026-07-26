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
    code, out = compose("ps", "--format", "json")
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
        print("brak dzialajacych kontenerow (czy stack jest podniesiony?)")
        return
    for row in sorted(rows, key=lambda r: r.get("Service", "")):
        name = row.get("Service", "?")
        state = row.get("State", "?")
        health = row.get("Health") or "-"
        flag = "OK " if state == "running" and health in ("healthy", "-") else "!! "
        print(f"  {flag}{name:<20} {state:<10} health={health}")


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


def check_database() -> None:
    section("BAZA DANYCH")
    user = env_value("DB_USER") or "trader"
    db = env_value("DB_NAME") or "trading_db"
    queries = {
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
    }
    for label, sql in queries.items():
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
            timeout=30,
        )
        value = out.strip().splitlines()[-1] if out.strip() else ""
        flag = "OK " if code == 0 else "!! "
        print(f"  {flag}{label:<18}: {value[:200]}")


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
    section("PROBNY FETCH (market-data)")
    url = (
        "http://localhost:8001/api/v1/market-data/fetch/AAPL"
        "?interval=1d&start_date=2024-01-02&end_date=2024-01-10"
    )
    status, body = http("POST", url, timeout=120)
    print(f"  POST /fetch/AAPL -> HTTP {status}")
    try:
        parsed = json.loads(body)
        print(f"  odpowiedz: {json.dumps(parsed, ensure_ascii=True)[:600]}")
    except json.JSONDecodeError:
        print(f"  odpowiedz (nie-JSON): {body[:600]!r}")


def recent_errors() -> None:
    section("OSTATNIE BLEDY W LOGACH market-data")
    code, out = compose("logs", "--tail", "400", "market-data", timeout=60)
    if code != 0:
        print(out.strip()[:400])
        return
    lines = out.splitlines()
    hits = [
        i
        for i, line in enumerate(lines)
        if "Traceback" in line or "Error" in line or "error" in line.lower()
    ]
    if not hits:
        print("  brak linii z bledami w ostatnich 400 liniach")
        return
    start = max(0, hits[-1] - 3)
    for line in lines[start : start + 30]:
        print("  " + line[:200])


def main() -> int:
    print("DIAGNOSTYKA STACKU - wklej cala ta sekcje w rozmowie")
    for step in (
        check_environment,
        check_containers,
        check_services,
        check_database,
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
