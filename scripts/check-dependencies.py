#!/usr/bin/env python3
"""Every runtime import must be a declared runtime dependency.

Tests run in an environment that has the [dev] extras installed, so a package
imported by src/ but declared only under dev - or not declared at all - passes
every check locally and then kills the container at import time. That is
exactly how signal-aggregator shipped with httpx missing: the service crashed
on boot with ModuleNotFoundError while its test suite was green.

Compares the third-party imports under each component's src/ against the
runtime dependencies in its pyproject.toml:

    python scripts/check-dependencies.py

Exit code 1 lists what is missing. Transitive availability does not count -
a package imported directly is declared directly.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parent.parent

# imported module name -> distribution name as written in pyproject
DISTRIBUTION = {
    "aiohttp": "aiohttp",
    "aiosqlite": "aiosqlite",
    "asyncpg": "asyncpg",
    "dotenv": "python-dotenv",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "mlflow": "mlflow",
    "nats": "nats-py",
    "numpy": "numpy",
    "pandas": "pandas",
    "prometheus_client": "prometheus-client",
    "prometheus_fastapi_instrumentator": "prometheus-fastapi-instrumentator",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "redis": "redis",
    "scipy": "scipy",
    "sqlalchemy": "sqlalchemy",
    "structlog": "structlog",
    "tenacity": "tenacity",
    "torch": "torch",
    "trading_common": "trading-common",
    "uvicorn": "uvicorn",
    "yfinance": "yfinance",
}


def third_party_imports(src: Path) -> set[str]:
    """Top-level modules imported under src/, minus stdlib and local packages."""
    modules: set[str] = set()
    for path in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            raise SystemExit(
                f"{path}: nie da sie sparsowac ({exc}). Projekt wymaga Pythona 3.12+ "
                f"- uruchamiasz {sys.version.split()[0]}."
            ) from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in sys.stdlib_module_names and m != "src"}


def declared_runtime(pyproject: Path) -> tuple[str, set[str]]:
    """(package name, declared runtime dependency names) from [project]."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data["project"]
    names = {
        spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for spec in project.get("dependencies", [])
    }
    return project["name"].lower(), names


def check(component: Path) -> list[str]:
    pyproject = component / "pyproject.toml"
    src = component / "src"
    if not pyproject.exists() or not src.exists():
        return []
    own_name, declared = declared_runtime(pyproject)
    problems = []
    for module in sorted(third_party_imports(src)):
        distribution = DISTRIBUTION.get(module)
        if distribution is None:
            problems.append(
                f"{component.name}: import '{module}' nieznany temu skryptowi "
                f"- dopisz go do DISTRIBUTION w scripts/check-dependencies.py"
            )
            continue
        if distribution.lower() == own_name:  # component importing itself
            continue
        if distribution.lower() not in declared:
            problems.append(
                f"{component.name}: importuje '{module}', ale '{distribution}' "
                f"nie jest w [project] dependencies (obraz go nie zainstaluje)"
            )
    return problems


def main() -> int:
    components = [REPO / "shared" / "trading-common"]
    components += sorted(p for p in (REPO / "services").iterdir() if p.is_dir())

    problems: list[str] = []
    for component in components:
        problems.extend(check(component))

    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} brakujacych deklaracji")
        return 1
    print(f"OK - {len(components)} komponentow, kazdy import zadeklarowany")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
