"""`.env.example` is a claim about the repository, and it was wrong both ways.

It documented three variables NOTHING reads (ANTHROPIC_API_KEY,
INITIAL_CAPITAL, PAPER_TRADING) and omitted twenty-two that docker-compose
does read — including FETCH_SYMBOLS and REFRESH_SYMBOLS, without which both
schedulers start and have nothing to pull. The stack then comes up "healthy"
and never fetches a bar on its own, so the 30-days-of-paper rule cannot begin
to accrue. The only warning is one INFO line at boot.

This is the failure mode CLAUDE.md's review checklist names directly: an
instruction that does not fail loudly, it just quietly does nothing. A setup
guide can only be trusted if something checks it against the compose file.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
COMPOSE = REPO / "infrastructure" / "docker-compose.yml"
COMPOSE_PROD = REPO / "infrastructure" / "docker-compose.prod.yml"
ENV_EXAMPLE = REPO / ".env.example"

# Injected by the compose runtime or by the templates themselves, never by the
# operator's .env — listing them would invite someone to "configure" them.
NOT_OPERATOR_SETTABLE: frozenset[str] = frozenset()


def compose_variables() -> set[str]:
    """Every ${VAR} the compose files interpolate from the environment."""
    found: set[str] = set()
    for path in (COMPOSE, COMPOSE_PROD):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        found.update(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)", text))
    return found - NOT_OPERATOR_SETTABLE


def documented_variables() -> set[str]:
    """Assignments in .env.example, ignoring comments."""
    return {
        match.group(1)
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if (match := re.match(r"^([A-Z_][A-Z0-9_]*)=", line.strip()))
    }


def test_every_variable_compose_reads_is_documented():
    """A variable compose reads but .env.example omits is invisible to setup.

    FETCH_SYMBOLS was the costly one: the scheduler is enabled by default, so
    the omission produced a system that looked configured and did nothing.
    """
    missing = compose_variables() - documented_variables()
    assert not missing, (
        "compose reads these but .env.example never mentions them: "
        + ", ".join(sorted(missing))
    )


def test_env_example_does_not_promise_settings_nothing_reads():
    """The other direction, which is quieter and therefore worse.

    A documented variable nobody consumes is a promise that setting it does
    something. ANTHROPIC_API_KEY, INITIAL_CAPITAL and PAPER_TRADING each sat
    there for months while no service or compose file read any of them.
    """
    undocumented_consumers = documented_variables() - compose_variables()
    assert not undocumented_consumers, (
        ".env.example documents variables nothing reads: "
        + ", ".join(sorted(undocumented_consumers))
    )


def test_the_two_schedulers_are_documented_with_their_symbol_lists():
    """Enabling a scheduler without naming symbols is the trap, so pin the pair.

    Documenting SCHEDULE_FETCH_ENABLED alone would be worse than documenting
    neither: it reads as "the scheduler is on" while the list it walks is empty.
    """
    documented = documented_variables()
    for flag, symbols in (
        ("SCHEDULE_FETCH_ENABLED", "FETCH_SYMBOLS"),
        ("SCHEDULE_REFRESH_ENABLED", "REFRESH_SYMBOLS"),
    ):
        assert flag in documented and symbols in documented, (
            f"{flag} and {symbols} must be documented together — the flag alone "
            "describes a scheduler that runs over nothing"
        )
