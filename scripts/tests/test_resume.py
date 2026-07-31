"""A 486-symbol, 20-year backfill runs for hours — it has to survive Ctrl-C.

Storage upserts idempotently on (symbol, interval, ts), so an interruption was
never a correctness problem; it was a cost problem, and the cost is the whole
download. These tests pin that a resumed run skips what it already has, that
progress survives a kill between symbols, and that changing the requested depth
does NOT silently reuse a shallower backfill's progress.
"""

import importlib.util
import json
import pathlib
from datetime import date

SPEC = importlib.util.spec_from_file_location(
    "bootstrap", pathlib.Path(__file__).resolve().parents[1] / "bootstrap-universe.py"
)
boot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boot)

START, END = date(2006, 1, 3), date(2026, 1, 2)


class FakeMarket:
    """Records which symbols were actually fetched; can fail on demand."""

    def __init__(self, fail_on: set[str] | None = None, raise_on: str | None = None):
        self.fetched: list[str] = []
        self.fail_on = fail_on or set()
        self.raise_on = raise_on

    def request(self, method, url, payload=None, timeout=0):
        symbol = url.split("/fetch/")[1].split("?")[0]
        if symbol == self.raise_on:
            raise OSError("connection reset")
        self.fetched.append(symbol)
        if symbol in self.fail_on:
            return 502, {"detail": "upstream unavailable"}
        return 200, {"rows": 5000}


def run(monkeypatch, market, symbols, progress, years=20.0, done=None):
    monkeypatch.setattr(boot, "_request", market.request)
    return boot.backfill(
        "http://md",
        symbols,
        START,
        END,
        pause_s=0.0,
        already_done=done if done is not None else boot.load_progress(progress, years),
        progress_path=progress,
        years=years,
    )


def test_a_resumed_run_fetches_only_what_is_missing(tmp_path, monkeypatch):
    """The whole point: Ctrl-C at symbol 3 of 5, re-run, and only 2 are fetched."""
    progress = str(tmp_path / "progress.json")
    first = FakeMarket()
    run(monkeypatch, first, ["A", "B", "C"], progress)
    assert first.fetched == ["A", "B", "C"]

    second = FakeMarket()
    rows = run(monkeypatch, second, ["A", "B", "C", "D", "E"], progress)
    assert second.fetched == ["D", "E"], "already-fetched symbols were re-downloaded"
    assert set(rows) == {"A", "B", "C", "D", "E"}, (
        "resumed symbols vanished from the result"
    )


def test_progress_is_written_after_every_symbol_not_at_the_end(tmp_path, monkeypatch):
    """A kill -9 mid-run must not lose the preceding hour, so the file has to be
    current between symbols rather than flushed once at the end."""
    progress = str(tmp_path / "progress.json")
    seen: list[int] = []

    market = FakeMarket()
    original = market.request

    def spy(method, url, payload=None, timeout=0):
        result = original(method, url, payload, timeout)
        saved = (
            json.loads(pathlib.Path(progress).read_text())
            if pathlib.Path(progress).exists()
            else {}
        )
        seen.append(len(saved.get("rows_by_symbol", {})))
        return result

    market.request = spy
    run(monkeypatch, market, ["A", "B", "C"], progress)
    # before A is recorded the file is empty; by the time C is requested, A and B are in it
    assert seen == [0, 1, 2], seen


def test_a_symbol_that_failed_is_retried_on_the_next_run(tmp_path, monkeypatch):
    """Only a symbol that actually stored rows counts as done. A 502 or a
    dropped connection must come back around, or a resumed run would quietly
    inherit the gap."""
    progress = str(tmp_path / "progress.json")
    run(monkeypatch, FakeMarket(fail_on={"B"}), ["A", "B", "C"], progress)
    retry = FakeMarket()
    run(monkeypatch, retry, ["A", "B", "C"], progress)
    assert retry.fetched == ["B"]

    dropped = str(tmp_path / "dropped.json")
    run(monkeypatch, FakeMarket(raise_on="B"), ["A", "B", "C"], dropped)
    again = FakeMarket()
    run(monkeypatch, again, ["A", "B", "C"], dropped)
    assert again.fetched == ["B"]


def test_asking_for_a_different_depth_starts_clean(tmp_path, monkeypatch):
    """A 6-year backfill's progress must not be read as a 20-year one — that
    would leave the universe silently short of history, which is exactly the
    failure the training data contract exists to catch late."""
    progress = str(tmp_path / "progress.json")
    run(monkeypatch, FakeMarket(), ["A", "B"], progress, years=6.0)
    deeper = FakeMarket()
    run(monkeypatch, deeper, ["A", "B"], progress, years=20.0)
    assert deeper.fetched == ["A", "B"]


def test_resume_can_be_turned_off(tmp_path, monkeypatch):
    progress = str(tmp_path / "progress.json")
    run(monkeypatch, FakeMarket(), ["A", "B"], progress)
    forced = FakeMarket()
    # --no-resume passes progress_path=None, so nothing is loaded or recorded
    monkeypatch.setattr(boot, "_request", forced.request)
    boot.backfill(
        "http://md", ["A", "B"], START, END, 0.0, already_done=None, progress_path=None
    )
    assert forced.fetched == ["A", "B"]


def test_a_corrupt_progress_file_is_ignored_not_fatal(tmp_path):
    """Unreadable progress is no progress. Refusing to start because a JSON file
    got truncated would turn a cosmetic problem into a blocked campaign."""
    path = tmp_path / "progress.json"
    path.write_text("{ this is not json")
    assert boot.load_progress(str(path), 20.0) == {}
    assert boot.load_progress(str(tmp_path / "absent.json"), 20.0) == {}
    assert boot.load_progress(None, 20.0) == {}


def test_progress_survives_a_crash_during_the_write(tmp_path):
    """Written via a temp file + atomic replace, so a crash mid-write leaves the
    previous complete file rather than a truncated one."""
    path = tmp_path / "nested" / "progress.json"
    boot.save_progress(str(path), 20.0, {"A": 100})
    boot.save_progress(str(path), 20.0, {"A": 100, "B": 200})
    assert boot.load_progress(str(path), 20.0) == {"A": 100, "B": 200}
    assert not (tmp_path / "nested" / "progress.json.tmp").exists()


# --- symbol parsing (a Windows PowerShell trap) ----------------------------


def test_symbols_split_on_spaces_as_well_as_commas():
    """Windows PowerShell parses a bare `A,B,C` as an ARRAY, and make.ps1 binds
    it to a [string[]] parameter, which stringifies back space-joined. So
    `--symbols SEE,K,HES` arrived as one token "SEE K HES": the run asked
    market-data for a ticker with spaces in its name and died on InvalidURL,
    after cheerfully announcing "Backfilling 1 symbols".
    """
    expected = ["SEE", "K", "HES", "CTRA"]
    assert boot.split_symbols("SEE,K,HES,CTRA") == expected
    assert boot.split_symbols("SEE K HES CTRA") == expected  # what PowerShell sends
    assert boot.split_symbols("SEE, K  HES ,CTRA") == expected


def test_symbol_parsing_normalises_case_and_drops_empties():
    assert boot.split_symbols(" see,,k ,  ") == ["SEE", "K"]
    assert boot.split_symbols("") == []
    assert boot.split_symbols("   ") == []


# --- what the report keeps about a failure ---------------------------------


def test_a_failed_symbol_records_why_not_just_that_it_failed(tmp_path, monkeypatch):
    """A bare list of names cannot distinguish a stale ticker from a broken
    service, and the real run produced exactly that: 41 names, no causes, on a
    universe that deliberately contains delisted companies. The console printed
    the reason and the report — the thing actually reviewed — dropped it.
    """
    progress = str(tmp_path / "progress.json")
    failures: dict[str, str] = {}
    market = FakeMarket(fail_on={"B"}, raise_on="C")
    monkeypatch.setattr(boot, "_request", market.request)

    rows = boot.backfill(
        "http://md",
        ["A", "B", "C"],
        START,
        END,
        pause_s=0.0,
        already_done={},
        progress_path=progress,
        years=20.0,
        failures=failures,
    )

    assert set(rows) == {"A"}
    assert set(failures) == {"B", "C"}
    assert "502" in failures["B"] and "upstream unavailable" in failures["B"]
    assert "OSError" in failures["C"] and "connection reset" in failures["C"]


def test_collecting_failures_stays_optional(tmp_path, monkeypatch):
    """The parameter is additive: a caller that does not care must still run."""
    market = FakeMarket(fail_on={"B"})
    monkeypatch.setattr(boot, "_request", market.request)
    rows = boot.backfill(
        "http://md",
        ["A", "B"],
        START,
        END,
        pause_s=0.0,
        already_done={},
        progress_path=str(tmp_path / "p.json"),
        years=20.0,
    )
    assert set(rows) == {"A"}
