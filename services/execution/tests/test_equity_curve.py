"""Testy historii kapitału brokera — szeregu, którego dotąd nikt nie trzymał."""

from datetime import date

import pytest

from src.core.paper_broker import PaperBroker


class FakeClock:
    def __init__(self, start: date) -> None:
        self.today = start

    def __call__(self) -> date:
        return self.today

    def advance(self, days: int = 1) -> None:
        self.today = date.fromordinal(self.today.toordinal() + days)


def broker(clock: FakeClock | None = None, **kwargs) -> PaperBroker:
    return PaperBroker(initial_cash=100_000.0, clock=clock or FakeClock(date(2024, 1, 1)), **kwargs)


def test_a_fresh_broker_already_has_its_starting_point():
    """Without a seed the first chart point would be whatever the first trade
    happened to leave behind, so the curve would start mid-story."""
    curve = broker().equity_curve()
    assert len(curve) == 1
    assert curve[0]["equity"] == pytest.approx(100_000.0)
    assert curve[0]["date"] == "2024-01-01"


def test_many_mutations_in_one_session_collapse_to_ONE_point():
    """Recording every mutation would make the retained window depend on how
    many symbols were marked that day — a busy day would push a quiet week off
    the end of an 'eight year' history."""
    b = broker()
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    for price in (101.0, 102.0, 103.0):
        b.mark("AAPL", price)
    curve = b.equity_curve()
    assert len(curve) == 1
    # ...and the single point carries the NEWEST value, not the first.
    assert curve[0]["equity"] == pytest.approx(b.equity)


def test_a_new_session_appends_a_new_point():
    clock = FakeClock(date(2024, 1, 1))
    b = broker(clock)
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    clock.advance()
    b.mark("AAPL", 120.0)
    clock.advance()
    b.mark("AAPL", 90.0)

    curve = b.equity_curve()
    assert [p["date"] for p in curve] == ["2024-01-01", "2024-01-02", "2024-01-03"]
    assert curve[1]["equity"] > curve[0]["equity"]
    assert curve[2]["equity"] < curve[1]["equity"]


def test_the_curve_is_bounded_and_drops_the_OLDEST_points():
    clock = FakeClock(date(2024, 1, 1))
    b = broker(clock, equity_history_limit=5)
    b.fill("o1", "AAPL", "BUY", 1, 100.0)
    for _ in range(10):
        clock.advance()
        b.mark("AAPL", 100.0)

    curve = b.equity_curve()
    assert len(curve) == 5
    # The window ends at today, not at the day trading started.
    assert curve[-1]["date"] == clock.today.isoformat()


def test_limit_keeps_the_most_recent_points():
    """Truncating from the front would show the start of trading forever and
    never today."""
    clock = FakeClock(date(2024, 1, 1))
    b = broker(clock)
    b.fill("o1", "AAPL", "BUY", 1, 100.0)
    for _ in range(9):
        clock.advance()
        b.mark("AAPL", 100.0)

    assert len(b.equity_curve()) == 10
    tail = b.equity_curve(limit=3)
    assert len(tail) == 3
    assert tail[-1]["date"] == clock.today.isoformat()


def test_marking_a_symbol_we_do_not_hold_records_nothing():
    """`mark` returns early for a flat symbol; it must not manufacture a point
    and make an idle day look like a session that traded."""
    b = broker()
    before = b.equity_curve()
    b.mark("TSLA", 250.0)
    assert b.equity_curve() == before


def test_a_duplicate_fill_does_not_move_the_curve():
    b = broker()
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    after_first = b.equity_curve()[-1]["equity"]
    assert b.fill("o1", "AAPL", "BUY", 10, 100.0) is None
    assert b.equity_curve()[-1]["equity"] == pytest.approx(after_first)


# --- persistence ----------------------------------------------------------


def test_the_curve_survives_a_snapshot_round_trip():
    clock = FakeClock(date(2024, 1, 1))
    b = broker(clock)
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    clock.advance()
    b.mark("AAPL", 130.0)
    snapshot = b.snapshot()

    restored = broker(FakeClock(date(2024, 1, 2)))
    restored.restore(snapshot)
    assert restored.equity_curve() == b.equity_curve()


def test_a_snapshot_written_before_the_curve_existed_still_restores():
    """The persisted layout changed; a container restarting on an old snapshot
    must come back up and simply start its history now."""
    b = broker()
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    legacy = b.snapshot()
    del legacy["equity_curve"]

    restored = broker()
    restored.restore(legacy)
    assert restored.equity_curve() == []
    assert restored.cash == pytest.approx(b.cash)


def test_restoring_then_trading_continues_the_same_series():
    clock = FakeClock(date(2024, 1, 1))
    b = broker(clock)
    b.fill("o1", "AAPL", "BUY", 10, 100.0)
    snapshot = b.snapshot()

    later = FakeClock(date(2024, 1, 5))
    restored = broker(later)
    restored.restore(snapshot)
    restored.mark("AAPL", 110.0)

    dates = [p["date"] for p in restored.equity_curve()]
    assert dates == ["2024-01-01", "2024-01-05"]
