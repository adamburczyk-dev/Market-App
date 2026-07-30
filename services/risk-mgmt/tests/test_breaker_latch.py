"""The two non-negotiable rules the breaker was not actually enforcing.

    "Daily loss > 5% → automatic trading halt UNTIL NEXT DAY"
    "Drawdown > 15% → flatten all positions, REQUIRE HUMAN RESTART"

A breaker that only reports the present metric satisfies neither. It lifted the
daily halt the moment an intraday bounce took the loss back under 5% — on the
same day — and it forgot a BLACK as soon as drawdown recovered, so the system
resumed trading after a catastrophic loss with nobody having looked at it.

Both failures can only occur after something already went badly wrong, which is
exactly why they were never noticed: the recovery looks like permission.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.events import CircuitBreakerLevel

from src.core.circuit_breaker import CircuitBreaker

DAY1 = datetime(2026, 7, 30, 15, 0, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)


def breaker(now: datetime = DAY1) -> CircuitBreaker:
    clock = {"t": now}
    b = CircuitBreaker(clock=lambda: clock["t"])
    b._clock_box = clock  # type: ignore[attr-defined]  # tests advance time
    return b


def advance(b: CircuitBreaker, to: datetime) -> None:
    b._clock_box["t"] = to  # type: ignore[attr-defined]


# --- BLACK: requires a human ------------------------------------------------


def test_black_does_not_clear_when_the_drawdown_recovers():
    """The headline defect. A 16% drawdown trips BLACK and flattens the book;
    the market bounces to 5% and the old breaker resumed trading by itself."""
    b = breaker()
    assert b.evaluate(0.16, 0.0).level is CircuitBreakerLevel.BLACK
    assert b.evaluate(0.05, 0.0).level is CircuitBreakerLevel.BLACK
    assert b.evaluate(0.0, 0.0).level is CircuitBreakerLevel.BLACK
    assert b.is_tripped and b.latched


def test_a_human_reset_clears_it_once_the_book_is_back_inside_the_limit():
    b = breaker()
    b.evaluate(0.16, 0.0)
    b.evaluate(0.04, 0.0)  # exposure reduced
    result = b.reset(0.04, 0.0)
    assert result.cleared and not b.latched
    assert b.evaluate(0.04, 0.0).level is None


def test_a_reset_is_refused_while_the_breach_still_stands():
    """Clearing the latch mid-breach would re-trip on the next update anyway,
    but in between it would let new orders through at the worst moment. The
    reset acknowledges a recovery; it does not perform one."""
    b = breaker()
    b.evaluate(0.16, 0.0)
    result = b.reset(0.16, 0.0)
    assert not result.cleared
    assert "reduce exposure first" in result.reason
    assert b.latched and b.is_tripped


def test_resetting_an_untripped_breaker_is_a_no_op_not_an_error():
    b = breaker()
    b.evaluate(0.01, 0.0)
    assert b.reset(0.01, 0.0).cleared is False


# --- the restart loophole ---------------------------------------------------


def test_a_restart_is_not_a_human_reset():
    """The easiest possible way to bypass "require human restart" is to restart
    the container, and nobody would notice. The latch is persisted, so a
    restored breaker comes back latched even though the metric has recovered."""
    b = breaker()
    b.evaluate(0.16, 0.0)
    snapshot = b.snapshot()

    restarted = breaker()
    restarted.restore(snapshot)
    restarted.evaluate(0.02, 0.0)  # drawdown has since recovered
    assert restarted.latched
    assert restarted.level is CircuitBreakerLevel.BLACK


def test_a_cleared_latch_is_also_persisted():
    """...and the converse: once an operator clears it, a restart must not
    resurrect the halt."""
    b = breaker()
    b.evaluate(0.16, 0.0)
    b.reset(0.04, 0.0)

    restarted = breaker()
    restarted.restore(b.snapshot())
    restarted.evaluate(0.04, 0.0)
    assert not restarted.latched
    assert restarted.level is None


def test_restoring_nothing_leaves_a_clean_breaker():
    b = breaker()
    b.restore(None)
    b.restore({})
    assert not b.latched and b.halted_session is None


# --- RED: holds for the session ---------------------------------------------


def test_the_daily_halt_holds_for_the_rest_of_the_day():
    """ "Until next day" is a statement about time, not about the metric. An
    intraday bounce under the threshold used to lift the halt immediately."""
    b = breaker()
    assert b.evaluate(0.0, 0.06).level is CircuitBreakerLevel.RED
    assert b.evaluate(0.0, 0.01).level is CircuitBreakerLevel.RED
    assert b.evaluate(0.0, 0.0).level is CircuitBreakerLevel.RED
    assert b.is_tripped


def test_the_daily_halt_lifts_on_the_next_session():
    b = breaker()
    b.evaluate(0.0, 0.06)
    advance(b, DAY2)
    assert b.evaluate(0.0, 0.0).level is None
    assert not b.is_tripped


def test_a_daily_halt_does_not_need_a_human():
    """Only BLACK requires a restart. Demanding one for the daily halt would
    make an ordinary bad day need an operator every time."""
    b = breaker()
    b.evaluate(0.0, 0.06)
    assert not b.latched
    advance(b, DAY2)
    b.evaluate(0.0, 0.0)
    assert b.level is None


def test_black_outranks_a_daily_halt_and_survives_the_day_rolling_over():
    b = breaker()
    b.evaluate(0.16, 0.06)
    advance(b, DAY2)
    assert b.evaluate(0.0, 0.0).level is CircuitBreakerLevel.BLACK


# --- levels below the halts still behave as before --------------------------


def test_yellow_is_advisory_and_self_clearing():
    b = breaker()
    assert b.evaluate(0.09, 0.0).level is CircuitBreakerLevel.YELLOW
    assert not b.is_tripped
    assert b.evaluate(0.01, 0.0).level is None


@pytest.mark.parametrize("drawdown", [0.15, 0.20, 0.99])
def test_the_flatten_threshold_is_inclusive(drawdown: float):
    b = breaker()
    assert b.evaluate(drawdown, 0.0).action == "flatten_all"
