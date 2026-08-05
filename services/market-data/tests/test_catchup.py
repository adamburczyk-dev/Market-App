"""The daily pull must survive a stack that is not up all day.

`PeriodicTask` aligns its first run to FETCH_AT_HOUR_UTC. For a container up
from 09:00 to 18:00 that hour never arrives, so the scheduled pull fires
NEVER — not late. Nothing errors, because nothing failed: the run was always
still in the future, and the whole event chain hangs off market_data.updated.
"""

from datetime import UTC, date, datetime

import pytest

from src.core.catchup import (
    MARKER_KEY,
    NullSyncMarker,
    RedisSyncMarker,
    coverage_date,
    needs_catchup,
)


class FakeRedis:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.data = dict(initial or {})

    async def get(self, name: str) -> str | None:
        return self.data.get(name)

    async def set(self, name: str, value: str) -> bool:
        self.data[name] = value
        return True


NOON = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
AFTER_CLOSE = datetime(2026, 8, 4, 23, 30, tzinfo=UTC)


def test_a_session_with_no_recorded_pull_needs_one():
    assert needs_catchup(None, NOON) is True
    assert needs_catchup(date(2026, 8, 2), NOON) is True


def test_a_covered_session_does_not_pull_again():
    """The heartbeat beats every 30 minutes; it must not re-pull 455 symbols
    each time."""
    # At noon the latest AVAILABLE session is the previous day's close.
    assert needs_catchup(date(2026, 8, 3), NOON) is False
    assert needs_catchup(date(2026, 8, 4), AFTER_CLOSE) is False


def test_a_pull_before_the_close_is_recorded_against_the_previous_session():
    """Otherwise a stack that only runs in the morning would mark the day done
    and never fetch today's close at all."""
    assert coverage_date(NOON, close_hour=23) == date(2026, 8, 3)
    assert coverage_date(AFTER_CLOSE, close_hour=23) == date(2026, 8, 4)


def test_the_close_is_picked_up_by_a_later_beat_the_same_day():
    """A morning pull covers yesterday; the stack still up after the close must
    fetch again — and then stop, or the heartbeat would loop all evening."""
    morning_marker = coverage_date(NOON, 23)
    assert needs_catchup(morning_marker, AFTER_CLOSE, 23) is True
    evening_marker = coverage_date(AFTER_CLOSE, 23)
    assert needs_catchup(evening_marker, AFTER_CLOSE, 23) is False


def test_a_marker_dated_in_the_future_counts_as_covered():
    """A clock jump or a restored snapshot must not trigger a pull on every
    beat until the calendar catches up."""
    assert needs_catchup(date(2026, 8, 10), NOON) is False


def test_suspending_the_host_delays_the_answer_but_cannot_corrupt_it():
    """The defect this replaced: `asyncio.sleep` measures MONOTONIC time, which
    stops while the host sleeps. Measured on the real machine — 20.0h of wall
    clock against 4.3h of monotonic — so a timer aimed at 23:00 was still
    counting down the next afternoon and the pull never ran.

    A date comparison has no such failure mode: however long the machine
    rested, the first beat after resume sees the same answer it would have
    seen without the suspension.
    """
    # Exactly the run that happened: a pull at 19:19 on the 4th, before the
    # close, so it covers the 3rd.
    ran_at = datetime(2026, 8, 4, 19, 19, tzinfo=UTC)
    marker = coverage_date(ran_at, 23)
    assert marker == date(2026, 8, 3)

    # Minutes later nothing is due — there is no newer session to fetch.
    assert needs_catchup(marker, ran_at, 23) is False

    # The host then slept ~16 hours. On the real machine the monotonic clock
    # advanced 4.3h against 20.0h of wall clock, so the aligned timer never
    # fired. The date question is unaffected: the 4th has closed, and the 4th
    # is not covered.
    after_a_long_sleep = datetime(2026, 8, 5, 15, 0, tzinfo=UTC)
    assert needs_catchup(marker, after_a_long_sleep, 23) is True

    # And once the 4th is covered, further beats that day stay quiet.
    assert needs_catchup(date(2026, 8, 4), after_a_long_sleep, 23) is False


@pytest.mark.asyncio
async def test_the_marker_round_trips_through_redis():
    redis = FakeRedis()
    marker = RedisSyncMarker(redis)

    assert await marker.last_sync() is None
    await marker.mark(date(2026, 8, 4))

    assert redis.data[MARKER_KEY] == "2026-08-04"
    assert await marker.last_sync() == date(2026, 8, 4)


@pytest.mark.asyncio
async def test_an_unreadable_marker_does_not_wedge_the_catchup():
    """A corrupt value must degrade to "unknown", not to "never run again"."""
    marker = RedisSyncMarker(FakeRedis({MARKER_KEY: "not-a-date"}))
    assert await marker.last_sync() is None
    assert needs_catchup(await marker.last_sync(), NOON) is True


@pytest.mark.asyncio
async def test_bytes_from_a_client_without_decode_responses_still_parse():
    """redis-py returns bytes unless decode_responses is set; the marker must
    not depend on how the client three layers up was configured."""
    marker = RedisSyncMarker(FakeRedis({MARKER_KEY: b"2026-08-04"}))  # type: ignore[dict-item]
    assert await marker.last_sync() == date(2026, 8, 4)


@pytest.mark.asyncio
async def test_without_redis_the_pull_runs_rather_than_being_skipped():
    """Erring toward a redundant pull is deliberate: the pull is incremental
    and idempotent, while a skipped day leaves a hole that only the next
    restatement check would notice."""
    marker = NullSyncMarker()
    assert await marker.last_sync() is None
    assert needs_catchup(await marker.last_sync(), NOON) is True
    await marker.mark(date(2026, 8, 4))  # must not raise
    assert needs_catchup(await marker.last_sync(), NOON) is True
