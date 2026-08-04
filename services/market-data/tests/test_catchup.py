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


def test_a_day_with_no_recorded_sync_needs_one():
    assert needs_catchup(None, NOON) is True
    assert needs_catchup(date(2026, 8, 3), NOON) is True


def test_a_day_already_synced_does_not_run_again():
    """The container may restart several times a day; each restart must not
    re-pull 455 symbols."""
    assert needs_catchup(date(2026, 8, 4), NOON) is False


def test_a_marker_dated_in_the_future_counts_as_synced():
    """A clock jump or a restored snapshot must not trigger a pull on every
    boot until the calendar catches up."""
    assert needs_catchup(date(2026, 8, 10), NOON) is False


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
