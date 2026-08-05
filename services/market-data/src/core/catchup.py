"""Has today's pull happened yet? — asked on a heartbeat, not on a timer.

Two things make a once-a-day aligned timer wrong for a stack that is not up
around the clock, and the second one is the reason this module exists at all.

A container up from 09:00 to 18:00 never reaches FETCH_AT_HOUR_UTC, so an
aligned first run fires NEVER — not late. That much is obvious once stated.

The second is not: `asyncio.sleep` measures MONOTONIC time, which stops while
the host is suspended. Measured on the user's laptop after one night — 20.0
hours of wall clock against 4.3 hours of monotonic — so a timer set for "in
four hours" was still counting down the following afternoon. On a machine that
sleeps, an aligned daily timer cannot hold its alignment: it drifts by exactly
as much as the machine rested, and never lands on the hour it was aimed at.

So the schedule is a short HEARTBEAT that asks a wall-clock question: is there
a completed pull recorded for today? Suspend just delays the next beat; it
cannot corrupt the answer, because the answer is a date comparison rather than
an elapsed-time one.

`FETCH_AT_HOUR_UTC` keeps its meaning as the hour after which a same-day pull
can see today's close. A pull that runs before it is recorded as covering the
PREVIOUS session, so one later beat the same day picks up the close — at most
two pulls a day, and the loop terminates because the second one is recorded
after the hour.
"""

from collections.abc import Awaitable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()

MARKER_KEY = "market-data:last-sync-date"


class MarkerStore(Protocol):
    """The slice of a Redis client this needs.

    Two details make this match redis-py rather than merely resemble it.
    Parameters are POSITIONAL-ONLY, because redis names them `name`/`value` and
    carries a dozen optional keywords. And they return `Awaitable` rather than
    being declared `async def`: redis-py's methods are ordinary functions
    returning an awaitable, which a coroutine-typed Protocol does not accept.
    """

    def get(self, name: str, /) -> Awaitable[Any]: ...

    def set(self, name: str, value: str, /) -> Awaitable[Any]: ...


class RedisSyncMarker:
    """Last successful sync date, surviving restarts.

    Redis rather than the database because this is bookkeeping about a JOB, not
    data about the market. No TTL: a marker that expires would silently mean
    "never synced" and trigger a redundant pull, which is harmless but noisy.
    """

    def __init__(self, client: MarkerStore) -> None:
        self._client = client

    async def last_sync(self) -> date | None:
        raw = await self._client.get(MARKER_KEY)
        if not raw:
            return None
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        try:
            return date.fromisoformat(raw)
        except ValueError:
            # A corrupt marker must not wedge the catch-up permanently; treat
            # it as "unknown" and let the pull decide.
            logger.warning("Ignoring unreadable sync marker", value=raw)
            return None

    async def mark(self, day: date) -> None:
        await self._client.set(MARKER_KEY, day.isoformat())


class NullSyncMarker:
    """Used when Redis is down. Reports "never synced", so the pull runs.

    Erring toward a redundant pull rather than a skipped one is deliberate: the
    pull is incremental and idempotent, while a skipped day leaves a hole that
    only the next restatement check would notice.
    """

    async def last_sync(self) -> date | None:
        return None

    async def mark(self, day: date) -> None:
        return None


def coverage_date(now: datetime, close_hour: int) -> date:
    """Which session a pull running at `now` can actually have covered.

    Before the close hour the provider has nothing for today, so the run is
    recorded against yesterday. That is what lets a later beat the same day
    fetch today's close instead of the marker claiming the day is finished.
    """
    return now.date() if now.hour >= close_hour else now.date() - timedelta(days=1)


def needs_catchup(
    last_sync: date | None,
    now: datetime | None = None,
    close_hour: int = 23,
) -> bool:
    """True when no completed pull covers the latest session available now.

    A date comparison, deliberately — not elapsed time. A suspended host makes
    elapsed time lie by however long it slept, while "is the marker older than
    the session we could fetch" survives any amount of suspension.

    A marker dated in the future — a clock that jumped, a restored snapshot —
    counts as covered rather than triggering a pull on every beat until the
    calendar catches up.
    """
    if last_sync is None:
        return True
    return last_sync < coverage_date(now or datetime.now(UTC), close_hour)
