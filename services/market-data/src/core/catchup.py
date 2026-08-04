"""Did today's pull already happen? — the question a part-time stack must ask.

`PeriodicTask` aligns its first run to `FETCH_AT_HOUR_UTC` so the daily pull
lands after a market close rather than wherever the container happened to
start. That is right for a stack that runs continuously, and WRONG for one that
does not: a container up from 09:00 to 18:00 never reaches 23:00 UTC, so the
scheduled pull never fires — not late, never. Nothing logs an error, because
nothing failed; the run was simply always still in the future.

So on boot we ask whether the pull already ran TODAY, and if not, run it now
instead of waiting. The regular schedule stays exactly as it was: if the
container happens to be up at the configured hour, that run happens too and is
a cheap incremental no-op, because `plan_fetch` resumes from the last stored
bar.

The marker is a DATE, not a timestamp, and it is written only after a run
finishes. A run that crashes half way leaves the day unmarked, so the next boot
retries it — the opposite bias would silently skip a day of data.
"""

from collections.abc import Awaitable
from datetime import UTC, date, datetime
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


def needs_catchup(last_sync: date | None, now: datetime | None = None) -> bool:
    """True when no successful pull has been recorded for today (UTC).

    Compared by DATE in UTC, matching how the schedule's hour is expressed. A
    marker dated in the future — a clock that jumped, a restored snapshot —
    counts as "already synced" rather than triggering a pull on every boot
    until the calendar catches up.
    """
    if last_sync is None:
        return True
    return last_sync < (now or datetime.now(UTC)).date()
