"""Lightweight in-process periodic scheduler for service background jobs.

Services run recurring work (weekly walk-forward revalidation, periodic macro
refresh, EDGAR universe refresh) as plain asyncio tasks inside their FastAPI
lifespan — no external cron, no new infrastructure. A failed run is logged and
the schedule keeps ticking (exception isolation); ``stop()`` cancels cleanly
on shutdown.

Single-replica semantics: every replica runs its own schedule, so scheduled
services must stay at 1 replica (or grow leader election) — consistent with
the push-consumer constraint already documented for subscribers.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import datetime, timedelta

import structlog

logger = structlog.get_logger()

Job = Callable[[], Awaitable[None]]

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY


def seconds_until_weekday_hour(now: datetime, weekday: int, hour: int) -> float:
    """Seconds from ``now`` to the next (weekday, hour):00 in ``now``'s timezone.

    ``weekday`` follows ``datetime.weekday()``: Monday=0 … Sunday=6. When
    ``now`` is exactly on (or past) the boundary, the *next* week's occurrence
    is returned — a job started at its own fire time must not double-fire.
    """
    days_ahead = (weekday - now.weekday()) % 7
    candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0) + timedelta(
        days=days_ahead
    )
    if candidate <= now:
        candidate += timedelta(days=7)
    return (candidate - now).total_seconds()


class PeriodicTask:
    """Run an async job every ``interval_s`` seconds.

    The first run fires after ``initial_delay_s`` (defaults to one full
    interval — a restart must not immediately hammer external APIs); pass an
    aligned delay (e.g. from ``seconds_until_weekday_hour``) for calendar
    schedules like "Saturday 06:00 UTC, then weekly".
    """

    def __init__(
        self,
        name: str,
        interval_s: float,
        job: Job,
        initial_delay_s: float | None = None,
    ) -> None:
        self._name = name
        self._interval_s = interval_s
        self._job = job
        self._initial_delay_s = interval_s if initial_delay_s is None else initial_delay_s
        self._task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self._loop(), name=f"periodic-{self._name}")
        logger.info(
            "Periodic task started",
            task=self._name,
            interval_s=self._interval_s,
            initial_delay_s=round(self._initial_delay_s, 1),
        )

    async def _loop(self) -> None:
        delay = self._initial_delay_s
        while True:
            await asyncio.sleep(delay)
            try:
                await self._job()
            except Exception as exc:  # noqa: BLE001 — one failed run must not kill the schedule
                logger.warning("Scheduled job failed", task=self._name, error=str(exc))
            delay = self._interval_s

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info("Periodic task stopped", task=self._name)
