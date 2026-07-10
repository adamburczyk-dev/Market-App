"""Tests for the in-process periodic scheduler."""

import asyncio
from datetime import UTC, datetime

import pytest

from trading_common.scheduler import (
    SECONDS_PER_WEEK,
    PeriodicTask,
    seconds_until_weekday_hour,
)

# --- seconds_until_weekday_hour (pure) ---

SATURDAY = 5


def test_later_same_week():
    now = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)  # Wednesday noon
    delay = seconds_until_weekday_hour(now, SATURDAY, 6)
    target = datetime(2026, 7, 11, 6, 0, tzinfo=UTC)  # Saturday 06:00
    assert delay == (target - now).total_seconds()


def test_same_day_before_hour():
    now = datetime(2026, 7, 11, 4, 0, tzinfo=UTC)  # Saturday 04:00
    assert seconds_until_weekday_hour(now, SATURDAY, 6) == 2 * 3600


def test_same_day_past_hour_wraps_a_week():
    now = datetime(2026, 7, 11, 7, 0, tzinfo=UTC)  # Saturday 07:00
    delay = seconds_until_weekday_hour(now, SATURDAY, 6)
    assert delay == SECONDS_PER_WEEK - 3600


def test_exact_boundary_schedules_next_week():
    now = datetime(2026, 7, 11, 6, 0, tzinfo=UTC)  # exactly Saturday 06:00
    assert seconds_until_weekday_hour(now, SATURDAY, 6) == SECONDS_PER_WEEK


# --- PeriodicTask ---


@pytest.mark.asyncio
async def test_job_fires_repeatedly():
    fired = 0

    async def job() -> None:
        nonlocal fired
        fired += 1

    task = PeriodicTask("t", interval_s=0.01, job=job)
    task.start()
    await asyncio.sleep(0.06)
    await task.stop()
    assert fired >= 3


@pytest.mark.asyncio
async def test_initial_delay_defers_first_run():
    fired = 0

    async def job() -> None:
        nonlocal fired
        fired += 1

    task = PeriodicTask("t", interval_s=0.01, job=job, initial_delay_s=10.0)
    task.start()
    await asyncio.sleep(0.05)
    await task.stop()
    assert fired == 0  # still inside the initial delay


@pytest.mark.asyncio
async def test_failing_job_keeps_the_schedule_alive():
    calls = 0

    async def job() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    task = PeriodicTask("t", interval_s=0.01, job=job)
    task.start()
    await asyncio.sleep(0.06)
    assert task.running  # exceptions are isolated, the loop keeps ticking
    await task.stop()
    assert calls >= 3


@pytest.mark.asyncio
async def test_stop_cancels_cleanly_and_is_idempotent():
    async def job() -> None:
        return None

    task = PeriodicTask("t", interval_s=60.0, job=job)
    task.start()
    assert task.running
    await task.stop()
    assert not task.running
    await task.stop()  # second stop is a no-op


@pytest.mark.asyncio
async def test_start_twice_keeps_one_loop():
    fired = 0

    async def job() -> None:
        nonlocal fired
        fired += 1

    task = PeriodicTask("t", interval_s=0.02, job=job)
    task.start()
    task.start()  # no second loop
    await asyncio.sleep(0.03)
    await task.stop()
    assert fired == 1
