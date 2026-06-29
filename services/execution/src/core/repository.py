"""Snapshot persistence for the paper broker (Redis, with a no-op fallback).

Cash, open positions and the equity highs (peak / day-start) are persisted so the
portfolio — and therefore the drawdown / daily-loss metrics fed to risk-mgmt — survives
a restart instead of resetting to the initial cash balance.
"""

import json
from typing import Protocol

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger()


class BrokerRepository(Protocol):
    async def load(self) -> dict | None: ...
    async def save(self, snapshot: dict) -> None: ...


class NullBrokerRepository:
    """No persistence. Used when Redis is unavailable (state is in-memory only)."""

    async def load(self) -> dict | None:
        return None

    async def save(self, snapshot: dict) -> None:
        return None


class RedisBrokerRepository:
    def __init__(self, redis: Redis, key: str = "execution:broker") -> None:
        self._redis = redis
        self._key = key

    async def load(self) -> dict | None:
        raw = await self._redis.get(self._key)
        if raw is None:
            return None
        try:
            data: dict = json.loads(raw)
            return data
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt broker snapshot, ignoring", error=str(exc))
            return None

    async def save(self, snapshot: dict) -> None:
        await self._redis.set(self._key, json.dumps(snapshot))
