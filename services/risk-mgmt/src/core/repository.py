"""Snapshot persistence for portfolio state (Redis, with a no-op fallback).

The circuit-breaker level is NOT persisted directly — it is re-derived from the
restored drawdown/daily-loss on startup, so a tripped halt survives a restart.
"""

import json
from typing import Protocol

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger()


class StateRepository(Protocol):
    async def load(self) -> dict | None: ...
    async def save(self, snapshot: dict) -> None: ...


class NullStateRepository:
    """No persistence. Used when Redis is unavailable (state is in-memory only)."""

    async def load(self) -> dict | None:
        return None

    async def save(self, snapshot: dict) -> None:
        return None


class RedisStateRepository:
    def __init__(self, redis: Redis, key: str = "risk_mgmt:portfolio") -> None:
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
            logger.warning("Corrupt portfolio snapshot, ignoring", error=str(exc))
            return None

    async def save(self, snapshot: dict) -> None:
        await self._redis.set(self._key, json.dumps(snapshot))
