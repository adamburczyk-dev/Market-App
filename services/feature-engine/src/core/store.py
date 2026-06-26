"""Latest-FeatureVector store. In-memory (single replica) or Redis (HA).

The interface is async so a Redis backend can be swapped in without changing
callers; the in-memory implementation just wraps a dict.
"""

from typing import Protocol

import structlog
from redis.asyncio import Redis
from trading_common.schemas import FeatureVector, Interval

logger = structlog.get_logger()


def _key(symbol: str, interval: Interval) -> str:
    return f"feat:{symbol}:{interval.value}"


def _as_str(value: str | bytes) -> str:
    """Redis stubs type members as bytes|str; with decode_responses they're str."""
    return value if isinstance(value, str) else value.decode()


class FeatureStore(Protocol):
    async def put(self, fv: FeatureVector) -> None: ...
    async def get(self, symbol: str, interval: Interval) -> FeatureVector | None: ...
    async def all_for_interval(self, interval: Interval) -> list[FeatureVector]: ...
    async def symbols(self) -> list[str]: ...


class InMemoryFeatureStore:
    """Process-local store. Single-replica; used in tests and degraded mode."""

    def __init__(self) -> None:
        self._store: dict[str, FeatureVector] = {}

    async def put(self, fv: FeatureVector) -> None:
        self._store[_key(fv.symbol, fv.interval)] = fv

    async def get(self, symbol: str, interval: Interval) -> FeatureVector | None:
        return self._store.get(_key(symbol, interval))

    async def all_for_interval(self, interval: Interval) -> list[FeatureVector]:
        return [v for v in self._store.values() if v.interval == interval]

    async def symbols(self) -> list[str]:
        return sorted({v.symbol for v in self._store.values()})


class RedisFeatureStore:
    """Redis-backed store — shared across feature-engine replicas (HA)."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def put(self, fv: FeatureVector) -> None:
        await self._redis.set(_key(fv.symbol, fv.interval), fv.model_dump_json())
        await self._redis.sadd(f"feat:index:{fv.interval.value}", fv.symbol)
        await self._redis.sadd("feat:index", fv.symbol)

    async def get(self, symbol: str, interval: Interval) -> FeatureVector | None:
        raw = await self._redis.get(_key(symbol, interval))
        return FeatureVector.model_validate_json(raw) if raw else None

    async def all_for_interval(self, interval: Interval) -> list[FeatureVector]:
        members = await self._redis.smembers(f"feat:index:{interval.value}")
        if not members:
            return []
        raws = await self._redis.mget([_key(_as_str(s), interval) for s in members])
        return [FeatureVector.model_validate_json(r) for r in raws if r]

    async def symbols(self) -> list[str]:
        return sorted(_as_str(s) for s in await self._redis.smembers("feat:index"))
