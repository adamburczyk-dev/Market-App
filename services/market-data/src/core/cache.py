"""OHLCV cache — Redis in production, in-memory for tests / degraded mode."""

import json
from typing import Protocol

import structlog
from redis.asyncio import Redis
from trading_common.schemas import Interval, OHLCVBar

logger = structlog.get_logger()


def _key(symbol: str, interval: Interval) -> str:
    return f"ohlcv:{symbol}:{interval.value}"


class Cache(Protocol):
    async def get_bars(self, symbol: str, interval: Interval) -> list[OHLCVBar] | None: ...
    async def set_bars(self, symbol: str, interval: Interval, bars: list[OHLCVBar]) -> None: ...
    async def invalidate(self, symbol: str, interval: Interval) -> None: ...


class InMemoryCache:
    """Process-local cache. Used in tests and when Redis is unavailable."""

    def __init__(self) -> None:
        self._store: dict[str, list[OHLCVBar]] = {}

    async def get_bars(self, symbol: str, interval: Interval) -> list[OHLCVBar] | None:
        return self._store.get(_key(symbol, interval))

    async def set_bars(self, symbol: str, interval: Interval, bars: list[OHLCVBar]) -> None:
        self._store[_key(symbol, interval)] = bars

    async def invalidate(self, symbol: str, interval: Interval) -> None:
        self._store.pop(_key(symbol, interval), None)


class RedisCache:
    """Redis-backed cache. Bars serialized as JSON with a TTL."""

    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def get_bars(self, symbol: str, interval: Interval) -> list[OHLCVBar] | None:
        raw = await self._redis.get(_key(symbol, interval))
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return [OHLCVBar.model_validate(item) for item in data]
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Corrupt cache entry, ignoring", symbol=symbol, error=str(exc))
            return None

    async def set_bars(self, symbol: str, interval: Interval, bars: list[OHLCVBar]) -> None:
        payload = json.dumps([bar.model_dump(mode="json") for bar in bars])
        await self._redis.set(_key(symbol, interval), payload, ex=self._ttl)

    async def invalidate(self, symbol: str, interval: Interval) -> None:
        await self._redis.delete(_key(symbol, interval))
