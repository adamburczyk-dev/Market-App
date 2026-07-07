"""Per-symbol attribute store — interval-agnostic Tier-2 features.

Fundamentals (Piotroski F-score, margin/leverage ratios) and company style
arrive as events per *symbol*, not per (symbol, interval) bar series, so they
live in their own store and are merged into feature vectors at read time.
``put`` merges: the fundamentals handler and the classifier handler each own a
disjoint set of keys and must not clobber each other.

Same backend pattern as FeatureStore: in-memory (single replica) or Redis (HA).
"""

import json
from typing import Protocol

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger()


def _key(symbol: str) -> str:
    return f"attr:{symbol.upper()}"


class SymbolAttributeStore(Protocol):
    async def put(self, symbol: str, attributes: dict[str, float]) -> None: ...
    async def get(self, symbol: str) -> dict[str, float]: ...


class InMemoryAttributeStore:
    """Process-local store. Single-replica; used in tests and degraded mode."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, float]] = {}

    async def put(self, symbol: str, attributes: dict[str, float]) -> None:
        self._store.setdefault(_key(symbol), {}).update(attributes)

    async def get(self, symbol: str) -> dict[str, float]:
        return dict(self._store.get(_key(symbol), {}))


class RedisAttributeStore:
    """Redis-backed store — shared across feature-engine replicas (HA)."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def put(self, symbol: str, attributes: dict[str, float]) -> None:
        merged = await self.get(symbol)
        merged.update(attributes)
        await self._redis.set(_key(symbol), json.dumps(merged))

    async def get(self, symbol: str) -> dict[str, float]:
        raw = await self._redis.get(_key(symbol))
        if not raw:
            return {}
        data = json.loads(raw)
        return {k: float(v) for k, v in data.items()}
