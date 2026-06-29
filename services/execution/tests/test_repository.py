"""Persistence: paper-broker state (cash / positions) survives restarts."""

import json

import pytest

from src.core.paper_broker import PaperBroker
from src.core.repository import NullBrokerRepository, RedisBrokerRepository

from .conftest import build_service
from .test_service import order


class FakeRepository:
    """In-memory BrokerRepository double (no Redis needed)."""

    def __init__(self, snapshot: dict | None = None) -> None:
        self.snapshot = snapshot
        self.saved: list[dict] = []

    async def load(self) -> dict | None:
        return self.snapshot

    async def save(self, snapshot: dict) -> None:
        self.saved.append(snapshot)
        self.snapshot = snapshot


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


def test_broker_snapshot_round_trip():
    broker = PaperBroker(initial_cash=100_000.0)
    broker.fill("o1", "AAPL", "BUY", 50.0, 100.0)  # cash 95_000, 50 AAPL @100
    snap = broker.snapshot()

    restored = PaperBroker(initial_cash=1.0)  # different cash → must be overwritten
    restored.restore(snap)
    assert restored.cash == broker.cash
    assert restored.positions() == broker.positions()
    assert restored.equity == broker.equity


@pytest.mark.asyncio
async def test_execute_persists_snapshot():
    repo = FakeRepository()
    service = build_service(repository=repo)
    await service.execute(order())  # BUY 50 AAPL @100
    assert repo.saved
    assert repo.saved[-1]["cash"] == 95_000.0
    assert repo.saved[-1]["positions"]["AAPL"]["quantity"] == 50.0


@pytest.mark.asyncio
async def test_restore_reapplies_broker_state():
    snapshot = {
        "cash": 95_000.0,
        "peak_equity": 100_000.0,
        "day_start_equity": 100_000.0,
        "positions": {"AAPL": {"quantity": 50.0, "last_price": 100.0}},
    }
    service = build_service(repository=FakeRepository(snapshot))
    await service.restore()
    assert service.broker.cash == 95_000.0
    assert service.broker.positions()["AAPL"]["quantity"] == 50.0
    assert service.broker.equity == 100_000.0


@pytest.mark.asyncio
async def test_restore_noop_when_no_snapshot():
    service = build_service(repository=FakeRepository(None))
    await service.restore()
    assert service.broker.cash == 100_000.0
    assert service.broker.positions() == {}


@pytest.mark.asyncio
async def test_defaults_to_null_repository():
    service = build_service()  # no repository → NullBrokerRepository, no crash
    await service.restore()
    await service.execute(order())
    assert service.broker.positions()["AAPL"]["quantity"] == 50.0


@pytest.mark.asyncio
async def test_end_to_end_persist_then_restore():
    # Fill on one instance, restore into a fresh one via the shared repo.
    repo = FakeRepository()
    first = build_service(repository=repo)
    await first.execute(order())  # BUY 50 AAPL @100

    second = build_service(repository=repo)
    await second.restore()
    assert second.broker.cash == 95_000.0
    assert second.broker.positions()["AAPL"]["quantity"] == 50.0


@pytest.mark.asyncio
async def test_null_repository_round_trip():
    repo = NullBrokerRepository()
    await repo.save({"cash": 1.0})
    assert await repo.load() is None


@pytest.mark.asyncio
async def test_redis_repository_round_trip():
    redis = FakeRedis()
    repo = RedisBrokerRepository(redis, key="test:broker")  # type: ignore[arg-type]
    snapshot = {"cash": 95_000.0, "positions": {}}
    await repo.save(snapshot)
    assert redis.store["test:broker"] == json.dumps(snapshot)
    assert await repo.load() == snapshot


@pytest.mark.asyncio
async def test_redis_repository_ignores_corrupt_snapshot():
    redis = FakeRedis()
    redis.store["execution:broker"] = "{not valid json"
    repo = RedisBrokerRepository(redis)  # type: ignore[arg-type]
    assert await repo.load() is None
