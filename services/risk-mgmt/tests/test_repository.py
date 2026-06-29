"""Persistence: portfolio snapshots survive restarts; the breaker level is re-derived."""

import json

import pytest

from src.core.repository import NullStateRepository, RedisStateRepository

from .conftest import build_service


class FakeRepository:
    """In-memory StateRepository double (no Redis needed)."""

    def __init__(self, snapshot: dict | None = None) -> None:
        self.snapshot = snapshot
        self.saved: list[dict] = []

    async def load(self) -> dict | None:
        return self.snapshot

    async def save(self, snapshot: dict) -> None:
        self.saved.append(snapshot)
        self.snapshot = snapshot


class FakeRedis:
    """Minimal async Redis double implementing get/set."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def set(self, key: str, value: str) -> None:
        self.store[key] = value


@pytest.mark.asyncio
async def test_update_portfolio_persists_snapshot():
    repo = FakeRepository()
    service = build_service(repository=repo)
    await service.update_portfolio(value=90_000.0, drawdown_pct=0.10, daily_loss_pct=0.01)
    assert repo.saved  # at least one save happened
    assert repo.saved[-1]["value"] == 90_000.0
    assert repo.saved[-1]["drawdown_pct"] == 0.10


@pytest.mark.asyncio
async def test_restore_reapplies_portfolio_state():
    repo = FakeRepository(
        {
            "value": 80_000.0,
            "exposure_pct": 0.3,
            "drawdown_pct": 0.04,
            "daily_loss_pct": 0.0,
            "regime": "contraction",
        }
    )
    service = build_service(repository=repo)
    await service.restore()
    assert service.portfolio.value == 80_000.0
    assert service.portfolio.exposure_pct == 0.3
    assert service.portfolio.regime == "contraction"


@pytest.mark.asyncio
async def test_restore_rederives_tripped_breaker():
    # A persisted RED-level daily loss must halt trading again after a restart.
    repo = FakeRepository(
        {
            "value": 95_000.0,
            "exposure_pct": 0.0,
            "drawdown_pct": 0.0,
            "daily_loss_pct": 0.06,  # > 5% → RED
            "regime": "expansion",
        }
    )
    service = build_service(repository=repo)
    await service.restore()
    assert service.breaker.is_tripped is True


@pytest.mark.asyncio
async def test_restore_noop_when_no_snapshot():
    service = build_service(repository=FakeRepository(None))
    await service.restore()
    # default state preserved
    assert service.portfolio.value == 100_000.0
    assert service.breaker.is_tripped is False


@pytest.mark.asyncio
async def test_defaults_to_null_repository():
    # No repository passed → no persistence, no crash.
    service = build_service()
    await service.restore()
    await service.update_portfolio(drawdown_pct=0.10)
    assert service.portfolio.drawdown_pct == 0.10


@pytest.mark.asyncio
async def test_null_repository_round_trip():
    repo = NullStateRepository()
    await repo.save({"value": 1.0})
    assert await repo.load() is None


@pytest.mark.asyncio
async def test_redis_repository_round_trip():
    redis = FakeRedis()
    repo = RedisStateRepository(redis, key="test:portfolio")  # type: ignore[arg-type]
    snapshot = {"value": 90_000.0, "drawdown_pct": 0.1}
    await repo.save(snapshot)
    assert redis.store["test:portfolio"] == json.dumps(snapshot)
    assert await repo.load() == snapshot


@pytest.mark.asyncio
async def test_redis_repository_ignores_corrupt_snapshot():
    redis = FakeRedis()
    redis.store["risk_mgmt:portfolio"] = "{not valid json"
    repo = RedisStateRepository(redis)  # type: ignore[arg-type]
    assert await repo.load() is None


@pytest.mark.asyncio
async def test_restore_then_persist_round_trip_end_to_end():
    # Save from one service instance, restore into a fresh one (shared repo) → state carries over.
    repo = FakeRepository()
    first = build_service(repository=repo)
    await first.update_portfolio(value=70_000.0, drawdown_pct=0.16)  # BLACK → flatten
    assert first.breaker.is_tripped is True

    second = build_service(repository=repo)
    await second.restore()
    assert second.portfolio.value == 70_000.0
    assert second.breaker.is_tripped is True
