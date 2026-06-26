"""Testy orkiestracji StrategyService."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import Interval

from src.core.service import PortfolioSnapshot
from src.events.publisher import NullPublisher

from .conftest import FakeFeatureClient, FakePortfolioClient, build_service, buy_client


@pytest.mark.asyncio
async def test_buy_signal_published():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)

    event = await service.evaluate_symbol("AAPL", Interval.D1)
    assert event is not None
    assert event.signal == "BUY"
    assert event.event_type == EventType.SIGNAL_GENERATED
    assert event.stop_loss == 95.0  # 100 * (1 - 0.05)
    assert event.take_profit == 110.0  # 100 + 5 * 2
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_sell_signal_published():
    client = FakeFeatureClient(ranked={"momentum_20": 0.1}, raw={"rsi_14": 50.0, "close": 100.0})
    service = build_service(client)
    event = await service.evaluate_symbol("AAPL", Interval.D1)
    assert event is not None
    assert event.signal == "SELL"
    assert event.stop_loss == 105.0
    assert event.take_profit == 90.0


@pytest.mark.asyncio
async def test_hold_publishes_nothing():
    publisher = NullPublisher()
    client = FakeFeatureClient(ranked={"momentum_20": 0.5}, raw={"rsi_14": 50.0, "close": 100.0})
    service = build_service(client, publisher=publisher)
    assert await service.evaluate_symbol("AAPL", Interval.D1) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_rejected_by_risk_envelope_drawdown():
    # Portfolio breaching the drawdown limit -> hard reject (not a sizing concern).
    publisher = NullPublisher()
    service = build_service(
        buy_client(), publisher=publisher, portfolio=PortfolioSnapshot(drawdown_pct=0.20)
    )
    assert await service.evaluate_symbol("AAPL", Interval.D1) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_filtered_by_cost_when_edge_too_small():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher, expected_edge_bps=10.0)
    assert await service.evaluate_symbol("AAPL", Interval.D1) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_inactive_strategy_suppresses_signals():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    # Deactivate via decay monitor (negative 30d sharpe).
    changed = await service.update_health(
        sharpe_30d=-0.5,
        sharpe_90d=0.0,
        sharpe_180d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
        excess_return_vs_spy_30d=0.0,
    )
    assert changed is not None
    assert service.health.status == "deactivated"
    assert await service.evaluate_symbol("AAPL", Interval.D1) is None


@pytest.mark.asyncio
async def test_update_health_publishes_status_change():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    event = await service.update_health(
        sharpe_30d=-0.5,
        sharpe_90d=0.0,
        sharpe_180d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
        excess_return_vs_spy_30d=0.0,
    )
    assert event is not None
    assert event.event_type == EventType.STRATEGY_STATUS_CHANGED
    assert event.new_status == "deactivated"
    assert any(e.event_type == EventType.STRATEGY_STATUS_CHANGED for e in publisher.published)


@pytest.mark.asyncio
async def test_handle_features_ready_event_triggers_signal():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    from trading_common.events import FeaturesReadyEvent

    event = FeaturesReadyEvent(symbol="AAPL", interval="1d", features_count=10, tier=1)
    await service.handle_features_ready_event(event.model_dump_json().encode())
    assert len(publisher.published) == 1
    assert publisher.published[0].symbol == "AAPL"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_live_portfolio_drawdown_blocks_signal():
    publisher = NullPublisher()
    breach = {"value": 100_000.0, "exposure_pct": 0.0, "drawdown_pct": 0.20, "daily_loss_pct": 0.0}
    service = build_service(
        buy_client(), publisher=publisher, portfolio_client=FakePortfolioClient(breach)
    )
    assert await service.evaluate_symbol("AAPL", Interval.D1) is None
    assert publisher.published == []


@pytest.mark.asyncio
async def test_falls_back_to_placeholder_when_portfolio_unavailable():
    publisher = NullPublisher()
    service = build_service(
        buy_client(), publisher=publisher, portfolio_client=FakePortfolioClient(None)
    )
    # client returns None → fall back to the (healthy) placeholder → signal still emitted
    assert await service.evaluate_symbol("AAPL", Interval.D1) is not None
    assert len(publisher.published) == 1
