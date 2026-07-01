"""Tests for SignalAggregatorService — weight, combine, cost-gate, publish."""

import pytest
from trading_common.cost_filter import CostAwareFilter, TransactionCosts
from trading_common.events import EventType

from src.events.publisher import NullPublisher

from .conftest import build_service, components


@pytest.mark.asyncio
async def test_aggregate_publishes_event():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    result = await service.aggregate(
        "AAPL", components(("strategy", "BUY", 0.9), ("ml", "BUY", 0.8), ("macro", "BUY", 0.7))
    )
    assert result.final_signal == "BUY"
    assert len(publisher.published) == 1
    event = publisher.published[0]
    assert event.event_type == EventType.SIGNAL_AGGREGATED
    assert event.symbol == "AAPL"
    assert event.final_signal == "BUY"
    assert event.components_count == 3
    assert event.source_service == "signal-aggregator"


@pytest.mark.asyncio
async def test_default_weights_are_equal():
    service = build_service()
    result = await service.aggregate("AAPL", components(("strategy", "BUY", 0.9)))
    assert result.weights == {"strategy": 1.0}  # single source renormalizes to 1


@pytest.mark.asyncio
async def test_weights_renormalize_over_present_subset():
    service = build_service()  # 3 configured sources, equal 1/3 each
    result = await service.aggregate(
        "AAPL", components(("strategy", "BUY", 0.5), ("ml", "BUY", 0.5))
    )
    # only 2 present → each renormalized to 0.5
    assert result.weights["strategy"] == pytest.approx(0.5)
    assert result.weights["ml"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_unknown_source_gets_baseline_weight():
    service = build_service()
    result = await service.aggregate("AAPL", components(("newsrc", "BUY", 0.9)))
    # unknown source still contributes (renormalized to 1.0 as the only component)
    assert result.weights == {"newsrc": 1.0}
    assert result.final_signal == "BUY"


@pytest.mark.asyncio
async def test_cost_filter_downgrades_marginal_signal():
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    # a real BUY consensus, but a tiny expected edge on a micro-cap → filtered to HOLD
    result = await service.aggregate(
        "PENNY",
        components(("strategy", "BUY", 0.9), ("ml", "BUY", 0.9), ("macro", "BUY", 0.9)),
        expected_return_bps=5.0,
        market_cap_tier="micro",
    )
    assert result.final_signal == "HOLD"
    assert result.cost_filtered is True
    assert publisher.published[0].final_signal == "HOLD"


@pytest.mark.asyncio
async def test_generous_edge_passes_cost_filter():
    service = build_service()
    result = await service.aggregate(
        "AAPL",
        components(("strategy", "BUY", 0.9), ("ml", "BUY", 0.9), ("macro", "BUY", 0.9)),
        expected_return_bps=500.0,
        market_cap_tier="large",
    )
    assert result.final_signal == "BUY"
    assert result.cost_filtered is False


@pytest.mark.asyncio
async def test_hold_is_not_cost_filtered():
    service = build_service()
    result = await service.aggregate(
        "AAPL", components(("strategy", "BUY", 0.8), ("ml", "SELL", 0.8))
    )
    assert result.final_signal == "HOLD"
    assert result.cost_filtered is False  # HOLD never goes through the cost gate


@pytest.mark.asyncio
async def test_record_outcome_adapts_weights():
    service = build_service()
    # varied returns (need non-zero variance for the information ratio): ml wins, strategy loses
    ml_returns = [0.01, 0.03, 0.02, 0.04, 0.02]
    strat_returns = [-0.01, -0.03, -0.02, -0.04, -0.02]
    for _ in range(6):
        for r in ml_returns:
            service.record_outcome("ml", r)
        for r in strat_returns:
            service.record_outcome("strategy", r)
    weights = service.weights()
    assert weights["ml"] > weights["strategy"]


@pytest.mark.asyncio
async def test_strict_cost_filter_via_custom_costs():
    strict = CostAwareFilter(
        costs=TransactionCosts(spread_bps=50, slippage_bps=50), min_edge_multiple=3.0
    )
    service = build_service(cost_filter=strict)
    result = await service.aggregate(
        "AAPL",
        components(("strategy", "BUY", 0.5)),
        expected_return_bps=100.0,
    )
    # required edge is very high now → downgraded
    assert result.final_signal == "HOLD"
    assert result.cost_filtered is True
