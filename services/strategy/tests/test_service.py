"""Testy orkiestracji StrategyService (wiele reguł na jednym zdarzeniu)."""

import pytest
from trading_common.events import EventType
from trading_common.schemas import Interval

from src.core.service import PortfolioSnapshot
from src.events.publisher import NullPublisher

from .conftest import (
    FakeFeatureClient,
    FakePortfolioClient,
    build_service,
    buy_client,
    rules_named,
)


@pytest.mark.asyncio
async def test_buy_signal_published():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)

    events = await service.evaluate_symbol("AAPL", Interval.D1)
    assert len(events) == 1
    event = events[0]
    assert event.signal == "BUY"
    assert event.strategy_name == "momentum_rank"
    assert event.event_type == EventType.SIGNAL_GENERATED
    # No ATR in this vector → the configured fallback stop (5%) applies.
    assert event.stop_loss == 95.0
    assert event.take_profit == 110.0  # 100 + 5 * RR 2.0
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_sell_signal_published():
    client = FakeFeatureClient(ranked={"momentum_20": 0.1}, raw={"rsi_14": 50.0, "close": 100.0})
    service = build_service(client)
    events = await service.evaluate_symbol("AAPL", Interval.D1)
    assert [e.signal for e in events] == ["SELL"]
    assert events[0].stop_loss == 105.0
    assert events[0].take_profit == 90.0


@pytest.mark.asyncio
async def test_hold_publishes_nothing():
    publisher = NullPublisher()
    client = FakeFeatureClient(ranked={"momentum_20": 0.5}, raw={"rsi_14": 50.0, "close": 100.0})
    service = build_service(client, publisher=publisher)
    assert await service.evaluate_symbol("AAPL", Interval.D1) == []
    assert publisher.published == []


# --- S2: many rules on one features.ready ---------------------------------


CONTESTED_RULES = ("donchian_breakout", "momentum_rank")


def contested_client() -> FakeFeatureClient:
    """One vector that momentum reads as BUY and the breakout rule as SELL.

    Top-decile 20-day momentum but a close below the PRIOR 20-day low — a name
    that ran up and then broke down. The pair matters: momentum and reversion
    cannot disagree by construction (momentum's overbought filter refuses the
    exact RSI that reversion needs to sell), so using them here would have
    tested the fixture, not the service.
    """
    return FakeFeatureClient(
        ranked={"momentum_20": 0.95},
        raw={"rsi_14": 55.0, "donchian_pos_20": -0.2, "close": 100.0, "atr_pct_14": 0.02},
    )


@pytest.mark.asyncio
async def test_each_active_rule_emits_its_own_event():
    publisher = NullPublisher()
    service = build_service(
        contested_client(), publisher=publisher, rules=rules_named(*CONTESTED_RULES)
    )
    events = await service.evaluate_symbol("AAPL", Interval.D1)

    by_strategy = {e.strategy_name: e.signal for e in events}
    assert by_strategy == {"momentum_rank": "BUY", "donchian_breakout": "SELL"}
    # Both reach the bus: resolving the disagreement is the aggregator's job,
    # and a service that picked a winner here would hide it.
    assert len(publisher.published) == 2


@pytest.mark.asyncio
async def test_deactivating_one_rule_leaves_the_other_running():
    publisher = NullPublisher()
    service = build_service(
        contested_client(), publisher=publisher, rules=rules_named(*CONTESTED_RULES)
    )
    await service.update_health(
        "momentum_rank",
        sharpe_30d=-0.5,
        sharpe_90d=0.0,
        sharpe_180d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
        excess_return_vs_spy_30d=0.0,
    )
    assert service.health_of("momentum_rank").status == "deactivated"
    assert service.health_of("donchian_breakout").status == "active"

    events = await service.evaluate_symbol("AAPL", Interval.D1)
    assert [e.strategy_name for e in events] == ["donchian_breakout"]


@pytest.mark.asyncio
async def test_the_stop_is_scaled_by_volatility_not_flat():
    """S6: same rule, same price, twice the ATR → twice the stop distance.
    A flat 5% was taking two different risks depending on the name."""
    calm = FakeFeatureClient(
        ranked={"momentum_20": 0.9}, raw={"rsi_14": 50.0, "close": 100.0, "atr_pct_14": 0.01}
    )
    wild = FakeFeatureClient(
        ranked={"momentum_20": 0.9}, raw={"rsi_14": 50.0, "close": 100.0, "atr_pct_14": 0.02}
    )
    calm_event = (await build_service(calm).evaluate_symbol("A", Interval.D1))[0]
    wild_event = (await build_service(wild).evaluate_symbol("A", Interval.D1))[0]
    # momentum_rank stops at 2.0 * ATR
    assert calm_event.stop_loss == pytest.approx(98.0)
    assert wild_event.stop_loss == pytest.approx(96.0)


@pytest.mark.asyncio
async def test_statuses_report_every_rule_with_its_inputs():
    service = build_service(buy_client(), rules=rules_named("donchian_breakout", "momentum_rank"))
    rows = service.statuses()
    assert [r["name"] for r in rows] == ["donchian_breakout", "momentum_rank"]
    assert all(r["status"] == "active" for r in rows)
    assert "donchian_pos_20" in rows[0]["required_features"]
    assert rows[1]["required_ranks"] == ["momentum_20"]


@pytest.mark.asyncio
async def test_health_of_an_unrun_strategy_names_what_is_running():
    service = build_service(buy_client())
    with pytest.raises(KeyError, match="momentum_rank"):
        service.health_of("donchian_breakout")


def test_a_service_with_no_rules_is_refused():
    """Silently running nothing looks exactly like running and finding nothing."""
    with pytest.raises(ValueError, match="at least one rule"):
        build_service(buy_client(), rules=[])


# --- risk / cost gates (unchanged behaviour, now per rule) -----------------


@pytest.mark.asyncio
async def test_rejected_by_risk_envelope_drawdown():
    # Portfolio breaching the drawdown limit -> hard reject (not a sizing concern).
    publisher = NullPublisher()
    service = build_service(
        buy_client(), publisher=publisher, portfolio=PortfolioSnapshot(drawdown_pct=0.20)
    )
    assert await service.evaluate_symbol("AAPL", Interval.D1) == []
    assert publisher.published == []


@pytest.mark.asyncio
async def test_filtered_by_cost_when_edge_too_small():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher, expected_edge_bps=10.0)
    assert await service.evaluate_symbol("AAPL", Interval.D1) == []
    assert publisher.published == []


@pytest.mark.asyncio
async def test_inactive_strategy_suppresses_signals():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    changed = await service.update_health(
        "momentum_rank",
        sharpe_30d=-0.5,
        sharpe_90d=0.0,
        sharpe_180d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
        excess_return_vs_spy_30d=0.0,
    )
    assert changed is not None
    assert service.health_of("momentum_rank").status == "deactivated"
    assert await service.evaluate_symbol("AAPL", Interval.D1) == []


@pytest.mark.asyncio
async def test_update_health_publishes_status_change():
    publisher = NullPublisher()
    service = build_service(buy_client(), publisher=publisher)
    event = await service.update_health(
        "momentum_rank",
        sharpe_30d=-0.5,
        sharpe_90d=0.0,
        sharpe_180d=0.0,
        win_rate_30d=0.5,
        profit_factor_30d=1.0,
        excess_return_vs_spy_30d=0.0,
    )
    assert event is not None
    assert event.event_type == EventType.STRATEGY_STATUS_CHANGED
    assert event.strategy_name == "momentum_rank"
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
    assert await service.evaluate_symbol("AAPL", Interval.D1) == []
    assert publisher.published == []


@pytest.mark.asyncio
async def test_falls_back_to_placeholder_when_portfolio_unavailable():
    publisher = NullPublisher()
    service = build_service(
        buy_client(), publisher=publisher, portfolio_client=FakePortfolioClient(None)
    )
    # client returns None → fall back to the (healthy) placeholder → signal still emitted
    assert len(await service.evaluate_symbol("AAPL", Interval.D1)) == 1
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_the_portfolio_is_fetched_once_per_event_not_once_per_rule():
    """Two rules must see one portfolio snapshot; re-querying per rule would let
    a mid-event change gate one rule and not the other."""

    class CountingPortfolioClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get_portfolio(self) -> dict | None:
            self.calls += 1
            return None

    counting = CountingPortfolioClient()
    service = build_service(
        contested_client(), rules=rules_named(*CONTESTED_RULES), portfolio_client=counting
    )
    await service.evaluate_symbol("AAPL", Interval.D1)
    assert counting.calls == 1
