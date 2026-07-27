"""N2: a re-published aggregate must never open the same position twice.

The aggregator re-decides whenever a component arrives (that is correct — a
regime change is new information). Risk-mgmt is the idempotent side: the second
BUY for the same symbol/session is refused, so a late ML vote enriches the
decision instead of doubling the exposure.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.events import SignalAggregatedEvent

from src.core.order_ledger import OrderLedger, session_of
from src.core.portfolio import PortfolioState
from src.events.publisher import NullPublisher

from .conftest import build_service


def aggregated(
    side: str = "BUY",
    *,
    symbol: str = "AAPL",
    components: list[str] | None = None,
    timestamp: datetime | None = None,
) -> SignalAggregatedEvent:
    return SignalAggregatedEvent(
        symbol=symbol,
        final_signal=side,
        confidence=0.8,
        components_count=len(components or ["strategy"]),
        components_present=components or ["strategy"],
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        strategy_name="momentum_rank",
        timestamp=timestamp or datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_ml_vote_rearrival_does_not_double_the_position():
    # The exact N2 sequence: strategy-only decision is sized, then the ML vote
    # lands and the aggregate is re-published as BUY with two components.
    publisher = NullPublisher()
    service = build_service(publisher=publisher)

    first = await service.process_aggregated(aggregated(components=["strategy"]))
    second = await service.process_aggregated(aggregated(components=["strategy", "ml"]))

    assert first is not None
    assert second is None
    assert len(publisher.published) == 1


@pytest.mark.asyncio
async def test_exit_on_the_same_session_is_allowed():
    # A SELL is the exit for the BUY — a different side, so it must go through.
    publisher = NullPublisher()
    service = build_service(publisher=publisher)
    await service.process_aggregated(aggregated("BUY"))
    exit_order = await service.process_aggregated(aggregated("SELL"))
    assert exit_order is not None
    assert len(publisher.published) == 2


@pytest.mark.asyncio
async def test_other_symbol_is_unaffected():
    service = build_service()
    await service.process_aggregated(aggregated(symbol="AAPL"))
    assert await service.process_aggregated(aggregated(symbol="MSFT")) is not None


@pytest.mark.asyncio
async def test_next_session_reopens_the_symbol():
    service = build_service()
    today = datetime.now(UTC)
    await service.process_aggregated(aggregated(timestamp=today))
    tomorrow = await service.process_aggregated(aggregated(timestamp=today + timedelta(days=1)))
    assert tomorrow is not None


@pytest.mark.asyncio
async def test_blocked_order_is_not_recorded():
    # Sizing refused the order → no exposure was opened, so a later component
    # (after the exposure cap frees up) must still be able to trade.
    service = build_service(portfolio=PortfolioState(exposure_pct=0.95, regime="expansion"))
    assert await service.process_aggregated(aggregated()) is None
    await service.update_portfolio(exposure_pct=0.10)
    assert await service.process_aggregated(aggregated()) is not None


@pytest.mark.asyncio
async def test_ledger_survives_a_restart():
    from .test_repository import FakeRepository

    repo = FakeRepository()
    first = build_service(repository=repo)
    event = aggregated()
    assert await first.process_aggregated(event) is not None

    second = build_service(repository=repo)
    await second.restore()
    # Same event redelivered to the restarted service (durables replay) → no order.
    assert await second.process_aggregated(event) is None


@pytest.mark.asyncio
async def test_session_key_uses_event_time_not_wall_clock():
    ts = datetime(2026, 7, 20, 23, 30, tzinfo=UTC)
    assert session_of(ts) == "2026-07-20"
    # A naive timestamp is read as-is rather than shifted into a different day.
    assert session_of(datetime(2026, 7, 20, 23, 30)) == "2026-07-20"


def test_ledger_prunes_old_sessions():
    ledger = OrderLedger(keep_sessions=2)
    for day in range(1, 6):
        ledger.record("AAPL", "BUY", f"2026-07-{day:02d}")
    snapshot = ledger.snapshot()
    assert set(snapshot.values()) == {"2026-07-04", "2026-07-05"}
    assert ledger.already_placed("AAPL", "BUY", "2026-07-05") is True
    assert ledger.already_placed("AAPL", "BUY", "2026-07-01") is False


def test_ledger_restore_ignores_empty_payload():
    ledger = OrderLedger()
    ledger.restore(None)
    ledger.restore({})
    assert ledger.snapshot() == {}
