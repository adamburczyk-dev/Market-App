"""Testy NatsPublisher (JetStream) z atrapą kontekstu JS — bez sieci."""

from types import SimpleNamespace

import pytest
from trading_common.events import OrderRequestedEvent

from src.events.publisher import NatsPublisher


class _FakeJS:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def publish(self, subject, payload, headers=None):  # type: ignore[no-untyped-def]
        self.calls.append((subject, payload, headers))
        return SimpleNamespace(stream="ORDERS", seq=len(self.calls))


@pytest.mark.asyncio
async def test_publish_order_with_dedup_header():
    js = _FakeJS()
    publisher = NatsPublisher(js)
    event = OrderRequestedEvent(
        symbol="AAPL", side="BUY", quantity=50.0, price=100.0, strategy_name="momentum_rank"
    )
    await publisher.publish(event)
    assert len(js.calls) == 1
    subject, payload, headers = js.calls[0]
    assert subject == "order.requested"
    assert b"AAPL" in payload
    assert headers["Nats-Msg-Id"] == event.event_id
