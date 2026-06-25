"""Testy NatsPublisher (JetStream) z atrapą kontekstu JS — bez sieci."""

from types import SimpleNamespace

import pytest
from trading_common.events import MarketDataUpdatedEvent

from src.events.publisher import NatsPublisher


class _FakeJS:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def publish(self, subject, payload, headers=None):  # type: ignore[no-untyped-def]
        self.calls.append((subject, payload, headers))
        return SimpleNamespace(stream="MARKET_DATA", seq=len(self.calls))


@pytest.mark.asyncio
async def test_publish_uses_subject_payload_and_dedup_header():
    js = _FakeJS()
    publisher = NatsPublisher(js)
    event = MarketDataUpdatedEvent(symbol="AAPL", interval="1d", rows_count=3)

    await publisher.publish(event)

    assert len(js.calls) == 1
    subject, payload, headers = js.calls[0]
    assert subject == "market_data.updated"
    assert b"AAPL" in payload
    # event_id jako Nats-Msg-Id => deduplikacja po stronie JetStream
    assert headers["Nats-Msg-Id"] == event.event_id
