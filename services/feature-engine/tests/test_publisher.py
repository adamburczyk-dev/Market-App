"""Testy NatsPublisher (JetStream) z atrapą kontekstu JS — bez sieci."""

from types import SimpleNamespace

import pytest
from trading_common.events import FeaturesReadyEvent

from src.events.publisher import NatsPublisher


class _FakeJS:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def publish(self, subject, payload, headers=None):  # type: ignore[no-untyped-def]
        self.calls.append((subject, payload, headers))
        return SimpleNamespace(stream="FEATURES", seq=len(self.calls))


@pytest.mark.asyncio
async def test_publish_features_ready_with_dedup_header():
    js = _FakeJS()
    publisher = NatsPublisher(js)
    event = FeaturesReadyEvent(symbol="AAPL", interval="1d", features_count=12, tier=1)

    await publisher.publish(event)

    assert len(js.calls) == 1
    subject, payload, headers = js.calls[0]
    assert subject == "features.ready"
    assert b"AAPL" in payload
    assert headers["Nats-Msg-Id"] == event.event_id
