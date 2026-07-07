"""Testy obsługi wiadomości w EventSubscriber (ack / term / nak)."""

import pytest

from src.events.subscriber import EventSubscriber


class _FakeMsg:
    def __init__(self, data: bytes = b"{}") -> None:
        self.data = data
        self.acked = False
        self.termed = False
        self.naked = False

    async def ack(self) -> None:
        self.acked = True

    async def term(self) -> None:
        self.termed = True

    async def nak(self) -> None:
        self.naked = True


def _subscriber(handler) -> EventSubscriber:  # type: ignore[no-untyped-def]
    return EventSubscriber(js=None, subject="s", durable="d", handler=handler)


@pytest.mark.asyncio
async def test_acks_on_success():
    received: list[bytes] = []

    async def handler(data: bytes) -> None:
        received.append(data)

    msg = _FakeMsg(b"payload")
    await _subscriber(handler)._on_message(msg)
    assert msg.acked is True
    assert received == [b"payload"]


@pytest.mark.asyncio
async def test_terminates_poison_message():
    async def handler(data: bytes) -> None:
        raise ValueError("malformed event")

    msg = _FakeMsg()
    await _subscriber(handler)._on_message(msg)
    assert msg.termed is True
    assert msg.acked is False
    assert msg.naked is False


@pytest.mark.asyncio
async def test_naks_on_transient_error():
    async def handler(data: bytes) -> None:
        raise RuntimeError("market-data temporarily down")

    msg = _FakeMsg()
    await _subscriber(handler)._on_message(msg)
    assert msg.naked is True
    assert msg.acked is False
    assert msg.termed is False
