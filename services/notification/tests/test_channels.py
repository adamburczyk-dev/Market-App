"""Tests for delivery channels (log + Slack/Telegram HTTP payloads)."""

import json

import httpx
import pytest

from src.core.channels import Alert, LogChannel, SlackChannel, TelegramChannel


def alert(severity: str = "critical") -> Alert:
    return Alert(
        severity=severity, title="Boom", message="something", source="risk.circuit_breaker"
    )


@pytest.mark.asyncio
async def test_log_channel_records():
    ch = LogChannel()
    await ch.send(alert("warning"))
    assert len(ch.sent) == 1
    assert ch.sent[0].title == "Boom"


@pytest.mark.asyncio
async def test_slack_channel_posts_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ch = SlackChannel("https://hooks.slack.test/abc", client=client)
    await ch.send(alert("critical"))
    await ch.aclose()
    assert captured["url"] == "https://hooks.slack.test/abc"
    assert "Boom" in captured["json"]["text"]
    assert "🔴" in captured["json"]["text"]  # critical emoji


@pytest.mark.asyncio
async def test_telegram_channel_posts_chat_id_and_text():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ch = TelegramChannel("BOTTOKEN", "12345", client=client)
    await ch.send(alert("warning"))
    await ch.aclose()
    assert "BOTTOKEN" in captured["url"]
    assert captured["json"]["chat_id"] == "12345"
    assert "WARNING" in captured["json"]["text"]


@pytest.mark.asyncio
async def test_slack_channel_raises_on_http_error():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    ch = SlackChannel("https://hooks.slack.test/abc", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        await ch.send(alert())
    await ch.aclose()
