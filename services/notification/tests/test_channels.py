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


@pytest.mark.asyncio
async def test_email_channel_builds_and_sends_message():
    from src.core.channels import EmailChannel

    sent = []
    ch = EmailChannel(
        "smtp.test",
        587,
        "alerts@trading.local",
        ["ops@trading.local", "risk@trading.local"],
        sender=sent.append,
    )
    await ch.send(
        Alert(
            severity="critical",
            title="Circuit breaker RED",
            message="daily_loss=6.00%",
            source="risk.circuit_breaker",
            metadata={"level": "red", "action": "halt"},
        )
    )
    await ch.aclose()
    assert len(sent) == 1
    msg = sent[0]
    assert msg["Subject"] == "[CRITICAL] Circuit breaker RED"
    assert msg["From"] == "alerts@trading.local"
    assert msg["To"] == "ops@trading.local, risk@trading.local"
    body = msg.get_content()
    assert "daily_loss=6.00%" in body
    assert "source: risk.circuit_breaker" in body
    assert "action: halt" in body  # metadata included


@pytest.mark.asyncio
async def test_email_channel_sender_failure_propagates():
    from src.core.channels import EmailChannel

    def broken(msg):  # noqa: ARG001
        raise ConnectionRefusedError("smtp down")

    ch = EmailChannel("smtp.test", 587, "a@b", ["c@d"], sender=broken)
    with pytest.raises(ConnectionRefusedError):
        await ch.send(alert())  # dispatch() isolates this per channel in the service
