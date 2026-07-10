"""Alert + delivery channels.

A channel turns an Alert into a delivered message. LogChannel is always on (works
without credentials); Slack/Telegram/Email are only constructed when their config
is present, so the service degrades gracefully to logging when nothing is wired.
"""

import asyncio
import smtplib
from collections.abc import Callable
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

import httpx
import structlog

logger = structlog.get_logger()

# info < warning < critical
SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Alert:
    severity: str  # "info" | "warning" | "critical"
    title: str
    message: str
    source: str  # originating event_type, e.g. "risk.circuit_breaker"
    metadata: dict[str, str] = field(default_factory=dict)

    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)


class Channel(Protocol):
    name: str

    async def send(self, alert: Alert) -> None: ...

    async def aclose(self) -> None: ...


class LogChannel:
    """Always-available sink — emits the alert through structlog at its level."""

    name = "log"

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    async def send(self, alert: Alert) -> None:
        self.sent.append(alert)
        log = logger.bind(channel=self.name, source=alert.source, **alert.metadata)
        if alert.severity == "critical":
            log.error(alert.title, message=alert.message, severity=alert.severity)
        elif alert.severity == "warning":
            log.warning(alert.title, message=alert.message, severity=alert.severity)
        else:
            log.info(alert.title, message=alert.message, severity=alert.severity)

    async def aclose(self) -> None:
        return None


class SlackChannel:
    """Posts to a Slack incoming-webhook URL."""

    name = "slack"

    def __init__(self, webhook_url: str, client: httpx.AsyncClient | None = None) -> None:
        self._url = webhook_url
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def send(self, alert: Alert) -> None:
        emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity, "🔵")
        text = f"{emoji} *{alert.title}*\n{alert.message}"
        resp = await self._client.post(self._url, json={"text": text})
        resp.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


class TelegramChannel:
    """Sends a message via the Telegram bot API."""

    name = "telegram"

    def __init__(
        self, bot_token: str, chat_id: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def send(self, alert: Alert) -> None:
        text = f"[{alert.severity.upper()}] {alert.title}\n{alert.message}"
        resp = await self._client.post(self._url, json={"chat_id": self._chat_id, "text": text})
        resp.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


class EmailChannel:
    """Sends the alert as a plain-text e-mail over SMTP.

    The blocking smtplib call runs in a worker thread. A fresh connection is
    opened per alert — volume is human-scale, so connection reuse isn't worth
    holding SMTP state. ``sender`` is injectable for tests.
    """

    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        from_addr: str,
        to_addrs: list[str],
        user: str | None = None,
        password: str | None = None,
        starttls: bool = True,
        timeout_s: float = 10.0,
        sender: Callable[[EmailMessage], None] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._from = from_addr
        self._to = to_addrs
        self._user = user
        self._password = password
        self._starttls = starttls
        self._timeout_s = timeout_s
        self._sender = sender or self._smtp_send

    def _smtp_send(self, msg: EmailMessage) -> None:
        with smtplib.SMTP(self._host, self._port, timeout=self._timeout_s) as smtp:
            if self._starttls:
                smtp.starttls()
            if self._user and self._password:
                smtp.login(self._user, self._password)
            smtp.send_message(msg)

    def _build(self, alert: Alert) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
        msg["From"] = self._from
        msg["To"] = ", ".join(self._to)
        lines = [alert.message, "", f"source: {alert.source}"]
        lines += [f"{key}: {value}" for key, value in sorted(alert.metadata.items())]
        msg.set_content("\n".join(lines))
        return msg

    async def send(self, alert: Alert) -> None:
        await asyncio.to_thread(self._sender, self._build(alert))

    async def aclose(self) -> None:
        return None
