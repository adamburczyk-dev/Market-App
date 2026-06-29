"""Alert + delivery channels.

A channel turns an Alert into a delivered message. LogChannel is always on (works
without credentials); Slack/Telegram are HTTP-based and only constructed when their
config is present, so the service degrades gracefully to logging when nothing is wired.
"""

from dataclasses import dataclass, field
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
