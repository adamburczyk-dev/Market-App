"""NotificationService — consume domain events, fan out alerts to channels."""

import asyncio
from collections import deque

import structlog
from trading_common.events import (
    CircuitBreakerTriggeredEvent,
    ModelDriftDetectedEvent,
    OrderFilledEvent,
    StrategyRevalidatedEvent,
    StrategyStatusChangedEvent,
)

from src.core import alerts
from src.core.channels import SEVERITY_RANK, Alert, Channel

logger = structlog.get_logger()


class NotificationService:
    def __init__(
        self,
        channels: list[Channel],
        min_severity: str = "info",
        recent_maxlen: int = 100,
    ) -> None:
        self._channels = channels
        self._min_rank = SEVERITY_RANK.get(min_severity, 0)
        self._recent: deque[Alert] = deque(maxlen=recent_maxlen)

    @property
    def channels(self) -> list[Channel]:
        return self._channels

    def recent(self, limit: int = 20) -> list[Alert]:
        return list(self._recent)[-limit:]

    async def dispatch(self, alert: Alert) -> None:
        """Send an alert to every channel, if it meets the min-severity threshold."""
        if alert.rank() < self._min_rank:
            logger.debug("Alert below threshold, suppressed", title=alert.title)
            return
        self._recent.append(alert)
        results = await asyncio.gather(
            *(ch.send(alert) for ch in self._channels), return_exceptions=True
        )
        for ch, res in zip(self._channels, results, strict=True):
            if isinstance(res, Exception):
                logger.warning("Channel delivery failed", channel=ch.name, error=str(res))

    # --- event handlers (one per subscribed subject) ---

    async def handle_circuit_breaker(self, data: bytes) -> None:
        event = CircuitBreakerTriggeredEvent.model_validate_json(data)
        await self.dispatch(alerts.from_circuit_breaker(event))

    async def handle_order_filled(self, data: bytes) -> None:
        event = OrderFilledEvent.model_validate_json(data)
        await self.dispatch(alerts.from_order_filled(event))

    async def handle_strategy_revalidated(self, data: bytes) -> None:
        event = StrategyRevalidatedEvent.model_validate_json(data)
        await self.dispatch(alerts.from_strategy_revalidated(event))

    async def handle_strategy_status_changed(self, data: bytes) -> None:
        event = StrategyStatusChangedEvent.model_validate_json(data)
        await self.dispatch(alerts.from_strategy_status_changed(event))

    async def handle_model_drift(self, data: bytes) -> None:
        event = ModelDriftDetectedEvent.model_validate_json(data)
        await self.dispatch(alerts.from_model_drift(event))
