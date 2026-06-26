"""ExecutionService — paper-fill OrderRequestedEvents and feed portfolio back."""

import uuid

import structlog
from trading_common.events import OrderFilledEvent, OrderRequestedEvent

from src.core.paper_broker import PaperBroker
from src.core.risk_client import RiskClient
from src.events.publisher import Publisher

logger = structlog.get_logger()


class ExecutionService:
    def __init__(self, broker: PaperBroker, publisher: Publisher, risk_client: RiskClient) -> None:
        self._broker = broker
        self._publisher = publisher
        self._risk_client = risk_client

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    async def handle_order_event(self, data: bytes) -> None:
        order = OrderRequestedEvent.model_validate_json(data)
        await self.execute(order)

    async def execute(self, order: OrderRequestedEvent) -> OrderFilledEvent:
        """Paper-fill an order, publish OrderFilledEvent, push portfolio to risk-mgmt."""
        order_id = str(uuid.uuid4())
        fill = self._broker.fill(order_id, order.symbol, order.side, order.quantity, order.price)
        event = OrderFilledEvent(
            order_id=order_id,
            symbol=fill.symbol,
            filled_quantity=fill.quantity,
            filled_price=fill.price,
        )
        await self._publisher.publish(event)
        await self._risk_client.push_portfolio(self._broker.metrics())
        logger.info(
            "Order filled",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill.price,
        )
        return event
