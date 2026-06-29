"""ExecutionService — paper-fill OrderRequestedEvents and feed portfolio back."""

import uuid

import structlog
from trading_common.events import (
    MarketDataUpdatedEvent,
    OrderFilledEvent,
    OrderRequestedEvent,
)
from trading_common.schemas import Interval

from src.core.market_data_client import MarketDataClient
from src.core.paper_broker import PaperBroker
from src.core.repository import BrokerRepository, NullBrokerRepository
from src.core.risk_client import RiskClient
from src.events.publisher import Publisher

logger = structlog.get_logger()


class ExecutionService:
    def __init__(
        self,
        broker: PaperBroker,
        publisher: Publisher,
        risk_client: RiskClient,
        market_client: MarketDataClient | None = None,
        repository: BrokerRepository | None = None,
    ) -> None:
        self._broker = broker
        self._publisher = publisher
        self._risk_client = risk_client
        self._market_client = market_client
        self._repository = repository or NullBrokerRepository()

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    async def restore(self) -> None:
        """Load persisted broker state (cash / positions) on startup."""
        snapshot = await self._repository.load()
        if snapshot is None:
            return
        self._broker.restore(snapshot)
        logger.info(
            "Restored broker",
            equity=self._broker.equity,
            positions=self._broker.positions(),
        )

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
        await self._repository.save(self._broker.snapshot())
        await self._risk_client.push_portfolio(self._broker.metrics())
        logger.info(
            "Order filled",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill.price,
        )
        return event

    async def handle_market_data_event(self, data: bytes) -> None:
        event = MarketDataUpdatedEvent.model_validate_json(data)
        await self.mark_position(event.symbol, Interval(event.interval))

    async def mark_position(self, symbol: str, interval: Interval) -> None:
        """Re-mark a held position to the latest market price; push portfolio if changed."""
        if self._market_client is None or symbol not in self._broker.positions():
            return
        close = await self._market_client.latest_close(symbol, interval)
        if close is None:
            return
        self._broker.mark(symbol, close)
        await self._repository.save(self._broker.snapshot())
        await self._risk_client.push_portfolio(self._broker.metrics())
        logger.info("Re-marked position", symbol=symbol, price=close)
