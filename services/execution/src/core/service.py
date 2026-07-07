"""ExecutionService — paper-fill OrderRequestedEvents and feed portfolio back.

Order semantics live here (the broker stays a plain ledger):
- Idempotent fills: the order's ``event_id`` is the order_id — a redelivered
  event is skipped, so a transient NATS/Redis error cannot double-fill (R3).
- Long-only: a SELL is an exit — capped at the held quantity and skipped when
  flat, so live behavior matches the long/flat backtest engine (R4).
- Protective exits: each re-mark checks the position's SL/TP and generates a
  paper exit when a level is breached (R5).
"""

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

    async def execute(self, order: OrderRequestedEvent) -> OrderFilledEvent | None:
        """Paper-fill an order, publish OrderFilledEvent, push portfolio to risk-mgmt.

        Returns None when the order is a duplicate delivery or a long-only
        violation (SELL with no position).
        """
        if self._broker.is_processed(order.event_id):
            logger.warning(
                "Duplicate order delivery — skipped",
                symbol=order.symbol,
                order_id=order.event_id,
            )
            return None

        quantity = order.quantity
        if order.side == "SELL":
            held = self._broker.position_qty(order.symbol)
            if held <= 0:
                logger.info("Long-only: SELL without a position skipped", symbol=order.symbol)
                return None
            quantity = min(quantity, held)

        fill = self._broker.fill(
            order.event_id,
            order.symbol,
            order.side,
            quantity,
            order.price,
            stop_loss=order.stop_loss if order.side == "BUY" else None,
            take_profit=order.take_profit if order.side == "BUY" else None,
        )
        if fill is None:  # raced duplicate — state untouched
            return None
        event = OrderFilledEvent(
            order_id=fill.order_id,
            symbol=fill.symbol,
            filled_quantity=fill.quantity,
            filled_price=fill.price,
        )
        # Save BEFORE publish: a crash between fill and save replays cleanly (the
        # restored snapshot predates both the fill and the dedup entry), while a
        # publish failure after save is deduped on redelivery — losing one
        # OrderFilled alert beats double-filling the position.
        await self._repository.save(self._broker.snapshot())
        await self._publisher.publish(event)
        await self._risk_client.push_portfolio(self._broker.metrics())
        logger.info(
            "Order filled",
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=fill.price,
        )
        return event

    async def handle_market_data_event(self, data: bytes) -> None:
        event = MarketDataUpdatedEvent.model_validate_json(data)
        await self.mark_position(event.symbol, Interval(event.interval))

    async def mark_position(self, symbol: str, interval: Interval) -> None:
        """Re-mark a held position; exit it if a protective level is breached."""
        if self._market_client is None or symbol not in self._broker.positions():
            return
        close = await self._market_client.latest_close(symbol, interval)
        if close is None:
            return
        self._broker.mark(symbol, close)
        exit_event = await self._maybe_protective_exit(symbol, close)
        await self._repository.save(self._broker.snapshot())
        await self._risk_client.push_portfolio(self._broker.metrics())
        logger.info(
            "Re-marked position",
            symbol=symbol,
            price=close,
            protective_exit=exit_event is not None,
        )

    async def _maybe_protective_exit(self, symbol: str, price: float) -> OrderFilledEvent | None:
        trigger = self._broker.protective_trigger(symbol)
        if trigger is None:
            return None
        quantity = self._broker.position_qty(symbol)
        fill = self._broker.fill(f"exit-{uuid.uuid4()}", symbol, "SELL", quantity, price)
        if fill is None:
            return None
        event = OrderFilledEvent(
            order_id=fill.order_id,
            symbol=symbol,
            filled_quantity=fill.quantity,
            filled_price=fill.price,
        )
        await self._publisher.publish(event)
        logger.warning(
            "Protective exit executed",
            symbol=symbol,
            trigger=trigger,
            quantity=quantity,
            price=fill.price,
        )
        return event
