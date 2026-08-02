"""S3: the buffer is keyed by (symbol, strategy), not by symbol.

Every test here fails on the previous implementation, and that is the point:
with one rule live, a symbol-keyed buffer and a hard-coded `source="strategy"`
were indistinguishable from correct behaviour.
"""

from datetime import UTC, datetime, timedelta

import pytest
from trading_common.events import SignalGeneratedEvent
from trading_common.strategies import strategy_names

from src.core.service import BufferedSignal, select_levels, strategy_source
from src.events.publisher import NullPublisher

from .conftest import build_service

MULTI_SOURCES = [
    strategy_source("momentum_rank"),
    strategy_source("donchian_breakout"),
    "ml",
    "macro",
]


def signal_event(
    strategy: str,
    side: str = "BUY",
    confidence: float = 0.9,
    symbol: str = "AAPL",
    price: float = 100.0,
    stop_loss: float = 95.0,
    take_profit: float = 110.0,
    ts: datetime | None = None,
) -> bytes:
    kwargs = {"timestamp": ts} if ts is not None else {}
    return (
        SignalGeneratedEvent(
            symbol=symbol,
            strategy_name=strategy,
            signal=side,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            **kwargs,
        )
        .model_dump_json()
        .encode()
    )


@pytest.mark.asyncio
async def test_a_second_strategy_does_not_overwrite_the_first():
    """THE regression. On the symbol-keyed buffer the later event replaced the
    earlier one, so which rule was heard depended on delivery order."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES)

    await service.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9))
    await service.handle_signal_generated(signal_event("donchian_breakout", "SELL", 0.8))

    event = publisher.published[-1]
    assert set(event.components_present) == {
        strategy_source("momentum_rank"),
        strategy_source("donchian_breakout"),
    }
    assert event.components_count == 2


@pytest.mark.asyncio
async def test_each_strategy_is_its_own_weighting_source():
    """FLOW: the adaptive loop can only learn per rule if the rules are
    distinguishable sources. As one lumped "strategy" source they never were."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES)
    # Varied returns: the information ratio is mean/std, so a constant series
    # has an undefined ratio and would leave both sources at the baseline.
    winning = [0.01, 0.03, 0.02, 0.04, 0.02]
    for _ in range(6):
        for r in winning:
            service.record_outcome(strategy_source("momentum_rank"), r)
            service.record_outcome(strategy_source("donchian_breakout"), -r)

    weights = service.weights()
    assert weights[strategy_source("momentum_rank")] > weights[strategy_source("donchian_breakout")]

    # ...and the sources the live path actually emits are the ones just weighted.
    await service.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9))
    await service.handle_signal_generated(signal_event("donchian_breakout", "BUY", 0.8))
    emitted = set(publisher.published[-1].components_present)
    assert emitted <= set(weights), f"unweighted source emitted: {emitted - set(weights)}"
    assert len(emitted) == 2


@pytest.mark.asyncio
async def test_opposite_votes_of_equal_weight_cancel_to_hold():
    """Two rules disagreeing is a real state of the world, not an error — and
    it has to be visible as HOLD rather than as whichever arrived last."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES)
    await service.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9))
    await service.handle_signal_generated(signal_event("donchian_breakout", "SELL", 0.9))

    event = publisher.published[-1]
    assert event.final_signal == "HOLD"
    assert event.stop_loss is None


# --- level selection ------------------------------------------------------


def entry(strategy: str, side: str, confidence: float, stop: float) -> BufferedSignal:
    from src.core.aggregator import SignalComponent

    return BufferedSignal(
        component=SignalComponent(strategy_source(strategy), side, confidence),
        price=100.0,
        stop_loss=stop,
        take_profit=110.0,
        strategy_name=strategy,
        at=datetime.now(UTC),
    )


def test_levels_come_from_the_most_confident_AGREEING_strategy():
    chosen = select_levels(
        [
            entry("a_rule", "SELL", 0.99, stop=999.0),  # disagrees — ineligible
            entry("b_rule", "BUY", 0.6, stop=94.0),
            entry("c_rule", "BUY", 0.8, stop=96.0),
        ],
        "BUY",
    )
    assert chosen is not None
    assert chosen.strategy_name == "c_rule"
    assert chosen.stop_loss == 96.0


def test_level_ties_break_on_the_strategy_name_not_on_arrival_order():
    """Without a deterministic tie-break the same inputs would produce
    different orders depending on what NATS delivered first."""
    a_first = [entry("a_rule", "BUY", 0.8, 94.0), entry("b_rule", "BUY", 0.8, 96.0)]
    assert select_levels(a_first, "BUY").strategy_name == "a_rule"
    assert select_levels(list(reversed(a_first)), "BUY").strategy_name == "a_rule"


def test_no_agreeing_strategy_means_no_levels():
    assert select_levels([entry("a_rule", "SELL", 0.9, 105.0)], "BUY") is None


@pytest.mark.asyncio
async def test_the_published_levels_belong_to_the_winning_strategy():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES)
    # The winner arrives FIRST on purpose: with the old symbol-keyed buffer the
    # later event simply overwrote it, so a test where the winner happens to
    # arrive last would pass on the broken code too.
    await service.handle_signal_generated(
        signal_event("donchian_breakout", "BUY", 0.9, stop_loss=96.0, take_profit=108.0)
    )
    await service.handle_signal_generated(
        signal_event("momentum_rank", "BUY", 0.6, stop_loss=94.0, take_profit=112.0)
    )

    event = publisher.published[-1]
    assert event.final_signal == "BUY"
    assert event.strategy_name == "donchian_breakout"
    assert event.stop_loss == 96.0
    assert event.take_profit == 108.0


# --- expiry is per entry --------------------------------------------------


@pytest.mark.asyncio
async def test_one_stale_strategy_does_not_retire_a_fresh_one():
    """Expiry used to be per symbol: an old entry took the whole name with it."""
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES, signal_ttl_s=3600.0)
    stale = datetime.now(UTC) - timedelta(hours=2)
    await service.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9, ts=stale))
    await service.handle_signal_generated(signal_event("donchian_breakout", "BUY", 0.8))

    event = publisher.published[-1]
    assert event.components_present == [strategy_source("donchian_breakout")]
    assert event.final_signal == "BUY"


@pytest.mark.asyncio
async def test_a_symbol_whose_entries_all_expired_stops_aggregating():
    publisher = NullPublisher()
    service = build_service(publisher=publisher, sources=MULTI_SOURCES, signal_ttl_s=3600.0)
    stale = datetime.now(UTC) - timedelta(hours=2)
    await service.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9, ts=stale))
    published_before = len(publisher.published)

    assert await service.aggregate_symbol("AAPL") is None
    assert len(publisher.published) == published_before
    # ...and the empty symbol key is gone, not left behind as a permanent no-op.
    assert "AAPL" not in service._buffer


@pytest.mark.asyncio
async def test_components_present_does_not_depend_on_delivery_order():
    """`components_present` is what tells an operator which rules were heard;
    an order-dependent list makes two identical decisions look different."""
    forward, backward = NullPublisher(), NullPublisher()
    a = build_service(publisher=forward, sources=MULTI_SOURCES)
    b = build_service(publisher=backward, sources=MULTI_SOURCES)

    await a.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9))
    await a.handle_signal_generated(signal_event("donchian_breakout", "BUY", 0.8))
    await b.handle_signal_generated(signal_event("donchian_breakout", "BUY", 0.8))
    await b.handle_signal_generated(signal_event("momentum_rank", "BUY", 0.9))

    assert forward.published[-1].components_present == backward.published[-1].components_present


def test_every_registered_strategy_gets_a_source_name():
    """The prefix is what keeps a rule named "ml" from colliding with the ML
    source — and the registry is the only list of rules there is."""
    sources = {strategy_source(name) for name in strategy_names()}
    assert len(sources) == len(strategy_names())
    assert "ml" not in sources and "macro" not in sources
