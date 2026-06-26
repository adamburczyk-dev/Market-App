"""Momentum-on-ranks signal rule.

Direction/strength comes from the cross-sectional percentile rank of momentum
(in [0,1]); the raw RSI provides an overbought/oversold sanity filter so we
don't chase already-stretched names.
"""

from dataclasses import dataclass

from trading_common.schemas import Signal


@dataclass
class MomentumParams:
    buy_rank: float = 0.80
    sell_rank: float = 0.20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


def generate_signal(
    momentum_rank: float, rsi: float, params: MomentumParams
) -> tuple[Signal, float]:
    """Return (signal, confidence in [0, 1]).

    BUY  — top of the universe on momentum and not yet overbought.
    SELL — bottom of the universe on momentum and not yet oversold.
    HOLD — otherwise (confidence 0.5).
    """
    if momentum_rank >= params.buy_rank and rsi < params.rsi_overbought:
        return Signal.BUY, momentum_rank
    if momentum_rank <= params.sell_rank and rsi > params.rsi_oversold:
        return Signal.SELL, 1.0 - momentum_rank
    return Signal.HOLD, 0.5
