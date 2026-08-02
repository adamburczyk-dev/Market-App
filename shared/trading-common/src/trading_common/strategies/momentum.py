"""Momentum-on-ranks — the rule that has been live since the first signal.

Direction and strength come from the cross-sectional percentile rank of
momentum; the raw RSI is an overbought/oversold sanity filter so the rule does
not chase a name that has already run.

Moved here from `services/strategy/src/core/momentum.py` unchanged in behaviour:
same thresholds, same confidence, same HOLD. The move is what makes it
evaluable by backtest as well as by the service.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_common.schemas import Signal
from trading_common.strategies.base import HOLD, RuleOutput, apply_params, pick, register

DEFAULTS: Mapping[str, float] = {
    "buy_rank": 0.80,
    "sell_rank": 0.20,
    "rsi_overbought": 70.0,
    "rsi_oversold": 30.0,
    "stop_atr_mult": 2.0,
    "take_profit_rr": 2.0,
}


@dataclass(frozen=True)
class MomentumRank:
    name: str = "momentum_rank"
    # The rank is the whole rule; the RSI level is only a filter on top of it.
    required_features: frozenset[str] = frozenset({"rsi_14"})
    required_ranks: frozenset[str] = frozenset({"momentum_20"})
    default_params: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULTS))

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        p = apply_params(self, params)
        rank = pick(ranked, "momentum_20")
        levels = pick(raw, "rsi_14")
        if rank is None or levels is None:
            return HOLD
        momentum_rank, rsi = rank["momentum_20"], levels["rsi_14"]
        meta = {"momentum_rank": momentum_rank, "rsi": rsi}

        if momentum_rank >= p["buy_rank"] and rsi < p["rsi_overbought"]:
            return RuleOutput(
                signal=Signal.BUY,
                confidence=momentum_rank,
                stop_atr_mult=p["stop_atr_mult"],
                take_profit_rr=p["take_profit_rr"],
                reason="top_of_universe_not_overbought",
                metadata=meta,
            )
        if momentum_rank <= p["sell_rank"] and rsi > p["rsi_oversold"]:
            return RuleOutput(
                signal=Signal.SELL,
                confidence=1.0 - momentum_rank,
                stop_atr_mult=p["stop_atr_mult"],
                take_profit_rr=p["take_profit_rr"],
                reason="bottom_of_universe_not_oversold",
                metadata=meta,
            )
        return HOLD


momentum_rank = register(MomentumRank())
