"""Mean-reversion rule: RSI extreme confirmed by the Bollinger band position.

This also settles **D5** (open since the audit): the RSI filter becomes a rule
of its own instead of another condition bolted onto momentum. The two are
opposite bets — momentum buys the top of the universe, this buys a name that
has been sold off — so folding them together would have averaged out the very
disagreement the aggregator exists to weigh.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_common.schemas import Signal
from trading_common.strategies.base import (
    HOLD,
    RuleOutput,
    apply_params,
    pick,
    register,
    saturating_confidence,
)

DEFAULTS: Mapping[str, float] = {
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    # %B at/below 0.05 means the close sits on or under the lower band.
    "band_lower": 0.05,
    "band_upper": 0.95,
    # RSI distance past the threshold that counts as full confidence.
    "full_confidence_rsi": 10.0,
    # A reversion entry is wrong the moment the move continues, so it gets a
    # tighter stop than the trend rules — and the target is the middle band,
    # not a trend, which is why the reward:risk is 1.5 rather than 2.5.
    "stop_atr_mult": 1.5,
    "take_profit_rr": 1.5,
}


@dataclass(frozen=True)
class RsiBollingerReversion:
    name: str = "rsi_bollinger_reversion"
    required_features: frozenset[str] = frozenset({"rsi_14", "bb_pct_b"})
    required_ranks: frozenset[str] = frozenset()
    default_params: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULTS))

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        p = apply_params(self, params)
        f = pick(raw, "rsi_14", "bb_pct_b")
        if f is None:
            return HOLD
        rsi, pct_b = f["rsi_14"], f["bb_pct_b"]
        meta = {"rsi": rsi, "bb_pct_b": pct_b}

        # Both conditions are required: RSI alone flags a strong trend as often
        # as an exhausted one, and the band position is what says the move is
        # stretched relative to the name's OWN recent range.
        if rsi <= p["rsi_oversold"] and pct_b <= p["band_lower"]:
            return RuleOutput(
                signal=Signal.BUY,
                confidence=saturating_confidence(p["rsi_oversold"] - rsi, p["full_confidence_rsi"]),
                stop_atr_mult=p["stop_atr_mult"],
                take_profit_rr=p["take_profit_rr"],
                reason="oversold_at_lower_band",
                metadata=meta,
            )
        if rsi >= p["rsi_overbought"] and pct_b >= p["band_upper"]:
            return RuleOutput(
                signal=Signal.SELL,
                confidence=saturating_confidence(
                    rsi - p["rsi_overbought"], p["full_confidence_rsi"]
                ),
                stop_atr_mult=p["stop_atr_mult"],
                take_profit_rr=p["take_profit_rr"],
                reason="overbought_at_upper_band",
                metadata=meta,
            )
        return HOLD


rsi_bollinger_reversion = register(RsiBollingerReversion())
