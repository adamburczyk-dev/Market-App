"""Trend-following rules: moving-average crossover, MACD confirmation, breakout."""

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

CROSSOVER_DEFAULTS: Mapping[str, float] = {
    # Separation (as a fraction of the slow average) at which the rule is fully
    # confident. 3% is roughly a month of trend at typical large-cap vol.
    "full_confidence_spread": 0.03,
    "stop_atr_mult": 2.0,
    "take_profit_rr": 2.0,
}


@dataclass(frozen=True)
class SmaEmaCrossover:
    """Two moving-average pairs must agree — the simplest possible baseline.

    Requiring BOTH the fast EMA pair and the slower SMA pair to point the same
    way is the whole rule: a single crossover fires constantly in a range, and
    a rule that trades every whipsaw tells you nothing about whether trend
    following works on this universe.

    Every input is on the adjusted price scale, so comparing them is valid; the
    output is a fraction, so applying it to the raw execution price is too.
    """

    name: str = "sma_ema_crossover"
    required_features: frozenset[str] = frozenset({"ema_12", "ema_26", "sma_20", "sma_50"})
    required_ranks: frozenset[str] = frozenset()
    default_params: Mapping[str, float] = field(default_factory=lambda: dict(CROSSOVER_DEFAULTS))

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        p = apply_params(self, params)
        f = pick(raw, "ema_12", "ema_26", "sma_20", "sma_50")
        if f is None or f["ema_26"] <= 0 or f["sma_50"] <= 0:
            return HOLD
        fast_spread = f["ema_12"] / f["ema_26"] - 1.0
        slow_spread = f["sma_20"] / f["sma_50"] - 1.0
        meta = {"fast_spread": fast_spread, "slow_spread": slow_spread}

        if fast_spread > 0 and slow_spread > 0:
            direction = Signal.BUY
        elif fast_spread < 0 and slow_spread < 0:
            direction = Signal.SELL
        else:
            return HOLD
        # The weaker of the two agreeing legs sets the strength — the rule is
        # only as convinced as its least convinced half.
        magnitude = min(abs(fast_spread), abs(slow_spread))
        return RuleOutput(
            signal=direction,
            confidence=saturating_confidence(magnitude, p["full_confidence_spread"]),
            stop_atr_mult=p["stop_atr_mult"],
            take_profit_rr=p["take_profit_rr"],
            reason="ema_and_sma_pairs_agree",
            metadata=meta,
        )


MACD_DEFAULTS: Mapping[str, float] = {
    # Histogram at which the rule is fully confident, as a fraction of price.
    "full_confidence_hist": 0.01,
    "stop_atr_mult": 2.0,
    "take_profit_rr": 2.0,
}


@dataclass(frozen=True)
class MacdConfirmation:
    """MACD histogram confirming the direction of the 20-day return.

    **Named for what it does.** The plan called this slot `macd_divergence`, and
    divergence is not implementable here: it compares the swings of price and
    oscillator over time, while a rule sees ONE feature vector with no history
    in it. Shipping a confirmation rule under a divergence name would have made
    every later report about it wrong. Real divergence needs lagged features —
    a serving-contract change, like `beta_60`.

    Confirmation is still worth its own rule: the histogram is the second
    derivative of trend, so it turns before the return does, and a return that
    the oscillator does not back is the one most likely to be ending.
    """

    name: str = "macd_confirmation"
    required_features: frozenset[str] = frozenset({"macd_hist", "close", "return_20d"})
    required_ranks: frozenset[str] = frozenset()
    default_params: Mapping[str, float] = field(default_factory=lambda: dict(MACD_DEFAULTS))

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        p = apply_params(self, params)
        f = pick(raw, "macd_hist", "close", "return_20d")
        if f is None or f["close"] <= 0:
            return HOLD
        # The histogram is in price units; scaling by price makes the threshold
        # mean the same thing for a $20 stock and a $2000 one.
        hist = f["macd_hist"] / f["close"]
        ret = f["return_20d"]
        meta = {"macd_hist_pct": hist, "return_20d": ret}

        if hist > 0 and ret > 0:
            direction = Signal.BUY
        elif hist < 0 and ret < 0:
            direction = Signal.SELL
        else:
            return HOLD
        return RuleOutput(
            signal=direction,
            confidence=saturating_confidence(hist, p["full_confidence_hist"]),
            stop_atr_mult=p["stop_atr_mult"],
            take_profit_rr=p["take_profit_rr"],
            reason="histogram_confirms_20d_return",
            metadata=meta,
        )


BREAKOUT_DEFAULTS: Mapping[str, float] = {
    # How far past the channel (in channel widths) counts as full confidence.
    "full_confidence_excess": 0.10,
    # Breakouts have to survive the pullback that follows them, so they get a
    # wider stop and a longer target than the mean-reversion rules.
    "stop_atr_mult": 2.5,
    "take_profit_rr": 2.5,
}


@dataclass(frozen=True)
class DonchianBreakout:
    """Close outside the PRIOR 20-session channel.

    `donchian_pos_20` is the close's position in that channel, so > 1.0 is a
    breakout and < 0.0 a breakdown. It excludes today's own bar — a channel
    containing today can never be broken, and a rule keyed on a condition that
    cannot occur is silent forever without ever erroring.
    """

    name: str = "donchian_breakout"
    required_features: frozenset[str] = frozenset({"donchian_pos_20"})
    required_ranks: frozenset[str] = frozenset()
    default_params: Mapping[str, float] = field(default_factory=lambda: dict(BREAKOUT_DEFAULTS))

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        p = apply_params(self, params)
        f = pick(raw, "donchian_pos_20")
        if f is None:
            return HOLD
        pos = f["donchian_pos_20"]
        meta = {"donchian_pos_20": pos}

        if pos > 1.0:
            excess = pos - 1.0
            direction = Signal.BUY
        elif pos < 0.0:
            excess = -pos
            direction = Signal.SELL
        else:
            return HOLD
        return RuleOutput(
            signal=direction,
            confidence=saturating_confidence(excess, p["full_confidence_excess"]),
            stop_atr_mult=p["stop_atr_mult"],
            take_profit_rr=p["take_profit_rr"],
            reason="closed_outside_prior_20d_channel",
            metadata=meta,
        )


sma_ema_crossover = register(SmaEmaCrossover())
macd_confirmation = register(MacdConfirmation())
donchian_breakout = register(DonchianBreakout())
