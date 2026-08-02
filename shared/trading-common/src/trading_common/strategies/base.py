"""Strategy rule protocol + registry — the SHARED definition of a rule.

This lives in trading-common, not in the strategy service, for the same reason
`features` and `ranking` do: **backtest has to evaluate the same rule the
service trades**. Today it does not — `backtest/src/core/engine.py` reimplements
momentum on a price series while the live path runs it on cross-sectional ranks,
so a weekly revalidation grades a strategy that is not the one in production.
A rule reachable from both sides is the only structural fix; services cannot
import each other.

A rule is a pure function of two feature vectors:

- ``ranked`` — cross-sectional percentiles for this session (López de Prado).
  Use these for anything that means "compared to the rest of the universe".
- ``raw`` — the symbol's own values. Use these for anything self-referential
  (RSI level, %B, whether MACD crossed its signal line).

Both are already fetched by the strategy service, so the shape adds no new
dependency — it just stops the rule from reaching for whichever one happens to
be in scope.

`required_features` is validated against the names the system actually computes
**at registration time**. A rule that reads a feature nobody produces is then a
startup failure with a name in it, instead of a `KeyError` on the first symbol
of the first session it ever runs.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from trading_common.features import TECHNICAL_FEATURES
from trading_common.fundamentals import FUNDAMENTAL_FEATURE_NAMES
from trading_common.schemas import Signal

# Tier-2 attributes merged into the served vectors by feature-engine
# (`core/enrichment.py`): the Piotroski score and the investment-style encoding.
TIER2_FEATURES: frozenset[str] = frozenset({"f_score", "style_growth", "style_value"})

#: Everything a rule is allowed to ask for.
KNOWN_FEATURES: frozenset[str] = (
    TECHNICAL_FEATURES | TIER2_FEATURES | frozenset(FUNDAMENTAL_FEATURE_NAMES)
)


@dataclass(frozen=True)
class RuleOutput:
    """What a rule decided, and how wide the protection around it should be.

    The stop is expressed in **ATR multiples**, not in currency or percent: a
    rule knows how much room its own idea needs (a breakout has to survive the
    pullback that follows it; a mean-reversion entry is wrong the moment the
    move continues), but it does not know the price the order will be filled at.
    Converting to a level is the caller's job — and has to happen against the
    RAW execution price, which is why `atr_pct_14` rather than `atr_14` is the
    scale-free input.
    """

    signal: Signal
    confidence: float
    stop_atr_mult: float = 2.0
    take_profit_rr: float = 2.0
    reason: str = ""
    metadata: Mapping[str, float] = field(default_factory=dict)


HOLD = RuleOutput(signal=Signal.HOLD, confidence=0.5, reason="no_setup")


@runtime_checkable
class StrategyRule(Protocol):
    """A named, self-describing signal rule.

    The three descriptors are declared read-only, which is both what a frozen
    dataclass provides and what the design wants: `name` is the identity every
    per-strategy account downstream is keyed on, so a rule that could rename
    itself at runtime would silently split its own history in two.
    """

    @property
    def name(self) -> str: ...

    @property
    def required_features(self) -> frozenset[str]:
        """Names read from the symbol's OWN vector (`raw`)."""
        ...

    @property
    def required_ranks(self) -> frozenset[str]:
        """Names read as cross-sectional percentiles (`ranked`).

        Declared separately because it decides where a rule can be evaluated at
        all: a rank only exists relative to a universe, so a rule with any entry
        here cannot be reproduced by a single-symbol backtest. Naming the ranks
        lets the refusal say WHICH input is missing instead of just "no".
        """
        ...

    @property
    def default_params(self) -> Mapping[str, float]: ...

    def generate(
        self,
        ranked: Mapping[str, float],
        raw: Mapping[str, float],
        params: Mapping[str, float] | None = None,
    ) -> RuleOutput:
        """Return the decision for ONE symbol. Must not raise on missing input.

        A rule that cannot see what it needs returns HOLD: a partially-computed
        vector is normal early in a symbol's history, and a rule is not the
        right place to decide that the universe is broken.
        """
        ...


_REGISTRY: dict[str, "StrategyRule"] = {}


def register[R: StrategyRule](rule: R) -> R:
    """Register a rule instance. Returns it, so it can be used as a decorator.

    Refuses a duplicate name (two rules answering to one name would make
    `strategy_name` on the event ambiguous, and the whole per-strategy
    accounting downstream depends on it being an identity) and refuses a rule
    asking for features nothing computes.
    """
    if rule.name in _REGISTRY:
        raise ValueError(f"strategy already registered: {rule.name}")
    unknown = (set(rule.required_features) | set(rule.required_ranks)) - KNOWN_FEATURES
    if unknown:
        raise ValueError(
            f"strategy {rule.name} requires features nothing computes: {sorted(unknown)}"
        )
    _REGISTRY[rule.name] = rule
    return rule


def get_strategy(name: str) -> StrategyRule:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy: {name} (known: {sorted(_REGISTRY)})") from None


def all_strategies() -> list[StrategyRule]:
    """Every registered rule, in a stable order.

    Sorted by name on purpose: the order rules are evaluated in decides which
    signal reaches the aggregator first, and an import-order-dependent sequence
    would make the same inputs produce different event streams across restarts.
    """
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def strategy_names() -> list[str]:
    return sorted(_REGISTRY)


def apply_params(rule: StrategyRule, params: Mapping[str, float] | None) -> Mapping[str, float]:
    """Rule defaults with the caller's overrides on top."""
    if not params:
        return rule.default_params
    return {**rule.default_params, **params}


def pick(source: Mapping[str, float], *names: str) -> dict[str, float] | None:
    """All of `names` from `source`, or None if any is missing.

    Missing is the normal case (a symbol early in its history has no `sma_50`),
    so this returns None instead of raising and the rule returns HOLD. What it
    must NOT do is substitute a default: a made-up MACD is a made-up trade.
    """
    out: dict[str, float] = {}
    for name in names:
        value = source.get(name)
        if value is None:
            return None
        out[name] = float(value)
    return out


def saturating_confidence(magnitude: float, full_scale: float) -> float:
    """0.5 where the evidence is marginal, approaching 1.0 as it gets strong.

    Rules that fire on a threshold would otherwise all report the same
    confidence, and the aggregator weighs components by exactly that number —
    a barely-crossed moving average and a runaway trend must not vote alike.
    """
    if full_scale <= 0:
        return 0.5
    return 0.5 + 0.5 * min(1.0, abs(magnitude) / full_scale)
