"""Combine directional signals from multiple sources into one weighted decision.

Each component contributes a signed confidence (+conf for BUY, −conf for SELL,
0 for HOLD); the weighted sum crosses a threshold to yield BUY / SELL / HOLD.
Weights come from the AdaptiveWeightOptimizer (performance-weighted per source);
the final signal is then gated by the shared CostAwareFilter.
"""

from dataclasses import dataclass, field


@dataclass
class SignalComponent:
    source: str
    signal: str  # "BUY" | "SELL" | "HOLD"
    confidence: float  # [0, 1]


@dataclass
class AggregationResult:
    symbol: str
    final_signal: str
    confidence: float
    score: float
    components_count: int
    weights: dict[str, float] = field(default_factory=dict)
    cost_filtered: bool = False


def _signed(component: SignalComponent) -> float:
    if component.signal == "BUY":
        return component.confidence
    if component.signal == "SELL":
        return -component.confidence
    return 0.0


def combine(
    components: list[SignalComponent],
    weights: dict[str, float],
    buy_threshold: float = 0.2,
) -> tuple[str, float, float]:
    """Return (final_signal, confidence, score) from weighted component votes."""
    if not components:
        return "HOLD", 0.0, 0.0
    score = sum(weights.get(c.source, 0.0) * _signed(c) for c in components)
    if score >= buy_threshold:
        signal = "BUY"
    elif score <= -buy_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"
    return signal, min(abs(score), 1.0), score
