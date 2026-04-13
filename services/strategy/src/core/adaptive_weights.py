"""Adaptive signal weight optimization — EWP method."""

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SignalPerformance:
    """Tracks rolling performance for a single signal source."""

    outcomes: deque[float] = field(default_factory=lambda: deque(maxlen=60))

    @property
    def hit_rate(self) -> float:
        """Fraction of positive outcomes."""
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o > 0) / len(self.outcomes)

    @property
    def avg_return(self) -> float:
        """Mean return across outcomes."""
        if not self.outcomes:
            return 0.0
        return sum(self.outcomes) / len(self.outcomes)

    @property
    def information_ratio(self) -> float:
        """Mean / std of outcomes. Returns 0 if insufficient data."""
        if len(self.outcomes) < 2:
            return 0.0
        mean = self.avg_return
        variance = sum((o - mean) ** 2 for o in self.outcomes) / (len(self.outcomes) - 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return mean / std


class AdaptiveWeightOptimizer:
    """
    Dynamically adjusts signal source weights based on rolling performance.

    Method: exponentially weighted performance (EWP).
    weight_i = exp(lambda * IR_i) / sum(exp(lambda * IR_j))
    Floored at min_weight, capped at max_weight, then re-normalized.

    Research: DeMiguel et al. (2009) — adaptive portfolio weights
    outperform static allocation in non-stationary environments.
    """

    def __init__(
        self,
        signal_sources: list[str],
        lookback_days: int = 60,
        smoothing_lambda: float = 2.0,
        min_weight: float = 0.05,
        max_weight: float = 0.60,
    ) -> None:
        self.sources = signal_sources
        self.lookback_days = lookback_days
        self.smoothing_lambda = smoothing_lambda
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.performance: dict[str, SignalPerformance] = {
            src: SignalPerformance(outcomes=deque(maxlen=lookback_days)) for src in signal_sources
        }

    def record_outcome(self, source: str, daily_return: float) -> None:
        """Record a daily return for a signal source."""
        if source in self.performance:
            self.performance[source].outcomes.append(daily_return)

    def compute_weights(self) -> dict[str, float]:
        """
        Compute normalized weights for all signal sources.

        Returns equal weights if no data available.
        """
        n = len(self.sources)
        if n == 0:
            return {}

        equal_weight = 1.0 / n

        # Check if any source has data
        has_data = any(len(self.performance[s].outcomes) > 1 for s in self.sources)
        if not has_data:
            return {s: equal_weight for s in self.sources}

        # Compute raw exponential weights from IR
        ir_values = {s: self.performance[s].information_ratio for s in self.sources}
        # Clamp exponent to avoid overflow (exp(709) ≈ max float64)
        max_exp = 500.0
        raw_weights = {
            s: math.exp(min(self.smoothing_lambda * ir, max_exp)) for s, ir in ir_values.items()
        }

        # Normalize
        total = sum(raw_weights.values())
        weights = {s: w / total for s, w in raw_weights.items()}

        # Apply floor and cap
        weights = {s: max(self.min_weight, min(self.max_weight, w)) for s, w in weights.items()}

        # Re-normalize after floor/cap
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}

        return weights
