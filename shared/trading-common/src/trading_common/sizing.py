"""Position size from a calibrated probability — fractional Kelly (P5-3).

Today every position is sized from a fixed risk budget: 2% of equity at risk,
scaled down by drawdown. That budget is the same whether the model says 51% or
95%, which throws away the one number a classifier actually produces. Kelly is
the correct answer to "how much, given a probability", and the reason it is not
simply *the* answer is that it is correct only when the probability is right.

**The bet.** With triple-barrier labels the profit and loss targets are
symmetric (±pt_mult·sigma·sqrt(h)), so the payoff ratio is 1 and the Kelly
fraction collapses to

    f* = p - (1 - p) = 2p - 1

which is a useful sanity anchor: p = 0.5 sizes nothing, and the fraction is
linear in the edge rather than in the confidence.

**Why fractional.** Full Kelly is optimal only if `p` is exactly right. It is
not: it comes from a model with estimation error, and Kelly's downside is
brutally asymmetric — overbetting by 2x turns the growth-optimal bet into a
zero-growth one, and beyond that into ruin. Halving (or quartering) it costs
little growth and buys a large margin, which is why practitioners run quarter to
half Kelly. The default here is a quarter.

**Preconditions that are not negotiable.** Kelly consumes a CALIBRATED
probability; a model whose probabilities are systematically overconfident sizes
systematically too large, and the error compounds. That is precisely what gate
condition G4 tests, so this must not be driven by a model that has not passed
it. The output is also never larger than the existing risk envelope allows —
`kelly_weight` composes with the drawdown-adaptive budget by taking the smaller
of the two, so switching sizing on can only ever reduce exposure relative to
the rules already in force, never raise it.
"""

from dataclasses import dataclass

__all__ = ["KellyParams", "kelly_fraction", "kelly_weight", "scale_to_exposure"]


@dataclass(frozen=True)
class KellyParams:
    """The assumptions behind a Kelly size, all of them explicit."""

    # Fraction of full Kelly to actually bet. 1.0 is growth-optimal ONLY under a
    # perfectly known probability; 0.25 is the usual defensive choice.
    fraction: float = 0.25
    # Hard per-position ceiling — mirrors the non-negotiable 5% rule so a
    # confident model can never talk the book into a concentrated position.
    max_position_weight: float = 0.05
    # Ignore edges below this. Near p = 0.5 the sign of (2p - 1) is estimation
    # noise, and a portfolio of tiny noise-driven positions is pure cost.
    min_edge: float = 0.02
    # Payoff ratio (win magnitude / loss magnitude). 1.0 matches symmetric
    # triple barriers; a different barrier geometry must pass its real ratio.
    payoff_ratio: float = 1.0


def kelly_fraction(probability: float, params: KellyParams | None = None) -> float:
    """Full Kelly fraction for a binary bet, clipped at zero (long-only).

    `f* = (p*b - (1-p)) / b`, which for the symmetric case `b = 1` is `2p - 1`.
    A probability at or below the break-even point returns 0.0 rather than a
    negative number: the book is long-only (R4), so "bet against" is not a size,
    it is a decision not to trade.
    """
    p = params or KellyParams()
    prob = min(max(float(probability), 0.0), 1.0)
    b = p.payoff_ratio
    if b <= 0:
        return 0.0
    edge = (prob * b - (1.0 - prob)) / b
    if edge <= 0 or abs(2.0 * prob - 1.0) < p.min_edge:
        return 0.0
    return float(edge)


def kelly_weight(
    probability: float,
    envelope_weight: float,
    params: KellyParams | None = None,
) -> float:
    """Portfolio weight for one position: fractional Kelly, capped by the envelope.

    ``envelope_weight`` is what the existing drawdown-adaptive sizer would have
    allowed. Taking the MINIMUM is the whole safety argument: enabling this can
    only shrink a position relative to the rules already in force. A sizing rule
    that could exceed them would be a change to the risk limits wearing the
    costume of a model improvement.
    """
    p = params or KellyParams()
    raw = kelly_fraction(probability, p) * max(p.fraction, 0.0)
    return float(max(0.0, min(raw, p.max_position_weight, max(envelope_weight, 0.0))))


def scale_to_exposure(weights: dict[str, float], max_exposure: float = 0.80) -> dict[str, float]:
    """Scale a book down proportionally if it breaches the gross-exposure cap.

    Kelly sizes each position as if it were the only bet. Ten confident
    positions at 5% is a 50% book, twenty is 100% — and the 80% cap is one of
    the non-negotiable rules. Scaling proportionally preserves the RELATIVE
    conviction the model expressed, which truncating the tail would not.
    """
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= max_exposure or total <= 0:
        return {symbol: max(w, 0.0) for symbol, w in weights.items()}
    scale = max_exposure / total
    return {symbol: max(w, 0.0) * scale for symbol, w in weights.items()}
