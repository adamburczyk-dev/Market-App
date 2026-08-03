"""Triple-barrier labeling (López de Prado) — the mandated label scheme.

Fixed-horizon returns mislabel volatile paths (a +1% 10-day return that drew
down −8% first is not a "win"). The triple barrier asks which came first: the
profit barrier, the loss barrier, or the time (vertical) barrier. Barriers
scale with trailing volatility so a label means the same thing for a calm
mega-cap and a volatile small-cap. Parameters per docs/ml_integration_plan.md
§4: ±2·σ₂₀·√h around the reference close, vertical barrier h = 10 sessions.
"""

import math
from dataclasses import dataclass

import numpy as np

# The vertical barrier, declared ONCE. Four objects need this number to agree:
# the label (how far the path is scanned), the walk-forward purge (how much of
# the training window overlaps the test labels), the published event (what the
# probability is about), and the outcome resolver (when a vote has matured).
# They used to carry four independent defaults, and the dangerous disagreement
# is silent: a label horizon LARGER than the purge horizon leaks label window
# into every test block and makes the metrics look BETTER. Nothing raises.
# Same class of defect as MAX_OHLCV_LIMIT declared twice in two values.
LABEL_HORIZON = 10


def outcome_drop_after_days(horizon: int = LABEL_HORIZON) -> int:
    """Calendar days after which an unresolved vote is given up on.

    Sessions are not days: `horizon` sessions span `horizon * 365.25/252`
    calendar days, and the cutoff is expressed in days because that is what a
    wall clock measures. Three horizons of slack absorbs holidays and a late
    monitor run. Derived rather than typed — a literal here is the same trap
    one horizon change later (at h=10 this returns 44, the 42 it replaces).
    """
    return math.ceil(3 * horizon * 365.25 / 252)


@dataclass(frozen=True)
class LabelParams:
    sigma_window: int = 20  # trailing sessions for the daily-vol estimate
    # E1/P1-1: 2.0 made the barriers unreachable — measured through the full
    # pipeline, 90.9% of labels timed out on the vertical barrier, which turns
    # the triple barrier into a fixed-horizon sign label with extra steps.
    # `calibrate_barriers` scans multipliers on real paths and reports the mix;
    # 1.0 is what it picks (about half the labels touch a horizontal barrier).
    pt_mult: float = 1.0  # profit barrier, in sigma*sqrt(horizon) units
    sl_mult: float = 1.0  # loss barrier, in sigma*sqrt(horizon) units
    horizon: int = LABEL_HORIZON  # vertical barrier (sessions)
    # E1/P1-3: label the return RELATIVE to the cross-section instead of the
    # absolute one. A cross-sectional model is asked "which names beat the
    # rest"; an absolute label asks it to predict the market too, and the
    # market is the part it has no information about. Excess labels scan
    # close-to-close cumulative excess (no intraday path exists for a
    # synthetic market leg), so they trade intraday precision for measuring
    # the right quantity.
    excess: bool = False


@dataclass(frozen=True)
class BarrierOutcome:
    label: int  # 1 = up barrier first / vertical with positive return, else 0
    touch_index: int  # bar index at which the label resolved
    barrier: str  # "upper" | "lower" | "vertical"


def trailing_sigma(closes: np.ndarray, i: int, window: int) -> float | None:
    """Std of daily log returns over the trailing ``window`` sessions ending at i.

    None when there is not enough history or the segment is flat/degenerate —
    a zero-width barrier would label noise.
    """
    if i < window:
        return None
    segment = closes[i - window : i + 1]  # window+1 prices → window returns
    if np.any(segment <= 0):
        return None
    returns = np.diff(np.log(segment))
    sigma = float(np.std(returns, ddof=1))
    return sigma if sigma > 0 else None


def triple_barrier_label(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    i: int,
    params: LabelParams | None = None,
) -> BarrierOutcome | None:
    """Label the sample whose features are known at ``close_i``.

    Path scanning starts at the NEXT bar (entry after the signal, consistent
    with the backtest engine's next-bar accounting) and uses daily highs/lows —
    intraday extremes touch barriers that closes miss. If both barriers are
    touched on the same bar the LOWER wins (conservative for a long label).
    A vertical hit resolves by the sign of the net close-to-close return.

    Returns None when the sample cannot be labeled honestly: not enough
    trailing history for the vol estimate, or the label window is truncated
    by the end of history before any barrier is touched.
    """
    p = params or LabelParams()
    sigma = trailing_sigma(closes, i, p.sigma_window)
    if sigma is None:
        return None
    n = len(closes)
    if i + 1 >= n:
        return None  # no future bars at all

    width = sigma * math.sqrt(p.horizon)
    upper = closes[i] * (1.0 + p.pt_mult * width)
    lower = closes[i] * (1.0 - p.sl_mult * width)

    end = min(i + p.horizon, n - 1)
    for j in range(i + 1, end + 1):
        if lows[j] <= lower:  # checked first: same-bar double touch counts as loss
            return BarrierOutcome(label=0, touch_index=j, barrier="lower")
        if highs[j] >= upper:
            return BarrierOutcome(label=1, touch_index=j, barrier="upper")

    if end < i + p.horizon:
        return None  # window truncated by end of history → unresolved, drop
    return BarrierOutcome(
        label=1 if closes[end] > closes[i] else 0, touch_index=end, barrier="vertical"
    )


def excess_barrier_label(
    closes: np.ndarray,
    market: np.ndarray,
    i: int,
    params: LabelParams | None = None,
) -> BarrierOutcome | None:
    """Triple barrier on the return RELATIVE to ``market`` (same index basis).

    ``market`` is the cross-section's own benchmark path (equal-weight median
    of the universe), so the label answers "did this name beat the others",
    which is the question a ranking model can actually answer. Barrier width
    scales with the trailing volatility of the EXCESS daily return — a name
    that tracks the market closely needs a narrower barrier to say something.

    Scanning is close-to-close: an intraday excess path would require intraday
    market levels aligned to each bar, which we do not have. The cost is that
    a barrier touched and reversed within one session is missed.
    """
    p = params or LabelParams()
    n = len(closes)
    if i + 1 >= n or len(market) != n:
        return None
    if i < p.sigma_window or np.any(closes[i - p.sigma_window : i + 1] <= 0):
        return None
    if np.any(market[i - p.sigma_window : i + 1] <= 0):
        return None

    stock_returns = np.diff(np.log(closes[i - p.sigma_window : i + 1]))
    market_returns = np.diff(np.log(market[i - p.sigma_window : i + 1]))
    sigma = float(np.std(stock_returns - market_returns, ddof=1))
    if sigma <= 0:
        return None

    width = sigma * math.sqrt(p.horizon)
    upper, lower = p.pt_mult * width, -p.sl_mult * width

    end = min(i + p.horizon, n - 1)
    for j in range(i + 1, end + 1):
        if market[j] <= 0 or closes[j] <= 0:
            return None
        excess = math.log(closes[j] / closes[i]) - math.log(market[j] / market[i])
        if excess <= lower:  # same-bar ambiguity resolves as a loss, as above
            return BarrierOutcome(label=0, touch_index=j, barrier="lower")
        if excess >= upper:
            return BarrierOutcome(label=1, touch_index=j, barrier="upper")

    if end < i + p.horizon:
        return None
    final = math.log(closes[end] / closes[i]) - math.log(market[end] / market[i])
    return BarrierOutcome(label=1 if final > 0 else 0, touch_index=end, barrier="vertical")
