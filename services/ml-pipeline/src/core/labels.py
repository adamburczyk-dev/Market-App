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


@dataclass(frozen=True)
class LabelParams:
    sigma_window: int = 20  # trailing sessions for the daily-vol estimate
    pt_mult: float = 2.0  # profit barrier, in sigma*sqrt(horizon) units
    sl_mult: float = 2.0  # loss barrier, in sigma*sqrt(horizon) units
    horizon: int = 10  # vertical barrier (sessions)


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
