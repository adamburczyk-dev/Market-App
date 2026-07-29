"""What trading actually costs (P5-2) — instead of a flat 5 bps.

Every evaluation so far charged 5 basis points per unit of turnover: the same
number for a mega-cap and for the thinnest name in the universe, and the same
number whether the book is $100k or $100M. That is not a cost model, it is a
placeholder, and it hides the one thing about costs that matters — **capacity**.
A strategy can be real at small size and impossible at large, and a flat rate
cannot express the difference.

Two components, both estimated from the OHLCV we already store.

**Market impact.** The square-root law: pushing a fraction `q` of a name's daily
volume through the book costs about `k · sigma · sqrt(q)` (Almgren et al.;
Torre). This is the term with real content here, because both of its inputs —
median dollar volume and daily volatility — are things we measure well, and
because it depends on ORDER SIZE. That is why the model takes an AUM. Stating
the book size is the honesty the flat rate was hiding: the same strategy has
different costs at different sizes, and now a report has to say which size it
was evaluated at.

**Half-spread.** Corwin & Schultz (2012) recover the bid-ask spread from daily
highs and lows: over one day the high/low range contains one day of volatility
plus the spread, over two days it contains twice the volatility and still one
spread, and the difference identifies them. Measured against synthetic paths
with a KNOWN injected spread, this implementation tracks it with roughly unit
slope — but two things limit what that is worth:

* its intercept moves with how finely the intraday path is monitored, which is
  not something daily bars tell us (a zero-spread series reads −9.9 bps at 390
  intraday observations and +9.1 at 4000), so the LEVEL is not identified; and
* the per-name noise is ±3.6 bps even over 252 sessions (±8.0 over 63).

Real S&P 500 half-spreads are on the order of 0.5–3 bps — below that noise. So
for a large-cap universe this term should be read as at-the-floor for nearly
every name, and the cost model is effectively impact-driven. That is not a
defect being hidden: `spread_identified_share` reports exactly how much of the
spread column is an estimate rather than the floor, and on a large-cap universe
the honest answer is "almost none of it".

Nothing here is calibrated against our own fills, because we have none. These
are literature estimators on our data: far better than 5 bps flat, and still
estimates. When paper trading produces fills, they get replaced by measurements.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from trading_common.prices import adjusted_closes
from trading_common.schemas import OHLCVBar

logger = structlog.get_logger()

# The estimator's level is not identified (see the module docstring), so the
# floor is what a liquid US large cap actually pays rather than what the
# estimator says. Above the cap an estimate is a bad-tick artefact, not a quote.
MIN_HALF_SPREAD_BPS = 1.0
MAX_HALF_SPREAD_BPS = 100.0
# Fewer bars than this and neither the spread nor the volatility input means
# anything; the name is reported untradeable rather than given a made-up number.
MIN_BARS = 21


@dataclass(frozen=True)
class CostParams:
    """Everything about cost that is an assumption rather than a measurement."""

    # Total book size. The whole reason impact can be computed at all — and the
    # number that decides whether a result survives at the size you want to run.
    aum_usd: float = 1_000_000.0
    # Square-root-law coefficient. Published estimates cluster around 0.5-1.0
    # for equities; 1.0 is the conservative end.
    impact_coefficient: float = 1.0
    # Trailing window for the liquidity and volatility inputs.
    window: int = 63
    # ...and a LONGER one for the spread, because it is a slower-moving property
    # and the estimator is noisy: measured on synthetic paths with a known
    # spread, the per-name standard deviation falls from 8.0 bps at 63 sessions
    # to 3.6 at 252 while the bias does not move. Sharing the 63-session window
    # cost half the precision for nothing — and the noise showed up as a
    # mega-cap's total cost swinging 2.5 -> 8.9 bps between two runs on the
    # same universe.
    spread_window: int = 252
    # A position is capped at this share of the portfolio (mirrors risk-mgmt's
    # 5% rule) — impact is charged on the money that actually moves.
    max_position_weight: float = 0.05
    # Refuse to model a trade beyond this share of a name's daily volume: past
    # it the square-root law stops being a reasonable extrapolation and the
    # honest answer is "this size is not tradeable in this name".
    max_participation: float = 0.10


@dataclass(frozen=True)
class SymbolCost:
    """One name's one-way cost at the configured book size."""

    symbol: str
    half_spread_bps: float
    impact_bps: float
    participation: float
    median_dollar_volume: float
    daily_volatility: float
    tradeable: bool
    # False when Corwin-Schultz returned a non-positive estimate and the floor
    # was used instead. Without this the floor is indistinguishable from a
    # measurement that happens to be small.
    spread_identified: bool

    @property
    def total_bps(self) -> float:
        return self.half_spread_bps + self.impact_bps

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "half_spread_bps": round(self.half_spread_bps, 2),
            "impact_bps": round(self.impact_bps, 2),
            "total_bps": round(self.total_bps, 2),
            "participation": round(self.participation, 5),
            "median_dollar_volume": round(self.median_dollar_volume, 0),
            "daily_volatility": round(self.daily_volatility, 5),
            "tradeable": self.tradeable,
            "spread_identified": self.spread_identified,
        }


_K = 3.0 - 2.0 * math.sqrt(2.0)


def _overnight_adjusted(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shift each day's range so the previous close sits inside it.

    Corwin-Schultz assumes a continuous price process. An overnight gap breaks
    that: the two-day range picks up the jump while the one-day ranges do not,
    which inflates `gamma` and drives the spread estimate down — measured here
    at 1.5% overnight volatility, an unadjusted zero-spread series reads −59 bps
    instead of −10. The adjustment removes the jump, not the range.
    """
    h, lo = highs.copy(), lows.copy()
    previous_close = closes[:-1]
    gap_up = np.where(lo[1:] > previous_close, lo[1:] - previous_close, 0.0)
    gap_down = np.where(h[1:] < previous_close, previous_close - h[1:], 0.0)
    shift = gap_down - gap_up
    h[1:] += shift
    lo[1:] += shift
    return h, lo


def corwin_schultz_half_spread(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray | None = None
) -> float:
    """RAW proportional half-spread in bps — may be negative (= not identified).

    With `beta` the sum of two consecutive days' squared log-ranges and `gamma`
    the squared log-range of the two days combined,

        alpha = (sqrt(2*beta) - sqrt(beta)) / k - sqrt(gamma / k),  k = 3 - 2*sqrt(2)
        S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

    `beta` and `gamma` are averaged over the window BEFORE the transform is
    applied, which is not cosmetic: the identity that makes the estimator work
    holds for the expectations, and applying it per two-day observation then
    aggregating turns a mean-zero estimator into a strictly positive one.
    Taking a median over only the positive per-observation values — the first
    version of this function — reads **65 bps on a zero-spread series**, because
    the negative half of a symmetric noise distribution has been thrown away.

    Callers floor the result; a non-positive return means the window did not
    identify a spread, which is information and must not be silently rounded up.
    """
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    if len(h) < 3 or len(lo) != len(h) or np.any(h <= 0) or np.any(lo <= 0):
        return 0.0
    if closes is not None:
        c = np.asarray(closes, dtype=float)
        if len(c) == len(h) and np.all(c > 0):
            h, lo = _overnight_adjusted(h, lo, c)
    if np.any(h <= 0) or np.any(lo <= 0):
        return 0.0

    log_hl = np.log(h / lo)
    beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2
    gamma = np.log(np.maximum(h[:-1], h[1:]) / np.minimum(lo[:-1], lo[1:])) ** 2
    usable = np.isfinite(beta) & np.isfinite(gamma)
    if not usable.any():
        return 0.0

    b = float(np.mean(beta[usable]))
    g = float(np.mean(gamma[usable]))
    if b <= 0 or g < 0:
        return 0.0
    alpha = (math.sqrt(2.0 * b) - math.sqrt(b)) / _K - math.sqrt(g / _K)
    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return float(min(spread / 2.0 * 10_000.0, MAX_HALF_SPREAD_BPS))


def estimate_symbol_cost(
    symbol: str,
    bars: list[OHLCVBar],
    params: CostParams | None = None,
) -> SymbolCost:
    """Per-name one-way cost in bps at the configured book size."""
    p = params or CostParams()
    by_time = sorted(bars, key=lambda b: b.timestamp)
    ordered = by_time[-p.window :]
    if len(ordered) < MIN_BARS:
        return SymbolCost(
            symbol=symbol,
            half_spread_bps=MAX_HALF_SPREAD_BPS,
            impact_bps=0.0,
            participation=0.0,
            median_dollar_volume=0.0,
            daily_volatility=0.0,
            tradeable=False,
            spread_identified=False,
        )

    spread_bars = by_time[-max(p.spread_window, p.window) :]
    highs = np.array([b.high for b in spread_bars], dtype=float)
    lows = np.array([b.low for b in spread_bars], dtype=float)
    raw_closes = np.array([b.close for b in spread_bars], dtype=float)
    raw_spread = corwin_schultz_half_spread(highs, lows, raw_closes)
    half_spread = max(raw_spread, MIN_HALF_SPREAD_BPS)

    # Raw close x raw volume: the money that actually changed hands. (Adjusted
    # prices are for measuring returns; this is measuring liquidity.)
    dollar_volume = np.array([b.close * b.volume for b in ordered], dtype=float)
    median_dv = float(np.median(dollar_volume))

    closes = adjusted_closes(ordered)
    returns = np.diff(closes) / closes[:-1]
    daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    order_usd = p.aum_usd * p.max_position_weight
    if median_dv <= 0:
        return SymbolCost(
            symbol=symbol,
            half_spread_bps=half_spread,
            impact_bps=0.0,
            participation=0.0,
            median_dollar_volume=0.0,
            daily_volatility=daily_vol,
            tradeable=False,
            spread_identified=raw_spread > MIN_HALF_SPREAD_BPS,
        )

    participation = order_usd / median_dv
    # Square-root law, evaluated at the capped participation so an untradeable
    # size does not silently produce a finite, comfortable-looking number.
    impact_bps = (
        p.impact_coefficient
        * daily_vol
        * math.sqrt(min(participation, p.max_participation))
        * 10_000.0
    )
    return SymbolCost(
        symbol=symbol,
        half_spread_bps=half_spread,
        impact_bps=impact_bps,
        participation=participation,
        median_dollar_volume=median_dv,
        daily_volatility=daily_vol,
        tradeable=participation <= p.max_participation,
        spread_identified=raw_spread > MIN_HALF_SPREAD_BPS,
    )


def estimate_costs(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    params: CostParams | None = None,
) -> dict[str, SymbolCost]:
    p = params or CostParams()
    costs = {s: estimate_symbol_cost(s, bars, p) for s, bars in bars_by_symbol.items() if bars}
    if costs:
        totals = [c.total_bps for c in costs.values()]
        logger.info(
            "Costs estimated",
            symbols=len(costs),
            aum_usd=p.aum_usd,
            median_total_bps=round(float(np.median(totals)), 2),
            untradeable=sum(1 for c in costs.values() if not c.tradeable),
        )
    return costs


def cost_table_bps(costs: dict[str, SymbolCost]) -> dict[str, float]:
    """symbol -> one-way cost in bps, for the portfolio evaluation."""
    return {s: c.total_bps for s, c in costs.items()}


def cost_summary(
    costs: dict[str, SymbolCost],
    params: CostParams | None = None,
    flat_baseline_bps: float = 5.0,
) -> dict[str, Any]:
    """The comparison that matters: modelled cost vs the flat rate we assumed.

    Reported at a stated AUM, because that is what the flat number hid. The same
    universe is cheap at small size and untradeable at large, and until now every
    evaluation implied the first without saying so.
    """
    p = params or CostParams()
    if not costs:
        return {"symbols": 0, "aum_usd": p.aum_usd}
    totals = np.array([c.total_bps for c in costs.values()], dtype=float)
    spreads = np.array([c.half_spread_bps for c in costs.values()], dtype=float)
    impacts = np.array([c.impact_bps for c in costs.values()], dtype=float)
    untradeable = sorted(s for s, c in costs.items() if not c.tradeable)
    identified = sum(1 for c in costs.values() if c.spread_identified)
    return {
        "symbols": len(costs),
        "aum_usd": p.aum_usd,
        "max_position_usd": p.aum_usd * p.max_position_weight,
        "impact_coefficient": p.impact_coefficient,
        "flat_baseline_bps": flat_baseline_bps,
        "half_spread_bps": _distribution(spreads),
        "impact_bps": _distribution(impacts),
        "total_bps": _distribution(totals),
        "cost_ratio_vs_flat": round(float(np.median(totals)) / flat_baseline_bps, 2),
        "untradeable": untradeable,
        "untradeable_share": round(len(untradeable) / len(costs), 4),
        "spread_identified_share": round(identified / len(costs), 4),
        "note": (
            "Literature estimators (Corwin-Schultz spread, square-root impact) on our own "
            "OHLCV, not measured fills. Read cost_ratio_vs_flat: above 1 and every past "
            "evaluation was optimistic. Impact scales with sqrt(AUM), so the number moves "
            "with book size — a result that survives at $1M may not at $50M. "
            "spread_identified_share says how much of the spread column is a real estimate "
            "rather than the floor: the estimator orders names by liquidity reliably, but "
            "its level is not identified from daily bars."
        ),
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "median": round(float(np.median(values)), 2),
        "p90": round(float(np.percentile(values, 90)), 2),
        "max": round(float(np.max(values)), 2),
    }


__all__ = [
    "MAX_HALF_SPREAD_BPS",
    "MIN_HALF_SPREAD_BPS",
    "CostParams",
    "SymbolCost",
    "corwin_schultz_half_spread",
    "cost_summary",
    "cost_table_bps",
    "estimate_costs",
    "estimate_symbol_cost",
]
