"""Rule-based company classification → investment style + model-stack routing.

Style (growth / value / blend) is scored from valuation & growth metrics when
available; with no metrics it falls back to a sector prior, then to blend. The
model stack (which ML ensemble serves this company) is routed from (style, cap
tier) — growth vs value stocks behave differently, and large- vs small-caps have
different data depth, so they get different stacks.
"""

from dataclasses import dataclass
from enum import StrEnum


class Style(StrEnum):
    GROWTH = "growth"
    VALUE = "value"
    BLEND = "blend"


# Sectors that lean a given style when valuation metrics are missing.
GROWTH_SECTORS = frozenset(
    {"Information Technology", "Communication Services", "Consumer Discretionary", "Health Care"}
)
VALUE_SECTORS = frozenset(
    {"Financials", "Energy", "Utilities", "Consumer Staples", "Materials", "Real Estate"}
)


@dataclass(frozen=True)
class StyleThresholds:
    growth_revenue: float = 0.15  # YoY revenue growth ≥ → growth signal
    growth_earnings: float = 0.15  # YoY earnings growth ≥ → growth signal
    growth_pe: float = 25.0  # rich P/E (market pricing growth)
    growth_max_dividend: float = 0.01  # low/no dividend → growth signal
    value_pe: float = 15.0  # cheap P/E → value signal
    value_pb: float = 1.5  # cheap P/B → value signal
    value_dividend: float = 0.03  # income → value signal


DEFAULT_THRESHOLDS = StyleThresholds()


@dataclass
class ValuationMetrics:
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    dividend_yield: float | None = None


@dataclass
class ClassificationResult:
    style: str
    model_stack: str
    cap_tier: str
    growth_score: int
    value_score: int
    basis: str  # "metrics" | "sector" | "default"


def cap_tier(market_cap: float | None) -> str:
    if market_cap is None:
        return "unknown"
    if market_cap >= 200e9:
        return "mega"
    if market_cap >= 10e9:
        return "large"
    if market_cap >= 2e9:
        return "mid"
    if market_cap >= 300e6:
        return "small"
    return "micro"


def _cap_bucket(tier: str) -> str:
    """Coarse liquidity bucket for model routing."""
    return "large" if tier in ("mega", "large") else "small"


def route_model_stack(style: str, tier: str) -> str:
    return f"{style}_{_cap_bucket(tier)}cap_v1"


def _score_style(metrics: ValuationMetrics, t: StyleThresholds) -> tuple[int, int]:
    """Return (growth_score, value_score) counting only present signals."""
    growth = 0
    value = 0
    if metrics.revenue_growth is not None and metrics.revenue_growth >= t.growth_revenue:
        growth += 1
    if metrics.earnings_growth is not None and metrics.earnings_growth >= t.growth_earnings:
        growth += 1
    if metrics.pe_ratio is not None and metrics.pe_ratio >= t.growth_pe:
        growth += 1
    if metrics.dividend_yield is not None and metrics.dividend_yield < t.growth_max_dividend:
        growth += 1
    if metrics.pe_ratio is not None and 0 < metrics.pe_ratio <= t.value_pe:
        value += 1
    if metrics.pb_ratio is not None and metrics.pb_ratio <= t.value_pb:
        value += 1
    if metrics.dividend_yield is not None and metrics.dividend_yield >= t.value_dividend:
        value += 1
    return growth, value


def _sector_style(sector: str | None) -> Style | None:
    if sector in GROWTH_SECTORS:
        return Style.GROWTH
    if sector in VALUE_SECTORS:
        return Style.VALUE
    return None


def classify(
    sector: str | None,
    market_cap: float | None,
    metrics: ValuationMetrics | None = None,
    thresholds: StyleThresholds = DEFAULT_THRESHOLDS,
) -> ClassificationResult:
    """Classify style + route the model stack for a company."""
    metrics = metrics or ValuationMetrics()
    growth, value = _score_style(metrics, thresholds)

    if growth == 0 and value == 0:
        # No metric signals — lean on the sector prior, else blend.
        sector_style = _sector_style(sector)
        style = sector_style if sector_style is not None else Style.BLEND
        basis = "sector" if sector_style is not None else "default"
    elif growth > value:
        style = Style.GROWTH
        basis = "metrics"
    elif value > growth:
        style = Style.VALUE
        basis = "metrics"
    else:
        style = Style.BLEND
        basis = "metrics"

    tier = cap_tier(market_cap)
    return ClassificationResult(
        style=style.value,
        model_stack=route_model_stack(style.value, tier),
        cap_tier=tier,
        growth_score=growth,
        value_score=value,
        basis=basis,
    )
