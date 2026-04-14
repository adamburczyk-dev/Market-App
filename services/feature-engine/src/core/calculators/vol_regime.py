"""VIX-based volatility regime classification and exposure scaling."""

import math

# VIX regime boundaries: (low, high, label)
VIX_REGIMES: list[tuple[float, float, str]] = [
    (0, 15, "low"),
    (15, 20, "normal"),
    (20, 30, "elevated"),
    (30, 40, "high"),
    (40, 100, "extreme"),
]

# Exposure scalar per regime — how much to scale position sizes
EXPOSURE_SCALAR: dict[str, float] = {
    "low": 1.20,
    "normal": 1.00,
    "elevated": 0.70,
    "high": 0.40,
    "extreme": 0.15,
}


def classify_vix(vix: float) -> str:
    """Classify VIX value into a volatility regime."""
    if vix < 0:
        raise ValueError(f"VIX cannot be negative, got {vix}")
    for low, high, label in VIX_REGIMES:
        if low <= vix < high:
            return label
    # VIX >= 100 — still extreme
    return "extreme"


def exposure_scalar(vix: float) -> float:
    """Return position size multiplier for current VIX level."""
    regime = classify_vix(vix)
    return EXPOSURE_SCALAR[regime]


def target_vol_position_size(
    portfolio_value: float,
    entry_price: float,
    realized_vol_annual: float,
    target_vol: float = 0.15,
) -> int:
    """
    Compute position size to target a specific portfolio volatility.

    shares = (portfolio_value * target_vol) / (entry_price * realized_vol)
    """
    if realized_vol_annual <= 0 or entry_price <= 0:
        return 0
    shares = (portfolio_value * target_vol) / (entry_price * realized_vol_annual)
    return int(math.floor(shares))
