"""Derive Tier-2 attribute features from fundamentals and company style.

Pure functions; every derivation is conservative — a missing or degenerate
input yields no feature rather than a guessed one. Downstream consumers must
use the cross-sectional *ranks* of these values (López de Prado), which the
existing ``/ranked`` endpoint provides once the attributes are merged in.
"""

from typing import Any

# style → (style_growth, style_value); blend sits between the poles
STYLE_ENCODING: dict[str, tuple[float, float]] = {
    "growth": (1.0, 0.0),
    "value": (0.0, 1.0),
    "blend": (0.5, 0.5),
}


def fundamental_features(payload: dict[str, Any]) -> dict[str, float]:
    """Features from a fundamental-data view (``{"statement": ..., "f_score": ...}``)."""
    out: dict[str, float] = {}
    f_score = payload.get("f_score")
    if isinstance(f_score, int | float):
        out["f_score"] = float(f_score)

    statement = payload.get("statement") or {}
    revenue = statement.get("revenue")
    net_income = statement.get("net_income")
    total_assets = statement.get("total_assets")
    total_liabilities = statement.get("total_liabilities")
    if net_income is not None and revenue:  # revenue 0/None → no margin
        out["fund_net_margin"] = net_income / revenue
    if net_income is not None and total_assets:
        out["fund_roa"] = net_income / total_assets
    if total_liabilities is not None and total_assets:
        out["fund_leverage"] = total_liabilities / total_assets
    return out


def style_features(style: str) -> dict[str, float]:
    """Numeric encoding of the classifier's investment style (unknown → none)."""
    encoding = STYLE_ENCODING.get(style)
    if encoding is None:
        return {}
    growth, value = encoding
    return {"style_growth": growth, "style_value": value}
