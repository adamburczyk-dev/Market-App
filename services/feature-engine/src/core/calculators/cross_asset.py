"""Cross-asset momentum and risk appetite scoring."""

import numpy as np


def compute_cross_asset_scores(
    asset_returns_60d: dict[str, float],
    spy_return_60d: float,
) -> dict[str, float | dict[str, float]]:
    """
    Compute relative strength and risk appetite from cross-asset returns.

    For each asset: relative_strength = asset_return - spy_return
    Composite risk_appetite_score = mean of all relative strengths.

    Positive composite → risk-on environment.
    Negative composite → risk-off environment.

    Research: Asness et al. (2013) "Value and Momentum Everywhere"
    """
    if not asset_returns_60d:
        return {"relative_strength": {}, "risk_appetite_score": 0.0}

    relative_strength = {asset: ret - spy_return_60d for asset, ret in asset_returns_60d.items()}

    risk_appetite_score = float(np.mean(list(relative_strength.values())))

    return {
        "relative_strength": relative_strength,
        "risk_appetite_score": risk_appetite_score,
    }
