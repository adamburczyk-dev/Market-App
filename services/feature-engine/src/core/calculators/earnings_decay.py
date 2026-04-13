"""Earnings decay and PEAD (Post-Earnings Announcement Drift) signals."""

import math


def decay_weight(days_since_earnings: int, half_life: float = 30.0) -> float:
    """
    Exponential decay weight for earnings signal freshness.

    weight = 0.5^(days / half_life)
    At half_life days: weight = 0.5
    At 0 days: weight = 1.0
    """
    if days_since_earnings < 0:
        return 0.0
    return math.pow(0.5, days_since_earnings / half_life)


def surprise_score(
    actual_eps: float,
    consensus_eps: float,
    historical_std: float,
) -> float:
    """
    Standardized Unexpected Earnings (SUE).

    SUE = (actual - consensus) / historical_std
    Returns 0 if historical_std is zero (no surprise measurable).
    """
    if historical_std <= 0:
        return 0.0
    return (actual_eps - consensus_eps) / historical_std


def pead_signal(sue_score: float, days_since_earnings: int) -> float:
    """
    Post-Earnings Announcement Drift signal.

    Combines SUE magnitude with time decay.
    Research: Bernard & Thomas (1989) — drift persists 60-90 days.
    """
    weight = decay_weight(days_since_earnings)
    return sue_score * weight
