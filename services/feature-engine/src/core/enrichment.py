"""Derive Tier-2 attribute features from company style.

The FUNDAMENTAL derivation moved to ``trading_common.fundamentals`` (P2-3):
ml-pipeline's dataset builder now derives the same features over history, and
two copies of that arithmetic across a service boundary is train/serve skew
waiting to happen — the same reason ``features``/``ranking`` are shared.
This module keeps the style encoding, which only the serving path has.

Pure functions; every derivation is conservative — a missing or degenerate
input yields no feature rather than a guessed one. Downstream consumers must
use the cross-sectional *ranks* of these values (López de Prado), which the
existing ``/ranked`` endpoint provides once the attributes are merged in.
"""

from typing import Any

import structlog
from pydantic import ValidationError
from trading_common.fundamentals import fundamental_features as _shared_features
from trading_common.schemas import FinancialStatements

logger = structlog.get_logger()

# style → (style_growth, style_value); blend sits between the poles
STYLE_ENCODING: dict[str, tuple[float, float]] = {
    "growth": (1.0, 0.0),
    "value": (0.0, 1.0),
    "blend": (0.5, 0.5),
}


def fundamental_features(payload: dict[str, Any]) -> dict[str, float]:
    """Adapt fundamental-data's HTTP view to the SHARED derivation.

    The view is ``{"statement": {...}, "f_score": n}``; the F-score is folded
    into the statement so one function computes the features on both paths.
    """
    raw = payload.get("statement")
    if not isinstance(raw, dict):
        return {}
    try:
        statement = FinancialStatements.model_validate(raw)
    except ValidationError as exc:
        # No features rather than an exception: this runs inside a NATS handler,
        # and raising here would NAK a payload that redelivery cannot fix.
        logger.warning("Unparseable fundamentals payload", error=str(exc))
        return {}
    f_score = payload.get("f_score")
    if statement.piotroski_f_score is None and isinstance(f_score, int):
        statement = statement.model_copy(update={"piotroski_f_score": f_score})
    return _shared_features(statement)


def style_features(style: str) -> dict[str, float]:
    """Numeric encoding of the classifier's investment style (unknown → none)."""
    encoding = STYLE_ENCODING.get(style)
    if encoding is None:
        return {}
    growth, value = encoding
    return {"style_growth": growth, "style_value": value}
