"""Seed ensembling (P4-2) — the cheapest variance reduction available.

Run #2's folds swung from Sharpe −1.61 to +4.54 on the same data and the same
configuration. Some of that is the market; some is purely which random
initialisation the optimiser happened to start from. Averaging the calibrated
probabilities of several seeds removes the second part without touching the
first, and costs nothing but training time.

The ensemble deliberately exposes the SPREAD across members. If members
disagree wildly, the averaged prediction is a summary of noise and the report
should say so rather than present a smooth number. `seed_disagreement` is that
statistic — mean per-row standard deviation across members, on the probability
scale, so it reads next to `pred_std`: a disagreement of the same size as the
signal means the ensemble is averaging coin flips.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import structlog

logger = structlog.get_logger()


class Predictor(Protocol):
    """What every downstream stage actually needs from a fitted model."""

    feature_names: list[str]
    diagnostics: dict[str, Any]

    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class EnsembleModel:
    """Averages member probabilities. Satisfies the same Predictor interface."""

    members: list[Predictor]
    feature_names: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        stacked = np.vstack([m.predict_proba(x).reshape(-1) for m in self.members])
        return np.asarray(stacked.mean(axis=0), dtype=float)

    def member_predictions(self, x: np.ndarray) -> np.ndarray:
        """(n_members, n_rows) — the raw disagreement, for diagnostics."""
        return np.vstack([m.predict_proba(x).reshape(-1) for m in self.members])


def train_ensemble(
    fit: Callable[[int], Predictor | None],
    seeds: Sequence[int],
    reference_x: np.ndarray | None = None,
) -> Predictor | None:
    """Fit one model per seed and average them.

    `fit` takes a seed and returns a fitted model or None (an untrainable
    window). A single successful member is returned as itself rather than
    wrapped: an "ensemble" of one is just a model, and pretending otherwise
    would put a disagreement of exactly zero in the report.
    """
    members: list[Predictor] = []
    for seed in seeds:
        member = fit(seed)
        if member is not None:
            members.append(member)
    if not members:
        return None
    if len(members) == 1:
        return members[0]

    diagnostics: dict[str, Any] = {
        "model_kind": f"ensemble[{members[0].diagnostics.get('model_kind', 'mlp')}]",
        "n_members": len(members),
        "seeds": list(seeds)[: len(members)],
        # Averaged member diagnostics that G0 reads. A single member that never
        # improved should not be hidden by the others, so this is the MINIMUM:
        # G0 must still catch "the fit never improved".
        "best_epoch": min(
            (int(m.diagnostics.get("best_epoch", 0)) for m in members),
            default=0,
        ),
        "epochs_run": max((int(m.diagnostics.get("epochs_run", 0)) for m in members), default=0),
    }
    if reference_x is not None and len(reference_x):
        preds = np.vstack([m.predict_proba(reference_x).reshape(-1) for m in members])
        diagnostics["seed_disagreement"] = round(float(np.mean(np.std(preds, axis=0))), 5)
        diagnostics["pred_std_ensemble"] = round(float(np.std(preds.mean(axis=0))), 5)
        diagnostics["pred_std_member_mean"] = round(float(np.mean([np.std(p) for p in preds])), 5)
    logger.info("Ensemble trained", **{k: v for k, v in diagnostics.items() if k != "seeds"})
    return EnsembleModel(
        members=members,
        feature_names=list(members[0].feature_names),
        diagnostics=diagnostics,
    )


__all__ = ["EnsembleModel", "Predictor", "train_ensemble"]
