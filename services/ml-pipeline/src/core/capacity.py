"""Capacity probe — does the data contain anything to fit at all?

Run #2 produced a train AUC of ~0.52: the model did not fit even the window it
was shown. Two very different causes look identical from that number:

  (a) optimization — capacity, learning rate, dropout, early stopping;
  (b) no signal — the features simply do not determine the labels.

The response to (a) is to fix training; the response to (b) is to change the
features or the labels, and no amount of extra symbols or history helps. This
module separates them with one deliberately unfair experiment: fit a large,
unregularized model on the training window and read the TRAIN AUC.

The reading requires a control. A big enough network memorizes random labels,
so a high train AUC alone proves nothing — it has to be compared against the
same model trained on SHUFFLED labels, which measures pure memorization
capacity for this feature matrix and sample size. What matters is the gap:

  real ≫ shuffled  → learnable structure exists; the production config is
                     leaving it on the table (cause (a))
  real ≈ shuffled  → the fit is memorization only; the features carry no
                     information about the labels (cause (b))

This is a diagnostic, never a model that may serve: it is fitted on the
training window with no validation, so its out-of-sample meaning is nil.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
import structlog

from src.core.dataset import Dataset
from src.core.evaluation import auc
from src.core.model import TrainConfig, train_classifier

logger = structlog.get_logger()

# Deliberately unfair: wide, no dropout, no weight decay, no early stopping
# (min_epochs == max_epochs), so the only limit is what the data supports.
# Sized to finish in minutes on ~40k rows: the probe is comparative (identical
# capacity on real and shuffled labels), so the absolute width matters far less
# than it being several times the production model's (32, 16).
OVERFIT_CONFIG = TrainConfig(
    hidden=(128, 64),
    dropout=0.0,
    weight_decay=0.0,
    lr=3e-3,
    batch_size=512,
    max_epochs=80,
    min_epochs=80,
    patience=10_000,
)


@dataclass(frozen=True)
class CapacityProbe:
    n_rows: int
    n_features: int
    auc_train_real: float  # mean over the real-label fits
    auc_train_shuffled: float  # mean over the shuffled-label controls
    auc_train_production: float
    gap: float  # real − shuffled: learnable structure beyond memorization
    real_runs: tuple[float, ...]
    shuffled_runs: tuple[float, ...]
    separated: bool  # every real fit beat every control fit
    margin: float  # worst real − best control; ≤ 0 means the runs overlap
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def _fit_and_score(
    x: np.ndarray, y: np.ndarray, config: TrainConfig, seed_offset: int = 0
) -> float:
    """Train on (x, y) and return AUC on the SAME rows — fit quality, not skill."""
    cfg = replace(config, seed=config.seed + seed_offset)
    # The validation split is the training data itself: this probe asks how well
    # the model CAN fit, so holding data out would answer a different question.
    model = train_classifier(x, y, x, y, [f"f{i}" for i in range(x.shape[1])], cfg)
    return auc(y, model.predict_proba(x))


def run_capacity_probe(
    ds: Dataset,
    production_config: TrainConfig | None = None,
    overfit_config: TrainConfig | None = None,
    max_rows: int = 60_000,
    gap_threshold: float = 0.05,
    n_real: int = 2,
    n_shuffles: int = 3,
) -> CapacityProbe:
    """Fit an over-parameterized model on real and on shuffled labels.

    Each side is fitted several times with different seeds. One fit per side
    cannot separate a small real gap from seed noise, and the first real run of
    this probe landed at +0.0518 against a 0.05 threshold — a verdict that
    turned on 0.0018 is not a verdict. The decision now needs the gap AND
    complete separation: the worst real fit must beat the best control fit.

    ``max_rows`` caps the probe (the most recent rows) so the run stays within
    an ops-call budget; every fit sees exactly the same matrix.
    """
    if ds.n_samples < 200:
        raise ValueError(f"capacity probe needs ≥ 200 rows, got {ds.n_samples}")

    x = ds.x[-max_rows:]
    y = ds.y[-max_rows:]
    if len(np.unique(y)) < 2:
        raise ValueError("capacity probe needs both classes present")

    over = overfit_config or OVERFIT_CONFIG
    real_runs = tuple(_fit_and_score(x, y, over, seed_offset=i) for i in range(n_real))

    rng = np.random.default_rng(over.seed)
    shuffled_runs: list[float] = []
    for i in range(n_shuffles):
        labels = y.copy()
        rng.shuffle(labels)  # fresh permutation each time, not one reused draw
        shuffled_runs.append(_fit_and_score(x, labels, over, seed_offset=100 + i))

    production = _fit_and_score(x, y, production_config or TrainConfig(), seed_offset=900)

    real = float(np.mean(real_runs))
    shuffled = float(np.mean(shuffled_runs))
    gap = real - shuffled
    margin = min(real_runs) - max(shuffled_runs)
    separated = margin > 0.0

    if gap >= gap_threshold and separated:
        verdict = (
            "learnable structure EXISTS — every real-label fit beat every shuffled-label "
            "control, by "
            f"{margin:.4f} at worst. A flat production train AUC is then an optimization/"
            "regularization problem (capacity, lr, dropout, early stopping), not an empty "
            "dataset. In-sample structure is NOT proof of an out-of-sample edge — it means "
            "the features are worth extracting harder."
        )
    elif gap >= gap_threshold:
        verdict = (
            f"INCONCLUSIVE — the mean gap is {gap:+.4f} but the real and control fits overlap "
            f"(worst real − best control = {margin:+.4f}), so seed noise explains it as well "
            "as signal does. Re-run with more repeats before acting on it."
        )
    else:
        verdict = (
            "NO learnable structure at this horizon — the big model does no better on real "
            "labels than on shuffled ones, so its fit is memorization. More symbols or more "
            "history will not change this; the features or the labels have to change."
        )

    logger.info(
        "Capacity probe complete",
        rows=int(x.shape[0]),
        auc_real=round(real, 4),
        auc_shuffled=round(shuffled, 4),
        gap=round(gap, 4),
        margin=round(margin, 4),
        separated=separated,
    )
    return CapacityProbe(
        n_rows=int(x.shape[0]),
        n_features=int(x.shape[1]),
        auc_train_real=real,
        auc_train_shuffled=shuffled,
        auc_train_production=production,
        gap=gap,
        real_runs=real_runs,
        shuffled_runs=tuple(shuffled_runs),
        separated=separated,
        margin=margin,
        verdict=verdict,
    )
