"""Shallow PyTorch MLP + temperature calibration (plan §5).

At this data scale the model class barely matters — the network is deliberately
small and heavily regularized (dropout + weight decay + early stopping), and
the aggregator consumes *calibrated* probabilities, so temperature scaling on
the validation fold is part of the model, not an afterthought.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
from torch import nn

logger = structlog.get_logger()


@dataclass(frozen=True)
class TrainConfig:
    hidden: tuple[int, int] = (32, 16)
    dropout: float = 0.3
    lr: float = 3e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 200
    min_epochs: int = 30  # warm-up before early stopping may trigger (dropout
    # makes early val loss noisy — a lucky epoch-3 minimum must not stop training)
    patience: int = 15  # early-stopping epochs without val-loss improvement
    seed: int = 7


class MlpClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, int], dropout: float) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainedModel:
    """A fitted classifier plus everything needed to reproduce its outputs."""

    module: MlpClassifier
    temperature: float
    feature_names: list[str]
    config: TrainConfig
    history: dict[str, float] = field(default_factory=dict)
    # T0-3: what happened during the fit. Needed to tell "the model could not
    # learn" from "there was nothing to learn" — the two produce identical
    # out-of-sample metrics but demand opposite responses.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Calibrated P(up-barrier-first) for a (n, n_features) matrix."""
        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.as_tensor(x, dtype=torch.float32))
            return torch.sigmoid(logits / self.temperature).numpy()


def _fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """1-D NLL minimisation over the softmax temperature (Guo et al. 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits / torch.exp(log_t), targets)
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]
    return float(torch.exp(log_t).item())


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: TrainConfig | None = None,
    sample_weights: np.ndarray | None = None,
) -> TrainedModel:
    """Fit the MLP with early stopping on the validation fold, then calibrate.

    The validation fold does double duty: early stopping and temperature
    calibration. Class imbalance is handled with ``pos_weight``.

    ``sample_weights`` are the average-uniqueness weights (P0-3): rows whose
    label windows overlap share one market episode, so they must not each count
    as full evidence. Passing None keeps every row at weight 1.
    """
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    xt = torch.as_tensor(x_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.float32)
    xv = torch.as_tensor(x_val, dtype=torch.float32)
    yv = torch.as_tensor(y_val, dtype=torch.float32)

    model = MlpClassifier(xt.shape[1], cfg.hidden, cfg.dropout)
    positives = float(yt.sum().item())
    negatives = float(len(yt) - positives)
    pos_weight = torch.tensor(negatives / positives if positives > 0 else 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    if sample_weights is None:
        wt = torch.ones(len(yt), dtype=torch.float32)
    else:
        wt = torch.as_tensor(np.asarray(sample_weights, dtype=float), dtype=torch.float32)
        if len(wt) != len(yt):
            raise ValueError(f"sample_weights has {len(wt)} rows, y_train has {len(yt)}")

    def weighted(logits: torch.Tensor, targets: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Normalize by the weight sum, not the row count: otherwise down-weighting
        # rows also shrinks the gradient, which is a learning-rate change in
        # disguise rather than a change in what the model is asked to fit.
        return (loss_fn(logits, targets) * w).sum() / w.sum().clamp(min=1e-12)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    since_best = 0
    epochs_run = 0
    early_stop_reason = "max_epochs"

    for epoch in range(cfg.max_epochs):
        epochs_run = epoch + 1
        model.train()
        permutation = torch.randperm(len(xt))
        for start in range(0, len(xt), cfg.batch_size):
            batch = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = weighted(model(xt[batch]), yt[batch], wt[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(weighted(model(xv), yv, torch.ones(len(yv))).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if epoch + 1 >= cfg.min_epochs and since_best >= cfg.patience:
                early_stop_reason = "patience"
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(xv)
        temperature = _fit_temperature(val_logits, yv)
        if not (0.05 <= temperature <= 20.0):  # degenerate calibration → identity
            temperature = 1.0
        # A collapsed model and a correctly-humbled one look identical AFTER
        # calibration: when validation AUC is ~0.5 the optimal temperature grows
        # and flattens every probability onto the base rate. Separating the two
        # requires the spread BEFORE calibration and the temperature itself.
        pred_std_pre = float(torch.sigmoid(val_logits).std().item())
        pred_std_post = float(torch.sigmoid(val_logits / temperature).std().item())
        train_loss = float(weighted(model(xt), yt, wt).item())

    logger.info(
        "Classifier trained",
        epochs=best_epoch + 1,
        best_val_loss=round(best_val, 5),
        temperature=round(temperature, 3),
        train_rows=len(xt),
        val_rows=len(xv),
    )
    return TrainedModel(
        module=model,
        temperature=temperature,
        feature_names=list(feature_names),
        config=cfg,
        history={"best_val_loss": best_val, "epochs": float(best_epoch + 1)},
        diagnostics={
            "epochs_run": epochs_run,
            "best_epoch": best_epoch + 1,
            "early_stop_reason": early_stop_reason,
            "loss_train_final": round(train_loss, 6),
            "loss_val_final": round(best_val, 6),
            "calibration_temperature": round(temperature, 4),
            "pred_std_pre_calibration": round(pred_std_pre, 6),
            "pred_std_post_calibration": round(pred_std_post, 6),
            "n_train_rows": int(len(xt)),
            "n_val_rows": int(len(xv)),
        },
    )
