"""Shallow PyTorch MLP + temperature calibration (plan §5).

At this data scale the model class barely matters — the network is deliberately
small and heavily regularized (dropout + weight decay + early stopping), and
the aggregator consumes *calibrated* probabilities, so temperature scaling on
the validation fold is part of the model, not an afterthought.
"""

from dataclasses import dataclass, field

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
) -> TrainedModel:
    """Fit the MLP with early stopping on the validation fold, then calibrate.

    The validation fold does double duty: early stopping and temperature
    calibration. Class imbalance is handled with ``pos_weight``.
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
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    since_best = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        permutation = torch.randperm(len(xt))
        for start in range(0, len(xt), cfg.batch_size):
            batch = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv), yv).item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if epoch + 1 >= cfg.min_epochs and since_best >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        temperature = _fit_temperature(model(xv), yv)
    if not (0.05 <= temperature <= 20.0):  # degenerate calibration → identity
        temperature = 1.0

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
    )
