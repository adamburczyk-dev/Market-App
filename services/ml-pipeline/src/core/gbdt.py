"""Gradient-boosted trees as a CHALLENGER to the MLP (P4-1).

At this sample size, on tabular cross-sectional features, boosted trees usually
beat a neural net — nets pull ahead when the cross-section is very wide and
carries many characteristics (Gu, Kelly & Xiu 2020). Our capacity probe asks
the same question from the MLP side; this answers it from the other.

Deliberately built to be COMPARABLE rather than merely good:

- the same fit/validation split, the same uniqueness weights, the same early
  stopping on the validation fold;
- the same temperature calibration applied to the raw margin. LightGBM's
  logloss output is already reasonable, but G4 judges calibration, and two
  models scored on a metric one of them was post-processed for is not a
  comparison;
- the same `predict_proba(x) -> np.ndarray` interface, so every downstream
  metric, the portfolio construction and the whole gate run unchanged.

What it is NOT: a production model. `MlflowModelStore` persists an MLP
`state_dict` and reconstructs `MlpClassifier`, so a GBDT cannot be registered
or promoted yet. That is on purpose — the registry format should follow the
measurement, not be widened for a challenger that has not yet won.
"""

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import structlog
import torch

from src.core.model import _fit_temperature

logger = structlog.get_logger()


@dataclass(frozen=True)
class GbdtConfig:
    """Shallow and heavily regularized on purpose.

    The effective sample here is a few hundred independent observations, not
    the ~50k rows the matrix reports (overlapping labels, correlated names).
    A deep forest would memorize that in seconds — the capacity probe already
    showed an unregularized model reaching 0.71 train AUC on pure noise.
    """

    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 200
    subsample: float = 0.7
    subsample_freq: int = 1
    colsample_bytree: float = 0.7
    reg_lambda: float = 10.0
    early_stopping_rounds: int = 50
    seed: int = 7


@dataclass
class GbdtModel:
    """A fitted booster plus everything needed to reproduce its outputs."""

    booster: lgb.Booster
    temperature: float
    feature_names: list[str]
    config: GbdtConfig
    history: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        margin = self.booster.predict(np.asarray(x, dtype=float), raw_score=True)
        return 1.0 / (1.0 + np.exp(-np.asarray(margin, dtype=float) / self.temperature))


def train_gbdt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    config: GbdtConfig | None = None,
    sample_weights: np.ndarray | None = None,
) -> GbdtModel:
    """Fit a booster with early stopping on the validation fold, then calibrate."""
    cfg = config or GbdtConfig()
    weights = (
        np.asarray(sample_weights, dtype=float)
        if sample_weights is not None
        else np.ones(len(y_train), dtype=float)
    )
    if len(weights) != len(y_train):
        raise ValueError(f"sample_weights has {len(weights)} rows, y_train has {len(y_train)}")

    positives = float(np.sum(y_train))
    negatives = float(len(y_train) - positives)
    # The native training API rather than the sklearn wrapper: the wrapper's
    # eval_set argument is already deprecated, and this is the interface that
    # does not churn between LightGBM majors.
    params = {
        "objective": "binary",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "min_child_samples": cfg.min_child_samples,
        "bagging_fraction": cfg.subsample,
        "bagging_freq": cfg.subsample_freq,
        "feature_fraction": cfg.colsample_bytree,
        "lambda_l2": cfg.reg_lambda,
        # same class balancing as the MLP's pos_weight
        "scale_pos_weight": negatives / positives if positives > 0 else 1.0,
        "seed": cfg.seed,
        "bagging_seed": cfg.seed,
        "feature_fraction_seed": cfg.seed,
        "num_threads": 1,  # determinism over speed: a run must be reproducible
        "verbose": -1,
        "metric": "binary_logloss",
    }
    train_set = lgb.Dataset(
        np.asarray(x_train, dtype=float),
        label=np.asarray(y_train, dtype=float),
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        np.asarray(x_val, dtype=float),
        label=np.asarray(y_val, dtype=float),
        reference=train_set,
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(booster.best_iteration or cfg.n_estimators)

    val_margin = booster.predict(np.asarray(x_val, dtype=float), raw_score=True)
    temperature = _fit_temperature(
        torch.as_tensor(np.asarray(val_margin, dtype=float), dtype=torch.float32),
        torch.as_tensor(np.asarray(y_val, dtype=float), dtype=torch.float32),
    )

    raw = np.asarray(val_margin, dtype=float)
    diagnostics: dict[str, Any] = {
        "model_kind": "gbdt",
        # `best_epoch` under the name G0 already checks: a booster that stops at
        # its first tree learned nothing, exactly like an MLP whose validation
        # loss never improved.
        "best_epoch": best_iteration,
        "epochs_run": int(cfg.n_estimators),
        "stopped_early": best_iteration < cfg.n_estimators,
        "temperature": round(temperature, 4),
        "pred_std_pre_calibration": round(float(np.std(1.0 / (1.0 + np.exp(-raw)))), 4),
        "pred_std_post_calibration": round(
            float(np.std(1.0 / (1.0 + np.exp(-raw / temperature)))), 4
        ),
        "n_features_used": int(np.sum(booster.feature_importance(importance_type="gain") > 0)),
    }
    logger.info("GBDT trained", **{k: v for k, v in diagnostics.items() if k != "model_kind"})
    return GbdtModel(
        booster=booster,
        temperature=temperature,
        feature_names=list(feature_names),
        config=cfg,
        history={"best_iteration": float(best_iteration)},
        diagnostics=diagnostics,
    )


def feature_gain(model: GbdtModel) -> dict[str, float]:
    """Normalized split gain per feature — the one diagnostic trees give free.

    Not a causal statement and not a substitute for the per-feature IC table:
    gain says what the booster leaned on, which on noise is still something.
    """
    gains = model.booster.feature_importance(importance_type="gain")
    total = float(np.sum(gains)) or 1.0
    return {
        name: round(float(g) / total, 4)
        for name, g in zip(model.feature_names, gains, strict=False)
    }


__all__ = ["GbdtConfig", "GbdtModel", "feature_gain", "train_gbdt"]
