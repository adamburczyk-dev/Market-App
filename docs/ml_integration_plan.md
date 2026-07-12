# ML Integration Plan (serwisy 10–13 → ml-pipeline)

> Authoritative design for the ML phase, written **before** the deep ML work per the standing
> rule in CLAUDE.md. Everything here is chosen for the data this system actually has — daily
> OHLCV for a small/medium US-equity universe, annual fundamentals, macro indicators — not for
> the data a hedge fund wishes it had. When a section says *deferred*, that is a deliberate,
> reviewable decision, not an omission.

## 1. Guiding principles

1. **Cross-sectional, not per-symbol time-series.** With daily bars and a universe of tens of
   symbols, per-symbol price prediction is statistically hopeless (a few thousand noisy samples
   per symbol). The tractable question is *relative*: "which names in the universe are likely to
   outperform over the next ~2 weeks?" — pooling all symbols into one training set multiplies the
   sample count by the universe size and matches how the features are already served
   (cross-sectional percentile ranks, López de Prado). The empirical asset-pricing literature
   (Gu–Kelly–Xiu 2020) finds even huge datasets yield OOS R² well under 1% — the edge, if any,
   is relative ordering, and every design choice below follows from that.
2. **Shallow model, deep validation.** At this data scale the model class barely matters; leakage
   control, labeling, and cost-adjusted out-of-sample evaluation matter enormously. Budget the
   engineering effort accordingly: the validation harness is the product, the model is a plug-in.
3. **ML is a *vote*, not an autopilot.** The signal-aggregator was built for exactly this: the ML
   signal enters as one weighted component next to the rules strategy and the macro bias, behind
   the shared cost gate, RiskEnvelope, sector/exposure caps, and the circuit breaker. The
   AdaptiveWeightOptimizer is the safety net — a model that stops working loses weight
   automatically. v1 ML signals carry **no protective levels**, so an ML-only aggregate cannot
   become an order (risk-mgmt blocks orders without a stop): ML can strengthen or veto
   strategy-led trades, never trade alone. Standalone ML-led orders are a v2 decision.
4. **Kill criteria are part of the design.** A model ships with the conditions under which it is
   demoted, and the plan defines them up front (§8).

## 2. Data & universe

| Source | Data | Cadence | Point-in-time discipline |
|---|---|---|---|
| market-data | daily OHLCV | daily | features at *t* use bars ≤ *t*; entry assumed next bar |
| fundamental-data | annual statements + 9-signal Piotroski | yearly (weekly refresh) | joined with a **90-day publication lag** after `period_end` (use `filed_at` when present) — annual reports are not knowable on their period-end date |
| company-classifier | style / cap tier / sector | on classify | static-slow; no lag needed |
| macro-data | regime (expansion…crisis) | 6h refresh | regime known at *t* uses indicators ≤ *t* |

- **Universe:** config-driven ticker list. Start with ~20–50 liquid US large caps (the 8-symbol
  bootstrap list is too small for meaningful ranks; 20 is the floor at which cross-sectional
  percentiles stop being coin flips). History target: ≥ 5 years of daily bars per symbol
  (≈ 1 250 sessions × universe ≈ 25k–60k training rows — adequate for a shallow model).
- **Bootstrap:** a one-shot market-data backfill (yfinance) into TimescaleDB for the configured
  universe precedes the first training run.

## 3. Features

The model consumes exactly what feature-engine already serves — **cross-sectional ranked
vectors** (Tier-1 technicals + Tier-2 attributes), plus a macro one-hot appended at dataset-build
time:

- Tier-1 (ranked): momentum_5/20/60, RSI(14), realized_vol_20, volume z-score, distance from
  52w high, short-term reversal (r_5), etc. — whatever `compute_feature_vector` emits, ranked.
- Tier-2 (ranked where cross-sectional): `f_score`, `fund_net_margin`, `fund_roa`,
  `fund_leverage`; `style_growth`/`style_value` pass through unranked (already in [0,1]).
- Macro context (global, unranked): one-hot of the 5 regimes at feature date (from macro-data
  history; "unknown" allowed).

**Refactor required (ML-0):** the pure feature and ranking functions move from
`services/feature-engine/src/core/{features,ranking}.py` to **`trading_common.features`** /
**`trading_common.ranking`**. Training must reproduce historical features bit-for-bit with the
serving path; duplicating the math across a service boundary guarantees drift, and services must
not import each other. feature-engine keeps orchestration, store, and API; the *definitions*
become shared contracts (same precedent as `cost_filter` / `RiskEnvelope`).

## 4. Labels — Triple Barrier (mandated)

Fixed-horizon returns mislabel volatile names and ignore path (a +1% 10-day return that drew down
−8% first is not a "win"). Triple-barrier labels (López de Prado):

- Volatility target: `σ_t` = std of daily log returns over the trailing 20 sessions.
- Reference price: `close_t` (signal is computed after the close, consistent with the backtest
  engine's next-bar accounting).
- Upper barrier: `close_t · (1 + 2.0·σ_t·√10)`, lower: `close_t · (1 − 2.0·σ_t·√10)`,
  vertical barrier: **h = 10 trading days**.
- Outcome: `1` if the upper barrier is touched first, `0` if the lower is touched first; on a
  vertical hit, the sign of the net return decides. Binary target = **P(up-barrier-first)**.
- Overlap: consecutive daily samples share label windows — handled by purging (§6), not by
  discarding samples (small data).

**Meta-labeling is the designated v2**, not v1: train a second model on "given the momentum rule
fired, did it pay?" — it is lower-variance and fits the aggregator naturally, but it needs a
history of rule signals first (generate from the backtest engine). v1 is the direct classifier
so the "ml" source is symmetric with "strategy" from day one.

## 5. Model — one global stack first

- **v1 model class:** small PyTorch MLP (input ≈ 20 ranked features → 32 → 16 → 1, ReLU,
  dropout 0.3, weight decay, early stopping on the purged validation fold, class-balanced BCE).
  PyTorch per the project rule; at this scale a gradient-boosted tree would perform comparably —
  if a later bake-off justifies it, a GBDT stack can be added as a *second* registered stack, but
  the plumbing (labels/splits/registry/serving) is identical and is what this phase builds.
- **Calibration:** temperature scaling on the validation fold; the aggregator receives
  `confidence = 2·|P(up) − 0.5|` and direction `BUY` if P(up) ≥ 0.55, `SELL` if ≤ 0.45, else
  `HOLD` (dead zone keeps marginal predictions out of the vote).
- **One global model** (`global_v1`) trained on the pooled universe. The company-classifier's
  model-stack routing stays in place, but **all stacks map to `global_v1` until per-style models
  demonstrably beat it** — with tens of symbols, splitting the training set by style/cap-tier is
  data starvation dressed up as sophistication. Revisit when the universe is ≥ 200 names.

## 6. Validation protocol (the actual product)

- **Split:** walk-forward only, never random (project rule). Rolling train window 756 sessions
  (~3y) → validation 63 sessions (~3m) → step forward; final holdout = the most recent 126
  sessions never touched during model selection.
- **Purging + embargo:** drop training samples whose label window overlaps the validation window
  (purge = h = 10 sessions) plus a 5-session embargo after it (López de Prado) — triple-barrier
  labels leak across naive split boundaries.
- **Metrics:** AUC and Brier (calibration) are diagnostics; the **decision metric is
  cost-adjusted OOS Sharpe** of a daily-rebalanced, equal-weight, long-only top-quintile
  portfolio built from OOS predictions, with the same 5 bps per-turn cost as the backtest engine.
  Baselines it must beat: the momentum rule itself and equal-weight buy-and-hold of the universe.
- **Activation gate (non-negotiable, mirrors the strategy rule):** OOS Sharpe > 0.5 on the
  holdout **and** on ≥ 2 of the 3 most recent walk-forward folds; calibration sane (Brier no
  worse than the base rate); only then may the serving path publish non-HOLD signals.

## 7. Registry & lifecycle — MLflow

- **MLflow with a local file/sqlite backend** inside the ml-pipeline container volume (tracking
  URI `file:./mlruns` equivalent). No dedicated MLflow server container in v1 — the registry API
  is what matters (log params/metrics/artifacts, stage transitions Production/Staging/Archived);
  a standalone MLflow service + artifact store is the documented scale-up path.
- The in-memory `ModelRegistry` becomes an MLflow-backed implementation with the same interface;
  drift baselines (feature distributions, baseline Sharpe/accuracy) are logged as artifacts next
  to the weights, so `restore()` at startup reloads both model and monitoring baseline.
- Artifact = TorchScript module + `metadata.json` (feature names/order, rank transform params,
  label params σ/h/multipliers, training window, OOS metrics, calibration temperature). The
  metadata is load-bearing: serving refuses to run if the served feature vector's keys don't
  match the artifact's feature list.
- **Retrain cadence:** weekly (scheduled via the existing `PeriodicTask`), plus on-demand
  `POST /models/train`. A retrained model lands in *Staging*; promotion to *Production* is manual
  in v1 (human reads the gate report), automatable later.

## 8. Serving & integration (activates R11)

- **Trigger:** ml-pipeline subscribes `features.ready` (durable `ml-pipeline-features`), pulls the
  symbol's **ranked** vector from feature-engine over HTTP (same pattern as strategy), runs the
  Production model of the routed stack, and publishes — contracts-first — a new event:

  ```
  MlSignalGeneratedEvent  (EventType.ML_SIGNAL_GENERATED = "ml.signal_generated", ML stream)
    symbol, model_id, model_stack,
    signal: BUY|SELL|HOLD, confidence: [0,1], probability_up: [0,1],
    horizon_days: int, source_service="ml-pipeline"
  ```

- **Aggregator:** third durable subscription (`signal-aggregator-ml`) buffers the latest ML
  component per symbol with the same emit-timestamp TTL as strategy signals → re-aggregates the
  symbol. The pre-provisioned "ml" source in `SIGNAL_SOURCES` goes live; renormalized adaptive
  weights and the shared cost gate apply unchanged. ML components carry **no SL/TP** (§1.3).
- **Risk path:** unchanged — sector caps, exposure caps, breaker, long-only execution, protective
  exits all sit downstream exactly as for strategy-led signals.

## 9. Monitoring (mandated: daily drift + weekly report)

- Serving keeps a rolling window (e.g. last 500 served vectors + predictions per model). The
  **deferred daily drift schedule now lands**: a `PeriodicTask` runs `check_drift` daily —
  per-feature PSI vs the training baseline, KS on the prediction distribution — and the existing
  `ModelDriftDetectedEvent` → notification path alerts (warning=investigate, critical=retrain).
- **Delayed-label accuracy:** triple-barrier outcomes resolve h≈10 sessions later; a daily job
  resolves matured predictions against market-data history and feeds rolling accuracy/Sharpe into
  the decay detector — this closes the `record_outcome` loop so the aggregator's adaptive weights
  learn from *realized* ML performance, not hopes.
- **Pause semantics:** `POST /models/{id}/pause` forces HOLD-only serving (v1 manual, mirroring
  the strategy probation philosophy); auto-pause on critical drift is v2.

## 10. Implementation roadmap (each increment: tests + lint/mypy + live verify + CLAUDE.md)

| # | Increment | Contents |
|---|---|---|
| ML-0 | Shared feature definitions + dataset builder | move pure `features`/`ranking` to trading-common (feature-engine re-exports); ml-pipeline `core/dataset.py`: OHLCV history → ranked feature matrix + triple-barrier labels (+ purge/embargo split helpers); synthetic-data tests |
| ML-1 | Training + registry | PyTorch MLP, purged walk-forward, calibration, gate report; MLflow local registry replaces in-memory; `POST /models/train`; torch+mlflow deps (heavier image — ml-pipeline only) |
| ML-2 | Serving + aggregation | contracts-first `MlSignalGeneratedEvent`; `features.ready` → infer → publish; aggregator ml subscription + TTL; live-verify full chain (features → ml vote → aggregate shifts) |
| ML-3 | Monitoring | rolling inference window; daily drift `PeriodicTask`; delayed-label resolution → `record_outcome` + decay detector; pause route |
| ML-4 (v2) | Extensions | meta-labeling bake-off, per-style stacks (gated on universe ≥ 200), ML-derived protective levels, auto-pause, GBDT challenger stack |

## 11. Explicitly deferred / rejected

- **Deep sequence models (LSTM/Transformer) on daily bars:** rejected for v1 — sample counts are
  2–3 orders of magnitude short of where these stop overfitting; revisit only with intraday data.
- **Per-symbol models:** rejected (see §1.1).
- **Reinforcement learning for execution/sizing:** rejected — sizing is a solved, risk-governed
  path here (drawdown-adaptive + regime caps); RL adds opacity exactly where auditability matters.
- **Sentiment (serwis "sentiment-data") features:** contracts exist; deferred until a reliable
  source is wired — no fabricated placeholder features.
- **Automatic Production promotion:** deferred until the gate report has been exercised manually
  a few times.
