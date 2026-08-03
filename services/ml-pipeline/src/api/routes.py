"""ml-pipeline HTTP API — training, model registry, baselines and drift checks."""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from trading_common.constants import MAX_OHLCV_LIMIT
from trading_common.schemas import Interval

from src.api.deps import get_service
from src.core.data_contract import TrainingDataContractError
from src.core.service import MLPipelineService
from src.core.training import TrainingParams
from src.core.universe import UniverseParams

logger = structlog.get_logger()
router = APIRouter()


@contextmanager
def _mapped_errors(operation: str) -> Iterator[None]:
    """Turn a failed study/training run into an answer the CALLER can act on.

    Every route here spends minutes pulling a universe over HTTP and then
    computing, so the ways it fails are worth naming. Previously only
    RuntimeError and ValueError were mapped and everything else escaped, which
    Starlette answers as a plain-text 500 with an EMPTY body: a whole campaign
    of five studies plus training reported `HTTP 500: {}` six times, and the
    actual cause — market-data refusing a 5293-bar request — was only legible
    in the container log. The report is the artifact; it has to carry the
    reason. Same fix market-data's POST /fetch already got, for the same
    reason.
    """
    try:
        yield
    except HTTPException:
        raise  # already a deliberate answer — do not re-wrap as a 500
    except TrainingDataContractError as exc:
        # T0-1: refuse to train on data that cannot produce an interpretable
        # model, and return the full report so the caller sees WHY.
        # (A RuntimeError subclass — must be matched before it.)
        raise HTTPException(
            status_code=422,
            detail={"error": "training data contract violated", **exc.report},
        ) from exc
    except httpx.HTTPStatusError as exc:
        # An upstream service answered, and said no. Report WHICH service, WHAT
        # it answered and for WHAT url — 502 also says plainly that the caller's
        # request was not itself the problem.
        detail = f"upstream {exc.response.status_code} from {exc.request.url}"
        logger.error("Upstream rejected request", operation=operation, detail=detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except httpx.HTTPError as exc:  # timeout, connection refused, DNS...
        detail = f"upstream unreachable: {type(exc).__name__}: {exc}"
        logger.error("Upstream call failed", operation=operation, detail=detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except RuntimeError as exc:  # market-data client not configured
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:  # dataset too small for the requested design
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Operation failed", operation=operation)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


class TrainRequest(BaseModel):
    symbols: list[str] = Field(min_length=2)  # cross-sectional — needs a universe
    interval: str = "1d"
    # Bars per symbol to pull from market-data. The ceiling is the SHARED one:
    # accepting a limit market-data will not serve turns a legitimate 20-year
    # request into an upstream 422 halfway through the run.
    limit: int = Field(default=1500, ge=300, le=MAX_OHLCV_LIMIT)
    # P2-3: join the point-in-time fundamentals panel. OFF by default — a family
    # enters the model only after the per-feature IC table says it earns a place,
    # and `dataset.fundamental_coverage` says whether it was really there at all.
    fundamentals: bool = False
    # P3-1: restrict each session's cross-section to the point-in-time universe
    # (top-N by trailing dollar volume, rebalanced quarterly). None = every
    # symbol with bars, which on a ticker list assembled today means ranking
    # among companies chosen for having survived.
    universe_top_n: int | None = Field(default=None, ge=20, le=2000)
    # P4-1: "mlp" (registrable) or "gbdt" (CHALLENGER — evaluated through the
    # identical walk-forward and gate, never registered, since the store
    # persists an MLP state_dict).
    model_kind: str = Field(default="mlp", pattern="^(mlp|gbdt)$")
    # P4-2: seeds to average. >1 removes initialisation variance from the fold
    # spread and reports the members' disagreement.
    n_seeds: int = Field(default=1, ge=1, le=10)


class SectorStudyRequest(TrainRequest):
    # symbol -> free-form sector string; normalized to GICS on the way in.
    # A symbol absent here is "unknown sector", which is a real answer.
    sectors: dict[str, str | None] = Field(default_factory=dict)


class CpcvRequest(TrainRequest):
    n_groups: int = Field(default=6, ge=3, le=12)
    test_groups: int = Field(default=2, ge=1, le=4)


class AlphaDecayRequest(TrainRequest):
    # Holding horizons to profile, in sessions. The label's own horizon is
    # always measured too, since that is what selects which features to profile.
    horizons: list[int] = Field(default=[1, 5, 10, 21, 63], min_length=1, max_length=8)
    # Entry delays, in sessions — how stale a signal may be before it stops
    # being worth acting on.
    delays: list[int] = Field(default=[0, 1, 2, 3, 5], min_length=1, max_length=8)
    # Admit the classic-TA block (CANDIDATE_FEATURES) into the comparison. Off
    # by default so the routine study reports the model's actual inputs; turned
    # on, this is how stage E2's "a family joins after the IC table says so"
    # is actually executed. Measuring costs no trials — the study is model-free.
    include_candidates: bool = False


class ImportanceRequest(TrainRequest):
    # Permutations per feature. More repeats average out the permutation draw;
    # they do NOT buy statistical power, which comes from the number of
    # holdout sessions the paired difference is taken over.
    n_repeats: int = Field(default=5, ge=1, le=20)
    # Admit the classic-TA block, so the study can answer whether the model
    # would USE it — the conditional counterpart to alpha-decay's marginal IC.
    include_candidates: bool = False


class CostStudyRequest(TrainRequest):
    # The reference book size the per-name table is priced at. The capacity
    # curve spans several sizes regardless — a single size cannot answer
    # "how large can this get".
    aum_usd: float = Field(default=1_000_000.0, gt=0)
    # Measured turnover from a training run, if there is one: it converts bps
    # into annual drag, which is the only unit comparable to a Sharpe.
    turnover_daily: float | None = Field(default=None, ge=0, le=1)


class TuneRequest(TrainRequest):
    n_folds: int = Field(default=4, ge=2, le=10)


class BaselineRequest(BaseModel):
    reference_features: dict[str, list[float]]
    baseline_sharpe: float
    prediction_reference: list[float] = Field(default_factory=list)


class DriftCheckRequest(BaseModel):
    current_features: dict[str, list[float]]
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    rolling_accuracy_30d: float
    prediction_current: list[float] = Field(default_factory=list)


@router.get("/status")
async def status() -> dict:
    return {"service": "ml-pipeline", "status": "ready"}


@router.get("/models")
async def list_models(service: MLPipelineService = Depends(get_service)) -> dict:
    versions = service.model_store.versions() if service.model_store is not None else []
    return {"models": service.registry.model_ids(), "registry_versions": versions}


@router.get("/runs")
async def list_runs(service: MLPipelineService = Depends(get_service)) -> dict:
    """Which long operations have completed since this container started.

    A full-universe training pass runs for hours and its report exists only in
    the HTTP response, so a client read timeout used to destroy work that had
    actually finished — the server does not stop when the caller goes away.
    The index is deliberately thin; fetch one payload at a time.
    """
    return {"runs": service.runs()}


@router.get("/runs/{operation}")
async def get_run(operation: str, service: MLPipelineService = Depends(get_service)) -> dict:
    """The full report of the last completed run of `operation`.

    404 means it has not finished (or the container restarted) — NOT that it
    failed. Poll here after a timeout instead of paying for the run twice.
    """
    entry = service.last_run(operation)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no completed run recorded for {operation}")
    return entry


@router.post("/models/train")
async def train(req: TrainRequest, service: MLPipelineService = Depends(get_service)) -> dict:
    """Run the full training pass (plan §6–§7): dataset → purged walk-forward →
    gate report → MLflow version + drift baseline. Synchronous and potentially
    minutes-long — ops/scheduled use; promotion to production stays manual."""
    with _mapped_errors("train"):
        result = await service.train(
            req.symbols,
            Interval(req.interval),
            limit=req.limit,
            fundamentals=req.fundamentals,
            params=TrainingParams(model_kind=req.model_kind, n_seeds=req.n_seeds),
            universe_params=(
                UniverseParams(top_n=req.universe_top_n) if req.universe_top_n is not None else None
            ),
        )
        return service.record_run("train", result)


@router.post("/models/target-study")
async def target_study(
    req: TrainRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """Calibrate the barrier width and rank candidate targets (horizon,
    absolute vs excess) by how well RAW features already predict them. No model
    is fitted, so this costs nothing against the gate's n_trials."""
    with _mapped_errors("target-study"):
        result = await service.target_study(req.symbols, Interval(req.interval), limit=req.limit)
        return service.record_run("target-study", result)


@router.post("/models/sector-study")
async def sector_study(
    req: SectorStudyRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """Compare each raw feature's standalone IC with and without sector
    neutralization. Model-free — no model is fitted, so no trials are consumed.
    Read `composition.share_neutralized_against_peers` first: with too few names
    per sector the transform barely applies and the comparison means nothing."""
    with _mapped_errors("sector-study"):
        result = await service.sector_study(
            req.symbols, Interval(req.interval), req.sectors, limit=req.limit
        )
        return service.record_run("sector-study", result)


@router.post("/models/alpha-decay")
async def alpha_decay(
    req: AlphaDecayRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """How long the predictive content lasts and what a delayed entry costs.
    The holding period and the tranche count currently follow the label horizon
    by assumption; this measures whether that is right. Model-free — no fit, no
    trials consumed. Read the t-statistics, not the IC levels."""
    with _mapped_errors("alpha-decay"):
        result = await service.alpha_decay(
            req.symbols,
            Interval(req.interval),
            limit=req.limit,
            horizons=tuple(req.horizons),
            delays=tuple(req.delays),
            include_candidates=req.include_candidates,
        )
        return service.record_run("alpha-decay", result)


@router.post("/models/feature-importance")
async def feature_importance(
    req: ImportanceRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """Which inputs the model uses, measured by within-session permutation on
    the untouched holdout, with a planted noise column as the empirical floor.
    Read `groups` alongside `features`: permutation splits credit between
    correlated columns, so a family can matter while none of its members does.
    Diagnostic only — the fitted model carries a column serving cannot
    produce, so nothing here is registered or promotable."""
    with _mapped_errors("feature-importance"):
        result = await service.feature_importance(
            req.symbols,
            Interval(req.interval),
            limit=req.limit,
            n_repeats=req.n_repeats,
            include_candidates=req.include_candidates,
            fundamentals=req.fundamentals,
        )
        return service.record_run("feature-importance", result)


@router.post("/models/cost-study")
async def cost_study(
    req: CostStudyRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """Price the universe per name and across book sizes instead of at a flat
    5 bps. Read `capacity_curve`: impact grows with the square root of size
    while the edge does not grow at all, so a result that survives at $1M need
    not survive at $50M. Model-free — no fit, no trials consumed."""
    with _mapped_errors("cost-study"):
        result = await service.cost_study(
            req.symbols,
            Interval(req.interval),
            limit=req.limit,
            aum_usd=req.aum_usd,
            turnover_daily=req.turnover_daily,
        )
        return service.record_run("cost-study", result)


@router.post("/models/cpcv")
async def cpcv(req: CpcvRequest, service: MLPipelineService = Depends(get_service)) -> dict:
    """Combinatorial purged CV: several out-of-sample paths from the same data,
    reported as a dispersion rather than a single number. Expensive (C(N,k)
    fits) and diagnostic only — nothing is registered or promotable from it."""
    with _mapped_errors("cpcv"):
        result = await service.cpcv(
            req.symbols,
            Interval(req.interval),
            limit=req.limit,
            n_groups=req.n_groups,
            test_groups=req.test_groups,
            params=TrainingParams(model_kind=req.model_kind, n_seeds=req.n_seeds),
        )
        return service.record_run("cpcv", result)


@router.post("/models/capacity-probe")
async def capacity_probe(
    req: TrainRequest, service: MLPipelineService = Depends(get_service)
) -> dict:
    """Fit a deliberately over-parameterized model on real and on SHUFFLED
    labels and compare the train AUCs. Answers the question a flat train AUC
    cannot: optimization problem, or nothing in the data to learn? Diagnostic
    only — nothing is registered, nothing can be promoted from it."""
    with _mapped_errors("capacity-probe"):
        result = await service.capacity_probe(req.symbols, Interval(req.interval), limit=req.limit)
        return service.record_run("capacity-probe", result)


@router.post("/models/tune")
async def tune(req: TuneRequest, service: MLPipelineService = Depends(get_service)) -> dict:
    """Compare training configurations on the walk-forward folds (never the
    holdout) and report the winner plus the number of configurations tried —
    that count is what the gate's deflated Sharpe must be told."""
    with _mapped_errors("tune"):
        result = await service.tune(
            req.symbols, Interval(req.interval), limit=req.limit, n_folds=req.n_folds
        )
        return service.record_run("tune", result)


@router.get("/serving")
async def serving_status(service: MLPipelineService = Depends(get_service)) -> dict:
    engine = service.serving
    if engine is None:
        return {"active": False, "paused": False, "model_id": None}
    return {"active": engine.active, "paused": engine.paused, "model_id": engine.model_id}


@router.post("/serving/pause")
async def pause_serving(service: MLPipelineService = Depends(get_service)) -> dict:
    """Force HOLD-only serving (manual ops action — plan §9)."""
    if service.serving is None:
        raise HTTPException(status_code=503, detail="serving engine unavailable")
    service.serving.pause()
    return {"paused": True, "model_id": service.serving.model_id}


@router.post("/serving/resume")
async def resume_serving(service: MLPipelineService = Depends(get_service)) -> dict:
    if service.serving is None:
        raise HTTPException(status_code=503, detail="serving engine unavailable")
    service.serving.resume()
    return {"paused": False, "model_id": service.serving.model_id}


@router.post("/monitor/run")
async def run_monitor(service: MLPipelineService = Depends(get_service)) -> dict:
    """Run the daily monitor now (ops; the scheduler runs it daily anyway)."""
    with _mapped_errors("monitor"):
        return await service.run_daily_monitor()


@router.post("/models/versions/{version}/promote")
async def promote(version: str, service: MLPipelineService = Depends(get_service)) -> dict:
    """Point the ``production`` alias at a version (manual gate sign-off).

    Hot-reloads the serving engine, so the new model starts voting on the next
    ``features.ready`` without a restart."""
    try:
        return service.promote(version)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/models/{model_id}/baseline")
async def register_baseline(
    model_id: str,
    req: BaselineRequest,
    service: MLPipelineService = Depends(get_service),
) -> dict:
    """Register a model's training reference distributions + baseline Sharpe."""
    service.register_baseline(
        model_id,
        req.reference_features,
        req.baseline_sharpe,
        req.prediction_reference,
    )
    return {"model_id": model_id, "registered": True, "features": sorted(req.reference_features)}


@router.post("/models/{model_id}/drift")
async def check_drift(
    model_id: str,
    req: DriftCheckRequest,
    service: MLPipelineService = Depends(get_service),
) -> dict:
    """Run the drift check; publishes ModelDriftDetectedEvent when actionable."""
    with _mapped_errors("drift"):
        report = await service.check_drift(
            model_id,
            req.current_features,
            req.rolling_sharpe_30d,
            req.rolling_sharpe_90d,
            req.rolling_accuracy_30d,
            prediction_current=req.prediction_current,
        )
    if report is None:
        raise HTTPException(status_code=404, detail=f"no baseline registered for {model_id}")
    return {
        "model_id": report.model_id,
        "report_date": report.report_date.isoformat(),
        "feature_psi_scores": report.feature_psi_scores,
        "features_drifted": report.features_drifted,
        "sharpe_decay_pct": report.sharpe_decay_pct,
        "needs_retrain": report.needs_retrain,
        "needs_investigation": report.needs_investigation,
        "recommended_action": report.recommended_action,
    }
