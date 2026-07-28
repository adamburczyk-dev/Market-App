"""Does neutralizing the sector raise the evidence? (E2 gate, P2-2)

The stage-2 rule is that a feature transform enters the model only if it makes
the ranking measurably better — never because the literature likes it. This
answers that for sector neutralization, and answers it MODEL-FREE: both
datasets are built through the real `build_dataset`, and each raw feature's
standalone IC (and its t-statistic) is measured against the same forward
returns. Nothing is fitted, so this costs nothing against the gate's
`n_trials`.

The comparison is only meaningful if the neutralization actually ran, which at
34 names over 11 sectors it largely does not: ~3 names per sector is below the
size at which a peer median means anything, so those names fall into the
residual group and are demeaned against a mixed bag. The report therefore leads
with the sector composition. "No improvement" and "barely applied" look
identical in the numbers and are completely different conclusions.

**Read the direction carefully — it is not "higher is better".** Building the
test fixture for this made the point concretely: on a universe where sectors
drift apart, the global mean |t| was 2.12 with 9 features clearing |t| >= 2,
and after neutralization it was 0.74 with NONE clearing. Nothing broke. The
evidence was the sector, and demeaning removed it, which is the job. So a
collapse here is a finding about the DATA — the ranking was largely a sector
ranking — not a verdict against the transform. The question the study actually
answers is how much evidence survives once the sector bet is taken away, since
that residue is the only part a cross-sectional stock-selection model can claim
as its own (and the only part a benchmark-relative book gets paid for, per D3).
"""

from dataclasses import asdict, dataclass
from typing import Any

import structlog
from trading_common.ranking import MIN_SECTOR_SIZE
from trading_common.schemas import OHLCVBar
from trading_common.sectors import normalize_sector

from src.core.dataset import Dataset, DatasetParams, build_dataset
from src.core.evaluation import per_feature_ic

logger = structlog.get_logger()


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    ic_plain: float
    t_plain: float
    ic_neutral: float
    t_neutral: float

    @property
    def t_gain(self) -> float:
        """Change in evidence, measured on |t| — a feature that ranks backwards
        more consistently has more evidence, not less, and the sign of the IC is
        the model's problem, not the transform's."""
        return abs(self.t_neutral) - abs(self.t_plain)

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "t_gain": round(self.t_gain, 2)}


def sector_composition(
    symbols: list[str],
    sector_by_symbol: dict[str, str | None],
    min_sector_size: int = MIN_SECTOR_SIZE,
) -> dict[str, Any]:
    """How the universe actually splits — the precondition for reading the rest."""
    counts: dict[str, int] = {}
    unknown: list[str] = []
    for symbol in symbols:
        canonical = normalize_sector(sector_by_symbol.get(symbol))
        if canonical is None:
            unknown.append(symbol)
            continue
        counts[canonical] = counts.get(canonical, 0) + 1
    usable = {s: n for s, n in counts.items() if n >= min_sector_size}
    pooled = sum(n for s, n in counts.items() if n < min_sector_size) + len(unknown)
    return {
        "symbols": len(symbols),
        "sectors": dict(sorted(counts.items())),
        "unknown_sector": sorted(unknown),
        "min_sector_size": min_sector_size,
        "sectors_large_enough": sorted(usable),
        "names_in_residual_group": pooled,
        # The one number that decides whether the comparison below means anything.
        "share_neutralized_against_peers": (
            round(sum(usable.values()) / len(symbols), 3) if symbols else 0.0
        ),
    }


def _ic_table(ds: Dataset) -> dict[str, tuple[float, float]]:
    table = per_feature_ic(ds.dates, ds.x, ds.feature_names, ds.next_returns)
    return {name: (f.mean, f.tstat) for name, f in table.items()}


def run_sector_study(
    bars_by_symbol: dict[str, list[OHLCVBar]],
    sector_by_symbol: dict[str, str | None],
    params: DatasetParams | None = None,
) -> dict[str, Any]:
    """Build the dataset both ways and compare per-feature evidence."""
    p = params or DatasetParams()
    plain = build_dataset(bars_by_symbol, p)
    neutral = build_dataset(bars_by_symbol, p, sector_by_symbol=sector_by_symbol)

    ic_plain = _ic_table(plain)
    ic_neutral = _ic_table(neutral)
    comparisons = [
        FeatureComparison(
            feature=name,
            ic_plain=round(ic_plain[name][0], 5),
            t_plain=round(ic_plain[name][1], 2),
            ic_neutral=round(ic_neutral[name][0], 5),
            t_neutral=round(ic_neutral[name][1], 2),
        )
        for name in sorted(set(ic_plain) & set(ic_neutral))
        if not name.startswith("macro_")
    ]
    # Ordered by the evidence that SURVIVES, not by the gain: what remains after
    # the sector bet is removed is the actionable column.
    ranked = sorted(comparisons, key=lambda c: abs(c.t_neutral), reverse=True)
    improved = [c for c in ranked if c.t_gain > 0]
    composition = sector_composition(sorted(bars_by_symbol), sector_by_symbol)

    mean_t_plain = _mean_abs([c.t_plain for c in comparisons])
    mean_t_neutral = _mean_abs([c.t_neutral for c in comparisons])
    logger.info(
        "Sector study",
        features=len(comparisons),
        improved=len(improved),
        mean_t_plain=mean_t_plain,
        mean_t_neutral=mean_t_neutral,
    )
    return {
        "composition": composition,
        "rows_plain": plain.n_samples,
        "rows_neutral": neutral.n_samples,
        "features": [c.as_dict() for c in ranked],
        "summary": {
            "n_features": len(comparisons),
            "n_improved": len(improved),
            "mean_abs_t_plain": mean_t_plain,
            "mean_abs_t_neutral": mean_t_neutral,
            "strong_plain": sum(1 for c in comparisons if abs(c.t_plain) >= 2.0),
            "strong_neutral": sum(1 for c in comparisons if abs(c.t_neutral) >= 2.0),
        },
        "note": (
            "Model-free: standalone IC of each raw feature against the next-session "
            "return, measured on the real build_dataset output. No model was fitted, "
            "so no trials were consumed. Read share_neutralized_against_peers FIRST — "
            "if it is low, most names were demeaned against a mixed residual group "
            "and this comparison says nothing about sector neutralization. Then read "
            "strong_neutral, NOT the change in mean |t|: a drop means the evidence "
            "was the sector, which is what demeaning is supposed to remove."
        ),
    }


def _mean_abs(values: list[float]) -> float:
    return round(sum(abs(v) for v in values) / len(values), 3) if values else 0.0


__all__ = ["FeatureComparison", "run_sector_study", "sector_composition"]
