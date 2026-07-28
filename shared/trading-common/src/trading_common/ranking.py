"""Cross-sectional percentile ranking of features across the symbol universe.

López de Prado: use cross-sectional percentile ranks, not raw values. For each
feature, a symbol's value is ranked against all other symbols in the same
snapshot; the result is a percentile in [0, 1] (0 = lowest in the universe,
1 = highest). This is the transform strategy/ML should consume, not raw values.

Shared (trading-common) for the same reason as ``trading_common.features``:
ml-pipeline's training must apply the exact rank transform the serving path
applies (docs/ml_integration_plan.md §3).
"""

import statistics

from trading_common.schemas import FeatureVector
from trading_common.sectors import normalize_sector

# A peer group smaller than this cannot supply a usable median: with two names
# the "median" is their midpoint and demeaning turns both into ±the same number;
# with one it is the name itself, which zeroes every feature it has. Undersized
# sectors are pooled into one residual group instead of being left on a
# different scale from the rest.
MIN_SECTOR_SIZE = 4
OTHER_SECTOR = "__other__"


def _percentile_ranks(values: list[float]) -> list[float]:
    """Average-rank percentile in [0, 1]; ties share the mean rank."""
    n = len(values)
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0  # 0-based mean rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank / (n - 1)
        i = j + 1
    return ranks


def cross_sectional_rank(vectors: list[FeatureVector]) -> list[FeatureVector]:
    """Rank-transform a universe of FeatureVectors (one interval/time snapshot).

    Each feature value is replaced by its cross-sectional percentile rank across
    the symbols that have that feature. Returns new vectors (rank_transformed=True);
    inputs are left unchanged. A single-symbol universe yields neutral ranks (0.5).
    """
    if not vectors:
        return []

    keys = {key for v in vectors for key in v.features}
    ranked_features: list[dict[str, float]] = [{} for _ in vectors]
    for key in keys:
        idx = [i for i, v in enumerate(vectors) if key in v.features]
        values = [vectors[i].features[key] for i in idx]
        for i, rank in zip(idx, _percentile_ranks(values), strict=True):
            ranked_features[i][key] = rank

    return [
        v.model_copy(update={"features": feats, "rank_transformed": True})
        for v, feats in zip(vectors, ranked_features, strict=True)
    ]


def sector_neutralize(
    vectors: list[FeatureVector],
    sector_by_symbol: dict[str, str | None],
    min_sector_size: int = MIN_SECTOR_SIZE,
) -> list[FeatureVector]:
    """Subtract each feature's sector median, so ranks compare peers to peers.

    With 11 sectors in the universe, a global rank is substantially a ranking of
    sectors: the model can score well by holding whichever sector happened to
    run, which is one undiversified macro bet wearing a factor's clothes.
    Neutralizing removes the sector's common component and leaves the part of
    each feature that distinguishes a name from its own peers.

    **Demeaning, not ranking within sector.** Ranking inside a sector is the
    textbook construction and it is the wrong one at this universe size: 34
    names over 11 sectors is ~3 per sector, and a percentile rank over 3 values
    is the set {0, 0.5, 1} — the same degeneracy `min_universe` exists to keep
    out of the training set. Demeaning the raw value and ranking GLOBALLY keeps
    the full cross-section's resolution while still removing the sector level.
    Sectors below ``min_sector_size`` are pooled into one residual group rather
    than left un-demeaned, so every name stays on the same "excess over peers"
    scale.

    Symbols with an unrecognized or missing sector join that residual group —
    an honest "unknown peer set", never a guess. Returns new vectors; the
    caller still applies ``cross_sectional_rank``.
    """
    if not vectors:
        return []

    groups: dict[str, list[int]] = {}
    for i, v in enumerate(vectors):
        canonical = normalize_sector(sector_by_symbol.get(v.symbol))
        groups.setdefault(canonical or OTHER_SECTOR, []).append(i)

    # Undersized sectors merge into the residual group (which may itself stay
    # small — then it is left alone below, since demeaning it would be noise).
    residual = groups.pop(OTHER_SECTOR, [])
    for sector in [s for s, members in groups.items() if len(members) < min_sector_size]:
        residual.extend(groups.pop(sector))
    if residual:
        groups[OTHER_SECTOR] = residual

    out: list[dict[str, float]] = [dict(v.features) for v in vectors]
    for members in groups.values():
        if len(members) < min_sector_size:
            continue  # too few peers to define a centre — leave the values as they are
        keys = {key for i in members for key in vectors[i].features}
        for key in keys:
            present = [i for i in members if key in vectors[i].features]
            if len(present) < min_sector_size:
                continue
            centre = statistics.median(vectors[i].features[key] for i in present)
            for i in present:
                out[i][key] = vectors[i].features[key] - centre

    return [v.model_copy(update={"features": feats}) for v, feats in zip(vectors, out, strict=True)]
