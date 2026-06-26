"""Cross-sectional percentile ranking of features across the symbol universe.

López de Prado: use cross-sectional percentile ranks, not raw values. For each
feature, a symbol's value is ranked against all other symbols in the same
snapshot; the result is a percentile in [0, 1] (0 = lowest in the universe,
1 = highest). This is the transform strategy/ML should consume, not raw values.
"""

from trading_common.schemas import FeatureVector


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
