"""Feature importance — what the model USES, not what merely correlates.

`Plan_Rozwoju` Faza 3 asks for a feature-importance report and the project has
never had one. The gap it fills is specific: `per_feature_ic` (the E2
instrument) measures each feature's evidence **on its own**, which answers "does
this column predict" and cannot answer "does the model need it, given the other
fourteen". Those come apart constantly — a feature with a strong standalone IC
can be redundant, and a feature with none can still carry the interaction the
model actually trades.

Four decisions make the difference between a number and a picture of one.

**Measured out of sample, on the holdout.** Permuting a column of the window
the model was FITTED on reports what it memorised. The holdout — the most
recent `holdout_size` sessions, untouched during selection — is the only window
where the drop means "this is how much predictive power the feature carries".

**Permuted WITHIN each session.** This is the capacity probe's lesson applied to
a different question (see core/capacity.py). A global shuffle of one column
moves values across dates, so it destroys two things at once: which NAME held
which value, and each session's marginal distribution of the feature — and the
features are persistent, so that marginal drifts with the regime. The score
here is a per-session cross-sectional IC, so a distributional change alone
would register as importance even for a column the model ignores. Permuting
inside a session keeps the day's distribution exactly and gives up only the
pairing, which is the entire question.

**Scored as an IC drop, paired session by session.** The IC is the quantity the
gate's rank conditions are built on, so importance is denominated in the same
unit as the decision it informs. The pairing matters more than the unit: an
unpaired comparison of two means over 126 sessions is buried in the IC's own
session-to-session variance, whereas the per-session difference removes exactly
that variance because both terms saw the same day. The standard error of the
paired difference is what turns a drop into evidence, and a bar corrected for
the number of features tested is what stops the largest of fifteen noise draws
from being read as the model's favourite input.

**Reported for FAMILIES as well as columns.** Permutation importance splits
credit between correlated features: two near-duplicates each look unimportant,
because the model recovers through the twin that was left alone. That is not a
flaw to apologise for, it is a reason to also permute the whole family at once
— and families are the unit this project makes decisions in ("a family enters
when the IC table says so"). Each entry therefore also carries its strongest
correlation with a feature OUTSIDE its own group, so a zero next to a 0.95
twin reads as redundancy rather than as irrelevance.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import structlog

from src.core.ensemble import Predictor
from src.core.evaluation import (
    auc,
    average_ranks,
    normal_ppf,
    session_groups,
    session_ic_series,
)

logger = structlog.get_logger()

# Below this the paired t-statistic has no sampling distribution worth reading.
MIN_SESSIONS = 20
# Family-wise error rate the reported bar controls.
FAMILY_ALPHA = 0.05
# A correlation at or above this makes "unimportant" ambiguous with "redundant".
REDUNDANCY_THRESHOLD = 0.9
# Below this an IC "drop" is arithmetic dust, not a change in the ranking.
#
# Load-bearing, and found by a test rather than by reading the code: a model
# that ignores a column exactly (a tree that never splits on it, a weight of
# zero) produces identical predictions before and after the permutation, and
# the repeats are summed and divided — which is not the identity in binary
# floating point. The measured drop came out at 1.7e-17 with a standard error
# of 5.2e-18, so a column the model provably never reads scored t = +3.23 and
# was announced as clearing the significance bar. Dust over dust is a ratio,
# not evidence.
IC_DROP_EPSILON = 1e-9
# Name of the synthetic column that measures what noise scores here.
NOISE_FEATURE = "noise_control"

# Column-name prefixes that identify a family. The remainder are grouped by the
# concept they measure — a grouping is a claim about what the columns share, so
# it is written down rather than inferred from a string.
_PREFIX_GROUPS = {"macro_": "macro_regime", "fund_": "fundamentals"}
_CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "momentum": ("return_1d", "return_5d", "return_20d", "momentum_6_1", "momentum_12_1"),
    "volatility": ("realized_vol_20", "downside_vol_20", "skew_60", "max_ret_1m", "atr_pct_14"),
    "liquidity": ("dollar_volume_20", "amihud_20", "volume_ratio", "obv_slope_20", "ad_slope_20"),
    "oscillators": (
        "rsi_14",
        "stoch_k_14",
        "stoch_d_14",
        "cci_20",
        "mfi_14",
        "bb_pct_b",
        "keltner_pos_20",
        "donchian_pos_20",
    ),
    "trend": (
        "price_to_sma50",
        "dist_52w_high",
        "macd",
        "macd_signal",
        "macd_hist",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "aroon_up_25",
        "aroon_down_25",
        "aroon_osc_25",
        "vwap_ratio_20",
    ),
}


def feature_groups(feature_names: list[str]) -> dict[str, tuple[str, ...]]:
    """Families present in this feature set, as {group: members}.

    Only families with at least two members are returned: permuting a
    one-column "family" reproduces that column's own entry exactly, and two
    identical rows in one table invite the reader to treat them as two pieces
    of evidence.
    """
    present = set(feature_names)
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        for prefix, group in _PREFIX_GROUPS.items():
            if name.startswith(prefix):
                groups.setdefault(group, []).append(name)
    for group, members in _CONCEPT_GROUPS.items():
        found = [m for m in members if m in present]
        if found:
            groups.setdefault(group, []).extend(found)
    return {
        group: tuple(sorted(set(members))) for group, members in groups.items() if len(members) > 1
    }


@dataclass(frozen=True)
class ImportanceEntry:
    """One column's — or one family's — contribution to the model's ranking."""

    name: str
    members: tuple[str, ...]
    ic_drop: float  # base IC − permuted IC; positive means the model relies on it
    ic_drop_se: float  # standard error of the PAIRED per-session difference
    tstat: float
    auc_drop: float
    max_abs_correlation: float  # with the most similar feature outside this entry
    most_correlated_with: str | None

    @property
    def redundant(self) -> bool:
        """A near-duplicate exists, so a small drop does not mean "carries nothing"."""
        return self.max_abs_correlation >= REDUNDANCY_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.name,
            "members": list(self.members),
            "ic_drop": round(self.ic_drop, 5),
            "ic_drop_se": round(self.ic_drop_se, 5),
            "t": round(self.tstat, 2),
            "auc_drop": round(self.auc_drop, 4),
            "max_abs_correlation": round(self.max_abs_correlation, 3),
            "most_correlated_with": self.most_correlated_with,
            "redundant": self.redundant,
        }


@dataclass(frozen=True)
class ImportanceReport:
    n_rows: int
    n_sessions: int
    base_ic: float
    base_auc: float
    n_repeats: int
    tstat_bar: float  # Šidák-corrected two-sided bar for the number of tests run
    features: tuple[ImportanceEntry, ...]
    groups: tuple[ImportanceEntry, ...]
    noise_control: ImportanceEntry | None
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_sessions": self.n_sessions,
            "base_ic": round(self.base_ic, 5),
            "base_auc": round(self.base_auc, 4),
            "n_repeats": self.n_repeats,
            "tstat_bar": round(self.tstat_bar, 2),
            "features": [e.as_dict() for e in self.features],
            "groups": [e.as_dict() for e in self.groups],
            "noise_control": self.noise_control.as_dict() if self.noise_control else None,
            "verdict": self.verdict,
        }


def sidak_tstat_bar(n_tests: int, alpha: float = FAMILY_ALPHA) -> float:
    """Two-sided |t| that keeps the family-wise error at `alpha` over n tests.

    Fifteen independent noise draws produce a largest |t| near 2.6 by
    construction; reading that as "the model's most important feature" is the
    obvious way to turn this report into a random-number generator. Šidák
    rather than Bonferroni because the correction is exact under independence
    and the tests here are close to it: the paired differences are taken over
    sessions whose forward returns span one session and therefore do not
    overlap.
    """
    n = max(1, int(n_tests))
    per_test = 1.0 - (1.0 - alpha) ** (1.0 / n)
    return float(normal_ppf(1.0 - per_test / 2.0))


def _permute_block(
    x: np.ndarray,
    columns: list[int],
    groups: list[list[int]],
    rng: np.random.Generator,
) -> None:
    """Permute `columns` within each session, IN PLACE, with a shared ordering.

    One permutation per session applied to the whole block, not an independent
    one per member: for a family the question is whether the family as a unit
    tells the model which name to buy, and permuting members independently
    would additionally destroy the relationships BETWEEN them — answering a
    harder question than the one asked, and inflating the family's importance
    for a reason that has nothing to do with prediction.
    """
    for rows in groups:
        order = rng.permutation(len(rows))
        source = [rows[i] for i in order]
        x[np.ix_(rows, columns)] = x[np.ix_(source, columns)]


def _rank_correlations(x: np.ndarray) -> np.ndarray:
    """|Spearman| between every pair of columns, computed once.

    Once, because the per-entry loop would otherwise re-rank the same 50k-row
    columns O(features²) times for a diagnostic. A constant column correlates
    with nothing — numpy answers NaN there, which is read as 0.0 rather than
    propagated into a comparison.
    """
    ranks = np.column_stack([average_ranks(x[:, j]) for j in range(x.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(ranks, rowvar=False)
    return np.abs(np.nan_to_num(np.atleast_2d(matrix), nan=0.0))


def _correlation_neighbour(
    correlations: np.ndarray, feature_names: list[str], columns: list[int]
) -> tuple[float, str | None]:
    """Strongest |rank correlation| between this entry's columns and any other.

    Measured against features OUTSIDE the entry: inside a family the members
    are expected to be correlated, and that is the reason the family row exists
    rather than a caveat on it.
    """
    inside = set(columns)
    outside = [j for j in range(correlations.shape[1]) if j not in inside]
    best, partner = 0.0, None
    for i in columns:
        for j in outside:
            value = float(correlations[i, j])
            if value > best:
                best, partner = value, feature_names[j]
    return best, partner


def _entry(
    name: str,
    members: tuple[str, ...],
    columns: list[int],
    x: np.ndarray,
    y: np.ndarray,
    next_returns: np.ndarray,
    groups: list[list[int]],
    base_ic: np.ndarray,
    base_auc: float,
    model: Predictor,
    feature_names: list[str],
    correlations: np.ndarray,
    n_repeats: int,
    rng: np.random.Generator,
) -> ImportanceEntry:
    scratch = x.copy()
    original = x[:, columns].copy()
    permuted_ic = np.zeros_like(base_ic)
    permuted_auc = 0.0
    for _ in range(n_repeats):
        scratch[:, columns] = original  # start each repeat from the real column
        _permute_block(scratch, columns, groups, rng)
        probs = model.predict_proba(scratch)
        permuted_ic += session_ic_series(probs, next_returns, groups)
        permuted_auc += auc(y, probs)
    permuted_ic /= n_repeats
    permuted_auc /= n_repeats

    paired = base_ic - permuted_ic
    mean = float(paired.mean()) if len(paired) else 0.0
    se = float(paired.std(ddof=1) / math.sqrt(len(paired))) if len(paired) > 1 else 0.0
    if abs(mean) < IC_DROP_EPSILON:  # see IC_DROP_EPSILON — a t-stat here is dust/dust
        mean, se = 0.0, 0.0
    correlation, partner = _correlation_neighbour(correlations, feature_names, columns)
    return ImportanceEntry(
        name=name,
        members=members,
        ic_drop=mean,
        ic_drop_se=se,
        tstat=mean / se if se > 0 else 0.0,
        auc_drop=base_auc - permuted_auc,
        max_abs_correlation=correlation,
        most_correlated_with=partner,
    )


def permutation_importance(
    model: Predictor,
    x: np.ndarray,
    y: np.ndarray,
    dates: list[datetime],
    next_returns: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
    seed: int = 7,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> ImportanceReport:
    """Within-session permutation importance of every column and every family.

    `x`/`y`/`dates`/`next_returns` must describe a window the model did NOT see
    while fitting — the holdout. Raises ValueError when the window is too short
    for the paired t-statistic to mean anything, rather than returning a table
    of ratios computed from three sessions.
    """
    if not feature_names:
        raise ValueError("no features to measure")
    sessions = session_groups(dates)
    if len(sessions) < MIN_SESSIONS:
        raise ValueError(
            f"permutation importance needs ≥ {MIN_SESSIONS} rankable sessions, got {len(sessions)}"
        )
    repeats = max(1, int(n_repeats))
    rng = np.random.default_rng(seed)

    base_probs = model.predict_proba(x)
    base_ic = session_ic_series(base_probs, next_returns, sessions)
    base_auc = auc(y, base_probs)

    index_of = {name: i for i, name in enumerate(feature_names)}
    families = groups if groups is not None else feature_groups(feature_names)
    correlations = _rank_correlations(x)

    def measure(name: str, members: tuple[str, ...]) -> ImportanceEntry:
        return _entry(
            name,
            members,
            [index_of[m] for m in members],
            x,
            y,
            next_returns,
            sessions,
            base_ic,
            base_auc,
            model,
            feature_names,
            correlations,
            repeats,
            rng,
        )

    entries = [measure(name, (name,)) for name in feature_names]
    family_entries = [
        measure(group, tuple(m for m in members if m in index_of))
        for group, members in sorted(families.items())
        if sum(1 for m in members if m in index_of) > 1
    ]
    entries.sort(key=lambda e: e.ic_drop, reverse=True)
    family_entries.sort(key=lambda e: e.ic_drop, reverse=True)

    # The noise column, when the caller planted one, is a MEASUREMENT and not a
    # test the bar has to be widened for — it is there precisely to say what
    # this machinery scores on a column that carries nothing.
    noise = next((e for e in entries if e.name == NOISE_FEATURE), None)
    bar = sidak_tstat_bar(sum(1 for e in entries if e.name != NOISE_FEATURE))

    report = ImportanceReport(
        n_rows=int(x.shape[0]),
        n_sessions=len(sessions),
        base_ic=float(base_ic.mean()) if len(base_ic) else 0.0,
        base_auc=base_auc,
        n_repeats=repeats,
        tstat_bar=bar,
        features=tuple(entries),
        groups=tuple(family_entries),
        noise_control=noise,
        verdict=_verdict(entries, family_entries, noise, bar),
    )
    logger.info(
        "Permutation importance measured",
        sessions=len(sessions),
        features=len(entries),
        base_ic=round(report.base_ic, 5),
        strongest=entries[0].name if entries else None,
    )
    return report


def _verdict(
    entries: list[ImportanceEntry],
    families: list[ImportanceEntry],
    noise: ImportanceEntry | None,
    bar: float,
) -> str:
    """What the table says, including the two ways it can be misread."""
    if not entries:
        return "No features were measured."
    real = [e for e in entries if e.name != NOISE_FEATURE]
    credible = [e for e in real if e.tstat > bar]
    parts: list[str] = []
    if noise is not None:
        parts.append(
            f"A planted column of pure noise scored t={noise.tstat:+.2f} "
            f"(IC drop {noise.ic_drop:+.5f}) — that is what this machinery reports for a "
            "feature that carries nothing."
        )
    if not credible:
        best = max(real, key=lambda e: e.tstat)
        parts.append(
            f"No feature's permutation drops the IC by more than noise at the corrected bar "
            f"(|t| > {bar:.2f} for {len(real)} tests); the strongest is {best.name} at "
            f"t={best.tstat:+.2f}. Read that as the model not depending on any single input — "
            "which, when the model itself has no measured edge, is the expected reading and "
            "not a finding about the features."
        )
    else:
        named = ", ".join(
            f"{e.name} (t={e.tstat:+.2f}, ΔIC {e.ic_drop:+.5f})" for e in credible[:5]
        )
        parts.append(
            f"{len(credible)} of {len(real)} features clear the corrected bar "
            f"(|t| > {bar:.2f}): {named}."
        )
    strong_family = [f for f in families if f.tstat > bar]
    if strong_family:
        parts.append(
            "By family: "
            + ", ".join(f"{f.name} (t={f.tstat:+.2f})" for f in strong_family[:4])
            + ". A family can matter while none of its members does on its own — that is "
            "redundancy inside the family, not a contradiction."
        )
    redundant = [e for e in real if e.redundant]
    if redundant:
        parts.append(
            "Credit is split for "
            + ", ".join(f"{e.name}~{e.most_correlated_with}" for e in redundant[:4])
            + ": each has a near-duplicate that stays unpermuted, so a small drop means "
            "'recoverable from its twin' rather than 'carries nothing', and two large ones "
            "are not two separate contributions to be added up. The family row is the "
            "number to read for these."
        )
    return " ".join(parts)


def noise_control_column(dates: list[datetime], seed: int = 11) -> np.ndarray:
    """A column that is a valid cross-sectional rank and predicts nothing.

    Built as a random permutation of the evenly spaced rank grid WITHIN each
    session, so it matches the construction of every real column
    (``cross_sectional_rank``) instead of merely matching its range — a control
    that is distributed differently from the thing it controls for measures the
    difference in distribution as well as the difference in information.
    """
    rng = np.random.default_rng(seed)
    column = np.full(len(dates), 0.5, dtype=float)
    for rows in session_groups(dates):
        n = len(rows)
        grid = (np.arange(n, dtype=float) + 0.5) / n
        column[rows] = rng.permutation(grid)
    return column


__all__ = [
    "NOISE_FEATURE",
    "ImportanceEntry",
    "ImportanceReport",
    "feature_groups",
    "noise_control_column",
    "permutation_importance",
    "sidak_tstat_bar",
]
