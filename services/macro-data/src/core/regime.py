"""Rule-based market-regime classification from macro indicators.

Maps a handful of macro signals to one of the five ``MacroRegime`` values that
risk-mgmt's RegimeAllocator already consumes (expansion/recovery/slowdown/
contraction/crisis → equity-exposure caps). Rules are evaluated in severity
order and tolerate missing inputs (only rules whose inputs are present fire).

Signals & research basis:
- Yield curve (10y-2y): inversion (< 0) is a classic recession lead
  (Estrella & Mishkin 1998).
- Credit spread (BAA-10y): widening reflects financial stress
  (Gilchrist & Zakrajšek 2012); very wide → crisis.
- PMI: the 50 line separates expansion from contraction (ISM convention).
"""

from dataclasses import dataclass

from trading_common.schemas import MacroRegime


@dataclass(frozen=True)
class RegimeThresholds:
    crisis_spread: float = 3.0  # BAA-10y spread (pp) → severe stress
    stress_spread: float = 2.0  # elevated but not crisis
    pmi_contraction: float = 48.0  # clearly below the 50 line
    pmi_slowdown: float = 52.0  # below-trend growth
    pmi_expansion: float = 55.0  # robust growth
    deep_inversion: float = -0.5  # 10y-2y this negative + weak PMI → crisis


DEFAULT_THRESHOLDS = RegimeThresholds()


def classify_regime(
    yield_curve_10y_2y: float | None = None,
    credit_spread_baa_10y: float | None = None,
    pmi: float | None = None,
    thresholds: RegimeThresholds = DEFAULT_THRESHOLDS,
) -> MacroRegime | None:
    """Classify the macro regime; returns ``None`` when no signal is available."""
    t = thresholds
    curve = yield_curve_10y_2y
    spread = credit_spread_baa_10y

    # 1) Crisis — severe credit stress, or a deep inversion alongside a weak PMI.
    if spread is not None and spread >= t.crisis_spread:
        return MacroRegime.CRISIS
    deep_inversion = curve is not None and curve <= t.deep_inversion
    if deep_inversion and pmi is not None and pmi < t.pmi_contraction:
        return MacroRegime.CRISIS

    # 2) Contraction — sub-48 PMI or an inverted curve (recession signal).
    if pmi is not None and pmi < t.pmi_contraction:
        return MacroRegime.CONTRACTION
    if curve is not None and curve < 0:
        return MacroRegime.CONTRACTION

    # 3) Slowdown — below-trend PMI or elevated (pre-crisis) spreads.
    if pmi is not None and pmi < t.pmi_slowdown:
        return MacroRegime.SLOWDOWN
    if spread is not None and spread >= t.stress_spread:
        return MacroRegime.SLOWDOWN

    # 4) PMI-driven growth split (curve already known non-inverted here).
    if pmi is not None:
        return MacroRegime.EXPANSION if pmi >= t.pmi_expansion else MacroRegime.RECOVERY

    # 5) No PMI — fall back to the curve when it is the only signal.
    if curve is not None:
        return MacroRegime.EXPANSION if curve > 0 else MacroRegime.CONTRACTION

    return None
