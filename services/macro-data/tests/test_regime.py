"""Tests for the rule-based regime classifier."""

from trading_common.schemas import MacroRegime

from src.core.regime import classify_regime


class TestFullSignals:
    def test_expansion(self):
        assert classify_regime(1.5, 1.2, 57) == MacroRegime.EXPANSION

    def test_recovery(self):
        assert classify_regime(1.0, 1.5, 53) == MacroRegime.RECOVERY

    def test_slowdown_from_pmi(self):
        assert classify_regime(0.5, 1.5, 51) == MacroRegime.SLOWDOWN

    def test_contraction_from_pmi(self):
        assert classify_regime(0.5, 1.5, 47) == MacroRegime.CONTRACTION

    def test_crisis_from_spread(self):
        assert classify_regime(1.0, 3.5, 57) == MacroRegime.CRISIS

    def test_crisis_from_deep_inversion_and_weak_pmi(self):
        assert classify_regime(-0.8, 1.5, 44) == MacroRegime.CRISIS


class TestPriorityOrder:
    def test_spread_crisis_beats_strong_pmi(self):
        # severe credit stress dominates even a booming PMI
        assert classify_regime(2.0, 4.0, 60) == MacroRegime.CRISIS

    def test_inverted_curve_forces_contraction(self):
        # inverted curve with an otherwise-OK PMI → contraction (recession signal)
        assert classify_regime(-0.2, 1.5, 53) == MacroRegime.CONTRACTION

    def test_elevated_spread_forces_slowdown(self):
        # PMI healthy but spreads elevated (>=2.0) → slowdown, not expansion
        assert classify_regime(1.0, 2.5, 57) == MacroRegime.SLOWDOWN


class TestMissingInputs:
    def test_no_signals_returns_none(self):
        assert classify_regime() is None

    def test_pmi_only_expansion(self):
        assert classify_regime(pmi=58) == MacroRegime.EXPANSION

    def test_pmi_only_recovery(self):
        assert classify_regime(pmi=53) == MacroRegime.RECOVERY

    def test_curve_only_positive_expansion(self):
        assert classify_regime(yield_curve_10y_2y=1.0) == MacroRegime.EXPANSION

    def test_curve_only_inverted_contraction(self):
        assert classify_regime(yield_curve_10y_2y=-0.3) == MacroRegime.CONTRACTION

    def test_spread_only_crisis(self):
        assert classify_regime(credit_spread_baa_10y=3.2) == MacroRegime.CRISIS


class TestBoundaries:
    def test_pmi_50_is_slowdown(self):
        # 50 is below the 52 slowdown line and >= 48 → slowdown
        assert classify_regime(0.5, 1.0, 50) == MacroRegime.SLOWDOWN

    def test_pmi_55_is_expansion(self):
        assert classify_regime(0.5, 1.0, 55) == MacroRegime.EXPANSION

    def test_flat_curve_zero_not_inverted(self):
        # curve exactly 0 is not < 0 → not forced to contraction
        assert classify_regime(0.0, 1.0, 56) == MacroRegime.EXPANSION
