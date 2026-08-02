"""Tests for backtesting a registered rule — the object the service trades."""

import numpy as np
import pytest
from trading_common.features import FEATURE_LOOKBACK
from trading_common.schemas import Signal
from trading_common.strategies import donchian_breakout, momentum_rank
from trading_common.strategies.base import HOLD, RuleOutput

from src.core.rule_engine import CrossSectionalRuleError, rule_positions, run_rule_backtest

from .conftest import make_bars, trending_closes


class ScriptedRule:
    """A rule whose decisions are fixed per bar index — lets the position path
    be asserted exactly, without reasoning about indicators."""

    name = "scripted"
    required_features: frozenset[str] = frozenset()
    required_ranks: frozenset[str] = frozenset()
    default_params: dict[str, float] = {}

    def __init__(self, script: list[Signal]) -> None:
        self.script = script
        self.calls = 0
        self.window_sizes: list[int] = []

    def generate(self, ranked, raw, params=None):  # type: ignore[no-untyped-def]
        decision = self.script[self.calls]
        self.calls += 1
        if decision is Signal.HOLD:
            return HOLD
        return RuleOutput(signal=decision, confidence=0.9)


def test_hold_carries_the_previous_position():
    """A rule that fires rarely must not read as flat. Recomputing the position
    from scratch each bar would make every HOLD an exit."""
    script = [Signal.HOLD, Signal.BUY, Signal.HOLD, Signal.HOLD, Signal.SELL, Signal.HOLD]
    bars = make_bars(trending_closes(n=len(script)))
    position = rule_positions(bars, ScriptedRule(script))
    assert list(position) == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_sell_is_an_exit_not_a_short():
    """Long/flat, matching the live path (R4). A short would need modelling
    end-to-end — engine, sizing and broker — not a sign flip here."""
    script = [Signal.SELL, Signal.SELL, Signal.SELL]
    bars = make_bars(trending_closes(n=3))
    assert list(rule_positions(bars, ScriptedRule(script))) == [0.0, 0.0, 0.0]


def test_the_rule_sees_the_same_features_serving_would_compute():
    """A different trailing window here would make the backtest grade a
    different function under the same name — the exact drift this module
    exists to remove."""
    from trading_common.features import compute_feature_vector

    n = FEATURE_LOOKBACK + 50
    bars = make_bars(trending_closes(n=n, seed=9))
    seen: list[dict] = []

    rule = ScriptedRule([Signal.HOLD] * n)
    rule.generate = lambda ranked, raw, params=None: (  # type: ignore[method-assign]
        seen.append(dict(raw)) or HOLD
    )
    rule_positions(bars, rule)

    assert len(seen) == n  # one decision per bar, none skipped
    serving = compute_feature_vector(bars[-FEATURE_LOOKBACK:]).features
    assert seen[-1] == pytest.approx(serving)


def test_a_rule_gets_no_ranks_here_and_must_not_be_a_ranked_rule():
    rule = ScriptedRule([Signal.HOLD])

    captured: list[dict] = []

    def capture(ranked, raw, params=None):  # type: ignore[no-untyped-def]
        captured.append(dict(ranked))
        return HOLD

    rule.generate = capture  # type: ignore[method-assign]
    rule_positions(make_bars(trending_closes(n=3)), rule)
    # Empty on purpose: a percentile only exists relative to a universe, and
    # handing a rule an empty mapping is what makes `pick` return None → HOLD
    # rather than inventing a neutral 0.5.
    assert captured == [{}, {}, {}]


def test_cross_sectional_rules_are_refused_with_the_missing_input_named():
    bars = make_bars(trending_closes(n=60))
    with pytest.raises(CrossSectionalRuleError) as exc:
        rule_positions(bars, momentum_rank)
    assert "momentum_20" in str(exc.value)
    assert "universe" in str(exc.value)


def test_run_rule_backtest_scores_a_real_rule():
    bars = make_bars(trending_closes(n=200, seed=5))
    prices = np.array([b.close for b in bars])
    result = run_rule_backtest(bars, donchian_breakout, prices, cost_bps=0.0)
    assert result.n_bars == 199
    assert result.n_trades >= 1


def test_too_few_bars_is_an_empty_result_not_a_crash():
    result = run_rule_backtest(make_bars([100.0]), donchian_breakout, np.array([100.0]))
    assert result.n_bars == 0
