"""Testy registry strategii i czterech reguł (czyste funkcje, bez I/O)."""

import pytest

from trading_common.features import TECHNICAL_FEATURES
from trading_common.schemas import Signal
from trading_common.strategies import (
    KNOWN_FEATURES,
    all_strategies,
    donchian_breakout,
    get_strategy,
    macd_confirmation,
    momentum_rank,
    register,
    rsi_bollinger_reversion,
    sma_ema_crossover,
    strategy_names,
)
from trading_common.strategies.base import RuleOutput, saturating_confidence

# --- registry --------------------------------------------------------------


def test_every_builtin_rule_is_registered_by_importing_the_package():
    assert strategy_names() == [
        "donchian_breakout",
        "macd_confirmation",
        "momentum_rank",
        "rsi_bollinger_reversion",
        "sma_ema_crossover",
    ]
    assert [r.name for r in all_strategies()] == strategy_names()


def test_order_is_stable_not_import_order():
    """Evaluation order decides which signal reaches the aggregator first; an
    import-order-dependent sequence would make the same inputs produce
    different event streams across restarts."""
    assert [r.name for r in all_strategies()] == sorted(r.name for r in all_strategies())


def test_get_strategy_names_the_alternatives_when_it_fails():
    assert get_strategy("momentum_rank") is momentum_rank
    with pytest.raises(KeyError) as exc:
        get_strategy("nie_ma_takiej")
    assert "momentum_rank" in str(exc.value)


def test_registration_refuses_a_rule_needing_a_feature_nothing_computes():
    class Bogus:
        name = "bogus"
        required_features = frozenset({"sentiment_score"})
        required_ranks: frozenset[str] = frozenset()
        default_params: dict[str, float] = {}

        def generate(self, ranked, raw, params=None):  # pragma: no cover - never called
            return RuleOutput(signal=Signal.HOLD, confidence=0.5)

    with pytest.raises(ValueError, match="sentiment_score"):
        register(Bogus())
    assert "bogus" not in strategy_names()


def test_registration_refuses_a_duplicate_name():
    class Clone:
        name = "momentum_rank"
        required_features = frozenset({"rsi_14"})
        required_ranks: frozenset[str] = frozenset()
        default_params: dict[str, float] = {}

        def generate(self, ranked, raw, params=None):  # pragma: no cover - never called
            return RuleOutput(signal=Signal.HOLD, confidence=0.5)

    with pytest.raises(ValueError, match="already registered"):
        register(Clone())


def test_declared_inputs_of_every_rule_are_actually_computable():
    for rule in all_strategies():
        declared = rule.required_features | rule.required_ranks
        assert declared <= KNOWN_FEATURES, rule.name
        assert declared, f"{rule.name} declares no inputs"


def test_a_rank_is_declared_as_a_rank_not_as_a_raw_feature():
    """The split decides where a rule can be evaluated at all: a percentile
    only exists relative to a universe, so a single-symbol backtest has to be
    able to refuse — by name, not by guessing."""
    assert momentum_rank.required_ranks == frozenset({"momentum_20"})
    assert "momentum_20" not in momentum_rank.required_features
    # Every other built-in reads only the symbol's own vector.
    for rule in all_strategies():
        if rule.name != "momentum_rank":
            assert rule.required_ranks == frozenset(), rule.name


def test_known_features_is_a_superset_of_the_technical_block():
    assert TECHNICAL_FEATURES <= KNOWN_FEATURES


# --- shared helpers --------------------------------------------------------


def test_saturating_confidence_spans_half_to_one():
    assert saturating_confidence(0.0, 0.03) == pytest.approx(0.5)
    assert saturating_confidence(0.03, 0.03) == pytest.approx(1.0)
    assert saturating_confidence(10.0, 0.03) == pytest.approx(1.0)  # clamped
    assert 0.5 < saturating_confidence(0.01, 0.03) < 1.0
    assert saturating_confidence(-0.03, 0.03) == pytest.approx(1.0)  # magnitude
    assert saturating_confidence(1.0, 0.0) == pytest.approx(0.5)  # degenerate scale


# --- momentum_rank (behaviour preserved from the service) ------------------


def test_momentum_rank_buys_the_top_unless_already_overbought():
    buy = momentum_rank.generate({"momentum_20": 0.9}, {"rsi_14": 55.0})
    assert buy.signal is Signal.BUY
    assert buy.confidence == pytest.approx(0.9)
    stretched = momentum_rank.generate({"momentum_20": 0.9}, {"rsi_14": 85.0})
    assert stretched.signal is Signal.HOLD


def test_momentum_rank_sells_the_bottom_unless_already_oversold():
    sell = momentum_rank.generate({"momentum_20": 0.1}, {"rsi_14": 45.0})
    assert sell.signal is Signal.SELL
    assert sell.confidence == pytest.approx(0.9)
    assert momentum_rank.generate({"momentum_20": 0.1}, {"rsi_14": 20.0}).signal is Signal.HOLD


def test_momentum_rank_reads_the_rank_from_ranked_and_the_level_from_raw():
    """The whole point of the two-vector signature: a rank means 'versus the
    universe', a level means 'versus itself'. Swapping them silently changes
    what the rule bets on."""
    assert momentum_rank.generate({}, {"momentum_20": 0.9, "rsi_14": 55.0}).signal is Signal.HOLD


# --- sma_ema_crossover -----------------------------------------------------


def test_crossover_requires_both_pairs_to_agree():
    up = {"ema_12": 103.0, "ema_26": 100.0, "sma_20": 102.0, "sma_50": 100.0}
    assert sma_ema_crossover.generate({}, up).signal is Signal.BUY
    down = {"ema_12": 97.0, "ema_26": 100.0, "sma_20": 98.0, "sma_50": 100.0}
    assert sma_ema_crossover.generate({}, down).signal is Signal.SELL
    # Fast says up, slow says down → no trade. This is the rule.
    mixed = {"ema_12": 103.0, "ema_26": 100.0, "sma_20": 98.0, "sma_50": 100.0}
    assert sma_ema_crossover.generate({}, mixed).signal is Signal.HOLD


def test_crossover_confidence_follows_the_WEAKER_leg():
    """The rule is only as convinced as its least convinced half — otherwise a
    wide fast spread would hide a barely-crossed slow pair."""
    strong_fast_weak_slow = {
        "ema_12": 110.0,
        "ema_26": 100.0,
        "sma_20": 100.1,
        "sma_50": 100.0,
    }
    both_strong = {"ema_12": 110.0, "ema_26": 100.0, "sma_20": 110.0, "sma_50": 100.0}
    weak = sma_ema_crossover.generate({}, strong_fast_weak_slow)
    strong = sma_ema_crossover.generate({}, both_strong)
    assert weak.confidence < strong.confidence
    assert strong.confidence == pytest.approx(1.0)


# --- rsi_bollinger_reversion (closes D5) -----------------------------------


def test_reversion_needs_the_rsi_extreme_AND_the_band():
    at_band = rsi_bollinger_reversion.generate({}, {"rsi_14": 25.0, "bb_pct_b": 0.02})
    assert at_band.signal is Signal.BUY
    # RSI alone flags a strong trend as often as an exhausted one.
    rsi_only = rsi_bollinger_reversion.generate({}, {"rsi_14": 25.0, "bb_pct_b": 0.5})
    assert rsi_only.signal is Signal.HOLD
    band_only = rsi_bollinger_reversion.generate({}, {"rsi_14": 50.0, "bb_pct_b": 0.01})
    assert band_only.signal is Signal.HOLD


def test_reversion_bets_against_momentum_on_the_same_input():
    """D5 settled: these are opposite bets, so they are separate rules. Folding
    the RSI filter into momentum would have averaged out the disagreement the
    aggregator exists to weigh."""
    raw = {"rsi_14": 78.0, "bb_pct_b": 0.99}
    assert rsi_bollinger_reversion.generate({}, raw).signal is Signal.SELL
    assert momentum_rank.generate({"momentum_20": 0.95}, raw).signal is Signal.HOLD


def test_reversion_stops_tighter_than_the_breakout_rule():
    """A reversion entry is wrong the moment the move continues; a breakout has
    to survive the pullback that follows it."""
    rev = rsi_bollinger_reversion.generate({}, {"rsi_14": 20.0, "bb_pct_b": 0.0})
    brk = donchian_breakout.generate({}, {"donchian_pos_20": 1.2})
    assert rev.stop_atr_mult < brk.stop_atr_mult
    assert rev.take_profit_rr < brk.take_profit_rr


# --- macd_confirmation -----------------------------------------------------


def test_macd_confirmation_requires_agreement_with_the_20d_return():
    agree = {"macd_hist": 1.0, "close": 100.0, "return_20d": 0.05}
    assert macd_confirmation.generate({}, agree).signal is Signal.BUY
    disagree = {"macd_hist": 1.0, "close": 100.0, "return_20d": -0.05}
    assert macd_confirmation.generate({}, disagree).signal is Signal.HOLD
    both_down = {"macd_hist": -1.0, "close": 100.0, "return_20d": -0.05}
    assert macd_confirmation.generate({}, both_down).signal is Signal.SELL


def test_macd_confirmation_scales_the_histogram_by_price():
    """Same histogram means something different on a $20 and a $2000 stock."""
    cheap = macd_confirmation.generate({}, {"macd_hist": 1.0, "close": 20.0, "return_20d": 0.05})
    rich = macd_confirmation.generate({}, {"macd_hist": 1.0, "close": 2000.0, "return_20d": 0.05})
    assert cheap.confidence > rich.confidence


# --- donchian_breakout -----------------------------------------------------


def test_breakout_fires_only_outside_the_channel():
    assert donchian_breakout.generate({}, {"donchian_pos_20": 1.05}).signal is Signal.BUY
    assert donchian_breakout.generate({}, {"donchian_pos_20": -0.05}).signal is Signal.SELL
    assert donchian_breakout.generate({}, {"donchian_pos_20": 0.99}).signal is Signal.HOLD
    assert donchian_breakout.generate({}, {"donchian_pos_20": 1.0}).signal is Signal.HOLD


# --- shared contract every rule has to honour ------------------------------


@pytest.mark.parametrize("rule", all_strategies(), ids=lambda r: r.name)
def test_a_rule_holds_on_a_missing_feature_instead_of_raising(rule):
    """A partially-computed vector is normal early in a symbol's history. A
    rule is not the place to decide the universe is broken — and it must never
    substitute a default, because a made-up MACD is a made-up trade."""
    assert rule.generate({}, {}).signal is Signal.HOLD
    partial = {name: 0.5 for name in list(rule.required_features)[:1]}
    assert rule.generate(partial, partial).signal in (Signal.HOLD, Signal.BUY, Signal.SELL)


@pytest.mark.parametrize("rule", all_strategies(), ids=lambda r: r.name)
def test_confidence_stays_in_range_and_protection_is_positive(rule):
    for ranked, raw in (
        (
            {"momentum_20": 0.99},
            {
                "rsi_14": 15.0,
                "bb_pct_b": 0.0,
                "donchian_pos_20": 5.0,
                "macd_hist": 50.0,
                "close": 100.0,
                "return_20d": 0.9,
                "ema_12": 200.0,
                "ema_26": 100.0,
                "sma_20": 200.0,
                "sma_50": 100.0,
            },
        ),
        (
            {"momentum_20": 0.01},
            {
                "rsi_14": 95.0,
                "bb_pct_b": 1.0,
                "donchian_pos_20": -5.0,
                "macd_hist": -50.0,
                "close": 100.0,
                "return_20d": -0.9,
                "ema_12": 50.0,
                "ema_26": 100.0,
                "sma_20": 50.0,
                "sma_50": 100.0,
            },
        ),
    ):
        out = rule.generate(ranked, raw)
        assert 0.0 <= out.confidence <= 1.0, rule.name
        assert out.stop_atr_mult > 0 and out.take_profit_rr > 0, rule.name


@pytest.mark.parametrize("rule", all_strategies(), ids=lambda r: r.name)
def test_params_can_be_overridden_without_reregistering(rule):
    """Config has to be able to retune a rule; a copy in the registry would
    make the tuned and the registered rule two different strategies under one
    name."""
    tuned = rule.generate({}, {}, {"stop_atr_mult": 9.0})
    assert tuned.signal is Signal.HOLD  # empty input still holds
    assert rule.default_params["stop_atr_mult"] != 9.0
