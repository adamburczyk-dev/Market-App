"""Shared strategy rules + registry.

Importing this package registers every built-in rule, so `all_strategies()` is
populated by the import alone — the strategy service and backtest do not each
maintain their own list of what exists.

Deliberately NOT here: pair trading. It needs a second price series (spread,
cointegration) and `compute_feature_vector` sees one symbol by construction —
the same structural block as `beta_60`/`idio_vol_60`. It is a serving-contract
change, not a fifth entry in this file.
"""

from trading_common.strategies.base import (
    HOLD,
    KNOWN_FEATURES,
    RuleOutput,
    StrategyRule,
    all_strategies,
    apply_params,
    get_strategy,
    pick,
    register,
    saturating_confidence,
    strategy_names,
)
from trading_common.strategies.momentum import MomentumRank, momentum_rank
from trading_common.strategies.reversion import RsiBollingerReversion, rsi_bollinger_reversion
from trading_common.strategies.trend import (
    DonchianBreakout,
    MacdConfirmation,
    SmaEmaCrossover,
    donchian_breakout,
    macd_confirmation,
    sma_ema_crossover,
)

__all__ = [
    "HOLD",
    "KNOWN_FEATURES",
    "DonchianBreakout",
    "MacdConfirmation",
    "MomentumRank",
    "RsiBollingerReversion",
    "RuleOutput",
    "SmaEmaCrossover",
    "StrategyRule",
    "all_strategies",
    "apply_params",
    "donchian_breakout",
    "get_strategy",
    "macd_confirmation",
    "momentum_rank",
    "pick",
    "register",
    "rsi_bollinger_reversion",
    "saturating_confidence",
    "sma_ema_crossover",
    "strategy_names",
]
