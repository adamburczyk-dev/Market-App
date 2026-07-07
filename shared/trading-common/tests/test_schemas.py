"""Testy kontraktów Pydantic — walidacja danych między serwisami."""

from datetime import UTC, date, datetime

import pytest

from trading_common.schemas import (
    CompanyProfile,
    FeatureVector,
    FinancialStatements,
    Interval,
    MacroRegime,
    MacroSnapshot,
    OHLCVBar,
    PortfolioMetrics,
    SentimentSnapshot,
    Signal,
    TradingSignal,
)


def make_bar(**kwargs) -> dict:
    defaults = {
        "symbol": "AAPL",
        "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        "interval": Interval.D1,
        "open": 150.0,
        "high": 155.0,
        "low": 149.0,
        "close": 153.0,
        "volume": 1_000_000.0,
    }
    return {**defaults, **kwargs}


class TestOHLCVBar:
    def test_valid_bar(self):
        bar = OHLCVBar(**make_bar())
        assert bar.symbol == "AAPL"
        assert bar.interval == Interval.D1

    def test_high_gte_low_valid(self):
        bar = OHLCVBar(**make_bar(high=155.0, low=149.0))
        assert bar.high > bar.low

    def test_high_less_than_low_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(high=148.0, low=149.0))

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(open=-1.0))

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(close=0.0))

    def test_negative_volume_raises(self):
        with pytest.raises(ValueError):
            OHLCVBar(**make_bar(volume=-100.0))

    def test_zero_volume_allowed(self):
        bar = OHLCVBar(**make_bar(volume=0.0))
        assert bar.volume == 0.0

    def test_optional_source(self):
        bar = OHLCVBar(**make_bar(source="yfinance"))
        assert bar.source == "yfinance"

    def test_interval_enum_values(self):
        assert Interval.M1 == "1m"
        assert Interval.D1 == "1d"

    def test_serialization_roundtrip(self):
        bar = OHLCVBar(**make_bar())
        restored = OHLCVBar.model_validate(bar.model_dump())
        assert restored == bar


class TestTradingSignal:
    def make_signal(self, **kwargs) -> dict:
        defaults = {
            "symbol": "AAPL",
            "strategy": "sma_crossover",
            "signal": Signal.BUY,
            "confidence": 0.75,
            "price": 153.0,
            "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            "stop_loss": 145.0,
        }
        return {**defaults, **kwargs}

    def test_buy_without_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss is required"):
            TradingSignal(**self.make_signal(signal=Signal.BUY, stop_loss=None))

    def test_sell_without_stop_loss_raises(self):
        with pytest.raises(ValueError, match="stop_loss is required"):
            TradingSignal(**self.make_signal(signal=Signal.SELL, stop_loss=None))

    def test_hold_without_stop_loss_allowed(self):
        sig = TradingSignal(**self.make_signal(signal=Signal.HOLD, stop_loss=None))
        assert sig.signal == Signal.HOLD
        assert sig.stop_loss is None

    def test_valid_signal(self):
        sig = TradingSignal(**self.make_signal())
        assert sig.signal == Signal.BUY
        assert sig.confidence == 0.75

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            TradingSignal(**self.make_signal(confidence=1.5))

    def test_negative_confidence_raises(self):
        with pytest.raises(ValueError):
            TradingSignal(**self.make_signal(confidence=-0.1))

    def test_confidence_boundary_values(self):
        TradingSignal(**self.make_signal(confidence=0.0))
        TradingSignal(**self.make_signal(confidence=1.0))

    def test_all_signal_values(self):
        for sig in [Signal.BUY, Signal.SELL, Signal.HOLD]:
            s = TradingSignal(**self.make_signal(signal=sig))
            assert s.signal == sig

    def test_optional_stop_loss_take_profit(self):
        sig = TradingSignal(**self.make_signal(stop_loss=145.0, take_profit=165.0))
        assert sig.stop_loss == 145.0
        assert sig.take_profit == 165.0

    def test_metadata_default_empty(self):
        sig = TradingSignal(**self.make_signal())
        assert sig.metadata == {}


class TestPortfolioMetrics:
    def test_valid_metrics(self):
        m = PortfolioMetrics(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            total_value=105_000.0,
            cash=50_000.0,
            positions_value=55_000.0,
            daily_pnl=500.0,
            daily_pnl_pct=0.005,
            sharpe_ratio=1.5,
            max_drawdown=-0.08,
        )
        assert m.total_value == 105_000.0
        assert m.sharpe_ratio == 1.5

    def test_optional_fields_default_none(self):
        m = PortfolioMetrics(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            total_value=100_000.0,
            cash=100_000.0,
            positions_value=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
        )
        assert m.sharpe_ratio is None
        assert m.var_95 is None


class TestCompanyProfile:
    def test_minimal(self):
        p = CompanyProfile(symbol="AAPL")
        assert p.symbol == "AAPL"
        assert p.model_stack is None

    def test_full(self):
        p = CompanyProfile(
            symbol="NVDA",
            name="NVIDIA",
            sector="Information Technology",
            market_cap=3_000_000_000_000.0,
            style="growth",
            model_stack="growth_tech_v1",
        )
        assert p.style == "growth"

    def test_negative_market_cap_raises(self):
        with pytest.raises(ValueError):
            CompanyProfile(symbol="X", market_cap=-1.0)


class TestFinancialStatements:
    def test_valid(self):
        fs = FinancialStatements(
            symbol="AAPL",
            period_end=date(2024, 3, 31),
            fiscal_period="Q1",
            revenue=90_000_000_000.0,
            piotroski_f_score=8,
        )
        assert fs.fiscal_period == "Q1"
        assert fs.piotroski_f_score == 8

    def test_f_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FinancialStatements(
                symbol="X", period_end=date(2024, 1, 1), fiscal_period="FY", piotroski_f_score=10
            )

    def test_balance_sheet_detail_for_full_piotroski(self):
        fs = FinancialStatements(
            symbol="AAPL",
            period_end=date(2024, 9, 28),
            fiscal_period="FY",
            current_assets=152_987.0,
            current_liabilities=176_392.0,
            shares_outstanding=15_116.0,
        )
        assert fs.current_assets == 152_987.0
        assert fs.current_liabilities == 176_392.0
        assert fs.shares_outstanding == 15_116.0

    def test_balance_sheet_detail_defaults_none(self):
        fs = FinancialStatements(symbol="X", period_end=date(2024, 1, 1), fiscal_period="FY")
        assert fs.current_assets is None
        assert fs.current_liabilities is None
        assert fs.shares_outstanding is None

    def test_negative_shares_outstanding_raises(self):
        with pytest.raises(ValueError):
            FinancialStatements(
                symbol="X", period_end=date(2024, 1, 1), fiscal_period="FY", shares_outstanding=-1.0
            )


class TestMacroSnapshot:
    def test_minimal(self):
        s = MacroSnapshot(timestamp=datetime(2024, 1, 1, tzinfo=UTC))
        assert s.regime is None

    def test_with_regime(self):
        s = MacroSnapshot(timestamp=datetime(2024, 1, 1, tzinfo=UTC), regime=MacroRegime.CRISIS)
        assert s.regime == MacroRegime.CRISIS

    def test_regime_values_match_allocator_keys(self):
        # Wartości muszą zgadzać się z risk-mgmt RegimeAllocator
        assert {r.value for r in MacroRegime} == {
            "expansion",
            "recovery",
            "slowdown",
            "contraction",
            "crisis",
        }


class TestSentimentSnapshot:
    def test_valid(self):
        s = SentimentSnapshot(
            symbol="TSLA", timestamp=datetime(2024, 1, 1, tzinfo=UTC), sentiment_score=0.3
        )
        assert s.news_count == 0

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            SentimentSnapshot(
                symbol="X", timestamp=datetime(2024, 1, 1, tzinfo=UTC), sentiment_score=1.5
            )


class TestFeatureVector:
    def test_valid(self):
        fv = FeatureVector(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            interval=Interval.D1,
            features={"rsi_14": 0.6, "macd": -0.2},
            tier=1,
            rank_transformed=True,
        )
        assert fv.features["rsi_14"] == 0.6
        assert fv.rank_transformed is True

    def test_defaults(self):
        fv = FeatureVector(
            symbol="MSFT", timestamp=datetime(2024, 1, 1, tzinfo=UTC), interval=Interval.H1
        )
        assert fv.features == {}
        assert fv.tier is None

    def test_tier_out_of_range_raises(self):
        with pytest.raises(ValueError):
            FeatureVector(
                symbol="X",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                interval=Interval.D1,
                tier=4,
            )
