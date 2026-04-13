# Framework Supplement — podniesienie systemu do poziomu 9/10

## Cel dokumentu

Uzupełnienia do trzech istniejących dokumentów (`Plan_Rozwoju`, `ml_integration_plan`, `CLAUDE.md`)
eliminujące zidentyfikowane luki w: ochronie kapitału, adaptacji strategii, monitoringu runtime
i maksymalizacji zysków.

**Zasada:** Każdy komponent poniżej ma przypisany serwis docelowy, schemat danych, eventy
i tydzień implementacji — gotowe do kodowania.

---

## CZĘŚĆ A: OCHRONA KAPITAŁU (podnosi Plan_Rozwoju z 6→9)

### A1. Risk Envelope od dnia 1 (nie od tygodnia 19)

**Problem:** Plan umieszcza `risk-mgmt-svc` w tygodniu 19. Do tego czasu system generuje sygnały
bez jakiejkolwiek kontroli ryzyka.

**Rozwiązanie:** Dwuwarstwowy risk — lekki `risk-envelope` wbudowany w `trading-common` od dnia 1,
pełny `risk-mgmt-svc` od tygodnia 19.

#### Warstwa 1: `RiskEnvelope` w trading-common (od tygodnia 1)

```python
# shared/trading-common/src/trading_common/risk_envelope.py

from dataclasses import dataclass
from trading_common.schemas import TradingSignal


@dataclass
class RiskLimits:
    """Minimalne reguły ryzyka — obowiązują KAŻDY serwis generujący sygnały."""
    max_position_pct: float = 0.05          # Max 5% portfela na pozycję
    max_portfolio_exposure_pct: float = 0.80 # Max 80% portfela w pozycjach (20% cash buffer)
    max_single_loss_pct: float = 0.02       # Max 2% straty na pojedynczej transakcji (risk per trade)
    max_daily_loss_pct: float = 0.05        # Max 5% straty dziennej → stop trading na dziś
    max_drawdown_pct: float = 0.15          # Max 15% drawdown od peak → FLAT all positions
    max_correlated_positions: int = 3       # Max 3 pozycje w tym samym sektorze
    min_confidence: float = 0.55            # Odrzuć sygnały poniżej progu confidence


class RiskEnvelope:
    """
    Lekki pre-trade check. Nie wymaga bazy danych — operuje na bieżącym stanie portfela.
    Każdy serwis generujący sygnały MUSI przepuścić je przez RiskEnvelope.
    """

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def check_signal(
        self,
        signal: TradingSignal,
        portfolio_value: float,
        current_exposure_pct: float,
        current_drawdown_pct: float,
        daily_loss_pct: float,
        sector_positions: dict[str, int],
    ) -> tuple[bool, str]:
        """
        Returns: (approved, reason)
        Wywoływany PRZED publikacją SignalGeneratedEvent.
        """
        # Hard stop: drawdown limit
        if abs(current_drawdown_pct) >= self.limits.max_drawdown_pct:
            return False, f"portfolio_drawdown_{current_drawdown_pct:.1%}_exceeds_limit"

        # Hard stop: daily loss
        if abs(daily_loss_pct) >= self.limits.max_daily_loss_pct:
            return False, f"daily_loss_{daily_loss_pct:.1%}_exceeds_limit"

        # Confidence threshold
        if signal.confidence < self.limits.min_confidence:
            return False, f"confidence_{signal.confidence:.2f}_below_threshold"

        # Exposure limit
        if current_exposure_pct >= self.limits.max_portfolio_exposure_pct:
            return False, f"exposure_{current_exposure_pct:.1%}_exceeds_limit"

        # Risk per trade (wymaga stop_loss)
        if signal.stop_loss:
            risk_per_share = abs(signal.price - signal.stop_loss)
            max_risk = portfolio_value * self.limits.max_single_loss_pct
            max_shares = max_risk / risk_per_share if risk_per_share > 0 else 0
            position_value = max_shares * signal.price
            if position_value / portfolio_value > self.limits.max_position_pct:
                return False, "position_size_exceeds_limit_after_risk_sizing"

        return True, "approved"
```

**Integracja:** `strategy-svc` i `signal-aggregator-svc` importują `RiskEnvelope`
i wywołują `check_signal()` przed publikacją `SignalGeneratedEvent`.

#### Warstwa 2: Pełny `risk-mgmt-svc` (tydzień 19 — bez zmian w planie)

Warstwa 2 dodaje: Kelly sizing, portfolio optimization (HRP), Monte Carlo VaR, correlation matrix.
Warstwa 1 **nie znika** — działa jako first-line defense nawet po wdrożeniu warstwy 2.

---

### A2. Portfolio Circuit Breaker

**Problem:** Brak mechanizmu zatrzymania systemu przy katastrofalnych stratach.

**Serwis:** `risk-mgmt-svc` (tydzień 19), ale event + schema od tygodnia 1.

#### Nowy event: `CircuitBreakerTriggeredEvent`

```python
# Dodać do trading_common/events.py

class CircuitBreakerLevel(StrEnum):
    YELLOW = "yellow"   # Reduced exposure
    RED = "red"         # Flat all, stop new positions
    BLACK = "black"     # System halt, human intervention required

class CircuitBreakerTriggeredEvent(BaseEvent):
    event_type: EventType = EventType.CIRCUIT_BREAKER_TRIGGERED
    level: CircuitBreakerLevel
    trigger_metric: str        # "drawdown", "daily_loss", "var_breach", "correlation_spike"
    current_value: float
    threshold_value: float
    action_taken: str          # "reduce_exposure_50pct", "flatten_all", "halt_system"
    source_service: str = "risk-mgmt"
```

#### Nowy EventType

```python
# Dodać do EventType enum:
CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
```

#### Progi circuit breaker

| Poziom | Trigger | Akcja |
|--------|---------|-------|
| YELLOW | DD > 8% LUB daily loss > 3% LUB VaR breach | Redukcja nowych pozycji o 50%, podwyżka min_confidence do 0.70 |
| RED | DD > 15% LUB daily loss > 5% | Flatten all positions, stop new orders, CRITICAL alert |
| BLACK | DD > 25% LUB anomalia systemowa (np. 10x avg volume spike) | System halt, wymaga ręcznego restartu |

#### Subskrypcja

```
risk-mgmt-svc → PUBLISH CircuitBreakerTriggeredEvent
execution-svc → SUBSCRIBE → enforce (reject new orders / flatten)
notification-svc → SUBSCRIBE → CRITICAL alert
strategy-svc → SUBSCRIBE → stop signal generation (RED/BLACK)
```

---

### A3. Drawdown-Adaptive Position Sizing

**Problem:** Stały position sizing (np. 5% portfela) nie uwzględnia bieżącej kondycji equity curve.

**Serwis:** `risk-mgmt-svc`

```python
# services/risk-mgmt/src/core/adaptive_sizing.py

class DrawdownAdaptiveSizer:
    """
    Im głębszy drawdown, tym mniejsze pozycje — ochrona kapitału w złych okresach.
    Bazuje na: Vince (1990) "Portfolio Management Formulas",
    Tharp (2008) "Position Sizing" — anti-martingale approach.
    """

    def __init__(
        self,
        base_risk_per_trade: float = 0.02,  # 2% risked per trade at equity peak
        dd_scaling_start: float = 0.05,      # Start reducing at 5% DD
        dd_scaling_end: float = 0.15,        # Zero new positions at 15% DD
    ):
        self.base_risk = base_risk_per_trade
        self.dd_start = dd_scaling_start
        self.dd_end = dd_scaling_end

    def compute_risk_budget(self, current_drawdown_pct: float) -> float:
        """
        Returns: fraction of portfolio to risk on next trade (0.0 to base_risk).
        Linear scaling between dd_start and dd_end.
        """
        dd = abs(current_drawdown_pct)
        if dd <= self.dd_start:
            return self.base_risk  # Full risk budget
        if dd >= self.dd_end:
            return 0.0  # No new positions

        # Linear interpolation
        scale = 1.0 - (dd - self.dd_start) / (self.dd_end - self.dd_start)
        return self.base_risk * scale

    def position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: float,
        current_drawdown_pct: float,
    ) -> int:
        """Returns: number of shares to buy."""
        risk_budget = self.compute_risk_budget(current_drawdown_pct)
        if risk_budget <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_loss_price)
        if risk_per_share <= 0:
            return 0

        max_risk_amount = portfolio_value * risk_budget
        shares = int(max_risk_amount / risk_per_share)
        # Cap at max_position_pct
        max_shares_by_position = int(portfolio_value * 0.05 / entry_price)
        return min(shares, max_shares_by_position)
```

**Wizualizacja skalowania:**
```
DD:    0%     5%      10%      15%
Risk:  2.0%   2.0%    1.0%     0.0%
       |======|-------|--------|
       full   scaling  scaling  STOP
```

---

### A4. Regime-Aware Cash Allocation

**Problem:** Signal aggregator identyfikuje CRISIS regime, ale nie wymusza przejścia do cash.

**Serwis:** `risk-mgmt-svc` + `signal-aggregator-svc`

```python
# services/risk-mgmt/src/core/regime_allocator.py

class RegimeAllocator:
    """
    Max equity exposure per market regime.
    Research: Ang & Bekaert (2004) — regime-conditional asset allocation
    improves Sharpe by 0.3-0.5 vs static allocation.
    """

    # Max % portfela w equity per regime
    MAX_EQUITY_EXPOSURE = {
        "expansion":   0.90,  # Full risk-on
        "recovery":    0.80,
        "slowdown":    0.60,  # Defensive tilt
        "contraction": 0.35,  # Mostly cash + defensive
        "crisis":      0.15,  # Minimal exposure, preserve capital
    }

    # Sektory dozwolone per regime (hard filter, nie soft scoring)
    ALLOWED_SECTORS = {
        "expansion":   None,  # All sectors allowed
        "recovery":    None,
        "slowdown":    {"Health Care", "Consumer Staples", "Utilities",
                        "Information Technology"},  # Quality growth + defensive
        "contraction": {"Consumer Staples", "Utilities", "Health Care"},
        "crisis":      {"Consumer Staples", "Utilities"},  # Only ultra-defensive
    }

    def max_exposure(self, regime: str) -> float:
        return self.MAX_EQUITY_EXPOSURE.get(regime, 0.60)

    def is_sector_allowed(self, regime: str, sector: str) -> bool:
        allowed = self.ALLOWED_SECTORS.get(regime)
        if allowed is None:
            return True  # No restrictions
        return sector in allowed

    def required_cash_pct(self, regime: str) -> float:
        return 1.0 - self.max_exposure(regime)
```

**Integracja z signal-aggregator:**
```python
# W aggregate() — dodaj hard gate:
regime = macro.regime.value
if not regime_allocator.is_sector_allowed(regime, sector):
    return {"final_score": 0.0, "reason": f"sector_{sector}_blocked_in_{regime}_regime"}
```

---

## CZĘŚĆ B: ADAPTACJA I MONITORING RUNTIME (podnosi ml_integration_plan z 8→9)

### B1. Live Model Drift Detection

**Problem:** Model wytrenowany na danych historycznych degraduje w czasie (concept drift,
feature drift, covariate shift). Bez monitoringu nie wiadomo kiedy retrenować.

**Serwis:** `ml-pipeline-svc`

```python
# services/ml-pipeline/src/core/monitoring/drift_detector.py

import numpy as np
from scipy import stats
from dataclasses import dataclass
from datetime import date


@dataclass
class DriftReport:
    model_id: str
    report_date: date
    # Feature drift (PSI — Population Stability Index)
    feature_psi_scores: dict[str, float]    # feature_name → PSI
    features_drifted: list[str]             # PSI > 0.2
    # Prediction drift
    prediction_distribution_shift: float     # KS test p-value
    # Performance degradation
    rolling_sharpe_30d: float
    rolling_sharpe_90d: float
    sharpe_decay_pct: float                 # (30d - 90d) / 90d
    rolling_accuracy_30d: float
    # Verdicts
    needs_retrain: bool
    needs_investigation: bool
    recommended_action: str


class DriftDetector:
    """
    Monitoruje 3 rodzaje drift:
    1. Feature drift (PSI): rozkład features się zmienił vs training set
    2. Prediction drift (KS test): model outputs zmieniły rozkład
    3. Performance decay: rolling metrics spadają

    Research: Webb et al. (2016) "Characterizing Concept Drift",
    Gama et al. (2014) "A survey on concept drift adaptation".
    """

    PSI_THRESHOLD = 0.20       # PSI > 0.2 = significant shift
    KS_PVALUE_THRESHOLD = 0.01 # p < 0.01 = prediction distribution shifted
    SHARPE_DECAY_THRESHOLD = -0.30  # 30% spadek Sharpe → retrain
    ACCURACY_MIN = 0.48        # Poniżej random → model uszkodzony

    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """
        Population Stability Index.
        < 0.10 = no shift, 0.10-0.20 = moderate, > 0.20 = significant.
        """
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bin_edges)

        # Normalize + epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = ref_hist / len(reference) + eps
        curr_pct = curr_hist / len(current) + eps

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def check_prediction_drift(
        self, reference_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> float:
        """KS test na prediction distributions. Returns p-value."""
        _, p_value = stats.ks_2samp(reference_predictions, current_predictions)
        return float(p_value)

    def generate_report(
        self,
        model_id: str,
        reference_features: dict[str, np.ndarray],
        current_features: dict[str, np.ndarray],
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        rolling_sharpe_30d: float,
        rolling_sharpe_90d: float,
        rolling_accuracy_30d: float,
    ) -> DriftReport:

        # Feature PSI
        psi_scores = {}
        drifted = []
        for feat_name in reference_features:
            if feat_name in current_features:
                psi = self.compute_psi(reference_features[feat_name],
                                       current_features[feat_name])
                psi_scores[feat_name] = psi
                if psi > self.PSI_THRESHOLD:
                    drifted.append(feat_name)

        # Prediction drift
        pred_drift_p = self.check_prediction_drift(
            reference_predictions, current_predictions)

        # Sharpe decay
        sharpe_decay = (
            (rolling_sharpe_30d - rolling_sharpe_90d) / abs(rolling_sharpe_90d)
            if rolling_sharpe_90d != 0 else 0
        )

        # Verdicts
        needs_retrain = (
            sharpe_decay < self.SHARPE_DECAY_THRESHOLD
            or rolling_accuracy_30d < self.ACCURACY_MIN
            or len(drifted) > len(psi_scores) * 0.3  # >30% features drifted
        )
        needs_investigation = (
            pred_drift_p < self.KS_PVALUE_THRESHOLD
            or len(drifted) > 0
        )

        if needs_retrain:
            action = "auto_retrain"
        elif needs_investigation:
            action = "alert_and_monitor"
        else:
            action = "no_action"

        return DriftReport(
            model_id=model_id,
            report_date=date.today(),
            feature_psi_scores=psi_scores,
            features_drifted=drifted,
            prediction_distribution_shift=pred_drift_p,
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_sharpe_90d=rolling_sharpe_90d,
            sharpe_decay_pct=sharpe_decay,
            rolling_accuracy_30d=rolling_accuracy_30d,
            needs_retrain=needs_retrain,
            needs_investigation=needs_investigation,
            recommended_action=action,
        )
```

#### Nowe eventy drift

```python
# Dodać do trading_common/events.py:

class ModelDriftDetectedEvent(BaseEvent):
    event_type: EventType = EventType.MODEL_DRIFT_DETECTED
    model_id: str
    drift_type: str             # "feature_drift", "prediction_drift", "performance_decay"
    severity: str               # "warning", "critical"
    recommended_action: str     # "monitor", "retrain", "deactivate"
    source_service: str = "ml-pipeline"

class ModelRetrainedEvent(BaseEvent):
    event_type: EventType = EventType.MODEL_RETRAINED
    model_id: str
    old_sharpe: float
    new_sharpe: float
    retrain_reason: str
    source_service: str = "ml-pipeline"
```

#### Harmonogram monitoringu

| Check | Częstotliwość | Trigger |
|-------|---------------|---------|
| Feature PSI | Daily (po market close) | APScheduler job |
| Prediction drift | Daily | APScheduler job |
| Rolling Sharpe/accuracy | Daily | APScheduler job |
| Full drift report | Weekly (sobota) | APScheduler job |
| Auto-retrain | On trigger (needs_retrain=True) | Event-driven |

---

### B2. Strategy Decay Detection & Auto-Deactivation

**Problem:** Strategia, która miała Sharpe 1.5 rok temu, może mieć teraz 0.3.
Bez automatycznej detekcji degradacji system traci pieniądze.

**Serwis:** `strategy-svc` + `backtest-svc`

```python
# services/strategy/src/core/decay_monitor.py

from dataclasses import dataclass
from datetime import date


@dataclass
class StrategyHealth:
    strategy_name: str
    check_date: date
    # Rolling performance
    sharpe_30d: float
    sharpe_90d: float
    sharpe_180d: float
    # Hit rate
    win_rate_30d: float
    profit_factor_30d: float     # gross profits / gross losses
    # Benchmark comparison
    excess_return_vs_spy_30d: float
    # Verdict
    status: str                  # "active", "probation", "deactivated"
    reason: str


class StrategyDecayMonitor:
    """
    Monitoruje rolling performance strategii i automatycznie deaktywuje
    te które tracą edge.

    3-tier system:
    - ACTIVE: Sharpe > 0.5, profit factor > 1.2 → full allocation
    - PROBATION: Sharpe 0.0-0.5, lub PF 0.8-1.2 → 50% allocation, 30-day review
    - DEACTIVATED: Sharpe < 0.0, lub PF < 0.8, lub 30 dni probation bez poprawy
    """

    ACTIVE_THRESHOLDS = {
        "min_sharpe_90d": 0.50,
        "min_profit_factor": 1.20,
        "min_win_rate": 0.40,
    }

    DEACTIVATION_THRESHOLDS = {
        "max_sharpe_90d": 0.0,   # Negative Sharpe → immediate deactivation
        "max_profit_factor": 0.80,
        "probation_days": 30,
    }

    def evaluate(
        self,
        strategy_name: str,
        sharpe_30d: float,
        sharpe_90d: float,
        sharpe_180d: float,
        win_rate_30d: float,
        profit_factor_30d: float,
        excess_return_vs_spy: float,
        days_on_probation: int = 0,
    ) -> StrategyHealth:

        # Immediate deactivation
        if sharpe_90d < self.DEACTIVATION_THRESHOLDS["max_sharpe_90d"]:
            return StrategyHealth(
                strategy_name=strategy_name, check_date=date.today(),
                sharpe_30d=sharpe_30d, sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d, win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy,
                status="deactivated", reason="negative_sharpe_90d",
            )

        if profit_factor_30d < self.DEACTIVATION_THRESHOLDS["max_profit_factor"]:
            return StrategyHealth(
                strategy_name=strategy_name, check_date=date.today(),
                sharpe_30d=sharpe_30d, sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d, win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy,
                status="deactivated", reason="profit_factor_below_0.8",
            )

        # Probation timeout
        if days_on_probation >= self.DEACTIVATION_THRESHOLDS["probation_days"]:
            return StrategyHealth(
                strategy_name=strategy_name, check_date=date.today(),
                sharpe_30d=sharpe_30d, sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d, win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy,
                status="deactivated", reason="probation_timeout_30d",
            )

        # Active check
        if (sharpe_90d >= self.ACTIVE_THRESHOLDS["min_sharpe_90d"]
                and profit_factor_30d >= self.ACTIVE_THRESHOLDS["min_profit_factor"]
                and win_rate_30d >= self.ACTIVE_THRESHOLDS["min_win_rate"]):
            return StrategyHealth(
                strategy_name=strategy_name, check_date=date.today(),
                sharpe_30d=sharpe_30d, sharpe_90d=sharpe_90d,
                sharpe_180d=sharpe_180d, win_rate_30d=win_rate_30d,
                profit_factor_30d=profit_factor_30d,
                excess_return_vs_spy_30d=excess_return_vs_spy,
                status="active", reason="all_thresholds_met",
            )

        # Probation
        return StrategyHealth(
            strategy_name=strategy_name, check_date=date.today(),
            sharpe_30d=sharpe_30d, sharpe_90d=sharpe_90d,
            sharpe_180d=sharpe_180d, win_rate_30d=win_rate_30d,
            profit_factor_30d=profit_factor_30d,
            excess_return_vs_spy_30d=excess_return_vs_spy,
            status="probation", reason="below_active_thresholds",
        )
```

#### Nowy event

```python
class StrategyStatusChangedEvent(BaseEvent):
    event_type: EventType = EventType.STRATEGY_STATUS_CHANGED
    strategy_name: str
    old_status: str
    new_status: str          # "active", "probation", "deactivated"
    reason: str
    sharpe_90d: float
    profit_factor_30d: float
    source_service: str = "strategy"
```

#### Flow

```
APScheduler (daily, po market close):
  → strategy-svc: oblicz rolling metrics per strategia
  → StrategyDecayMonitor.evaluate()
  → If status changed → PUBLISH StrategyStatusChangedEvent
  → execution-svc: SUBSCRIBE → adjust allocation (50% for probation, 0% for deactivated)
  → notification-svc: SUBSCRIBE → alert (probation=WARNING, deactivated=CRITICAL)
  → signal-aggregator-svc: SUBSCRIBE → remove deactivated from ensemble
```

---

### B3. Adaptive Signal Weights (Meta-Learner)

**Problem:** Signal aggregator ma hardcoded wagi (40/30/30). Optymalne wagi zmieniają się
w zależności od reżimu rynkowego i bieżącej skuteczności sygnałów.

**Serwis:** `signal-aggregator-svc`

```python
# services/signal-aggregator/src/core/adaptive_weights.py

import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SignalPerformance:
    """Tracks rolling performance of each signal source."""
    source_name: str
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=60))
    # Last 60 trading days

    @property
    def hit_rate(self) -> float:
        if not self.recent_returns:
            return 0.5
        wins = sum(1 for r in self.recent_returns if r > 0)
        return wins / len(self.recent_returns)

    @property
    def avg_return(self) -> float:
        if not self.recent_returns:
            return 0.0
        return float(np.mean(list(self.recent_returns)))

    @property
    def information_ratio(self) -> float:
        if len(self.recent_returns) < 10:
            return 0.0
        returns = list(self.recent_returns)
        mean = np.mean(returns)
        std = np.std(returns)
        return float(mean / std) if std > 0 else 0.0


class AdaptiveWeightOptimizer:
    """
    Dynamicznie alokuje wagi między źródłami sygnałów na podstawie
    rolling performance.

    Metoda: Exponentially Weighted Performance (EWP).
    Wagi proporcjonalne do exp(lambda * information_ratio).
    Lambda kontroluje agresywność adaptacji.

    Research: DeMiguel et al. (2009) "Optimal Versus Naive Diversification",
    Kolm et al. (2014) "60 Years of Portfolio Optimization".
    """

    def __init__(
        self,
        signal_sources: list[str],
        lookback_days: int = 60,
        smoothing_lambda: float = 2.0,
        min_weight: float = 0.05,   # Floor: każde źródło min 5%
        max_weight: float = 0.60,   # Cap: żadne źródło nie dominuje
    ):
        self.sources = signal_sources
        self.lookback = lookback_days
        self.lam = smoothing_lambda
        self.min_w = min_weight
        self.max_w = max_weight
        self.trackers: dict[str, SignalPerformance] = {
            s: SignalPerformance(source_name=s) for s in signal_sources
        }

    def record_outcome(self, source: str, daily_return: float) -> None:
        """Rejestruj outcome transakcji przypisanej do danego źródła sygnału."""
        if source in self.trackers:
            self.trackers[source].recent_returns.append(daily_return)

    def compute_weights(self) -> dict[str, float]:
        """
        Returns: {source_name: weight} summing to 1.0.
        """
        # Information ratio per source
        ir_scores = {}
        for name, tracker in self.trackers.items():
            ir = tracker.information_ratio
            ir_scores[name] = ir

        # Exponential weighting
        raw_weights = {}
        for name, ir in ir_scores.items():
            raw_weights[name] = np.exp(self.lam * ir)

        # Normalize
        total = sum(raw_weights.values())
        if total == 0:
            # Equal weights fallback
            n = len(self.sources)
            return {s: 1.0 / n for s in self.sources}

        weights = {name: w / total for name, w in raw_weights.items()}

        # Apply floor and cap
        weights = self._apply_constraints(weights)
        return weights

    def _apply_constraints(self, weights: dict[str, float]) -> dict[str, float]:
        """Apply min/max constraints and re-normalize."""
        constrained = {}
        for name, w in weights.items():
            constrained[name] = max(self.min_w, min(self.max_w, w))

        # Re-normalize to sum to 1.0
        total = sum(constrained.values())
        return {name: w / total for name, w in constrained.items()}
```

---

### B4. Transaction Cost Filter

**Problem:** Model ML może generować sygnały z pozytywnym expected return,
ale po uwzględnieniu kosztów transakcji (spread, commission, slippage, market impact) — negatywny.

**Serwis:** `signal-aggregator-svc` / `risk-mgmt-svc`

```python
# services/signal-aggregator/src/core/cost_filter.py

from dataclasses import dataclass


@dataclass
class TransactionCosts:
    """Realistyczny model kosztów transakcji."""
    commission_per_share: float = 0.005     # $0.005/share (IBKR tiered)
    min_commission: float = 1.00            # $1 minimum
    spread_bps: float = 5.0                 # 5 bps average spread (large cap)
    slippage_bps: float = 5.0              # 5 bps slippage estimate
    market_impact_bps: float = 2.0         # 2 bps market impact (small orders)
    sec_fee_per_million: float = 22.90     # SEC fee (sells only)

    @property
    def total_roundtrip_bps(self) -> float:
        """Total cost of opening and closing a position, in basis points."""
        one_way = self.spread_bps + self.slippage_bps + self.market_impact_bps
        return 2 * one_way  # roundtrip


class CostAwareFilter:
    """
    Odrzuca sygnały, których expected return nie pokrywa kosztów transakcji
    z odpowiednim marginesem bezpieczeństwa.

    Research: Novy-Marx & Velikov (2016) "A Taxonomy of Anomalies
    and Their Trading Costs" — wielu anomalii nie da się eksploatować
    po uwzględnieniu kosztów.
    """

    def __init__(
        self,
        costs: TransactionCosts | None = None,
        min_edge_multiple: float = 2.0,  # Expected return musi być >= 2x kosztów
    ):
        self.costs = costs or TransactionCosts()
        self.min_edge_multiple = min_edge_multiple

    def is_profitable_after_costs(
        self,
        expected_return_bps: float,
        holding_period_days: int = 21,
        market_cap_tier: str = "large",
    ) -> tuple[bool, dict]:
        """
        Returns: (is_viable, details)
        """
        # Adjust costs by market cap
        cost_multiplier = {
            "large": 1.0,
            "mid": 1.5,
            "small": 2.5,
            "micro": 5.0,
        }.get(market_cap_tier, 2.0)

        roundtrip_cost = self.costs.total_roundtrip_bps * cost_multiplier

        # Annualize expected return for comparison
        annualized_trades = 252 / holding_period_days
        annual_cost_bps = roundtrip_cost * annualized_trades

        net_return_bps = expected_return_bps - roundtrip_cost
        edge_multiple = expected_return_bps / roundtrip_cost if roundtrip_cost > 0 else 0

        is_viable = edge_multiple >= self.min_edge_multiple

        return is_viable, {
            "expected_return_bps": expected_return_bps,
            "roundtrip_cost_bps": roundtrip_cost,
            "net_return_bps": net_return_bps,
            "edge_multiple": edge_multiple,
            "annual_cost_drag_bps": annual_cost_bps,
            "market_cap_tier": market_cap_tier,
            "viable": is_viable,
        }
```

---

### B5. Automated Walk-Forward Revalidation

**Problem:** Strategia przeszła walk-forward analysis raz podczas tworzenia,
ale nigdy więcej nie była rewalidowana na nowych danych.

**Serwis:** `backtest-svc`

```python
# services/backtest/src/core/continuous_validation.py

class ContinuousWalkForward:
    """
    Co tydzień (sobota): automatyczny walk-forward re-test aktywnych strategii
    na ostatnich 6 miesiącach danych.

    Porównuje OOS Sharpe z momentu aktywacji vs OOS Sharpe na nowych danych.
    Jeśli degradacja > threshold → StrategyStatusChangedEvent.

    Research: Bailey et al. (2014) "The Deflated Sharpe Ratio" —
    backtested Sharpe jest zawyżony, ciągła walidacja OOS jest konieczna.
    """

    def __init__(
        self,
        oos_window_days: int = 126,  # 6 months
        is_window_days: int = 252,   # 1 year in-sample
        degradation_threshold: float = 0.40,  # 40% spadek OOS Sharpe → probation
    ):
        self.oos_window = oos_window_days
        self.is_window = is_window_days
        self.degradation_threshold = degradation_threshold

    async def revalidate(
        self,
        strategy_name: str,
        original_oos_sharpe: float,
        ohlcv_data: "pd.DataFrame",
        strategy_params: dict,
    ) -> dict:
        """
        Uruchamia walk-forward na najnowszych danych.
        Returns: {current_oos_sharpe, degradation_pct, action}
        """
        # Split: IS = [-378:-126], OOS = [-126:]
        is_data = ohlcv_data.iloc[-(self.is_window + self.oos_window):-self.oos_window]
        oos_data = ohlcv_data.iloc[-self.oos_window:]

        # Run strategy on OOS
        # (tu wywołanie istniejącego backtest engine)
        oos_sharpe = await self._run_backtest(strategy_name, oos_data, strategy_params)

        degradation = (
            (oos_sharpe - original_oos_sharpe) / abs(original_oos_sharpe)
            if original_oos_sharpe != 0 else -1.0
        )

        if degradation < -self.degradation_threshold:
            action = "probation"
        elif oos_sharpe < 0:
            action = "deactivate"
        else:
            action = "active"

        return {
            "strategy_name": strategy_name,
            "original_oos_sharpe": original_oos_sharpe,
            "current_oos_sharpe": oos_sharpe,
            "degradation_pct": degradation,
            "action": action,
        }

    async def _run_backtest(self, name: str, data: "pd.DataFrame", params: dict) -> float:
        """Delegate to existing backtest engine. Override in implementation."""
        raise NotImplementedError
```

---

## CZĘŚĆ C: MAKSYMALIZACJA ZYSKÓW — DODATKOWE EDGE (propozycje dodatkowe)

### C1. Volatility Regime Overlay

```python
# services/feature-engine/src/core/calculators/vol_regime.py

class VolatilityRegimeCalculator:
    """
    VIX regime → position sizing i strategy selection.

    Research: Moreira & Muir (2017) "Volatility-Managed Portfolios" —
    skalowanie pozycji odwrotną zmiennością podnosi Sharpe o 0.2-0.5
    bez dodatkowego ryzyka. Działa na WSZYSTKICH asset classes.

    Implementacja: target volatility approach.
    """

    VIX_REGIMES = {
        "low":      (0,    15),   # Calm market
        "normal":   (15,   20),   # Normal
        "elevated": (20,   30),   # Increased uncertainty
        "high":     (30,   40),   # Fear
        "extreme":  (40,   100),  # Panic
    }

    # Mnożnik ekspozycji per regime
    EXPOSURE_SCALAR = {
        "low":      1.20,  # Lekki overweight (niski VIX = niska vol = większe pozycje)
        "normal":   1.00,
        "elevated": 0.70,
        "high":     0.40,
        "extreme":  0.15,  # Minimal exposure
    }

    def classify_vix(self, vix: float) -> str:
        for regime, (low, high) in self.VIX_REGIMES.items():
            if low <= vix < high:
                return regime
        return "extreme"

    def exposure_scalar(self, vix: float) -> float:
        regime = self.classify_vix(vix)
        return self.EXPOSURE_SCALAR[regime]

    def target_vol_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        realized_vol_annual: float,
        target_vol: float = 0.15,  # 15% annualized target
    ) -> int:
        """
        Volatility-managed position sizing.
        Shares = (portfolio * target_vol) / (price * realized_vol)
        """
        if realized_vol_annual <= 0 or entry_price <= 0:
            return 0
        notional = portfolio_value * target_vol / realized_vol_annual
        shares = int(notional / entry_price)
        # Cap at 5% of portfolio
        max_shares = int(portfolio_value * 0.05 / entry_price)
        return min(shares, max_shares)
```

---

### C2. Earnings Momentum Decay Model

```python
# services/feature-engine/src/core/calculators/earnings_decay.py

class EarningsDecayCalculator:
    """
    Post-Earnings Announcement Drift (PEAD) — najstarsza i najtrwalsza anomalia
    w finansach (Ball & Brown 1968, potwierdzona 2024).

    Kluczowe: drift trwa ~60 dni po earnings, potem maleje.
    Ten kalkulator modeluje exponential decay siły sygnału.

    Research: Bernard & Thomas (1989): 6% excess return w 60 dni po earnings.
    Lucca & Moench (2015): pre-announcement drift (24h przed FOMC).
    """

    def __init__(
        self,
        half_life_days: int = 30,   # Połowa sygnału ginie po 30 dniach
        max_signal_age_days: int = 63,  # Ignoruj sygnały starsze niż kwartał
    ):
        self.half_life = half_life_days
        self.max_age = max_signal_age_days

    def decay_weight(self, days_since_earnings: int) -> float:
        """
        Exponential decay: weight = 0.5^(days / half_life)
        """
        if days_since_earnings > self.max_age:
            return 0.0
        if days_since_earnings <= 0:
            return 1.0

        import math
        return math.pow(0.5, days_since_earnings / self.half_life)

    def surprise_score(
        self,
        actual_eps: float,
        consensus_eps: float,
        historical_std: float,
    ) -> float:
        """
        Standardized Unexpected Earnings (SUE).
        Research: Foster et al. (1984).
        """
        if historical_std <= 0:
            return 0.0
        return (actual_eps - consensus_eps) / historical_std

    def pead_signal(
        self,
        sue_score: float,
        days_since_earnings: int,
    ) -> float:
        """
        Combined: SUE * decay_weight.
        Positive = bullish drift, Negative = bearish drift.
        Range: approximately [-3, +3] for practical purposes.
        """
        weight = self.decay_weight(days_since_earnings)
        return sue_score * weight
```

---

### C3. Cross-Asset Momentum (Inter-Market Signals)

```python
# services/feature-engine/src/core/calculators/cross_asset.py

class CrossAssetMomentumCalculator:
    """
    Sygnały z innych asset classes poprawiają equity predictions.

    Research:
    - Asness et al. (2013) "Value and Momentum Everywhere":
      momentum działa cross-asset (equity, bonds, FX, commodities)
    - Koijen et al. (2018): carry factor cross-asset

    Implementacja: relative strength commodities/bonds/USD vs equity.
    """

    # ETF proxies
    CROSS_ASSET_ETFS = {
        "bonds":       "TLT",   # 20+ Year Treasury
        "commodities": "DBC",   # Commodity Index
        "gold":        "GLD",
        "dollar":      "UUP",   # US Dollar Index
        "vix":         "VXX",   # VIX short-term futures (contrarian)
        "emerging":    "EEM",   # Emerging Markets
        "real_estate": "VNQ",   # REITs
    }

    def compute_cross_asset_scores(
        self,
        asset_returns_60d: dict[str, float],
        spy_return_60d: float,
    ) -> dict:
        """
        Returns cross-asset momentum signals relative to SPY.
        Positive = asset outperforming equity → bullish for that asset class.
        """
        scores = {}
        for asset, ret in asset_returns_60d.items():
            relative_strength = ret - spy_return_60d
            scores[f"{asset}_relative_60d"] = relative_strength

        # Composite risk-on/risk-off score
        risk_on_assets = ["emerging", "commodities"]
        risk_off_assets = ["bonds", "gold", "dollar"]

        risk_on_avg = np.mean([
            asset_returns_60d.get(a, 0) for a in risk_on_assets
        ])
        risk_off_avg = np.mean([
            asset_returns_60d.get(a, 0) for a in risk_off_assets
        ])

        scores["risk_appetite_score"] = risk_on_avg - risk_off_avg
        # Positive = risk-on environment, negative = risk-off

        return scores
```

---

## CZĘŚĆ D: OPERACYJNE UZUPEŁNIENIA CLAUDE.md (podnosi z 6→9)

### D1. Nowe sekcje do dodania w CLAUDE.md

#### Risk management rules (non-negotiable)

```markdown
## Risk rules (non-negotiable)

- Every signal MUST pass through RiskEnvelope before publishing
- No order without stop_loss — enforce in TradingSignal validation
- Circuit breaker events MUST be subscribed by ALL services that generate or execute orders
- Paper trading MUST run minimum 30 days with positive Sharpe before live capital
- Max 5% portfolio per position, max 80% total exposure — never override without human approval
- Daily loss > 5% → automatic trading halt until next day
- Drawdown > 15% → flatten all positions, require human restart
- Every strategy MUST have walk-forward OOS validation before activation
- No strategy goes live without backtested Sharpe > 0.5 on OOS data
```

#### Monitoring requirements

```markdown
## Monitoring requirements (every service)

- ML models: daily drift check (PSI + rolling Sharpe), weekly full report
- Strategies: daily decay check, auto-probation/deactivation
- Portfolio: real-time drawdown tracking, circuit breaker armed 24/7
- Prometheus alerts:
  - drawdown > 8% → WARNING
  - drawdown > 15% → CRITICAL
  - model drift PSI > 0.2 → WARNING
  - strategy Sharpe < 0 (90d) → CRITICAL
  - daily loss > 3% → WARNING
  - order fill rate < 90% → WARNING
```

#### New events to register

```markdown
## Extended event types

Add to EventType enum:
- CIRCUIT_BREAKER_TRIGGERED = "risk.circuit_breaker"
- MODEL_DRIFT_DETECTED = "ml.drift_detected"
- MODEL_RETRAINED = "ml.model_retrained"
- STRATEGY_STATUS_CHANGED = "strategy.status_changed"
- REGIME_CHANGED = "macro.regime_changed"
- FUNDAMENTALS_UPDATED = "fundamentals.updated"
- MACRO_UPDATED = "macro.updated"
- SENTIMENT_UPDATED = "sentiment.updated"
- COMPANY_CLASSIFIED = "company.classified"
- FEATURES_READY = "features.ready"
- SIGNAL_AGGREGATED = "signal.aggregated"
```

---

## CZĘŚĆ E: ZAKTUALIZOWANY HARMONOGRAM IMPLEMENTACJI

| Tydzień | Oryginał | + Supplement |
|---------|----------|--------------|
| 1 | Infra + DevOps | + `RiskEnvelope` w trading-common, nowe eventy (circuit breaker, drift) |
| 2 | market-data-svc | (bez zmian) |
| 3 | feature-engine-svc | + `VolatilityRegimeCalculator`, `CrossAssetMomentumCalculator` |
| 3-4 | (nowe z ml_plan) | fundamental-data, macro-data |
| 4 | strategy + backtest | + `StrategyDecayMonitor` scaffold |
| 5 | company-classifier | (bez zmian) |
| 7-8 | Backtest engine | + `ContinuousWalkForward` |
| 11-12 | Walk-forward | + `CostAwareFilter` w backtest metrics |
| 13-16 | ML pipeline | + `DriftDetector`, MLflow drift alerts |
| 17-18 | Sentiment + HMM | + `EarningsDecayCalculator` (PEAD) |
| 19-20 | risk-mgmt-svc | + `DrawdownAdaptiveSizer`, `RegimeAllocator`, `CircuitBreaker` |
| 19 | signal-aggregator | + `AdaptiveWeightOptimizer`, `CostAwareFilter` |
| 22 | Execution | + circuit breaker subscription enforcement |

---

## OCENA PO UZUPEŁNIENIACH

| Dokument | Przed | Po | Kluczowe zmiany |
|----------|-------|-----|-----------------|
| Plan_Rozwoju | 6/10 | **9/10** | Risk from day 1, circuit breakers, adaptive sizing, regime allocation |
| ml_integration_plan | 8/10 | **9/10** | Drift detection, strategy decay monitor, adaptive weights, cost filter |
| CLAUDE.md | 6/10 | **8/10** | Risk rules, monitoring requirements, extended events |
| **Framework łącznie** | 7/10 | **9/10** | Pełny cykl: generate → validate → execute → monitor → adapt → protect |
