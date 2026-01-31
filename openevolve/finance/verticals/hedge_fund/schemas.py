#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Schemas for Hedge Fund Alpha Discovery

Canonical data models for alpha signals, backtests, and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np


class CrisisPeriod(Enum):
    """Major market crisis periods for validation."""
    DOTCOM = "dotcom"  # 2000-2002
    GFC = "gfc"  # 2008-2009
    COVID = "covid"  # 2020
    INFLATION_2022 = "inflation_2022"  # 2022


class AlphaSource(Enum):
    """Source of alpha (why it works)."""
    BEHAVIORAL = "behavioral"  # Investor biases
    RISK_BASED = "risk_based"  # Risk premia
    INFORMATION = "information"  # Information advantage
    STRUCTURAL = "structural"  # Market structure
    COMBINATION = "combination"  # Multiple sources


@dataclass
class CrisisPerformance:
    """Performance during a specific crisis period."""
    crisis: CrisisPeriod
    start_date: datetime
    end_date: datetime
    return_pct: float  # Total return during crisis
    alpha_pct: float  # Risk-adjusted return
    max_drawdown_pct: float
    volatility_pct: float
    survived: bool
    notes: str = ""


@dataclass
class FeatureSet:
    """A set of features used in an alpha signal."""
    features: Dict[str, Any]  # Feature name -> value/computation
    feature_correlations: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features": self.features,
            "feature_correlations": self.feature_correlations,
            "feature_importance": self.feature_importance
        }


@dataclass
class BacktestResult:
    """Results from backtesting an alpha signal."""
    signal_id: str
    universe: str
    start_date: datetime
    end_date: datetime
    include_delisted: bool

    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    alpha: float  # Alpha vs benchmark
    beta: float  # Beta vs benchmark
    tracking_error: float

    # Risk metrics
    max_drawdown: float
    volatility: float
    var_95: float  # Value at risk at 95% confidence

    # Trading costs
    avg_market_cap: float  # Average market cap of holdings
    turnover: float  # Annual turnover rate
    trading_costs: float  # Estimated trading costs

    # Survival metrics
    delisting_rate: float  # Fraction of holdings that delisted
    survivorship_bias: float  # Performance with vs without delisted

    # Crisis performance
    crisis_performance: Dict[str, CrisisPerformance] = field(default_factory=dict)

    # Period-by-period returns
    returns_by_period: Dict[str, float] = field(default_factory=dict)

    # Metadata
    backtest_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "universe": self.universe,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "include_delisted": self.include_delisted,
            "performance": {
                "total_return": self.total_return,
                "annual_return": self.annual_return,
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "information_ratio": self.information_ratio,
                "alpha": self.alpha,
                "beta": self.beta,
                "tracking_error": self.tracking_error
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "var_95": self.var_95
            },
            "trading": {
                "avg_market_cap": self.avg_market_cap,
                "turnover": self.turnover,
                "trading_costs": self.trading_costs
            },
            "survival": {
                "delisting_rate": self.delisting_rate,
                "survivorship_bias": self.survivorship_bias
            },
            "crisis_performance": {
                name: {
                    "crisis": cp.crisis.value,
                    "start": cp.start_date.isoformat(),
                    "end": cp.end_date.isoformat(),
                    "return": cp.return_pct,
                    "alpha": cp.alpha_pct,
                    "max_drawdown": cp.max_drawdown_pct,
                    "survived": cp.survived,
                    "notes": cp.notes
                }
                for name, cp in self.crisis_performance.items()
            },
            "returns_by_period": self.returns_by_period,
            "backtest_timestamp": self.backtest_timestamp.isoformat()
        }


@dataclass
class AlphaSignal:
    """An alpha signal for trading."""
    signal_id: str
    name: str
    description: str
    formula: str

    # Features
    features: FeatureSet

    # Alpha source
    alpha_source: AlphaSource
    rationale: str  # Why this should work

    # Performance metrics
    backtest: BacktestResult
    sharpe_ratio: float
    information_ratio: float

    # Crisis performance
    crisis_performance: Dict[str, CrisisPerformance]

    # Feature correlations
    feature_correlation: Dict[str, float]

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "name": self.name,
            "description": self.description,
            "formula": self.formula,
            "alpha_source": self.alpha_source.value,
            "rationale": self.rationale,
            "features": self.features.to_dict(),
            "performance": {
                "sharpe_ratio": self.sharpe_ratio,
                "information_ratio": self.information_ratio
            },
            "crisis_performance": {
                name: {
                    "crisis": cp.crisis.value,
                    "return": cp.return_pct,
                    "alpha": cp.alpha_pct,
                    "survived": cp.survived
                }
                for name, cp in self.crisis_performance.items()
            },
            "feature_correlation": self.feature_correlation,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }


@dataclass
class FeatureHypothesis:
    """A hypothesis about a feature that might generate alpha."""
    hypothesis_id: str
    feature_name: str
    feature_definition: str  # Formula or computation
    alpha_source: AlphaSource
    rationale: str  # Why this should work
    crisis_resistance: str  # Why it might survive crises
    failure_modes: List[str]  # Potential ways it could fail
    expected_correlation: float  # Expected correlation with returns
    confidence: float  # Confidence level (0-1)


@dataclass
class SignalConstraints:
    """Constraints for alpha signal discovery."""
    min_market_cap: float  # Minimum market cap (e.g., 500_000_000)
    max_turnover: float  # Maximum annual turnover (e.g., 50 for 50%)
    crisis_periods: List[str]  # Crisis periods to survive
    min_sharpe: float = 1.0  # Minimum Sharpe ratio
    min_information_ratio: float = 0.5  # Minimum IR
    max_delisting_rate: float = 0.01  # Max 1% delisting rate
    max_drawdown: float = -0.30  # Max 30% drawdown
    sector_neutral: bool = False  # Sector-neutral constraints
    beta_neutral: bool = False  # Beta-neutral constraints


@dataclass
class SignalEvolutionResult:
    """Result from evolving a single signal hypothesis."""
    hypothesis: FeatureHypothesis
    best_signal: Optional[AlphaSignal]
    all_valid_signals: List[Dict[str, Any]]
    survival_rate: float  # Fraction of variants that survived

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis": {
                "id": self.hypothesis.hypothesis_id,
                "feature": self.hypothesis.feature_name,
                "definition": self.hypothesis.feature_definition,
                "alpha_source": self.hypothesis.alpha_source.value
            },
            "best_signal": self.best_signal.to_dict() if self.best_signal else None,
            "num_valid_signals": len(self.all_valid_signals),
            "survival_rate": self.survival_rate
        }


@dataclass
class AlphaDiscoveryResult:
    """Result from alpha signal discovery process."""
    surviving_signals: List[AlphaSignal]
    ensemble_signal: "EnsembleSignal"
    feature_importance: Dict[str, float]
    crisis_performance: Dict[str, Dict[str, CrisisPerformance]]

    # Discovery metadata
    total_hypotheses_tested: int
    total_variants_tested: int
    overall_survival_rate: float
    discovery_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "surviving_signals": [s.to_dict() for s in self.surviving_signals],
            "ensemble_signal": self.ensemble_signal.to_dict(),
            "feature_importance": self.feature_importance,
            "crisis_performance_summary": {
                crisis: {
                    "avg_return": np.mean([cp.return_pct for cp in performances]),
                    "avg_alpha": np.mean([cp.alpha_pct for cp in performances]),
                    "survival_rate": sum(1 for cp in performances if cp.survived) / len(performances)
                }
                for crisis, performances in self.crisis_performance.items()
            },
            "discovery_metadata": {
                "total_hypotheses_tested": self.total_hypotheses_tested,
                "total_variants_tested": self.total_variants_tested,
                "overall_survival_rate": self.overall_survival_rate,
                "num_surviving_signals": len(self.surviving_signals),
                "discovery_timestamp": self.discovery_timestamp.isoformat()
            }
        }


@dataclass
class EnsembleSignal:
    """Ensemble of multiple alpha signals."""
    ensemble_id: str
    name: str
    description: str
    signals: List[AlphaSignal]
    weights: Dict[str, float]  # signal_id -> weight
    combination_method: str

    # Performance metrics
    backtest: BacktestResult
    sharpe_ratio: float
    information_ratio: float

    # Diversification metrics
    signal_correlations: Dict[Tuple[str, str], float]  # Pairwise correlations
    diversification_ratio: float  # How diversified the ensemble is

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_id": self.ensemble_id,
            "name": self.name,
            "description": self.description,
            "num_signals": len(self.signals),
            "weights": self.weights,
            "combination_method": self.combination_method,
            "performance": {
                "sharpe_ratio": self.sharpe_ratio,
                "information_ratio": self.information_ratio
            },
            "diversification": {
                "diversification_ratio": self.diversification_ratio,
                "signal_correlations": {
                    f"{s1}_{s2}": corr
                    for (s1, s2), corr in self.signal_correlations.items()
                }
            },
            "signal_ids": [s.signal_id for s in self.signals]
        }
