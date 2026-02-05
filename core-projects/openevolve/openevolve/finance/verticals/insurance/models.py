"""
Data models for insurance vertical

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class CreditRating(Enum):
    """Credit rating scale with ordering support"""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"

    def __lt__(self, other):
        """Support comparison for credit quality (AAA is best/worst - higher index = better)"""
        if self.__class__ is not other.__class__:
            return NotImplemented
        # Order enum members by declaration order (AAA is first)
        # For credit quality: BBB (index 3) > AAA (index 0)
        # This way min() returns the WORST rating as expected
        order = list(CreditRating)
        return order.index(self) > order.index(other)

    def __le__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self == other or self < other

    def __gt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return not self <= other

    def __ge__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return not self < other

    @classmethod
    def from_string(cls, rating: str) -> 'CreditRating':
        """Parse rating string"""
        rating_map = {
            "AAA": cls.AAA,
            "AA+": cls.AA, "AA": cls.AA, "AA-": cls.AA,
            "A+": cls.A, "A": cls.A, "A-": cls.A,
            "BBB+": cls.BBB, "BBB": cls.BBB, "BBB-": cls.BBB,
            "BB+": cls.BB, "BB": cls.BB, "BB-": cls.BB,
            "B+": cls.B, "B": cls.B, "B-": cls.B,
        }
        return rating_map.get(rating.upper(), cls.BBB)


@dataclass
class Bond:
    """Bond position"""
    ticker: str
    rating: CreditRating
    par_value: float
    market_value: float
    book_value: float
    duration: float
    convexity: float
    yield_to_maturity: float
    sector: str
    coupon_rate: float
    maturity_date: datetime


@dataclass
class Portfolio:
    """Insurance reserve portfolio"""
    bonds: List[Bond] = field(default_factory=list)
    cash: float = 0.0
    total_value: float = 0.0

    @property
    def duration(self) -> float:
        """Calculate portfolio duration"""
        if not self.bonds or self.total_value == 0:
            return 0.0

        weighted_duration = sum(
            bond.duration * bond.market_value
            for bond in self.bonds
        )
        return weighted_duration / self.total_value

    @property
    def credit_quality(self) -> CreditRating:
        """Get minimum credit quality in portfolio"""
        if not self.bonds:
            return CreditRating.AAA

        return min(bond.rating for bond in self.bonds)


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio evolution"""
    max_duration: float = 7.0
    min_credit_quality: str = "BBB-"
    max_concentration: float = 0.30  # Max 30% in any sector
    min_diversification: int = 20  # Minimum number of bonds
    max_single_bond: float = 0.05  # Max 5% in any single bond
    liquidity_requirement: float = 0.10  # 10% cash or liquid assets


@dataclass
class StressScenario:
    """Stress test scenario"""
    name: str
    description: str
    duration_months: int
    shocks: Dict[str, Any] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    probability: float = 0.01  # 1% annual probability


@dataclass
class StressTestResult:
    """Result of stress test on portfolio"""
    scenario_name: str
    initial_value: float
    final_value: float
    loss_amount: float
    loss_percentage: float
    rbc_ratio_initial: float
    rbc_ratio_final: float
    breaches_rbc: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InsuranceEvolutionResult:
    """Result of insurance reserve portfolio evolution"""
    portfolio: Portfolio
    stress_test_results: Dict[str, StressTestResult]
    min_rbc_ratio: float
    regulatory_compliant: bool
    evolution_iterations: int
    scenarios_tested: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvolutionResult:
    """Result for single scenario evolution"""
    best_portfolio: Portfolio
    best_rbc: float
    all_results: List[tuple]  # (portfolio, result, rbc_ratio, score)


@dataclass
class RBCCalculationResult:
    """RBC calculation result"""
    tac: float  # Total Adjusted Capital
    rbc_required: float  # RBC Required
    rbc_ratio: float  # RBC ratio as percentage
    c0_risk: float  # Affiliate risk
    c1_risk: float  # Fixed income risk
    c2_risk: float  # Equity risk
    c3_risk: float  # Real estate risk
    c4_risk: float  # Off-balance sheet risk
    compliant: bool  # Meets 350% threshold
    details: Dict[str, Any] = field(default_factory=dict)
