"""
Insurance Vertical - LoongFlow-OpenEvolve Finance Platform

Insurance companies need portfolio strategies that survive regulatory stress tests
while maintaining Risk-Based Capital (RBC) ratios through crises.

Components:
- InsuranceReserveEvolver: Evolves bond portfolios surviving stress tests
- RBCCalculator: Calculates Risk-Based Capital per NAIC standards
- StressScenarioGenerator: Generates regulatory stress scenarios

Author: AI Architecture Team
Date: 2026-01-30
"""

from .reserve_evolver import (
    InsuranceReserveEvolver,
    InsuranceEvolutionResult,
    PortfolioConstraints,
    StressScenario,
    ScenarioEvolutionResult
)

from .rbc_calculator import (
    RBCCalculator,
    RBCCalculationResult
)

from .stress_generator import (
    StressScenarioGenerator,
    HistoricalCrises,
    RateShockScenario
)

from .models import (
    Portfolio,
    Bond,
    CreditRating
)

__all__ = [
    # Main evolver
    'InsuranceReserveEvolver',
    'InsuranceEvolutionResult',

    # RBC calculation
    'RBCCalculator',
    'RBCCalculationResult',

    # Stress scenarios
    'StressScenarioGenerator',
    'StressScenario',
    'HistoricalCrises',
    'RateShockScenario',

    # Data models
    'PortfolioConstraints',
    'ScenarioEvolutionResult',
    'Portfolio',
    'Bond',
    'CreditRating'
]
