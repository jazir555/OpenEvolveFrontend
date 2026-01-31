"""
Corporate Treasury Vertical
Liquidity management and crisis survival strategies

This module provides evolutionary optimization for corporate treasury liquidity management,
focusing on surviving liquidity crises while minimizing the cost of holding liquidity.

Main Components:
- LiquidityCrisisEvolver: Evolves liquidity management strategies
- LiquidityCalculator: Calculates liquidity metrics and costs
- LiquidityScenarioGenerator: Generates realistic stress scenarios

Author: AI Architecture Team
Date: 2026-01-30
"""

from .liquidity_evolver import (
    LiquidityCrisisEvolver,
    LiquidityEvolutionResult,
    LiquidityAllocation,
    LiquidityConstraints,
    CashFlowProfile,
    LiquidityScenario,
    LiquiditySimulationResult
)

from .liquidity_calculator import (
    LiquidityCalculator,
    LiquidityMetrics
)

from .scenario_generator import (
    LiquidityScenarioGenerator,
    ScenarioType
)

__all__ = [
    # Main evolver
    'LiquidityCrisisEvolver',
    'LiquidityEvolutionResult',

    # Data models
    'LiquidityAllocation',
    'LiquidityConstraints',
    'CashFlowProfile',
    'LiquidityScenario',
    'LiquiditySimulationResult',

    # Calculator
    'LiquidityCalculator',
    'LiquidityMetrics',

    # Scenario generator
    'LiquidityScenarioGenerator',
    'ScenarioType'
]

__version__ = '1.0.0'
