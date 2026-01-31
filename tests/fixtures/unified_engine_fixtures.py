"""
Test Fixtures for Unified Evolution Engine Integration Tests

Provides reusable fixtures, mock data, and helper functions for testing
the unified evolutionary optimization pipeline.

Copyright 2026 OpenEvolve
Licensed under Apache License 2.0
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, field
import json
import uuid
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


# ============================================================================
# DOMAIN CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def domain_configurations():
    """
    Domain-specific configurations for all 6 supported domains.

    Returns pre-configured settings for:
    - Finance: PES mode with risk management
    - Trading: PES mode with signal processing
    - Science: PES mode with experimental design
    - Engineering: Adversarial mode with safety constraints
    - Pharma: Adversarial mode with molecular constraints
    - Web Design: PES mode with conversion optimization
    """
    return {
        "finance": {
            "default_mode": "pes",
            "max_evaluations": 50,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["return", "risk", "liquidity"],
            "constraints": {
                "max_position_size": 0.1,
                "min_diversification": 0.7,
                "max_drawdown": 0.2
            },
            "evaluation_cost": "high",
            "time_per_eval": 300,  # 5 minutes for backtest
            "success_threshold": 0.7
        },
        "trading": {
            "default_mode": "pes",
            "max_evaluations": 80,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["sharpe_ratio", "max_drawdown", "win_rate"],
            "constraints": {
                "max_leverage": 2.0,
                "min_win_rate": 0.55,
                "max_slippage": 0.01
            },
            "evaluation_cost": "high",
            "time_per_eval": 120,  # 2 minutes for backtest
            "success_threshold": 0.75
        },
        "science": {
            "default_mode": "pes",
            "max_evaluations": 20,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["statistical_power", "effect_size", "cost"],
            "constraints": {
                "max_experiments": 20,
                "min_power": 0.8,
                "max_budget": 50000
            },
            "evaluation_cost": "very_high",
            "time_per_eval": 3600,  # 1 hour per experiment
            "success_threshold": 0.8
        },
        "engineering": {
            "default_mode": "adversarial",
            "max_evaluations": 100,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["weight", "strength", "cost"],
            "constraints": {
                "min_safety_factor": 1.5,
                "max_weight": 1000,
                "max_stress": 0.8
            },
            "evaluation_cost": "medium",
            "time_per_eval": 60,  # 1 minute for FEA simulation
            "success_threshold": 0.85,
            "adversarial_rounds": 20
        },
        "pharma": {
            "default_mode": "adversarial",
            "max_evaluations": 100,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["binding_affinity", "selectivity", "toxicity"],
            "constraints": {
                "min_affinity": 0.8,
                "max_toxicity": 0.3,
                "lipinski_rules": True
            },
            "evaluation_cost": "very-high",
            "time_per_eval": 1800,  # 30 minutes for docking
            "success_threshold": 0.9,
            "adversarial_rounds": 25
        },
        "web_design": {
            "default_mode": "pes",
            "max_evaluations": 200,
            "enable_planning": True,
            "enable_memory": True,
            "objectives": ["conversion_rate", "engagement", "load_time"],
            "constraints": {
                "max_load_time": 3.0,
                "min_accessibility": 0.9,
                "mobile_friendly": True
            },
            "evaluation_cost": "low",
            "time_per_eval": 5,  # 5 seconds for A/B test simulation
            "success_threshold": 0.8
        }
    }


@pytest.fixture
def problem_templates():
    """
    Pre-defined problem templates for each domain.

    Provides standardized problem descriptions with embedded
    domain-specific vocabulary and constraints.
    """
    return {
        "finance": [
            "Optimize portfolio allocation to maximize Sharpe ratio with minimum risk",
            "Design rebalancing strategy for tax-efficient portfolio management",
            "Develop risk parity strategy with leverage constraints"
        ],
        "trading": [
            "Develop momentum-based trading strategy with entry/exit signals",
            "Create mean reversion strategy for equity pairs trading",
            "Design machine learning strategy for cryptocurrency trading"
        ],
        "science": [
            "Design experiment to maximize statistical power with limited budget",
            "Optimize clinical trial parameters for maximum efficacy detection",
            "Create sampling strategy for environmental monitoring study"
        ],
        "engineering": [
            "Minimize structural weight while maintaining safety factor of 1.5",
            "Optimize heat sink design for maximum thermal dissipation",
            "Design suspension system balancing comfort and performance"
        ],
        "pharma": [
            "Optimize molecular structure for maximum binding affinity to target protein",
            "Design drug formulation with improved bioavailability and minimal toxicity",
            "Create screening strategy for identifying novel kinase inhibitors"
        ],
        "web_design": [
            "Optimize landing page layout to maximize conversion rate",
            "Design checkout flow to minimize cart abandonment",
            "Create responsive design balancing aesthetics and performance"
        ]
    }


# ============================================================================
# STRATEGY SELECTOR FIXTURES
# ============================================================================

@pytest.fixture
def strategy_selector_test_cases():
    """
    Test cases for strategy selection logic.

    Covers all decision paths in the strategy selection algorithm.
    """
    return [
        {
            "name": "expensive_evaluations",
            "problem": {
                "description": "Optimize with expensive backtesting",
                "estimated_time_per_eval": 300,
                "estimated_cost_per_eval": 100
            },
            "domain": "finance",
            "expected_mode": "pes",
            "min_confidence": 0.8
        },
        {
            "name": "multi_objective",
            "problem": {
                "description": "Optimize cost and quality",
                "objectives": ["cost", "quality", "time"]
            },
            "domain": "general",
            "expected_mode": "mo",
            "min_confidence": 0.8
        },
        {
            "name": "diversity_needed",
            "problem": {
                "description": "Explore diverse novel solutions",
                "require_diversity": True
            },
            "domain": "general",
            "expected_mode": "qd",
            "min_confidence": 0.7
        },
        {
            "name": "safety_critical",
            "problem": {
                "description": "Design safety-critical system",
                "safety_critical": True
            },
            "domain": "engineering",
            "expected_mode": "adversarial",
            "min_confidence": 0.8
        },
        {
            "name": "real_time_constraints",
            "problem": {
                "description": "Optimize with real-time requirements",
                "constraints": {"real_time": True}
            },
            "domain": "trading",
            "expected_mode": "pes",
            "min_confidence": 0.7
        },
        {
            "name": "default_fallback",
            "problem": {
                "description": "Simple optimization problem"
            },
            "domain": "general",
            "expected_mode": "pes",  # PES is default
            "min_confidence": 0.5
        }
    ]


# ============================================================================
# EVOLUTION RESULT FIXTURES
# ============================================================================

@pytest.fixture
def mock_evolution_results():
    """
    Mock evolution results for different modes.

    Provides realistic mock data with:
    - PES: 60% fewer evaluations, high efficiency
    - QD: High diversity, archive metrics
    - MO: Pareto front with multiple objectives
    - Adversarial: Robustness metrics
    - Standard: Baseline performance
    """
    return {
        "pes": {
            "mode": "pes",
            "best_solution": "def pes_optimized():\n    # PES-generated solution\n    return x * 2 + momentum",
            "fitness": 0.95,
            "evaluations": 30,  # 60% fewer than baseline
            "total_time": 45.0,
            "iterations": 3,
            "planning_success": 0.9,
            "efficiency_gain": 0.60,
            "artifacts": [
                {"type": "pes_pattern", "phase": "planning", "success": 0.95},
                {"type": "pes_pattern", "phase": "execution", "early_stops": [10, 20]},
                {"type": "efficiency", "gain": 0.60, "time_saved": 180}
            ]
        },
        "qd": {
            "mode": "qd",
            "best_solution": "def qd_diverse():\n    # QD-generated diverse solution\n    return archive[best_niche]",
            "fitness": 0.85,
            "evaluations": 100,
            "total_time": 60.0,
            "archive_size": 50,
            "niches_filled": 45,
            "coverage": 0.90,
            "artifacts": [
                {"type": "archive", "size": 50, "grid_resolution": [10, 10]},
                {"type": "niche", "filled": 45, "total": 50},
                {"type": "diversity", "score": 0.92}
            ]
        },
        "mo": {
            "mode": "mo",
            "best_solution": "def mo_pareto():\n    # MO-generated Pareto solution\n    return select_from_pareto()",
            "fitness": 0.88,
            "evaluations": 120,
            "total_time": 90.0,
            "pareto_front_size": 15,
            "artifacts": [
                {"type": "pareto_front", "size": 15},
                {"type": "objective_1", "name": "cost", "value": 0.75},
                {"type": "objective_2", "name": "quality", "value": 0.92}
            ],
            "pareto_front": [
                {"fitness": 0.88, "objectives": {"cost": 0.75, "quality": 0.92}},
                {"fitness": 0.85, "objectives": {"cost": 0.82, "quality": 0.95}},
                {"fitness": 0.82, "objectives": {"cost": 0.90, "quality": 0.98}}
            ]
        },
        "adversarial": {
            "mode": "adversarial",
            "best_solution": "def adversarial_robust():\n    # Adversarial-hardened solution\n    return robust_solution()",
            "fitness": 0.82,
            "evaluations": 150,
            "total_time": 120.0,
            "adversarial_rounds": 20,
            "attacks_survived": 18,
            "artifacts": [
                {"type": "attack", "round": 5, "survived": True, "severity": 0.8},
                {"type": "attack", "round": 10, "survived": True, "severity": 0.9},
                {"type": "robustness", "score": 0.85, "attacks_survived": 18}
            ]
        },
        "standard": {
            "mode": "standard",
            "best_solution": "def standard_baseline():\n    # Standard evolution solution\n    return x * 1.5",
            "fitness": 0.75,
            "evaluations": 200,  # Baseline
            "total_time": 90.0,
            "iterations": 50,
            "artifacts": [
                {"type": "population", "size": 100},
                {"type": "generations", "count": 50}
            ]
        }
    }


# ============================================================================
# GAUNTLET FIXTURES
# ============================================================================

@pytest.fixture
def gauntlet_round_configs():
    """
    Gauntlet round configurations for all 3 rounds.

    Defines scoring thresholds, weights, and evaluation criteria.
    """
    return {
        "round_1": {
            "name": "loongflow_ai_eval",
            "type": "automated",
            "weight": 0.2,
            "min_score": 0.5,
            "max_attempts": 1,
            "timeout": 60,
            "description": "Quick AI evaluation screen"
        },
        "round_2": {
            "name": "red_team",
            "type": "adversarial",
            "weight": 0.3,
            "min_score": 0.7,
            "max_attempts": 3,
            "timeout": 180,
            "description": "Adversarial attack testing"
        },
        "round_3": {
            "name": "gold_team",
            "type": "consensus",
            "weight": 0.5,
            "min_score": 0.9,
            "max_attempts": 2,
            "timeout": 300,
            "description": "Multi-judge consensus verification"
        }
    }


@pytest.fixture
def gauntlet_test_scenarios():
    """
    Gauntlet test scenarios with different solution qualities.

    Provides:
    - Excellent solution: Passes all rounds
    - Good solution: Passes rounds 1-2, marginal on round 3
    - Moderate solution: Passes round 1, fails round 2
    - Poor solution: Fails round 1 (early termination)
    """
    return {
        "excellent": {
            "solution": {
                "code": "def excellent_solution():\n    # Optimized approach\n    # Comprehensive error handling\n    # Well-documented\n    return optimal_result",
                "quality": "excellent",
                "metrics": {"correctness": 0.98, "robustness": 0.95, "efficiency": 0.92}
            },
            "expected_outcome": {
                "passed": True,
                "final_score": 0.92,
                "rounds_completed": 3,
                "round_scores": [0.90, 0.85, 0.95]
            }
        },
        "good": {
            "solution": {
                "code": "def good_solution():\n    # Solid approach\n    # Some error handling\n    return good_result",
                "quality": "good",
                "metrics": {"correctness": 0.90, "robustness": 0.85, "efficiency": 0.88}
            },
            "expected_outcome": {
                "passed": True,
                "final_score": 0.85,
                "rounds_completed": 3,
                "round_scores": [0.85, 0.80, 0.88]
            }
        },
        "moderate": {
            "solution": {
                "code": "def moderate_solution():\n    # Decent approach\n    # Limited error handling\n    return moderate_result",
                "quality": "moderate",
                "metrics": {"correctness": 0.75, "robustness": 0.70, "efficiency": 0.80}
            },
            "expected_outcome": {
                "passed": False,
                "final_score": 0.75,
                "rounds_completed": 3,
                "round_scores": [0.75, 0.70, 0.65],
                "failed_round": 3
            }
        },
        "poor": {
            "solution": {
                "code": "def poor_solution():\n    # Basic approach\n    # No error handling\n    # May have bugs",
                "quality": "poor",
                "metrics": {"correctness": 0.50, "robustness": 0.40, "efficiency": 0.60}
            },
            "expected_outcome": {
                "passed": False,
                "final_score": 0.40,
                "rounds_completed": 1,
                "round_scores": [0.40],
                "failed_round": 1,
                "early_termination": True
            }
        }
    }


# ============================================================================
# KNOWLEDGE ENGINE FIXTURES
# ============================================================================

@pytest.fixture
def knowledge_artifacts():
    """
    Sample knowledge artifacts from past evolutionary runs.

    Includes artifacts from:
    - PES runs with efficiency patterns
    - QD runs with diversity patterns
    - MO runs with Pareto patterns
    - Cross-domain patterns
    """
    return {
        "pes_artifacts": [
            {
                "run_id": "pes_finance_001",
                "domain": "finance",
                "mode": "pes",
                "patterns": [
                    "Momentum helps escape local optima",
                    "Early stopping after 30 evaluations is optimal",
                    "Adaptive learning rate improves convergence"
                ],
                "performance": {
                    "evaluations": 30,
                    "efficiency_gain": 0.60,
                    "fitness": 0.95
                },
                "timestamp": datetime.now(timezone.utc) - timedelta(days=1)
            },
            {
                "run_id": "pes_science_001",
                "domain": "science",
                "mode": "pes",
                "patterns": [
                    "Sequential experimental design reduces cost",
                    "Power analysis guides sample size optimization"
                ],
                "performance": {
                    "evaluations": 12,
                    "efficiency_gain": 0.70,
                    "fitness": 0.88
                },
                "timestamp": datetime.now(timezone.utc) - timedelta(days=2)
            }
        ],
        "qd_artifacts": [
            {
                "run_id": "qd_general_001",
                "domain": "general",
                "mode": "qd",
                "patterns": [
                    "Grid resolution of 10 provides good coverage",
                    "Feature selection critical for diversity"
                ],
                "performance": {
                    "archive_size": 50,
                    "coverage": 0.90,
                    "niches_filled": 45
                },
                "timestamp": datetime.now(timezone.utc) - timedelta(days=3)
            }
        ],
        "cross_domain_patterns": [
            {
                "pattern_id": "momentum_optimization",
                "domains": ["finance", "trading", "engineering"],
                "description": "Momentum-based improvements escape local optima",
                "success_rate": 0.85,
                "applicability": "gradient-based optimization"
            },
            {
                "pattern_id": "early_stopping",
                "domains": ["finance", "science", "trading"],
                "description": "Early stopping saves 60% of evaluations",
                "success_rate": 0.90,
                "applicability": "expensive evaluations"
            }
        ]
    }


@pytest.fixture
def strategy_recommendations():
    """
    Sample strategy recommendations from knowledge engine.

    Provides AI-generated recommendations based on
    historical performance across domains.
    """
    return {
        "finance": {
            "recommended_strategy": "pes",
            "confidence": 0.92,
            "expected_improvement": "60%",
            "reason": "Historical data shows PES reduces backtesting cost by 60%",
            "config": {
                "enable_planning": True,
                "enable_memory": True,
                "max_evaluations": 50
            },
            "expected_evaluations": 30,
            "expected_fitness": 0.92
        },
        "science": {
            "recommended_strategy": "pes",
            "confidence": 0.95,
            "expected_improvement": "70%",
            "reason": "Experimental design benefits from planning phase",
            "config": {
                "enable_planning": True,
                "enable_memory": True,
                "max_evaluations": 20
            },
            "expected_evaluations": 12,
            "expected_fitness": 0.88
        },
        "engineering": {
            "recommended_strategy": "adversarial",
            "confidence": 0.88,
            "expected_improvement": "40%",
            "reason": "Safety-critical systems require adversarial testing",
            "config": {
                "adversarial_rounds": 20,
                "enable_planning": True
            },
            "expected_evaluations": 150,
            "expected_fitness": 0.85
        },
        "general": {
            "recommended_strategy": "pes",
            "confidence": 0.80,
            "expected_improvement": "50%",
            "reason": "PES provides best general performance across benchmarks",
            "config": {
                "enable_planning": True,
                "enable_memory": True
            },
            "expected_evaluations": 50,
            "expected_fitness": 0.85
        }
    }


# ============================================================================
# PERFORMANCE BENCHMARK FIXTURES
# ============================================================================

@pytest.fixture
def performance_benchmarks():
    """
    Performance benchmarks for all domains.

    Defines target metrics for:
    - Maximum execution time
    - Maximum evaluations
    - Minimum fitness
    - Efficiency improvements
    """
    return {
        "general": {
            "target_time": 60,  # seconds
            "target_evals": 100,
            "min_fitness": 0.7,
            "efficiency_improvement": 0.40  # 40% vs baseline
        },
        "finance": {
            "target_time": 600,  # 10 minutes (backtests are expensive)
            "target_evals": 50,
            "min_fitness": 0.7,
            "efficiency_improvement": 0.60  # 60% vs baseline
        },
        "trading": {
            "target_time": 300,  # 5 minutes
            "target_evals": 80,
            "min_fitness": 0.75,
            "efficiency_improvement": 0.50
        },
        "science": {
            "target_time": 900,  # 15 minutes (experiments are very expensive)
            "target_evals": 20,
            "min_fitness": 0.8,
            "efficiency_improvement": 0.70  # 70% vs baseline
        },
        "engineering": {
            "target_time": 600,  # 10 minutes
            "target_evals": 100,
            "min_fitness": 0.8,
            "efficiency_improvement": 0.30
        },
        "pharma": {
            "target_time": 600,  # 10 minutes
            "target_evals": 100,
            "min_fitness": 0.85,
            "efficiency_improvement": 0.30
        },
        "web_design": {
            "target_time": 120,  # 2 minutes
            "target_evals": 200,
            "min_fitness": 0.75,
            "efficiency_improvement": 0.30
        }
    }


# ============================================================================
# MOCK OBJECTS HELPERS
# ============================================================================

@pytest.fixture
def create_mock_strategy_selector():
    """
    Factory function to create mock strategy selectors.

    Returns a function that creates configured mock selectors.
    """
    def _create_selector(responses: Optional[Dict] = None):
        """Create a mock strategy selector with custom responses."""
        selector = Mock()

        async def select_strategy(problem, domain, constraints):
            if responses and domain in responses:
                return responses[domain]
            else:
                # Default behavior
                return Mock(
                    mode="pes",
                    confidence=0.8,
                    reason="Default PES strategy"
                )

        selector.select_strategy = select_strategy
        return selector

    return _create_selector


@pytest.fixture
def create_mock_evolution_engine():
    """
    Factory function to create mock evolution engines.

    Returns a function that creates configured mock engines.
    """
    def _create_engine(result_mode: str = "pes"):
        """Create a mock evolution engine with specific result mode."""
        engine = Mock()

        async def run_evolution(problem, config, mode):
            await asyncio.sleep(0.01)  # Simulate work
            return Mock(
                best_solution=f"def {mode}_solution(): return optimal",
                fitness=0.85,
                evaluations=50 if mode == "pes" else 100,
                total_time=45.0,
                strategy_used=Mock(mode=mode, confidence=0.8, reason=f"Test {mode}"),
                evolution_artifacts=[{"type": "test", "mode": mode}]
            )

        engine.run_evolution = run_evolution
        return engine

    return _create_engine


# ============================================================================
# TEST HELPERS
# ============================================================================

@pytest.fixture
def verify_non_dominated():
    """
    Helper function to verify Pareto optimality.

    Returns a function that checks if a set of solutions
    forms a valid non-dominated Pareto front.
    """
    def _verify(solutions: List[Dict]) -> bool:
        """
        Verify that all solutions in the front are non-dominated.

        A solution is dominated if another solution is better in
        at least one objective and not worse in any objective.
        """
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i == j:
                    continue

                obj1 = sol1.get("objectives", {})
                obj2 = sol2.get("objectives", {})

                # Check if sol2 dominates sol1
                dominates = True
                for key in obj1.keys():
                    if obj2.get(key, 0) < obj1.get(key, 0):
                        dominates = False
                        break

                if dominates:
                    return False  # Found dominated solution

        return True

    return _verify


@pytest.fixture
def calculate_efficiency_gain():
    """
    Helper function to calculate efficiency improvements.

    Returns a function that computes the percentage reduction
    in evaluations compared to baseline.
    """
    def _calculate(baseline_evals: int, optimized_evals: int) -> float:
        """Calculate efficiency gain as percentage."""
        if baseline_evals == 0:
            return 0.0
        return (baseline_evals - optimized_evals) / baseline_evals

    return _calculate


# ============================================================================
# EXPORT ALL FIXTURES
# ============================================================================

__all__ = [
    # Domain configurations
    "domain_configurations",
    "problem_templates",

    # Strategy selector fixtures
    "strategy_selector_test_cases",

    # Evolution results
    "mock_evolution_results",

    # Gauntlet fixtures
    "gauntlet_round_configs",
    "gauntlet_test_scenarios",

    # Knowledge engine fixtures
    "knowledge_artifacts",
    "strategy_recommendations",

    # Performance benchmarks
    "performance_benchmarks",

    # Mock factories
    "create_mock_strategy_selector",
    "create_mock_evolution_engine",

    # Test helpers
    "verify_non_dominated",
    "calculate_efficiency_gain"
]
