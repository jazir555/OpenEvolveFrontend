"""
Evolutionary Test Data Fixtures

Provides realistic mock data for testing Knowledge Engine integration
with OpenEvolve and LoongFlow evolutionary systems.

Domains Covered:
- Finance (portfolio optimization)
- Trading (high-frequency strategies)
- Science (experimental design)
- Engineering (structural optimization)
- Pharma (drug dosage)
- Web Design (conversion optimization)

Copyright 2026 OpenEvolve
Licensed under Apache License 2.0
"""

from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta
import random


# ============================================================================
# LOONGFLOW TEST DATA
# ============================================================================

def get_loongflow_success_result(
    problem_type: str = "portfolio_optimization",
    success_rate: float = 0.85
) -> Dict[str, Any]:
    """
    Generate successful LoongFlow PES result.

    Args:
        problem_type: Type of problem
        success_rate: Success rate (0.0 to 1.0)

    Returns:
        Complete PES run result dictionary
    """
    return {
        "plan": {
            "strategy": "Use gradient descent with adaptive learning rate",
            "approach": "iterative_refinement",
            "success_rate": success_rate,
            "iterations": 50,
            "reasoning": "Adaptive learning rate allows faster convergence"
        },
        "execution": {
            "early_stops": [15, 25, 35],
            "convergence_rate": 0.95,
            "iterations_to_best": 35,
            "total_evaluations": 120,
            "efficiency_gain": 0.60,
            "time_saved": 180,
        },
        "summary": {
            "insights": "Momentum helps escape local optima. Early stopping saves 60% evaluations.",
            "what_worked": ["adaptive learning rate", "momentum", "early stopping"],
            "what_failed": ["fixed learning rate", "no momentum"],
            "recommendations": ["Use momentum in future runs", "Implement adaptive early stopping"]
        },
        "evolutionary_tree": {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 25,
            "best_path": [0, 2, 5, 8, 15, 22, 35, 48, 62, 78, 95],
            "solutions": [f"sol_{i}" for i in range(100)]
        },
        "best_solution": {
            "code": f"def optimize_{problem_type}():\n    return optimal_result",
            "fitness": 0.95,
            "iteration": 35,
            "improvement": 0.45
        }
    }


def get_loongflow_failure_result(
    problem_type: str = "portfolio_optimization"
) -> Dict[str, Any]:
    """Generate failed LoongFlow PES result."""
    return {
        "plan": {
            "strategy": "Failed strategy",
            "approach": "wrong_direction",
            "success_rate": 0.15,
            "iterations": 100
        },
        "execution": {
            "early_stops": [5],
            "convergence_rate": 0.20,
            "total_evaluations": 200,
            "efficiency_gain": -0.20  # Negative efficiency
        },
        "summary": {
            "insights": "Strategy failed due to wrong approach",
            "what_worked": [],
            "what_failed": ["wrong_direction", "no convergence"],
            "recommendations": ["Try opposite approach"]
        },
        "evolutionary_tree": {
            "generations": 5,
            "avg_branching": 1.0,
            "total_mutations": 5
        },
        "best_solution": {
            "code": f"def failed_{problem_type}():\n    return None",
            "fitness": 0.20,
            "iteration": 5,
            "improvement": -0.30
        }
    }


# ============================================================================
# OPENEVOLVE TEST DATA
# ============================================================================

def get_openevolve_qd_result() -> Dict[str, Any]:
    """Generate OpenEvolve Quality-Diversity result."""
    return {
        "evolution_mode": "qd",
        "iterations": 100,
        "evaluations": 500,
        "best_fitness": 0.85,
        "archive": {
            "size": 50,
            "grid_resolution": [10, 10],
            "feature_dimensions": 2,
            "solutions": [f"qd_sol_{i}" for i in range(50)]
        },
        "population_history": [
            {
                "generation": i,
                "diversity": 0.5 + i * 0.005,
                "best_fitness": 0.3 + i * 0.006
            }
            for i in range(100)
        ],
        "convergence_curve": [0.30, 0.36, 0.42, 0.48, 0.54, 0.60, 0.66, 0.72, 0.78, 0.85],
        "final_score": 0.85,
        "mode": "qd",
        "niches_filled": 45,
        "coverage": 0.90
    }


def get_openevolve_mo_result() -> Dict[str, Any]:
    """Generate OpenEvolve Multi-Objective result."""
    return {
        "evolution_mode": "mo",
        "iterations": 150,
        "evaluations": 750,
        "objectives": ["return", "risk", "liquidity"],
        "pareto_front": {
            "size": 30,
            "solutions": [f"pareto_{i}" for i in range(30)]
        },
        "convergence_curve": [0.25, 0.35, 0.45, 0.55, 0.65, 0.72, 0.78, 0.82],
        "hypervolume": 0.75,
        "final_scores": {
            "return": 0.80,
            "risk": 0.70,
            "liquidity": 0.85
        }
    }


def get_openevolve_adversarial_result() -> Dict[str, Any]:
    """Generate OpenEvolve Adversarial result."""
    return {
        "evolution_mode": "adversarial",
        "iterations": 200,
        "evaluations": 1000,
        "adversarial_rounds": 20,
        "red_team_attacks": 150,
        "defenses_survived": 120,
        "robustness_score": 0.80,
        "best_solution": {
            "code": "def robust_solution():\n    # Adversarially tested\n    pass",
            "fitness": 0.88,
            "attack_survival_rate": 0.80
        },
        "attack_types": ["gradient", "random", "boundary"],
        "vulnerabilities_found": ["edge_case_1", "edge_case_2"]
    }


# ============================================================================
# DOMAIN-SPECIFIC TEST DATA
# ============================================================================

DOMAIN_PROBLEMS = {
    "finance": {
        "description": "Optimize portfolio allocation for maximum return with minimum risk",
        "objectives": ["return", "risk", "liquidity"],
        "constraints": {
            "budget": 1000000,
            "max_position": 0.2,
            "min_diversification": 10
        },
        "evaluation_cost": "high",  # Backtesting is expensive
        "typical_evaluations": 500
    },
    "trading": {
        "description": "Design high-frequency trading strategy with Sharpe ratio > 2.0",
        "objectives": ["sharpe_ratio", "profit_factor", "max_drawdown"],
        "constraints": {
            "hold_time": "1-5min",
            "volume": "<1000",
            "slippage": "<0.1%"
        },
        "evaluation_cost": "high",
        "typical_evaluations": 1000
    },
    "science": {
        "description": "Optimize experimental parameters for chemical reaction yield",
        "objectives": ["yield", "purity", "cost"],
        "constraints": {
            "temperature": "20-100C",
            "time": "<24h",
            "safety": "high"
        },
        "evaluation_cost": "very_high",  # Experiments cost money
        "typical_evaluations": 50
    },
    "engineering": {
        "description": "Design lightweight bridge supporting 50 tons",
        "objectives": ["weight", "strength", "cost"],
        "constraints": {
            "safety_factor": ">2.0",
            "materials": "steel/concrete",
            "span": "100m"
        },
        "evaluation_cost": "medium",
        "typical_evaluations": 200
    },
    "pharma": {
        "description": "Optimize drug dosage for efficacy and minimal side effects",
        "objectives": ["efficacy", "safety", "bioavailability"],
        "constraints": {
            "toxicity": "<0.01",
            "half_life": "6-24h",
            "fda_compliance": True
        },
        "evaluation_cost": "very_high",
        "typical_evaluations": 30
    },
    "web_design": {
        "description": "Optimize landing page for conversion",
        "objectives": ["conversion_rate", "engagement", "load_time"],
        "constraints": {
            "mobile_friendly": True,
            "accessibility": "WCAG_AA",
            "browser_support": "all_major"
        },
        "evaluation_cost": "low",
        "typical_evaluations": 1000
    }
}


def get_domain_problem(domain: str) -> Dict[str, Any]:
    """Get problem definition for specific domain."""
    return DOMAIN_PROBLEMS.get(domain, DOMAIN_PROBLEMS["general"])


# ============================================================================
# TEMPORAL TEST DATA
# ============================================================================

def get_temporal_artifacts(num_points: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    Generate artifacts at different time points.

    Args:
        num_points: Number of time points to generate

    Returns:
        Dict with keys T1, T2, ... Tn
    """
    now = datetime.now(timezone.utc)
    artifacts = {}

    for i in range(num_points):
        days_ago = (num_points - i) * 15
        timestamp = now - timedelta(days=days_ago)

        artifacts[f"T{i+1}"] = {
            "content": f"Strategy at generation {i+1}: " + \
                       ["Simple gradient descent", "Add momentum", "Adaptive learning rate"][i] if i < 3 else \
                       f"Improved strategy v{i+1}",
            "valid_at": timestamp.isoformat(),
            "invalid_at": None,
            "created_at": timestamp.isoformat(),
            "metadata": {
                "success_rate": 0.6 + i * 0.12,
                "generation": i + 1,
                "improvements": ["improvement_1", "improvement_2"][:i+1]
            }
        }

    return artifacts


# ============================================================================
# PERFORMANCE BENCHMARK DATA
# ============================================================================

PERFORMANCE_BENCHMARKS = {
    "query": {
        "target_latency_ms": 100,
        "target_throughput_qps": 500,
        "dataset_sizes": [10, 100, 1000, 10000]
    },
    "storage": {
        "target_latency_ms": 200,
        "target_throughput_wps": 1000,
        "batch_sizes": [1, 10, 100, 1000]
    },
    "extraction": {
        "target_latency_ms": 1000,
        "artifacts_per_run": 5
    },
    "dual_run_analysis": {
        "target_latency_s": 5,
        "comparison_dimensions": 6
    }
}


def get_performance_expectations(operation: str) -> Dict[str, Any]:
    """Get performance expectations for operation."""
    return PERFORMANCE_BENCHMARKS.get(operation, {})


# ============================================================================
# EDGE CASE TEST DATA
# ============================================================================

EDGE_CASES = {
    "empty": {},
    "null_values": {
        "plan": None,
        "execution": None,
        "summary": None,
        "evolutionary_tree": None,
        "best_solution": None
    },
    "extreme_values": {
        "plan": {"success_rate": 1.5},
        "execution": {
            "early_stops": list(range(10000)),
            "efficiency_gain": -10.0
        },
        "summary": {"insights": "A" * 10000},
        "evolutionary_tree": {"generations": -1},
        "best_solution": {"fitness": 2.0}
    },
    "unicode": {
        "plan": {"strategy": "Strategy with emoji 🚀 and unicode 中文"},
        "summary": {"insights": "Multi-language: English, 中文, 日本語, 한국어"},
        "best_solution": {"code": "def test():\n    # Special chars: <>&\"'\n    pass"}
    },
    "malformed": {
        "plan": {"strategy": ["list", "instead", "of", "string"]},
        "execution": {"early_stops": "not_a_list"},
        "summary": {"insights": {"nested": "dict"}}
    }
}


def get_edge_case(case_name: str) -> Dict[str, Any]:
    """Get edge case test data."""
    return EDGE_CASES.get(case_name, {})


# ============================================================================
# GENERATOR FUNCTIONS
# ============================================================================

def generate_random_pes_result(
    success: bool = True,
    domain: str = "general"
) -> Dict[str, Any]:
    """Generate random PES result for testing."""
    if success:
        return get_loongflow_success_result(
            problem_type=domain,
            success_rate=random.uniform(0.6, 0.95)
        )
    else:
        return get_loongflow_failure_result(problem_type=domain)


def generate_multi_run_history(
    num_runs: int = 10,
    domain: str = "general"
) -> List[Dict[str, Any]]:
    """Generate history of multiple evolutionary runs."""
    history = []

    for i in range(num_runs):
        # Gradually improving success rates
        success_rate = 0.5 + (i / num_runs) * 0.4

        result = get_loongflow_success_result(
            problem_type=domain,
            success_rate=success_rate
        )

        # Add timestamp
        result["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(days=num_runs - i)
        ).isoformat()

        history.append(result)

    return history


def generate_comparison_pair(domain: str = "finance") -> tuple:
    """
    Generate paired LoongFlow and OpenEvolve results for comparison.

    Returns:
        (loongflow_result, openevolve_result)
    """
    loongflow = get_loongflow_success_result(problem_type=domain)
    openevolve = get_openevolve_qd_result()

    # Normalize to same domain
    openevolve["domain"] = domain
    loongflow["domain"] = domain

    return loongflow, openevolve


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_artifact_structure(artifact: Dict[str, Any]) -> bool:
    """Validate artifact has required fields."""
    required_fields = [
        "id", "content", "artifact_type", "valid_at",
        "created_at", "source", "metadata", "confidence"
    ]

    return all(field in artifact for field in required_fields)


def validate_pes_result(result: Dict[str, Any]) -> bool:
    """Validate PES result has all phases."""
    required_phases = ["plan", "execution", "summary", "evolutionary_tree", "best_solution"]

    return all(phase in result for phase in required_phases)


def calculate_improvement(old_value: float, new_value: float) -> float:
    """Calculate percentage improvement."""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    # LoongFlow data
    "get_loongflow_success_result",
    "get_loongflow_failure_result",

    # OpenEvolve data
    "get_openevolve_qd_result",
    "get_openevolve_mo_result",
    "get_openevolve_adversarial_result",

    # Domain data
    "DOMAIN_PROBLEMS",
    "get_domain_problem",

    # Temporal data
    "get_temporal_artifacts",

    # Performance data
    "PERFORMANCE_BENCHMARKS",
    "get_performance_expectations",

    # Edge cases
    "EDGE_CASES",
    "get_edge_case",

    # Generators
    "generate_random_pes_result",
    "generate_multi_run_history",
    "generate_comparison_pair",

    # Validators
    "validate_artifact_structure",
    "validate_pes_result",
    "calculate_improvement",
]
