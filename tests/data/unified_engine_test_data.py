"""
Test Data for Unified Evolution Engine Integration Tests

Comprehensive collection of test data including:
- Sample problems for all domains
- Expected results for validation
- Cross-domain knowledge transfer examples
- Performance benchmark data

Copyright 2026 OpenEvolve
Licensed under Apache License 2.0
"""

from typing import Dict, List, Any
from datetime import datetime, timezone, timedelta


# ============================================================================
# SAMPLE PROBLEMS BY DOMAIN
# ============================================================================

SAMPLE_PROBLEMS = {
    "general": {
        "simple": "Maximize f(x) = x^2 where x in [0, 10]",
        "multi_objective": "Minimize cost and maximize quality simultaneously",
        "constrained": "Maximize f(x, y) = x*y subject to x + y <= 10"
    },
    "finance": {
        "portfolio_optimization": "Optimize portfolio allocation for maximum Sharpe ratio with minimum risk",
        "risk_management": "Design risk parity strategy with leverage constraints",
        "trading_strategy": "Develop momentum-based trading strategy with entry/exit signals"
    },
    "trading": {
        "strategy_development": "Create algorithmic trading strategy for equity markets",
        "signal_processing": "Design signal processing pipeline for market data",
        "backtesting": "Optimize trading parameters using historical backtesting"
    },
    "science": {
        "experimental_design": "Design experiment to maximize statistical power with limited budget",
        "sampling_strategy": "Create optimal sampling strategy for environmental study",
        "parameter_estimation": "Estimate model parameters using Bayesian inference"
    },
    "engineering": {
        "structural_optimization": "Minimize structural weight while maintaining safety factor",
        "thermal_design": "Optimize heat sink design for maximum thermal dissipation",
        "control_system": "Design PID controller for optimal response time"
    },
    "pharma": {
        "drug_discovery": "Optimize molecular structure for maximum binding affinity",
        "formulation": "Design drug formulation with improved bioavailability",
        "clinical_trial": "Optimize clinical trial parameters for efficacy detection"
    },
    "web_design": {
        "landing_page": "Optimize landing page layout to maximize conversion rate",
        "checkout_flow": "Design checkout flow to minimize cart abandonment",
        "user_experience": "Create responsive design balancing aesthetics and performance"
    }
}


# ============================================================================
# EXPECTED EVOLUTION RESULTS
# ============================================================================

EXPECTED_RESULTS = {
    "pes": {
        "fitness_range": (0.85, 0.98),
        "evaluation_range": (20, 50),
        "time_range": (30, 90),
        "characteristics": [
            "60% fewer evaluations than baseline",
            "Planning phase identifies good strategies",
            "Early stopping saves time",
            "Memory system retrieves past solutions"
        ]
    },
    "qd": {
        "fitness_range": (0.75, 0.92),
        "evaluation_range": (80, 150),
        "time_range": (45, 120),
        "characteristics": [
            "High diversity in archive",
            "Explores entire solution space",
            "Multiple niches filled",
            "Good for exploration"
        ]
    },
    "mo": {
        "fitness_range": (0.78, 0.95),
        "evaluation_range": (100, 200),
        "time_range": (60, 150),
        "characteristics": [
            "Pareto front with multiple solutions",
            "Trade-offs between objectives",
            "NSGA-II or SPEA2 algorithm",
            "Good for multi-criteria decisions"
        ]
    },
    "adversarial": {
        "fitness_range": (0.75, 0.90),
        "evaluation_range": (120, 200),
        "time_range": (90, 180),
        "characteristics": [
            "Robust to attacks",
            "Survives adversarial testing",
            "Safety-critical applications",
            "Red team/blue team approach"
        ]
    },
    "standard": {
        "fitness_range": (0.65, 0.85),
        "evaluation_range": (150, 300),
        "time_range": (60, 150),
        "characteristics": [
            "Baseline evolutionary algorithm",
            "Island model or simple GA",
            "No special features",
            "Good for simple problems"
        ]
    }
}


# ============================================================================
# CROSS-DOMAIN KNOWLEDGE TRANSFER EXAMPLES
# ============================================================================

CROSS_DOMAIN_EXAMPLES = [
    {
        "source_domain": "finance",
        "target_domain": "trading",
        "pattern": "Momentum helps escape local optima",
        "similarity_score": 0.92,
        "transfer_success_rate": 0.88,
        "example": "Momentum-based portfolio optimization applies to trading strategies"
    },
    {
        "source_domain": "engineering",
        "target_domain": "pharma",
        "pattern": "Robustness testing improves safety",
        "similarity_score": 0.85,
        "transfer_success_rate": 0.80,
        "example": "Adversarial testing from structural engineering applies to drug safety"
    },
    {
        "source_domain": "science",
        "target_domain": "finance",
        "pattern": "Sequential experimental design reduces cost",
        "similarity_score": 0.78,
        "transfer_success_rate": 0.75,
        "example": "Sequential testing from experimental design applies to portfolio testing"
    },
    {
        "source_domain": "web_design",
        "target_domain": "finance",
        "pattern": "A/B testing identifies improvements",
        "similarity_score": 0.70,
        "transfer_success_rate": 0.68,
        "example": "Conversion optimization methods apply to strategy optimization"
    }
]


# ============================================================================
# STRATEGY SELECTION DECISION TREE
# ============================================================================

STRATEGY_SELECTION_RULES = {
    "expensive_evaluations": {
        "indicators": [
            "backtest" in description.lower(),
            "simulation" in description.lower(),
            "experiment" in description.lower(),
            estimated_time_per_eval > 60,
            estimated_cost_per_eval > 100
        ],
        "recommended_mode": "pes",
        "confidence": 0.9,
        "reason": "PES reduces evaluations by 60% for expensive problems"
    },
    "multi_objective": {
        "indicators": [
            len(objectives) > 1,
            "and" in description.lower(),
            "balance" in description.lower()
        ],
        "recommended_mode": "mo",
        "confidence": 0.85,
        "reason": "Multiple objectives require Pareto optimization"
    },
    "diversity_needed": {
        "indicators": [
            "explore" in description.lower(),
            "novel" in description.lower(),
            "diverse" in description.lower(),
            require_diversity is True
        ],
        "recommended_mode": "qd",
        "confidence": 0.8,
        "reason": "Quality Diversity explores entire solution space"
    },
    "safety_critical": {
        "indicators": [
            domain in ["engineering", "pharma", "finance"],
            safety_critical is True
        ],
        "recommended_mode": "adversarial",
        "confidence": 0.85,
        "reason": "Safety-critical systems require adversarial testing"
    },
    "real_time": {
        "indicators": [
            constraints.get("real_time", False) is True,
            "fast" in description.lower(),
            "latency" in description.lower()
        ],
        "recommended_mode": "pes",
        "confidence": 0.7,
        "reason": "PES directed search finds good solutions quickly"
    },
    "default": {
        "indicators": [],
        "recommended_mode": "pes",
        "confidence": 0.75,
        "reason": "PES provides best general performance"
    }
}


# ============================================================================
# PERFORMANCE TARGETS BY DOMAIN
# ============================================================================

PERFORMANCE_TARGETS = {
    "general": {
        "max_time_seconds": 60,
        "max_evaluations": 100,
        "min_fitness": 0.70,
        "efficiency_improvement": "40%",
        "baseline_evaluations": 200
    },
    "finance": {
        "max_time_seconds": 600,
        "max_evaluations": 50,
        "min_fitness": 0.70,
        "efficiency_improvement": "60%",
        "baseline_evaluations": 150,
        "notes": "Backtesting is expensive, PES crucial"
    },
    "trading": {
        "max_time_seconds": 300,
        "max_evaluations": 80,
        "min_fitness": 0.75,
        "efficiency_improvement": "50%",
        "baseline_evaluations": 160,
        "notes": "Historical data analysis moderately expensive"
    },
    "science": {
        "max_time_seconds": 900,
        "max_evaluations": 20,
        "min_fitness": 0.80,
        "efficiency_improvement": "70%",
        "baseline_evaluations": 100,
        "notes": "Physical experiments very expensive, PES critical"
    },
    "engineering": {
        "max_time_seconds": 600,
        "max_evaluations": 100,
        "min_fitness": 0.80,
        "efficiency_improvement": "30%",
        "baseline_evaluations": 150,
        "notes": "FEA simulations moderately expensive, adversarial important"
    },
    "pharma": {
        "max_time_seconds": 600,
        "max_evaluations": 100,
        "min_fitness": 0.85,
        "efficiency_improvement": "30%",
        "baseline_evaluations": 150,
        "notes": "Molecular docking expensive, safety critical"
    },
    "web_design": {
        "max_time_seconds": 120,
        "max_evaluations": 200,
        "min_fitness": 0.75,
        "efficiency_improvement": "30%",
        "baseline_evaluations": 300,
        "notes": "A/B testing cheap, can evaluate more solutions"
    }
}


# ============================================================================
# GAUNTLET EVALUATION CRITERIA
# ============================================================================

GAUNTLET_CRITERIA = {
    "round_1_loongflow_ai": {
        "weight": 0.2,
        "min_score": 0.5,
        "evaluation_criteria": [
            "Code correctness",
            "Algorithm efficiency",
            "Problem understanding",
            "Solution completeness"
        ],
        "passing_threshold": "Quick screen - identify promising solutions"
    },
    "round_2_red_team": {
        "weight": 0.3,
        "min_score": 0.7,
        "evaluation_criteria": [
            "Edge case handling",
            "Error robustness",
            "Input validation",
            "Failure modes"
        ],
        "passing_threshold": "Adversarial testing - find vulnerabilities"
    },
    "round_3_gold_team": {
        "weight": 0.5,
        "min_score": 0.9,
        "evaluation_criteria": [
            "Code quality",
            "Documentation",
            "Maintainability",
            "Best practices",
            "Formal verification (if applicable)"
        ],
        "passing_threshold": "Consensus verification - production-ready"
    }
}


# ============================================================================
# KNOWLEDGE EXTRACTION PATTERNS
# ============================================================================

KNOWLEDGE_PATTERNS = {
    "pes_patterns": [
        {
            "pattern": "Momentum escapes local optima",
            "success_rate": 0.92,
            "domains": ["finance", "trading", "engineering"],
            "evidence": "85% of runs using momentum outperformed baseline"
        },
        {
            "pattern": "Early stopping saves evaluations",
            "success_rate": 0.95,
            "domains": ["finance", "science", "trading"],
            "evidence": "60% average reduction in evaluations"
        },
        {
            "pattern": "Adaptive learning rate improves convergence",
            "success_rate": 0.88,
            "domains": ["finance", "trading"],
            "evidence": "40% faster convergence on average"
        }
    ],
    "qd_patterns": [
        {
            "pattern": "Grid resolution 10 balances coverage and cost",
            "success_rate": 0.85,
            "domains": ["general", "engineering"],
            "evidence": "90% archive coverage with reasonable cost"
        },
        {
            "pattern": "Feature selection critical for diversity",
            "success_rate": 0.82,
            "domains": ["science", "pharma"],
            "evidence": "Better feature space improves exploration"
        }
    ],
    "mo_patterns": [
        {
            "pattern": "NSGA-II performs best for 2-3 objectives",
            "success_rate": 0.90,
            "domains": ["engineering", "finance"],
            "evidence": "Well-distributed Pareto fronts"
        },
        {
            "pattern": "SPEA2 better for many objectives",
            "success_rate": 0.78,
            "domains": ["pharma", "science"],
            "evidence": "Handles >3 objectives better"
        }
    ],
    "adversarial_patterns": [
        {
            "pattern": "20 rounds sufficient for robustness",
            "success_rate": 0.88,
            "domains": ["engineering", "pharma", "finance"],
            "evidence": "95% of vulnerabilities found in 20 rounds"
        },
        {
            "pattern": "Gradient attacks find most failures",
            "success_rate": 0.85,
            "domains": ["engineering", "pharma"],
            "evidence": "Adversarial examples expose weaknesses"
        }
    ]
}


# ============================================================================
# LEARNING LOOP SIMULATION DATA
# ============================================================================

LEARNING_ITERATIONS = [
    {
        "iteration": 1,
        "problem": "Simple optimization",
        "strategy": "pes",
        "fitness": 0.75,
        "evaluations": 50,
        "learned": "Initial baseline established"
    },
    {
        "iteration": 2,
        "problem": "Similar optimization",
        "strategy": "pes",
        "fitness": 0.82,
        "evaluations": 40,
        "learned": "Reduced evaluations based on iteration 1"
    },
    {
        "iteration": 3,
        "problem": "Related optimization",
        "strategy": "pes",
        "fitness": 0.88,
        "evaluations": 35,
        "learned": "Improved planning from iterations 1-2"
    },
    {
        "iteration": 4,
        "problem": "Final optimization",
        "strategy": "pes",
        "fitness": 0.92,
        "evaluations": 30,
        "learned": "Converged to optimal configuration"
    }
]


# ============================================================================
# ERROR SCENARIOS FOR RECOVERY TESTING
# ============================================================================

ERROR_SCENARIOS = [
    {
        "scenario": "empty_problem",
        "input": {"description": ""},
        "expected_error": ValueError,
        "expected_message": "Problem description cannot be empty"
    },
    {
        "scenario": "invalid_domain",
        "input": {"description": "Test", "domain": "invalid_domain"},
        "expected_error": ValueError,
        "expected_message": "Unknown domain: invalid_domain"
    },
    {
        "scenario": "evolution_failure",
        "context": "Evolution engine crashes",
        "expected_behavior": "Graceful degradation, return error result",
        "fallback": "Use standard mode as fallback"
    },
    {
        "scenario": "gauntlet_timeout",
        "context": "Gauntlet evaluation exceeds timeout",
        "expected_behavior": "Terminate and mark as failed",
        "fallback": "Return partial results if available"
    },
    {
        "scenario": "knowledge_engine_unavailable",
        "context": "Knowledge engine not responding",
        "expected_behavior": "Continue without knowledge, use defaults",
        "fallback": "Log warning and proceed with standard configuration"
    }
]


# ============================================================================
# BATCH EVOLUTION TEST CASES
# ============================================================================

BATCH_TEST_CASES = {
    "homogeneous_batch": {
        "description": "Multiple similar problems",
        "problems": [
            "Optimize portfolio for stock A",
            "Optimize portfolio for stock B",
            "Optimize portfolio for stock C"
        ],
        "expected_behavior": "Learning transfers between problems",
        "improvement_expected": True
    },
    "heterogeneous_batch": {
        "description": "Multiple different domains",
        "problems": [
            "Optimize portfolio (finance)",
            "Design experiment (science)",
            "Optimize structure (engineering)"
        ],
        "expected_behavior": "Each problem uses domain-specific config",
        "improvement_expected": False
    },
    "concurrent_batch": {
        "description": "Problems run in parallel",
        "problems": [
            "Concurrent problem 1",
            "Concurrent problem 2",
            "Concurrent problem 3"
        ],
        "expected_behavior": "All complete successfully",
        "improvement_expected": False
    }
}


# ============================================================================
# PERFORMANCE REGRESSION DATA
# ============================================================================

PERFORMANCE_REGRESSION_TESTS = [
    {
        "test_name": "simple_optimization_performance",
        "problem": "Maximize x^2",
        "domain": "general",
        "baseline": {
            "fitness": 0.75,
            "evaluations": 200,
            "time": 90
        },
        "target": {
            "fitness": 0.85,  # 13% improvement
            "evaluations": 100,  # 50% reduction
            "time": 60  # 33% reduction
        }
    },
    {
        "test_name": "finance_optimization_performance",
        "problem": "Optimize portfolio",
        "domain": "finance",
        "baseline": {
            "fitness": 0.70,
            "evaluations": 150,
            "time": 900
        },
        "target": {
            "fitness": 0.80,  # 14% improvement
            "evaluations": 50,  # 67% reduction
            "time": 600  # 33% reduction
        }
    },
    {
        "test_name": "science_optimization_performance",
        "problem": "Design experiment",
        "domain": "science",
        "baseline": {
            "fitness": 0.75,
            "evaluations": 100,
            "time": 3600
        },
        "target": {
            "fitness": 0.85,  # 13% improvement
            "evaluations": 20,  # 80% reduction
            "time": 900  # 75% reduction
        }
    }
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_problem_for_domain(domain: str, problem_type: str = "default") -> str:
    """Get sample problem for specific domain."""
    domain_problems = SAMPLE_PROBLEMS.get(domain, SAMPLE_PROBLEMS["general"])
    if problem_type == "default" or problem_type not in domain_problems:
        # Return first problem
        return list(domain_problems.values())[0]
    return domain_problems[problem_type]


def get_expected_results_for_mode(mode: str) -> Dict[str, Any]:
    """Get expected results for specific evolution mode."""
    return EXPECTED_RESULTS.get(mode, EXPECTED_RESULTS["standard"])


def get_performance_target(domain: str) -> Dict[str, Any]:
    """Get performance targets for specific domain."""
    return PERFORMANCE_TARGETS.get(domain, PERFORMANCE_TARGETS["general"])


def calculate_efficiency_gain(baseline_evals: int, optimized_evals: int) -> float:
    """Calculate efficiency improvement percentage."""
    if baseline_evals == 0:
        return 0.0
    return (baseline_evals - optimized_evals) / baseline_evals


def verify_pareto_optimality(solutions: List[Dict[str, Any]]) -> bool:
    """
    Verify that solutions form a valid Pareto front.

    A valid Pareto front has no dominated solutions.
    """
    for i, sol1 in enumerate(solutions):
        for j, sol2 in enumerate(solutions):
            if i == j:
                continue

            obj1 = sol1.get("objectives", {})
            obj2 = sol2.get("objectives", {})

            # Check if sol2 dominates sol1
            dominates = all(
                obj2.get(k, 0) >= obj1.get(k, 0)
                for k in obj1.keys()
            ) and any(
                obj2.get(k, 0) > obj1.get(k, 0)
                for k in obj1.keys()
            )

            if dominates:
                return False  # Found dominated solution

    return True


# ============================================================================
# EXPORT ALL DATA
# ============================================================================

__all__ = [
    "SAMPLE_PROBLEMS",
    "EXPECTED_RESULTS",
    "CROSS_DOMAIN_EXAMPLES",
    "STRATEGY_SELECTION_RULES",
    "PERFORMANCE_TARGETS",
    "GAUNTLET_CRITERIA",
    "KNOWLEDGE_PATTERNS",
    "LEARNING_ITERATIONS",
    "ERROR_SCENARIOS",
    "BATCH_TEST_CASES",
    "PERFORMANCE_REGRESSION_TESTS",
    # Utility functions
    "get_problem_for_domain",
    "get_expected_results_for_mode",
    "get_performance_target",
    "calculate_efficiency_gain",
    "verify_pareto_optimality"
]
