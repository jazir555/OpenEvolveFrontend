"""
RESE Phase III: MCTS Search Adapter

This package implements RESE Phase III - Monte Carlo Refinement with MC-NEST algorithm.

Components:
- MCTSSearchExecutor: Main orchestrator
- SearchTreeBuilder: Tree management with idempotent updates
- HypothesisValidator: Statistical validation
- ConvergenceDetector: ACI-based convergence detection
- Phase3Adapter: REST API adapter

Author: RESE Team
Created: 2026-02-04
Phase: III - Monte Carlo Refinement
"""

from .phase3_executor import (
    Phase3Config,
    MCTSSearchExecutor,
    SearchTreeBuilder,
    HypothesisValidator,
    ConvergenceDetector,
    UCB1SelectionStrategy,
    HypothesisDLQ,
    ValidationMetrics,
)

from .phase3_adapter import (
    Phase3Adapter,
    create_adapter,
)

__all__ = [
    # Executor components
    "Phase3Config",
    "MCTSSearchExecutor",
    "SearchTreeBuilder",
    "HypothesisValidator",
    "ConvergenceDetector",
    "UCB1SelectionStrategy",
    "HypothesisDLQ",
    "ValidationMetrics",

    # Adapter
    "Phase3Adapter",
    "create_adapter",
]

__version__ = "1.0.0"
__author__ = "RESE Team"
__phase__ = "III - Monte Carlo Refinement"
