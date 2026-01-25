"""
RESE Phase III: Monte Carlo Refinement

Phase III focuses on:
- Γ₁: ACI (Algorithmic Complexity Index) Analysis
- Γ₂: MCTS (Monte Carlo Tree Search)
- Γ₃: Convergence Control
- N_max: Maximum iterations control

Main Components:
    - aci_analyzer: ACI calculation and analysis
        * ACIAnalyzer: Main analyzer class
        * ComplexityMetrics: Detailed complexity metrics
        * ACIResult: Analysis result container
    - mcts_search: Monte Carlo Tree Search implementation
        * MCTSSearch: Main search class
        * MCTSNode: Tree node
        * PlayoutStrategy: Simulation strategies
    - convergence_controller: Control convergence
        * ConvergenceController: Monitor and control convergence
    - stage3_integration: Complete Phase III integration
        * MonteCarloNest: Full nest implementation
        * NestConfig: Configuration
    - statistical_validator: Statistical validation
        * StatisticalValidator: Validate results statistically

Usage:
    from phase3.aci_analyzer import ACIAnalyzer, calculate_aci
    from phase3.mcts_search import MCTSSearch
    from phase3.convergence_controller import ConvergenceController
    from phase3.stage3_integration import MonteCarloNest
"""

__version__ = "1.0.0"

__all__ = [
    "aci_analyzer",
    "mcts_search",
    "convergence_controller",
    "stage3_integration",
    "statistical_validator",
]
