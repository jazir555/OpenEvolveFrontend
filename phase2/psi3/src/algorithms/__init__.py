"""
Ψ₃ Algorithms Module

Implements the 4-stage constraint reduction pipeline.
"""

from .preprocessing import syntactic_preprocessing, estimate_redundancy, PreprocessingResult
from .dependency_analyzer import (
    build_dependency_graph,
    DependencyGraph,
    find_redundant_constraints,
    find_independent_components,
    compute_closure,
    DependencyAnalysisResult
)

__all__ = [
    # Stage 1: Preprocessing
    "syntactic_preprocessing",
    "estimate_redundancy",
    "PreprocessingResult",

    # Stage 2: Dependency Analysis
    "build_dependency_graph",
    "DependencyGraph",
    "find_redundant_constraints",
    "find_independent_components",
    "compute_closure",
    "DependencyAnalysisResult",
]
