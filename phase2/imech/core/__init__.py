"""
I_mech Core Module
Mechanistic Isomorphism Validator - Core Data Structures

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from .fdg import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType,
    CausalModel
)

from .domain import Domain
from .result import SimilarityResult

__all__ = [
    'FunctionalDependencyGraph',
    'Node',
    'Edge',
    'EdgeType',
    'CausalModel',
    'Domain',
    'SimilarityResult'
]
