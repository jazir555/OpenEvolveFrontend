"""
I_mech: Mechanistic Isomorphism Validator

Detects mechanistic isomorphisms between problem domains and enables
reliable solution transfer using graph isomorphism, causal structure analysis,
and formal proof verification.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
Target: >80% transfer success correlation
"""

# Version info
__version__ = '1.0.0'
__author__ = 'Agent G3'

# Main interface
from .isomorphism_validator import (
    IMechValidator,
    compare_domains
)

# Core data structures
from .core import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType,
    Domain,
    SimilarityResult
)

# Algorithms
from .algorithms import (
    WeisfeilerLehman,
    VF2Matcher,
    SubgraphMatcher,
    InterventionSimulator
)

# Transfer components
from .transfer import (
    SolutionMapper,
    SolutionValidator,
    SolutionRepair
)

# Lean 4 integration
from .lean4 import ProofGenerator

__all__ = [
    # Main interface
    'IMechValidator',
    'compare_domains',

    # Core data structures
    'FunctionalDependencyGraph',
    'Node',
    'Edge',
    'EdgeType',
    'Domain',
    'SimilarityResult',

    # Algorithms
    'WeisfeilerLehman',
    'VF2Matcher',
    'SubgraphMatcher',
    'InterventionSimulator',

    # Transfer
    'SolutionMapper',
    'SolutionValidator',
    'SolutionRepair',

    # Proofs
    'ProofGenerator'
]


# Module metadata
MODULE_INFO = {
    'name': 'I_mech',
    'version': __version__,
    'description': 'Mechanistic Isomorphism Validator for analogy detection and solution transfer',
    'capabilities': [
        'Graph isomorphism detection (Weisfeiler-Lehman + VF2)',
        'Causal structure analysis',
        'Interventional equivalence testing',
        'Solution transfer between isomorphic domains',
        'Lean 4 proof generation and verification'
    ],
    'performance': {
        'target_accuracy': 0.80,
        'benchmarked': True
    }
}
