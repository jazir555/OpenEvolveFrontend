"""
CAV-NLP Integration for OpenEvolve

Canonical Arithmetic Verification via NLP - Integration Module

This module provides integration between CAV-NLP (mathematical formalization 
from natural language/LaTeX to verified Lean 4) and OpenEvolve.

Main Components:
- Z3LeanAideBridge: Main bridge class (preserves legacy API)
- Data structures: Z3Constraint, Lean4Constraint, TranslationResult, etc.
- Mappings: Type and operator mappings between Z3 and Lean
- Verification: Enhanced verification with CAV-NLP canonicalization

Quick Start:
    >>> from openevolve.cav_nlp_integration import Z3LeanAideBridge
    >>> bridge = Z3LeanAideBridge()
    >>> result = await bridge.verify("x + y = y + x")
"""

__version__ = "2.0.0"
__author__ = "OpenEvolve"

# Main adapter
from .adapter import (
    Z3LeanAideBridge,
    create_z3_lean_bridge,
    quick_verify,
)

# Data structures
from .data_structures import (
    TranslationDirection,
    ConstraintType,
    Z3Constraint,
    Lean4Constraint,
    TranslationResult,
    VerificationBridgeResult,
    HybridProofResult,
    CAVNLPContext,
    CanonicalizationResult,
)

# Mappings
from .mappings import (
    Z3_TO_LEAN_TYPES,
    LEAN_TO_Z3_TYPES,
    Z3_TO_LEAN_OPERATORS,
    LEAN_TO_Z3_OPERATORS,
    CONSTRAINT_TYPE_TACTICS,
    CANONICALIZATION_RULES,
    CANONICALIZATION_ORDER,
    LEAN_IMPORTS_BY_TYPE,
)

# Verification
from .verification import (
    Z3LeanVerificationBridge,
)

# CAV-NLP Core Components (re-exported for convenience)
# These use lazy imports to avoid circular dependency issues

def _import_flexible_semantic_parsing():
    """Lazy import flexible_semantic_parsing components."""
    try:
        from .flexible_semantic_parsing import (
            MathematicalTextParser,
            SemanticPrimitive,
            SemanticNormalizer,
        )
        return MathematicalTextParser, SemanticPrimitive, SemanticNormalizer
    except ImportError:
        return None, None, None

def _import_dependency_dag():
    """Lazy import dependency DAG components."""
    try:
        from .dependency_dag import (
            DependencyDAG,
            Statement,
            StatementKind,
        )
        return DependencyDAG, Statement, StatementKind
    except ImportError:
        return None, None, None

def _import_z3_semantic_synthesis():
    """Lazy import Z3 semantic synthesis components."""
    try:
        from .z3_semantic_synthesis import (
            Z3SemanticSynthesis,
            SemanticSketch,
            SemanticHole,
        )
        return Z3SemanticSynthesis, SemanticSketch, SemanticHole
    except ImportError:
        return None, None, None

def _import_canonical_lean_generator():
    """Lazy import canonical lean generator components."""
    try:
        from .canonical_lean_generator import (
            CanonicalLeanGenerator,
            SemanticGrammar,
        )
        return CanonicalLeanGenerator, SemanticGrammar
    except ImportError:
        return None, None

def _import_z3_canonicalizer():
    """Lazy import Z3 canonicalizer components."""
    try:
        from .z3_canonicalizer import Z3Canonicalizer
        return Z3Canonicalizer
    except ImportError:
        return None

# Initialize lazy imports on first access
MathematicalTextParser, SemanticPrimitive, SemanticNormalizer = _import_flexible_semantic_parsing()
DependencyDAG, Statement, StatementKind = _import_dependency_dag()
Z3SemanticSynthesis, SemanticSketch, SemanticHole = _import_z3_semantic_synthesis()
CanonicalLeanGenerator, SemanticGrammar = _import_canonical_lean_generator()
Z3Canonicalizer = _import_z3_canonicalizer()

__all__ = [
    # Version
    "__version__",
    
    # Main adapter
    "Z3LeanAideBridge",
    "create_z3_lean_bridge",
    "quick_verify",
    
    # Data structures
    "TranslationDirection",
    "ConstraintType",
    "Z3Constraint",
    "Lean4Constraint",
    "TranslationResult",
    "VerificationBridgeResult",
    "HybridProofResult",
    "CAVNLPContext",
    "CanonicalizationResult",
    
    # Mappings
    "Z3_TO_LEAN_TYPES",
    "LEAN_TO_Z3_TYPES",
    "Z3_TO_LEAN_OPERATORS",
    "LEAN_TO_Z3_OPERATORS",
    "CONSTRAINT_TYPE_TACTICS",
    "CANONICALIZATION_RULES",
    "CANONICALIZATION_ORDER",
    "LEAN_IMPORTS_BY_TYPE",
    
    # Verification
    "Z3LeanVerificationBridge",
    
    # CAV-NLP Core
    "MathematicalTextParser",
    "SemanticPrimitive",
    "SemanticNormalizer",
    "DependencyDAG",
    "Statement",
    "StatementKind",
    "Z3SemanticSynthesis",
    "SemanticSketch",
    "SemanticHole",
    "CanonicalLeanGenerator",
    "SemanticGrammar",
    "Z3Canonicalizer",
]

# Check dependencies
def _check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import z3
    except ImportError:
        missing.append("z3-solver")
    
    try:
        from lean4_integration import LeanAideService
    except ImportError:
        pass  # Lean integration is optional
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing optional dependencies: {', '.join(missing)}. "
            "Some features may be unavailable.",
            ImportWarning
        )

_check_dependencies()
