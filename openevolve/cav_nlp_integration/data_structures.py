"""
CAV-NLP Integration Data Structures Module

This module provides dataclasses and enums for the CAV-NLP (Computer-Aided Verification - Natural Language Processing)
integration system. It preserves the core data structures from the Z3-LeanAide bridge while adding CAV-NLP specific
enhancements for formal verification workflow management.

Components:
    - TranslationDirection: Enum for translation direction
    - ConstraintType: Enum for constraint classification
    - Z3Constraint: Z3 constraint representation
    - Lean4Constraint: Lean 4 constraint representation
    - TranslationResult: Translation operation metadata with CAV-NLP enhancements
    - VerificationBridgeResult: Dual verification result with CAV-NLP enhancements
    - HybridProofResult: Hybrid proof combining Z3 and Lean 4
    - CAVNLPContext: CAV-NLP specific context for theorem extraction
    - CanonicalizationResult: Canonical form verification result
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class TranslationDirection(Enum):
    """Direction of translation between Z3 and Lean 4 representations.
    
    Attributes:
        Z3_TO_LEAN: Translation from Z3 Python API to Lean 4 code
        LEAN_TO_Z3: Translation from Lean 4 code to Z3 Python API
        BIDIRECTIONAL: Translation supporting both directions
    """
    Z3_TO_LEAN = "z3_to_lean"
    LEAN_TO_Z3 = "lean_to_z3"
    BIDIRECTIONAL = "bidirectional"


class ConstraintType(Enum):
    """Types of constraints supported by the verification system.
    
    Attributes:
        BOOLEAN: Boolean logic constraints (and, or, not, implies)
        ARITHMETIC: Linear arithmetic constraints (+, -, *, / on reals/ints)
        ARRAY: Array theory constraints (select, store)
        BITVECTOR: Bit-vector constraints (bitwise operations)
        NONLINEAR: Non-linear arithmetic (polynomials, exponentials)
        QUANTIFIED: Quantified formulas (forall, exists)
    """
    BOOLEAN = "boolean"
    ARITHMETIC = "arithmetic"
    ARRAY = "array"
    BITVECTOR = "bitvector"
    NONLINEAR = "nonlinear"
    QUANTIFIED = "quantified"


@dataclass
class Z3Constraint:
    """Representation of a Z3 constraint.
    
    Attributes:
        expr: The Z3 expression object or string representation
        constraint_type: Classification of the constraint type
        variables: List of variable names used in the constraint
        is_assertion: Whether this is an assertion (vs definition)
    """
    expr: Any
    constraint_type: ConstraintType
    variables: List[str]
    is_assertion: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert the constraint to a dictionary representation.
        
        Returns:
            Dictionary containing constraint data with serializable types
        """
        return {
            "expr": str(self.expr) if self.expr is not None else None,
            "constraint_type": self.constraint_type.value,
            "variables": self.variables,
            "is_assertion": self.is_assertion
        }


@dataclass
class Lean4Constraint:
    """Representation of a Lean 4 constraint.
    
    Attributes:
        lean_code: The Lean 4 code string
        constraint_type: Classification of the constraint type
        variables: List of variable names used in the constraint
        theorem_statement: Optional theorem statement associated with this constraint
    """
    lean_code: str
    constraint_type: ConstraintType
    variables: List[str]
    theorem_statement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the constraint to a dictionary representation.
        
        Returns:
            Dictionary containing constraint data
        """
        return {
            "lean_code": self.lean_code,
            "constraint_type": self.constraint_type.value,
            "variables": self.variables,
            "theorem_statement": self.theorem_statement
        }


@dataclass
class TranslationResult:
    """Result of a translation operation between Z3 and Lean 4.
    
    This dataclass captures metadata about translation attempts, including
    source/target code, errors, and timing information.
    
    CAV-NLP Enhancements:
        dag: Optional dependency graph extracted during translation
        canonical_form: Optional canonical representation for comparison
        cegis_iterations: Optional counterexample-guided iteration count
    
    Attributes:
        success: Whether the translation succeeded
        source: Source system identifier (e.g., "z3", "lean4")
        target: Target system identifier
        direction: Direction of translation
        source_code: Original source code/expression
        target_code: Translated target code/expression
        errors: List of error messages if translation failed
        warnings: List of warning messages
        metadata: Additional translation metadata
        timestamp: When the translation occurred
        dag: Optional dependency graph for the translation (CAV-NLP)
        canonical_form: Optional canonical form string (CAV-NLP)
        cegis_iterations: Optional CEGIS iteration count (CAV-NLP)
    """
    success: bool
    source: str
    target: str
    direction: TranslationDirection
    source_code: str
    target_code: Optional[str]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    # CAV-NLP Enhancements
    dag: Optional[Any] = None
    canonical_form: Optional[str] = None
    cegis_iterations: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the translation result to a dictionary.
        
        Returns:
            Dictionary containing all translation result data
        """
        result = {
            "success": self.success,
            "source": self.source,
            "target": self.target,
            "direction": self.direction.name,
            "source_code": self.source_code,
            "target_code": self.target_code,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "dag": str(self.dag) if self.dag is not None else None,
            "canonical_form": self.canonical_form,
            "cegis_iterations": self.cegis_iterations
        }
        return result


@dataclass
class VerificationBridgeResult:
    """Result of dual verification using both Z3 and Lean 4.
    
    This dataclass captures the results of running verification through
    both Z3 and Lean 4 systems, comparing their outcomes for consistency.
    
    CAV-NLP Enhancements:
        dag: Optional dependency graph for verification context
        canonicalization_verified: Whether canonicalization validation passed
    
    Attributes:
        z3_result: Result from Z3 verification (SAT/UNSAT/UNKNOWN)
        lean_result: Result from Lean 4 verification (proved/disproved/unknown)
        agreed: Whether both systems agree on the result
        z3_model: Optional model from Z3 if satisfiable
        lean_proof: Optional proof term from Lean 4
        counterexample: Optional counterexample if verification failed
        confidence: Confidence score in the verification result (0-1)
        execution_time: Total execution time in seconds
        dag: Optional dependency graph for verification (CAV-NLP)
        canonicalization_verified: Whether canonical form was validated (CAV-NLP)
    """
    z3_result: str
    lean_result: str
    agreed: bool
    z3_model: Optional[Dict[str, Any]] = None
    lean_proof: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    execution_time: float = 0.0
    # CAV-NLP Enhancements
    dag: Optional[Any] = None
    canonicalization_verified: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the verification result to a dictionary.
        
        Returns:
            Dictionary containing all verification result data
        """
        return {
            "z3_result": self.z3_result,
            "lean_result": self.lean_result,
            "agreed": self.agreed,
            "z3_model": self.z3_model,
            "lean_proof": self.lean_proof,
            "counterexample": self.counterexample,
            "confidence": self.confidence,
            "execution_time": self.execution_time,
            "dag": str(self.dag) if self.dag is not None else None,
            "canonicalization_verified": self.canonicalization_verified
        }


@dataclass
class HybridProofResult:
    """Result of a hybrid proof combining Z3 and Lean 4.
    
    This dataclass represents proofs that leverage both Z3's automated
    reasoning and Lean 4's interactive theorem proving capabilities.
    
    Attributes:
        success: Whether the hybrid proof succeeded
        z3_component: Description of Z3's contribution to the proof
        lean_component: Description of Lean 4's contribution
        combined_proof: The combined proof script or term
        tactics_used: List of tactics used in the Lean component
        z3_time: Time spent in Z3 verification (seconds)
        lean_time: Time spent in Lean 4 verification (seconds)
        total_time: Total proof time (seconds)
    """
    success: bool
    z3_component: Optional[str] = None
    lean_component: Optional[str] = None
    combined_proof: Optional[str] = None
    tactics_used: List[str] = field(default_factory=list)
    z3_time: float = 0.0
    lean_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the hybrid proof result to a dictionary.
        
        Returns:
            Dictionary containing all hybrid proof data
        """
        return {
            "success": self.success,
            "z3_component": self.z3_component,
            "lean_component": self.lean_component,
            "combined_proof": self.combined_proof,
            "tactics_used": self.tactics_used,
            "z3_time": self.z3_time,
            "lean_time": self.lean_time,
            "total_time": self.total_time
        }


@dataclass
class CAVNLPContext:
    """Context information for CAV-NLP theorem extraction and processing.
    
    This dataclass captures metadata from academic papers and theorem
    contexts to support extraction and formalization of mathematical content.
    
    Attributes:
        paper_title: Optional title of the source paper
        section_context: Optional section or paragraph context
        theorem_number: Optional theorem number/index in the paper
        dependency_graph: Optional parsed dependency graph structure
    """
    paper_title: Optional[str] = None
    section_context: Optional[str] = None
    theorem_number: Optional[int] = None
    dependency_graph: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the CAV-NLP context to a dictionary.
        
        Returns:
            Dictionary containing context data
        """
        return {
            "paper_title": self.paper_title,
            "section_context": self.section_context,
            "theorem_number": self.theorem_number,
            "dependency_graph": str(self.dependency_graph) if self.dependency_graph is not None else None
        }


@dataclass
class CanonicalizationResult:
    """Result of canonical form transformation and verification.
    
    This dataclass captures the outcome of transforming an expression
    into its canonical form and verifying equivalence.
    
    Attributes:
        original: The original expression/formula
        canonical: The canonical form of the expression
        z3_validated: Whether Z3 confirmed semantic equivalence
        equivalent_by: Description of equivalence rule applied
                      (e.g., "commutativity", "associativity", "De Morgan")
    """
    original: str
    canonical: str
    z3_validated: bool = False
    equivalent_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the canonicalization result to a dictionary.
        
        Returns:
            Dictionary containing canonicalization data
        """
        return {
            "original": self.original,
            "canonical": self.canonical,
            "z3_validated": self.z3_validated,
            "equivalent_by": self.equivalent_by
        }


# Type alias for convenience
Constraint = Union[Z3Constraint, Lean4Constraint]
"""Type alias representing either a Z3 or Lean 4 constraint."""


__all__ = [
    "TranslationDirection",
    "ConstraintType", 
    "Z3Constraint",
    "Lean4Constraint",
    "TranslationResult",
    "VerificationBridgeResult",
    "HybridProofResult",
    "CAVNLPContext",
    "CanonicalizationResult",
    "Constraint",
]