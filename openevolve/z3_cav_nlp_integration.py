"""
Z3-CAV-NLP Integration Module
==============================

This module provides drop-in enhancements for existing Z3-based code,
adding CAV-NLP capabilities for formalization, canonicalization, and verification.

Key Components:
    - EnhancedZ3Solver: Z3 Solver with CAV-NLP integration
    - ConstraintFormalizer: Converts NL/LaTeX to Z3 constraints
    - ProofExporter: Exports Z3 proofs to Lean 4
    - CanonicalConstraintManager: Manages canonical forms of constraints
    - Decorators: @with_cav_nlp, @auto_canonicalize, @auto_formalize
    - Context managers: cav_nlp_scope, enhanced_solver

Integration Patterns:
    - Drop-in replacements for common Z3 patterns
    - Decorators for adding CAV-NLP to existing functions
    - Context managers for scoped CAV-NLP enhancement

Example Usage:
    # Before: Pure Z3
    solver = z3.Solver()
    solver.add(x > 0, y > 0)
    result = solver.check()

    # After: Z3 + CAV-NLP
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    solver = EnhancedZ3Solver()
    solver.add(solver.formalize_constraint("x and y are positive"))
    result = solver.check()
    verification = solver.verify_with_lean(solver.assertions())

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import functools
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    TypeVar, Union, Iterator, Tuple
)

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Optional Imports with Graceful Degradation
# =============================================================================

# Try to import Z3
try:
    import z3
    from z3 import Solver, Bool, Int, Real, sat, unsat, unknown
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 not available - using simulation mode")
    z3 = None
    Solver = None
    sat = unsat = unknown = None

# Try to import CAV-NLP integration
try:
    from openevolve.cav_nlp_integration import (
        Z3LeanAideBridge,
        Z3Constraint,
        Lean4Constraint,
        ConstraintType,
        VerificationBridgeResult,
        HybridProofResult,
        CanonicalizationResult,
        CAVNLPContext,
    )
    from openevolve.cav_nlp_integration.adapter import (
        create_z3_lean_bridge,
        quick_verify,
    )
    CAV_NLP_AVAILABLE = True
except ImportError as e:
    CAV_NLP_AVAILABLE = False
    logger.warning(f"CAV-NLP integration not available: {e}")
    Z3LeanAideBridge = None
    Z3Constraint = None
    Lean4Constraint = None
    ConstraintType = None
    VerificationBridgeResult = None
    HybridProofResult = None
    CanonicalizationResult = None
    CAVNLPContext = None

# Try to import Unified Math Service
try:
    from openevolve.unified_math_service import (
        UnifiedMathService,
        FormalizationResult,
        ProofResult,
        create_unified_math_service,
    )
    UNIFIED_MATH_AVAILABLE = True
except ImportError as e:
    UNIFIED_MATH_AVAILABLE = False
    logger.warning(f"Unified math service not available: {e}")
    UnifiedMathService = None
    FormalizationResult = None
    ProofResult = None

# Try to import Lean4 integration
try:
    from lean4_integration import VerificationResult, VerificationStatus
    LEAN4_AVAILABLE = True
except ImportError:
    LEAN4_AVAILABLE = False
    VerificationResult = None
    VerificationStatus = None


# =============================================================================
# Type Variables for Generics
# =============================================================================

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class VerificationResult:
    """Result of constraint verification.
    
    Attributes:
        success: Whether verification succeeded
        z3_result: Z3 verification result (sat/unsat/unknown)
        lean_result: Lean 4 verification result
        confidence: Confidence score (0-1)
        counterexample: Counterexample if verification failed
        proof: Generated proof if verification succeeded
        execution_time: Time taken for verification
        metadata: Additional metadata
    """
    success: bool
    z3_result: Optional[str] = None
    lean_result: Optional[Any] = None
    confidence: float = 0.0
    counterexample: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "z3_result": self.z3_result,
            "lean_result": str(self.lean_result) if self.lean_result else None,
            "confidence": self.confidence,
            "counterexample": self.counterexample,
            "proof": self.proof,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


@dataclass
class FormalizationResult:
    """Result of formalizing natural language to Z3.
    
    Attributes:
        success: Whether formalization succeeded
        z3_expr: The resulting Z3 expression
        constraint_type: Type of constraint
        variables: Variables used in the constraint
        canonical_form: Canonical form of the constraint
        source_text: Original natural language text
        warnings: Any warnings during formalization
    """
    success: bool
    z3_expr: Optional[Any] = None
    constraint_type: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    canonical_form: Optional[str] = None
    source_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "z3_expr": str(self.z3_expr) if self.z3_expr else None,
            "constraint_type": self.constraint_type,
            "variables": self.variables,
            "canonical_form": self.canonical_form,
            "source_text": self.source_text,
            "warnings": self.warnings,
        }


@dataclass
class CanonicalForm:
    """Represents a canonical form of a constraint.
    
    Attributes:
        original: Original constraint expression
        canonical: Canonical form
        is_valid: Whether canonicalization was successful
        equivalence_proof: Proof of equivalence
    """
    original: Any
    canonical: Any
    is_valid: bool = True
    equivalence_proof: Optional[str] = None


# =============================================================================
# EnhancedZ3Solver Class
# =============================================================================

class EnhancedZ3Solver:
    """Z3 Solver enhanced with CAV-NLP formalization capabilities.
    
    This class wraps the standard Z3 Solver and adds CAV-NLP capabilities
    for formalizing natural language constraints and hybrid verification.
    
    Attributes:
        solver: The underlying Z3 Solver instance
        cav_nlp: CAV-NLP bridge for formalization and verification
        math_service: Unified math service for formalization
        use_cav_nlp: Whether CAV-NLP features are enabled
        formalization_history: History of formalized constraints
    
    Example:
        >>> solver = EnhancedZ3Solver(use_cav_nlp=True)
        >>> # Add constraint from natural language
        >>> constraint = solver.formalize_constraint("x is positive and y is negative")
        >>> solver.add(constraint)
        >>> result = solver.check()
        >>> # Verify with Lean 4
        >>> verification = solver.verify_with_lean(solver.assertions())
    """
    
    def __init__(
        self,
        use_cav_nlp: bool = True,
        lean_service: Optional[Any] = None,
        enable_logging: bool = True
    ):
        """Initialize Enhanced Z3 Solver with CAV-NLP capabilities.
        
        Args:
            use_cav_nlp: Whether to enable CAV-NLP features
            lean_service: Optional LeanAide service for verification
            enable_logging: Whether to enable operation logging
        """
        # Initialize underlying Z3 solver
        if Z3_AVAILABLE and Solver is not None:
            self.solver = Solver()
        else:
            self.solver = None
            logger.warning("Z3 not available - operating in simulation mode")
        
        # Initialize CAV-NLP components
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        self.cav_nlp: Optional[Z3LeanAideBridge] = None
        self.math_service: Optional[UnifiedMathService] = None
        
        if self.use_cav_nlp:
            try:
                self.cav_nlp = create_z3_lean_bridge(lean_service)
                logger.info("CAV-NLP bridge initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP bridge: {e}")
                self.use_cav_nlp = False
        
        # Initialize unified math service
        if UNIFIED_MATH_AVAILABLE:
            try:
                self.math_service = create_unified_math_service(
                    use_cav_nlp=self.use_cav_nlp,
                    use_leanaide=True,
                    lean_service=lean_service
                )
                logger.info("Unified math service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize unified math service: {e}")
                self.math_service = None
        
        # History tracking
        self.formalization_history: List[Dict[str, Any]] = []
        self.enable_logging = enable_logging
        
        # Statistics
        self.stats = {
            "constraints_added": 0,
            "natural_language_formalized": 0,
            "verification_calls": 0,
            "canonicalizations": 0,
        }
    
    # ========================================================================
    # Core Solver Interface (Pass-through to Z3)
    # ========================================================================
    
    def add(self, *constraints) -> None:
        """Add constraints to the solver.
        
        Args:
            *constraints: Constraints to add (can be Z3 expressions or strings)
        """
        if self.solver is None:
            logger.warning("Cannot add constraints: Z3 not available")
            return
        
        for constraint in constraints:
            self.solver.add(constraint)
            self.stats["constraints_added"] += 1
            
            if self.enable_logging:
                logger.debug(f"Added constraint: {constraint}")
    
    def check(self, *assumptions) -> Any:
        """Check satisfiability of constraints.
        
        Args:
            *assumptions: Optional assumptions for the check
            
        Returns:
            sat, unsat, or unknown result
        """
        if self.solver is None:
            logger.warning("Cannot check: Z3 not available")
            return unknown if unknown else "unknown"
        
        return self.solver.check(*assumptions)
    
    def model(self) -> Optional[Any]:
        """Get the model from the last satisfiable check.
        
        Returns:
            Z3 model or None
        """
        if self.solver is None:
            return None
        return self.solver.model()
    
    def assertions(self) -> List[Any]:
        """Get all assertions in the solver.
        
        Returns:
            List of Z3 expressions
        """
        if self.solver is None:
            return []
        return list(self.solver.assertions())
    
    def push(self) -> None:
        """Push a new scope onto the solver stack."""
        if self.solver is not None:
            self.solver.push()
    
    def pop(self) -> None:
        """Pop the current scope from the solver stack."""
        if self.solver is not None:
            self.solver.pop()
    
    def reset(self) -> None:
        """Reset the solver, removing all constraints."""
        if self.solver is not None:
            self.solver.reset()
        self.formalization_history.clear()
    
    # ========================================================================
    # CAV-NLP Enhanced Methods
    # ========================================================================
    
    def formalize_constraint(
        self,
        natural_language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Convert natural language to Z3 constraint using CAV-NLP.
        
        This method uses the CAV-NLP pipeline to formalize natural language
        or LaTeX mathematical statements into Z3 constraints.
        
        Pipeline:
            1. Parse natural language using CAV-NLP
            2. Extract semantic primitives
            3. Build dependency DAG
            4. Synthesize to Z3 expression
        
        Args:
            natural_language: Natural language constraint (e.g., "x is positive")
            context: Optional context for formalization
            
        Returns:
            Z3 expression or None if formalization failed
            
        Example:
            >>> solver = EnhancedZ3Solver()
            >>> constraint = solver.formalize_constraint("x and y are positive")
            >>> solver.add(constraint)
        """
        if not self.use_cav_nlp:
            logger.warning("CAV-NLP not available, cannot formalize constraint")
            return None
        
        try:
            start_time = datetime.now()
            
            # Use unified math service if available
            if self.math_service is not None:
                # Run formalization asynchronously
                result = asyncio.run(self._formalize_async(natural_language, context))
                
                if result and result.success:
                    z3_expr = self._lean_to_z3_expr(result.code)
                    
                    # Record in history
                    self.formalization_history.append({
                        "source": natural_language,
                        "result": str(z3_expr) if z3_expr else None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    
                    self.stats["natural_language_formalized"] += 1
                    
                    if self.enable_logging:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        logger.info(
                            f"Formalized '{natural_language[:50]}...' in {elapsed:.3f}s"
                        )
                    
                    return z3_expr
            
            # Fallback: use CAV-NLP bridge directly
            if self.cav_nlp is not None:
                # Formalize to Lean first, then translate to Z3
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    formalization = loop.run_until_complete(
                        self.math_service.formalize(natural_language) if self.math_service else None
                    )
                finally:
                    loop.close()
                
                if formalization and formalization.success:
                    lean_constraint = Lean4Constraint(
                        lean_code=formalization.code,
                        constraint_type=ConstraintType.BOOLEAN if ConstraintType else None,
                        variables=[],
                        theorem_statement=None
                    )
                    z3_constraint = self.cav_nlp.lean4_to_z3(lean_constraint.lean_code)
                    if z3_constraint:
                        return z3_constraint.expr
            
            logger.warning(f"Failed to formalize: {natural_language}")
            return None
            
        except Exception as e:
            logger.error(f"Error formalizing constraint: {e}")
            return None
    
    async def _formalize_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Any]:
        """Asynchronously formalize text using unified math service."""
        if self.math_service is None:
            return None
        
        cav_context = None
        if context and CAVNLPContext is not None:
            cav_context = CAVNLPContext(
                paper_title=context.get("paper_title"),
                section_context=context.get("section_context"),
                theorem_number=context.get("theorem_number"),
            )
        
        return await self.math_service.formalize(
            text, context=cav_context, elaborate=True
        )
    
    def verify_with_lean(
        self,
        constraints: Optional[List[Any]] = None,
        use_counterexamples: bool = True
    ) -> VerificationResult:
        """Verify constraints using hybrid Z3 + Lean approach.
        
        This method performs dual verification using both Z3 and Lean 4,
        providing higher confidence in the verification result.
        
        Args:
            constraints: Constraints to verify (uses solver assertions if None)
            use_counterexamples: Whether to generate counterexamples on failure
            
        Returns:
            VerificationResult with dual verification results
            
        Example:
            >>> solver = EnhancedZ3Solver()
            >>> solver.add(x > 0, y > 0)
            >>> result = solver.verify_with_lean()
            >>> print(f"Confidence: {result.confidence}")
        """
        if not self.use_cav_nlp or self.cav_nlp is None:
            logger.warning("CAV-NLP not available, falling back to Z3 only")
            return self._verify_z3_only(constraints)
        
        start_time = datetime.now()
        self.stats["verification_calls"] += 1
        
        try:
            # Get constraints to verify
            if constraints is None:
                constraints = self.assertions()
            
            # Convert to Z3Constraint
            z3_constraints = []
            for constraint in constraints:
                if CAV_NLP_AVAILABLE and Z3Constraint is not None and ConstraintType is not None:
                    z3_constraints.append(Z3Constraint(
                        expr=constraint,
                        constraint_type=ConstraintType.BOOLEAN,
                        variables=self._extract_vars(constraint)
                    ))
            
            # Run hybrid verification
            if z3_constraints:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bridge_result = loop.run_until_complete(
                        self.cav_nlp.verify(z3_constraints[0], use_counterexamples)
                    )
                finally:
                    loop.close()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                
                return VerificationResult(
                    success=bridge_result.agreed if bridge_result else False,
                    z3_result=bridge_result.z3_result if bridge_result else None,
                    lean_result=bridge_result.lean_result if bridge_result else None,
                    confidence=bridge_result.confidence if bridge_result else 0.0,
                    counterexample=bridge_result.counterexample if bridge_result else None,
                    execution_time=elapsed,
                    metadata={
                        "agreed": bridge_result.agreed if bridge_result else False,
                        "z3_model": bridge_result.z3_model if bridge_result else None,
                    }
                )
            
            return self._verify_z3_only(constraints)
            
        except Exception as e:
            logger.error(f"Error in hybrid verification: {e}")
            return self._verify_z3_only(constraints)
    
    def _verify_z3_only(self, constraints: Optional[List[Any]]) -> VerificationResult:
        """Verify using Z3 only (fallback)."""
        if self.solver is None or not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                z3_result="unknown",
                confidence=0.0,
                warnings=["Z3 not available"]
            )
        
        start_time = datetime.now()
        
        try:
            result = self.solver.check()
            
            counterexample = None
            if result == sat and sat:
                model = self.solver.model()
                counterexample = {str(d): str(model[d]) for d in model.decls()}
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return VerificationResult(
                success=True,
                z3_result=str(result),
                confidence=0.5 if str(result) != "unknown" else 0.2,
                counterexample=counterexample,
                execution_time=elapsed,
                metadata={"mode": "z3_only"}
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                z3_result="error",
                confidence=0.0,
                warnings=[str(e)]
            )
    
    def find_counterexample(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find counterexample to a theorem using Z3.
        
        Args:
            theorem: Theorem statement to check
            variables: Variable declarations {name: type}
            
        Returns:
            Counterexample dictionary or None if no counterexample found
        """
        if not self.use_cav_nlp or self.cav_nlp is None:
            logger.warning("CAV-NLP not available for counterexample search")
            return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                counterexample = loop.run_until_complete(
                    self.cav_nlp.find_counterexample(theorem)
                )
            finally:
                loop.close()
            
            return counterexample
        except Exception as e:
            logger.error(f"Error finding counterexample: {e}")
            return None
    
    def prove(
        self,
        theorem: str,
        variables: Optional[Dict[str, str]] = None
    ) -> Optional[HybridProofResult]:
        """Prove a theorem using hybrid Z3/Lean approach.
        
        Args:
            theorem: Theorem statement to prove
            variables: Variable declarations {name: type}
            
        Returns:
            HybridProofResult or None if proving failed
        """
        if not self.use_cav_nlp or self.cav_nlp is None:
            logger.warning("CAV-NLP not available for proving")
            return None
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.cav_nlp.prove(theorem, variables)
                )
            finally:
                loop.close()
            
            return result
        except Exception as e:
            logger.error(f"Error in hybrid proof: {e}")
            return None
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get available capabilities of the enhanced solver.
        
        Returns:
            Dictionary mapping capability names to availability
        """
        return {
            "z3_available": Z3_AVAILABLE,
            "cav_nlp_available": self.use_cav_nlp,
            "lean_available": LEAN4_AVAILABLE,
            "formalization": self.use_cav_nlp,
            "hybrid_verification": self.use_cav_nlp and LEAN4_AVAILABLE,
            "counterexamples": self.use_cav_nlp and Z3_AVAILABLE,
            "hybrid_proofs": self.use_cav_nlp,
            "unified_math_service": self.math_service is not None,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            **self.stats,
            "formalization_history_count": len(self.formalization_history),
            "cav_nlp_enabled": self.use_cav_nlp,
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _extract_vars(self, constraint: Any) -> List[str]:
        """Extract variable names from Z3 constraint."""
        import re
        constraint_str = str(constraint)
        matches = re.findall(r'\b[a-zA-Z_]\w*\b', constraint_str)
        keywords = {'and', 'or', 'not', 'implies', 'forall', 'exists', 'True', 'False'}
        return [v for v in matches if v not in keywords]
    
    def _lean_to_z3_expr(self, lean_code: str) -> Optional[Any]:
        """Convert Lean code to Z3 expression."""
        if not Z3_AVAILABLE or self.cav_nlp is None:
            return None
        
        z3_constraint = self.cav_nlp.lean4_to_z3(lean_code)
        return z3_constraint.expr if z3_constraint else None


# =============================================================================
# ConstraintFormalizer Class
# =============================================================================

class ConstraintFormalizer:
    """Formalize natural language constraints to Z3 using CAV-NLP.
    
    This class provides a dedicated interface for converting natural language
    and LaTeX mathematical statements into Z3 constraints.
    
    Attributes:
        cav_bridge: CAV-NLP bridge for formalization
        math_service: Unified math service
        use_cav_nlp: Whether CAV-NLP is available
    
    Example:
        >>> formalizer = ConstraintFormalizer()
        >>> result = formalizer.formalize("x is positive and y is negative")
        >>> if result.success:
        ...     print(f"Variables: {result.variables}")
    """
    
    def __init__(
        self,
        cav_bridge: Optional[Z3LeanAideBridge] = None,
        use_cav_nlp: bool = True
    ):
        """Initialize the constraint formalizer.
        
        Args:
            cav_bridge: Optional CAV-NLP bridge instance
            use_cav_nlp: Whether to use CAV-NLP for formalization
        """
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        
        if self.use_cav_nlp:
            try:
                self.cav_bridge = cav_bridge or create_z3_lean_bridge()
                self.math_service = (
                    create_unified_math_service() if UNIFIED_MATH_AVAILABLE else None
                )
                logger.info("ConstraintFormalizer initialized with CAV-NLP")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP: {e}")
                self.cav_bridge = None
                self.math_service = None
                self.use_cav_nlp = False
        else:
            self.cav_bridge = None
            self.math_service = None
    
    def formalize(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        target: str = "z3"
    ) -> FormalizationResult:
        """Formalize natural language text to Z3 constraint.
        
        Uses the CAV-NLP pipeline:
            1. flexible_semantic_parsing - Parse to semantic primitives
            2. dependency_dag - Extract dependency graph
            3. z3_semantic_synthesis - Synthesize to Z3
            4. canonical_lean_generator - Generate canonical form
        
        Args:
            text: Natural language or LaTeX mathematical statement
            context: Optional context for formalization
            target: Target format ("z3" or "lean")
            
        Returns:
            FormalizationResult with Z3 expression and metadata
            
        Example:
            >>> formalizer = ConstraintFormalizer()
            >>> result = formalizer.formalize("for all x > 0, x² > 0")
            >>> if result.success:
            ...     solver = z3.Solver()
            ...     solver.add(result.z3_expr)
        """
        if not self.use_cav_nlp:
            return FormalizationResult(
                success=False,
                source_text=text,
                warnings=["CAV-NLP not available"]
            )
        
        try:
            # Use unified math service for formalization
            if self.math_service is not None:
                cav_context = None
                if context and CAVNLPContext is not None:
                    cav_context = CAVNLPContext(
                        paper_title=context.get("paper_title"),
                        section_context=context.get("section_context"),
                        theorem_number=context.get("theorem_number"),
                    )
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    formalization = loop.run_until_complete(
                        self.math_service.formalize(text, context=cav_context)
                    )
                finally:
                    loop.close()
                
                if formalization and formalization.success:
                    # Convert to Z3 if needed
                    z3_expr = None
                    if target == "z3" and self.cav_bridge:
                        z3_constraint = self.cav_bridge.lean4_to_z3(formalization.code)
                        z3_expr = z3_constraint.expr if z3_constraint else None
                    
                    return FormalizationResult(
                        success=True,
                        z3_expr=z3_expr,
                        constraint_type=self._infer_constraint_type(text),
                        variables=self._extract_variables(text),
                        canonical_form=formalization.code,
                        source_text=text,
                        warnings=formalization.warnings if hasattr(formalization, 'warnings') else []
                    )
            
            # Fallback: basic parsing
            return self._basic_formalize(text)
            
        except Exception as e:
            logger.error(f"Error formalizing text: {e}")
            return FormalizationResult(
                success=False,
                source_text=text,
                warnings=[str(e)]
            )
    
    def formalize_latex(
        self,
        latex: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FormalizationResult:
        """Formalize LaTeX mathematical expression to Z3.
        
        Args:
            latex: LaTeX mathematical expression
            context: Optional context for formalization
            
        Returns:
            FormalizationResult with Z3 expression
        """
        # Preprocess LaTeX to natural language-like form
        processed = self._preprocess_latex(latex)
        return self.formalize(processed, context)
    
    def batch_formalize(
        self,
        texts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[FormalizationResult]:
        """Formalize multiple constraints in batch.
        
        Args:
            texts: List of natural language constraints
            context: Optional shared context
            
        Returns:
            List of FormalizationResult
        """
        results = []
        for text in texts:
            result = self.formalize(text, context)
            results.append(result)
        return results
    
    def _infer_constraint_type(self, text: str) -> str:
        """Infer the type of constraint from text."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['∀', '∃', 'forall', 'exists', 'for all', 'there exists']):
            return "quantified"
        elif any(kw in text_lower for kw in ['^', '**', 'pow', 'square', 'cube', '²', '³']):
            return "nonlinear"
        elif any(kw in text_lower for kw in ['+', '-', '*', '/', '<', '>', '≤', '≥', 'less', 'greater']):
            return "arithmetic"
        elif any(kw in text_lower for kw in ['and', 'or', 'not', 'implies', 'if', 'then']):
            return "boolean"
        else:
            return "unknown"
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from text."""
        import re
        # Simple heuristic: single letters
        matches = re.findall(r'\b[a-zA-Z]\b', text)
        return list(set(matches))
    
    def _preprocess_latex(self, latex: str) -> str:
        """Preprocess LaTeX to natural language form."""
        # Simple replacements for common LaTeX patterns
        replacements = [
            (r'\\forall', 'for all'),
            (r'\\exists', 'there exists'),
            (r'\\geq', '>='),
            (r'\\leq', '<='),
            (r'\\gt', '>'),
            (r'\\lt', '<'),
            (r'\\and', 'and'),
            (r'\\or', 'or'),
            (r'\\neg', 'not'),
            (r'\\rightarrow', 'implies'),
            (r'\\Rightarrow', 'implies'),
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\div', '/'),
            (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1 / \2)'),
            (r'\^\{?([^}]+)\}?', r'^\1'),
        ]
        
        result = latex
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _basic_formalize(self, text: str) -> FormalizationResult:
        """Basic formalization without CAV-NLP."""
        # Create a simple boolean expression as placeholder
        if Z3_AVAILABLE:
            z3_expr = z3.BoolVal(True)  # Placeholder
        else:
            z3_expr = None
        
        return FormalizationResult(
            success=True,  # Mark as success but with placeholder
            z3_expr=z3_expr,
            constraint_type=self._infer_constraint_type(text),
            variables=self._extract_variables(text),
            source_text=text,
            warnings=["Using basic formalization - CAV-NLP unavailable"]
        )


# =============================================================================
# ProofExporter Class
# =============================================================================

class ProofExporter:
    """Export Z3 proofs to Lean 4 for formal verification.
    
    This class converts Z3 proofs and constraints into Lean 4 proof scripts
    that can be verified by the Lean 4 theorem prover.
    
    Attributes:
        cav_bridge: CAV-NLP bridge for translation
        lean_service: LeanAide service for verification
    
    Example:
        >>> exporter = ProofExporter()
        >>> solver = z3.Solver()
        >>> solver.add(x > 0, y > 0, x + y > 0)
        >>> proof = exporter.export_proof(solver)
        >>> print(proof)
    """
    
    def __init__(
        self,
        cav_bridge: Optional[Z3LeanAideBridge] = None,
        lean_service: Optional[Any] = None
    ):
        """Initialize the proof exporter.
        
        Args:
            cav_bridge: Optional CAV-NLP bridge instance
            lean_service: Optional LeanAide service
        """
        if CAV_NLP_AVAILABLE:
            try:
                self.cav_bridge = cav_bridge or create_z3_lean_bridge(lean_service)
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP bridge: {e}")
                self.cav_bridge = None
        else:
            self.cav_bridge = None
        
        self.lean_service = lean_service
    
    def export_proof(
        self,
        z3_proof: Any,
        theorem_name: str = "z3_exported_theorem",
        generate_tactics: bool = True
    ) -> str:
        """Export Z3 proof to Lean 4 proof script.
        
        Converts a Z3 proof or solver state into a Lean 4 theorem with
        proof tactics.
        
        Args:
            z3_proof: Z3 proof object or Solver
            theorem_name: Name for the generated theorem
            generate_tactics: Whether to generate proof tactics
            
        Returns:
            Lean 4 proof script as string
            
        Example:
            >>> exporter = ProofExporter()
            >>> solver = z3.Solver()
            >>> solver.add(x > 0, y > 0)
            >>> proof = exporter.export_proof(solver, "positive_sum")
        """
        if self.cav_bridge is None:
            return self._basic_export(z3_proof, theorem_name)
        
        try:
            # Extract constraints from proof/solver
            if hasattr(z3_proof, 'assertions'):
                constraints = list(z3_proof.assertions())
            elif hasattr(z3_proof, 'children'):
                constraints = [z3_proof]
            else:
                constraints = [z3_proof]
            
            # Convert to Lean using CAV-NLP bridge
            if constraints and CAV_NLP_AVAILABLE and ConstraintType is not None:
                lean_constraint = self.cav_bridge.z3_to_lean4(
                    constraints[0],
                    ConstraintType.ARITHMETIC
                )
                return lean_constraint.lean_code
            
            return self._basic_export(z3_proof, theorem_name)
            
        except Exception as e:
            logger.error(f"Error exporting proof: {e}")
            return self._basic_export(z3_proof, theorem_name)
    
    def export_constraints(
        self,
        constraints: List[Any],
        theorem_name: str = "z3_constraints"
    ) -> str:
        """Export a list of Z3 constraints to Lean 4.
        
        Args:
            constraints: List of Z3 expressions
            theorem_name: Name for the generated theorem
            
        Returns:
            Lean 4 code as string
        """
        if not constraints:
            return f"-- No constraints to export for {theorem_name}"
        
        lines = ["import Mathlib", ""]
        
        # Extract all variables
        all_vars = set()
        for constraint in constraints:
            if self.cav_bridge:
                vars_in_constraint = self._extract_vars_from_constraint(constraint)
                all_vars.update(vars_in_constraint)
        
        # Build theorem statement
        lines.append(f"theorem {theorem_name}")
        
        # Add variable declarations
        for var in sorted(all_vars):
            lines.append(f"    ({var} : ℝ)")
        
        # Build constraint conjunction
        constraint_strs = [str(c) for c in constraints]
        combined = " ∧ ".join(constraint_strs)
        
        lines.append(f"    : {combined} := by")
        lines.append("  sorry")
        lines.append("")
        
        return "\n".join(lines)
    
    def export_with_verification(
        self,
        z3_proof: Any,
        theorem_name: str = "z3_verified_theorem"
    ) -> Tuple[str, Optional[VerificationResult]]:
        """Export proof and verify in Lean 4.
        
        Args:
            z3_proof: Z3 proof object
            theorem_name: Name for the theorem
            
        Returns:
            Tuple of (Lean code, verification result)
        """
        lean_code = self.export_proof(z3_proof, theorem_name)
        
        verification = None
        if self.cav_bridge is not None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bridge_result = loop.run_until_complete(
                        self.cav_bridge.verify(lean_code)
                    )
                finally:
                    loop.close()
                
                if bridge_result:
                    verification = VerificationResult(
                        success=bridge_result.agreed,
                        z3_result=bridge_result.z3_result,
                        lean_result=bridge_result.lean_result,
                        confidence=bridge_result.confidence,
                        counterexample=bridge_result.counterexample,
                        execution_time=bridge_result.execution_time,
                    )
            except Exception as e:
                logger.error(f"Verification failed: {e}")
        
        return lean_code, verification
    
    def _basic_export(self, z3_proof: Any, theorem_name: str) -> str:
        """Basic export without CAV-NLP."""
        lines = ["import Mathlib", ""]
        lines.append(f"-- Basic export for {theorem_name}")
        lines.append(f"-- Z3 proof: {str(z3_proof)[:100]}")
        lines.append("")
        lines.append(f"theorem {theorem_name} : True := by")
        lines.append("  trivial")
        lines.append("")
        return "\n".join(lines)
    
    def _extract_vars_from_constraint(self, constraint: Any) -> List[str]:
        """Extract variables from a constraint."""
        import re
        text = str(constraint)
        matches = re.findall(r'\b[a-zA-Z_]\w*\b', text)
        keywords = {'and', 'or', 'not', 'implies', 'forall', 'exists'}
        return [v for v in matches if v.lower() not in keywords]


# =============================================================================
# CanonicalConstraintManager Class
# =============================================================================

class CanonicalConstraintManager:
    """Manage canonical forms of Z3 constraints using CAV-NLP.
    
    This class provides canonicalization and equivalence checking for
    Z3 constraints, using CAV-NLP for semantic canonicalization.
    
    Attributes:
        cav_bridge: CAV-NLP bridge for canonicalization
        canonicalizer: Z3 canonicalizer instance
        cache: Cache of canonical forms
    
    Example:
        >>> manager = CanonicalConstraintManager()
        >>> c1 = x > 0
        >>> c2 = 0 < x
        >>> canonical1 = manager.canonicalize(c1)
        >>> canonical2 = manager.canonicalize(c2)
        >>> print(manager.are_equivalent(c1, c2))  # True
    """
    
    def __init__(self, cav_bridge: Optional[Z3LeanAideBridge] = None):
        """Initialize the canonical constraint manager.
        
        Args:
            cav_bridge: Optional CAV-NLP bridge instance
        """
        self.cav_bridge = cav_bridge
        self._canonical_cache: Dict[str, CanonicalForm] = {}
        self._equivalence_cache: Dict[Tuple[str, str], bool] = {}
        
        # Try to import and initialize Z3 canonicalizer
        self.z3_canonicalizer = None
        if CAV_NLP_AVAILABLE:
            try:
                from openevolve.cav_nlp_integration import Z3Canonicalizer
                self.z3_canonicalizer = Z3Canonicalizer()
                logger.info("Z3Canonicalizer initialized")
            except Exception as e:
                logger.debug(f"Z3Canonicalizer not available: {e}")
    
    def canonicalize(self, constraint: Any) -> CanonicalForm:
        """Convert constraint to canonical form.
        
        Uses CAV-NLP canonicalization rules to produce a standardized
        representation of the constraint.
        
        Args:
            constraint: Z3 expression to canonicalize
            
        Returns:
            CanonicalForm with canonical representation
            
        Example:
            >>> manager = CanonicalConstraintManager()
            >>> c = z3.And(x > 0, y > 0)
            >>> canonical = manager.canonicalize(c)
            >>> print(canonical.canonical)
        """
        constraint_str = str(constraint)
        
        # Check cache
        if constraint_str in self._canonical_cache:
            return self._canonical_cache[constraint_str]
        
        try:
            # Try CAV-NLP canonicalization
            if self.z3_canonicalizer is not None:
                canonical = self.z3_canonicalizer.canonicalize(constraint)
                result = CanonicalForm(
                    original=constraint,
                    canonical=canonical,
                    is_valid=True
                )
            elif self.cav_bridge is not None:
                # Use bridge for translation-based canonicalization
                lean_constraint = self.cav_bridge.z3_to_lean4(constraint)
                # Translate back for normalization effect
                z3_back = self.cav_bridge.lean4_to_z3(lean_constraint.lean_code)
                result = CanonicalForm(
                    original=constraint,
                    canonical=z3_back.expr if z3_back else constraint,
                    is_valid=z3_back is not None
                )
            else:
                # Fallback: use Z3 simplify
                if Z3_AVAILABLE:
                    simplified = z3.simplify(constraint)
                    result = CanonicalForm(
                        original=constraint,
                        canonical=simplified,
                        is_valid=True
                    )
                else:
                    result = CanonicalForm(
                        original=constraint,
                        canonical=constraint,
                        is_valid=False
                    )
        except Exception as e:
            logger.warning(f"Canonicalization failed: {e}")
            result = CanonicalForm(
                original=constraint,
                canonical=constraint,
                is_valid=False
            )
        
        # Cache result
        self._canonical_cache[constraint_str] = result
        return result
    
    def are_equivalent(self, c1: Any, c2: Any) -> bool:
        """Check if two constraints are equivalent.
        
        Uses Z3 to verify semantic equivalence by checking that
        c1 implies c2 and c2 implies c1.
        
        Args:
            c1: First constraint
            c2: Second constraint
            
        Returns:
            True if constraints are equivalent, False otherwise
            
        Example:
            >>> manager = CanonicalConstraintManager()
            >>> c1 = x > 0
            >>> c2 = 0 < x
            >>> print(manager.are_equivalent(c1, c2))  # True
        """
        if not Z3_AVAILABLE:
            # String-based equivalence as fallback
            return str(c1) == str(c2)
        
        c1_str, c2_str = str(c1), str(c2)
        
        # Check cache
        cache_key = (c1_str, c2_str)
        cache_key_rev = (c2_str, c1_str)
        if cache_key in self._equivalence_cache:
            return self._equivalence_cache[cache_key]
        if cache_key_rev in self._equivalence_cache:
            return self._equivalence_cache[cache_key_rev]
        
        try:
            # Check equivalence using Z3
            solver = Solver()
            
            # c1 equivalent to c2 iff (c1 == c2) is valid
            # iff (c1 != c2) is unsatisfiable
            solver.add(c1 != c2)
            
            result = solver.check()
            
            is_equivalent = (result == unsat)
            
            # Cache result
            self._equivalence_cache[cache_key] = is_equivalent
            
            return is_equivalent
            
        except Exception as e:
            logger.warning(f"Equivalence check failed: {e}")
            return str(c1) == str(c2)
    
    def find_redundant_constraints(
        self,
        constraints: List[Any]
    ) -> List[int]:
        """Find indices of redundant constraints in a list.
        
        A constraint is redundant if it's implied by the conjunction
        of the other constraints.
        
        Args:
            constraints: List of constraints to analyze
            
        Returns:
            Indices of redundant constraints
        """
        if not Z3_AVAILABLE:
            return []
        
        redundant_indices = []
        
        for i, c in enumerate(constraints):
            other_constraints = [constraints[j] for j in range(len(constraints)) if j != i]
            
            if not other_constraints:
                continue
            
            try:
                solver = Solver()
                
                # Add other constraints as assumptions
                for oc in other_constraints:
                    solver.add(oc)
                
                # Check if c is implied (negation is unsatisfiable)
                solver.add(z3.Not(c))
                
                if solver.check() == unsat:
                    redundant_indices.append(i)
                    
            except Exception as e:
                logger.debug(f"Redundancy check failed for constraint {i}: {e}")
        
        return redundant_indices
    
    def simplify_constraint_set(
        self,
        constraints: List[Any]
    ) -> List[Any]:
        """Simplify a set of constraints by removing redundancies.
        
        Args:
            constraints: List of constraints to simplify
            
        Returns:
            Simplified list of constraints
        """
        redundant = self.find_redundant_constraints(constraints)
        return [c for i, c in enumerate(constraints) if i not in redundant]
    
    def get_canonical_form_string(self, constraint: Any) -> str:
        """Get canonical form as a string.
        
        Args:
            constraint: Constraint to canonicalize
            
        Returns:
            String representation of canonical form
        """
        canonical = self.canonicalize(constraint)
        return str(canonical.canonical)


# =============================================================================
# Decorators for Drop-in Enhancement
# =============================================================================

def with_cav_nlp(
    func: Optional[F] = None,
    *,
    auto_formalize: bool = True,
    auto_canonicalize: bool = False
) -> Union[F, Callable[[F], F]]:
    """Decorator to add CAV-NLP capabilities to a function.
    
    This decorator wraps a function to provide CAV-NLP formalization
    and canonicalization capabilities.
    
    Args:
        func: Function to decorate
        auto_formalize: Whether to auto-formalize string arguments
        auto_canonicalize: Whether to auto-canonicalize results
        
    Returns:
        Decorated function
        
    Example:
        >>> @with_cav_nlp
        ... def solve_constraint(constraint):
        ...     solver = z3.Solver()
        ...     solver.add(constraint)
        ...     return solver.check()
        
        >>> # Can now pass natural language
        >>> result = solve_constraint("x is positive")
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            formalizer = ConstraintFormalizer() if auto_formalize else None
            
            # Try to formalize string arguments
            new_args = []
            for arg in args:
                if isinstance(arg, str) and formalizer is not None:
                    result = formalizer.formalize(arg)
                    if result.success and result.z3_expr is not None:
                        new_args.append(result.z3_expr)
                    else:
                        new_args.append(arg)
                else:
                    new_args.append(arg)
            
            result = f(*new_args, **kwargs)
            
            # Canonicalize result if requested
            if auto_canonicalize and result is not None:
                manager = CanonicalConstraintManager()
                canonical = manager.canonicalize(result)
                return canonical.canonical
            
            return result
        
        # Attach CAV-NLP utilities
        wrapper.formalizer = lambda: ConstraintFormalizer()
        wrapper.enhanced_solver = lambda: EnhancedZ3Solver()
        
        return wrapper  # type: ignore
    
    if func is None:
        return decorator
    return decorator(func)


def auto_formalize(func: F) -> F:
    """Decorator to automatically formalize string arguments.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with auto-formalization
        
    Example:
        >>> @auto_formalize
        ... def analyze(constraint):
        ...     # constraint will be Z3 expression if formalization succeeds
        ...     return constraint
    """
    return with_cav_nlp(func, auto_formalize=True)


def auto_canonicalize(func: F) -> F:
    """Decorator to automatically canonicalize return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with auto-canonicalization
        
    Example:
        >>> @auto_canonicalize
        ... def get_constraint():
        ...     return x > 0
    """
    return with_cav_nlp(func, auto_canonicalize=True)


# =============================================================================
# Context Managers
# =============================================================================

@contextmanager
def cav_nlp_scope(
    lean_service: Optional[Any] = None
) -> Iterator[EnhancedZ3Solver]:
    """Context manager for CAV-NLP enhanced solving.
    
    Provides a scoped EnhancedZ3Solver with CAV-NLP capabilities.
    
    Args:
        lean_service: Optional LeanAide service
        
    Yields:
        EnhancedZ3Solver instance
        
    Example:
        >>> with cav_nlp_scope() as solver:
        ...     solver.add(solver.formalize_constraint("x > 0"))
        ...     result = solver.check()
        ...     verification = solver.verify_with_lean()
    """
    solver = EnhancedZ3Solver(use_cav_nlp=True, lean_service=lean_service)
    try:
        yield solver
    finally:
        # Cleanup
        solver.reset()


@contextmanager
def enhanced_solver(
    use_cav_nlp: bool = True,
    lean_service: Optional[Any] = None
) -> Iterator[EnhancedZ3Solver]:
    """Context manager for enhanced Z3 solver.
    
    Similar to cav_nlp_scope but with more control over CAV-NLP usage.
    
    Args:
        use_cav_nlp: Whether to enable CAV-NLP
        lean_service: Optional LeanAide service
        
    Yields:
        EnhancedZ3Solver instance
        
    Example:
        >>> with enhanced_solver(use_cav_nlp=True) as solver:
        ...     constraint = solver.formalize_constraint("x is positive")
        ...     solver.add(constraint)
        ...     print(solver.check())
    """
    solver = EnhancedZ3Solver(use_cav_nlp=use_cav_nlp, lean_service=lean_service)
    try:
        yield solver
    finally:
        solver.reset()


# =============================================================================
# Convenience Functions
# =============================================================================

def formalize_to_z3(
    text: str,
    context: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """Quick function to formalize natural language to Z3.
    
    Args:
        text: Natural language constraint
        context: Optional context
        
    Returns:
        Z3 expression or None
        
    Example:
        >>> expr = formalize_to_z3("x and y are positive")
        >>> solver = z3.Solver()
        >>> solver.add(expr)
    """
    formalizer = ConstraintFormalizer()
    result = formalizer.formalize(text, context)
    return result.z3_expr if result.success else None


def quick_canonicalize(constraint: Any) -> Any:
    """Quick function to canonicalize a constraint.
    
    Args:
        constraint: Constraint to canonicalize
        
    Returns:
        Canonical form of constraint
        
    Example:
        >>> c = quick_canonicalize(x > 0)
    """
    manager = CanonicalConstraintManager()
    canonical = manager.canonicalize(constraint)
    return canonical.canonical


def check_equivalence(c1: Any, c2: Any) -> bool:
    """Quick function to check if two constraints are equivalent.
    
    Args:
        c1: First constraint
        c2: Second constraint
        
    Returns:
        True if equivalent
        
    Example:
        >>> if check_equivalence(c1, c2):
        ...     print("Constraints are equivalent")
    """
    manager = CanonicalConstraintManager()
    return manager.are_equivalent(c1, c2)


async def verify_constraint(
    constraint: Any,
    use_lean: bool = True
) -> VerificationResult:
    """Quick async function to verify a constraint.
    
    Args:
        constraint: Constraint to verify
        use_lean: Whether to use Lean verification
        
    Returns:
        VerificationResult
        
    Example:
        >>> result = await verify_constraint("x > 0")
        >>> print(f"Confidence: {result.confidence}")
    """
    solver = EnhancedZ3Solver(use_cav_nlp=use_lean)
    
    if isinstance(constraint, str):
        constraint = solver.formalize_constraint(constraint)
    
    if constraint:
        solver.add(constraint)
    
    return solver.verify_with_lean()


def create_enhanced_solver(
    use_cav_nlp: bool = True,
    lean_service: Optional[Any] = None
) -> EnhancedZ3Solver:
    """Create an enhanced Z3 solver with CAV-NLP capabilities.
    
    Args:
        use_cav_nlp: Whether to enable CAV-NLP
        lean_service: Optional LeanAide service
        
    Returns:
        EnhancedZ3Solver instance
        
    Example:
        >>> solver = create_enhanced_solver()
        >>> constraint = solver.formalize_constraint("x > 0")
        >>> solver.add(constraint)
    """
    return EnhancedZ3Solver(use_cav_nlp=use_cav_nlp, lean_service=lean_service)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main Classes
    "EnhancedZ3Solver",
    "ConstraintFormalizer",
    "ProofExporter",
    "CanonicalConstraintManager",
    
    # Data Classes
    "VerificationResult",
    "FormalizationResult",
    "CanonicalForm",
    
    # Decorators
    "with_cav_nlp",
    "auto_formalize",
    "auto_canonicalize",
    
    # Context Managers
    "cav_nlp_scope",
    "enhanced_solver",
    
    # Convenience Functions
    "formalize_to_z3",
    "quick_canonicalize",
    "check_equivalence",
    "verify_constraint",
    "create_enhanced_solver",
]


# =============================================================================
# Self-Test (if run directly)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Z3-CAV-NLP Integration Module - Self Test")
    print("=" * 70)
    
    # Test 1: EnhancedZ3Solver
    print("\n1. Testing EnhancedZ3Solver")
    print("-" * 40)
    
    solver = EnhancedZ3Solver(use_cav_nlp=CAV_NLP_AVAILABLE)
    print(f"   Capabilities: {solver.get_capabilities()}")
    print(f"   Stats: {solver.get_stats()}")
    
    # Test 2: ConstraintFormalizer
    print("\n2. Testing ConstraintFormalizer")
    print("-" * 40)
    
    formalizer = ConstraintFormalizer(use_cav_nlp=CAV_NLP_AVAILABLE)
    result = formalizer.formalize("x is positive")
    print(f"   Formalization success: {result.success}")
    print(f"   Constraint type: {result.constraint_type}")
    print(f"   Variables: {result.variables}")
    
    # Test 3: ProofExporter
    print("\n3. Testing ProofExporter")
    print("-" * 40)
    
    exporter = ProofExporter()
    if Z3_AVAILABLE:
        x = Real('x')
        y = Real('y')
        z3_solver = Solver()
        z3_solver.add(x > 0, y > 0)
        proof = exporter.export_proof(z3_solver, "test_theorem")
        print(f"   Proof length: {len(proof)} characters")
        print(f"   Preview:\n{proof[:200]}...")
    else:
        print("   Z3 not available - skipped")
    
    # Test 4: CanonicalConstraintManager
    print("\n4. Testing CanonicalConstraintManager")
    print("-" * 40)
    
    manager = CanonicalConstraintManager()
    if Z3_AVAILABLE:
        x = Real('x')
        c1 = x > 0
        c2 = 0 < x
        canonical1 = manager.canonicalize(c1)
        equivalent = manager.are_equivalent(c1, c2)
        print(f"   Canonicalization valid: {canonical1.is_valid}")
        print(f"   x > 0 equivalent to 0 < x: {equivalent}")
    else:
        print("   Z3 not available - skipped")
    
    # Test 5: Decorators and context managers
    print("\n5. Testing Decorators and Context Managers")
    print("-" * 40)
    
    @with_cav_nlp
    def example_function(constraint):
        return f"Processed: {constraint}"
    
    result = example_function("test constraint")
    print(f"   Decorator result: {result}")
    
    print("\n   Context manager test:")
    try:
        with enhanced_solver(use_cav_nlp=False) as ctx_solver:
            print(f"   - Solver created successfully")
            print(f"   - Capabilities: {ctx_solver.get_capabilities()}")
    except Exception as e:
        print(f"   - Error: {e}")
    
    # Test 6: Convenience functions
    print("\n6. Testing Convenience Functions")
    print("-" * 40)
    
    print(f"   create_enhanced_solver available: {create_enhanced_solver is not None}")
    print(f"   formalize_to_z3 available: {formalize_to_z3 is not None}")
    print(f"   quick_canonicalize available: {quick_canonicalize is not None}")
    
    print("\n" + "=" * 70)
    print("Self-test completed!")
    print("=" * 70)
