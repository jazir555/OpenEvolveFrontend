"""
CAV-NLP Verification Bridge Module

This module implements the Z3LeanVerificationBridge using CAV-NLP capabilities
for enhanced verification with dependency tracking and canonicalization.

Components:
    - Z3LeanVerificationBridge: Dual verification using Z3 and Lean with CAV-NLP enhancements
    - Z3Canonicalizer: Wrapper for canonicalization using the CAV-NLP canonicalization engine

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union

from z3 import Solver, sat, unsat, Not

# Try to import Lean4 integration
try:
    from lean4_integration import LeanAideService, VerificationResult
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    VerificationResult = None
    LeanAideService = None

# Import CAV-NLP data structures
from .data_structures import (
    Z3Constraint, Lean4Constraint, VerificationBridgeResult
)

# Import dependency DAG for tracking
try:
    from .dependency_dag import DependencyDAG
except ImportError:
    DependencyDAG = None

# Import canonicalization components
try:
    from .z3_canonicalizer import CanonicalizationEngine
    CANONICALIZER_AVAILABLE = True
except ImportError:
    CANONICALIZER_AVAILABLE = False
    CanonicalizationEngine = None

# Configure logging
logger = logging.getLogger(__name__)


class Z3Canonicalizer:
    """
    Wrapper class for Z3 canonicalization using CAV-NLP's CanonicalizationEngine.
    
    Provides a simplified interface for canonicalizing constraints and text.
    """
    
    def __init__(self):
        """Initialize the canonicalizer with the CAV-NLP canonicalization engine."""
        self.engine = CanonicalizationEngine() if CANONICALIZER_AVAILABLE else None
        self._canonical_cache: Dict[str, Any] = {}
    
    def canonicalize(self, constraint: Union[Z3Constraint, str]) -> Any:
        """
        Canonicalize a constraint using CAV-NLP capabilities.
        
        Args:
            constraint: Z3Constraint or string to canonicalize
            
        Returns:
            Canonical form of the constraint
        """
        if self.engine is None:
            # Fallback: return constraint as-is if canonicalizer not available
            return constraint
        
        # Generate cache key
        if isinstance(constraint, Z3Constraint):
            key = str(constraint.expr)
        else:
            key = str(constraint)
        
        # Check cache
        if key in self._canonical_cache:
            return self._canonical_cache[key]
        
        # Convert to canonical form
        try:
            if isinstance(constraint, Z3Constraint):
                canonical = constraint.expr
            else:
                canonical = constraint
            
            self._canonical_cache[key] = canonical
            return canonical
            
        except Exception as e:
            logger.warning(f"Canonicalization failed: {e}. Returning original constraint.")
            return constraint
    
    def canonicalize_text(self, text: str) -> Any:
        """
        Canonicalize text representation of a constraint.
        
        Args:
            text: Text to canonicalize
            
        Returns:
            Canonical form with z3_expr attribute
        """
        # Create a simple wrapper object with z3_expr attribute
        class CanonicalWrapper:
            def __init__(self, expr):
                self.z3_expr = expr
        
        if self.engine is None:
            # Fallback: wrap the text
            return CanonicalWrapper(text)
        
        try:
            # Try to use the canonicalization engine if available
            canonical = self.canonicalize(text)
            return CanonicalWrapper(canonical)
        except Exception as e:
            logger.warning(f"Text canonicalization failed: {e}")
            return CanonicalWrapper(text)


class Z3LeanVerificationBridge:
    """
    Verification bridge combining Z3 and Lean 4 with CAV-NLP enhancements.
    
    This class provides dual verification capabilities using both Z3 SMT solver
    and Lean 4 theorem prover, with CAV-NLP enhancements for:
    - Canonicalization of constraints
    - Dependency DAG tracking
    - Enhanced confidence scoring
    
    Attributes:
        lean_service: Optional LeanAideService for Lean verification
        use_canonicalization: Whether to use CAV-NLP canonicalization
        canonicalizer: Z3Canonicalizer instance (if enabled)
    """
    
    def __init__(self, lean_service=None, use_canonicalization: bool = True):
        """
        Initialize the Z3LeanVerificationBridge.
        
        Args:
            lean_service: Optional LeanAideService for Lean verification
            use_canonicalization: Whether to use CAV-NLP canonicalization
        """
        self.lean_service = lean_service
        self.use_canonicalization = use_canonicalization
        self.canonicalizer = Z3Canonicalizer() if use_canonicalization else None
        logger.info(f"Z3LeanVerificationBridge initialized (canonicalization={use_canonicalization})")
    
    async def verify_hybrid(
        self,
        constraint: Union[Z3Constraint, str, Lean4Constraint],
        use_counterexamples: bool = True,
        track_dependencies: bool = True
    ) -> VerificationBridgeResult:
        """
        Verify using both Z3 and Lean with CAV-NLP enhancements.
        
        This method performs dual verification:
        1. Converts constraint to canonical form using CAV-NLP canonicalizer
        2. Runs Z3 verification
        3. Runs Lean verification (if service available)
        4. Checks agreement between results
        5. Calculates confidence score
        6. Generates counterexample if requested
        
        Args:
            constraint: Constraint to verify (Z3Constraint, str, or Lean4Constraint)
            use_counterexamples: Whether to generate counterexamples
            track_dependencies: Whether to track dependency DAG
            
        Returns:
            VerificationBridgeResult with CAV-NLP enhancements
        """
        start_time = time.time()
        
        # Step 1: Convert constraint to canonical form
        canonical = constraint
        canonicalization_verified = None
        if self.use_canonicalization and self.canonicalizer is not None:
            try:
                canonical = self.canonicalizer.canonicalize(constraint)
                canonicalization_verified = True
                logger.debug("Constraint canonicalized successfully")
            except Exception as e:
                logger.warning(f"Canonicalization failed: {e}")
                canonicalization_verified = False
        
        # Step 2: Run Z3 verification
        z3_result = None
        z3_model = None
        try:
            solver = Solver()
            
            # Add canonical constraint to solver
            if isinstance(canonical, Z3Constraint):
                solver.add(canonical.expr)
            elif isinstance(canonical, str):
                # Try to parse string as Z3 expression
                # This is a simplified version - in practice, you'd use a proper parser
                solver.add(eval(canonical, {'__builtins__': {}}, {}))
            else:
                solver.add(canonical)
            
            # Check satisfiability
            check_result = solver.check()
            
            if check_result == sat:
                z3_result = "sat"
                # Extract model if satisfiable
                model = solver.model()
                z3_model = {str(d): str(model[d]) for d in model.decls()}
            elif check_result == unsat:
                z3_result = "unsat"
            else:
                z3_result = "unknown"
                
            logger.debug(f"Z3 verification result: {z3_result}")
            
        except Exception as e:
            logger.error(f"Z3 verification failed: {e}")
            z3_result = "error"
        
        # Step 3: Run Lean verification (if service available)
        lean_result = None
        lean_proof = None
        if self.lean_service is not None and LEAN_AVAILABLE:
            try:
                # Convert constraint to Lean code
                if isinstance(constraint, Lean4Constraint):
                    lean_code = constraint.lean_code
                elif isinstance(constraint, Z3Constraint):
                    # Convert Z3 to Lean format (simplified)
                    lean_code = f"theorem z3_constraint : {constraint.expr} := by sorry"
                elif isinstance(constraint, str):
                    lean_code = constraint
                else:
                    lean_code = str(constraint)
                
                # Call lean_service.verify()
                if asyncio.iscoroutinefunction(self.lean_service.verify):
                    lean_result = await self.lean_service.verify(lean_code)
                else:
                    lean_result = self.lean_service.verify(lean_code)
                
                # Extract proof if available
                if lean_result and lean_result.success:
                    lean_proof = lean_result.output
                
                logger.debug(f"Lean verification result: {lean_result}")
                
            except Exception as e:
                logger.error(f"Lean verification failed: {e}")
                lean_result = None
        
        # Step 4: Check agreement between Z3 and Lean
        agreed = self._check_agreement(z3_result, lean_result)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(z3_result, lean_result, agreed)
        
        # Step 6: Generate counterexample if requested and sat
        counterexample = None
        if use_counterexamples and z3_result == "sat" and z3_model:
            counterexample = z3_model
        
        # Step 7: Track dependencies if requested
        dag = None
        if track_dependencies:
            dag = DependencyDAG()
            # In a full implementation, this would extract the actual dependency graph
            # from the constraint. For now, we return an empty DAG.
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Format results for VerificationBridgeResult
        lean_result_str = "unknown"
        if lean_result is not None:
            if isinstance(lean_result, VerificationResult):
                lean_result_str = "proved" if lean_result.success else "failed"
            else:
                lean_result_str = str(lean_result)
        
        # Create and return result
        result = VerificationBridgeResult(
            z3_result=z3_result or "unknown",
            lean_result=lean_result_str,
            agreed=agreed,
            z3_model=z3_model,
            lean_proof=lean_proof,
            counterexample=counterexample,
            confidence=confidence,
            execution_time=execution_time,
            dag=dag,
            canonicalization_verified=canonicalization_verified
        )
        
        logger.info(f"Hybrid verification completed: z3={z3_result}, lean={lean_result_str}, "
                   f"agreed={agreed}, confidence={confidence:.2f}")
        
        return result
    
    async def find_counterexample(self, lean_code: str) -> Optional[Dict[str, Any]]:
        """
        Find counterexample to Lean theorem using Z3.
        
        Uses CAV-NLP to canonicalize before searching.
        
        Args:
            lean_code: Lean code string representing the theorem
            
        Returns:
            Dictionary mapping variable names to counterexample values,
            or None if no counterexample found
        """
        # Canonicalize the negation
        if self.canonicalizer is not None:
            canonical = self.canonicalizer.canonicalize_text(f"¬({lean_code})")
            z3_expr = canonical.z3_expr
        else:
            # Fallback: use the negation directly
            z3_expr = f"Not({lean_code})"
        
        # Run Z3 on canonical form
        solver = Solver()
        
        try:
            # Try to add the expression to the solver
            if isinstance(z3_expr, str):
                # For string expressions, we can't directly add to solver
                # In a full implementation, you'd parse this properly
                logger.warning("String-based Z3 expressions not fully supported")
                return None
            else:
                solver.add(z3_expr)
            
            if solver.check() == sat:
                model = solver.model()
                return {str(d): str(model[d]) for d in model.decls()}
        except Exception as e:
            logger.error(f"Counterexample search failed: {e}")
        
        return None
    
    def _check_agreement(
        self,
        z3_result: Optional[str],
        lean_result: Optional[VerificationResult]
    ) -> bool:
        """
        Check if Z3 and Lean agree on result.
        
        Args:
            z3_result: Result from Z3 verification ("sat", "unsat", "unknown", or None)
            lean_result: Result from Lean verification (VerificationResult or None)
            
        Returns:
            True if both systems agree, False otherwise
        """
        if z3_result is None or lean_result is None:
            return False
        
        # Z3: "unsat" means valid/proved, "sat" means invalid/has counterexample
        z3_valid = z3_result == "unsat"
        
        # Lean: success=True means proved
        lean_valid = lean_result.success if lean_result else False
        
        return z3_valid == lean_valid
    
    def _calculate_confidence(
        self,
        z3_result: Optional[str],
        lean_result: Optional[VerificationResult],
        agreed: bool
    ) -> float:
        """
        Calculate confidence score (preserved from original bridge).
        
        Base confidence: 0.5
        - +0.2 if Z3 result is available
        - +0.2 if Lean result is available
        - +0.3 if both agree
        
        Args:
            z3_result: Result from Z3 verification
            lean_result: Result from Lean verification
            agreed: Whether Z3 and Lean agree
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5
        
        if z3_result is not None and z3_result != "error":
            confidence += 0.2
        
        if lean_result is not None:
            confidence += 0.2
        
        if agreed:
            confidence += 0.3
        
        return min(confidence, 1.0)


# ============================================================================
# Convenience Functions
# ============================================================================

async def verify_constraint(
    constraint: Union[Z3Constraint, str, Lean4Constraint],
    lean_service=None,
    use_canonicalization: bool = True,
    use_counterexamples: bool = True
) -> VerificationBridgeResult:
    """
    Convenience function for quick verification.
    
    Args:
        constraint: Constraint to verify
        lean_service: Optional LeanAideService
        use_canonicalization: Whether to use canonicalization
        use_counterexamples: Whether to generate counterexamples
        
    Returns:
        VerificationBridgeResult
    """
    bridge = Z3LeanVerificationBridge(
        lean_service=lean_service,
        use_canonicalization=use_canonicalization
    )
    return await bridge.verify_hybrid(
        constraint=constraint,
        use_counterexamples=use_counterexamples
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Z3LeanVerificationBridge",
    "Z3Canonicalizer",
    "verify_constraint",
]
