"""
Quality Gate Z3 Verifier - Complete Implementation

Integrates formal verification into the quality assurance pipeline.
Author: OpenEvolve
Created: 2026-02-02
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Constraint,
        Z3ConstraintType, Z3Config, DigitalTwinSandbox
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

# CAV-NLP integration for enhanced verification
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# **LEAN INTEGRATION**: Real Lean theorem proving for standalone verification
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False


class VerificationType(Enum):
    """Types of formal verification."""
    SAFETY_INVARIANT = "safety_invariant"
    PERFORMANCE_GUARANTEE = "performance_guarantee"
    SECURITY_PROPERTY = "security_property"


class VerificationStatus(Enum):
    """Status of verification."""
    VERIFIED = "verified"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    """Result of formal verification."""
    success: bool
    status: VerificationStatus
    verification_type: VerificationType
    constraint_id: str
    violations: List[Dict[str, Any]] = field(default_factory=list)
    proof: Optional[str] = None
    counterexample: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class Z3QualityGateVerifier:
    """Formal verification for quality gates using Z3 with Lean integration."""
    
    def __init__(self, config=None):
        self.config = config or (Z3Config(timeout=60.0) if Z3_AVAILABLE else None)
        self.solver = Z3SolverEngine(self.config) if Z3_AVAILABLE and self.config else None
        self.prover = Z3TheoremProver(self.config) if Z3_AVAILABLE and self.config else None
        self.sandbox = DigitalTwinSandbox(self.solver) if Z3_AVAILABLE and self.solver else None
        
        # CAV-NLP enhanced verification
        self.use_cav_nlp = config.get("use_cav_nlp", True) if isinstance(config, dict) else True
        self.use_cav_nlp = self.use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
        
        # **LEAN INTEGRATION**: Lean theorem proving client
        self._lean_client = None
        if LEAN_AVAILABLE:
            try:
                self._lean_client = LeanAideClient()
                logger.info("LeanAide client initialized in Z3 verifier")
            except Exception as e:
                logger.warning(f"Failed to initialize LeanAide client: {e}")
    
    async def verify_with_lean(self, content: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover (standalone method).
        
        Args:
            content: Content to verify
            criteria: Verification criteria
            
        Returns:
            Dict with verification results
        """
        if not LEAN_AVAILABLE or not self._lean_client:
            return {"verified": False, "reason": "Lean unavailable"}
        
        try:
            formalized = await self._lean_client.translate_thm(content)
            result = await self._lean_client.verify(formalized)
            
            return {
                "verified": result.verified if hasattr(result, 'verified') else False,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                "proof": result.proof_code if hasattr(result, 'proof_code') else None
            }
        except Exception as e:
            logger.error(f"Lean verification error: {e}")
            return {"verified": False, "reason": str(e)}
    
    def verify_sop_safety(self, sop_steps, safety_invariants):
        """Verify SOP satisfies safety invariants using Digital Twin Sandbox."""
        start_time = time.time()
        
        if not Z3_AVAILABLE or not self.sandbox:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.SAFETY_INVARIANT,
                constraint_id="sop_safety",
                execution_time_ms=(time.time() - start_time) * 1000,
                recommendations=["Z3 not available"]
            )
        
        try:
            passed, counterexample = self.sandbox.verify_fix_with_invariants(
                "\n".join(sop_steps),
                safety_invariants
            )
            
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED if passed else VerificationStatus.VIOLATED,
                verification_type=VerificationType.SAFETY_INVARIANT,
                constraint_id="sop_safety",
                violations=[{"counterexample": counterexample}] if counterexample else [],
                counterexample=counterexample,
                execution_time_ms=(time.time() - start_time) * 1000,
                recommendations=["SOP satisfies safety invariants"] if passed else ["Review SOP for safety violations"]
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.SAFETY_INVARIANT,
                constraint_id="sop_safety",
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_performance_guarantee(self, constraint_specs, system_model=None):
        """Verify performance guarantees are satisfiable."""
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.PERFORMANCE_GUARANTEE,
                constraint_id="perf_guarantee",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            constraints = []
            for spec in constraint_specs:
                expr = spec.get("expression", "")
                if expr:
                    constraints.append(Z3Constraint(expr, Z3ConstraintType.BOOLEAN))
            
            if system_model:
                for constraint_expr in system_model.get("constraints", []):
                    constraints.append(Z3Constraint(constraint_expr, Z3ConstraintType.BOOLEAN))
            
            if constraints:
                result = self.solver.solve_constraints([], constraints)
                is_sat = result.is_sat() if hasattr(result, "is_sat") else False
            else:
                is_sat = True
            
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED if is_sat else VerificationStatus.VIOLATED,
                verification_type=VerificationType.PERFORMANCE_GUARANTEE,
                constraint_id="perf_guarantee",
                recommendations=["Performance guarantees feasible"] if is_sat else ["Conflicting constraints"],
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.PERFORMANCE_GUARANTEE,
                constraint_id="perf_guarantee",
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def verify_security_property(self, property_spec, threat_model=None):
        """Verify security property holds against threat model."""
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.SECURITY_PROPERTY,
                constraint_id="security_prop",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        try:
            constraints = []
            
            if threat_model:
                for constraint_expr in threat_model.get("constraints", []):
                    constraints.append(Z3Constraint(constraint_expr, Z3ConstraintType.BOOLEAN))
            
            property_expr = property_spec.get("expression", "true")
            constraints.append(Z3Constraint("(not " + property_expr + ")", Z3ConstraintType.BOOLEAN))
            
            result = self.solver.solve_constraints([], constraints)
            is_unsat = result.is_unsat() if hasattr(result, "is_unsat") else False
            
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED if is_unsat else VerificationStatus.VIOLATED,
                verification_type=VerificationType.SECURITY_PROPERTY,
                constraint_id="security_prop",
                recommendations=["Security property verified"] if is_unsat else ["Security property can be violated"],
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=VerificationType.SECURITY_PROPERTY,
                constraint_id="security_prop",
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    async def verify_hybrid(self, constraints, verification_type=VerificationType.SAFETY_INVARIANT, context=None) -> VerificationResult:
        """
        Verify using hybrid Z3 + CAV-NLP approach.
        
        Args:
            constraints: List of constraints to verify
            verification_type: Type of verification to perform
            context: Optional context for verification
            
        Returns:
            VerificationResult from hybrid validation
        """
        start_time = time.time()
        
        if not Z3_AVAILABLE:
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=verification_type,
                constraint_id="hybrid",
                execution_time_ms=(time.time() - start_time) * 1000,
                recommendations=["Z3 not available"]
            )
        
        try:
            # Z3 validation
            z3_constraints = [
                Z3Constraint(c, Z3ConstraintType.BOOLEAN) for c in constraints
            ]
            z3_result = self.solver.solve_constraints([], z3_constraints)
            
            # CAV-NLP verification
            if self.use_cav_nlp and CAV_NLP_AVAILABLE:
                try:
                    cav_result = await self.math_service.verify(constraints)
                    return self._combine_verification_results(
                        z3_result, cav_result, verification_type,
                        execution_time=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    logger.warning(f"CAV-NLP verification failed, using Z3 only: {e}")
            
            # Return Z3-only result
            execution_time = (time.time() - start_time) * 1000
            is_verified = z3_result.is_unsat() if hasattr(z3_result, 'is_unsat') else False
            
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED if is_verified else VerificationStatus.VIOLATED,
                verification_type=verification_type,
                constraint_id="hybrid",
                execution_time_ms=execution_time,
                recommendations=["Z3 verification completed"]
            )
            
        except Exception as e:
            logger.error(f"Hybrid verification failed: {e}")
            return VerificationResult(
                success=False,
                status=VerificationStatus.UNKNOWN,
                verification_type=verification_type,
                constraint_id="hybrid",
                violations=[{"error": str(e)}],
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _combine_verification_results(self, z3_result, cav_result, verification_type, execution_time: float) -> VerificationResult:
        """Combine Z3 and CAV-NLP verification results."""
        z3_unsat = hasattr(z3_result, 'is_unsat') and z3_result.is_unsat()
        cav_verified = isinstance(cav_result, dict) and cav_result.get('verified', False)
        
        if z3_unsat and cav_verified:
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED,
                verification_type=verification_type,
                constraint_id="hybrid",
                execution_time_ms=execution_time,
                recommendations=[
                    "Z3: Constraints are unsatisfiable",
                    "CAV-NLP: Mathematically verified"
                ]
            )
        elif z3_unsat:
            return VerificationResult(
                success=True,
                status=VerificationStatus.VERIFIED,
                verification_type=verification_type,
                constraint_id="hybrid",
                execution_time_ms=execution_time,
                recommendations=["Z3 verified (CAV-NLP inconclusive)"]
            )
        else:
            violations = [{"issue": "Z3: Constraints are satisfiable"}]
            if isinstance(cav_result, dict) and cav_result.get('violations'):
                violations.extend(cav_result['violations'])
            
            return VerificationResult(
                success=True,
                status=VerificationStatus.VIOLATED,
                verification_type=verification_type,
                constraint_id="hybrid",
                violations=violations,
                execution_time_ms=execution_time,
                recommendations=["Review constraints for potential issues"]
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get verifier capabilities including CAV-NLP status."""
        return {
            "z3_available": Z3_AVAILABLE,
            "cav_nlp_available": CAV_NLP_AVAILABLE,
            "cav_nlp_enabled": self.use_cav_nlp,
            "hybrid_verification": Z3_AVAILABLE and CAV_NLP_AVAILABLE,
            "verification_types": [vt.value for vt in VerificationType],
            "capabilities": [
                "sop_safety_verification",
                "performance_guarantee",
                "security_property_verification",
                "hybrid_z3_cav_verification" if (Z3_AVAILABLE and CAV_NLP_AVAILABLE) else "z3_only_verification"
            ]
        }


def get_z3_quality_gate_verifier():
    """Get global verifier instance."""
    return Z3QualityGateVerifier()


if __name__ == "__main__":
    print("Quality Gate Z3 Verifier initialized")
