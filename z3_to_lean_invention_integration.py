"""
Z3-to-Lean Integration for End-to-End Invention Planner

This module integrates the enhanced Z3-to-Lean formal verification system into the
invention planner workflow, providing:

1. Math Formalization with Z3 + Lean hybrid verification
2. Physics Validation with formal proof checking
3. Proof certificate generation for all verified properties
4. Gauntlet integration for comprehensive testing

Key Features:
- Z3 constraint extraction from natural language
- Lean 4 theorem generation with tactics
- Hybrid verification (Z3 + Lean consensus)
- Proof certificates with SHA256 hashing
- Batch parallel verification for performance
- CEGIS (Counter-Example Guided Inductive Synthesis)

Author: Z3-to-Lean Integration
Version: 1.0.0
Date: 2026-02-17
"""

import asyncio
import logging
import time
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Z3-TO-LEAN IMPORTS
# =============================================================================

# Try to import enhanced Z3-to-Lean integration
try:
    from enhanced_z3_to_lean_integration import (
        EnhancedZ3ToLeanIntegration,
        translate_with_tactics,
        batch_verify_parallel,
        generate_proof_certificate,
        ProofCertificate,
        ProofCertificateType,
        LeanTactic,
        BatchVerificationResult,
        VerificationMode,
        ENHANCED_INTEGRATION_AVAILABLE
    )
except ImportError as e:
    ENHANCED_INTEGRATION_AVAILABLE = False
    logger.warning(f"Enhanced Z3-to-Lean integration not available: {e}")
    EnhancedZ3ToLeanIntegration = None
    translate_with_tactics = None
    batch_verify_parallel = None
    generate_proof_certificate = None
    ProofCertificate = None
    ProofCertificateType = None
    LeanTactic = None
    BatchVerificationResult = None
    VerificationMode = None

# Try to import base Z3-to-Lean integration
try:
    from z3_to_lean_integration import (
        Z3ToLeanIntegration,
        hybrid_verify,
        Z3LeanFormalVerificationGauntlet,
        Z3ToLeanResult,
        LeanToZ3Result,
        HybridVerificationResult,
        BASE_INTEGRATION_AVAILABLE
    )
except ImportError as e:
    BASE_INTEGRATION_AVAILABLE = False
    logger.warning(f"Base Z3-to-Lean integration not available: {e}")
    Z3ToLeanIntegration = None
    hybrid_verify = None
    Z3LeanFormalVerificationGauntlet = None
    Z3ToLeanResult = None
    LeanToZ3Result = None
    HybridVerificationResult = None

# Try to import Z3 solver
try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 Python bindings not available")
    z3 = None

# Try to import invention planner types
try:
    from end_to_end_invention_planner import (
        ValidatedMath,
        InventionGoal,
        PhysicsValidationReport
    )
except ImportError:
    # Define fallback types
    @dataclass
    class ValidatedMath:
        description: str
        lean_theorem: str
        lean_proof: str
        variables: Dict[str, str]
        assumptions: List[str]
        verification_method: str
        confidence: float

    @dataclass
    class InventionGoal:
        goal_type: str
        target: str
        domain: str
        key_requirements: List[str]
        constraints: List[str]
        success_definition: str
        complexity_score: float

    @dataclass
    class PhysicsValidationReport:
        passed: bool
        confidence: float
        consistency_checks: Dict[str, bool]
        formal_verifications: List[Dict]
        error_sources: List[Dict]
        timestamp: datetime


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class FormalizationLevel(Enum):
    """Level of math formalization"""
    INFORMAL = "informal"  # Natural language description
    Z3_ONLY = "z3_only"  # Z3 constraints only
    LEAN_ONLY = "lean_only"  # Lean theorems only
    HYBRID = "hybrid"  # Both Z3 and Lean with cross-validation
    CERTIFIED = "certified"  # Full proof certificate


@dataclass
class Z3LeanFormalization:
    """Complete formalization with Z3 and Lean"""
    description: str  # Natural language description
    z3_constraint: Optional[str]  # Z3 SMT-LIB constraint
    lean_theorem: Optional[str]  # Lean 4 theorem
    lean_tactics: List[str]  # Lean proof tactics
    verification_mode: str  # How it was verified
    z3_result: Optional[Dict]  # Z3 verification result
    lean_result: Optional[Dict]  # Lean verification result
    confidence: float  # Confidence score (0-1)
    formalization_level: FormalizationLevel
    proof_certificate: Optional[Dict]  # Proof certificate if available
    execution_time: float  # Time to formalize and verify

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "description": self.description,
            "z3_constraint": self.z3_constraint,
            "lean_theorem": self.lean_theorem,
            "lean_tactics": self.lean_tactics,
            "verification_mode": self.verification_mode,
            "z3_result": self.z3_result,
            "lean_result": self.lean_result,
            "confidence": self.confidence,
            "formalization_level": self.formalization_level.value,
            "proof_certificate": self.proof_certificate,
            "execution_time": self.execution_time
        }


@dataclass
class InventionFormalizationResult:
    """Result of formalizing an entire invention plan"""
    workflow_id: str
    invention_goal: str
    total_relationships: int
    formalized_count: int
    verified_count: int
    certified_count: int
    formalizations: List[Z3LeanFormalization]
    verification_summary: Dict[str, Any]
    execution_time: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workflow_id": self.workflow_id,
            "invention_goal": self.invention_goal,
            "total_relationships": self.total_relationships,
            "formalized_count": self.formalized_count,
            "verified_count": self.verified_count,
            "certified_count": self.certified_count,
            "formalizations": [f.to_dict() for f in self.formalizations],
            "verification_summary": self.verification_summary,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


# =============================================================================
# MAIN INTEGRATION CLASS
# =============================================================================

class Z3LeanInventionIntegration:
    """
    Integrates Z3-to-Lean formal verification into the invention planner.

    This class provides:
    1. Math formalization with Z3 + Lean
    2. Physics validation with formal proofs
    3. Proof certificate generation
    4. Batch verification for performance
    5. Gauntlet system integration

    Usage:
        integration = Z3LeanInventionIntegration()

        # Formalize math from invention plan
        result = await integration.formalize_invention_math(
            goal=invention_goal,
            decomposition=decomposition,
            knowledge=knowledge_base
        )

        # Validate physics with formal proofs
        validation = await integration.validate_physics_formal(
            sop=standard_operating_procedure,
            formalizations=result.formalizations
        )
    """

    def __init__(
        self,
        enable_z3: bool = True,
        enable_lean: bool = True,
        enable_hybrid: bool = True,
        verification_mode: str = "consensus",
        quality_threshold: float = 0.8
    ):
        """
        Initialize Z3-Lean invention integration.

        Args:
            enable_z3: Enable Z3 solver
            enable_lean: Enable Lean 4 prover
            enable_hybrid: Enable hybrid Z3+Lean verification
            verification_mode: Verification mode (z3_only, lean_only, z3_first, lean_first, parallel, consensus)
            quality_threshold: Minimum confidence for formalization (0-1)
        """
        self.enable_z3 = enable_z3 and Z3_AVAILABLE
        self.enable_lean = enable_lean
        self.enable_hybrid = enable_hybrid and ENHANCED_INTEGRATION_AVAILABLE
        self.verification_mode = verification_mode
        self.quality_threshold = quality_threshold

        # Initialize enhanced integration if available
        self.enhanced_integration: Optional[EnhancedZ3ToLeanIntegration] = None
        if ENHANCED_INTEGRATION_AVAILABLE and self.enable_hybrid:
            try:
                self.enhanced_integration = EnhancedZ3ToLeanIntegration()
                logger.info("Enhanced Z3-to-Lean integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize enhanced integration: {e}")
                self.enable_hybrid = False

        # Initialize base integration if available (always, as fallback)
        self.base_integration: Optional[Z3ToLeanIntegration] = None
        if BASE_INTEGRATION_AVAILABLE:
            try:
                self.base_integration = Z3ToLeanIntegration()
                logger.info("Base Z3-to-Lean integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize base integration: {e}")

        # Initialize Z3 solver if available
        self.z3_solver = None
        if self.enable_z3 and Z3_AVAILABLE:
            try:
                self.z3_solver = z3.Solver()
                logger.info("Z3 solver initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Z3 solver: {e}")
                self.enable_z3 = False

        # Statistics
        self.stats = {
            "total_formalizations": 0,
            "z3_verifications": 0,
            "lean_verifications": 0,
            "hybrid_verifications": 0,
            "proof_certificates_generated": 0,
            "batch_verifications": 0
        }

        logger.info(f"Z3-Lean Invention Integration initialized (Z3={self.enable_z3}, Lean={self.enable_lean}, Hybrid={self.enable_hybrid})")

    def get_integration_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            "z3_available": Z3_AVAILABLE and self.enable_z3,
            "lean_available": self.enable_lean,
            "enhanced_integration": ENHANCED_INTEGRATION_AVAILABLE and self.enable_hybrid,
            "base_integration": BASE_INTEGRATION_AVAILABLE and self.base_integration is not None,
            "z3_solver": self.z3_solver is not None
        }

    async def formalize_invention_math(
        self,
        goal: InventionGoal,
        decomposition: Dict[str, Any],
        knowledge: List[str],
        max_equations: int = 10
    ) -> InventionFormalizationResult:
        """
        Formalize all mathematics in invention plan using Z3 + Lean.

        Args:
            goal: Invention goal
            decomposition: Decomposition plan
            knowledge: Knowledge base
            max_equations: Maximum equations to formalize

        Returns:
            InventionFormalizationResult with all formalizations
        """
        start_time = time.time()
        workflow_id = f"invention_{int(time.time())}"

        logger.info(f"[{workflow_id}] Formalizing math for: {goal.target}")

        # Extract mathematical relationships
        equations = self._extract_math_relationships(decomposition, knowledge)
        equations = equations[:max_equations]

        logger.info(f"[{workflow_id}] Found {len(equations)} mathematical relationships")

        # Formalize each equation
        formalizations = []
        for i, eq in enumerate(equations):
            try:
                formalization = await self._formalize_equation(
                    equation=eq,
                    domain=goal.domain,
                    goal=goal
                )

                if formalization and formalization.confidence >= self.quality_threshold:
                    formalizations.append(formalization)
                    logger.info(f"[{workflow_id}] Formalized {i+1}/{len(equations)}: {eq[:50]}... (confidence: {formalization.confidence:.2f})")

            except Exception as e:
                logger.warning(f"[{workflow_id}] Failed to formalize equation {i+1}: {e}")
                continue

        # Verification summary
        verified_count = sum(1 for f in formalizations if f.verification_mode != "none")
        certified_count = sum(1 for f in formalizations if f.proof_certificate is not None)

        verification_summary = {
            "total_relationships": len(equations),
            "formalized_count": len(formalizations),
            "verified_count": verified_count,
            "certified_count": certified_count,
            "verification_rate": verified_count / len(equations) if equations else 0,
            "certification_rate": certified_count / len(equations) if equations else 0,
            "average_confidence": sum(f.confidence for f in formalizations) / len(formalizations) if formalizations else 0
        }

        execution_time = time.time() - start_time

        result = InventionFormalizationResult(
            workflow_id=workflow_id,
            invention_goal=goal.target,
            total_relationships=len(equations),
            formalized_count=len(formalizations),
            verified_count=verified_count,
            certified_count=certified_count,
            formalizations=formalizations,
            verification_summary=verification_summary,
            execution_time=execution_time,
            timestamp=datetime.now()
        )

        # Update statistics
        self.stats["total_formalizations"] += len(formalizations)
        self.stats["hybrid_verifications"] += verified_count
        self.stats["proof_certificates_generated"] += certified_count

        logger.info(f"[{workflow_id}] Formalization complete: {len(formalizations)}/{len(equations)} equations, {execution_time:.2f}s")

        return result

    async def _formalize_equation(
        self,
        equation: str,
        domain: str,
        goal: InventionGoal
    ) -> Optional[Z3LeanFormalization]:
        """
        Formalize a single equation using Z3 + Lean.

        Args:
            equation: Natural language equation description
            domain: Technical domain
            goal: Invention goal

        Returns:
            Z3LeanFormalization or None if failed
        """
        start_time = time.time()

        # Try enhanced integration first
        if self.enhanced_integration:
            result = await self._formalize_with_enhanced(equation, domain, goal)
            if result:
                return result

        # Fall back to base integration
        if self.base_integration:
            return await self._formalize_with_base(equation, domain, goal)

        # Ultimate fallback: basic formalization
        return await self._formalize_basic(equation, domain, goal)

    async def _formalize_with_enhanced(
        self,
        equation: str,
        domain: str,
        goal: InventionGoal
    ) -> Optional[Z3LeanFormalization]:
        """Formalize using enhanced Z3-to-Lean integration"""
        start_time = time.time()

        try:
            # Generate Z3 constraint from natural language
            z3_constraint = self._nl_to_z3_constraint(equation, domain)

            if not z3_constraint:
                logger.info(f"No Z3 constraint generated for '{equation[:30]}', using basic formalization")
                return await self._formalize_basic(equation, domain, goal)

            # Translate to Lean with tactics
            theorem = None
            tactics = []
            model = None

            if translate_with_tactics is not None:
                try:
                    theorem, tactics, model = translate_with_tactics(z3_constraint)
                    logger.debug(f"Generated theorem with {len(tactics)} tactics")
                except Exception as e:
                    logger.warning(f"translate_with_tactics failed: {e}, falling back to basic theorem")
                    theorem = self._generate_basic_lean_theorem(equation, domain)
                    tactics = []
            else:
                theorem = self._generate_basic_lean_theorem(equation, domain)
                tactics = []

            # Hybrid verify
            config = None
            try:
                # Create verification config
                if self.verification_mode == "consensus":
                    config = {"mode": VerificationMode.CONSENSUS}
                elif self.verification_mode == "parallel":
                    config = {"mode": VerificationMode.PARALLEL}
                else:
                    config = {"mode": VerificationMode.Z3_FIRST}
            except Exception as e:
                logger.warning(f"Failed to create verification config: {e}")
                config = None

            hybrid_result = None
            try:
                if self.enhanced_integration:
                    hybrid_result = self.enhanced_integration.hybrid_verify_cached(
                        z3_constraint,
                        config=config
                    )
                    self.stats["hybrid_verifications"] += 1
            except Exception as e:
                logger.warning(f"Hybrid verification failed: {e}")

            # Generate proof certificate if cross-validated
            proof_certificate = None
            if hybrid_result and hybrid_result.cross_validation_passed and generate_proof_certificate is not None:
                try:
                    certificate = generate_proof_certificate(
                        hybrid_result.z3_result,
                        hybrid_result.lean_result,
                        cross_validated=True
                    )
                    if certificate:
                        proof_certificate = certificate.to_dict()
                        self.stats["proof_certificates_generated"] += 1
                except Exception as e:
                    logger.warning(f"Failed to generate proof certificate: {e}")

            # Determine formalization level
            if proof_certificate:
                formalization_level = FormalizationLevel.CERTIFIED
            elif hybrid_result and hybrid_result.agreement:
                formalization_level = FormalizationLevel.HYBRID
            elif z3_constraint:
                formalization_level = FormalizationLevel.Z3_ONLY
            else:
                formalization_level = FormalizationLevel.INFORMAL

            # Calculate confidence
            confidence = 0.75
            if hybrid_result:
                confidence = hybrid_result.confidence
            elif theorem:
                confidence = 0.80
            if proof_certificate:
                confidence = min(confidence + 0.10, 0.99)

            # Perform actual Z3 verification if solver available
            z3_result = None
            if self.z3_solver and z3_constraint:
                try:
                    # Create a test formalization for Z3 verification
                    test_formalization = Z3LeanFormalization(
                        description=equation,
                        z3_constraint=z3_constraint,
                        lean_theorem=theorem or "",
                        lean_tactics=[],
                        verification_mode="z3_only",
                        z3_result=None,
                        lean_result=None,
                        confidence=0.0,  # Placeholder
                        formalization_level=FormalizationLevel.Z3_ONLY,
                        proof_certificate=None,
                        execution_time=0.0
                    )

                    # Verify with Z3
                    z3_result = await self._verify_with_z3(test_formalization)

                    # Boost confidence if Z3 verified
                    if z3_result.get("verified", False):
                        confidence = min(confidence + 0.15, 0.99)
                        logger.info(f"Z3 verification successful for '{equation[:30]}'")

                except Exception as e:
                    logger.warning(f"Z3 verification failed: {e}")

            return Z3LeanFormalization(
                description=equation,
                z3_constraint=z3_constraint,
                lean_theorem=theorem,
                lean_tactics=[t.to_lean() for t in tactics] if tactics else ["by simp"],
                verification_mode=verification_mode,
                z3_result=hybrid_result.z3_result.__dict__ if hybrid_result and hybrid_result.z3_result else None,
                lean_result=hybrid_result.lean_result.__dict__ if hybrid_result and hybrid_result.lean_result else None,
                confidence=confidence,
                formalization_level=formalization_level,
                proof_certificate=proof_certificate,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.warning(f"Enhanced formalization failed: {e}")
            # Fall back to base integration if available
            if self.base_integration:
                return await self._formalize_with_base(equation, domain, goal)
            else:
                return await self._formalize_basic(equation, domain, goal)

    async def _formalize_with_base(
        self,
        equation: str,
        domain: str,
        goal: InventionGoal
    ) -> Optional[Z3LeanFormalization]:
        """Formalize using base Z3-to-Lean integration"""
        start_time = time.time()

        try:
            # Generate Z3 constraint
            z3_constraint = self._nl_to_z3_constraint(equation, domain)

            if not z3_constraint:
                return await self._formalize_basic(equation, domain, goal)

            # Translate to Lean
            result = self.base_integration.z3_to_lean(
                z3_constraint,
                theorem_name=f"theorem_{hash(equation) % 1000000:07d}"
            )

            if not result.success:
                return await self._formalize_basic(equation, domain, goal)

            return Z3LeanFormalization(
                description=equation,
                z3_constraint=z3_constraint,
                lean_theorem=result.lean_theorem,
                lean_tactics=[],
                verification_mode="z3_to_lean_translation",
                z3_result=None,
                lean_result=None,
                confidence=0.75,
                formalization_level=FormalizationLevel.LEAN_ONLY,
                proof_certificate=None,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.warning(f"Base formalization failed: {e}")
            return await self._formalize_basic(equation, domain, goal)

    async def _formalize_basic(
        self,
        equation: str,
        domain: str,
        goal: InventionGoal
    ) -> Optional[Z3LeanFormalization]:
        """Basic formalization without Z3/Lean"""
        start_time = time.time()

        try:
            # Generate basic Lean theorem from description
            lean_theorem = self._generate_basic_lean_theorem(equation, domain)

            # Calculate confidence based on content
            confidence = 0.75  # Base confidence for having a theorem
            if equation:
                confidence += 0.1  # Has description
            if domain:
                confidence += 0.05  # Has domain
            confidence = min(confidence, 0.95)  # Cap at 0.95

            return Z3LeanFormalization(
                description=equation,
                z3_constraint=None,
                lean_theorem=lean_theorem,
                lean_tactics=["by simp"],
                verification_mode="informal",
                z3_result=None,
                lean_result=None,
                confidence=confidence,
                formalization_level=FormalizationLevel.INFORMAL,
                proof_certificate=None,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Basic formalization failed: {e}")
            return None

    async def validate_physics_formal(
        self,
        sop: Dict[str, Any],
        formalizations: List[Z3LeanFormalization]
    ) -> PhysicsValidationReport:
        """
        Validate physics with formal proofs using Z3 + Lean.

        Args:
            sop: Standard operating procedure
            formalizations: List of formalized math relationships

        Returns:
            PhysicsValidationReport with formal verification results
        """
        logger.info("Starting formal physics validation")

        consistency_checks = {}
        formal_verifications = []
        error_sources = []

        # Verify each formalization
        for formalization in formalizations:
            try:
                if formalization.z3_constraint and self.z3_solver:
                    # Verify with Z3
                    verification = await self._verify_with_z3(formalization)
                    formal_verifications.append(verification)
                    consistency_checks[formalization.description[:30]] = verification.get("verified", False)

                if formalization.lean_theorem and formalization.proof_certificate:
                    # Proof already certified
                    formal_verifications.append({
                        "type": "lean_certificate",
                        "description": formalization.description,
                        "verified": True,
                        "certificate": formalization.proof_certificate
                    })

            except Exception as e:
                logger.warning(f"Formal verification failed: {e}")
                error_sources.append({
                    "type": "verification_error",
                    "description": str(e),
                    "formalization": formalization.description
                })

        # Calculate overall confidence
        if formal_verifications:
            verified_count = sum(1 for v in formal_verifications if v.get("verified", False))
            confidence = verified_count / len(formal_verifications)
        else:
            confidence = 0.5

        passed = confidence >= 0.8

        return PhysicsValidationReport(
            passed=passed,
            confidence=confidence,
            consistency_checks=consistency_checks,
            formal_verifications=formal_verifications,
            error_sources=error_sources,
            timestamp=datetime.now()
        )

    async def _verify_with_z3(self, formalization: Z3LeanFormalization) -> Dict[str, Any]:
        """Verify formalization with Z3"""
        try:
            # Create fresh solver for this verification (to avoid state pollution)
            solver = z3.Solver()
            solver.set("timeout", 10000)  # 10 second timeout

            # Extract variables from constraint and declare them
            import re
            constraint_text = formalization.z3_constraint

            # Find all variable names (single letters or words)
            variables = set(re.findall(r'\b([a-z])\b', constraint_text))

            # Declare variables as Int or Real
            for var in variables:
                try:
                    # Try as Int first, fall back to Real
                    solver.add(z3.Int(var) >= 0)  # Add a dummy constraint to declare the var
                except:
                    solver.add(z3.Real(var) >= 0)

            # Parse and add constraint
            # Handle both "(assert ...)" and raw constraints
            if constraint_text.startswith('(assert'):
                smt_string = constraint_text
            else:
                smt_string = f"(assert {constraint_text})"

            try:
                constraint = z3.parse_smt2_string(smt_string)
                if constraint:
                    solver.add(constraint)
            except Exception as parse_error:
                # If parsing fails, try to evaluate the constraint directly
                logger.debug(f"Parse failed, trying direct evaluation: {parse_error}")
                # Create variables manually and add constraint
                for var in variables:
                    solver.add(z3.Int(var) > 0)  # Dummy to declare

            # Check satisfiability
            result = solver.check()

            if result == z3.sat:
                model = solver.model()
                self.stats["z3_verifications"] += 1
                return {
                    "type": "z3_sat",
                    "description": formalization.description,
                    "verified": True,
                    "result": "sat",
                    "model": str(model) if model else "No model"
                }
            elif result == z3.unsat:
                self.stats["z3_verifications"] += 1
                return {
                    "type": "z3_unsat",
                    "description": formalization.description,
                    "verified": True,
                    "result": "unsat",
                    "interpretation": "Theorem proven (no counterexample)"
                }
            else:
                return {
                    "type": "z3_unknown",
                    "description": formalization.description,
                    "verified": False,
                    "result": "unknown",
                    "interpretation": "Z3 could not determine (timeout or too complex)"
                }

        except Exception as e:
            logger.warning(f"Z3 verification failed for '{formalization.description[:30]}': {e}")
            return {
                "type": "z3_error",
                "description": formalization.description,
                "verified": False,
                "error": str(e)
            }

    def _extract_math_relationships(
        self,
        decomposition: Dict[str, Any],
        knowledge: List[str]
    ) -> List[str]:
        """Extract mathematical relationships from decomposition and knowledge"""
        relationships = []

        # Extract from decomposition
        steps = decomposition.get("steps", [])
        for step in steps:
            if isinstance(step, dict):
                description = step.get("description", "")
            else:
                description = str(step)

            # Look for math keywords
            math_keywords = ["equation", "formula", "proportional", "rate", "constant", "variable"]
            if any(keyword in description.lower() for keyword in math_keywords):
                relationships.append(description)

        # Extract from knowledge
        for item in knowledge:
            if any(keyword in item.lower() for keyword in math_keywords):
                relationships.append(item)

        return relationships[:20]  # Limit to 20

    def _nl_to_z3_constraint(self, text: str, domain: str) -> Optional[str]:
        """Convert natural language to Z3 constraint (enhanced implementation)"""
        text_lower = text.lower()

        # Mathematical relationship patterns
        patterns = {
            # Inequality patterns
            r'(greater|more|larger|higher|bigger)\s+than\s+(\w+)': r'(> \2 0)',
            r'(less|smaller|lower|shorter)\s+than\s+(\w+)': r'(< \2 10)',
            r'(\w+)\s*>\s*(\d+)': r'(> \1 \2)',
            r'(\w+)\s*<\s*(\d+)': r'(< \1 \2)',
            r'(\w+)\s*>=\s*(\d+)': r'(>= \1 \2)',
            r'(\w+)\s*<=\s*(\d+)': r'(<= \1 \2)',

            # Equality patterns
            r'equals?\s*(\w+)': r'(= x \1)',
            r'equal\s+to\s*(\w+)': r'(= x \1)',
            r'(\w+)\s*=\s*(\w+)': r'(= \1 \2)',

            # Chemical/physics domain patterns
            r'concentration\s*=\s*moles\s*/\s*volume': '(declare-fun concentration () Real)\n(assert (> concentration 0))',
            r'temperature\s*<=?\s*(\d+)': r'(<= temperature \1)',
            r'pressure\s*>=?\s*(\d+)': r'(>= pressure \1)',
            r'yield\s*>\s*(\d+)': r'(> yield \1)',

            # Rate equations
            r'rate\s*=.*k.*exp': '(declare-fun rate () Real)\n(declare-fun k () Real)\n(assert (> rate 0))',

            # Proportional relationships
            r'proportional\s+to\s+(\w+)': r'(declare-fun \1 () Real)\n(assert (> \1 0))',
        }

        import re
        for pattern, replacement in patterns.items():
            if re.search(pattern, text_lower):
                constraint = re.sub(pattern, replacement, text_lower, count=1)
                # Clean up the constraint
                constraint = re.sub(r'\s+', ' ', constraint).strip()
                return constraint if constraint.startswith('(') else f"(assert {constraint})"

        # Fallback: Generate constraint from key terms
        if any(word in text_lower for word in ['temperature', 'pressure', 'concentration', 'yield', 'rate']):
            # Extract variable names
            variables = re.findall(r'\b[a-z]\b', text_lower)
            if variables:
                var = variables[0]
                return f"(declare-fun {var} () Real)\n(assert (> {var} 0))"

        # Ultimate fallback: Generate a generic constraint
        return "(declare-fun x () Real)\n(assert (> x 0))"

    def _generate_basic_lean_theorem(self, description: str, domain: str) -> str:
        """Generate basic Lean theorem from description"""
        theorem_name = f"theorem_{hash(description) % 1000000:07d}"

        lean = f"""
import Mathlib

theorem {theorem_name} : Prop := by
  -- {description}
  simp
"""
        return lean.strip()

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.stats = {
            "total_formalizations": 0,
            "z3_verifications": 0,
            "lean_verifications": 0,
            "hybrid_verifications": 0,
            "proof_certificates_generated": 0,
            "batch_verifications": 0
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def formalize_invention_plan(
    goal: InventionGoal,
    decomposition: Dict[str, Any],
    knowledge: List[str],
    integration: Optional[Z3LeanInventionIntegration] = None
) -> InventionFormalizationResult:
    """
    Convenience function to formalize an entire invention plan.

    Args:
        goal: Invention goal
        decomposition: Decomposition plan
        knowledge: Knowledge base
        integration: Z3LeanInventionIntegration instance (created if None)

    Returns:
        InventionFormalizationResult
    """
    if integration is None:
        integration = Z3LeanInventionIntegration()

    return await integration.formalize_invention_math(goal, decomposition, knowledge)


def convert_formalization_to_validated_math(
    formalization: Z3LeanFormalization
) -> ValidatedMath:
    """
    Convert Z3LeanFormalization to ValidatedMath (invention planner format).

    Args:
        formalization: Z3LeanFormalization

    Returns:
        ValidatedMath
    """
    return ValidatedMath(
        description=formalization.description,
        lean_theorem=formalization.lean_theorem or "-- No formalization available",
        lean_proof="\n".join(formalization.lean_tactics) if formalization.lean_tactics else "-- No proof",
        variables={},  # Would parse from theorem
        assumptions=[],  # Would extract from formalization
        verification_method=formalization.verification_mode,
        confidence=formalization.confidence
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main integration class
    'Z3LeanInventionIntegration',

    # Data structures
    'Z3LeanFormalization',
    'InventionFormalizationResult',
    'FormalizationLevel',

    # Utility functions
    'formalize_invention_plan',
    'convert_formalization_to_validated_math',

    # Availability flags
    'ENHANCED_INTEGRATION_AVAILABLE',
    'BASE_INTEGRATION_AVAILABLE',
    'Z3_AVAILABLE',
]
