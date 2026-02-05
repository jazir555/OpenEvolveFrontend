"""
Tiered Verification Orchestrator

Main orchestrator for the 3-tier verification system:
- Tier 1: Z3 Fast Verification (<1 second, 0-100 constraints)
- Tier 2: LeanAide AI-Assisted Proving (<1 minute, 100-1000 constraints)
- Tier 3: Lean 4 Formal Verification (any time, 1000+ constraints)

The orchestrator:
1. Classifies problems using ProblemClassifier
2. Selects appropriate solver using SolverSelector
3. Executes verification with automatic tier escalation
4. Combines results from multiple tiers
5. Provides unified API for all RESE phases

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Runtime Truth: Verify solvers via probes before use
- Law of Idempotency: All operations safe to run 100x
- Circuit Breaker: Detect and handle solver failures
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC ISO-8601

Author: RESE Team
Created: 2026-02-04
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import verification result structures
try:
    from .verification_result import (
        VerificationTier,
        VerificationStatus,
        ProblemClass,
        ProblemDomain,
        Z3VerificationResult,
        LeanAideVerificationResult,
        Lean4VerificationResult,
        UnifiedVerificationResult,
    )

    # Import problem classifier and solver selector
    from .problem_classifier import ProblemClassifier, ClassifierConfig
    from .solver_selector import SolverSelector, SolverSelectorConfig, SelectionStrategy
except ImportError:
    from verification_result import (
        VerificationTier,
        VerificationStatus,
        ProblemClass,
        ProblemDomain,
        Z3VerificationResult,
        LeanAideVerificationResult,
        Lean4VerificationResult,
        UnifiedVerificationResult,
    )

    # Import problem classifier and solver selector
    from problem_classifier import ProblemClassifier, ClassifierConfig
    from solver_selector import SolverSelector, SolverSelectorConfig, SelectionStrategy


# =============================================================================
# CONFIGURATION
# =============================================================================

class TieredVerifierConfig:
    """
    Tiered verifier configuration

    Law of Configuration Explicitness: All config from environment
    """

    def __init__(self):
        # Z3 configuration (Tier 1)
        self.z3_base_url = os.getenv("Z3_BASE_URL", "http://localhost:8000")
        self.z3_timeout_ms = int(os.getenv("TIER1_TIMEOUT_MS", "1000"))
        self.z3_max_constraints = int(os.getenv("TIER1_MAX_CONSTRAINTS", "100"))

        # LeanAide configuration (Tier 2)
        self.leanaide_base_url = os.getenv("LEANAIDE_BASE_URL", "http://localhost:8001")
        self.leanaide_timeout_ms = int(os.getenv("TIER2_TIMEOUT_MS", "60000"))
        self.leanaide_max_constraints = int(os.getenv("TIER2_MAX_CONSTRAINTS", "1000"))

        # Lean 4 configuration (Tier 3)
        self.lean4_path = os.getenv("LEAN4_PATH", "lean")
        self.lean4_workspace_dir = os.getenv("LEAN4_WORKSPACE_DIR", "/workspace/lean4")
        self.lean4_timeout_ms = int(os.getenv("TIER3_TIMEOUT_MS", "300000"))  # 5 minutes

        # Auto-escalation configuration
        self.auto_escalate = os.getenv("AUTO_ESCALATE", "true").lower() == "true"
        self.max_tier = int(os.getenv("MAX_TIER", "3"))  # 1, 2, or 3

        # Selection strategy
        self.selection_strategy = os.getenv("SELECTION_STRATEGY", "adaptive")  # fast_first, accurate_first, parallel, adaptive

        # Performance monitoring
        self.enable_monitoring = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

    @classmethod
    def from_env(cls) -> 'TieredVerifierConfig':
        """Load configuration from environment variables"""
        return cls()


# =============================================================================
# TIERED VERIFIER
# =============================================================================

class TieredVerifier:
    """
    Main tiered verification orchestrator.

    Provides unified API for verification across all 3 tiers with automatic
    escalation based on problem complexity and solver performance.

    Usage:
        >>> verifier = TieredVerifier()
        >>> result = verifier.verify("forall x, P(x) -> Q(x)")
        >>> print(result.get_summary())
    """

    def __init__(self, config: Optional[TieredVerifierConfig] = None):
        """
        Initialize tiered verifier.

        Args:
            config: Verifier configuration (defaults to environment variables)
        """
        self.config = config or TieredVerifierConfig.from_env()

        # Setup logger
        self.logger = logging.getLogger("rese.verification.tiered_verifier")
        self.logger.setLevel(logging.INFO)

        # Setup classifier and selector
        classifier_config = ClassifierConfig.from_env()
        selector_config = SolverSelectorConfig.from_env()

        self.classifier = ProblemClassifier(classifier_config)
        self.selector = SolverSelector(selector_config, classifier_config)

        # Initialize solver clients (lazy initialization)
        self._z3_client = None
        self._leanaide_client = None
        self._lean4_interface = None

        self.logger.info(json.dumps({
            "level": "info",
            "component": "TieredVerifier",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Tiered verifier initialized",
            "config": {
                "z3_base_url": self.config.z3_base_url,
                "leanaide_base_url": self.config.leanaide_base_url,
                "lean4_path": self.config.lean4_path,
                "auto_escalate": self.config.auto_escalate,
                "max_tier": self.config.max_tier,
                "selection_strategy": self.config.selection_strategy,
            },
        }))

    # ========================================================================
    # MAIN VERIFICATION API
    # ========================================================================

    def verify(
        self,
        problem: str,
        constraints: Optional[List[Any]] = None,
        variables: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> UnifiedVerificationResult:
        """
        Main verification entry point.

        Automatically selects solver tier and escalates if needed.

        Args:
            problem: Problem statement (natural language or formal)
            constraints: Optional list of constraints
            variables: Optional list of variables
            metadata: Optional metadata
            correlation_id: Optional correlation ID for tracing

        Returns:
            UnifiedVerificationResult with combined results from all tiers
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        self.logger.info(json.dumps({
            "level": "info",
            "component": "TieredVerifier",
            "timestamp": start_time.isoformat(),
            "message": "Starting verification",
            "correlation_id": correlation_id,
            "problem": problem[:200] if len(problem) > 200 else problem,
        }))

        # Create unified result
        result = UnifiedVerificationResult(correlation_id=correlation_id)

        try:
            # Classify problem
            problem_class, problem_domain, complexity = self.classifier.classify(
                problem,
                constraints,
                variables,
                metadata
            )

            result.problem_class = problem_class
            result.problem_domain = problem_domain

            # Select solver
            selection = self.selector.select_solver(
                problem,
                constraints,
                variables,
                metadata,
                strategy=SelectionStrategy(self.config.selection_strategy),
            )

            self.logger.info(json.dumps({
                "level": "info",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Solver selected",
                "correlation_id": correlation_id,
                "selected_tier": selection.recommended_tier.value,
                "strategy": selection.strategy.value,
                "reasoning": selection.reasoning,
            }))

            # Execute verification with escalation
            self._verify_with_escalation(
                problem,
                constraints,
                variables,
                selection,
                result,
                correlation_id
            )

            # Calculate total time
            end_time = datetime.now(timezone.utc)
            result.total_execution_time_ms = (end_time - start_time).total_seconds() * 1000

            self.logger.info(json.dumps({
                "level": "info",
                "component": "TieredVerifier",
                "timestamp": end_time.isoformat(),
                "message": "Verification completed",
                "correlation_id": correlation_id,
                "final_status": result.final_status.value,
                "successful_tier": result.successful_tier.value if result.successful_tier else None,
                "confidence": result.confidence,
                "total_time_ms": result.total_execution_time_ms,
            }))

        except Exception as e:
            self.logger.error(json.dumps({
                "level": "error",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }))
            result.final_status = VerificationStatus.ERROR
            result.metadata["error"] = str(e)

        return result

    def verify_with_tier(
        self,
        problem: str,
        tier: VerificationTier,
        constraints: Optional[List[Any]] = None,
        variables: Optional[List[Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]:
        """
        Verify with a specific tier.

        Args:
            problem: Problem statement
            tier: Specific tier to use
            constraints: Optional list of constraints
            variables: Optional list of variables
            correlation_id: Optional correlation ID

        Returns:
            Tier-specific verification result
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        self.logger.info(json.dumps({
            "level": "info",
            "component": "TieredVerifier",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Starting verification with {tier.value}",
            "correlation_id": correlation_id,
        }))

        if tier == VerificationTier.TIER1_Z3:
            return self._verify_tier1(problem, constraints, variables, correlation_id)
        elif tier == VerificationTier.TIER2_LEANAIDE:
            return self._verify_tier2(problem, constraints, variables, correlation_id)
        elif tier == VerificationTier.TIER3_LEAN4:
            return self._verify_tier3(problem, constraints, variables, correlation_id)
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def escalate_tier(
        self,
        current_result: Union[Z3VerificationResult, LeanAideVerificationResult],
        problem: str,
        constraints: Optional[List[Any]] = None,
        variables: Optional[List[Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Union[LeanAideVerificationResult, Lean4VerificationResult]:
        """
        Escalate to next tier.

        Args:
            current_result: Result from current tier
            problem: Problem statement
            constraints: Optional list of constraints
            variables: Optional list of variables
            correlation_id: Optional correlation ID

        Returns:
            Result from next tier
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        if isinstance(current_result, Z3VerificationResult):
            self.logger.info(json.dumps({
                "level": "info",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Escalating from Tier 1 to Tier 2",
                "correlation_id": correlation_id,
                "reason": "Tier 1 did not produce definitive result",
            }))
            return self._verify_tier2(problem, constraints, variables, correlation_id)

        elif isinstance(current_result, LeanAideVerificationResult):
            self.logger.info(json.dumps({
                "level": "info",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Escalating from Tier 2 to Tier 3",
                "correlation_id": correlation_id,
                "reason": "Tier 2 did not produce definitive result",
            }))
            return self._verify_tier3(problem, constraints, variables, correlation_id)

        else:
            raise ValueError("Cannot escalate from Tier 3 (it's the final tier)")

    def get_verification_status(self, correlation_id: str) -> Dict[str, Any]:
        """
        Get verification status for a correlation ID.

        Args:
            correlation_id: Correlation ID to check

        Returns:
            Status dictionary
        """
        # This would typically query a persistent store
        # For now, return a placeholder
        return {
            "correlation_id": correlation_id,
            "status": "unknown",
            "message": "Status tracking not implemented (would require persistent store)",
        }

    def combine_results(
        self,
        results: List[Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]],
        correlation_id: Optional[str] = None,
    ) -> UnifiedVerificationResult:
        """
        Combine results from multiple tiers.

        Args:
            results: List of tier results
            correlation_id: Optional correlation ID

        Returns:
            Unified verification result
        """
        correlation_id = correlation_id or str(uuid.uuid4())
        unified = UnifiedVerificationResult(correlation_id=correlation_id)

        for result in results:
            unified.add_tier_result(result)

        return unified

    # ========================================================================
    # INTERNAL VERIFICATION METHODS
    # ========================================================================

    def _verify_with_escalation(
        self,
        problem: str,
        constraints: Optional[List[Any]],
        variables: Optional[List[Any]],
        selection,
        result: UnifiedVerificationResult,
        correlation_id: str
    ):
        """Verify with automatic tier escalation"""

        current_tier = selection.recommended_tier
        max_tier = selection.max_tier

        while current_tier:
            # Verify with current tier
            tier_result = self.verify_with_tier(
                problem,
                current_tier,
                constraints,
                variables,
                correlation_id
            )

            # Add result to unified result
            escalation_reason = f"Selected by {selection.strategy.value} strategy"
            result.add_tier_result(tier_result, escalation_reason)

            # Record performance
            self.selector.record_performance(
                current_tier,
                tier_result.is_successful(),
                tier_result.status == VerificationStatus.TIMEOUT,
                tier_result.execution_time_ms
            )

            # Check if successful
            if tier_result.is_successful():
                result.final_status = VerificationStatus.VERIFIED
                result.successful_tier = current_tier
                return

            # Check if should escalate
            if not self.config.auto_escalate:
                break

            if current_tier == max_tier:
                # Reached max tier, stop escalating
                result.final_status = VerificationStatus.REFUTED
                return

            # Check if result indicates need for escalation
            if hasattr(tier_result, 'should_escalate') and tier_result.should_escalate():
                # Escalate to next tier
                if current_tier == VerificationTier.TIER1_Z3:
                    current_tier = VerificationTier.TIER2_LEANAIDE
                elif current_tier == VerificationTier.TIER2_LEANAIDE:
                    current_tier = VerificationTier.TIER3_LEAN4
                else:
                    # No more tiers to escalate
                    result.final_status = VerificationStatus.REFUTED
                    return
            else:
                # Don't escalate
                result.final_status = VerificationStatus.REFUTED
                return

        # If we get here, verification failed
        result.final_status = VerificationStatus.REFUTED

    # ========================================================================
    # TIER-SPECIFIC VERIFICATION
    # ========================================================================

    def _verify_tier1(
        self,
        problem: str,
        constraints: Optional[List[Any]],
        variables: Optional[List[Any]],
        correlation_id: str
    ) -> Z3VerificationResult:
        """Verify with Tier 1: Z3 Fast Verification"""
        start_time = datetime.now(timezone.utc)

        try:
            # Import Z3 bridge
            from glue.adapters.rese_z3_bridge.src.rese_z3_bridge import RESEZ3Bridge
            from glue.adapters.rese_z3_bridge.src.rese_z3_schema import (
                CanonicalConstraint,
                ConstraintType,
                CanonicalVariable,
            )

            # Initialize Z3 bridge (lazy)
            if self._z3_client is None:
                self._z3_client = RESEZ3Bridge()

            # Convert to canonical format
            canonical_constraints = []
            if constraints:
                for c in constraints:
                    if isinstance(c, str):
                        canonical_constraints.append(
                            CanonicalConstraint(
                                expression=c,
                                constraint_type=ConstraintType.BOOLEAN,
                            )
                        )
                    elif isinstance(c, dict):
                        canonical_constraints.append(
                            CanonicalConstraint(**c)
                        )

            # Use Z3 to detect contradictions
            has_contradiction, counterexample = self._z3_client.detect_contradictions(
                canonical_constraints,
                correlation_id=correlation_id,
                timeout_ms=self.config.z3_timeout_ms
            )

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Determine status
            if has_contradiction:
                status = VerificationStatus.REFUTED
                z3_result = "unsat"
            else:
                status = VerificationStatus.VERIFIED
                z3_result = "sat"

            return Z3VerificationResult(
                status=status,
                z3_result=z3_result,
                model=counterexample,
                execution_time_ms=execution_time_ms,
                constraints_checked=len(canonical_constraints),
                correlation_id=correlation_id,
            )

        except Exception as e:
            self.logger.error(json.dumps({
                "level": "error",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Tier 1 verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return Z3VerificationResult(
                status=VerificationStatus.ERROR,
                z3_result="error",
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                errors=[str(e)],
            )

    def _verify_tier2(
        self,
        problem: str,
        constraints: Optional[List[Any]],
        variables: Optional[List[Any]],
        correlation_id: str
    ) -> LeanAideVerificationResult:
        """Verify with Tier 2: LeanAide AI-Assisted Proving"""
        start_time = datetime.now(timezone.utc)

        try:
            # Try to import LeanAide bridge
            try:
                from z3_leanaide_bridge import Z3LeanAideBridge
            except ImportError:
                # LeanAide not available, simulate result
                self.logger.warning(json.dumps({
                    "level": "warn",
                    "component": "TieredVerifier",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "LeanAide not available, simulating Tier 2",
                    "correlation_id": correlation_id,
                }))

                execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                return LeanAideVerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    proof_status="partial",
                    execution_time_ms=execution_time_ms,
                    constraints_checked=len(constraints) if constraints else 0,
                    correlation_id=correlation_id,
                    errors=["LeanAide not available"],
                )

            # Initialize LeanAide bridge
            if self._leanaide_client is None:
                self._leanaide_client = Z3LeanAideBridge()

            # Use LeanAide to prove
            import asyncio

            async def verify_with_leanaide():
                return await self._leanaide_client.prove(
                    problem,
                    variables
                )

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            proof_result = loop.run_until_complete(verify_with_leanaide())
            loop.close()

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Convert to LeanAideVerificationResult
            if proof_result.success:
                status = VerificationStatus.VERIFIED
                proof_status = "proved"
            else:
                status = VerificationStatus.REFUTED
                proof_status = "failed"

            return LeanAideVerificationResult(
                status=status,
                proof_status=proof_status,
                proof_script=proof_result.lean_component,
                tactics_used=proof_result.tactics_used,
                execution_time_ms=execution_time_ms,
                constraints_checked=len(constraints) if constraints else 0,
                correlation_id=correlation_id,
            )

        except Exception as e:
            self.logger.error(json.dumps({
                "level": "error",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Tier 2 verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return LeanAideVerificationResult(
                status=VerificationStatus.ERROR,
                proof_status="failed",
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                errors=[str(e)],
            )

    def _verify_tier3(
        self,
        problem: str,
        constraints: Optional[List[Any]],
        variables: Optional[List[Any]],
        correlation_id: str
    ) -> Lean4VerificationResult:
        """Verify with Tier 3: Lean 4 Formal Verification"""
        start_time = datetime.now(timezone.utc)

        try:
            # Import Lean 4 bridge
            try:
                from glue.lib.lean4_bridge.lean4_interface import Lean4Interface
            except ImportError:
                # Lean 4 not available, simulate result
                self.logger.warning(json.dumps({
                    "level": "warn",
                    "component": "TieredVerifier",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": "Lean 4 not available, simulating Tier 3",
                    "correlation_id": correlation_id,
                }))

                execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                return Lean4VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    verification_status="errors",
                    execution_time_ms=execution_time_ms,
                    constraints_checked=len(constraints) if constraints else 0,
                    correlation_id=correlation_id,
                    errors=["Lean 4 not available"],
                )

            # Initialize Lean 4 interface
            if self._lean4_interface is None:
                self._lean4_interface = Lean4Interface()

            # Formalize constraint
            formalization_result = self._lean4_interface.formalize_constraint(
                problem,
                constraint_type="proposition",
                correlation_id=correlation_id
            )

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Convert to Lean4VerificationResult
            if formalization_result["verification_status"] == "verified":
                status = VerificationStatus.VERIFIED
                verification_status = "verified"
            else:
                status = VerificationStatus.REFUTED
                verification_status = "errors"

            return Lean4VerificationResult(
                status=status,
                verification_status=verification_status,
                lean4_code=formalization_result.get("lean4_code"),
                theorem_name=formalization_result.get("theorem_name"),
                execution_time_ms=execution_time_ms,
                constraints_checked=len(constraints) if constraints else 0,
                lean_version="4.x",
                correlation_id=correlation_id,
                errors=formalization_result.get("errors", []),
            )

        except Exception as e:
            self.logger.error(json.dumps({
                "level": "error",
                "component": "TieredVerifier",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Tier 3 verification failed",
                "correlation_id": correlation_id,
                "error": str(e),
            }))

            execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return Lean4VerificationResult(
                status=VerificationStatus.ERROR,
                verification_status="errors",
                execution_time_ms=execution_time_ms,
                correlation_id=correlation_id,
                errors=[str(e)],
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def verify(
    problem: str,
    constraints: Optional[List[Any]] = None,
    config: Optional[TieredVerifierConfig] = None
) -> UnifiedVerificationResult:
    """
    Convenience function for verification.

    Args:
        problem: Problem statement
        constraints: Optional list of constraints
        config: Optional verifier configuration

    Returns:
        UnifiedVerificationResult
    """
    verifier = TieredVerifier(config)
    return verifier.verify(problem, constraints)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TieredVerifierConfig",
    "TieredVerifier",
    "verify",
]
