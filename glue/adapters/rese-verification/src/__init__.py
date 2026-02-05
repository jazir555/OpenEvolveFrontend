"""
RESE Tiered Verification System

A unified 3-tier verification system integrating:
- Tier 1: Z3 Fast Verification
- Tier 2: LeanAide AI-Assisted Proving
- Tier 3: Lean 4 Formal Verification

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Runtime Truth: Verify solvers via probes
- Law of Idempotency: All operations safe to run 100x
- Circuit Breaker: Detect and handle failures
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC ISO-8601

Usage:
    >>> from glue.adapters.rese_verification.src import TieredVerifier
    >>> verifier = TieredVerifier()
    >>> result = verifier.verify("forall x, P(x) -> Q(x)")
    >>> print(result.get_summary())

Author: RESE Team
Created: 2026-02-04
"""

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

from .problem_classifier import (
    ClassifierConfig,
    ProblemClassifier,
    classify_problem,
    should_escalate,
)

from .solver_selector import (
    SolverSelectorConfig,
    SolverPerformance,
    SelectionStrategy,
    SelectionResult,
    SolverSelector,
    select_solver,
)

from .tiered_verifier import (
    TieredVerifierConfig,
    TieredVerifier,
    verify,
)


__all__ = [
    # Verification results
    "VerificationTier",
    "VerificationStatus",
    "ProblemClass",
    "ProblemDomain",
    "Z3VerificationResult",
    "LeanAideVerificationResult",
    "Lean4VerificationResult",
    "UnifiedVerificationResult",

    # Problem classifier
    "ClassifierConfig",
    "ProblemClassifier",
    "classify_problem",
    "should_escalate",

    # Solver selector
    "SolverSelectorConfig",
    "SolverPerformance",
    "SelectionStrategy",
    "SelectionResult",
    "SolverSelector",
    "select_solver",

    # Tiered verifier
    "TieredVerifierConfig",
    "TieredVerifier",
    "verify",
]

__version__ = "1.0.0"
