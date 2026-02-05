"""
Unified Verification Result Data Structures

Defines canonical data structures for verification results across all 3 tiers:
- Tier 1: Z3 Fast Verification
- Tier 2: LeanAide AI-Assisted Proving
- Tier 3: Lean 4 Formal Verification

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of UTC: All timestamps in UTC ISO-8601
- Structured Logging: JSON with correlation_id
- Anti-Corruption Layer: Unified result schema

Author: RESE Team
Created: 2026-02-04
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# ENUMS
# =============================================================================

class VerificationTier(Enum):
    """Verification tier levels"""
    TIER1_Z3 = "tier1_z3"           # Fast SMT solving
    TIER2_LEANAIDE = "tier2_leanaide"  # AI-assisted proving
    TIER3_LEAN4 = "tier3_lean4"     # Formal verification


class VerificationStatus(Enum):
    """Verification status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    ERROR = "error"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"  # Escalated to next tier


class ProblemClass(Enum):
    """Problem classification"""
    CONSTRAINT_SAT = "constraint_sat"  # Constraint satisfaction
    THEOREM_PROVING = "theorem_proving"  # Theorem proving
    OPTIMIZATION = "optimization"  # Optimization problem
    CONTRADICTION_DETECTION = "contradiction_detection"  # Find contradictions
    MODEL_VALIDATION = "model_validation"  # Validate models


class ProblemDomain(Enum):
    """Problem domain classification"""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    LOGIC = "logic"
    PHYSICS = "physics"
    ARITHMETIC = "arithmetic"
    GEOMETRY = "geometry"
    GENERAL = "general"


# =============================================================================
# TIER-SPECIFIC RESULTS
# =============================================================================

@dataclass
class Z3VerificationResult:
    """
    Result from Tier 1: Z3 Fast Verification

    Characteristics:
    - Fast: <1 second
    - Scales: 0-100 constraints
    - Use case: Quick satisfiability, contradiction detection
    """
    tier: VerificationTier = field(default=VerificationTier.TIER1_Z3, init=False)
    status: VerificationStatus
    z3_result: str  # sat, unsat, unknown
    model: Optional[Dict[str, Any]] = None
    proof: Optional[str] = None
    execution_time_ms: float = 0.0
    constraints_checked: int = 0
    solver_version: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "status": self.status.value,
            "z3_result": self.z3_result,
            "model": self.model,
            "proof": self.proof,
            "execution_time_ms": self.execution_time_ms,
            "constraints_checked": self.constraints_checked,
            "solver_version": self.solver_version,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Z3VerificationResult':
        return cls(
            status=VerificationStatus(data["status"]),
            z3_result=data["z3_result"],
            model=data.get("model"),
            proof=data.get("proof"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            constraints_checked=data.get("constraints_checked", 0),
            solver_version=data.get("solver_version"),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )

    def is_successful(self) -> bool:
        """Check if verification was successful"""
        return self.status == VerificationStatus.VERIFIED

    def should_escalate(self) -> bool:
        """Check if should escalate to next tier"""
        # Escalate if unknown, timeout, or too many constraints
        return (
            self.status in [VerificationStatus.UNKNOWN, VerificationStatus.TIMEOUT] or
            self.constraints_checked > 100 or
            (self.execution_time_ms > 5000)  # Took too long
        )


@dataclass
class LeanAideVerificationResult:
    """
    Result from Tier 2: LeanAide AI-Assisted Proving

    Characteristics:
    - Medium speed: <1 minute
    - Scales: 100-1000 constraints
    - Use case: Theorem proving with AI guidance, autoformalization
    """
    tier: VerificationTier = field(default=VerificationTier.TIER2_LEANAIDE, init=False)
    status: VerificationStatus
    proof_status: str  # proved, failed, partial
    proof_script: Optional[str] = None
    tactics_used: List[str] = field(default_factory=list)
    autoformalization_confidence: float = 0.0  # 0.0 to 1.0
    suggested_tactics: List[str] = field(default_factory=list)
    goals_remaining: int = 0
    execution_time_ms: float = 0.0
    constraints_checked: int = 0
    ai_model_version: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "status": self.status.value,
            "proof_status": self.proof_status,
            "proof_script": self.proof_script,
            "tactics_used": self.tactics_used,
            "autoformalization_confidence": self.autoformalization_confidence,
            "suggested_tactics": self.suggested_tactics,
            "goals_remaining": self.goals_remaining,
            "execution_time_ms": self.execution_time_ms,
            "constraints_checked": self.constraints_checked,
            "ai_model_version": self.ai_model_version,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LeanAideVerificationResult':
        return cls(
            status=VerificationStatus(data["status"]),
            proof_status=data["proof_status"],
            proof_script=data.get("proof_script"),
            tactics_used=data.get("tactics_used", []),
            autoformalization_confidence=data.get("autoformalization_confidence", 0.0),
            suggested_tactics=data.get("suggested_tactics", []),
            goals_remaining=data.get("goals_remaining", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            constraints_checked=data.get("constraints_checked", 0),
            ai_model_version=data.get("ai_model_version"),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )

    def is_successful(self) -> bool:
        """Check if verification was successful"""
        return (
            self.status == VerificationStatus.VERIFIED and
            self.proof_status == "proved"
        )

    def should_escalate(self) -> bool:
        """Check if should escalate to next tier"""
        # Escalate if failed, partial, or too complex
        return (
            self.proof_status in ["failed", "partial"] or
            self.constraints_checked > 1000 or
            self.execution_time_ms > 60000  # Took too long
        )


@dataclass
class Lean4VerificationResult:
    """
    Result from Tier 3: Lean 4 Formal Verification

    Characteristics:
    - Any time: No strict limit
    - Scales: 1000+ constraints
    - Use case: Machine-checkable proofs, complete rigor
    """
    tier: VerificationTier = field(default=VerificationTier.TIER3_LEAN4, init=False)
    status: VerificationStatus
    verification_status: str  # verified, errors
    lean4_code: Optional[str] = None
    theorem_name: Optional[str] = None
    proof_object: Optional[str] = None  # Lean 4 proof object
    tactics_applied: List[str] = field(default_factory=list)
    goals_solved: int = 0
    total_goals: int = 0
    execution_time_ms: float = 0.0
    constraints_checked: int = 0
    lean_version: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "status": self.status.value,
            "verification_status": self.verification_status,
            "lean4_code": self.lean4_code,
            "theorem_name": self.theorem_name,
            "proof_object": self.proof_object,
            "tactics_applied": self.tactics_applied,
            "goals_solved": self.goals_solved,
            "total_goals": self.total_goals,
            "execution_time_ms": self.execution_time_ms,
            "constraints_checked": self.constraints_checked,
            "lean_version": self.lean_version,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Lean4VerificationResult':
        return cls(
            status=VerificationStatus(data["status"]),
            verification_status=data["verification_status"],
            lean4_code=data.get("lean4_code"),
            theorem_name=data.get("theorem_name"),
            proof_object=data.get("proof_object"),
            tactics_applied=data.get("tactics_applied", []),
            goals_solved=data.get("goals_solved", 0),
            total_goals=data.get("total_goals", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
            constraints_checked=data.get("constraints_checked", 0),
            lean_version=data.get("lean_version"),
            correlation_id=data.get("correlation_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )

    def is_successful(self) -> bool:
        """Check if verification was successful"""
        return (
            self.status == VerificationStatus.VERIFIED and
            self.verification_status == "verified"
        )

    def should_escalate(self) -> bool:
        """Lean 4 is the final tier - no escalation"""
        return False


# =============================================================================
# UNIFIED VERIFICATION RESULT
# =============================================================================

@dataclass
class UnifiedVerificationResult:
    """
    Unified verification result across all tiers

    Combines results from multiple tiers and provides a unified interface.
    """
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_class: Optional[ProblemClass] = None
    problem_domain: Optional[ProblemDomain] = None

    # Individual tier results
    tier1_result: Optional[Z3VerificationResult] = None
    tier2_result: Optional[LeanAideVerificationResult] = None
    tier3_result: Optional[Lean4VerificationResult] = None

    # Final combined result
    final_status: VerificationStatus = VerificationStatus.PENDING
    successful_tier: Optional[VerificationTier] = None
    confidence: float = 0.0  # 0.0 to 1.0

    # Escalation information
    escalation_path: List[VerificationTier] = field(default_factory=list)
    escalation_reasons: List[str] = field(default_factory=list)

    # Performance metrics
    total_execution_time_ms: float = 0.0
    total_constraints_checked: int = 0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "problem_class": self.problem_class.value if self.problem_class else None,
            "problem_domain": self.problem_domain.value if self.problem_domain else None,
            "tier1_result": self.tier1_result.to_dict() if self.tier1_result else None,
            "tier2_result": self.tier2_result.to_dict() if self.tier2_result else None,
            "tier3_result": self.tier3_result.to_dict() if self.tier3_result else None,
            "final_status": self.final_status.value,
            "successful_tier": self.successful_tier.value if self.successful_tier else None,
            "confidence": self.confidence,
            "escalation_path": [t.value for t in self.escalation_path],
            "escalation_reasons": self.escalation_reasons,
            "total_execution_time_ms": self.total_execution_time_ms,
            "total_constraints_checked": self.total_constraints_checked,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedVerificationResult':
        tier1 = None
        tier2 = None
        tier3 = None

        if data.get("tier1_result"):
            tier1 = Z3VerificationResult.from_dict(data["tier1_result"])
        if data.get("tier2_result"):
            tier2 = LeanAideVerificationResult.from_dict(data["tier2_result"])
        if data.get("tier3_result"):
            tier3 = Lean4VerificationResult.from_dict(data["tier3_result"])

        return cls(
            correlation_id=data["correlation_id"],
            problem_class=ProblemClass(data["problem_class"]) if data.get("problem_class") else None,
            problem_domain=ProblemDomain(data["problem_domain"]) if data.get("problem_domain") else None,
            tier1_result=tier1,
            tier2_result=tier2,
            tier3_result=tier3,
            final_status=VerificationStatus(data["final_status"]),
            successful_tier=VerificationTier(data["successful_tier"]) if data.get("successful_tier") else None,
            confidence=data.get("confidence", 0.0),
            escalation_path=[VerificationTier(t) for t in data.get("escalation_path", [])],
            escalation_reasons=data.get("escalation_reasons", []),
            total_execution_time_ms=data.get("total_execution_time_ms", 0.0),
            total_constraints_checked=data.get("total_constraints_checked", 0),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            metadata=data.get("metadata", {}),
        )

    def add_tier_result(
        self,
        result: Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult],
        reason: Optional[str] = None
    ):
        """Add a tier result and update escalation path"""
        if isinstance(result, Z3VerificationResult):
            self.tier1_result = result
            self.escalation_path.append(VerificationTier.TIER1_Z3)
        elif isinstance(result, LeanAideVerificationResult):
            self.tier2_result = result
            self.escalation_path.append(VerificationTier.TIER2_LEANAIDE)
        elif isinstance(result, Lean4VerificationResult):
            self.tier3_result = result
            self.escalation_path.append(VerificationTier.TIER3_LEAN4)

        if reason:
            self.escalation_reasons.append(reason)

        # Update totals
        self.total_execution_time_ms += result.execution_time_ms
        self.total_constraints_checked += result.constraints_checked

        # Update final status if this tier succeeded, but only set successful_tier if not already set
        # (first successful tier wins - lower tier is better)
        if result.is_successful():
            self.final_status = VerificationStatus.VERIFIED
            if self.successful_tier is None:
                self.successful_tier = result.tier
                self._calculate_confidence()

    def _calculate_confidence(self):
        """Calculate confidence based on successful tier"""
        if not self.successful_tier:
            self.confidence = 0.0
        elif self.successful_tier == VerificationTier.TIER1_Z3:
            # Z3 is fast but less rigorous
            self.confidence = 0.7
        elif self.successful_tier == VerificationTier.TIER2_LEANAIDE:
            # LeanAide provides AI-assisted proofs
            self.confidence = 0.85
        elif self.successful_tier == VerificationTier.TIER3_LEAN4:
            # Lean 4 is machine-checkable
            self.confidence = 1.0

    def is_successful(self) -> bool:
        """Check if overall verification was successful"""
        return self.final_status == VerificationStatus.VERIFIED

    def get_successful_result(self) -> Optional[Union[Z3VerificationResult, LeanAideVerificationResult, Lean4VerificationResult]]:
        """Get the successful tier result"""
        if self.successful_tier == VerificationTier.TIER1_Z3:
            return self.tier1_result
        elif self.successful_tier == VerificationTier.TIER2_LEANAIDE:
            return self.tier2_result
        elif self.successful_tier == VerificationTier.TIER3_LEAN4:
            return self.tier3_result
        return None

    def get_summary(self) -> str:
        """Get human-readable summary"""
        if self.final_status == VerificationStatus.PENDING:
            return f"Verification pending (correlation_id: {self.correlation_id})"

        if self.is_successful():
            tier_name = self.successful_tier.value.replace("_", " ").title() if self.successful_tier else "Unknown"
            return (
                f"✓ Verified via {tier_name} "
                f"(confidence: {self.confidence:.1%}, "
                f"time: {self.total_execution_time_ms:.0f}ms, "
                f"constraints: {self.total_constraints_checked})"
            )
        else:
            return (
                f"✗ Verification failed "
                f"(escalated through {len(self.escalation_path)} tiers, "
                f"time: {self.total_execution_time_ms:.0f}ms)"
            )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "VerificationTier",
    "VerificationStatus",
    "ProblemClass",
    "ProblemDomain",
    "Z3VerificationResult",
    "LeanAideVerificationResult",
    "Lean4VerificationResult",
    "UnifiedVerificationResult",
]
