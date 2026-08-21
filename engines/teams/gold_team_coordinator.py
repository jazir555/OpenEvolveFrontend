"""Gold Team Coordinator.

The Gold Team is the *final verification* team. After the Red Team has raised
challenges and the Blue Team has attempted fixes / proofs, the Gold Team
performs the authoritative verification using ``LeanAideClient`` and aggregates
confidence into a final verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from team_lean_bridge import get_lean_client, LEAN_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class VerificationVerdict:
    """Authoritative Gold Team verification result."""

    target: str
    verified: bool
    confidence: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "verified": self.verified,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "details": self.details,
        }


class GoldTeamCoordinator:
    """Coordinates final, authoritative verification for the Gold Team."""

    def __init__(self, client: Any = None, min_confidence: float = 0.6):
        self.client = client or get_lean_client()
        self.min_confidence = min_confidence
        self.verification_history: List[VerificationVerdict] = []

    def verify(self, lean_code: str, proof_attempt: Optional[Dict[str, Any]] = None) -> VerificationVerdict:
        """Formally verify ``lean_code`` (a theorem / solution) with the client."""
        result = self.client.prove(lean_code)
        proved = bool(result)
        status = getattr(result, "status", "unknown")
        tactics = list(getattr(result, "tactics", []))

        # Confidence: a proved formal statement is highly confident; a failed one
        # is only as confident as the evidence that the attempt was exhaustive.
        if proved:
            confidence = 1.0
        else:
            confidence = 0.2 if tactics else 0.0

        verdict = VerificationVerdict(
            target=lean_code,
            verified=proved,
            confidence=confidence,
            method="lean_formal_proof",
            details={
                "status": status,
                "tactics": tactics,
                "error": getattr(result, "error", None),
                "prior_proof": proof_attempt,
            },
        )
        self.verification_history.append(verdict)
        return verdict

    def final_review(self, content: str, artifacts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Produce a consolidated Gold Team verdict for a candidate solution.

        ``artifacts`` may carry prior round results (red findings, blue proof).
        A solution passes the Gold gate only when it is formally verified and
        confidence meets the configured threshold.
        """
        verdict = self.verify(content)
        passed = verdict.verified and verdict.confidence >= self.min_confidence

        return {
            "passed": passed,
            "verified": verdict.verified,
            "confidence": verdict.confidence,
            "method": verdict.method,
            "verdict": verdict.to_dict(),
            "artifacts_reviewed": artifacts or [],
            "recommendation": (
                "accept" if passed else "reject_or_revise"
            ),
        }

    def get_verification_stats(self) -> Dict[str, Any]:
        total = len(self.verification_history)
        verified = sum(1 for v in self.verification_history if v.verified)
        avg_conf = (
            sum(v.confidence for v in self.verification_history) / total
            if total
            else 0.0
        )
        return {
            "total": total,
            "verified": verified,
            "rejected": total - verified,
            "average_confidence": round(avg_conf, 4),
        }
