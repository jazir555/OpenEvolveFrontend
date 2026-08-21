"""Blue Team Coordinator.

The Blue Team are the *fixers / provers*. Given issues raised by the Red Team
(or a candidate solution / theorem), the coordinator uses ``LeanAideClient`` to
actually attempt formal proofs and verify proposed fixes.

When the full ``BlueTeam`` system (``blue_team.py``) is importable it is used to
generate non-formal fixes; otherwise the coordinator degrades to pure formal
verification so the module remains functional offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from team_lean_bridge import get_lean_client, LEAN_AVAILABLE, sha1

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional heavy dependency
    from blue_team import BlueTeam
    from red_team import IssueFinding
    BLUE_TEAM_AVAILABLE = True
except Exception:  # noqa: BLE001
    BLUE_TEAM_AVAILABLE = False
    BlueTeam = None  # type: ignore
    IssueFinding = None  # type: ignore


@dataclass
class ProofAttempt:
    """Result of a single formal proof attempt."""

    theorem: str
    tactics: List[str]
    proved: bool
    status: str
    tactic_count: int = 0
    error: Optional[str] = None
    elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theorem": self.theorem,
            "tactics": list(self.tactics),
            "proved": self.proved,
            "status": self.status,
            "tactic_count": self.tactic_count,
            "error": self.error,
            "elapsed": round(self.elapsed, 4),
        }


class BlueTeamCoordinator:
    """Coordinates proof attempts and fix verification for the Blue Team."""

    def __init__(
        self,
        client: Any = None,
        blue_team: Any = None,
        max_tactics: int = 20,
    ):
        self.client = client or get_lean_client()
        self.blue_team = blue_team or (BlueTeam() if BLUE_TEAM_AVAILABLE else None)
        self.max_tactics = max_tactics
        self.proof_history: List[ProofAttempt] = []

    # -- formal proof -----------------------------------------------------
    def attempt_proof(
        self, theorem: str, tactics: Optional[List[str]] = None
    ) -> ProofAttempt:
        """Attempt to prove ``theorem`` using the LeanAIDE client."""
        result = self.client.prove(theorem, tactics)
        attempt = ProofAttempt(
            theorem=theorem,
            tactics=list(getattr(result, "tactics", tactics or [])),
            proved=bool(result),
            status=getattr(result, "status", "unknown"),
            tactic_count=getattr(result, "tactic_count", len(tactics or [])),
            error=getattr(result, "error", None),
            elapsed=getattr(result, "elapsed", 0.0),
        )
        self.proof_history.append(attempt)
        return attempt

    def verify_fix(self, lean_code: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Verify a proposed fix / solution by attempting to prove it."""
        attempt = self.attempt_proof(lean_code)
        return {
            "verified": attempt.proved,
            "status": attempt.status,
            "tactics": attempt.tactics,
            "error": attempt.error,
            "formalized": lean_code,
            "metadata": properties or {},
        }

    # -- non-formal fixing ------------------------------------------------
    def fix_content(
        self, content: str, issues: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Apply Blue Team fixes to ``content``.

        Returns the fixed content plus the list of applied fixes. When the
        full ``BlueTeam`` system is unavailable, the content is returned
        unchanged with a note so the pipeline stays functional.
        """
        if self.blue_team is None:
            return {
                "fixed_content": content,
                "applied_fixes": [],
                "improvement_score": 0.0,
                "note": "BlueTeam system unavailable; returning content unchanged",
            }
        try:
            assessment = self.blue_team.apply_fixes(content, issues or [])
            fixed = getattr(assessment, "fixed_content", content)
            applied = getattr(assessment, "applied_fixes", []) or []
            score = getattr(assessment, "improvement_score", 0.0)
            return {
                "fixed_content": fixed,
                "applied_fixes": list(applied),
                "improvement_score": float(score),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Blue Team fix generation failed: %s", exc)
            return {
                "fixed_content": content,
                "applied_fixes": [],
                "improvement_score": 0.0,
                "error": str(exc),
            }

    # -- reporting --------------------------------------------------------
    def get_proof_stats(self) -> Dict[str, Any]:
        total = len(self.proof_history)
        proved = sum(1 for p in self.proof_history if p.proved)
        return {
            "total_attempts": total,
            "proved": proved,
            "failed": total - proved,
            "success_rate": (proved / total) if total else 0.0,
        }

    @staticmethod
    def _fingerprint(content: str) -> str:
        return sha1(content)
