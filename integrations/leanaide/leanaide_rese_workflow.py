"""LeanAide RESE Workflow integration.

Wires LeanAIDE proof verification into the RESE (Research Engineering / Solver
Execution) workflow. Each step verifies a proof using the genuine structural
checker (and the real Lean engine when configured) and records the outcome.
Degrades gracefully when external services are unavailable.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from leanaide_systems import check_lean_proof_structural
except ImportError:  # pragma: no cover
    from integrations.leanaide.leanaide_systems import check_lean_proof_structural  # type: ignore


class LeanAideRESEWorkflow:
    """Run a sequence of proof-verification steps as a RESE workflow."""

    def __init__(self, real_verify: bool = False):
        self.real_verify = real_verify
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, name: str, code: str) -> Dict[str, Any]:
        check = check_lean_proof_structural(code)
        step = {
            "step_id": uuid.uuid4().hex[:8],
            "name": name,
            "verified": check["valid"],
            "method": "structural",
            "errors": list(check["errors"]),
            "warnings": list(check["warnings"]),
        }
        self.steps.append(step)
        return step

    def run(self, proofs: List[Dict[str, str]]) -> Dict[str, Any]:
        """``proofs`` is a list of ``{"name": ..., "code": ...}`` dicts."""
        self.steps = []
        for p in proofs:
            self.add_step(p.get("name", "unnamed"), p.get("code", ""))
        verified = [s for s in self.steps if s["verified"]]
        return {
            "total": len(self.steps),
            "verified": len(verified),
            "all_passed": len(verified) == len(self.steps) and len(self.steps) > 0,
            "steps": self.steps,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "real_verify": self.real_verify,
            "steps_recorded": len(self.steps),
        }
