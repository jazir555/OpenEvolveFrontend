"""LeanAide Proof Integration.

Integrates a Lean proof through genuine structural verification and (when
configured) the real Lean 4 engine, producing a structured verification
record. Never returns a passing verdict without an actual check.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from leanaide_systems import check_lean_proof_structural
except ImportError:  # pragma: no cover
    from integrations.leanaide.leanaide_systems import check_lean_proof_structural  # type: ignore

try:
    from lean4_integration import Lean4VerificationEngine, Lean4ServerConfig
    LEAN4_AVAILABLE = True
except Exception:  # pragma: no cover
    LEAN4_AVAILABLE = False
    Lean4VerificationEngine = None
    Lean4ServerConfig = None


@dataclass
class ProofIntegrationResult:
    name: str
    verified: bool
    method: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "verified": self.verified,
            "method": self.method,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


class LeanAideProofIntegration:
    """Integrate a named proof into the verification pipeline."""

    def __init__(self, real_verify: bool = False):
        self.real_verify = real_verify
        self._engine = None
        if LEAN4_AVAILABLE and real_verify:
            try:
                self._engine = Lean4VerificationEngine(Lean4ServerConfig(real_verify=True))
            except Exception:  # pragma: no cover
                self._engine = None

    def integrate(self, name: str, code: str) -> ProofIntegrationResult:
        structural = check_lean_proof_structural(code)
        result = ProofIntegrationResult(
            name=name,
            verified=structural["valid"],
            method="structural",
            errors=list(structural["errors"]),
            warnings=list(structural["warnings"]),
            details=structural.get("details", {}),
        )
        if self._engine is not None:
            try:
                import asyncio

                real = asyncio.get_event_loop().run_until_complete(
                    self._engine.verify(code)
                )
                result.method = "lean4"
                result.verified = real.success
                result.errors = list(real.errors)
                result.warnings = list(real.warnings)
            except Exception as exc:  # pragma: no cover
                result.warnings.append(f"Real verification failed ({exc})")
        return result
