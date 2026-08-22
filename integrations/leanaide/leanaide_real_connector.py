"""LeanAide Real Connector.

Connector that prefers the REAL Lean 4 verification engine (compiler-based)
when the toolchain and a usable project are available, and degrades to the
structural checker otherwise. This mirrors the "real connector" intent: it
attempts genuine verification rather than returning success unconditionally.
"""

import logging
import os
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


class LeanAideRealConnector:
    """Prefer real Lean 4 verification, degrade gracefully when unavailable."""

    def __init__(self, real_verify: bool = True, config: Optional[Any] = None):
        self.real_verify = real_verify
        self._engine = None
        if LEAN4_AVAILABLE and real_verify:
            try:
                cfg = config or Lean4ServerConfig(real_verify=True)
                self._engine = Lean4VerificationEngine(cfg)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not build real Lean engine: %s", exc)
                self._engine = None

    def status(self) -> Dict[str, Any]:
        return {
            "lean4_available": LEAN4_AVAILABLE,
            "engine_active": self._engine is not None,
            "real_verify": self.real_verify,
        }

    def verify(self, code: str) -> Dict[str, Any]:
        structural = check_lean_proof_structural(code)
        if self._engine is None:
            structural["method"] = "structural"
            structural["warnings"] = list(structural.get("warnings", [])) + [
                "Real Lean engine unavailable; structural check only"
            ]
            return structural
        try:
            import asyncio

            real = asyncio.get_event_loop().run_until_complete(
                self._engine.verify(code)
            )
            return {
                "valid": real.success,
                "method": "lean4",
                "errors": list(real.errors),
                "warnings": list(real.warnings),
                "lean_status": real.status.value if hasattr(real.status, "value") else str(real.status),
            }
        except Exception as exc:  # pragma: no cover
            structural["method"] = "structural"
            structural["warnings"] = list(structural.get("warnings", [])) + [
                f"Real verification failed ({exc}); structural check only"
            ]
            return structural
