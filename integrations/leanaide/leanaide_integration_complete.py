"""LeanAide Integration Complete.

High-level coordinator that ties together the genuine structural proof checker
and (when available) the real Lean 4 verification engine from
``engines.other.lean4_integration``. It never reports success without an
actual verification step.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from leanaide_systems import check_lean_proof_structural
except ImportError:  # pragma: no cover
    from integrations.leanaide.leanaide_systems import check_lean_proof_structural  # type: ignore

try:
    from lean4_integration import (
        Lean4VerificationEngine,
        Lean4ServerConfig,
        VerificationResult,
    )
    LEAN4_ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover
    LEAN4_ENGINE_AVAILABLE = False
    Lean4VerificationEngine = None
    Lean4ServerConfig = None
    VerificationResult = None


class LeanAideIntegrationComplete:
    """End-to-end verification coordinator with graceful degradation."""

    def __init__(self, real_verify: bool = False, config: Optional[Any] = None):
        self.real_verify = real_verify
        self._engine = None
        if LEAN4_ENGINE_AVAILABLE and real_verify:
            try:
                cfg = config or Lean4ServerConfig(real_verify=True)
                self._engine = Lean4VerificationEngine(cfg)
                logger.info("LeanAideIntegrationComplete using real Lean4 engine")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not init Lean4 engine: %s", exc)
                self._engine = None
        else:
            logger.info("LeanAideIntegrationComplete using structural checker")

    async def verify(self, code: str) -> Dict[str, Any]:
        """Verify Lean code, preferring the real engine when configured."""
        structural = check_lean_proof_structural(code)
        result: Dict[str, Any] = {
            "valid": structural["valid"],
            "structural": structural,
            "method": "structural",
            "errors": list(structural["errors"]),
            "warnings": list(structural["warnings"]),
        }
        if self._engine is not None:
            try:
                real: VerificationResult = await self._engine.verify(code)
                result["method"] = "lean4"
                result["valid"] = real.success
                result["errors"] = real.errors
                result["warnings"] = real.warnings
                result["lean_status"] = real.status.value if hasattr(real.status, "value") else str(real.status)
            except Exception as exc:  # pragma: no cover
                result["warnings"].append(f"Real verification failed ({exc}); structural kept")
        return result

    def verify_sync(self, code: str) -> Dict[str, Any]:
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.verify(code))
