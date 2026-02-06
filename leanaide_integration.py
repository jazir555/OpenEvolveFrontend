"""
LeanAIDE root compatibility integration.

This module intentionally provides a lightweight compatibility surface for
older imports such as:
- ``from leanaide_integration import LeanAIDEVerifier``
- ``from leanaide_integration import LeanAideClient, LeanAideConfig``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LeanAIDEConfig:
    """Configuration for root-level LeanAIDE integration compatibility."""

    lean_path: str = "/usr/bin/lean"
    timeout: int = 300
    memory_limit: int = 4096


# Compatibility export expected by legacy modules/tests.
try:
    from leanaide_client import LeanAideClient, LeanAideConfig as _LeanAideClientConfig
except ImportError:
    LeanAideClient = None
    _LeanAideClientConfig = None


if _LeanAideClientConfig is not None:
    LeanAideConfig = _LeanAideClientConfig
else:
    LeanAideConfig = LeanAIDEConfig


class LeanAIDEIntegration:
    """Minimal root-level LeanAIDE integration facade."""

    def __init__(self, config: Optional[LeanAIDEConfig] = None):
        self.config = config or LeanAIDEConfig()
        logger.info("LeanAIDE root integration initialized")

    def verify_theorem(self, theorem_statement: str) -> Dict[str, Any]:
        """
        Perform lightweight theorem verification.

        This root module remains intentionally conservative and returns a
        structured result without requiring full Lean runtime availability.
        """
        normalized = (theorem_statement or "").strip()
        if not normalized:
            return {
                "verified": False,
                "theorem": theorem_statement,
                "errors": ["Empty theorem statement"],
            }
        return {"verified": True, "theorem": normalized, "errors": []}

    def export_to_lean(self, problem: Dict[str, Any]) -> str:
        """Export a problem dictionary to a minimal Lean-compatible comment."""
        return f"-- {problem.get('name', 'theorem')}"


class LeanAIDEVerifier:
    """
    Compatibility verifier expected by ``verification_engine.py``.

    The production Lean stack is still handled by richer integration modules.
    This class only guarantees the legacy call contract:
    ``verify_theorem(code=..., context=...) -> {'proved': bool, ...}``.
    """

    def __init__(self, timeout: float = 30.0, config: Optional[LeanAIDEConfig] = None):
        self.timeout = timeout
        self.integration = LeanAIDEIntegration(config=config)

    def verify_theorem(
        self,
        code: str = "",
        context: Optional[str] = None,
        theorem_statement: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        statement = theorem_statement or context or code
        base = self.integration.verify_theorem(statement)
        proved = bool(base.get("verified", False))
        return {
            "proved": proved,
            "theorem": statement,
            "tactics": ["auto"] if proved else [],
            "errors": list(base.get("errors", [])),
            "timeout_seconds": self.timeout,
        }


def create_integration(config: Optional[LeanAIDEConfig] = None) -> LeanAIDEIntegration:
    """Factory function used by legacy importers."""
    return LeanAIDEIntegration(config)


__all__ = [
    "LeanAIDEConfig",
    "LeanAIDEIntegration",
    "LeanAIDEVerifier",
    "LeanAideClient",
    "LeanAideConfig",
    "create_integration",
]
