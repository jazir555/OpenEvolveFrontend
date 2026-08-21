"""
Unified math service (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the real service fronts Z3, Lean 4 and the CAV-NLP
canonicalisers. :mod:`.bubblelabs_node_completion` imports this behind a
``try/except ImportError`` and sets ``CAV_NLP_AVAILABLE`` accordingly, so a
fail-closed stub is the honest option here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

__all__ = ["STUB", "UnifiedMathService"]


class UnifiedMathService:
    """
    Facade over the mathematical reasoning backends.

    Args:
        config: Optional backend configuration.

    Attributes:
        config: Backend configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})

    def available_backends(self) -> List[str]:
        """
        List reachable backends.

        Returns:
            An empty list - no backend is wired up in this stub.
        """
        return []

    def canonicalize(self, expression: str) -> str:
        """
        Canonicalise a mathematical expression.

        Args:
            expression: Expression to canonicalise.

        Returns:
            The canonical form.

        Raises:
            StubNotImplementedError: Always - requires the CAV-NLP canonicaliser.
        """
        raise_stub(
            "UnifiedMathService.canonicalize",
            hint="delegate to the CAV-NLP canonicalizer",
        )

    def solve(self, problem: str, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Solve or decide a mathematical problem.

        Args:
            problem: Problem statement.
            timeout: Solver timeout in seconds.

        Returns:
            Mapping describing the solution.

        Raises:
            StubNotImplementedError: Always - requires a solver backend.
        """
        raise_stub(
            "UnifiedMathService.solve",
            hint="dispatch to Z3 / Lean 4 via the real unified math service",
        )
