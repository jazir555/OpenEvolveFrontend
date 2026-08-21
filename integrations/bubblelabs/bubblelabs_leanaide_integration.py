"""
BubbleLabs <-> LeanAide bridge (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full bridge drives LeanAide, Lean 4 and the MCTS/MDAP
search stack. None of those backends are available from this package, so the
capability flags below are all ``False`` and
:meth:`LeanAideIntegrationBridge.execute_task` fails closed. Consumers in this
package already branch on the flags, so they degrade gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from ._stub_support import STUB, raise_stub
except ImportError:
    from _stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

__all__ = [
    "STUB",
    "LeanAideTaskType",
    "LeanAideResult",
    "LeanAideIntegrationBridge",
    "get_leanaide_bridge",
    "initialize_leanaide_integration",
    "LEANAIDE_AVAILABLE",
    "MCTS_AVAILABLE",
    "MDAP_AVAILABLE",
    "LEAN4_AVAILABLE",
]

#: No LeanAide server reachable from this stub.
LEANAIDE_AVAILABLE: bool = False
#: No MCTS search backend reachable from this stub.
MCTS_AVAILABLE: bool = False
#: No MDAP backend reachable from this stub.
MDAP_AVAILABLE: bool = False
#: No Lean 4 toolchain reachable from this stub.
LEAN4_AVAILABLE: bool = False


class LeanAideTaskType(str, Enum):
    """Kinds of task the LeanAide bridge can be asked to perform."""

    TRANSLATE_THEOREM = "translate_theorem"
    GENERATE_PROOF = "generate_proof"
    VERIFY_SOLUTION = "verify_solution"
    ELABORATE_CODE = "elaborate_code"
    MATH_QUERY = "math_query"
    MCTS_SEARCH = "mcts_search"


@dataclass
class LeanAideResult:
    """
    Outcome of a LeanAide task.

    Attributes:
        task_type: The task that was requested.
        success: Whether the task succeeded.
        output: Primary textual output (Lean code, proof, answer, ...).
        details: Additional structured result data.
        error: Failure message when ``success`` is ``False``.
    """

    task_type: LeanAideTaskType
    success: bool = False
    output: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class LeanAideIntegrationBridge:
    """
    Bridge to the LeanAide formal-mathematics stack.

    Attributes:
        available: Whether any LeanAide backend is reachable (always ``False``
            in this stub).
    """

    def __init__(self) -> None:
        self.available: bool = LEANAIDE_AVAILABLE

    def status(self) -> Dict[str, bool]:
        """
        Report backend availability.

        Returns:
            Mapping of backend name to availability flag.
        """
        return {
            "leanaide": LEANAIDE_AVAILABLE,
            "mcts": MCTS_AVAILABLE,
            "mdap": MDAP_AVAILABLE,
            "lean4": LEAN4_AVAILABLE,
        }

    def supported_tasks(self) -> List[LeanAideTaskType]:
        """Return every task type the real bridge understands."""
        return list(LeanAideTaskType)

    def execute_task(
        self,
        task_type: LeanAideTaskType,
        payload: Optional[Dict[str, Any]] = None,
    ) -> LeanAideResult:
        """
        Execute a LeanAide task.

        Args:
            task_type: Task to run.
            payload: Task inputs.

        Returns:
            The task result.

        Raises:
            StubNotImplementedError: Always - requires a LeanAide backend.
        """
        raise_stub(
            f"LeanAideIntegrationBridge.execute_task({task_type})",
            hint="dispatch to a running LeanAide server / Lean 4 toolchain",
        )


_bridge: Optional[LeanAideIntegrationBridge] = None


def get_leanaide_bridge() -> LeanAideIntegrationBridge:
    """
    Return the process-wide LeanAide bridge, creating it on first use.

    Returns:
        The shared :class:`LeanAideIntegrationBridge`.
    """
    global _bridge
    if _bridge is None:
        _bridge = LeanAideIntegrationBridge()
    return _bridge


def initialize_leanaide_integration() -> Dict[str, Any]:
    """
    Initialise the LeanAide integration and report what is available.

    Returns:
        Mapping with an ``available`` flag, per-backend ``backends`` status and
        a human-readable ``message``.
    """
    bridge = get_leanaide_bridge()
    backends = bridge.status()
    return {
        "available": any(backends.values()),
        "backends": backends,
        "message": "stub: implement - no LeanAide backend is reachable from integrations.bubblelabs",
    }
