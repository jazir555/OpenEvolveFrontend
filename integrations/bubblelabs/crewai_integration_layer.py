"""
CrewAI integration layer (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the real layer lives at
``integrations/crewai/crewai_integration_layer.py`` and orchestrates CrewAI
agents/crews. It is not importable from this package under the
``integrations.bubblelabs.*`` namespace, so :mod:`.bubblelab_crewai_mcp_server`
binds to this local stub instead. Crew execution fails closed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from ._stub_support import STUB, raise_stub
except ImportError:
    from _stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

__all__ = ["STUB", "CrewAIService", "get_crewai_service"]


class CrewAIService:
    """
    Facade over CrewAI crew orchestration.

    Args:
        config: Optional service configuration.

    Attributes:
        config: Service configuration.
        sessions: Session id -> session metadata for registered crews.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        self.sessions: Dict[str, Dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        """Whether a CrewAI backend is wired up (always ``False`` in this stub)."""
        return False

    def health(self) -> Dict[str, Any]:
        """
        Report service reachability.

        Returns:
            Mapping with a ``healthy`` flag and diagnostic ``detail``. Returned
            rather than raised so callers can probe availability cheaply.
        """
        return {
            "healthy": False,
            "detail": "stub: implement - CrewAI orchestration is not wired up in integrations.bubblelabs",
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List configured CrewAI agents.

        Returns:
            An empty list - no agents are configured in this stub.
        """
        return []

    def create_crew(self, name: str, agents: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create a CrewAI crew.

        Args:
            name: Crew name.
            agents: Agent specifications.

        Returns:
            The created crew's metadata.

        Raises:
            StubNotImplementedError: Always - requires the CrewAI runtime.
        """
        raise_stub(
            "CrewAIService.create_crew",
            hint="build a crewai.Crew from the agent specs and register it",
        )

    def execute_task(self, task: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delegate a task to a CrewAI crew.

        Args:
            task: Task specification.
            session_id: Optional existing session to run within.

        Returns:
            The task result mapping.

        Raises:
            StubNotImplementedError: Always - requires the CrewAI runtime.
        """
        raise_stub(
            "CrewAIService.execute_task",
            hint="kick off the crew and return its output",
        )


_service: Optional[CrewAIService] = None


def get_crewai_service(config: Optional[Dict[str, Any]] = None) -> CrewAIService:
    """
    Return the process-wide CrewAI service, creating it on first use.

    Args:
        config: Configuration used only when the service is first created.

    Returns:
        The shared :class:`CrewAIService`.
    """
    global _service
    if _service is None:
        _service = CrewAIService(config)
    return _service
