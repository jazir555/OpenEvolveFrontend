"""
BubbleLabs core integration (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full module talks to the BubbleLabs workflow service.
This stub keeps workflow definitions in memory (safe, no backend needed) and
fails closed on :meth:`BubbleLabsIntegration.execute_workflow`, which genuinely
requires the remote runtime.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ._stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

__all__ = ["STUB", "BubbleWorkflowDefinition", "BubbleLabsIntegration"]


@dataclass
class BubbleWorkflowDefinition:
    """
    Declarative BubbleLabs workflow (a graph of "bubbles").

    Attributes:
        id: Unique workflow identifier.
        name: Human-readable workflow name.
        description: Longer description of what the workflow does.
        nodes: Bubble/node dictionaries.
        edges: Edge dictionaries connecting nodes.
        parameters: Workflow-level parameters.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict view of the definition."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "parameters": dict(self.parameters),
        }


class BubbleLabsIntegration:
    """
    In-memory registry of :class:`BubbleWorkflowDefinition` objects.

    Attributes:
        workflows: Workflow id -> definition.
    """

    def __init__(self) -> None:
        self.workflows: Dict[str, BubbleWorkflowDefinition] = {}

    def create_workflow(
        self,
        name: str,
        description: str = "",
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> BubbleWorkflowDefinition:
        """
        Register a new workflow definition.

        Args:
            name: Workflow name.
            description: Workflow description.
            nodes: Bubble/node dictionaries.
            edges: Edge dictionaries.
            parameters: Workflow-level parameters.

        Returns:
            The stored :class:`BubbleWorkflowDefinition`.
        """
        definition = BubbleWorkflowDefinition(
            name=name,
            description=description,
            nodes=list(nodes or []),
            edges=list(edges or []),
            parameters=dict(parameters or {}),
        )
        self.workflows[definition.id] = definition
        return definition

    def get_workflow(self, workflow_id: str) -> Optional[BubbleWorkflowDefinition]:
        """
        Look up a workflow definition.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            The definition, or ``None`` when unknown.
        """
        return self.workflows.get(workflow_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """Return dict views of every registered workflow."""
        return [definition.to_dict() for definition in self.workflows.values()]

    def execute_workflow(self, workflow_id: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a workflow on the BubbleLabs runtime.

        Args:
            workflow_id: Workflow to execute.
            inputs: Run inputs.

        Returns:
            The execution result mapping.

        Raises:
            StubNotImplementedError: Always - requires the BubbleLabs runtime.
        """
        raise_stub(
            "BubbleLabsIntegration.execute_workflow",
            hint="POST the workflow to the BubbleLabs runtime and stream back results",
        )
