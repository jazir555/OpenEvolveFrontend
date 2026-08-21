"""
OpenEvolve <-> BubbleLabs workflow API (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full implementation drives real OpenEvolve evolution
runs. This stub is a faithful, purely in-memory model of the control surface:
workflow definitions and instances are created, listed and transitioned exactly
as the real API does, but no evolution is executed. That keeps the package and
its test suite runnable while making the missing execution backend explicit -
started instances stop at ``pending`` rather than pretending to make progress.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ._stub_support import STUB
from .workflow_structures import WorkflowStage, WorkflowState, WorkflowStatus

logger = logging.getLogger(__name__)

__all__ = [
    "STUB",
    "WorkflowStatus",
    "WorkflowMetrics",
    "OpenEvolveBubbleLabsIntegration",
]


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class WorkflowMetrics:
    """
    Aggregate metrics for one workflow instance.

    Attributes:
        instance_id: Identifier of the workflow instance.
        iterations: Iterations completed.
        best_fitness: Best fitness observed.
        avg_fitness: Mean fitness of the population.
        diversity: Population diversity in ``[0.0, 1.0]``.
        execution_time: Wall-clock seconds spent executing.
        collected_at: When the sample was taken.
    """

    instance_id: str
    iterations: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    execution_time: float = 0.0
    collected_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict view of the metrics."""
        return {
            "instance_id": self.instance_id,
            "iterations": self.iterations,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "diversity": self.diversity,
            "execution_time": self.execution_time,
            "collected_at": self.collected_at.isoformat(),
        }


class OpenEvolveBubbleLabsIntegration:
    """
    In-memory workflow definition/instance registry.

    Each instance owns its own state - nothing is shared at module level - so
    tests and concurrent callers stay isolated.

    Attributes:
        workflow_definitions: Definition id -> definition dict.
        workflow_instances: Instance id -> :class:`WorkflowState`.
    """

    def __init__(self) -> None:
        self.workflow_definitions: Dict[str, Dict[str, Any]] = {}
        self.workflow_instances: Dict[str, WorkflowState] = {}

    # -- definitions ----------------------------------------------------------

    def create_workflow_definition(
        self,
        name: str,
        description: str = "",
        workflow_type: str = "evolution",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a reusable workflow definition.

        Args:
            name: Human-readable definition name.
            description: Longer description of the definition.
            workflow_type: Workflow family, e.g. ``"evolution"``.
            parameters: Default parameters applied to instances of this
                definition.

        Returns:
            The new definition's identifier.
        """
        definition_id = str(uuid.uuid4())
        self.workflow_definitions[definition_id] = {
            "id": definition_id,
            "name": name,
            "description": description,
            "workflow_type": workflow_type,
            "parameters": dict(parameters or {}),
            "created_at": _utcnow().isoformat(),
        }
        logger.debug("Created workflow definition %s (%s)", definition_id, name)
        return definition_id

    def get_workflow_definition(self, definition_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up a workflow definition.

        Args:
            definition_id: Definition identifier.

        Returns:
            The definition dict, or ``None`` when it is unknown.
        """
        return self.workflow_definitions.get(definition_id)

    def list_workflow_definitions(self) -> List[Dict[str, Any]]:
        """
        List all registered workflow definitions.

        Returns:
            One dict per definition, each with ``id``, ``name``,
            ``workflow_type``, ``description`` and ``parameters``.
        """
        return list(self.workflow_definitions.values())

    # -- instances ------------------------------------------------------------

    def create_workflow_instance(
        self,
        definition_id: str,
        instance_name: str = "",
        inputs: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Instantiate a workflow from a definition.

        Definition parameters are used as defaults and overridden by
        ``parameters``.

        Args:
            definition_id: Definition to instantiate.
            instance_name: Friendly name for this instance.
            inputs: Run inputs (e.g. ``{"content": ...}``).
            parameters: Per-instance parameter overrides.

        Returns:
            The new instance's identifier.

        Raises:
            KeyError: If ``definition_id`` is not registered.
        """
        definition = self.workflow_definitions.get(definition_id)
        if definition is None:
            raise KeyError(f"Unknown workflow definition: {definition_id}")

        effective: Dict[str, Any] = dict(definition["parameters"])
        effective.update(parameters or {})

        instance_id = str(uuid.uuid4())
        state = WorkflowState(
            workflow_id=instance_id,
            workflow_type=definition["workflow_type"],
            problem_statement=str(effective.get("problem_statement", definition.get("description", ""))),
            current_stage=WorkflowStage.CREATED.value,
            status=WorkflowStatus.CREATED.value,
            max_iterations=int(effective.get("max_iterations", 100)),
            population_size=int(effective.get("population_size", 10)),
            parameters=effective,
            inputs=dict(inputs or {}),
        )
        self.workflow_instances[instance_id] = state
        logger.debug(
            "Created workflow instance %s (%s) from definition %s",
            instance_id,
            instance_name,
            definition_id,
        )
        return instance_id

    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowState]:
        """
        Look up a workflow instance's state object.

        Args:
            instance_id: Instance identifier.

        Returns:
            The :class:`WorkflowState`, or ``None`` when unknown.
        """
        return self.workflow_instances.get(instance_id)

    def get_workflow_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Return the status view of one workflow instance.

        Args:
            instance_id: Instance identifier.

        Returns:
            The instance's status mapping, or ``{"status": "unknown", ...}``
            when the instance is not registered.
        """
        state = self.workflow_instances.get(instance_id)
        if state is None:
            return {"instance_id": instance_id, "status": "unknown", "error": "instance not found"}
        return state.summary()

    def list_workflow_instances(self) -> List[Dict[str, Any]]:
        """
        List all workflow instances.

        Returns:
            One status mapping per instance.
        """
        return [state.summary() for state in self.workflow_instances.values()]

    # -- lifecycle transitions ------------------------------------------------

    def _transition(
        self,
        instance_id: str,
        status: WorkflowStatus,
        stage: WorkflowStage,
        message: str,
    ) -> Dict[str, Any]:
        """
        Apply a lifecycle transition to an instance.

        Args:
            instance_id: Instance identifier.
            status: New status.
            stage: New stage.
            message: Human-readable result message.

        Returns:
            Mapping with ``message``, ``instance_id`` and resulting ``status``.
        """
        state = self.workflow_instances.get(instance_id)
        if state is None:
            return {"message": f"Unknown workflow instance: {instance_id}", "instance_id": instance_id, "status": "unknown"}

        state.status = status.value
        state.current_stage = stage.value
        state.touch()
        return {"message": message, "instance_id": instance_id, "status": state.status}

    def start_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Queue a workflow instance for execution.

        No execution backend is wired up in this stub, so the instance is left
        in ``pending`` rather than transitioning to ``running``.

        Args:
            instance_id: Instance identifier.

        Returns:
            Mapping with ``message``, ``instance_id`` and ``status``.
        """
        return self._transition(
            instance_id,
            WorkflowStatus.PENDING,
            WorkflowStage.QUEUED,
            "Workflow queued (stub: no execution backend attached)",
        )

    def pause_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """Pause a workflow instance and return the resulting status mapping."""
        return self._transition(instance_id, WorkflowStatus.PAUSED, WorkflowStage.QUEUED, "Workflow paused")

    def resume_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """Resume a paused workflow instance and return the status mapping."""
        return self._transition(instance_id, WorkflowStatus.PENDING, WorkflowStage.QUEUED, "Workflow resumed")

    def stop_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """Cancel a workflow instance and return the resulting status mapping."""
        return self._transition(instance_id, WorkflowStatus.CANCELLED, WorkflowStage.DONE, "Workflow cancelled")

    # -- metrics --------------------------------------------------------------

    def get_workflow_metrics(self, instance_id: str) -> Optional[WorkflowMetrics]:
        """
        Snapshot the metrics of one workflow instance.

        Args:
            instance_id: Instance identifier.

        Returns:
            A :class:`WorkflowMetrics` sample, or ``None`` when unknown.
        """
        state = self.workflow_instances.get(instance_id)
        if state is None:
            return None
        return WorkflowMetrics(
            instance_id=instance_id,
            iterations=state.current_iteration,
            best_fitness=state.best_fitness,
            avg_fitness=state.avg_fitness,
            diversity=state.diversity,
            execution_time=state.execution_time,
        )
