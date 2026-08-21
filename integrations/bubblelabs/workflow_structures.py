"""
Workflow data structures (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full project defines these in ``engines/workflow`` and
``openevolve/kernel/schema.py``. This stub carries only the fields that the
BubbleLab modules and their tests in this package actually read and write, so the
package is self-contained and importable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ._stub_support import STUB

__all__ = ["STUB", "WorkflowStatus", "WorkflowStage", "WorkflowState"]


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkflowStatus(str, Enum):
    """Lifecycle status of a workflow instance."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(str, Enum):
    """Coarse-grained stage a workflow instance is currently in."""

    CREATED = "created"
    QUEUED = "queued"
    INITIALIZING = "initializing"
    EVOLVING = "evolving"
    EVALUATING = "evaluating"
    FINALIZING = "finalizing"
    DONE = "done"


@dataclass
class WorkflowState:
    """
    Mutable snapshot of one workflow instance.

    Only ``workflow_id`` is required; every other field has a sane default so
    callers can construct partial states and fill them in incrementally.

    Attributes:
        workflow_id: Unique identifier for the workflow instance.
        workflow_type: Workflow family, e.g. ``"evolution"`` or ``"adversarial"``.
        problem_statement: Human-readable description of the problem being solved.
        current_stage: Current :class:`WorkflowStage` value (as a plain string).
        status: Current :class:`WorkflowStatus` value (as a plain string).
        progress: Completion fraction in ``[0.0, 1.0]``.
        max_iterations: Iteration budget for the run.
        current_iteration: Iterations completed so far.
        population_size: Number of candidates held per generation.
        best_fitness: Best fitness observed so far.
        avg_fitness: Mean fitness of the current population.
        diversity: Population diversity metric in ``[0.0, 1.0]``.
        execution_time: Wall-clock seconds spent executing.
        parameters: Effective parameter set for the run.
        inputs: Caller-supplied inputs for the run.
        error: Failure message when ``status`` is ``"failed"``.
        created_at: Creation timestamp.
        updated_at: Timestamp of the last mutation recorded via :meth:`touch`.
    """

    workflow_id: str
    workflow_type: str = "evolution"
    problem_statement: str = ""
    current_stage: str = WorkflowStage.CREATED.value
    status: str = WorkflowStatus.CREATED.value
    progress: float = 0.0
    max_iterations: int = 100
    current_iteration: int = 0
    population_size: int = 10
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    execution_time: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def touch(self) -> None:
        """Record that the state was just mutated."""
        self.updated_at = _utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-friendly dict view of the state.

        Returns:
            Mapping of every field, with timestamps rendered as ISO-8601 strings.
        """
        data = asdict(self)
        for key in ("created_at", "updated_at"):
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    def summary(self) -> Dict[str, Any]:
        """
        Return the compact status view used by list/status endpoints.

        Returns:
            Mapping with the identity, lifecycle and progress fields.
        """
        return {
            "instance_id": self.workflow_id,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "status": self.status,
            "current_stage": self.current_stage,
            "progress": self.progress,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "best_fitness": self.best_fitness,
            "error": self.error,
        }


def list_status_values() -> List[str]:
    """Return every valid workflow status string."""
    return [status.value for status in WorkflowStatus]
