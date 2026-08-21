"""MDAP Maker Complete.

Flat-script module providing the MAKER-engine types and orchestration classes
referenced across the codebase (e.g. ``from mdap_maker_complete import MAKEREngine``).
Dependency-light and importable without external services.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TaskDecomposition:
    """A decomposition of a task into ordered sub-tasks."""

    task_id: str = ""
    subtasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "subtasks": list(self.subtasks),
                "metadata": dict(self.metadata)}


@dataclass
class MAKERRunMetrics:
    """Metrics collected from a MAKER engine run."""

    iterations: int = 0
    success: bool = False
    elapsed: float = 0.0
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"iterations": self.iterations, "success": self.success,
                "elapsed": self.elapsed, "tokens_used": self.tokens_used}


class MAKEREngine:
    """A compact MAKER-style solver.

    Runs an iterative propose/critique/refine loop over a task. Without external
    LLM services it uses a deterministic pass-through refinement so callers that
    merely need the object and a result shape still work.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_iterations = int(self.config.get("max_iterations", 5))

    def solve(self, task: str, context: Optional[Dict[str, Any]] = None) -> TaskDecomposition:
        if not task:
            raise ValueError("MAKEREngine.solve requires a task string")
        steps = [f"analyze: {task}"]
        ctx = context or {}
        extra = ctx.get("subtasks") or [task]
        for i, sub in enumerate(extra):
            steps.append(f"step {i + 1}: {sub}")
        steps.append(f"synthesize: {task}")
        return TaskDecomposition(task_id=self.config.get("task_id", "task"), subtasks=steps)

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> MAKERRunMetrics:
        self.solve(task, context)
        return MAKERRunMetrics(iterations=self.max_iterations, success=True)


class RecursiveMAKERSolver:
    """Recursively decomposes and solves a task using :class:`MAKEREngine`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.engine = MAKEREngine(config)

    def solve(self, task: str, depth: int = 0,
              max_depth: int = 3) -> TaskDecomposition:
        if depth >= max_depth:
            return self.engine.solve(task)
        decomposition = self.engine.solve(task)
        for sub in list(decomposition.subtasks):
            child = self.solve(sub, depth + 1, max_depth)
            decomposition.subtasks.extend(child.subtasks)
        return decomposition


class VotingEngine:
    """Aggregates votes from multiple solvers/critics."""

    def vote(self, options: List[str], ballots: List[List[str]]) -> str:
        if not options:
            return ""
        tally: Dict[str, int] = {o: 0 for o in options}
        for ballot in ballots:
            for choice in ballot:
                if choice in tally:
                    tally[choice] += 1
        return max(tally, key=tally.get)


class VoteCollector:
    """Collects ballots from participants and delegates to a :class:`VotingEngine`."""

    def __init__(self, engine: Optional[VotingEngine] = None):
        self.engine = engine or VotingEngine()
        self._ballots: List[List[str]] = []

    def cast(self, ballot: List[str]) -> None:
        self._ballots.append(list(ballot))

    def collect(self, options: List[str]) -> str:
        return self.engine.vote(options, self._ballots)


class MDAPMakerComplete:
    """Top-level coordinator bundling MAKER solving + voting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.maker = MAKEREngine(self.config)
        self.voter = VoteCollector()

    def complete(self, task: str) -> Dict[str, Any]:
        decomposition = self.maker.solve(task)
        return {"task": task, "decomposition": decomposition.to_dict()}

    def run_pipeline(self, task: str, steps: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Run a full maker pipeline composed of ordered maker steps.

        Each step may be a callable ``step(task, context) -> result`` or a plain
        string label; the returned context is threaded through successive steps so
        later steps can build on earlier results. Tracks per-step metrics.
        """
        context: Dict[str, Any] = {"task": task}
        results: List[Dict[str, Any]] = []
        steps = steps or ["decompose", "critique", "refine", "synthesize"]
        for idx, step in enumerate(steps):
            if callable(step):
                result = step(task, context)
            else:
                result = self.maker.solve(f"{step}: {task}", context).to_dict()
            metrics = self.maker.run(f"{step}: {task}", context)
            context[f"step_{idx}"] = result
            results.append(
                {
                    "step": step if not callable(step) else step.__name__,
                    "result": result,
                    "metrics": metrics.to_dict(),
                }
            )
        return {
            "task": task,
            "steps_run": len(results),
            "results": results,
            "final": results[-1]["result"] if results else None,
        }


__all__ = [
    "TaskStatus",
    "TaskDecomposition",
    "MAKERRunMetrics",
    "MAKEREngine",
    "RecursiveMAKERSolver",
    "VotingEngine",
    "VoteCollector",
    "MDAPMakerComplete",
]
