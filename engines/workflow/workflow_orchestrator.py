"""
Workflow Orchestrator Module

Provides workflow orchestration for OpenEvolve.

This module previously exposed only a thin facade (``orchestrate`` /
``execute`` returned static placeholders). It now implements a real,
dependency-light **DAG executor** that:

* parses a workflow definition (per ``WORKFLOW_ORCHESTRATION_SPEC.md``),
* topologically orders steps and detects cycles,
* runs steps in dependency order with per-step status tracking,
* applies per-step retry policies with exponential backoff,
* guards external ``service`` steps (no handler registered => graceful
  failure, never an unhandled exception),
* optionally executes independent steps concurrently (threaded).

Public names preserved: ``WorkflowOrchestratorConfig``,
``WorkflowOrchestrator``, ``create_orchestrator``.
"""
from __future__ import annotations


import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowOrchestratorConfig:
    """Configuration for workflow orchestrator"""
    max_concurrent: int = 10
    timeout: int = 300
    # Default behaviour when a step fails and declares no explicit policy.
    default_on_failure: str = "stop"  # "stop" | "continue"


# --------------------------------------------------------------------------- #
# DAG execution primitives
# --------------------------------------------------------------------------- #
class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class RetryPolicy:
    """Retry configuration for a single step (see spec: Error Handling)."""
    max_attempts: int = 1
    backoff_coefficient: float = 2.0
    initial_delay_ms: int = 0
    max_delay_ms: int = 60000
    retryable_errors: Optional[List[str]] = None
    non_retryable_errors: Optional[List[str]] = None
    timeout_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RetryPolicy":
        if not data:
            return cls()
        return cls(
            max_attempts=int(data.get("max_attempts", 1)),
            backoff_coefficient=float(data.get("backoff_coefficient", 2.0)),
            initial_delay_ms=int(data.get("initial_delay_ms", 0)),
            max_delay_ms=int(data.get("max_delay_ms", 60000)),
            retryable_errors=data.get("retryable_errors"),
            non_retryable_errors=data.get("non_retryable_errors"),
            timeout_ms=data.get("timeout_ms"),
        )


@dataclass
class StepResult:
    """Outcome of executing a single workflow step."""
    task_id: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    output: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    duration_ms: int = 0
    skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "skipped_reason": self.skipped_reason,
        }


class WorkflowDagError(Exception):
    """Raised for invalid DAGs (e.g. cycles, missing dependencies)."""


@dataclass
class WorkflowExecutionResult:
    """Aggregate result of running a workflow DAG."""
    workflow_id: str
    status: str = "pending"  # pending|running|completed|failed|partial
    steps: Dict[str, StepResult] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outputs": self.outputs,
            "steps": {tid: s.to_dict() for tid, s in self.steps.items()},
        }


class WorkflowDag:
    """
    Build and validate a workflow DAG from a list of task definitions.

    Each task is expected to be a dict with at least ``task_id`` and an optional
    ``dependencies`` list (mirroring the spec's ``definition.tasks`` schema).
    """

    def __init__(self, tasks: List[Dict[str, Any]]) -> None:
        self.tasks: List[Dict[str, Any]] = tasks
        self.task_ids: List[str] = [t.get("task_id") for t in tasks]
        self.by_id: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in tasks}
        self.adj: Dict[str, List[str]] = {}      # task -> dependents
        self.deps: Dict[str, List[str]] = {}      # task -> dependencies
        self._build()

    def _build(self) -> None:
        self.adj = {tid: [] for tid in self.task_ids}
        self.deps = {tid: [] for tid in self.task_ids}
        for t in self.tasks:
            tid = t["task_id"]
            for dep in t.get("dependencies", []) or []:
                if dep not in self.by_id:
                    raise WorkflowDagError(
                        f"Task '{tid}' depends on unknown task '{dep}'")
                self.deps[tid].append(dep)
                self.adj[dep].append(tid)

    def validate(self) -> None:
        """Validate the DAG; raises :class:`WorkflowDagError` on cycle."""
        self.topological_order()

    def topological_order(self) -> List[str]:
        """
        Kahn's algorithm. Returns task ids in dependency order.

        Raises:
            WorkflowDagError: if the graph contains a cycle.
        """
        in_degree = {tid: len(self.deps[tid]) for tid in self.task_ids}
        queue = [tid for tid in self.task_ids if in_degree[tid] == 0]
        order: List[str] = []
        while queue:
            # Deterministic ordering for stable execution.
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for dependent in sorted(self.adj[node]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        if len(order) != len(self.task_ids):
            remaining = [t for t in self.task_ids if t not in set(order)]
            raise WorkflowDagError(
                f"Workflow contains a cycle involving: {remaining}")
        return order

    def execution_levels(self) -> List[List[str]]:
        """Group tasks into dependency levels (level N runs after N-1)."""
        order = self.topological_order()
        level_of: Dict[str, int] = {}
        for tid in order:
            deps = self.deps[tid]
            level = 0 if not deps else (max(level_of[d] for d in deps) + 1)
            level_of[tid] = level
        max_level = max(level_of.values()) if level_of else -1
        levels: List[List[str]] = [[] for _ in range(max_level + 1)]
        for tid, lvl in level_of.items():
            levels[lvl].append(tid)
        return levels


class StepRunner:
    """
    Executes a single step with retry + timeout, guarding external calls.

    The handler is invoked inside a worker thread so that ``timeout_ms`` can be
    enforced cross-platform (signals only work on the main thread on Unix).
    """

    def __init__(self, default_on_failure: str = "stop") -> None:
        self.default_on_failure = default_on_failure

    def run(
        self,
        task: Dict[str, Any],
        handler: Optional[Callable[..., Any]],
        resolved_inputs: Dict[str, Any],
        retry_policy: RetryPolicy,
    ) -> StepResult:
        tid = task["task_id"]
        result = StepResult(task_id=tid)
        on_failure = task.get("on_failure") or self.default_on_failure

        # External service steps without a handler are guarded: they must not
        # raise. We record a clear, actionable error and let the policy decide.
        if handler is None:
            svc = task.get("service") or task.get("function") or task.get("type")
            result.status = StepStatus.FAILED
            result.error = (
                f"No handler registered for step '{tid}' "
                f"(service/function: {svc})")
            return result

        attempt = 0
        last_error: Optional[str] = None
        delay = retry_policy.initial_delay_ms / 1000.0

        while attempt < max(1, retry_policy.max_attempts):
            attempt += 1
            result.attempts = attempt
            result.status = StepStatus.RUNNING
            started = time.monotonic()
            result.started_at = self._now()
            try:
                output = self._invoke(handler, resolved_inputs, retry_policy.timeout_ms)
                result.output = output
                result.status = StepStatus.COMPLETED
                result.ended_at = self._now()
                result.duration_ms = int((time.monotonic() - started) * 1000)
                return result
            except Exception as exc:  # noqa: BLE001 - capture & classify
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Step '%s' attempt %d failed: %s",
                               tid, attempt, last_error)
                result.error = last_error
                if not self._is_retryable(exc, retry_policy):
                    break
                if attempt < retry_policy.max_attempts:
                    time.sleep(delay)
                    delay = min(delay * retry_policy.backoff_coefficient,
                                retry_policy.max_delay_ms / 1000.0)

        result.status = StepStatus.FAILED
        result.ended_at = self._now()
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    # ------------------------------------------------------------------ #
    def _invoke(self, handler, inputs, timeout_ms):
        call = _build_call(handler, inputs)
        if timeout_ms is None:
            return call()
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(call)
            return future.result(timeout=timeout_ms / 1000.0)

    @staticmethod
    def _is_retryable(exc: Exception, policy: RetryPolicy) -> bool:
        # Include the exception type name so class-based keywords
        # (e.g. "connection", "timeout") match even when the message alone
        # does not mention them.
        msg = f"{type(exc).__name__}: {exc}".lower()
        if policy.non_retryable_errors:
            if any(n.lower() in msg for n in policy.non_retryable_errors):
                return False
        if policy.retryable_errors:
            return any(r.lower() in msg for r in policy.retryable_errors)
        # Default classification (mirrors spec: transient keywords are retryable).
        transient = ("timeout", "connection", "network", "temporar",
                     "busy", "rate limit", "unavailable", "503", "429")
        permanent = ("invalid", "validation", "valueerror", "keyerror",
                     "typeerror", "not found", "404")
        if any(p in msg for p in permanent):
            return False
        return any(t in msg for t in transient)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def _build_call(handler: Callable, inputs: Dict[str, Any]) -> Callable[[], Any]:
    """
    Return a zero-argument callable that invokes ``handler`` with ``inputs``.

    Calling convention (best-effort, dependency-light):
    * handler accepts ``**kwargs`` -> ``handler(**inputs)``
    * handler accepts no parameters       -> ``handler()``
    * otherwise                            -> ``handler(inputs)`` (single dict)
    """
    import inspect
    try:
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params)
    except (ValueError, TypeError):
        has_var_kw = False
        params = []

    if has_var_kw:
        return lambda: handler(**inputs)
    if not params:
        return lambda: handler()
    return lambda: handler(inputs)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class WorkflowOrchestrator:
    """Workflow Orchestrator with a real DAG execution engine."""

    def __init__(self, config: Optional[WorkflowOrchestratorConfig] = None):
        self.config = config or WorkflowOrchestratorConfig()
        self.runner = StepRunner(self.config.default_on_failure)
        logger.info("Workflow Orchestrator initialized")

    # --- public facade (preserved signatures) -------------------------- #
    def orchestrate(
        self,
        workflow: Dict[str, Any],
        handlers: Optional[Dict[str, Callable[..., Any]]] = None,
        max_concurrent: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate a workflow.

        If ``workflow`` carries a spec-compliant ``definition`` (with ``tasks``),
        it is executed as a DAG; otherwise the previous passthrough behaviour is
        retained for backward compatibility. ``handlers`` maps step ids /
        function / service names to callables and is forwarded to the executor.
        """
        definition = workflow.get("definition")
        if isinstance(definition, dict) and definition.get("tasks"):
            result = self.execute_workflow(
                definition,
                inputs=workflow.get("inputs"),
                handlers=handlers,
                max_concurrent=max_concurrent or self.config.max_concurrent,
            )
            return result.to_dict()
        return {"orchestrated": True, "workflow": workflow, "handlers": bool(handlers)}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task via an ad-hoc DAG of one step."""
        definition = {
            "tasks": [task],
            "connections": [],
            "start_task": task.get("task_id"),
            "end_tasks": [task.get("task_id")],
        }
        result = self.execute_workflow(definition, inputs=task.get("inputs"))
        steps = result.steps
        tid = task.get("task_id")
        step = steps.get(tid)
        return {
            "executed": step.status == StepStatus.COMPLETED if step else False,
            "task": task,
            "result": step.to_dict() if step else None,
        }

    # --- real DAG execution ------------------------------------------- #
    def execute_workflow(
        self,
        definition: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        handlers: Optional[Dict[str, Callable[..., Any]]] = None,
        max_concurrent: Optional[int] = None,
        raise_on_failure: bool = False,
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow DAG.

        Args:
            definition: spec-compliant ``definition`` dict (``tasks`` etc.).
            inputs: top-level workflow inputs (referenced as
                ``${workflow.inputs.X}``).
            handlers: mapping used to run steps. Resolution order per step:
                1. ``handlers[task_id]``
                2. ``handlers[task["function"]]`` (function steps)
                3. ``handlers[task["service"]]`` (service steps)
                If none match, the step fails gracefully (external service
                guarded).
            max_concurrent: max threads per execution level (defaults to config).
            raise_on_failure: if True, raise on the first failed step instead of
                returning a FAILED result.

        Returns:
            :class:`WorkflowExecutionResult` with per-step statuses.
        """
        tasks = definition.get("tasks", [])
        workflow_id = definition.get("workflow_id") or "workflow"
        handlers = handlers or {}
        max_concurrent = max_concurrent or self.config.max_concurrent or 1

        result = WorkflowExecutionResult(workflow_id=workflow_id)
        result.started_at = WorkflowOrchestrator._now()
        for t in tasks:
            result.steps[t["task_id"]] = StepResult(task_id=t["task_id"])

        dag = WorkflowDag(tasks)
        try:
            dag.validate()
        except WorkflowDagError as exc:
            result.status = "failed"
            result.ended_at = self._now()
            if raise_on_failure:
                raise
            logger.error("Invalid workflow DAG: %s", exc)
            return result

        task_outputs: Dict[str, Any] = {}
        workflow_ctx = {"inputs": inputs or {}}

        for level in dag.execution_levels():
            if max_concurrent > 1 and len(level) > 1:
                with ThreadPoolExecutor(
                        max_workers=min(max_concurrent, len(level))) as ex:
                    futures = {
                        tid: ex.submit(
                            self._run_step, dag.by_id[tid], handlers,
                            workflow_ctx, task_outputs, result)
                        for tid in level
                    }
                    for tid, fut in futures.items():
                        result.steps[tid] = fut.result()
            else:
                for tid in level:
                    result.steps[tid] = self._run_step(
                        dag.by_id[tid], handlers, workflow_ctx,
                        task_outputs, result)

        self._finalize(result, dag, raise_on_failure)
        return result

    # ------------------------------------------------------------------ #
    def _run_step(self, task, handlers, workflow_ctx, task_outputs, result):
        tid = task["task_id"]
        step = result.steps[tid]

        # Dependency gate: skip if any dependency did not complete.
        deps = task.get("dependencies", []) or []
        failed_deps = [d for d in deps
                       if result.steps[d].status != StepStatus.COMPLETED]
        if failed_deps:
            step.status = StepStatus.SKIPPED
            step.skipped_reason = (
                f"dependencies not completed: {failed_deps}")
            step.error = step.skipped_reason
            logger.info("Skipping step '%s': %s", tid, step.skipped_reason)
            return step

        handler = self._resolve_handler(task, handlers)
        resolved = self._resolve_inputs(task, workflow_ctx, task_outputs)
        policy = RetryPolicy.from_dict(task.get("retry_policy"))
        step = self.runner.run(task, handler, resolved, policy)
        result.steps[tid] = step

        if step.status == StepStatus.COMPLETED:
            task_outputs[tid] = self._store_outputs(task, step.output)
        return step

    def _resolve_handler(self, task, handlers):
        tid = task["task_id"]
        if tid in handlers:
            return handlers[tid]
        if task.get("type") == "function" and task.get("function") in handlers:
            return handlers[task["function"]]
        if task.get("type") == "service" and task.get("service") in handlers:
            return handlers[task["service"]]
        # condition tasks may resolve to a plain boolean function
        if task.get("condition") and task.get("function") in handlers:
            return handlers[task["function"]]
        return None

    def _resolve_inputs(self, task, workflow_ctx, task_outputs):
        raw = task.get("inputs", {}) or {}
        resolved = {}
        for key, val in raw.items():
            resolved[key] = self._resolve_ref(val, workflow_ctx, task_outputs)
        return resolved

    def _resolve_ref(self, value, workflow_ctx, task_outputs):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            expr = value[2:-1].strip()
            parts = expr.split(".")
            if parts[0] == "workflow" and parts[1] == "inputs":
                return _deep_get(workflow_ctx["inputs"], parts[2:])
            if parts[0] == "tasks":
                tid = parts[1]
                out = task_outputs.get(tid, {})
                if len(parts) >= 4 and parts[2] == "outputs":
                    return _deep_get(out, parts[3:])
                return out
        return value

    @staticmethod
    def _store_outputs(task, output):
        declared = task.get("outputs", []) or []
        if not declared:
            return output
        if isinstance(output, dict):
            return {name: output.get(name, output) for name in declared}
        return {name: output for name in declared}

    def _finalize(self, result, dag, raise_on_failure):
        statuses = [s.status for s in result.steps.values()]
        failed = [tid for tid, s in result.steps.items()
                  if s.status == StepStatus.FAILED]
        skipped = [tid for tid, s in result.steps.items()
                   if s.status == StepStatus.SKIPPED]

        # Collect outputs of completed steps for the workflow result.
        for tid, s in result.steps.items():
            if s.status == StepStatus.COMPLETED:
                result.outputs[tid] = s.output

        if failed:
            result.status = "failed"
        elif skipped and not [s for s in result.steps.values()
                              if s.status == StepStatus.COMPLETED]:
            result.status = "failed"
        elif skipped:
            result.status = "partial"
        else:
            result.status = "completed"
        result.ended_at = self._now()

        if raise_on_failure and failed:
            raise WorkflowDagError(
                f"Workflow '{result.workflow_id}' failed at steps: {failed}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


def _deep_get(obj, keys):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def create_orchestrator(config: Optional[WorkflowOrchestratorConfig] = None) -> WorkflowOrchestrator:
    """Factory function to create orchestrator instance"""
    return WorkflowOrchestrator(config)
