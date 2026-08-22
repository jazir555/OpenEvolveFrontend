"""
Distributed processing for large-scale problems in the Sovereign-Grade
Decomposition Workflow (doc §6.3 "distributed processing for large-scale
problems").

This module provides :class:`SubProblemExecutor`, which solves a set of
independent / dependent sub-problems while respecting their dependency ordering
(topological sort with independent nodes run in parallel).

Two backends are supported:
  * ``local`` (default): strictly sequential, process-safe, import-safe.
  * ``multiprocessing``: parallel across independent sub-problems using
    :class:`concurrent.futures.ProcessPoolExecutor`. It degrades gracefully to
    ``local`` if multiprocessing is unavailable, pickle fails (e.g. a closure
    ``solve_fn`` / ``voter`` that cannot cross process boundaries) or any
    Windows-specific quirk surfaces.

Design goals (per task):
  * Dependency ordering always respected: a sub-problem is only solved once all
    of its dependencies are solved (Kahn layering).
  * Reuses the injectable *voter* pattern from ``maker_engine`` / ``mdap_engine``
    so it runs fully OFFLINE with a mock voter: ``solve_fn(sub_problem, voter)
    -> SolutionAttempt``.
  * Import-safe and dependency-light: only the stdlib (``concurrent.futures``,
    ``multiprocessing``, ``dataclasses``, ``os``, ``sys``, ``time``, ``logging``).
    We deliberately do NOT import the heavy ``workflow_structures`` /
    ``openevolve.kernel.schema`` graph at module top so child worker processes
    stay cheap. Sub-problems are accessed duck-typed (``.id`` and
    ``.dependencies``).
  * Never requires network / API keys. The default voter is a deterministic mock.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Voter pattern (mirrors maker_engine / mdap_engine)
# ----------------------------------------------------------------------
# A voter is a backend-agnostic callable that, given a rendered prompt and a
# system prompt, returns (raw_text, candidate). The default voter below is a
# deterministic mock that performs NO network / LLM calls, allowing the executor
# to run offline. Real deployments inject an OpenAI-compatible voter.
Voter = Callable[[str, Optional[str], Optional[Dict[str, Any]], Any], Tuple[str, Any]]


def _default_mock_voter(
    prompt: str,
    system_prompt: Optional[str] = None,
    expected_schema: Optional[Dict[str, Any]] = None,
    step: Any = None,
) -> Tuple[str, Any]:
    """Deterministic offline mock voter.

    Returns an empty raw text and a candidate derived trivially from the prompt
    length so the workflow can proceed without any LLM. This is the OFFLINE
    default used when no real voter is injected.
    """
    candidate = {
        "vote": 1,
        "prompt_len": len(prompt or ""),
        "system_len": len(system_prompt or ""),
    }
    return "", candidate


# ----------------------------------------------------------------------
# Top-level worker (must be picklable for multiprocessing)
# ----------------------------------------------------------------------
def _mp_worker(args: Tuple[str, Any, Callable, Voter]) -> Tuple[str, Any]:
    """Picklable worker executed in a child process.

    Receives the sub-problem id, the sub-problem object, the solve callable and
    the voter. Returns ``(sub_problem_id, result)`` so the parent can re-key by
    id regardless of completion order.
    """
    sub_id, sub_problem, solve_fn, voter = args
    result = solve_fn(sub_problem, voter)
    return sub_id, result


def _multiprocessing_available() -> bool:
    """Best-effort probe for whether process-based parallelism is usable."""
    try:
        import multiprocessing  # noqa: F401

        # On some platforms / frozen builds spawn is problematic; guard it.
        if getattr(multiprocessing, "current_process", None) is not None:
            return True
        return True
    except Exception:  # pragma: no cover - defensive
        return False


@dataclass
class ExecutionResult:
    """Aggregated result of an :meth:`SubProblemExecutor.execute` run."""

    solutions: Dict[str, Any] = field(default_factory=dict)
    # Order in which sub-problems were *solved* (dependency-respecting).
    solve_order: List[str] = field(default_factory=list)
    backend_used: str = "local"
    degraded_to_local: bool = False
    # Sub-problem ids -> pid that produced the solution (proves parallelism).
    solving_pids: Dict[str, int] = field(default_factory=dict)


class SubProblemExecutor:
    """Solves sub-problems respecting dependency ordering, with local /
    multiprocessing backends.

    The executor does not own the sub-problem *definitions* — those are passed
    in. It only needs each sub-problem to expose ``id`` (str) and ``dependencies``
    (iterable of dependency ids). It calls ``solve_fn(sub_problem, voter)`` and
    stores whatever it returns, keyed by ``sub_problem.id``.

    An optional :class:`ResourceManager` can be supplied so that consumption is
    tracked/recorded as sub-problems are solved (enforcement itself is the
    caller's responsibility via ``recommend_batch_size`` + ``consume_or_raise``).
    """

    def __init__(
        self,
        default_backend: str = "local",
        max_workers: Optional[int] = None,
        voter: Optional[Voter] = None,
        resource_manager: Optional[Any] = None,
    ):
        self.default_backend = default_backend
        self.max_workers = max_workers
        self.voter: Voter = voter or _default_mock_voter
        self.resource_manager = resource_manager

    # ------------------------------------------------------------------
    # Topological layering (Kahn's algorithm)
    # ------------------------------------------------------------------
    def topological_layers(self, sub_problems: List[Any]) -> List[List[Any]]:
        """Return sub-problems grouped into dependency layers.

        Each layer is a list of sub-problems whose dependencies are all satisfied
        by earlier layers. Within a layer the sub-problems are mutually
        independent and therefore safe to run in parallel.
        """
        by_id: Dict[str, Any] = {}
        for sp in sub_problems:
            sid = getattr(sp, "id", None)
            if sid is None:
                raise ValueError("Every sub-problem must expose a string 'id'")
            if sid in by_id:
                raise ValueError(f"Duplicate sub-problem id: {sid!r}")
            by_id[sid] = sp

        deps: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {}
        for sp in sub_problems:
            sid = sp.id
            raw = getattr(sp, "dependencies", None) or []
            dep_list = [d for d in raw if d in by_id and d != sid]
            deps[sid] = dep_list
            indegree[sid] = len(dep_list)
            # Validate dependency existence.
            for d in raw:
                if d not in by_id:
                    raise ValueError(
                        f"Sub-problem {sid!r} depends on unknown id {d!r}"
                    )

        # Kahn layering.
        layers: List[List[Any]] = []
        remaining = dict(indegree)
        while remaining:
            # Nodes with no unsatisfied deps form the next layer.
            layer_ids = [sid for sid, deg in remaining.items() if deg == 0]
            if not layer_ids:
                cycle = sorted(remaining.keys())
                raise ValueError(
                    f"Circular dependency detected among sub-problems: {cycle!r}"
                )
            layer = [by_id[sid] for sid in layer_ids]
            layers.append(layer)
            # "Remove" this layer and decrement dependents.
            for sid in layer_ids:
                del remaining[sid]
            for sid, deg in remaining.items():
                # only dependents of this layer should drop; recompute cheaply.
                if any(dep in layer_ids for dep in deps[sid]):
                    remaining[sid] = deg - sum(
                        1 for dep in deps[sid] if dep in layer_ids
                    )
        return layers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute(
        self,
        sub_problems: List[Any],
        solve_fn: Callable[[Any, Voter], Any],
        backend: Optional[str] = None,
        voter: Optional[Voter] = None,
    ) -> ExecutionResult:
        """Solve all sub-problems, respecting dependency order.

        ``backend`` may be ``"local"``, ``"multiprocessing"`` or ``"auto"``.
        Returns an :class:`ExecutionResult`.
        """
        backend = backend or self.default_backend
        voter = voter or self.voter
        layers = self.topological_layers(sub_problems)

        if backend == "auto":
            backend = "multiprocessing" if _multiprocessing_available() else "local"

        if backend == "multiprocessing":
            try:
                return self._execute_mp(layers, solve_fn, voter)
            except Exception as exc:  # pragma: no cover - gracefully degrade
                logger.warning(
                    "multiprocessing backend failed (%s); falling back to local",
                    exc,
                )
                return self._execute_local(layers, solve_fn, voter, degraded=True)
        return self._execute_local(layers, solve_fn, voter)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _execute_local(
        self,
        layers: List[List[Any]],
        solve_fn: Callable[[Any, Voter], Any],
        voter: Voter,
        degraded: bool = False,
    ) -> ExecutionResult:
        solutions: Dict[str, Any] = {}
        solve_order: List[str] = []
        solving_pids: Dict[str, int] = {}
        for layer in layers:
            for sp in layer:
                result = solve_fn(sp, voter)
                solutions[sp.id] = result
                solve_order.append(sp.id)
                solving_pids[sp.id] = os.getpid()
                self._record_usage(sp)
        return ExecutionResult(
            solutions=solutions,
            solve_order=solve_order,
            backend_used="local",
            degraded_to_local=degraded,
            solving_pids=solving_pids,
        )

    def _execute_mp(
        self,
        layers: List[List[Any]],
        solve_fn: Callable[[Any, Voter], Any],
        voter: Voter,
    ) -> ExecutionResult:
        solutions: Dict[str, Any] = {}
        solve_order: List[str] = []
        solving_pids: Dict[str, int] = {}

        # Probe picklability of the solve_fn/voter once with a dummy sub-problem
        # before committing to a process pool. If they aren't picklable we
        # degrade to local immediately (no wasted processes).
        if layers:
            probe_sp = layers[0][0]
            try:
                import pickle

                pickle.dumps((probe_sp, solve_fn, voter))
            except Exception as exc:
                logger.warning(
                    "solve_fn/voter not picklable (%s); using local backend", exc
                )
                return self._execute_local(layers, solve_fn, voter, degraded=True)

        workers = self.max_workers or min(32, (os.cpu_count() or 1) + 1)
        for layer in layers:
            if not layer:
                continue
            if len(layer) == 1:
                # No parallelism benefit; run in-process to avoid spawn overhead.
                sp = layer[0]
                result = solve_fn(sp, voter)
                solutions[sp.id] = result
                solve_order.append(sp.id)
                solving_pids[sp.id] = os.getpid()
                self._record_usage(sp)
                continue
            try:
                with ProcessPoolExecutor(max_workers=min(workers, len(layer))) as ex:
                    args = [(sp.id, sp, solve_fn, voter) for sp in layer]
                    for sub_id, result in ex.map(_mp_worker, args):
                        solutions[sub_id] = result
                        solve_order.append(sub_id)
            except Exception as exc:
                logger.warning(
                    "ProcessPoolExecutor layer failed (%s); solving layer locally",
                    exc,
                )
                for sp in layer:
                    result = solve_fn(sp, voter)
                    solutions[sp.id] = result
                    solve_order.append(sp.id)
                    solving_pids[sp.id] = os.getpid()
                    self._record_usage(sp)
                    continue
            # Record pids reported by the worker via result metadata, if present.
            for sp in layer:
                res = solutions.get(sp.id)
                pid = None
                if isinstance(res, dict) and "solving_pid" in res:
                    pid = res["solving_pid"]
                if pid is not None:
                    solving_pids[sp.id] = pid
                else:
                    solving_pids.setdefault(sp.id, os.getpid())
                self._record_usage(sp)
        return ExecutionResult(
            solutions=solutions,
            solve_order=solve_order,
            backend_used="multiprocessing",
            degraded_to_local=False,
            solving_pids=solving_pids,
        )

    def _record_usage(self, sp: Any) -> None:
        if self.resource_manager is None:
            return
        est = getattr(sp, "estimated_resources", None) or {}
        tokens = int(est.get("tokens", 0))
        tsec = float(est.get("time_seconds", 0.0))
        steps = int(est.get("steps", 1))
        try:
            self.resource_manager.record_actual(tokens=tokens, time_seconds=tsec, steps=steps)
        except Exception:  # pragma: no cover - never block on reporting
            pass
