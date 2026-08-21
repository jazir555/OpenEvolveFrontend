"""
workflow_extraction.py - Extract an executable workflow from a decomposition plan.

Turns a :class:`DecompositionPlan` (sub-problems + dependency graph) into a
runnable workflow specification:

  * topological execution order,
  * parallelizable batches (sub-problems that can run concurrently),
  * explicit dependency edges,
  * a serializable JSON spec (``to_spec``) and a linear step list.

Uses the shared ``SubProblem`` symbol (imported, never redefined).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from subproblem import SubProblem
except ImportError:  # pragma: no cover
    SubProblem = None  # type: ignore

try:  # pragma: no cover
    from strategies import DecompositionPlan, topo_order  # type: ignore
except ImportError:  # pragma: no cover
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _spec = _ilu.spec_from_file_location(
        "_decomposition_strategies",
        _os.path.join(_os.path.dirname(__file__), "strategies.py"),
    )
    _strat = _ilu.module_from_spec(_spec)
    _sys.modules["_decomposition_strategies"] = _strat
    _spec.loader.exec_module(_strat)
    DecompositionPlan = _strat.DecompositionPlan
    topo_order = _strat.topo_order


def _id_of(sp: Any) -> str:
    return getattr(sp, "id", None) or getattr(sp, "sub_problem_id", "") or ""


def _title_of(sp: Any) -> str:
    return getattr(sp, "title", "") or _id_of(sp)


@dataclass
class WorkflowNode:
    id: str
    title: str
    depends_on: List[str] = field(default_factory=list)
    strategy: Optional[str] = None
    domain: Optional[str] = None
    metric: Optional[str] = None


@dataclass
class Workflow:
    problem_id: str = ""
    strategy: str = ""
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    parallel_batches: List[List[str]] = field(default_factory=list)

    def to_spec(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "strategy": self.strategy,
            "nodes": [n.__dict__ for n in self.nodes],
            "edges": {k: list(v) for k, v in self.edges.items()},
            "execution_order": list(self.execution_order),
            "parallel_batches": [list(b) for b in self.parallel_batches],
        }

    def linear_steps(self) -> List[str]:
        steps = []
        for idx, nid in enumerate(self.execution_order, start=1):
            node = next((n for n in self.nodes if n.id == nid), None)
            title = node.title if node else nid
            deps = ", ".join(node.depends_on) if node and node.depends_on else "none"
            steps.append(f"{idx}. [{nid}] {title} (depends on: {deps})")
        return steps


def extract_workflow(plan: DecompositionPlan) -> Workflow:
    """Build a :class:`Workflow` from a decomposition plan."""
    nodes: List[WorkflowNode] = []
    edges: Dict[str, List[str]] = {}
    id_to_node: Dict[str, WorkflowNode] = {}

    for sp in plan.sub_problems:
        sid = _id_of(sp)
        meta = getattr(sp, "metadata", None) or {}
        deps = list(plan.dependencies.get(sid, []))
        node = WorkflowNode(
            id=sid,
            title=_title_of(sp),
            depends_on=deps,
            strategy=getattr(sp, "strategy", None),
            domain=meta.get("domain"),
            metric=meta.get("metric"),
        )
        nodes.append(node)
        id_to_node[sid] = node
        # Edge: dependency -> this node.
        for dep in deps:
            edges.setdefault(dep, []).append(sid)

    order = plan.execution_order()
    batches = _parallel_batches(nodes, edges)

    return Workflow(
        problem_id=plan.problem_id,
        strategy=plan.strategy,
        nodes=nodes,
        edges=edges,
        execution_order=order,
        parallel_batches=batches,
    )


def _parallel_batches(nodes: List[WorkflowNode], edges: Dict[str, List[str]]) -> List[List[str]]:
    """Group nodes into levels; nodes on the same level share no intra-level deps."""
    # Map each node to its set of direct dependencies.
    deps_of = {n.id: set(n.depends_on) for n in nodes}
    # Compute depth = longest dependency chain length.
    depth_cache: Dict[str, int] = {}

    def depth(nid: str, _seen: Optional[set] = None) -> int:
        if nid in depth_cache:
            return depth_cache[nid]
        _seen = _seen or set()
        if nid in _seen:
            return 0
        _seen.add(nid)
        parents = deps_of.get(nid, set())
        d = 0 if not parents else 1 + max((depth(p, set(_seen)) for p in parents), default=0)
        depth_cache[nid] = d
        return d

    levels: Dict[int, List[str]] = {}
    for n in nodes:
        levels.setdefault(depth(n.id), []).append(n.id)
    return [levels[k] for k in sorted(levels)]


__all__ = [
    "WorkflowNode",
    "Workflow",
    "extract_workflow",
]
