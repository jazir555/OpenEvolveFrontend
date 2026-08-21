"""
strategies.py - Concrete decomposition strategies for the problem-decomposition engine.

This module implements real decomposition logic. It builds :class:`SubProblem`
instances (imported from the shared ``subproblem`` module, owned by another
agent) organized into a dependency graph, and exposes:
  * multiple decomposition strategies (hierarchical, semantic, flow, dependency),
  * adaptive strategy selection based on a problem analysis,
  * a serializable :class:`DecompositionPlan` (to_dict / from_dict),
  * a topologically-sorted execution order.

The shared symbols are imported defensively so the module compiles and imports
cleanly even before the parallel agent wave has created them; when they exist
they are used directly.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared-symbol imports (provided by parallel agents; do not redefine here).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on parallel agent output
    from subproblem import SubProblem
except ImportError:  # pragma: no cover
    SubProblem = None  # type: ignore

try:  # pragma: no cover
    from math_domain import MathematicalDomain, EvaluationMetric
except ImportError:  # pragma: no cover
    MathematicalDomain = None  # type: ignore
    EvaluationMetric = None  # type: ignore

try:  # pragma: no cover
    from decomposition_engine import DecompositionEngine
except ImportError:  # pragma: no cover
    DecompositionEngine = None  # type: ignore


# --------------------------------------------------------------------------- #
# Enums / helpers
# --------------------------------------------------------------------------- #
class StrategyKind(str, Enum):
    HIERARCHICAL = "hierarchical"
    SEMANTIC = "semantic"
    FLOW = "flow"
    DEPENDENCY = "dependency"
    ADAPTIVE = "adaptive"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def normalize_problem(problem: Any) -> Dict[str, Any]:
    """Coerce a problem (dict or object) into a uniform dict view."""
    if isinstance(problem, dict):
        get = problem.get
    else:
        def get(key, default=None):
            return getattr(problem, key, default)
    return {
        "id": get("id") or get("problem_id") or "",
        "title": get("title") or "",
        "description": get("description") or "",
        "domain": get("domain") or "general",
        "constraints": _as_list(get("constraints")),
        "requirements": _as_list(get("requirements")),
        "success_criteria": _as_list(get("success_criteria")) or _as_list(get("objectives")),
    }


def topo_order(nodes: List[str], deps: Dict[str, List[str]]) -> List[str]:
    """Return ``nodes`` topologically ordered so dependencies precede dependents.

    ``deps[node]`` lists the node ids that ``node`` depends on (must run first).
    Cycles are broken by emitting remaining nodes in insertion order.
    """
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    adj: Dict[str, List[str]] = defaultdict(list)
    present = set(nodes)
    for node, parents in deps.items():
        for parent in _as_list(parents):
            if parent in present and node in present and parent != node:
                adj[parent].append(node)
                indeg[node] += 1
    queue = deque(sorted((n for n in nodes if indeg[n] == 0), key=lambda x: nodes.index(x)))
    ordered: List[str] = []
    while queue:
        n = queue.popleft()
        ordered.append(n)
        for child in adj[n]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if len(ordered) != len(nodes):
        # Break cycles: append any not yet emitted.
        seen = set(ordered)
        ordered.extend(n for n in nodes if n not in seen)
    return ordered


# --------------------------------------------------------------------------- #
# Tolerant SubProblem / domain / metric construction
# --------------------------------------------------------------------------- #
def make_subproblem(
    sub_id: str,
    title: str,
    description: str,
    parent_id: Optional[str] = None,
    order: int = 0,
    strategy: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
    domain: Optional[str] = None,
    metric: Optional[str] = None,
    **extra: Any,
) -> Any:
    """Construct a shared :class:`SubProblem` tolerantly.

    Different ``subproblem`` implementations expose different field names; this
    helper constructs with no required args (dataclasses default) and then sets
    whatever attributes the concrete class accepts, stashing everything else in
    ``metadata``.
    """
    if SubProblem is None:  # pragma: no cover - shared symbol not present yet
        raise RuntimeError("SubProblem is not available; import the shared subproblem module")

    try:
        sp = SubProblem()
    except Exception:
        try:
            sp = SubProblem(id=sub_id, title=title, description=description)
        except Exception:
            sp = SubProblem()

    core = {
        "id": sub_id,
        "sub_problem_id": sub_id,
        "title": title,
        "description": description,
        "parent_id": parent_id,
        "strategy": strategy,
        "order": order,
    }
    for key, val in core.items():
        if val is None:
            continue
        try:
            setattr(sp, key, val)
        except Exception:
            pass

    # Attach domain / evaluation metric objects when available.
    meta: Dict[str, Any] = {}
    try:
        meta = getattr(sp, "metadata")
        if meta is None:
            meta = {}
            setattr(sp, "metadata", meta)
    except Exception:
        meta = {}

    if domain is not None:
        meta["domain"] = domain
        dom_obj = _build_domain(domain, description)
        if dom_obj is not None:
            meta["domain_obj"] = dom_obj
    if metric is not None:
        meta["metric"] = metric
        m_obj = _build_metric(metric)
        if m_obj is not None:
            meta["metric_obj"] = m_obj

    if dependencies:
        if hasattr(sp, "dependencies") and isinstance(getattr(sp, "dependencies"), list):
            try:
                setattr(sp, "dependencies", list(dependencies))
            except Exception:
                meta["dependencies"] = list(dependencies)
        else:
            meta["dependencies"] = list(dependencies)

    if extra:
        meta.update(extra)
    return sp


def _build_domain(name: str, description: str) -> Any:
    if MathematicalDomain is None:
        return None
    for attempt in (
        lambda: MathematicalDomain(name=name, description=description),
        lambda: MathematicalDomain(name=name),
        lambda: MathematicalDomain(),
    ):
        try:
            return attempt()
        except Exception:
            continue
    return None


def _build_metric(name: str) -> Any:
    if EvaluationMetric is None:
        return None
    for attempt in (
        lambda: EvaluationMetric(name=name, target=0.9),
        lambda: EvaluationMetric(name=name),
        lambda: EvaluationMetric(),
    ):
        try:
            return attempt()
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# Decomposition plan
# --------------------------------------------------------------------------- #
@dataclass
class DecompositionPlan:
    problem_id: str = ""
    strategy: str = "hierarchical"
    sub_problems: List[Any] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _id_of(self, sp: Any) -> str:
        return getattr(sp, "id", None) or getattr(sp, "sub_problem_id", "") or ""

    def execution_order(self) -> List[str]:
        ids = [self._id_of(sp) for sp in self.sub_problems]
        deps = {i: self.dependencies.get(i, []) for i in ids}
        return topo_order(ids, deps)

    def to_dict(self) -> Dict[str, Any]:
        serialized = []
        for sp in self.sub_problems:
            if hasattr(sp, "to_dict"):
                serialized.append(sp.to_dict())
            else:
                serialized.append({
                    "id": getattr(sp, "id", None) or getattr(sp, "sub_problem_id", ""),
                    "title": getattr(sp, "title", ""),
                    "description": getattr(sp, "description", ""),
                    "parent_id": getattr(sp, "parent_id", None),
                    "strategy": getattr(sp, "strategy", None),
                    "metadata": getattr(sp, "metadata", {}),
                })
        return {
            "problem_id": self.problem_id,
            "strategy": self.strategy,
            "sub_problems": serialized,
            "dependencies": {k: list(v) for k, v in self.dependencies.items()},
            "execution_order": self.execution_order(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionPlan":
        plan = cls(
            problem_id=data.get("problem_id", ""),
            strategy=data.get("strategy", "hierarchical"),
            dependencies={k: list(v) for k, v in data.get("dependencies", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )
        for spd in data.get("sub_problems", []):
            plan.sub_problems.append(make_subproblem(
                sub_id=spd.get("id") or spd.get("sub_problem_id", ""),
                title=spd.get("title", ""),
                description=spd.get("description", ""),
                parent_id=spd.get("parent_id"),
                strategy=spd.get("strategy"),
                **(spd.get("metadata", {}) or {}),
            ))
        return plan


# --------------------------------------------------------------------------- #
# Strategy base + concrete strategies
# --------------------------------------------------------------------------- #
class DecompositionStrategy:
    """Base class for a decomposition strategy."""

    kind: StrategyKind = StrategyKind.HIERARCHICAL

    def decompose(self, problem: Any) -> DecompositionPlan:
        raise NotImplementedError


class HierarchicalStrategy(DecompositionStrategy):
    kind = StrategyKind.HIERARCHICAL

    def decompose(self, problem: Any) -> DecompositionPlan:
        p = normalize_problem(problem)
        subs: List[Any] = []
        deps: Dict[str, List[str]] = {}

        root_id = "sp_root"
        subs.append(make_subproblem(
            root_id, "Analyze problem",
            f"Understand the problem '{p['title']}', its domain ('{p['domain']}') "
            f"and success criteria before producing solution components.",
            order=0, strategy="analysis", domain=p["domain"], metric="coverage",
        ))

        criteria = p["success_criteria"] or [{"description": c} if isinstance(c, str) else c
                                             for c in p["requirements"]]
        prev_ids: List[str] = [root_id]
        for idx, criterion in enumerate(criteria, start=1):
            if isinstance(criterion, dict):
                desc = criterion.get("description") or criterion.get("name") or "objective"
                thr = criterion.get("threshold")
            else:
                desc = str(criterion)
                thr = None
            sid = f"sp_obj_{idx}"
            subs.append(make_subproblem(
                sid, f"Objective {idx}: {desc}",
                f"Satisfy the success criterion: {desc}"
                + (f" (threshold={thr})" if thr is not None else ""),
                parent_id=root_id, order=idx, strategy="implementation",
                domain=p["domain"], metric="correctness",
            ))
            deps[sid] = [root_id]
            prev_ids.append(sid)

        if len(subs) > 2:
            synth_id = "sp_synthesize"
            subs.append(make_subproblem(
                synth_id, "Synthesize and validate integrated solution",
                "Integrate objective sub-problems, verify against constraints and "
                "success criteria, and resolve conflicts.",
                parent_id=root_id, order=len(subs), strategy="synthesis",
                domain=p["domain"], metric="integration",
            ))
            deps[synth_id] = [s for s in prev_ids if s != synth_id]

        return DecompositionPlan(
            problem_id=p["id"], strategy=self.kind.value,
            sub_problems=subs, dependencies=deps,
            metadata={"strategy": self.kind.value, "criteria_count": len(criteria)},
        )


class SemanticStrategy(DecompositionStrategy):
    kind = StrategyKind.SEMANTIC

    def decompose(self, problem: Any) -> DecompositionPlan:
        import re
        p = normalize_problem(problem)
        text = f"{p['title']} {p['description']} {' '.join(p['requirements'])}"
        words = re.findall(r"[A-Za-z][A-Za-z_]+", text.lower())
        freq: Dict[str, int] = defaultdict(int)
        for w in words:
            if len(w) > 3:
                freq[w] += 1
        concepts = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:8]
        if not concepts:
            concepts = [("general", 1)]

        subs: List[Any] = []
        deps: Dict[str, List[str]] = {}
        prev = None
        for idx, (concept, _) in enumerate(concepts):
            sid = f"sp_sem_{concept}"
            subs.append(make_subproblem(
                sid, f"Semantic unit: {concept}",
                f"Address the semantic cluster '{concept}' within the {p['domain']} domain.",
                order=idx, strategy="semantic", domain=p["domain"], metric="relevance",
            ))
            if prev is not None:
                deps[sid] = [prev]
            prev = sid

        return DecompositionPlan(
            problem_id=p["id"], strategy=self.kind.value,
            sub_problems=subs, dependencies=deps,
            metadata={"strategy": self.kind.value, "concepts": [c for c, _ in concepts]},
        )


class FlowStrategy(DecompositionStrategy):
    kind = StrategyKind.FLOW

    def decompose(self, problem: Any) -> DecompositionPlan:
        p = normalize_problem(problem)
        stages = [
            ("Ingest & scope", "Collect requirements, constraints and context."),
            ("Transform", "Produce the core solution transformation."),
            ("Evaluate", "Validate against success criteria and constraints."),
            ("Deliver", "Package and hand off the verified solution."),
        ]
        subs: List[Any] = []
        deps: Dict[str, List[str]] = {}
        prev = None
        for idx, (title, desc) in enumerate(stages):
            sid = f"sp_flow_{idx}"
            subs.append(make_subproblem(
                sid, title, desc, order=idx, strategy="flow",
                domain=p["domain"], metric="throughput" if idx == 1 else "coverage",
            ))
            if prev is not None:
                deps[sid] = [prev]
            prev = sid

        return DecompositionPlan(
            problem_id=p["id"], strategy=self.kind.value,
            sub_problems=subs, dependencies=deps,
            metadata={"strategy": self.kind.value},
        )


class DependencyStrategy(DecompositionStrategy):
    kind = StrategyKind.DEPENDENCY

    def decompose(self, problem: Any) -> DecompositionPlan:
        p = normalize_problem(problem)
        subs: List[Any] = []
        deps: Dict[str, List[str]] = {}

        # One sub-problem per explicit constraint (must hold), then per requirement.
        for idx, constraint in enumerate(p["constraints"], start=1):
            cid = f"sp_con_{idx}"
            subs.append(make_subproblem(
                cid, f"Constraint {idx}", f"Honor constraint: {constraint}",
                order=idx, strategy="constraint", domain=p["domain"], metric="feasibility",
            ))
        base = len(subs)
        for idx, req in enumerate(p["requirements"], start=1):
            rid = f"sp_req_{idx}"
            subs.append(make_subproblem(
                rid, f"Requirement {idx}", f"Implement requirement: {req}",
                order=base + idx, strategy="requirement", domain=p["domain"], metric="correctness",
            ))
            # Requirements depend on all constraints being satisfied first.
            if base:
                deps[rid] = [f"sp_con_{j}" for j in range(1, base + 1)]

        if not subs:
            subs.append(make_subproblem(
                "sp_core", "Core problem", p["description"] or "Solve the problem.",
                order=0, strategy="core", domain=p["domain"], metric="coverage",
            ))

        return DecompositionPlan(
            problem_id=p["id"], strategy=self.kind.value,
            sub_problems=subs, dependencies=deps,
            metadata={"strategy": self.kind.value},
        )


_STRATEGY_REGISTRY: Dict[str, type] = {
    StrategyKind.HIERARCHICAL.value: HierarchicalStrategy,
    StrategyKind.SEMANTIC.value: SemanticStrategy,
    StrategyKind.FLOW.value: FlowStrategy,
    StrategyKind.DEPENDENCY.value: DependencyStrategy,
}


def select_strategy(problem: Any) -> StrategyKind:
    """Pick a strategy from problem characteristics (no external services)."""
    p = normalize_problem(problem)
    n_constraints = len(p["constraints"])
    n_obj = len(p["success_criteria"])
    desc_len = len(p["description"] or "")
    if n_constraints >= 3 or n_obj >= 4:
        return StrategyKind.DEPENDENCY if n_constraints >= 3 else StrategyKind.HIERARCHICAL
    if desc_len > 800 or len(p["requirements"]) >= 4:
        return StrategyKind.HIERARCHICAL
    if desc_len < 200:
        return StrategyKind.SEMANTIC
    return StrategyKind.FLOW


def decompose(problem: Any, strategy: Optional[str] = None) -> DecompositionPlan:
    """Decompose ``problem`` into a :class:`DecompositionPlan`."""
    kind = StrategyKind(strategy) if strategy else select_strategy(problem)
    if kind == StrategyKind.ADAPTIVE:
        kind = select_strategy(problem)
    strategy_cls = _STRATEGY_REGISTRY.get(kind.value, HierarchicalStrategy)
    return strategy_cls().decompose(problem)


# If the shared DecompositionEngine exists, expose a thin adapter so it can also
# be used to produce plans (kept defensive to avoid clashing with the other agent).
def decompose_with_engine(problem: Any, strategy: Optional[str] = None) -> DecompositionPlan:
    """Use the shared :class:`DecompositionEngine` if available, else fall back."""
    if DecompositionEngine is not None:
        try:
            engine = DecompositionEngine()
            engine_plan = engine.decompose(problem, strategy or "hierarchical")
            # Bridge the engine's plan into our serializable plan.
            plan = DecompositionPlan(
                problem_id=getattr(engine_plan, "problem_id", ""),
                strategy=getattr(engine_plan, "strategy", "hierarchical"),
                metadata=dict(getattr(engine_plan, "metadata", {}) or {}),
            )
            deps: Dict[str, List[str]] = {}
            for sp in getattr(engine_plan, "sub_problems", []):
                plan.sub_problems.append(sp)
                sid = getattr(sp, "id", None) or getattr(sp, "sub_problem_id", "")
                pid = getattr(sp, "parent_id", None)
                if sid and pid:
                    deps.setdefault(sid, []).append(pid)
            plan.dependencies = deps
            return plan
        except Exception as exc:  # pragma: no cover
            logger.warning("DecompositionEngine bridge failed, using local strategies: %s", exc)
    return decompose(problem, strategy)


__all__ = [
    "StrategyKind",
    "DecompositionPlan",
    "DecompositionStrategy",
    "HierarchicalStrategy",
    "SemanticStrategy",
    "FlowStrategy",
    "DependencyStrategy",
    "select_strategy",
    "decompose",
    "decompose_with_engine",
    "topo_order",
    "make_subproblem",
    "normalize_problem",
]
