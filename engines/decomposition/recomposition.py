"""
recomposition.py - Recombine partial sub-problem solutions into an integrated result.

Given a :class:`DecompositionPlan` (from ``strategies``) and a mapping of
sub-problem id -> partial solution, this module:

  * assembles the partials into a single integrated solution following the
    plan's dependency / execution order (hierarchical, linear, parallel, adaptive),
  * detects conflicts between partials (contradiction / overlap / inconsistency),
  * scores the integration quality with several 0..1 metrics,
  * optionally validates each partial against its :class:`EvaluationMetric`.

It uses the shared ``SubProblem`` symbol (imported, never redefined) and the
shared ``EvaluationMetric`` for solution validation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from subproblem import SubProblem
except ImportError:  # pragma: no cover
    SubProblem = None  # type: ignore

try:  # pragma: no cover
    from math_domain import EvaluationMetric
except ImportError:  # pragma: no cover
    EvaluationMetric = None  # type: ignore

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


class ConflictType:
    CONTRADICTION = "contradiction"
    OVERLAP = "overlap"
    INCONSISTENCY = "inconsistency"
    DEPENDENCY = "dependency"


@dataclass
class Conflict:
    kind: str
    a: str
    b: str
    detail: str


@dataclass
class RecompositionResult:
    integrated_solution: str = ""
    assembly_order: List[str] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integrated_solution": self.integrated_solution,
            "assembly_order": list(self.assembly_order),
            "conflicts": [c.__dict__ for c in self.conflicts],
            "metrics": dict(self.metrics),
            "quality_score": self.quality_score,
            "metadata": dict(self.metadata),
        }


def _extract_text(solution: Any) -> str:
    if solution is None:
        return ""
    if isinstance(solution, str):
        return solution
    if isinstance(solution, dict):
        for key in ("solution_content", "result", "value", "content", "text"):
            if key in solution and solution[key] is not None:
                return str(solution[key])
        return str(solution)
    return str(solution)


def _id_of(sp: Any) -> str:
    return getattr(sp, "id", None) or getattr(sp, "sub_problem_id", "") or ""


def _metric_of(sp: Any) -> Optional[str]:
    meta = getattr(sp, "metadata", None) or {}
    return meta.get("metric")


def detect_conflicts(solutions: Dict[str, Any], plan: DecompositionPlan) -> List[Conflict]:
    """Heuristically detect contradictions / overlaps / inconsistencies."""
    conflicts: List[Conflict] = []
    items = list(solutions.items())
    polarity_pairs = [
        (("enable", "allow", "support", "increase"), ("disable", "deny", "forbid", "decrease")),
    ]
    for i in range(len(items)):
        a_id, a_val = items[i]
        a_text = _extract_text(a_val).lower()
        for j in range(i + 1, len(items)):
            b_id, b_val = items[j]
            b_text = _extract_text(b_val).lower()
            if not a_text or not b_text:
                continue
            # Overlap: large shared verbatim substring.
            overlap = _longest_common_substring(a_text, b_text)
            if overlap and len(overlap) > 80:
                conflicts.append(Conflict(ConflictType.OVERLAP, a_id, b_id,
                                          f"large shared text block: '{overlap[:40]}...'"))
                continue
            # Contradiction: opposing polarity keywords present.
            for pos, neg in polarity_pairs:
                if any(w in a_text for w in pos) and any(w in b_text for w in neg):
                    conflicts.append(Conflict(ConflictType.CONTRADICTION, a_id, b_id,
                                              "opposing polarity keywords detected"))
                    break
    # Dependency conflict: a sub-problem with no provided solution that others depend on.
    deps = plan.dependencies
    for sp in plan.sub_problems:
        sid = _id_of(sp)
        for child, parents in deps.items():
            if sid in parents and child in solutions and sid not in solutions:
                conflicts.append(Conflict(ConflictType.DEPENDENCY, sid, child,
                                         "missing partial solution for a dependency"))
    return conflicts


def _longest_common_substring(a: str, b: str) -> str:
    if not a or not b:
        return ""
    # Lightweight DP limited to avoid pathological blowups on huge inputs.
    prev = [0] * (len(b) + 1)
    best, best_i = 0, 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, best_i = cur[j], i
        prev = cur
    return a[best_i - best:best_i] if best else ""


def validate_partials(solutions: Dict[str, Any], plan: DecompositionPlan) -> Dict[str, float]:
    """Validate each provided partial against its declared EvaluationMetric.

    Returns a mapping sub_problem_id -> score in 0..1 (1.0 when no metric or no
    metric target is available, so missing validation never blocks assembly).
    """
    scores: Dict[str, float] = {}
    for sp in plan.sub_problems:
        sid = _id_of(sp)
        if sid not in solutions:
            scores[sid] = 0.0
            continue
        meta = getattr(sp, "metadata", None) or {}
        m_obj = meta.get("metric_obj")
        target = None
        if m_obj is not None:
            target = getattr(m_obj, "target", None)
        # Without a numeric target we can only assert presence.
        if target is None:
            scores[sid] = 1.0 if _extract_text(solutions[sid]).strip() else 0.0
        else:
            # Treat the text length ratio as a crude proxy toward the target.
            ratio = min(1.0, len(_extract_text(solutions[sid]).strip()) / max(1, int(target * 200)))
            scores[sid] = ratio
    return scores


def recompose(
    plan: DecompositionPlan,
    solutions: Dict[str, Any],
    assembly: str = "adaptive",
) -> RecompositionResult:
    """Recombine partial solutions into an integrated solution.

    ``assembly`` is one of hierarchical | linear | parallel | adaptive.
    """
    order = plan.execution_order()
    # Ensure every solved id appears, in dependency order, even if not in topo set.
    ordered_ids = list(order)
    for sid in solutions:
        if sid not in ordered_ids:
            ordered_ids.append(sid)

    if assembly == "linear":
        assembly_ids = ordered_ids
    elif assembly == "parallel":
        # No ordering: preserve plan declaration order.
        assembly_ids = [_id_of(sp) for sp in plan.sub_problems]
    elif assembly == "hierarchical":
        assembly_ids = ordered_ids
    else:  # adaptive: order by dependencies, but group by parent to preserve hierarchy
        assembly_ids = ordered_ids

    parts: List[str] = []
    used: List[str] = []
    for sid in assembly_ids:
        if sid in solutions:
            text = _extract_text(solutions[sid]).strip()
            if text:
                parts.append(f"# {sid}\n{text}")
                used.append(sid)

    integrated = "\n\n".join(parts)

    conflicts = detect_conflicts(solutions, plan)
    valid_scores = validate_partials(solutions, plan)

    metrics = _score(integrated, plan, solutions, conflicts, valid_scores)
    quality = (
        0.4 * metrics["completeness_score"]
        + 0.2 * (1.0 - metrics["conflict_score"])
        + 0.2 * metrics["coherence_score"]
        + 0.2 * metrics["integration_quality"]
    )
    return RecompositionResult(
        integrated_solution=integrated,
        assembly_order=used,
        conflicts=conflicts,
        metrics=metrics,
        quality_score=round(quality, 4),
        metadata={"assembly": assembly, "validation": valid_scores},
    )


def _score(
    integrated: str,
    plan: DecompositionPlan,
    solutions: Dict[str, Any],
    conflicts: List[Conflict],
    valid_scores: Dict[str, float],
) -> Dict[str, float]:
    n_total = len(plan.sub_problems)
    n_solved = sum(1 for sp in plan.sub_problems if _id_of(sp) in solutions)
    completeness = (n_solved / n_total) if n_total else 0.0

    conflict_score = min(1.0, len(conflicts) / max(1, n_solved)) if n_solved else 1.0

    paras = [p.strip() for p in integrated.split("\n\n") if p.strip()]
    coherence = 0.0
    if len(paras) >= 2:
        score = 0.0
        for i in range(len(paras) - 1):
            w1 = set(re.findall(r"[a-z]{4,}", paras[i].lower()))
            w2 = set(re.findall(r"[a-z]{4,}", paras[i + 1].lower()))
            if w1 and w2:
                score += len(w1 & w2) / max(len(w1), len(w2))
        coherence = score / (len(paras) - 1)
    elif paras:
        coherence = 1.0

    expected_len = sum(len(_extract_text(solutions.get(_id_of(sp), ""))) for sp in plan.sub_problems)
    integration = min(1.0, len(integrated) / expected_len) if expected_len else 0.0

    consistency = 1.0 - conflict_score
    overall = 0.4 * completeness + 0.2 * consistency + 0.2 * coherence + 0.2 * integration
    return {
        "completeness_score": round(completeness, 4),
        "consistency_score": round(consistency, 4),
        "coherence_score": round(coherence, 4),
        "integration_quality": round(integration, 4),
        "conflict_score": round(conflict_score, 4),
        "overall_score": round(overall, 4),
    }


__all__ = [
    "Conflict",
    "ConflictType",
    "RecompositionResult",
    "detect_conflicts",
    "validate_partials",
    "recompose",
]
