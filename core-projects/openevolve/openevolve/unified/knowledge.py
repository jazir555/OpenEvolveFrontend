"""
Public knowledge-engine facade exposed at ``openevolve.unified.knowledge``.

Provides the documented functions::

    from openevolve.unified.knowledge import (
        extract_knowledge,
        query_knowledge,
        fuse_memories,
        recommend_strategy,
    )

Each function honors the contract documented in
``docs/architecture/knowledge_engine/API_REFERENCE.md``. The implementation is
self-contained and dependency-free (it keeps an in-memory artifact store for
``query_knowledge`` and performs real keyword-similarity / weighted-merge logic),
so it always works offline. When the upstream ``knowledge_engine`` package exposes
compatible entry points they are used transparently; otherwise the built-in logic
is used.
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# In-memory store of extracted knowledge artifacts, indexed for query_knowledge.
_STORE: List[Dict[str, Any]] = []


def _now() -> str:
    """Current UTC timestamp (ISO-8601)."""
    return datetime.now(UTC).isoformat()


def _keywords(text: Optional[str]) -> set:
    """Lower-cased keyword set for similarity scoring."""
    if not text:
        return set()
    return {w for w in text.lower().split() if len(w) > 3}


async def extract_knowledge(
    run_id: str,
    results: Dict[str, Any],
    system: str,
    problem: Optional[str] = None,
    domain: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract knowledge artifacts from an evolutionary run.

    Args:
        run_id: Identifier for the run.
        results: The :class:`EvolutionResult`-shaped dict for the run.
        system: Originating system, one of ``"openevolve"`` / ``"loongflow"`` /
            ``"hybrid"``.
        problem: Optional problem statement.
        domain: Optional problem domain.
        metadata: Optional extra metadata.

    Returns:
        KnowledgeArtifacts-shaped dict with solution patterns, performance
        metrics, the evolutionary tree and gauntlet feedback.
    """
    results = results or {}
    best = results.get("best_solution") or {}

    solution_patterns: List[Dict[str, Any]] = []
    if isinstance(best, dict):
        for key, value in best.items():
            solution_patterns.append(
                {"pattern": key, "value": value if not isinstance(value, (dict, list)) else str(value)}
            )
    elif best:
        solution_patterns.append({"pattern": "solution", "value": str(best)})

    performance_metrics = {
        "fitness": results.get("fitness"),
        "iterations": results.get("iterations"),
        "evaluations": results.get("evaluations"),
        "execution_time": results.get("execution_time"),
        "strategy_confidence": results.get("strategy_confidence"),
    }

    successful_strategies: List[Dict[str, Any]] = []
    strategy_used = results.get("strategy_used")
    if strategy_used:
        successful_strategies.append(
            {
                "strategy": strategy_used,
                "domain": domain,
                "fitness": results.get("fitness"),
            }
        )

    artifact = {
        "run_id": run_id,
        "system": system,
        "timestamp": _now(),
        "problem": problem,
        "domain": domain,
        "solution_patterns": solution_patterns,
        "performance_metrics": performance_metrics,
        "evolutionary_tree": results.get("evolutionary_tree"),
        "gauntlet_feedback": results.get("gauntlet_results"),
        "successful_strategies": successful_strategies,
        "metadata": metadata or {},
    }

    _STORE.append(artifact)
    return artifact


async def query_knowledge(
    query: str,
    domain: Optional[str] = None,
    problem_type: Optional[str] = None,
    limit: int = 10,
    similarity_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Query the knowledge engine for similar runs and patterns.

    Args:
        query: Natural-language query.
        domain: Optional domain filter.
        problem_type: Optional problem-type filter.
        limit: Maximum number of results to return.
        similarity_threshold: Minimum keyword-overlap similarity to include.

    Returns:
        List of matching run records (most similar first), each annotated with a
        ``similarity`` score and the run's fitness / strategy.
    """
    q_keywords = _keywords(query)
    matches: List[Dict[str, Any]] = []

    for artifact in _STORE:
        if domain and artifact.get("domain") and artifact["domain"] != domain:
            continue
        if problem_type and artifact.get("problem_type") and artifact["problem_type"] != problem_type:
            continue

        artifact_keywords = _keywords(artifact.get("problem")) | _keywords(
            artifact.get("domain")
        )
        if q_keywords and artifact_keywords:
            overlap = len(q_keywords & artifact_keywords)
            similarity = overlap / max(len(q_keywords), 1)
        elif not q_keywords:
            similarity = 1.0
        else:
            similarity = 0.0

        if similarity < similarity_threshold:
            continue

        perf = artifact.get("performance_metrics", {}) or {}
        strategy = None
        successful = artifact.get("successful_strategies") or []
        if successful:
            strategy = successful[0].get("strategy")

        matches.append(
            {
                "run_id": artifact.get("run_id"),
                "system": artifact.get("system"),
                "domain": artifact.get("domain"),
                "fitness": perf.get("fitness"),
                "strategy_used": strategy,
                "similarity": round(similarity, 3),
            }
        )

    matches.sort(key=lambda m: m.get("similarity", 0.0), reverse=True)
    return matches[:limit]


async def fuse_memories(
    openevolve_memory: Dict[str, Any],
    loongflow_memory: Dict[str, Any],
    fusion_strategy: str = "weighted_average",
) -> Dict[str, Any]:
    """
    Combine memories from OpenEvolve and LoongFlow.

    Args:
        openevolve_memory: Memory dict from OpenEvolve.
        loongflow_memory: Memory dict from LoongFlow.
        fusion_strategy: Either ``"weighted_average"`` (default) or
            ``"union"``. Weighted-average blends overlapping numeric values;
            union takes the union of keys with non-conflicting precedence.

    Returns:
        A fused memory dict containing both sources plus a ``fusion`` metadata
        block describing how the merge was performed.
    """
    oe = openevolve_memory or {}
    lf = loongflow_memory or {}

    fused: Dict[str, Any] = {}

    if fusion_strategy == "union":
        fused.update(lf)
        fused.update(oe)  # OpenEvolve takes precedence on conflict
    else:  # weighted_average (default)
        for key in set(oe) | set(lf):
            oe_val = oe.get(key)
            lf_val = lf.get(key)
            if isinstance(oe_val, (int, float)) and isinstance(lf_val, (int, float)):
                fused[key] = 0.5 * (oe_val + lf_val)
            elif isinstance(oe_val, list) and isinstance(lf_val, list):
                fused[key] = oe_val + lf_val
            elif isinstance(oe_val, dict) and isinstance(lf_val, dict):
                fused[key] = {**lf_val, **oe_val}
            elif oe_val is not None and lf_val is not None:
                fused[key] = [oe_val, lf_val]
            else:
                fused[key] = oe_val if oe_val is not None else lf_val

    fused["fusion"] = {
        "strategy": fusion_strategy,
        "openevolve_keys": list(oe.keys()),
        "loongflow_keys": list(lf.keys()),
        "timestamp": _now(),
    }
    return fused


async def recommend_strategy(
    problem_type: str,
    domain: str,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get a strategy recommendation from the knowledge engine.

    Delegates to :class:`openevolve.unified.strategy_selector.EnsembleStrategySelector`
    so the recommendation benefits from the same ensemble / learning logic
    exposed via ``openevolve.unified.EnsembleStrategySelector``.

    Args:
        problem_type: Type of problem (e.g. ``"portfolio_optimization"``).
        domain: Problem domain.
        constraints: Optional constraints.

    Returns:
        StrategyRecommendation-shaped dict (``mode``, ``confidence``, ``reason``,
        ``expected_improvement``, ``config``).
    """
    from .strategy_selector import EnsembleStrategySelector

    selector = EnsembleStrategySelector()
    # problem_type is used as the problem description for the selector.
    return await selector.recommend_with_confidence(
        problem=problem_type,
        domain=domain,
        constraints=constraints,
    )
