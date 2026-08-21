"""
Self-Play <-> Knowledge Engine Integration Bridge

Implements the three adapters described in
``docs/architecture/selfplay_knowledgebase_integration_spec.md`` so the PSV
self-play system (``engines/other/psv_selfplay.py``) can leverage the
Knowledge Engine (``knowledge_engine/engine.py`` - ``KnowledgeEngine``).

The bridge is deliberately dependency-light:

* It never imports the heavy ``knowledge_engine`` package at module load time.
* It talks to the engine through a *duck-typed* retrieval protocol, trying the
  real public query methods in priority order
  (``retrieve_knowledge`` -> ``retrieve`` -> ``query`` -> ``search`` ->
  ``query_index_by_keyword``).
* Every adapter degrades gracefully: if the engine is ``None`` or exposes no
  compatible query API, the adapter still returns a complete, sensible
  structure (``available=False``) instead of crashing or returning ``None``.

Only the three spec adapters are exposed here. They are real, importable and
runnable (no network / no LLM required when passed a lightweight engine).
"""

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional

from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Best-effort import of the PSV dataclasses for isinstance checks.
# Failure here is non-fatal - the adapters accept duck-typed inputs anyway.
# ---------------------------------------------------------------------------
try:
    from psv_selfplay import (
        MathematicalProblem,
        SolutionAttempt,
        VerificationResult,
    )
except Exception:
    try:  # pragma: no cover - only when imported as a package
        from engines.other.psv_selfplay import (
            MathematicalProblem,
            SolutionAttempt,
            VerificationResult,
        )
    except Exception:  # pragma: no cover - fall back to duck typing
        MathematicalProblem = None
        SolutionAttempt = None
        VerificationResult = None


# Real query methods tried, in priority order.
_RETRIEVE_METHODS: tuple = (
    "retrieve_knowledge",
    "retrieve",
    "query",
    "search",
    "query_index_by_keyword",
)


# ===========================================================================
# Low-level helpers
# ===========================================================================

def _call_maybe_async(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``method``, running it to completion if it returns a coroutine."""
    result = method(*args, **kwargs)
    if inspect.isawaitable(result):
        try:
            return asyncio.run(result)
        except RuntimeError:
            # Already inside an event loop - fall back to a fresh loop in a
            # dedicated thread so we never block the caller's loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(result)).result()
    return result


def _result_to_text(item: Any) -> str:
    """Coerce a single retrieved item into a human-readable string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "content", "summary", "statement", "answer", "value", "chunk"):
            if item.get(key) is not None:
                return str(item[key])
        return str(item)
    return str(item)


def _coerce_results(raw: Any) -> List[Any]:
    """Normalise whatever an engine returns into a list of result items."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("results", "hits", "matches", "items", "documents", "passages"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        return [raw]
    if isinstance(raw, list):
        return raw
    return [raw]


def _retrieve_knowledge(
    knowledge_engine: Any,
    query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve knowledge from the engine using its real query API.

    Degrades gracefully: returns ``available=False`` when the engine is missing
    or has no compatible retrieval method.
    """
    if knowledge_engine is None:
        return {
            "available": False,
            "engine_type": None,
            "via": None,
            "query": query,
            "results": [],
            "error": "no knowledge engine provided",
        }

    engine_type = type(knowledge_engine).__name__

    for name in _RETRIEVE_METHODS:
        method = getattr(knowledge_engine, name, None)
        if not callable(method):
            continue

        try:
            if name == "query_index_by_keyword":
                index = getattr(knowledge_engine, "last_loaded_index", None) or {}
                raw = _call_maybe_async(method, index, query)
            else:
                try:
                    raw = _call_maybe_async(method, query)
                except TypeError:
                    raw = _call_maybe_async(method, query, limit=limit)
        except Exception as exc:  # engine method blew up - try the next one
            logger.warning("Knowledge engine retrieval via %s failed: %s", name, exc)
            continue

        results = _coerce_results(raw)
        return {
            "available": True,
            "engine_type": engine_type,
            "via": name,
            "query": query,
            "results": results,
            "error": None,
        }

    return {
        "available": False,
        "engine_type": engine_type,
        "via": None,
        "query": query,
        "results": [],
        "error": "engine exposes no compatible query/retrieve API",
    }


def _problem_fields(problem: Any):
    """Extract (id, domain, statement) from a duck-typed problem object."""
    if problem is None:
        return None, None, None
    if is_dataclass(problem):
        data = asdict(problem)
    elif isinstance(problem, dict):
        data = problem
    elif hasattr(problem, "__dict__"):
        data = vars(problem)
    else:
        data = {}
    pid = data.get("id")
    domain = data.get("domain")
    statement = data.get("statement") or data.get("text") or data.get("query")
    return pid, domain, statement


def _solution_fields(solution: Any):
    """Extract (solution_id, problem_id, text) from a duck-typed solution."""
    if solution is None:
        return "unknown", None, ""
    if is_dataclass(solution):
        data = asdict(solution)
    elif isinstance(solution, dict):
        data = solution
    elif hasattr(solution, "__dict__"):
        data = vars(solution)
    else:
        return "unknown", None, str(solution)

    sid = data.get("solver_id") or data.get("id") or "unknown"
    pid = data.get("problem_id")
    text = data.get("solution") or data.get("text") or data.get("answer") or ""
    if not isinstance(text, str):
        text = str(text)
    return sid, pid, text


# ===========================================================================
# Spec adapters
# ===========================================================================

def generate_knowledge_enhanced_specification(
    problem: Any,
    knowledge_engine: Any,
    context_query: Optional[str] = None,
    target_difficulty: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Knowledge-Augmented Specification Generation (spec section 1).

    Pulls relevant knowledge from the engine to enrich a problem's
    specification with real-world context, patterns and constraints.

    Returns a complete specification dict (never ``None``).
    """
    pid, domain, statement = _problem_fields(problem)
    statement = statement or (problem if isinstance(problem, str) else "unspecified problem")

    query = context_query or " ".join(filter(None, [str(domain or ""), str(statement)]))

    knowledge = _retrieve_knowledge(knowledge_engine, query)

    sources: List[str] = []
    if knowledge["available"]:
        for item in knowledge["results"]:
            text = _result_to_text(item)
            if text:
                sources.append(text)

    enhanced_statement = str(statement)
    if sources:
        context_block = "\n".join(f"- {s}" for s in sources)
        enhanced_statement = (
            f"{statement}\n\n"
            "[Knowledge-Enhanced Context]\n"
            "Use the following retrieved knowledge when interpreting and "
            f"constraining this specification:\n{context_block}"
        )

    return {
        "problem_id": pid,
        "domain": domain,
        "target_difficulty": target_difficulty,
        "original_statement": str(statement),
        "enhanced_specification": enhanced_statement,
        "enhanced": bool(sources),
        "knowledge": knowledge,
        "sources": sources,
    }


def solve_with_knowledge_context(
    problem: Any,
    knowledge_engine: Any,
    solver_model: Any = None,
) -> Dict[str, Any]:
    """
    Context-Aware Solution Generation (spec section 2).

    Retrieves similar solved problems / code patterns / proof techniques and
    bundles them into a ``solver_context`` that the solver can consume.

    Returns a complete context dict (never ``None``).
    """
    pid, domain, statement = _problem_fields(problem)
    statement = statement or (problem if isinstance(problem, str) else "unspecified problem")

    query = " ".join(filter(None, [str(domain or ""), str(statement)]))
    knowledge = _retrieve_knowledge(knowledge_engine, query)

    hints: List[str] = []
    for item in knowledge["results"]:
        text = _result_to_text(item)
        if text:
            hints.append(text)

    if hints:
        solver_context = (
            "Retrieved knowledge to guide the solution:\n"
            + "\n".join(f"- {h}" for h in hints)
        )
    else:
        solver_context = "No knowledge retrieved; solving without external context."

    return {
        "problem_id": pid,
        "context_available": knowledge["available"],
        "knowledge": knowledge,
        "solver_context": solver_context,
        "solver_hints": hints,
        "solver_model": getattr(solver_model, "__name__", str(solver_model))
        if solver_model is not None
        else None,
    }


def verify_with_knowledge(
    solution: Any,
    knowledge_engine: Any,
    problem: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Knowledge-Based Verification Enhancement (spec section 3).

    Cross-checks a solution against knowledge-base facts. The check is a
    transparent heuristic (substring / token overlap) so it runs with zero
    network or LLM dependency; it still reports which retrieved facts support
    or conflict with the proposed solution.

    Returns a complete verification dict (never ``None``).
    """
    sid, pid, solution_text = _solution_fields(solution)
    solution_text_lower = solution_text.lower()

    if problem is not None:
        _, domain, statement = _problem_fields(problem)
        query = " ".join(filter(None, [str(domain or ""), str(statement)]))
    else:
        query = solution_text[:280] or sid

    knowledge = _retrieve_knowledge(knowledge_engine, query)

    supported: List[str] = []
    conflicts: List[str] = []
    notes: List[str] = []

    for item in knowledge["results"]:
        fact = _result_to_text(item).strip()
        if not fact:
            continue
        # Heuristic support: an explicit "answer"/"=" fact that is echoed in
        # the solution text, or any fact whose salient tokens appear verbatim.
        if fact.lower() in solution_text_lower:
            supported.append(fact)
            continue
        if "=" in fact:
            rhs = fact.split("=", 1)[1].strip()
            if rhs and rhs.lower() in solution_text_lower:
                supported.append(fact)
                continue
        # Otherwise the retrieved fact is unconfirmed by the solution text.
        conflicts.append(fact)

    knowledge_available = knowledge["available"]
    verified_by_knowledge = bool(knowledge_available and supported and not conflicts)

    if not knowledge_available:
        notes.append("No knowledge engine available; verification not augmented.")
    elif supported and not conflicts:
        notes.append("Solution is consistent with all retrieved knowledge facts.")
    elif conflicts:
        notes.append(
            "Solution does not reflect some retrieved knowledge facts; "
            "manual review recommended."
        )
    else:
        notes.append("Knowledge retrieved but no overlap with the solution was found.")

    return {
        "solution_id": sid,
        "problem_id": pid,
        "verified_by_knowledge": verified_by_knowledge,
        "knowledge_available": knowledge_available,
        "knowledge_facts": knowledge["results"],
        "supported_facts": supported,
        "conflicts": conflicts,
        "notes": notes,
    }


__all__ = [
    "generate_knowledge_enhanced_specification",
    "solve_with_knowledge_context",
    "verify_with_knowledge",
    "_retrieve_knowledge",
]
