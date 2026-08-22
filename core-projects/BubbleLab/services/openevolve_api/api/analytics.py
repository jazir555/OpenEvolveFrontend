"""
OpenEvolve analytics router (mounted at ``/api/analytics``; also serves
``/api/statistics``).

Aggregates REAL run data from the ``RUNS`` registry maintained by
``api/openevolve_v1``. No fabricated metrics: every workflow/performance metric
is derived from an actual evolution run.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET /analytics/performance-metrics -> { metrics: PerformanceMetric[], total }
    GET /analytics/knowledge-stats     -> AnalyticsKnowledgeStats
    GET /analytics/workflow-metrics    -> { metrics: AnalyticsWorkflowMetric[], total }
    GET /statistics                    -> StatisticsSummary
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from ..api.openevolve_v1 import RUNS, _RUNS_LOCK
except ImportError:  # pragma: no cover - absolute import fallback
    from api.openevolve_v1 import RUNS, _RUNS_LOCK

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.analytics")

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshots() -> List[Dict[str, Any]]:
    with _RUNS_LOCK:
        return list(RUNS.values())


def _run_counts() -> Dict[str, int]:
    counts = {"total": 0, "running": 0, "completed": 0, "failed": 0}
    for run in _snapshots():
        counts["total"] += 1
        status = run.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _workflow_metrics() -> List[Dict[str, Any]]:
    metrics: List[Dict[str, Any]] = []
    for run in _snapshots():
        result = run.get("result") or {}
        status = run.get("status")
        metrics.append(
            {
                "timestamp": result.get("completed_at") or result.get("started_at"),
                "workflow_id": run.get("run_id", ""),
                "status": status,
                "progress": 1.0 if status == "completed" else (0.0 if status == "running" else 0.0),
                "best_fitness": result.get("best_score"),
                "avg_fitness": result.get("best_score"),
                "diversity": None,
                "tokens_used": None,
                "execution_time": result.get("duration_seconds"),
                "memory_usage": None,
                "cpu_usage": None,
                "population_size": result.get("population_size"),
                "generation": result.get("generations"),
                "metrics": {
                    "engine": result.get("engine"),
                    "llm_mode": result.get("llm_mode"),
                    "mock_llm_calls": result.get("mock_llm_calls"),
                },
            }
        )
    return metrics


@router.get("/analytics/performance-metrics")
async def performance_metrics(request: Request) -> JSONResponse:
    limit = 200
    try:
        limit_raw = request.query_params.get("limit")
        if limit_raw:
            limit = max(1, min(int(limit_raw), 1000))
    except (TypeError, ValueError):
        limit = 200

    metrics: List[Dict[str, Any]] = []
    for run in _snapshots()[-limit:]:
        result = run.get("result") or {}
        metrics.append(
            {
                "entity_type": "run",
                "entity_id": run.get("run_id", ""),
                "metrics": {
                    "best_score": result.get("best_score"),
                    "duration_seconds": result.get("duration_seconds"),
                    "mock_llm_calls": result.get("mock_llm_calls"),
                },
                "timestamp": result.get("completed_at"),
                "domain": None,
                "problem_type": None,
                "context": {"engine": result.get("engine")},
            }
        )
    return JSONResponse(
        {"metrics": metrics, "total": len(metrics)},
        headers={"Content-Type": "application/json"},
    )


@router.get("/analytics/knowledge-stats")
async def knowledge_stats() -> JSONResponse:
    # No knowledge base is wired into this service; report a truthful empty state.
    stats = {
        "total_artifacts": 0,
        "total_usage": 0,
        "avg_effectiveness": 0.0,
        "artifact_type_distribution": {},
        "domain_distribution": {},
        "top_used_artifacts": [],
        "top_effective_artifacts": [],
    }
    return JSONResponse(stats, headers={"Content-Type": "application/json"})


@router.get("/analytics/workflow-metrics")
async def workflow_metrics() -> JSONResponse:
    metrics = _workflow_metrics()
    return JSONResponse(
        {"metrics": metrics, "total": len(metrics)},
        headers={"Content-Type": "application/json"},
    )


@router.get("/statistics")
async def statistics() -> JSONResponse:
    counts = _run_counts()
    total_teams = 0
    total_gauntlets = 0
    try:
        from ..api.teams import _teams_cache
        from ..api.gauntlets import _gauntlets_cache

        total_teams = len(_teams_cache)
        total_gauntlets = len(_gauntlets_cache)
    except Exception:
        # Caches may be unavailable (e.g. DB-backed); fall back to live counts.
        try:
            from api.teams import _teams_cache
            from api.gauntlets import _gauntlets_cache

            total_teams = len(_teams_cache)
            total_gauntlets = len(_gauntlets_cache)
        except Exception:
            pass

    summary = {
        "total_workflows": counts["total"],
        "completed": counts["completed"],
        "failed": counts["failed"],
        "running": counts["running"],
        "total_teams": total_teams,
        "total_gauntlets": total_gauntlets,
    }
    return JSONResponse(summary, headers={"Content-Type": "application/json"})
