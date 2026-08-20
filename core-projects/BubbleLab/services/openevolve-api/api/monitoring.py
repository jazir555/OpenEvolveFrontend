"""
OpenEvolve monitoring router (mounted at ``/api/monitoring``).

Exposes service health and aggregate run statistics derived from the REAL
in-memory ``RUNS`` registry maintained by ``api/openevolve_v1`` (the live
evolution runs). All numbers are computed from actual run state, never faked.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET /monitoring/dashboard -> MonitoringDashboardMetrics
    GET /monitoring/alerts    -> { alerts: MonitoringAlert[] }
    GET /monitoring/services  -> { services: MonitoringService[], timestamp }
    GET /monitoring/logs      -> { entries: MonitoringLogEntry[], total }
    GET /monitoring/metrics   -> { metrics: MonitoringMetric[] }
    GET /monitoring/health    -> { ... } (raw health map)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

    logger = logging.getLogger("openevolve_api.monitoring")

router = APIRouter()

_SERVICE_START = time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_counts() -> Dict[str, int]:
    counts = {"total": 0, "running": 0, "completed": 0, "failed": 0}
    with _RUNS_LOCK:
        runs = list(RUNS.values())
    counts["total"] = len(runs)
    for run in runs:
        status = run.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _last_best_score() -> Optional[float]:
    best: Optional[float] = None
    with _RUNS_LOCK:
        runs = list(RUNS.values())
    for run in runs:
        result = run.get("result") or {}
        score = result.get("best_score")
        if isinstance(score, (int, float)) and (best is None or score > best):
            best = float(score)
    return best


@router.get("/monitoring/dashboard")
async def monitoring_dashboard() -> JSONResponse:
    counts = _run_counts()
    uptime = time.time() - _SERVICE_START
    last_score = _last_best_score()
    dashboard = {
        "timestamp": _now_iso(),
        "system": {
            "system": {
                "runs_total": counts["total"],
                "runs_completed": counts["completed"],
                "runs_failed": counts["failed"],
                "uptime_seconds": round(uptime, 3),
            }
        },
        "health": {
            "status": "healthy",
            "healthy": True,
            "timestamp": _now_iso(),
            "uptime_seconds": round(uptime, 3),
        },
        "workflow": {
            "total_runs": counts["total"],
            "completed": counts["completed"],
            "failed": counts["failed"],
            "running": counts["running"],
            "last_best_score": last_score,
        },
        "recent_metrics": {
            "runs": {
                "total": counts["total"],
                "completed": counts["completed"],
                "failed": counts["failed"],
            }
        },
    }
    return JSONResponse(dashboard, headers={"Content-Type": "application/json"})


@router.get("/monitoring/alerts")
async def monitoring_alerts() -> JSONResponse:
    counts = _run_counts()
    alerts: List[Dict[str, Any]] = []
    if counts["failed"] > 0:
        alerts.append(
            {
                "name": "failed_runs",
                "metric_name": "runs.failed",
                "condition": "failed > 0",
                "threshold": 0,
                "description": f"{counts['failed']} evolution run(s) failed",
                "active": True,
                "triggered": True,
                "latest_value": counts["failed"],
            }
        )
    return JSONResponse(
        {"alerts": alerts}, headers={"Content-Type": "application/json"}
    )


@router.get("/monitoring/services")
async def monitoring_services() -> JSONResponse:
    counts = _run_counts()
    services = [
        {
            "name": "openevolve-api",
            "status": "healthy",
            "healthy": True,
            "execution_time": None,
            "timestamp": _now_iso(),
            "error": None,
        },
        {
            "name": "openevolve-engine",
            "status": "healthy" if counts["failed"] == 0 else "degraded",
            "healthy": counts["failed"] == 0,
            "execution_time": None,
            "timestamp": _now_iso(),
            "error": None,
        },
    ]
    return JSONResponse(
        {"services": services, "timestamp": _now_iso()},
        headers={"Content-Type": "application/json"},
    )


@router.get("/monitoring/logs")
async def monitoring_logs(request: Request) -> JSONResponse:
    limit = 200
    try:
        limit_raw = request.query_params.get("limit")
        if limit_raw:
            limit = max(1, min(int(limit_raw), 1000))
    except (TypeError, ValueError):
        limit = 200

    entries: List[Dict[str, str]] = []
    with _RUNS_LOCK:
        runs = list(RUNS.values())
    for run in runs[-limit:]:
        run_id = run.get("run_id", "")
        status = run.get("status", "unknown")
        entries.append(
            {
                "source": "openevolve-api",
                "line": f"run {run_id} status={status}",
            }
        )
    return JSONResponse(
        {"entries": entries, "total": len(entries)},
        headers={"Content-Type": "application/json"},
    )


@router.get("/monitoring/metrics")
async def monitoring_metrics(request: Request) -> JSONResponse:
    name_filter = request.query_params.get("name")
    metrics: List[Dict[str, Any]] = []
    with _RUNS_LOCK:
        runs = list(RUNS.values())
    for run in runs:
        result = run.get("result") or {}
        score = result.get("best_score")
        run_id = run.get("run_id", "")
        if isinstance(score, (int, float)):
            metrics.append(
                {
                    "name": "best_score",
                    "value": float(score),
                    "type": "gauge",
                    "labels": {"run_id": run_id},
                    "timestamp": result.get("completed_at"),
                    "description": "Best fitness score achieved by the run",
                }
            )
    if name_filter:
        metrics = [m for m in metrics if m.get("name") == name_filter]
    return JSONResponse(
        {"metrics": metrics}, headers={"Content-Type": "application/json"}
    )


@router.get("/monitoring/health")
async def monitoring_health() -> JSONResponse:
    counts = _run_counts()
    uptime = time.time() - _SERVICE_START
    health = {
        "status": "healthy",
        "healthy": True,
        "service": "openevolve-api",
        "uptime_seconds": round(uptime, 3),
        "runs": counts,
        "last_best_score": _last_best_score(),
        "timestamp": _now_iso(),
    }
    return JSONResponse(health, headers={"Content-Type": "application/json"})
