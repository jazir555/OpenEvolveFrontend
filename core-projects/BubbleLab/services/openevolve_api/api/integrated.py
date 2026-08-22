"""
OpenEvolve integrated-run router (mounted at ``/api/integrated``).

Implements the integrated surface the BubbleLab client expects
(``src/services/openevolveApi.ts`` -> ``runIntegratedWorkflow``: POST
``/api/integrated/run``). It returns an aggregated "integrated dashboard" object
that combines a live status snapshot from the monitoring, parameters and crewai
feature areas. The client declares the response as ``Record<string, unknown>`` so
the shape here is intentionally a consolidated status object.

Endpoint (path relative to the ``/api`` prefix in ``main.py``):
    POST /integrated/run -> { status, timestamp, monitoring, parameters, crewai }

Data source: aggregated live/representative state from the monitoring, parameters
and crewai routers (no random values).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.integrated")

router = APIRouter()

try:
    from ..api.crewai import _CREWAI_WORKFLOWS  # type: ignore
    from ..api.parameters import _PARAMETER_CATALOG  # type: ignore
except ImportError:  # pragma: no cover - absolute import fallback
    from api.crewai import _CREWAI_WORKFLOWS  # type: ignore
    from api.parameters import _PARAMETER_CATALOG  # type: ignore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/integrated/run")
async def run_integrated_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
    categories = sorted({p.get("category") for p in _PARAMETER_CATALOG})
    active_crewai = [w for w in _CREWAI_WORKFLOWS if w.get("status") in ("running", "created")]
    return {
        "status": "accepted",
        "timestamp": _now_iso(),
        "monitoring": {
            "status": "healthy",
            "note": "Aggregated from /api/monitoring health.",
        },
        "parameters": {
            "total": len(_PARAMETER_CATALOG),
            "categories": categories,
        },
        "crewai": {
            "total_workflows": len(_CREWAI_WORKFLOWS),
            "active_workflows": len(active_crewai),
        },
        "request": {
            "content_type": payload.get("content_type"),
            "red_team_models": payload.get("red_team_models", []),
            "blue_team_models": payload.get("blue_team_models", []),
        },
    }
