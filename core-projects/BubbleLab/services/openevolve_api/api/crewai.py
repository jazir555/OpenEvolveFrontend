"""
OpenEvolve CrewAI router (mounted at ``/api/crewai``).

The BubbleLab client expects the CrewAI workflow surface
(``src/services/openevolveApi.ts`` -> ``listCrewaiWorkflows`` /
``getCrewaiWorkflow`` / ``getCrewaiWorkflowTickets``). This service has no live
CrewAI execution backend, so the endpoints return a structured, representative
catalog of CrewAI workflows and tickets that matches the client's
``CrewAIWorkflowSummary`` / ``CrewAIWorkflowTicket`` wire shapes. Values are
seeded from the documented CrewAI Unified Flow (6 phases) rather than generated
at random, so the UI renders meaningfully instead of 404ing.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET  /crewai/workflows                       -> { workflows, total }
    GET  /crewai/workflows/{workflow_id}         -> CrewAIWorkflowSummary
    GET  /crewai/workflows/{workflow_id}/tickets -> { tickets, total, status_breakdown }

Data source: a static, representative catalog (no live CrewAI backend).
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.crewai")

router = APIRouter()

# --------------------------------------------------------------------------- #
# Structured-static catalog (representative of the CrewAI Unified Flow phases)
# --------------------------------------------------------------------------- #
_PHASES = 6

_CREWAI_WORKFLOWS: List[Dict[str, Any]] = [
    {
        "workflow_id": "crewai-problem-decomp",
        "problem_statement": "Decompose a research protocol into verifiable sub-problems.",
        "phase": 2,
        "status": "completed",
        "execution_method": "ROMA",
        "created_at": "2026-07-14T09:12:00Z",
        "updated_at": "2026-07-14T09:41:00Z",
        "has_decomposition_plan": True,
        "num_sub_solutions": 5,
        "num_critiques": 12,
        "num_verification_results": 5,
        "has_reassembly_result": True,
        "has_final_validation": True,
    },
    {
        "workflow_id": "crewai-mdap-maker",
        "problem_statement": "Generate a minimum-description protocol artifact for a trading strategy.",
        "phase": 4,
        "status": "running",
        "execution_method": "ROMA_MDAP_MAKER",
        "created_at": "2026-08-02T15:30:00Z",
        "updated_at": "2026-08-20T02:10:00Z",
        "has_decomposition_plan": True,
        "num_sub_solutions": 3,
        "num_critiques": 7,
        "num_verification_results": 2,
        "has_reassembly_result": False,
        "has_final_validation": False,
    },
    {
        "workflow_id": "crewai-claudiomiro",
        "problem_statement": "Run the Claudiomiro ensemble pass over a long-horizon planning task.",
        "phase": 1,
        "status": "created",
        "execution_method": "Claudiomiro",
        "created_at": "2026-08-19T22:05:00Z",
        "updated_at": "2026-08-19T22:05:00Z",
        "has_decomposition_plan": False,
        "num_sub_solutions": 0,
        "num_critiques": 0,
        "num_verification_results": 0,
        "has_reassembly_result": False,
        "has_final_validation": False,
    },
]

_CREWAI_TICKETS: Dict[str, List[Dict[str, Any]]] = {
    "crewai-problem-decomp": [
        {
            "id": "T-1001",
            "title": "Draft decomposition plan",
            "description": "Split protocol into 5 sub-problems.",
            "status": "done",
            "assigned_agent_id": "agent-solver",
            "created_at": "2026-07-14T09:13:00Z",
            "updated_at": "2026-07-14T09:20:00Z",
            "sub_problem_id": "sp-1",
            "dependencies": [],
            "priority": 1,
        },
        {
            "id": "T-1002",
            "title": "Validate final reassembly",
            "description": "Confirm merged solution passes final validation.",
            "status": "done",
            "assigned_agent_id": "agent-judge",
            "created_at": "2026-07-14T09:35:00Z",
            "updated_at": "2026-07-14T09:41:00Z",
            "sub_problem_id": "sp-5",
            "dependencies": ["T-1001"],
            "priority": 2,
        },
    ],
    "crewai-mdap-maker": [
        {
            "id": "T-2001",
            "title": "Generate MDAP candidate",
            "description": "Produce minimum-description protocol artifact.",
            "status": "in_progress",
            "assigned_agent_id": "agent-mdap",
            "created_at": "2026-08-02T15:31:00Z",
            "updated_at": "2026-08-20T02:10:00Z",
            "sub_problem_id": "sp-2",
            "dependencies": [],
            "priority": 1,
        },
    ],
}


def _find_workflow(workflow_id: str) -> Dict[str, Any] | None:
    for wf in _CREWAI_WORKFLOWS:
        if wf["workflow_id"] == workflow_id:
            return wf
    return None


@router.get("/crewai/workflows")
async def list_crewai_workflows() -> Dict[str, Any]:
    return {"workflows": _CREWAI_WORKFLOWS, "total": len(_CREWAI_WORKFLOWS)}


@router.get("/crewai/workflows/{workflow_id}")
async def get_crewai_workflow(workflow_id: str) -> Dict[str, Any]:
    wf = _find_workflow(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail=f"CrewAI workflow not found: {workflow_id}")
    return wf


@router.get("/crewai/workflows/{workflow_id}/tickets")
async def get_crewai_workflow_tickets(workflow_id: str) -> Dict[str, Any]:
    if _find_workflow(workflow_id) is None:
        raise HTTPException(status_code=404, detail=f"CrewAI workflow not found: {workflow_id}")
    tickets = _CREWAI_TICKETS.get(workflow_id, [])
    breakdown: Dict[str, int] = {}
    for t in tickets:
        breakdown[t.get("status", "unknown")] = breakdown.get(t.get("status", "unknown"), 0) + 1
    return {"tickets": tickets, "total": len(tickets), "status_breakdown": breakdown}
