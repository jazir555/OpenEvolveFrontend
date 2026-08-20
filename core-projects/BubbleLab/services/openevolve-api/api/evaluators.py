"""
OpenEvolve evaluators catalog router (mounted at ``/api/evaluators``).

Implements the evaluator surface the BubbleLab client expects
(``src/services/openevolveApi.ts`` -> ``listEvaluators`` / ``uploadEvaluator`` /
``deleteEvaluator``). Returns a catalog of available evaluators keyed by id, with
real, representative evaluator source snippets (correctness, performance, security)
rather than random content.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET    /evaluators              -> { evaluators: Record<id, source> }
    POST   /evaluators              -> { evaluator_id }   (body: { code })
    DELETE /evaluators/{evaluator_id} -> { success, evaluator_id }

Data source: in-memory store seeded with representative evaluator definitions.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.evaluators")

router = APIRouter()

_REPRESENTATIVE_EVALUATORS: Dict[str, str] = {
    "correctness": (
        "def evaluate(solution: str, problem: str) -> float:\n"
        "    # Score 0..1 based on unit-test pass rate / expected-output match.\n"
        "    return 1.0 if _runs_clean(solution) else 0.0\n"
    ),
    "performance": (
        "def evaluate(solution: str, problem: str) -> float:\n"
        "    # Score 0..1 from normalized runtime / memory efficiency.\n"
        "    return _normalize_latency(solution)\n"
    ),
    "security": (
        "def evaluate(solution: str, problem: str) -> float:\n"
        "    # Score 0..1: penalize unsafe patterns (eval/exec, secrets leakage).\n"
        "    return 0.0 if _has_insecure_patterns(solution) else 1.0\n"
    ),
}

_EVALUATORS: Dict[str, str] = dict(_REPRESENTATIVE_EVALUATORS)


@router.get("/evaluators")
async def list_evaluators() -> Dict[str, Any]:
    return {"evaluators": _EVALUATORS}


@router.post("/evaluators")
async def upload_evaluator(payload: Dict[str, Any]) -> Dict[str, Any]:
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail="Field 'code' (non-empty string) is required.")
    evaluator_id = f"eval-{uuid.uuid4().hex[:8]}"
    _EVALUATORS[evaluator_id] = code
    return {"evaluator_id": evaluator_id}


@router.delete("/evaluators/{evaluator_id}")
async def delete_evaluator(evaluator_id: str) -> Dict[str, Any]:
    if evaluator_id in _REPRESENTATIVE_EVALUATORS:
        raise HTTPException(status_code=400, detail="Cannot delete a built-in evaluator.")
    if evaluator_id not in _EVALUATORS:
        raise HTTPException(status_code=404, detail=f"Evaluator not found: {evaluator_id}")
    del _EVALUATORS[evaluator_id]
    return {"success": True, "evaluator_id": evaluator_id}
