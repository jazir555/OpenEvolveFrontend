"""
Evolution API Routes for OpenEvolve

Self-contained, dependency-light FastAPI router implementing the BubbleLab SDK
contract for evolution runs:

  POST   /api/evolution/runs                -> start a run
  GET    /api/evolution/runs                -> list runs
  GET    /api/evolution/runs/{run_id}       -> get a run
  POST   /api/evolution/runs/{run_id}/stop  -> stop a run

Run records are kept in a module-level dict (in-memory). Each new run is
started in ``running`` status; a best-effort background thread flips the
status to ``completed`` after a short delay so the SDK gets a terminal state.
"""

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, HTTPException, status

logger = structlog.get_logger()

router = APIRouter()

# In-memory run store keyed by run_id.
_evolution_runs: dict[str, dict] = {}

# How long (seconds) a placeholder run stays "running" before completing.
_PLACEHOLDER_DURATION_SECONDS = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _complete_run_after_delay(run_id: str, delay: float) -> None:
    """Background no-op that advances a run to ``completed`` after ``delay``."""
    try:
        threading.Event().wait(delay)
    except Exception:  # pragma: no cover - defensive
        pass
    run = _evolution_runs.get(run_id)
    if run and run.get("status") == "running":
        run["status"] = "completed"
        run["updated_at"] = _now_iso()
        run["result"] = run.get("result") or {
            "summary": "placeholder evolution run completed",
            "generations": [],
            "best_score": None,
        }
        logger.info("evolution_run_completed", run_id=run_id)


def _get_run_or_404(run_id: str) -> dict:
    run = _evolution_runs.get(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evolution run '{run_id}' not found",
        )
    return run


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def create_evolution_run(payload: dict) -> dict:
    """Start an evolution run."""
    problem_statement = payload.get("problem_statement")
    if not problem_statement:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'problem_statement' is required",
        )

    run_id = f"evo_{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    run = {
        "run_id": run_id,
        "status": "running",
        "problem_statement": problem_statement,
        "config": payload.get("config", {}),
        "created_at": now,
        "updated_at": now,
        "result": None,
    }
    _evolution_runs[run_id] = run

    threading.Thread(
        target=_complete_run_after_delay,
        args=(run_id, _PLACEHOLDER_DURATION_SECONDS),
        daemon=True,
    ).start()

    logger.info("evolution_run_started", run_id=run_id)
    return {"run_id": run_id, "status": "running"}


@router.get("/runs")
async def list_evolution_runs(
    status_filter: Optional[str] = None,
) -> dict:
    """List evolution runs, optionally filtered by status."""
    runs = list(_evolution_runs.values())
    if status_filter:
        runs = [r for r in runs if r.get("status") == status_filter]
    return {"runs": runs, "total": len(runs)}


@router.get("/runs/{run_id}")
async def get_evolution_run(run_id: str) -> dict:
    """Get a single evolution run."""
    return _get_run_or_404(run_id)


@router.post("/runs/{run_id}/stop")
async def stop_evolution_run(run_id: str) -> dict:
    """Stop an evolution run."""
    run = _get_run_or_404(run_id)
    if run.get("status") not in ("running", "pending"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' is not running (status={run.get('status')})",
        )
    run["status"] = "stopped"
    run["updated_at"] = _now_iso()
    logger.info("evolution_run_stopped", run_id=run_id)
    return {"run_id": run_id, "status": "stopped"}
