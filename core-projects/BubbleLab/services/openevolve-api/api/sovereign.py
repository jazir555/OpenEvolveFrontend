"""
Sovereign subsystem API routes for OpenEvolve.

Surfaces the real SQLite-backed sovereign decomposition subsystem
(engines/other/sovereign_persistence.py -> SovereignDatabase) through :8000.

NOTE on reads: the engine's ``ProblemDefinition.from_dict`` / read path is
currently broken upstream (the model only implements ``to_dict``), so the
``/problems`` and ``/plans`` endpoints read rows directly via
``SovereignDatabase.get_connection()`` and JSON-decode the TEXT columns. This is
robust and returns the real persisted data. The ``/run`` endpoint performs a
real problem analysis + persistence via ``ProblemAnalyzer`` + ``SovereignDatabase``
and degrades to HTTP 501 with a clear message when the engine is unavailable.
"""

import json
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = structlog.get_logger()
router = APIRouter()

# Ensure the sovereign engine modules (which live at <repo_root>/engines/other)
# are importable. This file is located in:
#   core-projects/BubbleLab/services/openevolve-api/api/sovereign.py
# so parents[5] is the repository root (OpenEvolveFrontend) and the engines
# package lives at <repo_root>/engines/other.
_SERVICE_DIR = Path(__file__).resolve().parents[1]  # .../services/openevolve-api
_REPO_ROOT = Path(__file__).resolve().parents[5]    # OpenEvolveFrontend
_ENGINES_OTHER = _REPO_ROOT / "engines" / "other"
_OPEVOLVE_SRC = _REPO_ROOT / "core-projects" / "openevolve"

for _p in (str(_ENGINES_OTHER), str(_OPEVOLVE_SRC), str(_REPO_ROOT), str(_SERVICE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from sovereign_persistence import SovereignDatabase
except Exception as exc:  # pragma: no cover - depends on engine source tree
    logger.warning("sovereign_persistence_unavailable", error=str(exc))
    SovereignDatabase = None

try:
    from sovereign_reliability import HealthMonitor
except Exception as exc:  # pragma: no cover
    logger.warning("sovereign_reliability_unavailable", error=str(exc))
    HealthMonitor = None

try:
    from problem_analyzer import ProblemAnalyzer
except Exception as exc:  # pragma: no cover
    logger.warning("sovereign_problem_analyzer_unavailable", error=str(exc))
    ProblemAnalyzer = None

# Absolute path to the shared sovereign DB file (same file the standalone
# api_server.py would create when run from engines/other).
_SOVEREIGN_DB_PATH = str(_ENGINES_OTHER / "sovereign_decomposition.db")

sovereign_db = None
if SovereignDatabase is not None:
    try:
        sovereign_db = SovereignDatabase(_SOVEREIGN_DB_PATH)
    except Exception as exc:  # pragma: no cover
        logger.warning("sovereign_database_init_failed", error=str(exc))
        sovereign_db = None

health_monitor = None
if HealthMonitor is not None:
    try:
        health_monitor = HealthMonitor()
    except Exception:  # pragma: no cover
        health_monitor = None

# In-memory tracking of decomposition runs started through /run.
_sovereign_runs: Dict[str, Dict[str, Any]] = {}

# Columns stored as JSON TEXT per table (decoded on read).
_JSON_COLUMNS = {
    "problems": {
        "domain_context", "complexity_score", "constraints",
        "success_criteria", "stakeholders", "resources_available", "metadata",
    },
    "decomposition_plans": {
        "sub_problems", "dependency_graph", "validation_checkpoints",
        "quality_scores", "metadata",
    },
    "sub_problems": {"dependencies", "success_criteria", "metadata"},
    "solution_attempts": {"validation_results", "feedback", "metadata"},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_sovereign_db() -> SovereignDatabase:
    if sovereign_db is None:
        raise HTTPException(
            status_code=501,
            detail="Sovereign database is not available in this environment.",
        )
    return sovereign_db


def _row_to_dict(row: sqlite3.Row, table: str) -> Dict[str, Any]:
    json_cols = _JSON_COLUMNS.get(table, set())
    out: Dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in json_cols and value is not None:
            try:
                out[key] = json.loads(value) if isinstance(value, str) else value
            except (json.JSONDecodeError, TypeError):
                out[key] = value
        else:
            out[key] = value
    return out


def _query_rows(table: str, where: Optional[str] = None,
                params: tuple = ()) -> List[Dict[str, Any]]:
    db = _ensure_sovereign_db()
    with db.get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        cursor.execute(sql, params)
        return [_row_to_dict(r, table) for r in cursor.fetchall()]


class SovereignRunRequest(BaseModel):
    problem_statement: str
    title: Optional[str] = None
    strategy: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


@router.get("/health")
async def sovereign_health() -> Dict[str, Any]:
    if health_monitor is not None:
        try:
            return health_monitor.run_health_checks()
        except Exception as exc:
            logger.warning("sovereign_health_check_failed", error=str(exc))
    return {
        "timestamp": _now_iso(),
        "overall_healthy": sovereign_db is not None,
        "checks": {"database": sovereign_db is not None},
    }


@router.get("/problems")
async def list_problems(problem_type: Optional[str] = None) -> Dict[str, Any]:
    if problem_type:
        rows = _query_rows("problems", "problem_type = ?", (problem_type,))
    else:
        rows = _query_rows("problems")
    return {"problems": rows}


@router.get("/problems/{problem_id}")
async def get_problem(problem_id: str) -> Dict[str, Any]:
    rows = _query_rows("problems", "id = ?", (problem_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Problem '{problem_id}' not found")
    return rows[0]


@router.get("/plans")
async def list_plans(status_filter: Optional[str] = None) -> Dict[str, Any]:
    if status_filter:
        rows = _query_rows("decomposition_plans", "status = ?", (status_filter,))
    else:
        rows = _query_rows("decomposition_plans")
    return {"plans": rows}


@router.get("/plans/{plan_id}")
async def get_plan(plan_id: str) -> Dict[str, Any]:
    rows = _query_rows("decomposition_plans", "id = ?", (plan_id,))
    if not rows:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found")
    return rows[0]


@router.get("/subproblems/{parent_id}")
async def list_subproblems(parent_id: str) -> Dict[str, Any]:
    rows = _query_rows("sub_problems", "parent_id = ?", (parent_id,))
    return {"sub_problems": rows}


@router.get("/solution-attempts/{sub_problem_id}")
async def list_solution_attempts(sub_problem_id: str) -> Dict[str, Any]:
    rows = _query_rows("solution_attempts", "sub_problem_id = ?", (sub_problem_id,))
    return {"solution_attempts": rows}


@router.get("/stats")
async def sovereign_stats() -> Dict[str, Any]:
    db = _ensure_sovereign_db()
    return db.get_database_stats()


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_sovereign_decomposition(request: SovereignRunRequest) -> Dict[str, Any]:
    """Analyze a problem and persist it through the sovereign subsystem."""
    if ProblemAnalyzer is None:
        raise HTTPException(
            status_code=501,
            detail="Sovereign problem analyzer (ProblemAnalyzer) is not available.",
        )
    db = _ensure_sovereign_db()

    try:
        analyzer = ProblemAnalyzer()
        problem = analyzer.analyze_problem(
            request.problem_statement, title=request.title or ""
        )
    except Exception as exc:
        logger.error("sovereign_analysis_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Problem analysis failed: {exc}")

    if problem is None:
        raise HTTPException(status_code=500, detail="Problem analysis returned None")

    try:
        db.create_problem(problem)
        problem_id = problem.id
    except Exception as exc:
        logger.error("sovereign_problem_persist_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Failed to persist problem: {exc}")

    run_id = f"sov_{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    run = {
        "run_id": run_id,
        "status": "completed",
        "problem_id": problem_id,
        "problem_statement": request.problem_statement,
        "strategy": request.strategy,
        "created_at": now,
        "updated_at": now,
    }
    _sovereign_runs[run_id] = run
    logger.info("sovereign_run_completed", run_id=run_id, problem_id=problem_id)

    return {
        "run_id": run_id,
        "problem_id": problem_id,
        "status": "completed",
        "problem": problem.to_dict(),
    }


@router.get("/runs")
async def list_sovereign_runs(status_filter: Optional[str] = None) -> Dict[str, Any]:
    runs = list(_sovereign_runs.values())
    if status_filter:
        runs = [r for r in runs if r.get("status") == status_filter]
    return {"runs": runs, "total": len(runs)}


@router.get("/runs/{run_id}")
async def get_sovereign_run(run_id: str) -> Dict[str, Any]:
    run = _sovereign_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Sovereign run '{run_id}' not found")
    return run
