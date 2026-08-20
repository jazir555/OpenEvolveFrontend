"""
OpenEvolve ``/api/v1/*`` dialect router for THIS primary service.

This mirrors the contract exposed by ``openevolve/server_stdlib.py`` (the
library's stdlib HTTP server) so that the BubbleLab integration bubbles can
target the primary OpenEvolve API service instead of a separate server.

Endpoints (mounted under ``/api/v1`` in ``main.py``):
    GET  /health                        -> {"status":"healthy","version": ...}
    POST /evolve                        -> spawn evolution run (202)
    GET  /runs/{run_id}                 -> poll run status/result (404 if unknown)
    POST /workflows/orchestrate         -> spawn workflow-style run (202)

All evolution is delegated to the REAL openevolve library via
``core.openevolve_bridge.run_openevolve_workflow`` (offline mock LLM by
default). Runs are executed on a daemon worker thread and tracked in an
in-memory registry.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

try:
    from ..core.openevolve_bridge import (
        OpenEvolveBridgeError,
        run_openevolve_workflow,
    )

    OPENEVOLVE_BRIDGE_AVAILABLE = True
    _OPENEVOLVE_IMPORT_ERROR: Optional[str] = None
except ImportError as _bridge_import_error:  # pragma: no cover - env dependent
    OpenEvolveBridgeError = RuntimeError  # type: ignore[assignment,misc]
    run_openevolve_workflow = None  # type: ignore[assignment]
    OPENEVOLVE_BRIDGE_AVAILABLE = False
    _OPENEVOLVE_IMPORT_ERROR = str(_bridge_import_error)


logger = structlog.get_logger()

router = APIRouter()

# In-memory registry of runs: run_id -> {"run_id", "status", "result", "error"}.
RUNS: Dict[str, Dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()

# Offline mock runs finish fast when kept tiny.
_DEFAULT_MAX_ITERATIONS = 3
_DEFAULT_POPULATION_SIZE = 6


def _svc_version() -> str:
    """Service version, preferring the real openevolve library version."""
    try:
        import openevolve

        return getattr(openevolve, "__version__", "0.1.0")
    except Exception:
        return "0.1.0"


def _get_run(run_id: str) -> Optional[Dict[str, Any]]:
    with _RUNS_LOCK:
        run = RUNS.get(run_id)
        return dict(run) if run else None


def _run_worker(run_id: str, bridge_request: Dict[str, Any]) -> None:
    """Execute the bridge call off the request thread; record outcome."""
    try:
        result = run_openevolve_workflow(bridge_request)
        with _RUNS_LOCK:
            RUNS[run_id]["status"] = "completed"
            RUNS[run_id]["result"] = result
    except Exception as exc:  # Never crash the worker thread.
        logger.error("openevolve_v1_run_failed", run_id=run_id, error=str(exc))
        with _RUNS_LOCK:
            RUNS[run_id]["status"] = "failed"
            RUNS[run_id]["error"] = str(exc)


def _spawn(bridge_request: Dict[str, Any]) -> str:
    """Register and start a background evolution run; return its run_id."""
    run_id = uuid.uuid4().hex
    with _RUNS_LOCK:
        RUNS[run_id] = {
            "run_id": run_id,
            "status": "running",
            "result": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_run_worker, args=(run_id, bridge_request), daemon=True
    )
    thread.start()
    return run_id


def _evolve_request_to_bridge(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a ``/evolve`` body into a bridge request."""
    initial_program = body.get("initial_program")
    evaluator = body.get("evaluator")
    if not isinstance(initial_program, str) or not initial_program.strip():
        raise ValueError("'initial_program' (str) is required")
    if not isinstance(evaluator, str) or not evaluator.strip():
        raise ValueError("'evaluator' (str) is required")
    if "def evaluate" not in evaluator:
        raise ValueError("'evaluator' must contain an 'evaluate(program_path)' function")

    parameters: Dict[str, Any] = {}
    iterations = body.get("iterations")
    if isinstance(iterations, int):
        parameters["max_iterations"] = iterations

    config = body.get("config")
    if isinstance(config, dict):
        for key in ("max_iterations", "population_size", "temperature", "seed", "log_level"):
            if key in config:
                parameters[key] = config[key]

    return {
        "system": "evolutionary",
        "initial_program": initial_program,
        "evaluator": evaluator,
        "parameters": parameters,
        # No live LLM credentials -> the bridge selects the offline mock backend.
        "llm": {},
    }


def _orchestrate_request_to_bridge(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a ``/workflows/orchestrate`` body into a bridge request."""
    system = body.get("system", "evolutionary")
    problem = body.get("problemStatement")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("'problemStatement' is required and must be a non-empty string")

    parameters: Dict[str, Any] = {}
    generations = body.get("generations")
    if isinstance(generations, int):
        parameters["max_iterations"] = generations
    population_size = body.get("populationSize")
    if isinstance(population_size, int):
        parameters["population_size"] = population_size

    # Pass through any extra evolution params the bubble might send.
    extra = body.get("parameters")
    if isinstance(extra, dict):
        parameters.update(extra)

    if not parameters:
        parameters = {
            "max_iterations": _DEFAULT_MAX_ITERATIONS,
            "population_size": _DEFAULT_POPULATION_SIZE,
        }

    return {
        "system": system,
        "problem_statement": problem,
        "parameters": parameters,
        "llm": {},
    }


async def _json_payload(request: Request) -> Dict[str, Any]:
    """Read and parse the request body as a JSON object (robust)."""
    import json

    raw = b""
    try:
        raw = await request.body()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON body: {exc}")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/health")
async def health() -> Response:
    return JSONResponse(
        {"status": "healthy", "version": _svc_version()},
        headers={"Content-Type": "application/json"},
    )


@router.post("/evolve")
async def evolve(request: Request) -> Response:
    if not OPENEVOLVE_BRIDGE_AVAILABLE:
        return JSONResponse(
            status_code=500,
            content={"error": f"openevolve bridge unavailable: {_OPENEVOLVE_IMPORT_ERROR}"},
            headers={"Content-Type": "application/json"},
        )
    try:
        body = await _json_payload(request)
        bridge_request = _evolve_request_to_bridge(body)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
            headers={"Content-Type": "application/json"},
        )
    except Exception as exc:  # Robust parse failures -> 500.
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to parse request: {exc}"},
            headers={"Content-Type": "application/json"},
        )

    run_id = _spawn(bridge_request)
    return JSONResponse(
        status_code=202,
        content={"run_id": run_id, "status": "running"},
        headers={"Content-Type": "application/json"},
    )


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> Response:
    run = _get_run(run_id)
    if run is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run not found: {run_id}"},
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(
        {
            "run_id": run["run_id"],
            "status": run["status"],
            "result": run["result"],
            "error": run["error"],
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/orchestrate")
async def orchestrate(request: Request) -> Response:
    if not OPENEVOLVE_BRIDGE_AVAILABLE:
        return JSONResponse(
            status_code=500,
            content={"error": f"openevolve bridge unavailable: {_OPENEVOLVE_IMPORT_ERROR}"},
            headers={"Content-Type": "application/json"},
        )
    try:
        body = await _json_payload(request)
        bridge_request = _orchestrate_request_to_bridge(body)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
            headers={"Content-Type": "application/json"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to parse request: {exc}"},
            headers={"Content-Type": "application/json"},
        )

    run_id = _spawn(bridge_request)
    # Mirror the workflowId convention the integration bubble reads.
    return JSONResponse(
        status_code=202,
        content={"workflowId": run_id, "status": "running"},
        headers={"Content-Type": "application/json"},
    )
