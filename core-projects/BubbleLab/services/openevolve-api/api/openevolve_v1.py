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

# In-memory default run configuration applied to FUTURE orchestrate calls.
# Populated by POST /workflows/configure (e.g. {"max_iterations": N, "population_size": M}).
DEFAULT_RUN_CONFIG: Dict[str, Any] = {}

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
            # Preserve an externally-set status (stopped/paused) if the run was
            # halted while the worker was in flight.
            if RUNS.get(run_id, {}).get("status") == "running":
                RUNS[run_id]["status"] = "completed"
                RUNS[run_id]["result"] = result
    except Exception as exc:  # Never crash the worker thread.
        logger.error("openevolve_v1_run_failed", run_id=run_id, error=str(exc))
        with _RUNS_LOCK:
            if RUNS.get(run_id, {}).get("status") == "running":
                RUNS[run_id]["status"] = "failed"
                RUNS[run_id]["error"] = str(exc)


def _chain_worker(chain_id: str, steps: list) -> None:
    """Run each chain step sequentially; halt the chain on first failure."""
    for step in steps:
        step_id = step["run_id"]
        with _RUNS_LOCK:
            if RUNS.get(step_id, {}).get("status") in ("stopped", "paused"):
                continue  # skip steps halted before they began
            RUNS[step_id]["status"] = "running"
        try:
            bridge_request = _orchestrate_request_to_bridge(step["spec"])
            result = run_openevolve_workflow(bridge_request)
            with _RUNS_LOCK:
                RUNS[step_id]["status"] = "completed"
                RUNS[step_id]["result"] = result
        except Exception as exc:  # Halt the chain on any step failure.
            logger.error("openevolve_v1_chain_step_failed", chain_id=chain_id, step_id=step_id, error=str(exc))
            with _RUNS_LOCK:
                RUNS[step_id]["status"] = "failed"
                RUNS[step_id]["error"] = str(exc)
            break


def _update_run_status(run_id: str, new_status: str) -> Optional[Dict[str, Any]]:
    """Best-effort status transition for a run.

    Only transitions *from* ``running`` (or ``pending``); if the run already
    reached a terminal state, its current status is preserved and returned.
    Returns ``None`` when the run_id is unknown.
    """
    with _RUNS_LOCK:
        run = RUNS.get(run_id)
        if run is None:
            return None
        current = run["status"]
        if current not in ("completed", "failed"):
            run["status"] = new_status
        return dict(run)


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

    # Apply any config overrides registered via /workflows/configure (without
    # clobbering an explicit value supplied on this request).
    for key, value in DEFAULT_RUN_CONFIG.items():
        parameters.setdefault(key, value)

    # Optional real-LLM config. The bridge only uses a live model when a name
    # AND an api_key are supplied; otherwise it falls back to the offline mock.
    llm = body.get("llm")
    if isinstance(llm, dict):
        llm = dict(llm)
    else:
        llm = {}

    # Allow a top-level server ``config`` object to carry additional LLM/params
    # overrides without changing the canonical bridge ``parameters`` shape.
    config = body.get("config")
    if isinstance(config, dict):
        for key in ("max_iterations", "population_size", "temperature", "seed", "log_level"):
            if key in config and key not in parameters:
                parameters[key] = config[key]
        if not llm and isinstance(config.get("llm"), dict):
            llm = dict(config["llm"])

    return {
        "system": system,
        "problem_statement": problem,
        "parameters": parameters,
        "llm": llm,
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


@router.post("/workflows/{run_id}/stop")
async def stop_workflow(run_id: str) -> Response:
    run = _update_run_status(run_id, "stopped")
    if run is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run not found: {run_id}"},
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(
        {"workflowId": run_id, "status": run["status"], "operation": "stop_workflow"},
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/{run_id}/pause")
async def pause_workflow(run_id: str) -> Response:
    run = _update_run_status(run_id, "paused")
    if run is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run not found: {run_id}"},
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(
        {"workflowId": run_id, "status": run["status"], "operation": "pause_workflow"},
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/{run_id}/resume")
async def resume_workflow(run_id: str) -> Response:
    run = _update_run_status(run_id, "running")
    if run is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Run not found: {run_id}"},
            headers={"Content-Type": "application/json"},
        )
    return JSONResponse(
        {"workflowId": run_id, "status": run["status"], "operation": "resume_workflow"},
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/configure")
async def configure(request: Request) -> Response:
    try:
        body = await _json_payload(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
            headers={"Content-Type": "application/json"},
        )

    definition = body.get("definition")
    if not isinstance(definition, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "'definition' (object) is required"},
            headers={"Content-Type": "application/json"},
        )

    # Extract recognized overrides that the bridge understands.
    overrides: Dict[str, Any] = {}
    alias_map = {
        "iterations": "max_iterations",
        "maxIterations": "max_iterations",
        "max_iterations": "max_iterations",
        "generations": "max_iterations",
        "populationSize": "population_size",
        "population_size": "population_size",
        "seed": "seed",
        "temperature": "temperature",
    }
    for src, dst in alias_map.items():
        value = definition.get(src)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            overrides[dst] = value

    with _RUNS_LOCK:
        DEFAULT_RUN_CONFIG.update(overrides)
        effective = dict(DEFAULT_RUN_CONFIG)

    return JSONResponse(
        {
            "workflowName": body.get("workflowName"),
            "config": effective,
            "status": "configured",
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/batch")
async def batch_execute(request: Request) -> Response:
    if not OPENEVOLVE_BRIDGE_AVAILABLE:
        return JSONResponse(
            status_code=500,
            content={"error": f"openevolve bridge unavailable: {_OPENEVOLVE_IMPORT_ERROR}"},
            headers={"Content-Type": "application/json"},
        )
    try:
        body = await _json_payload(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
            headers={"Content-Type": "application/json"},
        )

    workflows = body.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        return JSONResponse(
            status_code=400,
            content={"error": "'workflows' (non-empty array) is required"},
            headers={"Content-Type": "application/json"},
        )

    run_ids: list = []
    for spec in workflows:
        if not isinstance(spec, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "each workflow spec must be an object"},
                headers={"Content-Type": "application/json"},
            )
        try:
            bridge_request = _orchestrate_request_to_bridge(spec)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
                headers={"Content-Type": "application/json"},
            )
        run_ids.append(_spawn(bridge_request))

    return JSONResponse(
        {
            "batchId": uuid.uuid4().hex,
            "run_ids": run_ids,
            "count": len(run_ids),
            "status": "running",
        },
        headers={"Content-Type": "application/json"},
    )


@router.post("/workflows/chain")
async def chain_workflows(request: Request) -> Response:
    if not OPENEVOLVE_BRIDGE_AVAILABLE:
        return JSONResponse(
            status_code=500,
            content={"error": f"openevolve bridge unavailable: {_OPENEVOLVE_IMPORT_ERROR}"},
            headers={"Content-Type": "application/json"},
        )
    try:
        body = await _json_payload(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
            headers={"Content-Type": "application/json"},
        )

    chain = body.get("chain")
    if not isinstance(chain, list) or not chain:
        return JSONResponse(
            status_code=400,
            content={"error": "'chain' (non-empty array) is required"},
            headers={"Content-Type": "application/json"},
        )

    steps: list = []
    with _RUNS_LOCK:
        for spec in chain:
            if not isinstance(spec, dict):
                return JSONResponse(
                    status_code=400,
                    content={"error": "each chain step must be an object"},
                    headers={"Content-Type": "application/json"},
                )
            step_id = uuid.uuid4().hex
            RUNS[step_id] = {
                "run_id": step_id,
                "status": "pending",
                "result": None,
                "error": None,
            }
            steps.append({"run_id": step_id, "spec": spec})

    chain_id = uuid.uuid4().hex
    thread = threading.Thread(
        target=_chain_worker, args=(chain_id, steps), daemon=True
    )
    thread.start()

    return JSONResponse(
        {
            "chainId": chain_id,
            "run_ids": [s["run_id"] for s in steps],
            "count": len(steps),
            "status": "running",
        },
        headers={"Content-Type": "application/json"},
    )
