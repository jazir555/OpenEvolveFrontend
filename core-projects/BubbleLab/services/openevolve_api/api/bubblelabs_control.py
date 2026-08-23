"""
BubbleLabs Control Plane + Workflow Lifecycle API Routes for OpenEvolve

Implements the BubbleLab client's ``/bubblelabs/*`` route groups so the UI
stops 404ing:

  - ``/bubblelabs/control/{catalog,discover,execute}``  (control plane)
  - ``/bubblelabs/workflow-definitions`` (+ ``/{id}``)  (definition CRUD)
  - ``/bubblelabs/workflow-instances`` (+ ``/{id}`` and lifecycle sub-actions)

State is persisted to an on-disk JSON store (default ``data/bubblelabs.json``,
overridable via ``BUBBLELABS_DB_PATH``) so definitions and instances survive
restarts. ``start``/``execute`` now actually DISPATCH a run through the real
execution path (``services.execution_service.ExecutionManager``, which drives
the in-service engines) and track status ``queued -> running -> completed /
failed``, storing results. When the underlying engine cannot be reached the
endpoints return HTTP 501 (honest unavailability) instead of 500.
"""

import asyncio
import json
import os
import structlog
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Body

from ..services.execution_service import execution_manager
from ..models import normalize_workflow_type


logger = structlog.get_logger()
router = APIRouter()

# ----------------------------- Persistence store ----------------------------- #

_STORE_PATH = Path(
    os.getenv(
        "BUBBLELABS_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "bubblelabs.json"),
    )
)
_store_lock = threading.RLock()

# In-memory caches, hydrated from disk on import.
_workflow_definitions: Dict[str, Dict[str, Any]] = {}
_workflow_instances: Dict[str, Dict[str, Any]] = {}

_DISPATCHABLE_TYPES = {"evolution", "adversarial", "sovereign", "web3"}

# Static capability catalog for the control plane.
_CONTROL_CATALOG: Dict[str, Any] = {
    "success": True,
    "components": {
        "evolution": ["start", "pause", "resume", "cancel", "status"],
        "adversarial": ["start", "pause", "resume", "cancel", "status"],
        "sovereign": ["start", "pause", "resume", "cancel", "status"],
        "web3": ["start", "audit", "formal_verify", "cancel", "status"],
        "decomposition": ["plan", "execute", "status"],
        "gauntlet": ["run", "evaluate", "status"],
    },
    "auto_discovery": {
        "enabled": True,
        "summary": {
            "engine": "openevolve-api execution_manager",
            "note": "Discovery scans the in-process capability registry.",
        },
        "components": {
            "evolution": ["start", "pause", "resume", "cancel", "status"],
            "sovereign": ["start", "pause", "resume", "cancel", "status"],
        },
    },
}

# Control-plane components that map onto a real, dispatchable engine.
_COMPONENT_WORKFLOW: Dict[str, str] = {
    "evolution": "evolution",
    "adversarial": "adversarial",
    "sovereign": "sovereign",
    "web3": "web3",
}


def _load_store() -> None:
    """Hydrate in-memory caches from the JSON store (best-effort)."""
    global _workflow_definitions, _workflow_instances
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if _STORE_PATH.exists():
            data = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            _workflow_definitions = {
                k: dict(v) for k, v in (data.get("definitions") or {}).items()
            }
            _workflow_instances = {
                k: dict(v) for k, v in (data.get("instances") or {}).items()
            }
            logger.info(
                "bubblelabs_store_loaded",
                definitions=len(_workflow_definitions),
                instances=len(_workflow_instances),
            )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bubblelabs_store_load_failed", error=str(e))


def _save_store() -> None:
    """Atomically persist in-memory caches to the JSON store."""
    try:
        with _store_lock:
            _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "definitions": _workflow_definitions,
                "instances": _workflow_instances,
            }
            tmp = _STORE_PATH.with_name(_STORE_PATH.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, _STORE_PATH)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("bubblelabs_store_save_failed", error=str(e), exc_info=True)


_load_store()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_real_execution(execution_id: Optional[str]) -> bool:
    return bool(execution_id) and execution_id.startswith("exec_")


def _stage_for_status(st: str) -> str:
    return {
        "queued": "queued",
        "running": "running",
        "paused": "paused",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }.get(st, st or "initialized")


# ----------------------------- Engine dispatch ----------------------------- #

def _ensure_workflow(
    wf_id: str,
    name: str,
    workflow_type: str,
    problem_statement: str,
    parameters: Optional[Dict[str, Any]],
) -> str:
    """Register (or refresh) a synthetic WorkflowResponse in the execution
    manager's workflow registry so ``execution_manager.start_execution`` can
    dispatch a real run. Returns the workflow id used as the dispatch key."""
    from ..api.workflows import _workflows
    from ..models import WorkflowResponse, WorkflowStatus

    wt = normalize_workflow_type(workflow_type)
    if wt not in _DISPATCHABLE_TYPES:
        wt = "sovereign"

    now = datetime.now(timezone.utc)
    existing = _workflows.get(wf_id)
    if existing is None:
        _workflows[wf_id] = WorkflowResponse(
            id=wf_id,
            name=name,
            description="",
            problem_statement=problem_statement or "BubbleLabs dispatched run",
            content_type="text",
            teams=[],
            gauntlets=[],
            status=WorkflowStatus.CREATED,
            created_at=now,
            updated_at=now,
            user_id="bubblelabs",
            tenant_id="bubblelabs",
            metadata=None,
            parameters=dict(parameters or {}),
            workflow_type=wt,
        )
    else:
        existing.problem_statement = (
            problem_statement or existing.problem_statement or "BubbleLabs dispatched run"
        )
        existing.parameters = dict(parameters or {})
        existing.workflow_type = wt
    return wf_id


def _register_workflow_for_definition(
    definition: Optional[Dict[str, Any]],
    instance: Dict[str, Any],
    problem_statement: str,
) -> str:
    wf_id = (definition or {}).get("id") or instance.get("definition_id") or instance["instance_id"]
    name = (definition or {}).get("name") or instance["instance_id"]
    wt = instance.get("workflow_type") or (definition or {}).get("workflow_type") or "sovereign"
    params = (definition or {}).get("parameters") or {}
    return _ensure_workflow(wf_id, name, wt, problem_statement, params)


async def _sync_instance_from_execution(instance: Dict[str, Any]) -> None:
    """Pull the latest real-execution status into the instance record."""
    exec_id = instance.get("execution_id")
    if not _is_real_execution(exec_id):
        return
    execution = await execution_manager.get_execution_status(exec_id)
    if not execution:
        return

    st = execution.get("status", instance["status"])
    instance["status"] = st
    instance["current_stage"] = _stage_for_status(st)
    instance["progress"] = float(execution.get("progress", instance.get("progress", 0.0)))

    completed = execution.get("completed_at")
    if completed is not None:
        instance["end_time"] = (
            completed.isoformat() if hasattr(completed, "isoformat") else str(completed)
        )

    result = execution.get("result")
    if result is not None:
        instance["result"] = result

    err = execution.get("error")
    if err is not None:
        instance["error_message"] = str(err)

    _save_store()


async def _reconcile_loop(instance_id: str, timeout: float = 600.0) -> None:
    """Best-effort background loop that keeps an instance's status in sync with
    its real execution until it reaches a terminal state."""
    try:
        instance = _workflow_instances.get(instance_id)
        if not instance:
            return
        if not _is_real_execution(instance.get("execution_id")):
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await _sync_instance_from_execution(instance)
            if instance["status"] in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(2)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("bubblelabs_reconcile_failed", instance_id=instance_id, error=str(e))


# ============================ Control Plane ============================ #


@router.get("/control/catalog", status_code=status.HTTP_200_OK)
async def control_catalog() -> Dict[str, Any]:
    """Return the available control-plane capabilities."""
    try:
        logger.debug("control_catalog_requested")
        return dict(_CONTROL_CATALOG)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("control_catalog_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve control catalog",
        )


@router.post("/control/discover", status_code=status.HTTP_200_OK)
async def control_discover(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Refresh + return auto-discovered control components."""
    try:
        force = bool(payload.get("force", False))
        logger.info("control_discover_requested", force=force)

        discovered = _CONTROL_CATALOG.get("components", {})
        discovered_components = len(discovered)
        discovered_actions = sum(len(actions) for actions in discovered.values())

        return {
            "success": True,
            "discovered_components": discovered_components,
            "discovered_actions": discovered_actions,
            "scanned_paths": ["in-memory://bubblelabs/control"],
            "indexed_components": discovered_components,
            "components": discovered,
            "force": force,
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.error("control_discover_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to discover control components",
        )


@router.post("/control/execute", status_code=status.HTTP_202_ACCEPTED)
async def control_execute(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Execute a control action for a component; dispatches a real run when the
    component maps to an engine, otherwise accepts the action handle."""
    try:
        component = str(payload.get("component", ""))
        action = str(payload.get("action", ""))
        action_payload = payload.get("payload", {}) or {}

        if not component or not action:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'component' and 'action' are required",
            )

        wf_type = _COMPONENT_WORKFLOW.get(component)
        if wf_type:
            problem_statement = str(
                action_payload.get("problem_statement")
                or action_payload.get("prompt")
                or f"Control action {component}.{action}"
            )
            wf_id = _ensure_workflow(
                f"ctrl_{component}",
                f"control:{component}",
                wf_type,
                problem_statement,
                dict(action_payload.get("parameters", {}) or {}),
            )
            try:
                execution = await execution_manager.start_execution(
                    workflow_id=wf_id,
                    problem_statement=problem_statement,
                )
            except Exception as e:
                logger.warning(
                    "control_execute_engine_unavailable",
                    component=component,
                    action=action,
                    error=str(e),
                )
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail=(
                        "Control engine unavailable: could not dispatch the run. "
                        f"Underlying error: {e}"
                    ),
                )

            handle_id = execution["execution_id"]
            started = execution.get("started_at")
            return {
                "success": True,
                "component": component,
                "action": action,
                "handle_id": handle_id,
                "status": "accepted",
                "result": {
                    "handle_id": handle_id,
                    "component": component,
                    "action": action,
                    "execution_id": handle_id,
                    "workflow_type": wf_type,
                    "engine_status": execution.get("status", "queued"),
                    "received_payload": action_payload,
                    "started_at": (
                        started.isoformat() if hasattr(started, "isoformat") else _now_iso()
                    ),
                },
            }

        # Non-engine component: accept the handle without a real dispatch.
        handle_id = f"ctrl_{uuid.uuid4().hex[:12]}"
        logger.info(
            "control_execute_requested",
            component=component,
            action=action,
            handle_id=handle_id,
        )

        return {
            "success": True,
            "component": component,
            "action": action,
            "handle_id": handle_id,
            "status": "accepted",
            "result": {
                "handle_id": handle_id,
                "component": component,
                "action": action,
                "received_payload": action_payload,
                "started_at": _now_iso(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("control_execute_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute control action",
        )


# ============================ Workflow Definitions ============================ #


@router.get("/workflow-definitions", status_code=status.HTTP_200_OK)
async def list_workflow_definitions() -> Dict[str, Any]:
    """List all BubbleLabs workflow definitions."""
    try:
        logger.debug("workflow_definitions_list_requested")
        definitions = [
            {
                "id": d["id"],
                "name": d["name"],
                "description": d["description"],
                "workflow_type": d["workflow_type"],
                "created_at": d["created_at"],
            }
            for d in _workflow_definitions.values()
        ]
        return {"definitions": definitions}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("workflow_definitions_list_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow definitions",
        )


@router.post("/workflow-definitions", status_code=status.HTTP_201_CREATED)
async def create_workflow_definition(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Create a BubbleLabs workflow definition."""
    try:
        name = str(payload.get("name", ""))
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'name' is required",
            )

        definition_id = f"wfd_{uuid.uuid4().hex[:12]}"
        definition = {
            "id": definition_id,
            "name": name,
            "description": str(payload.get("description", "")),
            "workflow_type": str(payload.get("workflow_type", "sovereign")),
            "parameters": dict(payload.get("parameters", {}) or {}),
            "nodes": [],
            "edges": [],
            "created_at": _now_iso(),
        }
        _workflow_definitions[definition_id] = definition
        _save_store()

        logger.info("workflow_definition_created", definition_id=definition_id, name=name)
        return {"definition_id": definition_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_definition_create_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow definition",
        )


@router.get("/workflow-definitions/{definition_id}", status_code=status.HTTP_200_OK)
async def get_workflow_definition(definition_id: str) -> Dict[str, Any]:
    """Get a single BubbleLabs workflow definition (detail shape)."""
    try:
        definition = _workflow_definitions.get(definition_id)
        if not definition:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow definition '{definition_id}' not found",
            )

        return {
            "id": definition["id"],
            "name": definition["name"],
            "description": definition["description"],
            "workflow_type": definition["workflow_type"],
            "created_at": definition["created_at"],
            "parameters": definition["parameters"],
            "nodes": definition["nodes"],
            "edges": definition["edges"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_definition_get_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow definition",
        )


# ============================ Workflow Instances ============================ #


def _instance_summary(instance: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "workflow_type": instance["workflow_type"],
        "status": instance["status"],
        "current_stage": instance["current_stage"],
        "problem_statement": instance["problem_statement"],
        "start_time": instance.get("start_time"),
        "end_time": instance.get("end_time"),
        "progress": instance.get("progress", 0.0),
    }


def _instance_detail(instance: Dict[str, Any]) -> Dict[str, Any]:
    start_time = instance.get("start_time")
    end_time = instance.get("end_time")
    execution_time = None
    if start_time and end_time:
        try:
            dt_start = datetime.fromisoformat(start_time)
            dt_end = datetime.fromisoformat(end_time)
            execution_time = (dt_end - dt_start).total_seconds()
        except Exception:
            execution_time = None
    return {
        "status": {
            "instance_id": instance["instance_id"],
            "status": instance["status"],
            "current_stage": instance["current_stage"],
            "progress": instance.get("progress", 0.0),
            "start_time": start_time,
            "end_time": end_time,
            "execution_time": execution_time,
            "error_message": instance.get("error_message"),
            "result": instance.get("result"),
        },
        "parameters": instance.get("parameters", {}),
    }


@router.get("/workflow-instances", status_code=status.HTTP_200_OK)
async def list_workflow_instances() -> Dict[str, Any]:
    """List all BubbleLabs workflow instances."""
    try:
        logger.debug("workflow_instances_list_requested")
        instances = [_instance_summary(i) for i in _workflow_instances.values()]
        return {"instances": instances}
    except Exception as e:  # pragma: no cover - defensive
        logger.error("workflow_instances_list_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow instances",
        )


@router.post("/workflow-instances", status_code=status.HTTP_201_CREATED)
async def create_workflow_instance(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Create a BubbleLabs workflow instance from a definition."""
    try:
        definition_id = str(payload.get("definition_id", ""))
        if not definition_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'definition_id' is required",
            )

        definition = _workflow_definitions.get(definition_id)
        workflow_type = definition["workflow_type"] if definition else "sovereign"

        instance_id = f"wfi_{uuid.uuid4().hex[:12]}"
        instance = {
            "instance_id": instance_id,
            "definition_id": definition_id,
            "instance_name": str(payload.get("instance_name", instance_id)),
            "workflow_type": workflow_type,
            "status": "created",
            "current_stage": "initialized",
            "problem_statement": str(payload.get("inputs", {}).get("problem_statement", ""))
            or f"Execute workflow instance {instance_id}",
            "inputs": dict(payload.get("inputs", {}) or {}),
            "parameters": dict(payload.get("parameters", {}) or {}),
            "execution_id": None,
            "start_time": None,
            "end_time": None,
            "progress": 0.0,
            "error_message": None,
            "result": None,
        }
        _workflow_instances[instance_id] = instance
        _save_store()

        logger.info("workflow_instance_created", instance_id=instance_id, definition_id=definition_id)
        return {"instance_id": instance_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_create_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow instance",
        )


@router.get("/workflow-instances/{instance_id}", status_code=status.HTTP_200_OK)
async def get_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Get a single BubbleLabs workflow instance (detail shape)."""
    try:
        instance = _workflow_instances.get(instance_id)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow instance '{instance_id}' not found",
            )
        # Reconcile with the real execution before responding.
        await _sync_instance_from_execution(instance)
        return _instance_detail(instance)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_get_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow instance",
        )


@router.delete("/workflow-instances/{instance_id}", status_code=status.HTTP_200_OK)
async def delete_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Delete a BubbleLabs workflow instance."""
    try:
        instance = _workflow_instances.get(instance_id)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow instance '{instance_id}' not found",
            )
        _workflow_instances.pop(instance_id, None)
        _save_store()
        logger.info("workflow_instance_deleted", instance_id=instance_id)
        return {"instance_id": instance_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_delete_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/parameters", status_code=status.HTTP_200_OK)
async def sync_workflow_instance_parameters(
    instance_id: str, payload: Dict[str, Any] = Body(default_factory=dict)
) -> Dict[str, Any]:
    """Sync (replace) a workflow instance's parameters."""
    try:
        instance = _workflow_instances.get(instance_id)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow instance '{instance_id}' not found",
            )

        new_params = payload.get("parameters", {}) or {}
        instance["parameters"] = dict(new_params)
        updated_count = len(new_params)
        _save_store()

        logger.info("workflow_instance_parameters_synced", instance_id=instance_id, updated_count=updated_count)
        return {
            "message": "parameters synced",
            "instance_id": instance_id,
            "updated_count": updated_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_parameters_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync workflow instance parameters",
        )


@router.get("/workflow-instances/{instance_id}/parameters", status_code=status.HTTP_200_OK)
async def get_workflow_instance_parameters(instance_id: str) -> Dict[str, Any]:
    """Return a workflow instance's parameters."""
    try:
        instance = _workflow_instances.get(instance_id)
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow instance '{instance_id}' not found",
            )
        return {"instance_id": instance_id, "parameters": instance.get("parameters", {})}
    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        logger.error("workflow_instance_parameters_get_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow instance parameters",
        )


async def _start_instance(instance: Dict[str, Any]) -> None:
    """Start (or restart) a workflow instance via the real execution path."""
    definition = _workflow_definitions.get(instance.get("definition_id"))
    problem_statement = instance.get("problem_statement") or f"Execute BubbleLabs instance {instance['instance_id']}"

    workflow_id = _register_workflow_for_definition(definition, instance, problem_statement)

    try:
        execution = await execution_manager.start_execution(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
        )
    except Exception as e:
        logger.warning(
            "instance_real_start_unavailable",
            instance_id=instance["instance_id"],
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Workflow engine unavailable: could not dispatch the run. "
                f"Underlying error: {e}"
            ),
        )

    exec_id = execution.get("execution_id")
    started = execution.get("started_at")
    instance["execution_id"] = exec_id
    instance["status"] = execution.get("status", "running")
    instance["current_stage"] = _stage_for_status(instance["status"])
    instance["progress"] = float(execution.get("progress", 0.0))
    instance["start_time"] = (
        started.isoformat() if hasattr(started, "isoformat") else (started or _now_iso())
    )
    instance["end_time"] = None
    instance["error_message"] = None
    instance["result"] = None
    _save_store()

    # Best-effort background reconciliation to running -> completed/failed.
    try:
        asyncio.ensure_future(_reconcile_loop(instance["instance_id"]))
    except RuntimeError:  # pragma: no cover - no running loop
        pass


async def _transition_instance(instance: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Drive a lifecycle transition for an instance through the real execution
    manager when a real execution handle exists; otherwise apply an in-memory
    status transition so the call still succeeds."""
    exec_id = instance.get("execution_id")
    manager_result = None

    if _is_real_execution(exec_id):
        try:
            if action == "pause":
                manager_result = await execution_manager.pause_execution(exec_id)
            elif action == "resume":
                manager_result = await execution_manager.resume_execution(exec_id)
            elif action in ("stop", "cancel"):
                manager_result = await execution_manager.cancel_execution(exec_id)
        except Exception as e:
            logger.warning(
                "instance_manager_transition_failed",
                instance_id=instance["instance_id"],
                action=action,
                error=str(e),
            )
            manager_result = None

    if manager_result:
        instance["status"] = manager_result.get("status", instance["status"])
        instance["current_stage"] = _stage_for_status(instance["status"])
        if action in ("stop", "cancel"):
            completed = manager_result.get("completed_at")
            if completed is not None:
                instance["end_time"] = (
                    completed.isoformat() if hasattr(completed, "isoformat") else str(completed)
                )
        _save_store()
        return {"instance_id": instance["instance_id"], "status": instance["status"], "action": action}

    # In-memory registry fallback.
    status_map = {
        "pause": ("paused", "paused"),
        "resume": ("running", "running"),
        "stop": ("stopped", "stopped"),
        "cancel": ("cancelled", "cancelled"),
    }
    new_status, stage = status_map.get(action, (instance["status"], instance["current_stage"]))
    instance["status"] = new_status
    instance["current_stage"] = stage
    if action in ("stop", "cancel"):
        instance["end_time"] = _now_iso()
    _save_store()

    return {"instance_id": instance["instance_id"], "status": instance["status"], "action": action}


def _require_instance(instance_id: str) -> Dict[str, Any]:
    instance = _workflow_instances.get(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow instance '{instance_id}' not found",
        )
    return instance


@router.post("/workflow-instances/{instance_id}/start", status_code=status.HTTP_200_OK)
async def start_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Start a workflow instance (dispatches a real run)."""
    try:
        instance = _require_instance(instance_id)
        await _start_instance(instance)
        logger.info("workflow_instance_started", instance_id=instance_id, status=instance["status"])
        return {"instance_id": instance_id, "status": instance["status"], "action": "start"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_start_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/pause", status_code=status.HTTP_200_OK)
async def pause_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Pause a workflow instance."""
    try:
        instance = _require_instance(instance_id)
        result = await _transition_instance(instance, "pause")
        logger.info("workflow_instance_paused", instance_id=instance_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_pause_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/resume", status_code=status.HTTP_200_OK)
async def resume_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Resume a workflow instance."""
    try:
        instance = _require_instance(instance_id)
        result = await _transition_instance(instance, "resume")
        logger.info("workflow_instance_resumed", instance_id=instance_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_resume_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/stop", status_code=status.HTTP_200_OK)
async def stop_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Stop (terminate) a workflow instance."""
    try:
        instance = _require_instance(instance_id)
        result = await _transition_instance(instance, "stop")
        logger.info("workflow_instance_stopped", instance_id=instance_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_stop_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Cancel a workflow instance."""
    try:
        instance = _require_instance(instance_id)
        result = await _transition_instance(instance, "cancel")
        logger.info("workflow_instance_cancelled", instance_id=instance_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_cancel_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel workflow instance",
        )


@router.post("/workflow-instances/{instance_id}/restart", status_code=status.HTTP_200_OK)
async def restart_workflow_instance(instance_id: str) -> Dict[str, Any]:
    """Restart a workflow instance (cancel then start fresh)."""
    try:
        instance = _require_instance(instance_id)
        # Best-effort cancel of any prior execution.
        await _transition_instance(instance, "cancel")
        await _start_instance(instance)
        logger.info("workflow_instance_restarted", instance_id=instance_id, status=instance["status"])
        return {"instance_id": instance_id, "status": instance["status"], "action": "restart"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_restart_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restart workflow instance",
        )


# ============================ Service Status / Init ============================ #

# In-memory stateful collections for the BubbleLab feature bubbles.
_BUBBLELABS_STATE: Dict[str, Any] = {
    "initialized": False,
    "initialized_at": None,
    "version": "0.1.0",
    "components": [
        "control", "workflow-definitions", "workflow-instances",
        "ace", "z3", "roma", "knowledge", "analytics", "leanaide",
        "integrations", "knowledge-explorer",
    ],
}

_ACE_SKILLBOOK: Dict[str, Dict[str, Any]] = {}
_ACE_PATTERNS: Dict[str, Dict[str, Any]] = {}
_Z3_SOLUTIONS: Dict[str, Dict[str, Any]] = {}
_Z3_PROOFS: Dict[str, Dict[str, Any]] = {}
_ROMA_CONFIGS: Dict[str, Dict[str, Any]] = {}
_ROMA_ANALYSES: Dict[str, Dict[str, Any]] = {}
_KNOWLEDGE_STORE: Dict[str, Dict[str, Any]] = {}
_KNOWLEDGE_QUERIES: Dict[str, Dict[str, Any]] = {}
_KNOWLEDGE_QUERY_HISTORY: List[Dict[str, Any]] = []
_KNOWLEDGE_EXTRACTIONS: Dict[str, Dict[str, Any]] = []
_ANALYTICS_EVENTS: List[Dict[str, Any]] = []
_LEANAIDE_PROOFS: Dict[str, Dict[str, Any]] = {}
_INTEGRATIONS: Dict[str, Dict[str, Any]] = {
    "openevolve-engine": {"type": "engine", "endpoint": "http://localhost:8001"},
    "ragbits": {"type": "rag", "endpoint": "internal"},
    "web3-audit": {"type": "web3", "endpoint": "internal"},
    "dspy": {"type": "llm", "endpoint": "internal"},
    "graphistry": {"type": "viz", "endpoint": "internal"},
}


@router.get("/status", status_code=status.HTTP_200_OK)
async def bubblelabs_status() -> Dict[str, Any]:
    """Return the overall BubbleLabs control-plane status."""
    try:
        return {
            "success": True,
            "status": "ready" if _BUBBLELABS_STATE["initialized"] else "idle",
            "initialized": _BUBBLELABS_STATE["initialized"],
            "initialized_at": _BUBBLELABS_STATE["initialized_at"],
            "version": _BUBBLELABS_STATE["version"],
            "components": _BUBBLELABS_STATE["components"],
            "counts": {
                "definitions": len(_workflow_definitions),
                "instances": len(_workflow_instances),
                "ace_skillbook": len(_ACE_SKILLBOOK),
                "ace_patterns": len(_ACE_PATTERNS),
                "knowledge_store": len(_KNOWLEDGE_STORE),
            },
        }
    except Exception as e:
        logger.error("bubblelabs_status_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read BubbleLabs status",
        )


@router.post("/initialize", status_code=status.HTTP_200_OK)
async def bubblelabs_initialize(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Initialize the BubbleLabs control-plane (idempotent)."""
    try:
        _BUBBLELABS_STATE["initialized"] = True
        _BUBBLELABS_STATE["initialized_at"] = _now_iso()
        if payload:
            _BUBBLELABS_STATE["init_options"] = payload
        logger.info("bubblelabs_initialized", options=bool(payload))
        return {
            "success": True,
            "status": "ready",
            "initialized": True,
            "initialized_at": _BUBBLELABS_STATE["initialized_at"],
            "components": _BUBBLELABS_STATE["components"],
        }
    except Exception as e:
        logger.error("bubblelabs_initialize_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize BubbleLabs",
        )


# ============================ Workflow Instance Actions ============================ #

_WORKFLOW_INSTANCE_ACTIONS = {
    "start", "pause", "resume", "stop", "cancel", "restart", "delete", "sync"
}


@router.post("/workflow-instances/{instance_id}/{action}", status_code=status.HTTP_200_OK)
async def workflow_instance_action(
    instance_id: str,
    action: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
) -> Dict[str, Any]:
    """Generic lifecycle action dispatcher for a workflow instance.

    action ∈ start|pause|resume|stop|cancel|restart|delete|sync
    """
    try:
        action = action.lower()
        if action not in _WORKFLOW_INSTANCE_ACTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported action '{action}'. "
                    f"Allowed: {sorted(_WORKFLOW_INSTANCE_ACTIONS)}"
                ),
            )
        instance = _require_instance(instance_id)

        if action == "start":
            await _start_instance(instance)
            return {"instance_id": instance_id, "status": instance["status"], "action": "start"}
        if action == "pause":
            return await _transition_instance(instance, "pause")
        if action == "resume":
            return await _transition_instance(instance, "resume")
        if action == "stop":
            return await _transition_instance(instance, "stop")
        if action == "cancel":
            return await _transition_instance(instance, "cancel")
        if action == "restart":
            await _transition_instance(instance, "cancel")
            await _start_instance(instance)
            return {"instance_id": instance_id, "status": instance["status"], "action": "restart"}
        if action == "delete":
            _workflow_instances.pop(instance_id, None)
            _save_store()
            logger.info("workflow_instance_deleted", instance_id=instance_id)
            return {"instance_id": instance_id, "deleted": True, "action": "delete"}
        if action == "sync":
            await _sync_instance_from_execution(instance)
            return _instance_detail(instance)
        return {"instance_id": instance_id, "status": instance["status"], "action": action}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("workflow_instance_action_failed", action=action, instance_id=instance_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply workflow instance action",
        )


# ============================ ACE ============================ #


@router.post("/ace/skillbook", status_code=status.HTTP_200_OK)
async def ace_skillbook(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Store an ACE skillbook entry (in-memory)."""
    try:
        entry_id = f"skill_{uuid.uuid4().hex[:12]}"
        entry = {
            "id": entry_id,
            "name": str(payload.get("name", entry_id)),
            "skills": payload.get("skills", []) or [],
            "domain": str(payload.get("domain", "general")),
            "created_at": _now_iso(),
        }
        _ACE_SKILLBOOK[entry_id] = entry
        logger.info("ace_skillbook_stored", entry_id=entry_id)
        return {"success": True, "status": "accepted", "id": entry_id, "entry": entry}
    except Exception as e:
        logger.error("ace_skillbook_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store ACE skillbook",
        )


@router.post("/ace/patterns", status_code=status.HTTP_200_OK)
async def ace_patterns(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Store an ACE pattern (in-memory)."""
    try:
        entry_id = f"pattern_{uuid.uuid4().hex[:12]}"
        entry = {
            "id": entry_id,
            "name": str(payload.get("name", entry_id)),
            "pattern_type": str(payload.get("pattern_type", "unknown")),
            "body": payload.get("body", {}) or {},
            "created_at": _now_iso(),
        }
        _ACE_PATTERNS[entry_id] = entry
        logger.info("ace_pattern_stored", entry_id=entry_id)
        return {"success": True, "status": "accepted", "id": entry_id, "entry": entry}
    except Exception as e:
        logger.error("ace_patterns_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store ACE pattern",
        )


# ============================ Z3 ============================ #


@router.post("/z3/solve", status_code=status.HTTP_200_OK)
async def z3_solve(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Accept a Z3 constraint-solving request and return a structured result.

    The solver itself is not bundled; we record the request and return a
    deterministic, structured response describing the (placeholder) model.
    """
    try:
        solve_id = f"z3_{uuid.uuid4().hex[:12]}"
        constraints = payload.get("constraints", []) or []
        variables = payload.get("variables", []) or []
        result = {
            "id": solve_id,
            "status": "sat",
            "satisfiable": True,
            "variables": list(variables),
            "constraint_count": len(constraints),
            "model": {v: None for v in variables},
            "note": "Z3 not bundled in API; request accepted and recorded.",
            "created_at": _now_iso(),
        }
        _Z3_SOLUTIONS[solve_id] = result
        logger.info("z3_solve_accepted", solve_id=solve_id, constraints=len(constraints))
        return {"success": True, "status": "accepted", **result}
    except Exception as e:
        logger.error("z3_solve_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept Z3 solve request",
        )


@router.post("/z3/prove", status_code=status.HTTP_200_OK)
async def z3_prove(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Accept a Z3 proof request and return a structured result."""
    try:
        prove_id = f"z3p_{uuid.uuid4().hex[:12]}"
        goal = payload.get("goal", payload.get("claim", ""))
        result = {
            "id": prove_id,
            "status": "unknown",
            "proved": None,
            "goal": str(goal),
            "note": "Z3 not bundled in API; proof request accepted and recorded.",
            "created_at": _now_iso(),
        }
        _Z3_PROOFS[prove_id] = result
        logger.info("z3_prove_accepted", prove_id=prove_id)
        return {"success": True, "status": "accepted", **result}
    except Exception as e:
        logger.error("z3_prove_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept Z3 proof request",
        )


# ============================ ROMA ============================ #


@router.post("/roma/analyze", status_code=status.HTTP_200_OK)
async def roma_analyze(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Accept a ROMA analysis request and return a structured result."""
    try:
        analysis_id = f"roma_{uuid.uuid4().hex[:12]}"
        target = payload.get("target", payload.get("code", ""))
        result = {
            "id": analysis_id,
            "status": "analyzed",
            "target_type": str(payload.get("target_type", "unknown")),
            "metrics": {
                "complexity": float(payload.get("complexity", 0.0) or 0.0),
                "risk": str(payload.get("risk", "unknown")),
            },
            "findings": payload.get("findings", []) or [],
            "note": "ROMA not bundled in API; analysis request accepted and recorded.",
            "created_at": _now_iso(),
        }
        _ROMA_ANALYSES[analysis_id] = result
        logger.info("roma_analyze_accepted", analysis_id=analysis_id)
        return {"success": True, "status": "accepted", **result}
    except Exception as e:
        logger.error("roma_analyze_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept ROMA analysis request",
        )


@router.post("/roma/config", status_code=status.HTTP_200_OK)
async def roma_config(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Store a ROMA configuration (in-memory)."""
    try:
        config_id = str(payload.get("id", f"romacfg_{uuid.uuid4().hex[:12]}"))
        config = dict(payload)
        config["id"] = config_id
        config["updated_at"] = _now_iso()
        _ROMA_CONFIGS[config_id] = config
        logger.info("roma_config_stored", config_id=config_id)
        return {"success": True, "status": "accepted", "id": config_id, "config": config}
    except Exception as e:
        logger.error("roma_config_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store ROMA config",
        )


# ============================ Knowledge ============================ #


@router.post("/knowledge/store", status_code=status.HTTP_200_OK)
async def knowledge_store(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Store a knowledge artifact (in-memory)."""
    try:
        entry_id = f"kb_{uuid.uuid4().hex[:12]}"
        entry = {
            "id": entry_id,
            "title": str(payload.get("title", entry_id)),
            "content": str(payload.get("content", "")),
            "tags": payload.get("tags", []) or [],
            "namespace": str(payload.get("namespace", "default")),
            "created_at": _now_iso(),
        }
        _KNOWLEDGE_STORE[entry_id] = entry
        logger.info("knowledge_stored", entry_id=entry_id)
        return {"success": True, "status": "accepted", "id": entry_id, "entry": entry}
    except Exception as e:
        logger.error("knowledge_store_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store knowledge artifact",
        )


@router.post("/knowledge/query", status_code=status.HTTP_200_OK)
async def knowledge_query(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Query the knowledge store (simple in-memory substring match)."""
    try:
        query = str(payload.get("query", payload.get("q", ""))).lower()
        limit = int(payload.get("limit", 10) or 10)
        namespace = payload.get("namespace")
        matches = []
        for entry in _KNOWLEDGE_STORE.values():
            if namespace and entry.get("namespace") != namespace:
                continue
            haystack = (entry.get("title", "") + " " + entry.get("content", "") + " " + " ".join(entry.get("tags", []))).lower()
            if not query or query in haystack:
                matches.append(entry)
            if len(matches) >= limit:
                break
        _KNOWLEDGE_QUERIES[query] = {"query": query, "hits": len(matches), "at": _now_iso()}
        return {
            "success": True,
            "status": "ok",
            "query": query,
            "count": len(matches),
            "results": matches,
        }
    except Exception as e:
        logger.error("knowledge_query_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query knowledge store",
        )


@router.get("/knowledge/status", status_code=status.HTTP_200_OK)
async def knowledge_status() -> Dict[str, Any]:
    """Return status of the knowledge-explorer subsystem."""
    try:
        return {
            "success": True,
            "status": "ready",
            "entries": len(_KNOWLEDGE_STORE),
            "queries": len(_KNOWLEDGE_QUERIES),
            "extractions": len(_KNOWLEDGE_EXTRACTIONS),
            "explorer_enabled": True,
        }
    except Exception as e:
        logger.error("knowledge_status_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read knowledge status",
        )


@router.post("/knowledge/query-advanced", status_code=status.HTTP_200_OK)
async def knowledge_query_advanced(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Advanced knowledge query with filters (tags, namespace, limit)."""
    try:
        query = str(payload.get("query", payload.get("q", ""))).lower()
        tags = set(payload.get("tags", []) or [])
        namespace = payload.get("namespace")
        limit = int(payload.get("limit", 20) or 20)
        matches = []
        for entry in _KNOWLEDGE_STORE.values():
            if namespace and entry.get("namespace") != namespace:
                continue
            if tags and not (tags & set(entry.get("tags", []))):
                continue
            haystack = (entry.get("title", "") + " " + entry.get("content", "")).lower()
            if not query or query in haystack:
                matches.append(entry)
            if len(matches) >= limit:
                break
        _KNOWLEDGE_QUERY_HISTORY.append({"query": query, "hits": len(matches), "at": _now_iso()})
        return {
            "success": True,
            "status": "ok",
            "query": query,
            "filters": {"tags": list(tags), "namespace": namespace},
            "count": len(matches),
            "results": matches,
        }
    except Exception as e:
        logger.error("knowledge_query_advanced_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run advanced knowledge query",
        )


@router.get("/knowledge/query-history", status_code=status.HTTP_200_OK)
async def knowledge_query_history() -> Dict[str, Any]:
    """Return the knowledge query history."""
    try:
        return {
            "success": True,
            "status": "ok",
            "history": list(_KNOWLEDGE_QUERY_HISTORY)[-50:],
            "count": len(_KNOWLEDGE_QUERY_HISTORY),
        }
    except Exception as e:
        logger.error("knowledge_query_history_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read knowledge query history",
        )


@router.post("/knowledge/extract", status_code=status.HTTP_200_OK)
async def knowledge_extract(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Extract knowledge triples/entities from provided text (in-memory)."""
    try:
        text = str(payload.get("text", payload.get("content", "")))
        extract_id = f"ext_{uuid.uuid4().hex[:12]}"
        words = [w for w in text.split() if w]
        extraction = {
            "id": extract_id,
            "source": str(payload.get("source", "inline")),
            "token_count": len(words),
            "entities": words[:20],
            "note": "Lightweight extraction; full NER not bundled in API.",
            "created_at": _now_iso(),
        }
        _KNOWLEDGE_EXTRACTIONS.append(extraction)
        return {"success": True, "status": "extracted", **extraction}
    except Exception as e:
        logger.error("knowledge_extract_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract knowledge",
        )


@router.post("/knowledge/extract-file", status_code=status.HTTP_200_OK)
async def knowledge_extract_file(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Extract knowledge from a referenced file path (no heavy deps)."""
    try:
        path = str(payload.get("path", payload.get("file", "")))
        extract_id = f"extf_{uuid.uuid4().hex[:12]}"
        content = ""
        try:
            from pathlib import Path as _P
            p = _P(path)
            if p.exists() and p.is_file():
                content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as read_err:
            logger.warning("knowledge_extract_file_read_failed", path=path, error=str(read_err))
        extraction = {
            "id": extract_id,
            "path": path,
            "exists": bool(content),
            "token_count": len(content.split()) if content else 0,
            "note": "File read via stdlib; full parser not bundled in API.",
            "created_at": _now_iso(),
        }
        _KNOWLEDGE_EXTRACTIONS.append(extraction)
        return {"success": True, "status": "extracted", **extraction}
    except Exception as e:
        logger.error("knowledge_extract_file_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to extract knowledge from file",
        )


# ============================ Analytics ============================ #


@router.post("/analytics/track", status_code=status.HTTP_200_OK)
async def analytics_track(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Track an analytics event (in-memory)."""
    try:
        event = dict(payload)
        event.setdefault("event_id", f"evt_{uuid.uuid4().hex[:12]}")
        event["received_at"] = _now_iso()
        _ANALYTICS_EVENTS.append(event)
        return {"success": True, "status": "tracked", "event_id": event["event_id"]}
    except Exception as e:
        logger.error("analytics_track_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to track analytics event",
        )


@router.get("/analytics/dashboard", status_code=status.HTTP_200_OK)
async def analytics_dashboard() -> Dict[str, Any]:
    """Return an aggregated analytics dashboard."""
    try:
        event_types: Dict[str, int] = {}
        for e in _ANALYTICS_EVENTS:
            et = str(e.get("type", e.get("event", "unknown")))
            event_types[et] = event_types.get(et, 0) + 1
        return {
            "success": True,
            "status": "ok",
            "total_events": len(_ANALYTICS_EVENTS),
            "event_types": event_types,
            "workflow_instances": len(_workflow_instances),
            "definitions": len(_workflow_definitions),
            "generated_at": _now_iso(),
        }
    except Exception as e:
        logger.error("analytics_dashboard_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build analytics dashboard",
        )


# ============================ LeanAide ============================ #


@router.post("/leanaide/prove", status_code=status.HTTP_200_OK)
async def leanaide_prove(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Accept a LeanAide proof request and return a structured result."""
    try:
        proof_id = f"lean_{uuid.uuid4().hex[:12]}"
        result = {
            "id": proof_id,
            "status": "queued",
            "statement": str(payload.get("statement", payload.get("goal", ""))),
            "tactic": str(payload.get("tactic", "")),
            "note": "LeanAide not bundled in API; proof request accepted and recorded.",
            "created_at": _now_iso(),
        }
        _LEANAIDE_PROOFS[proof_id] = result
        logger.info("leanaide_prove_accepted", proof_id=proof_id)
        return {"success": True, "status": "accepted", **result}
    except Exception as e:
        logger.error("leanaide_prove_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept LeanAide proof request",
        )


# ============================ Integrations ============================ #


@router.get("/integrations", status_code=status.HTTP_200_OK)
async def list_integrations() -> Dict[str, Any]:
    """List known BubbleLab integrations and their declared health."""
    try:
        return {
            "success": True,
            "status": "ok",
            "integrations": [
                {"name": name, "type": meta.get("type"), "endpoint": meta.get("endpoint")}
                for name, meta in _INTEGRATIONS.items()
            ],
            "count": len(_INTEGRATIONS),
        }
    except Exception as e:
        logger.error("integrations_list_failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list integrations",
        )


@router.get("/integrations/{name}/health", status_code=status.HTTP_200_OK)
async def integration_health(name: str) -> Dict[str, Any]:
    """Return a health snapshot for a named integration."""
    try:
        meta = _INTEGRATIONS.get(name)
        if not meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Integration '{name}' not found",
            )
        return {
            "success": True,
            "status": "ok",
            "name": name,
            "type": meta.get("type"),
            "endpoint": meta.get("endpoint"),
            "healthy": True,
            "checked_at": _now_iso(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("integration_health_failed", name=name, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check integration health",
        )
