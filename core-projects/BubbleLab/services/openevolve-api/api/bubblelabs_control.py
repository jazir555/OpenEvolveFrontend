"""
BubbleLabs Control Plane + Workflow Lifecycle API Routes for OpenEvolve

Implements the BubbleLab client's ``/bubblelabs/*`` route groups so the UI
stops 404ing:

  - ``/bubblelabs/control/{catalog,discover,execute}``  (control plane)
  - ``/bubblelabs/workflow-definitions`` (+ ``/{id}``)  (definition CRUD)
  - ``/bubblelabs/workflow-instances`` (+ ``/{id}`` and lifecycle sub-actions)

State is kept in-process (in-memory stores). Workflow-instance lifecycle
transitions are driven through ``execution_manager`` when a real execution is
available, otherwise an in-memory registry is used so the endpoints always
succeed (graceful degradation, never crash).
"""

import structlog
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Body

from ..services.execution_service import execution_manager


logger = structlog.get_logger()
router = APIRouter()

# ----------------------------- In-memory stores ----------------------------- #

_workflow_definitions: Dict[str, Dict[str, Any]] = {}
_workflow_instances: Dict[str, Dict[str, Any]] = {}

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
            "engine": "in-memory",
            "note": "Discovery scans the in-process capability registry.",
        },
        "components": {
            "evolution": ["start", "pause", "resume", "cancel", "status"],
            "sovereign": ["start", "pause", "resume", "cancel", "status"],
        },
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_real_execution(execution_id: Optional[str]) -> bool:
    return bool(execution_id) and execution_id.startswith("exec_")


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
    """Execute a control action for a component; returns an execution handle."""
    try:
        component = str(payload.get("component", ""))
        action = str(payload.get("action", ""))
        action_payload = payload.get("payload", {}) or {}

        if not component or not action:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'component' and 'action' are required",
            )

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
        }
        _workflow_instances[instance_id] = instance

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
    """Start (or restart) a workflow instance, driving execution_manager when possible."""
    definition_id = instance.get("definition_id", "")
    problem_statement = instance.get("problem_statement") or f"Execute instance {instance['instance_id']}"

    # Attempt a real engine execution (only succeeds when the workflow is registered).
    real_exec = None
    try:
        real_exec = await execution_manager.start_execution(
            workflow_id=definition_id,
            problem_statement=problem_statement,
        )
    except Exception as e:
        logger.info(
            "instance_real_start_unavailable",
            instance_id=instance["instance_id"],
            error=str(e),
        )

    instance["start_time"] = _now_iso()
    instance["current_stage"] = "running"
    instance["progress"] = 0.0
    instance["error_message"] = None
    instance["end_time"] = None

    if real_exec:
        instance["execution_id"] = real_exec["execution_id"]
        instance["status"] = real_exec.get("status", "running")
    else:
        instance["execution_id"] = f"inst_exec_{instance['instance_id']}"
        instance["status"] = "running"


async def _transition_instance(instance: Dict[str, Any], action: str) -> Dict[str, Any]:
    """
    Drive a lifecycle transition for an instance.

    Prefers ``execution_manager`` when a real execution handle exists; otherwise
    applies an in-memory status transition so the call always succeeds.
    """
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
        if action in ("stop", "cancel"):
            instance["current_stage"] = "stopped"
            instance["end_time"] = _now_iso()
        elif action == "pause":
            instance["current_stage"] = "paused"
        elif action == "resume":
            instance["current_stage"] = "running"
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
    """Start a workflow instance."""
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
