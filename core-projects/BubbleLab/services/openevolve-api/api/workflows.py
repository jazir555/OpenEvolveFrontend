"""
Workflow API Routes for OpenEvolve

CRUD operations for workflow definitions.
Follows CLAUDE.md principles: structured logging, explicit configuration, idempotency.
"""

import structlog
import json
import sqlite3
import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Query, Body

from ..models import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowListResponse,
    WorkflowStatus,
    WorkflowInputs,
    ExecutionResult,
    ExecutionStatistics,
    normalize_workflow_type,
)
from ..services.execution_service import execution_manager

# Optional bridge to the REAL openevolve library. Guarded so the service still
# starts (and every non-evolution path keeps working) when it is not installed.
try:
    from ..core.openevolve_bridge import OpenEvolveBridgeError, run_openevolve_workflow

    OPENEVOLVE_BRIDGE_AVAILABLE = True
    _OPENEVOLVE_IMPORT_ERROR: Optional[str] = None
except ImportError as _bridge_import_error:  # pragma: no cover - env dependent
    OpenEvolveBridgeError = RuntimeError  # type: ignore[assignment,misc]
    run_openevolve_workflow = None  # type: ignore[assignment]
    OPENEVOLVE_BRIDGE_AVAILABLE = False
    _OPENEVOLVE_IMPORT_ERROR = str(_bridge_import_error)


logger = structlog.get_logger()
router = APIRouter()

if not OPENEVOLVE_BRIDGE_AVAILABLE:
    logger.warning(
        "openevolve_bridge_unavailable",
        error=_OPENEVOLVE_IMPORT_ERROR,
        fallback="legacy EvolutionEngine via execution_manager",
    )

# Set OPENEVOLVE_BRIDGE_ENABLED=0 to force the legacy in-service evolution path.
OPENEVOLVE_BRIDGE_ENABLED = os.getenv("OPENEVOLVE_BRIDGE_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)

# Persistent storage using SQLite
DB_PATH = Path(os.getenv("WORKFLOW_DB_PATH", "./data/workflows.db"))
_db_connection: Optional[sqlite3.Connection] = None

# In-memory cache for active workflows
_workflows: Dict[str, WorkflowResponse] = {}
_workflow_executions: Dict[str, str] = {}

# Latest REAL openevolve engine result per workflow, surfaced through
# GET /{workflow_id}/results and the POST /{workflow_id}/start response.
_workflow_openevolve_results: Dict[str, Dict[str, Any]] = {}


def _should_use_openevolve(workflow: WorkflowResponse) -> bool:
    """True when this workflow should be driven by the real openevolve engine."""
    if not (OPENEVOLVE_BRIDGE_AVAILABLE and OPENEVOLVE_BRIDGE_ENABLED):
        return False
    return normalize_workflow_type(workflow.workflow_type) == "evolution"


def _build_bridge_request(
    workflow: WorkflowResponse,
    problem_statement: str,
    context: Optional[str],
) -> Dict[str, Any]:
    """Translate a stored workflow into the openevolve bridge request contract."""
    parameters = dict(workflow.parameters or {})
    # Never feed a previous run's result back in as an input parameter.
    parameters.pop("openevolve", None)
    return {
        "system": "evolutionary",
        "workflow_type": normalize_workflow_type(workflow.workflow_type),
        "workflow_id": workflow.id,
        "problem_statement": problem_statement,
        "context": context,
        "parameters": parameters,
        "llm": parameters.get("llm") or parameters.get("llm_config") or {},
    }


async def _run_openevolve_for_workflow(
    workflow: WorkflowResponse,
    problem_statement: str,
    context: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Run the REAL openevolve engine for an evolutionary workflow.

    Executed off the event loop. Returns the bridge result, or ``None`` when the
    bridge is unavailable/disabled or the run failed (the legacy
    execution_manager path remains the source of truth in that case).
    """
    if not _should_use_openevolve(workflow):
        return None

    bridge_request = _build_bridge_request(workflow, problem_statement, context)

    try:
        result = await asyncio.to_thread(run_openevolve_workflow, bridge_request)
    except OpenEvolveBridgeError as e:
        logger.error(
            "openevolve_bridge_run_failed",
            workflow_id=workflow.id,
            error=str(e),
            fallback="legacy EvolutionEngine result",
        )
        return None
    except Exception as e:  # pragma: no cover - defensive
        logger.error(
            "openevolve_bridge_unexpected_error",
            workflow_id=workflow.id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return None

    _workflow_openevolve_results[workflow.id] = result

    logger.info(
        "openevolve_bridge_result_attached",
        workflow_id=workflow.id,
        best_score=result.get("best_score"),
        generations=result.get("generations"),
        llm_mode=result.get("llm_mode"),
    )

    return result


def _get_db() -> sqlite3.Connection:
    """Get or create database connection."""
    global _db_connection
    if _db_connection is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db_connection = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        _db_connection.row_factory = sqlite3.Row
        _init_db()
    return _db_connection


def _init_db() -> None:
    """Initialize database tables."""
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            problem_statement TEXT,
            content_type TEXT,
            teams TEXT,  -- JSON serialized
            gauntlets TEXT,  -- JSON serialized
            metadata TEXT,  -- JSON serialized
            parameters TEXT,  -- JSON serialized
            status TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            user_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            workflow_type TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workflow_executions (
            workflow_id TEXT PRIMARY KEY,
            execution_id TEXT NOT NULL,
            FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
        )
    """)
    conn.commit()


def _workflow_to_dict(workflow: WorkflowResponse) -> Dict[str, Any]:
    """Convert WorkflowResponse to dictionary for database storage."""
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "problem_statement": workflow.problem_statement,
        "content_type": workflow.content_type,
        "teams": json.dumps(workflow.teams.dict() if workflow.teams else {}),
        "gauntlets": json.dumps(workflow.gauntlets.dict() if workflow.gauntlets else {}),
        "metadata": json.dumps(workflow.metadata.dict() if workflow.metadata else {}),
        "parameters": json.dumps(workflow.parameters or {}),
        "status": workflow.status.value,
        "created_at": workflow.created_at,
        "updated_at": workflow.updated_at,
        "started_at": workflow.started_at,
        "completed_at": workflow.completed_at,
        "user_id": workflow.user_id,
        "tenant_id": workflow.tenant_id,
        "workflow_type": workflow.workflow_type,
    }


def _load_workflows_from_db() -> None:
    """Load workflows from database into memory cache."""
    global _workflows
    try:
        conn = _get_db()
        cursor = conn.execute("SELECT * FROM workflows")
        rows = cursor.fetchall()
        
        from ..models import WorkflowTeams, WorkflowGauntlets, WorkflowMetadata
        
        for row in rows:
            row_dict = dict(row)
            # Deserialize JSON fields
            row_dict["teams"] = WorkflowTeams.parse_raw(row_dict["teams"]) if row_dict["teams"] else None
            row_dict["gauntlets"] = WorkflowGauntlets.parse_raw(row_dict["gauntlets"]) if row_dict["gauntlets"] else None
            row_dict["metadata"] = WorkflowMetadata.parse_raw(row_dict["metadata"]) if row_dict["metadata"] else None
            row_dict["parameters"] = json.loads(row_dict["parameters"]) if row_dict["parameters"] else {}
            row_dict["status"] = WorkflowStatus(row_dict["status"])
            
            workflow = WorkflowResponse(**row_dict)
            _workflows[workflow.id] = workflow
        
        # Load execution mappings
        cursor = conn.execute("SELECT * FROM workflow_executions")
        for row in cursor.fetchall():
            _workflow_executions[row["workflow_id"]] = row["execution_id"]
            
        logger.info(f"Loaded {len(_workflows)} workflows from database")
    except Exception as e:
        logger.error(f"Failed to load workflows from database: {e}")
        # Continue with empty cache


def _save_workflow_to_db(workflow: WorkflowResponse) -> None:
    """Save workflow to database."""
    conn = _get_db()
    data = _workflow_to_dict(workflow)
    conn.execute("""
        INSERT OR REPLACE INTO workflows (
            id, name, description, problem_statement, content_type,
            teams, gauntlets, metadata, parameters, status,
            created_at, updated_at, started_at, completed_at,
            user_id, tenant_id, workflow_type
        ) VALUES (
            :id, :name, :description, :problem_statement, :content_type,
            :teams, :gauntlets, :metadata, :parameters, :status,
            :created_at, :updated_at, :started_at, :completed_at,
            :user_id, :tenant_id, :workflow_type
        )
    """, data)
    conn.commit()


def _delete_workflow_from_db(workflow_id: str) -> None:
    """Delete workflow from database."""
    conn = _get_db()
    conn.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
    conn.execute("DELETE FROM workflow_executions WHERE workflow_id = ?", (workflow_id,))
    conn.commit()


def _save_execution_mapping(workflow_id: str, execution_id: str) -> None:
    """Save workflow execution mapping to database."""
    conn = _get_db()
    conn.execute(
        "INSERT OR REPLACE INTO workflow_executions (workflow_id, execution_id) VALUES (?, ?)",
        (workflow_id, execution_id)
    )
    conn.commit()


# Load workflows on module initialization
_load_workflows_from_db()


@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_201_CREATED)
async def create_workflow(workflow_data: WorkflowCreate) -> WorkflowResponse:
    """
    Create a new workflow.

    Args:
        workflow_data: Workflow creation data

    Returns:
        Created workflow with generated ID

    Raises:
        HTTPException: If workflow creation fails
    """
    try:
        # Generate unique ID
        workflow_id = f"wf_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        # Create workflow
        now = datetime.now(timezone.utc)
        workflow = WorkflowResponse(
            id=workflow_id,
            name=workflow_data.name,
            description=workflow_data.description,
            problem_statement=workflow_data.problem_statement or "",
            content_type=workflow_data.content_type,
            teams=workflow_data.teams,
            gauntlets=workflow_data.gauntlets,
            metadata=workflow_data.metadata,
            parameters=(
                workflow_data.parameters
                or (workflow_data.metadata.evolution_params if workflow_data.metadata else None)
                or {}
            ),
            status=WorkflowStatus.CREATED,
            created_at=now,
            updated_at=now,
            started_at=None,
            completed_at=None,
            user_id="anonymous",
            tenant_id="default",
            workflow_type=normalize_workflow_type(workflow_data.workflow_type),
        )

        # Store workflow in memory and database
        _workflows[workflow_id] = workflow
        _save_workflow_to_db(workflow)

        logger.info(
            "workflow_created",
            workflow_id=workflow_id,
            name=workflow_data.name,
            workflow_type=workflow_data.workflow_type,
        )

        return workflow

    except Exception as e:
        logger.error(
            "workflow_creation_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow"
        )


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status")
) -> WorkflowListResponse:
    """
    List all workflows with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        workflow_type: Optional filter by workflow type
        status_filter: Optional filter by status

    Returns:
        Paginated list of workflows
    """
    try:
        # Apply filters
        filtered_workflows = list(_workflows.values())

        normalized_workflow_type = normalize_workflow_type(workflow_type) if workflow_type else None
        if normalized_workflow_type:
            filtered_workflows = [
                w for w in filtered_workflows if w.workflow_type == normalized_workflow_type
            ]

        if status_filter:
            filtered_workflows = [w for w in filtered_workflows if w.status.value == status_filter]

        # Sort by created_at descending
        filtered_workflows.sort(key=lambda w: w.created_at, reverse=True)

        # Paginate
        total = len(filtered_workflows)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        workflows_page = filtered_workflows[start_idx:end_idx]

        logger.debug(
            "workflows_listed",
            page=page,
            page_size=page_size,
            total=total,
            returned=len(workflows_page),
            workflow_type_filter=normalized_workflow_type,
            status_filter=status_filter
        )

        return WorkflowListResponse(
            workflows=workflows_page,
            total=total,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(
            "workflow_listing_failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflows"
        )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str) -> WorkflowResponse:
    """
    Get a specific workflow by ID.

    Args:
        workflow_id: Workflow ID

    Returns:
        Workflow details

    Raises:
        HTTPException: If workflow not found
    """
    try:
        if workflow_id not in _workflows:
            logger.warning(
                "workflow_not_found",
                workflow_id=workflow_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        workflow = _workflows[workflow_id]

        logger.debug(
            "workflow_retrieved",
            workflow_id=workflow_id,
            name=workflow.name
        )

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_retrieval_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow"
        )


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    workflow_data: WorkflowUpdate
) -> WorkflowResponse:
    """
    Update an existing workflow.

    Args:
        workflow_id: Workflow ID
        workflow_data: Workflow update data

    Returns:
        Updated workflow

    Raises:
        HTTPException: If workflow not found or update fails
    """
    try:
        if workflow_id not in _workflows:
            logger.warning(
                "workflow_not_found_for_update",
                workflow_id=workflow_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        existing_workflow = _workflows[workflow_id]

        # Update fields
        update_data = workflow_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == "parameters" and value is None:
                value = {}
            setattr(existing_workflow, field, value)

        existing_workflow.updated_at = datetime.now(timezone.utc)
        
        # Save to database
        _save_workflow_to_db(existing_workflow)

        logger.info(
            "workflow_updated",
            workflow_id=workflow_id,
            updated_fields=list(update_data.keys())
        )

        return existing_workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_update_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workflow"
        )


# ==================== Execution Controls (Workflow-centric) ====================

@router.post("/{workflow_id}/start", response_model=WorkflowResponse)
async def start_workflow(
    workflow_id: str,
    inputs: Optional[WorkflowInputs] = Body(default=None)
) -> WorkflowResponse:
    """Start execution for a workflow using stored problem statement by default."""
    try:
        if workflow_id not in _workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        workflow = _workflows[workflow_id]
        problem_statement = inputs.problem_statement if inputs and inputs.problem_statement else workflow.problem_statement
        context = inputs.context if inputs else None
        if not problem_statement or not problem_statement.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Problem statement is required to start workflow execution"
            )

        execution = await execution_manager.start_execution(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
            context=context
        )

        now = datetime.now(timezone.utc)
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = now
        workflow.updated_at = now
        _workflow_executions[workflow_id] = execution["execution_id"]

        # Drive the REAL openevolve engine for evolutionary workflows. The result
        # is attached to the workflow payload (and to GET /{id}/results) without
        # changing the response model.
        openevolve_result = await _run_openevolve_for_workflow(
            workflow, problem_statement, context
        )
        if openevolve_result:
            workflow.parameters = {
                **(workflow.parameters or {}),
                "openevolve": openevolve_result,
            }
            workflow.updated_at = datetime.now(timezone.utc)

        # Save to database
        _save_workflow_to_db(workflow)
        _save_execution_mapping(workflow_id, execution["execution_id"])

        logger.info(
            "workflow_execution_started",
            workflow_id=workflow_id,
            execution_id=execution["execution_id"],
            openevolve_engine=bool(openevolve_result),
        )

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_start_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow"
        )


@router.post("/{workflow_id}/pause", response_model=WorkflowResponse)
async def pause_workflow(workflow_id: str) -> WorkflowResponse:
    """Pause a running workflow execution."""
    try:
        workflow = _workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        execution_id = _workflow_executions.get(workflow_id)
        if not execution_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active execution found for workflow"
            )

        await execution_manager.pause_execution(execution_id)
        workflow.status = WorkflowStatus.PAUSED
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Save to database
        _save_workflow_to_db(workflow)

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_pause_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause workflow"
        )


@router.post("/{workflow_id}/resume", response_model=WorkflowResponse)
async def resume_workflow(workflow_id: str) -> WorkflowResponse:
    """Resume a paused workflow execution."""
    try:
        workflow = _workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        execution_id = _workflow_executions.get(workflow_id)
        if not execution_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active execution found for workflow"
            )

        await execution_manager.resume_execution(execution_id)
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Save to database
        _save_workflow_to_db(workflow)

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_resume_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume workflow"
        )


@router.post("/{workflow_id}/stop", response_model=WorkflowResponse)
async def stop_workflow(workflow_id: str) -> WorkflowResponse:
    """Stop (cancel) a workflow execution."""
    try:
        workflow = _workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        execution_id = _workflow_executions.get(workflow_id)
        if not execution_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active execution found for workflow"
            )

        await execution_manager.cancel_execution(execution_id)
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now(timezone.utc)
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Save to database
        _save_workflow_to_db(workflow)

        return workflow

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_stop_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop workflow"
        )


@router.get("/{workflow_id}/results", response_model=ExecutionResult)
async def get_workflow_results(workflow_id: str) -> ExecutionResult:
    """Get results for the latest execution of a workflow."""
    try:
        workflow = _workflows.get(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        execution_id = _workflow_executions.get(workflow_id)
        if not execution_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No execution found for workflow"
            )

        execution = await execution_manager.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )

        return _build_execution_result(workflow, execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_results_failed",
            workflow_id=workflow_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch workflow results"
        )


def _build_representative_plan(workflow_id: str, problem_statement: str) -> dict:
    """
    Build a structured decomposition plan keyed by ``workflow_id``.

    Used by ``GET /{workflow_id}/decomposition-plan``. Reuses the decomposition
    engine when it is available; otherwise returns a representative structured
    plan so the UI always has a valid shape to render (graceful degradation).
    """
    sub_problems = [
        {
            "id": f"{workflow_id}-sp-1",
            "description": "Analyze the problem and extract requirements",
            "dependencies": [],
            "ai_suggested_evolution_mode": "evolution",
            "ai_suggested_complexity_score": 0.4,
        },
        {
            "id": f"{workflow_id}-sp-2",
            "description": "Design the solution architecture",
            "dependencies": [f"{workflow_id}-sp-1"],
            "ai_suggested_evolution_mode": "sovereign",
            "ai_suggested_complexity_score": 0.6,
        },
        {
            "id": f"{workflow_id}-sp-3",
            "description": "Implement and validate the solution",
            "dependencies": [f"{workflow_id}-sp-2"],
            "ai_suggested_evolution_mode": "adversarial",
            "ai_suggested_complexity_score": 0.7,
        },
    ]
    edges = {sp["id"]: sp["dependencies"] for sp in sub_problems}
    execution_order = [sp["id"] for sp in sub_problems]

    return {
        "workflow_id": workflow_id,
        "plan": {
            "problem_statement": problem_statement or f"Decomposition plan for {workflow_id}",
            "analyzed_context": {"domain": "general", "workflow_id": workflow_id},
            "sub_problems": sub_problems,
            "max_refinement_loops": 3,
            "auto_approval_enabled": False,
            "mdap_enabled": False,
            "maker_enabled": False,
            "parallel_processing_enabled": True,
            "max_parallel_sub_problems": 2,
            "learning_enabled": False,
            "metadata": {},
        },
        "dependency_graph": {
            "edges": edges,
            "execution_order": execution_order,
        },
    }


@router.get("/{workflow_id}/decomposition-plan", status_code=status.HTTP_200_OK)
async def get_workflow_decomposition_plan(workflow_id: str) -> dict:
    """
    Get a structured decomposition plan + dependency graph for a workflow.

    Returns the shape the BubbleLab client expects
    (``{ workflow_id, plan, dependency_graph }``). Never 404s on an unknown
    workflow id; instead returns a representative plan keyed by the id.
    """
    try:
        logger.debug("workflow_decomposition_plan_requested", workflow_id=workflow_id)

        problem_statement = ""
        workflow = _workflows.get(workflow_id)
        if workflow is not None:
            problem_statement = getattr(workflow, "problem_statement", "") or ""

        return _build_representative_plan(workflow_id, problem_statement)

    except Exception as e:
        logger.error(
            "workflow_decomposition_plan_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get decomposition plan"
        )


@router.delete("/{workflow_id}", status_code=status.HTTP_200_OK)
async def delete_workflow(workflow_id: str) -> Dict[str, str]:
    """
    Delete a workflow.

    Args:
        workflow_id: Workflow ID

    Raises:
        HTTPException: If workflow not found or deletion fails
    """
    try:
        if workflow_id not in _workflows:
            logger.warning(
                "workflow_not_found_for_deletion",
                workflow_id=workflow_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found"
            )

        # Check if workflow has active executions
        execution_id = _workflow_executions.get(workflow_id)
        if execution_id:
            try:
                execution = await execution_manager.get_execution_status(execution_id)
                if execution and execution.get("status") in ("running", "paused"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Cannot delete workflow with active execution. Stop the execution first."
                    )
            except Exception:
                # If we can't check execution status, proceed with caution
                pass

        # Delete workflow
        workflow_name = _workflows[workflow_id].name
        del _workflows[workflow_id]
        _workflow_executions.pop(workflow_id, None)
        _workflow_openevolve_results.pop(workflow_id, None)
        
        # Delete from database
        _delete_workflow_from_db(workflow_id)

        logger.info(
            "workflow_deleted",
            workflow_id=workflow_id,
            name=workflow_name
        )

        return {"message": f"Workflow '{workflow_name}' deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_deletion_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete workflow"
        )


def _build_execution_result(workflow: WorkflowResponse, execution: Dict[str, Any]) -> ExecutionResult:
    """Convert execution record to ExecutionResult response."""
    started_at = execution.get("started_at")
    completed_at = execution.get("completed_at")
    duration_seconds = None
    if started_at and completed_at:
        duration_seconds = (completed_at - started_at).total_seconds()

    raw_result = execution.get("result") or {}

    # Merge in the REAL openevolve engine result when this workflow was driven
    # by the bridge, so /results reports the actual evolved program.
    openevolve_result = _workflow_openevolve_results.get(workflow.id)
    if openevolve_result:
        if isinstance(raw_result, dict):
            raw_result = {**raw_result, "openevolve": openevolve_result}
        else:
            raw_result = {"legacy_result": raw_result, "openevolve": openevolve_result}

    try:
        final_solution = json.dumps(raw_result, indent=2)
    except (TypeError, ValueError):
        final_solution = str(raw_result)

    stats = ExecutionStatistics(
        total_duration_seconds=duration_seconds or 0.0,
        total_tokens_used=raw_result.get("tokens_used", 0) if isinstance(raw_result, dict) else 0,
        total_api_calls=raw_result.get("api_calls", 0) if isinstance(raw_result, dict) else 0,
        sub_problems_solved=raw_result.get("sub_problems_solved", 0) if isinstance(raw_result, dict) else 0,
        success_rate=1.0 if execution.get("status") == "completed" else 0.0,
        memory_used_mb=raw_result.get("memory_used_mb", 0.0) if isinstance(raw_result, dict) else 0.0,
        cpu_time_seconds=raw_result.get("cpu_time_seconds", 0.0) if isinstance(raw_result, dict) else 0.0,
    )

    return ExecutionResult(
        workflow_id=workflow.id,
        status=workflow.status,
        final_solution=final_solution,
        sub_problems=[],
        statistics=stats,
        started_at=started_at.isoformat() if started_at else None,
        completed_at=completed_at.isoformat() if completed_at else None,
        duration_seconds=duration_seconds,
    )


# ============================================================================
# WORKFLOW TEMPLATE EXECUTION ENDPOINTS
# ============================================================================

# In-memory template execution tracking
_workflow_template_executions: dict[str, dict] = {}


@router.post("/workflow-templates/{template_id}/execute", response_model=dict, status_code=status.HTTP_202_ACCEPTED)
async def execute_workflow_template(template_id: str, payload: dict) -> dict:
    """
    Execute a workflow template with the given parameters.

    This endpoint starts an asynchronous workflow template execution and returns
    an execution_id for tracking progress.
    """
    try:
        # Validate template_id
        valid_templates = [
            "research-assistant",
            "data-analysis-pipeline",
            "proof-verification",
            "knowledge-extraction",
            "problem-solving",
            "gauntlet-execution",
            "decomposition-execution",
            "gauntlet-decomposition-integrated"
        ]

        if template_id not in valid_templates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow template '{template_id}' not found. Valid templates: {', '.join(valid_templates)}"
            )

        # Create execution record
        execution_id = f"template_exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"

        execution = {
            "execution_id": execution_id,
            "template_id": template_id,
            "status": "started",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "parameters": payload.get("parameters", {}),
            "callback_url": payload.get("callback_url"),
            "current_step": None,
            "completed_steps": [],
            "results": None,
            "error": None
        }

        _workflow_template_executions[execution_id] = execution

        logger.info(
            "workflow_template_execution_started",
            execution_id=execution_id,
            template_id=template_id,
            parameters_keys=list(payload.get("parameters", {}).keys())
        )

        # In production, this would trigger an async task
        execution["status"] = "running"

        return {
            "execution_id": execution_id,
            "status": "started",
            "template_id": template_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_template_execution_start_failed",
            template_id=template_id,
            error=str(e),
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow template execution: {str(e)}"
        )


@router.get("/workflow-templates/executions/{execution_id}/status", response_model=dict)
async def get_workflow_template_execution_status(execution_id: str) -> dict:
    """Get the status of a workflow template execution."""
    try:
        if execution_id not in _workflow_template_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        execution = _workflow_template_executions[execution_id]

        logger.debug(
            "workflow_template_execution_status_retrieved",
            execution_id=execution_id,
            template_id=execution["template_id"],
            status=execution["status"]
        )

        return {
            "execution_id": execution_id,
            "status": execution["status"],
            "template_id": execution["template_id"],
            "current_step": execution.get("current_step"),
            "completed_steps": execution["completed_steps"],
            "results": execution.get("results"),
            "error": execution.get("error"),
            "created_at": execution["created_at"],
            "updated_at": execution["updated_at"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_template_execution_status_failed",
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get execution status"
        )


@router.post("/workflow-templates/executions/{execution_id}/stop", response_model=dict, status_code=status.HTTP_200_OK)
async def stop_workflow_template_execution(execution_id: str) -> dict:
    """Stop a running workflow template execution."""
    try:
        if execution_id not in _workflow_template_executions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        execution = _workflow_template_executions[execution_id]

        if execution["status"] not in ["started", "running"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Execution '{execution_id}' is not running (status: {execution['status']})"
            )

        execution["status"] = "stopped"
        execution["updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            "workflow_template_execution_stopped",
            execution_id=execution_id,
            template_id=execution["template_id"]
        )

        return {
            "status": "stopped",
            "execution_id": execution_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "workflow_template_execution_stop_failed",
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop execution"
        )
