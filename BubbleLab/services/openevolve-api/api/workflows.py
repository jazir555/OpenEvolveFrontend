"""
Workflow API Routes for OpenEvolve

CRUD operations for workflow definitions.
Follows CLAUDE.md principles: structured logging, explicit configuration, idempotency.
"""

import structlog
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone
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
)
from ..services.execution_service import execution_manager


logger = structlog.get_logger()
router = APIRouter()

# In-memory storage (TODO: Replace with persistent storage)
_workflows: dict[str, WorkflowResponse] = {}
_workflow_executions: Dict[str, str] = {}


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
            workflow_type=workflow_data.workflow_type,
        )

        # Store workflow
        _workflows[workflow_id] = workflow

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

        if workflow_type:
            filtered_workflows = [w for w in filtered_workflows if w.workflow_type == workflow_type]

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
            workflow_type_filter=workflow_type,
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

        logger.info(
            "workflow_execution_started",
            workflow_id=workflow_id,
            execution_id=execution["execution_id"]
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
        # TODO: Add execution check when execution service is integrated

        # Delete workflow
        workflow_name = _workflows[workflow_id].name
        del _workflows[workflow_id]
        _workflow_executions.pop(workflow_id, None)

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
    try:
        final_solution = json.dumps(raw_result, indent=2)
    except Exception:
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


