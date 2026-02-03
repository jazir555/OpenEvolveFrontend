"""
Execution API Routes for OpenEvolve

Workflow execution management: start, pause, resume, cancel, status.
Follows CLAUDE.md principles: structured logging, background tasks, failure isolation.
"""

import structlog
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse

from ..models import (
    WorkflowInputs,
    ExecutionResponse,
    ExecutionStatusResponse,
    ExecutionLogsResponse,
    ExecutionStartRequest,
)
from ..services.execution_service import execution_manager


logger = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=ExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_execution_direct(request: ExecutionStartRequest) -> ExecutionResponse:
    """
    Start execution of a workflow (direct execution endpoint).

    Args:
        request: Execution start request containing workflow_id and inputs

    Returns:
        Execution response with execution_id and initial status
    """
    try:
        logger.info(
            "execution_direct_start_requested",
            workflow_id=request.workflow_id,
            problem_statement_length=len(request.problem_statement or ""),
        )

        # If problem statement missing, attempt to pull from stored workflow
        problem_statement = request.problem_statement
        if not problem_statement:
            from .workflows import _workflows
            workflow = _workflows.get(request.workflow_id)
            if not workflow:
                raise ValueError(f"Workflow '{request.workflow_id}' not found")
            problem_statement = workflow.problem_statement

        if not problem_statement or not problem_statement.strip():
            raise ValueError("Problem statement is required to start execution")

        execution = await execution_manager.start_execution(
            workflow_id=request.workflow_id,
            problem_statement=problem_statement,
            context=request.context,
        )

        return ExecutionResponse(**execution)

    except ValueError as e:
        logger.error(
            "execution_direct_start_validation_failed",
            workflow_id=request.workflow_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "execution_direct_start_failed",
            workflow_id=request.workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow execution",
        )

@router.post("/workflows/{workflow_id}/execute", response_model=ExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_execution(workflow_id: str, inputs: WorkflowInputs) -> ExecutionResponse:
    """
    Start execution of a workflow.

    Args:
        workflow_id: Workflow ID to execute
        inputs: Workflow execution inputs

    Returns:
        Execution response with execution_id and initial status

    Raises:
        HTTPException: If workflow not found or execution fails to start
    """
    try:
        logger.info(
            "workflow_execution_requested",
            workflow_id=workflow_id,
            problem_statement_length=len(inputs.problem_statement),
            context_provided=inputs.context is not None
        )

        # Determine problem statement (use stored workflow if omitted)
        problem_statement = inputs.problem_statement
        if not problem_statement:
            from .workflows import _workflows
            workflow = _workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow '{workflow_id}' not found")
            problem_statement = workflow.problem_statement

        # Start execution through execution manager
        execution = await execution_manager.start_execution(
            workflow_id=workflow_id,
            problem_statement=problem_statement,
            context=inputs.context
        )

        logger.info(
            "workflow_execution_started",
            workflow_id=workflow_id,
            execution_id=execution["execution_id"],
            status=execution["status"]
        )

        return ExecutionResponse(**execution)

    except ValueError as e:
        logger.error(
            "workflow_execution_start_validation_failed",
            workflow_id=workflow_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "workflow_execution_start_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start workflow execution"
        )


@router.get("/workflows/{workflow_id}/executions/{execution_id}", response_model=ExecutionStatusResponse)
async def get_execution_status(
    workflow_id: str,
    execution_id: str
) -> ExecutionStatusResponse:
    """
    Get status of a workflow execution.

    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID

    Returns:
        Current execution status

    Raises:
        HTTPException: If execution not found
    """
    try:
        logger.debug(
            "execution_status_requested",
            workflow_id=workflow_id,
            execution_id=execution_id
        )

        # Get execution status from manager
        execution = await execution_manager.get_execution_status(execution_id)

        if not execution:
            logger.warning(
                "execution_not_found",
                workflow_id=workflow_id,
                execution_id=execution_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        logger.debug(
            "execution_status_retrieved",
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=execution["status"]
        )

        return ExecutionStatusResponse(**execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_status_retrieval_failed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution status"
        )


@router.get("/{execution_id}", response_model=ExecutionStatusResponse)
async def get_execution_status_direct(execution_id: str) -> ExecutionStatusResponse:
    """Get execution status by execution ID (direct endpoint)."""
    try:
        execution = await execution_manager.get_execution_status(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )
        return ExecutionStatusResponse(**execution)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_status_direct_failed",
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution status"
        )


@router.post("/workflows/{workflow_id}/executions/{execution_id}/pause", status_code=status.HTTP_200_OK)
async def pause_execution(
    workflow_id: str,
    execution_id: str
) -> ExecutionStatusResponse:
    """
    Pause a running workflow execution.

    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID

    Returns:
        Updated execution status

    Raises:
        HTTPException: If execution not found or cannot be paused
    """
    try:
        logger.info(
            "execution_pause_requested",
            workflow_id=workflow_id,
            execution_id=execution_id
        )

        # Pause execution through manager
        execution = await execution_manager.pause_execution(execution_id)

        if not execution:
            logger.warning(
                "execution_not_found_for_pause",
                workflow_id=workflow_id,
                execution_id=execution_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        logger.info(
            "execution_paused",
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=execution["status"]
        )

        return ExecutionStatusResponse(**execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_pause_failed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause execution"
        )


@router.post("/{execution_id}/pause", status_code=status.HTTP_200_OK)
async def pause_execution_direct(execution_id: str) -> ExecutionStatusResponse:
    """Pause execution by execution ID (direct endpoint)."""
    try:
        execution = await execution_manager.pause_execution(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )
        return ExecutionStatusResponse(**execution)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_pause_direct_failed",
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause execution"
        )


@router.post("/workflows/{workflow_id}/executions/{execution_id}/resume", status_code=status.HTTP_200_OK)
async def resume_execution(
    workflow_id: str,
    execution_id: str
) -> ExecutionStatusResponse:
    """
    Resume a paused workflow execution.

    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID

    Returns:
        Updated execution status

    Raises:
        HTTPException: If execution not found or cannot be resumed
    """
    try:
        logger.info(
            "execution_resume_requested",
            workflow_id=workflow_id,
            execution_id=execution_id
        )

        # Resume execution through manager
        execution = await execution_manager.resume_execution(execution_id)

        if not execution:
            logger.warning(
                "execution_not_found_for_resume",
                workflow_id=workflow_id,
                execution_id=execution_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        logger.info(
            "execution_resumed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=execution["status"]
        )

        return ExecutionStatusResponse(**execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_resume_failed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume execution"
        )


@router.post("/{execution_id}/resume", status_code=status.HTTP_200_OK)
async def resume_execution_direct(execution_id: str) -> ExecutionStatusResponse:
    """Resume execution by execution ID (direct endpoint)."""
    try:
        execution = await execution_manager.resume_execution(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )
        return ExecutionStatusResponse(**execution)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_resume_direct_failed",
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resume execution"
        )


@router.post("/workflows/{workflow_id}/executions/{execution_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_execution(
    workflow_id: str,
    execution_id: str
) -> ExecutionStatusResponse:
    """
    Cancel a workflow execution.

    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID

    Returns:
        Updated execution status

    Raises:
        HTTPException: If execution not found or cannot be cancelled
    """
    try:
        logger.info(
            "execution_cancel_requested",
            workflow_id=workflow_id,
            execution_id=execution_id
        )

        # Cancel execution through manager
        execution = await execution_manager.cancel_execution(execution_id)

        if not execution:
            logger.warning(
                "execution_not_found_for_cancel",
                workflow_id=workflow_id,
                execution_id=execution_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        logger.info(
            "execution_cancelled",
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=execution["status"]
        )

        return ExecutionStatusResponse(**execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_cancel_failed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel execution"
        )


@router.post("/{execution_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_execution_direct(execution_id: str) -> ExecutionStatusResponse:
    """Cancel execution by execution ID (direct endpoint)."""
    try:
        execution = await execution_manager.cancel_execution(execution_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )
        return ExecutionStatusResponse(**execution)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_cancel_direct_failed",
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel execution"
        )


@router.get("/workflows/{workflow_id}/executions/{execution_id}/logs", response_model=ExecutionLogsResponse)
async def get_execution_logs(
    workflow_id: str,
    execution_id: str,
    since: Optional[datetime] = Query(None, description="Filter logs since this timestamp")
) -> ExecutionLogsResponse:
    """
    Get logs for a workflow execution.

    Args:
        workflow_id: Workflow ID
        execution_id: Execution ID
        since: Optional datetime filter

    Returns:
        Execution logs

    Raises:
        HTTPException: If execution not found
    """
    try:
        logger.debug(
            "execution_logs_requested",
            workflow_id=workflow_id,
            execution_id=execution_id,
            since=since.isoformat() if since else None
        )

        # Get logs from execution manager
        logs_response = await execution_manager.get_execution_logs(execution_id, since)

        if not logs_response:
            logger.warning(
                "execution_not_found_for_logs",
                workflow_id=workflow_id,
                execution_id=execution_id
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )

        logger.debug(
            "execution_logs_retrieved",
            workflow_id=workflow_id,
            execution_id=execution_id,
            logs_count=logs_response["total"]
        )

        return ExecutionLogsResponse(**logs_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_logs_retrieval_failed",
            workflow_id=workflow_id,
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution logs"
        )


@router.get("/{execution_id}/logs", response_model=ExecutionLogsResponse)
async def get_execution_logs_direct(
    execution_id: str,
    since: Optional[datetime] = Query(None, description="Filter logs since this timestamp")
) -> ExecutionLogsResponse:
    """Get logs for an execution by execution ID (direct endpoint)."""
    try:
        logs_response = await execution_manager.get_execution_logs(execution_id, since)
        if not logs_response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution '{execution_id}' not found"
            )
        return ExecutionLogsResponse(**logs_response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "execution_logs_direct_failed",
            execution_id=execution_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution logs"
        )


@router.get("/workflows/{workflow_id}/executions", response_model=list[ExecutionStatusResponse])
async def list_workflow_executions(
    workflow_id: str,
    limit: int = Query(10, ge=1, le=100, description="Maximum number of executions to return")
) -> list[ExecutionStatusResponse]:
    """
    List all executions for a workflow.

    Args:
        workflow_id: Workflow ID
        limit: Maximum number of executions to return

    Returns:
        List of execution statuses

    Raises:
        HTTPException: If workflow not found
    """
    try:
        logger.debug(
            "workflow_executions_list_requested",
            workflow_id=workflow_id,
            limit=limit
        )

        # Get executions from manager
        executions = await execution_manager.list_workflow_executions(workflow_id, limit)

        logger.debug(
            "workflow_executions_listed",
            workflow_id=workflow_id,
            count=len(executions)
        )

        return [ExecutionStatusResponse(**exec) for exec in executions]

    except Exception as e:
        logger.error(
            "workflow_executions_list_failed",
            workflow_id=workflow_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow executions"
        )
