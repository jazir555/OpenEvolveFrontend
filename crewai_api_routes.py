"""
CrewAI API Routes - Enhanced API Endpoints for CrewAI Integration

This module provides comprehensive API endpoints for CrewAI integration
that can be plugged into the main API server.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from crewai_hub import (
    execute_crewai_task,
    get_crewai_workflow_state,
    list_crewai_workflows,
    get_crewai_workflow_metrics,
    delegate_to_crewai,
    sync_crewai_delegations,
    get_crewai_status
)
from crewai_state_management import WorkflowStatus, ExecutionMethod

# Import the verification dependency from the main server
from api_server import verify_api_key  # Assuming this exists in the main server

router = APIRouter(prefix="/crewai", tags=["crewai"])


# Request/Response Models
class CrewAITaskRequest(BaseModel):
    problem_statement: str
    execution_method: str = "auto"
    agents_config: Optional[List[Dict[str, Any]]] = None
    tasks_config: Optional[List[Dict[str, Any]]] = None
    enable_learning: bool = True
    enable_zero_error: bool = True


class CrewAIDelegateRequest(BaseModel):
    task_name: str
    task_description: str
    workflow_epic_id: Optional[str] = None


class CrewAIWorkflowResponse(BaseModel):
    success: bool
    workflow_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = datetime.now().isoformat()


@router.post("/execute", dependencies=[Depends(verify_api_key)])
async def execute_crewai_task_endpoint(request: CrewAITaskRequest):
    """
    Execute a CrewAI task with the specified configuration.
    
    This endpoint allows clients to execute complex multi-agent workflows
    using CrewAI with various execution methods and configurations.
    """
    try:
        # Convert execution method string to enum
        try:
            execution_method = ExecutionMethod(request.execution_method.lower())
        except ValueError:
            execution_method = ExecutionMethod.AUTO
        
        # Execute the task
        result = await execute_crewai_task(
            problem_statement=request.problem_statement,
            execution_method=execution_method,
            agents_config=request.agents_config,
            tasks_config=request.tasks_config
        )
        
        response = CrewAIWorkflowResponse(
            success=result.get('success', True),
            workflow_id=result.get('workflow_id'),
            result=result,
            created_at=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@router.post("/delegate", dependencies=[Depends(verify_api_key)])
def delegate_task_to_crewai_endpoint(request: CrewAIDelegateRequest):
    """
    Delegate a task to the CrewAI workflow system.
    
    This endpoint allows clients to delegate tasks to be handled by the
    CrewAI workflow system with proper tracking and status management.
    """
    try:
        result = delegate_to_crewai(
            task_name=request.task_name,
            task_description=request.task_description,
            workflow_epic_id=request.workflow_epic_id
        )
        
        if result and result.get("success"):
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "Delegation failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task delegation failed: {str(e)}")


@router.get("/workflows", dependencies=[Depends(verify_api_key)])
def list_crewai_workflows_endpoint(status: Optional[str] = None):
    """
    List all CrewAI workflows, optionally filtered by status.
    
    This endpoint provides visibility into all active and historical
    CrewAI workflows in the system.
    """
    try:
        status_enum = None
        if status:
            try:
                status_enum = WorkflowStatus(status.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        workflow_ids = list_crewai_workflows(status=status_enum)
        return {
            "workflows": workflow_ids,
            "count": len(workflow_ids),
            "filter": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


@router.get("/workflows/{workflow_id}", dependencies=[Depends(verify_api_key)])
def get_crewai_workflow_endpoint(workflow_id: str):
    """
    Get the state of a specific CrewAI workflow.
    
    This endpoint provides detailed information about a specific workflow
    including its current phase, status, and execution details.
    """
    try:
        state = get_crewai_workflow_state(workflow_id)
        if not state:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "workflow_id": workflow_id,
            "state": state.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")


@router.get("/workflows/{workflow_id}/metrics", dependencies=[Depends(verify_api_key)])
def get_crewai_workflow_metrics_endpoint(workflow_id: str):
    """
    Get comprehensive metrics for a specific CrewAI workflow.
    
    This endpoint provides detailed metrics and analytics for a workflow
    including performance indicators, resource usage, and execution details.
    """
    try:
        metrics = get_crewai_workflow_metrics(workflow_id)
        if "error" in metrics and "not found" in metrics.get("error", "").lower():
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow metrics: {str(e)}")


@router.get("/workflows/{workflow_id}/tickets", dependencies=[Depends(verify_api_key)])
def get_crewai_workflow_tickets_endpoint(workflow_id: str):
    """
    Get ticket-like entries derived from a CrewAI workflow.
    
    This endpoint provides a ticket-based view of a workflow's sub-tasks
    and their current status, compatible with project management tools.
    """
    try:
        # Get the client to access the ticket functionality
        from crewai_hub import get_crewai_hub
        hub = get_crewai_hub()
        
        tickets = hub.client.get_workflow_tickets(workflow_id)
        if not tickets and not hub.get_workflow_state(workflow_id):
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "workflow_id": workflow_id,
            "tickets": tickets,
            "count": len(tickets)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow tickets: {str(e)}")


@router.post("/sync", dependencies=[Depends(verify_api_key)])
def sync_crewai_delegations_endpoint():
    """
    Sync all delegations with the CrewAI workflow system.
    
    This endpoint forces synchronization of all delegated tasks with
    the CrewAI system to ensure status consistency.
    """
    try:
        synced_count = sync_crewai_delegations()
        return {
            "synced_count": synced_count,
            "message": f"Successfully synced {synced_count} delegations"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync delegations: {str(e)}")


@router.get("/status", dependencies=[Depends(verify_api_key)])
def get_crewai_status_endpoint():
    """
    Get the status of all CrewAI components.
    
    This endpoint provides a comprehensive health check of all CrewAI
    integration components and their availability.
    """
    try:
        status = get_crewai_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/workflows/{workflow_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_crewai_workflow_endpoint(workflow_id: str):
    """
    Cancel a running CrewAI workflow.
    
    This endpoint allows for cancellation of long-running workflows
    that are no longer needed.
    """
    try:
        from crewai_hub import get_crewai_hub
        hub = get_crewai_hub()
        
        # Try to cancel the workflow
        success = hub.integration.cancel_workflow(workflow_id)
        
        if success:
            return {
                "workflow_id": workflow_id,
                "status": "cancelled",
                "message": f"Workflow {workflow_id} cancelled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found or could not be cancelled")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {str(e)}")


# Health check endpoint
@router.get("/health", include_in_schema=False)
def crewai_health_check():
    """
    Health check for CrewAI integration endpoints.
    
    This endpoint provides a lightweight health check for the CrewAI
    integration without performing heavy operations.
    """
    try:
        from crewai_hub import get_crewai_hub
        hub = get_crewai_hub()
        
        # Just check if the hub is accessible
        status = hub.get_crewai_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": len(status.get("components", {}))
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"CrewAI integration unhealthy: {str(e)}")


# Utility function to register these routes with the main app
def register_crewai_routes(app):
    """
    Register CrewAI API routes with the main FastAPI application.
    
    Args:
        app: The main FastAPI application instance
    """
    app.include_router(router)
    print("CrewAI API routes registered successfully")


# Example usage for testing
async def test_crewai_endpoints():
    """
    Test function to verify the CrewAI endpoints work correctly.
    """
    print("Testing CrewAI API Endpoints...")
    
    # Test basic status
    try:
        status = get_crewai_status()
        print(f"CrewAI Status: {status['hub']['initialized']}")
        print(f"Components: {len(status['components'])}")
    except Exception as e:
        print(f"Status check failed: {e}")
    
    # Test workflow listing
    try:
        workflows = list_crewai_workflows()
        print(f"Workflows found: {len(workflows)}")
    except Exception as e:
        print(f"Workflow listing failed: {e}")
    
    print("CrewAI endpoint testing completed.")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_crewai_endpoints())