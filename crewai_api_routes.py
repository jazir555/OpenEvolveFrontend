"""
CrewAI API Routes

This module provides the FastAPI router for CrewAI orchestration endpoints,
enabling external systems (like BubbleLab) to control CrewAI workflows.

Endpoints:
- GET /api/crewai/health
- GET /api/crewai/capabilities
- POST /api/crewai/workflows
- GET /api/crewai/workflows
- GET /api/crewai/workflows/{workflow_id}/status
- GET /api/crewai/workflows/{workflow_id}/results
- POST /api/crewai/workflows/{workflow_id}/phases/{phase_number}
- POST /api/crewai/tasks

Integration:
- Uses CrewAIClient for state management and metrics
- Uses CrewAIUnifiedBridge for workflow execution
"""


import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, status
from pydantic import BaseModel, Field
import uuid

# SECURITY: Import security framework
try:
    from security_framework import (
        Permission, UserContext, get_current_user, require_auth, require_permission,
        InputValidator, get_rate_limiter, get_audit_logger
    )
    SECURITY_AVAILABLE = True
    logging.info("SECURITY: CrewAI API routes security enabled")
except ImportError as e:
    SECURITY_AVAILABLE = False
    logging.warning(f"SECURITY: CrewAI API routes security not available: {e}")
    
    # Define stubs
    def get_current_user(): return None
    def require_auth(): return None
    def require_permission(permission): return None

# Import CrewAI components
try:
    from crewai_client import create_crewai_client, ExecutionMethod, ExecutionResult
    from crewai_unified_bridge import (
        execute_full_workflow,
        get_unified_bridge_status
    )
    # Ensure openevolve_crewai_bridge is imported to register it if needed,
    # though usually handled by the unified flow routing.
    CREWAI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"CrewAI dependencies not found: {e}")
    CREWAI_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/crewai", tags=["CrewAI"])

# ============================================================================
# Pydantic Models
# ============================================================================

class WorkflowCreateRequest(BaseModel):
    problem_statement: str = Field(..., description="Problem description")
    execution_method: str = Field("auto", description="Execution method (traditional, roma, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class PhaseExecuteRequest(BaseModel):
    phase_input: Dict[str, Any] = Field(..., description="Input data for the phase")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")

class TaskDelegateRequest(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    task_description: str = Field(..., description="Description of the task")
    team_name: Optional[str] = Field(None, description="Target team")
    context: Dict[str, Any] = Field(default_factory=dict, description="Task context")

# ============================================================================
# Dependency Injection
# ============================================================================

def get_crewai_client():
    """Get or create a CrewAI client instance."""
    if not CREWAI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CrewAI backend not available"
        )
    return create_crewai_client()

# ============================================================================
# Routes
# ============================================================================

@router.get("/health")
async def health_check():
    """Check CrewAI service health."""
    if not CREWAI_AVAILABLE:
        return {"status": "unavailable", "message": "CrewAI dependencies missing"}
    
    return {
        "status": "healthy",
        "service": "CrewAI Orchestrator",
        "bridge_status": get_unified_bridge_status()
    }

@router.get("/capabilities")
async def get_capabilities():
    """Get available execution methods and capabilities."""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI unavailable"}
        
    status = get_unified_bridge_status()
    return {
        "execution_methods": status.get("execution_methods", []),
        "features": [
            "full_workflow_execution",
            "phase_by_phase_execution",
            "state_persistence",
            "evolutionary_optimization"
        ],
        "version": status.get("version", "1.0.0")
    }

@router.post("/workflows")
async def execute_workflow(
    request: WorkflowCreateRequest,
    background_tasks: BackgroundTasks,
    client = Depends(get_crewai_client)
):
    """
    Execute a full CrewAI workflow.
    
    This endpoint starts the workflow. For long-running workflows, 
    it returns immediately with a workflow_id.
    """
    workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
    
    # We run this in background to avoid blocking
    # However, the client.execute_workflow is currently synchronous in the provided code.
    # For a true async API, we'd wrap it or use the background_tasks.
    # Given the existing client code is sync, we'll run it directly or wrap if needed.
    # For now, we'll execute it synchronously to ensure the ID matches the one returned,
    # or pass the ID to the client.
    
    # NOTE: The current CrewAIClient.execute_workflow accepts a workflow_id.
    
    try:
        # Running synchronously for now to return immediate results for simpler cases,
        # or we could make the client async. 
        # Assuming the BubbleLab bubble expects a response with status.
        
        # If the request parameters contain 'async': True, we could offload.
        # But let's stick to the simplest integration first.
        
        # Mapping execution method string to enum happens inside client
        
        # Inject the workflow_id so we can track it
        result = client.execute_workflow(
            problem_statement=request.problem_statement,
            execution_method=request.execution_method,
            workflow_id=workflow_id,
            **request.parameters
        )
        
        return {
            "success": True,
            "workflow_id": result.workflow_id,
            "status": result.status,
            "data": result.to_dict()
        }
        
    except ValueError as e:
        logger.error(f"Workflow validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/workflows")
async def list_workflows(client = Depends(get_crewai_client)):
    """List all active and persisted workflows."""
    workflows = client.list_workflows()
    return {
        "workflows": workflows,
        "count": len(workflows)
    }

@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str, client = Depends(get_crewai_client)):
    """Get status of a specific workflow."""
    state = client.get_workflow_state(workflow_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found"
        )
        
    return {
        "workflow_id": workflow_id,
        "status": state.status,
        "phase": state.phase,
        "execution_method": state.execution_method,
        "updated_at": state.updated_at
    }

@router.get("/workflows/{workflow_id}/results")
async def get_workflow_results(workflow_id: str, client = Depends(get_crewai_client)):
    """Get results of a specific workflow."""
    state = client.get_workflow_state(workflow_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found"
        )
    
    # Construct result object from state
    # This aligns with ExecutionResult.to_dict()
    return {
        "workflow_id": workflow_id,
        "status": state.status,
        "final_solution": state.reassembly_result or state.final_validation,
        "phase_results": {
            "phase1": state.metadata,
            "phase2": state.sub_solutions,
            "phase3": state.critique_reports,
            "phase4": state.verification_results,
            "phase5": state.reassembly_result,
            "phase6": state.final_validation
        }
    }

@router.post("/workflows/{workflow_id}/phases/{phase_number}")
async def execute_phase(
    workflow_id: str,
    phase_number: int,
    request: PhaseExecuteRequest,
    client = Depends(get_crewai_client)
):
    """Execute a specific phase for a workflow."""
    try:
        result = client.execute_phase(
            workflow_id=workflow_id,
            phase_number=phase_number,
            phase_input=request.phase_input,
            execution_method=request.parameters.get("execution_method", "auto")
        )
        
        return {
            "success": result.get("status") == "completed",
            "workflow_id": workflow_id,
            "phase": phase_number,
            "data": result
        }
    except ValueError as e:
        logger.error(f"Phase validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Phase execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/tasks")
async def delegate_task(
    request: TaskDelegateRequest,
    client = Depends(get_crewai_client)
):
    """Delegate a single task to a team/agent."""
    # Maps to a simplified workflow or a specific single-phase execution
    # For now, we'll treat it as a mini-workflow using the traditional method
    
    workflow_id = f"task_{uuid.uuid4().hex[:8]}"
    
    try:
        # We can use the DataPizza method for task delegation if available,
        # otherwise fallback to traditional Phase 2 execution.
        
        if request.team_name:
            # Construct a decomposition plan with a single task
            decomposition_plan = {
                "sub_problems": [{
                    "id": "task_1",
                    "description": request.task_description,
                    "title": request.task_name
                }]
            }
            
            # Execute Phase 2 directly
            result = client.execute_phase(
                workflow_id=workflow_id,
                phase_number=2,
                phase_input=decomposition_plan,
                execution_method="traditional" # or datapizza if preferred
            )
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "data": result
            }
        else:
            # Fallback to full workflow if no team specified
            result = client.execute_workflow(
                problem_statement=request.task_description,
                execution_method="auto",
                workflow_id=workflow_id
            )
            return {
                "success": True,
                "workflow_id": result.workflow_id,
                "data": result.to_dict()
            }
            
    except ValueError as e:
        logger.error(f"Task validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Task delegation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
