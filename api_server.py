"""
REST API Server for External System Integration

This module provides a REST API for external systems to interact with the
Decomposition Workflow system.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
import uvicorn
import os
import re
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)

# Import environment helpers
from env_helpers import is_production

from workflow_structures import (
    DecompositionPlan, SubProblem, Team, GauntletDefinition, GauntletRoundRule,
    WorkflowState, ModelConfig
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow


# Initialize FastAPI app
app = FastAPI(
    title="Decomposition Workflow API",
    description="""
    REST API for the Sovereign-Grade Decomposition Workflow system.
    
    ## Features
    
    * **Workflow Management**: Create, monitor, pause, resume, and retrieve workflow results
    * **Team Management**: Configure AI teams for different roles (Blue, Red, Gold)
    * **Gauntlet Management**: Define evaluation gauntlets with programmable rules
    * **Webhooks**: Subscribe to workflow events for real-time notifications
    * **Authentication**: API key and JWT token-based authentication with RBAC
    
    ## Authentication
    
    Use one of the following methods:
    
    1. **API Key**: Include `X-API-Key` header with your API key
    2. **JWT Token**: Get a token from `/auth/token` and include `Authorization: Bearer <token>` header
    
    ## Roles
    
    * **ADMIN**: Full access to all endpoints
    * **USER**: Can create and manage workflows, teams, and gauntlets
    * **READONLY**: Can only view resources
    """,
    version="1.0.0",
    contact={
        "name": "Decomposition Workflow Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )

# Initialize default managers (legacy fallback, use tenant-scoped managers in handlers)
team_manager = TeamManager()
gauntlet_manager = GauntletManager()

# In-memory storage for workflows (replace with database in production)
workflows: Dict[str, WorkflowState] = {}

# In-memory audit log (replace with persistent storage in production)
AUDIT_LOGS: List[Dict[str, Any]] = []


def record_audit_event(
    user: "AuthUser",
    operation: str,
    resource: str,
    resource_id: str,
    success: bool,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Record an audit event for workflow lifecycle actions."""
    AUDIT_LOGS.append({
        "timestamp": datetime.now().isoformat(),
        "user": user.name,
        "role": user.role,
        "operation": operation,
        "resource": resource,
        "resource_id": resource_id,
        "success": success,
        "details": details or {}
    })


def _normalize_tenant_id(tenant_id: str) -> str:
    """Normalize tenant ID for safe filesystem usage."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", tenant_id.strip())
    return normalized or "default"


def get_tenant_id(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Get tenant ID from request headers (defaults to 'default')."""
    if not x_tenant_id:
        return "default"
    return _normalize_tenant_id(x_tenant_id)


def _get_tenant_storage_dir(tenant_id: str) -> str:
    """Get or create the tenant storage directory."""
    base_dir = os.path.join("data", "tenants", tenant_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_tenant_team_manager(tenant_id: str) -> TeamManager:
    """Get a tenant-scoped TeamManager."""
    base_dir = _get_tenant_storage_dir(tenant_id)
    return TeamManager(teams_file=os.path.join(base_dir, "teams.json"))


def get_tenant_gauntlet_manager(tenant_id: str) -> GauntletManager:
    """Get a tenant-scoped GauntletManager."""
    base_dir = _get_tenant_storage_dir(tenant_id)
    return GauntletManager(gauntlets_file=os.path.join(base_dir, "gauntlets.json"))


# Pydantic models for API requests/responses
class WorkflowCreateRequest(BaseModel):
    problem_statement: str = Field(..., min_length=10, description="The problem to solve")
    content_analyzer_team: str = Field(..., description="Team for content analysis")
    planner_team: str = Field(..., description="Team for planning")
    solver_team: str = Field(..., description="Team for solving sub-problems")
    patcher_team: str = Field(..., description="Team for patching solutions")
    assembler_team: str = Field(..., description="Team for assembling final solution")
    sub_problem_red_gauntlet: str = Field(..., description="Red gauntlet for sub-problems")
    sub_problem_gold_gauntlet: str = Field(..., description="Gold gauntlet for sub-problems")
    final_red_gauntlet: str = Field(..., description="Red gauntlet for final solution")
    final_gold_gauntlet: str = Field(..., description="Gold gauntlet for final solution")
    solver_generation_gauntlet: str = Field(..., description="Gauntlet for solution generation")
    max_refinement_loops: int = Field(3, ge=1, le=10, description="Maximum refinement loops")
    mdap_enabled: bool = Field(False, description="Enable MDAP for solution generation")
    mdap_config: Dict[str, Any] = Field(default_factory=dict, description="MDAP configuration overrides")
    maker_enabled: bool = Field(False, description="Enable MAKER for solution generation")
    maker_config: Dict[str, Any] = Field(default_factory=dict, description="MAKER configuration overrides")
    
    @validator('problem_statement')
    def validate_problem_statement(cls, v):
        if not v or not v.strip():
            raise ValueError('Problem statement cannot be empty')
        return v.strip()


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    current_stage: str
    progress: float
    created_at: str


class WorkflowDetailResponse(BaseModel):
    workflow_id: str
    problem_statement: str
    status: str
    current_stage: str
    progress: float
    start_time: float
    end_time: Optional[float]
    refinement_loop_count: int
    solved_sub_problems: int
    total_sub_problems: int


class TeamCreateRequest(BaseModel):
    name: str
    role: str
    description: Optional[str] = None
    members: List[Dict[str, Any]]


class GauntletCreateRequest(BaseModel):
    name: str
    team_name: str
    description: Optional[str] = None
    rounds: List[Dict[str, Any]]


# Authentication and Authorization
from enum import Enum
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import timedelta


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


# API Keys - Load from environment, NO HARDCODED KEYS
# In production, these should be stored in a database with proper encryption
def _load_api_keys() -> Dict[str, Dict[str, Any]]:
    """
    Load API keys from environment variables.

    Format: API_KEY_<name>=<key>:<role>
    Example: API_KEY_ADMIN=sk-admin123:admin

    Returns:
        Dictionary mapping API keys to user info
    """
    api_keys = {}
    prefix = "API_KEY_"

    for env_var, value in os.environ.items():
        if env_var.startswith(prefix):
            try:
                # Parse format: key:role
                if ":" in value:
                    key, role = value.split(":", 1)
                    name = env_var[len(prefix):].lower().replace("_", " ")
                    api_keys[key] = {"role": UserRole(role), "name": name}
                else:
                    logger.warning(f"Invalid API key format in {env_var}. Expected 'key:role'")
            except ValueError as e:
                logger.warning(f"Failed to parse API key from {env_var}: {e}")

    return api_keys


API_KEYS = _load_api_keys()

# JWT Configuration - MUST be set from environment
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if is_production():
        raise RuntimeError(
            "SECRET_KEY environment variable must be set in production. "
            "Generate a secure key with: python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    else:
        # Generate a temporary key for development only
        import secrets
        SECRET_KEY = secrets.token_hex(32)
        logger.warning(
            "Using auto-generated SECRET_KEY for development. "
            "This will change on restart! Set SECRET_KEY environment variable for persistence."
        )

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthUser(BaseModel):
    """Authenticated user information."""
    api_key: str
    role: UserRole
    name: str


def verify_api_key(x_api_key: str = Header(...)) -> AuthUser:
    """Verify API key from header and return user info."""
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    key_info = API_KEYS[x_api_key]
    return AuthUser(
        api_key=x_api_key,
        role=key_info["role"],
        name=key_info["name"]
    )


def require_role(required_role: UserRole):
    """Dependency to require specific role."""
    def role_checker(user: AuthUser = Depends(verify_api_key)) -> AuthUser:
        # Admin can do everything
        if user.role == UserRole.ADMIN:
            return user
        
        # Check if user has required role
        role_hierarchy = {
            UserRole.ADMIN: 3,
            UserRole.USER: 2,
            UserRole.READONLY: 1
        }
        
        if role_hierarchy.get(user.role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return user
    
    return role_checker


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str = Header(..., alias="Authorization")) -> dict:
    """Verify JWT token."""
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


# API Endpoints

@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Decomposition Workflow API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


class TokenRequest(BaseModel):
    """Request for JWT token."""
    api_key: str = Field(..., description="API key for authentication")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: str


@app.post("/auth/token", response_model=TokenResponse)
def get_token(request: TokenRequest):
    """Get JWT token using API key."""
    if request.api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    key_info = API_KEYS[request.api_key]
    
    # Create token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": key_info["name"], "role": key_info["role"]},
        expires_delta=access_token_expires
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=key_info["role"]
    )


# Workflow endpoints

@app.post(
    "/workflows",
    response_model=WorkflowResponse,
    dependencies=[Depends(require_role(UserRole.USER))],
    summary="Create a new workflow",
    description="Create a new decomposition workflow with specified teams and gauntlets",
    responses={
        200: {
            "description": "Workflow created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "created",
                        "current_stage": "INITIALIZING",
                        "progress": 0.0,
                        "created_at": "2025-10-21T12:00:00"
                    }
                }
            }
        },
        400: {"description": "Invalid request or missing teams/gauntlets"},
        401: {"description": "Invalid API key"},
        500: {"description": "Internal server error"}
    }
)
def create_workflow(
    request: WorkflowCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new workflow."""
    logger.info(f"Workflow creation request by {user.name} for problem: {request.problem_statement[:50]}...")
    try:
        tenant_team_manager = get_tenant_team_manager(tenant_id)
        tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)

        # Get teams and gauntlets
        content_analyzer_team = tenant_team_manager.get_team(request.content_analyzer_team)
        planner_team = tenant_team_manager.get_team(request.planner_team)
        solver_team = tenant_team_manager.get_team(request.solver_team)
        patcher_team = tenant_team_manager.get_team(request.patcher_team)
        assembler_team = tenant_team_manager.get_team(request.assembler_team)
        
        sub_problem_red_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.sub_problem_red_gauntlet)
        sub_problem_gold_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.sub_problem_gold_gauntlet)
        final_red_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.final_red_gauntlet)
        final_gold_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.final_gold_gauntlet)
        solver_generation_gauntlet = tenant_gauntlet_manager.get_gauntlet(request.solver_generation_gauntlet)
        
        # Validate all exist
        if not all([
            content_analyzer_team, planner_team, solver_team, patcher_team, assembler_team,
            sub_problem_red_gauntlet, sub_problem_gold_gauntlet,
            final_red_gauntlet, final_gold_gauntlet, solver_generation_gauntlet
        ]):
            raise HTTPException(status_code=400, detail="One or more teams/gauntlets not found")
        
        # Create workflow state
        workflow_id = str(uuid.uuid4())
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type="sovereign_decomposition",
            problem_statement=request.problem_statement,
            current_stage="INITIALIZING",
            status="created",
            tenant_id=tenant_id,
            mdap_enabled=request.mdap_enabled,
            mdap_config=request.mdap_config,
            maker_enabled=request.maker_enabled,
            maker_config=request.maker_config
        )
        
        # Store workflow
        workflows[workflow_id] = workflow_state

        record_audit_event(
            user=user,
            operation="CREATE_WORKFLOW",
            resource="workflow",
            resource_id=workflow_id,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            status=workflow_state.status,
            current_stage=workflow_state.current_stage,
            progress=workflow_state.progress,
            created_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/workflows", dependencies=[Depends(verify_api_key)])
def list_workflows(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all workflows."""
    logger.info(f"User {user.name} listed workflows.")
    record_audit_event(
        user=user,
        operation="LIST_WORKFLOWS",
        resource="workflow",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "workflows": [
            {
                "workflow_id": wf.workflow_id,
                "status": wf.status,
                "current_stage": wf.current_stage,
                "progress": wf.progress
            }
            for wf in workflows.values()
            if (wf.tenant_id or "default") == tenant_id
        ],
        "total": len([wf for wf in workflows.values() if (wf.tenant_id or "default") == tenant_id])
    }


@app.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse, dependencies=[Depends(verify_api_key)])
def get_workflow(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get workflow details."""
    logger.info(f"User {user.name} requested details for workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")

    record_audit_event(
        user=user,
        operation="GET_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    total_sub_problems = len(wf.decomposition_plan.sub_problems) if wf.decomposition_plan else 0
    solved_sub_problems = len(wf.solved_sub_problem_ids)
    
    return WorkflowDetailResponse(
        workflow_id=wf.workflow_id,
        problem_statement=wf.problem_statement,
        status=wf.status,
        current_stage=wf.current_stage,
        progress=wf.progress,
        start_time=wf.start_time,
        end_time=wf.end_time,
        refinement_loop_count=wf.refinement_loop_count,
        solved_sub_problems=solved_sub_problems,
        total_sub_problems=total_sub_problems
    )


@app.post("/workflows/{workflow_id}/pause", dependencies=[Depends(require_role(UserRole.USER))])
def pause_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Pause a running workflow."""
    logger.info(f"User {user.name} requested to pause workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "running":
        raise HTTPException(status_code=400, detail=f"Cannot pause workflow in status: {wf.status}")
    
    wf.status = "paused"

    record_audit_event(
        user=user,
        operation="PAUSE_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"status": wf.status, "tenant_id": tenant_id}
    )
    
    return {
        "message": "Workflow paused",
        "workflow_id": workflow_id,
        "status": wf.status
    }


@app.post("/workflows/{workflow_id}/resume", dependencies=[Depends(require_role(UserRole.USER))])
def resume_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Resume a paused workflow."""
    logger.info(f"User {user.name} requested to resume workflow {workflow_id}.")
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "paused":
        raise HTTPException(status_code=400, detail=f"Cannot resume workflow in status: {wf.status}")
    
    wf.status = "running"

    record_audit_event(
        user=user,
        operation="RESUME_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"status": wf.status, "tenant_id": tenant_id}
    )
    
    return {
        "message": "Workflow resumed",
        "workflow_id": workflow_id,
        "status": wf.status
    }


@app.get("/workflows/{workflow_id}/results", dependencies=[Depends(verify_api_key)])
def get_workflow_results(
    workflow_id: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get workflow results."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    
    if wf.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow not completed yet. Current status: {wf.status}"
        )
    
    # Prepare results
    results = {
        "workflow_id": wf.workflow_id,
        "problem_statement": wf.problem_statement,
        "status": wf.status,
        "final_solution": None,
        "sub_problem_solutions": {},
        "execution_time": None,
        "refinement_loops": wf.refinement_loop_count
    }
    
    # Add final solution if available
    if wf.final_solution:
        results["final_solution"] = {
            "content": wf.final_solution.content,
            "generated_by": wf.final_solution.generated_by_model,
            "timestamp": wf.final_solution.timestamp
        }
    
    # Add sub-problem solutions
    for sp_id, solution in wf.sub_problem_solutions.items():
        results["sub_problem_solutions"][sp_id] = {
            "content": solution.content,
            "generated_by": solution.generated_by_model,
            "timestamp": solution.timestamp
        }
    
    # Calculate execution time
    if wf.start_time and wf.end_time:
        results["execution_time"] = wf.end_time - wf.start_time
    
    record_audit_event(
        user=user,
        operation="GET_WORKFLOW_RESULTS",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return results


@app.delete("/workflows/{workflow_id}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_workflow(
    workflow_id: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Cancel and delete a workflow."""
    if workflow_id not in workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    wf = workflows[workflow_id]
    if (wf.tenant_id or "default") != tenant_id:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # If running, mark as cancelled
    if wf.status == "running":
        wf.status = "cancelled"
    
    del workflows[workflow_id]

    record_audit_event(
        user=user,
        operation="DELETE_WORKFLOW",
        resource="workflow",
        resource_id=workflow_id,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {"message": "Workflow deleted", "workflow_id": workflow_id}


# Team endpoints

@app.get("/teams", dependencies=[Depends(verify_api_key)])
def list_teams(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all teams."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    teams = tenant_team_manager.get_all_teams()
    record_audit_event(
        user=user,
        operation="LIST_TEAMS",
        resource="team",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "teams": [
            {
                "name": team.name,
                "role": team.role,
                "description": team.description,
                "member_count": len(team.members)
            }
            for team in teams
        ],
        "total": len(teams)
    }


@app.get("/teams/{team_name}", dependencies=[Depends(verify_api_key)])
def get_team(
    team_name: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get team details."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    team = tenant_team_manager.get_team(team_name)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    record_audit_event(
        user=user,
        operation="GET_TEAM",
        resource="team",
        resource_id=team_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "name": team.name,
        "role": team.role,
        "description": team.description,
        "members": [
            {
                "model_id": m.model_id,
                "temperature": m.temperature,
                "max_tokens": m.max_tokens
            }
            for m in team.members
        ]
    }


@app.post("/teams", dependencies=[Depends(require_role(UserRole.USER))])
def create_team(
    request: TeamCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new team (requires USER role)."""
    try:
        # Convert members to ModelConfig objects
        members = [ModelConfig(**member) for member in request.members]
        
        team = Team(
            name=request.name,
            tenant_id=tenant_id,
            role=request.role,
            description=request.description,
            members=members
        )
        
        tenant_team_manager = get_tenant_team_manager(tenant_id)
        tenant_team_manager.save_team(team)
        
        logger.info(f"Team '{team.name}' created by {user.name}")
        record_audit_event(
            user=user,
            operation="CREATE_TEAM",
            resource="team",
            resource_id=team.name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Team created", "team_name": team.name}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/teams/{team_name}", dependencies=[Depends(require_role(UserRole.USER))])
def update_team(
    team_name: str,
    request: TeamCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Update an existing team (requires USER role)."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    existing_team = tenant_team_manager.get_team(team_name)
    if not existing_team:
        raise HTTPException(status_code=404, detail="Team not found")

    try:
        # Convert members to ModelConfig objects
        members = [ModelConfig(**member) for member in request.members]
        
        updated_team = Team(
            name=team_name, # Ensure name from path is used
            tenant_id=tenant_id,
            role=request.role,
            description=request.description,
            members=members
        )
        
        tenant_team_manager.save_team(updated_team) # Overwrite existing
        
        logger.info(f"Team '{team_name}' updated by {user.name}")
        record_audit_event(
            user=user,
            operation="UPDATE_TEAM",
            resource="team",
            resource_id=team_name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Team updated", "team_name": team_name}
    
    except Exception as e:
        logger.error(f"Error updating team '{team_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/teams/{team_name}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_team(
    team_name: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a team (requires ADMIN role)."""
    tenant_team_manager = get_tenant_team_manager(tenant_id)
    if not tenant_team_manager.get_team(team_name):
        raise HTTPException(status_code=404, detail="Team not found")
    
    tenant_team_manager.delete_team(team_name)
    logger.info(f"Team '{team_name}' deleted by {user.name}")
    record_audit_event(
        user=user,
        operation="DELETE_TEAM",
        resource="team",
        resource_id=team_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    return {"message": "Team deleted", "team_name": team_name}


# Gauntlet endpoints

@app.get("/gauntlets", dependencies=[Depends(verify_api_key)])
def list_gauntlets(
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """List all gauntlets."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    gauntlets = tenant_gauntlet_manager.get_all_gauntlets()
    record_audit_event(
        user=user,
        operation="LIST_GAUNTLETS",
        resource="gauntlet",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "gauntlets": [
            {
                "name": g.name,
                "team_name": g.team_name,
                "description": g.description,
                "round_count": len(g.rounds)
            }
            for g in gauntlets
        ],
        "total": len(gauntlets)
    }


@app.get("/gauntlets/{gauntlet_name}", dependencies=[Depends(verify_api_key)])
def get_gauntlet(
    gauntlet_name: str,
    user: AuthUser = Depends(verify_api_key),
    tenant_id: str = Depends(get_tenant_id)
):
    """Get gauntlet details."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    gauntlet = tenant_gauntlet_manager.get_gauntlet(gauntlet_name)
    if not gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    record_audit_event(
        user=user,
        operation="GET_GAUNTLET",
        resource="gauntlet",
        resource_id=gauntlet_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "name": gauntlet.name,
        "team_name": gauntlet.team_name,
        "description": gauntlet.description,
        "rounds": [
            {
                "round_number": r.round_number,
                "quorum_required_approvals": r.quorum_required_approvals,
                "quorum_from_panel_size": r.quorum_from_panel_size,
                "min_overall_confidence": r.min_overall_confidence
            }
            for r in gauntlet.rounds
        ]
    }


@app.post("/gauntlets", dependencies=[Depends(require_role(UserRole.USER))])
def create_gauntlet(
    request: GauntletCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new gauntlet (requires USER role)."""
    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        gauntlet = GauntletDefinition(
            name=request.name,
            tenant_id=tenant_id,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
        tenant_gauntlet_manager.save_gauntlet(gauntlet)
        
        logger.info(f"Gauntlet '{gauntlet.name}' created by {user.name}")
        record_audit_event(
            user=user,
            operation="CREATE_GAUNTLET",
            resource="gauntlet",
            resource_id=gauntlet.name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Gauntlet created", "gauntlet_name": gauntlet.name}
    
    except Exception as e:
        logger.error(f"Error creating gauntlet: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/gauntlets/{gauntlet_name}", dependencies=[Depends(require_role(UserRole.USER))])
def update_gauntlet(
    gauntlet_name: str,
    request: GauntletCreateRequest,
    user: AuthUser = Depends(require_role(UserRole.USER)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Update an existing gauntlet (requires USER role)."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    existing_gauntlet = tenant_gauntlet_manager.get_gauntlet(gauntlet_name)
    if not existing_gauntlet:
        raise HTTPException(status_code=404, detail="Gauntlet not found")

    try:
        # Convert rounds to GauntletRoundRule objects
        rounds = [GauntletRoundRule(**round_data) for round_data in request.rounds]
        
        updated_gauntlet = GauntletDefinition(
            name=gauntlet_name, # Ensure name from path is used
            tenant_id=tenant_id,
            team_name=request.team_name,
            description=request.description,
            rounds=rounds
        )
        
        tenant_gauntlet_manager.save_gauntlet(updated_gauntlet) # Overwrite existing
        
        logger.info(f"Gauntlet '{gauntlet_name}' updated by {user.name}")
        record_audit_event(
            user=user,
            operation="UPDATE_GAUNTLET",
            resource="gauntlet",
            resource_id=gauntlet_name,
            success=True,
            details={"tenant_id": tenant_id}
        )
        
        return {"message": "Gauntlet updated", "gauntlet_name": gauntlet_name}
    
    except Exception as e:
        logger.error(f"Error updating gauntlet '{gauntlet_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/gauntlets/{gauntlet_name}", dependencies=[Depends(require_role(UserRole.ADMIN))])
def delete_gauntlet(
    gauntlet_name: str,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """Delete a gauntlet (requires ADMIN role)."""
    tenant_gauntlet_manager = get_tenant_gauntlet_manager(tenant_id)
    if not tenant_gauntlet_manager.get_gauntlet(gauntlet_name):
        raise HTTPException(status_code=404, detail="Gauntlet not found")
    
    tenant_gauntlet_manager.delete_gauntlet(gauntlet_name)
    logger.info(f"Gauntlet '{gauntlet_name}' deleted by {user.name}")
    record_audit_event(
        user=user,
        operation="DELETE_GAUNTLET",
        resource="gauntlet",
        resource_id=gauntlet_name,
        success=True,
        details={"tenant_id": tenant_id}
    )
    
    return {"message": "Gauntlet deleted", "gauntlet_name": gauntlet_name}


# Webhook system
import asyncio
import aiohttp
from typing import Set


class WebhookRegistration(BaseModel):
    """Webhook registration."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for webhook verification")


class WebhookManager:
    """Manages webhook registrations and delivery."""
    
    def __init__(self):
        """Initialize webhook manager."""
        self.webhooks: Dict[str, WebhookRegistration] = {}
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    def register(self, webhook_id: str, registration: WebhookRegistration) -> None:
        """Register a webhook."""
        self.webhooks[webhook_id] = registration
        logger.info(f"Registered webhook {webhook_id} for events: {registration.events}")
    
    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False
    
    async def trigger(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger webhooks for an event."""
        matching_webhooks = [
            (wid, wh) for wid, wh in self.webhooks.items()
            if event in wh.events or "*" in wh.events
        ]
        
        if not matching_webhooks:
            return
        
        logger.info(f"Triggering {len(matching_webhooks)} webhooks for event: {event}")
        
        # Trigger all webhooks concurrently
        tasks = [
            self._deliver_webhook(wid, wh, event, data)
            for wid, wh in matching_webhooks
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_webhook(
        self,
        webhook_id: str,
        webhook: WebhookRegistration,
        event: str,
        data: Dict[str, Any]
    ) -> None:
        """Deliver webhook with retry logic."""
        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            # Add signature for verification
            import hmac
            import hashlib
            import json
            
            signature = hmac.new(
                webhook.secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook.url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status < 300:
                            logger.info(f"Webhook {webhook_id} delivered successfully")
                            return
                        else:
                            logger.warning(
                                f"Webhook {webhook_id} returned status {response.status}"
                            )
            except Exception as e:
                logger.error(f"Webhook {webhook_id} delivery failed (attempt {attempt + 1}): {e}")
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        logger.error(f"Webhook {webhook_id} failed after {self.max_retries} attempts")


# Initialize webhook manager
webhook_manager = WebhookManager()


# Webhook endpoints

@app.post("/webhooks", dependencies=[Depends(require_role(UserRole.USER))])
def register_webhook(registration: WebhookRegistration, user: AuthUser = Depends(verify_api_key)):
    """Register a webhook."""
    webhook_id = str(uuid.uuid4())
    webhook_manager.register(webhook_id, registration)
    
    logger.info(f"Webhook registered by {user.name}: {webhook_id}")
    
    return {
        "webhook_id": webhook_id,
        "url": registration.url,
        "events": registration.events,
        "message": "Webhook registered successfully"
    }


@app.get("/webhooks", dependencies=[Depends(verify_api_key)])
def list_webhooks():
    """List all registered webhooks."""
    return {
        "webhooks": [
            {
                "webhook_id": wid,
                "url": wh.url,
                "events": wh.events
            }
            for wid, wh in webhook_manager.webhooks.items()
        ],
        "total": len(webhook_manager.webhooks)
    }


@app.delete("/webhooks/{webhook_id}", dependencies=[Depends(require_role(UserRole.USER))])
def unregister_webhook(webhook_id: str, user: AuthUser = Depends(verify_api_key)):
    """Unregister a webhook."""
    if not webhook_manager.unregister(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    logger.info(f"Webhook unregistered by {user.name}: {webhook_id}")
    
    return {"message": "Webhook unregistered", "webhook_id": webhook_id}


@app.get("/audit/logs", dependencies=[Depends(require_role(UserRole.ADMIN))])
def list_audit_logs(
    limit: int = 200,
    user: AuthUser = Depends(require_role(UserRole.ADMIN)),
    tenant_id: str = Depends(get_tenant_id)
):
    """List audit logs for the current tenant (admin only)."""
    tenant_logs = [
        log for log in AUDIT_LOGS
        if log.get("details", {}).get("tenant_id") == tenant_id
    ]
    record_audit_event(
        user=user,
        operation="LIST_AUDIT_LOGS",
        resource="audit",
        resource_id="*",
        success=True,
        details={"tenant_id": tenant_id}
    )
    return {
        "logs": tenant_logs[-limit:],
        "total": len(tenant_logs)
    }


# Helper function to trigger webhooks from workflow events
async def trigger_workflow_event(event: str, workflow_id: str, data: Dict[str, Any] = None):
    """Trigger webhook for workflow event."""
    payload = {
        "workflow_id": workflow_id,
        **(data or {})
    }
    await webhook_manager.trigger(event, payload)


# Statistics endpoints

@app.get("/statistics", dependencies=[Depends(verify_api_key)])
def get_statistics():
    """Get system statistics."""
    completed_workflows = [wf for wf in workflows.values() if wf.status == "completed"]
    failed_workflows = [wf for wf in workflows.values() if wf.status == "failed"]
    running_workflows = [wf for wf in workflows.values() if wf.status == "running"]
    
    return {
        "total_workflows": len(workflows),
        "completed": len(completed_workflows),
        "failed": len(failed_workflows),
        "running": len(running_workflows),
        "total_teams": len(team_manager.get_all_teams()),
        "total_gauntlets": len(gauntlet_manager.get_all_gauntlets())
    }


server = None

def start_api_server(host: str = "0.0.0.0", port: int = 8001):
    """Start the API server."""
    global server
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    server.run()

def stop_api_server():
    """Stop the API server."""
    global server
    if server:
        server.should_exit = True
        server.force_exit = True

if __name__ == "__main__":
    start_api_server()
