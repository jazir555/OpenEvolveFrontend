"""
REST API Server for External System Integration

This module provides a REST API for external systems to interact with the
Decomposition Workflow system.
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from collections import deque
from enum import Enum
import uvicorn
import os
import re
import base64
from datetime import datetime, timedelta
import uuid
import logging

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for API Server
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: API Server
def _trigger_api_alerts(operation, success, request_id=None, error=None, metadata=None):
    """Trigger alerts for API server operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"API {operation} Failed",
            message=f"API server operation '{operation}' failed: {error}",
            severity=severity,
            source="APIServer",
            metadata=metadata or {"request_id": request_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger API alert: {e}")


def _extract_api_knowledge(operation, request_id, endpoint, result):
    """Extract knowledge from API operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"api_{operation}_{request_id}",
            artifact_type="api_execution",
            source_component="APIServer",
            content={
                "operation": operation,
                "request_id": request_id,
                "endpoint": endpoint,
                "status": result.get("status", "unknown") if result else "unknown",
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract API knowledge: {e}")


def _track_api_performance(operation, success, duration_seconds, endpoint, status_code=200):
    """Track performance of API operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name=f"api_{endpoint}",
            component_name="APIServer",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "endpoint": endpoint,
                "status_code": status_code
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track API performance: {e}")


def _request_openai_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    extra_headers: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 1024,
    seed: Optional[int] = None,
) -> str:
    """Make a request to an OpenAI-compatible API."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed
        )
        return response.choices[0].message.content
    except ImportError:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        if extra_headers:
            headers.update(extra_headers)
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            data["seed"] = seed
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]


# Import environment helpers
from env_helpers import is_production

from workflow_structures import (
    DecompositionPlan, SubProblem, Team, GauntletDefinition, GauntletRoundRule,
    WorkflowState, ModelConfig
)
from knowledge_manager import KnowledgeManager
from template_manager import TemplateManager
from parameter_manager import ParameterManager
from sovereign_persistence import SovereignDatabase
from sovereign_reliability import HealthMonitor
from providercatalogue import PROVIDERS as PROVIDERS_MAP
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow
from determinism_stack import (
    DeterministicPipeline,
    DeterminismConfig,
    HybridDeterministicSystem,
    LLMConfig,
    build_llm,
)


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


# Initialize templates for dashboard
templates = Jinja2Templates(directory="templates")


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
knowledge_manager = KnowledgeManager()
template_manager = TemplateManager()
parameter_manager = ParameterManager()
sovereign_db = SovereignDatabase()
sovereign_health_monitor = HealthMonitor()

# Auto-approval configuration (in-memory)
AUTO_APPROVAL_CONFIG: Dict[str, Any] = {"enabled": False, "rules": []}
AUTO_APPROVAL_AUDIT_LOG: List[Dict[str, Any]] = []

# In-memory storage for workflows (replace with database in production)
workflows: Dict[str, WorkflowState] = {}

# In-memory audit log (replace with persistent storage in production)
AUDIT_LOGS: List[Dict[str, Any]] = []

# ICR event queues (in-memory, best-effort)
ICR_REFINEMENT_EVENTS: deque = deque(maxlen=200)
ICR_REWARD_CALIBRATION_QUEUE: deque = deque(maxlen=100)
ICR_REWARD_CALIBRATION_RESPONSES: Dict[str, Dict[str, Any]] = {}
ICR_HEATMAP_SNAPSHOTS: deque = deque(maxlen=100)


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


def _evaluate_auto_approval_rule(rule: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    """Evaluate a single auto-approval rule against a plan."""
    conditions = rule.get("conditions", [])
    if not conditions:
        return False

    results = []
    for condition in conditions:
        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")
        plan_value = plan.get(field)

        try:
            if operator == "<":
                result = float(plan_value) < float(value)
            elif operator == ">":
                result = float(plan_value) > float(value)
            elif operator == "==":
                result = str(plan_value) == str(value)
            elif operator == "!=":
                result = str(plan_value) != str(value)
            elif operator == "contains":
                result = str(value).lower() in str(plan_value).lower()
            else:
                result = False
        except (TypeError, ValueError):
            result = False

        results.append(result)

    final_result = results[0]
    for index, condition in enumerate(conditions[:-1]):
        logical_op = condition.get("logical_op", "AND")
        if logical_op == "AND":
            final_result = final_result and results[index + 1]
        else:
            final_result = final_result or results[index + 1]

    return final_result


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


class KnowledgeArtifactCreateRequest(BaseModel):
    artifact_type: str
    content: Any
    source_workflow_id: Optional[str] = "manual"
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    related_artifacts: Optional[List[str]] = None


class KnowledgeSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    artifact_types: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=100)


class KnowledgeRecommendationsRequest(BaseModel):
    problem_statement: str
    domain: Optional[str] = None


class KnowledgeImportRequest(BaseModel):
    artifacts: Dict[str, Any]


class AutoApprovalConditionModel(BaseModel):
    field: str
    operator: str
    value: Any
    logical_op: Optional[str] = "AND"


class AutoApprovalRuleModel(BaseModel):
    name: str
    priority: int = Field(0, ge=0, le=100)
    action: str = Field("approve")
    enabled: bool = True
    conditions: List[AutoApprovalConditionModel]
    created_at: Optional[str] = None


class AutoApprovalConfigModel(BaseModel):
    enabled: bool = False
    rules: List[AutoApprovalRuleModel] = Field(default_factory=list)


class AutoApprovalTestRequest(BaseModel):
    plan: Dict[str, Any]


class WorkflowTemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    config: Dict[str, Any]
    tags: Optional[List[str]] = None


class WorkflowTemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class ProviderModelsRequest(BaseModel):
    api_key: Optional[str] = None


class ParameterValidateRequest(BaseModel):
    parameters: Dict[str, Any]


class SuggestionRequest(BaseModel):
    content: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    extra_headers: Optional[Dict[str, str]] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 1024
    seed: Optional[int] = None


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


class IcrRefinementEvent(BaseModel):
    """Event signaling a refinement is needed."""
    reason: Optional[str] = None
    overall_score: Optional[float] = None
    weaknesses: Optional[List[str]] = None
    friction_points: Optional[List[str]] = None
    auto_refine: Optional[bool] = None


class IcrRewardCalibrationRequest(BaseModel):
    """Reward calibration request payload."""
    request_id: Optional[str] = None
    option_a: str
    option_b: str
    confidence: Optional[float] = None
    prompt: Optional[str] = None


class IcrRewardCalibrationResponse(BaseModel):
    """Reward calibration response payload."""
    request_id: Optional[str] = None
    choice: str


class IcrHeatmapPoint(BaseModel):
    """Heatmap point from GenerativeUI."""
    x: float
    y: float
    intensity: float = 0.0
    dwellMs: Optional[float] = None
    timestamp: Optional[float] = None
    type: Optional[str] = None


class IcrHeatmapSnapshot(BaseModel):
    """Heatmap snapshot payload for multimodal analysis."""
    snapshot_id: Optional[str] = None
    timestamp: Optional[float] = None
    screen_html: str
    heatmap_data_url: Optional[str] = None
    composite_data_url: Optional[str] = None
    points: List[IcrHeatmapPoint] = Field(default_factory=list)
    manual_code_delta: Optional[float] = None
    context_text: Optional[str] = None
    auto_refine: Optional[bool] = None

def verify_api_key(x_api_key: str = Header(...)) -> AuthUser:
    """Verify API key from header and return user info."""
    import time
    start_time = time.time()
    success = False

    try:
        if x_api_key not in API_KEYS:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            duration = time.time() - start_time
            _trigger_api_alerts("verify_api_key", False, None, "Invalid API key")
            _track_api_performance("verify_api_key", False, duration, "verify_api_key", 401)

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )

        key_info = API_KEYS[x_api_key]
        user = AuthUser(
            api_key=x_api_key,
            role=key_info["role"],
            name=key_info["name"]
        )

        # **ACTUAL INTEGRATION**: Track performance on success
        success = True
        duration = time.time() - start_time
        _track_api_performance("verify_api_key", True, duration, "verify_api_key", 200)

        return user

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # **ACTUAL INTEGRATION**: Trigger alert and track unexpected errors
        duration = time.time() - start_time
        _trigger_api_alerts("verify_api_key", False, None, str(e))
        _track_api_performance("verify_api_key", False, duration, "verify_api_key", 500)
        raise


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
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
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
    
    except (ValueError, TypeError, RuntimeError) as e:
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
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
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
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
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
    
    except (ValueError, TypeError, KeyError, AttributeError) as e:
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
            except (OSError, IOError, RuntimeError) as e:
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


# Deterministic LLM endpoints (Bubblelab UI + CLI control)

class DeterminismGenerateRequest(BaseModel):
    prompt: str
    schema: Optional[Dict[str, Any]] = None
    constraints: Optional[str] = None
    context_document: Optional[str] = None
    mode: str = "auto"  # auto | cloud | local | hybrid | consensus
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_base_url: Optional[str] = None
    local_provider: Optional[str] = "hf"
    local_model: Optional[str] = None
    local_device: Optional[str] = "cpu"
    local_dtype: Optional[str] = "auto"
    config: Optional[Dict[str, Any]] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None


class DeterminismCheckRequest(BaseModel):
    prompt: str
    tier: int = 2
    runs: int = 3
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None
    device: Optional[str] = "cpu"
    dtype: Optional[str] = "auto"


def _build_llm(
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    device: Optional[str] = None,
    dtype: Optional[str] = None,
):
    if not provider or not model:
        return None
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        device=device or "cpu",
        dtype=dtype or "auto",
    )
    return build_llm(config)


def _build_config(overrides: Optional[Dict[str, Any]], detllm_backend: Optional[str], detllm_model: Optional[str]) -> DeterminismConfig:
    config = DeterminismConfig()
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    if detllm_backend:
        config.detllm_backend = detllm_backend
    if detllm_model:
        config.detllm_model = detllm_model
    return config


@app.post("/determinism/generate", dependencies=[Depends(verify_api_key)])
def determinism_generate(req: DeterminismGenerateRequest):
    config = _build_config(req.config, req.detllm_backend, req.detllm_model)

    if req.mode in {"hybrid", "consensus"}:
        cloud_llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url)
        local_llm = _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)
        if cloud_llm is None or local_llm is None:
            raise HTTPException(status_code=400, detail="Hybrid mode requires both cloud and local LLM configs")
        system = HybridDeterministicSystem(cloud_llm=cloud_llm, local_llm=local_llm)
        result = system.generate(req.prompt, mode=req.mode)
        return result.__dict__

    if req.mode == "cloud":
        llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url)
    elif req.mode == "local":
        llm = _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)
    else:
        llm = _build_llm(req.cloud_provider, req.cloud_model, req.cloud_api_key, req.cloud_base_url) or _build_llm(req.local_provider, req.local_model, None, None, req.local_device, req.local_dtype)

    pipeline = DeterministicPipeline(llm=llm, config=config)
    result = pipeline.generate_with_all_layers(req.prompt, schema=req.schema, constraints=req.constraints, context_document=req.context_document)
    return result.__dict__


@app.post("/determinism/check", dependencies=[Depends(verify_api_key)])
def determinism_check(req: DeterminismCheckRequest):
    llm = _build_llm(req.provider, req.model, req.api_key, req.base_url, req.device, req.dtype)
    pipeline = DeterministicPipeline(llm=llm, config=_build_config(None, req.detllm_backend, req.detllm_model))
    result = pipeline.reproducibility.check(
        prompt=req.prompt,
        llm=llm,
        tier=req.tier,
        runs=req.runs,
        backend=req.detllm_backend,
        model=req.detllm_model,
    )
    return result


# --- ICR Heatmap helpers ---

def _decode_data_url(data_url: Optional[str]) -> Optional[bytes]:
    if not data_url:
        return None
    if "," not in data_url:
        return None
    try:
        _, b64_data = data_url.split(",", 1)
        return base64.b64decode(b64_data)
    except (ValueError, TypeError, base64.binascii.Error):
        return None


async def _analyze_heatmap_composite(data_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Analyze a heatmap composite image using VLM (Vision Language Model).

    This function provides UI interaction insights from heatmap composite images
    using configurable VLM providers (OpenAI, Anthropic, etc.).

    Args:
        data_url: Base64-encoded data URL of the heatmap composite image

    Returns:
        Dictionary containing VLM analysis results, or None if analysis is disabled/unavailable.
        Returns:
        {
            "summary": str,
            "insights": List[str],
            "friction_points": List[str],
            "recommendations": List[str],
            "confidence": float,
            "provider": str,
            "model": str
        }
    """
    # Check if VLM analysis is enabled
    if os.getenv("ICR_VLM_ENABLED", "").lower() not in {"1", "true", "yes"}:
        logger.debug("VLM analysis is disabled via ICR_VLM_ENABLED")
        return None

    if not data_url:
        logger.debug("No composite data URL provided for VLM analysis")
        return None

    # Decode the image
    image_bytes = _decode_data_url(data_url)
    if not image_bytes:
        logger.warning("Failed to decode composite data URL for VLM analysis")
        return None

    try:
        from vision_language_monitor import VLMAnalyzer, VLMConfig, AnalysisType, VLMProvider
    except ImportError:
        logger.warning("vision_language_monitor module not available; skipping VLM heatmap analysis")
        return None

    # Load configuration from environment variables
    provider_env = os.getenv("ICR_VLM_PROVIDER", "openai").lower()
    provider = VLMProvider.OPENAI
    if provider_env:
        for candidate in VLMProvider:
            if candidate.value == provider_env:
                provider = candidate
                break

    model = os.getenv("ICR_VLM_MODEL", "gpt-4o")
    temperature = float(os.getenv("ICR_VLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("ICR_VLM_MAX_TOKENS", "1024"))
    api_key = os.getenv("ICR_VLM_API_KEY")
    base_url = os.getenv("ICR_VLM_BASE_URL")

    # Create VLM config
    config = VLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url
    )

    # Initialize analyzer
    analyzer = VLMAnalyzer(config)

    # Check if VLM is properly configured
    if not analyzer.is_configured():
        logger.warning("VLM is not properly configured (missing API key). Skipping analysis.")
        return None

    # Build analysis prompt
    prompt = (
        "Analyze this UI snapshot with an interaction heatmap overlay.\n"
        "Identify cognitive friction points, confusing placements, and areas of repeated interaction.\n"
        "Provide concise, actionable UI refinement suggestions."
    )

    # Run analysis
    try:
        analysis = await analyzer.analyze(image_bytes, prompt, AnalysisType.LAYOUT_ANALYSIS)
        return analysis.to_dict()
    except Exception as e:
        logger.error(f"VLM analysis failed: {e}")
        return None


# ICR Event Bridge Endpoints (optional, unauthenticated for local UI polling)

@app.post("/icr/events/refinement-needed")
def icr_emit_refinement_needed(event: IcrRefinementEvent):
    payload = event.model_dump()
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REFINEMENT_EVENTS.append(payload)
    return {"queued": True}


@app.get("/icr/events/refinement-needed")
def icr_get_refinement_needed(limit: int = 5):
    items = []
    while ICR_REFINEMENT_EVENTS and len(items) < limit:
        items.append(ICR_REFINEMENT_EVENTS.popleft())
    return items


@app.post("/icr/reward-calibration/request")
def icr_queue_reward_calibration(request: IcrRewardCalibrationRequest):
    payload = request.model_dump()
    if not payload.get("request_id"):
        payload["request_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_QUEUE.append(payload)
    return {"queued": True, "request_id": payload["request_id"]}


@app.get("/icr/reward-calibration/next")
def icr_next_reward_calibration():
    if not ICR_REWARD_CALIBRATION_QUEUE:
        return {}
    return ICR_REWARD_CALIBRATION_QUEUE.popleft()


@app.post("/icr/reward-calibration/respond")
def icr_reward_calibration_respond(response: IcrRewardCalibrationResponse):
    request_id = response.request_id or str(uuid.uuid4())
    payload = response.model_dump()
    payload["request_id"] = request_id
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_RESPONSES[request_id] = payload
    return {"received": True, "request_id": request_id}


@app.get("/icr/reward-calibration/response/{request_id}")
def icr_reward_calibration_response(request_id: str):
    return ICR_REWARD_CALIBRATION_RESPONSES.get(request_id, {})


@app.post("/icr/heatmap/snapshot")
async def icr_heatmap_snapshot(snapshot: IcrHeatmapSnapshot):
    """
    Store heatmap snapshot for ICR pattern analysis.
    
    Accepts heatmap snapshot data from GenerativeUI and stores it for:
    - Pattern analysis and learning
    - Multimodal healing prompt generation
    - Vision-language model analysis of UI interactions
    
    Args:
        snapshot: Heatmap snapshot containing screen HTML, heatmap data, and interaction points
        
    Returns:
        Success response with snapshot_id and optional analysis results
        
    Environment Variables for VLM:
        - ICR_VLM_ENABLED: Enable/disable VLM analysis (default: false)
        - ICR_VLM_PROVIDER: VLM provider - openai, anthropic, mock (default: openai)
        - ICR_VLM_MODEL: Model name (default: gpt-4o for OpenAI, claude-3-5-sonnet-20241022 for Anthropic)
        - ICR_VLM_API_KEY: API key for the VLM provider (optional if using provider's default env var)
        - ICR_VLM_TEMPERATURE: Temperature for VLM (default: 0.2)
        - ICR_VLM_MAX_TOKENS: Max tokens for VLM response (default: 1024)
        - ICR_VLM_BASE_URL: Custom base URL for VLM API (optional)
    """
    payload = snapshot.model_dump()
    if not payload.get("snapshot_id"):
        payload["snapshot_id"] = str(uuid.uuid4())
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().timestamp()
    payload["received_at"] = datetime.utcnow().isoformat()
    ICR_HEATMAP_SNAPSHOTS.append(payload)

    analysis = None
    vlm_analysis = None

    # Generate multimodal healing prompt if analytics_manager is available
    try:
        from analytics_manager import analytics_manager
        heatmap_payload = {
            "points": payload.get("points", []),
            "manual_code_delta": payload.get("manual_code_delta")
        }
        analysis = analytics_manager.generate_multimodal_healing_prompt(
            payload.get("context_text", "") or "",
            heatmap_snapshot=heatmap_payload,
            auto_refine_enabled=bool(payload.get("auto_refine"))
        )
    except Exception as exc:
        logger.warning("Failed to generate multimodal healing prompt: %s", exc)

    # Run VLM analysis if enabled and composite data is available
    try:
        vlm_analysis = await _analyze_heatmap_composite(payload.get("composite_data_url"))
        if vlm_analysis and analysis is not None:
            analysis["vlm_analysis"] = vlm_analysis
    except Exception as exc:
        logger.warning("Failed to run VLM heatmap analysis: %s", exc)

    return {
        "queued": True,
        "snapshot_id": payload["snapshot_id"],
        "analysis": analysis,
        "vlm_analysis": vlm_analysis,
    }


@app.get("/icr/vlm/config")
def icr_vlm_config():
    """
    Get VLM configuration status.
    
    Returns current VLM configuration and whether it's properly set up.
    Useful for debugging and checking if VLM analysis is available.
    
    Returns:
        Dictionary with VLM configuration information
    """
    try:
        from vision_language_monitor import VLMAnalyzer, VLMConfig
    except ImportError:
        return {
            "available": False,
            "error": "vision_language_monitor module not available",
            "message": "VLM analysis is not available"
        }

    config = VLMAnalyzer._load_config_from_env()
    analyzer = VLMAnalyzer(config)
    
    return {
        "available": True,
        "enabled": os.getenv("ICR_VLM_ENABLED", "").lower() in {"1", "true", "yes"},
        "configured": analyzer.is_configured(),
        "config": analyzer.get_config_info()
    }


# =============================================================================
# ICR ANALYTICS DASHBOARD ENDPOINTS
# =============================================================================

# In-memory storage for ICR analytics data (simulated)
ICR_ANALYTICS_DATA = {
    "overview": {
        "icr_enabled": True,
        "total_patterns": 0,
        "overall_success_rate": 0.0,
        "active_components": 5,
        "total_refinements": 0
    },
    "components": {
        "quality_gate_engine": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "workflow_orchestrator": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "robustness_coordinator": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "bubblelab": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        },
        "roma": {
            "active": True,
            "total_patterns": 0,
            "overall_pass_rate": 0.0,
            "overall_quality": 0.0
        }
    },
    "patterns": {
        "pattern_types": {},
        "trends": {
            "timestamps": [],
            "values": []
        },
        "by_content_type": {},
        "by_quality_level": {},
        "by_complexity": {}
    },
    "vlm": {
        "available": False,
        "enabled": False,
        "total_analyses": 0,
        "total_tokens": 0,
        "avg_confidence": 0.0,
        "cache_hit_rate": 0.0,
        "by_provider": {},
        "config": None
    },
    "heatmap": {
        "points": []
    },
    "config": {
        "enabled": True,
        "enable_prediction": True,
        "enable_learning": True,
        "quality_gate_enabled": True,
        "workflow_orchestrator_enabled": True,
        "gauntlet_system_enabled": True,
        "robustness_enabled": True,
        "roma_modules_enabled": True
    }
}


@app.get("/icr/dashboard")
async def icr_dashboard(request: Request):
    """
    Serve the ICR Analytics Dashboard.
    
    Returns the HTML template for the ICR analytics dashboard.
    """
    return templates.TemplateResponse("icr_dashboard.html", {"request": request})


@app.get("/icr/analytics/overview")
async def icr_analytics_overview():
    """
    Get ICR overview statistics.
    
    Returns:
        - Total patterns stored
        - Overall success rate
        - Active components count
        - Total refinements applied
        - ICR enabled status
    """
    # Calculate total patterns from all components
    total_patterns = sum(
        comp["total_patterns"]
        for comp in ICR_ANALYTICS_DATA["components"].values()
    )
    
    # Calculate overall success rate
    component_rates = [
        comp["overall_pass_rate"]
        for comp in ICR_ANALYTICS_DATA["components"].values()
        if comp["overall_pass_rate"] > 0
    ]
    overall_success_rate = sum(component_rates) / len(component_rates) if component_rates else 0.0
    
    # Count active components
    active_components = sum(
        1 for comp in ICR_ANALYTICS_DATA["components"].values()
        if comp["active"]
    )
    
    return {
        "icr_enabled": ICR_ANALYTICS_DATA["overview"]["icr_enabled"],
        "total_patterns": total_patterns,
        "overall_success_rate": overall_success_rate,
        "active_components": active_components,
        "total_refinements": ICR_ANALYTICS_DATA["overview"]["total_refinements"]
    }


@app.get("/icr/analytics/components")
async def icr_analytics_components():
    """
    Get component-specific ICR statistics.
    
    Returns statistics for each ICR component:
        - QualityGateEngine
        - SGDWorkflowOrchestrator
        - RobustnessCoordinator
        - BubbleLab
        - ROMA
    """
    return ICR_ANALYTICS_DATA["components"]


@app.get("/icr/analytics/patterns")
async def icr_analytics_patterns():
    """
    Get pattern analysis data.
    
    Returns:
        - Pattern types distribution
        - Success rate trends over time
        - Patterns by content type
        - Patterns by quality level
        - Patterns by complexity
    """
    return ICR_ANALYTICS_DATA["patterns"]


@app.get("/icr/analytics/vlm")
async def icr_analytics_vlm():
    """
    Get VLM analytics data.
    
    Returns:
        - Total analyses performed
        - Total tokens consumed
        - Average confidence
        - Cache hit rate
        - Analysis count by provider
        - Current VLM configuration
    """
    # Get VLM status from existing endpoint
    vlm_status = icr_vlm_config()
    
    return {
        "available": vlm_status.get("available", False),
        "enabled": vlm_status.get("enabled", False),
        "total_analyses": ICR_ANALYTICS_DATA["vlm"]["total_analyses"],
        "total_tokens": ICR_ANALYTICS_DATA["vlm"]["total_tokens"],
        "avg_confidence": ICR_ANALYTICS_DATA["vlm"]["avg_confidence"],
        "cache_hit_rate": ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"],
        "by_provider": ICR_ANALYTICS_DATA["vlm"]["by_provider"],
        "config": vlm_status.get("config")
    }


@app.get("/icr/analytics/refinements")
async def icr_analytics_refinements(limit: int = 10):
    """
    Get recent refinement events.
    
    Args:
        limit: Maximum number of events to return (default: 10)
    
    Returns:
        - List of recent refinement events with details
    """
    # Get events from the global queue
    events = []
    while ICR_REFINEMENT_EVENTS and len(events) < limit:
        event = ICR_REFINEMENT_EVENTS.popleft()
        events.append(event)
        # Put it back for other consumers
        ICR_REFINEMENT_EVENTS.appendleft(event)
    
    return {
        "events": events[:limit],
        "total_count": len(ICR_REFINEMENT_EVENTS)
    }


@app.get("/icr/analytics/heatmap")
async def icr_analytics_heatmap():
    """
    Get ICR pattern heatmap data.
    
    Returns:
        - Heatmap points with coordinates and intensity
        - Snapshot metadata
    """
    # Aggregate heatmap data from snapshots
    heatmap_points = []
    
    for snapshot in list(ICR_HEATMAP_SNAPSHOTS):
        points = snapshot.get("points", [])
        heatmap_points.extend(points)
    
    return {
        "points": heatmap_points,
        "total_snapshots": len(ICR_HEATMAP_SNAPSHOTS)
    }


@app.get("/icr/config")
async def icr_get_config():
    """
    Get current ICR configuration.
    
    Returns:
        - ICR enabled status
        - Component enablement flags
        - Feature flags (prediction, learning)
    """
    return ICR_ANALYTICS_DATA["config"]


# =============================================================================
# ICR DATA UPDATE HELPERS (for component integration)
# =============================================================================

def update_icr_component_stats(component_name: str, stats: dict):
    """
    Update statistics for a specific ICR component.
    
    Args:
        component_name: Name of the component (e.g., "quality_gate_engine")
        stats: Statistics dictionary with keys:
            - total_patterns: int
            - overall_pass_rate: float
            - overall_quality: float
            - active: bool
    """
    if component_name in ICR_ANALYTICS_DATA["components"]:
        ICR_ANALYTICS_DATA["components"][component_name].update(stats)


def update_icr_pattern_data(pattern_type: str, content_type: str = None,
                           quality_level: str = None, complexity: int = None):
    """
    Update pattern analysis data when new patterns are stored.
    
    Args:
        pattern_type: Type of pattern (e.g., "content_type", "metric")
        content_type: Content type (e.g., "code", "text")
        quality_level: Quality level (e.g., "standard", "high")
        complexity: Complexity score (1-10)
    """
    # Update pattern types
    if pattern_type not in ICR_ANALYTICS_DATA["patterns"]["pattern_types"]:
        ICR_ANALYTICS_DATA["patterns"]["pattern_types"][pattern_type] = 0
    ICR_ANALYTICS_DATA["patterns"]["pattern_types"][pattern_type] += 1
    
    # Update by content type
    if content_type:
        if content_type not in ICR_ANALYTICS_DATA["patterns"]["by_content_type"]:
            ICR_ANALYTICS_DATA["patterns"]["by_content_type"][content_type] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_content_type"][content_type] += 1
    
    # Update by quality level
    if quality_level:
        if quality_level not in ICR_ANALYTICS_DATA["patterns"]["by_quality_level"]:
            ICR_ANALYTICS_DATA["patterns"]["by_quality_level"][quality_level] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_quality_level"][quality_level] += 1
    
    # Update by complexity
    if complexity:
        complexity_key = str(complexity)
        if complexity_key not in ICR_ANALYTICS_DATA["patterns"]["by_complexity"]:
            ICR_ANALYTICS_DATA["patterns"]["by_complexity"][complexity_key] = 0
        ICR_ANALYTICS_DATA["patterns"]["by_complexity"][complexity_key] += 1
    
    # Update trend
    now = datetime.utcnow()
    ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"].append(now.isoformat())
    # Placeholder for actual success rate calculation
    ICR_ANALYTICS_DATA["patterns"]["trends"]["values"].append(0.8)
    
    # Keep only last 50 trend points
    if len(ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"]) > 50:
        ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"] = \
            ICR_ANALYTICS_DATA["patterns"]["trends"]["timestamps"][-50:]
        ICR_ANALYTICS_DATA["patterns"]["trends"]["values"] = \
            ICR_ANALYTICS_DATA["patterns"]["trends"]["values"][-50:]


def update_icr_vlm_stats(provider: str, tokens_used: int = 0,
                        confidence: float = 0.0, cached: bool = False):
    """
    Update VLM analytics statistics.
    
    Args:
        provider: VLM provider name (e.g., "openai", "anthropic")
        tokens_used: Number of tokens consumed
        confidence: Analysis confidence score
        cached: Whether the result was from cache
    """
    ICR_ANALYTICS_DATA["vlm"]["total_analyses"] += 1
    ICR_ANALYTICS_DATA["vlm"]["total_tokens"] += tokens_used
    
    # Update average confidence
    current_avg = ICR_ANALYTICS_DATA["vlm"]["avg_confidence"]
    total = ICR_ANALYTICS_DATA["vlm"]["total_analyses"]
    ICR_ANALYTICS_DATA["vlm"]["avg_confidence"] = \
        (current_avg * (total - 1) + confidence) / total
    
    # Update cache hit rate
    if cached:
        cache_hits = ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] * (total - 1) + 1
    else:
        cache_hits = ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] * (total - 1)
    ICR_ANALYTICS_DATA["vlm"]["cache_hit_rate"] = cache_hits / total
    
    # Update by provider
    if provider not in ICR_ANALYTICS_DATA["vlm"]["by_provider"]:
        ICR_ANALYTICS_DATA["vlm"]["by_provider"][provider] = 0
    ICR_ANALYTICS_DATA["vlm"]["by_provider"][provider] += 1


def record_icr_refinement(refinement_type: str, component: str,
                          reason: str, success: bool, confidence: float):
    """
    Record a refinement event.
    
    Args:
        refinement_type: Type of refinement (e.g., "threshold_adjustment")
        component: Component that triggered the refinement
        reason: Reason for the refinement
        success: Whether the refinement was successful
        confidence: Confidence in the refinement
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "refinement_type": refinement_type,
        "component": component,
        "reason": reason,
        "success": success,
        "confidence": confidence
    }
    
    ICR_REFINEMENT_EVENTS.append(event)
    ICR_ANALYTICS_DATA["overview"]["total_refinements"] += 1


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


# Knowledge Base endpoints

@app.get("/knowledge/artifacts", dependencies=[Depends(verify_api_key)])
def list_knowledge_artifacts():
    artifacts = knowledge_manager.get_all_artifacts()
    return {"artifacts": [a.__dict__ for a in artifacts]}


@app.get("/knowledge/artifacts/{artifact_id}", dependencies=[Depends(verify_api_key)])
def get_knowledge_artifact(artifact_id: str):
    artifact = knowledge_manager.artifacts.get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifact.__dict__


@app.post("/knowledge/artifacts", dependencies=[Depends(verify_api_key)])
def create_knowledge_artifact(request: KnowledgeArtifactCreateRequest):
    from workflow_structures import KnowledgeArtifact as KnowledgeArtifactModel
    artifact_id = uuid.uuid4().hex[:16]
    artifact = KnowledgeArtifactModel(
        id=artifact_id,
        artifact_type=request.artifact_type,
        content=request.content,
        source_workflow_id=request.source_workflow_id or "manual",
        extraction_timestamp=datetime.now().isoformat(),
        domain=request.domain,
        problem_type=request.problem_type,
        usage_count=0,
        effectiveness_score=0.0,
        related_artifacts=request.related_artifacts or []
    )
    knowledge_manager.store_knowledge_artifact(artifact)
    return artifact.__dict__


@app.delete("/knowledge/artifacts/{artifact_id}", dependencies=[Depends(verify_api_key)])
def delete_knowledge_artifact(artifact_id: str):
    success = knowledge_manager.delete_artifact(artifact_id)
    if not success:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return {"success": True}


@app.post("/knowledge/search", dependencies=[Depends(verify_api_key)])
def search_knowledge(request: KnowledgeSearchRequest):
    results = knowledge_manager.retrieve_relevant_knowledge(
        problem_statement=request.query,
        domain=request.domain,
        artifact_types=request.artifact_types,
        limit=request.limit
    )
    return {"results": [a.__dict__ for a in results]}


@app.get("/knowledge/graph", dependencies=[Depends(verify_api_key)])
def get_knowledge_graph():
    artifacts = knowledge_manager.get_all_artifacts()
    nodes = [
        {
            "id": artifact.id,
            "type": artifact.artifact_type,
            "domain": artifact.domain,
            "usage": artifact.usage_count
        }
        for artifact in artifacts
    ]
    edges = []
    artifact_ids = {artifact.id for artifact in artifacts}
    for artifact in artifacts:
        for related_id in artifact.related_artifacts or []:
            if related_id in artifact_ids:
                edges.append({"source": artifact.id, "target": related_id})
    return {"nodes": nodes, "edges": edges}


@app.get("/knowledge/stats", dependencies=[Depends(verify_api_key)])
def get_knowledge_stats():
    artifacts = knowledge_manager.get_all_artifacts()
    total_usage = sum(a.usage_count for a in artifacts)
    avg_effectiveness = (
        sum(a.effectiveness_score for a in artifacts) / len(artifacts)
        if artifacts else 0.0
    )
    by_type: Dict[str, int] = {}
    for artifact in artifacts:
        by_type[artifact.artifact_type] = by_type.get(artifact.artifact_type, 0) + 1
    return {
        "total_artifacts": len(artifacts),
        "total_usage": total_usage,
        "average_effectiveness": avg_effectiveness,
        "by_type": by_type
    }


@app.post("/knowledge/recommendations", dependencies=[Depends(verify_api_key)])
def get_knowledge_recommendations(request: KnowledgeRecommendationsRequest):
    recommendations = knowledge_manager.apply_learned_patterns(
        request.problem_statement,
        domain=request.domain
    )
    return recommendations


@app.get("/knowledge/export", dependencies=[Depends(verify_api_key)])
def export_knowledge_base():
    artifacts = knowledge_manager.get_all_artifacts()
    export_data = {artifact.id: artifact.__dict__ for artifact in artifacts}
    return export_data


@app.post("/knowledge/import", dependencies=[Depends(verify_api_key)])
def import_knowledge_base(request: KnowledgeImportRequest):
    # File-based import to leverage existing KnowledgeManager logic
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", "knowledge_import.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        import json
        json.dump(request.artifacts, f, indent=2)
    knowledge_manager.import_knowledge_base(temp_path)
    return {"success": True}


# Auto-approval endpoints

@app.get("/auto-approval/config", dependencies=[Depends(verify_api_key)])
def get_auto_approval_config():
    return AUTO_APPROVAL_CONFIG


@app.put("/auto-approval/config", dependencies=[Depends(verify_api_key)])
def update_auto_approval_config(request: AutoApprovalConfigModel):
    AUTO_APPROVAL_CONFIG["enabled"] = request.enabled
    AUTO_APPROVAL_CONFIG["rules"] = [rule.dict() for rule in request.rules]
    return AUTO_APPROVAL_CONFIG


@app.post("/auto-approval/test", dependencies=[Depends(verify_api_key)])
def test_auto_approval_rules(request: AutoApprovalTestRequest):
    results = []
    for rule in AUTO_APPROVAL_CONFIG.get("rules", []):
        if not rule.get("enabled", True):
            continue
        matched = _evaluate_auto_approval_rule(rule, request.plan)
        results.append({
            "rule_name": rule.get("name", "Unnamed Rule"),
            "action": rule.get("action", "approve"),
            "matched": matched
        })
        AUTO_APPROVAL_AUDIT_LOG.append({
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule.get("name", "Unnamed Rule"),
            "action": rule.get("action", "approve"),
            "matched": matched,
            "plan": request.plan
        })
    return {"results": results}


@app.get("/auto-approval/audit", dependencies=[Depends(verify_api_key)])
def get_auto_approval_audit():
    return {"logs": AUTO_APPROVAL_AUDIT_LOG}


# Workflow template endpoints

@app.get("/workflow-templates", dependencies=[Depends(verify_api_key)])
def list_workflow_templates():
    return {"templates": template_manager.get_all_templates()}


@app.post("/workflow-templates", dependencies=[Depends(verify_api_key)])
def create_workflow_template(request: WorkflowTemplateCreateRequest):
    template_id = template_manager.create_template(
        name=request.name,
        description=request.description or "",
        config=request.config,
        tags=request.tags or []
    )
    template = template_manager.get_template(template_id)
    if not template:
        raise HTTPException(status_code=500, detail="Failed to create template")
    return template


@app.put("/workflow-templates/{template_id}", dependencies=[Depends(verify_api_key)])
def update_workflow_template(template_id: str, request: WorkflowTemplateUpdateRequest):
    success = template_manager.update_template(
        template_id=template_id,
        name=request.name,
        description=request.description,
        config=request.config,
        tags=request.tags
    )
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_manager.get_template(template_id)


@app.delete("/workflow-templates/{template_id}", dependencies=[Depends(verify_api_key)])
def delete_workflow_template(template_id: str):
    success = template_manager.delete_template(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"success": True}


@app.get("/workflow-templates/export", dependencies=[Depends(verify_api_key)])
def export_workflow_templates():
    templates = template_manager.get_all_templates()
    return {"templates": templates}


@app.post("/workflow-templates/import", dependencies=[Depends(verify_api_key)])
def import_workflow_templates(request: Dict[str, Any]):
    templates = request.get("templates", [])
    imported = []
    for template in templates:
        template_id = template_manager.create_template(
            name=template.get("name", "Imported Template"),
            description=template.get("description", ""),
            config=template.get("config", {}),
            tags=template.get("tags", []),
        )
        imported.append(template_id)
    return {"success": True, "imported": imported}


# Providers and parameters

@app.get("/providers", dependencies=[Depends(verify_api_key)])
def list_providers():
    providers = []
    for provider_id, data in PROVIDERS_MAP.items():
        providers.append({
            "id": provider_id,
            "name": data.get("name"),
            "api_base": data.get("api_base"),
            "models_endpoint": data.get("models_endpoint"),
            "default_model": data.get("default_model")
        })
    return {"providers": providers}


@app.post("/providers/{provider_id}/models", dependencies=[Depends(verify_api_key)])
def get_provider_models(provider_id: str, request: ProviderModelsRequest):
    provider = PROVIDERS_MAP.get(provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    loader = provider.get("loader")
    if not callable(loader):
        return {"models": [provider.get("default_model")]}
    try:
        models = loader(request.api_key)
        return {"models": models}
    except Exception as e:
        logger.warning(f"Failed to fetch models for {provider_id}: {e}")
        return {"models": [provider.get("default_model")]}


@app.get("/parameters/schema", dependencies=[Depends(verify_api_key)])
def get_parameter_schema():
    params = []
    for param in parameter_manager.schema.parameters.values():
        params.append({
            "name": param.name,
            "type": param.type.value,
            "default": param.default,
            "description": param.description,
            "category": param.category,
            "min_value": param.min_value,
            "max_value": param.max_value,
            "options": param.options,
            "required": param.required
        })
    return {"parameters": params}


@app.get("/parameters/defaults", dependencies=[Depends(verify_api_key)])
def get_parameter_defaults():
    return parameter_manager.get_defaults()


@app.get("/parameters/categories", dependencies=[Depends(verify_api_key)])
def get_parameter_categories():
    return {"categories": parameter_manager.get_categories()}


@app.post("/parameters/validate", dependencies=[Depends(verify_api_key)])
def validate_parameters(request: ParameterValidateRequest):
    result = parameter_manager.validate(request.parameters)
    return {"valid": result.valid, "errors": result.errors, "warnings": result.warnings}


# Sovereign dashboard endpoints

@app.get("/sovereign/health", dependencies=[Depends(verify_api_key)])
def get_sovereign_health():
    return sovereign_health_monitor.run_health_checks()


@app.get("/sovereign/problems", dependencies=[Depends(verify_api_key)])
def list_sovereign_problems():
    problems = sovereign_db.list_problems()
    return {"problems": [p.to_dict() for p in problems]}


@app.get("/sovereign/plans", dependencies=[Depends(verify_api_key)])
def list_sovereign_plans():
    plans = sovereign_db.list_plans()
    return {"plans": [p.to_dict() for p in plans]}


# Suggestions endpoints

@app.post("/suggestions/content", dependencies=[Depends(verify_api_key)])
def get_content_suggestions(request: SuggestionRequest):
    system_prompt = (
        "You are an AI assistant that provides suggestions for improving the given content. "
        "Provide a list of suggestions in a clear and concise manner."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    suggestions = [line.strip() for line in response.split("\n") if line.strip()]
    return {"suggestions": suggestions}


@app.post("/suggestions/classification", dependencies=[Depends(verify_api_key)])
def get_content_classification(request: SuggestionRequest):
    system_prompt = (
        "You are an AI assistant that classifies the given content and suggests relevant tags. "
        "Provide the classification and a list of tags in JSON format."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    try:
        import json
        parsed = json.loads(response)
    except Exception:
        parsed = {"classification": "", "tags": []}
    return parsed


@app.post("/suggestions/security", dependencies=[Depends(verify_api_key)])
def get_security_suggestions(request: SuggestionRequest):
    system_prompt = (
        "You are a security expert. Analyze the following code for common security vulnerabilities "
        "and provide a list of potential issues."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.content},
    ]
    response = _request_openai_chat(
        api_key=request.api_key,
        base_url=request.base_url,
        model=request.model,
        messages=messages,
        extra_headers=request.extra_headers,
        temperature=request.temperature,
        top_p=request.top_p,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        max_tokens=request.max_tokens,
        seed=request.seed,
    )
    vulnerabilities = [line.strip() for line in response.split("\n") if line.strip()]
    return {"vulnerabilities": vulnerabilities}


@app.post("/suggestions/improvement", dependencies=[Depends(verify_api_key)])
def get_improvement_potential(request: SuggestionRequest):
    suggestions = get_content_suggestions(request)
    classification = get_content_classification(request)
    score = 0.0
    score += len(suggestions.get("suggestions", [])) * 0.1
    score += len(classification.get("tags", [])) * 0.05
    score = min(1.0, score)
    return {"score": score}


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

# PyGraphistry Visualization Endpoint for BubbleLab Integration

class PyGraphistryVisualizationRequest(BaseModel):
    """Request model for PyGraphistry visualization."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None


@app.post("/api/openevolve/visualize/pygraphistry", dependencies=[Depends(verify_api_key)])
async def get_pygraphistry_visualization(request: PyGraphistryVisualizationRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Get a PyGraphistry visualization for knowledge graph data.
    This endpoint is specifically designed for BubbleLab integration.

    Args:
        request: Request body containing nodes and edges data
        user: Authenticated user information

    Returns:
        Dictionary with visualization URL or path
    """
    try:
        from openevolve_visualization import get_pygraphistry_viz

        # Call the visualization function with nodes and edges from request
        result = get_pygraphistry_viz(request.nodes, request.edges, request.config)

        if result:
            record_audit_event(
                user=user,
                operation="VISUALIZE_PYGRAPHISTRY",
                resource="visualization",
                resource_id="pygraphistry_viz",
                success=True
            )
            return {
                "status": "success",
                "visualization_url": result,
                "message": "PyGraphistry visualization generated successfully"
            }
        else:
            record_audit_event(
                user=user,
                operation="VISUALIZE_PYGRAPHISTRY_FAILED",
                resource="visualization",
                resource_id="pygraphistry_viz",
                success=False
            )
            return {
                "status": "error",
                "message": "Failed to generate PyGraphistry visualization"
            }

    except ImportError as e:
        logger.error(f"PyGraphistry import error: {e}")
        return {
            "status": "error",
            "message": "PyGraphistry integration not available"
        }
    except Exception as e:
        logger.error(f"Error in PyGraphistry visualization endpoint: {e}")
        record_audit_event(
            user=user,
            operation="VISUALIZE_PYGRAPHISTRY_ERROR",
            resource="visualization",
            resource_id="pygraphistry_viz",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error generating visualization: {str(e)}"
        }


# =============================================================================
# DSPY ENHANCED ASSESSMENT ENDPOINT
# =============================================================================

class DSPyAssessmentRequest(BaseModel):
    """Request model for DSPy-enhanced assessment."""
    content: str = Field(..., description="Content to assess")
    content_type: str = Field("general", description="Type of content (code, document, legal, etc.)")
    assessment_type: str = Field("comprehensive", description="Type of assessment (comprehensive, security, performance, logic)")


class DSPyAssessmentResponse(BaseModel):
    """Response model for DSPy-enhanced assessment."""
    status: str
    assessment_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    issues_found: Optional[int] = None
    recommendations: Optional[List[str]] = None
    message: Optional[str] = None


@app.post("/api/openevolve/assess/dspy", dependencies=[Depends(verify_api_key)], response_model=DSPyAssessmentResponse)
async def assess_content_with_dspy(request: DSPyAssessmentRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Assess content using DSPy for enhanced programmatic prompting and structured analysis.

    Args:
        request: Assessment request containing content and parameters
        user: Authenticated user information

    Returns:
        Assessment results from DSPy-enhanced analysis
    """
    try:
        from dspy_integration import DSPY_AVAILABLE

        if not DSPY_AVAILABLE:
            # Fallback to standard assessment if DSPy not available
            from quality_assessment import QualityAssessmentEngine

            engine = QualityAssessmentEngine()
            result = engine.assess_quality(request.content, request.content_type)

            record_audit_event(
                user=user,
                operation="ASSESS_CONTENT_DSPY_FALLBACK",
                resource="assessment",
                resource_id="dspy_fallback",
                success=True,
                details={"content_type": request.content_type, "assessment_type": request.assessment_type}
            )

            return {
                "status": "success",
                "assessment_result": {
                    "scores": {dim.value: score for dim, score in result.scores.items()},
                    "composite_score": result.composite_score,
                    "issues_count": len(result.issues),
                    "recommendations_count": len(result.recommendations)
                },
                "confidence_score": result.confidence,
                "issues_found": len(result.issues),
                "recommendations": result.recommendations[:5],  # First 5 recommendations
                "message": "DSPy not available, using standard assessment"
            }

        # Use DSPy-enhanced assessment
        from quality_assessment import QualityAssessmentEngine

        engine = QualityAssessmentEngine()
        result = engine.assess_quality_with_dspy(request.content, request.content_type)

        record_audit_event(
            user=user,
            operation="ASSESS_CONTENT_DSPY",
            resource="assessment",
            resource_id="dspy_enhanced",
            success=True,
            details={"content_type": request.content_type, "assessment_type": request.assessment_type}
        )

        return {
            "status": "success",
            "assessment_result": {
                "scores": {dim.value: score for dim, score in result.scores.items()},
                "composite_score": result.composite_score,
                "issues_count": len(result.issues),
                "recommendations_count": len(result.recommendations),
                "assessment_method": result.assessment_method
            },
            "confidence_score": result.confidence,
            "issues_found": len(result.issues),
            "recommendations": result.recommendations[:5],  # First 5 recommendations
            "message": "DSPy-enhanced assessment completed"
        }

    except ImportError as e:
        logger.error(f"DSPy assessment import error: {e}")
        return {
            "status": "error",
            "message": "DSPy assessment not available"
        }
    except Exception as e:
        logger.error(f"Error in DSPy assessment endpoint: {e}")
        record_audit_event(
            user=user,
            operation="ASSESS_CONTENT_DSPY_ERROR",
            resource="assessment",
            resource_id="dspy_error",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error performing DSPy assessment: {str(e)}"
        }


# =============================================================================
# DSPY ENHANCED FIX GENERATION ENDPOINT
# =============================================================================

class DSPyFixGenerationRequest(BaseModel):
    """Request model for DSPy-enhanced fix generation."""
    content: str = Field(..., description="Content to fix")
    content_type: str = Field("general", description="Type of content (code, document, legal, etc.)")
    issues: Optional[List[Dict[str, Any]]] = Field(None, description="List of issues to address")


class DSPyFixGenerationResponse(BaseModel):
    """Response model for DSPy-enhanced fix generation."""
    status: str
    fixed_content: Optional[str] = None
    suggested_fixes: Optional[List[Dict[str, Any]]] = None
    confidence_score: Optional[float] = None
    fixes_applied: Optional[int] = None
    message: Optional[str] = None


@app.post("/api/openevolve/fix/dspy", dependencies=[Depends(verify_api_key)], response_model=DSPyFixGenerationResponse)
async def generate_fixes_with_dspy(request: DSPyFixGenerationRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Generate fixes using DSPy for enhanced programmatic prompting and structured analysis.

    Args:
        request: Fix generation request containing content and issues
        user: Authenticated user information

    Returns:
        Fix generation results from DSPy-enhanced analysis
    """
    try:
        from dspy_integration import DSPY_AVAILABLE
        from blue_team import BlueTeam, IssueFinding
        from quality_assessment import SeverityLevel
        from red_team import IssueCategory

        if not DSPY_AVAILABLE:
            # Fallback to standard fix generation if DSPy not available
            blue_team = BlueTeam()

            # Convert issues to IssueFinding objects if provided
            issues = []
            if request.issues:
                for issue in request.issues:
                    issue_finding = IssueFinding(
                        title=issue.get("title", "Issue"),
                        description=issue.get("description", ""),
                        severity=SeverityLevel.MEDIUM,
                        category=IssueCategory.LOGICAL_ERROR,
                        confidence=issue.get("confidence", 0.5),
                        suggested_fix=issue.get("suggested_fix", ""),
                        location=issue.get("location", "")
                    )
                    issues.append(issue_finding)

            result = blue_team.apply_fixes(request.content, issues, content_type=request.content_type)

            record_audit_event(
                user=user,
                operation="GENERATE_FIXES_DSPY_FALLBACK",
                resource="fix_generation",
                resource_id="dspy_fallback",
                success=True,
                details={"content_type": request.content_type, "issues_count": len(issues)}
            )

            return {
                "status": "success",
                "fixed_content": result.fixed_content,
                "suggested_fixes": [fix.fix_description for fix in result.fix_suggestions],
                "confidence_score": result.confidence_score,
                "fixes_applied": len(result.applied_fixes),
                "message": "DSPy not available, using standard fix generation"
            }

        # Use DSPy-enhanced fix generation
        blue_team = BlueTeam()

        # Convert issues to IssueFinding objects if provided
        issues = []
        if request.issues:
            for issue in request.issues:
                issue_finding = IssueFinding(
                    title=issue.get("title", "Issue"),
                    description=issue.get("description", ""),
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.LOGICAL_ERROR,
                    confidence=issue.get("confidence", 0.5),
                    suggested_fix=issue.get("suggested_fix", ""),
                    location=issue.get("location", "")
                )
                issues.append(issue_finding)

        result = blue_team.generate_fixes_with_dspy(
            content=request.content,
            content_type=request.content_type,
            issues=issues
        )

        record_audit_event(
            user=user,
            operation="GENERATE_FIXES_DSPY",
            resource="fix_generation",
            resource_id="dspy_enhanced",
            success=True,
            details={"content_type": request.content_type, "issues_count": len(issues)}
        )

        return {
            "status": "success",
            "fixed_content": result.get("fixed_content", request.content),
            "suggested_fixes": result.get("suggested_fixes", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "fixes_applied": result.get("fix_count", 0),
            "message": "DSPy-enhanced fix generation completed"
        }

    except ImportError as e:
        logger.error(f"DSPy fix generation import error: {e}")
        return {
            "status": "error",
            "message": "DSPy fix generation not available"
        }
    except Exception as e:
        logger.error(f"Error in DSPy fix generation endpoint: {e}")
        record_audit_event(
            user=user,
            operation="GENERATE_FIXES_DSPY_ERROR",
            resource="fix_generation",
            resource_id="dspy_error",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Error performing DSPy fix generation: {str(e)}"
        }


# =============================================================================
# RAGBITS INTEGRATION ENDPOINTS
# =============================================================================

class RAGBitsSearchRequest(BaseModel):
    """Request model for RAGBits search."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class RAGBitsIngestRequest(BaseModel):
    """Request model for RAGBits ingest."""
    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    source: str = Field("manual", description="Document source identifier")


@app.post("/openevolve/ragbits/search", dependencies=[Depends(verify_api_key)])
async def ragbits_search(request: RAGBitsSearchRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Search documents using RAGBits semantic search.

    Args:
        request: Search request containing query and parameters
        user: Authenticated user information

    Returns:
        Search results from RAGBits
    """
    try:
        from knowledge_engine.ragbits_retriever import get_ragbits_retriever

        retriever = get_ragbits_retriever()

        # Perform search
        results = await retriever.search_similar_solutions(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            enable_hybrid_search=True
        )

        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH",
            resource="ragbits",
            resource_id="search",
            success=True
        )

        return {
            "status": "success",
            "results": results,
            "total_results": len(results),
            "query": request.query
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH_FAILED",
            resource="ragbits",
            resource_id="search",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error performing RAGBits search: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_SEARCH_ERROR",
            resource="ragbits",
            resource_id="search",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


@app.post("/openevolve/ragbits/ingest", dependencies=[Depends(verify_api_key)])
async def ragbits_ingest(request: RAGBitsIngestRequest, user: AuthUser = Depends(verify_api_key)):
    """
    Ingest a document into the RAGBits system.

    Args:
        request: Ingest request containing content and metadata
        user: Authenticated user information

    Returns:
        Ingestion result
    """
    try:
        from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, RAGBitsProcessorConfig

        # Initialize processor
        config = RAGBitsProcessorConfig()
        processor = RAGBitsDocumentProcessor(config)
        await processor.initialize()

        # Ingest document
        result = await processor.ingest_text(
            text=request.content,
            metadata=request.metadata,
            source=request.source
        )

        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST",
            resource="ragbits",
            resource_id=result.document_id,
            success=result.success
        )

        return {
            "status": "success" if result.success else "error",
            "document_id": result.document_id,
            "chunks_ingested": result.chunks_ingested,
            "processing_time": result.processing_time,
            "error": result.error
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST_FAILED",
            resource="ragbits",
            resource_id="ingest",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error ingesting document to RAGBits: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_INGEST_ERROR",
            resource="ragbits",
            resource_id="ingest",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


@app.get("/openevolve/ragbits/stats", dependencies=[Depends(verify_api_key)])
async def ragbits_stats(user: AuthUser = Depends(verify_api_key)):
    """
    Get RAGBits system statistics.

    Args:
        user: Authenticated user information

    Returns:
        System statistics
    """
    try:
        from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, RAGBitsProcessorConfig
        from knowledge_engine.ragbits_retriever import get_ragbits_retriever

        # Get processor stats
        config = RAGBitsProcessorConfig()
        processor = RAGBitsDocumentProcessor(config)
        await processor.initialize()
        processor_stats = await processor.get_statistics()

        # Get retriever stats
        retriever = get_ragbits_retriever()
        retriever_stats = await retriever.get_statistics()

        record_audit_event(
            user=user,
            operation="RAGBITS_STATS",
            resource="ragbits",
            resource_id="stats",
            success=True
        )

        return {
            "status": "success",
            "processor": processor_stats,
            "retriever": retriever_stats
        }

    except ImportError:
        error_msg = "RAGBits integration not available"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_STATS_FAILED",
            resource="ragbits",
            resource_id="stats",
            success=False,
            details={"error": error_msg}
        )
        return {
            "status": "error",
            "message": error_msg
        }
    except Exception as e:
        error_msg = f"Error getting RAGBits stats: {str(e)}"
        logger.error(error_msg)
        record_audit_event(
            user=user,
            operation="RAGBITS_STATS_ERROR",
            resource="ragbits",
            resource_id="stats",
            success=False,
            details={"error": str(e)}
        )
        return {
            "status": "error",
            "message": error_msg
        }


# =============================================================================
# Adaptive MDAP Endpoints
# =============================================================================

try:
    from adaptive_mdap import (
        TaskComplexityClassifier,
        AdaptiveMDAPAllocator,
        CostCalculator,
        APIPricing,
        get_health_checker,
        get_dashboard,
        ConfigProfile,
        get_profile_config,
    )
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    logger.warning("Adaptive MDAP not available, endpoints disabled")


class ComplexityRequest(BaseModel):
    """Request to classify problem complexity."""
    description: str
    domain: str = "general"
    depth: int = 0
    dependencies: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class AllocationRequest(BaseModel):
    """Request to allocate resources based on complexity."""
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None


class CostCalculationRequest(BaseModel):
    """Request to calculate costs."""
    num_problems: int = Field(..., ge=1, le=1000000)
    workload_distribution: Optional[Dict[str, float]] = None
    model: str = "gpt-4o-mini"


@app.get("/adaptive-mdap/health", dependencies=[Depends(verify_api_key)])
def adaptive_mdap_health():
    """Get Adaptive MDAP system health."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    health = get_health_checker()
    report = health.get_status_report()
    return report


@app.post("/adaptive-mdap/complexity", dependencies=[Depends(verify_api_key)])
def classify_complexity(request: ComplexityRequest):
    """Classify problem complexity."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        from adaptive_mdap.core.types import SubProblem
        
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id=f"api-{uuid.uuid4().hex[:8]}",
            description=request.description,
            domain=request.domain,
            depth=request.depth,
            dependencies=request.dependencies,
            metadata={
                "constraints": request.constraints,
                "success_criteria": request.success_criteria,
            },
        )
        
        complexity = classifier.compute_complexity(subproblem)
        
        return {
            "overall_score": complexity.overall_score,
            "text_length_score": complexity.text_length_score,
            "domain_rarity_score": complexity.domain_rarity_score,
            "depth_score": complexity.depth_score,
            "historical_error_score": complexity.historical_error_score,
            "dependency_score": complexity.dependency_score,
            "keyword_score": complexity.keyword_score,
            "constraint_score": complexity.constraint_score,
            "feature_weights": complexity.feature_weights,
        }
    except Exception as e:
        logger.error(f"Error classifying complexity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adaptive-mdap/allocate", dependencies=[Depends(verify_api_key)])
def allocate_resources(request: AllocationRequest):
    """Allocate resources based on complexity score."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        allocator = AdaptiveMDAPAllocator()
        
        from adaptive_mdap.allocators.resource_allocator import AllocationContext
        
        context = None
        if request.context:
            context = AllocationContext(
                system_load=request.context.get("system_load"),
                budget_remaining=request.context.get("budget_remaining"),
                quality_requirements=request.context.get("quality_requirements"),
            )
        
        config = allocator.allocate_resources(request.complexity_score, context)
        
        return {
            "strategy": config.strategy.value,
            "n_agents": config.n_agents,
            "k_ahead": config.k_ahead,
            "max_retries": config.max_retries,
            "timeout_ms": config.timeout_ms,
        }
    except Exception as e:
        logger.error(f"Error allocating resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adaptive-mdap/cost", dependencies=[Depends(verify_api_key)])
def calculate_cost(request: CostCalculationRequest):
    """Calculate costs for adaptive allocation."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        # Get pricing model
        pricing_map = {
            "gpt-4o-mini": APIPricing.gpt_4o_mini,
            "gpt-4o": APIPricing.gpt_4o,
            "gpt-4": APIPricing.gpt_4,
            "claude-3-5-sonnet": APIPricing.claude_3_5_sonnet,
            "claude-3-5-haiku": APIPricing.claude_3_5_haiku,
            "gemini-1-5-pro": APIPricing.gemini_1_5_pro,
            "gemini-1-5-flash": APIPricing.gemini_1_5_flash,
        }
        
        pricing = pricing_map.get(request.model, APIPricing.gpt_4o_mini)()
        calculator = CostCalculator(pricing=pricing)
        
        # Get workload distribution
        from adaptive_mdap.tools.cost_calculator import WorkloadDistribution
        
        if request.workload_distribution:
            workload = WorkloadDistribution(
                easy_percentage=request.workload_distribution.get("easy", 0.3),
                medium_percentage=request.workload_distribution.get("medium", 0.4),
                hard_percentage=request.workload_distribution.get("hard", 0.3),
            )
        else:
            workload = WorkloadDistribution.default()
        
        result = calculator.calculate_adaptive_cost(request.num_problems, workload)
        
        return {
            "model": request.model,
            "num_problems": request.num_problems,
            "baseline_cost": result["baseline_cost"],
            "adaptive_cost": result["adaptive_cost"],
            "savings": result["savings"],
            "savings_percent": result["savings_percent"],
            "breakdown": result["breakdown"],
        }
    except Exception as e:
        logger.error(f"Error calculating cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adaptive-mdap/dashboard", dependencies=[Depends(verify_api_key)])
def get_adaptive_dashboard():
    """Get Adaptive MDAP dashboard data."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    try:
        dashboard = get_dashboard()
        full_dashboard = dashboard.generate_full_dashboard()
        return full_dashboard
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/adaptive-mdap/profiles", dependencies=[Depends(verify_api_key)])
def get_adaptive_profiles():
    """Get available configuration profiles."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    profiles = {
        "conservative": "Favors quality over cost (lower thresholds)",
        "balanced": "Default balance between cost and quality",
        "aggressive": "Favors cost savings over quality (higher thresholds)",
        "cloud_conservative": "Cloud-optimized conservative profile",
        "cloud_balanced": "Cloud-optimized balanced profile",
        "cloud_aggressive": "Cloud-optimized aggressive profile",
    }
    
    return {
        "profiles": profiles,
        "default": "balanced",
    }


@app.get("/adaptive-mdap/profiles/{profile_name}", dependencies=[Depends(verify_api_key)])
def get_adaptive_profile_config(profile_name: str):
    """Get specific configuration profile."""
    if not ADAPTIVE_MDAP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Adaptive MDAP not available")
    
    profile_map = {
        "conservative": ConfigProfile.CONSERVATIVE,
        "balanced": ConfigProfile.BALANCED,
        "aggressive": ConfigProfile.AGGRESSIVE,
        "cloud_conservative": ConfigProfile.CLOUD_CONSERVATIVE,
        "cloud_balanced": ConfigProfile.CLOUD_BALANCED,
        "cloud_aggressive": ConfigProfile.CLOUD_AGGRESSIVE,
    }
    
    if profile_name not in profile_map:
        raise HTTPException(status_code=404, detail=f"Profile not found: {profile_name}")
    
    try:
        config = get_profile_config(profile_map[profile_name])
        return config
    except Exception as e:
        logger.error(f"Error getting profile config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    start_api_server()
