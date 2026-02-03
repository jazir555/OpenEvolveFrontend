"""
Pydantic models for API Gateway request/response validation
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Schemas
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


# =============================================================================
# AUTHENTICATION MODELS
# =============================================================================

class UserRegister(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class Token(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str


class UserProfile(BaseModel):
    """User profile"""
    user_id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: str = "user"
    created_at: datetime
    updated_at: Optional[datetime] = None
    preferences: Dict[str, Any] = {}


class UserUpdate(BaseModel):
    """User profile update"""
    full_name: Optional[str] = Field(None, max_length=100)
    preferences: Optional[Dict[str, Any]] = None


# =============================================================================
# EVOLUTION MODELS
# =============================================================================

class EvolutionConfig(BaseModel):
    """Evolution parameters"""
    max_iterations: int = Field(100, ge=1, le=1000)
    population_size: int = Field(50, ge=10, le=500)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0)
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0)
    branching_mode: Literal["root", "lineage"] = "lineage"
    children_per_parent: int = Field(3, ge=1, le=100)
    survival_threshold: float = Field(0.6, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """LLM model configuration"""
    provider: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    api_key: str = Field(..., min_length=1)
    api_base: Optional[str] = None


class EvolutionStart(BaseModel):
    """Start evolution request"""
    content: str = Field(..., min_length=1)
    mode: Literal["standard", "quality_diversity", "island_model"] = "standard"
    parameters: EvolutionConfig
    models: List[ModelConfig] = Field(..., min_length=1)
    constraints: Optional[Dict[str, Any]] = None


class EvolutionStatus(BaseModel):
    """Evolution status response"""
    evolution_id: str
    status: Literal["running", "completed", "paused", "stopped", "error"]
    progress: Dict[str, Any]
    population: List[Dict[str, Any]]
    best_individual: Optional[Dict[str, Any]] = None
    metrics: Dict[str, float]
    started_at: datetime
    updated_at: datetime


class EvolutionListItem(BaseModel):
    """Evolution list item"""
    evolution_id: str
    status: str
    mode: str
    created_at: datetime
    updated_at: datetime
    best_fitness: Optional[float] = None
    iterations_completed: int


class EvolutionListResponse(BaseModel):
    """Evolution list response"""
    evolutions: List[EvolutionListItem]
    total: int
    limit: int
    offset: int


# =============================================================================
# ADVERSARIAL TESTING MODELS
# =============================================================================

class AdversarialStart(BaseModel):
    """Start adversarial testing request"""
    content: str = Field(..., min_length=1)
    attack_modes: List[Literal["prompt_injection", "jailbreak", "adversarial_example", "data_poisoning"]]
    parameters: Dict[str, Any]


class AdversarialStatus(BaseModel):
    """Adversarial test status response"""
    test_id: str
    status: Literal["running", "completed", "stopped", "error"]
    current_round: int
    total_rounds: int
    red_team_results: List[Dict[str, Any]]
    blue_team_results: List[Dict[str, Any]]
    vulnerabilities_found: int
    patches_generated: int
    patches_approved: int


class PatchApproval(BaseModel):
    """Patch approval request"""
    round: int = Field(..., ge=1)
    approved: bool
    feedback: Optional[str] = None


# =============================================================================
# CONTENT MANAGEMENT MODELS
# =============================================================================

class ContentCreate(BaseModel):
    """Create content request"""
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    language: Optional[str] = "python"
    tags: List[str] = []


class ContentUpdate(BaseModel):
    """Update content request"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None


class ContentResponse(BaseModel):
    """Content response"""
    content_id: str
    title: str
    content: str
    language: str
    tags: List[str]
    version: int
    created_at: datetime
    updated_at: datetime


class ContentListResponse(BaseModel):
    """Content list response"""
    content: List[ContentResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# VERSION CONTROL MODELS
# =============================================================================

class VersionInfo(BaseModel):
    """Version information"""
    version: int
    created_at: datetime
    created_by: str
    comment: Optional[str] = None


class VersionListResponse(BaseModel):
    """Version list response"""
    versions: List[VersionInfo]


class BranchCreate(BaseModel):
    """Create branch request"""
    branch_name: str = Field(..., min_length=1, max_length=100)
    from_version: int = Field(..., ge=1)


class BranchResponse(BaseModel):
    """Branch response"""
    branch_id: str
    branch_name: str
    version: int
    created_at: datetime


class DiffResponse(BaseModel):
    """Diff response"""
    version1: int
    version2: int
    diff: str


# =============================================================================
# COLLABORATION MODELS
# =============================================================================

class RoomCreate(BaseModel):
    """Create collaboration room request"""
    content_id: str
    room_name: Optional[str] = None


class RoomResponse(BaseModel):
    """Collaboration room response"""
    room_id: str
    room_name: str
    websocket_url: str
    created_at: datetime


class UserInfo(BaseModel):
    """User information"""
    user_id: str
    username: str
    joined_at: datetime
    cursor_position: Optional[Dict[str, int]] = None


class RoomUsersResponse(BaseModel):
    """Room users response"""
    users: List[UserInfo]


class CommentCreate(BaseModel):
    """Create comment request"""
    comment: str = Field(..., min_length=1)
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    parent_comment_id: Optional[str] = None


class CommentResponse(BaseModel):
    """Comment response"""
    comment_id: str
    user_id: str
    username: str
    comment: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    created_at: datetime
    replies: List["CommentResponse"] = []


class CommentsListResponse(BaseModel):
    """Comments list response"""
    comments: List[CommentResponse]


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class ProviderInfo(BaseModel):
    """Provider information"""
    provider: str
    name: str
    models: List[str]
    requires_api_key: bool


class ProvidersResponse(BaseModel):
    """Providers list response"""
    providers: List[ProviderInfo]


class APIKeySave(BaseModel):
    """Save API key request"""
    api_key: str = Field(..., min_length=1)


class APIKeyResponse(BaseModel):
    """API key response"""
    provider: str
    api_key_last_four: str
    saved_at: datetime


class ParametersConfig(BaseModel):
    """Parameters configuration"""
    generation: Optional[Dict[str, Any]] = None
    evolution: Optional[Dict[str, Any]] = None
    adversarial: Optional[Dict[str, Any]] = None


# =============================================================================
# ANALYTICS MODELS
# =============================================================================

class MetricsResponse(BaseModel):
    """Metrics response"""
    period: Dict[str, Any]
    metrics: Dict[str, float]
    time_series: List[Dict[str, Any]]


class PerformanceResponse(BaseModel):
    """Performance response"""
    model_performance: List[Dict[str, Any]]
    cost_analysis: Dict[str, float]


# =============================================================================
# MONITORING MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, str]
    resource_usage: Dict[str, float]
    active_operations: Dict[str, int]


class LogEntry(BaseModel):
    """Log entry"""
    timestamp: datetime
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    message: str
    context: Optional[Dict[str, Any]] = None


class LogsResponse(BaseModel):
    """Logs response"""
    logs: List[LogEntry]
    total: int


# =============================================================================
# WORKFLOW MODELS
# =============================================================================

class WorkflowStart(BaseModel):
    """Start workflow request"""
    problem_statement: str = Field(..., min_length=1)
    workflow_template: Literal["standard", "accelerated", "thorough"] = "standard"
    parameters: Optional[Dict[str, Any]] = {}


class WorkflowStatus(BaseModel):
    """Workflow status response"""
    workflow_id: str
    status: Literal["running", "completed", "failed", "stopped"]
    current_stage: str
    stages: List[Dict[str, Any]]


# =============================================================================
# FILE OPERATIONS MODELS
# =============================================================================

class FileMetadata(BaseModel):
    """File metadata"""
    file_id: str
    filename: str
    size: int
    mime_type: str
    uploaded_at: datetime


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: ErrorDetail


# =============================================================================
# WEBSOCKET MESSAGE MODELS
# =============================================================================

class WSMessage(BaseModel):
    """WebSocket message base"""
    type: str
    data: Dict[str, Any]


class WSProgressUpdate(WSMessage):
    """Progress update message"""
    type: Literal["progress_update"]
    data: Dict[str, Any]


class WSError(WSMessage):
    """Error message"""
    type: Literal["error"]
    data: Dict[str, Any]


# =============================================================================
# PAGINATION MODELS
# =============================================================================

class PaginatedResponse(BaseModel):
    """Paginated response base"""
    total: int
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    has_more: bool = False
