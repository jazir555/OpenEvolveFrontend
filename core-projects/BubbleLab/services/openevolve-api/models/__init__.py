"""Pydantic models for OpenEvolve API"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime


# ==================== Workflow & Execution Models ====================

class WorkflowStatus(str, Enum):
    """Workflow lifecycle status (mirrors frontend types)"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowMetadata(BaseModel):
    """Optional workflow metadata for advanced features"""
    mdap_enabled: Optional[bool] = None
    maker_enabled: Optional[bool] = None
    maker_config: Optional[Dict[str, Any]] = None
    adaptive_config: Optional[Dict[str, Any]] = None
    evolution_params: Optional[Dict[str, Any]] = None
    performance_params: Optional[Dict[str, Any]] = None


class WorkflowCreate(BaseModel):
    """Workflow creation request (OpenEvolve UI)"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    problem_statement: Optional[str] = Field(None, min_length=10, max_length=10000)
    content_type: str = Field(default="text", max_length=50)
    teams: List[str] = Field(default_factory=list)
    gauntlets: List[str] = Field(default_factory=list)
    metadata: Optional[WorkflowMetadata] = None
    parameters: Optional[Dict[str, Any]] = None
    workflow_type: Literal["evolution", "adversarial", "sovereign"] = "sovereign"


class WorkflowUpdate(BaseModel):
    """Workflow update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    problem_statement: Optional[str] = Field(None, min_length=10, max_length=10000)
    content_type: Optional[str] = Field(None, max_length=50)
    teams: Optional[List[str]] = None
    gauntlets: Optional[List[str]] = None
    metadata: Optional[WorkflowMetadata] = None
    parameters: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    """Workflow response"""
    id: str
    name: str
    description: Optional[str] = None
    problem_statement: str
    content_type: str
    teams: List[str]
    gauntlets: List[str]
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    user_id: str
    tenant_id: str
    metadata: Optional[WorkflowMetadata] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    workflow_type: Literal["evolution", "adversarial", "sovereign"] = "sovereign"


class ExecutionStartRequest(BaseModel):
    """Direct execution start request"""
    workflow_id: str = Field(..., min_length=1)
    problem_statement: Optional[str] = None
    context: Optional[str] = None


class WorkflowListResponse(BaseModel):
    """Workflow list response"""
    workflows: List[WorkflowResponse]
    total: int
    page: int = 1
    page_size: int = 50


class WorkflowInputs(BaseModel):
    """Inputs for workflow execution"""
    problem_statement: Optional[str] = Field(
        None,
        description="Problem statement to solve (defaults to workflow stored problem statement)"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context or constraints"
    )


class ExecutionResponse(BaseModel):
    """Execution response"""
    execution_id: str
    workflow_id: str
    status: Literal["queued", "running", "paused", "completed", "failed", "cancelled"]
    progress: float = Field(ge=0.0, le=1.0)
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


class ExecutionStatusResponse(ExecutionResponse):
    """Execution status response (same as ExecutionResponse)"""
    pass


class ExecutionLogsResponse(BaseModel):
    """Execution logs response"""
    logs: List[Dict[str, Any]]
    total: int
    since: Optional[datetime]


class SubProblemResult(BaseModel):
    """Sub-problem result"""
    subproblem_id: str
    problem: str
    solution: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None


class ExecutionStatistics(BaseModel):
    """Execution statistics"""
    total_duration_seconds: float = 0.0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    sub_problems_solved: int = 0
    success_rate: float = 0.0
    memory_used_mb: float = 0.0
    cpu_time_seconds: float = 0.0


class ExecutionResult(BaseModel):
    """Execution result response"""
    workflow_id: str
    status: WorkflowStatus
    final_solution: str
    sub_problems: List[SubProblemResult]
    statistics: ExecutionStatistics
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None


# ==================== Team Models ====================

class TeamMember(BaseModel):
    """Team member definition"""
    id: Optional[str] = None
    name: str
    role: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_iterations: int = 5


class TeamCreate(BaseModel):
    """Team creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    members: List[TeamMember]


class TeamUpdate(BaseModel):
    """Team update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    members: Optional[List[TeamMember]] = None


class TeamResponse(BaseModel):
    """Team response"""
    id: str
    name: str
    description: Optional[str] = None
    members: List[TeamMember]
    created_at: datetime
    updated_at: datetime
    user_id: str
    tenant_id: str


class TeamListResponse(BaseModel):
    """Team list response"""
    teams: List[TeamResponse]
    total: int


# ==================== Gauntlet Models ====================

class GauntletRound(BaseModel):
    """Gauntlet round definition"""
    id: Optional[str] = None
    name: str
    quorum_threshold: float = Field(ge=0.0, le=1.0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    evaluation_type: str
    required_consensus: bool = False
    max_iterations: int = 1


class GauntletCreate(BaseModel):
    """Gauntlet creation request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    rounds: List[GauntletRound]


class GauntletUpdate(BaseModel):
    """Gauntlet update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    rounds: Optional[List[GauntletRound]] = None


class GauntletResponse(BaseModel):
    """Gauntlet response"""
    id: str
    name: str
    description: Optional[str] = None
    rounds: List[GauntletRound]
    created_at: datetime
    updated_at: datetime
    user_id: str
    tenant_id: str


class GauntletListResponse(BaseModel):
    """Gauntlet list response"""
    gauntlets: List[GauntletResponse]
    total: int


# ==================== Settings Models ====================

class LLMConfig(BaseModel):
    """LLM provider configuration"""
    provider: Literal["openai", "anthropic", "cohere", "custom"] = "openai"
    api_key: str = ""
    base_url: Optional[str] = None
    model_leanaide: str = "gpt-4"
    model_text: str = "gpt-4"
    model_img: str = "gpt-4-vision"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class UpdateLLMConfig(BaseModel):
    """LLM config update request"""
    provider: Optional[Literal["openai", "anthropic", "cohere", "custom"]] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_leanaide: Optional[str] = None
    model_text: Optional[str] = None
    model_img: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ICRConfig(BaseModel):
    """ICR configuration"""
    auto_refine_enabled: bool = False
    reward_calibration_enabled: bool = True
    reward_calibration_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    heatmap_analysis_enabled: bool = True
    heatmap_snapshot_interval: int = Field(default=10, ge=1, le=1000)
    vlm_provider: Optional[str] = None
    vlm_model: Optional[str] = None


class UpdateICRConfig(BaseModel):
    """ICR config update request"""
    auto_refine_enabled: Optional[bool] = None
    reward_calibration_enabled: Optional[bool] = None
    reward_calibration_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    heatmap_analysis_enabled: Optional[bool] = None
    heatmap_snapshot_interval: Optional[int] = Field(default=None, ge=1, le=1000)
    vlm_provider: Optional[str] = None
    vlm_model: Optional[str] = None


class DeterminismDefaults(BaseModel):
    """Defaults for determinism endpoints"""
    mode: str = "auto"
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None
    cloud_base_url: Optional[str] = None
    local_provider: Optional[str] = "hf"
    local_model: Optional[str] = None
    local_device: Optional[str] = "cpu"
    local_dtype: Optional[str] = "auto"
    config: Dict[str, Any] = Field(default_factory=dict)
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None
    check_tier: int = Field(default=2, ge=1, le=5)
    check_runs: int = Field(default=3, ge=1, le=20)
    check_provider: Optional[str] = None
    check_model: Optional[str] = None
    check_base_url: Optional[str] = None
    check_device: Optional[str] = "cpu"
    check_dtype: Optional[str] = "auto"


class UpdateDeterminismDefaults(BaseModel):
    """Determinism defaults update request"""
    mode: Optional[str] = None
    cloud_provider: Optional[str] = None
    cloud_model: Optional[str] = None
    cloud_base_url: Optional[str] = None
    local_provider: Optional[str] = None
    local_model: Optional[str] = None
    local_device: Optional[str] = None
    local_dtype: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    detllm_backend: Optional[str] = None
    detllm_model: Optional[str] = None
    check_tier: Optional[int] = Field(default=None, ge=1, le=5)
    check_runs: Optional[int] = Field(default=None, ge=1, le=20)
    check_provider: Optional[str] = None
    check_model: Optional[str] = None
    check_base_url: Optional[str] = None
    check_device: Optional[str] = None
    check_dtype: Optional[str] = None


class DecompositionDefaults(BaseModel):
    """Defaults for decomposition planning + workflow creation"""
    strategy: Optional[str] = None
    enable_adaptive_selection: bool = True
    maker_config: Dict[str, Any] = Field(default_factory=dict)
    openevolve_client_config: Dict[str, Any] = Field(default_factory=dict)
    mdap_enabled: bool = False
    mdap_config: Dict[str, Any] = Field(default_factory=dict)
    maker_enabled: bool = False
    workflow_max_refinement_loops: int = Field(default=3, ge=1, le=10)


class UpdateDecompositionDefaults(BaseModel):
    """Decomposition defaults update request"""
    strategy: Optional[str] = None
    enable_adaptive_selection: Optional[bool] = None
    maker_config: Optional[Dict[str, Any]] = None
    openevolve_client_config: Optional[Dict[str, Any]] = None
    mdap_enabled: Optional[bool] = None
    mdap_config: Optional[Dict[str, Any]] = None
    maker_enabled: Optional[bool] = None
    workflow_max_refinement_loops: Optional[int] = Field(default=None, ge=1, le=10)


class AdaptiveDecompositionDefaults(BaseModel):
    """Defaults for adaptive decomposition integration"""
    strategy_hint: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class UpdateAdaptiveDecompositionDefaults(BaseModel):
    """Adaptive decomposition defaults update request"""
    strategy_hint: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class MDAPMakerDefaults(BaseModel):
    """Defaults for MDAP/MAKER recomposition workflow"""
    use_mdap: bool = True
    use_associative: bool = True
    num_mdap_agents: int = Field(default=5, ge=1, le=20)
    llm_config: Dict[str, Any] = Field(default_factory=dict)


class UpdateMDAPMakerDefaults(BaseModel):
    """MDAP/MAKER defaults update request"""
    use_mdap: Optional[bool] = None
    use_associative: Optional[bool] = None
    num_mdap_agents: Optional[int] = Field(default=None, ge=1, le=20)
    llm_config: Optional[Dict[str, Any]] = None


class ROMAMDAPMakerDefaults(BaseModel):
    """Defaults for ROMA-MDAP-MAKER associative pipeline"""
    recursive: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)


class UpdateROMAMDAPMakerDefaults(BaseModel):
    """ROMA-MDAP-MAKER defaults update request"""
    recursive: Optional[bool] = None
    config: Optional[Dict[str, Any]] = None
