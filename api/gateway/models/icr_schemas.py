"""
ICR (Iterative Contextual Refinements) Schemas

This module provides Pydantic v2 schemas for ICR-related data structures
across multiple components including QualityGateEngine, SGDWorkflowOrchestrator,
GauntletSystem, ROMA modules, and VLM analysis.

All schemas are compatible with Pydantic v2 and include proper validation
and type hints for type safety and consistency across the ICR integration.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum


# =============================================================================
# ICR ENUMS
# =============================================================================

class ICRPatternType(str, Enum):
    """Types of ICR patterns stored for learning."""
    CONTENT_TYPE = "content_type"
    QUALITY_LEVEL = "quality_level"
    METRIC = "metric"
    WORKFLOW = "workflow"
    PROBLEM_TYPE = "problem_type"
    COMPLEXITY = "complexity"
    TEAM_CONFIG = "team_config"
    GAUNTLET_CONFIG = "gauntlet_config"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    PLANNING = "planning"
    ATOMIZATION = "atomization"
    ROUTING = "routing"
    RESEARCH = "research"
    BLUE_TEAM_FIX = "blue_team_fix"
    BLUE_TEAM_RESEARCH = "blue_team_research"


class ICRRefinementType(str, Enum):
    """Types of refinements that can be applied."""
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_CHANGE = "strategy_change"
    CONFIG_UPDATE = "config_update"
    PROMPT_REFACTOR = "prompt_refactor"
    TEAM_RECONFIGURATION = "team_reconfiguration"
    GAUNTLET_RECONFIGURATION = "gauntlet_reconfiguration"


class ICRStatus(str, Enum):
    """Status of ICR operations."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    LEARNING = "learning"
    PREDICTING = "predicting"
    REFINING = "refining"


class VLMProvider(str, Enum):
    """Supported VLM providers for ICR analysis."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    MOCK = "mock"


class VLMAnalysisType(str, Enum):
    """Types of VLM analysis for ICR."""
    LAYOUT_ANALYSIS = "layout_analysis"
    INTERACTION_PATTERNS = "interaction_patterns"
    FRICTION_DETECTION = "friction_detection"
    HEATMAP_INTERPRETATION = "heatmap_interpretation"
    COMPREHENSIVE = "comprehensive"


# =============================================================================
# ICR PATTERN SCHEMAS
# =============================================================================

class ICRPatternMetrics(BaseModel):
    """Metrics associated with an ICR pattern."""
    model_config = ConfigDict(frozen=True)

    total_count: int = Field(..., ge=0, description="Total number of patterns in this group")
    success_count: int = Field(..., ge=0, description="Number of successful patterns")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Pass rate (0.0 to 1.0)")
    average_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average quality score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in pattern reliability")


class ICRPattern(BaseModel):
    """Base ICR pattern for storing learned patterns."""
    model_config = ConfigDict(extra="allow")

    pattern_id: str = Field(..., description="Unique identifier for this pattern")
    pattern_type: ICRPatternType = Field(..., description="Type of pattern")
    pattern_key: str = Field(..., description="Key used to group patterns (e.g., content_type_complexity)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When pattern was stored")

    # Context information
    content_type: Optional[str] = Field(None, description="Content type (code, text, design, etc.)")
    quality_level: Optional[str] = Field(None, description="Quality level (standard, high, premium)")
    complexity_score: Optional[int] = Field(None, ge=1, le=10, description="Problem complexity (1-10)")
    problem_type: Optional[str] = Field(None, description="Type of problem (design, implementation, etc.)")

    # Outcome information
    passed: bool = Field(..., description="Whether the evaluation passed")
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Historical pass rate for this pattern")

    # Additional context
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Individual metric scores")


class ICRPatternStore(BaseModel):
    """Container for ICR pattern storage."""
    model_config = ConfigDict(extra="allow")

    content_type_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by content_type and quality_level"
    )
    quality_level_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by quality level"
    )
    metric_patterns: Dict[str, Dict[str, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Patterns for individual metrics (metric -> score_range -> stats)"
    )
    problem_type_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by problem type"
    )
    complexity_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by complexity level"
    )
    team_config_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by team configuration hash"
    )
    gauntlet_config_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns grouped by gauntlet configuration hash"
    )
    execution_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for code execution operations"
    )
    verification_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for verification operations"
    )
    planning_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for planning operations"
    )
    atomization_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for atomization operations"
    )
    routing_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for routing operations"
    )
    research_patterns: Dict[str, List[ICRPattern]] = Field(
        default_factory=dict,
        description="Patterns for research operations"
    )
    operation_history: List[ICRPattern] = Field(
        default_factory=list,
        description="Chronological history of all patterns"
    )
    refinement_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of refinements applied"
    )


# =============================================================================
# ICR STATISTICS SCHEMAS
# =============================================================================

class ICRPatternStatistics(BaseModel):
    """Statistics for a specific pattern group."""
    model_config = ConfigDict(frozen=True)

    pattern_type: str = Field(..., description="Type of pattern")
    pattern_key: str = Field(..., description="Key for this pattern group")
    count: int = Field(..., ge=0, description="Number of patterns in this group")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Pass rate for this group")
    average_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Average score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in statistics")


class ICRComponentStatistics(BaseModel):
    """Statistics for a specific ICR component."""
    model_config = ConfigDict(frozen=True)

    component_name: str = Field(..., description="Name of the component")
    total_patterns: int = Field(..., ge=0, description="Total patterns stored")
    pattern_groups: Dict[str, int] = Field(
        default_factory=dict,
        description="Pattern count by group key"
    )
    overall_pass_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall pass rate")
    overall_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall quality score")
    history_size: int = Field(..., ge=0, description="Size of operation history")


class ICRStatistics(BaseModel):
    """Complete ICR statistics response."""
    model_config = ConfigDict(frozen=True)

    icr_enabled: bool = Field(..., description="Whether ICR is enabled")
    icr_status: ICRStatus = Field(default=ICRStatus.ENABLED, description="Current ICR status")

    # Overall statistics
    total_patterns: int = Field(..., ge=0, description="Total number of patterns across all stores")
    total_workflows: Optional[int] = Field(None, ge=0, description="Total number of workflows processed")
    average_duration_seconds: Optional[float] = Field(None, ge=0.0, description="Average workflow duration")

    # Pattern group statistics
    patterns_by_content_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Pattern count by content type"
    )
    patterns_by_quality_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Pattern count by quality level"
    )
    patterns_by_complexity: Dict[str, int] = Field(
        default_factory=dict,
        description="Pattern count by complexity level"
    )
    patterns_by_problem_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Pattern count by problem type"
    )

    # Configuration statistics
    unique_team_configs: Optional[int] = Field(None, ge=0, description="Number of unique team configurations")
    unique_gauntlet_configs: Optional[int] = Field(None, ge=0, description="Number of unique gauntlet configurations")

    # Component-specific statistics
    quality_gate_stats: Optional[ICRComponentStatistics] = Field(None, description="QualityGateEngine statistics")
    workflow_orchestrator_stats: Optional[ICRComponentStatistics] = Field(None, description="SGDWorkflowOrchestrator statistics")
    gauntlet_system_stats: Optional[ICRComponentStatistics] = Field(None, description="GauntletSystem statistics")
    robustness_stats: Optional[ICRComponentStatistics] = Field(None, description="RobustnessIntegration statistics")
    atomizer_stats: Optional[ICRComponentStatistics] = Field(None, description="Atomizer statistics")
    verifier_stats: Optional[ICRComponentStatistics] = Field(None, description="Verifier statistics")
    planner_stats: Optional[ICRComponentStatistics] = Field(None, description="Planner statistics")
    executor_stats: Optional[ICRComponentStatistics] = Field(None, description="Executor statistics")
    aggregator_stats: Optional[ICRComponentStatistics] = Field(None, description="Aggregator statistics")

    # Adaptive thresholds
    adaptive_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Current adaptive threshold adjustments"
    )

    # Refinement statistics
    total_refinements: Optional[int] = Field(None, ge=0, description="Total number of refinements applied")
    refinements_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Refinement count by type"
    )

    # Timestamps
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="When statistics were last updated")
    patterns_last_cleared: Optional[datetime] = Field(None, description="When patterns were last cleared")


# =============================================================================
# ICR CONFIGURATION SCHEMAS
# =============================================================================

class ICRVLMConfig(BaseModel):
    """VLM configuration for ICR analysis."""
    model_config = ConfigDict(extra="allow")

    provider: VLMProvider = Field(default=VLMProvider.OPENAI, description="VLM provider")
    model: str = Field(default="gpt-4o", description="Model name")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL for API")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=1024, ge=1, le=8192, description="Maximum tokens to generate")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")


class ICRHeatmapConfig(BaseModel):
    """Heatmap analysis configuration."""
    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable heatmap analysis")
    snapshot_interval: int = Field(default=10, ge=1, le=1000, description="Snapshot interval in seconds")
    max_snapshots: int = Field(default=100, ge=1, le=10000, description="Maximum snapshots to store")
    auto_analyze: bool = Field(default=True, description="Automatically analyze snapshots")
    vlm_analysis_enabled: bool = Field(default=False, description="Enable VLM analysis of heatmaps")


class ICRRefinementConfig(BaseModel):
    """Refinement configuration for ICR."""
    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable automatic refinements")
    max_cycles: int = Field(default=3, ge=1, le=10, description="Maximum refinement cycles")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Refinement trigger threshold")
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for refinement")
    auto_apply: bool = Field(default=False, description="Automatically apply refinements")


class ICRRewardCalibrationConfig(BaseModel):
    """Reward calibration configuration for ICR."""
    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(default=True, description="Enable reward calibration")
    threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Calibration threshold")
    max_queue_size: int = Field(default=100, ge=1, le=1000, description="Maximum queue size")
    timeout_seconds: int = Field(default=300, ge=10, le=3600, description="Request timeout in seconds")


class ICRPatternStorageConfig(BaseModel):
    """Pattern storage configuration for ICR."""
    model_config = ConfigDict(extra="allow")

    max_patterns_per_key: int = Field(default=100, ge=10, le=10000, description="Max patterns per key")
    max_history_size: int = Field(default=500, ge=100, le=10000, description="Max history size")
    max_refinement_history: int = Field(default=200, ge=50, le=5000, description="Max refinement history")
    persist_to_disk: bool = Field(default=False, description="Persist patterns to disk")
    storage_path: Optional[str] = Field(None, description="Path for pattern storage")


class ICRConfig(BaseModel):
    """Complete ICR configuration."""
    model_config = ConfigDict(extra="allow")

    # Core settings
    enabled: bool = Field(default=True, description="Enable ICR functionality")
    enable_prediction: bool = Field(default=True, description="Enable pass/fail prediction")
    enable_learning: bool = Field(default=True, description="Enable pattern learning")

    # Component-specific settings
    quality_gate_enabled: bool = Field(default=True, description="Enable QualityGateEngine ICR")
    workflow_orchestrator_enabled: bool = Field(default=True, description="Enable SGDWorkflowOrchestrator ICR")
    gauntlet_system_enabled: bool = Field(default=True, description="Enable GauntletSystem ICR")
    robustness_enabled: bool = Field(default=True, description="Enable RobustnessIntegration ICR")
    roma_modules_enabled: bool = Field(default=True, description="Enable ROMA module ICR")

    # Sub-configurations
    vlm: ICRVLMConfig = Field(default_factory=ICRVLMConfig, description="VLM configuration")
    heatmap: ICRHeatmapConfig = Field(default_factory=ICRHeatmapConfig, description="Heatmap configuration")
    refinement: ICRRefinementConfig = Field(default_factory=ICRRefinementConfig, description="Refinement configuration")
    reward_calibration: ICRRewardCalibrationConfig = Field(
        default_factory=ICRRewardCalibrationConfig,
        description="Reward calibration configuration"
    )
    pattern_storage: ICRPatternStorageConfig = Field(
        default_factory=ICRPatternStorageConfig,
        description="Pattern storage configuration"
    )

    # Adaptive settings
    adaptive_thresholds_enabled: bool = Field(default=True, description="Enable adaptive thresholds")
    min_pattern_count_for_adaptation: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Min patterns before adaptive thresholds apply"
    )


class UpdateICRConfig(BaseModel):
    """Update request for ICR configuration."""
    model_config = ConfigDict(extra="allow")

    enabled: Optional[bool] = Field(None, description="Enable ICR functionality")
    enable_prediction: Optional[bool] = Field(None, description="Enable pass/fail prediction")
    enable_learning: Optional[bool] = Field(None, description="Enable pattern learning")

    quality_gate_enabled: Optional[bool] = Field(None, description="Enable QualityGateEngine ICR")
    workflow_orchestrator_enabled: Optional[bool] = Field(None, description="Enable SGDWorkflowOrchestrator ICR")
    gauntlet_system_enabled: Optional[bool] = Field(None, description="Enable GauntletSystem ICR")
    robustness_enabled: Optional[bool] = Field(None, description="Enable RobustnessIntegration ICR")
    roma_modules_enabled: Optional[bool] = Field(None, description="Enable ROMA module ICR")

    vlm: Optional[ICRVLMConfig] = Field(None, description="VLM configuration")
    heatmap: Optional[ICRHeatmapConfig] = Field(None, description="Heatmap configuration")
    refinement: Optional[ICRRefinementConfig] = Field(None, description="Refinement configuration")
    reward_calibration: Optional[ICRRewardCalibrationConfig] = Field(None, description="Reward calibration configuration")
    pattern_storage: Optional[ICRPatternStorageConfig] = Field(None, description="Pattern storage configuration")

    adaptive_thresholds_enabled: Optional[bool] = Field(None, description="Enable adaptive thresholds")
    min_pattern_count_for_adaptation: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Min patterns before adaptive thresholds apply"
    )


# =============================================================================
# ICR PREDICTION SCHEMAS
# =============================================================================

class ICRPredictionRequest(BaseModel):
    """Request for ICR pass/fail prediction."""
    model_config = ConfigDict(extra="allow")

    assessments: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of metric assessments"
    )
    content_type: str = Field(..., description="Content type (code, text, design, etc.)")
    quality_level: str = Field(..., description="Quality level (standard, high, premium)")
    complexity_score: int = Field(..., ge=1, le=10, description="Problem complexity (1-10)")

    # Optional context
    problem_type: Optional[str] = Field(None, description="Type of problem")
    team_config: Optional[Dict[str, Any]] = Field(None, description="Team configuration")
    gauntlet_config: Optional[Dict[str, Any]] = Field(None, description="Gauntlet configuration")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ICRPredictionResponse(BaseModel):
    """Response from ICR pass/fail prediction."""
    model_config = ConfigDict(frozen=True)

    predicted_outcome: Literal["pass", "fail"] = Field(..., description="Predicted outcome")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of pass")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    reason: str = Field(..., description="Reason for prediction")

    # Additional information
    pattern_count: int = Field(..., ge=0, description="Number of patterns used for prediction")
    similar_patterns: List[ICRPattern] = Field(
        default_factory=list,
        description="Similar patterns used for prediction"
    )
    suggested_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Suggested threshold")
    recommended_action: Optional[str] = Field(None, description="Recommended action")


class ICRRecommendation(BaseModel):
    """ICR recommendation for configuration."""
    model_config = ConfigDict(frozen=True)

    recommendation_type: str = Field(..., description="Type of recommendation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    recommendation: str = Field(..., description="The recommendation itself")
    reasoning: str = Field(..., description="Reasoning behind recommendation")

    # Specific recommendations
    suggested_team_config: Optional[Dict[str, Any]] = Field(None, description="Suggested team configuration")
    suggested_gauntlet_config: Optional[Dict[str, Any]] = Field(None, description="Suggested gauntlet configuration")
    suggested_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Suggested threshold")

    # Metadata
    based_on_pattern_count: int = Field(..., ge=0, description="Number of patterns used")
    icr_insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional ICR insights"
    )


# =============================================================================
# ICR HEATMAP SNAPSHOT SCHEMAS
# =============================================================================

class IcrHeatmapPoint(BaseModel):
    """Heatmap point from UI interaction logging."""
    model_config = ConfigDict(frozen=True)

    x: float = Field(..., description="X coordinate (0-1 normalized)")
    y: float = Field(..., description="Y coordinate (0-1 normalized)")
    intensity: float = Field(default=0.0, ge=0.0, le=1.0, description="Interaction intensity")
    dwell_ms: Optional[float] = Field(None, ge=0.0, description="Dwell time in milliseconds")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    type: Optional[str] = Field(None, description="Interaction type (click, hover, etc.)")
    element_id: Optional[str] = Field(None, description="DOM element identifier")


class IcrHeatmapSnapshot(BaseModel):
    """Heatmap snapshot payload for multimodal analysis."""
    model_config = ConfigDict(extra="allow")

    snapshot_id: Optional[str] = Field(None, description="Unique snapshot identifier")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    screen_html: str = Field(..., description="Screen HTML content")
    heatmap_data_url: Optional[str] = Field(None, description="Base64-encoded heatmap image")
    composite_data_url: Optional[str] = Field(None, description="Base64-encoded composite image")
    points: List[IcrHeatmapPoint] = Field(
        default_factory=list,
        description="Heatmap interaction points"
    )
    manual_code_delta: Optional[float] = Field(None, description="Manual code change delta")
    context_text: Optional[str] = Field(None, description="Contextual text description")
    auto_refine: Optional[bool] = Field(None, description="Auto-refine flag")

    # Additional metadata
    page_url: Optional[str] = Field(None, description="Page URL")
    viewport_size: Optional[Dict[str, int]] = Field(None, description="Viewport dimensions")
    user_agent: Optional[str] = Field(None, description="User agent string")


class IcrHeatmapSnapshotResponse(BaseModel):
    """Response for heatmap snapshot submission."""
    model_config = ConfigDict(frozen=True)

    queued: bool = Field(..., description="Whether snapshot was queued")
    snapshot_id: str = Field(..., description="Snapshot identifier")
    received_at: str = Field(..., description="ISO timestamp when received")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    vlm_analysis: Optional[Dict[str, Any]] = Field(None, description="VLM analysis results")


# =============================================================================
# ICR VLM ANALYSIS SCHEMAS
# =============================================================================

class VLMAnalysisRequest(BaseModel):
    """Request for VLM analysis."""
    model_config = ConfigDict(extra="allow")

    image_data: str = Field(..., description="Base64-encoded image data")
    prompt: Optional[str] = Field(None, description="Custom analysis prompt")
    analysis_type: VLMAnalysisType = Field(
        default=VLMAnalysisType.COMPREHENSIVE,
        description="Type of analysis to perform"
    )
    provider: Optional[VLMProvider] = Field(None, description="VLM provider override")
    model: Optional[str] = Field(None, description="Model override")


class VLMAnalysisResult(BaseModel):
    """Result of VLM analysis."""
    model_config = ConfigDict(frozen=True)

    summary: str = Field(..., description="Summary of analysis")
    insights: List[str] = Field(default_factory=list, description="Key insights from analysis")
    friction_points: List[str] = Field(default_factory=list, description="Detected friction points")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    provider: str = Field(..., description="VLM provider used")
    model: str = Field(..., description="Model used for analysis")
    tokens_used: int = Field(default=0, ge=0, description="Tokens consumed")
    raw_response: str = Field(default="", description="Raw VLM response")

    # Additional metadata
    analysis_type: Optional[str] = Field(None, description="Type of analysis performed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When analysis was performed")
    cached: bool = Field(default=False, description="Whether result was from cache")


class VLMConfigStatus(BaseModel):
    """VLM configuration status."""
    model_config = ConfigDict(frozen=True)

    available: bool = Field(..., description="Whether VLM is available")
    enabled: bool = Field(..., description="Whether VLM is enabled")
    configured: bool = Field(..., description="Whether VLM is properly configured")
    config: Optional[ICRVLMConfig] = Field(None, description="Current VLM configuration")
    error: Optional[str] = Field(None, description="Error message if not available")


# =============================================================================
# ICR REFINEMENT SCHEMAS
# =============================================================================

class ICRRefinementEvent(BaseModel):
    """Event signaling a refinement is needed."""
    model_config = ConfigDict(extra="allow")

    event_id: Optional[str] = Field(None, description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When event occurred")

    reason: Optional[str] = Field(None, description="Reason for refinement")
    overall_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall score")
    weaknesses: Optional[List[str]] = Field(None, description="Identified weaknesses")
    friction_points: Optional[List[str]] = Field(None, description="Detected friction points")
    auto_refine: Optional[bool] = Field(None, description="Auto-refine flag")

    # Context
    component: Optional[str] = Field(None, description="Component that triggered refinement")
    pattern_key: Optional[str] = Field(None, description="Pattern key for this event")


class ICRRefinement(BaseModel):
    """Applied refinement."""
    model_config = ConfigDict(frozen=True)

    refinement_id: str = Field(..., description="Unique refinement identifier")
    refinement_type: ICRRefinementType = Field(..., description="Type of refinement")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When refinement was applied")

    # Before and after
    before: Dict[str, Any] = Field(..., description="State before refinement")
    after: Dict[str, Any] = Field(..., description="State after refinement")
    delta: Dict[str, Any] = Field(..., description="Changes applied")

    # Metadata
    triggered_by: str = Field(..., description="What triggered the refinement")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in refinement")
    success: bool = Field(..., description="Whether refinement was successful")


# =============================================================================
# ICR REWARD CALIBRATION SCHEMAS
# =============================================================================

class IcrRewardCalibrationRequest(BaseModel):
    """Reward calibration request payload."""
    model_config = ConfigDict(extra="allow")

    request_id: Optional[str] = Field(None, description="Unique request identifier")
    option_a: str = Field(..., description="Option A")
    option_b: str = Field(..., description="Option B")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in preference")
    prompt: Optional[str] = Field(None, description="Additional prompt context")


class IcrRewardCalibrationResponse(BaseModel):
    """Reward calibration response payload."""
    model_config = ConfigDict(extra="allow")

    request_id: Optional[str] = Field(None, description="Request identifier being responded to")
    choice: str = Field(..., description="Chosen option (a or b)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in choice")
    reasoning: Optional[str] = Field(None, description="Reasoning for choice")


class IcrRewardCalibrationQueueItem(BaseModel):
    """Item in the reward calibration queue."""
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., description="Unique request identifier")
    option_a: str = Field(..., description="Option A")
    option_b: str = Field(..., description="Option B")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in preference")
    prompt: Optional[str] = Field(None, description="Additional prompt context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When queued")


# =============================================================================
# ICR ERROR SCHEMAS
# =============================================================================

class ICRErrorResponse(BaseModel):
    """Error response for ICR operations."""
    model_config = ConfigDict(frozen=True)

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    request_id: Optional[str] = Field(None, description="Associated request identifier")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ICRPatternType",
    "ICRRefinementType",
    "ICRStatus",
    "VLMProvider",
    "VLMAnalysisType",
    
    # Pattern schemas
    "ICRPatternMetrics",
    "ICRPattern",
    "ICRPatternStore",
    
    # Statistics schemas
    "ICRPatternStatistics",
    "ICRComponentStatistics",
    "ICRStatistics",
    
    # Configuration schemas
    "ICRVLMConfig",
    "ICRHeatmapConfig",
    "ICRRefinementConfig",
    "ICRRewardCalibrationConfig",
    "ICRPatternStorageConfig",
    "ICRConfig",
    "UpdateICRConfig",
    
    # Prediction schemas
    "ICRPredictionRequest",
    "ICRPredictionResponse",
    "ICRRecommendation",
    
    # Heatmap schemas
    "IcrHeatmapPoint",
    "IcrHeatmapSnapshot",
    "IcrHeatmapSnapshotResponse",
    
    # VLM schemas
    "VLMAnalysisRequest",
    "VLMAnalysisResult",
    "VLMConfigStatus",
    
    # Refinement schemas
    "ICRRefinementEvent",
    "ICRRefinement",
    
    # Reward calibration schemas
    "IcrRewardCalibrationRequest",
    "IcrRewardCalibrationResponse",
    "IcrRewardCalibrationQueueItem",
    
    # Error schemas
    "ICRErrorResponse",
]
