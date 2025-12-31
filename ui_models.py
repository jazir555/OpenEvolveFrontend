"""
Data models for UI components.
These models represent the data structures used across UI components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# Graph and Visualization Models
# ============================================================================

class NodeStatus(Enum):
    """Status of a node in dependency graph"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DependencyType(Enum):
    """Type of dependency relationship"""
    REQUIRES = "requires"
    BLOCKS = "blocks"
    RELATES_TO = "relates_to"


@dataclass
class GraphNode:
    """Node in a dependency graph"""
    id: str
    label: str
    status: NodeStatus
    team: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """Edge in a dependency graph"""
    source: str
    target: str
    dependency_type: DependencyType


@dataclass
class DependencyGraphData:
    """Complete dependency graph data"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    critical_path: List[str] = field(default_factory=list)
    circular_dependencies: List[List[str]] = field(default_factory=list)


# ============================================================================
# Analytics Models
# ============================================================================

@dataclass
class ResourceUsage:
    """Resource usage metrics"""
    cpu_percent: float
    memory_percent: float
    api_calls: int
    tokens_used: int
    cost: float


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics"""
    total_workflows: int
    success_rate: float
    avg_duration: float
    resource_usage: ResourceUsage
    throughput: float
    openevolve_api_calls: int = 0
    openevolve_tokens: int = 0
    openevolve_cost: float = 0.0
    evolution_iterations: int = 0


@dataclass
class TeamMetrics:
    """Team performance metrics"""
    team_type: str
    success_rate: float
    avg_quality_score: float
    resource_efficiency: float
    total_tasks: int
    openevolve_quality_scores: List[float] = field(default_factory=list)


@dataclass
class GauntletMetrics:
    """Gauntlet effectiveness metrics"""
    gauntlet_type: str
    detection_rate: float
    verification_accuracy: float
    false_positive_rate: float
    avg_execution_time: float


@dataclass
class QualityDataPoint:
    """Quality metric data point"""
    timestamp: datetime
    quality_score: float
    workflow_id: str


@dataclass
class KnowledgeStats:
    """Knowledge base statistics"""
    total_artifacts: int
    usage_frequency: Dict[str, int]
    effectiveness_scores: Dict[str, float]
    growth_rate: float


@dataclass
class AnalyticsData:
    """Complete analytics data"""
    workflow_metrics: WorkflowMetrics
    team_metrics: Dict[str, TeamMetrics]
    gauntlet_metrics: Dict[str, GauntletMetrics]
    quality_trends: List[QualityDataPoint]
    knowledge_stats: KnowledgeStats


# ============================================================================
# Knowledge Base Models
# ============================================================================

class ArtifactType(Enum):
    """Type of knowledge artifact"""
    PATTERN = "pattern"
    SOLUTION = "solution"
    ERROR = "error"
    BEST_PRACTICE = "best_practice"


@dataclass
class KnowledgeArtifact:
    """Knowledge artifact in the knowledge base"""
    id: str
    name: str
    type: ArtifactType
    content: str
    tags: List[str]
    source_workflow: str
    usage_count: int
    effectiveness_score: float
    related_artifacts: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Auto-Approval Models
# ============================================================================

class RuleOperator(Enum):
    """Operators for rule conditions"""
    LESS_THAN = "<"
    GREATER_THAN = ">"
    EQUALS = "=="
    NOT_EQUALS = "!="
    CONTAINS = "contains"
    IN = "in"


class LogicalOperator(Enum):
    """Logical operators for combining conditions"""
    AND = "AND"
    OR = "OR"


class RuleAction(Enum):
    """Action to take when rule matches"""
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass
class RuleCondition:
    """Condition in an auto-approval rule"""
    field: str
    operator: RuleOperator
    value: Any
    logical_op: LogicalOperator = LogicalOperator.AND


@dataclass
class AutoApprovalRule:
    """Auto-approval rule"""
    id: str
    name: str
    conditions: List[RuleCondition]
    action: RuleAction
    priority: int
    enabled: bool
    created_at: datetime


@dataclass
class AuditLogEntry:
    """Entry in auto-approval audit log"""
    timestamp: datetime
    rule_id: str
    rule_name: str
    plan_id: str
    decision: RuleAction
    details: Dict[str, Any]


# ============================================================================
# Template Models
# ============================================================================

@dataclass
class WorkflowTemplate:
    """Workflow configuration template"""
    id: str
    name: str
    description: str
    version: str
    config: Dict[str, Any]
    usage_count: int
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)


# ============================================================================
# Monitoring Models
# ============================================================================

class WorkflowState(Enum):
    """State of workflow execution"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class AlertSeverity(Enum):
    """Severity level of alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Alert:
    """Alert notification"""
    timestamp: datetime
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecutionState:
    """Current state of workflow execution"""
    workflow_id: str
    state: WorkflowState
    progress: float
    resource_usage: ResourceUsage
    completed_tasks: int
    total_tasks: int
    quality_scores: List[float]
    alerts: List[Alert]
    start_time: datetime
    last_update: datetime


# ============================================================================
# Batch Operations Models
# ============================================================================

@dataclass
class BatchOperation:
    """Batch operation to perform"""
    operation_type: str  # assign_team, assign_gauntlet, update_params
    target_ids: List[str]
    parameters: Dict[str, Any]
    timestamp: datetime


@dataclass
class BatchOperationResult:
    """Result of batch operation"""
    operation: BatchOperation
    succeeded: List[str]
    failed: Dict[str, str]  # id -> error message
    rollback_data: Dict[str, Any]


# OpenEvolve UI Models
@dataclass
class OpenEvolveConfig:
    """OpenEvolve configuration"""
    evolution_mode: str = "standard"
    max_iterations: int = 10
    population_size: int = 20
    temperature: float = 0.7
    max_tokens: int = 2048
    archive_size: Optional[int] = None
    feature_dimensions: Optional[List[str]] = None
    objectives: Optional[List[str]] = None

@dataclass
class OpenEvolveMetrics:
    """OpenEvolve operation metrics"""
    operation_id: str
    iterations_completed: int
    max_iterations: int
    best_fitness: float
    population_diversity: float
    api_calls: int
    tokens_total: int
    cost_usd: float
    time_taken: float

@dataclass
class EvolutionProgress:
    """Real-time evolution progress"""
    operation_id: str
    current_iteration: int
    max_iterations: int
    best_fitness: float
    fitness_history: List[float]
    population_diversity: float
    estimated_time_remaining: float

@dataclass
class ArchiveState:
    """Quality diversity archive state"""
    archive_size: int
    max_archive_size: int
    coverage: float
    avg_fitness: float
    entries: List[Dict[str, Any]]

@dataclass
class ParetoFront:
    """Multi-objective Pareto front"""
    solutions: List[Dict[str, Any]]
    num_objectives: int
    hypervolume: float
    spread: float
