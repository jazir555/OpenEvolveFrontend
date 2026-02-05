"""
Sovereign-Grade Problem Decomposition System - Core Data Models
Implements all data structures for semantic decomposition with validation and serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json
import uuid


def generate_id(prefix: str = "item") -> str:
    """Generate a unique ID with optional prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# ENUMS - Task 1.1
# ============================================================================

class ProblemType(Enum):
    """Types of problems that can be decomposed"""
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    DESIGN = "design"


class SubProblemType(Enum):
    """Types of sub-problems"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    INTEGRATION = "integration"


class DecompositionStrategy(Enum):
    """Strategies for decomposing problems"""
    SEMANTIC = "semantic"
    DEPENDENCY = "dependency"
    COMPLEXITY = "complexity"
    RESEARCH = "research"
    HYBRID = "hybrid"


class SubProblemStatus(Enum):
    """Status of sub-problem resolution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"
    BLOCKED = "blocked"
    ERROR = "error"


class PlanStatus(Enum):
    """Status of decomposition plan"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    IN_EXECUTION = "in_execution"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# CORE DATA MODELS - Task 1.2
# ============================================================================

@dataclass
class Constraint:
    """Represents a problem constraint"""
    id: str
    description: str
    type: str  # time, resource, quality, technical
    severity: str  # hard, soft
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if self.type not in ["time", "resource", "quality", "technical"]:
            errors.append(f"Invalid constraint type: {self.type}")
        if self.severity not in ["hard", "soft"]:
            errors.append(f"Invalid constraint severity: {self.severity}")
        return errors


@dataclass
class SuccessCriterion:
    """Defines measurable success criteria"""
    id: str
    description: str
    metric: str
    threshold: float
    validation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuccessCriterion':
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if not (0.0 <= self.threshold <= 1.0):
            errors.append(f"SuccessCriterion threshold must be between 0.0 and 1.0, but got {self.threshold}")
        return errors


@dataclass
class DomainContext:
    """Problem domain information"""
    domain: str
    subdomain: Optional[str] = None
    related_domains: List[str] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainContext':
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if not self.domain:
            errors.append("DomainContext domain cannot be empty.")
        return errors


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment"""
    explanation: str
    cognitive_complexity: float  # 0-10
    computational_complexity: float  # 0-10
    domain_complexity: float  # 0-10
    integration_complexity: float  # 0-10
    overall_complexity: float  # 0-10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexityScore':
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        scores = {
            "cognitive_complexity": self.cognitive_complexity,
            "computational_complexity": self.computational_complexity,
            "domain_complexity": self.domain_complexity,
            "integration_complexity": self.integration_complexity,
            "overall_complexity": self.overall_complexity
        }
        for name, score in scores.items():
            if not (0.0 <= score <= 10.0):
                errors.append(f"ComplexityScore {name} must be between 0.0 and 10.0, but got {score}")
        return errors


@dataclass
class ProblemDefinition:
    """Complete problem definition"""
    id: str
    title: str
    description: str
    problem_type: ProblemType
    domain_context: DomainContext
    complexity_score: ComplexityScore
    parent_id: Optional[str] = None
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    resources_available: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['problem_type'] = self.problem_type.value
        data['domain_context'] = self.domain_context.to_dict()
        data['complexity_score'] = self.complexity_score.to_dict()
        data['constraints'] = [c.to_dict() for c in self.constraints]
        data['success_criteria'] = [s.to_dict() for s in self.success_criteria]
        if self.deadline:
            data['deadline'] = self.deadline.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.parent_id:
            data['parent_id'] = self.parent_id
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProblemDefinition':
        data = data.copy()
        data['problem_type'] = ProblemType(data['problem_type'])
        data['domain_context'] = DomainContext.from_dict(data['domain_context'])
        data['complexity_score'] = ComplexityScore.from_dict(data['complexity_score'])
        data['constraints'] = [Constraint.from_dict(c) for c in data.get('constraints', [])]
        data['success_criteria'] = [SuccessCriterion.from_dict(s) for s in data.get('success_criteria', [])]
        if data.get('deadline'):
            data['deadline'] = datetime.fromisoformat(data['deadline'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['parent_id'] = data.get('parent_id')
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if not self.title:
            errors.append("ProblemDefinition title cannot be empty.")
        if not self.description:
            errors.append("ProblemDefinition description cannot be empty.")
        errors.extend(self.domain_context.validate())
        errors.extend(self.complexity_score.validate())
        for constraint in self.constraints:
            errors.extend(constraint.validate())
        for criterion in self.success_criteria:
            errors.extend(criterion.validate())
        return errors


@dataclass
class SubProblem:
    """Verifiable sub-problem with clear success criteria"""
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType
    complexity_score: ComplexityScore
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    validation_gauntlet: str = ""
    assigned_team: Optional[str] = None
    estimated_effort: int = 1  # person-hours
    priority: int = 5  # 1-10
    execution_order: int = 0  # Execution sequence order
    dependency_outputs: Dict[str, Any] = field(default_factory=dict)  # Outputs from dependencies
    status: SubProblemStatus = SubProblemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    solution_attempts: List['SolutionAttempt'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set_execution_order(self, order: int) -> None:
        """Set the execution order for this sub-problem."""
        self.execution_order = order
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['type'] = self.type.value
        data['complexity_score'] = self.complexity_score.to_dict()
        data['success_criteria'] = [s.to_dict() for s in self.success_criteria]
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubProblem':
        data = data.copy()
        data['type'] = SubProblemType(data['type'])
        data['complexity_score'] = ComplexityScore.from_dict(data['complexity_score'])
        data['success_criteria'] = [SuccessCriterion.from_dict(s) for s in data.get('success_criteria', [])]
        data['status'] = SubProblemStatus(data.get('status', 'pending'))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data.pop('solution_attempts', None)  # Handle separately
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if not self.title:
            errors.append("SubProblem title cannot be empty.")
        if not self.description:
            errors.append("SubProblem description cannot be empty.")
        if not self.parent_id:
            errors.append("SubProblem parent_id cannot be empty.")
        errors.extend(self.complexity_score.validate())
        for criterion in self.success_criteria:
            errors.extend(criterion.validate())
        return errors


@dataclass
class DependencyGraph:
    """Represents dependency relationships"""
    nodes: Dict[str, SubProblem] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    critical_path: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'edges': self.edges,
            'critical_path': self.critical_path,
            'parallel_groups': self.parallel_groups,
            'execution_order': self.execution_order,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyGraph':
        data = data.copy()
        data['nodes'] = {k: SubProblem.from_dict(v) for k, v in data.get('nodes', {}).items()}
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        # Check for circular dependencies
        for node_id in self.nodes:
            path = {node_id}
            stack = [iter(self.edges.get(node_id, []))]
            while stack:
                children = stack[-1]
                try:
                    child = next(children)
                    if child in path:
                        errors.append(f"Circular dependency detected: {child} is already in path {path}")
                        continue
                    path.add(child)
                    stack.append(iter(self.edges.get(child, [])))
                except StopIteration:
                    path.remove(list(path)[-1])
                    stack.pop()

        # Check that all dependencies are valid sub-problems
        for node_id, dependencies in self.edges.items():
            if node_id not in self.nodes:
                errors.append(f"DependencyGraph edge source {node_id} is not a valid node.")
            for dep_id in dependencies:
                if dep_id not in self.nodes:
                    errors.append(f"DependencyGraph edge target {dep_id} is not a valid node.")
        return errors


@dataclass
class ValidationResult:
    """Result of validation check"""
    validator: str
    passed: bool
    score: float
    feedback: str
    improvements: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class QualityScores:
    """Comprehensive quality metrics"""
    coherence_score: float
    completeness_score: float
    feasibility_score: float
    integration_score: float
    overall_score: float
    meets_thresholds: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityScores':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class DecompositionPlan:
    """Complete decomposition plan"""
    id: str
    problem_id: str
    strategy: DecompositionStrategy
    sub_problems: List[SubProblem] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    validation_checkpoints: List['ValidationCheckpoint'] = field(default_factory=list)
    quality_scores: Optional[QualityScores] = None
    confidence_level: float = 0.0
    created_by: str = "system"
    approved_by: Optional[str] = None
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['strategy'] = self.strategy.value
        data['sub_problems'] = [sp.to_dict() for sp in self.sub_problems]
        if self.dependency_graph:
            data['dependency_graph'] = self.dependency_graph.to_dict()
        if self.quality_scores:
            data['quality_scores'] = self.quality_scores.to_dict()
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.error_message:
            data['error_message'] = self.error_message
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecompositionPlan':
        data = data.copy()
        data['strategy'] = DecompositionStrategy(data['strategy'])
        data['sub_problems'] = [SubProblem.from_dict(sp) for sp in data.get('sub_problems', [])]
        if data.get('dependency_graph'):
            data['dependency_graph'] = DependencyGraph.from_dict(data['dependency_graph'])
        if data.get('quality_scores'):
            data['quality_scores'] = QualityScores.from_dict(data['quality_scores'])
        data['status'] = PlanStatus(data.get('status', 'draft'))
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data.pop('validation_checkpoints', None)  # Handle separately
        data['error_message'] = data.get('error_message')
        return cls(**data)

    def validate(self) -> List[str]:
        errors = []
        if not self.problem_id:
            errors.append("DecompositionPlan problem_id cannot be empty.")
        for sub_problem in self.sub_problems:
            errors.extend(sub_problem.validate())
        if self.dependency_graph:
            errors.extend(self.dependency_graph.validate())
        return errors


@dataclass
class SolutionAttempt:
    """Tracks solution attempts for sub-problems"""
    id: str
    sub_problem_id: str
    approach: str
    solution_content: str
    team_id: str
    confidence_score: float
    validation_results: List[ValidationResult] = field(default_factory=list)
    feedback: List['Feedback'] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['validation_results'] = [vr.to_dict() for vr in self.validation_results]
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolutionAttempt':
        data = data.copy()
        data['validation_results'] = [ValidationResult.from_dict(vr) for vr in data.get('validation_results', [])]
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data.pop('feedback', None)  # Handle separately
        return cls(**data)


@dataclass
class Pattern:
    """Learned decomposition pattern"""
    id: str
    problem_type: ProblemType
    strategy: DecompositionStrategy
    pattern_description: str
    success_rate: float
    usage_count: int
    avg_quality_score: float
    applicable_domains: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['problem_type'] = self.problem_type.value
        data['strategy'] = self.strategy.value
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        data = data.copy()
        data['problem_type'] = ProblemType(data['problem_type'])
        data['strategy'] = DecompositionStrategy(data['strategy'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class TeamAssignment:
    """Team assignment for validation"""
    id: str
    task_id: str
    team: str
    assigned_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    status: str = "assigned"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['assigned_at'] = self.assigned_at.isoformat()
        if self.due_date:
            data['due_date'] = self.due_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamAssignment':
        data = data.copy()
        data['assigned_at'] = datetime.fromisoformat(data['assigned_at'])
        if data.get('due_date'):
            data['due_date'] = datetime.fromisoformat(data['due_date'])
        return cls(**data)


@dataclass
class Feedback:
    """Feedback from teams or gauntlets"""
    id: str
    source: str
    feedback_type: str
    content: str
    severity: str
    actionable: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ValidationCheckpoint:
    """Validation checkpoint in decomposition plan"""
    id: str
    name: str
    description: str
    validation_type: str
    required: bool = True
    passed: bool = False
    results: List[ValidationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['results'] = [r.to_dict() for r in self.results]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationCheckpoint':
        data = data.copy()
        data['results'] = [ValidationResult.from_dict(r) for r in data.get('results', [])]
        return cls(**data)



@dataclass
class HealthIssue:
    """Represents a health issue found in decomposition or solution.
    
    Attributes:
        id: Unique identifier for the issue
        issue_type: Type of issue (circular_dependency, complexity_imbalance, missing_dependency, etc.)
        severity: Severity level (critical, high, medium, low)
        description: Human-readable description of the issue
        affected_items: List of IDs of affected items
        suggested_fix: Optional suggested fix
        metadata: Additional metadata
    """
    id: str
    issue_type: str
    severity: str
    description: str
    affected_items: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealthIssue':
        return cls(**data)


@dataclass
class HealingResult:
    """Result of a healing operation.
    
    Attributes:
        id: Unique identifier for the healing result
        issue_id: ID of the issue that was addressed
        success: Whether the healing was successful
        strategy: Strategy used for healing
        changes_made: Description of changes made
        timestamp: When the healing was performed
        before_state: Snapshot of state before healing
        after_state: Snapshot of state after healing
        metadata: Additional metadata
    """
    id: str
    issue_id: str
    success: bool
    strategy: str
    changes_made: str
    timestamp: datetime = field(default_factory=datetime.now)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HealingResult':
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class EnhancedQualityScores:
    """Enhanced quality scores for decomposition and solutions.
    
    Attributes:
        overall_score: Overall quality score (0-100)
        coherence: Coherence score (0-100)
        completeness: Completeness score (0-100)
        correctness: Correctness score (0-100)
        clarity: Clarity score (0-100)
        complexity_balance: Complexity balance score (0-100)
        dependency_structure: Dependency structure score (0-100)
        testability: Testability score (0-100)
        maintainability: Maintainability score (0-100)
        metadata: Additional metadata
    """
    overall_score: float = 0.0
    coherence: float = 0.0
    completeness: float = 0.0
    correctness: float = 0.0
    clarity: float = 0.0
    complexity_balance: float = 0.0
    dependency_structure: float = 0.0
    testability: float = 0.0
    maintainability: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedQualityScores':
        return cls(**data)
    
    def get_average(self) -> float:
        """Calculate average of all scores."""
        scores = [
            self.coherence, self.completeness, self.correctness,
            self.clarity, self.complexity_balance, self.dependency_structure,
            self.testability, self.maintainability
        ]
        return sum(scores) / len(scores) if scores else 0.0

@dataclass
class GauntletRoundRule:
    """Rule for a single round in a gauntlet."""
    rule_id: str
    rule_type: str  # automated, red_team, gold_team, human
    description: str
    validation_type: str  # acceptance, quality, security, performance
    min_score: float
    max_attempts: int = 1
    evaluator: str = "automated"
    evaluation_prompt: str = ""
    success_criteria: List[str] = field(default_factory=list)
    is_required: bool = True
    can_fail_gracefully: bool = False
    retry_on_failure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GauntletDefinition:
    """Definition of a complete gauntlet."""
    gauntlet_id: str
    name: str
    description: str
    rounds: List[GauntletRoundRule]
    execution_order: str = "sequential"  # sequential, parallel, adaptive
    stop_on_first_failure: bool = False
    require_all_rounds: bool = True
    red_team_required: bool = False
    gold_team_required: bool = False
    blue_team_participation: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> List[str]:
        errors = []
        if not self.rounds:
            errors.append("Gauntlet must have at least one round")
        return errors

@dataclass
class GauntletExecution:
    """Record of a gauntlet execution."""
    execution_id: str
    gauntlet_definition: GauntletDefinition
    sub_problem_id: str
    solution_attempt: Any  # SolutionAttempt
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_duration: float = 0.0
    round_results: List[Dict[str, Any]] = field(default_factory=list)
    rounds_passed: int = 0
    rounds_failed: int = 0
    final_score: float = 0.0
    overall_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GauntletAssignment:
    """Assignment of a gauntlet to a task."""
    assignment_id: str
    gauntlet_id: str
    task_id: str
    assigned_by: str
    assigned_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

@dataclass
class CritiqueReport:
    """Detailed critique report from validation."""
    report_id: str
    solution_id: str
    critic_id: str
    overall_score: float
    passed: bool
    flaws: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    feedback: str = ""
    improvements: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
