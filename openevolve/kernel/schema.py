"""
OpenEvolve Kernel Schema (Unified)
Merges sovereign_data_models.py, openevolve_structures.py, and workflow_structures.py
into a single source of truth for the OpenEvolve Mega-Structure.
"""

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Literal, Set, Union
import time
import uuid
from datetime import datetime
from enum import Enum
import json

# ============================================================================
# ENUMS
# ============================================================================

class ProblemType(Enum):
    """Types of problems that can be decomposed"""
    RESEARCH = "research"
    IMPLEMENTATION = "implementation"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    DESIGN = "design"
    VALIDATION = "validation"


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
    REQUIRES_REWORK = "requires_rework"


class PlanStatus(Enum):
    """Status of decomposition plan"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    IN_EXECUTION = "in_execution"
    COMPLETED = "completed"
    FAILED = "failed"


class MathematicalDomain(Enum):
    """Enumeration of mathematical domains for classification and verification."""
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    GEOMETRY = "geometry"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    CATEGORY_THEORY = "category_theory"
    LINEAR_ALGEBRA = "linear_algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    GENERAL = "general"


class VerificationMethod(Enum):
    """Enumeration of verification methods available in the system."""
    MANUAL = "manual"
    AUTOMATED_TESTING = "automated_testing"
    PEER_REVIEW = "peer_review"
    LEAN4 = "lean4"
    Z3 = "z3"
    HYBRID = "hybrid"
    STATISTICAL = "statistical"
    CROSS_VALIDATION = "cross_validation"


class LeanProofStatus(Enum):
    """Status of a Lean 4 proof verification."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class GauntletRoundStatus(Enum):
    """Status of a gauntlet round execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# ============================================================================
# BASIC TYPES
# ============================================================================

def generate_id(prefix: str = "item") -> str:
    """Generate a unique ID with optional prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


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


@dataclass
class Web3Context:
    """Web3 specific context information"""
    network: str = "ethereum"
    contract_address: Optional[str] = None
    source_code: Optional[str] = None
    compiler_version: Optional[str] = None
    audit_findings: List[Dict[str, Any]] = field(default_factory=list)
    formal_verification_status: str = "pending"
    gas_optimization_score: float = 0.0
    security_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Web3Context':
        return cls(**data)


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


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment"""
    explanation: str
    cognitive_complexity: float = 0.0  # 0-10
    computational_complexity: float = 0.0 # 0-10
    domain_complexity: float = 0.0 # 0-10
    integration_complexity: float = 0.0 # 0-10
    overall_complexity: float = 0.0 # 0-10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplexityScore':
        return cls(**data)


@dataclass
class ModelConfig:
    """Configuration for a single AI model within a team."""
    model_id: str
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    n: Optional[int] = 1
    logit_bias: Optional[Dict[int, int]] = None
    reasoning_effort: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    stream: Optional[bool] = None
    user: Optional[str] = None
    max_retries: int = 5
    timeout: int = 120
    organization: Optional[str] = None
    response_model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    system_fingerprint: Optional[str] = None
    deployment_id: Optional[str] = None
    encoding_format: Optional[str] = None
    max_input_tokens: Optional[int] = None
    stop_token: Optional[str] = None
    best_of: Optional[int] = None
    logprobs_offset: Optional[int] = None
    suffix: Optional[str] = None
    presence_penalty_range: Optional[List[float]] = None
    frequency_penalty_range: Optional[List[float]] = None
    stop_token_id: Optional[int] = None
    response_json_format: Optional[bool] = None
    max_output_tokens: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None
    logprobs_type: Optional[str] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature_fallback: Optional[float] = None
    top_p_fallback: Optional[float] = None
    max_time: Optional[int] = None
    return_full_text: Optional[bool] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    domain_specialization: Optional[List[str]] = None
    problem_type_specialization: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    cost_per_token: Optional[float] = None


@dataclass
class Team:
    """A user-defined group of AI models assigned to a specific role."""
    name: str
    role: Literal["Blue", "Red", "Gold"]
    members: List[ModelConfig]
    tenant_id: Optional[str] = None
    description: Optional[str] = None
    content_analysis_system_prompt: Optional[str] = None
    content_analysis_user_prompt_template: Optional[str] = None
    decomposition_system_prompt: Optional[str] = None
    decomposition_user_prompt_template: Optional[str] = None
    solver_system_prompt: Optional[str] = None
    solver_user_prompt_template: Optional[str] = None
    patcher_system_prompt: Optional[str] = None
    patcher_user_prompt_template: Optional[str] = None
    assembler_system_prompt: Optional[str] = None
    assembler_user_prompt_template: Optional[str] = None
    red_team_system_prompt: Optional[str] = None
    red_team_user_prompt_template: Optional[str] = None
    gold_team_system_prompt: Optional[str] = None
    gold_team_user_prompt_template: Optional[str] = None
    sub_role: Optional[str] = None
    domain_specialization: Optional[List[str]] = None
    problem_type_specialization: Optional[List[str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    team_config: Optional[Dict[str, Any]] = None
    openevolve_metrics: Optional[List[Dict[str, Any]]] = None
    team_type: Literal["standard", "swarm", "sovereign"] = "standard"


# ============================================================================
# LEAN 4 / MATHEMATICAL TYPES
# ============================================================================

@dataclass
class LeanProof:
    """Represents a Lean 4 formal proof with metadata."""
    proof_id: str
    theorem_name: str
    lean_code: str
    natural_language_statement: str
    proof_status: LeanProofStatus = LeanProofStatus.PENDING
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity_score: int = 1
    proof_steps: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    verification_time: float = 0.0
    elaborated_type: str = ""
    proof_obligations: List[str] = field(default_factory=list)
    tactics_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeanTheorem:
    """Represents a mathematical theorem with Lean 4 formalization."""
    theorem_id: str
    name: str
    statement: str
    lean_code: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    keywords: List[str] = field(default_factory=list)
    difficulty: int = 5
    is_verified: bool = False
    proof: Optional[LeanProof] = None
    related_theorems: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeanVerificationResult:
    """Result of Lean 4 formal verification."""
    verification_id: str
    success: bool
    theorem_id: str
    proof_id: Optional[str] = None
    verification_method: VerificationMethod = VerificationMethod.LEAN4
    status: LeanProofStatus = LeanProofStatus.PENDING
    confidence_score: float = 0.0
    verification_time: float = 0.0
    proof_steps: List[str] = field(default_factory=list)
    remaining_obligations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    server_used: bool = True
    fallback_used: bool = False
    lean_output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MathematicalComponent:
    """A mathematical component extracted from a problem or solution."""
    component_id: str
    type: str
    name: str
    statement: str
    domain: MathematicalDomain = MathematicalDomain.GENERAL
    complexity: int = 1
    dependencies: List[str] = field(default_factory=list)
    formalized: bool = False
    lean_code: str = ""
    verification_status: Optional[LeanProofStatus] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DECOMPOSITION & GAUNTLET TYPES
# ============================================================================

@dataclass
class GauntletRoundRule:
    """Defines the specific rules and criteria for a single round within a Gauntlet."""
    round_number: int
    quorum_required_approvals: int
    quorum_from_panel_size: int
    min_overall_confidence: float = 0.0
    max_score_variance: Optional[float] = None
    per_judge_requirements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    collaboration_mode: Literal["independent", "share_previous_feedback"] = "independent"
    time_limit_seconds: Optional[int] = None
    max_api_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    adaptive_rules: Optional[Dict[str, Any]] = None
    voting_strategy: Literal["fixed_quorum", "first_to_ahead_by_k"] = "fixed_quorum"
    margin_k: Optional[int] = None
    max_dynamic_votes: Optional[int] = 100
    required_mathematical_properties: List[str] = field(default_factory=list)
    proof_obligation_threshold: float = 0.0
    mathematical_complexity_level: int = 1
    proof_generation_enabled: bool = False
    proof_verification_enabled: bool = False
    mathematical_approach: str = "direct_proof"
    verification_timeout: int = 300
    proof_storage_enabled: bool = False
    mathematical_quality_threshold: float = 0.0


@dataclass
class GauntletDefinition:
    """A programmable, multi-round process that a piece of content must pass to be approved."""
    name: str
    team_name: str
    rounds: List[GauntletRoundRule]
    gauntlet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: Optional[str] = None
    description: Optional[str] = None
    attack_modes: List[str] = field(default_factory=list)
    generation_mode: Literal["single_candidate", "multi_candidate_peer_review", "evolutionary", "hybrid"] = "single_candidate"
    gauntlet_type: Literal["standard", "adaptive", "hierarchical", "competitive", "collaborative"] = "standard"
    performance_metrics: Optional[Dict[str, float]] = None
    gauntlet_config: Optional[Dict[str, Any]] = None
    red_flags: Dict[str, Any] = field(default_factory=lambda: {
        "max_token_length": 2000,
        "strict_format_adherence": True,
        "forbidden_phrases": ["I apologize", "I'm confused", "As an AI"]
    })
    formal_verification_enabled: bool = False
    verification_methods: List[VerificationMethod] = field(default_factory=lambda: [VerificationMethod.PEER_REVIEW])
    mathematical_requirements: Dict[str, Any] = field(default_factory=dict)
    proof_generation_enabled: bool = False
    automatic_formalization: bool = False
    formal_verification_threshold: float = 0.9
    lean_verification_config: Dict[str, Any] = field(default_factory=dict)
    
    # Validation flags
    execution_order: str = "sequential"
    stop_on_first_failure: bool = False
    require_all_rounds: bool = True
    red_team_required: bool = False
    gold_team_required: bool = False
    blue_team_participation: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletRoundResult:
    """
    Result from executing a single gauntlet round.

    Attributes:
        rule_id: ID of the gauntlet round rule
        round_number: Round number (1-indexed)
        status: Final status of the round
        score: Score achieved (0.0-1.0+)
        feedback: Human-readable feedback
        details: Additional evaluation details
        execution_time: Time taken for evaluation in seconds
        timestamp: When the evaluation was performed
    """
    rule_id: str
    round_number: int
    status: GauntletRoundStatus
    score: float
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'rule_id': self.rule_id,
            'round_number': self.round_number,
            'status': self.status.value if isinstance(self.status, Enum) else self.status,
            'score': self.score,
            'feedback': self.feedback,
            'details': self.details,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


@dataclass
class GauntletExecution:
    """
    Complete execution result for a gauntlet.

    Attributes:
        gauntlet_id: ID of the gauntlet definition
        solution_id: ID of the solution evaluated
        rounds_results: Results from each round
        rounds_passed: List of round IDs that passed
        rounds_failed: List of round IDs that failed
        final_score: Final aggregated score
        overall_passed: Whether the gauntlet was passed
        execution_time: Total time for all rounds
        timestamp: When the execution was performed
    """
    gauntlet_id: str
    solution_id: str
    rounds_results: List[GauntletRoundResult] = field(default_factory=list)
    rounds_passed: List[str] = field(default_factory=list)
    rounds_failed: List[str] = field(default_factory=list)
    final_score: float = 0.0
    overall_passed: bool = False
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gauntlet_id': self.gauntlet_id,
            'solution_id': self.solution_id,
            'rounds_results': [r.to_dict() for r in self.rounds_results],
            'rounds_passed': self.rounds_passed,
            'rounds_failed': self.rounds_failed,
            'final_score': self.final_score,
            'overall_passed': self.overall_passed,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }


@dataclass
class SubProblem:
    """Represents a single sub-problem in the decomposition plan."""
    id: str
    description: str
    parent_id: Optional[str] = None # Added from sovereign
    title: str = "" # Added from sovereign
    type: Union[SubProblemType, str] = SubProblemType.RESEARCH
    complexity_score: Union[ComplexityScore, int] = 5
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list) # Added from sovereign
    validation_gauntlet: str = ""
    assigned_team: Optional[str] = None
    estimated_effort: int = 1
    priority: int = 5
    execution_order: int = 0
    
    # Workflow specific
    ai_suggested_evolution_mode: str = "standard"
    ai_suggested_complexity_score: int = 5
    ai_suggested_evaluation_prompt: str = ""
    content_type: str = "text_general"
    solver_team_name: str = ""
    red_team_gauntlet_name: Optional[str] = None
    gold_team_gauntlet_name: str = ""
    solver_generation_gauntlet_name: Optional[str] = None
    patcher_team_name: str = ""
    evolution_params: Dict[str, Any] = field(default_factory=dict)
    
    # AI suggestions
    ai_suggested_team_assignment: Optional[str] = None
    ai_suggested_gauntlet_assignment: Optional[Dict[str, str]] = None
    estimated_resources: Optional[Dict[str, Any]] = None
    potential_approaches: Optional[List[str]] = None
    
    status: Union[SubProblemStatus, str] = SubProblemStatus.PENDING
    
    solution_attempts: List['SolutionAttempt'] = field(default_factory=list)
    performance_metrics: Optional[Dict[str, float]] = None
    openevolve_metrics: Optional[Dict[str, Any]] = None
    
    # Atomic & Recursive
    atomic_mode: bool = False
    decomposition_depth: int = 0
    micro_steps: List['SubProblem'] = field(default_factory=list)
    
    # Context Slicer
    acceptance_criteria: List[str] = field(default_factory=list)
    solution_requirements: Dict[str, Any] = field(default_factory=dict)
    specific_constraints: List[str] = field(default_factory=list)
    dependency_outputs: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Mathematical
    mathematical_components: List[MathematicalComponent] = field(default_factory=list)
    requires_formal_verification: bool = False
    mathematical_domain: Optional[MathematicalDomain] = None
    formal_verification_enabled: bool = False
    mathematical_properties: List[str] = field(default_factory=list)
    lean_theorems: List[LeanTheorem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.type, Enum):
            data['type'] = self.type.value
        if isinstance(self.status, Enum):
            data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        data['updated_at'] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return data


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


@dataclass
class DecompositionPlan:
    """Complete decomposition plan"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_id: str = ""
    problem_statement: str = ""
    analyzed_context: Dict[str, Any] = field(default_factory=dict)
    strategy: DecompositionStrategy = DecompositionStrategy.SEMANTIC
    
    sub_problems: List[SubProblem] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    
    # Workflow config
    max_refinement_loops: int = 3
    auto_approval_enabled: bool = False
    auto_approval_criteria: Optional[Dict[str, Any]] = None
    mdap_enabled: bool = False
    mdap_config: Dict[str, Any] = field(default_factory=dict)
    maker_enabled: bool = False
    maker_config: Dict[str, Any] = field(default_factory=dict)
    
    # Integration
    crewai_workflow_id: Optional[str] = None
    id_to_ticket_id_map: Dict[str, str] = field(default_factory=dict)
    ticket_id_to_subproblem_id_map: Dict[str, str] = field(default_factory=dict)
    
    resource_limits: Optional[Dict[str, Any]] = None
    parallel_processing_enabled: bool = False
    max_parallel_sub_problems: int = 1
    learning_enabled: bool = False
    learning_config: Optional[Dict[str, Any]] = None
    
    # Teams
    content_analyzer_team_name: str = ""
    planner_team_name: str = ""
    assembler_team_name: str = ""
    final_red_team_gauntlet_name: Optional[str] = None
    final_gold_team_gauntlet_name: str = ""
    
    # Status
    confidence_level: float = 0.0
    created_by: str = "system"
    approved_by: Optional[str] = None
    status: PlanStatus = PlanStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.strategy, Enum):
            data['strategy'] = self.strategy.value
        if isinstance(self.status, Enum):
            data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        data['updated_at'] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return data


@dataclass
class CritiqueReport:
    """Report generated by a Red Team Gauntlet."""
    solution_attempt_id: str = ""
    gauntlet_name: str = ""
    is_approved: bool = False
    reports_by_judge: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    critique_timestamp: float = field(default_factory=time.time)
    overall_score: float = 0.0
    flaw_severity_scores: Dict[str, float] = field(default_factory=dict)
    identified_flaws: List[Dict[str, Any]] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    resource_usage: Optional[Dict[str, Any]] = None
    
    # Compatibility fields from sovereign
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    solution_id: str = ""
    critic_id: str = ""
    passed: bool = False
    flaws: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    feedback: str = ""
    improvements: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Report generated by a Gold Team Gauntlet."""
    solution_attempt_id: str = ""
    gauntlet_name: str = ""
    is_approved: bool = False
    reports_by_judge: List[Dict[str, Any]] = field(default_factory=list)
    average_score: float = 0.0
    score_variance: float = 0.0
    summary: str = ""
    verification_timestamp: float = field(default_factory=time.time)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    criteria_met: List[str] = field(default_factory=list)
    criteria_not_met: List[str] = field(default_factory=list)
    targeted_feedback: Optional[str] = None
    resource_usage: Optional[Dict[str, Any]] = None
    
    # Lean 4 / Mathematical
    lean_verification: Optional[LeanVerificationResult] = None
    verification_method: VerificationMethod = VerificationMethod.PEER_REVIEW
    mathematical_verified: bool = False
    formal_proof_available: bool = False
    mathematical_confidence: float = 0.0
    mathematical_components_verified: List[str] = field(default_factory=list)
    
    # Compatibility fields from sovereign
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    solution_id: str = ""
    verified: bool = False
    confidence: float = 0.0
    method: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionAttempt:
    """Represents a candidate solution."""
    sub_problem_id: str
    content: str
    generated_by_model: str = ""
    timestamp: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)
    solution_type: Optional[str] = None
    solution_approach: Optional[str] = None
    quality_metrics: Optional[Dict[str, float]] = None
    resource_usage: Optional[Dict[str, Any]] = None
    status: Union[Literal["generated", "critiqued", "verified", "rejected", "patched"], str] = "generated"
    critique_reports: List[CritiqueReport] = field(default_factory=list)
    verification_reports: List[VerificationReport] = field(default_factory=list)
    openevolve_metrics: Optional[Dict[str, Any]] = None
    
    # Compatibility
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approach: str = ""
    solution_content: str = "" # Alias for content
    team_id: str = ""
    confidence_score: float = 0.0
    validation_results: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# KNOWLEDGE ARTIFACTS
# ============================================================================

@dataclass
class KnowledgeArtifact:
    """Represents a piece of knowledge extracted from a workflow execution."""
    artifact_id: str
    artifact_type: Literal[
        "solution_pattern", 
        "team_performance", 
        "gauntlet_effectiveness", 
        "critique_insight", 
        "decomposition_strategy", 
        "verification_method",
        "adr",
        "refinement_template"
    ]
    source_workflow_id: str
    source_stage: Literal[0, 1, 2, 3, 4, 5, 6]
    timestamp: datetime
    confidence: float
    title: str
    description: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_artifacts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
    
    # Legacy compatibility
    id: str = "" # Should map to artifact_id

    def __post_init__(self):
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())
        if not self.id:
            self.id = self.artifact_id
        if not self.timestamp:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for workflow execution and evaluation."""
    workflow_id: str = ""
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    error_count: int = 0
    throughput: float = 0.0
    latency: float = 0.0
    accuracy: float = 0.0
    efficiency: float = 0.0
    quality_score: float = 0.0
    reliability: float = 0.0
    scalability: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


# ============================================================================
# WORKFLOW STATE
# ============================================================================

@dataclass
class WorkflowState:
    """Manages the state of an active Sovereign-Grade Decomposition Workflow."""
    workflow_id: str
    workflow_type: Any
    problem_statement: str
    current_stage: str
    tenant_id: Optional[str] = None
    current_sub_problem_id: Optional[str] = None
    current_gauntlet_name: Optional[str] = None
    status: str = "running"
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_problem_solutions: Dict[str, SolutionAttempt] = field(default_factory=dict)
    solved_sub_problem_ids: Set[str] = field(default_factory=set)
    rejected_sub_problems: Dict[str, Any] = field(default_factory=dict)
    final_solution: Optional[SolutionAttempt] = None
    refinement_loop_count: int = 0

    # Advanced features
    auto_refine_enabled: bool = False
    entanglement_matrix: Dict[str, Set[str]] = field(default_factory=dict)
    entanglement_strict_mode: bool = False

    # Teams & Gauntlets
    content_analyzer_team: Optional[Team] = None
    planner_team: Optional[Team] = None
    solver_team: Optional[Team] = None
    patcher_team: Optional[Team] = None
    assembler_team: Optional[Team] = None
    sub_problem_red_gauntlet: Optional[GauntletDefinition] = None
    sub_problem_gold_gauntlet: Optional[GauntletDefinition] = None
    solver_generation_gauntlet: Optional[GauntletDefinition] = None
    final_red_gauntlet: Optional[GauntletDefinition] = None
    final_gold_gauntlet: Optional[GauntletDefinition] = None

    max_refinement_loops: int = 3
    all_critique_reports: List[CritiqueReport] = field(default_factory=list)
    all_verification_reports: List[VerificationReport] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    knowledge_artifacts: List[KnowledgeArtifact] = field(default_factory=list)
    openevolve_metrics: Dict[str, Any] = field(default_factory=dict)

    mdap_enabled: bool = False
    mdap_config: Dict[str, Any] = field(default_factory=dict)
    maker_enabled: bool = False
    maker_config: Dict[str, Any] = field(default_factory=dict)

    # Configs
    openevolve_parameters: Dict[str, Any] = field(default_factory=dict)

    # LeanAide
    leanaide_enabled: bool = False
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    leanaide_confidence_threshold: float = 0.7
    leanaide_auto_detect_math: bool = True
    leanaide_require_formal_proof: bool = False
    leanaide_store_proofs: bool = True
    leanaide_verification_method: Literal["leanaide_only", "leanaide_primary", "standard_primary"] = "standard_primary"
    leanaide_timeout: int = 300

    # CrewAI
    CrewAI_workflow_id: Optional[str] = None
    id_to_ticket_id_map: Dict[str, str] = field(default_factory=dict)
    ticket_id_to_subproblem_id_map: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# SOVEREIGN DATA MODELS (Legacy Compatibility)
# ============================================================================

@dataclass
class ProblemDefinition:
    """Legacy compatibility class for sovereign problem definitions."""
    id: str
    title: str
    description: str
    problem_type: Union[str, ProblemType]
    domain_context: Union[DomainContext, str]
    complexity_score: Union[ComplexityScore, str]
    constraints: Optional[List[Constraint]] = None
    success_criteria: Optional[List[SuccessCriterion]] = None
    stakeholders: Optional[List[str]] = None
    resources_available: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.domain_context, DomainContext):
            data['domain_context'] = self.domain_context.to_dict()
        if isinstance(self.complexity_score, ComplexityScore):
            data['complexity_score'] = self.complexity_score.to_dict()
        if self.constraints:
            data['constraints'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.constraints]
        if self.success_criteria:
            data['success_criteria'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.success_criteria]
        data['created_at'] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        data['updated_at'] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return data


@dataclass
class Pattern:
    """Legacy compatibility class for problem patterns."""
    id: str
    name: str
    description: str
    pattern_type: str
    domain: str
    complexity: int = 5
    example_solutions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamAssignment:
    """Legacy compatibility class for team assignments."""
    id: str
    team_id: str
    sub_problem_id: str
    assigned_at: datetime = field(default_factory=datetime.now)
    assigned_by: str = "system"
    status: str = "assigned"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    """Legacy compatibility class for feedback."""
    id: str
    source: str
    target_id: str
    content: str
    feedback_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Legacy compatibility class for validation results."""
    id: str
    is_valid: bool
    confidence: float
    validation_method: str
    validated_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScores:
    """Legacy compatibility class for quality scores."""
    clarity: float = 0.0
    completeness: float = 0.0
    correctness: float = 0.0
    efficiency: float = 0.0
    maintainability: float = 0.0
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationCheckpoint:
    """Legacy compatibility class for validation checkpoints."""
    id: str
    name: str
    description: str
    checkpoint_type: str
    criteria: List[str] = field(default_factory=list)
    required_score: float = 0.0
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SOVEREIGN DATA MODELS (Legacy Compatibility)
# ============================================================================

@dataclass
class ProblemDefinition:
    """Legacy compatibility class for sovereign problem definitions."""
    id: str
    title: str
    description: str
    problem_type: str
    domain_context: Union[DomainContext, str]
    complexity_score: Union[ComplexityScore, str]
    constraints: Optional[List[Constraint]] = None
    success_criteria: Optional[List[SuccessCriterion]] = None
    stakeholders: Optional[List[str]] = None
    resources_available: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.domain_context, DomainContext):
            data['domain_context'] = self.domain_context.to_dict()
        if isinstance(self.complexity_score, ComplexityScore):
            data['complexity_score'] = self.complexity_score.to_dict()
        if self.constraints:
            data['constraints'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.constraints]
        if self.success_criteria:
            data['success_criteria'] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.success_criteria]
        data['created_at'] = self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        data['updated_at'] = self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at
        return data


@dataclass
class Pattern:
    """Legacy compatibility class for problem patterns."""
    id: str
    name: str
    description: str
    pattern_type: str
    domain: str
    complexity: int = 5
    example_solutions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamAssignment:
    """Legacy compatibility class for team assignments."""
    id: str
    team_id: str
    sub_problem_id: str
    assigned_at: datetime = field(default_factory=datetime.now)
    assigned_by: str = "system"
    status: str = "assigned"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    """Legacy compatibility class for feedback."""
    id: str
    source: str
    target_id: str
    content: str
    feedback_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Legacy compatibility class for validation results."""
    id: str
    is_valid: bool
    confidence: float
    validation_method: str
    validated_at: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityScores:
    """Legacy compatibility class for quality scores."""
    clarity: float = 0.0
    completeness: float = 0.0
    correctness: float = 0.0
    efficiency: float = 0.0
    maintainability: float = 0.0
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationCheckpoint:
    """Legacy compatibility class for validation checkpoints."""
    id: str
    name: str
    description: str
    checkpoint_type: str
    criteria: List[str] = field(default_factory=list)
    required_score: float = 0.0
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
