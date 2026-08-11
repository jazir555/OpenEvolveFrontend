"""
Z3 and LeanAIDE Bubbles for BubbleLab

This module provides BubbleLab workflow nodes (bubbles) for Z3 SMT solving
and LeanAIDE theorem proving operations. These bubbles enable visualization
and control of formal verification workflows through the BubbleLabs UI.

Features:
- Z3 constraint solving bubbles
- Z3 theorem proving bubbles  
- LeanAIDE proof visualization bubbles
- Cross-verification bubbles
- Sub-problem loop bubbles for entangled workflows
- Entanglement matrix integration and visualization
- Flexible workflow builder for arbitrary patterns

Usage:
    from z3_leanaide_bubbles import (
        create_z3_solver_bubble,
        create_z3_prover_bubble,
        create_leanaide_proof_bubble,
        create_cross_verification_bubble,
        create_z3_workflow,
        create_subproblem_loop_bubble,
        create_entanglement_visualization_bubble,
        create_z3_workflow_with_entanglement
    )
"""


import uuid
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Import CAV-NLP integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False


# =============================================================================
# Bubble Configuration Constants
# =============================================================================

Z3_NODE_POSITIONS = {
    "input": {"x": 0, "y": 0},
    "classification": {"x": 150, "y": 0},
    "z3_solver": {"x": 300, "y": 0},
    "z3_prover": {"x": 300, "y": 100},
    "leanaide_proof": {"x": 500, "y": 0},
    "cross_verify": {"x": 650, "y": 0},
    "result": {"x": 800, "y": 0},
}

Z3_NODE_COLORS = {
    "z3_solver": "#FF6B6B",
    "z3_prover": "#E17055",
    "leanaide_proof": "#00B894",
    "cross_verify": "#6C5CE7",
    "classification": "#0984E3",
    "input": "#74B9FF",
    "result": "#00B894",
    "subproblem_loop": "#FDCB6E",
    "entanglement_viz": "#E84393",
    "subproblem": "#81ECEC",
    "super_node": "#A29BFE",
    # Additional bubbles
    "adaptive_strategy": "#FD79A8",
    "quality_assessment": "#00CEC9",
    "knowledge_extraction": "#FAB1A0",
    "metrics_analytics": "#74B9FF",
    "cache": "#DFE6E9",
    "agent_coordination": "#B2BEC3",
    "error_handler": "#D63031",
    "refinement": "#E17055",
    "composition": "#00B894",
    "validation": "#0984E3",
    "parallelization": "#6C5CE7",
    "convergence": "#FDCB6E",
    "divergence": "#FF7675",
    # Extended bubbles
    "adversarial_testing": "#FF4757",
    "mdap": "#3742FA",
    "decomposition": "#2ED573",
    "recomposition": "#FFA502",
    "monitoring": "#70A1FF",
    "alerting": "#FF6B81",
    "security": "#2F3542",
    "api_gateway": "#5352ED",
    "logging": "#A4B0BE",
    "visualization": "#FF7F50",
    "batch_processing": "#1E90FF",
    "experiment_tracking": "#32CD32",
    "prompt_engineering": "#DA70D6",
    "evaluation": "#FFD700",
    "deployment": "#00CED1",
    "optimization": "#FF69B4",
    "heuristic": "#8A2BE2",
    "sampling": "#20B2AA",
    "ensemble": "#FF6347",
}

Z3_NODE_ICONS = {
    "z3_solver": "🔐",
    "z3_prover": "📐",
    "leanaide_proof": "📚",
    "cross_verify": "⚖️",
    "classification": "🔍",
    "input": "📥",
    "result": "[OK]",
    "subproblem_loop": "🔄",
    "entanglement_viz": "🕸️",
    "subproblem": "📦",
    "super_node": "🔗",
    # Additional bubbles
    "adaptive_strategy": "🎯",
    "quality_assessment": "📊",
    "knowledge_extraction": "🧠",
    "metrics_analytics": "📈",
    "cache": "💾",
    "agent_coordination": "🤝",
    "error_handler": "🚨",
    "refinement": "🔧",
    "composition": "🧩",
    "validation": "[OK]",
    "parallelization": "⚡",
    "convergence": "🎯",
    "divergence": "[WARN]",
    # Extended bubbles
    "adversarial_testing": "⚔️",
    "mdap": "📊",
    "decomposition": "✂️",
    "recomposition": "🧩",
    "monitoring": "📡",
    "alerting": "🔔",
    "security": "🛡️",
    "api_gateway": "🌐",
    "logging": "📝",
    "visualization": "🎨",
    "batch_processing": "📚",
    "experiment_tracking": "🧪",
    "prompt_engineering": "💬",
    "evaluation": "📋",
    "deployment": "🚀",
    "optimization": "⚙️",
    "heuristic": "🔍",
    "sampling": "🎲",
    "ensemble": "👥",
}


# =============================================================================
# Bubble Data Classes
# =============================================================================

@dataclass
class Z3SolverBubbleConfig:
    """Configuration for a Z3 constraint solver bubble."""
    problem_text: str
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: int = 30
    strategy: str = "auto"


@dataclass
class Z3ProverBubbleConfig:
    """Configuration for a Z3 theorem prover bubble."""
    theorem_statement: str
    proof_strategy: str = "default"
    timeout_seconds: int = 60


@dataclass
class LeanAideProofBubbleConfig:
    """Configuration for a LeanAIDE proof visualization bubble."""
    theorem_name: str
    proof_type: str = "theorem"  # theorem, definition, lemma
    mcts_enabled: bool = True
    timeout_seconds: int = 120


@dataclass
class CrossVerificationBubbleConfig:
    """Configuration for cross-verification bubble."""
    problem_text: str
    z3_strategy: str = "adaptive"
    lean_strategy: str = "auto"
    timeout_seconds: int = 60


@dataclass
class ProblemClassificationBubbleConfig:
    """Configuration for problem classification bubble."""
    problem_text: str
    auto_classify: bool = True


@dataclass
class SubProblemLoopBubbleConfig:
    """Configuration for sub-problem loop bubble (handles entangled sub-problems)."""
    sub_problems: List[Dict[str, Any]]
    entanglement_matrix: Dict[str, List[str]] = field(default_factory=dict)
    loop_strategy: str = "sequential"  # sequential, parallel, super_node
    max_iterations: int = 10
    convergence_threshold: float = 0.95
    
    def get_sub_problem_ids(self) -> List[str]:
        """Extract sub-problem IDs from the list."""
        return [sp.get("id", f"sp_{i}") for i, sp in enumerate(self.sub_problems)]
    
    def get_entangled_pairs(self) -> List[Tuple[str, str]]:
        """Get all entangled pairs from the matrix."""
        pairs = []
        for source, targets in self.entanglement_matrix.items():
            for target in targets:
                pairs.append((source, target))
        return pairs


@dataclass
class EntanglementVisualizationConfig:
    """Configuration for entanglement matrix visualization bubble."""
    entanglement_matrix: Dict[str, List[str]]
    sub_problems: List[Dict[str, Any]] = field(default_factory=list)
    show_coupling_strength: bool = True
    highlight_super_nodes: bool = True
    
    def get_coupling_density(self) -> float:
        """Calculate coupling density (entanglements / max possible)."""
        n = len(self.sub_problems)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        actual_edges = sum(len(targets) for targets in self.entanglement_matrix.values()) // 2
        return actual_edges / max_edges if max_edges > 0 else 0.0


@dataclass
class SubProblemBubbleConfig:
    """Configuration for an individual sub-problem bubble."""
    sub_problem_id: str
    problem_text: str
    entangled_with: List[str] = field(default_factory=list)
    entanglement_source: str = "symbolic_overlap"
    is_super_node: bool = False
    super_node_partner: Optional[str] = None
    

@dataclass
class AdaptiveStrategyBubbleConfig:
    """Configuration for adaptive strategy selection bubble."""
    problem_text: str
    available_strategies: List[str] = field(default_factory=lambda: ["qd", "mo", "pes", "adversarial"])
    selection_criteria: str = "auto"  # auto, performance, complexity, robustness
    enable_fallback: bool = True
    max_strategy_switches: int = 3


@dataclass
class QualityAssessmentBubbleConfig:
    """Configuration for quality assessment bubble."""
    problem_text: str
    assessment_dimensions: List[str] = field(default_factory=lambda: ["correctness", "efficiency", "robustness", "elegance"])
    threshold_mode: str = "adaptive"  # adaptive, strict, lenient
    min_quality_score: float = 0.7


@dataclass
class KnowledgeExtractionBubbleConfig:
    """Configuration for knowledge extraction bubble."""
    solution_text: str
    extraction_type: str = "patterns"  # patterns, constraints, heuristics, theorems
    store_in_graph: bool = True
    confidence_threshold: float = 0.8
    max_patterns: int = 100


@dataclass
class MetricsAnalyticsBubbleConfig:
    """Configuration for metrics and analytics bubble."""
    metrics_types: List[str] = field(default_factory=lambda: ["time", "iterations", "quality", "diversity"])
    track_historical: bool = True
    aggregation_level: str = "detailed"  # minimal, standard, detailed
    export_format: str = "json"  # json, csv, prometheus


@dataclass
class CacheBubbleConfig:
    """Configuration for cache bubble."""
    cache_key: str
    cache_type: str = "memory"  # memory, disk, distributed
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    compression_enabled: bool = False


@dataclass
class AgentCoordinationBubbleConfig:
    """Configuration for agent coordination bubble."""
    agent_ids: List[str]
    coordination_type: str = "hierarchical"  # hierarchical, peer, auction, consensus
    task_distribution: str = "load_balanced"  # load_balanced, round_robin, priority
    conflict_resolution: str = "voting"  # voting, arbitration, priority
    sync_interval_ms: int = 100


@dataclass
class ErrorHandlerBubbleConfig:
    """Configuration for error handler bubble."""
    error_types: List[str] = field(default_factory=lambda: ["timeout", "validation", "resource"])
    recovery_strategy: str = "retry"  # retry, fallback, skip, abort
    max_retries: int = 3
    escalation_threshold: int = 2
    notify_on_escalation: bool = True


@dataclass
class RefinementBubbleConfig:
    """Configuration for solution refinement bubble."""
    solution_text: str
    refinement_type: str = "iterative"  # iterative, adaptive, targeted
    max_refinements: int = 5
    convergence_check: bool = True
    preserve_constraints: bool = True


@dataclass
class CompositionBubbleConfig:
    """Configuration for solution composition bubble."""
    sub_solutions: List[Dict[str, Any]]
    composition_strategy: str = "merge"  # merge, override, weighted, voting
    conflict_resolution: str = "consensus"  # consensus, priority, expert
    validate_composition: bool = True


@dataclass
class ValidationBubbleConfig:
    """Configuration for validation bubble."""
    problem_text: str
    solution_text: str
    validation_type: str = "comprehensive"  # basic, comprehensive, formal
    check_soundness: bool = True
    check_completeness: bool = True
    timeout_seconds: int = 60


@dataclass
class ParallelizationBubbleConfig:
    """Configuration for parallelization bubble."""
    tasks: List[Dict[str, Any]]
    parallel_mode: str = "data"  # data, task, hybrid
    max_workers: int = 4
    load_balancing: bool = True
    result_aggregation: str = "reduce"  # reduce, collect, scatter_gather


@dataclass
class ConvergenceDivergenceBubbleConfig:
    """Configuration for convergence/divergence detection bubble."""
    metric_history: List[Dict[str, Any]]
    detection_type: str = "both"  # convergence, divergence, both
    window_size: int = 10
    threshold: float = 0.01
    trend_analysis: bool = True


# =============================================================================
# Additional Extended Config Classes
# =============================================================================

@dataclass
class AdversarialTestingBubbleConfig:
    """Configuration for adversarial testing bubble."""
    problem_text: str
    attack_strategies: List[str] = field(default_factory=lambda: ["fuzzing", "boundary", "edge_case"])
    attack_intensity: float = 0.7
    num_attacks: int = 10
    auto_fix: bool = True


@dataclass
class MDAPBubbleConfig:
    """Configuration for Multi-Dimensional Adaptive Planning bubble."""
    objectives: List[Dict[str, Any]]
    planning_horizon: int = 5
    adaptive_rate: float = 0.3
    exploration_factor: float = 0.2
    resource_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecompositionBubbleConfig:
    """Configuration for problem decomposition bubble."""
    problem_text: str
    decomposition_type: str = "semantic"  # semantic, syntactic, hybrid
    max_sub_problems: int = 10
    min_sub_problem_size: int = 1
    overlap_allowed: bool = False
    recursive_depth: int = 3


@dataclass
class RecompositionBubbleConfig:
    """Configuration for solution recomposition bubble."""
    sub_solutions: List[Dict[str, Any]]
    recomposition_strategy: str = "hierarchical"  # hierarchical, sequential, parallel
    validation_mode: str = "strict"  # strict, lenient, adaptive
    merge_conflicts: str = "resolve"  # resolve, flag, abort


@dataclass
class MonitoringBubbleConfig:
    """Configuration for monitoring bubble."""
    metrics: List[str] = field(default_factory=lambda: ["time", "memory", "accuracy"])
    sampling_interval: float = 1.0
    alerting_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    persistent_logging: bool = True


@dataclass
class AlertingBubbleConfig:
    """Configuration for alerting bubble."""
    alert_types: List[str] = field(default_factory=lambda: ["error", "warning", "info"])
    alert_channels: List[str] = field(default_factory=lambda: ["console", "log"])
    escalation_policy: str = "linear"  # linear, exponential, immediate
    max_escalation_level: int = 5
    quiet_period_seconds: int = 300


@dataclass
class SecurityBubbleConfig:
    """Configuration for security bubble."""
    security_level: str = "standard"  # minimal, standard, high
    vulnerability_scan: bool = True
    injection_prevention: bool = True
    authentication_required: bool = False
    encryption_enabled: bool = True


@dataclass
class APIGatewayBubbleConfig:
    """Configuration for API gateway bubble."""
    endpoints: List[str]
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    authentication: str = "optional"  # required, optional, none
    request_validation: bool = True
    response_caching: bool = True


@dataclass
class LoggingBubbleConfig:
    """Configuration for logging bubble."""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_format: str = "json"  # json, text, structured
    log_outputs: List[str] = field(default_factory=lambda: ["console", "file"])
    log_rotation: bool = True
    max_log_size_mb: int = 100
    retention_days: int = 30


@dataclass
class VisualizationBubbleConfig:
    """Configuration for visualization bubble."""
    data_source: str
    visualization_type: str = "graph"  # graph, timeline, heatmap, tree
    interactive: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "svg"])
    animation_enabled: bool = False
    theme: str = "default"  # default, dark, light


@dataclass
class CachingBubbleConfig:
    """Configuration for caching bubble."""
    cache_type: str = "memory"  # memory, disk, distributed
    ttl_seconds: int = 3600
    max_size_mb: int = 100
    compression_enabled: bool = False
    cache_invalidation: str = "ttl"  # ttl, manual, adaptive


@dataclass
class BatchProcessingBubbleConfig:
    """Configuration for batch processing bubble."""
    batch_items: List[Dict[str, Any]]
    batch_size: int = 50
    parallelization_mode: str = "auto"  # auto, sequential, parallel
    progress_tracking: bool = True
    error_handling: str = "skip"  # skip, abort, retry
    checkpoint_interval: int = 10


@dataclass
class ExperimentTrackingBubbleConfig:
    """Configuration for experiment tracking bubble."""
    experiment_name: str
    parameters: Dict[str, Any]
    metrics: List[str]
    log_artifacts: bool = True
    comparison_mode: bool = True
    export_results: bool = True


@dataclass
class PromptEngineeringBubbleConfig:
    """Configuration for prompt engineering bubble."""
    base_prompt: str
    prompt_variations: List[str] = field(default_factory=list)
    optimization_strategy: str = "auto"  # manual, auto, learned
    evaluation_criteria: List[str] = field(default_factory=lambda: ["accuracy", "relevance"])
    few_shot_examples: int = 3
    temperature: float = 0.7


@dataclass
class EvaluationBubbleConfig:
    """Configuration for evaluation bubble."""
    criteria: List[str]
    evaluation_type: str = "comprehensive"  # basic, comprehensive, automated
    scoring_method: str = "weighted"  # weighted, ranking, threshold
    benchmark_comparison: bool = True
    detailed_reporting: bool = True


@dataclass
class DeploymentBubbleConfig:
    """Configuration for deployment bubble."""
    deployment_target: str  # local, staging, production
    environment_config: Dict[str, Any]
    health_check_enabled: bool = True
    rollback_enabled: bool = True
    monitoring_integration: bool = True
    blue_green_deployment: bool = False


@dataclass
class OptimizationBubbleConfig:
    """Configuration for optimization bubble."""
    optimization_target: str  # speed, memory, accuracy, cost
    optimization_algorithm: str = "auto"  # auto, grid, random, bayesian
    constraint_bounds: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6


@dataclass
class HeuristicBubbleConfig:
    """Configuration for heuristic search bubble."""
    search_space: Dict[str, Any]
    heuristic_function: str = "default"  # default, custom, learned
    search_depth: int = 10
    branching_factor: int = 4
    pruning_enabled: bool = True
    ordering_strategy: str = "best_first"  # best_first, breadth_first, depth_first


@dataclass
class SamplingBubbleConfig:
    """Configuration for sampling bubble."""
    population_size: int
    sample_size: int
    sampling_method: str = "random"  # random, stratified, systematic, reservoir
    stratification_field: Optional[str] = None
    replacement: bool = False
    seed: Optional[int] = None


@dataclass
class EnsembleBubbleConfig:
    """Configuration for ensemble bubble."""
    models: List[Dict[str, Any]]
    ensemble_method: str = "voting"  # voting, stacking, bagging, boosting
    weight_strategy: str = "equal"  # equal, performance_based, learned
    cross_validation_folds: int = 5
    parallel_training: bool = True


# =============================================================================
# Entanglement Matrix Utilities (compatible with utils/entanglement_utils)
# =============================================================================

def normalize_entanglement_matrix_z3(
    matrix: Dict[str, Any],
    allowed_ids: Optional[List[str]] = None,
    enforce_symmetry: bool = True,
    strict: bool = False,
) -> Dict[str, Set[str]]:
    """
    Normalize entanglement matrices to Dict[str, Set[str]].
    Compatible with utils/entanglement_utils.normalize_entanglement_matrix.
    
    Args:
        matrix: Raw entanglement matrix
        allowed_ids: Allowed sub-problem IDs
        enforce_symmetry: Ensure bidirectional entanglements
        strict: Raise on validation errors
    
    Returns:
        Normalized matrix
    """
    allowed_set = set(allowed_ids or [])
    raw_map: Dict[str, Set[str]] = {}
    
    if matrix:
        for key, value in matrix.items():
            if allowed_set and key not in allowed_set:
                if strict:
                    raise ValueError(f"Entanglement matrix key not allowed: {key}")
                continue
            if isinstance(value, (set, list, tuple)):
                items = value
            elif value is None:
                items = []
            else:
                items = [value]
            
            raw_set: Set[str] = set()
            for item in items:
                if item is None:
                    continue
                if item == key:
                    if strict:
                        raise ValueError(f"Self-entanglement detected for {key}")
                    continue
                if allowed_set and item not in allowed_set:
                    if strict:
                        raise ValueError(f"Entanglement partner not allowed: {item}")
                    continue
                raw_set.add(item)
            raw_map[key] = raw_set
    
    if not allowed_set:
        allowed_set = set(raw_map.keys())
    
    normalized: Dict[str, Set[str]] = {key: set() for key in allowed_set}
    for key, partners in raw_map.items():
        if allowed_set and key not in allowed_set:
            continue
        normalized.setdefault(key, set()).update(partners)
    
    if enforce_symmetry:
        for key, partners in list(normalized.items()):
            for partner in list(partners):
                normalized.setdefault(partner, set()).add(key)
    
    for key in normalized:
        normalized[key].discard(key)
    
    return normalized


def serialize_entanglement_matrix_z3(matrix: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Serialize normalized matrix to JSON-safe format."""
    return {key: sorted(list(value)) for key, value in matrix.items()}


def build_entanglement_from_subproblems(sub_problems: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build entanglement matrix from sub-problem shared symbols.
    
    Args:
        sub_problems: List of sub-problem dicts with optional 'shared_symbols' field
    
    Returns:
        Entanglement matrix
    """
    matrix: Dict[str, Set[str]] = {}
    
    # Initialize empty sets for all sub-problems
    ids = [sub_problems[i].get("id", f"sp_{i}") for i in range(len(sub_problems))]
    for sp_id in ids:
        matrix[sp_id] = set()
    
    # Find shared symbols and create entanglements
    for i, sp1 in enumerate(sub_problems):
        id1 = sp1.get("id", f"sp_{i}")
        symbols1 = set(sp1.get("shared_symbols", []))
        
        for j, sp2 in enumerate(sub_problems):
            if i >= j:
                continue
            id2 = sp2.get("id", f"sp_{j}")
            symbols2 = set(sp2.get("shared_symbols", []))
            
            # Check for symbol overlap
            overlap = symbols1 & symbols2
            if overlap:
                matrix[id1].add(id2)
                matrix[id2].add(id1)
    
    return serialize_entanglement_matrix_z3(matrix)


def create_z3_solver_bubble(
    config: Z3SolverBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Z3 Solver"
) -> Dict[str, Any]:
    """
    Create a Z3 constraint solver bubble.
    
    Args:
        config: Z3SolverBubbleConfig with solver configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a Z3 solver bubble
    """
    position = position or Z3_NODE_POSITIONS.get("z3_solver", {"x": 300, "y": 0})
    icon = Z3_NODE_ICONS.get("z3_solver", "🔐")
    color = Z3_NODE_COLORS.get("z3_solver", "#FF6B6B")
    
    bubble = {
        "id": f"z3_solver_{uuid.uuid4().hex[:8]}",
        "type": "z3_solver",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "variables": config.variables,
            "constraints": config.constraints,
            "timeout_seconds": config.timeout_seconds,
            "strategy": config.strategy,
            "status": "pending",
            "node_color": color,
            "result": None,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created Z3 solver bubble: {bubble['id']}")
    return bubble


def create_z3_prover_bubble(
    config: Z3ProverBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Z3 Prover"
) -> Dict[str, Any]:
    """
    Create a Z3 theorem prover bubble.
    
    Args:
        config: Z3ProverBubbleConfig with prover configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a Z3 prover bubble
    """
    position = position or Z3_NODE_POSITIONS.get("z3_prover", {"x": 300, "y": 100})
    icon = Z3_NODE_ICONS.get("z3_prover", "📐")
    color = Z3_NODE_COLORS.get("z3_prover", "#E17055")
    
    bubble = {
        "id": f"z3_prover_{uuid.uuid4().hex[:8]}",
        "type": "z3_prover",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "theorem_statement": config.theorem_statement,
            "proof_strategy": config.proof_strategy,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "proven": False,
            "proof_steps": [],
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created Z3 prover bubble: {bubble['id']}")
    return bubble


def create_leanaide_proof_bubble(
    config: LeanAideProofBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a LeanAIDE proof visualization bubble.
    
    Args:
        config: LeanAideProofBubbleConfig with proof configuration
        position: Optional position override
        label: Display label (defaults to theorem name)
    
    Returns:
        Dict representing a LeanAIDE proof bubble
    """
    position = position or Z3_NODE_POSITIONS.get("leanaide_proof", {"x": 500, "y": 0})
    icon = Z3_NODE_ICONS.get("leanaide_proof", "📚")
    color = Z3_NODE_COLORS.get("leanaide_proof", "#00B894")
    label = label or f"{icon} {config.theorem_name}"
    
    bubble = {
        "id": f"leanaide_proof_{uuid.uuid4().hex[:8]}",
        "type": "leanaide_proof",
        "position": position,
        "data": {
            "label": label,
            "theorem_name": config.theorem_name,
            "proof_type": config.proof_type,
            "mcts_enabled": config.mcts_enabled,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "proof_steps": [],
            "proven": False,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created LeanAIDE proof bubble: {bubble['id']}")
    return bubble


def create_cross_verification_bubble(
    config: CrossVerificationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Cross Verification"
) -> Dict[str, Any]:
    """
    Create a cross-verification bubble (Z3 + LeanAIDE).
    
    Args:
        config: CrossVerificationBubbleConfig with verification configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a cross-verification bubble
    """
    position = position or Z3_NODE_POSITIONS.get("cross_verify", {"x": 650, "y": 0})
    icon = Z3_NODE_ICONS.get("cross_verify", "⚖️")
    color = Z3_NODE_COLORS.get("cross_verify", "#6C5CE7")
    
    bubble = {
        "id": f"cross_verify_{uuid.uuid4().hex[:8]}",
        "type": "cross_verification",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "z3_strategy": config.z3_strategy,
            "lean_strategy": config.lean_strategy,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "z3_status": None,
            "lean_status": None,
            "agreement": None,
            "confidence_score": 0.0,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created cross-verification bubble: {bubble['id']}")
    return bubble


def create_problem_classification_bubble(
    config: ProblemClassificationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Problem Classification"
) -> Dict[str, Any]:
    """
    Create a problem classification bubble.
    
    Args:
        config: ProblemClassificationBubbleConfig with classification configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a classification bubble
    """
    position = position or Z3_NODE_POSITIONS.get("classification", {"x": 150, "y": 0})
    icon = Z3_NODE_ICONS.get("classification", "🔍")
    color = Z3_NODE_COLORS.get("classification", "#0984E3")
    
    bubble = {
        "id": f"classification_{uuid.uuid4().hex[:8]}",
        "type": "problem_classification",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "auto_classify": config.auto_classify,
            "status": "pending",
            "node_color": color,
            "classification": None,
            "confidence": 0.0,
            "recommended_solver": None,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created classification bubble: {bubble['id']}")
    return bubble


def create_z3_result_bubble(
    result_status: str = "pending",
    position: Dict[str, float] = None,
    label: str = "Result"
) -> Dict[str, Any]:
    """
    Create a result bubble for Z3/LeanAIDE workflows.
    
    Args:
        result_status: Result status (pending, success, failed)
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a result bubble
    """
    position = position or Z3_NODE_POSITIONS.get("result", {"x": 800, "y": 0})
    icon = Z3_NODE_ICONS.get("result", "[OK]")
    color = Z3_NODE_COLORS.get("result", "#00B894")
    
    status_colors = {
        "pending": "#FDCB6E",
        "success": "#00B894",
        "failed": "#FF7675",
        "verified": "#00B894",
        "unverified": "#FF7675"
    }
    color = status_colors.get(result_status, color)
    
    bubble = {
        "id": f"z3_result_{uuid.uuid4().hex[:8]}",
        "type": "z3_result",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "status": result_status,
            "node_color": color,
            "summary": None,
            "details": {}
        }
    }
    
    logger.debug(f"Created Z3 result bubble: {bubble['id']}")
    return bubble


# =============================================================================
# Sub-Problem Loop and Entanglement Bubbles
# =============================================================================

def create_subproblem_loop_bubble(
    config: SubProblemLoopBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a sub-problem loop bubble for handling entangled sub-problems.
    
    This bubble manages iterative solving of entangled sub-problems,
    supporting convergence-based refinement.
    
    Args:
        config: SubProblemLoopBubbleConfig with loop configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a sub-problem loop bubble
    """
    position = position or {"x": 300, "y": 200}
    icon = Z3_NODE_ICONS.get("subproblem_loop", "🔄")
    color = Z3_NODE_COLORS.get("subproblem_loop", "#FDCB6E")
    
    sub_problem_ids = config.get_sub_problem_ids()
    entangled_pairs = config.get_entangled_pairs()
    
    label = label or f"{icon} Sub-Problem Loop ({len(sub_problem_ids)} problems)"
    
    bubble = {
        "id": f"subproblem_loop_{uuid.uuid4().hex[:8]}",
        "type": "subproblem_loop",
        "position": position,
        "data": {
            "label": label,
            "sub_problems": config.sub_problems,
            "sub_problem_ids": sub_problem_ids,
            "entanglement_matrix": config.entanglement_matrix,
            "entangled_pairs": entangled_pairs,
            "loop_strategy": config.loop_strategy,
            "max_iterations": config.max_iterations,
            "convergence_threshold": config.convergence_threshold,
            "status": "pending",
            "node_color": color,
            "current_iteration": 0,
            "converged": False,
            "refined_sub_problems": []
        }
    }
    
    logger.debug(f"Created sub-problem loop bubble: {bubble['id']}")
    return bubble


def create_entanglement_visualization_bubble(
    config: EntanglementVisualizationConfig,
    position: Dict[str, float] = None,
    label: str = "Entanglement Matrix"
) -> Dict[str, Any]:
    """
    Create a bubble for visualizing the entanglement matrix.
    
    This bubble displays the coupling between sub-problems and
    highlights super-nodes (tightly coupled groups).
    
    Args:
        config: EntanglementVisualizationConfig with visualization settings
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing an entanglement visualization bubble
    """
    position = position or {"x": 150, "y": 200}
    icon = Z3_NODE_ICONS.get("entanglement_viz", "🕸️")
    color = Z3_NODE_COLORS.get("entanglement_viz", "#E84393")
    
    coupling_density = config.get_coupling_density()
    
    # Identify super-nodes (nodes with high degree)
    super_nodes = []
    if config.highlight_super_nodes:
        matrix = normalize_entanglement_matrix_z3(config.entanglement_matrix)
        avg_degree = sum(len(neighbors) for neighbors in matrix.values()) / max(len(matrix), 1)
        for sp_id, neighbors in matrix.items():
            if len(neighbors) > avg_degree * 1.5:
                super_nodes.append(sp_id)
    
    bubble = {
        "id": f"entanglement_viz_{uuid.uuid4().hex[:8]}",
        "type": "entanglement_viz",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "entanglement_matrix": config.entanglement_matrix,
            "sub_problems": config.sub_problems,
            "coupling_density": coupling_density,
            "super_nodes": super_nodes,
            "show_coupling_strength": config.show_coupling_strength,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created entanglement visualization bubble: {bubble['id']}")
    return bubble


def create_subproblem_bubble(
    config: SubProblemBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create an individual sub-problem bubble for detailed viewing.
    
    Args:
        config: SubProblemBubbleConfig with sub-problem details
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a sub-problem bubble
    """
    position = position or {"x": 400, "y": 200}
    icon = Z3_NODE_ICONS.get("super_node", "📦") if config.is_super_node else Z3_NODE_ICONS.get("subproblem", "📦")
    color = Z3_NODE_COLORS.get("super_node", "#A29BFE") if config.is_super_node else Z3_NODE_COLORS.get("subproblem", "#81ECEC")
    
    label = label or f"{icon} {config.sub_problem_id}"
    
    bubble = {
        "id": f"subproblem_{config.sub_problem_id}_{uuid.uuid4().hex[:8]}",
        "type": "subproblem",
        "sub_problem_id": config.sub_problem_id,
        "position": position,
        "data": {
            "label": label,
            "problem_text": config.problem_text,
            "entangled_with": config.entangled_with,
            "entanglement_source": config.entanglement_source,
            "is_super_node": config.is_super_node,
            "super_node_partner": config.super_node_partner,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created sub-problem bubble: {bubble['id']}")
    return bubble


def create_super_node_bubble(
    sub_problem_ids: List[str],
    problem_text: str = "Super Node",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a super-node bubble for tightly coupled sub-problems.
    
    Super-nodes are groups of sub-problems that are highly entangled
    and should be solved together.
    
    Args:
        sub_problem_ids: List of sub-problem IDs in the super node
        problem_text: Description of the super node
        position: Optional position override
    
    Returns:
        Dict representing a super-node bubble
    """
    position = position or {"x": 500, "y": 200}
    icon = Z3_NODE_ICONS.get("super_node", "🔗")
    color = Z3_NODE_COLORS.get("super_node", "#A29BFE")
    
    bubble = {
        "id": f"super_node_{uuid.uuid4().hex[:8]}",
        "type": "super_node",
        "position": position,
        "data": {
            "label": f"{icon} Super Node ({len(sub_problem_ids)} problems)",
            "sub_problem_ids": sub_problem_ids,
            "problem_text": problem_text,
            "status": "pending",
            "node_color": color,
            "member_count": len(sub_problem_ids)
        }
    }
    
    logger.debug(f"Created super-node bubble: {bubble['id']}")
    return bubble


# =============================================================================
# Additional Bubble Types
# =============================================================================

def create_adaptive_strategy_bubble(
    config: AdaptiveStrategyBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create an adaptive strategy selection bubble.
    
    Automatically selects the best optimization strategy based on problem characteristics.
    
    Args:
        config: AdaptiveStrategyBubbleConfig with strategy configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing an adaptive strategy bubble
    """
    position = position or {"x": 200, "y": 300}
    icon = Z3_NODE_ICONS.get("adaptive_strategy", "🎯")
    color = Z3_NODE_COLORS.get("adaptive_strategy", "#FD79A8")
    label = label or f"{icon} Adaptive Strategy"
    
    bubble = {
        "id": f"adaptive_strategy_{uuid.uuid4().hex[:8]}",
        "type": "adaptive_strategy",
        "position": position,
        "data": {
            "label": label,
            "problem_text": config.problem_text,
            "available_strategies": config.available_strategies,
            "selection_criteria": config.selection_criteria,
            "enable_fallback": config.enable_fallback,
            "max_strategy_switches": config.max_strategy_switches,
            "selected_strategy": None,
            "strategy_history": [],
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created adaptive strategy bubble: {bubble['id']}")
    return bubble


def create_quality_assessment_bubble(
    config: QualityAssessmentBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a quality assessment bubble.
    
    Evaluates solution quality across multiple dimensions.
    
    Args:
        config: QualityAssessmentBubbleConfig with assessment configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a quality assessment bubble
    """
    position = position or {"x": 250, "y": 350}
    icon = Z3_NODE_ICONS.get("quality_assessment", "📊")
    color = Z3_NODE_COLORS.get("quality_assessment", "#00CEC9")
    label = label or f"{icon} Quality Assessment"
    
    bubble = {
        "id": f"quality_assessment_{uuid.uuid4().hex[:8]}",
        "type": "quality_assessment",
        "position": position,
        "data": {
            "label": label,
            "problem_text": config.problem_text,
            "assessment_dimensions": config.assessment_dimensions,
            "threshold_mode": config.threshold_mode,
            "min_quality_score": config.min_quality_score,
            "scores": {},
            "overall_score": 0.0,
            "passed": False,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created quality assessment bubble: {bubble['id']}")
    return bubble


def create_knowledge_extraction_bubble(
    config: KnowledgeExtractionBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a knowledge extraction bubble.
    
    Extracts patterns, constraints, and heuristics from solutions.
    
    Args:
        config: KnowledgeExtractionBubbleConfig with extraction configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a knowledge extraction bubble
    """
    position = position or {"x": 300, "y": 400}
    icon = Z3_NODE_ICONS.get("knowledge_extraction", "🧠")
    color = Z3_NODE_COLORS.get("knowledge_extraction", "#FAB1A0")
    label = label or f"{icon} Knowledge Extraction"
    
    bubble = {
        "id": f"knowledge_extraction_{uuid.uuid4().hex[:8]}",
        "type": "knowledge_extraction",
        "position": position,
        "data": {
            "label": label,
            "solution_text": config.solution_text,
            "extraction_type": config.extraction_type,
            "store_in_graph": config.store_in_graph,
            "confidence_threshold": config.confidence_threshold,
            "max_patterns": config.max_patterns,
            "extracted_patterns": [],
            "extraction_count": 0,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created knowledge extraction bubble: {bubble['id']}")
    return bubble


def create_metrics_analytics_bubble(
    config: MetricsAnalyticsBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a metrics and analytics bubble.
    
    Collects and aggregates workflow metrics.
    
    Args:
        config: MetricsAnalyticsBubbleConfig with metrics configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a metrics analytics bubble
    """
    position = position or {"x": 350, "y": 450}
    icon = Z3_NODE_ICONS.get("metrics_analytics", "📈")
    color = Z3_NODE_COLORS.get("metrics_analytics", "#74B9FF")
    label = label or f"{icon} Metrics & Analytics"
    
    bubble = {
        "id": f"metrics_analytics_{uuid.uuid4().hex[:8]}",
        "type": "metrics_analytics",
        "position": position,
        "data": {
            "label": label,
            "metrics_types": config.metrics_types,
            "track_historical": config.track_historical,
            "aggregation_level": config.aggregation_level,
            "export_format": config.export_format,
            "metrics_data": {},
            "historical_data": [],
            "aggregates": {},
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created metrics analytics bubble: {bubble['id']}")
    return bubble


def create_cache_bubble(
    config: CacheBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a cache bubble.
    
    Manages caching of intermediate results.
    
    Args:
        config: CacheBubbleConfig with cache configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a cache bubble
    """
    position = position or {"x": 400, "y": 500}
    icon = Z3_NODE_ICONS.get("cache", "💾")
    color = Z3_NODE_COLORS.get("cache", "#DFE6E9")
    label = label or f"{icon} Cache ({config.cache_type})"
    
    bubble = {
        "id": f"cache_{uuid.uuid4().hex[:8]}",
        "type": "cache",
        "position": position,
        "data": {
            "label": label,
            "cache_key": config.cache_key,
            "cache_type": config.cache_type,
            "ttl_seconds": config.ttl_seconds,
            "max_size_mb": config.max_size_mb,
            "compression_enabled": config.compression_enabled,
            "hits": 0,
            "misses": 0,
            "cached_data": None,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created cache bubble: {bubble['id']}")
    return bubble


def create_agent_coordination_bubble(
    config: AgentCoordinationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create an agent coordination bubble.
    
    Coordinates multiple agents in the workflow.
    
    Args:
        config: AgentCoordinationBubbleConfig with coordination configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing an agent coordination bubble
    """
    position = position or {"x": 450, "y": 550}
    icon = Z3_NODE_ICONS.get("agent_coordination", "🤝")
    color = Z3_NODE_COLORS.get("agent_coordination", "#B2BEC3")
    label = label or f"{icon} Agent Coordination"
    
    bubble = {
        "id": f"agent_coordination_{uuid.uuid4().hex[:8]}",
        "type": "agent_coordination",
        "position": position,
        "data": {
            "label": label,
            "agent_ids": config.agent_ids,
            "coordination_type": config.coordination_type,
            "task_distribution": config.task_distribution,
            "conflict_resolution": config.conflict_resolution,
            "sync_interval_ms": config.sync_interval_ms,
            "agent_status": {},
            "task_assignments": [],
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created agent coordination bubble: {bubble['id']}")
    return bubble


def create_error_handler_bubble(
    config: ErrorHandlerBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create an error handler bubble.
    
    Handles and recovers from errors in the workflow.
    
    Args:
        config: ErrorHandlerBubbleConfig with error handling configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing an error handler bubble
    """
    position = position or {"x": 500, "y": 600}
    icon = Z3_NODE_ICONS.get("error_handler", "🚨")
    color = Z3_NODE_COLORS.get("error_handler", "#D63031")
    label = label or f"{icon} Error Handler"
    
    bubble = {
        "id": f"error_handler_{uuid.uuid4().hex[:8]}",
        "type": "error_handler",
        "position": position,
        "data": {
            "label": label,
            "error_types": config.error_types,
            "recovery_strategy": config.recovery_strategy,
            "max_retries": config.max_retries,
            "escalation_threshold": config.escalation_threshold,
            "notify_on_escalation": config.notify_on_escalation,
            "error_count": 0,
            "recovery_count": 0,
            "escalation_count": 0,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created error handler bubble: {bubble['id']}")
    return bubble


def create_refinement_bubble(
    config: RefinementBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a solution refinement bubble.
    
    Iteratively refines solutions to improve quality.
    
    Args:
        config: RefinementBubbleConfig with refinement configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a refinement bubble
    """
    position = position or {"x": 550, "y": 650}
    icon = Z3_NODE_ICONS.get("refinement", "🔧")
    color = Z3_NODE_COLORS.get("refinement", "#E17055")
    label = label or f"{icon} Refinement"
    
    bubble = {
        "id": f"refinement_{uuid.uuid4().hex[:8]}",
        "type": "refinement",
        "position": position,
        "data": {
            "label": label,
            "solution_text": config.solution_text,
            "refinement_type": config.refinement_type,
            "max_refinements": config.max_refinements,
            "convergence_check": config.convergence_check,
            "preserve_constraints": config.preserve_constraints,
            "current_refinement": 0,
            "refined_solutions": [],
            "converged": False,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created refinement bubble: {bubble['id']}")
    return bubble


def create_composition_bubble(
    config: CompositionBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a solution composition bubble.
    
    Combines multiple sub-solutions into a complete solution.
    
    Args:
        config: CompositionBubbleConfig with composition configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a composition bubble
    """
    position = position or {"x": 600, "y": 700}
    icon = Z3_NODE_ICONS.get("composition", "🧩")
    color = Z3_NODE_COLORS.get("composition", "#00B894")
    label = label or f"{icon} Composition"
    
    bubble = {
        "id": f"composition_{uuid.uuid4().hex[:8]}",
        "type": "composition",
        "position": position,
        "data": {
            "label": label,
            "sub_solutions": config.sub_solutions,
            "composition_strategy": config.composition_strategy,
            "conflict_resolution": config.conflict_resolution,
            "validate_composition": config.validate_composition,
            "composed_solution": None,
            "conflicts": [],
            "composition_valid": False,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created composition bubble: {bubble['id']}")
    return bubble


def create_validation_bubble(
    config: ValidationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a validation bubble.
    
    Validates solutions against problem requirements.
    
    Args:
        config: ValidationBubbleConfig with validation configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a validation bubble
    """
    position = position or {"x": 650, "y": 750}
    icon = Z3_NODE_ICONS.get("validation", "[OK]")
    color = Z3_NODE_COLORS.get("validation", "#0984E3")
    label = label or f"{icon} Validation"
    
    bubble = {
        "id": f"validation_{uuid.uuid4().hex[:8]}",
        "type": "validation",
        "position": position,
        "data": {
            "label": label,
            "problem_text": config.problem_text,
            "solution_text": config.solution_text,
            "validation_type": config.validation_type,
            "check_soundness": config.check_soundness,
            "check_completeness": config.check_completeness,
            "timeout_seconds": config.timeout_seconds,
            "soundness_check": None,
            "completeness_check": None,
            "valid": False,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created validation bubble: {bubble['id']}")
    return bubble


def create_parallelization_bubble(
    config: ParallelizationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a parallelization bubble.
    
    Manages parallel execution of tasks.
    
    Args:
        config: ParallelizationBubbleConfig with parallelization configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a parallelization bubble
    """
    position = position or {"x": 700, "y": 800}
    icon = Z3_NODE_ICONS.get("parallelization", "⚡")
    color = Z3_NODE_COLORS.get("parallelization", "#6C5CE7")
    label = label or f"{icon} Parallelization"
    
    bubble = {
        "id": f"parallelization_{uuid.uuid4().hex[:8]}",
        "type": "parallelization",
        "position": position,
        "data": {
            "label": label,
            "tasks": config.tasks,
            "parallel_mode": config.parallel_mode,
            "max_workers": config.max_workers,
            "load_balancing": config.load_balancing,
            "result_aggregation": config.result_aggregation,
            "active_tasks": [],
            "completed_tasks": [],
            "results_aggregated": None,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created parallelization bubble: {bubble['id']}")
    return bubble


def create_convergence_divergence_bubble(
    config: ConvergenceDivergenceBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a convergence/divergence detection bubble.
    
    Detects convergence (solutions improving) or divergence (solutions degrading).
    
    Args:
        config: ConvergenceDivergenceBubbleConfig with detection configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a convergence/divergence bubble
    """
    position = position or {"x": 750, "y": 850}
    icon = Z3_NODE_ICONS.get("convergence", "🎯")
    color = Z3_NODE_COLORS.get("convergence", "#FDCB6E")
    label = label or f"{icon} Convergence/Divergence"
    
    bubble = {
        "id": f"convergence_divergence_{uuid.uuid4().hex[:8]}",
        "type": "convergence_divergence",
        "position": position,
        "data": {
            "label": label,
            "metric_history": config.metric_history,
            "detection_type": config.detection_type,
            "window_size": config.window_size,
            "threshold": config.threshold,
            "trend_analysis": config.trend_analysis,
            "convergence_detected": False,
            "divergence_detected": False,
            "trend": None,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created convergence/divergence bubble: {bubble['id']}")
    return bubble


# =============================================================================
# Edge Creation Functions
# =============================================================================

def create_z3_edge(
    source_id: str,
    target_id: str,
    edge_type: str = "default",
    source_handle: str = "output",
    target_handle: str = "input"
) -> Dict[str, Any]:
    """
    Create an edge connecting Z3/LeanAIDE bubbles.
    
    Args:
        source_id: ID of the source bubble
        target_id: ID of the target bubble
        edge_type: Type of edge (default, conditional, feedback)
        source_handle: Handle on source bubble
        target_handle: Handle on target bubble
    
    Returns:
        Dict representing an edge
    """
    edge = {
        "id": f"edge_{source_id}_{target_id}_{uuid.uuid4().hex[:8]}",
        "source": source_id,
        "target": target_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
        "type": edge_type,
        "animated": edge_type == "default",
        "style": {
            "stroke": get_z3_edge_color(edge_type),
            "strokeWidth": 2
        }
    }
    
    return edge


def get_z3_edge_color(edge_type: str) -> str:
    """Get color for edge type."""
    colors = {
        "default": "#888888",
        "conditional": "#FF6B6B",
        "feedback": "#9B59B6",
        "success": "#00B894",
        "error": "#FF7675",
    }
    return colors.get(edge_type, "#888888")


def create_conditional_z3_edge(
    source_id: str,
    target_id: str,
    condition: str
) -> Dict[str, Any]:
    """Create a conditional edge with labeled condition."""
    edge = create_z3_edge(source_id, target_id, "conditional")
    edge["label"] = condition
    edge["labelStyle"] = {"fill": "#FF6B6B", "fontSize": 12}
    return edge


def create_feedback_z3_edge(
    source_id: str,
    target_id: str
) -> Dict[str, Any]:
    """Create a feedback edge for iterative verification."""
    return create_z3_edge(
        source_id, target_id, "feedback",
        "feedback", "retry"
    )


# =============================================================================
# Complete Z3 Workflow Creation
# =============================================================================

def create_z3_solver_workflow(
    problem_text: str,
    workflow_name: str = "Z3 Solver Workflow",
    variables: List[Dict[str, Any]] = None,
    constraints: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete Z3 constraint solving workflow.
    
    Args:
        problem_text: The constraint problem to solve
        workflow_name: Name of the workflow
        variables: Optional list of variables
        constraints: Optional list of constraints
    
    Returns:
        Dict with complete workflow definition
    """
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": Z3_NODE_POSITIONS["input"],
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create solver bubble
    solver_config = Z3SolverBubbleConfig(
        problem_text=problem_text,
        variables=variables or [],
        constraints=constraints or []
    )
    solver_bubble = create_z3_solver_bubble(solver_config)
    nodes.append(solver_bubble)
    
    # Connect input to solver
    edges.append(create_z3_edge(input_bubble["id"], solver_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    
    # Connect solver to result
    edges.append(create_z3_edge(solver_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Z3 solver workflow for: {problem_text[:50]}...",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "workflow_type": "z3_solver",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created Z3 workflow: {workflow['id']}")
    return workflow


def create_z3_leanaide_workflow(
    problem_text: str,
    workflow_name: str = "Z3-LeanAIDE Verification",
    include_proof: bool = True,
    include_cross_verify: bool = True
) -> Dict[str, Any]:
    """
    Create a complete Z3 + LeanAIDE verification workflow.
    
    Args:
        problem_text: The problem to verify
        workflow_name: Name of the workflow
        include_proof: Whether to include LeanAIDE proof
        include_cross_verify: Whether to include cross-verification
    
    Returns:
        Dict with complete workflow definition
    """
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": Z3_NODE_POSITIONS["input"],
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create classification bubble
    class_config = ProblemClassificationBubbleConfig(problem_text=problem_text)
    class_bubble = create_problem_classification_bubble(class_config)
    nodes.append(class_bubble)
    edges.append(create_z3_edge(input_bubble["id"], class_bubble["id"]))
    
    # Create Z3 solver bubble
    solver_config = Z3SolverBubbleConfig(problem_text=problem_text)
    solver_bubble = create_z3_solver_bubble(solver_config)
    nodes.append(solver_bubble)
    edges.append(create_z3_edge(class_bubble["id"], solver_bubble["id"]))
    
    # Create LeanAIDE proof bubble (optional)
    if include_proof:
        proof_config = LeanAideProofBubbleConfig(
            theorem_name=workflow_name,
            proof_type="theorem"
        )
        proof_bubble = create_leanaide_proof_bubble(proof_config)
        nodes.append(proof_bubble)
        edges.append(create_z3_edge(solver_bubble["id"], proof_bubble["id"]))
    
    # Create cross-verification bubble (optional)
    if include_cross_verify:
        cross_config = CrossVerificationBubbleConfig(problem_text=problem_text)
        cross_bubble = create_cross_verification_bubble(cross_config)
        nodes.append(cross_bubble)
        
        last_node = proof_bubble["id"] if include_proof else solver_bubble["id"]
        edges.append(create_z3_edge(last_node, cross_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    
    # Connect last node to result
    if include_cross_verify:
        edges.append(create_z3_edge(cross_bubble["id"], result_bubble["id"]))
    elif include_proof:
        edges.append(create_z3_edge(proof_bubble["id"], result_bubble["id"]))
    else:
        edges.append(create_z3_edge(solver_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Z3-LeanAIDE verification for: {problem_text[:50]}...",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "workflow_type": "z3_leanaide",
            "include_proof": include_proof,
            "include_cross_verify": include_cross_verify,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created Z3-LeanAIDE workflow: {workflow['id']}")
    return workflow


def create_z3_workflow_with_entanglement(
    problem_text: str,
    sub_problems: List[Dict[str, Any]],
    entanglement_matrix: Dict[str, List[str]] = None,
    workflow_name: str = "Entangled Z3 Workflow",
    loop_strategy: str = "sequential"
) -> Dict[str, Any]:
    """
    Create a Z3 workflow with sub-problem loops and entanglement matrix integration.
    
    This workflow supports:
    - Iterative solving of entangled sub-problems
    - Convergence-based refinement
    - Super-node detection for tightly coupled problems
    
    Args:
        problem_text: The overall problem description
        sub_problems: List of sub-problem dicts with 'id', 'problem_text', optional 'shared_symbols'
        entanglement_matrix: Dict mapping sub-problem IDs to lists of entangled IDs
        workflow_name: Name of the workflow
        loop_strategy: Strategy for sub-problem loops (sequential, parallel, super_node)
    
    Returns:
        Dict with complete workflow definition including entanglement visualization
    """
    # Build entanglement matrix if not provided
    if entanglement_matrix is None:
        entanglement_matrix = build_entanglement_from_subproblems(sub_problems)
    
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": {"x": 0, "y": 0},
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create entanglement visualization bubble
    ent_viz_config = EntanglementVisualizationConfig(
        entanglement_matrix=entanglement_matrix,
        sub_problems=sub_problems,
        show_coupling_strength=True,
        highlight_super_nodes=True
    )
    ent_viz_bubble = create_entanglement_visualization_bubble(ent_viz_config)
    nodes.append(ent_viz_bubble)
    edges.append(create_z3_edge(input_bubble["id"], ent_viz_bubble["id"]))
    
    # Create sub-problem loop bubble
    loop_config = SubProblemLoopBubbleConfig(
        sub_problems=sub_problems,
        entanglement_matrix=entanglement_matrix,
        loop_strategy=loop_strategy,
        max_iterations=10,
        convergence_threshold=0.95
    )
    loop_bubble = create_subproblem_loop_bubble(loop_config)
    nodes.append(loop_bubble)
    edges.append(create_z3_edge(ent_viz_bubble["id"], loop_bubble["id"]))
    
    # Create individual sub-problem bubbles
    sub_problem_bubbles = []
    for i, sp in enumerate(sub_problems):
        sp_id = sp.get("id", f"sp_{i}")
        entangled_with = entanglement_matrix.get(sp_id, [])
        
        # Check if this is a super-node
        matrix = normalize_entanglement_matrix_z3(entanglement_matrix)
        avg_degree = sum(len(neighbors) for neighbors in matrix.values()) / max(len(matrix), 1)
        is_super_node = len(matrix.get(sp_id, set())) > avg_degree * 1.5
        
        sp_config = SubProblemBubbleConfig(
            sub_problem_id=sp_id,
            problem_text=sp.get("problem_text", f"Sub-problem {sp_id}"),
            entangled_with=entangled_with,
            entanglement_source=sp.get("entanglement_source", "symbolic_overlap"),
            is_super_node=is_super_node
        )
        sp_bubble = create_subproblem_bubble(
            sp_config,
            position={"x": 400, "y": 200 + i * 80}
        )
        nodes.append(sp_bubble)
        sub_problem_bubbles.append(sp_bubble)
        
        # Connect loop to each sub-problem
        edges.append(create_z3_edge(loop_bubble["id"], sp_bubble["id"], "feedback"))
    
    # Create cross-verification for entangled pairs
    cross_bubble = create_cross_verification_bubble(
        CrossVerificationBubbleConfig(problem_text=problem_text)
    )
    nodes.append(cross_bubble)
    
    # Connect all sub-problems to cross-verification
    for sp_bubble in sub_problem_bubbles:
        edges.append(create_z3_edge(sp_bubble["id"], cross_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    edges.append(create_z3_edge(cross_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Entangled Z3 workflow with {len(sub_problems)} sub-problems",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "sub_problems": sub_problems,
            "entanglement_matrix": entanglement_matrix,
            "workflow_type": "z3_entangled",
            "loop_strategy": loop_strategy,
            "coupling_density": ent_viz_config.get_coupling_density(),
            "super_nodes": ent_viz_bubble["data"]["super_nodes"],
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created entangled Z3 workflow: {workflow['id']}")
    return workflow


# =============================================================================
# Flexible Workflow Builder for Z3/LeanAIDE
# =============================================================================

@dataclass
class Z3BubbleDefinition:
    """Definition of a Z3/LeanAIDE bubble for user-defined workflows."""
    bubble_type: str  # z3_solver, z3_prover, leanaide_proof, cross_verification, classification
    label: str
    position: Dict[str, float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    node_color: str = "#888888"


@dataclass  
class Z3EdgeDefinition:
    """Definition of an edge for Z3/LeanAIDE workflows."""
    source_label: str
    target_label: str
    edge_type: str = "default"
    condition: str = ""
    source_handle: str = "output"
    target_handle: str = "input"


class Z3FlexibleWorkflowBuilder:
    """Builder for creating arbitrary Z3/LeanAIDE workflow patterns with CAV-NLP support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.bubbles: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.bubble_map: Dict[str, Dict[str, Any]] = {}
        
        # Initialize CAV-NLP components
        config = config or {}
        self.use_cav_nlp = config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        self.enhanced_solver = None
        self.math_service = None
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP enhancement enabled for Z3FlexibleWorkflowBuilder")
    
    def add_bubble(self, bubble_def: Z3BubbleDefinition) -> str:
        """Add a bubble to the workflow."""
        bubble_id = f"{bubble_def.bubble_type}_{uuid.uuid4().hex[:8]}"
        
        position = bubble_def.position or {"x": len(self.bubbles) * 150, "y": 0}
        
        bubble = {
            "id": bubble_id,
            "type": bubble_def.bubble_type,
            "position": position,
            "data": {
                "label": bubble_def.label,
                "status": "pending",
                "node_color": bubble_def.node_color,
                **bubble_def.config
            }
        }
        
        self.bubbles.append(bubble)
        self.bubble_map[bubble_def.label] = bubble
        
        return bubble_id
    
    def add_edge(self, edge_def: Z3EdgeDefinition) -> str:
        """Add an edge connecting two bubbles."""
        source_bubble = self.bubble_map.get(edge_def.source_label)
        target_bubble = self.bubble_map.get(edge_def.target_label)
        
        if not source_bubble:
            raise ValueError(f"Source bubble not found: {edge_def.source_label}")
        if not target_bubble:
            raise ValueError(f"Target bubble not found: {edge_def.target_label}")
        
        edge_id = f"edge_{source_bubble['id']}_{target_bubble['id']}_{uuid.uuid4().hex[:8]}"
        
        edge = {
            "id": edge_id,
            "source": source_bubble["id"],
            "target": target_bubble["id"],
            "sourceHandle": edge_def.source_handle,
            "targetHandle": edge_def.target_handle,
            "type": edge_def.edge_type,
            "animated": edge_def.edge_type == "default",
            "style": {
                "stroke": get_z3_edge_color(edge_def.edge_type),
                "strokeWidth": 2
            }
        }
        
        if edge_def.condition:
            edge["label"] = edge_def.condition
            edge["labelStyle"] = {"fill": "#FF6B6B", "fontSize": 12}
        
        self.edges.append(edge)
        return edge_id
    
    def build(self, workflow_name: str, problem_text: str = "") -> Dict[str, Any]:
        """Build the complete workflow."""
        return {
            "id": str(uuid.uuid4()),
            "name": workflow_name,
            "description": problem_text or f"Z3/LeanAIDE workflow: {workflow_name}",
            "nodes": self.bubbles,
            "edges": self.edges,
            "metadata": {
                "workflow_type": "z3_leanaide_custom",
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
    
    def reset(self):
        """Reset the builder."""
        self.bubbles = []
        self.edges = []
        self.bubble_map = {}
    
    def export_proof_to_lean(self, z3_proof) -> str:
        """Export Z3 proof to Lean 4 for bubble workflows.
        
        This method converts Z3 proof objects to Lean 4 compatible format,
        enabling integration with LeanAIDE workflows in BubbleLabs.
        
        Args:
            z3_proof: Z3 proof object to export
            
        Returns:
            Lean 4 formatted proof string
        """
        if not self.use_cav_nlp or not self.enhanced_solver:
            logger.warning("CAV-NLP not available for proof export")
            return "-- CAV-NLP not available for proof export"
        
        try:
            return self.enhanced_solver.proof_exporter.export_proof(z3_proof)
        except Exception as e:
            logger.error(f"Proof export failed: {e}")
            return f"-- Proof export failed: {str(e)}"


def create_custom_z3_workflow(
    workflow_name: str,
    problem_text: str,
    bubble_labels: List[str],
    bubble_types: List[str],
    team_config: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a custom Z3/LeanAIDE workflow from label/type lists.
    
    Args:
        workflow_name: Name of the workflow
        problem_text: Problem description
        bubble_labels: Ordered list of bubble labels
        bubble_types: Ordered list of bubble types
        team_config: Optional team mapping
    
    Returns:
        Dict with workflow definition
    """
    builder = Z3FlexibleWorkflowBuilder()
    team_config = team_config or {}
    
    for i, (label, btype) in enumerate(zip(bubble_labels, bubble_types)):
        color = Z3_NODE_COLORS.get(btype, "#888888")
        config = {"problem_text": problem_text} if i == 0 else {}
        
        if btype == "z3_solver":
            config = {"problem_text": problem_text, "variables": [], "constraints": []}
        elif btype == "classification":
            config = {"problem_text": problem_text, "auto_classify": True}
        
        bubble_def = Z3BubbleDefinition(
            bubble_type=btype,
            label=label,
            node_color=color,
            config=config
        )
        builder.add_bubble(bubble_def)
    
    # Create sequential edges
    for i in range(len(bubble_labels) - 1):
        edge_def = Z3EdgeDefinition(
            source_label=bubble_labels[i],
            target_label=bubble_labels[i + 1]
        )
        builder.add_edge(edge_def)
    
    return builder.build(workflow_name, problem_text)


# =============================================================================
# Bubble Update Functions
# =============================================================================

def update_z3_bubble_status(
    bubble: Dict[str, Any],
    status: str,
    additional_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Update the status of a Z3/LeanAIDE bubble."""
    bubble["data"]["status"] = status
    
    if additional_data:
        bubble["data"].update(additional_data)
    
    status_colors = {
        "pending": "#FDCB6E",
        "running": "#74B9FF",
        "success": "#00B894",
        "failed": "#FF7675",
        "verified": "#00B894",
    }
    
    if status in status_colors:
        bubble["data"]["node_color"] = status_colors[status]
    
    return bubble


def add_z3_result_to_bubble(
    bubble: Dict[str, Any],
    success: bool,
    result_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Add result data to a Z3/LeanAIDE bubble."""
    bubble["data"]["result"] = result_data
    bubble["data"]["status"] = "success" if success else "failed"
    bubble["data"]["node_color"] = Z3_NODE_COLORS.get("result", "#00B894")
    
    return bubble


# =============================================================================
# Serialization and Export
# =============================================================================

def serialize_z3_bubble(bubble: Dict[str, Any]) -> str:
    """Serialize a Z3 bubble to JSON string."""
    import json
    return json.dumps(bubble, indent=2)


def serialize_z3_workflow(workflow: Dict[str, Any]) -> str:
    """Serialize a Z3 workflow to JSON string."""
    import json
    return json.dumps(workflow, indent=2)


def export_z3_workflow_to_json(
    workflow: Dict[str, Any],
    output_path: str
) -> bool:
    """Export a Z3 workflow to a JSON file."""
    import json
    import os
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        logger.info(f"Exported Z3 workflow to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export workflow: {e}")
        return False


# =============================================================================
# Example Usage
# =============================================================================

def example_z3_workflow():
    """Example: Create and export a Z3 solver workflow."""
    workflow = create_z3_solver_workflow(
        problem_text="Find x, y such that: x + y = 10, x * y = 24",
        workflow_name="Equation Solver"
    )
    
    export_z3_workflow_to_json(workflow, "z3_workflow_example.json")
    
    return workflow


def example_z3_leanaide_workflow():
    """Example: Create and export a Z3-LeanAIDE workflow."""
    workflow = create_z3_leanaide_workflow(
        problem_text="Prove that for all natural numbers n, n^2 >= n",
        workflow_name="Theorem Verification",
        include_proof=True,
        include_cross_verify=True
    )
    
    export_z3_workflow_to_json(workflow, "z3_leanaide_workflow_example.json")
    
    return workflow


def example_entangled_z3_workflow():
    """Example: Create and export an entangled Z3 workflow with sub-problem loops."""
    # Define sub-problems with shared symbols for entanglement
    sub_problems = [
        {
            "id": "sp_A",
            "problem_text": "Solve for x: x + y = 10",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_B", 
            "problem_text": "Solve for y: x * y = 24",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_C",
            "problem_text": "Verify x > 0 and y > 0",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_D",
            "problem_text": "Calculate x^2 + y^2",
            "shared_symbols": ["x", "y"]
        }
    ]
    
    # Entanglement matrix (automatically computed from shared_symbols)
    workflow = create_z3_workflow_with_entanglement(
        problem_text="System of equations with entangled variables",
        sub_problems=sub_problems,
        workflow_name="Entangled Equation Solver",
        loop_strategy="sequential"
    )
    
    export_z3_workflow_to_json(workflow, "z3_entangled_workflow_example.json")
    
    return workflow

if __name__ == "__main__":
    # Run examples
    workflow = example_z3_workflow()
    print(f"Created workflow: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    print(f"Edges: {len(workflow['edges'])}")
    
    # Run entanglement example
    ent_workflow = example_entangled_z3_workflow()
    print(f"\nCreated entangled workflow: {ent_workflow['name']}")
    print(f"Nodes: {len(ent_workflow['nodes'])}")
    print(f"Edges: {len(ent_workflow['edges'])}")
    print(f"Coupling density: {ent_workflow['metadata'].get('coupling_density', 'N/A')}")
    print(f"Super nodes: {ent_workflow['metadata'].get('super_nodes', [])}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    'Z3SolverBubbleConfig',
    'Z3ProverBubbleConfig',
    'LeanAideProofBubbleConfig',
    'CrossVerificationBubbleConfig',
    'ProblemClassificationBubbleConfig',
    'SubProblemLoopBubbleConfig',
    'EntanglementVisualizationConfig',
    'SubProblemBubbleConfig',
    
    # Additional config classes
    'AdaptiveStrategyBubbleConfig',
    'QualityAssessmentBubbleConfig',
    'KnowledgeExtractionBubbleConfig',
    'MetricsAnalyticsBubbleConfig',
    'CacheBubbleConfig',
    'AgentCoordinationBubbleConfig',
    'ErrorHandlerBubbleConfig',
    'RefinementBubbleConfig',
    'CompositionBubbleConfig',
    'ValidationBubbleConfig',
    'ParallelizationBubbleConfig',
    'ConvergenceDivergenceBubbleConfig',
    
    # Additional extended config classes
    'AdversarialTestingBubbleConfig',
    'MDAPBubbleConfig',
    'DecompositionBubbleConfig',
    'RecompositionBubbleConfig',
    'MonitoringBubbleConfig',
    'AlertingBubbleConfig',
    'SecurityBubbleConfig',
    'APIGatewayBubbleConfig',
    'LoggingBubbleConfig',
    'VisualizationBubbleConfig',
    'CachingBubbleConfig',
    'BatchProcessingBubbleConfig',
    'ExperimentTrackingBubbleConfig',
    'PromptEngineeringBubbleConfig',
    'EvaluationBubbleConfig',
    'DeploymentBubbleConfig',
    'OptimizationBubbleConfig',
    'HeuristicBubbleConfig',
    'SamplingBubbleConfig',
    'EnsembleBubbleConfig',
    
    # Builder definition classes
    'Z3BubbleDefinition',
    'Z3EdgeDefinition',
    
    # Entanglement utilities
    'normalize_entanglement_matrix_z3',
    'serialize_entanglement_matrix_z3',
    'build_entanglement_from_subproblems',
    
    # Core bubble creation
    'create_z3_solver_bubble',
    'create_z3_prover_bubble',
    'create_leanaide_proof_bubble',
    'create_cross_verification_bubble',
    'create_problem_classification_bubble',
    'create_z3_result_bubble',
    
    # Entanglement bubble creation
    'create_subproblem_loop_bubble',
    'create_entanglement_visualization_bubble',
    'create_subproblem_bubble',
    'create_super_node_bubble',
    
    # Additional bubble creation
    'create_adaptive_strategy_bubble',
    'create_quality_assessment_bubble',
    'create_knowledge_extraction_bubble',
    'create_metrics_analytics_bubble',
    'create_cache_bubble',
    'create_agent_coordination_bubble',
    'create_error_handler_bubble',
    'create_refinement_bubble',
    'create_composition_bubble',
    'create_validation_bubble',
    'create_parallelization_bubble',
    'create_convergence_divergence_bubble',
    
    # Edge creation
    'create_z3_edge',
    'create_conditional_z3_edge',
    'create_feedback_z3_edge',
    
    # Workflow creation
    'create_z3_solver_workflow',
    'create_z3_leanaide_workflow',
    'create_z3_workflow_with_entanglement',
    
    # Flexible builder
    'Z3FlexibleWorkflowBuilder',
    'create_custom_z3_workflow',
    
    # Updates
    'update_z3_bubble_status',
    'add_z3_result_to_bubble',
    
    # Serialization
    'serialize_z3_bubble',
    'serialize_z3_workflow',
    'export_z3_workflow_to_json',
]
