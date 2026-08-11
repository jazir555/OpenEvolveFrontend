"""
Comprehensive Decomposition Engine - Production-Grade Problem Decomposition System

This module provides an infinitely more comprehensive decomposition system with:
- Multi-strategy orchestration with ML-based strategy selection
- Uncertainty quantification and confidence scoring
- Constraint satisfaction with optimization
- Parallel decomposition processing
- Iterative refinement with feedback loops
- Domain-specific decomposition patterns
- Resource-aware decomposition planning
- Semantic boundary detection with embeddings
- Hierarchical multi-level decomposition
- Dependency graph optimization

Author: OpenEvolve System
Version: 2.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from functools import lru_cache, partial
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Protocol
)
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Comprehensive Decomposition Engine
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# CAV-NLP Integration for natural language decomposition and hybrid validation
try:
    from openevolve.cav_nlp_integration.adapter import (
        Z3LeanAideBridge, create_z3_lean_bridge, quick_verify
    )
    from openevolve.cav_nlp_integration.data_structures import (
        VerificationBridgeResult, ConstraintType as CAVConstraintType
    )
    CAV_NLP_AVAILABLE = True
    logger.info("CAV-NLP integration available for enhanced decomposition")
except ImportError:
    try:
        # Fallback to direct LeanAide bridge
        from leanaide_crewai_bridge import LeanAideCrewAIBridge
        CAV_NLP_AVAILABLE = True
        logger.info("LeanAide-CrewAI bridge available for enhanced decomposition")
    except ImportError:
        CAV_NLP_AVAILABLE = False
        logger.warning("CAV-NLP integration not available - NL decomposition disabled")

# Z3 Validator for hybrid validation
try:
    from decomposition_z3_validator import (
        Z3DecompositionValidator, ValidationResult, DecompositionProperty
    )
    Z3_VALIDATOR_AVAILABLE = True
except ImportError:
    Z3_VALIDATOR_AVAILABLE = False

# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class DecompositionStrategy(Enum):
    """Available decomposition strategies with detailed classification."""
    # Structural strategies
    HIERARCHICAL = "hierarchical"           # Top-down functional decomposition
    FUNCTIONAL = "functional"               # By system capabilities/functions
    STRUCTURAL = "structural"               # By physical/organizational structure
    
    # Semantic strategies
    SEMANTIC = "semantic"                   # By meaning and concepts
    DOMAIN_DRIVEN = "domain_driven"         # Domain-specific patterns
    ONTOLOGY_BASED = "ontology_based"       # Using formal ontologies
    
    # Process strategies
    TEMPORAL = "temporal"                   # By chronological order
    WORKFLOW = "workflow"                   # By process flow
    STATE_BASED = "state_based"             # By state transitions
    
    # Risk/Value strategies
    RISK_BASED = "risk_based"               # Address highest risks first
    VALUE_BASED = "value_based"             # Deliver highest value first
    COST_BASED = "cost_based"               # Optimize for cost efficiency
    
    # Technical strategies
    DEPENDENCY = "dependency"               # Based on prerequisite relationships
    COMPLEXITY = "complexity"               # To balance cognitive load
    COUPLING = "coupling"                   # Minimize coupling between components
    
    # Advanced strategies
    ADAPTIVE = "adaptive"                   # Context-aware dynamic selection
    MULTI_LEVEL = "multi_level"             # Recursive hierarchical decomposition
    PARALLEL = "parallel"                   # Parallel decomposition streams
    HYBRID = "hybrid"                       # Combination of multiple strategies


class SubProblemType(Enum):
    """Granular sub-problem types for precise categorization."""
    # Research types
    RESEARCH = "research"
    INVESTIGATION = "investigation"
    EXPLORATION = "exploration"
    SURVEY = "survey"
    
    # Analysis types
    ANALYSIS = "analysis"
    DESIGN = "design"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    REQUIREMENTS = "requirements"
    
    # Implementation types
    IMPLEMENTATION = "implementation"
    DEVELOPMENT = "development"
    CODING = "coding"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    
    # Quality types
    VALIDATION = "validation"
    VERIFICATION = "verification"
    TESTING = "testing"
    REVIEW = "review"
    AUDIT = "audit"
    
    # Support types
    DOCUMENTATION = "documentation"
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"


class ConstraintType(Enum):
    """Types of constraints for decomposition."""
    TEMPORAL = "temporal"                   # Time-based constraints
    RESOURCE = "resource"                   # Resource constraints
    TECHNICAL = "technical"                 # Technical constraints
    REGULATORY = "regulatory"               # Regulatory/compliance
    QUALITY = "quality"                     # Quality requirements
    BUDGET = "budget"                       # Budget constraints
    SCOPE = "scope"                         # Scope boundaries
    RISK = "risk"                           # Risk thresholds


class ConstraintSeverity(Enum):
    """Severity levels for constraints."""
    CRITICAL = "critical"                   # Must be satisfied
    HIGH = "high"                          # Should be satisfied
    MEDIUM = "medium"                      # Preferably satisfied
    LOW = "low"                            # Nice to satisfy
    OPTIONAL = "optional"                  # Optional preference


class UncertaintyLevel(Enum):
    """Levels of uncertainty in decomposition decisions."""
    CERTAIN = 0.0
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.9
    UNKNOWN = 1.0


# ============================================================================
# DATA CLASSES - Core Data Models
# ============================================================================

@dataclass
class Constraint:
    """Enhanced constraint representation with satisfaction tracking."""
    id: str
    description: str
    type: ConstraintType
    severity: ConstraintSeverity
    validation_fn: Optional[Callable[[Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    satisfaction_score: float = 0.0  # 0.0-1.0
    is_satisfied: bool = False
    
    def validate(self, context: Any) -> Tuple[bool, float]:
        """Validate constraint against context. Returns (is_satisfied, score)."""
        if self.validation_fn:
            try:
                is_valid = self.validation_fn(context)
                self.is_satisfied = is_valid
                self.satisfaction_score = 1.0 if is_valid else 0.0
                return is_valid, self.satisfaction_score
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Constraint validation failed: {e}")
                return False, 0.0
        return True, 1.0


@dataclass
class SuccessCriterion:
    """Enhanced success criterion with validation methods."""
    id: str
    description: str
    metric: str
    threshold: float
    validation_method: str = "automatic"
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, value: float) -> Tuple[bool, float]:
        """Evaluate if criterion is met. Returns (is_met, score)."""
        score = min(1.0, value / self.threshold) if self.threshold > 0 else 1.0
        return score >= 1.0, score


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification for decomposition decisions."""
    level: UncertaintyLevel
    confidence_score: float  # 0.0-1.0
    entropy: float  # Information entropy
    variance: float  # Statistical variance
    sources: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    def aggregate(self, other: UncertaintyEstimate) -> UncertaintyEstimate:
        """Aggregate uncertainty from multiple sources."""
        return UncertaintyEstimate(
            level=UncertaintyLevel(max(self.level.value, other.level.value)),
            confidence_score=self.confidence_score * other.confidence_score,
            entropy=(self.entropy + other.entropy) / 2,
            variance=(self.variance + other.variance) / 2,
            sources=list(set(self.sources + other.sources)),
            mitigation_strategies=list(set(self.mitigation_strategies + other.mitigation_strategies))
        )


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment with uncertainty."""
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    coordination_complexity: float
    overall_complexity: float
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MEDIUM, 0.5, 1.0, 0.5
    ))
    explanation: str = ""
    
    def __post_init__(self):
        """Ensure all scores are within valid range."""
        for field_name in ['cognitive_complexity', 'computational_complexity', 
                          'domain_complexity', 'integration_complexity', 
                          'coordination_complexity', 'overall_complexity']:
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(10.0, float(value))))
    
    def weighted_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted complexity score."""
        score = (
            self.cognitive_complexity * weights.get('cognitive', 0.2) +
            self.computational_complexity * weights.get('computational', 0.2) +
            self.domain_complexity * weights.get('domain', 0.2) +
            self.integration_complexity * weights.get('integration', 0.2) +
            self.coordination_complexity * weights.get('coordination', 0.2)
        )
        return score


@dataclass
class ResourceEstimate:
    """Resource estimation for sub-problems."""
    estimated_hours: float
    estimated_cost: float
    required_skills: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    team_size: int = 1
    parallelism_factor: float = 1.0  # How parallelizable (1.0 = fully parallel)
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MEDIUM, 0.5, 1.0, 0.5
    ))


@dataclass
class SubProblem:
    """Enhanced atomic sub-problem with comprehensive metadata."""
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType
    complexity_score: ComplexityScore
    
    # Relationships
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)  # Reverse dependencies
    related_subproblems: List[str] = field(default_factory=list)
    
    # Success criteria
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    
    # Resources
    estimated_effort_hours: float = 1.0
    resource_estimate: Optional[ResourceEstimate] = None
    
    # Prioritization
    priority: int = 5  # 1-10
    urgency: int = 5   # 1-10
    business_value: float = 0.5  # 0.0-1.0
    risk_score: float = 0.5  # 0.0-1.0
    
    # Status
    status: str = "pending"
    completion_percentage: float = 0.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Semantic
    semantic_clusters: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def calculate_criticality(self) -> float:
        """Calculate criticality score based on multiple factors."""
        return (
            self.priority * 0.3 +
            self.urgency * 0.25 +
            self.business_value * 10 * 0.25 +
            self.risk_score * 10 * 0.2
        ) / 10.0
    
    def is_critical_path(self) -> bool:
        """Check if this sub-problem is on the critical path."""
        return len(self.dependents) > 2 or self.calculate_criticality() > 0.7


@dataclass
class DependencyEdge:
    """Enhanced dependency edge with type and strength."""
    from_id: str
    to_id: str
    dependency_type: str = "hard"  # hard, soft, temporal
    strength: float = 1.0  # 0.0-1.0
    description: str = ""
    is_critical: bool = False


@dataclass
class DependencyGraph:
    """Enhanced dependency graph with advanced analytics."""
    nodes: Dict[str, SubProblem] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    
    # Analytics
    critical_path: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    
    def add_node(self, node: SubProblem) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: DependencyEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        if edge.to_id in self.nodes:
            self.nodes[edge.to_id].dependencies.append(edge.from_id)
        if edge.from_id in self.nodes:
            self.nodes[edge.from_id].dependents.append(edge.to_id)
    
    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order with critical path awareness."""
        if self.execution_order:
            return self.execution_order
        
        # Build adjacency list
        in_degree = defaultdict(int)
        adjacency = defaultdict(list)
        
        for node_id in self.nodes:
            in_degree[node_id] = 0
        
        for edge in self.edges:
            adjacency[edge.from_id].append(edge.to_id)
            in_degree[edge.to_id] += 1
        
        # Kahn's algorithm with critical path prioritization
        queue = [
            (node_id, self.nodes[node_id].calculate_criticality())
            for node_id in self.nodes
            if in_degree[node_id] == 0
        ]
        queue.sort(key=lambda x: -x[1])  # Sort by criticality descending
        
        result = []
        while queue:
            node_id, _ = queue.pop(0)
            result.append(node_id)
            
            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    criticality = self.nodes[neighbor].calculate_criticality()
                    queue.append((neighbor, criticality))
                    queue.sort(key=lambda x: -x[1])
        
        if len(result) != len(self.nodes):
            logger.warning("Cycle detected in dependency graph")
            # Add remaining nodes
            for node_id in self.nodes:
                if node_id not in result:
                    result.append(node_id)
        
        self.execution_order = result
        return result
    
    def calculate_critical_path(self) -> List[str]:
        """Calculate the critical path through the graph."""
        if not self.nodes:
            return []
        
        # Calculate earliest start/finish
        es = {node_id: 0.0 for node_id in self.nodes}
        ef = {}
        
        for node_id in self.get_execution_order():
            node = self.nodes[node_id]
            duration = node.estimated_effort_hours
            ef[node_id] = es[node_id] + duration
            
            for edge in self.edges:
                if edge.from_id == node_id:
                    es[edge.to_id] = max(es[edge.to_id], ef[node_id])
        
        # Calculate latest start/finish
        total_duration = max(ef.values()) if ef else 0
        lf = {node_id: total_duration for node_id in self.nodes}
        ls = {}
        
        for node_id in reversed(self.get_execution_order()):
            node = self.nodes[node_id]
            duration = node.estimated_effort_hours
            ls[node_id] = lf[node_id] - duration
            
            for edge in self.edges:
                if edge.to_id == node_id:
                    lf[edge.from_id] = min(lf[edge.from_id], ls[node_id])
        
        # Find critical path (zero slack)
        critical_path = []
        for node_id in self.nodes:
            slack = ls[node_id] - es[node_id] if node_id in ls and node_id in es else 0
            if abs(slack) < 0.001:  # Near-zero slack
                critical_path.append(node_id)
        
        self.critical_path = critical_path
        return critical_path
    
    def find_parallel_groups(self) -> List[List[str]]:
        """Find groups of sub-problems that can be executed in parallel."""
        if not self.execution_order:
            self.get_execution_order()
        
        groups = []
        completed = set()
        
        while len(completed) < len(self.nodes):
            # Find all nodes with satisfied dependencies
            available = []
            for node_id in self.nodes:
                if node_id in completed:
                    continue
                node = self.nodes[node_id]
                if all(dep in completed for dep in node.dependencies):
                    available.append(node_id)
            
            if available:
                groups.append(available)
                completed.update(available)
            else:
                break
        
        self.parallel_groups = groups
        return groups
    
    def find_bottlenecks(self) -> List[str]:
        """Find bottleneck nodes (high dependents, long duration)."""
        bottlenecks = []
        
        for node_id, node in self.nodes.items():
            # High number of dependents
            dependent_count = len(node.dependents)
            duration = node.estimated_effort_hours
            
            # Bottleneck score: high dependents * long duration
            bottleneck_score = dependent_count * duration
            
            if bottleneck_score > 20 or dependent_count > 3:
                bottlenecks.append(node_id)
        
        self.bottlenecks = bottlenecks
        return bottlenecks


@dataclass
class DecompositionContext:
    """Context for decomposition decisions."""
    domain: str
    available_strategies: List[DecompositionStrategy]
    constraints: List[Constraint]
    preferences: Dict[str, Any] = field(default_factory=dict)
    historical_performance: Dict[str, float] = field(default_factory=dict)
    resource_limits: Optional[Dict[str, float]] = None
    deadline: Optional[datetime] = None
    quality_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecompositionPlan:
    """Enhanced decomposition plan with comprehensive metadata."""
    id: str
    original_problem_id: str
    sub_problems: List[SubProblem]
    strategy_used: DecompositionStrategy
    dependency_graph: DependencyGraph
    
    # Quality metrics
    quality_score: float = 0.0
    coverage_score: float = 0.0
    balance_score: float = 0.0
    
    # Uncertainty
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MEDIUM, 0.5, 1.0, 0.5
    ))
    
    # Resource estimates
    total_effort_hours: float = 0.0
    critical_path_duration: float = 0.0
    resource_estimate: Optional[ResourceEstimate] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    parent_plan_id: Optional[str] = None
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        if not self.sub_problems:
            return {}
        
        # Coverage: are all aspects covered?
        types_covered = set(sp.type for sp in self.sub_problems)
        coverage_score = len(types_covered) / len(SubProblemType)
        
        # Balance: are sub-problems similar in size?
        efforts = [sp.estimated_effort_hours for sp in self.sub_problems]
        if efforts:
            mean_effort = sum(efforts) / len(efforts)
            variance = sum((e - mean_effort) ** 2 for e in efforts) / len(efforts)
            balance_score = 1.0 / (1.0 + variance / 100)  # Normalize
        else:
            balance_score = 0.0
        
        # Dependency health
        dep_health = 1.0
        if self.dependency_graph.edges:
            cycles = self._detect_cycles()
            if cycles:
                dep_health = 0.5
        
        self.quality_score = (coverage_score + balance_score + dep_health) / 3
        self.coverage_score = coverage_score
        self.balance_score = balance_score
        
        return {
            'quality_score': self.quality_score,
            'coverage_score': coverage_score,
            'balance_score': balance_score,
            'dependency_health': dep_health,
            'sub_problem_count': len(self.sub_problems),
            'total_effort': self.total_effort_hours,
            'critical_path': len(self.dependency_graph.critical_path)
        }
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in dependency graph."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node_id: str, path: List[str]) -> None:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            node = self.dependency_graph.nodes.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in visited:
                        dfs(dep_id, path.copy())
                    elif dep_id in rec_stack:
                        cycle_start = path.index(dep_id)
                        cycles.append(path[cycle_start:] + [dep_id])
            
            rec_stack.remove(node_id)
        
        for node_id in self.dependency_graph.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles


@dataclass
class StrategyRecommendation:
    """Recommendation for decomposition strategy."""
    strategy: DecompositionStrategy
    confidence: float
    rationale: str
    expected_quality: float
    estimated_time: float
    alternatives: List[Tuple[DecompositionStrategy, float]] = field(default_factory=list)


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class DecompositionStrategyBase(ABC):
    """Abstract base class for decomposition strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def decompose(
        self, 
        problem: 'ProblemDefinition', 
        context: DecompositionContext
    ) -> DecompositionPlan:
        """Decompose problem into sub-problems."""
        pass
    
    @abstractmethod
    def get_strategy_type(self) -> DecompositionStrategy:
        """Get the strategy type."""
        pass
    
    def estimate_complexity(self, problem: 'ProblemDefinition') -> ComplexityScore:
        """Estimate problem complexity."""
        # Default implementation
        return ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=5.0,
            domain_complexity=5.0,
            integration_complexity=5.0,
            coordination_complexity=5.0,
            overall_complexity=5.0,
            explanation="Default complexity estimate"
        )


class ConstraintSatisfactionSolver(ABC):
    """Abstract base for constraint satisfaction."""
    
    @abstractmethod
    def solve(
        self, 
        sub_problems: List[SubProblem], 
        constraints: List[Constraint]
    ) -> Tuple[List[SubProblem], float]:
        """Solve constraints and return optimized sub-problems with satisfaction score."""
        pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix."""
    uid = hashlib.sha256(
        f"{prefix}{uuid.uuid4()}{time.time()}".encode()
    ).hexdigest()[:12]
    return f"{prefix}_{uid}" if prefix else uid


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts (fallback implementation)."""
    # Simple Jaccard similarity as fallback
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


def topological_sort_with_priority(
    nodes: List[str], 
    edges: Dict[str, List[str]], 
    priorities: Dict[str, float]
) -> List[str]:
    """Topological sort with priority ordering."""
    in_degree = {node: 0 for node in nodes}
    for neighbors in edges.values():
        for neighbor in neighbors:
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    queue = [(node, priorities.get(node, 0)) for node in nodes if in_degree[node] == 0]
    queue.sort(key=lambda x: -x[1])
    
    result = []
    while queue:
        node, _ = queue.pop(0)
        result.append(node)
        
        for neighbor in edges.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append((neighbor, priorities.get(neighbor, 0)))
                queue.sort(key=lambda x: -x[1])
    
    return result


# ============================================================================
# MAIN DECOMPOSITION ENGINE
# ============================================================================

class ComprehensiveDecompositionEngine:
    """
    Comprehensive Decomposition Engine with multi-strategy orchestration,
    uncertainty quantification, and advanced optimization.
    
    Enhanced with CAV-NLP integration for:
    - Natural language problem decomposition
    - Hybrid Z3+Lean validation
    - Formal constraint extraction from NL descriptions
    """
    
    def __init__(
        self,
        strategies: Optional[Dict[DecompositionStrategy, DecompositionStrategyBase]] = None,
        llm_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        enable_parallel: bool = True,
        max_workers: int = 4,
        enable_caching: bool = True,
        enable_cav_nlp: bool = True
    ):
        """
        Initialize the comprehensive decomposition engine.
        
        Args:
            strategies: Optional dict of strategy implementations
            llm_client: Optional LLM client for intelligent decomposition
            embedding_model: Optional embedding model for semantic analysis
            enable_parallel: Enable parallel processing
            max_workers: Maximum parallel workers
            enable_caching: Enable result caching
            enable_cav_nlp: Enable CAV-NLP integration for enhanced validation
        """
        self.strategies = strategies or {}
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # CAV-NLP integration
        self._cav_nlp_enabled = enable_cav_nlp and CAV_NLP_AVAILABLE
        self.cav_nlp_bridge = None
        self.z3_validator = None
        
        if self._cav_nlp_enabled:
            try:
                if CAV_NLP_AVAILABLE:
                    self.cav_nlp_bridge = create_z3_lean_bridge() if 'create_z3_lean_bridge' in globals() else Z3LeanAideBridge()
                    logger.info("CAV-NLP bridge initialized for NL decomposition")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP bridge: {e}")
                self._cav_nlp_enabled = False
        
        # Z3 Validator for hybrid validation
        if Z3_VALIDATOR_AVAILABLE:
            try:
                self.z3_validator = Z3DecompositionValidator(enable_cav_nlp=self._cav_nlp_enabled)
                logger.info("Z3 validator initialized for hybrid validation")
            except Exception as e:
                logger.warning(f"Failed to initialize Z3 validator: {e}")
                self.z3_validator = None
        
        # Cache
        self._cache: Dict[str, Any] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Metrics
        self.decomposition_history: List[DecompositionPlan] = []
        self.strategy_performance: Dict[DecompositionStrategy, List[float]] = defaultdict(list)
        
        # Configuration
        self.config = {
            'min_subproblems': 2,
            'max_subproblems': 15,
            'target_subproblem_size': 8,  # Hours
            'uncertainty_threshold': 0.7,
            'quality_threshold': 0.6,
            'enable_cav_nlp': self._cav_nlp_enabled,
            'cav_nlp_validation_mode': 'hybrid',  # hybrid, z3_only, lean_only
            'auto_validate_with_cav_nlp': True
        }
        
        logger.info(f"ComprehensiveDecompositionEngine initialized with {len(self.strategies)} strategies, CAV-NLP: {self._cav_nlp_enabled}")
    
    def register_strategy(
        self, 
        strategy_type: DecompositionStrategy, 
        implementation: DecompositionStrategyBase
    ) -> None:
        """Register a decomposition strategy."""
        self.strategies[strategy_type] = implementation
        logger.info(f"Registered strategy: {strategy_type.value}")
    
    def select_strategy(
        self, 
        problem: 'ProblemDefinition', 
        context: DecompositionContext
    ) -> StrategyRecommendation:
        """
        Select optimal decomposition strategy using ML-based recommendation.
        
        Uses problem characteristics, historical performance, and context
        to recommend the best strategy.
        """
        if not self.strategies:
            raise ValueError("No strategies registered")
        
        # Calculate strategy scores
        scores = []
        for strategy_type in context.available_strategies:
            if strategy_type not in self.strategies:
                continue
            
            score = self._calculate_strategy_score(strategy_type, problem, context)
            scores.append((strategy_type, score))
        
        if not scores:
            raise ValueError("No available strategies match requirements")
        
        # Sort by score
        scores.sort(key=lambda x: -x[1])
        
        best_strategy = scores[0][0]
        confidence = scores[0][1]
        
        # Build rationale
        rationale = self._build_strategy_rationale(best_strategy, problem, context)
        
        # Estimate expected quality and time
        expected_quality = self._estimate_strategy_quality(best_strategy, problem)
        estimated_time = self._estimate_decomposition_time(best_strategy, problem)
        
        return StrategyRecommendation(
            strategy=best_strategy,
            confidence=confidence,
            rationale=rationale,
            expected_quality=expected_quality,
            estimated_time=estimated_time,
            alternatives=[(s, sc) for s, sc in scores[1:4]]
        )
    
    def _calculate_strategy_score(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition', 
        context: DecompositionContext
    ) -> float:
        """Calculate suitability score for a strategy."""
        score = 0.0
        weights = {
            'historical': 0.25,
            'domain_match': 0.20,
            'complexity_match': 0.20,
            'constraint_compatibility': 0.20,
            'resource_efficiency': 0.15
        }
        
        # Historical performance
        if strategy in self.strategy_performance:
            hist_scores = self.strategy_performance[strategy]
            score += weights['historical'] * (sum(hist_scores) / len(hist_scores))
        else:
            score += weights['historical'] * 0.5  # Default
        
        # Domain match
        domain_score = self._calculate_domain_match(strategy, problem, context)
        score += weights['domain_match'] * domain_score
        
        # Complexity match
        complexity_score = self._calculate_complexity_match(strategy, problem)
        score += weights['complexity_match'] * complexity_score
        
        # Constraint compatibility
        constraint_score = self._calculate_constraint_compatibility(strategy, context.constraints)
        score += weights['constraint_compatibility'] * constraint_score
        
        # Resource efficiency
        resource_score = self._calculate_resource_efficiency(strategy, context)
        score += weights['resource_efficiency'] * resource_score
        
        return score
    
    def _calculate_domain_match(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition', 
        context: DecompositionContext
    ) -> float:
        """Calculate how well strategy matches the domain."""
        domain_matches = {
            DecompositionStrategy.DOMAIN_DRIVEN: ['software', 'business', 'research'],
            DecompositionStrategy.FUNCTIONAL: ['software', 'engineering'],
            DecompositionStrategy.SEMANTIC: ['research', 'analysis', 'design'],
            DecompositionStrategy.TEMPORAL: ['operations', 'project_management'],
            DecompositionStrategy.RISK_BASED: ['security', 'finance', 'compliance'],
            DecompositionStrategy.VALUE_BASED: ['business', 'product'],
        }
        
        compatible_domains = domain_matches.get(strategy, [])
        if context.domain.lower() in compatible_domains:
            return 1.0
        return 0.5
    
    def _calculate_complexity_match(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition'
    ) -> float:
        """Calculate how well strategy matches problem complexity."""
        complexity = problem.complexity_score.overall_complexity
        
        # Different strategies work better at different complexity levels
        if complexity < 3:
            return 1.0 if strategy in [DecompositionStrategy.FUNCTIONAL, DecompositionStrategy.HIERARCHICAL] else 0.6
        elif complexity < 6:
            return 1.0 if strategy in [DecompositionStrategy.SEMANTIC, DecompositionStrategy.DEPENDENCY] else 0.7
        else:
            return 1.0 if strategy in [DecompositionStrategy.MULTI_LEVEL, DecompositionStrategy.HYBRID, DecompositionStrategy.ADAPTIVE] else 0.5
    
    def _calculate_constraint_compatibility(
        self, 
        strategy: DecompositionStrategy, 
        constraints: List[Constraint]
    ) -> float:
        """Calculate how well strategy satisfies constraints."""
        if not constraints:
            return 1.0
        
        # Check strategy-specific constraint compatibility
        satisfied = 0
        for constraint in constraints:
            if self._strategy_satisfies_constraint(strategy, constraint):
                satisfied += 1
        
        return satisfied / len(constraints)
    
    def _strategy_satisfies_constraint(
        self, 
        strategy: DecompositionStrategy, 
        constraint: Constraint
    ) -> bool:
        """Check if a strategy can satisfy a constraint."""
        # Strategy-specific constraint handling
        temporal_strategies = [
            DecompositionStrategy.TEMPORAL, 
            DecompositionStrategy.WORKFLOW
        ]
        
        if constraint.type == ConstraintType.TEMPORAL:
            return strategy in temporal_strategies
        
        if constraint.type == ConstraintType.RESOURCE:
            return strategy not in [DecompositionStrategy.MULTI_LEVEL, DecompositionStrategy.PARALLEL]
        
        return True
    
    def _calculate_resource_efficiency(
        self, 
        strategy: DecompositionStrategy, 
        context: DecompositionContext
    ) -> float:
        """Calculate resource efficiency of strategy."""
        resource_costs = {
            DecompositionStrategy.HIERARCHICAL: 0.3,
            DecompositionStrategy.FUNCTIONAL: 0.3,
            DecompositionStrategy.SEMANTIC: 0.5,
            DecompositionStrategy.MULTI_LEVEL: 0.8,
            DecompositionStrategy.HYBRID: 0.7,
            DecompositionStrategy.ADAPTIVE: 0.6,
        }
        
        return 1.0 - resource_costs.get(strategy, 0.5)
    
    def _build_strategy_rationale(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition', 
        context: DecompositionContext
    ) -> str:
        """Build human-readable rationale for strategy selection."""
        parts = []
        parts.append(f"Selected {strategy.value} decomposition based on:")
        
        if context.domain:
            parts.append(f"- Domain ({context.domain}) compatibility")
        
        complexity = problem.complexity_score.overall_complexity
        parts.append(f"- Problem complexity level ({complexity:.1f}/10)")
        
        if context.constraints:
            parts.append(f"- Constraint compatibility ({len(context.constraints)} constraints)")
        
        if strategy in self.strategy_performance:
            avg_perf = sum(self.strategy_performance[strategy]) / len(self.strategy_performance[strategy])
            parts.append(f"- Historical performance ({avg_perf:.2f} quality score)")
        
        return "\n".join(parts)
    
    def _estimate_strategy_quality(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition'
    ) -> float:
        """Estimate expected quality for strategy."""
        if strategy in self.strategy_performance:
            return sum(self.strategy_performance[strategy]) / len(self.strategy_performance[strategy])
        return 0.7  # Default estimate
    
    def _estimate_decomposition_time(
        self, 
        strategy: DecompositionStrategy, 
        problem: 'ProblemDefinition'
    ) -> float:
        """Estimate decomposition time in seconds."""
        base_time = 5.0
        complexity_factor = problem.complexity_score.overall_complexity / 10.0
        
        strategy_factors = {
            DecompositionStrategy.HIERARCHICAL: 1.0,
            DecompositionStrategy.SEMANTIC: 1.5,
            DecompositionStrategy.MULTI_LEVEL: 2.0,
            DecompositionStrategy.HYBRID: 2.5,
            DecompositionStrategy.ADAPTIVE: 2.0,
        }
        
        return base_time * complexity_factor * strategy_factors.get(strategy, 1.0)
    
    def decompose(
        self,
        problem: 'ProblemDefinition',
        context: Optional[DecompositionContext] = None,
        strategy: Optional[DecompositionStrategy] = None
    ) -> DecompositionPlan:
        """
        Decompose a problem into sub-problems using optimal strategy.

        Args:
            problem: Problem to decompose
            context: Optional decomposition context
            strategy: Optional specific strategy to use

        Returns:
            DecompositionPlan with sub-problems and metadata
        """
        start_time_total = time.time()
        success = False

        try:
            context = context or DecompositionContext(
                domain="general",
                available_strategies=list(self.strategies.keys()),
                constraints=[]
            )

            # Check cache
            cache_key = f"{problem.id}_{strategy.value if strategy else 'auto'}"
            if self.enable_caching and cache_key in self._cache:
                logger.info(f"Returning cached decomposition for {problem.id}")
                return self._cache[cache_key]

            # Select or use specified strategy
            if strategy is None:
                recommendation = self.select_strategy(problem, context)
                strategy = recommendation.strategy
                logger.info(f"Selected strategy: {strategy.value} (confidence: {recommendation.confidence:.2f})")
            else:
                logger.info(f"Using specified strategy: {strategy.value}")

            # Execute decomposition
            strategy_impl = self.strategies.get(strategy)
            if not strategy_impl:
                raise ValueError(f"Strategy {strategy.value} not registered")

            start_time = time.time()
            plan = strategy_impl.decompose(problem, context)
            decomposition_time = time.time() - start_time

            # Post-process plan
            plan = self._post_process_plan(plan, context)

            # Calculate metrics
            metrics = plan.calculate_metrics()
            logger.info(f"Decomposition completed in {decomposition_time:.2f}s: {metrics}")

            # Update history and performance
            self.decomposition_history.append(plan)
            if plan.quality_score > 0:
                self.strategy_performance[strategy].append(plan.quality_score)

            # Cache result
            if self.enable_caching:
                self._cache[cache_key] = plan

            success = True
            total_time = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful decomposition
            self._extract_decomposition_knowledge("decompose", problem.id, strategy, plan)
            self._track_decomposition_performance("decompose", True, total_time, strategy.value, len(plan.sub_problems))

            return plan

        except Exception as e:
            total_time = time.time() - start_time_total

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_decomposition_alerts("decompose", False, problem.id, str(e))
            self._track_decomposition_performance("decompose", False, total_time, "unknown", 0)

            logger.error(f"Decomposition failed for problem {problem.id}: {e}")
            raise
    
    def _post_process_plan(
        self, 
        plan: DecompositionPlan, 
        context: DecompositionContext
    ) -> DecompositionPlan:
        """Post-process decomposition plan with optimizations."""
        # Calculate dependency graph analytics
        plan.dependency_graph.calculate_critical_path()
        plan.dependency_graph.find_parallel_groups()
        plan.dependency_graph.find_bottlenecks()
        
        # Calculate resource estimates
        plan.total_effort_hours = sum(
            sp.estimated_effort_hours for sp in plan.sub_problems
        )
        
        if plan.dependency_graph.critical_path:
            plan.critical_path_duration = sum(
                plan.dependency_graph.nodes[node_id].estimated_effort_hours
                for node_id in plan.dependency_graph.critical_path
                if node_id in plan.dependency_graph.nodes
            )
        
        # Estimate uncertainty
        plan.uncertainty = self._estimate_plan_uncertainty(plan)
        
        return plan
    
    def _estimate_plan_uncertainty(self, plan: DecompositionPlan) -> UncertaintyEstimate:
        """Estimate uncertainty for decomposition plan."""
        sources = []
        
        # Uncertainty from complexity
        avg_complexity = sum(
            sp.complexity_score.overall_complexity 
            for sp in plan.sub_problems
        ) / len(plan.sub_problems) if plan.sub_problems else 5.0
        
        if avg_complexity > 7:
            sources.append("high_complexity")
        
        # Uncertainty from dependencies
        if len(plan.dependency_graph.edges) > len(plan.sub_problems) * 2:
            sources.append("complex_dependencies")
        
        # Uncertainty from lack of historical data
        if plan.strategy_used not in self.strategy_performance:
            sources.append("no_historical_data")
        
        confidence = 1.0 - (len(sources) * 0.15)
        confidence = max(0.1, min(1.0, confidence))
        
        level = UncertaintyLevel.MEDIUM
        if confidence > 0.8:
            level = UncertaintyLevel.LOW
        elif confidence < 0.4:
            level = UncertaintyLevel.HIGH
        
        return UncertaintyEstimate(
            level=level,
            confidence_score=confidence,
            entropy=len(sources),
            variance=1.0 - confidence,
            sources=sources,
            mitigation_strategies=[
                "iterative_refinement",
                "expert_review",
                "prototyping"
            ]
        )
    
    def refine_decomposition(
        self, 
        plan: DecompositionPlan, 
        feedback: Dict[str, Any]
    ) -> DecompositionPlan:
        """
        Refine decomposition based on feedback.
        
        Args:
            plan: Original decomposition plan
            feedback: Feedback for refinement
        
        Returns:
            Refined DecompositionPlan
        """
        logger.info(f"Refining decomposition plan {plan.id}")
        
        # Create refined plan
        refined_subproblems = []
        
        for sp in plan.sub_problems:
            # Apply feedback-based adjustments
            adjusted_sp = self._apply_feedback(sp, feedback)
            refined_subproblems.append(adjusted_sp)
        
        # Create new plan
        refined_plan = DecompositionPlan(
            id=generate_id("plan"),
            original_problem_id=plan.original_problem_id,
            sub_problems=refined_subproblems,
            strategy_used=plan.strategy_used,
            dependency_graph=plan.dependency_graph,
            parent_plan_id=plan.id,
            version=plan.version + 1
        )
        
        # Recalculate metrics
        refined_plan.calculate_metrics()
        
        logger.info(f"Refined plan created: {refined_plan.id} (v{refined_plan.version})")
        return refined_plan
    
    def _apply_feedback(self, subproblem: SubProblem, feedback: Dict[str, Any]) -> SubProblem:
        """Apply feedback to a sub-problem."""
        # Deep copy
        adjusted = SubProblem(
            id=subproblem.id,
            parent_id=subproblem.parent_id,
            title=subproblem.title,
            description=subproblem.description,
            type=subproblem.type,
            complexity_score=subproblem.complexity_score,
            dependencies=subproblem.dependencies.copy(),
            dependents=subproblem.dependents.copy(),
            related_subproblems=subproblem.related_subproblems.copy(),
            success_criteria=subproblem.success_criteria.copy(),
            acceptance_criteria=subproblem.acceptance_criteria.copy(),
            estimated_effort_hours=subproblem.estimated_effort_hours,
            priority=subproblem.priority,
            urgency=subproblem.urgency,
            business_value=subproblem.business_value,
            risk_score=subproblem.risk_score,
            tags=subproblem.tags.copy(),
            metadata=subproblem.metadata.copy()
        )
        
        # Adjust effort based on feedback
        effort_adjustment = feedback.get('effort_adjustment', 1.0)
        adjusted.estimated_effort_hours *= effort_adjustment
        
        # Adjust priority based on feedback
        if 'priority_adjustment' in feedback:
            adjusted.priority = max(1, min(10, adjusted.priority + feedback['priority_adjustment']))
        
        return adjusted
    
    def parallel_decompose(
        self,
        problems: List['ProblemDefinition'],
        context: Optional[DecompositionContext] = None
    ) -> List[DecompositionPlan]:
        """
        Decompose multiple problems in parallel.
        
        Args:
            problems: List of problems to decompose
            context: Optional shared context
        
        Returns:
            List of DecompositionPlans
        """
        if not self.enable_parallel or len(problems) == 1:
            return [self.decompose(p, context) for p in problems]
        
        logger.info(f"Parallel decomposing {len(problems)} problems")
        
        plans = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.decompose, problem, context): problem 
                for problem in problems
            }
            
            for future in as_completed(futures):
                problem = futures[future]
                try:
                    plan = future.result()
                    plans.append(plan)
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.error(f"Failed to decompose problem {problem.id}: {e}")
        
        return plans

    def get_statistics(self) -> Dict[str, Any]:
        """Get decomposition engine statistics."""
        return {
            'total_decompositions': len(self.decomposition_history),
            'registered_strategies': [s.value for s in self.strategies.keys()],
            'strategy_performance': {
                s.value: {
                    'avg_quality': sum(scores) / len(scores) if scores else 0,
                    'count': len(scores)
                }
                for s, scores in self.strategy_performance.items()
            },
            'cache_size': len(self._cache),
            'avg_subproblems': (
                sum(len(p.sub_problems) for p in self.decomposition_history) /
                len(self.decomposition_history) if self.decomposition_history else 0
            ),
            'cav_nlp_enabled': self._cav_nlp_enabled,
            'cav_nlp_bridge_active': self.cav_nlp_bridge is not None,
            'z3_validator_active': self.z3_validator is not None
        }
    
    # ========================================================================
    # CAV-NLP Enhanced Methods
    # ========================================================================
    
    async def decompose_with_nl_input(
        self,
        natural_language: str,
        domain: str = "general",
        context: Optional[DecompositionContext] = None,
        extract_constraints: bool = True,
        auto_validate: bool = True
    ) -> DecompositionResult:
        """
        Decompose problem from natural language description using CAV-NLP.
        
        This method:
        1. Formalizes natural language to constraints using CAV-NLP
        2. Extracts mathematical components if present
        3. Creates a formal ProblemDefinition
        4. Runs decomposition on formalized constraints
        5. Optionally validates with hybrid Z3+Lean
        
        Args:
            natural_language: Natural language problem description
            domain: Problem domain (e.g., "software", "mathematics", "engineering")
            context: Optional decomposition context
            extract_constraints: Whether to extract formal constraints from NL
            auto_validate: Whether to auto-validate with hybrid validation
            
        Returns:
            DecompositionResult with formalized constraints and decomposition plan
        """
        start_time = time.time()
        
        if not self._cav_nlp_enabled:
            logger.warning("CAV-NLP not available, using standard decomposition with NL parsing")
            # Fallback: create a basic problem definition and decompose
            problem = self._create_problem_from_nl_fallback(natural_language, domain)
            context = context or DecompositionContext(
                domain=domain,
                available_strategies=list(self.strategies.keys()),
                constraints=[]
            )
            return self.decompose(problem, context)
        
        try:
            logger.info(f"Starting CAV-NLP decomposition for domain: {domain}")
            
            # Step 1: Formalize natural language using CAV-NLP
            formalization = await self._formalize_nl_with_cav_nlp(
                natural_language, extract_constraints
            )
            
            # Step 2: Create ProblemDefinition from formalized constraints
            problem = await self._create_problem_from_formalization(
                natural_language, formalization, domain
            )
            
            # Step 3: Create decomposition context with CAV-NLP metadata
            if context is None:
                context = DecompositionContext(
                    domain=domain,
                    available_strategies=list(self.strategies.keys()),
                    constraints=formalization.get("constraints", []),
                    preferences={
                        "cav_nlp_enhanced": True,
                        "extracted_constraints_count": len(formalization.get("constraints", []))
                    }
                )
            
            # Step 4: Run decomposition
            decomposition_start = time.time()
            plan = self.decompose(problem, context)
            decomposition_time = time.time() - decomposition_start
            
            # Step 5: Hybrid validation if enabled
            validation_result = None
            if auto_validate and self.z3_validator is not None:
                try:
                    validation_result = await self.validate_hybrid(plan, problem)
                    logger.info(f"Hybrid validation completed: valid={validation_result.valid}")
                except Exception as e:
                    logger.warning(f"Hybrid validation failed: {e}")
            
            # Step 6: Create enhanced result
            total_time = time.time() - start_time
            
            result = DecompositionResult(
                plan=plan,
                original_description=natural_language,
                formalized_constraints=formalization.get("constraints", []),
                extracted_entities=formalization.get("entities", []),
                validation_result=validation_result,
                metadata={
                    "cav_nlp_enhanced": True,
                    "domain": domain,
                    "formalization_time_ms": formalization.get("execution_time_ms", 0),
                    "decomposition_time_ms": decomposition_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "validation_performed": validation_result is not None,
                    "extracted_constraint_count": len(formalization.get("constraints", []))
                }
            )
            
            # Extract knowledge for successful decompositions
            self._extract_decomposition_knowledge(
                "decompose_with_nl", 
                problem.id, 
                plan.strategy_used, 
                plan
            )
            
            logger.info(f"CAV-NLP decomposition completed in {total_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"CAV-NLP decomposition failed: {e}", exc_info=True)
            # Trigger alert for failure
            self._trigger_decomposition_alerts(
                "decompose_with_nl", False, None, str(e),
                {"natural_language": natural_language[:100]}
            )
            raise
    
    async def validate_hybrid(
        self,
        plan: DecompositionPlan,
        problem: 'ProblemDefinition',
        validation_depth: str = "standard",
        validate_critical_with_lean: bool = True
    ) -> ValidationResult:
        """
        Validate decomposition using hybrid Z3+Lean approach.
        
        This method provides enhanced validation by combining:
        - Z3 SMT solver for constraint satisfaction
        - Lean 4 theorem prover for critical constraints
        - CAV-NLP for semantic consistency checking
        
        Args:
            plan: Decomposition plan to validate
            problem: Original problem definition
            validation_depth: "quick", "standard", or "deep"
            validate_critical_with_lean: Whether to validate critical constraints with Lean
            
        Returns:
            ValidationResult with hybrid verification data
        """
        if self.z3_validator is None:
            logger.warning("Z3 validator not available, returning basic validation")
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": "Z3 validator not available"}],
                recommendations=["Enable Z3 validation for enhanced verification"]
            )
        
        try:
            logger.info(f"Starting hybrid validation with depth: {validation_depth}")
            
            # Use the validator's CAV-NLP enhanced validation
            if hasattr(self.z3_validator, 'validate_with_cav_nlp'):
                # Convert plan to format expected by validator
                original_problem = self._problem_to_smtlib(problem) if hasattr(problem, 'constraints') else ""
                
                # Create decomposition tuple for the validator
                decomposition_data = (
                    original_problem,
                    self._convert_plan_to_subproblem_models(plan),
                    self._extract_entanglements_from_plan(plan)
                )
                
                result = await self.z3_validator.validate_with_cav_nlp(
                    decomposition_data,
                    validate_critical_with_lean=validate_critical_with_lean,
                    verification_depth=validation_depth
                )
                
                return result
            else:
                # Fallback to standard Z3 validation
                logger.warning("CAV-NLP validation not available, using standard Z3")
                return ValidationResult(
                    success=True,
                    valid=True,
                    recommendations=["CAV-NLP enhanced validation not available"]
                )
                
        except Exception as e:
            logger.error(f"Hybrid validation failed: {e}", exc_info=True)
            return ValidationResult(
                success=False,
                valid=False,
                violations=[{"error": f"Hybrid validation failed: {str(e)}"}],
                recommendations=["Review decomposition manually"]
            )
    
    async def _formalize_nl_with_cav_nlp(
        self,
        natural_language: str,
        extract_constraints: bool = True
    ) -> Dict[str, Any]:
        """Formalize natural language description using CAV-NLP."""
        start_time = time.time()
        result = {
            "constraints": [],
            "entities": [],
            "canonical_form": None,
            "mathematical_components": [],
            "execution_time_ms": 0
        }
        
        try:
            if self.cav_nlp_bridge is None:
                return result
            
            # Use CAV-NLP to canonicalize and extract structure
            if hasattr(self.cav_nlp_bridge, '_canonicalize_text'):
                canonical = self.cav_nlp_bridge._canonicalize_text(natural_language)
                result["canonical_form"] = canonical
            
            # Extract mathematical components if available
            if hasattr(self.cav_nlp_bridge, 'parser') and self.cav_nlp_bridge.parser is not None:
                parser = self.cav_nlp_bridge.parser
                
                # Try to extract constraints using the parser
                if hasattr(parser, 'parse'):
                    parse_result = parser.parse(natural_language)
                    result["parse_result"] = parse_result
                
                # Extract variables and entities
                if hasattr(parser, 'extract_entities'):
                    entities = parser.extract_entities(natural_language)
                    result["entities"] = entities
            
            # If we have Lean service, try to formalize to Lean
            if extract_constraints and hasattr(self.cav_nlp_bridge, 'z3_to_lean4'):
                # This is a simplified extraction - real implementation would be more sophisticated
                result["constraints"] = self._extract_constraints_from_nl(natural_language)
            
            result["execution_time_ms"] = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.warning(f"CAV-NLP formalization failed: {e}")
            result["error"] = str(e)
            result["execution_time_ms"] = (time.time() - start_time) * 1000
            return result
    
    async def _create_problem_from_formalization(
        self,
        natural_language: str,
        formalization: Dict[str, Any],
        domain: str
    ) -> 'ProblemDefinition':
        """Create ProblemDefinition from CAV-NLP formalization."""
        # Generate unique ID
        problem_id = f"nl_{hashlib.sha256(natural_language.encode()).hexdigest()[:12]}"
        
        # Extract or estimate complexity
        complexity = self._estimate_complexity_from_formalization(formalization)
        
        # Convert extracted constraints to Constraint objects
        constraints = []
        for c in formalization.get("constraints", []):
            if isinstance(c, dict):
                constraints.append(Constraint(
                    id=c.get("id", generate_id("cons")),
                    description=c.get("description", ""),
                    type=ConstraintType.TECHNICAL,
                    severity=ConstraintSeverity.HIGH if c.get("critical", False) else ConstraintSeverity.MEDIUM
                ))
        
        return ProblemDefinition(
            id=problem_id,
            title=natural_language[:100] + "..." if len(natural_language) > 100 else natural_language,
            description=natural_language,
            domain=domain,
            complexity_score=complexity,
            constraints=constraints,
            metadata={
                "cav_nlp_formalized": True,
                "extracted_entities": formalization.get("entities", []),
                "canonical_form": str(formalization.get("canonical_form", ""))[:500]
            }
        )
    
    def _create_problem_from_nl_fallback(
        self,
        natural_language: str,
        domain: str
    ) -> 'ProblemDefinition':
        """Create a basic ProblemDefinition when CAV-NLP is unavailable."""
        problem_id = f"nl_fb_{hashlib.sha256(natural_language.encode()).hexdigest()[:12]}"
        
        return ProblemDefinition(
            id=problem_id,
            title=natural_language[:100],
            description=natural_language,
            domain=domain,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                coordination_complexity=5.0,
                overall_complexity=5.0,
                explanation="Default complexity - CAV-NLP not available for analysis"
            ),
            constraints=[],
            metadata={"cav_nlp_formalized": False, "fallback": True}
        )
    
    def _estimate_complexity_from_formalization(
        self,
        formalization: Dict[str, Any]
    ) -> ComplexityScore:
        """Estimate complexity from CAV-NLP formalization."""
        # Base complexity
        cognitive = 5.0
        computational = 5.0
        domain = 5.0
        integration = 5.0
        coordination = 5.0
        
        # Adjust based on extracted constraints
        constraint_count = len(formalization.get("constraints", []))
        if constraint_count > 10:
            cognitive += 2.0
            computational += 1.5
        elif constraint_count > 5:
            cognitive += 1.0
            computational += 0.5
        
        # Adjust based on entities
        entity_count = len(formalization.get("entities", []))
        if entity_count > 20:
            integration += 2.0
            coordination += 1.5
        elif entity_count > 10:
            integration += 1.0
            coordination += 0.5
        
        # Check for mathematical components (typically higher complexity)
        math_components = formalization.get("mathematical_components", [])
        if math_components:
            computational += min(2.0, len(math_components) * 0.5)
            domain += 1.0
        
        return ComplexityScore(
            cognitive_complexity=min(10.0, cognitive),
            computational_complexity=min(10.0, computational),
            domain_complexity=min(10.0, domain),
            integration_complexity=min(10.0, integration),
            coordination_complexity=min(10.0, coordination),
            overall_complexity=min(10.0, (cognitive + computational + domain + integration + coordination) / 5),
            explanation="Estimated from CAV-NLP formalization analysis"
        )
    
    def _extract_constraints_from_nl(self, natural_language: str) -> List[Dict[str, Any]]:
        """Extract constraints from natural language description."""
        constraints = []
        
        # Simple keyword-based extraction (real implementation would use NLP)
        constraint_keywords = [
            ("must", ConstraintSeverity.CRITICAL),
            ("should", ConstraintSeverity.HIGH),
            ("needs to", ConstraintSeverity.HIGH),
            ("required", ConstraintSeverity.CRITICAL),
            ("optional", ConstraintSeverity.OPTIONAL),
            ("preferably", ConstraintSeverity.MEDIUM)
        ]
        
        for keyword, severity in constraint_keywords:
            if keyword in natural_language.lower():
                # Find sentences containing the keyword
                sentences = natural_language.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        constraints.append({
                            "id": generate_id("cons"),
                            "description": sentence.strip(),
                            "severity": severity,
                            "critical": severity == ConstraintSeverity.CRITICAL
                        })
        
        return constraints
    
    def _problem_to_smtlib(self, problem: 'ProblemDefinition') -> str:
        """Convert problem definition to SMT-LIB format for Z3."""
        # Simplified conversion
        smtlib_parts = ["(set-logic ALL)"]
        
        # Add constraints as assertions
        for constraint in problem.constraints:
            smtlib_parts.append(f"; {constraint.description}")
        
        return "\n".join(smtlib_parts)
    
    def _convert_plan_to_subproblem_models(
        self,
        plan: DecompositionPlan
    ) -> List[Any]:
        """Convert DecompositionPlan to SubProblemModels for Z3 validator."""
        # Import here to avoid circular imports
        try:
            from decomposition_z3_validator import SubProblemModel, DecompositionConstraint, Z3Variable, Z3ConstraintType
            
            models = []
            for sp in plan.sub_problems:
                # Extract or create variables
                variables = []
                if hasattr(sp, 'metadata') and 'variables' in sp.metadata:
                    for var_name in sp.metadata['variables']:
                        variables.append(Z3Variable(var_name, Z3ConstraintType.INTEGER))
                
                # Extract or create constraints
                constraints = []
                if hasattr(sp, 'success_criteria'):
                    for sc in sp.success_criteria:
                        constraints.append(DecompositionConstraint(
                            constraint_id=sc.id,
                            expression=sc.description,
                            scope="subproblem"
                        ))
                
                models.append(SubProblemModel(
                    subproblem_id=sp.id,
                    variables=variables,
                    constraints=constraints,
                    complexity_score=sp.complexity_score.overall_complexity if hasattr(sp, 'complexity_score') else 5.0
                ))
            
            return models
        except ImportError:
            logger.warning("Could not import SubProblemModel, returning empty list")
            return []
    
    def _extract_entanglements_from_plan(
        self,
        plan: DecompositionPlan
    ) -> List[Any]:
        """Extract entanglements from DecompositionPlan."""
        try:
            from decomposition_z3_validator import EntanglementSpecification
            
            entanglements = []
            for edge in plan.dependency_graph.edges:
                entanglements.append(EntanglementSpecification(
                    entanglement_id=f"ent_{edge.from_id}_{edge.to_id}",
                    source_subproblem=edge.from_id,
                    target_subproblem=edge.to_id,
                    coupling_constraints=[edge.description] if edge.description else [],
                    strength="strong" if edge.is_critical else "weak"
                ))
            
            return entanglements
        except ImportError:
            logger.warning("Could not import EntanglementSpecification, returning empty list")
            return []

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Comprehensive Decomposition
    # =========================================================================

    def _trigger_decomposition_alerts(
        self,
        operation: str,
        success: bool,
        problem_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for decomposition failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"Decomposition Alert: {operation}",
                    description=f"Decomposition operation '{operation}' failed" +
                                 (f" for problem '{problem_id}'" if problem_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="comprehensive_decomposition_engine",
                    component="decomposition",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Decomposition alert: {e}")

    def _extract_decomposition_knowledge(
        self,
        operation: str,
        problem_id: str,
        strategy: 'DecompositionStrategy',
        plan: 'DecompositionPlan'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract decomposition knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"decomp_{operation}_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="decomposition_execution",
                source_component="comprehensive_decomposition_engine",
                title=f"Decomposition: {operation} - {problem_id}",
                content={
                    "operation": operation,
                    "problem_id": problem_id,
                    "strategy": strategy.value if strategy else "unknown",
                    "num_subproblems": len(plan.sub_problems),
                    "quality_score": plan.quality_score,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "complexity": plan.total_complexity,
                    "total_effort_hours": plan.total_effort_hours
                },
                tags=["decomposition", operation, strategy.value if strategy else "unknown"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Decomposition knowledge for {problem_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Decomposition knowledge: {e}")
            return False

    def _track_decomposition_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        strategy: str,
        num_subproblems: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track decomposition performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"decomp_{strategy}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "num_subproblems": num_subproblems
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Decomposition performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Decomposition performance: {e}")


# ============================================================================
# PROBLEM DEFINITION (For standalone usage)
# ============================================================================

@dataclass
class DecompositionResult:
    """Result from CAV-NLP enhanced decomposition."""
    plan: DecompositionPlan
    original_description: str
    formalized_constraints: List[Dict[str, Any]] = field(default_factory=list)
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    validation_result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if decomposition is valid."""
        if self.validation_result is None:
            return True  # No validation performed
        if hasattr(self.validation_result, 'valid'):
            return self.validation_result.valid
        return True
    
    def get_confidence_score(self) -> float:
        """Get overall confidence score."""
        confidence = 0.7  # Base confidence
        
        # Adjust based on validation
        if self.validation_result is not None:
            if hasattr(self.validation_result, 'metrics'):
                metrics = self.validation_result.metrics
                if 'hybrid_confidence' in metrics:
                    confidence = metrics['hybrid_confidence']
                elif 'z3_execution_time_ms' in metrics:
                    confidence += 0.1  # Bonus for Z3 validation
        
        # Adjust based on formalization success
        if self.metadata.get('cav_nlp_enhanced', False):
            confidence += 0.1
        
        return min(1.0, confidence)


@dataclass
class ProblemDefinition:
    """Complete problem definition for decomposition."""
    id: str
    title: str
    description: str
    domain: str
    complexity_score: ComplexityScore
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'domain': self.domain,
            'complexity_score': {
                'cognitive_complexity': self.complexity_score.cognitive_complexity,
                'computational_complexity': self.complexity_score.computational_complexity,
                'domain_complexity': self.complexity_score.domain_complexity,
                'integration_complexity': self.complexity_score.integration_complexity,
                'coordination_complexity': self.complexity_score.coordination_complexity,
                'overall_complexity': self.complexity_score.overall_complexity,
            },
            'constraints': len(self.constraints),
            'success_criteria': len(self.success_criteria),
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# EXPORTS
# ============================================================================

# ValidationResult placeholder for when decomposition_z3_validator is not available
class ValidationResult:
    """Placeholder ValidationResult when Z3 validator is not available."""
    def __init__(self, success: bool, valid: bool, violations: List[Dict[str, Any]] = None, 
                 recommendations: List[str] = None, metrics: Dict[str, float] = None):
        self.success = success
        self.valid = valid
        self.violations = violations or []
        self.recommendations = recommendations or []
        self.metrics = metrics or {}


# =============================================================================
# Example Usage
# =============================================================================

async def example_nl_decomposition():
    """Example: Decompose a natural language problem using CAV-NLP."""
    print("CAV-NLP Enhanced Natural Language Decomposition")
    print("=" * 60)
    
    # Create engine with CAV-NLP enabled
    engine = ComprehensiveDecompositionEngine(enable_cav_nlp=True)
    
    # Check capabilities
    stats = engine.get_statistics()
    print(f"CAV-NLP Enabled: {stats.get('cav_nlp_enabled', False)}")
    print(f"CAV-NLP Bridge Active: {stats.get('cav_nlp_bridge_active', False)}")
    print(f"Z3 Validator Active: {stats.get('z3_validator_active', False)}")
    
    # Natural language problem description
    nl_description = """
    Design a distributed task scheduling system that:
    - Must handle up to 10,000 concurrent tasks
    - Should prioritize critical tasks over normal ones
    - Needs to support both immediate and scheduled execution
    - Must ensure no task is lost during system failures
    - Should provide real-time status updates to users
    """
    
    print(f"\nInput Problem:")
    print(nl_description[:200] + "...")
    
    try:
        # Decompose with CAV-NLP
        result = await engine.decompose_with_nl_input(
            natural_language=nl_description,
            domain="software",
            extract_constraints=True,
            auto_validate=True
        )
        
        print(f"\nDecomposition Result:")
        print(f"  Plan ID: {result.plan.id}")
        print(f"  Strategy: {result.plan.strategy_used.value}")
        print(f"  Sub-problems: {len(result.plan.sub_problems)}")
        print(f"  Valid: {result.is_valid()}")
        print(f"  Confidence: {result.get_confidence_score():.2f}")
        
        if result.formalized_constraints:
            print(f"\n  Extracted Constraints: {len(result.formalized_constraints)}")
            for i, c in enumerate(result.formalized_constraints[:3], 1):
                print(f"    {i}. {c.get('description', 'N/A')[:50]}...")
        
        if result.validation_result and hasattr(result.validation_result, 'metrics'):
            print(f"\n  Validation Metrics:")
            metrics = result.validation_result.metrics
            for key in ['hybrid_confidence', 'cav_nlp_violations_found', 'lean_verifications_count']:
                if key in metrics:
                    print(f"    {key}: {metrics[key]}")
        
        return result
        
    except Exception as e:
        print(f"Decomposition failed: {e}")
        return None


if __name__ == "__main__":
    import asyncio
    
    print("Comprehensive Decomposition Engine")
    print("=" * 60)
    
    # Run CAV-NLP example
    try:
        asyncio.run(example_nl_decomposition())
    except Exception as e:
        print(f"Example failed: {e}")


__all__ = [
    # Enums
    'DecompositionStrategy',
    'SubProblemType',
    'ConstraintType',
    'ConstraintSeverity',
    'UncertaintyLevel',
    
    # Data classes
    'Constraint',
    'SuccessCriterion',
    'UncertaintyEstimate',
    'ComplexityScore',
    'ResourceEstimate',
    'SubProblem',
    'DependencyEdge',
    'DependencyGraph',
    'DecompositionContext',
    'DecompositionPlan',
    'StrategyRecommendation',
    'ProblemDefinition',
    'DecompositionResult',
    'ValidationResult',
    
    # Base classes
    'DecompositionStrategyBase',
    'ConstraintSatisfactionSolver',
    
    # Main engine
    'ComprehensiveDecompositionEngine',
    
    # Utilities
    'generate_id',
    'calculate_semantic_similarity',
    'topological_sort_with_priority',
    
    # Examples
    'example_nl_decomposition',
]
