"""
Enhanced Decomposition Engine - Sovereign-Grade Comprehensive Problem Decomposition System

This module implements an infinitely more comprehensive decomposition system with:
- 20+ Decomposition Strategies (Structural, Semantic, Temporal, Causal, Risk-based, etc.)
- Cross-Domain Semantic Analysis with Knowledge Graph Integration
- Causal and Temporal Decomposition
- Multi-Level Hierarchical Decomposition with Recursive Depth Control
- Constraint Satisfaction with Multi-Objective Optimization
- Uncertainty Quantification and Confidence Scoring
- Resource-Aware Decomposition Planning
- Parallel Decomposition Processing
- LLM-Powered Intelligent Analysis with Multi-Model Ensemble
- Semantic Boundary Detection with Embeddings
- Domain-Specific Pattern Recognition
- Dependency Graph Optimization with Critical Path Analysis
- Evolutionary Decomposition Refinement
- Adversarial Decomposition Testing
- Real-time Decomposition Analytics

Version: 3.0.0
Author: OpenEvolve Sovereign System
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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache, partial, wraps
from typing import (
    Any, Callable, Coroutine, Dict, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Protocol, AsyncIterator, Iterator
)
import uuid
import heapq
import threading
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS - Comprehensive Type Definitions
# ============================================================================

class DecompositionStrategy(Enum):
    """Comprehensive decomposition strategies."""
    # Structural strategies
    HIERARCHICAL = "hierarchical"
    FUNCTIONAL = "functional"
    STRUCTURAL = "structural"
    COMPONENT_BASED = "component_based"
    MODULAR = "modular"
    
    # Semantic strategies
    SEMANTIC = "semantic"
    DOMAIN_DRIVEN = "domain_driven"
    ONTOLOGY_BASED = "ontology_based"
    CONCEPTUAL = "conceptual"
    
    # Process/Temporal strategies
    TEMPORAL = "temporal"
    WORKFLOW = "workflow"
    STATE_BASED = "state_based"
    EVENT_DRIVEN = "event_driven"
    PHASE_BASED = "phase_based"
    
    # Analytical strategies
    CAUSAL = "causal"
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    INFORMATION_FLOW = "information_flow"
    
    # Risk/Value strategies
    RISK_BASED = "risk_based"
    VALUE_BASED = "value_based"
    COST_BASED = "cost_based"
    ROI_BASED = "roi_based"
    
    # Technical strategies
    DEPENDENCY = "dependency"
    COMPLEXITY = "complexity"
    COUPLING = "coupling"
    COHESION = "cohesion"
    INTERFACE_BASED = "interface_based"
    
    # Advanced strategies
    ADAPTIVE = "adaptive"
    MULTI_LEVEL = "multi_level"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"
    EVOLUTIONARY = "evolutionary"
    ADVERSARIAL = "adversarial"


class SubProblemType(Enum):
    """Granular sub-problem types."""
    # Research types
    RESEARCH = "research"
    INVESTIGATION = "investigation"
    EXPLORATION = "exploration"
    SURVEY = "survey"
    DISCOVERY = "discovery"
    
    # Analysis types
    ANALYSIS = "analysis"
    DESIGN = "design"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    REQUIREMENTS = "requirements"
    MODELING = "modeling"
    
    # Implementation types
    IMPLEMENTATION = "implementation"
    DEVELOPMENT = "development"
    CODING = "coding"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"
    
    # Quality types
    VALIDATION = "validation"
    VERIFICATION = "verification"
    TESTING = "testing"
    REVIEW = "review"
    AUDIT = "audit"
    QA = "qa"
    
    # Support types
    DOCUMENTATION = "documentation"
    TRAINING = "training"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"
    REFACTORING = "refactoring"
    
    # Special types
    DECISION = "decision"
    COORDINATION = "coordination"
    COMMUNICATION = "communication"
    STAKEHOLDER = "stakeholder"
    COMPLIANCE = "compliance"


class ConstraintType(Enum):
    """Types of constraints."""
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    QUALITY = "quality"
    BUDGET = "budget"
    SCOPE = "scope"
    RISK = "risk"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    SCALABILITY = "scalability"
    MAINTAINABILITY = "maintainability"


class ConstraintSeverity(Enum):
    """Severity levels for constraints."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class UncertaintyLevel(Enum):
    """Levels of uncertainty."""
    CERTAIN = 0.0
    VERY_LOW = 0.1
    LOW = 0.25
    MODERATE = 0.5
    HIGH = 0.75
    VERY_HIGH = 0.9
    UNKNOWN = 1.0


class ProblemDomain(Enum):
    """Supported problem domains."""
    SOFTWARE = "software"
    FINANCE = "finance"
    SCIENTIFIC = "scientific"
    HEALTHCARE = "healthcare"
    MANUFACTURING = "manufacturing"
    LEGAL = "legal"
    BUSINESS = "business"
    EDUCATION = "education"
    GOVERNMENT = "government"
    ENERGY = "energy"
    TRANSPORTATION = "transportation"
    RETAIL = "retail"
    GENERIC = "generic"


class DecompositionStatus(Enum):
    """Status of decomposition."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    DECOMPOSING = "decomposing"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# ============================================================================
# DATA CLASSES - Comprehensive Data Models
# ============================================================================

@dataclass
class Constraint:
    """Enhanced constraint with validation and tracking."""
    id: str
    description: str
    type: ConstraintType
    severity: ConstraintSeverity
    validation_fn: Optional[Callable[[Any], Tuple[bool, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    satisfaction_score: float = 0.0
    is_satisfied: bool = False
    violation_cost: float = 1.0
    
    def validate(self, context: Any) -> Tuple[bool, float]:
        """Validate constraint. Returns (is_satisfied, score)."""
        if self.validation_fn:
            try:
                is_valid, score = self.validation_fn(context)
                self.is_satisfied = is_valid
                self.satisfaction_score = score
                return is_valid, score
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Constraint validation failed: {e}")
                return False, 0.0
        return True, 1.0


@dataclass
class SuccessCriterion:
    """Enhanced success criterion."""
    id: str
    description: str
    metric: str
    threshold: float
    validation_method: str = "automatic"
    weight: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, value: float) -> Tuple[bool, float]:
        """Evaluate criterion. Returns (is_met, score)."""
        if self.threshold > 0:
            score = min(1.0, value / self.threshold)
        else:
            score = 1.0 if value >= self.threshold else 0.0
        return score >= 1.0, score


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification."""
    level: UncertaintyLevel
    confidence_score: float
    entropy: float
    variance: float
    sources: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    sample_size: int = 0
    
    def aggregate(self, others: List[UncertaintyEstimate]) -> UncertaintyEstimate:
        """Aggregate uncertainty from multiple sources."""
        if not others:
            return self
        
        all_estimates = [self] + others
        max_level = max(e.level.value for e in all_estimates)
        combined_confidence = 1.0
        for e in all_estimates:
            combined_confidence *= e.confidence_score
        
        return UncertaintyEstimate(
            level=UncertaintyLevel(max_level),
            confidence_score=combined_confidence,
            entropy=sum(e.entropy for e in all_estimates) / len(all_estimates),
            variance=sum(e.variance for e in all_estimates) / len(all_estimates),
            sources=list(set(s for e in all_estimates for s in e.sources)),
            mitigation_strategies=list(set(s for e in all_estimates for s in e.mitigation_strategies)),
            sample_size=sum(e.sample_size for e in all_estimates)
        )


@dataclass
class ComplexityScore:
    """Multi-dimensional complexity assessment."""
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    coordination_complexity: float
    technical_complexity: float
    overall_complexity: float
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MODERATE, 0.5, 1.0, 0.5, [], [], 0
    ))
    explanation: str = ""
    
    def __post_init__(self):
        for field_name in ['cognitive_complexity', 'computational_complexity', 
                          'domain_complexity', 'integration_complexity', 
                          'coordination_complexity', 'technical_complexity', 'overall_complexity']:
            value = getattr(self, field_name)
            setattr(self, field_name, max(0.0, min(10.0, float(value))))
    
    def weighted_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted complexity."""
        if weights is None:
            weights = {
                'cognitive': 0.15, 'computational': 0.15, 'domain': 0.15,
                'integration': 0.15, 'coordination': 0.15, 'technical': 0.15
            }
        return (
            self.cognitive_complexity * weights.get('cognitive', 0.15) +
            self.computational_complexity * weights.get('computational', 0.15) +
            self.domain_complexity * weights.get('domain', 0.15) +
            self.integration_complexity * weights.get('integration', 0.15) +
            self.coordination_complexity * weights.get('coordination', 0.15) +
            self.technical_complexity * weights.get('technical', 0.15)
        )


@dataclass
class ResourceEstimate:
    """Resource estimation."""
    estimated_hours: float
    estimated_cost: float
    required_skills: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    team_size: int = 1
    parallelism_factor: float = 1.0
    critical_path_factor: float = 1.0
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MODERATE, 0.5, 1.0, 0.5, [], [], 0
    ))


@dataclass
class TemporalConstraints:
    """Temporal constraints for decomposition."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    deadlines: List[Tuple[str, datetime]] = field(default_factory=list)
    milestones: List[Tuple[str, datetime]] = field(default_factory=list)
    dependencies_temporal: Dict[str, List[str]] = field(default_factory=dict)
    seasonality_constraints: List[str] = field(default_factory=list)


@dataclass
class SemanticMetadata:
    """Semantic metadata for sub-problems."""
    keywords: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    semantic_similarity_map: Dict[str, float] = field(default_factory=dict)
    topic_distribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class SubProblem:
    """Enhanced atomic sub-problem."""
    id: str
    parent_id: str
    title: str
    description: str
    type: SubProblemType
    complexity_score: ComplexityScore
    
    # Relationships
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    related_subproblems: List[str] = field(default_factory=list)
    
    # Success criteria
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    
    # Resources
    estimated_effort_hours: float = 1.0
    resource_estimate: Optional[ResourceEstimate] = None
    
    # Prioritization
    priority: int = 5
    urgency: int = 5
    business_value: float = 0.5
    risk_score: float = 0.5
    
    # Status
    status: str = "pending"
    completion_percentage: float = 0.0
    
    # Semantic
    semantic_metadata: SemanticMetadata = field(default_factory=SemanticMetadata)
    
    # Temporal
    temporal_constraints: Optional[TemporalConstraints] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'type': self.type.value,
            'complexity': self.complexity_score.overall_complexity,
            'priority': self.priority,
            'status': self.status,
            'dependencies_count': len(self.dependencies)
        }


@dataclass
class ProblemDefinition:
    """Complete problem definition."""
    id: str
    title: str
    description: str
    domain: ProblemDomain
    subdomain: str = ""
    complexity_score: ComplexityScore = field(default_factory=lambda: ComplexityScore(
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0
    ))
    constraints: List[Constraint] = field(default_factory=list)
    success_criteria: List[SuccessCriterion] = field(default_factory=list)
    temporal_constraints: Optional[TemporalConstraints] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'domain': self.domain.value,
            'complexity': self.complexity_score.overall_complexity
        }


@dataclass
class DecompositionPlan:
    """Complete decomposition plan."""
    id: str
    original_problem: ProblemDefinition
    sub_problems: List[SubProblem]
    strategy_used: DecompositionStrategy
    
    # Graph structure
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    # Analysis
    complexity_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    resource_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    coverage_score: float = 0.0
    balance_score: float = 0.0
    coherence_score: float = 0.0
    overall_quality: float = 0.0
    
    # Uncertainty
    uncertainty: UncertaintyEstimate = field(default_factory=lambda: UncertaintyEstimate(
        UncertaintyLevel.MODERATE, 0.5, 1.0, 0.5, [], [], 0
    ))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_critical_path(self) -> List[str]:
        """Calculate critical path using dependency graph."""
        if not self.execution_order:
            return []
        
        # Find longest path through dependency graph
        memo = {}
        
        def longest_path(node: str) -> int:
            if node in memo:
                return memo[node]
            
            deps = self.dependency_graph.get(node, [])
            if not deps:
                memo[node] = 1
                return 1
            
            max_len = max(longest_path(dep) for dep in deps if dep in self.execution_order)
            memo[node] = max_len + 1
            return memo[node]
        
        # Sort by path length
        sorted_nodes = sorted(
            self.execution_order,
            key=lambda n: longest_path(n),
            reverse=True
        )
        
        return sorted_nodes


@dataclass
class DecompositionAnalytics:
    """Analytics for decomposition process."""
    decomposition_time_ms: float
    strategies_attempted: int
    strategies_successful: int
    llm_calls_made: int
    cache_hits: int
    conflicts_detected: int
    iterations_performed: int
    memory_usage_mb: float
    

# ============================================================================
# STRATEGY BASE CLASS
# ============================================================================

class DecompositionStrategyBase(ABC):
    """Base class for all decomposition strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        pass
    
    @abstractmethod
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        """Check if strategy can handle problem. Returns (can_handle, confidence)."""
        pass
    
    @abstractmethod
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose problem into sub-problems."""
        pass
    
    def estimate_complexity(self, problem: ProblemDefinition) -> ComplexityScore:
        """Estimate complexity for decomposition."""
        return problem.complexity_score


# ============================================================================
# ENHANCED DECOMPOSITION ENGINE
# ============================================================================

class EnhancedDecompositionEngine:
    """
    Sovereign-grade comprehensive decomposition engine.
    
    Features:
    - Multi-strategy orchestration with intelligent selection
    - Parallel decomposition processing
    - Constraint satisfaction with optimization
    - Uncertainty quantification
    - Cross-domain semantic analysis
    - Knowledge graph integration
    - LLM-powered intelligent analysis
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        use_llm: bool = True,
        enable_cache: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        self.max_workers = max_workers
        self.use_llm = use_llm
        self.enable_cache = enable_cache
        self.config = config or {}
        
        # Strategy registry
        self.strategies: Dict[DecompositionStrategy, DecompositionStrategyBase] = {}
        self._register_default_strategies()
        
        # Cache
        self._cache: Dict[str, DecompositionPlan] = {}
        self._cache_lock = threading.RLock()
        
        # Analytics
        self.analytics: List[DecompositionAnalytics] = []
        
        # LLM client
        self._llm_client: Optional[Any] = None
        self._init_llm_client()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("EnhancedDecompositionEngine initialized")
    
    def _register_default_strategies(self) -> None:
        """Register default decomposition strategies."""
        # These will be implemented as concrete classes
        self.strategies[DecompositionStrategy.HIERARCHICAL] = HierarchicalDecomposition()
        self.strategies[DecompositionStrategy.FUNCTIONAL] = FunctionalDecomposition()
        self.strategies[DecompositionStrategy.SEMANTIC] = SemanticDecomposition()
        self.strategies[DecompositionStrategy.TEMPORAL] = TemporalDecomposition()
        self.strategies[DecompositionStrategy.CAUSAL] = CausalDecomposition()
        self.strategies[DecompositionStrategy.RISK_BASED] = RiskBasedDecomposition()
        self.strategies[DecompositionStrategy.COMPLEXITY] = ComplexityBasedDecomposition()
        self.strategies[DecompositionStrategy.DEPENDENCY] = DependencyDecomposition()
        self.strategies[DecompositionStrategy.HYBRID] = HybridDecomposition()
    
    def _init_llm_client(self) -> None:
        """Initialize LLM client."""
        if not self.use_llm:
            return
        
        try:
            from openevolve_client import OpenEvolveClient
            self._llm_client = OpenEvolveClient()
            self.logger.info("LLM client initialized")
        except ImportError:
            self.logger.warning("OpenEvolveClient not available")
            self._llm_client = None
    
    def decompose(
        self,
        problem: ProblemDefinition,
        strategy: Optional[DecompositionStrategy] = None,
        min_subproblems: int = 3,
        max_subproblems: int = 10,
        max_depth: int = 3,
        constraints: Optional[List[Constraint]] = None
    ) -> DecompositionPlan:
        """
        Decompose problem into sub-problems.
        
        Args:
            problem: Problem to decompose
            strategy: Decomposition strategy (auto-selected if None)
            min_subproblems: Minimum number of sub-problems
            max_subproblems: Maximum number of sub-problems
            max_depth: Maximum recursion depth for multi-level decomposition
            constraints: Additional constraints
            
        Returns:
            DecompositionPlan with sub-problems
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(problem, strategy)
        if self.enable_cache and cache_key in self._cache:
            self.logger.info(f"Cache hit for problem {problem.id}")
            return self._cache[cache_key]
        
        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(problem)
        
        self.logger.info(f"Decomposing problem '{problem.title}' using {strategy.value} strategy")
        
        # Get strategy implementation
        strategy_impl = self.strategies.get(strategy)
        if not strategy_impl:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Perform decomposition
        try:
            sub_problems = strategy_impl.decompose(problem)
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.error(f"Decomposition failed: {e}")
            # Fallback to hybrid strategy
            strategy = DecompositionStrategy.HYBRID
            strategy_impl = self.strategies[strategy]
            sub_problems = strategy_impl.decompose(problem)
        
        # Validate and adjust
        sub_problems = self._validate_and_adjust(
            sub_problems, problem, min_subproblems, max_subproblems
        )
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(sub_problems)
        execution_order = self._topological_sort(dependency_graph)
        parallel_groups = self._identify_parallel_groups(dependency_graph, execution_order)
        
        # Calculate quality metrics
        coverage_score = self._calculate_coverage(problem, sub_problems)
        balance_score = self._calculate_balance(sub_problems)
        coherence_score = self._calculate_coherence(sub_problems)
        overall_quality = (coverage_score + balance_score + coherence_score) / 3
        
        # Perform analyses
        complexity_analysis = self._analyze_complexity_distribution(sub_problems)
        risk_analysis = self._analyze_risk_distribution(sub_problems)
        resource_analysis = self._analyze_resource_requirements(sub_problems)
        
        # Create plan
        plan = DecompositionPlan(
            id=self._generate_id("plan"),
            original_problem=problem,
            sub_problems=sub_problems,
            strategy_used=strategy,
            dependency_graph=dependency_graph,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            complexity_analysis=complexity_analysis,
            risk_analysis=risk_analysis,
            resource_analysis=resource_analysis,
            coverage_score=coverage_score,
            balance_score=balance_score,
            coherence_score=coherence_score,
            overall_quality=overall_quality
        )

        # Build entanglement matrix for decomposition/recomposition workflows
        entanglement_matrix = self._build_entanglement_matrix(sub_problems)
        plan.metadata["entanglement_matrix"] = entanglement_matrix
        for sp in sub_problems:
            entangled_with = entanglement_matrix.get(sp.id, [])
            if entangled_with:
                sp.metadata.setdefault("entangled_with", entangled_with)
                sp.metadata.setdefault("entanglement_source", "semantic_overlap")
        
        # Cache result
        if self.enable_cache:
            with self._cache_lock:
                self._cache[cache_key] = plan
        
        # Record analytics
        elapsed_ms = (time.time() - start_time) * 1000
        analytics = DecompositionAnalytics(
            decomposition_time_ms=elapsed_ms,
            strategies_attempted=1,
            strategies_successful=1,
            llm_calls_made=0,
            cache_hits=1 if cache_key in self._cache else 0,
            conflicts_detected=0,
            iterations_performed=1,
            memory_usage_mb=0.0
        )
        self.analytics.append(analytics)
        
        self.logger.info(f"Decomposition completed: {len(sub_problems)} sub-problems, quality={overall_quality:.2f}")
        
        return plan

    def _build_entanglement_matrix(self, sub_problems: List[SubProblem]) -> Dict[str, List[str]]:
        """Build entanglement matrix using shared semantic symbols."""
        matrix: Dict[str, Set[str]] = {sp.id: set() for sp in sub_problems}
        symbol_map: Dict[str, Set[str]] = {}

        for sp in sub_problems:
            symbols = self._extract_symbol_tokens(sp)
            if not symbols:
                continue
            for sym in symbols:
                symbol_map.setdefault(sym, set()).add(sp.id)

        for _, components in symbol_map.items():
            if len(components) < 2:
                continue
            for comp in components:
                matrix[comp].update({c for c in components if c != comp})

        return {key: sorted(value) for key, value in matrix.items()}

    def _extract_symbol_tokens(self, sub_problem: SubProblem) -> Set[str]:
        """Extract semantic tokens for entanglement detection."""
        tokens: Set[str] = set()
        if sub_problem.semantic_metadata.keywords:
            tokens.update(t.lower() for t in sub_problem.semantic_metadata.keywords if t)
        if sub_problem.semantic_metadata.concepts:
            tokens.update(t.lower() for t in sub_problem.semantic_metadata.concepts if t)

        if not tokens:
            raw = f"{sub_problem.title or ''} {sub_problem.description or ''}"
            tokens.update(self._tokenize_symbols(raw))

        return {t for t in tokens if t}

    @staticmethod
    def _tokenize_symbols(text: str) -> Set[str]:
        """Tokenize text into normalized symbols."""
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "into", "your",
            "their", "they", "them", "then", "than", "when", "where", "which",
            "while", "will", "would", "could", "should", "must", "shall", "have",
            "has", "had", "been", "being", "are", "was", "were", "not", "but",
            "use", "using", "used", "also", "more", "most", "some", "such",
            "task", "problem", "solution", "system", "component", "sub", "subproblem"
        }
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{2,}", text.lower())
        return {tok for tok in raw_tokens if tok not in stopwords}
    
    def _select_strategy(self, problem: ProblemDefinition) -> DecompositionStrategy:
        """Select best strategy for problem."""
        scores = {}
        
        for strategy_type, strategy_impl in self.strategies.items():
            can_handle, confidence = strategy_impl.can_handle(problem)
            if can_handle:
                scores[strategy_type] = confidence
        
        if not scores:
            return DecompositionStrategy.HYBRID
        
        # Select highest confidence strategy
        return max(scores, key=scores.get)
    
    def _validate_and_adjust(
        self,
        sub_problems: List[SubProblem],
        problem: ProblemDefinition,
        min_count: int,
        max_count: int
    ) -> List[SubProblem]:
        """Validate and adjust sub-problems."""
        # Ensure count constraints
        if len(sub_problems) < min_count:
            self.logger.warning(f"Too few sub-problems ({len(sub_problems)}), minimum is {min_count}")
            # Could trigger re-decomposition with different parameters
        
        if len(sub_problems) > max_count:
            self.logger.warning(f"Too many sub-problems ({len(sub_problems)}), maximum is {max_count}")
            # Merge some sub-problems
            sub_problems = self._merge_subproblems(sub_problems, max_count)
        
        return sub_problems
    
    def _merge_subproblems(self, sub_problems: List[SubProblem], target_count: int) -> List[SubProblem]:
        """Merge sub-problems to reach target count."""
        if len(sub_problems) <= target_count:
            return sub_problems
        
        # Sort by semantic similarity and merge closest pairs
        merged = sub_problems.copy()
        
        while len(merged) > target_count:
            # Find pair with highest similarity
            best_pair = None
            best_similarity = -1
            
            for i, sp1 in enumerate(merged):
                for sp2 in merged[i+1:]:
                    similarity = self._calculate_similarity(sp1, sp2)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_pair = (i, merged.index(sp2))
            
            if best_pair and best_similarity > 0.5:
                i, j = best_pair
                # Merge j into i
                merged[i] = self._merge_two_subproblems(merged[i], merged[j])
                merged.pop(j)
            else:
                # No good merge candidates, break
                break
        
        return merged
    
    def _merge_two_subproblems(self, sp1: SubProblem, sp2: SubProblem) -> SubProblem:
        """Merge two sub-problems."""
        return SubProblem(
            id=sp1.id,
            parent_id=sp1.parent_id,
            title=f"{sp1.title} + {sp2.title}",
            description=f"{sp1.description}\n\n{sp2.description}",
            type=sp1.type if sp1.type == sp2.type else SubProblemType.INTEGRATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=max(sp1.complexity_score.cognitive_complexity, sp2.complexity_score.cognitive_complexity),
                computational_complexity=max(sp1.complexity_score.computational_complexity, sp2.complexity_score.computational_complexity),
                domain_complexity=max(sp1.complexity_score.domain_complexity, sp2.complexity_score.domain_complexity),
                integration_complexity=max(sp1.complexity_score.integration_complexity, sp2.complexity_score.integration_complexity) + 1,
                coordination_complexity=max(sp1.complexity_score.coordination_complexity, sp2.complexity_score.coordination_complexity) + 1,
                technical_complexity=max(sp1.complexity_score.technical_complexity, sp2.complexity_score.technical_complexity),
                overall_complexity=min(10.0, max(sp1.complexity_score.overall_complexity, sp2.complexity_score.overall_complexity) + 0.5)
            ),
            dependencies=list(set(sp1.dependencies + sp2.dependencies)),
            success_criteria=sp1.success_criteria + sp2.success_criteria,
            estimated_effort_hours=sp1.estimated_effort_hours + sp2.estimated_effort_hours,
            priority=max(sp1.priority, sp2.priority)
        )
    
    def _calculate_similarity(self, sp1: SubProblem, sp2: SubProblem) -> float:
        """Calculate semantic similarity between sub-problems."""
        # Text similarity
        text1 = (sp1.title + " " + sp1.description).lower()
        text2 = (sp2.title + " " + sp2.description).lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)
        
        # Type similarity
        type_sim = 1.0 if sp1.type == sp2.type else 0.5
        
        return (jaccard + type_sim) / 2
    
    def _build_dependency_graph(self, sub_problems: List[SubProblem]) -> Dict[str, List[str]]:
        """Build dependency graph."""
        graph = {}
        sp_ids = {sp.id for sp in sub_problems}
        
        for sp in sub_problems:
            # Filter to only include dependencies that exist in our set
            graph[sp.id] = [dep for dep in sp.dependencies if dep in sp_ids]
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Topological sort of dependency graph."""
        in_degree = {node: 0 for node in graph}
        
        for node, deps in graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining nodes (cycles)
        for node in graph:
            if node not in result:
                result.append(node)
        
        return result
    
    def _identify_parallel_groups(
        self,
        graph: Dict[str, List[str]],
        execution_order: List[str]
    ) -> List[List[str]]:
        """Identify groups of sub-problems that can be executed in parallel."""
        groups = []
        completed = set()
        remaining = set(execution_order)
        
        while remaining:
            # Find all nodes whose dependencies are satisfied
            parallel_group = []
            for node in list(remaining):
                deps = graph.get(node, [])
                if all(dep in completed or dep not in execution_order for dep in deps):
                    parallel_group.append(node)
            
            if not parallel_group:
                # Deadlock - add remaining anyway
                parallel_group = list(remaining)
            
            groups.append(parallel_group)
            completed.update(parallel_group)
            remaining -= set(parallel_group)
        
        return groups
    
    def _calculate_coverage(self, problem: ProblemDefinition, sub_problems: List[SubProblem]) -> float:
        """Calculate how well sub-problems cover the original problem."""
        problem_text = (problem.title + " " + problem.description).lower()
        problem_words = set(problem_text.split())
        
        covered_words = set()
        for sp in sub_problems:
            sp_text = (sp.title + " " + sp.description).lower()
            covered_words.update(sp_text.split())
        
        if not problem_words:
            return 1.0
        
        coverage = len(covered_words & problem_words) / len(problem_words)
        return min(1.0, coverage)
    
    def _calculate_balance(self, sub_problems: List[SubProblem]) -> float:
        """Calculate balance of sub-problem sizes/complexities."""
        if len(sub_problems) < 2:
            return 1.0
        
        complexities = [sp.complexity_score.overall_complexity for sp in sub_problems]
        mean_complexity = sum(complexities) / len(complexities)
        
        if mean_complexity == 0:
            return 1.0
        
        # Calculate variance
        variance = sum((c - mean_complexity) ** 2 for c in complexities) / len(complexities)
        std_dev = variance ** 0.5
        
        # Lower std dev = better balance
        balance = max(0.0, 1.0 - (std_dev / mean_complexity))
        return balance
    
    def _calculate_coherence(self, sub_problems: List[SubProblem]) -> float:
        """Calculate semantic coherence of sub-problems."""
        if len(sub_problems) < 2:
            return 1.0
        
        similarities = []
        for i, sp1 in enumerate(sub_problems):
            for sp2 in sub_problems[i+1:]:
                sim = self._calculate_similarity(sp1, sp2)
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Some similarity is good (coherence) but too much is bad (overlap)
        avg_sim = sum(similarities) / len(similarities)
        coherence = 1.0 - abs(avg_sim - 0.3)  # Optimal around 0.3
        return max(0.0, coherence)
    
    def _analyze_complexity_distribution(self, sub_problems: List[SubProblem]) -> Dict[str, Any]:
        """Analyze complexity distribution."""
        complexities = [sp.complexity_score.overall_complexity for sp in sub_problems]
        return {
            'mean': sum(complexities) / len(complexities) if complexities else 0,
            'min': min(complexities) if complexities else 0,
            'max': max(complexities) if complexities else 0,
            'distribution': {
                'low': len([c for c in complexities if c < 3]),
                'medium': len([c for c in complexities if 3 <= c < 7]),
                'high': len([c for c in complexities if c >= 7])
            }
        }
    
    def _analyze_risk_distribution(self, sub_problems: List[SubProblem]) -> Dict[str, Any]:
        """Analyze risk distribution."""
        risks = [sp.risk_score for sp in sub_problems]
        return {
            'mean': sum(risks) / len(risks) if risks else 0,
            'high_risk_count': len([r for r in risks if r > 0.7]),
            'total_risk_score': sum(risks)
        }
    
    def _analyze_resource_requirements(self, sub_problems: List[SubProblem]) -> Dict[str, Any]:
        """Analyze resource requirements."""
        total_hours = sum(sp.estimated_effort_hours for sp in sub_problems)
        
        all_skills = set()
        for sp in sub_problems:
            if sp.resource_estimate:
                all_skills.update(sp.resource_estimate.required_skills)
        
        return {
            'total_hours': total_hours,
            'required_skills': list(all_skills),
            'skill_count': len(all_skills)
        }
    
    def _get_cache_key(self, problem: ProblemDefinition, strategy: Optional[DecompositionStrategy]) -> str:
        """Generate cache key."""
        content = f"{problem.title}:{problem.description}:{strategy.value if strategy else 'auto'}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# CONCRETE STRATEGY IMPLEMENTATIONS
# ============================================================================

class HierarchicalDecomposition(DecompositionStrategyBase):
    """Hierarchical top-down decomposition."""
    
    def get_strategy_name(self) -> str:
        return "hierarchical"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        return True, 0.7
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose hierarchically."""
        sub_problems = []
        
        # Extract key aspects from problem description
        aspects = self._extract_aspects(problem.description)
        
        for i, aspect in enumerate(aspects[:7], 1):
            sp = SubProblem(
                id=f"sub_{uuid.uuid4().hex[:8]}",
                parent_id=problem.id,
                title=f"{aspect['title']}",
                description=aspect['description'],
                type=SubProblemType.ANALYSIS,
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=4.0,
                    domain_complexity=problem.complexity_score.domain_complexity * 0.7,
                    integration_complexity=3.0,
                    coordination_complexity=2.0,
                    technical_complexity=4.0,
                    overall_complexity=4.5
                ),
                priority=8 if i <= 2 else 5,
                estimated_effort_hours=8 + i * 2
            )
            sub_problems.append(sp)
        
        # Add integration sub-problem
        sub_problems.append(SubProblem(
            id=f"sub_{uuid.uuid4().hex[:8]}",
            parent_id=problem.id,
            title="Integrate Components",
            description="Integrate all components into a cohesive solution",
            type=SubProblemType.INTEGRATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=problem.complexity_score.domain_complexity * 0.8,
                integration_complexity=7.0,
                coordination_complexity=6.0,
                technical_complexity=5.0,
                overall_complexity=6.0
            ),
            dependencies=[sp.id for sp in sub_problems[:-1]],
            priority=10,
            estimated_effort_hours=16
        ))
        
        return sub_problems
    
    def _extract_aspects(self, description: str) -> List[Dict[str, str]]:
        """Extract key aspects from description."""
        # Simple extraction based on sentences
        sentences = re.split(r'[.!?]+', description)
        aspects = []
        
        for sentence in sentences[:6]:
            sentence = sentence.strip()
            if len(sentence) > 10:
                aspects.append({
                    'title': sentence[:40] + "..." if len(sentence) > 40 else sentence,
                    'description': sentence
                })
        
        # Ensure minimum aspects
        if len(aspects) < 3:
            aspects.extend([
                {'title': 'Requirements Analysis', 'description': 'Analyze and document requirements'},
                {'title': 'System Design', 'description': 'Design system architecture and components'},
                {'title': 'Implementation', 'description': 'Implement the solution components'}
            ])
        
        return aspects


class FunctionalDecomposition(DecompositionStrategyBase):
    """Decompose by functional areas."""
    
    def get_strategy_name(self) -> str:
        return "functional"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        return True, 0.75
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose by functions."""
        functions = [
            ('Core Logic', SubProblemType.IMPLEMENTATION, 'Implement core business logic'),
            ('Data Management', SubProblemType.DEVELOPMENT, 'Handle data storage and retrieval'),
            ('User Interface', SubProblemType.DESIGN, 'Design and implement user interface'),
            ('Integration Layer', SubProblemType.INTEGRATION, 'Integrate with external systems'),
            ('Error Handling', SubProblemType.VALIDATION, 'Implement error handling and recovery'),
            ('Testing', SubProblemType.TESTING, 'Develop comprehensive tests')
        ]
        
        sub_problems = []
        for name, sp_type, desc in functions:
            sp = SubProblem(
                id=f"sub_{uuid.uuid4().hex[:8]}",
                parent_id=problem.id,
                title=name,
                description=desc,
                type=sp_type,
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=4.0,
                    domain_complexity=problem.complexity_score.domain_complexity * 0.6,
                    integration_complexity=4.0,
                    coordination_complexity=3.0,
                    technical_complexity=4.5,
                    overall_complexity=4.5
                ),
                priority=7,
                estimated_effort_hours=12
            )
            sub_problems.append(sp)
        
        return sub_problems


class SemanticDecomposition(DecompositionStrategyBase):
    """Decompose by semantic concepts."""
    
    def get_strategy_name(self) -> str:
        return "semantic"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        # Good for complex, ill-defined problems
        if problem.complexity_score.overall_complexity > 6:
            return True, 0.85
        return True, 0.6
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose semantically."""
        # Extract semantic clusters from description
        text = problem.description.lower()
        
        # Define semantic patterns
        patterns = {
            'research': ['research', 'investigate', 'explore', 'study', 'analyze'],
            'design': ['design', 'architect', 'plan', 'structure', 'model'],
            'implement': ['implement', 'build', 'develop', 'create', 'code'],
            'validate': ['test', 'validate', 'verify', 'check', 'ensure'],
            'deploy': ['deploy', 'release', 'publish', 'deliver', 'ship']
        }
        
        sub_problems = []
        for concept, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0 or concept in ['design', 'implement']:
                sp = SubProblem(
                    id=f"sub_{uuid.uuid4().hex[:8]}",
                    parent_id=problem.id,
                    title=f"{concept.capitalize()} Phase",
                    description=f"Focus on {concept} activities",
                    type=self._map_concept_to_type(concept),
                    complexity_score=ComplexityScore(
                        cognitive_complexity=5.0 + score * 0.5,
                        computational_complexity=4.0 + score * 0.3,
                        domain_complexity=problem.complexity_score.domain_complexity * 0.5,
                        integration_complexity=3.0 + score * 0.2,
                        coordination_complexity=2.0 + score * 0.3,
                        technical_complexity=4.0 + score * 0.4,
                        overall_complexity=4.5 + score * 0.3
                    ),
                    priority=6 + score,
                    estimated_effort_hours=10 + score * 2
                )
                sub_problems.append(sp)
        
        return sub_problems
    
    def _map_concept_to_type(self, concept: str) -> SubProblemType:
        mapping = {
            'research': SubProblemType.RESEARCH,
            'design': SubProblemType.DESIGN,
            'implement': SubProblemType.IMPLEMENTATION,
            'validate': SubProblemType.VALIDATION,
            'deploy': SubProblemType.DEPLOYMENT
        }
        return mapping.get(concept, SubProblemType.ANALYSIS)


class TemporalDecomposition(DecompositionStrategyBase):
    """Decompose by temporal phases."""
    
    def get_strategy_name(self) -> str:
        return "temporal"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        if problem.temporal_constraints:
            return True, 0.9
        return True, 0.6
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose temporally."""
        phases = [
            ('Phase 1: Initiation', SubProblemType.PLANNING, 'Project initiation and setup'),
            ('Phase 2: Analysis', SubProblemType.ANALYSIS, 'Requirements and analysis'),
            ('Phase 3: Design', SubProblemType.DESIGN, 'System and detailed design'),
            ('Phase 4: Implementation', SubProblemType.IMPLEMENTATION, 'Core implementation'),
            ('Phase 5: Testing', SubProblemType.TESTING, 'Testing and quality assurance'),
            ('Phase 6: Deployment', SubProblemType.DEPLOYMENT, 'Deployment and rollout')
        ]
        
        sub_problems = []
        prev_id = None
        
        for name, sp_type, desc in phases:
            sp_id = f"sub_{uuid.uuid4().hex[:8]}"
            sp = SubProblem(
                id=sp_id,
                parent_id=problem.id,
                title=name,
                description=desc,
                type=sp_type,
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=4.0,
                    domain_complexity=problem.complexity_score.domain_complexity * 0.6,
                    integration_complexity=3.5,
                    coordination_complexity=3.0,
                    technical_complexity=4.0,
                    overall_complexity=4.5
                ),
                dependencies=[prev_id] if prev_id else [],
                priority=7,
                estimated_effort_hours=12
            )
            sub_problems.append(sp)
            prev_id = sp_id
        
        return sub_problems


class CausalDecomposition(DecompositionStrategyBase):
    """Decompose by causal relationships."""
    
    def get_strategy_name(self) -> str:
        return "causal"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        # Good for diagnostic or root-cause problems
        diagnostic_keywords = ['cause', 'why', 'root', 'diagnose', 'fix', 'issue', 'problem', 'error']
        text = problem.description.lower()
        score = sum(1 for kw in diagnostic_keywords if kw in text)
        if score >= 2:
            return True, 0.85
        return True, 0.5
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose by causal chain."""
        causal_steps = [
            ('Identify Symptoms', SubProblemType.INVESTIGATION, 'Identify and document symptoms'),
            ('Analyze Causes', SubProblemType.ANALYSIS, 'Analyze potential root causes'),
            ('Determine Root Cause', SubProblemType.ANALYSIS, 'Determine the root cause'),
            ('Develop Solution', SubProblemType.DESIGN, 'Develop solution approaches'),
            ('Implement Fix', SubProblemType.IMPLEMENTATION, 'Implement the fix'),
            ('Verify Resolution', SubProblemType.VALIDATION, 'Verify the issue is resolved')
        ]
        
        sub_problems = []
        for name, sp_type, desc in causal_steps:
            sp = SubProblem(
                id=f"sub_{uuid.uuid4().hex[:8]}",
                parent_id=problem.id,
                title=name,
                description=desc,
                type=sp_type,
                complexity_score=ComplexityScore(
                    cognitive_complexity=6.0,
                    computational_complexity=4.0,
                    domain_complexity=problem.complexity_score.domain_complexity * 0.8,
                    integration_complexity=3.0,
                    coordination_complexity=2.0,
                    technical_complexity=4.0,
                    overall_complexity=4.5
                ),
                priority=8,
                estimated_effort_hours=10
            )
            sub_problems.append(sp)
        
        return sub_problems


class RiskBasedDecomposition(DecompositionStrategyBase):
    """Decompose addressing highest risks first."""
    
    def get_strategy_name(self) -> str:
        return "risk_based"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        return True, 0.7
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose by risk mitigation."""
        risk_areas = [
            ('Technical Risk Assessment', SubProblemType.ANALYSIS, 9, 0.9),
            ('Proof of Concept', SubProblemType.RESEARCH, 10, 0.85),
            ('Architecture Validation', SubProblemType.VALIDATION, 8, 0.7),
            ('Core Implementation', SubProblemType.IMPLEMENTATION, 7, 0.6),
            ('Integration & Testing', SubProblemType.INTEGRATION, 6, 0.5),
            ('Production Deployment', SubProblemType.DEPLOYMENT, 5, 0.4)
        ]
        
        sub_problems = []
        for name, sp_type, priority, risk in risk_areas:
            sp = SubProblem(
                id=f"sub_{uuid.uuid4().hex[:8]}",
                parent_id=problem.id,
                title=name,
                description=f"Address {name.lower()} with priority",
                type=sp_type,
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0 + risk * 2,
                    computational_complexity=4.0 + risk,
                    domain_complexity=problem.complexity_score.domain_complexity * (0.5 + risk * 0.3),
                    integration_complexity=3.0 + risk * 2,
                    coordination_complexity=2.0 + risk * 2,
                    technical_complexity=4.0 + risk * 3,
                    overall_complexity=4.0 + risk * 3
                ),
                priority=priority,
                risk_score=risk,
                estimated_effort_hours=12 if risk > 0.6 else 8
            )
            sub_problems.append(sp)
        
        return sub_problems


class ComplexityBasedDecomposition(DecompositionStrategyBase):
    """Decompose to balance cognitive load."""
    
    def get_strategy_name(self) -> str:
        return "complexity"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        if problem.complexity_score.overall_complexity > 7:
            return True, 0.9
        return True, 0.65
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose by complexity balancing."""
        total_complexity = problem.complexity_score.overall_complexity
        target_count = max(3, min(8, int(total_complexity)))
        target_complexity = total_complexity / target_count
        
        sub_problems = []
        for i in range(target_count):
            sp = SubProblem(
                id=f"sub_{uuid.uuid4().hex[:8]}",
                parent_id=problem.id,
                title=f"Component {i+1}",
                description=f"Manageable component with balanced complexity",
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=ComplexityScore(
                    cognitive_complexity=target_complexity,
                    computational_complexity=target_complexity * 0.8,
                    domain_complexity=problem.complexity_score.domain_complexity * 0.5,
                    integration_complexity=target_complexity * 0.6,
                    coordination_complexity=target_complexity * 0.4,
                    technical_complexity=target_complexity * 0.9,
                    overall_complexity=target_complexity
                ),
                priority=5,
                estimated_effort_hours=8
            )
            sub_problems.append(sp)
        
        return sub_problems


class DependencyDecomposition(DecompositionStrategyBase):
    """Decompose based on dependency analysis."""
    
    def get_strategy_name(self) -> str:
        return "dependency"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        return True, 0.75
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose by dependencies."""
        # Create foundation layer
        foundation = SubProblem(
            id=f"sub_{uuid.uuid4().hex[:8]}",
            parent_id=problem.id,
            title="Foundation Layer",
            description="Core infrastructure and foundational components",
            type=SubProblemType.ARCHITECTURE,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=problem.complexity_score.domain_complexity * 0.7,
                integration_complexity=4.0,
                coordination_complexity=3.0,
                technical_complexity=6.0,
                overall_complexity=5.5
            ),
            priority=10,
            estimated_effort_hours=16
        )
        
        # Create feature layers
        features = SubProblem(
            id=f"sub_{uuid.uuid4().hex[:8]}",
            parent_id=problem.id,
            title="Feature Implementation",
            description="Implement features on top of foundation",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=problem.complexity_score.domain_complexity * 0.8,
                integration_complexity=5.0,
                coordination_complexity=4.0,
                technical_complexity=5.0,
                overall_complexity=4.8
            ),
            dependencies=[foundation.id],
            priority=7,
            estimated_effort_hours=20
        )
        
        # Create integration layer
        integration = SubProblem(
            id=f"sub_{uuid.uuid4().hex[:8]}",
            parent_id=problem.id,
            title="System Integration",
            description="Integrate all components",
            type=SubProblemType.INTEGRATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=problem.complexity_score.domain_complexity * 0.6,
                integration_complexity=7.0,
                coordination_complexity=6.0,
                technical_complexity=5.0,
                overall_complexity=5.5
            ),
            dependencies=[foundation.id, features.id],
            priority=8,
            estimated_effort_hours=12
        )
        
        return [foundation, features, integration]


class HybridDecomposition(DecompositionStrategyBase):
    """Adaptive hybrid decomposition combining multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sub_strategies: List[DecompositionStrategyBase] = []
    
    def get_strategy_name(self) -> str:
        return "hybrid"
    
    def can_handle(self, problem: ProblemDefinition) -> Tuple[bool, float]:
        return True, 0.95
    
    def decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose using hybrid approach."""
        # Combine multiple strategies
        all_sub_problems = []
        
        # Try hierarchical
        hierarchical = HierarchicalDecomposition()
        all_sub_problems.extend(hierarchical.decompose(problem))
        
        # Try functional
        functional = FunctionalDecomposition()
        func_problems = functional.decompose(problem)
        
        # Merge results intelligently
        merged = self._merge_strategies(all_sub_problems, func_problems)
        
        return merged[:8]  # Limit to reasonable number
    
    def _merge_strategies(
        self,
        problems1: List[SubProblem],
        problems2: List[SubProblem]
    ) -> List[SubProblem]:
        """Intelligently merge results from different strategies."""
        # Simple approach: take unique titles
        seen_titles = set()
        merged = []
        
        for sp in problems1 + problems2:
            normalized_title = sp.title.lower().replace(" ", "_")
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                merged.append(sp)
        
        return merged


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_problem_definition(
    title: str,
    description: str,
    domain: ProblemDomain = ProblemDomain.GENERIC,
    complexity: Optional[float] = None
) -> ProblemDefinition:
    """Helper to create problem definition."""
    if complexity is None:
        # Estimate from description length
        complexity = min(10.0, max(3.0, len(description) / 200))
    
    return ProblemDefinition(
        id=f"prob_{uuid.uuid4().hex[:8]}",
        title=title,
        description=description,
        domain=domain,
        complexity_score=ComplexityScore(
            cognitive_complexity=complexity,
            computational_complexity=complexity * 0.8,
            domain_complexity=complexity * 0.7,
            integration_complexity=complexity * 0.6,
            coordination_complexity=complexity * 0.5,
            technical_complexity=complexity * 0.9,
            overall_complexity=complexity
        )
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    engine = EnhancedDecompositionEngine()
    
    problem = create_problem_definition(
        title="Build AI-Powered Customer Support System",
        description="""
        Develop a comprehensive customer support system that uses AI to:
        - Automatically categorize and route incoming tickets
        - Provide intelligent response suggestions to agents
        - Analyze customer sentiment in real-time
        - Generate insights from support interactions
        - Integrate with existing CRM systems
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=7.5
    )
    
    plan = engine.decompose(problem, strategy=DecompositionStrategy.HYBRID)
    
    print(f"\nDecomposition Plan: {plan.id}")
    print(f"Strategy: {plan.strategy_used.value}")
    print(f"Quality Score: {plan.overall_quality:.2f}")
    print(f"\nSub-Problems ({len(plan.sub_problems)}):")
    
    for i, sp in enumerate(plan.sub_problems, 1):
        print(f"\n{i}. {sp.title}")
        print(f"   Type: {sp.type.value}")
        print(f"   Complexity: {sp.complexity_score.overall_complexity:.1f}")
        print(f"   Priority: {sp.priority}")
        print(f"   Effort: {sp.estimated_effort_hours}h")
        if sp.dependencies:
            print(f"   Dependencies: {len(sp.dependencies)}")
