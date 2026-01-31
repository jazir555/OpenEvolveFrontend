"""
Long-Horizon Learning Schemas

Data structures for long-horizon agentic workflow learning.
Supports online learning, A/B testing, causal modeling, and meta-learning.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC


class OutcomeType(Enum):
    """Types of learning outcomes"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


class AdaptationActionType(Enum):
    """Types of adaptation actions"""
    CHANGE_STRATEGY = "change_strategy"
    TUNE_PARAMETERS = "tune_parameters"
    SWITCH_MODE = "switch_mode"
    ROLLBACK = "rollback"
    NO_ACTION = "no_action"


class ExperimentStatus(Enum):
    """A/B experiment status"""
    RUNNING = "running"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    PAUSED = "paused"


class ExplorationStrategy(Enum):
    """Exploration vs exploitation strategies"""
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    BAYESIAN_UCB = "bayesian_ucb"


@dataclass
class LearningOutcome:
    """
    Single learning outcome from workflow execution

    Attributes:
        workflow_id: Workflow that generated this outcome
        strategy_used: Strategy that was applied
        outcome_type: Success, failure, partial, or error
        metrics: Performance metrics (fitness, cost, time, etc.)
        context: Execution context (domain, problem, config)
        timestamp: When outcome occurred (UTC)
        outcome_id: Unique identifier for this outcome
    """
    workflow_id: str
    strategy_used: str
    outcome_type: OutcomeType
    metrics: Dict[str, float]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    outcome_id: Optional[str] = None

    def __post_init__(self):
        """Generate outcome_id if not provided"""
        if self.outcome_id is None:
            self.outcome_id = f"{self.workflow_id}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "outcome_id": self.outcome_id,
            "workflow_id": self.workflow_id,
            "strategy_used": self.strategy_used,
            "outcome_type": self.outcome_type.value,
            "metrics": self.metrics,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class StrategyPerformance:
    """
    Performance tracking for a strategy over time

    Attributes:
        strategy_id: Unique strategy identifier
        performance_history: List of performance scores over time
        moving_average: Exponential moving average of performance
        confidence_interval: (lower, upper) confidence bounds
        last_updated: Last time this was updated (UTC)
        total_outcomes: Total number of outcomes recorded
        success_rate: Fraction of successful outcomes
        decay_rate: Performance decay rate (negative means improving)
    """
    strategy_id: str
    performance_history: List[float] = field(default_factory=list)
    moving_average: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_outcomes: int = 0
    success_rate: float = 0.0
    decay_rate: float = 0.0  # Negative = improving, Positive = degrading
    variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "strategy_id": self.strategy_id,
            "performance_history": self.performance_history[-100:],  # Last 100
            "moving_average": self.moving_average,
            "confidence_interval": self.confidence_interval,
            "last_updated": self.last_updated.isoformat(),
            "total_outcomes": self.total_outcomes,
            "success_rate": self.success_rate,
            "decay_rate": self.decay_rate,
            "variance": self.variance
        }


@dataclass
class AdaptationAction:
    """
    Recommended adaptation action

    Attributes:
        action_type: Type of action to take
        description: Human-readable description
        parameters: Parameters for the action
        expected_improvement: Expected improvement (0-1)
        confidence: Confidence in recommendation (0-1)
        rollback_plan: How to rollback if things go wrong
        priority: Priority score (0-100)
    """
    action_type: AdaptationActionType
    description: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    rollback_plan: Optional[str] = None
    priority: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action_type": self.action_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "rollback_plan": self.rollback_plan,
            "priority": self.priority
        }


@dataclass
class VariantStats:
    """
    Statistics for an A/B test variant

    Attributes:
        variant_id: Variant identifier
        sample_size: Number of observations
        mean_outcome: Average outcome score
        variance: Outcome variance
        conversion_rate: Success rate (for binary outcomes)
        confidence_interval: Statistical confidence interval
        observations: List of individual outcomes
    """
    variant_id: str
    sample_size: int = 0
    mean_outcome: float = 0.0
    variance: float = 0.0
    conversion_rate: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    observations: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "variant_id": self.variant_id,
            "sample_size": self.sample_size,
            "mean_outcome": self.mean_outcome,
            "variance": self.variance,
            "conversion_rate": self.conversion_rate,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class Experiment:
    """
    A/B experiment for testing agent behaviors

    Attributes:
        experiment_id: Unique identifier
        name: Human-readable name
        description: What is being tested
        variants: Map of variant_id -> VariantStats
        start_time: When experiment started (UTC)
        end_time: When experiment ended (UTC)
        status: Current status
        significance_level: Statistical threshold (default 0.05)
        min_sample_size: Minimum samples per variant
        winner: Winning variant (if determined)
    """
    experiment_id: str
    name: str
    description: str
    variants: Dict[str, VariantStats]
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.RUNNING
    significance_level: float = 0.05
    min_sample_size: int = 100
    winner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "significance_level": self.significance_level,
            "min_sample_size": self.min_sample_size,
            "winner": self.winner
        }


@dataclass
class ExperimentResults:
    """
    Results from an A/B test

    Attributes:
        experiment_id: Experiment identifier
        winner: Winning variant
        confidence: Statistical confidence
        improvement: Relative improvement over control
        significance: Whether result is statistically significant
        test_statistic: Test statistic value
        p_value: Statistical p-value
        recommendation: What to do next
    """
    experiment_id: str
    winner: Optional[str]
    confidence: float
    improvement: float
    significance: bool
    test_statistic: float
    p_value: float
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "winner": self.winner,
            "confidence": self.confidence,
            "improvement": self.improvement,
            "significance": self.significance,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "recommendation": self.recommendation
        }


@dataclass
class CausalRelationship:
    """
    Causal relationship between factors

    Attributes:
        cause: Causal factor (what changes)
        effect: Effect (what gets influenced)
        strength: Causal strength (0-1)
        confidence: Statistical confidence (0-1)
        mechanism: How the cause produces the effect
        evidence: Supporting evidence
    """
    cause: str
    effect: str
    strength: float
    confidence: float
    mechanism: Optional[str] = None
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "cause": self.cause,
            "effect": self.effect,
            "strength": self.strength,
            "confidence": self.confidence,
            "mechanism": self.mechanism,
            "evidence": self.evidence
        }


@dataclass
class CausalModel:
    """
    Causal model of a domain

    Attributes:
        model_id: Unique identifier
        domain: Problem domain
        relationships: List of causal relationships
        factors: All factors in the model
        outcomes: All outcomes in the model
        graph_data: Serialized graph structure
        created_at: When model was created (UTC)
        updated_at: Last update time (UTC)
    """
    model_id: str
    domain: str
    relationships: List[CausalRelationship]
    factors: List[str]
    outcomes: List[str]
    graph_data: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "domain": self.domain,
            "relationships": [r.to_dict() for r in self.relationships],
            "factors": self.factors,
            "outcomes": self.outcomes,
            "graph_data": self.graph_data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class EffectPrediction:
    """
    Prediction of intervention effect

    Attributes:
        intervention: What is being changed
        predicted_effect: Expected outcome
        confidence: Prediction confidence
        alternative_outcomes: Other possible outcomes
        risk_assessment: Potential risks
    """
    intervention: str
    predicted_effect: float
    confidence: float
    alternative_outcomes: List[Tuple[str, float]]
    risk_assessment: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "intervention": self.intervention,
            "predicted_effect": self.predicted_effect,
            "confidence": self.confidence,
            "alternative_outcomes": self.alternative_outcomes,
            "risk_assessment": self.risk_assessment
        }


@dataclass
class Explanation:
    """
    Explanation of an outcome using causal model

    Attributes:
        outcome: What is being explained
        causes: Identified causes
        contribution: How much each cause contributed
        confidence: Explanation confidence
        counterfactuals: What would have happened if causes were different
    """
    outcome: str
    causes: List[str]
    contribution: Dict[str, float]
    confidence: float
    counterfactuals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "outcome": self.outcome,
            "causes": self.causes,
            "contribution": self.contribution,
            "confidence": self.confidence,
            "counterfactuals": self.counterfactuals
        }


@dataclass
class MetaPattern:
    """
    Meta-learning pattern across workflows

    Attributes:
        pattern_id: Unique identifier
        description: What the pattern is
        applicable_domains: Where this applies
        evidence: Workflow runs where this worked
        confidence: Pattern confidence (0-1)
        feature_signature: Problem features that indicate this pattern
        expected_benefit: Expected improvement when using this pattern
    """
    pattern_id: str
    description: str
    applicable_domains: List[str]
    evidence: List[str]  # Workflow IDs
    confidence: float
    feature_signature: Dict[str, Any]
    expected_benefit: float
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "applicable_domains": self.applicable_domains,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "feature_signature": self.feature_signature,
            "expected_benefit": self.expected_benefit,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class StrategyRecommendation:
    """
    Strategy recommendation for a new problem

    Attributes:
        problem_id: Problem identifier
        recommended_strategy: Strategy to use
        confidence: Recommendation confidence
        rationale: Why this strategy
        expected_performance: Expected outcome
        alternative_strategies: Other strategies to consider
        transfer_source: Where this learning came from
    """
    problem_id: str
    recommended_strategy: str
    confidence: float
    rationale: str
    expected_performance: float
    alternative_strategies: List[Tuple[str, float]]  # (strategy, expected_performance)
    transfer_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "problem_id": self.problem_id,
            "recommended_strategy": self.recommended_strategy,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "expected_performance": self.expected_performance,
            "alternative_strategies": self.alternative_strategies,
            "transfer_source": self.transfer_source
        }


@dataclass
class StoredCausalModel:
    """
    Causal model stored in knowledge engine with persistent storage

    Attributes:
        model_id: Unique identifier
        domain: Problem domain
        neo4j_id: Neo4j node ID (if stored in graph database)
        qdrant_id: Qdrant point ID (if stored in vector database)
        metadata: Additional metadata about the stored model
    """
    model_id: str
    domain: str
    neo4j_id: Optional[str] = None
    qdrant_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if None"""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "domain": self.domain,
            "neo4j_id": self.neo4j_id,
            "qdrant_id": self.qdrant_id,
            "metadata": self.metadata
        }


@dataclass
class CounterfactualResult:
    """
    Result from counterfactual query on causal model

    Attributes:
        intervention: What was changed
        outcome: Variable of interest
        predicted_value: Predicted outcome under intervention
        actual_value: Actual outcome without intervention
        effect: Causal effect (predicted - actual)
        confidence: Confidence in prediction
        method: Method used for prediction
    """
    intervention: Dict[str, Any]
    outcome: str
    predicted_value: float
    actual_value: Optional[float] = None
    effect: Optional[float] = None
    confidence: float = 0.0
    method: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "intervention": self.intervention,
            "outcome": self.outcome,
            "predicted_value": self.predicted_value,
            "actual_value": self.actual_value,
            "effect": self.effect,
            "confidence": self.confidence,
            "method": self.method
        }
