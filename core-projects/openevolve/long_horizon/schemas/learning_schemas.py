"""
Learning & Adaptation Schemas

Canonical schemas for learning from outcomes and adapting strategies.
All timestamps in UTC. All operations idempotent.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator


class LearningOutcome(BaseModel):
    """
    Result of a learning experience.

    Captures what the agent learned from a workflow execution.
    """
    outcome_id: str = Field(..., description="Unique outcome identifier")

    # Context
    workflow_id: str = Field(..., description="Workflow where learning occurred")
    execution_id: str = Field(..., description="Execution instance")

    # Learning content
    lesson_type: str = Field(..., description="Type: success, failure, optimization, insight")
    lesson_description: str = Field(..., description="What was learned")

    # Performance metrics
    success: bool = Field(..., description="Whether the outcome was successful")
    performance_score: float = Field(
        ...,
        description="Performance score achieved",
        ge=0.0,
        le=1.0
    )
    baseline_score: Optional[float] = Field(
        None,
        description="Baseline for comparison"
    )

    # Contextual factors
    strategy_used: str = Field(..., description="Strategy that produced this outcome")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used"
    )
    environmental_factors: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant environmental context"
    )

    # Causal links
    causal_factors: List[str] = Field(
        default_factory=list,
        description="Factors that caused this outcome"
    )

    # Generalization
    generalizable: bool = Field(True, description="Whether this lesson generalizes")
    applicable_contexts: List[str] = Field(
        default_factory=list,
        description="Contexts where this applies"
    )

    # Confidence
    confidence: float = Field(
        default=0.5,
        description="Confidence in learning 0-1",
        ge=0.0,
        le=1.0
    )

    # Metadata
    learned_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When learning occurred (UTC)"
    )
    learned_by: str = Field(..., description="Agent that learned")

    @validator('learned_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class StrategyPerformance(BaseModel):
    """
    Performance tracking for strategies.

    Enables strategy selection and adaptation.
    """
    performance_id: str = Field(..., description="Unique performance record")
    strategy_id: str = Field(..., description="Strategy being tracked")

    # Performance metrics
    total_uses: int = Field(0, description="Total times strategy was used")
    successful_uses: int = Field(0, description="Successful executions")
    failed_uses: int = Field(0, description="Failed executions")

    # Average performance
    avg_performance_score: float = Field(
        0.0,
        description="Average performance score",
        ge=0.0,
        le=1.0
    )
    avg_execution_time: Optional[float] = Field(None, description="Average execution time (seconds)")

    # Recent performance (last N uses)
    recent_performance: List[float] = Field(
        default_factory=list,
        description="Recent performance scores"
    )
    recent_trend: str = Field(
        default="stable",
        description="Trend: improving, stable, declining"
    )

    # Context-specific performance
    context_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance in different contexts"
    )

    # Cost tracking
    avg_resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Average resource consumption"
    )

    # Metadata
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time (UTC)"
    )
    first_used: Optional[datetime] = Field(None, description="First use time (UTC)")

    @validator('last_updated', 'first_used')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses


class ABTestResult(BaseModel):
    """
    Results from A/B testing strategies.

    Enables evidence-based strategy selection.
    """
    test_id: str = Field(..., description="Unique test identifier")

    # Test configuration
    test_name: str = Field(..., description="Test name/description")
    hypothesis: str = Field(..., description="Hypothesis being tested")

    # Variants
    control_strategy: str = Field(..., description="Control strategy ID")
    treatment_strategy: str = Field(..., description="Treatment strategy ID")

    # Results
    control_performance: float = Field(..., description="Control performance score")
    treatment_performance: float = Field(..., description="Treatment performance score")
    performance_delta: float = Field(..., description="Treatment - Control")

    # Statistical significance
    sample_size: int = Field(..., description="Total sample size")
    p_value: Optional[float] = Field(None, description="Statistical significance")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None,
        description="Confidence interval for delta"
    )
    is_significant: bool = Field(False, description="Whether result is statistically significant")

    # Context
    test_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context in which test was run"
    )

    # Recommendation
    recommended_strategy: str = Field(..., description="Strategy to use going forward")
    recommendation_confidence: float = Field(
        ...,
        description="Confidence in recommendation",
        ge=0.0,
        le=1.0
    )

    # Metadata
    started_at: datetime = Field(..., description="Test start time (UTC)")
    completed_at: datetime = Field(..., description="Test completion time (UTC)")
    conducted_by: str = Field(..., description="Agent that ran test")

    @validator('started_at', 'completed_at')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class AdaptationAction(BaseModel):
    """
    Action taken to adapt agent behavior.

    Tracks what adaptations were made and why.
    """
    action_id: str = Field(..., description="Unique action identifier")
    action_type: str = Field(
        ...,
        description="Type: parameter_change, strategy_switch, new_strategy, rule_addition"
    )

    # What changed
    target_component: str = Field(..., description="Component being adapted")
    previous_value: Optional[Any] = Field(None, description="Value before adaptation")
    new_value: Any = Field(..., description="Value after adaptation")

    # Why changed
    trigger_reason: str = Field(..., description="Reason for adaptation")
    trigger_outcome_id: Optional[str] = Field(None, description="Outcome that triggered this")
    confidence: float = Field(
        default=0.5,
        description="Confidence this adaptation will help",
        ge=0.0,
        le=1.0
    )

    # Expected impact
    expected_improvement: str = Field(..., description="Expected improvement description")
    expected_impact: str = Field(
        default="medium",
        description="Impact level: none, low, medium, high"
    )

    # Validation
    validated: bool = Field(False, description="Whether adaptation has been validated")
    validation_result: Optional[Dict[str, Any]] = Field(None, description="Validation results")

    # Rollback
    can_rollback: bool = Field(True, description="Whether this can be rolled back")
    rolled_back: bool = Field(False, description="Whether this has been rolled back")

    # Metadata
    adapted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When adaptation was made (UTC)"
    )
    adapted_by: str = Field(..., description="Agent making adaptation")

    @validator('adapted_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v
