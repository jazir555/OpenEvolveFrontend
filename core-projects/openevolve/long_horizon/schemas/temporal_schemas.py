"""
Temporal Context Schemas

Canonical schemas for time-aware reasoning and temporal knowledge graphs.
All timestamps in UTC. All operations idempotent.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator


class TemporalEvent(BaseModel):
    """
    Event with temporal information.

    Core unit for temporal knowledge graphs.
    All times in UTC ISO-8601 format.
    """
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(..., description="Type of event")

    # Temporal information
    timestamp: datetime = Field(..., description="Event occurrence time (UTC)")
    time_window_start: Optional[datetime] = Field(None, description="Window start if applicable")
    time_window_end: Optional[datetime] = Field(None, description="Window end if applicable")

    # Event data
    event_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload"
    )

    # Context
    workflow_id: Optional[str] = Field(None, description="Associated workflow")
    agent_id: Optional[str] = Field(None, description="Associated agent")
    source: str = Field(..., description="Event source (system/human/agent)")

    # Importance
    importance: float = Field(
        default=0.5,
        description="Importance score 0-1",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in event data 0-1",
        ge=0.0,
        le=1.0
    )

    # Recurrence
    is_recurring: bool = Field(False, description="Whether this is a recurring event")
    recurrence_pattern: Optional[str] = Field(None, description="Recurrence pattern if applicable")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event creation time (UTC)"
    )
    tags: List[str] = Field(default_factory=list, description="Event tags")

    @validator('timestamp', 'time_window_start', 'time_window_end', 'created_at')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware (UTC)")
        return v


class CausalLink(BaseModel):
    """
    Causal relationship between events.

    Enables reasoning about cause and effect over time.
    """
    link_id: str = Field(..., description="Unique link identifier")
    cause_event_id: str = Field(..., description="Event that is the cause")
    effect_event_id: str = Field(..., description="Event that is the effect")

    # Causal relationship
    causal_type: str = Field(
        ...,
        description="Type: direct, indirect, correlation, precondition"
    )
    strength: float = Field(
        default=0.5,
        description="Causal strength 0-1",
        ge=0.0,
        le=1.0
    )

    # Temporal aspect
    time_lag_seconds: Optional[float] = Field(
        None,
        description="Time lag between cause and effect"
    )

    # Evidence
    evidence_count: int = Field(1, description="Number of observations")
    confidence: float = Field(
        default=0.5,
        description="Confidence in causal link 0-1",
        ge=0.0,
        le=1.0
    )

    # Metadata
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When link was discovered (UTC)"
    )
    discovered_by: str = Field(..., description="How link was discovered (agent/algorithm)")

    @validator('discovered_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class TemporalPattern(BaseModel):
    """
    Recurring pattern detected in temporal data.

    Enables learning from historical patterns.
    """
    pattern_id: str = Field(..., description="Unique pattern identifier")
    pattern_type: str = Field(
        ...,
        description="Type: periodic, sequential, trend, seasonal"
    )

    # Pattern description
    description: str = Field(..., description="Pattern description")
    pattern_expression: str = Field(..., description="Formal pattern expression")

    # Temporal characteristics
    period_seconds: Optional[float] = Field(None, description="Period if periodic")
    phase_shift: Optional[float] = Field(None, description="Phase offset if applicable")

    # Events in pattern
    event_types: List[str] = Field(..., description="Event types involved in pattern")
    typical_sequence: List[str] = Field(
        default_factory=list,
        description="Typical event sequence"
    )

    # Statistics
    occurrence_count: int = Field(1, description="Number of times observed")
    confidence: float = Field(
        default=0.5,
        description="Confidence in pattern 0-1",
        ge=0.0,
        le=1.0
    )

    # Prediction
    next_occurrence: Optional[datetime] = Field(None, description="Predicted next occurrence (UTC)")

    # Metadata
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When pattern was discovered (UTC)"
    )
    last_observed: Optional[datetime] = Field(None, description="Most recent observation (UTC)")

    @validator('discovered_at', 'last_observed', 'next_occurrence')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class TimeWindow(BaseModel):
    """
    Time range for queries and analysis.

    Used for temporal context retrieval.
    """
    window_id: str = Field(..., description="Unique window identifier")

    # Time bounds
    start_time: datetime = Field(..., description="Window start (UTC)")
    end_time: datetime = Field(..., description="Window end (UTC)")

    # Window type
    window_type: str = Field(
        default="absolute",
        description="Type: absolute, relative, sliding"
    )
    relative_offset: Optional[str] = Field(None, description="Relative offset if applicable")

    # Filters
    event_types: Optional[List[str]] = Field(None, description="Event types to include")
    importance_threshold: float = Field(
        0.0,
        description="Minimum importance score",
        ge=0.0,
        le=1.0
    )

    @validator('start_time', 'end_time')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware (UTC)")
        return v

    @validator('end_time')
    def validate_order(cls, v, values):
        """Validate end_time is after start_time"""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError("end_time must be after start_time")
        return v

    @property
    def duration_seconds(self) -> float:
        """Calculate window duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()


class TrendAnalysis(BaseModel):
    """
    Trend or anomaly detected in temporal data.

    Enables adaptive behavior based on patterns.
    """
    analysis_id: str = Field(..., description="Unique analysis identifier")

    # Trend information
    trend_type: str = Field(..., description="Type: increasing, decreasing, stable, anomaly")
    metric_name: str = Field(..., description="Metric being analyzed")

    # Statistical properties
    slope: Optional[float] = Field(None, description="Trend slope (change per time unit)")
    correlation: Optional[float] = Field(None, description="Correlation coefficient")
    p_value: Optional[float] = Field(None, description="Statistical significance")

    # Time range
    analysis_window: TimeWindow = Field(..., description="Time window analyzed")

    # Anomaly detection
    is_anomaly: bool = Field(False, description="Whether this is an anomaly")
    anomaly_score: Optional[float] = Field(None, description="Anomaly score if applicable")
    threshold: Optional[float] = Field(None, description="Threshold for anomaly detection")

    # Confidence
    confidence: float = Field(
        default=0.5,
        description="Confidence in analysis 0-1",
        ge=0.0,
        le=1.0
    )

    # Impact assessment
    impact_level: str = Field(
        default="none",
        description="Impact: none, low, medium, high, critical"
    )
    recommended_action: Optional[str] = Field(None, description="Suggested action")

    # Metadata
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When analysis was performed (UTC)"
    )
    analyzed_by: str = Field(..., description="Agent/algorithm that performed analysis")

    @validator('analyzed_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v
