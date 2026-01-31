"""
Canonical Schemas for Long-Horizon Agentic Framework

These schemas define the Anti-Corruption Layer (ACL) for all state representations.
All external data must be normalized to these canonical forms before processing.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All fields required
- Law of UTC: All timestamps in UTC
- Law of Idempotency: All operations replay-safe
- Anti-Corruption Layer: No external data formats leak through

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from .state_schemas import (
    StateLevel,
    StateSnapshot,
    StateDelta,
    StateVersion,
    StateCheckpoint
)
from .workflow_schemas import (
    WorkflowStatus,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowDependency,
    HumanHandoff
)
from .temporal_schemas import (
    TemporalEvent,
    CausalLink,
    TemporalPattern,
    TimeWindow,
    TrendAnalysis
)
from .learning_schemas import (
    LearningOutcome,
    StrategyPerformance,
    ABTestResult,
    AdaptationAction
)
from .checkpoint_schemas import (
    CheckpointMetadata,
    CheckpointIntegrity,
    ReplaySession
)

__all__ = [
    # State Schemas
    "StateLevel",
    "StateSnapshot",
    "StateDelta",
    "StateVersion",
    "StateCheckpoint",

    # Workflow Schemas
    "WorkflowStatus",
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowDependency",
    "HumanHandoff",

    # Temporal Schemas
    "TemporalEvent",
    "CausalLink",
    "TemporalPattern",
    "TimeWindow",
    "TrendAnalysis",

    # Learning Schemas
    "LearningOutcome",
    "StrategyPerformance",
    "ABTestResult",
    "AdaptationAction",

    # Checkpoint Schemas
    "CheckpointMetadata",
    "CheckpointIntegrity",
    "ReplaySession",
]
