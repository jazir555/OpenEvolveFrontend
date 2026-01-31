"""
Long-Horizon Agentic Framework

Core framework for agents that maintain state across days/weeks/months.

This framework provides:
- Persistent state management with versioning
- Time-aware workflow orchestration
- Temporal context and reasoning
- Online learning and adaptation
- Checkpoint and replay capabilities

All components follow CLAUDE.md principles:
- Law of Runtime Truth: Verify everything with execution
- Law of Idempotency: All operations replay-safe
- Law of UTC: All timestamps in UTC
- Law of Configuration Explicitness: All settings via environment variables
- Anti-Corruption Layer: Canonical schemas for all data

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from .state_manager import StateManager
from .workflow_orchestrator import WorkflowOrchestrator
from .temporal_context import TemporalContextManager
from .learning_engine import LearningEngine
from .checkpoint_replay import CheckpointManager, ReplayEngine, CheckpointValidator

# Schemas
from .schemas import (
    StateLevel,
    StateSnapshot,
    StateDelta,
    StateVersion,
    StateCheckpoint,
    WorkflowStatus,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowDependency,
    HumanHandoff,
    TemporalEvent,
    CausalLink,
    TemporalPattern,
    TimeWindow,
    TrendAnalysis,
    LearningOutcome,
    StrategyPerformance,
    ABTestResult,
    AdaptationAction,
    CheckpointMetadata,
    CheckpointIntegrity,
    ReplaySession
)

__version__ = "1.0.0"
__author__ = "Claude (Sonnet 4.5)"

__all__ = [
    # Core Components
    "StateManager",
    "WorkflowOrchestrator",
    "TemporalContextManager",
    "LearningEngine",
    "CheckpointManager",
    "ReplayEngine",
    "CheckpointValidator",

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


def create_framework(config: dict = None) -> dict:
    """
    Factory function to create the complete long-horizon framework.

    Environment Variables Required:
    - MONGODB_URL: MongoDB connection string
    - NEO4J_URL: Neo4j connection string
    - NEO4J_USER: Neo4j username
    - NEO4J_PASSWORD: Neo4j password

    Optional Environment Variables:
    - STATE_COMPRESSION_ENABLED: Enable compression (default: true)
    - STATE_MAX_VERSIONS: Max versions to keep (default: 1000)
    - WORKFLOW_TIMEOUT_DEFAULT: Default timeout in seconds (default: 3600)
    - WORKFLOW_MAX_RETRIES: Default max retry count (default: 3)

    Args:
        config: Optional configuration dict (overrides env vars)

    Returns:
        Dictionary with all framework components:
            - state_manager: StateManager instance
            - workflow_orchestrator: WorkflowOrchestrator instance
            - temporal_context: TemporalContextManager instance
            - learning_engine: LearningEngine instance
            - checkpoint_manager: CheckpointManager instance
            - replay_engine: ReplayEngine instance

    Example:
        ```python
        from openvolve.long_horizon import create_framework

        # Create framework
        framework = create_framework()

        # Access components
        state_manager = framework['state_manager']
        orchestrator = framework['workflow_orchestrator']

        # Use components
        await state_manager.save_snapshot(
            state_data={'key': 'value'},
            level='session',
            workflow_id='my_workflow'
        )
        ```
    """
    import os

    # Validate required environment variables
    required_vars = ['MONGODB_URL', 'NEO4J_URL', 'NEO4J_USER', 'NEO4J_PASSWORD']
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {missing}. "
            "Please set MONGODB_URL, NEO4J_URL, NEO4J_USER, and NEO4J_PASSWORD."
        )

    # Initialize state manager first (other components depend on it)
    state_manager = StateManager(config=config)

    # Initialize other components
    workflow_orchestrator = WorkflowOrchestrator(
        state_manager=state_manager,
        config=config
    )

    temporal_context = TemporalContextManager(config=config)

    learning_engine = LearningEngine(config=config)

    checkpoint_manager = CheckpointManager(
        state_manager=state_manager,
        config=config
    )

    replay_engine = ReplayEngine(
        state_manager=state_manager,
        config=config
    )

    return {
        'state_manager': state_manager,
        'workflow_orchestrator': workflow_orchestrator,
        'temporal_context': temporal_context,
        'learning_engine': learning_engine,
        'checkpoint_manager': checkpoint_manager,
        'replay_engine': replay_engine
    }
