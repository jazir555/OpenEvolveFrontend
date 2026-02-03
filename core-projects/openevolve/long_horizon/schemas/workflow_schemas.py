"""
Workflow Management Schemas

Canonical schemas for workflow execution and orchestration.
All timestamps in UTC. All state transitions idempotent.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator


class WorkflowStatus(str, Enum):
    """Workflow execution states"""
    PENDING = "pending"  # Waiting to start
    RUNNING = "running"  # Currently executing
    PAUSED = "paused"  # Paused by human or logic
    WAITING = "waiting"  # Waiting for dependencies or human input
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user
    TIMEOUT = "timeout"  # Exceeded time limit


class WorkflowDefinition(BaseModel):
    """
    Workflow template and execution plan.

    Defines the structure, dependencies, and execution parameters.
    """
    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable workflow name")
    description: str = Field(..., description="Workflow description")

    # Workflow structure
    steps: List[Dict[str, Any]] = Field(
        ...,
        description="Workflow steps in execution order"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of workflows this depends on"
    )

    # Scheduling
    schedule_type: str = Field(
        default="manual",
        description="Schedule type: manual, cron, event_driven"
    )
    schedule_expression: Optional[str] = Field(
        None,
        description="Cron expression or event pattern"
    )
    timeout_seconds: Optional[int] = Field(
        None,
        description="Maximum execution time (None for unlimited)"
    )

    # Configuration
    retry_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Retry policy: max_retries, backoff_multiplier, etc."
    )
    resource_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="CPU, memory, GPU requirements"
    )

    # Human-in-the-loop
    human_handoff_points: List[str] = Field(
        default_factory=list,
        description="Step IDs requiring human intervention"
    )

    # Metadata
    version: int = Field(1, description="Workflow definition version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time (UTC)"
    )
    created_by: str = Field(..., description="Creator (agent/human)")
    tags: List[str] = Field(
        default_factory=list,
        description="Workflow tags for categorization"
    )

    @validator('created_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class WorkflowExecution(BaseModel):
    """
    Single execution instance of a workflow.

    Tracks runtime state, progress, and results.
    """
    execution_id: str = Field(..., description="Unique execution identifier")
    workflow_id: str = Field(..., description="Workflow definition ID")

    # State
    status: WorkflowStatus = Field(
        default=WorkflowStatus.PENDING,
        description="Current execution status"
    )
    current_step: int = Field(0, description="Current step index")
    completed_steps: List[str] = Field(
        default_factory=list,
        description="IDs of completed steps"
    )

    # Progress tracking
    total_steps: int = Field(..., description="Total number of steps")
    progress_percentage: float = Field(0.0, description="Progress 0-100")

    # Checkpoint support
    current_checkpoint_id: Optional[str] = Field(
        None,
        description="Most recent checkpoint ID"
    )
    can_resume_from: List[str] = Field(
        default_factory=list,
        description="Checkpoint IDs valid for resumption"
    )

    # Timing
    started_at: Optional[datetime] = Field(None, description="Start time (UTC)")
    completed_at: Optional[datetime] = Field(None, description="Completion time (UTC)")
    last_heartbeat: Optional[datetime] = Field(None, description="Last activity timestamp (UTC)")
    estimated_completion: Optional[datetime] = Field(None, description="ETA (UTC)")

    # Execution context
    input_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for this execution"
    )
    output_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output results from execution"
    )

    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(0, description="Number of retry attempts")
    retry_limit: int = Field(3, description="Maximum retry attempts")

    # Human interaction
    awaiting_human_input: bool = Field(False, description="Whether waiting for human")
    human_handoff_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of human interactions"
    )

    # Metadata
    execution_agent: Optional[str] = Field(None, description="Agent executing this workflow")
    parent_execution_id: Optional[str] = Field(None, description="Parent workflow if nested")
    child_execution_ids: List[str] = Field(
        default_factory=list,
        description="Child workflow executions"
    )

    @validator('started_at', 'completed_at', 'last_heartbeat', 'estimated_completion')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamps must be timezone-aware (UTC)")
        return v

    @validator('progress_percentage')
    def validate_progress(cls, v):
        """Validate progress is 0-100"""
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v


class WorkflowDependency(BaseModel):
    """
    Dependency relationship between workflows.

    Controls execution order and prerequisites.
    """
    dependency_id: str = Field(..., description="Unique dependency identifier")
    depends_on_workflow_id: str = Field(..., description="Workflow ID to depend on")
    dependent_workflow_id: str = Field(..., description="Workflow ID that depends")

    # Dependency type
    dependency_type: str = Field(
        default="completion",
        description="Type: completion, output, milestone"
    )
    required_output_key: Optional[str] = Field(
        None,
        description="Specific output key required if type='output'"
    )

    # Conditions
    wait_for_success: bool = Field(True, description="Only proceed if dependency succeeds")
    timeout_seconds: Optional[int] = Field(None, description="Max time to wait for dependency")

    # Status
    satisfied: bool = Field(False, description="Whether dependency is satisfied")
    satisfied_at: Optional[datetime] = Field(None, description="When satisfied (UTC)")

    @validator('satisfied_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class HumanHandoff(BaseModel):
    """
    Human-in-the-loop interaction point.

    Defines where and how humans interact with workflow.
    """
    handoff_id: str = Field(..., description="Unique handoff identifier")
    workflow_id: str = Field(..., description="Associated workflow ID")
    execution_id: str = Field(..., description="Associated execution ID")
    step_id: str = Field(..., description="Step ID for handoff")

    # Handoff configuration
    handoff_type: str = Field(..., description="Type: approval, input, review, guidance")
    timeout_seconds: Optional[int] = Field(
        None,
        description="Timeout for human response (None for indefinite)"
    )

    # Context
    context_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information provided to human"
    )
    request_message: str = Field(..., description="Message/request to human")

    # Human response
    response: Optional[str] = Field(None, description="Human's response")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Structured response data")
    responded_at: Optional[datetime] = Field(None, description="Response time (UTC)")
    responded_by: Optional[str] = Field(None, description="Human identifier")

    # Status
    status: str = Field(default="pending", description="pending, responded, skipped, timeout")
    assigned_to: Optional[str] = Field(None, description="Assigned human/user")

    # Timing
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Handoff creation time (UTC)"
    )
    expires_at: Optional[datetime] = Field(None, description="Expiration time (UTC)")

    @validator('created_at', 'responded_at', 'expires_at')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v
