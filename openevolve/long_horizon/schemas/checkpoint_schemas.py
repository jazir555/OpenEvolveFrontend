"""
Checkpoint & Replay Schemas

Canonical schemas for checkpoint creation and replay functionality.
All timestamps in UTC. All operations idempotent.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator
import hashlib
import json


class CheckpointMetadata(BaseModel):
    """
    Metadata for a checkpoint.

    Contains all information needed to identify and manage checkpoints.
    """
    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    checkpoint_name: str = Field(..., description="Human-readable checkpoint name")

    # Association
    workflow_id: str = Field(..., description="Associated workflow ID")
    execution_id: str = Field(..., description="Associated execution ID")
    snapshot_id: str = Field(..., description="Associated state snapshot ID")

    # Checkpoint type
    checkpoint_type: str = Field(
        ...,
        description="Type: milestone, error, handoff, manual, periodic"
    )
    checkpoint_reason: str = Field(..., description="Why checkpoint was created")

    # Position in workflow
    step_number: int = Field(..., description="Current step when checkpointed")
    step_description: str = Field(..., description="Description of current step")

    # Timing
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Checkpoint creation time (UTC)"
    )
    execution_time_seconds: float = Field(
        ...,
        description="Time elapsed since execution start"
    )

    # Creator
    created_by: str = Field(..., description="Creator (agent/human/system)")

    # State information
    state_size_bytes: int = Field(0, description="Size of state in bytes")
    is_compressed: bool = Field(False, description="Whether state is compressed")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio achieved")

    # Restoration capability
    can_restore: bool = Field(True, description="Whether checkpoint can be restored")
    restoration_count: int = Field(0, description="Number of times restored")

    # Dependencies
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other checkpoints this depends on"
    )

    # Tags and metadata
    tags: List[str] = Field(default_factory=list, description="Checkpoint tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    @validator('created_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class CheckpointIntegrity(BaseModel):
    """
    Integrity information for a checkpoint.

    Enables validation and trust in checkpoint data.
    """
    integrity_id: str = Field(..., description="Unique integrity record")
    checkpoint_id: str = Field(..., description="Associated checkpoint ID")

    # Checksums
    sha256_hash: str = Field(..., description="SHA-256 hash of state data")
    md5_hash: Optional[str] = Field(None, description="MD5 hash (legacy support)")

    # Validation
    is_valid: bool = Field(True, description="Whether checkpoint passes validation")
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Validation errors if any"
    )

    # Integrity checks performed
    checks_performed: List[str] = Field(
        default_factory=list,
        description="Integrity checks performed"
    )

    # Metadata
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last validation time (UTC)"
    )
    validation_method: str = Field(..., description="How validation was performed")

    @validator('validated_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @staticmethod
    def compute_hash(state_data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of state data.

        Args:
            state_data: State data dictionary

        Returns:
            Hex-encoded SHA-256 hash
        """
        # Ensure consistent serialization
        json_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class ReplaySession(BaseModel):
    """
    Session for replaying from a checkpoint.

    Tracks replay operations and their results.
    """
    replay_id: str = Field(..., description="Unique replay session identifier")
    checkpoint_id: str = Field(..., description="Checkpoint being replayed from")

    # Replay context
    replay_reason: str = Field(..., description="Why replay is being performed")
    replay_type: str = Field(
        ...,
        description="Type: debug, analysis, retry, branch"
    )

    # Execution state
    original_execution_id: str = Field(..., description="Original execution being replayed")
    new_execution_id: Optional[str] = Field(None, description="New execution ID if branched")

    # Status
    status: str = Field(
        default="initialized",
        description="Status: initialized, running, paused, completed, failed"
    )
    current_step: int = Field(0, description="Current step in replay")

    # Modifications
    modifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Modifications made during replay"
    )
    divergence_point: Optional[str] = Field(
        None,
        description="Step where replay diverges from original"
    )

    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Replay start time (UTC)"
    )
    completed_at: Optional[datetime] = Field(None, description="Replay completion time (UTC)")
    duration_seconds: Optional[float] = Field(None, description="Replay duration")

    # Results
    replay_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from replay"
    )
    comparison_to_original: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparison with original execution"
    )

    # Metadata
    replayed_by: str = Field(..., description="Agent/human initiating replay")
    notes: str = Field("", description="Notes on replay session")

    @validator('started_at', 'completed_at')
    def ensure_utc(cls, v):
        """Validate timestamps are in UTC"""
        if v is not None and v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    def mark_complete(self) -> None:
        """Mark replay as complete"""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
