"""
State Management Schemas

Canonical schemas for persistent state storage across long-horizon workflows.
All timestamps in UTC. All operations idempotent.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator


class StateLevel(str, Enum):
    """Levels of state hierarchy"""
    SESSION = "session"  # Single execution session
    WORKFLOW = "workflow"  # Across workflow instances
    AGENT = "agent"  # Agent-specific state
    GLOBAL = "global"  # Cross-agent, cross-workflow state


class StateSnapshot(BaseModel):
    """
    Complete state snapshot at a point in time.

    Immutable record of agent/workflow state.
    All timestamps in UTC ISO-8601 format.
    """
    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    level: StateLevel = Field(..., description="State hierarchy level")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    session_id: Optional[str] = Field(None, description="Session identifier")

    # State data (arbitrary JSON-serializable dict)
    state_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Actual state payload"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Snapshot creation time (UTC)"
    )
    created_by: Optional[str] = Field(None, description="Creator (agent/human)")
    version: int = Field(1, description="State version number")
    parent_snapshot_id: Optional[str] = Field(None, description="Parent snapshot for versioning")

    # Checkpoint information
    is_checkpoint: bool = Field(False, description="Whether this is a marked checkpoint")
    checkpoint_name: Optional[str] = Field(None, description="Checkpoint label if applicable")

    # Compression
    is_compressed: bool = Field(False, description="Whether state_data is compressed")
    compression_algorithm: Optional[str] = Field(None, description="Compression method used")

    @validator('created_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @validator('state_data')
    def ensure_serializable(cls, v):
        """Validate state data is JSON-serializable"""
        try:
            import json
            json.dumps(v)
        except Exception as e:
            raise ValueError(f"State data must be JSON-serializable: {e}")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StateDelta(BaseModel):
    """
    Difference between two state snapshots.

    Enables efficient storage and transmission of state changes.
    Deltas are replay-safe and idempotent.
    """
    delta_id: str = Field(..., description="Unique delta identifier")
    from_snapshot_id: str = Field(..., description="Source snapshot ID")
    to_snapshot_id: str = Field(..., description="Target snapshot ID")

    # Changes
    added_keys: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keys added in target state"
    )
    modified_keys: Dict[str, tuple] = Field(
        default_factory=dict,
        description="Keys changed: {key: (old_value, new_value)}"
    )
    deleted_keys: List[str] = Field(
        default_factory=list,
        description="Keys removed from target state"
    )

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Delta creation time (UTC)"
    )
    applied: bool = Field(False, description="Whether delta has been applied")

    @validator('created_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    def invert(self) -> 'StateDelta':
        """
        Create inverse delta for rollback.

        Returns:
            StateDelta that reverses this delta
        """
        return StateDelta(
            delta_id=f"{self.delta_id}_inverse",
            from_snapshot_id=self.to_snapshot_id,
            to_snapshot_id=self.from_snapshot_id,
            added_keys={k: v for k, v in self.modified_keys.items()},
            modified_keys={k: (v[1], v[0]) for k, v in self.modified_keys.items()},
            deleted_keys=self.added_keys,
            created_at=datetime.now(timezone.utc)
        )


class StateVersion(BaseModel):
    """
    Git-like version tracking for state snapshots.

    Enables branching, merging, and history traversal.
    """
    version_id: str = Field(..., description="Unique version identifier")
    snapshot_id: str = Field(..., description="Associated snapshot ID")

    # Version graph
    parent_version_id: Optional[str] = Field(None, description="Parent version")
    child_version_ids: List[str] = Field(
        default_factory=list,
        description="Child versions (branches)"
    )
    branch_name: Optional[str] = Field(None, description="Branch name if applicable")
    merge_source_id: Optional[str] = Field(None, description="Source version if merge commit")

    # Metadata
    commit_message: str = Field(..., description="Description of changes")
    commit_author: str = Field(..., description="Author (agent/human)")
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Commit timestamp (UTC)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Version tags (e.g., 'milestone', 'stable')"
    )

    @validator('committed_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v


class StateCheckpoint(BaseModel):
    """
    Named checkpoint for easy rollback and analysis.

    Checkpoints are significant states (milestones, errors, handoffs).
    """
    checkpoint_id: str = Field(..., description="Unique checkpoint identifier")
    snapshot_id: str = Field(..., description="Associated snapshot ID")
    checkpoint_name: str = Field(..., description="Human-readable checkpoint name")

    # Checkpoint type
    checkpoint_type: str = Field(..., description="Type: milestone, error, handoff, manual")
    workflow_id: Optional[str] = Field(None, description="Associated workflow")

    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Checkpoint creation time (UTC)"
    )
    created_by: str = Field(..., description="Creator (agent/human)")
    description: str = Field(..., description="Checkpoint description")

    # State statistics
    state_size_bytes: int = Field(0, description="Size of state in bytes")
    compression_ratio: Optional[float] = Field(None, description="Compression ratio if compressed")

    @validator('created_at')
    def ensure_utc(cls, v):
        """Validate timestamp is in UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v
