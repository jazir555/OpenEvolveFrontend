"""
Workflow State Manager Module

Manages workflow state persistence and resumption with:
- Save workflow state at any point
- Resume from saved state
- State versioning
- Audit trail
- Rollback support
- Branch and merge functionality
"""

from __future__ import annotations

import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import uuid

from sovereign_data_models import (
    WorkflowState, CheckpointInfo, AuditTrail, AuditEvent,
    WorkflowProgress
)
from workflow_persistence import WorkflowPersistence, generate_workflow_id, generate_state_id

logger = logging.getLogger(__name__)


class WorkflowStateManager:
    """
    Manages workflow state persistence and resumption.

    Features:
    - Save workflow state at any point
    - Resume from saved state
    - State versioning
    - Audit trail
    - Rollback support
    """

    def __init__(self, storage_backend: str = "file", storage_path: str = "workflow_states"):
        """
        Initialize with storage directory.

        Args:
            storage_backend: Storage backend ("file", "sqlite", "postgres")
            storage_path: Path for storage
        """
        self.persistence = WorkflowPersistence(storage_backend, storage_path)
        self.branches: Dict[str, Dict[str, WorkflowState]] = {}  # workflow_id -> {branch_name -> state}

    def save_state(
        self,
        workflow_id: str,
        state: WorkflowState,
        checkpoint_name: str = None
    ) -> str:
        """
        Save current workflow state.

        Creates checkpoint that can be resumed later.

        Args:
            workflow_id: Workflow identifier
            state: WorkflowState to save
            checkpoint_name: Optional checkpoint name

        Returns:
            checkpoint_id
        """
        try:
            # Ensure state has correct workflow_id
            state.workflow_id = workflow_id
            state.updated_at = datetime.now()

            # Persist state
            state_id = self.persistence.persist_state(state)

            # Create checkpoint metadata
            checkpoint_id = f"checkpoint_{uuid.uuid4().hex[:12]}"
            state_size = len(json.dumps(state.to_dict()))

            checkpoint = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                workflow_id=workflow_id,
                checkpoint_name=checkpoint_name or f"Checkpoint at {state.current_stage}",
                created_at=datetime.now(),
                stage=state.current_stage,
                progress=state.stage_progress,
                state_size=state_size,
                parent_checkpoint_id=getattr(state, 'current_checkpoint_id', None),  # Track parent checkpoint
                branch_name=state.branch_name
            )

            self.persistence.save_checkpoint(checkpoint)

            # Add to audit trail
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if not audit_trail:
                audit_trail = AuditTrail(
                    workflow_id=workflow_id,
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )

            event = AuditEvent(
                event_id=f"event_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                event_type="checkpoint",
                actor="system",
                description=f"Created checkpoint: {checkpoint.checkpoint_name}",
                from_state_id=None,
                to_state_id=state_id,
                stage=state.current_stage,
                progress_after=state.stage_progress,
                metadata={'checkpoint_id': checkpoint_id}
            )

            audit_trail.add_event(event)
            self.persistence.save_audit_trail(audit_trail)

            logger.info(f"Saved checkpoint {checkpoint_id} for workflow {workflow_id}")
            return checkpoint_id

        except (OSError, IOError, TypeError, AttributeError) as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
            raise

    def load_state(
        self,
        workflow_id: str,
        checkpoint_id: str = None
    ) -> Optional[WorkflowState]:
        """
        Load workflow state from checkpoint.

        If checkpoint_id is None, loads latest checkpoint.

        Args:
            workflow_id: Workflow identifier
            checkpoint_id: Optional checkpoint ID (loads latest if None)

        Returns:
            WorkflowState or None if not found
        """
        try:
            # Load state
            if checkpoint_id:
                # Load specific checkpoint
                checkpoint = self._get_checkpoint_by_id(workflow_id, checkpoint_id)
                if not checkpoint:
                    logger.warning(f"Checkpoint {checkpoint_id} not found for workflow {workflow_id}")
                    return None

                # Load the state that was saved at this checkpoint
                state = self.persistence.retrieve_state(workflow_id, checkpoint_id)
                if not state:
                    # Fallback to latest state if specific checkpoint state not found
                    state = self.persistence.retrieve_state(workflow_id, None)
            else:
                # Load latest state
                state = self.persistence.retrieve_state(workflow_id, None)

            if not state:
                logger.warning(f"No state found for workflow {workflow_id}")
                return None

            # Log to audit trail
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if audit_trail:
                event = AuditEvent(
                    event_id=f"event_{uuid.uuid4().hex[:12]}",
                    timestamp=datetime.now(),
                    event_type="state_change",
                    actor="user",
                    description=f"Loaded state for workflow",
                    from_state_id=None,
                    to_state_id=state.state_id,
                    stage=state.current_stage,
                    progress_after=state.stage_progress,
                    metadata={'checkpoint_id': checkpoint_id}
                )
                audit_trail.add_event(event)
                self.persistence.save_audit_trail(audit_trail)

            logger.info(f"Loaded state {state.state_id} for workflow {workflow_id}")
            return state

        except (OSError, IOError, json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)
            return None

    def _get_checkpoint_by_id(
        self,
        workflow_id: str,
        checkpoint_id: str
    ) -> Optional[CheckpointInfo]:
        """Get checkpoint by ID."""
        checkpoints = self.persistence.list_checkpoints(workflow_id)
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None

    def list_checkpoints(self, workflow_id: str) -> List[CheckpointInfo]:
        """
        List all checkpoints for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of CheckpointInfo objects
        """
        try:
            checkpoints = self.persistence.list_checkpoints(workflow_id)
            logger.info(f"Found {len(checkpoints)} checkpoints for workflow {workflow_id}")
            return checkpoints
        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to list checkpoints: {e}", exc_info=True)
            return []

    def rollback_to_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str
    ) -> Optional[WorkflowState]:
        """
        Rollback workflow to previous checkpoint.

        Useful when current state is problematic.

        Args:
            workflow_id: Workflow identifier
            checkpoint_id: Checkpoint to rollback to

        Returns:
            Rolled back WorkflowState or None
        """
        try:
            # Load checkpoint
            checkpoint = self._get_checkpoint_by_id(workflow_id, checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint {checkpoint_id} not found for workflow {workflow_id}")
                return None

            # Retrieve the specific state version from the checkpoint
            state = self.persistence.retrieve_state(workflow_id, checkpoint_id)
            if not state:
                # If specific version not found, fall back to latest
                state = self.persistence.retrieve_state(workflow_id, None)
                if not state:
                    return None

            # Update the current state to match the checkpoint
            self.current_states[workflow_id] = state

            # Log rollback to audit trail
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if not audit_trail:
                audit_trail = AuditTrail(workflow_id=workflow_id)

            event = AuditEvent(
                event_id=f"event_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                event_type="rollback",
                actor="user",
                description=f"Rolled back to checkpoint: {checkpoint.checkpoint_name}",
                from_state_id=state.state_id,
                to_state_id=state.state_id,  # Same state after rollback
                stage=state.current_stage,
                progress_before=state.stage_progress,
                progress_after=state.stage_progress,
                metadata={
                    'checkpoint_id': checkpoint_id,
                    'rollback_from': state.state_id
                }
            )

            audit_trail.add_event(event)
            self.persistence.save_audit_trail(audit_trail)

            logger.info(f"Rolled back workflow {workflow_id} to checkpoint {checkpoint_id}")
            return state

        except (OSError, IOError, TypeError, AttributeError) as e:
            logger.error(f"Failed to rollback: {e}", exc_info=True)
            return None

    def create_checkpoint_branch(
        self,
        workflow_id: str,
        checkpoint_id: str,
        branch_name: str
    ) -> Optional[str]:
        """
        Create experimental branch from checkpoint.

        Allows exploration without affecting main workflow.

        Args:
            workflow_id: Workflow identifier
            checkpoint_id: Checkpoint to branch from
            branch_name: Name for the new branch

        Returns:
            New workflow_id for the branch
        """
        try:
            # Load checkpoint state
            state = self.load_state(workflow_id, checkpoint_id)
            if not state:
                logger.error(f"Failed to load state for branching from checkpoint {checkpoint_id}")
                return None

            # Create new workflow for branch
            branch_workflow_id = f"{workflow_id}_branch_{branch_name}"

            # Copy state to branch
            state.workflow_id = branch_workflow_id
            state.state_id = generate_state_id()
            state.branch_name = branch_name
            state.parent_state_id = state.state_id  # Mark as branch point

            # Persist branch state
            self.persistence.persist_state(state)

            # Track branch
            if workflow_id not in self.branches:
                self.branches[workflow_id] = {}
            self.branches[workflow_id][branch_name] = state

            # Log to audit trail
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if not audit_trail:
                audit_trail = AuditTrail(workflow_id=workflow_id)

            event = AuditEvent(
                event_id=f"event_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                event_type="branch",
                actor="user",
                description=f"Created branch '{branch_name}' from checkpoint",
                from_state_id=state.parent_state_id,
                to_state_id=state.state_id,
                stage=state.current_stage,
                progress_after=state.stage_progress,
                metadata={
                    'checkpoint_id': checkpoint_id,
                    'branch_name': branch_name,
                    'branch_workflow_id': branch_workflow_id
                }
            )

            audit_trail.add_event(event)
            self.persistence.save_audit_trail(audit_trail)

            logger.info(f"Created branch '{branch_name}' (workflow: {branch_workflow_id}) from checkpoint {checkpoint_id}")
            return branch_workflow_id

        except (OSError, IOError, TypeError, AttributeError) as e:
            logger.error(f"Failed to create branch: {e}", exc_info=True)
            return None

    def merge_branch(
        self,
        workflow_id: str,
        branch_name: str,
        strategy: str = "keep_main"
    ) -> Optional[WorkflowState]:
        """
        Merge experimental branch back into main workflow.

        Args:
            workflow_id: Main workflow ID
            branch_name: Branch name to merge
            strategy: Merge strategy ("keep_main", "use_branch", "merge")

        Returns:
            Merged WorkflowState or None
        """
        try:
            # Get branch state
            if workflow_id not in self.branches or branch_name not in self.branches[workflow_id]:
                logger.error(f"Branch '{branch_name}' not found for workflow {workflow_id}")
                return None

            branch_state = self.branches[workflow_id][branch_name]
            main_state = self.load_state(workflow_id, None)

            if not main_state:
                logger.error(f"Main workflow state not found for {workflow_id}")
                return None

            # Apply merge strategy
            if strategy == "keep_main":
                merged_state = main_state
                description = f"Merged branch '{branch_name}' - kept main state"
            elif strategy == "use_branch":
                merged_state = branch_state
                merged_state.workflow_id = workflow_id
                merged_state.branch_name = None
                description = f"Merged branch '{branch_name}' - used branch state"
            elif strategy == "merge":
                # Intelligent merge - combine states preserving unique values
                merged_state = main_state.copy()  # Start with main state
                # Update with branch values where they differ or are more recent
                for attr in dir(branch_state):
                    if not attr.startswith('_'):  # Skip private attributes
                        branch_val = getattr(branch_state, attr)
                        main_val = getattr(main_state, attr, None)
                        # Only update if the branch value is different or more recent
                        if branch_val != main_val:
                            setattr(merged_state, attr, branch_val)

                merged_state.workflow_id = workflow_id
                merged_state.branch_name = None
                description = f"Merged branch '{branch_name}' - intelligent merge"
            else:
                logger.error(f"Unknown merge strategy: {strategy}")
                return None

            # Persist merged state
            self.persistence.persist_state(merged_state)

            # Remove branch
            del self.branches[workflow_id][branch_name]

            # Log to audit trail
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if not audit_trail:
                audit_trail = AuditTrail(workflow_id=workflow_id)

            event = AuditEvent(
                event_id=f"event_{uuid.uuid4().hex[:12]}",
                timestamp=datetime.now(),
                event_type="merge",
                actor="user",
                description=description,
                from_state_id=main_state.state_id,
                to_state_id=merged_state.state_id,
                stage=merged_state.current_stage,
                progress_before=main_state.stage_progress,
                progress_after=merged_state.stage_progress,
                metadata={
                    'branch_name': branch_name,
                    'strategy': strategy
                }
            )

            audit_trail.add_event(event)
            self.persistence.save_audit_trail(audit_trail)

            logger.info(f"Merged branch '{branch_name}' into workflow {workflow_id} using strategy '{strategy}'")
            return merged_state

        except (OSError, IOError, TypeError, AttributeError) as e:
            logger.error(f"Failed to merge branch: {e}", exc_info=True)
            return None

    def get_audit_trail(self, workflow_id: str) -> Optional[AuditTrail]:
        """
        Get complete audit trail for workflow.

        Shows all state transitions, who made changes,
        when changes were made, and why.

        Args:
            workflow_id: Workflow identifier

        Returns:
            AuditTrail or None if not found
        """
        try:
            audit_trail = self.persistence.load_audit_trail(workflow_id)
            if audit_trail:
                logger.info(f"Loaded audit trail for workflow {workflow_id} with {len(audit_trail.events)} events")
            else:
                logger.info(f"No audit trail found for workflow {workflow_id}")
            return audit_trail
        except (OSError, IOError, TypeError, AttributeError) as e:
            logger.error(f"Failed to get audit trail: {e}", exc_info=True)
            return None

    def get_workflow_progress(self, workflow_id: str) -> Optional[WorkflowProgress]:
        """
        Get current workflow progress.

        Args:
            workflow_id: Workflow identifier

        Returns:
            WorkflowProgress or None if not found
        """
        try:
            state = self.load_state(workflow_id, None)
            if not state:
                return None

            # Calculate progress details
            total_sub_problems = 0
            completed_sub_problems = 0

            if state.decomposition_plan:
                total_sub_problems = len(state.decomposition_plan.sub_problems)
                completed_sub_problems = sum(
                    1 for sp in state.decomposition_plan.sub_problems
                    if sp.status.value in ["solved", "completed"]
                )

            progress = WorkflowProgress(
                workflow_id=workflow_id,
                current_stage=state.current_stage,
                stage_progress=state.stage_progress,
                status=state.status,
                created_at=state.created_at,
                updated_at=state.updated_at,
                total_sub_problems=total_sub_problems,
                completed_sub_problems=completed_sub_problems
            )

            return progress

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Failed to get workflow progress: {e}", exc_info=True)
            return None

    def list_all_workflows(self) -> List[str]:
        """
        List all workflow IDs.

        Returns:
            List of workflow IDs
        """
        try:
            if self.persistence.storage_backend == "file":
                workflows_dir = self.persistence.workflows_dir
                if not workflows_dir.exists():
                    return []
                return [d.name for d in workflows_dir.iterdir() if d.is_dir()]
            elif self.persistence.storage_backend == "sqlite":
                # Query the database for all workflow IDs
                conn = sqlite3.connect(self.persistence.db_path)
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT DISTINCT workflow_id FROM workflow_states")
                    rows = cursor.fetchall()
                    return [row[0] for row in rows]
                finally:
                    conn.close()
            else:
                return []
        except (OSError, IOError, sqlite3.Error) as e:
            logger.error(f"Failed to list workflows: {e}", exc_info=True)
            return []

    def delete_workflow(self, workflow_id: str):
        """
        Delete all data for a workflow.

        Args:
            workflow_id: Workflow identifier
        """
        try:
            # Delete states
            if self.persistence.storage_backend == "file":
                workflow_dir = self.persistence.workflows_dir / workflow_id
                if workflow_dir.exists():
                    import shutil
                    shutil.rmtree(workflow_dir)

                checkpoint_dir = self.persistence.checkpoints_dir / workflow_id
                if checkpoint_dir.exists():
                    import shutil
                    shutil.rmtree(checkpoint_dir)

                audit_file = self.persistence.audit_dir / f"{workflow_id}.json"
                if audit_file.exists():
                    audit_file.unlink()

            # Remove from branches tracking
            if workflow_id in self.branches:
                del self.branches[workflow_id]

            logger.info(f"Deleted workflow {workflow_id}")

        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to delete workflow: {e}", exc_info=True)
            raise
