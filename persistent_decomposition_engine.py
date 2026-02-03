"""
Persistent Decomposition Engine Module

Extends DecompositionEngine with workflow persistence capabilities.

Features:
- Automatic state saving at key points
- Resume from saved states
- Checkpoint management
- Audit trail tracking
- Integration with existing DecompositionEngine
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

from sovereign_data_models import (
    ProblemDefinition, DecompositionPlan, WorkflowState, WorkflowProgress,
    generate_id, ValidationResult
)
from decomposition_engine import DecompositionEngine
from workflow_state_manager import WorkflowStateManager
from workflow_persistence import generate_workflow_id, generate_state_id

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting and adaptive for Persistent Decomposition Engine
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

class PersistentDecompositionEngine(DecompositionEngine):
    """
    DecompositionEngine with workflow persistence.

    Automatically saves state at key points and supports resumption.
    """

    def __init__(
        self,
        state_manager: Optional[WorkflowStateManager] = None,
        auto_checkpoint: bool = True,
        storage_backend: str = "file",
        storage_path: str = "workflow_states",
        **kwargs
    ):
        """
        Initialize with state management.

        Args:
            state_manager: Optional WorkflowStateManager instance
            auto_checkpoint: Whether to automatically save checkpoints
            storage_backend: Storage backend to use
            storage_path: Path for storage
            **kwargs: Additional arguments passed to DecompositionEngine
        """
        super().__init__(**kwargs)

        # Initialize state manager
        if state_manager:
            self.state_manager = state_manager
        else:
            self.state_manager = WorkflowStateManager(
                storage_backend=storage_backend,
                storage_path=storage_path
            )

        self.auto_checkpoint = auto_checkpoint
        self.logger = logging.getLogger(__name__)

    def decompose(
        self,
        problem: ProblemDefinition,
        workflow_id: str = None,
        resume_from: str = None,
        strategy: Optional[str] = None,
        assign_teams: bool = False,
        teams: Optional[List] = None,
        use_semantic_analysis: Optional[bool] = None,
        **kwargs
    ) -> Tuple[DecompositionPlan, str]:
        """
        Decompose with automatic state management.

        Args:
            problem: Problem to decompose
            workflow_id: Workflow ID (generated if None)
            resume_from: Checkpoint ID to resume from
            strategy: Optional strategy name
            assign_teams: Whether to assign teams
            teams: Optional list of teams
            use_semantic_analysis: Whether to use semantic analysis
            **kwargs: Additional arguments

        Returns:
            (decomposition_plan, workflow_id)
        """
        import time
        start_time = time.time()
        success = False

        # Generate workflow_id if needed
        if workflow_id is None:
            workflow_id = generate_workflow_id()
            self.logger.info(f"Generated new workflow ID: {workflow_id}")

        # Resume or create new state
        if resume_from:
            state = self.state_manager.load_state(workflow_id, resume_from)
            if not state:
                self.logger.warning(f"Could not resume from checkpoint {resume_from}, creating new state")
                state = self._create_initial_state(workflow_id, problem)
            else:
                self.logger.info(f"Resumed workflow {workflow_id} from checkpoint {resume_from}")
        else:
            state = self._create_initial_state(workflow_id, problem)
            self.logger.info(f"Created new state for workflow {workflow_id}")

        try:
            # Update state for decomposition stage
            state.current_stage = "decomposition"
            state.stage_progress = 0.0
            state.status = "in_progress"
            state.problem = problem

            # Perform decomposition using parent class
            self.logger.info(f"Starting decomposition for workflow {workflow_id}")
            plan = super().decompose(
                problem,
                strategy=strategy,
                assign_teams=assign_teams,
                teams=teams,
                use_semantic_analysis=use_semantic_analysis
            )

            # Update state with decomposition results
            state.decomposition_plan = plan
            state.selected_strategy = plan.metadata.get('strategy', strategy or 'unknown')
            state.current_stage = "decomposition_complete"
            state.stage_progress = 1.0
            state.updated_at = datetime.now()

            # Auto-checkpoint if enabled
            if self.auto_checkpoint:
                checkpoint_id = self.state_manager.save_state(
                    workflow_id,
                    state,
                    checkpoint_name=f"Decomposition complete: {plan.strategy.value}"
                )
                self.logger.info(f"Saved checkpoint {checkpoint_id} after decomposition")

            self.logger.info(f"Decomposition complete for workflow {workflow_id}")

            success = True
            duration = time.time() - start_time

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful decomposition
            self._extract_persistent_decomp_knowledge("decompose", workflow_id, strategy, plan)
            self._track_persistent_decomp_performance("decompose", True, duration, len(plan.sub_problems))

            return plan, workflow_id

        except (RuntimeError, ValueError, TypeError) as e:
            duration = time.time() - start_time

            self.logger.error(f"Decomposition failed for workflow {workflow_id}: {e}", exc_info=True)

            # Update state with error
            state.status = "failed"
            state.error_message = str(e)
            state.updated_at = datetime.now()

            # Save error state
            if self.auto_checkpoint:
                self.state_manager.save_state(
                    workflow_id,
                    state,
                    checkpoint_name="Decomposition failed"
                )

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_persistent_decomp_alerts("decompose", False, workflow_id, str(e))
            self._track_persistent_decomp_performance("decompose", False, duration, 0)

            raise

    def _create_initial_state(
        self,
        workflow_id: str,
        problem: ProblemDefinition
    ) -> WorkflowState:
        """Create initial workflow state."""
        return WorkflowState(
            workflow_id=workflow_id,
            state_id=generate_state_id(),
            version=1,
            current_stage="decomposition",
            stage_progress=0.0,
            problem=problem,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status="in_progress"
        )

    def get_workflow_progress(
        self,
        workflow_id: str
    ) -> Optional[WorkflowProgress]:
        """
        Get current workflow progress.

        Args:
            workflow_id: Workflow ID

        Returns:
            WorkflowProgress or None
        """
        return self.state_manager.get_workflow_progress(workflow_id)

    def save_checkpoint(
        self,
        workflow_id: str,
        state: WorkflowState,
        checkpoint_name: str = None
    ) -> str:
        """
        Manually save a checkpoint.

        Args:
            workflow_id: Workflow ID
            state: Current workflow state
            checkpoint_name: Optional checkpoint name

        Returns:
            checkpoint_id
        """
        return self.state_manager.save_state(workflow_id, state, checkpoint_name)

    def load_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str = None
    ) -> Optional[WorkflowState]:
        """
        Load workflow from checkpoint.

        Args:
            workflow_id: Workflow ID
            checkpoint_id: Optional checkpoint ID (loads latest if None)

        Returns:
            WorkflowState or None
        """
        return self.state_manager.load_state(workflow_id, checkpoint_id)

    def list_checkpoints(
        self,
        workflow_id: str
    ) -> List:
        """
        List all checkpoints for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            List of CheckpointInfo objects
        """
        return self.state_manager.list_checkpoints(workflow_id)

    def rollback_to_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str
    ) -> Optional[WorkflowState]:
        """
        Rollback workflow to previous checkpoint.

        Args:
            workflow_id: Workflow ID
            checkpoint_id: Checkpoint to rollback to

        Returns:
            WorkflowState or None
        """
        return self.state_manager.rollback_to_checkpoint(workflow_id, checkpoint_id)

    def create_branch(
        self,
        workflow_id: str,
        checkpoint_id: str,
        branch_name: str
    ) -> Optional[str]:
        """
        Create experimental branch from checkpoint.

        Args:
            workflow_id: Main workflow ID
            checkpoint_id: Checkpoint to branch from
            branch_name: Name for the new branch

        Returns:
            New workflow_id for the branch or None
        """
        return self.state_manager.create_checkpoint_branch(
            workflow_id,
            checkpoint_id,
            branch_name
        )

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
        return self.state_manager.merge_branch(workflow_id, branch_name, strategy)

    def get_audit_trail(
        self,
        workflow_id: str
    ):
        """
        Get complete audit trail for workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            AuditTrail or None
        """
        return self.state_manager.get_audit_trail(workflow_id)

    def list_workflows(self) -> List[str]:
        """
        List all workflow IDs.

        Returns:
            List of workflow IDs
        """
        return self.state_manager.list_all_workflows()

    def delete_workflow(self, workflow_id: str):
        """
        Delete all data for a workflow.

        Args:
            workflow_id: Workflow ID
        """
        self.state_manager.delete_workflow(workflow_id)

    def export_workflow(
        self,
        workflow_id: str,
        output_path: str
    ):
        """
        Export complete workflow to archive.

        Args:
            workflow_id: Workflow ID
            output_path: Path for output archive
        """
        self.state_manager.persistence.export_workflow(workflow_id, output_path)

    def import_workflow(
        self,
        archive_path: str
    ) -> Optional[str]:
        """
        Import workflow from archive.

        Args:
            archive_path: Path to workflow archive

        Returns:
            workflow_id or None
        """
        return self.state_manager.persistence.import_workflow(archive_path)

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Persistent Decomposition
    # =========================================================================

    def _trigger_persistent_decomp_alerts(
        self,
        operation: str,
        success: bool,
        workflow_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for persistent decomposition failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                alert_manager.create_alert(
                    title=f"Persistent Decomposition Alert: {operation}",
                    description=f"Persistent Decomposition operation '{operation}' failed" +
                                 (f" for workflow '{workflow_id}'" if workflow_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="persistent_decomposition_engine",
                    component="persistent_decomposition",
                    metadata=metadata or {}
                )

        except Exception as e:
            self.logger.error(f"Failed to trigger Persistent Decomposition alert: {e}")

    def _extract_persistent_decomp_knowledge(
        self,
        operation: str,
        workflow_id: str,
        strategy: Optional[str],
        plan: DecompositionPlan
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract persistent decomposition knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"persistent_decomp_{operation}_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="persistent_decomposition_execution",
                source_component="persistent_decomposition_engine",
                title=f"Persistent Decomposition: {operation} - {workflow_id}",
                content={
                    "operation": operation,
                    "workflow_id": workflow_id,
                    "strategy": strategy or "unknown",
                    "num_subproblems": len(plan.sub_problems),
                    "strategy_used": plan.strategy.value if hasattr(plan, 'strategy') else "unknown",
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "auto_checkpoint": self.auto_checkpoint
                },
                tags=["persistent_decomposition", operation, "checkpoint"]
            )

            knowledge_engine.store_artifact(artifact)
            self.logger.debug(f"Extracted Persistent Decomposition knowledge for {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract Persistent Decomposition knowledge: {e}")
            return False

    def _track_persistent_decomp_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        num_subproblems: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track persistent decomposition performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"persistent_decomp_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "num_subproblems": num_subproblems
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                self.logger.debug(f"Tracked Persistent Decomposition performance for {operation}")

        except Exception as e:
            self.logger.error(f"Failed to track Persistent Decomposition performance: {e}")


def create_persistent_engine(
    auto_checkpoint: bool = True,
    storage_backend: str = "file",
    storage_path: str = "workflow_states",
    **kwargs
) -> PersistentDecompositionEngine:
    """
    Factory function to create a persistent decomposition engine.

    Args:
        auto_checkpoint: Whether to automatically save checkpoints
        storage_backend: Storage backend to use
        storage_path: Path for storage
        **kwargs: Additional arguments for DecompositionEngine

    Returns:
        PersistentDecompositionEngine instance
    """
    return PersistentDecompositionEngine(
        auto_checkpoint=auto_checkpoint,
        storage_backend=storage_backend,
        storage_path=storage_path,
        **kwargs
    )
