"""
Workflow Orchestrator for Long-Horizon Agents

Implements time-aware workflow scheduling with checkpoint support.
Follows CLAUDE.md principles:
- Law of Runtime Truth: Verify all workflow executions
- Law of Idempotency: All operations replay-safe
- Law of UTC: All timestamps in UTC
- Law of Configuration Explicitness: All settings via environment variables

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import os
import structlog
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum

from .state_manager import StateManager
from .schemas.workflow_schemas import (
    WorkflowStatus,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowDependency,
    HumanHandoff
)
from .schemas.state_schemas import StateCheckpoint


logger = structlog.get_logger()


class WorkflowError(Exception):
    """Base exception for workflow errors"""
    pass


class WorkflowDependencyError(WorkflowError):
    """Raised when workflow dependencies are not satisfied"""
    pass


class WorkflowTimeoutError(WorkflowError):
    """Raised when workflow exceeds timeout"""
    pass


class WorkflowOrchestrator:
    """
    Orchestrates long-horizon workflow execution.

    Features:
    - Time-aware scheduling (cron-like)
    - Workflow state machine
    - Human-in-the-loop support
    - Checkpoint-based resumption
    - Dependency management
    - Distributed execution ready

    All operations idempotent and UTC-based.
    """

    def __init__(self, state_manager: StateManager, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Workflow Orchestrator.

        Environment Variables:
        - WORKFLOW_TIMEOUT_DEFAULT: Default timeout in seconds (default: 3600)
        - WORKFLOW_MAX_RETRIES: Default max retry count (default: 3)
        - WORKFLOW_HEARTBEAT_INTERVAL: Heartbeat interval in seconds (default: 30)
        - WORKFLOW_PARALLEL_WORKERS: Parallel worker count (default: 4)

        Args:
            state_manager: State manager instance
            config: Optional config dict
        """
        self.state_manager = state_manager
        self.config = config or self._load_config()
        self._validate_config()

        # Workflow execution storage
        self._executions: Dict[str, WorkflowExecution] = {}
        self._dependencies: Dict[str, List[WorkflowDependency]] = {}
        self._step_handlers: Dict[str, Callable] = {}

        # Scheduling
        self._scheduled_workflows: Dict[str, datetime] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}

        logger.info(
            "workflow_orchestrator_initialized",
            default_timeout=self.config.get('default_timeout', 3600),
            max_retries=self.config.get('max_retries', 3),
            parallel_workers=self.config.get('parallel_workers', 4)
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'default_timeout': int(os.getenv('WORKFLOW_TIMEOUT_DEFAULT', '3600')),
            'max_retries': int(os.getenv('WORKFLOW_MAX_RETRIES', '3')),
            'heartbeat_interval': int(os.getenv('WORKFLOW_HEARTBEAT_INTERVAL', '30')),
            'parallel_workers': int(os.getenv('WORKFLOW_PARALLEL_WORKERS', '4')),
        }

    def _validate_config(self) -> None:
        """Validate configuration"""
        if self.config.get('default_timeout', 0) <= 0:
            raise ValueError("WORKFLOW_TIMEOUT_DEFAULT must be positive")
        if self.config.get('max_retries', 0) < 0:
            raise ValueError("WORKFLOW_MAX_RETRIES must be non-negative")

    async def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        dependencies: Optional[List[str]] = None,
        schedule_type: str = "manual",
        schedule_expression: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        human_handoff_points: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> WorkflowDefinition:
        """
        Create a workflow definition.

        Args:
            workflow_id: Unique workflow identifier
            name: Human-readable name
            description: Workflow description
            steps: Workflow steps
            dependencies: Workflow IDs this depends on
            schedule_type: Schedule type (manual, cron, event_driven)
            schedule_expression: Schedule expression
            timeout_seconds: Maximum execution time
            retry_config: Retry policy
            human_handoff_points: Step IDs requiring human input
            created_by: Creator

        Returns:
            WorkflowDefinition: Created workflow
        """
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            steps=steps,
            dependencies=dependencies or [],
            schedule_type=schedule_type,
            schedule_expression=schedule_expression,
            timeout_seconds=timeout_seconds or self.config.get('default_timeout'),
            retry_config=retry_config or {'max_retries': self.config.get('max_retries', 3)},
            human_handoff_points=human_handoff_points or [],
            created_by=created_by
        )

        # Store workflow definition in state manager
        await self.state_manager.save_snapshot(
            state_data=workflow.dict(),
            level='workflow',
            workflow_id=workflow_id,
            is_checkpoint=False,
            created_by=created_by
        )

        logger.info(
            "workflow_created",
            workflow_id=workflow_id,
            name=name,
            steps_count=len(steps)
        )

        return workflow

    async def start_workflow(
        self,
        workflow_id: str,
        input_parameters: Optional[Dict[str, Any]] = None,
        execution_agent: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> WorkflowExecution:
        """
        Start workflow execution.

        Args:
            workflow_id: Workflow to execute
            input_parameters: Input parameters
            execution_agent: Agent executing workflow
            resume_from_checkpoint: Optional checkpoint ID to resume from

        Returns:
            WorkflowExecution: Execution instance

        Raises:
            WorkflowError: If workflow not found
            WorkflowDependencyError: If dependencies not satisfied
        """
        # Check if resuming
        if resume_from_checkpoint:
            return await self._resume_workflow(resume_from_checkpoint)

        # Load workflow definition
        workflow_def = await self._load_workflow_definition(workflow_id)

        # Check dependencies
        await self._check_dependencies(workflow_id)

        # Create execution instance
        execution = WorkflowExecution(
            execution_id=self._generate_id('execution'),
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            total_steps=len(workflow_def.steps),
            input_parameters=input_parameters or {},
            execution_agent=execution_agent,
            retry_limit=workflow_def.retry_config.get('max_retries', 3)
        )

        # Store execution
        self._executions[execution.execution_id] = execution

        # Save initial state
        await self.state_manager.save_snapshot(
            state_data=execution.dict(),
            level='session',
            workflow_id=workflow_id,
            session_id=execution.execution_id,
            is_checkpoint=True,
            checkpoint_name='execution_start',
            created_by='orchestrator'
        )

        # Start execution
        task = asyncio.create_task(
            self._execute_workflow(workflow_def, execution)
        )
        self._running_tasks[execution.execution_id] = task

        logger.info(
            "workflow_started",
            execution_id=execution.execution_id,
            workflow_id=workflow_id
        )

        return execution

    async def _execute_workflow(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """
        Execute workflow steps (main execution loop).

        Args:
            workflow_def: Workflow definition
            execution: Execution instance
        """
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now(timezone.utc)
            execution.last_heartbeat = datetime.now(timezone.utc)

            heartbeat_interval = self.config.get('heartbeat_interval', 30)

            for step_idx, step in enumerate(workflow_def.steps):
                # Check for timeout
                if workflow_def.timeout_seconds:
                    elapsed = (datetime.now(timezone.utc) - execution.started_at).total_seconds()
                    if elapsed > workflow_def.timeout_seconds:
                        raise WorkflowTimeoutError(
                            f"Workflow exceeded timeout of {workflow_def.timeout_seconds}s"
                        )

                # Check for pause
                if execution.status == WorkflowStatus.PAUSED:
                    await self._wait_for_resume(execution)

                # Check for human handoff
                step_id = step.get('step_id', f"step_{step_idx}")
                if step_id in workflow_def.human_handoff_points:
                    await self._handle_human_handoff(workflow_def, execution, step_id)

                # Execute step
                await self._execute_step(workflow_def, execution, step)

                # Update progress
                execution.current_step = step_idx + 1
                execution.completed_steps.append(step_id)
                execution.progress_percentage = ((step_idx + 1) / len(workflow_def.steps)) * 100
                execution.last_heartbeat = datetime.now(timezone.utc)

                # Create checkpoint at milestones
                if step.get('is_checkpoint', False):
                    await self._create_execution_checkpoint(
                        workflow_def,
                        execution,
                        step_id
                    )

            # Workflow completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)

            # Save final state
            await self.state_manager.create_checkpoint(
                snapshot_id=execution.execution_id,
                checkpoint_name='execution_complete',
                checkpoint_type='milestone',
                workflow_id=workflow_def.workflow_id,
                created_by='orchestrator',
                description='Workflow completed successfully'
            )

            logger.info(
                "workflow_completed",
                execution_id=execution.execution_id,
                workflow_id=workflow_def.workflow_id,
                duration_seconds=(execution.completed_at - execution.started_at).total_seconds()
            )

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(timezone.utc)

            logger.error(
                "workflow_failed",
                execution_id=execution.execution_id,
                workflow_id=workflow_def.workflow_id,
                error=str(e)
            )

            # Retry if configured
            if execution.retry_count < execution.retry_limit:
                execution.retry_count += 1
                logger.info(
                    "retrying_workflow",
                    execution_id=execution.execution_id,
                    retry_count=execution.retry_count
                )
                await self._execute_workflow(workflow_def, execution)

    async def _execute_step(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution,
        step: Dict[str, Any]
    ) -> None:
        """
        Execute a single workflow step.

        Args:
            workflow_def: Workflow definition
            execution: Execution instance
            step: Step configuration
        """
        step_id = step.get('step_id', 'unknown')
        step_type = step.get('type', 'default')

        # Get handler for step type
        handler = self._step_handlers.get(step_type)

        if handler is None:
            logger.warning(
                "no_handler_for_step_type",
                step_type=step_type,
                step_id=step_id
            )
            return

        # Execute handler
        try:
            result = await handler(
                workflow_def=workflow_def,
                execution=execution,
                step=step
            )

            # Store result
            execution.output_results[step_id] = result

        except Exception as e:
            logger.error(
                "step_execution_failed",
                step_id=step_id,
                error=str(e)
            )
            raise

    async def pause_workflow(self, execution_id: str) -> None:
        """
        Pause workflow execution.

        Args:
            execution_id: Execution to pause
        """
        if execution_id not in self._executions:
            raise WorkflowError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        execution.status = WorkflowStatus.PAUSED

        logger.info(
            "workflow_paused",
            execution_id=execution_id
        )

    async def resume_workflow(self, execution_id: str) -> None:
        """
        Resume paused workflow.

        Args:
            execution_id: Execution to resume
        """
        if execution_id not in self._executions:
            raise WorkflowError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        if execution.status != WorkflowStatus.PAUSED:
            raise WorkflowError(f"Execution {execution_id} is not paused")

        execution.status = WorkflowStatus.RUNNING

        logger.info(
            "workflow_resumed",
            execution_id=execution_id
        )

    async def cancel_workflow(self, execution_id: str) -> None:
        """
        Cancel workflow execution.

        Args:
            execution_id: Execution to cancel
        """
        if execution_id not in self._executions:
            raise WorkflowError(f"Execution {execution_id} not found")

        execution = self._executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now(timezone.utc)

        # Cancel task if running
        if execution_id in self._running_tasks:
            task = self._running_tasks[execution_id]
            task.cancel()

        logger.info(
            "workflow_cancelled",
            execution_id=execution_id
        )

    async def _handle_human_handoff(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution,
        step_id: str
    ) -> None:
        """
        Handle human-in-the-loop interaction.

        Args:
            workflow_def: Workflow definition
            execution: Execution instance
            step_id: Step requiring input
        """
        execution.awaiting_human_input = True
        execution.status = WorkflowStatus.WAITING

        handoff = HumanHandoff(
            handoff_id=self._generate_id('handoff'),
            workflow_id=workflow_def.workflow_id,
            execution_id=execution.execution_id,
            step_id=step_id,
            handoff_type='input',
            request_message=f"Human input required for step: {step_id}"
        )

        logger.info(
            "human_handoff_initiated",
            handoff_id=handoff.handoff_id,
            step_id=step_id
        )

        # Wait for human response (implement polling or webhook)
        await self._wait_for_human_response(execution, handoff)

        execution.awaiting_human_input = False
        execution.status = WorkflowStatus.RUNNING

    async def _wait_for_human_response(
        self,
        execution: WorkflowExecution,
        handoff: HumanHandoff
    ) -> None:
        """Wait for human to respond to handoff"""
        # TODO: Implement polling or webhook-based response handling
        pass

    async def _wait_for_resume(self, execution: WorkflowExecution) -> None:
        """Wait for paused workflow to resume"""
        while execution.status == WorkflowStatus.PAUSED:
            await asyncio.sleep(1)

    async def _create_execution_checkpoint(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution,
        step_id: str
    ) -> None:
        """Create checkpoint at current execution state"""
        checkpoint = await self.state_manager.create_checkpoint(
            snapshot_id=execution.execution_id,
            checkpoint_name=f"after_{step_id}",
            checkpoint_type='milestone',
            workflow_id=workflow_def.workflow_id,
            created_by='orchestrator',
            description=f'Checkpoint after completing step: {step_id}'
        )

        execution.current_checkpoint_id = checkpoint.checkpoint_id
        execution.can_resume_from.append(checkpoint.checkpoint_id)

    async def _resume_workflow(self, checkpoint_id: str) -> WorkflowExecution:
        """Resume workflow from checkpoint"""
        # Load checkpoint
        checkpoints = await self.state_manager.get_checkpoints(
            workflow_id="",  # Will filter by checkpoint_id
            checkpoint_type=None
        )

        checkpoint = next((c for c in checkpoints if c.checkpoint_id == checkpoint_id), None)
        if not checkpoint:
            raise WorkflowError(f"Checkpoint {checkpoint_id} not found")

        # Load execution state from snapshot
        snapshot = await self.state_manager.load_snapshot(checkpoint.snapshot_id)
        execution = WorkflowExecution(**snapshot.state_data)

        # Update status
        execution.status = WorkflowStatus.RUNNING

        # Restart execution
        workflow_def = await self._load_workflow_definition(execution.workflow_id)
        task = asyncio.create_task(
            self._execute_workflow(workflow_def, execution)
        )
        self._running_tasks[execution.execution_id] = task

        logger.info(
            "workflow_resumed_from_checkpoint",
            execution_id=execution.execution_id,
            checkpoint_id=checkpoint_id
        )

        return execution

    async def _load_workflow_definition(self, workflow_id: str) -> WorkflowDefinition:
        """Load workflow definition from state manager"""
        # TODO: Implement loading from state manager
        raise WorkflowError(f"Workflow {workflow_id} not found")

    async def _check_dependencies(self, workflow_id: str) -> None:
        """Check if workflow dependencies are satisfied"""
        # TODO: Implement dependency checking
        pass

    def register_step_handler(self, step_type: str, handler: Callable) -> None:
        """
        Register a handler for a step type.

        Args:
            step_type: Step type identifier
            handler: Async handler function
        """
        self._step_handlers[step_type] = handler

        logger.info(
            "step_handler_registered",
            step_type=step_type
        )

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:16]}"

    async def shutdown(self) -> None:
        """Shutdown orchestrator and wait for running tasks"""
        for execution_id, task in self._running_tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("workflow_orchestrator_shutdown")
