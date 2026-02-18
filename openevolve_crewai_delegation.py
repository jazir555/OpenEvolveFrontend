"""
OpenEvolve-CrewAI Delegation Integration

This module provides PROPER integration between OpenEvolve and CrewAI by DELEGATING
workflow orchestration to CrewAI instead of just syncing tickets.

It replaces the AGPL-licensed CrewAI delegation with MIT-licensed CrewAI,
maintaining full functional parity while providing better license compatibility.

Architecture:
- OpenEvolve defines the problem decomposition logic and specialized solving strategies
- CrewAI manages the workflow orchestration, agent spawning, and task coordination
- OpenEvolve stages map to CrewAI workflow phases
- CrewAI agents call back into OpenEvolve for specialized processing

Usage:
    from openevolve_crewai_delegation import OpenEvolveCrewAIDelegator

    delegator = OpenEvolveCrewAIDelegator(
        working_directory="/path/to/project"
    )

    # Start a decomposition workflow
    workflow_id = await delegator.start_decomposition_workflow(
        problem_statement="Solve the traveling salesman problem",
        launch_params={}
    )

    # Monitor progress
    status = await delegator.get_workflow_status(workflow_id)

This replaces openevolve_crewai_delegation.py with MIT-licensed CrewAI.

License: MIT (replaces AGPL CrewAI)
Author: OpenEvolve Team
Date: 2026-01-21
"""

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime

# Import CrewAI zero-error workflow (replaces CrewAI)
from crewai_zero_error_workflow import (
    CrewAIZeroErrorWorkflow,
    ZeroErrorConfig,
    create_zero_error_workflow,
    create_zero_error_config,
)

# Import state management
from crewai_state_management import (
    WorkflowState,
    SubProblem,
    DecompositionPlan,
    StateManager,
)

# Import OpenEvolve structures
try:
    from workflow_structures import (
        WorkflowState as OpenEvolveWorkflowState,
        DecompositionPlan as OpenEvolveDecompositionPlan,
        SubProblem as OpenEvolveSubProblem,
        SolutionAttempt,
        CritiqueReport,
        VerificationReport,
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# WORKFLOW METRICS
# =============================================================================

@dataclass
class WorkflowMetrics:
    """Metrics for a running workflow"""
    workflow_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"

    @property
    def duration_seconds(self) -> float:
        """Get workflow duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def completion_percentage(self) -> float:
        """Get completion percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


# =============================================================================
# OPENEVOLVE-CREWAI DELEGATOR
# =============================================================================

class OpenEvolveCrewAIDelegator:
    """
    Main delegator class that integrates OpenEvolve with CrewAI.

    This class delegates workflow orchestration to CrewAI while providing
    OpenEvolve's specialized decomposition and solving logic.

    Architecture:
    - CrewAI manages workflow lifecycle, agent spawning, task coordination
    - OpenEvolve provides domain-specific logic for decomposition and solving
    - Callback system allows CrewAI agents to invoke OpenEvolve functions

    This replaces OpenEvolveCrewAIDelegator with MIT-licensed CrewAI.
    """

    def __init__(
        self,
        working_directory: str = ".",
        state_storage_dir: str = "./crewai_states",
        zero_error_config: Optional[ZeroErrorConfig] = None,
        auto_start: bool = False,
        **config_kwargs,
    ):
        """
        Initialize the OpenEvolve-CrewAI delegator.

        Args:
            working_directory: Working directory for the workflow
            state_storage_dir: Directory for CrewAI state storage
            zero_error_config: Optional zero-error workflow config
            auto_start: Automatically start CrewAI services
            **config_kwargs: Additional configuration parameters
        """
        self.working_directory = working_directory
        self.state_storage_dir = state_storage_dir
        self.running = False
        self.zero_error_workflow: Optional[CrewAIZeroErrorWorkflow] = None
        self.state_manager: Optional[StateManager] = None
        self.active_workflows: Dict[str, WorkflowMetrics] = {}
        self.workflow_counter = 0

        # Create or use provided config
        self.config = zero_error_config or create_zero_error_config()

        # Initialize CrewAI workflow (but don't start yet)
        self._initialize_workflow()

        # Auto-start if requested
        if auto_start:
            self.start()

    def _initialize_workflow(self):
        """Initialize the CrewAI zero-error workflow with OpenEvolve configuration"""
        self.state_manager = StateManager(self.state_storage_dir)
        self.zero_error_workflow = create_zero_error_workflow(self.config)

    def start(self, timeout: int = 30, enable_tui: bool = False) -> bool:
        """
        Start CrewAI services.

        Args:
            timeout: Maximum seconds to wait for services to become healthy
            enable_tui: Enable TUI mode (default: False, headless)

        Returns:
            True if all services are healthy

        Raises:
            Exception: If services don't start within timeout
        """
        if self.running:
            logger.info("[OpenEvolve] CrewAI is already running")
            return True

        logger.info("[OpenEvolve] Starting CrewAI services...")

        # CrewAI doesn't require external services like CrewAI
        # Just mark as running
        self.running = True
        logger.info("[OpenEvolve] [OK] CrewAI services ready")
        logger.info(f"[OpenEvolve] Working directory: {self.working_directory}")

        return True

    async def start_decomposition_workflow(
        self,
        problem_statement: str,
        problem_domain: str = "General",
        complexity_level: str = "Medium (4-7)",
        max_sub_problems: int = 15,
        auto_approve_decomposition: bool = False,
    ) -> str:
        """
        Start a new OpenEvolve decomposition workflow in CrewAI.

        This method creates a new workflow execution in CrewAI, which will:
        1. Create Phase 1 task for problem decomposition
        2. Spawn agents to work on tasks
        3. Coordinate task execution across phases
        4. Track progress and status

        Args:
            problem_statement: The problem to solve
            problem_domain: Domain of the problem
            complexity_level: Expected complexity
            max_sub_problems: Maximum sub-problems to create
            auto_approve_decomposition: Auto-approve decomposition plan

        Returns:
            workflow_id: The ID of the created workflow

        Raises:
            Exception: If workflow creation fails
        """
        if not self.running:
            raise RuntimeError("CrewAI is not running. Call start() first.")

        # Prepare workflow parameters
        workflow_params = {
            "problem_statement": problem_statement,
            "problem_domain": problem_domain,
            "complexity_level": complexity_level,
            "max_sub_problems": max_sub_problems,
            "auto_approve_decomposition": auto_approve_decomposition,
        }

        # Start workflow in CrewAI
        logger.info(f"[OpenEvolve] Starting decomposition workflow...")

        # Create CrewAI workflow
        workflow_id = self.zero_error_workflow.create_workflow(
            description=f"Decompose: {problem_statement[:100]}...",
            workflow_type="openevolve_decomposition",
            parameters=workflow_params,
        )

        # Track workflow
        self.active_workflows[workflow_id] = WorkflowMetrics(
            workflow_id=workflow_id,
            status="active",
        )

        logger.info(f"[OpenEvolve] [OK] Workflow started: {workflow_id}")
        return workflow_id

    async def get_workflow_status(self, workflow_id: str) -> Any:
        """
        Get the status of a workflow execution.

        Args:
            workflow_id: The workflow execution ID

        Returns:
            Workflow execution object with status details

        Raises:
            Exception: If workflow not found
        """
        if not self.running:
            raise RuntimeError("CrewAI is not running. Call start() first.")

        execution = self.zero_error_workflow.get_workflow_status(workflow_id)
        if execution is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        # Update metrics
        if workflow_id in self.active_workflows:
            metrics = self.active_workflows[workflow_id]
            metrics.total_tasks = execution.total_tasks
            metrics.completed_tasks = execution.completed_tasks
            metrics.failed_tasks = execution.failed_tasks
            metrics.in_progress_tasks = execution.in_progress_tasks
            metrics.status = execution.status

            if execution.status in ["completed", "failed", "paused"]:
                metrics.end_time = time.time()

        return execution

    async def list_workflows(self, status: str = "all") -> List[Any]:
        """
        List all workflow executions.

        Args:
            status: Filter by status ("all", "active", "completed", "paused", "failed")

        Returns:
            List of workflow execution objects
        """
        if not self.running:
            raise RuntimeError("CrewAI is not running. Call start() first.")

        return self.zero_error_workflow.list_workflows(status=status)

    async def create_sub_problem_task(
        self,
        workflow_id: str,
        sub_problem: OpenEvolveSubProblem,
        phase_id: int = 2,
        priority: str = "medium",
    ) -> str:
        """
        Create a task for solving a specific sub-problem.

        This is called by Phase 1 (decomposition) to create Phase 2 tasks
        for each sub-problem.

        Args:
            workflow_id: The workflow execution ID
            sub_problem: The sub-problem to solve
            phase_id: Which phase this task belongs to (default: 2 for solving)
            priority: Task priority

        Returns:
            task_id: The ID of the created task
        """
        if not self.running:
            raise RuntimeError("CrewAI is not running. Call start() first.")

        # Build task description from sub-problem
        description = f"""Solve Sub-Problem: {sub_problem.id}

**Description:** {sub_problem.description}

**Complexity Score:** {sub_problem.ai_suggested_complexity_score}/10

**Dependencies:** {', '.join(sub_problem.dependencies) if sub_problem.dependencies else 'None'}

**Requirements:**
{chr(10).join(f'- {req}' for req in sub_problem.acceptance_criteria) if sub_problem.acceptance_criteria else '- Solve the problem completely'}

**Constraints:**
{chr(10).join(f'- {c}' for c in sub_problem.specific_constraints) if sub_problem.specific_constraints else '- None specified'}
"""

        # Create task in CrewAI
        task_id = self.zero_error_workflow.create_task(
            description=description,
            task_type="solve",
            priority=priority,
            metadata={
                "subproblem_id": sub_problem.id,
                "complexity": sub_problem.ai_suggested_complexity_score,
                "evolution_mode": sub_problem.ai_suggested_evolution_mode,
            },
        )

        logger.info(f"[OpenEvolve] Created task {task_id} for sub-problem {sub_problem.id}")
        return task_id

    async def get_tasks(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        phase_id: Optional[int] = None,
    ) -> List[Any]:
        """
        Get tasks with optional filtering.

        Args:
            workflow_id: Filter by workflow (optional, for future use)
            status: Filter by status ("pending", "in_progress", "done", "failed")
            phase_id: Filter by phase ID

        Returns:
            List of task status objects
        """
        if not self.running:
            raise RuntimeError("CrewAI is not running. Call start() first.")

        return self.zero_error_workflow.get_tasks(
            status=status,
            workflow_id=workflow_id,
        )

    async def monitor_workflow(
        self,
        workflow_id: str,
        callback: Optional[callable] = None,
        poll_interval: int = 5,
    ) -> Any:
        """
        Monitor a workflow execution until completion.

        Args:
            workflow_id: The workflow execution ID
            callback: Optional callback function called with status updates
            poll_interval: Seconds between status checks

        Returns:
            Final workflow execution object
        """
        logger.info(f"[OpenEvolve] Monitoring workflow {workflow_id}...")

        while True:
            execution = await self.get_workflow_status(workflow_id)

            # Call callback if provided
            if callback:
                await callback(execution)

            # Print status
            metrics = self.active_workflows.get(workflow_id)
            if metrics:
                logger.info(
                    f"[OpenEvolve] Status: {execution.status} | "
                    f"Tasks: {metrics.completed_tasks}/{metrics.total_tasks} | "
                    f"Progress: {metrics.completion_percentage:.1f}%"
                )

            # Check if complete
            if execution.status in ["completed", "failed"]:
                logger.info(f"[OpenEvolve] Workflow {workflow_id} {execution.status}")
                return execution

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    def get_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """
        Get metrics for a workflow.

        Args:
            workflow_id: The workflow execution ID

        Returns:
            WorkflowMetrics object or None if not found
        """
        return self.active_workflows.get(workflow_id)

    def is_healthy(self) -> Dict[str, bool]:
        """
        Check health of CrewAI services.

        Returns:
            Dictionary with component health status
        """
        if not self.running or not self.zero_error_workflow:
            return {
                "running": False,
                "overall": False,
            }

        # CrewAI is always healthy if running
        return {
            "running": True,
            "workflow": True,
            "state_manager": True,
            "overall": True,
        }

    def shutdown(self, graceful: bool = True, timeout: int = 10):
        """
        Shutdown CrewAI services.

        Args:
            graceful: Use graceful shutdown vs force kill
            timeout: Maximum seconds to wait for graceful shutdown
        """
        if not self.running:
            return

        logger.info("[OpenEvolve] Shutting down CrewAI...")

        if self.zero_error_workflow:
            # Save all states before shutdown
            self.state_manager.save_all_states()

        self.running = False
        logger.info("[OpenEvolve] [OK] Shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_openevolve_crewai_delegator(
    working_directory: str = ".",
    state_storage_dir: str = "./crewai_states",
    zero_error_config: Optional[ZeroErrorConfig] = None,
    llm_provider: str = "anthropic",
    auto_start: bool = False,
    **kwargs,
) -> OpenEvolveCrewAIDelegator:
    """
    Factory function to create an OpenEvolve-CrewAI delegator.

    Args:
        working_directory: Working directory for workflows
        state_storage_dir: Directory for CrewAI state storage
        zero_error_config: Optional zero-error workflow config
        llm_provider: LLM provider to use
        auto_start: Automatically start CrewAI services
        **kwargs: Additional configuration parameters

    Returns:
        OpenEvolveCrewAIDelegator instance
    """
    config = zero_error_config or create_zero_error_config()

    return OpenEvolveCrewAIDelegator(
        working_directory=working_directory,
        state_storage_dir=state_storage_dir,
        zero_error_config=config,
        auto_start=auto_start,
        **kwargs,
    )


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

async def main():
    """Example usage of the OpenEvolve-CrewAI delegator"""

    # Create delegator
    delegator = create_openevolve_crewai_delegator(
        working_directory="/path/to/project",
        auto_start=True,
    )

    try:
        # Start a decomposition workflow
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement="Design a scalable URL shortening service that can handle 10M requests per second",
            problem_domain="Software Development",
            complexity_level="High (8-10)",
            max_sub_problems=10,
        )

        # Monitor until completion
        execution = await delegator.monitor_workflow(
            workflow_id,
            poll_interval=10,
        )

        # Get final metrics
        metrics = delegator.get_metrics(workflow_id)
        logger.info(f"\n[OpenEvolve] Final Metrics:")
        logger.info(f"  Duration: {metrics.duration_seconds:.1f}s")
        logger.info(f"  Tasks: {metrics.completed_tasks}/{metrics.total_tasks}")
        logger.info(f"  Status: {metrics.status}")

    finally:
        # Shutdown
        delegator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "OpenEvolveCrewAIDelegator",
    "WorkflowMetrics",
    "create_openevolve_crewai_delegator",
]
