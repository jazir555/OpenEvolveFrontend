"""
OpenEvolve-CrewAI Adapter

This module provides adapter functions to bridge the existing OpenEvolve workflow
engine with CrewAI's workflow orchestration. It replaces the AGPL-licensed
Hephaestus adapter with MIT-licensed CrewAI.

It allows existing OpenEvolve code to optionally delegate orchestration to CrewAI
while maintaining backward compatibility.

This replaces openevolve_hephaestus_adapter.py with local CrewAI execution.

Usage:
    # Instead of running workflow locally, delegate to CrewAI
    from openevolve_crewai_adapter import delegate_workflow_to_crewai

    workflow_state = delegate_workflow_to_crewai(
        problem_statement="Solve the TSP problem",
        delegator=crewai_delegator,
    )

License: MIT (replaces AGPL Hephaestus)
Author: OpenEvolve Team
Date: 2026-01-21
"""

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

# Import CrewAI zero-error workflow (replaces Hephaestus)
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
    from team_manager import TeamManager
    from gauntlet_manager import GauntletManager
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    TeamManager = None
    GauntletManager = None

logger = logging.getLogger(__name__)


class CrewAIBackendConfig:
    """Configuration for CrewAI backend (replaces HephaestusBackendConfig)"""

    def __init__(
        self,
        enabled: bool = False,
        working_directory: str = ".",
        state_storage_dir: str = "./crewai_states",
        zero_error_config: Optional[ZeroErrorConfig] = None,
        llm_provider: str = "anthropic",
        auto_start: bool = True,
    ):
        self.enabled = enabled
        self.working_directory = working_directory
        self.state_storage_dir = state_storage_dir
        self.zero_error_config = zero_error_config
        self.llm_provider = llm_provider
        self.auto_start = auto_start


# Global delegator instance (lazy initialized)
_crewai_delegator = None
_crewai_config = None


def initialize_crewai_backend(config: CrewAIBackendConfig):
    """
    Initialize the CrewAI backend with the given configuration.

    Call this once at application startup to enable CrewAI delegation.

    Args:
        config: CrewAIBackendConfig with settings
    """
    global _crewai_config, _crewai_delegator

    _crewai_config = config

    if config.enabled:
        logger.info("Initializing CrewAI backend...")
        from openevolve_crewai_delegation import create_openevolve_crewai_delegator

        _crewai_delegator = create_openevolve_crewai_delegator(
            working_directory=config.working_directory,
            state_storage_dir=config.state_storage_dir,
            zero_error_config=config.zero_error_config,
            llm_provider=config.llm_provider,
            auto_start=config.auto_start,
        )
        logger.info("CrewAI backend initialized")
    else:
        logger.info("CrewAI backend disabled, using local execution")


def get_crewai_delegator():
    """
    Get the global CrewAI delegator instance.

    Returns:
        OpenEvolveCrewAIDelegator or None if not enabled
    """
    return _crewai_delegator


def is_crewai_enabled() -> bool:
    """Check if CrewAI backend is enabled"""
    return _crewai_config is not None and _crewai_config.enabled


def delegate_workflow_to_crewai(
    problem_statement: str,
    problem_domain: str = "General",
    complexity_level: str = "Medium (4-7)",
    max_sub_problems: int = 15,
    auto_approve: bool = False,
    monitor: bool = True,
    poll_interval: int = 5,
) -> OpenEvolveWorkflowState:
    """
    Delegate a workflow execution to CrewAI.

    This is the main entry point for delegating OpenEvolve workflows to CrewAI.

    Args:
        problem_statement: The problem to solve
        problem_domain: Domain of the problem
        complexity_level: Expected complexity
        max_sub_problems: Maximum sub-problems to create
        auto_approve: Auto-approve decomposition plan
        monitor: Wait for workflow completion
        poll_interval: Seconds between status checks

    Returns:
        OpenEvolveWorkflowState with results (if monitor=True) or with workflow_id (if monitor=False)
    """
    if not is_crewai_enabled():
        raise RuntimeError("CrewAI backend is not enabled. Call initialize_crewai_backend() first.")

    delegator = get_crewai_delegator()
    if delegator is None:
        raise RuntimeError("CrewAI delegator not initialized")

    async def _delegate():
        # Start workflow
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement=problem_statement,
            problem_domain=problem_domain,
            complexity_level=complexity_level,
            max_sub_problems=max_sub_problems,
            auto_approve_decomposition=auto_approve,
        )

        logger.info(f"Delegated workflow to CrewAI: {workflow_id}")

        # Create workflow state
        workflow_state = OpenEvolveWorkflowState(
            workflow_id=workflow_id,
            workflow_type="openevolve_decomposition",
            problem_statement=problem_statement,
            current_stage="delegated_to_crewai",
            status="running",
            start_time=time.time(),
        )

        # Store reference to delegator for monitoring
        workflow_state.crewai_delegator = delegator
        workflow_state.crewai_workflow_id = workflow_id

        if monitor:
            # Monitor until completion
            execution = await delegator.monitor_workflow(
                workflow_id,
                poll_interval=poll_interval,
            )

            # Update workflow state
            workflow_state.status = execution.status
            workflow_state.current_stage = execution.status
            workflow_state.end_time = time.time()

            if execution.status == "completed":
                workflow_state.completed_stages = [
                    "decomposition",
                    "solving",
                    "critique",
                    "verification",
                    "reassembly",
                    "final_verification",
                ]

        return workflow_state

    # Run async function
    return asyncio.run(_delegate())


def continue_crewai_workflow(workflow_id: str) -> Optional[OpenEvolveWorkflowState]:
    """
    Continue monitoring a previously delegated workflow.

    Args:
        workflow_id: The CrewAI workflow ID

    Returns:
        Updated OpenEvolveWorkflowState or None if not found
    """
    if not is_crewai_enabled():
        return None

    delegator = get_crewai_delegator()
    if delegator is None:
        return None

    async def _continue():
        try:
            execution = await delegator.get_workflow_status(workflow_id)
            if execution is None:
                return None

            workflow_state = OpenEvolveWorkflowState(
                workflow_id=workflow_id,
                workflow_type="openevolve_decomposition",
                problem_statement=execution.description,
                current_stage=execution.status,
                status=execution.status,
            )
            workflow_state.crewai_delegator = delegator
            workflow_state.crewai_workflow_id = workflow_id

            if execution.status in ["completed", "failed"]:
                workflow_state.end_time = time.time()

            return workflow_state

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None

    return asyncio.run(_continue())


def list_crewai_workflows(status: str = "all") -> List[Dict[str, Any]]:
    """
    List all CrewAI workflows.

    Args:
        status: Filter by status ("all", "active", "completed", "failed")

    Returns:
        List of workflow dictionaries
    """
    if not is_crewai_enabled():
        return []

    delegator = get_crewai_delegator()
    if delegator is None:
        return []

    async def _list():
        executions = await delegator.list_workflows(status=status)
        return [
            {
                "id": wf.id,
                "description": wf.description,
                "status": wf.status,
                "total_tasks": wf.total_tasks,
                "done_tasks": wf.done_tasks,
                "failed_tasks": wf.failed_tasks,
                "active_tasks": wf.active_tasks,
                "active_agents": wf.active_agents,
                "created_at": wf.created_at.isoformat(),
            }
            for wf in executions
        ]

    return asyncio.run(_list())


def shutdown_crewai_backend():
    """Shutdown the CrewAI backend"""
    global _crewai_delegator

    if _crewai_delegator is not None:
        logger.info("Shutting down CrewAI backend...")
        _crewai_delegator.shutdown(graceful=True, timeout=10)
        _crewai_delegator = None
        logger.info("CrewAI backend shut down")


# =============================================================================
# ADAPTER FOR EXISTING WORKFLOW ENGINE
# =============================================================================

def should_delegate_to_crewai(workflow_config: Dict[str, Any]) -> bool:
    """
    Determine if a workflow should be delegated to CrewAI based on configuration.

    Args:
        workflow_config: Workflow configuration dictionary

    Returns:
        True if should delegate, False otherwise
    """
    # Check if backend is enabled
    if not is_crewai_enabled():
        return False

    # Check if workflow config specifies backend preference
    backend = workflow_config.get("backend", "local")
    if backend == "crewai":
        return True
    elif backend == "local":
        return False

    # Auto-decide based on problem complexity
    complexity = workflow_config.get("complexity_estimate", 5)
    max_sub_problems = workflow_config.get("max_sub_problems", 10)

    # Delegate if complex or many sub-problems
    return complexity >= 7 or max_sub_problems >= 8


def run_workflow_with_backend_selection(
    problem_statement: str,
    workflow_config: Dict[str, Any],
    team_manager: Optional[TeamManager] = None,
    gauntlet_manager: Optional[GauntletManager] = None,
) -> OpenEvolveWorkflowState:
    """
    Run a workflow using either CrewAI backend or local execution.

    This adapter function allows existing code to optionally use CrewAI
    without changing the call signature.

    Args:
        problem_statement: The problem to solve
        workflow_config: Workflow configuration
        team_manager: TeamManager instance
        gauntlet_manager: GauntletManager instance

    Returns:
        OpenEvolveWorkflowState with results
    """
    # Decide which backend to use
    if should_delegate_to_crewai(workflow_config):
        logger.info("Delegating workflow to CrewAI...")

        # Extract parameters for CrewAI
        params = workflow_config.get("crewai_params", {})

        workflow_state = delegate_workflow_to_crewai(
            problem_statement=problem_statement,
            problem_domain=params.get("problem_domain", "General"),
            complexity_level=params.get("complexity_level", "Medium (4-7)"),
            max_sub_problems=workflow_config.get("max_sub_problems", 15),
            auto_approve=params.get("auto_approve", False),
            monitor=params.get("monitor", True),
            poll_interval=params.get("poll_interval", 5),
        )

        return workflow_state

    else:
        logger.info("Running workflow locally...")

        # Import local workflow engine (avoid circular import)
        from decomposition_engine import DecompositionEngine

        # Run locally using existing workflow engine
        if not OPENEVOLVE_AVAILABLE:
            raise RuntimeError("OpenEvolve components not available for local execution")

        engine = DecompositionEngine(team_manager, gauntlet_manager)
        workflow_state = engine.run_workflow(
            problem_statement=problem_statement,
            config=workflow_config,
        )

        return workflow_state


# =============================================================================
# CONTEXT MANAGER FOR TEMPORARY CREWAI USAGE
# =============================================================================

class CrewAIBackendContext:
    """Context manager for temporarily using CrewAI backend"""

    def __init__(
        self,
        working_directory: str = ".",
        state_storage_dir: str = "./temp_crewai_states",
        llm_provider: str = "anthropic",
    ):
        self.config = CrewAIBackendConfig(
            enabled=True,
            working_directory=working_directory,
            state_storage_dir=state_storage_dir,
            llm_provider=llm_provider,
            auto_start=True,
        )
        self.previous_config = None
        self.previous_delegator = None

    def __enter__(self):
        """Enable CrewAI backend temporarily"""
        global _crewai_config, _crewai_delegator

        # Save previous state
        self.previous_config = _crewai_config
        self.previous_delegator = _crewai_delegator

        # Initialize new backend
        initialize_crewai_backend(self.config)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous state"""
        global _crewai_config, _crewai_delegator

        # Shutdown temporary backend
        shutdown_crewai_backend()

        # Restore previous state
        _crewai_config = self.previous_config
        _crewai_delegator = self.previous_delegator

        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_crewai_health() -> Dict[str, bool]:
    """Get health status of CrewAI backend"""
    if not is_crewai_enabled():
        return {"enabled": False, "overall": False}

    delegator = get_crewai_delegator()
    if delegator is None:
        return {"enabled": True, "initialized": False, "overall": False}

    health = delegator.is_healthy()
    health["enabled"] = True
    health["initialized"] = True
    return health


def get_crewai_metrics(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get metrics for a specific workflow"""
    if not is_crewai_enabled():
        return None

    delegator = get_crewai_delegator()
    if delegator is None:
        return None

    metrics = delegator.get_metrics(workflow_id)
    if metrics is None:
        return None

    return {
        "workflow_id": metrics.workflow_id,
        "total_tasks": metrics.total_tasks,
        "completed_tasks": metrics.completed_tasks,
        "failed_tasks": metrics.failed_tasks,
        "in_progress_tasks": metrics.in_progress_tasks,
        "duration_seconds": metrics.duration_seconds,
        "completion_percentage": metrics.completion_percentage,
        "status": metrics.status,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example 1: Initialize and use CrewAI backend
    print("Example 1: Initialize CrewAI backend")

    config = CrewAIBackendConfig(
        enabled=True,
        working_directory="./workspace/example",
        auto_start=True,
    )

    initialize_crewai_backend(config)

    # Check health
    health = get_crewai_health()
    print(f"Health: {health}")

    # Delegate a workflow
    workflow_state = delegate_workflow_to_crewai(
        problem_statement="Implement a binary search tree",
        monitor=True,
        poll_interval=3,
    )

    print(f"Workflow status: {workflow_state.status}")

    # Shutdown
    shutdown_crewai_backend()

    # Example 2: Use context manager
    print("\nExample 2: Use context manager")

    with CrewAIBackendContext(
        working_directory="./workspace/temp",
    ) as ctx:
        workflow_state = delegate_workflow_to_crewai(
            problem_statement="Solve the N-Queens problem",
            monitor=False,  # Don't wait for completion
        )
        print(f"Started workflow: {workflow_state.workflow_id}")

    print("Context manager exited, backend shut down")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CrewAIBackendConfig",
    "CrewAIBackendContext",
    "initialize_crewai_backend",
    "get_crewai_delegator",
    "is_crewai_enabled",
    "delegate_workflow_to_crewai",
    "continue_crewai_workflow",
    "list_crewai_workflows",
    "shutdown_crewai_backend",
    "should_delegate_to_crewai",
    "run_workflow_with_backend_selection",
    "get_crewai_health",
    "get_crewai_metrics",
]
