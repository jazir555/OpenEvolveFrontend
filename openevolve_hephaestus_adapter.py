"""
OpenEvolve-Hephaestus Adapter

This module provides adapter functions to bridge the existing OpenEvolve workflow
engine with the new Hephaestus delegation approach.

It allows existing OpenEvolve code to optionally delegate orchestration to Hephaestus
while maintaining backward compatibility.

Usage:
    # Instead of running workflow locally, delegate to Hephaestus
    from openevolve_hephaestus_adapter import delegate_workflow_to_hephaestus

    workflow_state = delegate_workflow_to_hephaestus(
        problem_statement="Solve the TSP problem",
        delegator=hephaestus_delegator,
    )
"""

import asyncio
import logging
import os
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from workflow_structures import (
    WorkflowState,
    DecompositionPlan,
    SubProblem,
    SolutionAttempt,
    CritiqueReport,
    VerificationReport,
)
from team_manager import TeamManager
from gauntlet_manager import GauntletManager

logger = logging.getLogger(__name__)


class HephaestusBackendConfig:
    """Configuration for Hephaestus backend"""
    def __init__(
        self,
        enabled: bool = False,
        working_directory: str = ".",
        database_path: str = "./openevolve_hephaestus.db",
        qdrant_url: str = "http://localhost:6333",
        mcp_port: int = 8000,
        llm_provider: str = "anthropic",
        auto_start: bool = True,
    ):
        self.enabled = enabled
        self.working_directory = working_directory
        self.database_path = database_path
        self.qdrant_url = qdrant_url
        self.mcp_port = mcp_port
        self.llm_provider = llm_provider
        self.auto_start = auto_start


# Global delegator instance (lazy initialized)
_hephaestus_delegator = None
_hephaestus_config = None


def initialize_hephaestus_backend(config: HephaestusBackendConfig):
    """
    Initialize the Hephaestus backend with the given configuration.

    Call this once at application startup to enable Hephaestus delegation.

    Args:
        config: HephaestusBackendConfig with settings
    """
    global _hephaestus_config, _hephaestus_delegator

    _hephaestus_config = config

    if config.enabled:
        logger.info("Initializing Hephaestus backend...")
        from openevolve_hephaestus_delegation import create_openevolve_delegator

        _hephaestus_delegator = create_openevolve_delegator(
            working_directory=config.working_directory,
            database_path=config.database_path,
            qdrant_url=config.qdrant_url,
            mcp_port=config.mcp_port,
            llm_provider=config.llm_provider,
            auto_start=config.auto_start,
        )
        logger.info("Hephaestus backend initialized")
    else:
        logger.info("Hephaestus backend disabled, using local execution")


def get_hephaestus_delegator():
    """
    Get the global Hephaestus delegator instance.

    Returns:
        OpenEvolveHephaestusDelegator or None if not enabled
    """
    return _hephaestus_delegator


def is_hephaestus_enabled() -> bool:
    """Check if Hephaestus backend is enabled"""
    return _hephaestus_config is not None and _hephaestus_config.enabled


def delegate_workflow_to_hephaestus(
    problem_statement: str,
    problem_domain: str = "General",
    complexity_level: str = "Medium (4-7)",
    max_sub_problems: int = 15,
    auto_approve: bool = False,
    monitor: bool = True,
    poll_interval: int = 5,
) -> WorkflowState:
    """
    Delegate a workflow execution to Hephaestus.

    This is the main entry point for delegating OpenEvolve workflows to Hephaestus.

    Args:
        problem_statement: The problem to solve
        problem_domain: Domain of the problem
        complexity_level: Expected complexity
        max_sub_problems: Maximum sub-problems to create
        auto_approve: Auto-approve decomposition plan
        monitor: Wait for workflow completion
        poll_interval: Seconds between status checks

    Returns:
        WorkflowState with results (if monitor=True) or with workflow_id (if monitor=False)
    """
    if not is_hephaestus_enabled():
        raise RuntimeError("Hephaestus backend is not enabled. Call initialize_hephaestus_backend() first.")

    delegator = get_hephaestus_delegator()
    if delegator is None:
        raise RuntimeError("Hephaestus delegator not initialized")

    async def _delegate():
        # Start workflow
        workflow_id = await delegator.start_decomposition_workflow(
            problem_statement=problem_statement,
            problem_domain=problem_domain,
            complexity_level=complexity_level,
            max_sub_problems=max_sub_problems,
            auto_approve_decomposition=auto_approve,
        )

        logger.info(f"Delegated workflow to Hephaestus: {workflow_id}")

        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type="openevolve_decomposition",
            problem_statement=problem_statement,
            current_stage="delegated_to_hephaestus",
            status="running",
            start_time=time.time(),
        )

        # Store reference to delegator for monitoring
        workflow_state.hephaestus_delegator = delegator
        workflow_state.hephaestus_workflow_id = workflow_id

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


def continue_hephaestus_workflow(workflow_id: str) -> Optional[WorkflowState]:
    """
    Continue monitoring a previously delegated workflow.

    Args:
        workflow_id: The Hephaestus workflow ID

    Returns:
        Updated WorkflowState or None if not found
    """
    if not is_hephaestus_enabled():
        return None

    delegator = get_hephaestus_delegator()
    if delegator is None:
        return None

    async def _continue():
        try:
            execution = await delegator.get_workflow_status(workflow_id)
            if execution is None:
                return None

            workflow_state = WorkflowState(
                workflow_id=workflow_id,
                workflow_type="openevolve_decomposition",
                problem_statement=execution.description,
                current_stage=execution.status,
                status=execution.status,
            )
            workflow_state.hephaestus_delegator = delegator
            workflow_state.hephaestus_workflow_id = workflow_id

            if execution.status in ["completed", "failed"]:
                workflow_state.end_time = time.time()

            return workflow_state

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None

    return asyncio.run(_continue())


def list_hephaestus_workflows(status: str = "all") -> List[Dict[str, Any]]:
    """
    List all Hephaestus workflows.

    Args:
        status: Filter by status ("all", "active", "completed", "failed")

    Returns:
        List of workflow dictionaries
    """
    if not is_hephaestus_enabled():
        return []

    delegator = get_hephaestus_delegator()
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


def shutdown_hephaestus_backend():
    """Shutdown the Hephaestus backend"""
    global _hephaestus_delegator

    if _hephaestus_delegator is not None:
        logger.info("Shutting down Hephaestus backend...")
        _hephaestus_delegator.shutdown(graceful=True, timeout=10)
        _hephaestus_delegator = None
        logger.info("Hephaestus backend shut down")


# =============================================================================
# ADAPTER FOR EXISTING WORKFLOW ENGINE
# =============================================================================

def should_delegate_to_hephaestus(workflow_config: Dict[str, Any]) -> bool:
    """
    Determine if a workflow should be delegated to Hephaestus based on configuration.

    Args:
        workflow_config: Workflow configuration dictionary

    Returns:
        True if should delegate, False otherwise
    """
    # Check if backend is enabled
    if not is_hephaestus_enabled():
        return False

    # Check if workflow config specifies backend preference
    backend = workflow_config.get("backend", "local")
    if backend == "hephaestus":
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
    team_manager: TeamManager,
    gauntlet_manager: GauntletManager,
) -> WorkflowState:
    """
    Run a workflow using either Hephaestus backend or local execution.

    This adapter function allows existing code to optionally use Hephaestus
    without changing the call signature.

    Args:
        problem_statement: The problem to solve
        workflow_config: Workflow configuration
        team_manager: TeamManager instance
        gauntlet_manager: GauntletManager instance

    Returns:
        WorkflowState with results
    """
    # Decide which backend to use
    if should_delegate_to_hephaestus(workflow_config):
        logger.info("Delegating workflow to Hephaestus...")

        # Extract parameters for Hephaestus
        params = workflow_config.get("hephaestus_params", {})

        workflow_state = delegate_workflow_to_hephaestus(
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
        engine = DecompositionEngine(team_manager, gauntlet_manager)
        workflow_state = engine.run_workflow(
            problem_statement=problem_statement,
            config=workflow_config,
        )

        return workflow_state


# =============================================================================
# CONTEXT MANAGER FOR TEMPORARY HEPHAESTUS USAGE
# =============================================================================

class HephaestusBackendContext:
    """Context manager for temporarily using Hephaestus backend"""

    def __init__(
        self,
        working_directory: str = ".",
        database_path: str = "./temp_hephaestus.db",
        qdrant_url: str = "http://localhost:6333",
        mcp_port: int = 8001,  # Different port to avoid conflicts
        llm_provider: str = "anthropic",
    ):
        self.config = HephaestusBackendConfig(
            enabled=True,
            working_directory=working_directory,
            database_path=database_path,
            qdrant_url=qdrant_url,
            mcp_port=mcp_port,
            llm_provider=llm_provider,
            auto_start=True,
        )
        self.previous_config = None
        self.previous_delegator = None

    def __enter__(self):
        """Enable Hephaestus backend temporarily"""
        global _hephaestus_config, _hephaestus_delegator

        # Save previous state
        self.previous_config = _hephaestus_config
        self.previous_delegator = _hephaestus_delegator

        # Initialize new backend
        initialize_hephaestus_backend(self.config)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous state"""
        global _hephaestus_config, _hephaestus_delegator

        # Shutdown temporary backend
        shutdown_hephaestus_backend()

        # Restore previous state
        _hephaestus_config = self.previous_config
        _hephaestus_delegator = self.previous_delegator

        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_hephaestus_health() -> Dict[str, bool]:
    """Get health status of Hephaestus backend"""
    if not is_hephaestus_enabled():
        return {"enabled": False, "overall": False}

    delegator = get_hephaestus_delegator()
    if delegator is None:
        return {"enabled": True, "initialized": False, "overall": False}

    health = delegator.is_healthy()
    health["enabled"] = True
    health["initialized"] = True
    return health


def get_hephaestus_metrics(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get metrics for a specific workflow"""
    if not is_hephaestus_enabled():
        return None

    delegator = get_hephaestus_delegator()
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
    # Example 1: Initialize and use Hephaestus backend
    print("Example 1: Initialize Hephaestus backend")

    config = HephaestusBackendConfig(
        enabled=True,
        working_directory="./workspace/example",
        auto_start=True,
    )

    initialize_hephaestus_backend(config)

    # Check health
    health = get_hephaestus_health()
    print(f"Health: {health}")

    # Delegate a workflow
    workflow_state = delegate_workflow_to_hephaestus(
        problem_statement="Implement a binary search tree",
        monitor=True,
        poll_interval=3,
    )

    print(f"Workflow status: {workflow_state.status}")

    # Shutdown
    shutdown_hephaestus_backend()

    # Example 2: Use context manager
    print("\nExample 2: Use context manager")

    with HephaestusBackendContext(
        working_directory="./workspace/temp",
        mcp_port=8002,
    ) as ctx:
        workflow_state = delegate_workflow_to_hephaestus(
            problem_statement="Solve the N-Queens problem",
            monitor=False,  # Don't wait for completion
        )
        print(f"Started workflow: {workflow_state.workflow_id}")

    print("Context manager exited, backend shut down")
