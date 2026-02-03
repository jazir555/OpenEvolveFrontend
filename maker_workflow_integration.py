"""
MAKER Workflow Integration Module

This module provides updated MAKER integration functions for workflow_engine.py
that use the new OpenEvolve-integrated MAKER implementation.

Replaces the existing _generate_solution_with_maker() function in workflow_engine.py
with an implementation that uses the complete MAKER framework (arXiv:2511.09030).

Usage in workflow_engine.py:
    from maker_workflow_integration import generate_solution_with_maker_v2

    # In generate_solution_for_sub_problem(), replace:
    # maker_result = _generate_solution_with_maker(...)
    # with:
    # maker_result = generate_solution_with_maker_v2(...)
"""

import json
import logging
import os
from typing import Any, Dict, Optional

# Workflow structures
from workflow_structures import (
    SubProblem, Team, WorkflowState, SolutionAttempt
)

# Import the new MAKER integration
from openevolve_maker_integration import (
    MAKERWorkflowIntegrator,
    MAKERWorkflowConfig,
    MAKERMode,
    create_maker_config_from_workflow,
    create_maker_integrator,
    solve_subproblem_with_maker
)

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Adaptive MDAP for maker workflow complexity
try:
    from adaptive_mdap import TaskComplexityClassifier
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    SubProblem = None


# =============================================================================
# UPDATED MAKER GENERATION FUNCTION
# =============================================================================

def generate_solution_with_maker_v2(
    sub_problem: SubProblem,
    team: Team,
    formatted_user_prompt: str,
    system_message: str,
    workflow_state: WorkflowState,
    emit_info: Optional[callable] = None,
    emit_success: Optional[callable] = None,
    emit_warning: Optional[callable] = None
) -> Optional[str]:
    """
    Generate solution for sub-problem using MAKER framework.

    This is the updated version of _generate_solution_with_maker() that uses
    the complete MAKER implementation from arXiv:2511.09030.

    Key improvements over original:
    1. Uses all 4 MAKER algorithms (not just sequential)
    2. Integrates with OpenEvolve client
    3. Supports recursive decomposition (Algorithm 4)
    4. Proper red-flagging and error correction
    5. Returns SolutionAttempt with full metrics

    Args:
        sub_problem: The sub-problem to solve
        team: The team assigned to solve
        formatted_user_prompt: Formatted prompt template
        system_message: System prompt for LLM
        workflow_state: Current workflow state
        emit_info: Optional info logging function
        emit_success: Optional success logging function
        emit_warning: Optional warning logging function

    Returns:
        Solution string if successful, None otherwise
    """
    if emit_info:
        emit_info(f"  - Using MAKER v2 engine for {sub_problem.id}...")

    logger.info(f"Generating solution for {sub_problem.id} using MAKER v2")

    try:
        # Create MAKER integrator
        integrator = create_maker_integrator(workflow_state, team)

        # Solve sub-problem
        solution_attempt = integrator.solve_subproblem(sub_problem, workflow_state)

        # Extract solution content
        if solution_attempt and solution_attempt.content:
            if emit_success:
                emit_success(f"Solution generated for {sub_problem.id} using MAKER v2.")

            # Log metrics
            metrics = solution_attempt.metadata or {}
            logger.info(f"MAKER v2 metrics for {sub_problem.id}:")
            logger.info(f"  - Mode: {metrics.get('maker_mode', 'unknown')}")
            logger.info(f"  - Execution time: {metrics.get('execution_time', 0):.2f}s")
            logger.info(f"  - Total steps: {metrics.get('total_steps', 0)}")
            logger.info(f"  - Total votes: {metrics.get('total_votes', 0)}")
            logger.info(f"  - Red flags: {metrics.get('red_flags', 0)}")

            return solution_attempt.content
        else:
            if emit_warning:
                emit_warning(f"  - MAKER v2 produced no solution for {sub_problem.id}.")
            return None

    except Exception as e:
        logger.error(f"MAKER v2 failed for {sub_problem.id}: {e}", exc_info=True)
        if emit_warning:
            emit_warning(f"  - MAKER v2 failed for {sub_problem.id}: {str(e)}")
        return None


# =============================================================================
# CONFIGURATION BUILDERS
# =============================================================================

def build_maker_config_from_workflow(
    workflow_state: WorkflowState,
    sub_problem: SubProblem
) -> MAKERWorkflowConfig:
    """
    Build MAKER configuration from workflow state and sub-problem.

    This replaces _build_maker_config() from workflow_engine.py with
    a configuration that supports all MAKER modes and features.

    Args:
        workflow_state: Current workflow state
        sub_problem: Sub-problem to solve

    Returns:
        MAKERWorkflowConfig object
    """
    # Extract MAKER mode from maker_config (not metadata)
    maker_config = workflow_state.maker_config if hasattr(workflow_state, 'maker_config') else {}
    maker_mode_str = maker_config.get("maker_mode", "recursive")

    # Determine mode based on sub-problem characteristics
    if sub_problem and sub_problem.estimated_effort and sub_problem.estimated_effort > 20:
        # Large effort tasks benefit from recursive decomposition
        maker_mode_str = "recursive"
    elif sub_problem and hasattr(sub_problem, 'dependencies') and len(sub_problem.dependencies) > 2:
        # Complex dependencies benefit from recursive decomposition
        maker_mode_str = "recursive"

    try:
        maker_mode = MAKERMode(maker_mode_str)
    except ValueError:
        maker_mode = MAKERMode.RECURSIVE

    # Extract other parameters from maker_config
    k_ahead = maker_config.get("maker_k_ahead", 3)
    max_depth = maker_config.get("maker_max_depth", 5)
    enable_red_flagging = maker_config.get("maker_enable_red_flagging", True)
    max_token_length = maker_config.get("maker_max_token_length", 750)

    return MAKERWorkflowConfig(
        mode=maker_mode,
        k_ahead=k_ahead,
        max_depth=max_depth,
        enable_red_flagging=enable_red_flagging,
        max_token_length=max_token_length
    )


def resolve_maker_enabled(
    workflow_state: WorkflowState,
    sub_problem: SubProblem
) -> bool:
    """
    Determine if MAKER should be enabled for this sub-problem.

    This replaces _resolve_maker_enabled() from workflow_engine.py
    with logic that considers the new MAKER capabilities.

    Args:
        workflow_state: Current workflow state
        sub_problem: Sub-problem to check

    Returns:
        True if MAKER should be used, False otherwise
    """
    # Check explicit MAKER flag in workflow state
    if hasattr(workflow_state, 'maker_enabled') and workflow_state.maker_enabled is not None:
        return workflow_state.maker_enabled

    # Check maker_config (not metadata)
    maker_config = workflow_state.maker_config if hasattr(workflow_state, 'maker_config') else {}
    if maker_config.get("maker_enabled"):
        return True

    # Check sub-problem metadata
    if hasattr(sub_problem, 'metadata') and sub_problem.metadata:
        if sub_problem.metadata.get("maker_enabled"):
            return True
        if sub_problem.metadata.get("use_maker"):
            return True

    # Auto-enable for complex sub-problems
    if sub_problem and sub_problem.estimated_effort and sub_problem.estimated_effort > 16:
        return True

    if sub_problem and sub_problem.ai_suggested_complexity_score and sub_problem.ai_suggested_complexity_score > 7:
        return True

    # Default: disabled
    return False


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def generate_solutions_with_maker_batch(
    sub_problems: list[SubProblem],
    team: Team,
    workflow_state: WorkflowState,
    emit_info: Optional[callable] = None,
    emit_success: Optional[callable] = None,
    emit_warning: Optional[callable] = None
) -> Dict[str, Optional[str]]:
    """
    Generate solutions for multiple sub-problems using MAKER.

    This function processes multiple sub-problems in batch, which is more
    efficient than processing them individually.

    Args:
        sub_problems: List of sub-problems to solve
        team: The team assigned to solve
        workflow_state: Current workflow state
        emit_info: Optional info logging function
        emit_success: Optional success logging function
        emit_warning: Optional warning logging function

    Returns:
        Dict mapping sub_problem_id to solution string (or None if failed)
    """
    if emit_info:
        emit_info(f"  - Using MAKER v2 batch processing for {len(sub_problems)} sub-problems...")

    logger.info(f"Batch processing {len(sub_problems)} sub-problems with MAKER v2")

    results = {}

    # Create integrator once (more efficient)
    integrator = create_maker_integrator(workflow_state, team)

    for sub_problem in sub_problems:
        try:
            solution_attempt = integrator.solve_subproblem(sub_problem, workflow_state)

            if solution_attempt and solution_attempt.content:
                results[sub_problem.id] = solution_attempt.content

                if emit_success:
                    emit_success(f"Solution generated for {sub_problem.id} using MAKER v2 batch.")
            else:
                results[sub_problem.id] = None

                if emit_warning:
                    emit_warning(f"  - MAKER v2 batch produced no solution for {sub_problem.id}.")

        except Exception as e:
            logger.error(f"MAKER v2 batch failed for {sub_problem.id}: {e}", exc_info=True)
            results[sub_problem.id] = None

            if emit_warning:
                emit_warning(f"  - MAKER v2 batch failed for {sub_problem.id}: {str(e)}")

    return results


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def migrate_to_maker_v2(
    old_maker_result: Optional[str],
    sub_problem: SubProblem,
    workflow_state: WorkflowState
) -> Optional[str]:
    """
    Migrate old MAKER result to new format if needed.

    This helper function ensures backward compatibility when migrating
    from the old MAKER implementation to the new one.

    Args:
        old_maker_result: Result from old MAKER implementation
        sub_problem: Sub-problem that was solved
        workflow_state: Current workflow state

    Returns:
        Solution string (possibly reprocessed)
    """
    if old_maker_result is None:
        return None

    # If old result exists, try to enhance it with new MAKER metrics
    # For now, just return the old result
    # Future: could re-run with new MAKER for better metrics

    return old_maker_result


# =============================================================================
# STATUS AND INFO
# =============================================================================

def get_maker_integration_info() -> Dict[str, Any]:
    """
    Get information about the MAKER integration.

    Useful for UI display and debugging.

    Returns:
        Dict with integration status and capabilities
    """
    from openevolve_maker_integration import get_maker_status

    base_status = get_maker_status()

    return {
        **base_status,
        "integration_version": "2.0",
        "workflow_integration": "complete",
        "supported_functions": [
            "generate_solution_with_maker_v2",
            "build_maker_config_from_workflow",
            "resolve_maker_enabled",
            "generate_solutions_with_maker_batch"
        ],
        "algorithm_implementations": {
            "algorithm_1": "generate_solution (sequential)",
            "algorithm_2": "do_voting (first-to-ahead-by-k)",
            "algorithm_3": "get_vote (red-flagging)",
            "algorithm_4": "recursive_solve (decomposition)"
        },
        "modes_supported": [
            "sequential",
            "recursive",
            "hybrid"
        ],
        "default_mode": "recursive",
        "paper_reference": "arXiv:2511.09030"
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main functions
    "generate_solution_with_maker_v2",
    "generate_solutions_with_maker_batch",

    # Configuration
    "build_maker_config_from_workflow",
    "resolve_maker_enabled",

    # Migration
    "migrate_to_maker_v2",

    # Info
    "get_maker_integration_info",
]
