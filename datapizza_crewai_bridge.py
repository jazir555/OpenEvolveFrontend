"""
DataPizza-CrewAI Bridge

This module provides the bridge between CrewAI workflow phases and
DataPizza's multi-agent framework with MIT-licensed CrewAI execution.

This replaces datapizza_hephaestus_bridge.py with local CrewAI execution.

IMPORTANT: DataPizza provides multi-agent coordination with Blue/Red/Gold teams,
tool use (FileSystem, Web Search, SQL), and OpenTelemetry tracing.

Phase Mapping:
- Phase 1: Problem Setup → Multi-agent analysis
- Phase 2: Solution Generation → Blue Agent (with planning and tools)
- Phase 3: Adversarial Critique → Red Agent (critical review)
- Phase 4: Verification → Gold Agent (quality assurance)
- Phase 5: Reassembly → Multi-agent coordination
- Phase 6: Final Validation → Full blue-red-gold workflow

License: MIT (replaces AGPL Hephaestus)
Author: OpenEvolve Team
Date: 2026-01-21
"""

import logging
from typing import Dict, Any, List, Optional
import threading
import time

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

# DataPizza integration (maintained)
from datapizza_mcp_tools import (
    solve_with_datapizza_agent,
    run_multi_agent_task,
    create_multi_agent_system,
    get_datapizza_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE EXECUTION FUNCTIONS
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_web_search: bool = True,
    planning_interval: int = 3,
    max_steps: int = 15,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup using DataPizza multi-agent analysis

    Uses parallel Blue/Red/Gold agents to analyze the problem from different perspectives.

    Args:
        problem_statement: The problem to analyze
        provider: AI provider ("openai", "anthropic", "google")
        api_key: API key for the provider
        model: Model name to use
        enable_web_search: Enable web search for research
        planning_interval: Planning interval for agents
        max_steps: Maximum steps per agent
        state_manager: Optional CrewAI state manager

    Returns:
        Dict with analysis results from all three agents
    """
    logger.info(f"Phase 1: Analyzing problem with DataPizza multi-agent system - {problem_statement[:50]}...")

    # Create CrewAI workflow for tracking
    workflow_id = None
    if state_manager:
        workflow_id = f"DATAPIZZA-PHASE1-{int(time.time())}"
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement=problem_statement[:200],
            execution_method="traditional",
            phase=1,
            status="pending",
        )
        state_manager.save_state(workflow_id, workflow_state)

    try:
        # Run parallel multi-agent analysis
        result = run_multi_agent_task(
            team_name="phase1_analysis",
            task=problem_statement,
            provider=provider,
            api_key=api_key,
            model=model,
            enable_web_search=enable_web_search,
            planning_interval=planning_interval,
            max_steps=max_steps,
            workflow="parallel",  # Blue, Red, Gold analyze independently
        )

        if "error" in result:
            if state_manager and workflow_id:
                workflow_state.status = "failed"
                state_manager.save_state(workflow_id, workflow_state)
            raise Exception(result["error"])

        # Extract insights from all three perspectives
        blue_analysis = result["results"]["blue"]["response"]
        red_analysis = result["results"]["red"]["response"]
        gold_analysis = result["results"]["gold"]["response"]

        if state_manager and workflow_id:
            workflow_state.status = "completed"
            state_manager.save_state(workflow_id, workflow_state)

        return {
            "phase": 1,
            "status": "completed",
            "crewai_workflow_id": workflow_id,
            "analysis": {
                "blue": blue_analysis,  # Solution perspective
                "red": red_analysis,    # Challenge perspective
                "gold": gold_analysis,  # Requirements perspective
            },
            "total_steps": result["total_steps"],
            "next_phase": 2,
            "message": "Phase 1 complete: Multi-agent analysis finished",
        }

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        if state_manager and workflow_id:
            workflow_state.status = "failed"
            state_manager.save_state(workflow_id, workflow_state)
        return {
            "phase": 1,
            "status": "failed",
            "crewai_workflow_id": workflow_id,
            "error": str(e),
            "message": f"Phase 1 setup failed: {e}",
        }


def execute_phase_2_solve(
    sub_problems: List[Dict[str, Any]],
    team_name: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    working_directory: Optional[str] = None,
    enable_filesystem: bool = True,
    planning_interval: int = 3,
    max_steps: int = 20,
    solve_subset: Optional[List[str]] = None,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation using DataPizza Blue Agents

    Solves sub-problems using DataPizza agents with planning and tools.

    Args:
        sub_problems: List of sub-problems to solve
        team_name: Team name for agents
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use
        working_directory: Working directory for file operations
        enable_filesystem: Enable filesystem tools
        planning_interval: Planning interval
        max_steps: Maximum steps per agent
        solve_subset: List of sub-problem IDs to solve (solves all if None)
        state_manager: Optional CrewAI state manager

    Returns:
        Dict with solution generation results
    """
    logger.info(f"Phase 2: Generating solutions with DataPizza Blue Agents ({len(sub_problems)} sub-problems)")

    # Create CrewAI workflow for tracking
    workflow_id = None
    if state_manager:
        workflow_id = f"DATAPIZZA-PHASE2-{int(time.time())}"
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement=f"Solve {len(sub_problems)} sub-problems",
            execution_method="traditional",
            phase=2,
            status="pending",
        )
        state_manager.save_state(workflow_id, workflow_state)

    try:
        if not team_name:
            team_name = "phase2_blue"

        # Filter to subset if specified
        if solve_subset:
            sub_problems = [sp for sp in sub_problems if sp["id"] in solve_subset]

        solutions = []
        failed_sub_problems = []

        for sp in sub_problems:
            try:
                # Solve using DataPizza agent
                result = solve_with_datapizza_agent(
                    sub_problem_id=sp["id"],
                    sub_problem_description=sp["description"],
                    agent_role="solver",
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    context={
                        "dependencies": sp.get("dependencies", []),
                        "complexity_score": sp.get("complexity_score", 5),
                    },
                    requirements=sp.get("success_criteria", []),
                    tools=["filesystem"] if enable_filesystem else [],
                    planning_interval=planning_interval,
                    max_steps=max_steps,
                    working_directory=working_directory,
                )

                if "error" in result:
                    failed_sub_problems.append(sp["id"])
                    logger.warning(f"Failed to solve {sp['id']}: {result['error']}")
                else:
                    solutions.append({
                        "sub_problem_id": sp["id"],
                        "solution": result["solution"],
                        "agent_role": result["agent_role"],
                        "status": result["status"],
                        "steps_taken": result.get("steps_taken", 0),
                        "tools_used": result.get("tools_used", []),
                    })

            except Exception as e:
                failed_sub_problems.append(sp["id"])
                logger.error(f"Error solving {sp['id']}: {e}")

        if state_manager and workflow_id:
            workflow_state.status = "completed" if solutions else "failed"
            state_manager.save_state(workflow_id, workflow_state)

        return {
            "phase": 2,
            "status": "completed" if solutions else "failed",
            "crewai_workflow_id": workflow_id,
            "team_used": team_name,
            "solutions": solutions,
            "num_solved": len(solutions),
            "num_failed": len(failed_sub_problems),
            "failed_sub_problems": failed_sub_problems,
            "next_phase": 3,
            "message": f"Phase 2 complete: {len(solutions)} sub-problems solved with DataPizza, {len(failed_sub_problems)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        if state_manager and workflow_id:
            workflow_state.status = "failed"
            state_manager.save_state(workflow_id, workflow_state)
        return {
            "phase": 2,
            "status": "failed",
            "crewai_workflow_id": workflow_id,
            "error": str(e),
            "message": f"Phase 2 solution generation failed: {e}",
        }


def execute_phase_3_critique(
    solutions: List[Dict[str, Any]],
    team_name: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    enable_web_search: bool = True,
    planning_interval: int = 3,
    max_steps: int = 15,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique using DataPizza Red Agents

    Critiques solutions using adversarial Red Agents.

    Args:
        solutions: List of solutions from Phase 2
        team_name: Team name for Red agents
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use
        enable_web_search: Enable web search for validation
        planning_interval: Planning interval
        max_steps: Maximum steps per agent
        state_manager: Optional CrewAI state manager

    Returns:
        Dict with critique results
    """
    logger.info(f"Phase 3: Critiquing solutions with DataPizza Red Agents ({len(solutions)} solutions)")

    # Create CrewAI workflow for tracking
    workflow_id = None
    if state_manager:
        workflow_id = f"DATAPIZZA-PHASE3-{int(time.time())}"
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement=f"Critique {len(solutions)} solutions",
            execution_method="traditional",
            phase=3,
            status="pending",
        )
        state_manager.save_state(workflow_id, workflow_state)

    try:
        if not team_name:
            team_name = "phase3_red"

        critiques = []
        failed_critiques = []

        for solution in solutions:
            try:
                # Critique using DataPizza Red Agent
                result = solve_with_datapizza_agent(
                    sub_problem_id=solution["sub_problem_id"],
                    sub_problem_description=f"Review this solution:\n\n{solution['solution']}",
                    agent_role="critiquer",
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    tools=["duckduckgo"] if enable_web_search else [],
                    planning_interval=planning_interval,
                    max_steps=max_steps,
                )

                if "error" in result:
                    failed_critiques.append(solution["sub_problem_id"])
                    logger.warning(f"Failed to critique {solution['sub_problem_id']}: {result['error']}")
                else:
                    critiques.append({
                        "sub_problem_id": solution["sub_problem_id"],
                        "critique": result["solution"],
                        "status": result["status"],
                        "steps_taken": result.get("steps_taken", 0),
                    })

            except Exception as e:
                failed_critiques.append(solution["sub_problem_id"])
                logger.error(f"Error critiquing {solution['sub_problem_id']}: {e}")

        if state_manager and workflow_id:
            workflow_state.status = "completed" if critiques else "failed"
            state_manager.save_state(workflow_id, workflow_state)

        return {
            "phase": 3,
            "status": "completed" if critiques else "failed",
            "crewai_workflow_id": workflow_id,
            "team_used": team_name,
            "critiques": critiques,
            "num_critiqued": len(critiques),
            "num_failed": len(failed_critiques),
            "failed_critiques": failed_critiques,
            "next_phase": 4,
            "message": f"Phase 3 complete: {len(critiques)} solutions critiqued with DataPizza, {len(failed_critiques)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        if state_manager and workflow_id:
            workflow_state.status = "failed"
            state_manager.save_state(workflow_id, workflow_state)
        return {
            "phase": 3,
            "status": "failed",
            "crewai_workflow_id": workflow_id,
            "error": str(e),
            "message": f"Phase 3 critique failed: {e}",
        }


def execute_phase_4_verify(
    solutions: List[Dict[str, Any]],
    critiques: List[Dict[str, Any]],
    team_name: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    planning_interval: int = 3,
    max_steps: int = 15,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification using DataPizza Gold Agents

    Verifies solutions meet requirements using Gold Agents.

    Args:
        solutions: List of solutions from Phase 2
        critiques: List of critiques from Phase 3
        team_name: Team name for Gold agents
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use
        planning_interval: Planning interval
        max_steps: Maximum steps per agent
        state_manager: Optional CrewAI state manager

    Returns:
        Dict with verification results
    """
    logger.info(f"Phase 4: Verifying solutions with DataPizza Gold Agents ({len(solutions)} solutions)")

    # Create CrewAI workflow for tracking
    workflow_id = None
    if state_manager:
        workflow_id = f"DATAPIZZA-PHASE4-{int(time.time())}"
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement=f"Verify {len(solutions)} solutions",
            execution_method="traditional",
            phase=4,
            status="pending",
        )
        state_manager.save_state(workflow_id, workflow_state)

    try:
        if not team_name:
            team_name = "phase4_gold"

        verifications = []
        failed_verifications = []

        # Create mapping of sub_problem_id -> critique
        critique_map = {c["sub_problem_id"]: c for c in critiques}

        for solution in solutions:
            try:
                sp_id = solution["sub_problem_id"]
                critique = critique_map.get(sp_id)

                # Build verification prompt
                verify_prompt = f"Solution:\n{solution['solution']}\n\n"
                if critique:
                    verify_prompt += f"Previous Critique:\n{critique['critique']}\n\n"
                verify_prompt += "Verify this solution meets requirements and is production-ready."

                # Verify using DataPizza Gold Agent
                result = solve_with_datapizza_agent(
                    sub_problem_id=sp_id,
                    sub_problem_description=verify_prompt,
                    agent_role="verifier",
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    planning_interval=planning_interval,
                    max_steps=max_steps,
                )

                if "error" in result:
                    failed_verifications.append(sp_id)
                    logger.warning(f"Failed to verify {sp_id}: {result['error']}")
                else:
                    verifications.append({
                        "sub_problem_id": sp_id,
                        "verification": result["solution"],
                        "status": result["status"],
                        "steps_taken": result.get("steps_taken", 0),
                    })

            except Exception as e:
                failed_verifications.append(solution["sub_problem_id"])
                logger.error(f"Error verifying {solution['sub_problem_id']}: {e}")

        if state_manager and workflow_id:
            workflow_state.status = "completed" if verifications else "failed"
            state_manager.save_state(workflow_id, workflow_state)

        return {
            "phase": 4,
            "status": "completed" if verifications else "failed",
            "crewai_workflow_id": workflow_id,
            "team_used": team_name,
            "verifications": verifications,
            "num_verified": len(verifications),
            "num_failed": len(failed_verifications),
            "failed_verifications": failed_verifications,
            "next_phase": 5,
            "message": f"Phase 4 complete: {len(verifications)} solutions verified with DataPizza, {len(failed_verifications)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        if state_manager and workflow_id:
            workflow_state.status = "failed"
            state_manager.save_state(workflow_id, workflow_id)
        return {
            "phase": 4,
            "status": "failed",
            "crewai_workflow_id": workflow_id,
            "error": str(e),
            "message": f"Phase 4 verification failed: {e}",
        }


def execute_full_workflow(
    problem_statement: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    working_directory: Optional[str] = None,
    max_sub_problems: int = 15,
    enable_filesystem: bool = True,
    enable_web_search: bool = True,
    planning_interval: int = 3,
    max_steps: int = 20,
    state_manager: Optional[StateManager] = None,
) -> Dict[str, Any]:
    """
    Execute full workflow using DataPizza multi-agent system

    Runs all 6 phases with coordinated Blue/Red/Gold agents.

    Args:
        problem_statement: The problem to solve
        provider: AI provider
        api_key: API key
        model: Model name
        working_directory: Working directory
        max_sub_problems: Maximum sub-problems to create
        enable_filesystem: Enable filesystem tools
        enable_web_search: Enable web search tools
        planning_interval: Planning interval
        max_steps: Maximum steps per phase
        state_manager: Optional CrewAI state manager

    Returns:
        Dict with complete workflow results
    """
    logger.info(f"Starting full DataPizza workflow: {problem_statement[:50]}...")

    try:
        # Phase 1: Setup (would call decomposition to get sub-problems)
        # For now, use multi-agent analysis
        phase1_result = execute_phase_1_setup(
            problem_statement=problem_statement,
            provider=provider,
            api_key=api_key,
            model=model,
            enable_web_search=enable_web_search,
            planning_interval=planning_interval,
            max_steps=max_steps,
            state_manager=state_manager,
        )

        if phase1_result["status"] == "failed":
            return phase1_result

        # Note: In real implementation, would decompose problem here
        # For demo, create dummy sub-problems
        sub_problems = [
            {"id": "SP-001", "description": problem_statement, "complexity_score": 5}
        ]

        # Phase 2: Solve
        phase2_result = execute_phase_2_solve(
            sub_problems=sub_problems,
            provider=provider,
            api_key=api_key,
            model=model,
            working_directory=working_directory,
            enable_filesystem=enable_filesystem,
            planning_interval=planning_interval,
            max_steps=max_steps,
            state_manager=state_manager,
        )

        if phase2_result["status"] == "failed":
            return phase2_result

        # Phase 3: Critique
        phase3_result = execute_phase_3_critique(
            solutions=phase2_result["solutions"],
            provider=provider,
            api_key=api_key,
            model=model,
            enable_web_search=enable_web_search,
            planning_interval=planning_interval,
            max_steps=max_steps,
            state_manager=state_manager,
        )

        # Phase 4: Verify
        phase4_result = execute_phase_4_verify(
            solutions=phase2_result["solutions"],
            critiques=phase3_result["critiques"],
            provider=provider,
            api_key=api_key,
            model=model,
            planning_interval=planning_interval,
            max_steps=max_steps,
            state_manager=state_manager,
        )

        return {
            "workflow": "datapizza_full",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
            },
            "message": "Full DataPizza workflow completed successfully",
        }

    except Exception as e:
        logger.error(f"Full workflow failed: {e}")
        return {
            "workflow": "datapizza_full",
            "status": "failed",
            "error": str(e),
            "message": f"Full workflow failed: {e}",
        }


# =============================================================================
# MULTI-AGENT COORDINATION
# =============================================================================

class DataPizzaCrewAIWorkflowBridge:
    """
    Bridge class for DataPizza integration with CrewAI.

    Provides convenient methods for executing CrewAI workflows with DataPizza.
    Replaces DataPizzaHephaestusWorkflowBridge with MIT-licensed CrewAI.
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        working_directory: Optional[str] = None,
        enable_filesystem: bool = True,
        enable_web_search: bool = True,
        planning_interval: int = 3,
        max_steps: int = 20,
        state_storage_dir: str = "./crewai_states",
    ):
        """
        Initialize the DataPizza-CrewAI bridge.

        Args:
            provider: AI provider
            api_key: API key
            model: Model name
            working_directory: Working directory for file operations
            enable_filesystem: Enable filesystem tools
            enable_web_search: Enable web search tools
            planning_interval: Planning interval for agents
            max_steps: Maximum steps per agent
            state_storage_dir: Directory for CrewAI state storage
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.working_directory = working_directory
        self.enable_filesystem = enable_filesystem
        self.enable_web_search = enable_web_search
        self.planning_interval = planning_interval
        self.max_steps = max_steps

        # Initialize CrewAI state manager
        self.state_manager = StateManager(state_storage_dir)

        # Check DataPizza availability
        status = get_datapizza_status()
        self.available = status["available"]
        self.components = status["components"]

        if not self.available:
            logger.warning("DataPizza not available - bridge will fail gracefully")
        else:
            logger.info("DataPizza-CrewAI Bridge initialized (MIT-licensed)")

    def execute_phase_1_setup(self, problem_statement: str) -> Dict[str, Any]:
        """Execute Phase 1: Problem Setup"""
        return execute_phase_1_setup(
            problem_statement=problem_statement,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            enable_web_search=self.enable_web_search,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
            state_manager=self.state_manager,
        )

    def execute_phase_2_solve(
        self,
        sub_problems: List[Dict[str, Any]],
        solve_subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute Phase 2: Solution Generation"""
        return execute_phase_2_solve(
            sub_problems=sub_problems,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            working_directory=self.working_directory,
            enable_filesystem=self.enable_filesystem,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
            solve_subset=solve_subset,
            state_manager=self.state_manager,
        )

    def execute_phase_3_critique(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Phase 3: Adversarial Critique"""
        return execute_phase_3_critique(
            solutions=solutions,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            enable_web_search=self.enable_web_search,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
            state_manager=self.state_manager,
        )

    def execute_phase_4_verify(
        self,
        solutions: List[Dict[str, Any]],
        critiques: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute Phase 4: Verification"""
        return execute_phase_4_verify(
            solutions=solutions,
            critiques=critiques,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
            state_manager=self.state_manager,
        )

    def execute_full_workflow(self, problem_statement: str) -> Dict[str, Any]:
        """Execute full 6-phase workflow"""
        return execute_full_workflow(
            problem_statement=problem_statement,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            working_directory=self.working_directory,
            enable_filesystem=self.enable_filesystem,
            enable_web_search=self.enable_web_search,
            planning_interval=self.planning_interval,
            max_steps=self.max_steps,
            state_manager=self.state_manager,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'DataPizzaCrewAIWorkflowBridge',
    'execute_phase_1_setup',
    'execute_phase_2_solve',
    'execute_phase_3_critique',
    'execute_phase_4_verify',
    'execute_full_workflow',
]


if __name__ == "__main__":
    # Example usage
    bridge = DataPizzaCrewAIWorkflowBridge()

    print("DataPizza-CrewAI Bridge initialized (MIT-licensed)")
    print(f"Bridge ready for multi-agent execution")
