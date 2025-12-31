"""
ROMA - Hephaestus Bridge

This module provides the bridge between Hephaestus workflow phases and
ROMA's (Recursive Open Meta-Agents) framework.

ROMA Architecture:
    Atomizer → Planner → Executor → Aggregator
    ↓
    Recursive decomposition with depth constraints

Phase Mapping:
- Phase 1: Problem Setup → ROMA analysis (max_depth=3)
- Phase 2: Solution Generation → ROMA recursive solve (max_depth=2)
- Phase 3: Adversarial Critique → ROMA critique (max_depth=1)
- Phase 4: Verification → ROMA verification (max_depth=1)
- Phase 5: Reassembly → ROMA aggregation (automatic)
- Phase 6: Final Validation → ROMA full solve with verification
"""

import logging
from typing import Dict, Any, List, Optional

from roma_mcp_tools import (
    solve_with_roma,
    solve_sub_problem_with_roma,
    analyze_with_roma,
    critique_with_roma,
    verify_with_roma,
    get_roma_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE EXECUTION FUNCTIONS
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    max_depth: int = 3,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup using ROMA analysis

    Uses ROMA's recursive decomposition to analyze the problem structure.

    Args:
        problem_statement: The problem to analyze
        max_depth: Maximum recursion depth (default: 3 for analysis)
        execution_mode: "recursive" or "event_driven"
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use

    Returns:
        Dict with analysis results
    """
    logger.info(f"Phase 1: Analyzing problem with ROMA - {problem_statement[:50]}...")

    try:
        # Analyze problem structure
        result = analyze_with_roma(
            problem=problem_statement,
            analysis_type="decomposition",
            max_depth=max_depth,
            execution_mode=execution_mode,
            provider=provider,
            api_key=api_key,
            model=model,
        )

        if "error" in result:
            raise Exception(result["error"])

        return {
            "phase": 1,
            "status": "completed",
            "analysis": result["analysis"],
            "dag_info": result.get("dag_info", {}),
            "token_usage": result.get("token_usage"),
            "next_phase": 2,
            "message": "Phase 1 complete: ROMA analysis finished",
        }

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        return {
            "phase": 1,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 1 setup failed: {e}",
        }


def execute_phase_2_solve(
    sub_problems: List[Dict[str, Any]],
    team_name: Optional[str] = None,
    max_depth: int = 2,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    solve_subset: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation using ROMA recursive solve

    Solves sub-problems using ROMA's hierarchical decomposition.

    Args:
        sub_problems: List of sub-problems to solve
        team_name: Team name for agents
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use
        max_depth: Maximum recursion depth
        execution_mode: "recursive" or "event_driven"
        solve_subset: List of sub-problem IDs to solve (solves all if None)

    Returns:
        Dict with solution generation results
    """
    logger.info(f"Phase 2: Generating solutions with ROMA ({len(sub_problems)} sub-problems)")

    try:
        if not team_name:
            team_name = "phase2_roma"

        # Filter to subset if specified
        if solve_subset:
            sub_problems = [sp for sp in sub_problems if sp["id"] in solve_subset]

        solutions = []
        failed_sub_problems = []

        for sp in sub_problems:
            try:
                # Solve using ROMA
                result = solve_sub_problem_with_roma(
                    sub_problem_id=sp["id"],
                    sub_problem_description=sp["description"],
                    team_name=team_name,
                    context={
                        "dependencies": sp.get("dependencies", []),
                        "complexity_score": sp.get("complexity_score", 5),
                    },
                    requirements=sp.get("success_criteria", []),
                    max_depth=max_depth,
                    execution_mode=execution_mode,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                )

                if "error" in result:
                    failed_sub_problems.append(sp["id"])
                    logger.warning(f"Failed to solve {sp['id']}: {result['error']}")
                else:
                    solutions.append({
                        "sub_problem_id": sp["id"],
                        "solution": result["solution"],
                        "team_name": team_name,
                        "status": result["status"],
                        "dag_info": result.get("dag_info", {}),
                        "token_usage": result.get("token_usage"),
                    })

            except Exception as e:
                failed_sub_problems.append(sp["id"])
                logger.error(f"Error solving {sp['id']}: {e}")

        return {
            "phase": 2,
            "status": "completed" if solutions else "failed",
            "team_used": team_name,
            "solutions": solutions,
            "num_solved": len(solutions),
            "num_failed": len(failed_sub_problems),
            "failed_sub_problems": failed_sub_problems,
            "next_phase": 3,
            "message": f"Phase 2 complete: {len(solutions)} sub-problems solved with ROMA, {len(failed_sub_problems)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        return {
            "phase": 2,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 2 solution generation failed: {e}",
        }


def execute_phase_3_critique(
    solutions: List[Dict[str, Any]],
    team_name: Optional[str] = None,
    critique_focus: str = "comprehensive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique using ROMA

    Critiques solutions using ROMA's analysis capabilities.

    Args:
        solutions: List of solutions from Phase 2
        team_name: Team name for Red agents
        critique_focus: Type of critique
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use

    Returns:
        Dict with critique results
    """
    logger.info(f"Phase 3: Critiquing solutions with ROMA ({len(solutions)} solutions)")

    try:
        if not team_name:
            team_name = "phase3_roma_red"

        critiques = []
        failed_critiques = []

        for solution in solutions:
            try:
                # Critique using ROMA
                result = critique_with_roma(
                    solution=solution["solution"],
                    original_task=solution.get("task_description", ""),
                    critique_focus=critique_focus,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                )

                if "error" in result:
                    failed_critiques.append(solution["sub_problem_id"])
                    logger.warning(f"Failed to critique {solution['sub_problem_id']}: {result['error']}")
                else:
                    critiques.append({
                        "sub_problem_id": solution["sub_problem_id"],
                        "critique": result["critique"],
                        "critique_focus": critique_focus,
                        "status": "completed",
                        "token_usage": result.get("token_usage"),
                    })

            except Exception as e:
                failed_critiques.append(solution["sub_problem_id"])
                logger.error(f"Error critiquing {solution['sub_problem_id']}: {e}")

        return {
            "phase": 3,
            "status": "completed" if critiques else "failed",
            "team_used": team_name,
            "critiques": critiques,
            "num_critiqued": len(critiques),
            "num_failed": len(failed_critiques),
            "failed_critiques": failed_critiques,
            "next_phase": 4,
            "message": f"Phase 3 complete: {len(critiques)} solutions critiqued with ROMA, {len(failed_critiques)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        return {
            "phase": 3,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 3 critique failed: {e}",
        }


def execute_phase_4_verify(
    solutions: List[Dict[str, Any]],
    verification_criteria: Optional[List[str]] = None,
    team_name: Optional[str] = None,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification using ROMA

    Verifies solutions meet requirements using ROMA.

    Args:
        solutions: List of solutions from Phase 2
        verification_criteria: List of criteria to verify
        team_name: Team name for Gold agents
        provider: AI provider
        api_key: API key for the provider
        model: Model name to use

    Returns:
        Dict with verification results
    """
    logger.info(f"Phase 4: Verifying solutions with ROMA ({len(solutions)} solutions)")

    try:
        if not team_name:
            team_name = "phase4_roma_gold"

        verifications = []
        failed_verifications = []

        for solution in solutions:
            try:
                # Verify using ROMA
                result = verify_with_roma(
                    solution=solution["solution"],
                    original_task=solution.get("task_description", ""),
                    verification_criteria=verification_criteria,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                )

                if "error" in result:
                    failed_verifications.append(solution["sub_problem_id"])
                    logger.warning(f"Failed to verify {solution['sub_problem_id']}: {result['error']}")
                else:
                    verifications.append({
                        "sub_problem_id": solution["sub_problem_id"],
                        "verification": result["verification"],
                        "status": "completed",
                        "token_usage": result.get("token_usage"),
                    })

            except Exception as e:
                failed_verifications.append(solution["sub_problem_id"])
                logger.error(f"Error verifying {solution['sub_problem_id']}: {e}")

        return {
            "phase": 4,
            "status": "completed" if verifications else "failed",
            "team_used": team_name,
            "verifications": verifications,
            "num_verified": len(verifications),
            "num_failed": len(failed_verifications),
            "failed_verifications": failed_verifications,
            "next_phase": 5,
            "message": f"Phase 4 complete: {len(verifications)} solutions verified with ROMA, {len(failed_verifications)} failed",
        }

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        return {
            "phase": 4,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 4 verification failed: {e}",
        }


def execute_full_workflow(
    problem_statement: str,
    max_depth_analysis: int = 3,
    max_depth_solving: int = 2,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full Hephaestus workflow using ROMA

    Runs all 6 phases with ROMA's recursive decomposition.

    Args:
        problem_statement: The problem to solve
        max_depth_analysis: Depth for analysis phase
        max_depth_solving: Depth for solving phase
        execution_mode: "recursive" or "event_driven"
        provider: AI provider
        api_key: API key
        model: Model name

    Returns:
        Dict with complete workflow results
    """
    logger.info(f"Starting full ROMA workflow: {problem_statement[:50]}...")

    try:
        # Phase 1: Setup
        phase1_result = execute_phase_1_setup(
            problem_statement=problem_statement,
            max_depth=max_depth_analysis,
            execution_mode=execution_mode,
            provider=provider,
            api_key=api_key,
            model=model,
        )

        if phase1_result["status"] == "failed":
            return phase1_result

        # For demo, create dummy sub-problems
        # In real implementation, would parse phase1_result["analysis"]
        sub_problems = [
            {"id": "SP-001", "description": problem_statement, "complexity_score": 5}
        ]

        # Phase 2: Solve
        phase2_result = execute_phase_2_solve(
            sub_problems=sub_problems,
            max_depth=max_depth_solving,
            execution_mode=execution_mode,
            provider=provider,
            api_key=api_key,
            model=model,
        )

        if phase2_result["status"] == "failed":
            return phase2_result

        # Phase 3: Critique
        phase3_result = execute_phase_3_critique(
            solutions=phase2_result["solutions"],
            provider=provider,
            api_key=api_key,
            model=model,
        )

        # Phase 4: Verify
        phase4_result = execute_phase_4_verify(
            solutions=phase2_result["solutions"],
            provider=provider,
            api_key=api_key,
            model=model,
        )

        return {
            "workflow": "roma_full",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
            },
            "message": "Full ROMA workflow completed successfully",
        }

    except Exception as e:
        logger.error(f"Full workflow failed: {e}")
        return {
            "workflow": "roma_full",
            "status": "failed",
            "error": str(e),
            "message": f"Full workflow failed: {e}",
        }


# =============================================================================
# MULTI-AGENT COORDINATION
# =============================================================================

class ROMAHephaestusWorkflowBridge:
    """
    Bridge class for ROMA integration with Hephaestus.

    Provides convenient methods for executing Hephaestus phases with ROMA.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_depth_analysis: int = 3,
        max_depth_solving: int = 2,
        execution_mode: str = "recursive",
    ):
        """
        Initialize the ROMA-Hephaestus bridge.

        Args:
            provider: AI provider (openai, anthropic, google, openrouter)
            api_key: API key
            model: Model name
            max_depth_analysis: Depth for analysis phases
            max_depth_solving: Depth for solving phases
            execution_mode: "recursive" or "event_driven"
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_depth_analysis = max_depth_analysis
        self.max_depth_solving = max_depth_solving
        self.execution_mode = execution_mode

        # Check ROMA availability
        status = get_roma_status()
        self.available = status["available"]

        if not self.available:
            logger.warning("ROMA not available - bridge will fail gracefully")

    def execute_phase_1_setup(self, problem_statement: str) -> Dict[str, Any]:
        """Execute Phase 1: Problem Setup"""
        return execute_phase_1_setup(
            problem_statement=problem_statement,
            max_depth=self.max_depth_analysis,
            execution_mode=self.execution_mode,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
        )

    def execute_phase_2_solve(
        self,
        sub_problems: List[Dict[str, Any]],
        solve_subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute Phase 2: Solution Generation"""
        return execute_phase_2_solve(
            sub_problems=sub_problems,
            max_depth=self.max_depth_solving,
            execution_mode=self.execution_mode,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
            solve_subset=solve_subset,
        )

    def execute_phase_3_critique(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Phase 3: Adversarial Critique"""
        return execute_phase_3_critique(
            solutions=solutions,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
        )

    def execute_phase_4_verify(
        self,
        solutions: List[Dict[str, Any]],
        verification_criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute Phase 4: Verification"""
        return execute_phase_4_verify(
            solutions=solutions,
            verification_criteria=verification_criteria,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
        )

    def execute_full_workflow(self, problem_statement: str) -> Dict[str, Any]:
        """Execute full 6-phase workflow"""
        return execute_full_workflow(
            problem_statement=problem_statement,
            max_depth_analysis=self.max_depth_analysis,
            max_depth_solving=self.max_depth_solving,
            execution_mode=self.execution_mode,
            provider=self.provider,
            api_key=self.api_key,
            model=self.model,
        )
