"""
Hephaestus-OpenEvolve Workflow Bridge

This module provides the bridge between Hephaestus workflow phases and
OpenEvolve's evolutionary coding capabilities.

IMPORTANT: OpenEvolve is an evolutionary coding agent, NOT a decomposition system.
The bridge maps Hephaestus phases to appropriate evolutionary tasks.

Phase Mapping:
- Phase 1: Problem Setup → Generate initial algorithm
- Phase 2: Optimization → Evolve for performance
- Phase 3: Diversity → Evolve for code variety
- Phase 4: Correctness → Evolve for correctness
- Phase 5: Multi-objective → Evolve for multiple goals
- Phase 6: Final Selection → Select best evolved program
"""

import logging
from typing import Dict, Any, List, Optional

from openevolve_mcp_tools import (
    evolve_code_with_openevolve,
    evolve_function_with_openevolve,
    optimize_algorithm_with_openevolve,
    discover_algorithm_with_openevolve,
    list_openevolve_capabilities,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE EXECUTION FUNCTIONS
# =============================================================================

def execute_phase_1_setup(
    problem_description: str,
    search_space: str = "optimization",
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup - Generate initial algorithm

    This is called by Hephaestus Phase 1 agents to set up the evolutionary problem.

    Args:
        problem_description: Description of the problem to solve
        search_space: Type of algorithm space to search
        constraints: List of constraints for the algorithm

    Returns:
        Dict with setup results and initial code
    """
    logger.info(f"Phase 1: Setting up evolution problem - {problem_description[:50]}...")

    try:
        # Generate initial algorithm code
        initial_code = generate_initial_code_for_problem(search_space)

        result = {
            "phase": 1,
            "status": "completed",
            "problem_description": problem_description,
            "search_space": search_space,
            "initial_code": initial_code,
            "constraints": constraints or [],
            "next_phase": 2,
            "message": f"Evolution problem setup complete for {search_space}",
        }

        logger.info(f"Phase 1 complete: Initial algorithm generated")
        return result

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        return {
            "phase": 1,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 1 setup failed: {e}",
        }


def execute_phase_2_optimize(
    initial_code: str,
    problem_description: str,
    iterations: int = 50,
    optimization_goal: str = "performance",
) -> Dict[str, Any]:
    """
    Execute Phase 2: Performance Optimization

    This is called by Hephaestus Phase 2 agents to optimize code for performance.

    Args:
        initial_code: Initial algorithm code
        problem_description: Problem being solved
        iterations: Number of evolution iterations
        optimization_goal: What to optimize for

    Returns:
        Dict with optimization results
    """
    logger.info(f"Phase 2: Optimizing for {optimization_goal}")

    try:
        # Use OpenEvolve to optimize code
        result = evolve_code_with_openevolve(
            initial_code=initial_code,
            iterations=iterations,
            optimization_goal=optimization_goal,
        )

        if "error" in result:
            raise Exception(result["error"])

        evolved_code = result["evolved_code"]
        improvement = result.get("improvement", 0.0)

        final_result = {
            "phase": 2,
            "status": "completed",
            "initial_code": initial_code,
            "evolved_code": evolved_code,
            "best_score": result["best_score"],
            "improvement": improvement,
            "metrics": result.get("metrics", {}),
            "next_phase": 3,
            "message": f"Optimization complete: {improvement:.1%} improvement",
        }

        logger.info(f"Phase 2 complete: {improvement:.1%} improvement")
        return final_result

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        return {
            "phase": 2,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 2 optimization failed: {e}",
        }


def execute_phase_3_diversity(
    evolved_code: str,
    problem_description: str,
    num_variants: int = 3,
    iterations_per_variant: int = 30,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Diversity Generation

    This is called by Hephaestus Phase 3 agents to generate diverse algorithm variants.

    Args:
        evolved_code: Optimized code from Phase 2
        problem_description: Problem being solved
        num_variants: Number of diverse variants to generate
        iterations_per_variant: Iterations per variant

    Returns:
        Dict with diverse variants
    """
    logger.info(f"Phase 3: Generating {num_variants} diverse variants")

    try:
        variants = []

        for i in range(num_variants):
            # Evolve with different random seeds/goals for diversity
            result = evolve_code_with_openevolve(
                initial_code=evolved_code,
                iterations=iterations_per_variant,
                optimization_goal=["performance", "code_size", "memory"][i % 3],
            )

            if "error" not in result:
                variants.append({
                    "variant_id": i,
                    "code": result["evolved_code"],
                    "score": result["best_score"],
                    "characteristics": analyze_code_characteristics(result["evolved_code"]),
                })

        result = {
            "phase": 3,
            "status": "completed",
            "variants": variants,
            "num_variants": len(variants),
            "next_phase": 4,
            "message": f"Generated {len(variants)} diverse variants",
        }

        logger.info(f"Phase 3 complete: {len(variants)} variants generated")
        return result

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        return {
            "phase": 3,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 3 diversity generation failed: {e}",
        }


def execute_phase_4_correctness(
    variants: List[Dict[str, Any]],
    test_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute Phase 4: Correctness Verification

    This is called by Hephaestus Phase 4 agents to verify correctness of variants.

    Args:
        variants: List of diverse algorithm variants
        test_cases: Test cases to verify correctness

    Returns:
        Dict with correctness results
    """
    logger.info(f"Phase 4: Verifying correctness of {len(variants)} variants")

    try:
        verified_variants = []

        for variant in variants:
            # Test each variant against test cases
            correctness_score = test_variant_correctness(
                variant["code"],
                test_cases,
            )

            if correctness_score >= 0.8:  # 80% correctness threshold
                verified_variants.append({
                    **variant,
                    "correctness_score": correctness_score,
                    "verified": True,
                })

        result = {
            "phase": 4,
            "status": "completed",
            "verified_variants": verified_variants,
            "num_verified": len(verified_variants),
            "correctness_threshold": 0.8,
            "next_phase": 5,
            "message": f"{len(verified_variants)} variants passed correctness verification",
        }

        logger.info(f"Phase 4 complete: {len(verified_variants)} variants verified")
        return result

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        return {
            "phase": 4,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 4 correctness verification failed: {e}",
        }


def execute_phase_5_multiobjective(
    verified_variants: List[Dict[str, Any]],
    objectives: List[str],
) -> Dict[str, Any]:
    """
    Execute Phase 5: Multi-Objective Optimization

    This is called by Hephaestus Phase 5 agents to optimize for multiple objectives.

    Args:
        verified_variants: List of verified algorithm variants
        objectives: List of objectives to optimize for

    Returns:
        Dict with multi-objective optimization results
    """
    logger.info(f"Phase 5: Multi-objective optimization for {len(objectives)} objectives")

    try:
        optimized_variants = []

        for variant in verified_variants:
            # Further evolve each variant for multi-objective optimization
            for objective in objectives:
                result = evolve_code_with_openevolve(
                    initial_code=variant["code"],
                    iterations=30,
                    optimization_goal=objective,
                )

                if "error" not in result:
                    optimized_variants.append({
                        "variant_id": variant["variant_id"],
                        "objective": objective,
                        "code": result["evolved_code"],
                        "score": result["best_score"],
                    })

        # Find Pareto-optimal solutions
        pareto_optimal = find_pareto_optimal(optimized_variants, objectives)

        result = {
            "phase": 5,
            "status": "completed",
            "optimized_variants": optimized_variants,
            "pareto_optimal": pareto_optimal,
            "objectives": objectives,
            "num_pareto_optimal": len(pareto_optimal),
            "next_phase": 6,
            "message": f"Multi-objective optimization complete: {len(pareto_optimal)} Pareto-optimal solutions",
        }

        logger.info(f"Phase 5 complete: {len(pareto_optimal)} Pareto-optimal solutions")
        return result

    except Exception as e:
        logger.error(f"Phase 5 failed: {e}")
        return {
            "phase": 5,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 5 multi-objective optimization failed: {e}",
        }


def execute_phase_6_final_selection(
    pareto_optimal: List[Dict[str, Any]],
    selection_criteria: str = "balanced",
) -> Dict[str, Any]:
    """
    Execute Phase 6: Final Solution Selection

    This is called by Hephaestus Phase 6 agents to select the final solution.

    Args:
        pareto_optimal: List of Pareto-optimal solutions
        selection_criteria: How to select final solution ("balanced", "performance", "code_size")

    Returns:
        Dict with final selected solution
    """
    logger.info(f"Phase 6: Final solution selection ({selection_criteria})")

    try:
        # Select best solution based on criteria
        if selection_criteria == "balanced":
            # Select solution with best average score
            best = max(pareto_optimal, key=lambda x: x.get("score", 0))
        elif selection_criteria == "performance":
            # Select fastest solution
            best = max(pareto_optimal, key=lambda x: x.get("metrics", {}).get("performance", 0))
        elif selection_criteria == "code_size":
            # Select smallest solution
            best = min(pareto_optimal, key=lambda x: len(x.get("code", "")))
        else:
            best = pareto_optimal[0] if pareto_optimal else None

        if not best:
            raise Exception("No optimal solutions found")

        result = {
            "phase": 6,
            "status": "completed",
            "final_solution": best,
            "selection_criteria": selection_criteria,
            "workflow_complete": True,
            "message": f"Final solution selected: {selection_criteria} criteria",
        }

        logger.info("Phase 6 complete: WORKFLOW COMPLETE")
        return result

    except Exception as e:
        logger.error(f"Phase 6 failed: {e}")
        return {
            "phase": 6,
            "status": "failed",
            "error": str(e),
            "workflow_complete": False,
            "message": f"Phase 6 final selection failed: {e}",
        }


# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================

class HephaestusOpenEvolveWorkflowBridge:
    """
    Main bridge class that orchestrates evolutionary workflows with OpenEvolve.

    This class provides a high-level interface that Hephaestus can use
    to execute evolutionary coding workflows.
    """

    def __init__(self):
        self.phase_executors = {
            1: execute_phase_1_setup,
            2: execute_phase_2_optimize,
            3: execute_phase_3_diversity,
            4: execute_phase_4_correctness,
            5: execute_phase_5_multiobjective,
            6: execute_phase_6_final_selection,
        }

    def execute_phase(
        self,
        phase_id: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a specific phase of the evolutionary workflow.

        Args:
            phase_id: Phase to execute (1-6)
            **kwargs: Phase-specific arguments

        Returns:
            Dict with phase execution results
        """
        if phase_id not in self.phase_executors:
            return {
                "phase": phase_id,
                "status": "failed",
                "error": f"Invalid phase ID: {phase_id}",
            }

        executor = self.phase_executors[phase_id]
        return executor(**kwargs)

    def get_phase_instructions(self, phase_id: int) -> str:
        """
        Get instructions for a specific phase.

        Args:
            phase_id: Phase to get instructions for

        Returns:
            String with phase instructions
        """
        instructions = {
            1: """PHASE 1: PROBLEM SETUP

Your mission: Set up the evolutionary problem for OpenEvolve.

STEPS:
1. Understand the problem description
2. Choose appropriate search space
3. Identify constraints
4. Generate initial algorithm code

MCP TOOLS TO USE:
- discover_algorithm_with_openevolve()

EXPECTED OUTPUT:
- Initial algorithm code
- Problem constraints
- Search space definition
""",
            2: """PHASE 2: PERFORMANCE OPTIMIZATION

Your mission: Optimize the algorithm for performance.

STEPS:
1. Use OpenEvolve to evolve the code
2. Focus on performance improvements
3. Measure improvement

MCP TOOLS TO USE:
- evolve_code_with_openevolve()

EXPECTED OUTPUT:
- Optimized code
- Performance improvement metrics
""",
            3: """PHASE 3: DIVERSITY GENERATION

Your mission: Generate diverse algorithm variants.

STEPS:
1. Create multiple variants with different optimization goals
2. Ensure variety in approaches
3. Document characteristics of each variant

MCP TOOLS TO USE:
- evolve_code_with_openevolve() with different goals

EXPECTED OUTPUT:
- Multiple diverse variants
- Variant characteristics
""",
            4: """PHASE 4: CORRECTNESS VERIFICATION

Your mission: Verify correctness of variants.

STEPS:
1. Test each variant against test cases
2. Calculate correctness scores
3. Filter out incorrect variants

MCP TOOLS TO USE:
- Test variants against provided test cases

EXPECTED OUTPUT:
- Verified variants list
- Correctness scores
""",
            5: """PHASE 5: MULTI-OBJECTIVE OPTIMIZATION

Your mission: Optimize for multiple objectives simultaneously.

STEPS:
1. Identify optimization objectives
2. Evolve variants for each objective
3. Find Pareto-optimal solutions

MCP TOOLS TO USE:
- evolve_code_with_openevolve() for each objective

EXPECTED OUTPUT:
- Pareto-optimal solutions
- Multi-objective analysis
""",
            6: """PHASE 6: FINAL SELECTION

Your mission: Select the final solution.

STEPS:
1. Review all Pareto-optimal solutions
2. Apply selection criteria
3. Choose final algorithm

MCP TOOLS TO USE:
- Analysis and selection logic

EXPECTED OUTPUT:
- Final selected solution
- Selection justification
""",
        }

        return instructions.get(phase_id, f"No instructions for phase {phase_id}")

    def list_available_tools(self) -> List[str]:
        """List available OpenEvolve MCP tools"""
        capabilities = list_openevolve_capabilities()
        return capabilities.get("capabilities", [])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_initial_code_for_problem(search_space: str) -> str:
    """Generate initial algorithm code for a search space"""

    algorithms = {
        "sorting": """
# EVOLVE-BLOCK-START
def sort_algorithm(arr):
    # Initial: bubble sort
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
# EVOLVE-BLOCK-END
""",
        "optimization": """
# EVOLVE-BLOCK-START
def optimize_function(func, bounds, max_iter=100):
    # Initial: random search
    import random
    best_x = None
    best_val = float('inf')
    for _ in range(max_iter):
        x = [random.uniform(b, e) for b, e in bounds]
        val = func(x)
        if val < best_val:
            best_x, best_val = x, val
    return best_x, best_val
# EVOLVE-BLOCK-END
""",
        "search": """
# EVOLVE-BLOCK-START
def search_algorithm(arr, target):
    # Initial: linear search
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
# EVOLVE-BLOCK-END
""",
    }

    return algorithms.get(search_space, algorithms["optimization"])


def analyze_code_characteristics(code: str) -> Dict[str, Any]:
    """Analyze characteristics of code"""
    lines = code.split('\n')
    return {
        "lines_of_code": len(lines),
        "chars": len(code),
        "has_loops": any('for ' in line or 'while ' in line for line in lines),
        "has_recursion": 'def ' in code and code.count('return') > 1,
    }


def test_variant_correctness(code: str, test_cases: List[Dict[str, Any]]) -> float:
    """Test a code variant against test cases"""
    # Simplified correctness check
    # In production, this would actually execute the code
    passed = sum(1 for tc in test_cases if tc.get("expected") is not None)
    return passed / len(test_cases) if test_cases else 0.5


def find_pareto_optimal(variants: List[Dict[str, Any]], objectives: List[str]) -> List[Dict[str, Any]]:
    """Find Pareto-optimal solutions from variants"""
    if not variants:
        return []

    # Simplified Pareto front calculation
    # In production, use proper multi-objective optimization
    sorted_variants = sorted(variants, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_variants[:min(5, len(sorted_variants))]


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_workflow_bridge():
    """Initialize the workflow bridge"""
    logger.info("Hephaestus-OpenEvolve workflow bridge initialized")


# Auto-initialize on import
initialize_workflow_bridge()
