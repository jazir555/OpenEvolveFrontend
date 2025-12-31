"""
Decomposition Workflow - Hephaestus Bridge

This module provides the bridge between Hephaestus workflow phases and
the Sovereign-Grade Decomposition Workflow (teams, gauntlets, problem decomposition).

IMPORTANT: The Decomposition Workflow is a separate system (not OpenEvolve) that provides:
- Problem decomposition into sub-problems
- Team-based solving (Blue, Red, Gold teams)
- Gauntlet critiques and verification
- Multi-stage workflow (Stages 0-6)

Phase Mapping:
- Phase 1: Problem Setup → Stage 0 (Content Analysis) + Stage 1 (Decomposition)
- Phase 2: Solution Generation → Stage 3A (Blue Team Solving)
- Phase 3: Adversarial Critique → Stage 3B (Red Team Gauntlet)
- Phase 4: Verification → Stage 3C (Gold Team Gauntlet)
- Phase 5: Reassembly → Stage 4 (Configurable Reassembly)
- Phase 6: Final Validation → Stage 5 (Final Verification) + Stage 6 (Knowledge Extraction)
"""

import logging
from typing import Dict, Any, List, Optional
import json

from decomposition_mcp_tools import (
    analyze_problem_for_decomposition,
    decompose_problem_into_sub_problems,
    create_decomposition_plan,
    solve_sub_problem_with_team,
    critique_solution_with_gauntlet,
    verify_solution_with_gauntlet,
    list_available_teams,
    list_available_gauntlets,
    get_decomposition_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE EXECUTION FUNCTIONS
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    max_sub_problems: int = 15,
    decomposition_strategy: str = "semantic",
    use_evolution: bool = True,
    evolution_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup - Analyze and decompose the problem

    This is called by Hephaestus Phase 1 agents to set up the decomposition workflow.
    Maps to Stage 0 (Content Analysis) and Stage 1 (AI-Assisted Decomposition).

    Args:
        problem_statement: The problem to solve
        problem_type: Type of problem (optimization, design, research, etc.)
        domain: Problem domain (software, mathematics, system design, etc.)
        max_sub_problems: Maximum number of sub-problems to create
        decomposition_strategy: Strategy for decomposition ("semantic", "hierarchical", "flow")
        use_evolution: Whether to use OpenEvolve for evolutionary processing
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with setup results and decomposition plan
    """
    logger.info(f"Phase 1: Setting up decomposition workflow - {problem_statement[:50]}...")

    try:
        # Stage 0: Content Analysis (with OpenEvolve if enabled)
        logger.info(f"  Stage 0: Analyzing problem content (evolution={use_evolution})...")
        analysis = analyze_problem_for_decomposition(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )

        if "error" in analysis:
            raise Exception(analysis["error"])

        # Stage 1: AI-Assisted Decomposition (with OpenEvolve if enabled)
        logger.info(f"  Stage 1: Decomposing problem into sub-problems (evolution={use_evolution})...")
        decomposition = decompose_problem_into_sub_problems(
            problem_statement=problem_statement,
            analysis=analysis,
            max_sub_problems=max_sub_problems,
            decomposition_strategy=decomposition_strategy,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )

        if "error" in decomposition:
            raise Exception(decomposition["error"])

        # Create complete decomposition plan
        plan = create_decomposition_plan(
            problem_statement=problem_statement,
            sub_problems=decomposition["sub_problems"],
            dependencies=decomposition["dependencies"],
        )

        if "error" in plan:
            raise Exception(plan["error"])

        result = {
            "phase": 1,
            "status": "completed",
            "problem_statement": problem_statement,
            "analysis": analysis,
            "decomposition": decomposition,
            "plan": plan,
            "next_phase": 2,
            "message": f"Phase 1 complete: Decomposed into {len(decomposition['sub_problems'])} sub-problems",
        }

        logger.info(f"Phase 1 complete: {len(decomposition['sub_problems'])} sub-problems created")
        return result

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        return {
            "phase": 1,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 1 setup failed: {e}",
        }


def execute_phase_2_solve(
    decomposition_plan: Dict[str, Any],
    team_name: Optional[str] = None,
    solve_subset: Optional[List[str]] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 100,
    # Execution method selection (SOVEREIGN CHOICE)
    execution_method: str = "traditional",  # "traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker", "auto"
    use_claudiomiro: bool = False,
    use_datapizza: bool = False,
    use_roma: bool = False,
    use_hybrid: bool = False,
    # Claudiomiro parameters
    claudiomiro_provider: str = "claude",
    claudiomiro_backend: Optional[str] = None,
    claudiomiro_frontend: Optional[str] = None,
    working_dir: str = ".",
    max_cycles: int = 20,
    # DataPizza parameters
    datapizza_provider: str = "openai",
    datapizza_api_key: Optional[str] = None,
    datapizza_model: Optional[str] = None,
    datapizza_tools: Optional[List[str]] = None,
    datapizza_planning_interval: int = 3,
    datapizza_max_steps: int = 20,
    # ROMA parameters
    roma_max_depth: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    # ROMA-Decomposition Hybrid parameters
    hybrid_max_depth_analysis: int = 3,
    hybrid_max_depth_solving: int = 2,
    hybrid_execution_mode: str = "recursive",
    hybrid_provider: Optional[str] = None,
    hybrid_api_key: Optional[str] = None,
    hybrid_model: Optional[str] = None,
    hybrid_enable_gauntlets: bool = True,
    hybrid_enable_evolution: bool = True,
    hybrid_evolution_iterations: int = 50,
    # ROMA-MDAP-MAKER parameters
    use_roma_mdap_maker: bool = False,
    roma_mdap_maker_max_depth: int = 2,
    roma_mdap_maker_k_ahead: int = 3,
    roma_mdap_maker_enable_red_flagging: bool = True,
    roma_mdap_maker_max_samples: int = 100,
    roma_mdap_maker_enable_adaptive_k: bool = True,
    roma_mdap_maker_provider: str = "openai",
    roma_mdap_maker_api_key: Optional[str] = None,
    roma_mdap_maker_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation - Solve sub-problems with Blue Teams

    This is called by Hephaestus Phase 2 agents to generate solutions.
    Maps to Stage 3A (Blue Team Solution Generation).

    **SOVEREIGN CHOICE**: Seven execution methods available:
    1. "traditional" - AI-assisted decomposition with LLM prompts (existing method)
    2. "claudiomiro" - Autonomous development with Claudiomiro CLI
    3. "datapizza" - Multi-agent problem solving with DataPizza
    4. "roma" - Recursive meta-agent decomposition with ROMA
    5. "hybrid" - ROMA automatic decomposition + Decomposition Workflow teams
    6. "roma_mdap_maker" - ROMA + MAKER zero-error voting (NEW)
    7. "auto" - Automatically choose based on sub-problem characteristics

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        team_name: Specific Blue Team to use (auto-selected if None)
        solve_subset: List of sub-problem IDs to solve (solves all if None)
        use_evolution: Whether to use OpenEvolve for evolutionary solution generation
        evolution_iterations: Number of evolution iterations
        execution_method: How to execute (all 7 methods available)
        use_claudiomiro: Explicitly enable/disable Claudiomiro
        use_datapizza: Explicitly enable/disable DataPizza
        use_roma: Explicitly enable/disable ROMA
        use_hybrid: Explicitly enable/disable ROMA-Decomposition hybrid
        claudiomiro_provider: AI provider for Claudiomiro
        claudiomiro_backend: Backend directory for multi-repo projects
        claudiomiro_frontend: Frontend directory for multi-repo projects
        working_dir: Working directory for Claudiomiro execution
        max_cycles: Maximum Claudiomiro execution cycles
        datapizza_provider: AI provider for DataPizza
        datapizza_api_key: API key for DataPizza provider
        datapizza_model: Model name for DataPizza
        datapizza_tools: List of tools to enable
        datapizza_planning_interval: Planning interval for DataPizza agents
        datapizza_max_steps: Maximum steps for DataPizza agents
        roma_max_depth: Maximum recursion depth for ROMA
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA provider
        roma_model: Model name for ROMA
        hybrid_max_depth_analysis: Max depth for ROMA analysis phase (hybrid mode)
        hybrid_max_depth_solving: Max depth for ROMA solving phase (hybrid mode)
        hybrid_execution_mode: ROMA execution mode for hybrid
        hybrid_provider: AI provider for hybrid mode
        hybrid_api_key: API key for hybrid mode provider
        hybrid_model: Model name for hybrid mode
        hybrid_enable_gauntlets: Enable Decomposition Workflow gauntlets in hybrid mode
        hybrid_enable_evolution: Enable evolution in hybrid mode
        hybrid_evolution_iterations: Evolution iterations for hybrid mode
        use_roma_mdap_maker: Explicitly enable/disable ROMA-MDAP-MAKER (zero-error mode)
        roma_mdap_maker_max_depth: Max depth for ROMA-MDAP-MAKER
        roma_mdap_maker_k_ahead: K-ahead threshold for MAKER voting
        roma_mdap_maker_enable_red_flagging: Enable MAKER red-flagging
        roma_mdap_maker_max_samples: Max samples for MAKER voting
        roma_mdap_maker_enable_adaptive_k: Enable adaptive k-ahead selection
        roma_mdap_maker_provider: AI provider for ROMA-MDAP-MAKER
        roma_mdap_maker_api_key: API key for ROMA-MDAP-MAKER provider
        roma_mdap_maker_model: Model name for ROMA-MDAP-MAKER

    Returns:
        Dict with solution generation results
    """
    logger.info(f"Phase 2: Generating solutions with Blue Teams (method={execution_method}, evolution={use_evolution})")

    try:
        # Get available teams
        teams_info = list_available_teams()
        if "error" in teams_info:
            raise Exception(teams_info["error"])

        blue_teams = [t for t in teams_info["teams"] if t["role"] == "Blue"]
        if not blue_teams:
            raise Exception("No Blue Teams available")

        # Select team
        if not team_name:
            team_name = blue_teams[0]["name"]

        sub_problems = decomposition_plan["plan"]["sub_problems"]
        dependencies = decomposition_plan["plan"]["dependencies"]
        team_assignments = decomposition_plan["plan"]["team_assignments"]

        # Filter to subset if specified
        if solve_subset:
            sub_problems = [sp for sp in sub_problems if sp["id"] in solve_subset]

        solutions = []
        failed_sub_problems = []

        for sp in sub_problems:
            try:
                # Solve this sub-problem (with all execution methods available)
                solution = solve_sub_problem_with_team(
                    sub_problem_id=sp["id"],
                    sub_problem_description=sp["description"],
                    team_name=team_assignments.get(sp["id"], team_name),
                    context={
                        "dependencies": dependencies.get(sp["id"], []),
                        "complexity_score": sp.get("complexity_score", 5),
                    },
                    requirements=sp.get("success_criteria", []),
                    # Execution method selection (SOVEREIGN CHOICE)
                    execution_method=execution_method,
                    use_claudiomiro=use_claudiomiro,
                    use_datapizza=use_datapizza,
                    use_roma=use_roma,
                    use_hybrid=use_hybrid,
                    # OpenEvolve parameters
                    use_evolution=use_evolution,
                    evolution_iterations=evolution_iterations,
                    # Claudiomiro parameters
                    claudiomiro_provider=claudiomiro_provider,
                    claudiomiro_backend=claudiomiro_backend,
                    claudiomiro_frontend=claudiomiro_frontend,
                    working_dir=working_dir,
                    max_cycles=max_cycles,
                    # DataPizza parameters
                    datapizza_provider=datapizza_provider,
                    datapizza_api_key=datapizza_api_key,
                    datapizza_model=datapizza_model,
                    datapizza_tools=datapizza_tools,
                    datapizza_planning_interval=datapizza_planning_interval,
                    datapizza_max_steps=datapizza_max_steps,
                    # ROMA parameters
                    roma_max_depth=roma_max_depth,
                    roma_execution_mode=roma_execution_mode,
                    roma_provider=roma_provider,
                    roma_api_key=roma_api_key,
                    roma_model=roma_model,
                    # ROMA-Decomposition Hybrid parameters
                    hybrid_max_depth_analysis=hybrid_max_depth_analysis,
                    hybrid_max_depth_solving=hybrid_max_depth_solving,
                    hybrid_execution_mode=hybrid_execution_mode,
                    hybrid_provider=hybrid_provider,
                    hybrid_api_key=hybrid_api_key,
                    hybrid_model=hybrid_model,
                    hybrid_enable_gauntlets=hybrid_enable_gauntlets,
                    hybrid_enable_evolution=hybrid_enable_evolution,
                    hybrid_evolution_iterations=hybrid_evolution_iterations,
                    # ROMA-MDAP-MAKER parameters
                    use_roma_mdap_maker=use_roma_mdap_maker,
                    roma_mdap_maker_max_depth=roma_mdap_maker_max_depth,
                    roma_mdap_maker_k_ahead=roma_mdap_maker_k_ahead,
                    roma_mdap_maker_enable_red_flagging=roma_mdap_maker_enable_red_flagging,
                    roma_mdap_maker_max_samples=roma_mdap_maker_max_samples,
                    roma_mdap_maker_enable_adaptive_k=roma_mdap_maker_enable_adaptive_k,
                    roma_mdap_maker_provider=roma_mdap_maker_provider,
                    roma_mdap_maker_api_key=roma_mdap_maker_api_key,
                    roma_mdap_maker_model=roma_mdap_maker_model,
                )

                if "error" in solution:
                    failed_sub_problems.append(sp["id"])
                    logger.warning(f"Failed to solve {sp['id']}: {solution['error']}")
                else:
                    solutions.append({
                        "sub_problem_id": sp["id"],
                        "solution": solution["solution"],
                        "team_name": solution["team_name"],
                        "status": solution["status"],
                        "execution_method_used": solution.get("execution_method_used", "unknown"),
                    })

            except Exception as e:
                failed_sub_problems.append(sp["id"])
                logger.error(f"Error solving {sp['id']}: {e}")

        result = {
            "phase": 2,
            "status": "completed" if solutions else "failed",
            "team_used": team_name,
            "solutions": solutions,
            "num_solved": len(solutions),
            "num_failed": len(failed_sub_problems),
            "failed_sub_problems": failed_sub_problems,
            "next_phase": 3,
            "message": f"Phase 2 complete: {len(solutions)} sub-problems solved, {len(failed_sub_problems)} failed",
        }

        logger.info(f"Phase 2 complete: {len(solutions)} solutions generated")
        return result

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
    decomposition_plan: Dict[str, Any],
    gauntlet_name: Optional[str] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique - Critique solutions with Red Team Gauntlets

    This is called by Hephaestus Phase 3 agents for adversarial critique.
    Maps to Stage 3B (Red Team Gauntlet Critique).

    Args:
        solutions: List of solutions from Phase 2
        decomposition_plan: Complete decomposition plan
        gauntlet_name: Specific Red Team gauntlet to use (auto-selected if None)
        use_evolution: Whether to use OpenEvolve for evolutionary critique
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with critique results
    """
    logger.info(f"Phase 3: Critiquing solutions with Red Team Gauntlets (evolution={use_evolution})")

    try:
        # Get available gauntlets
        gauntlets_info = list_available_gauntlets()
        if "error" in gauntlets_info:
            raise Exception(gauntlets_info["error"])

        # Find Red Team gauntlets
        red_gauntlets = [g for g in gauntlets_info["gauntlets"] if "red" in g["name"].lower()]
        if not red_gauntlets:
            # Fallback to any gauntlet
            red_gauntlets = gauntlets_info["gauntlets"]

        if not red_gauntlets:
            raise Exception("No gauntlets available")

        # Select gauntlet
        if not gauntlet_name:
            gauntlet_name = red_gauntlets[0]["name"]

        # Build sub-problem lookup
        sub_problems = {
            sp["id"]: sp
            for sp in decomposition_plan["plan"]["sub_problems"]
        }

        critiques = []
        failed_critiques = []

        for solution in solutions:
            sub_problem_id = solution["sub_problem_id"]
            sub_problem = sub_problems.get(sub_problem_id)

            if not sub_problem:
                logger.warning(f"Sub-problem {sub_problem_id} not found in plan")
                continue

            try:
                # Critique with OpenEvolve if enabled
                critique = critique_solution_with_gauntlet(
                    solution=solution["solution"],
                    sub_problem_id=sub_problem_id,
                    gauntlet_name=gauntlet_name,
                    sub_problem_description=sub_problem["description"],
                    use_evolution=use_evolution,
                    evolution_iterations=evolution_iterations,
                )

                if "error" in critique:
                    failed_critiques.append(sub_problem_id)
                    logger.warning(f"Failed to critique {sub_problem_id}: {critique['error']}")
                else:
                    critiques.append({
                        "sub_problem_id": sub_problem_id,
                        "approved": critique["approved"],
                        "issues_found": critique.get("issues_found", []),
                        "overall_score": critique.get("overall_score", 0.0),
                        "feedback": critique.get("feedback", ""),
                    })

            except Exception as e:
                failed_critiques.append(sub_problem_id)
                logger.error(f"Error critiquing {sub_problem_id}: {e}")

        # Calculate statistics
        approved_count = sum(1 for c in critiques if c["approved"])
        avg_score = sum(c["overall_score"] for c in critiques) / len(critiques) if critiques else 0.0

        result = {
            "phase": 3,
            "status": "completed",
            "gauntlet_used": gauntlet_name,
            "critiques": critiques,
            "num_approved": approved_count,
            "num_rejected": len(critiques) - approved_count,
            "num_failed": len(failed_critiques),
            "avg_score": avg_score,
            "next_phase": 4,
            "message": f"Phase 3 complete: {approved_count}/{len(critiques)} solutions approved (avg score: {avg_score:.2f})",
        }

        logger.info(f"Phase 3 complete: {approved_count}/{len(critiques)} approved")
        return result

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
    critiques: List[Dict[str, Any]],
    decomposition_plan: Dict[str, Any],
    gauntlet_name: Optional[str] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification - Verify solutions with Gold Team Gauntlets

    This is called by Hephaestus Phase 4 agents for solution verification.
    Maps to Stage 3C (Gold Team Gauntlet Verification).

    Args:
        solutions: List of solutions from Phase 2
        critiques: List of critiques from Phase 3
        decomposition_plan: Complete decomposition plan
        gauntlet_name: Specific Gold Team gauntlet to use (auto-selected if None)
        use_evolution: Whether to use OpenEvolve for evolutionary verification
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with verification results
    """
    logger.info(f"Phase 4: Verifying solutions with Gold Team Gauntlets (evolution={use_evolution})")

    try:
        # Get available gauntlets
        gauntlets_info = list_available_gauntlets()
        if "error" in gauntlets_info:
            raise Exception(gauntlets_info["error"])

        # Find Gold Team gauntlets
        gold_gauntlets = [g for g in gauntlets_info["gauntlets"] if "gold" in g["name"].lower()]
        if not gold_gauntlets:
            # Fallback to any gauntlet
            gold_gauntlets = gauntlets_info["gauntlets"]

        if not gold_gauntlets:
            raise Exception("No gauntlets available")

        # Select gauntlet
        if not gauntlet_name:
            gauntlet_name = gold_gauntlets[0]["name"]

        # Build lookups
        solutions_lookup = {s["sub_problem_id"]: s for s in solutions}
        critiques_lookup = {c["sub_problem_id"]: c for c in critiques}
        sub_problems = {
            sp["id"]: sp
            for sp in decomposition_plan["plan"]["sub_problems"]
        }

        verifications = []
        failed_verifications = []

        for critique in critiques:
            sub_problem_id = critique["sub_problem_id"]
            solution = solutions_lookup.get(sub_problem_id)
            sub_problem = sub_problems.get(sub_problem_id)

            if not solution or not sub_problem:
                logger.warning(f"Missing data for sub-problem {sub_problem_id}")
                continue

            try:
                # Verify with OpenEvolve if enabled
                verification = verify_solution_with_gauntlet(
                    solution=solution["solution"],
                    critique=critique,
                    sub_problem_id=sub_problem_id,
                    gauntlet_name=gauntlet_name,
                    requirements=sub_problem.get("success_criteria", []),
                    use_evolution=use_evolution,
                    evolution_iterations=evolution_iterations,
                )

                if "error" in verification:
                    failed_verifications.append(sub_problem_id)
                    logger.warning(f"Failed to verify {sub_problem_id}: {verification['error']}")
                else:
                    verifications.append({
                        "sub_problem_id": sub_problem_id,
                        "approved": verification["approved"],
                        "correctness_score": verification.get("correctness_score", 0.0),
                        "completeness_score": verification.get("completeness_score", 0.0),
                        "quality_score": verification.get("quality_score", 0.0),
                        "requirements_met": verification.get("requirements_met", {}),
                    })

            except Exception as e:
                failed_verifications.append(sub_problem_id)
                logger.error(f"Error verifying {sub_problem_id}: {e}")

        # Calculate statistics
        approved_count = sum(1 for v in verifications if v["approved"])
        avg_correctness = sum(v["correctness_score"] for v in verifications) / len(verifications) if verifications else 0.0
        avg_quality = sum(v["quality_score"] for v in verifications) / len(verifications) if verifications else 0.0

        result = {
            "phase": 4,
            "status": "completed",
            "gauntlet_used": gauntlet_name,
            "verifications": verifications,
            "num_approved": approved_count,
            "num_rejected": len(verifications) - approved_count,
            "num_failed": len(failed_verifications),
            "avg_correctness": avg_correctness,
            "avg_quality": avg_quality,
            "next_phase": 5,
            "message": f"Phase 4 complete: {approved_count}/{len(verifications)} solutions verified (avg quality: {avg_quality:.2f})",
        }

        logger.info(f"Phase 4 complete: {approved_count}/{len(verifications)} verified")
        return result

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        return {
            "phase": 4,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 4 verification failed: {e}",
        }


def execute_phase_5_reassemble(
    decomposition_plan: Dict[str, Any],
    solutions: List[Dict[str, Any]],
    verifications: List[Dict[str, Any]],
    reassembly_strategy: str = "verified_only",
) -> Dict[str, Any]:
    """
    Execute Phase 5: Reassembly - Reassemble verified solutions into final output

    This is called by Hephaestus Phase 5 agents to reassemble solutions.
    Maps to Stage 4 (Configurable Reassembly).

    Args:
        decomposition_plan: Complete decomposition plan
        solutions: List of solutions from Phase 2
        verifications: List of verifications from Phase 4
        reassembly_strategy: Strategy for reassembly ("verified_only", "best_effort", "all")

    Returns:
        Dict with reassembly results
    """
    logger.info(f"Phase 5: Reassembling solutions into final output")

    try:
        # Build lookups
        solutions_lookup = {s["sub_problem_id"]: s for s in solutions}
        verifications_lookup = {v["sub_problem_id"]: v for v in verifications}
        sub_problems = decomposition_plan["plan"]["sub_problems"]
        dependencies = decomposition_plan["plan"]["dependencies"]

        # Select solutions based on strategy
        assembled_solutions = []
        skipped_sub_problems = []

        for sp in sub_problems:
            sub_problem_id = sp["id"]
            verification = verifications_lookup.get(sub_problem_id)

            if not verification:
                logger.warning(f"No verification for {sub_problem_id}")
                if reassembly_strategy == "all":
                    solution = solutions_lookup.get(sub_problem_id)
                    if solution:
                        assembled_solutions.append({
                            "sub_problem_id": sub_problem_id,
                            "solution": solution["solution"],
                            "verified": False,
                        })
                continue

            # Apply strategy
            include_solution = False
            if reassembly_strategy == "verified_only":
                include_solution = verification["approved"]
            elif reassembly_strategy == "best_effort":
                include_solution = verification["quality_score"] > 0.5
            elif reassembly_strategy == "all":
                include_solution = True

            if include_solution:
                solution = solutions_lookup.get(sub_problem_id)
                if solution:
                    assembled_solutions.append({
                        "sub_problem_id": sub_problem_id,
                        "solution": solution["solution"],
                        "verified": verification["approved"],
                        "quality_score": verification.get("quality_score", 0.0),
                    })
            else:
                skipped_sub_problems.append(sub_problem_id)

        # Calculate assembly order based on dependencies
        assembly_order = calculate_assembly_order(
            [s["sub_problem_id"] for s in assembled_solutions],
            dependencies,
        )

        # Sort solutions by assembly order
        assembled_solutions.sort(
            key=lambda x: assembly_order.index(x["sub_problem_id"])
            if x["sub_problem_id"] in assembly_order else 999
        )

        # Build final output
        final_output_parts = []
        for item in assembled_solutions:
            final_output_parts.append(f"""
## Sub-Problem: {item['sub_problem_id']}

{item['solution']}

**Verified:** {item['verified']}
**Quality Score:** {item.get('quality_score', 0.0):.2f}
""")

        final_output = "\n".join(final_output_parts)

        result = {
            "phase": 5,
            "status": "completed",
            "reassembly_strategy": reassembly_strategy,
            "assembled_solutions": assembled_solutions,
            "num_assembled": len(assembled_solutions),
            "num_skipped": len(skipped_sub_problems),
            "skipped_sub_problems": skipped_sub_problems,
            "assembly_order": assembly_order,
            "final_output": final_output,
            "next_phase": 6,
            "message": f"Phase 5 complete: {len(assembled_solutions)} solutions reassembled",
        }

        logger.info(f"Phase 5 complete: {len(assembled_solutions)} solutions reassembled")
        return result

    except Exception as e:
        logger.error(f"Phase 5 failed: {e}")
        return {
            "phase": 5,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 5 reassembly failed: {e}",
        }


def execute_phase_6_final_validation(
    reassembly_result: Dict[str, Any],
    decomposition_plan: Dict[str, Any],
    validation_criteria: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 6: Final Validation - Validate final output and extract knowledge

    This is called by Hephaestus Phase 6 agents for final validation.
    Maps to Stage 5 (Final Verification) and Stage 6 (Knowledge Extraction).

    Args:
        reassembly_result: Result from Phase 5 reassembly
        decomposition_plan: Complete decomposition plan
        validation_criteria: Optional list of validation criteria

    Returns:
        Dict with final validation results
    """
    logger.info(f"Phase 6: Final validation and knowledge extraction")

    try:
        # Calculate metrics
        assembled = reassembly_result["assembled_solutions"]
        num_verified = sum(1 for s in assembled if s["verified"])
        avg_quality = sum(s.get("quality_score", 0.0) for s in assembled) / len(assembled) if assembled else 0.0

        # Validate against criteria
        validation_passed = True
        validation_results = {}

        if validation_criteria:
            for criterion in validation_criteria:
                if criterion == "all_verified":
                    passed = num_verified == len(assembled)
                elif criterion == "min_quality_threshold":
                    passed = avg_quality >= 0.7
                elif criterion == "max_skipped":
                    passed = reassembly_result["num_skipped"] <= len(assembled) * 0.1
                else:
                    passed = True  # Unknown criteria, pass by default

                validation_results[criterion] = passed
                if not passed:
                    validation_passed = False

        # Extract knowledge artifacts
        knowledge_artifacts = []
        for item in assembled:
            if item["verified"]:
                knowledge_artifacts.append({
                    "sub_problem_id": item["sub_problem_id"],
                    "type": "verified_solution",
                    "quality": item.get("quality_score", 0.0),
                })

        result = {
            "phase": 6,
            "status": "completed",
            "validation_passed": validation_passed,
            "validation_results": validation_results,
            "num_verified_solutions": num_verified,
            "num_total_solutions": len(assembled),
            "avg_quality_score": avg_quality,
            "knowledge_artifacts": knowledge_artifacts,
            "workflow_complete": validation_passed,
            "message": f"Phase 6 complete: Validation {'PASSED' if validation_passed else 'FAILED'}",
        }

        logger.info(f"Phase 6 complete: Validation {'PASSED' if validation_passed else 'FAILED'}")
        return result

    except Exception as e:
        logger.error(f"Phase 6 failed: {e}")
        return {
            "phase": 6,
            "status": "failed",
            "error": str(e),
            "workflow_complete": False,
            "message": f"Phase 6 final validation failed: {e}",
        }


# =============================================================================
# WORKFLOW ORCHESTRATION
# =============================================================================

class DecompositionHephaestusWorkflowBridge:
    """
    Main bridge class that orchestrates decomposition workflows with Hephaestus.

    This class provides a high-level interface that Hephaestus can use
    to execute Sovereign-Grade Decomposition workflows.
    """

    def __init__(self):
        self.phase_executors = {
            1: execute_phase_1_setup,
            2: execute_phase_2_solve,
            3: execute_phase_3_critique,
            4: execute_phase_4_verify,
            5: execute_phase_5_reassemble,
            6: execute_phase_6_final_validation,
        }

    def execute_phase(
        self,
        phase_id: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute a specific phase of the decomposition workflow.

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

    def execute_full_workflow(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        max_sub_problems: int = 15,
        decomposition_strategy: str = "semantic",
        reassembly_strategy: str = "verified_only",
        validation_criteria: Optional[List[str]] = None,
        use_evolution: bool = True,
        evolution_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full decomposition workflow through all phases.

        Args:
            problem_statement: The problem to solve
            problem_type: Type of problem
            domain: Problem domain
            max_sub_problems: Maximum sub-problems to create
            decomposition_strategy: Decomposition strategy
            reassembly_strategy: Reassembly strategy
            validation_criteria: Validation criteria for final phase
            use_evolution: Whether to use OpenEvolve for all evolutionary stages
            evolution_iterations: Number of evolution iterations (uses defaults per phase if None)

        Returns:
            Dict with complete workflow results
        """
        logger.info(f"Starting full decomposition workflow (evolution={use_evolution})...")

        # Set default iterations per phase if not specified
        if evolution_iterations is None:
            phase1_iters = 50
            phase2_iters = 100
            phase3_iters = 30
            phase4_iters = 30
        else:
            phase1_iters = phase2_iters = phase3_iters = phase4_iters = evolution_iterations

        # Phase 1: Setup
        phase1_result = execute_phase_1_setup(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            max_sub_problems=max_sub_problems,
            decomposition_strategy=decomposition_strategy,
            use_evolution=use_evolution,
            evolution_iterations=phase1_iters,
        )

        if phase1_result["status"] == "failed":
            return {
                "workflow_status": "failed",
                "failed_at_phase": 1,
                "error": phase1_result.get("error"),
            }

        # Phase 2: Solve
        phase2_result = execute_phase_2_solve(
            decomposition_plan=phase1_result,
            use_evolution=use_evolution,
            evolution_iterations=phase2_iters,
        )

        if phase2_result["status"] == "failed":
            return {
                "workflow_status": "failed",
                "failed_at_phase": 2,
                "error": phase2_result.get("error"),
            }

        # Phase 3: Critique
        phase3_result = execute_phase_3_critique(
            solutions=phase2_result["solutions"],
            decomposition_plan=phase1_result,
            use_evolution=use_evolution,
            evolution_iterations=phase3_iters,
        )

        if phase3_result["status"] == "failed":
            return {
                "workflow_status": "failed",
                "failed_at_phase": 3,
                "error": phase3_result.get("error"),
            }

        # Phase 4: Verify
        phase4_result = execute_phase_4_verify(
            solutions=phase2_result["solutions"],
            critiques=phase3_result["critiques"],
            decomposition_plan=phase1_result,
            use_evolution=use_evolution,
            evolution_iterations=phase4_iters,
        )

        if phase4_result["status"] == "failed":
            return {
                "workflow_status": "failed",
                "failed_at_phase": 4,
                "error": phase4_result.get("error"),
            }

        # Phase 5: Reassemble
        phase5_result = execute_phase_5_reassemble(
            decomposition_plan=phase1_result,
            solutions=phase2_result["solutions"],
            verifications=phase4_result["verifications"],
            reassembly_strategy=reassembly_strategy,
        )

        if phase5_result["status"] == "failed":
            return {
                "workflow_status": "failed",
                "failed_at_phase": 5,
                "error": phase5_result.get("error"),
            }

        # Phase 6: Final Validation
        phase6_result = execute_phase_6_final_validation(
            reassembly_result=phase5_result,
            decomposition_plan=phase1_result,
            validation_criteria=validation_criteria,
        )

        return {
            "workflow_status": "completed" if phase6_result["status"] == "completed" else "failed",
            "phases": {
                1: phase1_result,
                2: phase2_result,
                3: phase3_result,
                4: phase4_result,
                5: phase5_result,
                6: phase6_result,
            },
            "final_output": phase5_result.get("final_output"),
            "validation_passed": phase6_result.get("validation_passed", False),
        }

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

Your mission: Analyze the problem and create a decomposition plan.

STEPS:
1. Analyze the problem statement
2. Identify constraints and success criteria
3. Decompose into sub-problems
4. Create team and gauntlet assignments

MCP TOOLS TO USE:
- analyze_problem_for_decomposition()
- decompose_problem_into_sub_problems()
- create_decomposition_plan()

EXPECTED OUTPUT:
- Complete decomposition plan with sub-problems
- Team assignments
- Gauntlet assignments
""",
            2: """PHASE 2: SOLUTION GENERATION

Your mission: Generate solutions for sub-problems using Blue Teams.

STEPS:
1. Review decomposition plan
2. Assign sub-problems to Blue Teams
3. Generate solutions for each sub-problem

MCP TOOLS TO USE:
- solve_sub_problem_with_team()

EXPECTED OUTPUT:
- Solutions for all sub-problems
- Team assignment tracking
""",
            3: """PHASE 3: ADVERSARIAL CRITIQUE

Your mission: Critique solutions using Red Team Gauntlets.

STEPS:
1. Review generated solutions
2. Run Red Team gauntlets on each solution
3. Collect critique feedback

MCP TOOLS TO USE:
- critique_solution_with_gauntlet()

EXPECTED OUTPUT:
- Critique reports for all solutions
- Issues found and severity scores
""",
            4: """PHASE 4: VERIFICATION

Your mission: Verify solutions using Gold Team Gauntlets.

STEPS:
1. Review critiques from Phase 3
2. Run Gold Team gauntlets for verification
3. Check requirements satisfaction

MCP TOOLS TO USE:
- verify_solution_with_gauntlet()

EXPECTED OUTPUT:
- Verification reports
- Correctness, completeness, and quality scores
""",
            5: """PHASE 5: REASSEMBLY

Your mission: Reassemble verified solutions into final output.

STEPS:
1. Review all verified solutions
2. Determine assembly order based on dependencies
3. Combine solutions into coherent final output

MCP TOOLS TO USE:
- Internal reassembly logic

EXPECTED OUTPUT:
- Final assembled solution
- Assembly order and strategy
""",
            6: """PHASE 6: FINAL VALIDATION

Your mission: Validate final output and extract knowledge.

STEPS:
1. Review final assembled output
2. Validate against criteria
3. Extract knowledge artifacts

MCP TOOLS TO USE:
- Internal validation logic

EXPECTED OUTPUT:
- Final validation result
- Knowledge artifacts for learning
""",
        }

        return instructions.get(phase_id, f"No instructions for phase {phase_id}")

    def list_available_tools(self) -> List[str]:
        """List available Decomposition MCP tools"""
        from decomposition_mcp_tools import list_mcp_tools
        return list_mcp_tools()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_assembly_order(
    sub_problem_ids: List[str],
    dependencies: Dict[str, List[str]],
) -> List[str]:
    """Calculate optimal assembly order based on dependencies"""
    # Topological sort
    visited = set()
    order = []

    def visit(sp_id: str):
        if sp_id in visited:
            return
        visited.add(sp_id)
        for dep in dependencies.get(sp_id, []):
            if dep in sub_problem_ids:
                visit(dep)
        order.append(sp_id)

    for sp_id in sub_problem_ids:
        if sp_id not in visited:
            visit(sp_id)

    return order


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_workflow_bridge():
    """Initialize the workflow bridge"""
    status = get_decomposition_status()
    logger.info(f"Decomposition-Hephaestus workflow bridge initialized (available: {status['available']})")


# Auto-initialize on import
initialize_workflow_bridge()
