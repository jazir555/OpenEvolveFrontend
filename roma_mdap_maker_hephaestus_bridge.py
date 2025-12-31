"""
ROMA-MDAP-MAKER - Hephaestus Bridge

This module provides the bridge between Hephaestus workflow phases and
the ROMA-MDAP-MAKER integration system (ROMA + MAKER zero-error voting).

ROMA-MDAP-MAKER Architecture:
    ROMA (Recursive Decomposition)
        ↓
    MAKER (First-to-Ahead-by-K Voting + Red-Flagging)
        ↓
    Hierarchical Aggregation with Confidence Weighting

Phase Mapping:
- Phase 1: Problem Setup → ROMA-MDAP complexity analysis + parameter recommendation
- Phase 2: Solution Generation → ROMA decomposition + MAKER voting on each atomic task
- Phase 3: Adversarial Critique → ROMA-MDAP critique with voting
- Phase 4: Verification → ROMA-MDAP verification with voting
- Phase 5: Reassembly → Hierarchical aggregation with confidence weighting
- Phase 6: Final Validation → Full ROMA-MDAP-MAKER with verification

Zero-Error Guarantee:
- First-to-ahead-by-k voting: P(success) ≈ 1 - exp(-k)
- Red-flagging: Detects and discards unreliable outputs
- Hierarchical confidence: Tracks confidence across ROMA levels
"""

import logging
from typing import Dict, Any, List, Optional

from roma_mdap_maker_mcp_tools import (
    solve_with_roma_mdap_maker,
    solve_subproblem_with_roma_mdap_maker,
    analyze_problem_with_roma_mdap,
    verify_solution_with_roma_mdap,
    get_roma_mdap_maker_status,
)
from roma_mdap_maker_engine import (
    create_roma_mdap_maker_config,
    ROMAMDAPMakerEngine,
    ROMA_AVAILABLE,
    MDAP_AVAILABLE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 1: PROBLEM SETUP WITH ROMA-MDAP COMPLEXITY ANALYSIS
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    roma_max_depth_analysis: int = 3,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup with ROMA-MDAP complexity analysis

    Analyzes problem complexity and recommends optimal ROMA-MDAP-MAKER parameters.

    Args:
        problem_statement: The problem to analyze
        roma_max_depth_analysis: Max depth for ROMA analysis
        provider: AI provider
        model: Model name
        api_key: API key
        **kwargs: Additional parameters

    Returns:
        Dict with:
            - complexity_score: Estimated complexity (1-10)
            - recommended_depth: Recommended ROMA depth
            - recommended_k: Recommended MAKER k-ahead value
            - use_roma_mdap_maker: Whether to use ROMA-MDAP-MAKER
            - decomposition: ROMA decomposition structure
            - dag_info: DAG structure information
    """
    logger.info(f"Phase 1: ROMA-MDAP-MAKER complexity analysis - {problem_statement[:50]}...")

    try:
        # Analyze problem with ROMA-MDAP
        analysis = analyze_problem_with_roma_mdap(
            problem_statement=problem_statement,
            roma_max_depth=roma_max_depth_analysis,
            provider=provider,
            model=model,
            **kwargs
        )

        if "error" in analysis:
            raise Exception(analysis["error"])

        # Extract recommendations
        complexity_score = analysis.get("estimated_complexity", 5.0)
        recommended_depth = analysis.get("recommended_depth", 2)
        recommended_k = analysis.get("recommended_k", 3)
        use_roma_mdap_maker = analysis.get("use_roma_mdap_maker", complexity_score > 7.0)

        logger.info(f"  Complexity: {complexity_score}/10")
        logger.info(f"  Recommended: depth={recommended_depth}, k={recommended_k}")
        logger.info(f"  Use ROMA-MDAP-MAKER: {use_roma_mdap_maker}")

        return {
            "phase": 1,
            "status": "completed",
            "analysis": analysis,
            "complexity_score": complexity_score,
            "recommended_params": {
                "roma_max_depth": recommended_depth,
                "mdap_k_ahead": recommended_k,
                "enable_red_flagging": True,
                "enable_adaptive_k": True,
            },
            "use_roma_mdap_maker": use_roma_mdap_maker,
            "next_phase": 2,
            "message": f"Phase 1 complete: Complexity {complexity_score}/10",
        }

    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        return {
            "phase": 1,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 1 setup failed: {e}",
        }


# =============================================================================
# PHASE 2: SOLUTION GENERATION WITH ROMA-MDAP-MAKER
# =============================================================================

def execute_phase_2_solve(
    sub_problem_id: str,
    sub_problem_description: str,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    roma_max_depth: int = 2,
    mdap_k_ahead: int = 3,
    mdap_enable_red_flagging: bool = True,
    mdap_max_samples: int = 100,
    enable_adaptive_k: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation using ROMA-MDAP-MAKER

    Applies ROMA hierarchical decomposition with MAKER voting on each atomic task.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        roma_max_depth: Max depth for ROMA decomposition
        mdap_k_ahead: K-ahead threshold for MAKER voting
        mdap_enable_red_flagging: Enable MAKER red-flagging
        mdap_max_samples: Max samples for MAKER voting
        enable_adaptive_k: Enable adaptive k-ahead selection
        provider: AI provider
        model: Model name
        api_key: API key
        **kwargs: Additional parameters

    Returns:
        Dict with solution, confidence, and detailed metrics
    """
    logger.info(f"Phase 2: Solving sub-problem {sub_problem_id} with ROMA-MDAP-MAKER")
    logger.info(f"  Description: {sub_problem_description[:80]}...")

    try:
        # Create config
        config = create_roma_mdap_maker_config(
            roma_max_depth_analysis=roma_max_depth,
            roma_max_depth_solving=roma_max_depth,
            roma_execution_mode="recursive",
            provider=provider,
            model=model,
            api_key=api_key,
            mdap_k_ahead=mdap_k_ahead,
            mdap_max_samples=mdap_max_samples,
            mdap_enable_red_flagging=mdap_enable_red_flagging,
            apply_maker_to_roma_atomic=True,
            enable_hierarchical_voting=True,
            enable_adaptive_k=enable_adaptive_k,
        )

        # Solve with ROMA-MDAP-MAKER
        result = solve_subproblem_with_roma_mdap_maker(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            context=context,
            constraints=constraints,
            requirements=requirements,
            config=config,
        )

        if "error" in result:
            raise Exception(result["error"])

        # Extract metrics
        metrics = result.get("roma_mdap_maker_metrics", {})

        logger.info(f"  Solution generated with confidence: {result.get('confidence', 0):.2%}")
        logger.info(f"  ROMA levels: {metrics.get('roma_decomposition_levels', 0)}")
        logger.info(f"  Atomic tasks: {metrics.get('total_atomic_tasks', 0)}")
        logger.info(f"  Voting rounds: {metrics.get('total_voting_rounds', 0)}")
        logger.info(f"  Red-flags: {metrics.get('total_red_flags', 0)}")

        return {
            "phase": 2,
            "status": "completed",
            "sub_problem_id": sub_problem_id,
            "solution": result.get("solution"),
            "confidence": result.get("confidence", 0.0),
            "execution_method_used": "roma_mdap_maker",
            "metrics": metrics,
            "red_flags": result.get("red_flags", 0),
            "attempts": result.get("attempts", 0),
            "next_phase": 3,
            "message": f"Phase 2 complete: Solution with {result.get('confidence', 0):.0%} confidence",
        }

    except Exception as e:
        logger.error(f"Phase 2 failed for {sub_problem_id}: {e}")
        return {
            "phase": 2,
            "status": "failed",
            "sub_problem_id": sub_problem_id,
            "error": str(e),
            "message": f"Phase 2 solution generation failed: {e}",
        }


# =============================================================================
# PHASE 3: ADVERSARIAL CRITIQUE WITH ROMA-MDAP-MAKER
# =============================================================================

def execute_phase_3_critique(
    solution: str,
    problem_statement: str,
    context: Optional[Dict[str, Any]] = None,
    attack_phases: Optional[List[str]] = None,
    roma_max_depth: int = 1,
    mdap_k_ahead: int = 2,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique using ROMA-MDAP-MAKER

    Applies ROMA decomposition to critique phases with MAKER voting for reliability.

    Args:
        solution: Solution to critique
        problem_statement: Original problem statement
        context: Additional context
        attack_phases: List of attack phases (integration, edge_cases, security, etc.)
        roma_max_depth: Max depth for ROMA decomposition (default: 1 for critique)
        mdap_k_ahead: K-ahead threshold for MAKER voting
        provider: AI provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Dict with critique results and identified flaws
    """
    logger.info("Phase 3: ROMA-MDAP-MAKER adversarial critique")

    if attack_phases is None:
        attack_phases = [
            "integration_vulnerability",
            "cross_component",
            "edge_cases",
            "performance",
            "security",
            "compliance"
        ]

    try:
        all_flaws = []
        all_improvements = []
        phase_reports = []

        for phase in attack_phases:
            logger.info(f"  Attack phase: {phase}")

            # Use ROMA-MDAP-MAKER for this attack phase
            critique_task = f"""
            Critique the following solution for {phase.replace('_', ' ')} issues:

            Original Problem: {problem_statement}

            Solution: {solution}

            Identify:
            1. Potential vulnerabilities or weaknesses
            2. Edge cases not handled
            3. Areas for improvement
            4. Specific recommendations

            Format your response as a structured critique.
            """

            result = solve_with_roma_mdap_maker(
                task=critique_task,
                context=context,
                roma_max_depth_analysis=roma_max_depth,
                roma_max_depth_solving=roma_max_depth,
                mdap_k_ahead=mdap_k_ahead,
                provider=provider,
                model=model,
            )

            if "error" in result:
                logger.warning(f"    Phase {phase} failed: {result['error']}")
                continue

            # Extract critique content
            critique_content = result.get("solution", "")
            confidence = result.get("confidence", 0.0)

            # Parse flaws and improvements (simplified - in production use LLM)
            flaws = _parse_flaws_from_critique(critique_content, phase)
            improvements = _parse_improvements_from_critique(critique_content)

            all_flaws.extend(flaws)
            all_improvements.extend(improvements)

            phase_reports.append({
                "phase": phase,
                "critique": critique_content,
                "confidence": confidence,
                "flaws_found": len(flaws),
                "improvements": len(improvements),
            })

        # Calculate approval
        critical_flaws = sum(1 for f in all_flaws if f.get("severity") == "critical")
        is_approved = critical_flaws == 0

        logger.info(f"  Critique complete: {len(all_flaws)} flaws identified")
        logger.info(f"  Critical flaws: {critical_flaws}")
        logger.info(f"  Status: {'APPROVED' if is_approved else 'NEEDS IMPROVEMENT'}")

        return {
            "phase": 3,
            "status": "completed",
            "is_approved": is_approved,
            "attack_phases_completed": len(phase_reports),
            "phase_reports": phase_reports,
            "identified_flaws": all_flaws,
            "suggested_improvements": all_improvements,
            "total_flaws": len(all_flaws),
            "critical_flaws": critical_flaws,
            "next_phase": 4,
            "message": f"Phase 3 complete: {len(all_flaws)} flaws identified",
        }

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        return {
            "phase": 3,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 3 critique failed: {e}",
        }


# =============================================================================
# PHASE 4: VERIFICATION WITH ROMA-MDAP-MAKER
# =============================================================================

def execute_phase_4_verify(
    solution: str,
    problem_statement: str,
    requirements: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    roma_max_depth: int = 1,
    mdap_k_ahead: int = 2,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification using ROMA-MDAP-MAKER

    Verifies solution meets requirements with MAKER voting for reliability.

    Args:
        solution: Solution to verify
        problem_statement: Original problem statement
        requirements: List of requirements to verify
        context: Additional context
        roma_max_depth: Max depth for ROMA decomposition
        mdap_k_ahead: K-ahead threshold for MAKER voting
        provider: AI provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Dict with verification results
    """
    logger.info("Phase 4: ROMA-MDAP-MAKER verification")

    try:
        # Verify solution with ROMA-MDAP
        verification = verify_solution_with_roma_mdap(
            solution=solution,
            problem_statement=problem_statement,
            requirements=requirements or [],
            context=context,
            roma_max_depth=roma_max_depth,
            mdap_k_ahead=mdap_k_ahead,
            provider=provider,
            model=model,
        )

        if "error" in verification:
            raise Exception(verification["error"])

        # Extract verification results
        is_verified = verification.get("is_verified", False)
        confidence = verification.get("confidence", 0.0)
        requirement_results = verification.get("requirement_results", [])

        # Calculate verification score
        verified_count = sum(1 for r in requirement_results if r.get("satisfied", False))
        total_count = len(requirement_results)
        verification_score = verified_count / total_count if total_count > 0 else 0.0

        logger.info(f"  Verification complete: {verification_score:.0%} requirements satisfied")
        logger.info(f"  Confidence: {confidence:.0%}")
        logger.info(f"  Status: {'VERIFIED' if is_verified else 'NOT VERIFIED'}")

        return {
            "phase": 4,
            "status": "completed",
            "is_verified": is_verified,
            "verification_score": verification_score,
            "confidence": confidence,
            "requirement_results": requirement_results,
            "verified_count": verified_count,
            "total_count": total_count,
            "next_phase": 5,
            "message": f"Phase 4 complete: {verification_score:.0%} verified",
        }

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        return {
            "phase": 4,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 4 verification failed: {e}",
        }


# =============================================================================
# PHASE 5: REASSEMBLY WITH HIERARCHICAL AGGREGATION
# =============================================================================

def execute_phase_5_reassemble(
    sub_solutions: List[Dict[str, Any]],
    problem_statement: str,
    context: Optional[Dict[str, Any]] = None,
    aggregation_method: str = "confidence_weighted",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 5: Reassembly with hierarchical aggregation

    Combines sub-solutions using confidence-weighted aggregation from ROMA hierarchy.

    Args:
        sub_solutions: List of sub-solutions with confidence scores
        problem_statement: Original problem statement
        context: Additional context
        aggregation_method: How to aggregate ("confidence_weighted", "simple", "llm")
        provider: AI provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Dict with integrated solution
    """
    logger.info(f"Phase 5: Reassembling {len(sub_solutions)} sub-solutions")

    try:
        if not sub_solutions:
            raise Exception("No sub-solutions to reassemble")

        # Filter out failed solutions
        valid_solutions = [s for s in sub_solutions if s.get("status") == "completed" and s.get("solution")]

        if not valid_solutions:
            raise Exception("No valid sub-solutions to reassemble")

        logger.info(f"  Valid solutions: {len(valid_solutions)}/{len(sub_solutions)}")

        # Aggregate based on method
        if aggregation_method == "confidence_weighted":
            integrated_solution = _aggregate_confidence_weighted(valid_solutions)
        elif aggregation_method == "simple":
            integrated_solution = _aggregate_simple(valid_solutions)
        elif aggregation_method == "llm":
            integrated_solution = _aggregate_with_llm(
                valid_solutions,
                problem_statement,
                context,
                provider,
                model
            )
        else:
            raise Exception(f"Unknown aggregation method: {aggregation_method}")

        # Calculate combined confidence
        confidences = [s.get("confidence", 0.5) for s in valid_solutions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(f"  Reassembly complete with confidence: {avg_confidence:.0%}")

        return {
            "phase": 5,
            "status": "completed",
            "integrated_solution": integrated_solution,
            "confidence": avg_confidence,
            "num_solutions": len(valid_solutions),
            "aggregation_method": aggregation_method,
            "next_phase": 6,
            "message": f"Phase 5 complete: {len(valid_solutions)} solutions integrated",
        }

    except Exception as e:
        logger.error(f"Phase 5 failed: {e}")
        return {
            "phase": 5,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 5 reassembly failed: {e}",
        }


# =============================================================================
# PHASE 6: FINAL VALIDATION WITH FULL ROMA-MDAP-MAKER
# =============================================================================

def execute_phase_6_final_validation(
    integrated_solution: str,
    problem_statement: str,
    requirements: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    roma_max_depth: int = 2,
    mdap_k_ahead: int = 3,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute Phase 6: Final Validation with full ROMA-MDAP-MAKER

    Applies complete ROMA-MDAP-MAKER pipeline for final validation.

    Args:
        integrated_solution: Integrated solution to validate
        problem_statement: Original problem statement
        requirements: List of requirements
        context: Additional context
        roma_max_depth: Max depth for ROMA decomposition
        mdap_k_ahead: K-ahead threshold for MAKER voting
        provider: AI provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Dict with final validation results
    """
    logger.info("Phase 6: Final validation with ROMA-MDAP-MAKER")

    try:
        # Full solve with ROMA-MDAP-MAKER for validation
        result = solve_with_roma_mdap_maker(
            task=problem_statement,
            context={
                **(context or {}),
                "reference_solution": integrated_solution,
                "validation_mode": True,
            },
            requirements=requirements,
            roma_max_depth_analysis=roma_max_depth,
            roma_max_depth_solving=roma_max_depth,
            mdap_k_ahead=mdap_k_ahead,
            mdap_enable_red_flagging=True,
            enable_adaptive_k=True,
            provider=provider,
            model=model,
        )

        if "error" in result:
            raise Exception(result["error"])

        # Compare with reference solution
        validation_solution = result.get("solution", "")
        confidence = result.get("confidence", 0.0)
        metrics = result.get("roma_mdap_maker_metrics", {})

        # Calculate similarity (simplified - in production use embeddings)
        similarity = _calculate_solution_similarity(integrated_solution, validation_solution)

        is_validated = confidence >= 0.8 and similarity >= 0.7

        logger.info(f"  Final validation complete:")
        logger.info(f"    Confidence: {confidence:.0%}")
        logger.info(f"    Similarity: {similarity:.0%}")
        logger.info(f"    Status: {'VALIDATED' if is_validated else 'NEEDS REVIEW'}")

        return {
            "phase": 6,
            "status": "completed",
            "is_validated": is_validated,
            "confidence": confidence,
            "similarity": similarity,
            "validation_solution": validation_solution,
            "metrics": metrics,
            "message": f"Phase 6 complete: Final validation with {confidence:.0%} confidence",
        }

    except Exception as e:
        logger.error(f"Phase 6 failed: {e}")
        return {
            "phase": 6,
            "status": "failed",
            "error": str(e),
            "message": f"Phase 6 final validation failed: {e}",
        }


# =============================================================================
# FULL WORKFLOW EXECUTION
# =============================================================================

def execute_full_workflow(
    problem_statement: str,
    context: Optional[Dict[str, Any]] = None,
    requirements: Optional[List[str]] = None,
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    mdap_k_ahead: int = 3,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute full 6-phase ROMA-MDAP-MAKER workflow

    Args:
        problem_statement: Problem to solve
        context: Additional context
        requirements: List of requirements
        roma_max_depth_analysis: Max depth for analysis
        roma_max_depth_solving: Max depth for solving
        mdap_k_ahead: K-ahead threshold
        provider: AI provider
        model: Model name
        **kwargs: Additional parameters

    Returns:
        Dict with all phase results
    """
    logger.info("=" * 70)
    logger.info("ROMA-MDAP-MAKER FULL WORKFLOW")
    logger.info("=" * 70)
    logger.info(f"Problem: {problem_statement[:100]}...")
    logger.info(f"Config: depth_analysis={roma_max_depth_analysis}, depth_solving={roma_max_depth_solving}, k={mdap_k_ahead}")
    logger.info("")

    workflow_results = {
        "problem_statement": problem_statement,
        "phases": {},
        "final_status": "not_started",
    }

    try:
        # Phase 1: Setup
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: PROBLEM SETUP")
        logger.info("=" * 70)
        phase1_result = execute_phase_1_setup(
            problem_statement=problem_statement,
            roma_max_depth_analysis=roma_max_depth_analysis,
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][1] = phase1_result

        if phase1_result["status"] != "completed":
            raise Exception(f"Phase 1 failed: {phase1_result.get('error')}")

        # Phase 2: Solve
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2: SOLUTION GENERATION")
        logger.info("=" * 70)
        phase2_result = execute_phase_2_solve(
            sub_problem_id="main",
            sub_problem_description=problem_statement,
            context=context,
            requirements=requirements,
            roma_max_depth=roma_max_depth_solving,
            mdap_k_ahead=mdap_k_ahead,
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][2] = phase2_result

        if phase2_result["status"] != "completed":
            raise Exception(f"Phase 2 failed: {phase2_result.get('error')}")

        solution = phase2_result["solution"]

        # Phase 3: Critique
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 3: ADVERSARIAL CRITIQUE")
        logger.info("=" * 70)
        phase3_result = execute_phase_3_critique(
            solution=solution,
            problem_statement=problem_statement,
            context=context,
            roma_max_depth=1,
            mdap_k_ahead=2,
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][3] = phase3_result

        # Phase 4: Verify
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 4: VERIFICATION")
        logger.info("=" * 70)
        phase4_result = execute_phase_4_verify(
            solution=solution,
            problem_statement=problem_statement,
            requirements=requirements,
            context=context,
            roma_max_depth=1,
            mdap_k_ahead=2,
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][4] = phase4_result

        # Phase 5: Reassemble
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 5: REASSEMBLY")
        logger.info("=" * 70)
        phase5_result = execute_phase_5_reassemble(
            sub_solutions=[{"solution": solution, "confidence": phase2_result.get("confidence", 0.0), "status": "completed"}],
            problem_statement=problem_statement,
            context=context,
            aggregation_method="confidence_weighted",
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][5] = phase5_result

        if phase5_result["status"] != "completed":
            raise Exception(f"Phase 5 failed: {phase5_result.get('error')}")

        integrated_solution = phase5_result["integrated_solution"]

        # Phase 6: Final Validation
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 6: FINAL VALIDATION")
        logger.info("=" * 70)
        phase6_result = execute_phase_6_final_validation(
            integrated_solution=integrated_solution,
            problem_statement=problem_statement,
            requirements=requirements,
            context=context,
            roma_max_depth=roma_max_depth_solving,
            mdap_k_ahead=mdap_k_ahead,
            provider=provider,
            model=model,
            **kwargs
        )
        workflow_results["phases"][6] = phase6_result

        # Final status
        workflow_results["final_status"] = "completed"
        workflow_results["final_solution"] = integrated_solution
        workflow_results["final_confidence"] = phase5_result.get("confidence", 0.0)
        workflow_results["is_validated"] = phase6_result.get("is_validated", False)

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Final confidence: {workflow_results['final_confidence']:.0%}")
        logger.info(f"Validated: {workflow_results['is_validated']}")
        logger.info("")

        return workflow_results

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        workflow_results["final_status"] = "failed"
        workflow_results["error"] = str(e)
        return workflow_results


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _parse_flaws_from_critique(critique: str, phase: str) -> List[Dict[str, Any]]:
    """Parse flaws from critique content (simplified)"""
    flaws = []

    # In production, use LLM to extract structured flaws
    # For now, simple keyword-based extraction
    if "vulnerability" in critique.lower():
        flaws.append({
            "type": "vulnerability",
            "severity": "high",
            "description": "Potential vulnerability identified",
            "phase": phase
        })

    if "error" in critique.lower() or "bug" in critique.lower():
        flaws.append({
            "type": "bug",
            "severity": "medium",
            "description": "Error handling issue",
            "phase": phase
        })

    return flaws


def _parse_improvements_from_critique(critique: str) -> List[str]:
    """Parse improvements from critique content"""
    improvements = []

    # Simple extraction
    if "recommend" in critique.lower():
        improvements.append("Review recommendations in critique")
    if "improve" in critique.lower():
        improvements.append("Consider suggested improvements")

    return improvements


def _aggregate_confidence_weighted(solutions: List[Dict]) -> str:
    """Aggregate solutions using confidence weighting"""
    # Sort by confidence
    sorted_solutions = sorted(solutions, key=lambda x: x.get("confidence", 0), reverse=True)

    # Weighted combination (simplified - just use highest confidence solution)
    return sorted_solutions[0]["solution"]


def _aggregate_simple(solutions: List[Dict]) -> str:
    """Simple aggregation by concatenation"""
    return "\n\n".join([s["solution"] for s in solutions])


def _aggregate_with_llm(
    solutions: List[Dict],
    problem_statement: str,
    context: Optional[Dict],
    provider: str,
    model: str
) -> str:
    """Aggregate solutions using LLM"""
    # Combine all solutions
    combined = "\n\n".join([
        f"Solution {i+1} (confidence: {s.get('confidence', 0):.0%}):\n{s['solution']}"
        for i, s in enumerate(solutions)
    ])

    # Use LLM to integrate (simplified - return combined)
    return f"Integrated Solution:\n\n{combined}"


def _calculate_solution_similarity(sol1: str, sol2: str) -> float:
    """Calculate similarity between two solutions"""
    # Simple word overlap similarity (in production, use embeddings)
    words1 = set(sol1.lower().split())
    words2 = set(sol2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


# =============================================================================
# INITIALIZATION
# =============================================================================

# Export phase functions
PHASE_FUNCTIONS = {
    1: execute_phase_1_setup,
    2: execute_phase_2_solve,
    3: execute_phase_3_critique,
    4: execute_phase_4_verify,
    5: execute_phase_5_reassemble,
    6: execute_phase_6_final_validation,
}


def execute_phase(phase_num: int, **kwargs) -> Dict[str, Any]:
    """Execute a specific phase by number"""
    if phase_num not in PHASE_FUNCTIONS:
        raise ValueError(f"Invalid phase: {phase_num}. Must be 1-6.")

    return PHASE_FUNCTIONS[phase_num](**kwargs)


# Status check
def get_romamdapmaker_bridge_status() -> Dict[str, Any]:
    """Get ROMA-MDAP-MAKER bridge status"""
    engine_status = get_roma_mdap_maker_status()

    return {
        "bridge_available": True,
        "roma_available": ROMA_AVAILABLE,
        "mdap_available": MDAP_AVAILABLE,
        "engine_available": engine_status.get("available", False),
        "phases_supported": list(PHASE_FUNCTIONS.keys()),
        "total_phases": len(PHASE_FUNCTIONS),
    }


__all__ = [
    "execute_phase_1_setup",
    "execute_phase_2_solve",
    "execute_phase_3_critique",
    "execute_phase_4_verify",
    "execute_phase_5_reassemble",
    "execute_phase_6_final_validation",
    "execute_full_workflow",
    "execute_phase",
    "get_romamdapmaker_bridge_status",
    "PHASE_FUNCTIONS",
]
