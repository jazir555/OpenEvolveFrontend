"""
Decomposition Workflow CrewAI Tools

This module provides Model Context Protocol (MCP) tools that CrewAI agents
can use to execute the Sovereign-Grade Decomposition Workflow.

CrewAI Integration:
    - Uses decomposition_crewai_bridge for execution
    - Replaces Hephaestus bridge with local CrewAI workflow
    - Maintains same MCP tool API for backward compatibility

CRITICAL ARCHITECTURE:
    CrewAI (Orchestrator) -> Decomposition Workflow -> CrewAI Zero-Error Engine

The Decomposition Workflow leverages CrewAI's zero-error workflow for:
    - Problem analysis and decomposition
    - Team-based solution generation (Blue, Red, Gold teams)
    - Gauntlet critiques and verification
    - Multi-stage workflow (Stages 0-6)

Architecture:
    CrewAI Agent -> MCP Tool -> Decomposition Bridge -> CrewAI Zero-Error Workflow -> Result
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Import CrewAI bridge (replaces Hephaestus)
from decomposition_crewai_bridge import (
    execute_phase_1_setup as decomp_phase1,
    execute_phase_2_generation as decomp_phase2,
    execute_phase_3_critique as decomp_phase3,
    execute_phase_4_verification as decomp_phase4,
    execute_phase_5_reassembly as decomp_phase5,
    execute_phase_6_validation as decomp_phase6,
)

try:
    from decomposition_mcp_tools import get_mcp_tool_inventory as _get_mcp_tool_inventory
except Exception:
    _get_mcp_tool_inventory = None

# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered Decomposition CrewAI tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered Decomposition CrewAI tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# STAGE 0: CONTENT ANALYSIS TOOLS
# =============================================================================

@mcp_tool("analyze_problem_for_decomposition")
def analyze_problem_for_decomposition(
    problem_statement: str,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 20,
) -> Dict[str, Any]:
    """
    Analyze a problem statement to extract structured context for decomposition (CrewAI version)

    This is used by CrewAI Phase 1 agents for content analysis (Stage 0).

    Args:
        problem_statement: The problem to analyze
        problem_type: Type of problem (optimization, design, research, etc.)
        domain: Problem domain (software, mathematics, system design, etc.)
        use_evolution: Whether to use evolutionary processing (CrewAI has this built-in)
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with analysis results:
        {
            "domain": str,
            "complexity": Dict[str, int],
            "constraints": List[str],
            "success_criteria": List[str],
            "estimated_sub_problems": int,
            "required_expertise": List[str],
            "execution_engine": "crewai"
        }
    """
    logger.info(f"[CrewAI] Analyzing problem for decomposition: {problem_statement[:100]}...")

    try:
        # Use Phase 1 setup for analysis
        phase1_result = decomp_phase1(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )

        if phase1_result.get("error"):
            return {
                "error": phase1_result["error"],
                "domain": domain or "Unknown",
                "complexity": {"overall": 5},
                "execution_engine": "crewai",
            }

        # Extract analysis
        analysis = phase1_result.get("analysis", {})

        return {
            "domain": analysis.get("domain", domain or "General"),
            "complexity": analysis.get("complexity", {
                "overall": 5,
                "cognitive": 5,
                "computational": 5,
                "domain": 5,
                "integration": 5,
            }),
            "constraints": analysis.get("constraints", []),
            "success_criteria": analysis.get("success_criteria", []),
            "estimated_sub_problems": analysis.get("estimated_sub_problems", 5),
            "required_expertise": analysis.get("required_expertise", []),
            "key_challenges": analysis.get("challenges", []),
            "evolution_metrics": phase1_result.get("evolution_metrics"),
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Problem analysis failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "domain": domain or "Unknown",
            "complexity": {"overall": 5},
            "execution_engine": "crewai",
        }


# =============================================================================
# STAGE 1: DECOMPOSITION TOOLS
# =============================================================================

@mcp_tool("decompose_problem_into_sub_problems")
def decompose_problem_into_sub_problems(
    problem_statement: str,
    analysis: Optional[Dict[str, Any]] = None,
    max_sub_problems: int = 15,
    decomposition_strategy: str = "semantic",
    complexity_target: int = 5,
    use_evolution: bool = True,
    evolution_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Decompose a complex problem into solvable sub-problems (CrewAI version)

    This is used by CrewAI Phase 1 agents for AI-assisted decomposition (Stage 1).

    Args:
        problem_statement: The problem to decompose
        analysis: Problem analysis from analyze_problem_for_decomposition()
        max_sub_problems: Maximum number of sub-problems to create
        decomposition_strategy: Strategy to use ("semantic", "hierarchical", "flow")
        complexity_target: Target complexity per sub-problem (1-10)
        use_evolution: Whether to use evolutionary processing
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with decomposition results:
        {
            "sub_problems": List[Dict],
            "dependencies": Dict[str, List[str]],
            "estimated_total_complexity": int,
            "decomposition_strategy": str,
            "execution_engine": "crewai"
        }
    """
    logger.info(f"[CrewAI] Decomposing problem using {decomposition_strategy} strategy (evolution={use_evolution})")

    try:
        # Use Phase 1 setup for decomposition
        phase1_result = decomp_phase1(
            problem_statement=problem_statement,
            max_sub_problems=max_sub_problems,
            decomposition_strategy=decomposition_strategy,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )

        if phase1_result.get("error"):
            return {
                "error": phase1_result["error"],
                "sub_problems": [],
                "dependencies": {},
                "execution_engine": "crewai",
            }

        # Extract decomposition plan
        decomposition_plan = phase1_result.get("decomposition_plan", {})

        return {
            "sub_problems": decomposition_plan.get("sub_problems", []),
            "dependencies": decomposition_plan.get("dependencies", {}),
            "estimated_total_complexity": decomposition_plan.get("total_complexity", 0),
            "decomposition_strategy": decomposition_strategy,
            "decomposition_depth": decomposition_plan.get("decomposition_depth", 0),
            "evolution_metrics": phase1_result.get("evolution_metrics"),
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Decomposition failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "sub_problems": [],
            "dependencies": {},
            "execution_engine": "crewai",
        }


@mcp_tool("create_decomposition_plan")
def create_decomposition_plan(
    problem_statement: str,
    sub_problems: List[Dict[str, Any]],
    dependencies: Optional[Dict[str, List[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a formal decomposition plan (CrewAI version)

    Args:
        problem_statement: The original problem
        sub_problems: List of sub-problem dictionaries
        dependencies: Dependency mapping between sub-problems
        metadata: Additional metadata

    Returns:
        Dict with decomposition plan
    """
    logger.info(f"[CrewAI] Creating decomposition plan with {len(sub_problems)} sub-problems")

    return {
        "problem_statement": problem_statement,
        "sub_problems": sub_problems,
        "dependencies": dependencies or {},
        "metadata": metadata or {},
        "total_sub_problems": len(sub_problems),
        "execution_engine": "crewai",
    }


# =============================================================================
# STAGE 3A: BLUE TEAM SOLUTION GENERATION
# =============================================================================

@mcp_tool("solve_sub_problem_with_team")
def solve_sub_problem_with_team(
    sub_problem_id: str,
    sub_problem_title: str,
    sub_problem_description: str,
    team_type: str = "blue",  # "blue", "red", "gold"
    context: Optional[Dict[str, Any]] = None,
    requirements: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    solution_approach: Optional[str] = None,
    use_roma_mdap: bool = False,
    use_evolution: bool = True,
    evolution_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using a team approach (CrewAI version)

    This is used by CrewAI Phase 2 agents for Stage 3A (Blue Team Solution Generation).

    Args:
        sub_problem_id: Sub-problem identifier
        sub_problem_title: Sub-problem title
        sub_problem_description: Sub-problem description
        team_type: Type of team ("blue", "red", "gold")
        context: Additional context
        requirements: Requirements to satisfy
        constraints: Constraints to respect
        solution_approach: Specific solution approach to use
        use_roma_mdap: Whether to use ROMA-MDAP-MAKER for solving
        use_evolution: Whether to use evolutionary processing
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with solution attempt
    """
    logger.info(f"[CrewAI] Solving sub-problem {sub_problem_id} with {team_type} team")

    try:
        # Build problem context
        problem_context = {
            "sub_problem_id": sub_problem_id,
            "title": sub_problem_title,
            "description": sub_problem_description,
            "team_type": team_type,
            "context": context or {},
            "requirements": requirements or [],
            "constraints": constraints or [],
            "solution_approach": solution_approach,
        }

        if use_roma_mdap:
            try:
                from roma_mdap_maker_crewai_bridge import execute_phase_2_solve as roma_mdap_solve
                phase2_result = roma_mdap_solve(
                    sub_problem_id=sub_problem_id,
                    sub_problem_description=sub_problem_description,
                    context=problem_context,
                    requirements=requirements,
                    constraints=constraints,
                )
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error(f"[CrewAI] ROMA-MDAP-MAKER solve failed: {e}", exc_info=True)
                return {
                    "error": str(e),
                    "sub_problem_id": sub_problem_id,
                    "team_type": team_type,
                    "execution_engine": "crewai",
                }
        else:
            decomposition_plan = {
                "analysis": {
                    "problem_statement": sub_problem_description,
                },
                "sub_problems": [
                    {
                        "id": sub_problem_id,
                        "title": sub_problem_title or sub_problem_id,
                        "description": sub_problem_description,
                        "dependencies": [],
                    }
                ],
            }

            phase2_result = decomp_phase2(
                decomposition_plan=decomposition_plan,
                team_name=f"{team_type}_team",
                solve_subset=[sub_problem_id],
                use_evolution=use_evolution,
                evolution_iterations=evolution_iterations,
            )

        if phase2_result.get("error"):
            return {
                "error": phase2_result["error"],
                "sub_problem_id": sub_problem_id,
                "team_type": team_type,
                "execution_engine": "crewai",
            }

        solutions = phase2_result.get("solutions") or []
        solution_entry = solutions[0] if solutions else {}

        return {
            "sub_problem_id": sub_problem_id,
            "solution": solution_entry.get("solution") or phase2_result.get("solution"),
            "team_type": team_type,
            "confidence": solution_entry.get("confidence", phase2_result.get("confidence", 0.8)),
            "approach_used": solution_approach or "default",
            "requirements_satisfied": phase2_result.get("requirements_satisfied", []),
            "evolution_metrics": phase2_result.get("metrics"),
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Sub-problem solving failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "sub_problem_id": sub_problem_id,
            "team_type": team_type,
            "execution_engine": "crewai",
        }


# =============================================================================
# STAGE 3B: RED TEAM GAUNTLET (ADVERSARIAL CRITIQUE)
# =============================================================================

@mcp_tool("critique_solution_with_gauntlet")
def critique_solution_with_gauntlet(
    solution: str,
    problem_statement: str,
    gauntlet_type: str = "comprehensive",  # "comprehensive", "security", "performance", "correctness"
    critique_focus: Optional[List[str]] = None,
    severity: str = "thorough",  # "quick", "standard", "thorough"
    team_type: str = "red",
    use_evolution: bool = True,
    evolution_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Critique a solution using gauntlet (CrewAI version)

    This is used by CrewAI Phase 3 agents for Stage 3B (Red Team Gauntlet).

    Args:
        solution: The solution to critique
        problem_statement: The original problem
        gauntlet_type: Type of gauntlet to run
        critique_focus: Specific focus areas for critique
        severity: Critique thoroughness level
        team_type: Team running the critique
        use_evolution: Whether to use evolutionary processing
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with critique results
    """
    logger.info(f"[CrewAI] Running {gauntlet_type} gauntlet critique (severity={severity})")

    try:
        phase3_result = decomp_phase3(
            solutions=[
                {
                    "id": "gauntlet_target",
                    "solution": solution,
                    "problem_statement": problem_statement,
                }
            ],
            gauntlet_type=gauntlet_type,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )

        if phase3_result.get("error"):
            return {
                "error": phase3_result["error"],
                "gauntlet_type": gauntlet_type,
                "execution_engine": "crewai",
            }

        critiques = phase3_result.get("critiques", [])
        critique_entry = critiques[0] if critiques else {}
        findings = critique_entry.get("findings", [])
        return {
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "critique": critique_entry.get("critique") or critique_entry.get("summary", ""),
            "issues_found": findings,
            "severity": severity,
            "gauntlet_type": gauntlet_type,
            "team_type": team_type,
            "improvement_suggestions": [],
            "evolution_metrics": phase3_result.get("metrics"),
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Gauntlet critique failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "gauntlet_type": gauntlet_type,
            "execution_engine": "crewai",
        }


# =============================================================================
# STAGE 3C: GOLD TEAM GAUNTLET (VERIFICATION)
# =============================================================================

@mcp_tool("verify_solution_with_gauntlet")
def verify_solution_with_gauntlet(
    solution: str,
    problem_statement: str,
    requirements: List[str],
    verification_criteria: Optional[List[str]] = None,
    verification_depth: str = "standard",  # "quick", "standard", "thorough"
    team_type: str = "gold",
    use_romamdap: bool = False,
) -> Dict[str, Any]:
    """
    Verify a solution using gauntlet (CrewAI version)

    This is used by CrewAI Phase 4 agents for Stage 3C (Gold Team Gauntlet).

    Args:
        solution: The solution to verify
        problem_statement: The original problem
        requirements: Requirements to verify against
        verification_criteria: Additional verification criteria
        verification_depth: Verification thoroughness
        team_type: Team running the verification
        use_romamdap: Whether to use ROMA-MDAP-MAKER for verification

    Returns:
        Dict with verification results
    """
    logger.info(f"[CrewAI] Running gauntlet verification (depth={verification_depth})")

    try:
        if use_romamdap:
            from roma_mdap_maker_crewai_tools import verify_solution_with_roma_mdap
            roma_result = verify_solution_with_roma_mdap(
                solution=solution,
                requirements=requirements,
                problem_statement=problem_statement,
            )
            if roma_result.get("error"):
                return {
                    "error": roma_result["error"],
                    "verification_depth": verification_depth,
                    "execution_engine": "crewai",
                }
            return {
                "solution": solution[:200] + "..." if len(solution) > 200 else solution,
                "is_verified": roma_result.get("is_verified", False),
                "passed": roma_result.get("passed", False),
                "confidence": roma_result.get("confidence", 0.5),
                "verification_criteria": verification_criteria or [],
                "requirement_results": roma_result.get("requirement_results", []),
                "findings": roma_result.get("findings", []),
                "total_checks": roma_result.get("total_checks", 0),
                "passed_checks": roma_result.get("passed_checks", 0),
                "verification_depth": verification_depth,
                "team_type": team_type,
                "execution_engine": "crewai",
            }

        # Use Phase 4 for verification
        phase4_result = decomp_phase4(
            solutions=[
                {
                    "id": "gauntlet_target",
                    "solution": solution,
                    "problem_statement": problem_statement,
                    "requirements": requirements,
                }
            ],
            requirements=requirements,
        )

        if phase4_result.get("error"):
            return {
                "error": phase4_result["error"],
                "verification_depth": verification_depth,
                "execution_engine": "crewai",
            }

        verifications = phase4_result.get("verifications", [])
        verification_entry = verifications[0] if verifications else {}
        return {
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "is_verified": verification_entry.get("is_verified", False),
            "passed": verification_entry.get("is_verified", False),
            "confidence": verification_entry.get("verification_score", 0.5),
            "verification_criteria": verification_criteria or [],
            "requirement_results": verification_entry.get("criteria_results", []),
            "findings": verification_entry.get("report", {}).get("findings", []),
            "total_checks": verification_entry.get("report", {}).get("total_checks", 0),
            "passed_checks": verification_entry.get("report", {}).get("passed_checks", 0),
            "verification_depth": verification_depth,
            "team_type": team_type,
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Gauntlet verification failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "verification_depth": verification_depth,
            "execution_engine": "crewai",
        }


# =============================================================================
# UTILITY TOOLS
# =============================================================================

@mcp_tool("list_available_teams")
def list_available_teams() -> Dict[str, Any]:
    """List available teams for solving"""
    return {
        "teams": [
            {
                "name": "blue",
                "description": "Solution generation team",
                "capabilities": ["solve", "generate", "implement"],
            },
            {
                "name": "red",
                "description": "Adversarial critique team",
                "capabilities": ["critique", "attack", "find_issues"],
            },
            {
                "name": "gold",
                "description": "Verification and validation team",
                "capabilities": ["verify", "validate", "test"],
            },
        ],
        "total_teams": 3,
        "execution_engine": "crewai",
    }


@mcp_tool("list_available_gauntlets")
def list_available_gauntlets() -> Dict[str, Any]:
    """List available gauntlets for critique and verification"""
    return {
        "gauntlets": [
            {
                "name": "comprehensive",
                "description": "Full comprehensive critique",
                "stage": "3b",
            },
            {
                "name": "security",
                "description": "Security-focused critique",
                "stage": "3b",
            },
            {
                "name": "performance",
                "description": "Performance-focused critique",
                "stage": "3b",
            },
            {
                "name": "correctness",
                "description": "Correctness verification",
                "stage": "3c",
            },
        ],
        "total_gauntlets": 4,
        "execution_engine": "crewai",
    }


@mcp_tool("get_decomposition_status")
def get_decomposition_status() -> Dict[str, Any]:
    """Get decomposition workflow status (CrewAI version)"""
    web3_tools: List[str] = []
    web3_ingestion_tools: List[str] = []
    web3_formal_tools: List[str] = []
    formal_capabilities: Dict[str, bool] = {
        "solidity_invariant_translation": False,
        "invariant_translation_verification": False,
        "symbolic_exploit_witness": False,
        "composite_exploit_verification": False,
    }

    if _get_mcp_tool_inventory is not None:
        try:
            inventory = _get_mcp_tool_inventory() or {}
            web3_tools = list(inventory.get("web3_tools", []) or [])
            web3_ingestion_tools = list(inventory.get("web3_ingestion_tools", []) or [])
            web3_formal_tools = list(inventory.get("web3_formal_tools", []) or [])
            existing_capabilities = inventory.get("formal_capabilities")
            if isinstance(existing_capabilities, dict):
                formal_capabilities.update(existing_capabilities)
        except Exception as exc:
            logger.debug("Unable to load MCP tool inventory for CrewAI tools status: %s", exc)

    if not web3_ingestion_tools:
        inferred_ingestion_tools = sorted(
            tool
            for tool in web3_tools
            if tool
            in {
                "web3_ingest_contract_audit_stack",
                "web3_ingest_slither_static_analysis",
                "web3_ingest_foundry_fuzzing",
            }
        )
        web3_ingestion_tools = inferred_ingestion_tools

    if not web3_formal_tools:
        if formal_capabilities.get("solidity_invariant_translation"):
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities.get("symbolic_exploit_witness"):
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities.get("composite_exploit_verification"):
            web3_formal_tools.append("z3_web3_audit_exploit_verification")
    web3_formal_tools = sorted(set(web3_formal_tools))

    if not web3_ingestion_tools:
        web3_ingestion_tools = sorted(
            {
                "web3_ingest_contract_audit_stack",
                "web3_ingest_slither_static_analysis",
                "web3_ingest_foundry_fuzzing",
            }
        )
    web3_ingestion_tools = sorted(set(web3_ingestion_tools))

    if not web3_tools:
        web3_tools = sorted(
            {
                *web3_ingestion_tools,
                *web3_formal_tools,
            }
        )
    web3_tools = sorted(set(web3_tools + web3_ingestion_tools + web3_formal_tools))
    web3_formal_available = bool(web3_formal_tools) or any(
        bool(v) for v in formal_capabilities.values()
    )

    return {
        "available": True,
        "engine": "CrewAI",
        "version": "crewai-bridge-1.0",
        "stages": {
            "stage_0": "Content Analysis",
            "stage_1": "Decomposition",
            "stage_3a": "Blue Team Solving",
            "stage_3b": "Red Team Gauntlet",
            "stage_3c": "Gold Team Gauntlet",
            "stage_4": "Reassembly",
            "stage_5": "Final Verification",
            "stage_6": "Knowledge Extraction",
        },
        "total_stages": 7,
        "capabilities": [
            "problem_analysis",
            "decomposition",
            "team_solving",
            "gauntlet_critique",
            "gauntlet_verification",
            "reassembly",
            "validation",
        ],
        "web3_tools": web3_tools,
        "web3_ingestion_tools": web3_ingestion_tools,
        "web3_formal_tools": web3_formal_tools,
        "formal_capabilities": formal_capabilities,
        "web3_ingestion_available": bool(web3_ingestion_tools),
        "web3_formal_available": web3_formal_available,
        "web3_formal_verification_available": web3_formal_available,
        "audit_exploit_verification_available": bool(
            formal_capabilities.get("composite_exploit_verification")
        ),
        "web3_domain_extension_available": bool(web3_tools),
        "execution_engine": "crewai",
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all Decomposition CrewAI MCP tools"""
    logger.info("Initializing Decomposition CrewAI MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} Decomposition CrewAI MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
        "execution_engine": "crewai",
    }


# Auto-initialize on import
initialize_mcp_tools()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "analyze_problem_for_decomposition",
    "decompose_problem_into_sub_problems",
    "create_decomposition_plan",
    "solve_sub_problem_with_team",
    "critique_solution_with_gauntlet",
    "verify_solution_with_gauntlet",
    "list_available_teams",
    "list_available_gauntlets",
    "get_decomposition_status",
    "mcp_tool",
    "register_mcp_tool",
    "get_mcp_tool",
    "list_mcp_tools",
]
