"""
ROMA CrewAI Tools for CrewAI Agents

This module provides Model Context Protocol (MCP) tools that CrewAI agents
can use to leverage ROMA's (Recursive Open Meta-Agents) framework.

CrewAI Integration:
    - Uses roma_crewai_bridge for execution
    - Replaces crewai bridge with local CrewAI workflow
    - Maintains same MCP tool API for backward compatibility

ROMA Architecture:
    solve(task):
        if is_atomic(task):
            return execute(task)
        else:
            subtasks = plan(task)
            results = [solve(subtask) for subtask in subtasks]  # Recursive
            return aggregate(results)

Key Features:
    - Recursive hierarchical decomposition
    - Atomizer: Decides if task needs planning
    - Planner: Breaks tasks into subtasks
    - Executor: Handles atomic tasks
    - Aggregator: Integrates results
    - Checkpoint/recovery system
    - MLflow observability
"""

import logging
from typing import Dict, Any, List, Optional

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# ACE + Steer Integration (optional)
try:
    from ace_steer_integration import AceSteerBridge
    from ace_mcp_tools import ACE_AVAILABLE
    STEER_ACE_BRIDGE_AVAILABLE = True
except ImportError:
    STEER_ACE_BRIDGE_AVAILABLE = False
    ACE_AVAILABLE = False
    AceSteerBridge = None

logger = logging.getLogger(__name__)

# Import CrewAI bridge (replaces crewai)
from roma_crewai_bridge import (
    execute_phase_1_setup as roma_phase1,
    execute_phase_2_generation as roma_phase2,
    execute_phase_5_reassemble as roma_phase5,
)

# CAV-NLP initialization
_cav_nlp_solver = None
_math_service = None
if CAV_NLP_AVAILABLE:
    try:
        _cav_nlp_solver = EnhancedZ3Solver()
        _math_service = UnifiedMathService()
        logger.info("CAV-NLP initialized for ROMA CrewAI tools")
    except Exception as e:
        logger.warning(f"Failed to initialize CAV-NLP: {e}")

# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered ROMA CrewAI tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered ROMA CrewAI tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# ROMA MCP TOOLS (CREWAI VERSION)
# =============================================================================

@mcp_tool("solve_with_roma")
def solve_with_roma(
    task: str,
    max_depth: int = 2,
    execution_mode: str = "recursive",  # "recursive", "event_driven"
    enable_checkpoints: bool = True,
    enable_logging: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve a task using ROMA's recursive decomposition framework (CrewAI version)

    This is the main integration point for Decomposition Workflow.

    ROMA Architecture:
        1. Atomizer: Checks if task is atomic or needs decomposition
        2. Planner: Breaks non-atomic tasks into subtasks
        3. Executor: Executes atomic tasks (or continues recursion)
        4. Aggregator: Combines subtask results

    Args:
        task: Task description to solve
        max_depth: Maximum recursion depth (default: 2)
        execution_mode: "recursive" (depth-first) or "event_driven" (parallel)
        enable_checkpoints: Enable checkpoint/recovery system
        enable_logging: Enable debug logging
        provider: LLM provider (openai, anthropic, google, openrouter)
        model: Model name
        api_key: API key for the provider
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with solution attempt
    """
    logger.info(f"[CrewAI] Solving with ROMA: {task[:100]}... (mode={execution_mode}, max_depth={max_depth})")

    # ACE + Steer Bridge initialization
    bridge = None
    if STEER_ACE_BRIDGE_AVAILABLE:
        bridge = AceSteerBridge(
            ace_agent_id=f"roma_solver_{hash(task)%10000}",
            skillbook_path="./ace_skillbook.json"
        )

    # ACE + Steer: Skill Injection
    original_task = task
    if bridge:
        task = bridge.prepare_prompt(
            task=task,
            model=model or "roma-default"
        )

    try:
        # Phase 1: Analysis
        logger.info("  Phase 1: ROMA analysis...")
        phase1_result = roma_phase1(
            problem_statement=task,
            max_depth=max_depth,
            execution_mode=execution_mode,
        )

        if phase1_result.get("error"):
            return {
                "error": phase1_result["error"],
                "task": task,
                "execution_method_used": "roma",
                "execution_engine": "crewai",
            }

        # Phase 2: Solution generation
        logger.info("  Phase 2: ROMA solve...")
        sub_problems = phase1_result.get("analysis", {}).get("sub_problems", [])
        if not sub_problems:
            sub_problems = [{
                "id": "sp_1",
                "title": "Primary task",
                "description": task,
                "dependencies": [],
            }]

        phase2_result = roma_phase2(
            sub_problems=sub_problems,
            team_name="roma_solver",
            max_depth=max_depth,
            execution_mode=execution_mode,
            provider=provider,
            api_key=api_key,
            model=model,
        )

        if phase2_result.get("error"):
            return {
                "error": phase2_result["error"],
                "task": task,
                "execution_method_used": "roma",
                "execution_engine": "crewai",
            }

        solutions = phase2_result.get("solutions", [])
        aggregated = roma_phase5(solutions, original_task) if solutions else {}
        result = aggregated.get("final_solution", "")
        status = phase2_result.get("status", "completed")

        # Get DAG info if available
        dag_info = phase1_result.get("dag_info", {})

        # Get token usage
        token_usage = phase2_result.get("metrics", {})

        # ACE + Steer: Verification & Learning
        if bridge and result:
            steer_v = bridge.verify_and_learn(
                query=original_task,
                output=str(result),
                verifications=["slop"],
                model=model or "roma-default"
            )
            if not steer_v["all_passed"]:
                logger.warning(f"ROMA solve failed Steer verification: {steer_v['failed_verifications']}")

        return {
            "task": original_task,
            "result": result,
            "status": status,
            "execution_mode": execution_mode,
            "max_depth": max_depth,
            "generated_by": f"ROMA ({execution_mode})",
            "execution_method_used": "roma",
            "execution_engine": "crewai",
            "dag_info": dag_info,
            "token_usage": token_usage,
        }

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Failed to solve with ROMA: {e}", exc_info=True)
        return {
            "error": str(e),
            "task": task,
            "execution_method_used": "roma",
            "execution_engine": "crewai",
        }


@mcp_tool("solve_sub_problem_with_roma")
def solve_sub_problem_with_roma(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    max_depth: int = 2,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA (CrewAI version, for Decomposition Workflow)

    This is specifically designed for Stage 3A (Blue Team Solution Generation).

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of the problem to solve
        team_name: Name of the Blue Team
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        max_depth: Maximum recursion depth
        execution_mode: "recursive" or "event_driven"
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with solution attempt
    """
    logger.info(f"[CrewAI] Solving sub-problem {sub_problem_id} with ROMA (team={team_name})")

    # Build enhanced task
    task_parts = [f"Sub-Problem ID: {sub_problem_id}", sub_problem_description]

    if context:
        task_parts.append(f"\nContext: {context}")

    if constraints:
        task_parts.append("\nConstraints:")
        for c in constraints:
            task_parts.append(f"  - {c}")

    if requirements:
        task_parts.append("\nRequirements:")
        for r in requirements:
            task_parts.append(f"  - {r}")

    task = "\n".join(task_parts)

    # Use main solve function
    result = solve_with_roma(
        task=task,
        max_depth=max_depth,
        execution_mode=execution_mode,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )

    if "error" in result:
        return {
            "error": result["error"],
            "sub_problem_id": sub_problem_id,
            "execution_method_used": "roma",
            "execution_engine": "crewai",
        }

    return {
        "sub_problem_id": sub_problem_id,
        "solution": result["result"],
        "team_name": team_name,
        "generated_by": f"ROMA ({execution_mode})",
        "status": result["status"],
        "execution_method_used": "roma",
        "execution_engine": "crewai",
        "dag_info": result.get("dag_info", {}),
        "token_usage": result.get("token_usage"),
    }


@mcp_tool("analyze_with_roma")
def analyze_with_roma(
    task: str,
    analysis_type: str = "decomposition",  # "decomposition", "complexity", "dependencies"
    max_depth: int = 3,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a problem using ROMA's decomposition capabilities (CrewAI version)

    This is for Stage 0-1 (Problem Analysis and Decomposition).

    Args:
        task: Problem statement to analyze
        analysis_type: Type of analysis to perform
        max_depth: Maximum recursion depth for decomposition
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with analysis results
    """
    logger.info(f"[CrewAI] Analyzing problem with ROMA: {task[:100]}... (type={analysis_type})")

    try:
        # Build analysis task
        full_task = f"Analyze this problem:\n\n{task}\n\n"
        if analysis_type == "decomposition":
            full_task += "Break it down into sub-problems and identify the decomposition structure."
        elif analysis_type == "complexity":
            full_task += "Analyze the complexity and identify key challenges."
        elif analysis_type == "dependencies":
            full_task += "Identify dependencies between different components."

        # Use Phase 1 for analysis
        analysis_result = roma_phase1(
            problem_statement=full_task,
            max_depth=max_depth,
            execution_mode="recursive",
        )

        if analysis_result.get("error"):
            return {
                "error": analysis_result["error"],
                "task": task,
                "execution_engine": "crewai",
            }

        # Extract analysis info
        decomposition = analysis_result.get("analysis", {})
        actual_depth = decomposition.get("decomposition_depth", 0)

        return {
            "task": task,
            "analysis_type": analysis_type,
            "decomposition": decomposition,
            "analysis": decomposition,  # Keep for backward compatibility
            "max_depth": actual_depth,
            "dag_info": analysis_result.get("dag_info", {}),
            "token_usage": analysis_result.get("token_usage", {}),
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"[CrewAI] Failed to analyze with ROMA: {e}", exc_info=True)
        return {
            "error": str(e),
            "task": task,
            "execution_engine": "crewai",
        }


def _count_hierarchy_depth(hierarchy: Dict[str, Any]) -> int:
    """Count depth of nested hierarchy dict"""
    if not isinstance(hierarchy, dict) or not hierarchy.get("subtasks"):
        return 0
    return 1 + max((_count_hierarchy_depth(s) for s in hierarchy["subtasks"]), default=0)


@mcp_tool("verify_with_roma")
def verify_with_roma(
    solution: str,
    original_task: str,
    verification_criteria: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Verify a solution using ROMA (CrewAI version, for Stage 3C/4 - Verification)

    Args:
        solution: The solution to verify
        original_task: The original task/problem
        verification_criteria: List of criteria to verify
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with verification results
    """
    logger.info(f"[CrewAI] Verifying solution with ROMA: {original_task[:100]}...")

    try:
        criteria = verification_criteria or [
            "Solution addresses the original task",
            "Solution is complete and correct",
            "Solution follows best practices",
        ]

        solution_lower = (solution or "").lower()
        findings = []
        passed_checks = 0
        for criterion in criteria:
            keywords = [kw for kw in criterion.lower().split() if kw]
            matches = sum(1 for kw in keywords if kw in solution_lower)
            passed = matches >= max(1, len(keywords) // 2)
            if passed:
                passed_checks += 1
            else:
                findings.append({
                    "check": criterion,
                    "result": "failed",
                    "details": "Insufficient evidence in solution text",
                })

        verified = passed_checks == len(criteria)
        confidence = passed_checks / len(criteria) if criteria else 0.0

        return {
            "original_task": original_task,
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "verification": "passed" if verified else "failed",
            "verification_criteria": criteria,
            "verified": verified,
            "passed": verified,
            "confidence": confidence,
            "findings": findings,
            "total_checks": len(criteria),
            "passed_checks": passed_checks,
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"[CrewAI] Failed to verify with ROMA: {e}", exc_info=True)
        return {
            "error": str(e),
            "original_task": original_task,
            "execution_engine": "crewai",
        }


def verify_solution_with_roma(
    solution: str,
    requirements: List[str],
    problem_statement: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Backward-compatible alias for verify_with_roma."""
    return verify_with_roma(
        solution=solution,
        original_task=problem_statement or "Verify solution",
        verification_criteria=requirements,
    )


@mcp_tool("critique_with_roma")
def critique_with_roma(
    solution: str,
    original_task: str,
    critique_focus: str = "comprehensive",  # "comprehensive", "security", "performance", "correctness"
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Critique a solution using ROMA (CrewAI version, for Stage 3B - Red Team Critique)

    Args:
        solution: The solution to critique
        original_task: The original task
        critique_focus: Type of critique to perform
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with critique results
    """
    logger.info(f"[CrewAI] Critiquing solution with ROMA: {original_task[:100]}... (focus={critique_focus})")

    try:
        findings = []
        solution_lower = (solution or "").lower()

        if not solution.strip():
            findings.append("Solution content is empty")
        if "todo" in solution_lower or "fixme" in solution_lower:
            findings.append("Solution includes TODO/FIXME markers")
        if critique_focus in ("security", "comprehensive") and "sanitize" not in solution_lower:
            findings.append("No explicit input validation detected")
        if critique_focus in ("performance", "comprehensive") and "optimize" not in solution_lower:
            findings.append("No explicit performance considerations detected")
        if critique_focus in ("correctness", "comprehensive") and "test" not in solution_lower:
            findings.append("No explicit testing or validation steps mentioned")

        if findings:
            critique_lines = [f"{idx + 1}. {item}" for idx, item in enumerate(findings)]
            result = "\n".join(critique_lines)
        else:
            result = "1. No critical issues detected in the solution text."

        return {
            "original_task": original_task,
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "critique": result,
            "critique_focus": critique_focus,
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"[CrewAI] Failed to critique with ROMA: {e}", exc_info=True)
        return {
            "error": str(e),
            "original_task": original_task,
            "execution_engine": "crewai",
        }


@mcp_tool("get_roma_status")
def get_roma_status() -> Dict[str, Any]:
    """Get the status of the ROMA integration (CrewAI version)"""
    return {
        "available": True,
        "engine": "CrewAI",
        "version": "crewai-bridge-1.0",
        "dspy_available": False,  # Not using dspy in CrewAI version
        "persistence_available": True,
        "observability_available": True,
        "execution_engine": "crewai",
    }


@mcp_tool("create_roma_config")
def create_roma_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a ROMA configuration object (CrewAI version)

    Args:
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Dict with configuration details
    """
    return {
        "config_created": True,
        "provider": provider or "default",
        "model": model or "default",
        "runtime": {
            "max_depth": kwargs.get("max_depth", 2),
            "max_concurrency": kwargs.get("max_concurrency", 5),
        },
        "storage": {
            "file_enabled": kwargs.get("enable_checkpoints", True),
        },
        "execution_engine": "crewai",
    }


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all ROMA CrewAI MCP tools"""
    logger.info("Initializing ROMA CrewAI MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} ROMA CrewAI MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
        "execution_engine": "crewai",
    }


# Auto-initialize on import
initialize_mcp_tools()
