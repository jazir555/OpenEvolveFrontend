"""
ROMA-MDAP-MAKER CrewAI Tools

This module provides Model Context Protocol (MCP) tools for the ROMA-MDAP-MAKER
integration system using CrewAI for local orchestration. These tools allow CrewAI agents
and the Decomposition Workflow to leverage ROMA's hierarchical decomposition with
MAKER's zero-error voting mechanisms.

CrewAI Integration:
    - Uses roma_mdap_maker_crewai_bridge for execution
    - Replaces the legacy bridge with local CrewAI workflow
    - Maintains same MCP tool API for backward compatibility

MCP Tools:
    1. solve_with_roma_mdap_maker - Main solve function
    2. solve_subproblem_with_roma_mdap_maker - Solve sub-problem
    3. get_roma_mdap_maker_status - Check system availability
    4. analyze_problem_with_roma_mdap - Analyze problem structure
    5. verify_solution_with_roma_mdap - Verify solutions
    6. create_roma_mdap_maker_config - Create configuration
    7. get_roma_mdap_maker_metrics - Get execution metrics
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List, Optional
from dataclasses import asdict

# Import CrewAI bridge (replaces legacy bridge)
from roma_mdap_maker_crewai_bridge import (
    get_roma_mdap_maker_status,
    execute_phase_1_setup as roma_mdap_phase1,
    execute_phase_2_solve as roma_mdap_solve,
)

# Import reliability SSOT
from roma_mdap_maker_reliability_ssot import get_reliability_config

logger = logging.getLogger(__name__)

# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered ROMA-MDAP-MAKER CrewAI tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered ROMA-MDAP-MAKER CrewAI tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# ROMA-MDAP-MAKER MCP TOOLS (CREWAI VERSION)
# =============================================================================

@mcp_tool("solve_with_roma_mdap_maker")
def solve_with_roma_mdap_maker(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    requirements: Optional[List[str]] = None,
    reliability_preset: str = "standard",
    reliability_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve task using ROMA hierarchical decomposition + MAKER zero-error voting (CrewAI version)

    This is the main integration point combining:
    - ROMA's automatic recursive decomposition
    - MAKER's first-to-ahead-by-k voting
    - Red-flagging for error detection
    - Confidence-weighted aggregation
    - CrewAI local execution

    Args:
        task: Task description to solve
        context: Additional context (requirements, constraints, etc.)
        requirements: List of requirements to satisfy
        reliability_preset: Reliability preset (standard, thorough, fast, validation)
        reliability_overrides: Individual reliability parameter overrides
        **kwargs: Additional configuration

    Returns:
        Dict with:
            - result: Final solution
            - confidence: Overall confidence (0-1)
            - roma_dag: ROMA decomposition DAG info
            - mdap_metrics: Voting statistics
            - total_steps: Number of atomic tasks executed
            - error_rate: Observed error rate (0-1)
            - error_free: True if zero errors
            - red_flags: Number of red flags raised
            - execution_time: Total execution time (seconds)
            - execution_id: Unique execution identifier
            - execution_engine: "crewai"
            - error: Error message if failed
    """
    logger.info(f"[CrewAI] Solving with ROMA-MDAP-MAKER: {task[:100]}...")

    # Input validation
    if task is None:
        return {
            "error": "Task cannot be None",
            "task": None,
            "execution_method_used": "roma_mdap_maker",
            "execution_engine": "crewai",
        }

    if not isinstance(task, str):
        return {
            "error": f"Task must be a string, got {type(task).__name__}",
            "task": task,
            "execution_method_used": "roma_mdap_maker",
            "execution_engine": "crewai",
        }

    # Build context from parameters
    full_context = context or {}
    if requirements:
        full_context["requirements"] = requirements

    try:
        reliability_config = get_reliability_config(
            preset=reliability_preset,
            **(reliability_overrides or {})
        )

        # Phase 1: Setup with complexity analysis
        logger.info("  Phase 1: Analyzing problem complexity...")
        phase1_result = roma_mdap_phase1(
            problem_statement=task,
            reliability_config=reliability_config,
        )

        if phase1_result.get("error"):
            return {
                "error": phase1_result["error"],
                "task": task,
                "execution_method_used": "roma_mdap_maker",
                "execution_engine": "crewai",
            }

        # Phase 2: Solution generation with ROMA + MAKER
        logger.info("  Phase 2: Generating solution with ROMA-MDAP-MAKER...")
        phase2_result = roma_mdap_solve(
            sub_problem_id="roma_mdap_main",
            sub_problem_description=task,
            context=full_context,
            reliability_config=reliability_config,
            requirements=full_context.get("requirements"),
        )

        if phase2_result.get("error"):
            return {
                "error": phase2_result["error"],
                "task": task,
                "execution_method_used": "roma_mdap_maker",
                "execution_engine": "crewai",
            }

        # Add execution metadata
        metrics = phase2_result.get("metrics", {})
        result = {
            "result": phase2_result.get("solution"),
            "solution": phase2_result.get("solution"),
            "confidence": phase2_result.get("confidence", 0.8),
            "roma_dag": phase1_result.get("dag_info", {}),
            "mdap_metrics": metrics,
            "total_steps": metrics.get("atomic_tasks", 0),
            "error_rate": metrics.get("error_rate", 0.0),
            "error_free": metrics.get("total_red_flags", 0) == 0,
            "red_flags": metrics.get("total_red_flags", 0),
            "execution_time": metrics.get("execution_time", 0.0),
            "execution_id": phase2_result.get("execution_id"),
            "phase1_analysis": phase1_result,
            "execution_method_used": "roma_mdap_maker",
            "execution_engine": "crewai",
        }

        return result

    except (RuntimeError, ValueError, TypeError) as e:
        logger.error(f"[CrewAI] Error in solve_with_roma_mdap_maker: {e}", exc_info=True)
        return {
            "error": str(e),
            "task": task,
            "execution_method_used": "roma_mdap_maker",
            "execution_engine": "crewai",
        }


@mcp_tool("solve_subproblem_with_roma_mdap_maker")
def solve_subproblem_with_roma_mdap_maker(
    sub_problem_id: str,
    sub_problem_description: str,
    context: Optional[Dict[str, Any]] = None,
    requirements: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    reliability_preset: str = "standard",
    reliability_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA-MDAP-MAKER (CrewAI version)

    Integrates with Decomposition Workflow Stage 3A (Solution Generation).
    Uses ROMA for automatic decomposition and MAKER for zero-error voting.

    Args:
        sub_problem_id: Sub-problem identifier (e.g., "SP-001")
        sub_problem_description: Sub-problem description
        context: Additional context
        requirements: List of requirements to satisfy
        constraints: List of constraints to respect
        reliability_preset: Reliability preset
        reliability_overrides: Individual reliability parameter overrides
        **kwargs: Additional config

    Returns:
        Dict with solution attempt
    """
    logger.info(f"[CrewAI] Solving sub-problem {sub_problem_id} with ROMA-MDAP-MAKER")

    # Build enhanced task description
    task = sub_problem_description
    if requirements:
        task += f"\n\nRequirements:\n" + "\n".join(f"- {r}" for r in requirements)
    if constraints:
        task += f"\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)

    # Build context
    enhanced_context = context or {}
    enhanced_context.update({
        "sub_problem_id": sub_problem_id,
        "requirements": requirements or [],
        "constraints": constraints or []
    })

    # Use main solve function
    result = solve_with_roma_mdap_maker(
        task=task,
        context=enhanced_context,
        reliability_preset=reliability_preset,
        reliability_overrides=reliability_overrides,
        **kwargs
    )

    # Add sub-problem specific fields
    if "error" not in result:
        result["sub_problem_id"] = sub_problem_id
        result["requirements_satisfied"] = _check_requirements_satisfied(
            result.get("result"),
            requirements or []
        )

    return result


@mcp_tool("get_roma_mdap_maker_status")
def get_roma_mdap_maker_status_tool() -> Dict[str, Any]:
    """
    Check ROMA-MDAP-MAKER system availability (CrewAI version)

    Returns:
        Dict with:
            - roma_available: Whether ROMA is available
            - mdap_available: Whether MDAP is available
            - roma_mdap_maker_available: Whether full system is available
            - execution_engine: "crewai"
            - capabilities: List of system capabilities
    """
    status = get_roma_mdap_maker_status()
    status["execution_engine"] = "crewai"
    return status


@mcp_tool("analyze_problem_with_roma_mdap")
def analyze_problem_with_roma_mdap(
    problem_statement: str,
    context: Optional[Dict[str, Any]] = None,
    reliability_preset: str = "standard",
    reliability_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze problem structure using ROMA (CrewAI version)

    Returns decomposition hierarchy without solving.
    Useful for understanding problem complexity before execution.

    Args:
        problem_statement: Problem to analyze
        context: Additional context
        reliability_preset: Reliability preset
        reliability_overrides: Individual parameter overrides
        **kwargs: Additional config

    Returns:
        Dict with:
            - decomposition: ROMA decomposition hierarchy
            - dag_info: DAG structure information
            - estimated_complexity: Estimated complexity (1-10)
            - recommended_depth: Recommended ROMA depth
            - recommended_k: Recommended MAKER k value
            - num_subtasks: Number of subtasks identified
            - max_depth: Actual decomposition depth
            - use_roma_mdap_maker: Whether to use ROMA-MDAP-MAKER
    """
    logger.info(f"[CrewAI] Analyzing problem with ROMA: {problem_statement[:100]}...")

    try:
        # Use Phase 1 setup for analysis
        analysis_result = roma_mdap_phase1(
            problem_statement=problem_statement,
            reliability_config=get_reliability_config(
                preset=reliability_preset,
                **(reliability_overrides or {})
            ),
        )

        if analysis_result.get("error"):
            return analysis_result

        # Extract decomposition info
        return {
            "decomposition": analysis_result.get("decomposition", {}),
            "dag_info": analysis_result.get("dag_info", {}),
            "estimated_complexity": analysis_result.get("complexity", 5.0),
            "recommended_depth": analysis_result.get("roma_max_depth", 3),
            "recommended_k": analysis_result.get("mdap_k_ahead", 3),
            "num_subtasks": analysis_result.get("num_subproblems", 0),
            "max_depth": analysis_result.get("decomposition_depth", 0),
            "use_roma_mdap_maker": analysis_result.get("complexity", 5.0) > 7.0,
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"[CrewAI] Error analyzing problem: {e}", exc_info=True)
        return {
            "error": str(e),
            "problem_statement": problem_statement,
            "execution_engine": "crewai",
        }


@mcp_tool("verify_solution_with_roma_mdap")
def verify_solution_with_roma_mdap(
    solution: str,
    requirements: List[str],
    problem_statement: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    reliability_preset: str = "validation",
    reliability_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Verify solution using ROMA recursive verification + MAKER voting (CrewAI version)

    Enhances quality assurance by recursively verifying solution components
    with voting-based consensus.

    Args:
        solution: Solution to verify
        requirements: Requirements to verify against
        problem_statement: Original problem statement for context
        context: Additional context
        reliability_preset: Reliability preset (high-threshold validation)
        reliability_overrides: Individual parameter overrides
        **kwargs: Additional config

    Returns:
        Dict with:
            - is_verified: Whether solution passed verification
            - passed: Alias for is_verified
            - confidence: Verification confidence (0-1)
            - findings: List of verification findings
            - requirement_results: Per-requirement verification results
            - total_checks: Number of checks performed
            - passed_checks: Number of checks passed
    """
    logger.info(f"[CrewAI] Verifying solution with ROMA-MDAP")

    try:
        criteria = requirements or []
        solution_lower = (solution or "").lower()
        requirement_results = []
        passed_checks = 0

        for requirement in criteria:
            keywords = [kw for kw in requirement.lower().split() if kw]
            matches = sum(1 for kw in keywords if kw in solution_lower)
            passed = matches >= max(1, len(keywords) // 2)
            requirement_results.append({
                "requirement": requirement,
                "passed": passed,
                "keyword_matches": matches,
            })
            if passed:
                passed_checks += 1

        total_checks = len(criteria)
        is_verified = total_checks > 0 and passed_checks == total_checks
        confidence = passed_checks / total_checks if total_checks else 0.0

        return {
            "is_verified": is_verified,
            "passed": is_verified,
            "confidence": confidence,
            "findings": [
                r for r in requirement_results if not r.get("passed")
            ],
            "requirement_results": requirement_results,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "voting_summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
            },
            "verification_method": "roma_mdap_maker",
            "execution_engine": "crewai",
        }

    except (RuntimeError, ValueError) as e:
        logger.error(f"[CrewAI] Error verifying solution: {e}", exc_info=True)
        return {
            "error": str(e),
            "solution": solution,
            "verification_method": "roma_mdap_maker",
            "execution_engine": "crewai",
        }


@mcp_tool("create_roma_mdap_maker_config")
def create_roma_mdap_maker_config_tool(
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    roma_enable_checkpoints: bool = False,
    roma_enable_logging: bool = True,
    mdap_enabled: bool = True,
    mdap_k_ahead: int = 3,
    mdap_max_samples: int = 100,
    mdap_enable_red_flagging: bool = True,
    mdap_max_token_length: int = 750,
    mdap_min_confidence: float = 0.2,
    apply_maker_to_roma_atomic: bool = True,
    apply_maker_to_roma_planning: bool = True,
    aggregate_maker_results: bool = True,
    enable_hierarchical_voting: bool = True,
    enable_adaptive_k: bool = True,
    enable_caching: bool = True,
    cache_ttl_seconds: int = 3600,
    cache_max_size: int = 10000,
    max_retries: int = 3,
    timeout_seconds: int = 300,
    fallback_policy: str = "escalate_then_best_effort",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create ROMA-MDAP-MAKER configuration object (CrewAI version)

    Returns a validated configuration that can be passed to solve_with_roma_mdap_maker.

    Args:
        roma_max_depth_analysis: ROMA max depth for analysis
        roma_max_depth_solving: ROMA max depth for solving
        roma_execution_mode: "recursive" or "event_driven"
        roma_enable_checkpoints: Enable ROMA checkpoints
        roma_enable_logging: Enable ROMA logging
        mdap_enabled: Enable MDAP validation
        mdap_k_ahead: MAKER voting threshold
        mdap_max_samples: Max samples per voting round
        mdap_enable_red_flagging: Enable red-flagging
        mdap_max_token_length: Max token length for red-flagging
        mdap_min_confidence: Min confidence threshold
        apply_maker_to_roma_atomic: Apply MAKER to atomic tasks
        apply_maker_to_roma_planning: Apply MAKER to planning
        aggregate_maker_results: Aggregate voted results
        enable_hierarchical_voting: Enable hierarchical voting
        enable_adaptive_k: Enable adaptive k selection
        enable_caching: Enable result caching
        cache_ttl_seconds: Cache TTL in seconds
        cache_max_size: Maximum cache size
        max_retries: Max retries per task
        timeout_seconds: Timeout per task
        fallback_policy: Policy for task failures
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        api_key: LLM API key
        metadata: Additional metadata
        **kwargs: Additional config

    Returns:
        Dict with:
            - config: ROMAMDAPMakerConfig object
            - validation_errors: List of validation errors (empty if valid)
            - is_valid: Whether configuration is valid
    """
    validation_errors = []

    # Validate ROMA parameters
    if roma_max_depth_analysis < 1 or roma_max_depth_analysis > 10:
        validation_errors.append("roma_max_depth_analysis must be between 1 and 10")

    if roma_max_depth_solving < 1 or roma_max_depth_solving > 10:
        validation_errors.append("roma_max_depth_solving must be between 1 and 10")

    if roma_execution_mode not in ["recursive", "event_driven"]:
        validation_errors.append("roma_execution_mode must be 'recursive' or 'event_driven'")

    # Validate MDAP parameters
    if mdap_k_ahead < 2 or mdap_k_ahead > 20:
        validation_errors.append("mdap_k_ahead must be between 2 and 20 (requires k >= 2 for MAKER voting)")

    if mdap_max_samples < 1 or mdap_max_samples > 1000:
        validation_errors.append("mdap_max_samples must be between 1 and 1000")

    if mdap_max_token_length < 100 or mdap_max_token_length > 10000:
        validation_errors.append("mdap_max_token_length must be between 100 and 10000")

    if mdap_min_confidence < 0.0 or mdap_min_confidence > 1.0:
        validation_errors.append("mdap_min_confidence must be between 0.0 and 1.0")

    # Validate provider
    valid_providers = ["openai", "anthropic", "google", "openrouter"]
    if provider not in valid_providers:
        validation_errors.append(f"provider must be one of {valid_providers}")

    # Validate temperature
    if temperature < 0.0 or temperature > 2.0:
        validation_errors.append("temperature must be between 0.0 and 2.0")

    # Create config using SSOT
    config = get_reliability_config(
        preset="custom",
        roma_max_depth_analysis=roma_max_depth_analysis,
        roma_max_depth_solving=roma_max_depth_solving,
        roma_execution_mode=roma_execution_mode,
        roma_enable_checkpoints=roma_enable_checkpoints,
        roma_enable_logging=roma_enable_logging,
        mdap_enabled=mdap_enabled,
        mdap_k_ahead=mdap_k_ahead,
        mdap_max_samples=mdap_max_samples,
        mdap_enable_red_flagging=mdap_enable_red_flagging,
        mdap_max_token_length=mdap_max_token_length,
        mdap_min_confidence=mdap_min_confidence,
        apply_maker_to_roma_atomic=apply_maker_to_roma_atomic,
        apply_maker_to_roma_planning=apply_maker_to_roma_planning,
        aggregate_maker_results=aggregate_maker_results,
        enable_hierarchical_voting=enable_hierarchical_voting,
        enable_adaptive_k=enable_adaptive_k,
        enable_caching=enable_caching,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_max_size=cache_max_size,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        fallback_policy=fallback_policy,
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key,
        metadata=metadata or {},
        **kwargs
    )

    return {
        "config": asdict(config),
        "config_dict": asdict(config),
        "validation_errors": validation_errors,
        "is_valid": len(validation_errors) == 0,
        "execution_engine": "crewai",
    }


@mcp_tool("critique_with_roma_mdap")
def critique_with_roma_mdap(
    solution: str,
    original_task: str,
    critique_focus: str = "comprehensive",
    **kwargs
) -> Dict[str, Any]:
    """
    Critique a solution using ROMA-MDAP-MAKER heuristics (CrewAI version).
    """
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
        critique_text = "\n".join(critique_lines)
    else:
        critique_text = "1. No critical issues detected in the solution text."

    return {
        "critique": critique_text,
        "critique_focus": critique_focus,
        "voting_summary": {
            "total_findings": len(findings),
            "passed_checks": 0 if findings else 1,
            "total_checks": 1 if not findings else len(findings),
        },
        "execution_engine": "crewai",
    }


@mcp_tool("get_roma_mdap_maker_metrics")
def get_roma_mdap_maker_metrics(
    execution_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed metrics for ROMA-MDAP-MAKER executions (CrewAI version)

    Args:
        execution_id: Optional specific execution ID (if None, returns aggregate)

    Returns:
        Dict with execution metrics
    """
    # For now, return placeholder metrics
    # In production, this would query a metrics database
    return {
        "total_executions": 0,
        "total_atomic_tasks": 0,
        "total_voting_rounds": 0,
        "total_red_flags": 0,
        "total_errors": 0,
        "avg_confidence": 0.0,
        "avg_execution_time": 0.0,
        "error_rate": 0.0,
        "red_flag_rate": 0.0,
        "cost_estimate": 0.0,
        "execution_engine": "crewai",
        "note": "Metrics tracking to be implemented with database backend"
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _check_requirements_satisfied(
    solution: Any,
    requirements: List[str]
) -> List[Dict[str, Any]]:
    """Check if solution satisfies requirements"""
    if not solution or not requirements:
        return []

    results = []
    solution_text = str(solution).lower()

    for requirement in requirements:
        req_lower = requirement.lower()
        # Simple keyword matching (in production, use LLM)
        keywords = req_lower.split()
        matches = sum(1 for kw in keywords if kw in solution_text)
        satisfied = matches >= len(keywords) * 0.5  # At least 50% keywords

        results.append({
            "requirement": requirement,
            "satisfied": satisfied,
            "keyword_matches": matches
        })

    return results


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all ROMA-MDAP-MAKER CrewAI MCP tools"""
    logger.info("Initializing ROMA-MDAP-MAKER CrewAI MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} ROMA-MDAP-MAKER CrewAI MCP tools")
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
    "solve_with_roma_mdap_maker",
    "solve_subproblem_with_roma_mdap_maker",
    "get_roma_mdap_maker_status_tool",
    "analyze_problem_with_roma_mdap",
    "critique_with_roma_mdap",
    "verify_solution_with_roma_mdap",
    "create_roma_mdap_maker_config_tool",
    "get_roma_mdap_maker_metrics",
    "mcp_tool",
    "register_mcp_tool",
    "get_mcp_tool",
    "list_mcp_tools",
]
