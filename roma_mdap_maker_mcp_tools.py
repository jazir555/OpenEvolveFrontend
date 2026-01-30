"""
ROMA-MDAP-MAKER MCP Tools

This module provides Model Context Protocol (MCP) tools for the ROMA-MDAP-MAKER
integration system. These tools allow CREWAI agents and the Decomposition
Workflow to leverage ROMA's hierarchical decomposition with MAKER's zero-error
voting mechanisms.

MCP Tools:
    1. solve_with_roma_mdap_maker - Main solve function
    2. solve_subproblem_with_roma_mdap_maker - Solve sub-problem (for Decomposition Workflow)
    3. get_roma_mdap_maker_status - Check system availability
    4. analyze_problem_with_roma_mdap - Analyze problem structure
    5. verify_solution_with_roma_mdap - Verify solutions
    6. create_roma_mdap_maker_config - Create configuration
    7. get_roma_mdap_maker_metrics - Get execution metrics
"""

import logging
from typing import Dict, Any, List, Optional, Union

from roma_mdap_maker_engine import (
    ROMAMDAPMakerEngine,
    ROMAMDAPMakerConfig,
    create_roma_mdap_maker_config,
    get_roma_mdap_maker_status,
    ROMA_AVAILABLE,
)

logger = logging.getLogger(__name__)

# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered ROMA-MDAP-MAKER MCP tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered ROMA-MDAP-MAKER MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# ROMA-MDAP-MAKER MCP TOOLS
# =============================================================================

@mcp_tool("solve_with_roma_mdap_maker")
def solve_with_roma_mdap_maker(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    mdap_k_ahead: int = 3,
    mdap_max_samples: int = 100,
    mdap_enable_red_flagging: bool = True,
    enable_adaptive_k: bool = True,
    enable_caching: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_retries: int = 3,
    timeout_seconds: int = 300,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve task using ROMA hierarchical decomposition + MAKER zero-error voting

    This is the main integration point combining:
    - ROMA's automatic recursive decomposition
    - MAKER's first-to-ahead-by-k voting
    - Red-flagging for error detection
    - Confidence-weighted aggregation

    Args:
        task: Task description to solve
        context: Additional context (requirements, constraints, etc.)
        roma_max_depth_analysis: ROMA max depth for analysis (default: 3)
        roma_max_depth_solving: ROMA max depth for solving (default: 2)
        roma_execution_mode: "recursive" or "event_driven" (default: "recursive")
        mdap_k_ahead: MAKER voting threshold k (default: 3)
        mdap_max_samples: Max samples per voting round (default: 100)
        mdap_enable_red_flagging: Enable red-flagging (default: True)
        enable_adaptive_k: Enable adaptive k selection (default: True)
        enable_caching: Enable result caching (default: True)
        provider: LLM provider (default: "openai")
        model: Model name (default: "gpt-4o-mini")
        api_key: API key (optional if set in environment)
        temperature: Sampling temperature (default: 0.1)
        max_retries: Max retries per task (default: 3)
        timeout_seconds: Timeout per task (default: 300)
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
            - error: Error message if failed

    Example:
        >>> result = solve_with_roma_mdap_maker(
        ...     task="Design a scalable authentication system",
        ...     roma_max_depth_solving=2,
        ...     mdap_k_ahead=3
        ... )
        >>> print(f"Solution: {result['result']}")
        >>> print(f"Error-free: {result['error_free']}")
        >>> print(f"Confidence: {result['confidence']}")
    """
    # Validate inputs
    if task is None:
        return {
            "error": "Task cannot be None",
            "task": None,
            "execution_method_used": "roma_mdap_maker",
        }

    if not isinstance(task, str):
        return {
            "error": f"Task must be a string, got {type(task).__name__}",
            "task": task,
            "execution_method_used": "roma_mdap_maker",
        }

    # Validate mdap_k_ahead
    if mdap_k_ahead < 2:
        return {
            "error": f"mdap_k_ahead must be at least 2 for voting, got {mdap_k_ahead}",
            "task": task,
            "execution_method_used": "roma_mdap_maker",
        }

    if mdap_k_ahead > 20:
        return {
            "error": f"mdap_k_ahead too large (max 20), got {mdap_k_ahead}",
            "task": task,
            "execution_method_used": "roma_mdap_maker",
        }

    logger.info(f"Solving with ROMA-MDAP-MAKER: {task[:100]}...")

    # Check ROMA availability
    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available - falling back to basic execution",
            "task": task,
            "execution_method_used": "roma_mdap_maker",
            "fallback": True
        }

    try:
        # Create configuration
        config = create_roma_mdap_maker_config(
            roma_max_depth_analysis=roma_max_depth_analysis,
            roma_max_depth_solving=roma_max_depth_solving,
            roma_execution_mode=roma_execution_mode,
            mdap_k_ahead=mdap_k_ahead,
            mdap_max_samples=mdap_max_samples,
            mdap_enable_red_flagging=mdap_enable_red_flagging,
            enable_adaptive_k=enable_adaptive_k,
            enable_caching=enable_caching,
            provider=provider,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

        # Add API key if provided
        if api_key:
            config.api_key = api_key

        # Create engine
        engine = ROMAMDAPMakerEngine(config)

        # Execute
        result = engine.solve_with_roma_mdap_maker(
            task=task,
            context=context
        )

        # Add execution metadata
        result["config_used"] = {
            "roma_max_depth_analysis": roma_max_depth_analysis,
            "roma_max_depth_solving": roma_max_depth_solving,
            "roma_execution_mode": roma_execution_mode,
            "mdap_k_ahead": mdap_k_ahead,
            "mdap_enable_red_flagging": mdap_enable_red_flagging,
            "enable_adaptive_k": enable_adaptive_k,
            "provider": provider,
            "model": model
        }

        return result

    except Exception as e:
        logger.error(f"Error in solve_with_roma_mdap_maker: {e}", exc_info=True)
        return {
            "error": str(e),
            "task": task,
            "execution_method_used": "roma_mdap_maker"
        }


@mcp_tool("solve_subproblem_with_roma_mdap_maker")
def solve_subproblem_with_roma_mdap_maker(
    sub_problem_id: str,
    sub_problem_description: str,
    context: Optional[Dict[str, Any]] = None,
    requirements: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    roma_max_depth: int = 2,
    roma_execution_mode: str = "recursive",
    maker_k_ahead: int = 3,
    maker_max_samples: int = 100,
    enable_red_flagging: bool = True,
    enable_adaptive_k: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA-MDAP-MAKER

    Integrates with Decomposition Workflow Stage 3A (Solution Generation).
    Uses ROMA for automatic decomposition and MAKER for zero-error voting.

    Args:
        sub_problem_id: Sub-problem identifier (e.g., "SP-001")
        sub_problem_description: Sub-problem description
        context: Additional context
        requirements: List of requirements to satisfy
        constraints: List of constraints to respect
        roma_max_depth: ROMA max depth (default: 2)
        roma_execution_mode: "recursive" or "event_driven"
        maker_k_ahead: MAKER voting threshold
        maker_max_samples: Max samples per voting round
        enable_red_flagging: Enable red-flagging
        enable_adaptive_k: Enable adaptive k selection
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional config

    Returns:
        Dict with solution attempt

    Example:
        >>> result = solve_subproblem_with_roma_mdap_maker(
        ...     sub_problem_id="SP-001",
        ...     sub_problem_description="Implement OAuth2 authentication",
        ...     roma_max_depth=2,
        ...     maker_k_ahead=3
        ... )
        >>> print(f"Solution: {result['solution']}")
        >>> print(f"Error-free: {result['error_free']}")
    """
    logger.info(f"Solving sub-problem {sub_problem_id} with ROMA-MDAP-MAKER")

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
        roma_max_depth_solving=roma_max_depth,
        roma_execution_mode=roma_execution_mode,
        mdap_k_ahead=maker_k_ahead,
        mdap_max_samples=maker_max_samples,
        mdap_enable_red_flagging=enable_red_flagging,
        enable_adaptive_k=enable_adaptive_k,
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )

    # Add sub-problem specific fields
    result["sub_problem_id"] = sub_problem_id
    result["solution"] = result.get("result")
    result["requirements_satisfied"] = _check_requirements_satisfied(
        result.get("result"),
        requirements or []
    )

    return result


@mcp_tool("get_roma_mdap_maker_status")
def get_roma_mdap_maker_status() -> Dict[str, Any]:
    """
    Check ROMA-MDAP-MAKER system availability and configuration

    Returns:
        Dict with:
            - roma_available: Whether ROMA is available
            - mdap_available: Whether MDAP is available
            - roma_mdap_maker_available: Whether full system is available
            - total_execution_methods: Number of execution methods
            - execution_methods: List of execution method names
            - capabilities: List of system capabilities

    Example:
        >>> status = get_roma_mdap_maker_status()
        >>> print(f"Available: {status['roma_mdap_maker_available']}")
        >>> print(f"Execution methods: {status['execution_methods']}")
    """
    from roma_mdap_maker_engine import get_roma_mdap_maker_status as engine_status
    return engine_status()


@mcp_tool("analyze_problem_with_roma_mdap")
def analyze_problem_with_roma_mdap(
    problem_statement: str,
    roma_max_depth: int = 3,
    context: Optional[Dict[str, Any]] = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze problem structure using ROMA

    Returns decomposition hierarchy without solving.
    Useful for understanding problem complexity before execution.

    Args:
        problem_statement: Problem to analyze
        roma_max_depth: Max decomposition depth (default: 3)
        context: Additional context
        provider: LLM provider
        model: Model name
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

    Example:
        >>> analysis = analyze_problem_with_roma_mdap(
        ...     problem_statement="Design a microservices architecture"
        ... )
        >>> print(f"Complexity: {analysis['estimated_complexity']}")
        >>> print(f"Subtasks: {analysis['num_subtasks']}")
    """
    logger.info(f"Analyzing problem with ROMA: {problem_statement[:100]}...")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "problem_statement": problem_statement
        }

    try:
        from roma_mcp_tools import analyze_with_roma

        # Analyze with ROMA
        roma_result = analyze_with_roma(
            task=problem_statement,
            max_depth=roma_max_depth,
            provider=provider,
            model=model,
            **kwargs
        )

        if roma_result.get("error"):
            return roma_result

        # Extract decomposition info
        decomposition = roma_result.get("decomposition", {})
        dag_info = roma_result.get("dag_info", {})

        # Estimate complexity
        num_subtasks = _count_subtasks(decomposition)
        max_depth = roma_result.get("max_depth", 0)
        estimated_complexity = _estimate_complexity_from_decomposition(
            num_subtasks, max_depth, problem_statement
        )

        # Recommend parameters
        recommended_depth = min(max_depth + 1, 5)
        recommended_k = _recommend_k_ahead(estimated_complexity)

        return {
            "decomposition": decomposition,
            "dag_info": dag_info,
            "estimated_complexity": estimated_complexity,
            "recommended_depth": recommended_depth,
            "recommended_k": recommended_k,
            "num_subtasks": num_subtasks,
            "max_depth": max_depth,
            "use_roma_mdap_maker": estimated_complexity > 7.0
        }

    except Exception as e:
        logger.error(f"Error analyzing problem: {e}", exc_info=True)
        return {
            "error": str(e),
            "problem_statement": problem_statement
        }


@mcp_tool("verify_solution_with_roma_mdap")
def verify_solution_with_roma_mdap(
    solution: str,
    requirements: List[str],
    verification_depth: int = 2,
    context: Optional[Dict[str, Any]] = None,
    maker_k_ahead: int = 2,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    **kwargs
) -> Dict[str, Any]:
    """
    Verify solution using ROMA recursive verification + MAKER voting

    Enhances quality assurance by recursively verifying solution components
    with voting-based consensus.

    Args:
        solution: Solution to verify
        requirements: Requirements to verify against
        verification_depth: Recursion depth for verification (default: 2)
        context: Additional context
        maker_k_ahead: MAKER voting threshold (default: 2, lower for verification)
        provider: LLM provider
        model: Model name
        **kwargs: Additional config

    Returns:
        Dict with:
            - passed: Whether solution passed verification
            - confidence: Verification confidence (0-1)
            - findings: List of verification findings
            - requirements_check: Per-requirement verification results
            - total_checks: Number of checks performed
            - passed_checks: Number of checks passed

    Example:
        >>> result = verify_solution_with_roma_mdap(
        ...     solution="OAuth2 implementation with PKCE",
        ...     requirements=["Security", "Scalability", "Usability"]
        ... )
        >>> print(f"Passed: {result['passed']}")
        >>> print(f"Confidence: {result['confidence']}")
    """
    logger.info(f"Verifying solution with ROMA-MDAP")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available for verification",
            "solution": solution
        }

    try:
        # Verify each requirement with ROMA + MAKER
        requirements_check = []
        passed_checks = 0

        for requirement in requirements:
            # Create verification task
            verification_task = f"""
            Verify the following solution against the requirement:

            Requirement: {requirement}

            Solution: {solution}

            Provide:
            1. Pass/Fail determination
            2. Confidence score (0-1)
            3. Specific findings
            """

            # Use ROMA-MDAP-MAKER for verification
            verification_result = solve_with_roma_mdap_maker(
                task=verification_task,
                roma_max_depth_solving=1,  # Shallow for verification
                mdap_k_ahead=maker_k_ahead,
                provider=provider,
                model=model,
                **kwargs
            )

            # Extract result
            result_text = verification_result.get("result", "")
            confidence = verification_result.get("confidence", 0.5)

            # Parse pass/fail (simple keyword matching)
            passed = "pass" in result_text.lower() or "✓" in result_text or "✓" in result_text

            if passed:
                passed_checks += 1

            requirements_check.append({
                "requirement": requirement,
                "passed": passed,
                "confidence": confidence,
                "findings": result_text
            })

        # Overall verification
        total_checks = len(requirements)
        overall_passed = passed_checks == total_checks
        overall_confidence = sum(rc["confidence"] for rc in requirements_check) / max(1, total_checks)

        # Aggregate findings
        findings = []
        for rc in requirements_check:
            if not rc["passed"]:
                findings.append(f"Failed: {rc['requirement']}")

        return {
            "passed": overall_passed,
            "confidence": overall_confidence,
            "findings": findings,
            "requirements_check": requirements_check,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "verification_method": "roma_mdap_maker"
        }

    except Exception as e:
        logger.error(f"Error verifying solution: {e}", exc_info=True)
        return {
            "error": str(e),
            "solution": solution,
            "verification_method": "roma_mdap_maker"
        }


@mcp_tool("create_roma_mdap_maker_config")
def create_roma_mdap_maker_config_tool(
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    mdap_k_ahead: int = 3,
    mdap_max_samples: int = 100,
    mdap_enable_red_flagging: bool = True,
    mdap_max_token_length: int = 750,
    mdap_min_confidence: float = 0.2,
    enable_adaptive_k: bool = True,
    enable_caching: bool = True,
    enable_hierarchical_voting: bool = True,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_retries: int = 3,
    timeout_seconds: int = 300,
    **kwargs
) -> Dict[str, Any]:
    """
    Create ROMA-MDAP-MAKER configuration object

    Returns a validated configuration that can be passed to solve_with_roma_mdap_maker.

    Args:
        roma_max_depth_analysis: ROMA max depth for analysis
        roma_max_depth_solving: ROMA max depth for solving
        roma_execution_mode: "recursive" or "event_driven"
        mdap_k_ahead: MAKER voting threshold
        mdap_max_samples: Max samples per voting round
        mdap_enable_red_flagging: Enable red-flagging
        mdap_max_token_length: Max token length for red-flagging
        mdap_min_confidence: Min confidence threshold
        enable_adaptive_k: Enable adaptive k selection
        enable_caching: Enable result caching
        enable_hierarchical_voting: Enable hierarchical voting
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature
        max_retries: Max retries per task
        timeout_seconds: Timeout per task
        **kwargs: Additional config

    Returns:
        Dict with:
            - config: ROMAMDAPMakerConfig object
            - validation_errors: List of validation errors (empty if valid)
            - is_valid: Whether configuration is valid

    Example:
        >>> config_result = create_roma_mdap_maker_config_tool(
        ...     roma_max_depth_solving=3,
        ...     mdap_k_ahead=5
        ... )
        >>> if config_result['is_valid']:
        ...     config = config_result['config']
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

    # Create config
    config = create_roma_mdap_maker_config(
        roma_max_depth_analysis=roma_max_depth_analysis,
        roma_max_depth_solving=roma_max_depth_solving,
        roma_execution_mode=roma_execution_mode,
        mdap_k_ahead=mdap_k_ahead,
        mdap_max_samples=mdap_max_samples,
        mdap_enable_red_flagging=mdap_enable_red_flagging,
        enable_adaptive_k=enable_adaptive_k,
        enable_caching=enable_caching,
        enable_hierarchical_voting=enable_hierarchical_voting,
        provider=provider,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        **kwargs
    )

    # Add MDAP-specific settings
    config.mdap_max_token_length = mdap_max_token_length
    config.mdap_min_confidence = mdap_min_confidence

    return {
        "config": config,
        "config_dict": {
            "roma_max_depth_analysis": config.roma_max_depth_analysis,
            "roma_max_depth_solving": config.roma_max_depth_solving,
            "roma_execution_mode": config.roma_execution_mode,
            "mdap_k_ahead": config.mdap_k_ahead,
            "mdap_max_samples": config.mdap_max_samples,
            "mdap_enable_red_flagging": config.mdap_enable_red_flagging,
            "enable_adaptive_k": config.enable_adaptive_k,
            "enable_caching": config.enable_caching,
            "provider": config.provider,
            "model": config.model
        },
        "validation_errors": validation_errors,
        "is_valid": len(validation_errors) == 0
    }


@mcp_tool("get_roma_mdap_maker_metrics")
def get_roma_mdap_maker_metrics(
    execution_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get detailed metrics for ROMA-MDAP-MAKER executions

    Args:
        execution_id: Optional specific execution ID (if None, returns aggregate)

    Returns:
        Dict with:
            - total_executions: Total number of executions
            - total_atomic_tasks: Total atomic tasks executed
            - total_voting_rounds: Total voting rounds
            - total_red_flags: Total red flags raised
            - total_errors: Total errors encountered
            - avg_confidence: Average confidence across all executions
            - avg_execution_time: Average execution time
            - error_rate: Overall error rate
            - red_flag_rate: Red flags per voting round
            - cost_estimate: Estimated cost in USD

    Example:
        >>> metrics = get_roma_mdap_maker_metrics()
        >>> print(f"Total executions: {metrics['total_executions']}")
        >>> print(f"Avg confidence: {metrics['avg_confidence']}")
        >>> print(f"Error rate: {metrics['error_rate']}")
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
        "note": "Metrics tracking to be implemented with database backend"
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _count_subtasks(decomposition: Dict[str, Any]) -> int:
    """Count total subtasks in decomposition"""
    if not decomposition or not decomposition.get("subtasks"):
        return 1  # Atomic task

    count = 0
    for subtask in decomposition.get("subtasks", []):
        count += _count_subtasks(subtask)

    return count


def _estimate_complexity_from_decomposition(
    num_subtasks: int,
    max_depth: int,
    problem_statement: str
) -> float:
    """Estimate complexity (1-10) from decomposition"""
    complexity = 5.0  # Base

    # Subtask count
    if num_subtasks > 20:
        complexity += 2.0
    elif num_subtasks > 10:
        complexity += 1.0

    # Depth
    if max_depth > 4:
        complexity += 1.5
    elif max_depth > 2:
        complexity += 0.5

    # Statement length
    if len(problem_statement) > 1000:
        complexity += 1.0

    return min(complexity, 10.0)


def _recommend_k_ahead(complexity: float) -> int:
    """Recommend k-ahead value based on complexity"""
    if complexity > 8.0:
        return 5
    elif complexity > 6.0:
        return 4
    elif complexity > 4.0:
        return 3
    else:
        return 2


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
# EXPORTS
# =============================================================================

__all__ = [
    "solve_with_roma_mdap_maker",
    "solve_subproblem_with_roma_mdap_maker",
    "get_roma_mdap_maker_status",
    "analyze_problem_with_roma_mdap",
    "verify_solution_with_roma_mdap",
    "create_roma_mdap_maker_config_tool",
    "get_roma_mdap_maker_metrics",
    "mcp_tool",
    "register_mcp_tool",
    "get_mcp_tool",
    "list_mcp_tools",
]
