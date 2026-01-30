"""
ROMA MCP Tools for CREWAI Agents

This module provides Model Context Protocol (MCP) tools that CREWAI agents
can use to leverage ROMA's (Recursive Open Meta-Agents) framework.

ROMA Architecture:
    solve(task):
        if is_atomic(task):
            return execute(task)
        else:
            subtasks = plan(task)
            results = [solve(subtask) for subtask in subtasks]  # Recursive
            return aggregate(results)

Key Features:
    - Recursive hierarchical decomposition (like Decomposition Workflow)
    - Atomizer: Decides if task needs planning
    - Planner: Breaks tasks into subtasks
    - Executor: Handles atomic tasks
    - Aggregator: Integrates results
    - Checkpoint/recovery system
    - MLflow observability
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import ROMA components
try:
    from roma_dspy.core.engine.solve import solve, async_solve, event_solve, RecursiveSolver
    from roma_dspy.core.signatures import TaskNode
    from roma_dspy.config.schemas.root import ROMAConfig
    from roma_dspy.core.engine import TaskDAG
    ROMA_AVAILABLE = True
    logger.info("ROMA core imported successfully")
except ImportError as e:
    logger.warning(f"ROMA not available: {e}")
    ROMA_AVAILABLE = False
    solve = None
    async_solve = None
    event_solve = None
    RecursiveSolver = None
    TaskNode = None
    ROMAConfig = None
    TaskDAG = None


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered ROMA MCP tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered ROMA MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# ROMA MCP TOOLS
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
    Solve a task using ROMA's recursive decomposition framework.

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
    logger.info(f"Solving with ROMA: {task[:100]}... (mode={execution_mode}, max_depth={max_depth})")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "task": task,
            "execution_method_used": "roma",
        }

    try:
        # Create ROMA config
        config = _create_roma_config(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )

        # Create solver
        solver = RecursiveSolver(
            config=config,
            max_depth=max_depth,
            enable_logging=enable_logging,
            enable_checkpoints=enable_checkpoints,
        )

        # Execute based on mode
        if execution_mode == "event_driven":
            # Event-driven: parallel execution with DAG
            result_task_node = solver.event_solve(task)
        else:
            # Recursive: depth-first execution
            result_task_node = solver.solve(task)

        # Extract results
        result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)
        status = result_task_node.status.value if hasattr(result_task_node, 'status') else "unknown"

        # Get DAG info if available
        dag_info = {}
        if solver.last_dag:
            dag_info = {
                "total_tasks": len(solver.last_dag.get_all_tasks()),
                "execution_id": solver.last_dag.execution_id,
            }

        # Get token usage
        token_usage = {
            "input_tokens": solver.get_total_input_tokens(),
            "output_tokens": solver.get_total_output_tokens(),
        }

        return {
            "task": task,
            "result": result,
            "status": status,
            "execution_mode": execution_mode,
            "max_depth": max_depth,
            "generated_by": f"ROMA ({execution_mode})",
            "execution_method_used": "roma",
            "dag_info": dag_info,
            "token_usage": token_usage,
        }

    except Exception as e:
        logger.error(f"Failed to solve with ROMA: {e}")
        return {
            "error": str(e),
            "task": task,
            "execution_method_used": "roma",
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
    Solve a sub-problem using ROMA (for Decomposition Workflow integration).

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
    logger.info(f"Solving sub-problem {sub_problem_id} with ROMA (team={team_name})")

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
        }

    return {
        "sub_problem_id": sub_problem_id,
        "solution": result["result"],
        "team_name": team_name,
        "generated_by": f"ROMA ({execution_mode})",
        "status": result["status"],
        "execution_method_used": "roma",
        "dag_info": result.get("dag_info", {}),
        "token_usage": result.get("token_usage"),
    }


@mcp_tool("analyze_with_roma")
def analyze_with_roma(
    problem: str,
    analysis_type: str = "decomposition",  # "decomposition", "complexity", "dependencies"
    max_depth: int = 3,
    provider: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze a problem using ROMA's decomposition capabilities.

    This is for Stage 0-1 (Problem Analysis and Decomposition).

    Args:
        problem: Problem statement to analyze
        analysis_type: Type of analysis to perform
        max_depth: Maximum recursion depth for decomposition
        provider: LLM provider
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with analysis results
    """
    logger.info(f"Analyzing problem with ROMA: {problem[:100]}... (type={analysis_type})")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "problem": problem,
        }

    try:
        config = _create_roma_config(provider=provider, **kwargs)
        solver = RecursiveSolver(config=config, max_depth=max_depth)

        # Build analysis task
        task = f"Analyze this problem:\n\n{problem}\n\n"
        if analysis_type == "decomposition":
            task += "Break it down into sub-problems and identify the decomposition structure."
        elif analysis_type == "complexity":
            task += "Analyze the complexity and identify key challenges."
        elif analysis_type == "dependencies":
            task += "Identify dependencies between different components."

        # Solve with ROMA
        result_task_node = solver.solve(task)

        # Extract results
        result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)

        return {
            "problem": problem,
            "analysis_type": analysis_type,
            "analysis": result,
            "dag_info": {
                "total_tasks": len(solver.last_dag.get_all_tasks()) if solver.last_dag else 1,
                "execution_id": solver.last_dag.execution_id if solver.last_dag else None,
            },
            "token_usage": {
                "input_tokens": solver.get_total_input_tokens(),
                "output_tokens": solver.get_total_output_tokens(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to analyze with ROMA: {e}")
        return {
            "error": str(e),
            "problem": problem,
        }


@mcp_tool("verify_with_roma")
def verify_with_roma(
    solution: str,
    original_task: str,
    verification_criteria: Optional[List[str]] = None,
    provider: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Verify a solution using ROMA (for Stage 3C/4 - Verification).

    Args:
        solution: The solution to verify
        original_task: The original task/problem
        verification_criteria: List of criteria to verify
        provider: LLM provider
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with verification results
    """
    logger.info(f"Verifying solution with ROMA: {original_task[:100]}...")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "original_task": original_task,
        }

    try:
        config = _create_roma_config(provider=provider, **kwargs)
        solver = RecursiveSolver(config=config, max_depth=1)

        # Build verification task
        task = f"""Verify this solution against the original task:

Original Task:
{original_task}

Solution:
{solution}
"""

        if verification_criteria:
            task += "\n\nVerification Criteria:\n"
            for criteria in verification_criteria:
                task += f"  - {criteria}\n"

        task += "\n\nProvide a detailed verification report with pass/fail for each criterion."

        # Solve with ROMA
        result_task_node = solver.solve(task)
        result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)

        return {
            "original_task": original_task,
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "verification": result,
            "verification_criteria": verification_criteria or [],
            "token_usage": {
                "input_tokens": solver.get_total_input_tokens(),
                "output_tokens": solver.get_total_output_tokens(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to verify with ROMA: {e}")
        return {
            "error": str(e),
            "original_task": original_task,
        }


@mcp_tool("critique_with_roma")
def critique_with_roma(
    solution: str,
    original_task: str,
    critique_focus: str = "comprehensive",  # "comprehensive", "security", "performance", "correctness"
    provider: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Critique a solution using ROMA (for Stage 3B - Red Team Critique).

    Args:
        solution: The solution to critique
        original_task: The original task
        critique_focus: Type of critique to perform
        provider: LLM provider
        **kwargs: Additional ROMA configuration

    Returns:
        Dict with critique results
    """
    logger.info(f"Critiquing solution with ROMA: {original_task[:100]}... (focus={critique_focus})")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "original_task": original_task,
        }

    try:
        config = _create_roma_config(provider=provider, **kwargs)
        solver = RecursiveSolver(config=config, max_depth=1)

        # Build critique task
        task = f"""Critique this solution from a Red Team perspective:

Original Task:
{original_task}

Solution:
{solution}

Focus your critique on: {critique_focus}

Identify:
1. Potential flaws or weaknesses
2. Missing edge cases
3. Security concerns (if applicable)
4. Performance issues
5. Areas for improvement
"""

        # Solve with ROMA
        result_task_node = solver.solve(task)
        result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)

        return {
            "original_task": original_task,
            "solution": solution[:200] + "..." if len(solution) > 200 else solution,
            "critique": result,
            "critique_focus": critique_focus,
            "token_usage": {
                "input_tokens": solver.get_total_input_tokens(),
                "output_tokens": solver.get_total_output_tokens(),
            },
        }

    except Exception as e:
        logger.error(f"Failed to critique with ROMA: {e}")
        return {
            "error": str(e),
            "original_task": original_task,
        }


@mcp_tool("get_roma_status")
def get_roma_status() -> Dict[str, Any]:
    """Get the status of the ROMA integration"""
    status = {
        "available": ROMA_AVAILABLE,
    }

    if ROMA_AVAILABLE:
        # Try to get more details
        try:
            from roma_dspy import __version__
            status["version"] = __version__
        except:
            status["version"] = "unknown"

        # Check for optional dependencies
        try:
            import dspy
            status["dspy_available"] = True
        except ImportError:
            status["dspy_available"] = False

        try:
            from roma_dspy.core.storage import PostgresStorage
            status["persistence_available"] = True
        except ImportError:
            status["persistence_available"] = False

        try:
            from roma_dspy.core.observability import MLflowManager
            status["observability_available"] = True
        except ImportError:
            status["observability_available"] = False

    return status


@mcp_tool("create_roma_config")
def create_roma_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a ROMA configuration object.

    Args:
        provider: LLM provider
        model: Model name
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        Dict with configuration details
    """
    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
        }

    try:
        config = _create_roma_config(
            provider=provider,
            model=model,
            api_key=api_key,
            **kwargs
        )

        return {
            "config_created": True,
            "provider": provider or "default",
            "model": model or "default",
            "runtime": {
                "max_depth": config.runtime.max_depth,
                "max_concurrency": config.runtime.max_concurrency,
            },
            "storage": {
                "file_enabled": config.storage.file.enabled if config.storage else True,
            },
        }
    except Exception as e:
        return {
            "error": str(e),
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_roma_config(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Optional["ROMAConfig"]:
    """
    Create a ROMAConfig instance with custom settings.

    Args:
        provider: LLM provider (openai, anthropic, google, openrouter)
        model: Model name
        api_key: API key
        **kwargs: Additional configuration

    Returns:
        ROMAConfig instance or None if ROMA not available
    """
    if not ROMA_AVAILABLE:
        return None

    try:
        # Create default config
        config = ROMAConfig()

        # Configure LLM provider if specified
        if provider:
            # ROMA uses environment variables by default
            # We can override by setting them temporarily
            import os
            if api_key:
                if provider.lower() == "openai":
                    os.environ["OPENAI_API_KEY"] = api_key
                elif provider.lower() == "anthropic":
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                elif provider.lower() == "google":
                    os.environ["GOOGLE_API_KEY"] = api_key
                elif provider.lower() == "openrouter":
                    os.environ["OPENROUTER_API_KEY"] = api_key

        return config

    except Exception as e:
        logger.error(f"Failed to create ROMA config: {e}")
        return None


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all ROMA MCP tools"""
    logger.info("Initializing ROMA MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} ROMA MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()
