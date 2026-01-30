"""
OpenEvolve MCP Tools for CREWAI Agents

This module provides Model Context Protocol (MCP) tools that CREWAI agents
can use to call OpenEvolve's evolutionary coding capabilities.

IMPORTANT: OpenEvolve is an evolutionary coding agent that optimizes code through
iterative mutations using LLMs. It does NOT provide decomposition, team management,
or gauntlet functionality - those are separate systems in the Frontend project.

OpenEvolve API:
- run_evolution() - Main evolution function
- evolve_function() - Evolve Python functions
- evolve_algorithm() - Evolve algorithm classes
- evolve_code() - Evolve arbitrary code

Architecture:
    CREWAI Agent → MCP Tool → OpenEvolve API → Optimized Code
"""

import json
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import OpenEvolve API
try:
    from openevolve.api import (
        run_evolution,
        evolve_function,
        evolve_algorithm,
        evolve_code,
        EvolutionResult,
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    logger.warning("OpenEvolve not available - MCP tools will be stubs")
    OPENEVOLVE_AVAILABLE = False


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        register_mcp_tool(name, func)
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered OpenEvolve MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# EVOLUTION MCP TOOLS
# =============================================================================

@mcp_tool("evolve_code_with_openevolve")
def evolve_code_with_openevolve(
    initial_code: str,
    evaluator_function: Optional[str] = None,
    iterations: int = 50,
    optimization_goal: str = "performance",
) -> Dict[str, Any]:
    """
    Evolve/optimize code using OpenEvolve's evolutionary algorithm.

    This tool is used by CREWAI agents to optimize code segments
    through iterative LLM-based mutations.

    Args:
        initial_code: The initial code to evolve (Python, Rust, R, etc.)
        evaluator_function: Optional evaluator code (Python function that returns metrics)
        iterations: Number of evolution iterations (default: 50)
        optimization_goal: What to optimize for ("performance", "correctness", "memory", "code_size")

    Returns:
        Dict with evolution results:
        {
            "evolved_code": str,
            "best_score": float,
            "metrics": Dict,
            "improvement": float,
            "iterations": int
        }
    """
    logger.info(f"Evolving code with OpenEvolve: {iterations} iterations, goal={optimization_goal}")

    if not OPENEVOLVE_AVAILABLE:
        return {
            "error": "OpenEvolve not available",
            "evolved_code": initial_code,
            "best_score": 0.0,
            "metrics": {},
            "improvement": 0.0,
        }

    try:
        # Create evaluator if not provided
        if evaluator_function is None:
            # Default evaluator that checks for syntax errors and basic metrics
            evaluator_function = create_default_evaluator(optimization_goal)

        # Run evolution
        result: EvolutionResult = evolve_code(
            initial_code=initial_code,
            evaluator=evaluator_function,
            iterations=iterations,
        )

        # Calculate improvement
        improvement = result.best_score  # Normalized score

        return {
            "evolved_code": result.best_code,
            "best_score": result.best_score,
            "metrics": result.metrics,
            "improvement": improvement,
            "iterations": iterations,
            "output_dir": result.output_dir,
        }

    except Exception as e:
        logger.error(f"Code evolution failed: {e}")
        return {
            "error": str(e),
            "evolved_code": initial_code,
            "best_score": 0.0,
            "metrics": {},
            "improvement": 0.0,
        }


@mcp_tool("evolve_function_with_openevolve")
def evolve_function_with_openevolve(
    function_name: str,
    function_code: str,
    test_cases: List[Dict[str, Any]],
    iterations: int = 50,
) -> Dict[str, Any]:
    """
    Evolve a Python function based on test cases using OpenEvolve.

    This tool is used by CREWAI agents to optimize specific functions
    through test-driven evolution.

    Args:
        function_name: Name of the function to evolve
        function_code: Source code of the function
        test_cases: List of test cases as dicts with "input" and "expected_output"
        iterations: Number of evolution iterations

    Returns:
        Dict with evolution results
    """
    logger.info(f"Evolving function '{function_name}' with {len(test_cases)} test cases")

    if not OPENEVOLVE_AVAILABLE:
        return {
            "error": "OpenEvolve not available",
            "evolved_code": function_code,
            "best_score": 0.0,
        }

    try:
        # Reconstruct function from code
        func_code = f"def {function_name}(...):\n    {function_code}"

        # Convert test cases to tuples
        test_tuples = [
            (tc.get("input"), tc.get("expected_output"))
            for tc in test_cases
        ]

        # Create a temporary function object
        import types
        func = types.FunctionType(
            compile(func_code, "<string>", "exec").co_consts[0],
            globals(),
            function_name,
        )

        # Run evolution
        result: EvolutionResult = evolve_function(
            func=func,
            test_cases=test_tuples,
            iterations=iterations,
        )

        return {
            "evolved_code": result.best_code,
            "best_score": result.best_score,
            "metrics": result.metrics,
            "function_name": function_name,
            "test_cases_passed": result.metrics.get("tests_passed", 0),
            "total_test_cases": result.metrics.get("total_tests", 0),
        }

    except Exception as e:
        logger.error(f"Function evolution failed: {e}")
        return {
            "error": str(e),
            "evolved_code": function_code,
            "best_score": 0.0,
            "metrics": {},
        }


@mcp_tool("optimize_algorithm_with_openevolve")
def optimize_algorithm_with_openevolve(
    algorithm_name: str,
    algorithm_code: str,
    benchmark_description: str,
    iterations: int = 100,
    performance_metric: str = "runtime",
) -> Dict[str, Any]:
    """
    Evolve an algorithm class using OpenEvolve with a custom benchmark.

    This tool is used by CREWAI agents to discover optimized algorithms
    through evolutionary search.

    Args:
        algorithm_name: Name of the algorithm class
        algorithm_code: Source code of the algorithm class
        benchmark_description: Description of the benchmark to run
        iterations: Number of evolution iterations
        performance_metric: Metric to optimize ("runtime", "accuracy", "memory")

    Returns:
        Dict with evolution results
    """
    logger.info(f"Optimizing algorithm '{algorithm_name}' for {performance_metric}")

    if not OPENEVOLVE_AVAILABLE:
        return {
            "error": "OpenEvolve not available",
            "evolved_code": algorithm_code,
            "best_score": 0.0,
        }

    try:
        # Create a benchmark function based on description
        benchmark_func = create_benchmark_from_description(benchmark_description, performance_metric)

        # Reconstruct class from code
        import types
        code_obj = compile(algorithm_code, "<string>", "exec")
        namespace = {}
        exec(code_obj, namespace)

        if algorithm_name not in namespace:
            return {
                "error": f"Algorithm class '{algorithm_name}' not found in code",
                "evolved_code": algorithm_code,
                "best_score": 0.0,
            }

        algorithm_class = namespace[algorithm_name]

        # Run evolution
        result: EvolutionResult = evolve_algorithm(
            algorithm_class=algorithm_class,
            benchmark=benchmark_func,
            iterations=iterations,
        )

        return {
            "evolved_code": result.best_code,
            "best_score": result.best_score,
            "metrics": result.metrics,
            "algorithm_name": algorithm_name,
            "performance_improvement": result.metrics.get("performance", 0.0),
        }

    except Exception as e:
        logger.error(f"Algorithm optimization failed: {e}")
        return {
            "error": str(e),
            "evolved_code": algorithm_code,
            "best_score": 0.0,
            "metrics": {},
        }


@mcp_tool("discover_algorithm_with_openevolve")
def discover_algorithm_with_openevolve(
    problem_description: str,
    constraints: List[str],
    search_space: str,
    iterations: int = 200,
    num_islands: int = 5,
) -> Dict[str, Any]:
    """
    Use OpenEvolve to discover novel algorithms for a problem.

    This tool is used by CREWAI agents to perform algorithm discovery
    through open-ended evolutionary search.

    Args:
        problem_description: Description of the problem to solve
        constraints: List of constraints the algorithm must satisfy
        search_space: Description of the search space (e.g., "sorting algorithms", "optimization")
        iterations: Number of evolution iterations
        num_islands: Number of parallel evolution islands

    Returns:
        Dict with discovered algorithm
    """
    logger.info(f"Discovering algorithm for: {problem_description[:100]}...")

    if not OPENEVOLVE_AVAILABLE:
        return {
            "error": "OpenEvolve not available",
            "discovered_code": "",
            "best_score": 0.0,
        }

    try:
        # Generate initial random algorithm code
        initial_code = generate_initial_algorithm(search_space)

        # Create evaluator for the problem
        evaluator = create_problem_evaluator(problem_description, constraints)

        # Configure OpenEvolve for discovery
        from openevolve.config import Config

        config = Config()
        config.database.num_islands = num_islands
        config.database.population_size = 500

        # Run evolution
        result: EvolutionResult = run_evolution(
            initial_program=initial_code,
            evaluator=evaluator,
            config=config,
            iterations=iterations,
        )

        return {
            "discovered_code": result.best_code,
            "best_score": result.best_score,
            "metrics": result.metrics,
            "problem_description": problem_description,
            "search_space": search_space,
            "iterations": iterations,
        }

    except Exception as e:
        logger.error(f"Algorithm discovery failed: {e}")
        return {
            "error": str(e),
            "discovered_code": "",
            "best_score": 0.0,
            "metrics": {},
        }


@mcp_tool("optimize_prompt_with_openevolve")
def optimize_prompt_with_openevolve(
    initial_prompt: str,
    test_cases: List[Dict[str, Any]],
    evaluation_criteria: List[str],
    iterations: int = 50,
) -> Dict[str, Any]:
    """
    Evolve a prompt using OpenEvolve for better LLM performance.

    This tool is used by CREWAI agents to optimize prompts
    through evolutionary search.

    Args:
        initial_prompt: The initial prompt to optimize
        test_cases: List of test cases with input and expected behavior
        evaluation_criteria: Criteria for evaluating prompt performance
        iterations: Number of evolution iterations

    Returns:
        Dict with optimized prompt
    """
    logger.info(f"Optimizing prompt with {len(test_cases)} test cases")

    if not OPENEVOLVE_AVAILABLE:
        return {
            "error": "OpenEvolve not available",
            "optimized_prompt": initial_prompt,
            "best_score": 0.0,
        }

    try:
        # Create evaluator for prompt
        def prompt_evaluator(prompt_path: str) -> Dict[str, Any]:
            # Load evolved prompt
            with open(prompt_path, 'r') as f:
                prompt = f.read()

            # Test prompt against test cases
            scores = []
            for tc in test_cases:
                # Simulate prompt evaluation
                score = evaluate_prompt(prompt, tc, evaluation_criteria)
                scores.append(score)

            return {
                "score": sum(scores) / len(scores),
                "avg_score": sum(scores) / len(scores),
                "test_count": len(test_cases),
            }

        # Run evolution
        result: EvolutionResult = run_evolution(
            initial_program=initial_prompt,
            evaluator=prompt_evaluator,
            iterations=iterations,
        )

        return {
            "optimized_prompt": result.best_code,
            "best_score": result.best_score,
            "metrics": result.metrics,
            "improvement": result.best_score,
        }

    except Exception as e:
        logger.error(f"Prompt optimization failed: {e}")
        return {
            "error": str(e),
            "optimized_prompt": initial_prompt,
            "best_score": 0.0,
            "metrics": {},
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@mcp_tool("list_openevolve_capabilities")
def list_openevolve_capabilities() -> Dict[str, Any]:
    """List OpenEvolve's capabilities and configuration options"""
    return {
        "capabilities": [
            "evolve_code",
            "evolve_function",
            "evolve_algorithm",
            "discover_algorithm",
            "optimize_prompt",
        ],
        "supported_languages": ["Python", "Rust", "R", "Metal", "C++", "JavaScript"],
        "optimization_targets": [
            "performance",
            "correctness",
            "memory_usage",
            "code_size",
            "energy_efficiency",
        ],
        "available": OPENEVOLVE_AVAILABLE,
    }


@mcp_tool("get_openevolve_status")
def get_openevolve_status() -> Dict[str, Any]:
    """Get OpenEvolve installation and configuration status"""
    if not OPENEVOLVE_AVAILABLE:
        return {
            "available": False,
            "installed": False,
            "version": None,
            "error": "OpenEvolve not installed or not accessible",
            "components": {
                "api": False,
                "config": False,
            },
        }

    try:
        import openevolve
        from openevolve._version import __version__

        return {
            "available": True,
            "installed": True,
            "version": __version__,
            "components": {
                "api": True,
                "config": True,
            },
        }
    except Exception as e:
        return {
            "available": False,
            "installed": True,
            "version": "unknown",
            "error": str(e),
            "components": {
                "api": False,
                "config": False,
            },
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_default_evaluator(goal: str) -> Callable:
    """Create a default evaluator function based on optimization goal"""

    def evaluator(program_path: str) -> Dict[str, Any]:
        import subprocess
        import time

        # Basic syntax check
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", program_path],
                capture_output=True,
                timeout=10,
            )
            syntax_valid = result.returncode == 0
        except Exception as e:
            return {
                "score": 0.0,
                "error": f"Syntax check failed: {e}",
            }

        if not syntax_valid:
            return {
                "score": 0.0,
                "syntax_valid": False,
            }

        # Read code for metrics
        with open(program_path, 'r') as f:
            code = f.read()

        lines = len(code.split('\n'))
        chars = len(code)

        # Calculate score based on goal
        if goal == "code_size":
            # Prefer smaller code
            score = max(0, 1 - (chars / 10000))
        elif goal == "performance":
            # Placeholder - would need actual benchmark
            score = 0.8  # Assume reasonable performance
        else:
            # Generic score
            score = 0.7

        return {
            "score": score,
            "syntax_valid": True,
            "lines_of_code": lines,
            "chars": chars,
        }

    return evaluator


def create_benchmark_from_description(description: str, metric: str) -> Callable:
    """Create a benchmark function from description"""

    def benchmark(instance: Any) -> Dict[str, Any]:
        import time

        # Placeholder benchmark - would be customized based on description
        start = time.time()

        try:
            # Try to run the algorithm
            if hasattr(instance, 'run'):
                result = instance.run([1, 2, 3, 4, 5])
            elif hasattr(instance, 'sort'):
                result = instance.sort([5, 2, 8, 1, 3])
            elif hasattr(instance, 'compute'):
                result = instance.compute(100)
            else:
                return {
                    "score": 0.0,
                    "error": "Unknown algorithm interface",
                }

            duration = time.time() - start

            if metric == "runtime":
                return {
                    "score": 1.0,
                    "runtime": duration,
                    "performance": 1.0 / (duration + 0.001),
                }
            else:
                return {
                    "score": 1.0,
                    "runtime": duration,
                }

        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e),
            }

    return benchmark


def generate_initial_algorithm(search_space: str) -> str:
    """Generate initial random algorithm code"""

    algorithms = {
        "sorting": """
# EVOLVE-BLOCK-START
def sort_algorithm(arr):
    # Initial: simple bubble sort
    for i in range(len(arr)):
        for j in range(len(arr) - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
# EVOLVE-BLOCK-END
""",
        "optimization": """
# EVOLVE-BLOCK-START
def optimize_function(func, bounds, max_iter=100):
    # Initial: random search
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
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
# EVOLVE-BLOCK-END
""",
    }

    return algorithms.get(search_space, algorithms["sorting"])


def create_problem_evaluator(description: str, constraints: List[str]) -> Callable:
    """Create evaluator for a specific problem"""

    def evaluator(program_path: str) -> Dict[str, Any]:
        import subprocess
        import sys

        # Syntax check
        try:
            subprocess.run(
                ["python", "-m", "py_compile", program_path],
                capture_output=True,
                timeout=10,
                check=True,
            )
        except Exception:
            return {
                "score": 0.0,
                "syntax_error": True,
            }

        # Load and test program
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_program", program_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Basic functionality check
            score = 0.5  # Base score for valid syntax

            # Check constraints
            for constraint in constraints:
                if "performance" in constraint.lower():
                    score += 0.1
                if "memory" in constraint.lower():
                    score += 0.1

            return {
                "score": min(1.0, score),
                "syntax_valid": True,
            }

        except Exception as e:
            return {
                "score": 0.0,
                "error": str(e),
            }

    return evaluator


def evaluate_prompt(prompt: str, test_case: Dict[str, Any], criteria: List[str]) -> float:
    """Evaluate a prompt against a test case"""
    # Placeholder - would use actual LLM to test prompt
    score = 0.5  # Base score

    # Check prompt characteristics
    if "clear" in criteria:
        if len(prompt.split('.')) >= 3:  # At least 3 sentences
            score += 0.1

    if "specific" in criteria:
        if any(keyword in prompt.lower() for keyword in ["example", "format", "ensure"]):
            score += 0.1

    if "concise" in criteria:
        if len(prompt) < 1000:  # Reasonable length
            score += 0.1

    return min(1.0, score)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all OpenEvolve MCP tools"""
    logger.info("Initializing OpenEvolve MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} OpenEvolve MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()
