"""
Test fixtures for PES testing

Provides reusable test fixtures for pytest, including:
- Sample configurations
- Mock problems and solutions
- Test data generators
- Helper functions
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import pytest


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for test files

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """
    Sample configuration for testing

    Returns:
        Dict with sample configuration
    """
    return {
        "max_generations": 10,
        "population_size": 5,
        "llm": {
            "model_name": "mock-model",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "evaluator": {
            "timeout": 10,
            "num_evaluations": 3
        },
        "database": {
            "enable_persistence": False
        },
        "island_model": {
            "num_islands": 1
        },
        "map_elites": {
            "enabled": False
        }
    }


@pytest.fixture
def simple_optimization_problem():
    """
    Simple optimization problem for testing

    Returns:
        Dict with problem definition
    """
    return {
        "type": "optimization",
        "objective": "minimize f(x) = x^2",
        "domain": "math",
        "constraints": {
            "x_range": [-10, 10],
            "precision": 0.01
        },
        "target_fitness": 0.0
    }


@pytest.fixture
def sample_program_code():
    """
    Sample program code for testing

    Returns:
        String with sample Python code
    """
    return '''def evaluate(x):
    """
    Simple quadratic function to minimize
    """
    return x * x

def main():
    x = 5.0
    return evaluate(x)
'''


@pytest.fixture
def sample_evaluation_script(temp_dir):
    """
    Create a sample evaluation script

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to evaluation script
    """
    eval_script = temp_dir / "eval_program.py"
    eval_script.write_text('''
import sys

def evaluate(program_output):
    """
    Evaluate the output of the evolved program
    Lower is better (minimization problem)
    """
    try:
        result = float(program_output.strip())
        # Minimize x^2, so fitness is just the result
        return result
    except (ValueError, TypeError):
        return 1e6  # High penalty for invalid output

if __name__ == "__main__":
    # Read output from evolved program
    output = sys.stdin.read()
    fitness = evaluate(output)
    print(f"{{'fitness': {fitness}}}")
''')
    return eval_script


@pytest.fixture
def sample_initial_program(temp_dir):
    """
    Create a sample initial program

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to initial program
    """
    program_file = temp_dir / "initial_program.py"
    program_file.write_text('''
def solve():
    """
    Initial solution for optimization
    """
    x = 5.0  # Starting point
    result = x * x
    return result

if __name__ == "__main__":
    print(solve())
''')
    return program_file


@pytest.fixture
def sample_config_file(temp_dir, sample_config):
    """
    Create a sample configuration file

    Args:
        temp_dir: Temporary directory fixture
        sample_config: Sample configuration fixture

    Returns:
        Path to config file
    """
    import yaml
    config_file = temp_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def mock_llm_client():
    """
    Mock LLM client for testing

    Returns:
        MockLLMClient instance
    """
    from openevolve.llm.mocks import MockLLMClient
    return MockLLMClient()


@pytest.fixture
def mock_llm_ensemble():
    """
    Mock LLM ensemble for testing

    Returns:
        MockLLMEnsemble instance
    """
    from openevolve.llm.mocks import MockLLMEnsemble
    return MockLLMEnsemble(num_models=3)


@pytest.fixture
def evolutionary_test_data():
    """
    Sample evolutionary data for testing

    Returns:
        Dict with test evolutionary data
    """
    return {
        "generation_0": {
            "programs": [
                {
                    "id": "prog_0_0",
                    "code": "def solve():\n    return 100",
                    "fitness": 100.0,
                    "parent_id": None
                },
                {
                    "id": "prog_0_1",
                    "code": "def solve():\n    return 50",
                    "fitness": 50.0,
                    "parent_id": None
                }
            ]
        },
        "generation_1": {
            "programs": [
                {
                    "id": "prog_1_0",
                    "code": "def solve():\n    return 25",
                    "fitness": 25.0,
                    "parent_id": "prog_0_1"
                },
                {
                    "id": "prog_1_1",
                    "code": "def solve():\n    return 10",
                    "fitness": 10.0,
                    "parent_id": "prog_0_1"
                }
            ]
        }
    }


@pytest.fixture
def sample_trace_data():
    """
    Sample evolution trace data

    Returns:
        Dict with trace data
    """
    return {
        "iterations": [
            {
                "iteration": 0,
                "best_fitness": 100.0,
                "avg_fitness": 75.0,
                "num_programs": 5,
                "improvement": 0.0
            },
            {
                "iteration": 1,
                "best_fitness": 50.0,
                "avg_fitness": 60.0,
                "num_programs": 5,
                "improvement": 50.0
            },
            {
                "iteration": 2,
                "best_fitness": 25.0,
                "avg_fitness": 45.0,
                "num_programs": 5,
                "improvement": 25.0
            }
        ],
        "best_overall": {
            "fitness": 25.0,
            "program_id": "prog_2_0"
        },
        "convergence_detected": False
    }


@pytest.fixture
def complex_optimization_problem():
    """
    More complex optimization problem

    Returns:
        Dict with complex problem definition
    """
    return {
        "type": "multi_objective",
        "objectives": [
            "minimize f1(x, y) = x^2 + y^2",
            "minimize f2(x, y) = (x-1)^2 + (y-1)^2"
        ],
        "domain": "math",
        "constraints": {
            "x_range": [-5, 5],
            "y_range": [-5, 5]
        },
        "pareto_front": True
    }


def create_mock_program(fitness: float, generation: int, program_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to create mock program data

    Args:
        fitness: Program fitness value
        generation: Generation number
        program_id: Optional program ID

    Returns:
        Dict with program data
    """
    if program_id is None:
        program_id = f"prog_{generation}_{fitness}"

    return {
        "id": program_id,
        "code": f"def solve():\n    return {fitness}",
        "fitness": fitness,
        "generation": generation,
        "parent_id": None,
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "evaluated": True,
            "evaluation_time": 0.1
        }
    }


def create_mock_trace(num_iterations: int, initial_fitness: float = 100.0) -> Dict[str, Any]:
    """
    Helper function to create mock evolution trace

    Args:
        num_iterations: Number of iterations to generate
        initial_fitness: Starting fitness value

    Returns:
        Dict with trace data
    """
    iterations = []
    current_best = initial_fitness

    for i in range(num_iterations):
        # Simulate improvement
        improvement = current_best * 0.5  # 50% improvement each iteration
        current_best = max(0.1, current_best - improvement)

        iterations.append({
            "iteration": i,
            "best_fitness": current_best,
            "avg_fitness": current_best * 1.2,
            "num_programs": 5,
            "improvement": improvement
        })

    return {
        "iterations": iterations,
        "best_overall": {
            "fitness": current_best,
            "program_id": f"prog_{num_iterations-1}_best"
        },
        "convergence_detected": num_iterations > 5
    }


@pytest.fixture
def convergence_test_config():
    """
    Configuration for testing convergence detection

    Returns:
        Dict with convergence test config
    """
    return {
        "max_generations": 100,
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "min_improvement": 0.01
        },
        "convergence_threshold": 0.001
    }


@pytest.fixture
def performance_benchmark_data():
    """
    Sample performance benchmark data

    Returns:
        Dict with benchmark data
    """
    return {
        "test_cases": [
            {
                "name": "simple_convex",
                "target_fitness": 0.01,
                "max_iterations": 20,
                "success_rate": 0.95
            },
            {
                "name": "multi_modal",
                "target_fitness": 0.1,
                "max_iterations": 50,
                "success_rate": 0.80
            }
        ],
        "benchmarks": {
            "avg_time_per_iteration": 0.5,
            "avg_improvement_rate": 0.3,
            "memory_usage_mb": 100
        }
    }
