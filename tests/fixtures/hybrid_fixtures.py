"""
Test fixtures for hybrid system E2E tests.

This module provides reusable fixtures and helper functions for testing
the hybrid OpenEvolve LoongFlow PES system.

Fixtures:
- Test problems (various types and complexities)
- Hybrid tasks (PES + Evolution)
- Adapter mocks and configuration
- Knowledge artifacts
- Test data generators

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class ProblemType(Enum):
    """Problem types for testing."""
    OPTIMIZATION = "OPTIMIZATION"
    REASONING = "REASONING"
    COMPLEX_OPTIMIZATION = "COMPLEX_OPTIMIZATION"
    FORMAL_VERIFICATION = "FORMAL_VERIFICATION"
    LONG_OPTIMIZATION = "LONG_OPTIMIZATION"
    LARGE_SCALE_OPTIMIZATION = "LARGE_SCALE_OPTIMIZATION"


class IntegrationStrategy(Enum):
    """Integration strategies for hybrid workflows."""
    SEQUENTIAL = "SEQUENTIAL"
    PARALLEL = "PARALLEL"
    ADAPTIVE = "ADAPTIVE"
    ITERATIVE = "ITERATIVE"


class TaskStatus(Enum):
    """Task status values."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TestProblem:
    """Test problem data structure."""
    id: str
    type: str
    description: str
    context: Dict[str, Any]
    constraints: List[str]
    success_criteria: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestProblem':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class HybridTask:
    """Hybrid task data structure."""
    id: str
    type: str
    problem: Dict[str, Any]
    pes_config: Dict[str, Any]
    evolution_config: Dict[str, Any]
    integration_strategy: str
    created_at: str
    status: str = TaskStatus.PENDING.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExecutionResult:
    """Execution result data structure."""
    execution_id: str
    task_id: str
    status: str
    result: Dict[str, Any]
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class KnowledgeFragment:
    """Knowledge fragment data structure."""
    id: str
    source_type: str
    problem_id: Optional[str]
    pattern: str
    success_rate: float
    avg_score: float
    usage_count: int
    extracted_at: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize defaults."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# FIXTURE GENERATORS
# ============================================================================

def create_test_problem(
    problem_type: str = ProblemType.OPTIMIZATION.value,
    description: str = "Test optimization problem",
    context: Dict[str, Any] = None,
    constraints: List[str] = None,
    success_criteria: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a test problem with default values.

    Args:
        problem_type: Type of problem
        description: Problem description
        context: Additional context
        constraints: Problem constraints
        success_criteria: Success criteria
        **kwargs: Additional fields

    Returns:
        Test problem dictionary
    """
    return {
        'id': str(uuid.uuid4()),
        'type': problem_type,
        'description': description,
        'context': context or {},
        'constraints': constraints or [],
        'success_criteria': success_criteria or [],
        'created_at': datetime.now(timezone.utc).isoformat(),
        **kwargs
    }


def create_hybrid_task(
    task_type: str = "PES_OPTIMIZE",
    problem_type: str = ProblemType.OPTIMIZATION.value,
    problem_description: str = "Test optimization problem",
    pes_iterations: int = 10,
    pes_islands: int = 3,
    evolution_generations: int = 5,
    evolution_population: int = 20,
    integration_strategy: str = IntegrationStrategy.SEQUENTIAL.value,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a hybrid task for testing.

    Args:
        task_type: Type of hybrid task
        problem_type: Type of problem to solve
        problem_description: Problem description
        pes_iterations: Number of PES iterations
        pes_islands: Number of PES islands
        evolution_generations: Number of evolution generations
        evolution_population: Evolution population size
        integration_strategy: How to integrate PES and Evolution
        **kwargs: Additional fields

    Returns:
        Hybrid task dictionary
    """
    return {
        'id': str(uuid.uuid4()),
        'type': task_type,
        'problem': create_test_problem(
            problem_type=problem_type,
            description=problem_description
        ),
        'pes_config': {
            'iterations': pes_iterations,
            'islands': pes_islands,
            'population_size': 50
        },
        'evolution_config': {
            'generations': evolution_generations,
            'population_size': evolution_population,
            'mutation_rate': 0.1
        },
        'integration_strategy': integration_strategy,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': TaskStatus.PENDING.value,
        **kwargs
    }


def create_execution_result(
    execution_id: str = None,
    task_id: str = None,
    status: str = TaskStatus.COMPLETED.value,
    result: Dict[str, Any] = None,
    error: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create an execution result for testing.

    Args:
        execution_id: Execution ID
        task_id: Task ID
        status: Execution status
        result: Execution result data
        error: Error message if failed
        **kwargs: Additional fields

    Returns:
        Execution result dictionary
    """
    now = datetime.now(timezone.utc).isoformat()

    return {
        'execution_id': execution_id or str(uuid.uuid4()),
        'task_id': task_id or str(uuid.uuid4()),
        'status': status,
        'result': result or {},
        'started_at': now,
        'completed_at': now if status == TaskStatus.COMPLETED.value else None,
        'error': error,
        **kwargs
    }


def create_knowledge_fragment(
    source_type: str = "LOONGFLOW_SOLUTION",
    problem_id: str = None,
    pattern: str = "Test pattern",
    success_rate: float = 0.9,
    avg_score: float = 0.85,
    usage_count: int = 0,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a knowledge fragment for testing.

    Args:
        source_type: Type of knowledge source
        problem_id: Associated problem ID
        pattern: Knowledge pattern
        success_rate: Success rate (0-1)
        avg_score: Average score (0-1)
        usage_count: Number of times used
        metadata: Additional metadata

    Returns:
        Knowledge fragment dictionary
    """
    return {
        'id': str(uuid.uuid4()),
        'source_type': source_type,
        'problem_id': problem_id,
        'pattern': pattern,
        'success_rate': success_rate,
        'avg_score': avg_score,
        'usage_count': usage_count,
        'extracted_at': datetime.now(timezone.utc).isoformat(),
        'metadata': metadata or {}
    }


# ============================================================================
# PRE-DEFINED FIXTURES
# ============================================================================

# Optimization problems
OPTIMIZATION_PROBLEMS = {
    'simple': create_test_problem(
        problem_type=ProblemType.OPTIMIZATION.value,
        description="Maximize f(x) = x^2 for x in [0, 10]",
        context={'domain': 'mathematical_optimization', 'difficulty': 'easy'},
        constraints=['x >= 0', 'x <= 10'],
        success_criteria=['fitness > 0.9']
    ),
    'portfolio': create_test_problem(
        problem_type=ProblemType.OPTIMIZATION.value,
        description="Optimize portfolio allocation for maximum returns",
        context={'domain': 'finance', 'difficulty': 'medium'},
        constraints=['sum_weights = 1.0', 'all_weights >= 0'],
        success_criteria=['sharpe_ratio > 1.5', 'volatility < 0.2']
    ),
    'neural_network': create_test_problem(
        problem_type=ProblemType.OPTIMIZATION.value,
        description="Optimize neural network architecture",
        context={'domain': 'machine_learning', 'difficulty': 'hard'},
        constraints=['layers <= 10', 'parameters <= 1M'],
        success_criteria=['accuracy > 0.95', 'training_time < 1000s']
    )
}

# Reasoning problems
REASONING_PROBLEMS = {
    'math_proof': create_test_problem(
        problem_type=ProblemType.REASONING.value,
        description="Prove that sqrt(2) is irrational",
        context={'domain': 'mathematics', 'difficulty': 'medium'},
        success_criteria=['proof_valid', 'steps_complete']
    ),
    'algorithm': create_test_problem(
        problem_type=ProblemType.REASONING.value,
        description="Design an efficient sorting algorithm",
        context={'domain': 'computer_science', 'difficulty': 'medium'},
        success_criteria=['time_complexity = O(n log n)', 'space_complexity = O(n)']
    )
}

# Hybrid tasks
HYBRID_TASKS = {
    'sequential': create_hybrid_task(
        task_type="PES_OPTIMIZE",
        problem_type=ProblemType.OPTIMIZATION.value,
        problem_description="Optimize with sequential PES then evolution",
        pes_iterations=10,
        evolution_generations=5,
        integration_strategy=IntegrationStrategy.SEQUENTIAL.value
    ),
    'adaptive': create_hybrid_task(
        task_type="PES_OPTIMIZE_ADAPTIVE",
        problem_type=ProblemType.COMPLEX_OPTIMIZATION.value,
        problem_description="Optimize with adaptive paradigm switching",
        pes_iterations=15,
        evolution_generations=7,
        integration_strategy=IntegrationStrategy.ADAPTIVE.value
    ),
    'iterative': create_hybrid_task(
        task_type="PES_EVOLVE_ITERATE",
        problem_type=ProblemType.OPTIMIZATION.value,
        problem_description="Iteratively apply PES and evolution",
        pes_iterations=5,
        evolution_generations=3,
        integration_strategy=IntegrationStrategy.ITERATIVE.value
    )
}


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_batch_problems(count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate a batch of test problems.

    Args:
        count: Number of problems to generate

    Returns:
        List of test problems
    """
    problems = []
    problem_types = [
        ProblemType.OPTIMIZATION.value,
        ProblemType.REASONING.value,
        ProblemType.COMPLEX_OPTIMIZATION.value
    ]

    for i in range(count):
        problem_type = problem_types[i % len(problem_types)]
        problems.append(
            create_test_problem(
                problem_type=problem_type,
                description=f"Batch problem {i}",
                context={'batch_index': i}
            )
        )

    return problems


def generate_evolutionary_population(
    size: int = 20,
    solution_template: str = "def solve(x): return x"
) -> List[Dict[str, Any]]:
    """
    Generate an evolutionary population.

    Args:
        size: Population size
        solution_template: Template for solutions

    Returns:
        List of individuals with fitness
    """
    population = []
    for i in range(size):
        population.append({
            'solution': f"{solution_template} * {i}",
            'fitness': 0.5 + (i / size) * 0.5  # 0.5 to 1.0
        })
    return population


def generate_knowledge_base(
    count: int = 50,
    source_types: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate a knowledge base.

    Args:
        count: Number of knowledge fragments
        source_types: Types of knowledge sources

    Returns:
        List of knowledge fragments
    """
    if source_types is None:
        source_types = ['LOONGFLOW_SOLUTION', 'EVOLUTION_RESULT', 'USER_FEEDBACK']

    knowledge_base = []
    patterns = [
        'Use gradient descent',
        'Apply evolutionary strategy',
        'Combine multiple approaches',
        'Use adaptive learning rates',
        'Implement early stopping'
    ]

    for i in range(count):
        source_type = source_types[i % len(source_types)]
        pattern = patterns[i % len(patterns)]
        success_rate = 0.6 + (i % 40) / 100  # 0.6 to 1.0

        knowledge_base.append(
            create_knowledge_fragment(
                source_type=source_type,
                pattern=pattern,
                success_rate=success_rate,
                avg_score=success_rate - 0.05,
                metadata={'index': i}
            )
        )

    return knowledge_base


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_problem(problem: Dict[str, Any]) -> bool:
    """
    Validate a test problem structure.

    Args:
        problem: Problem to validate

    Returns:
        True if valid
    """
    required_fields = ['id', 'type', 'description', 'created_at']
    return all(field in problem for field in required_fields)


def validate_hybrid_task(task: Dict[str, Any]) -> bool:
    """
    Validate a hybrid task structure.

    Args:
        task: Task to validate

    Returns:
        True if valid
    """
    required_fields = [
        'id', 'type', 'problem', 'pes_config',
        'evolution_config', 'integration_strategy'
    ]
    return all(field in task for field in required_fields)


def validate_execution_result(result: Dict[str, Any]) -> bool:
    """
    Validate an execution result structure.

    Args:
        result: Result to validate

    Returns:
        True if valid
    """
    required_fields = ['execution_id', 'task_id', 'status', 'started_at']
    return all(field in result for field in required_fields)


def validate_knowledge_fragment(fragment: Dict[str, Any]) -> bool:
    """
    Validate a knowledge fragment structure.

    Args:
        fragment: Fragment to validate

    Returns:
        True if valid
    """
    required_fields = ['id', 'source_type', 'pattern', 'success_rate']
    valid = all(field in fragment for field in required_fields)

    if valid:
        # Validate success_rate range
        success_rate = fragment.get('success_rate', 0)
        valid = 0 <= success_rate <= 1

    return valid


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'ProblemType',
    'IntegrationStrategy',
    'TaskStatus',

    # Data classes
    'TestProblem',
    'HybridTask',
    'ExecutionResult',
    'KnowledgeFragment',

    # Generators
    'create_test_problem',
    'create_hybrid_task',
    'create_execution_result',
    'create_knowledge_fragment',

    # Pre-defined fixtures
    'OPTIMIZATION_PROBLEMS',
    'REASONING_PROBLEMS',
    'HYBRID_TASKS',

    # Test data generators
    'generate_batch_problems',
    'generate_evolutionary_population',
    'generate_knowledge_base',

    # Validators
    'validate_problem',
    'validate_hybrid_task',
    'validate_execution_result',
    'validate_knowledge_fragment'
]
