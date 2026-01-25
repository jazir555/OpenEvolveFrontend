# API Documentation for Gaunt lets System

Complete API reference for all Gauntlet system components.

## Table of Contents

1. [Core Solving APIs](#core-solving-apis)
2. [Parallel Execution APIs](#parallel-execution-apis)
3. [Caching APIs](#caching-apis)
4. [Checkpointing APIs](#checkpointing-apis)
5. [Visualization APIs](#visualization-apis)
6. [Configuration APIs](#configuration-apis)
7. [Metrics APIs](#metrics-apis)
8. [Testing APIs](#testing-apis)

---

## Core Solving APIs

### solveProblem()

Main entry point for solving problems with automatic parallelization.

```python
async def solveProblem(
    problem: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    enable_parallel: bool = True,
    use_worker_pool: bool = False
) -> Dict[str, Any]
```

**Parameters:**
- `problem` (Dict): Problem to solve. Must contain `id` field
  - `id` (str): Unique problem identifier
  - `statement` (str): Problem description
  - `subproblems` (List[Dict], optional): Subproblems
  - `dependencies` (List[str], optional): Problem dependencies
- `context` (Dict, optional): Execution context
- `enable_parallel` (bool): Enable automatic parallel execution
- `use_worker_pool` (bool): Use worker pool instead of asyncio

**Returns:**
```python
{
    'problem_id': str,           # Problem identifier
    'success': bool,             # True if solved successfully
    'score': float,              # Solution quality score (0-1)
    'solution': Any,             # Solution object
    'confidence': float,         # Confidence in solution
    'team_id': str,              # Solving team identifier
    'timestamp': str,            # ISO timestamp
    'num_solutions': int,        # Number of sub-solutions (if applicable)
    'solutions': List[Dict]      # List of sub-solutions
}
```

**Example:**
```python
from bubblelabs_nodes import solveProblem

problem = {
    'id': 'my_problem',
    'statement': 'Solve this complex problem',
    'subproblems': [
        {'id': 's1', 'statement': 'Sub 1'},
        {'id': 's2', 'statement': 'Sub 2'},
    ]
}

solution = await solveProblem(problem, enable_parallel=True)

print(f"Success: {solution['success']}")
print(f"Score: {solution['score']}")
```

### GauntletSolver Class

```python
class GauntletSolver:
    def __init__(
        self,
        cache: Optional[AtomicSolutionCache] = None,
        parallel_executor: Optional[ParallelProblemExecutor] = None,
        worker_pool: Optional[WorkerPoolExecutor] = None,
        enable_parallel: bool = True,
        parallel_threshold: int = 3,
        use_worker_pool: bool = False
    )

    async def solve_problem(
        self,
        problem: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        force_sequential: bool = False
    ) -> Dict[str, Any]
```

**Methods:**
- `solve_problem()`: Solve a single problem
- `_solve_parallel()`: Solve using parallel execution
- `_solve_sequential()`: Solve sequentially
- `_detect_parallelizable_subproblems()`: Check if parallelizable
- `_should_use_parallel()`: Decide execution strategy

---

## Parallel Execution APIs

### ParallelProblemExecutor Class

```python
class ParallelProblemExecutor:
    def __init__(
        self,
        max_parallelism: int = 10,
        timeout_seconds: float = 300.0,
        stop_on_first_error: bool = False
    )

    async def execute_in_parallel(
        self,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any] = None
    ) -> ParallelExecutionSummary
```

**Parameters:**
- `max_parallelism`: Maximum concurrent operations
- `timeout_seconds`: Timeout per operation
- `stop_on_first_error`: Stop on first error

**Returns:** `ParallelExecutionSummary`
```python
{
    'total_count': int,           # Total problems
    'successful_count': int,       # Successfully solved
    'failed_count': int,           # Failed to solve
    'success_rate': float,         # Success rate (0-1)
    'results': List[Any],          # Results
    'errors': List[str],           # Error messages
    'total_time': float            # Total execution time
}
```

**Example:**
```python
from bubblelabs_nodes import get_parallel_executor

executor = get_parallel_executor(max_parallelism=5)

async def solve_func(problem):
    return {'id': problem['id'], 'solved': True}

result = await executor.execute_in_parallel(
    problems=[{'id': 'p1'}, {'id': 'p2'}],
    executor_func=solve_func,
    context={}
)

print(f"Solved: {result.successful_count}/{result.total_count}")
```

### ProblemDependencyAnalyzer Class

```python
class ProblemDependencyAnalyzer:
    def find_independent_problems(
        self,
        problems: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]

    def build_dependency_graph(
        self,
        problems: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]

    def topological_sort(
        self,
        graph: Dict[str, List[str]]
    ) -> List[str]
```

**Example:**
```python
from bubblelabs_nodes import ProblemDependencyAnalyzer

analyzer = ProblemDependencyAnalyzer()

# Find independent problems
problems = [
    {'id': 'p1', 'dependencies': []},
    {'id': 'p2', 'dependencies': ['p1']},
    {'id': 'p3', 'dependencies': []},
]

independent = analyzer.find_independent_problems(problems)
print(f"Independent: {[p['id'] for p in independent]}")

# Build dependency graph
graph = analyzer.build_dependency_graph(problems)
print(f"Graph: {graph}")

# Get execution order
ordered = analyzer.topological_sort(graph)
print(f"Execution order: {ordered}")
```

### WorkerPoolExecutor Class

```python
class WorkerPoolExecutor:
    def __init__(
        self,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
        enable_work_stealing: bool = True
    )

    async def execute_in_parallel(
        self,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any] = None
    ) -> PoolExecutionSummary

    async def execute_with_work_stealing(
        self,
        problems: List[Dict[str, Any]],
        executor_func: Callable,
        context: Dict[str, Any] = None
    ) -> PoolExecutionSummary
```

**Example:**
```python
from bubblelabs_nodes import create_worker_pool_executor

pool = create_worker_pool_executor(max_workers=4)

def cpu_bound_solve(problem):
    return {'id': problem['id'], 'result': 'computed'}

summary = await pool.execute_in_parallel(
    problems=[{'id': f'p{i}'} for i in range(10)],
    executor_func=cpu_bound_solve,
    context={}
)

print(f"Completed: {summary.successful_tasks}/{summary.total_tasks}")
```

---

## Caching APIs

### AtomicSolutionCache Class

```python
class AtomicSolutionCache:
    def __init__(
        self,
        cache_type: CacheType = CacheType.MEMORY,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        redis_url: Optional[str] = None
    )

    async def get(
        self,
        problem: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]

    async def set(
        self,
        problem: Dict[str, Any],
        solution: Dict[str, Any]
    ) -> bool

    async def has(
        self,
        problem_id: str
    ) -> bool

    async def invalidate(
        self,
        problem_id: str
    ) -> bool

    async def clear(self) -> bool

    def get_statistics(self) -> CacheStatistics
```

**Example:**
```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache(
    cache_type="memory",
    ttl_seconds=3600,
    max_size=1000
)

# Cache a solution
await cache.set(problem, solution)

# Retrieve a solution
cached = await cache.get(problem)

# Check if exists
exists = await cache.has(problem['id'])

# Get statistics
stats = cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### ProblemHasher Class

```python
class ProblemHasher:
    def normalize_problem(
        self,
        problem: Dict[str, Any]
    ) -> Dict[str, Any]

    def generate_hash(
        self,
        problem: Dict[str, Any]
    ) -> str
```

---

## Checkpointing APIs

### CheckpointManager Class

```python
class CheckpointManager:
    def __init__(
        self,
        storage_path: str = "./gauntlet_checkpoints",
        compression: bool = True,
        frequency: CheckpointFrequency = CheckpointFrequency.MAJOR,
        retention_count: int = 5
    )

    async def create_checkpoint(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any],
        solutions: Dict[str, Any],
        level: int,
        stage: str
    ) -> Optional[str]

    async def load_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[PipelineState]

    async def list_checkpoints(
        self,
        problem_id: Optional[str] = None
    ) -> List[Dict[str, Any]]

    async def delete_checkpoint(
        self,
        checkpoint_id: str
    ) -> bool

    async def cleanup_checkpoints(
        self,
        problem_id: str,
        keep_last_n: int = 5
    ) -> int
```

**Example:**
```python
from bubblelabs_nodes import create_checkpoint_manager

manager = create_checkpoint_manager(
    storage_path="./checkpoints",
    compression=True
)

# Create checkpoint
checkpoint_id = await manager.create_checkpoint(
    problem=problem,
    context={'stage': 'solving'},
    solutions={},
    level=0,
    stage='before_solve'
)

# Load checkpoint
state = await manager.load_checkpoint(checkpoint_id)

# List checkpoints
checkpoints = await manager.list_checkpoints(problem_id)

# Cleanup
deleted = await manager.cleanup_checkpoints(problem_id, keep_last_n=5)
```

---

## Configuration APIs

### create_config() Function

```python
def create_config(
    config_file: Optional[str] = None,
    profile: Optional[StrategyProfile] = None,
    from_env: bool = True
) -> GauntletConfig
```

**Example:**
```python
from bubblelabs_nodes import create_config, StrategyProfile

# From environment
config = create_config(from_env=True)

# With profile
config = create_config(profile=StrategyProfile.CONSERVATIVE)

# From file
config = create_config(config_file="./config.json")

# Combined
config = create_config(
    config_file="./base_config.json",
    profile=StrategyProfile.AGGRESSIVE,
    from_env=True
)
```

### GauntletConfig Class

```python
@dataclass
class GauntletConfig:
    # Sub-configurations
    cache: CacheConfig
    checkpointing: CheckpointConfig
    parallel_execution: ParallelExecutionConfig
    circuit_breaker: CircuitBreakerConfig
    fuzzing: FuzzingConfig
    difficulty: DifficultyConfig
    ml_decomposition: MLDecompositionConfig
    plugin: PluginConfig

    # General settings
    max_gauntlet_rounds: int
    pass_threshold: float
    max_decomposition_depth: int
    log_level: str
    strategy_profile: StrategyProfile

    def validate(self) -> Tuple[bool, List[str]]

    def apply_profile(self, profile: StrategyProfile) -> 'GauntletConfig'

    def save_to_file(self, filepath: str)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'GauntletConfig'
```

---

## Metrics APIs

### MetricsCollector Class

```python
class MetricsCollector:
    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None
    )

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    )

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    )

    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        metadata: Dict[str, Any] = None
    )

    def record_team_performance(
        self,
        team_id: str,
        problem_id: str,
        domain: str,
        difficulty: int,
        success: bool,
        score: float,
        execution_time: float
    )

    def get_all_metrics(self) -> Dict[str, Any]

    def start_resource_monitoring(self, interval_seconds: float = 1.0)

    def stop_resource_monitoring(self)
```

**Example:**
```python
from bubblelabs_nodes import get_metrics_collector

collector = get_metrics_collector()

# Record metrics
collector.increment("problems_solved")
collector.set_gauge("active_problems", 5)
collector.record_histogram("solve_time_ms", 150.5)
collector.record_performance("solve", 150.5, True)

# Team performance
collector.record_team_performance(
    team_id="blue_1",
    problem_id="p123",
    domain="web",
    difficulty=3,
    success=True,
    score=0.85,
    execution_time=150.0
)

# Get report
metrics = collector.get_all_metrics()
```

---

## Testing APIs

### TestDataGenerator Class

```python
class TestDataGenerator:
    def __init__(self, seed: Optional[int] = None)

    def generate_problem(
        self,
        complexity: str = "medium",
        domain: str = "general"
    ) -> Dict[str, Any]

    def generate_solution(
        self,
        problem: Dict[str, Any],
        success: bool = True,
        score: float = 0.8
    ) -> Dict[str, Any]

    def generate_decomposition_tree(
        self,
        depth: int = 3,
        branching_factor: int = 2
    ) -> Dict[str, Any]
```

**Example:**
```python
from bubblelabs_nodes import create_test_generator

generator = create_test_generator(seed=42)

# Generate test problem
problem = generator.generate_problem("medium", "web")

# Generate test solution
solution = generator.generate_solution(problem)

# Generate test tree
tree = generator.generate_decomposition_tree(depth=3)
```

### ValidationHelper Class

```python
class ValidationHelper:
    @staticmethod
    def validate_problem(problem: Dict[str, Any]) -> ValidationReport

    @staticmethod
    def validate_solution(solution: Dict[str, Any]) -> ValidationReport

    @staticmethod
    def validate_decomposition_tree(tree: Dict[str, Any]) -> ValidationReport

    @staticmethod
    def validate_checkpoint_state(state: Any) -> ValidationReport
```

**Example:**
```python
from bubblelabs_nodes import ValidationHelper

# Validate problem
report = ValidationHelper.validate_problem(problem)

if not report.is_valid:
    print(f"Errors: {report.errors}")
```

---

## Summary

This API documentation covers:
- ✅ Core solving APIs (solveProblem, GauntletSolver)
- ✅ Parallel execution APIs (ParallelProblemExecutor, WorkerPoolExecutor)
- ✅ Dependency analysis APIs (ProblemDependencyAnalyzer)
- ✅ Caching APIs (AtomicSolutionCache, ProblemHasher)
- ✅ Checkpointing APIs (CheckpointManager)
- ✅ Configuration APIs (create_config, GauntletConfig)
- ✅ Metrics APIs (MetricsCollector)
- ✅ Testing APIs (TestDataGenerator, ValidationHelper)

For implementation details, see:
- `bubblelabs_nodes/` for source code
- `CONFIGURATION_GUIDE.md` for configuration
- `METRICS_GUIDE.md` for monitoring
