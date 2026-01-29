# Parallel Execution Usage Guide

Complete guide for using parallel execution in the OpenEvolve Gauntlet System.

## Table of Contents

1. [Overview](#overview)
2. [When to Use Parallel vs Sequential](#when-to-use-parallel-vs-sequential)
3. [API Reference](#api-reference)
4. [Configuration](#configuration)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Overview

Parallel execution solves multiple independent problems concurrently, providing 50-80% performance improvement on multi-problem hierarchies.

### Key Benefits

- **Speed**: 50-80% faster for 3+ problems
- **Efficiency**: Automatic dependency detection
- **Safety**: Circuit breaker fault isolation
- **Scalability**: Configurable concurrency limits

### Quick Start

```python
from bubblelabs_nodes import solveProblem

# Problem with 4 independent subproblems
problem = {
    'type': 'hierarchy',
    'subproblems': [
        {'id': 'p1', 'data': {...}},
        {'id': 'p2', 'data': {...}},
        {'id': 'p3', 'data': {...}},
        {'id': 'p4', 'data': {...}},
    ]
}

# Automatic parallel execution
result = await solveProblem(problem)
```

---

## When to Use Parallel vs Sequential

### Use Parallel When:

1. **Multiple Independent Problems**
   ```python
   # GOOD: 4 independent problems
   problem = {
       'subproblems': [
           {'task': 'task1'},
           {'task': 'task2'},
           {'task': 'task3'},
           {'task': 'task4'},
       ]
   }
   ```

2. **No Dependencies Between Problems**
   ```python
   # GOOD: Problems don't reference each other
   problem = {
       'subproblems': [
           {'input': 'data1'},
           {'input': 'data2'},
       ]
   }
   ```

3. **CPU-Bound Operations**
   ```python
   # GOOD: Heavy computation
   problem = {
       'subproblems': [
           {'computation': 'heavy'},
       ]
   }
   ```

### Use Sequential When:

1. **Dependencies Exist**
   ```python
   # BAD: p2 depends on p1
   problem = {
       'subproblems': [
           {'id': 'p1'},
           {'id': 'p2', 'requires': 'p1'},
       ]
   }
   ```

2. **Shared State**
   ```python
   # BAD: Problems modify shared resource
   problem = {
       'subproblems': [
           {'resource': '/tmp/file'},
           {'resource': '/tmp/file'},  # Conflict!
       ]
   }
   ```

3. **Few Problems (< 3)**
   ```python
   # Not worth parallel overhead
   problem = {
       'subproblems': [
           {'task': 'task1'},
           {'task': 'task2'},
       ]
   }
   ```

---

## API Reference

### solveProblem()

```python
async def solveProblem(
    problem: Dict[str, Any],
    context: Dict[str, Any] = None,
    force_sequential: bool = False
) -> Dict[str, Any]:
    """
    Solve a problem (possibly with parallel subproblems).

    Args:
        problem: Problem definition
        context: Execution context
        force_sequential: Force sequential execution

    Returns:
        Solution result
    """
```

### ParallelProblemExecutor

```python
from bubblelabs_nodes import ParallelProblemExecutor

executor = ParallelProblemExecutor(
    max_parallelism=10,
    timeout_seconds=300
)

async def solve_multiple(problems):
    results = await executor.execute_in_parallel(
        problems=[p1, p2, p3],
        executor_func=solve_atomic,
        context={}
    )
    return results
```

### Configuration

```python
from bubblelabs_nodes import ParallelExecutionConfig

config = ParallelExecutionConfig(
    enabled=True,
    max_parallelism=10,
    timeout_seconds=300,
    use_worker_pool=False,
    worker_pool_size=4
)
```

---

## Configuration

### Environment Variables

```bash
# Enable parallel execution
PARALLEL_EXECUTION_ENABLED=true

# Maximum concurrent operations
PARALLEL_MAX_PARALLELISM=10

# Operation timeout (seconds)
PARALLEL_TIMEOUT_SECONDS=300

# Use worker pool for CPU-bound tasks
PARALLEL_USE_WORKER_POOL=false

# Worker pool size
PARALLEL_WORKER_POOL_SIZE=4
```

### Code Configuration

```python
from bubblelabs_nodes import GauntletSolver

solver = GauntletSolver(
    enable_parallel=True,
    parallel_threshold=3,
    use_worker_pool=False
)
```

---

## Best Practices

### 1. Set Appropriate Parallelism

```python
# Too low: Underutilizes resources
executor = ParallelProblemExecutor(max_parallelism=2)

# Good: Balanced
executor = ParallelProblemExecutor(max_parallelism=10)

# Too high: Can overwhelm system
executor = ParallelProblemExecutor(max_parallelism=100)
```

### 2. Use Timeouts

```python
# Prevents hanging operations
executor = ParallelProblemExecutor(
    max_parallelism=10,
    timeout_seconds=300  # 5 minute timeout
)
```

### 3. Handle Partial Failures

```python
results = await executor.execute_in_parallel(
    problems=[p1, p2, p3],
    executor_func=solve,
    context={}
)

# Check individual results
for i, (success, result, error) in enumerate(results):
    if success:
        print(f"Problem {i}: SUCCESS")
    else:
        print(f"Problem {i}: FAILED - {error}")
```

### 4. Monitor Performance

```python
from bubblelabs_nodes import get_metrics_collector

metrics = get_metrics_collector()

# Track parallel execution
await metrics.increment_counter('parallel_executions_total')
await metrics.set_gauge('parallel_active_tasks', 5)
await metrics.observe_histogram('parallel_execution_time', 15.3)
```

---

## Troubleshooting

### Issue 1: No Performance Improvement

**Symptoms:** Parallel execution same speed as sequential

**Diagnosis:**
```python
summary = await executor.execute_in_parallel(...)
print(f"Speedup: {summary.actual_time / summary.sequential_time}")
```

**Solutions:**
- Check if problems are truly independent
- Increase parallelism threshold
- Verify no shared resources
- Check for lock contention

### Issue 2: Timeouts

**Symptoms:** Operations timing out

**Diagnosis:**
```python
# Check timeout setting
print(f"Timeout: {executor.timeout_seconds}s")
```

**Solutions:**
- Increase timeout
- Optimize slow operations
- Reduce parallelism
- Use worker pool for CPU-bound tasks

### Issue 3: High Memory Usage

**Symptoms:** Memory usage increasing

**Diagnosis:**
```python
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

**Solutions:**
- Reduce max_parallelism
- Process problems in batches
- Clear caches between batches

### Issue 4: Inconsistent Results

**Symptoms:** Different results on each run

**Diagnosis:**
```python
# Check for race conditions
for result in results:
    print(f"Result: {result}")
```

**Solutions:**
- Verify no shared state
- Check for proper synchronization
- Use atomic operations
- Add locks where needed

---

## Performance Tips

### 1. Profile Before Optimizing

```python
import cProfile
import pstats

async def profile_solve():
    profiler = cProfile.Profile()
    profiler.enable()

    result = await solveProblem(problem)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(10)
```

### 2. Use Appropriate Granularity

```python
# GOOD: Coarse-grained parallelism
problem = {
    'subproblems': [p1, p2, p3, p4]  # 4 big tasks
}

# BAD: Fine-grained overhead
problem = {
    'subproblems': [t1, t2, ..., t100]  # 100 tiny tasks
}
```

### 3. Batch Similar Operations

```python
# Process in batches
problems = [...all_problems...]

batch_size = 10
for i in range(0, len(problems), batch_size):
    batch = problems[i:i+batch_size]
    await execute_batch(batch)
```

---

## Migration Guide

### From Sequential to Parallel

**Before (Sequential):**
```python
async def solve_all_problems(problems):
    results = []
    for problem in problems:
        result = await solve(problem)
        results.append(result)
    return results
```

**After (Parallel):**
```python
async def solve_all_problems(problems):
    executor = ParallelProblemExecutor()
    results = await executor.execute_in_parallel(
        problems=problems,
        executor_func=solve,
        context={}
    )
    return results
```

---

## Summary

Parallel execution provides:
- ✅ 50-80% speedup on multi-problem hierarchies
- ✅ Automatic dependency detection
- ✅ Configurable concurrency
- ✅ Fault isolation with circuit breakers
- ✅ Comprehensive monitoring

For more information:
- `bubblelabs_nodes/parallel_executor.py` - Implementation
- `bubblelabs_nodes/gauntlet_solver.py` - High-level API
- `METRICS_GUIDE.md` - Monitoring guide
