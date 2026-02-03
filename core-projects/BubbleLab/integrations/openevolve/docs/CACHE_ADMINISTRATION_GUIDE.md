# Cache Administration Guide

Complete guide for monitoring, managing, and troubleshooting the solution cache system in the OpenEvolve Gauntlet System.

## Table of Contents

1. [Overview](#overview)
2. [Monitoring Cache Performance](#monitoring-cache-performance)
3. [Cache Management Operations](#cache-management-operations)
4. [Cache Warming Strategies](#cache-warming-strategies)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Overview

The solution cache stores computed solutions to avoid redundant computation, providing 100x speedup on cache hits.

### Cache Architecture

```
┌─────────────────────────────────────────────────┐
│                   Application                    │
├─────────────────────────────────────────────────┤
│              solveAtomicProblem()                │
│                   ↓                               │
│            ┌─────────────┐                       │
│            │  Cache Hit? │                       │
│            └──────┬──────┘                       │
│              ↓           ↓                        │
│           Yes            No                      │
│           ↓               ↓                       │
│      Return Solution    Solve Problem            │
│                          ↓                        │
│                    Store Solution                 │
└─────────────────────────────────────────────────┘
```

### Cache Statistics

- **Hit Rate**: Percentage of cache requests that return cached solutions
- **Miss Rate**: Percentage of cache requests that require solving
- **Evictions**: Number of entries removed due to size limit
- **Size**: Current number of cached solutions

---

## Monitoring Cache Performance

### Prometheus Metrics

The cache exposes the following metrics to Prometheus:

```prometheus
# Hit rate as percentage (0-100)
cache_hit_rate{cache_type="memory"}

# Total cache hits
cache_hit_count_total

# Total cache misses
cache_miss_count_total

# Current cache size
cache_size{cache_type="memory"}

# Total evictions
cache_eviction_count_total
```

### Monitoring Dashboards

#### Grafana Dashboard Example

```json
{
  "title": "Solution Cache Performance",
  "panels": [
    {
      "title": "Cache Hit Rate",
      "targets": [
        {
          "expr": "cache_hit_rate"
        }
      ]
    },
    {
      "title": "Cache Size Over Time",
      "targets": [
        {
          "expr": "cache_size"
        }
      ]
    },
    {
      "title": "Hit/Miss Ratio",
      "targets": [
        {
          "expr": "rate(cache_hit_count_total[5m]) / rate(cache_miss_count_total[5m])"
        }
      ]
    }
  ]
}
```

### Health Status

Check cache health using the `CacheMonitor`:

```python
from bubblelabs_nodes import create_monitored_cache

cache = create_monitored_cache()
health = await cache.monitor.get_health_status()

print(f"Health: {health['health']}")
print(f"Hit Rate: {health['hit_rate']:.2%}")
print(f"Size: {health['size']}")
print(f"Uptime: {health['uptime_seconds']:.0f}s")
```

**Health Levels**:
- **healthy**: Hit rate >= 50%
- **degraded**: Hit rate >= 30% and < 50%
- **unhealthy**: Hit rate < 30%

---

## Cache Management Operations

### Manual Cache Invalidation

#### Invalidate Specific Problem

```python
from bubblelabs_nodes import create_monitored_cache

cache = create_monitored_cache()

problem = {'type': 'math', 'operation': 'add', 'a': 5, 'b': 3}
await cache.invalidate(problem)
```

#### Clear Entire Cache

```python
# Clears all cached solutions
await cache.clear()
```

#### Invalidate by Pattern

```python
from bubblelabs_nodes.solution_cache import ProblemHasher

# Invalidate all math operations
hasher = ProblemHasher()
# Get all keys and filter by pattern
# (Requires custom implementation for pattern matching)
```

### Cache Statistics

```python
# Get comprehensive statistics
stats = await cache.get_statistics()

print(f"Hits: {stats['hits']}")
print(f"Misses: {stats['misses']}")
print(f"Hit Rate: {stats['hit_rate']:.2%}")
print(f"Miss Rate: {stats['miss_rate']:.2%}")
print(f"Size: {stats['size']}")
print(f"Evictions: {stats['evictions']}")
```

### Cache Inspection

#### Check if Problem is Cached

```python
problem = {'type': 'math', 'operation': 'add', 'a': 5, 'b': 3}

is_cached = await cache.has(problem)
if is_cached:
    solution, hit = await cache.get(problem)
    print(f"Cached solution: {solution}")
else:
    print("Not in cache")
```

#### Get Cache Size

```python
size = await cache.size()
print(f"Cache contains {size} solutions")
```

---

## Cache Warming Strategies

Cache warming pre-loads frequently used problems into the cache to improve performance.

### Strategy 1: Pre-load Common Problems

```python
from bubblelabs_nodes import create_monitored_cache
import asyncio

async def warm_cache_common_problems():
    """Pre-load common problem types"""
    cache = create_monitored_cache()

    common_problems = [
        {'type': 'math', 'operation': 'add', 'a': 1, 'b': 1},
        {'type': 'math', 'operation': 'multiply', 'a': 2, 'b': 2},
        # Add more common problems...
    ]

    for problem in common_problems:
        # Solve and cache
        solution = await solve_problem(problem)
        await cache.set(problem, solution)

    print(f"Warmed cache with {len(common_problems)} solutions")

asyncio.run(warm_cache_common_problems())
```

### Strategy 2: Load from Historical Data

```python
async def warm_cache_from_history():
    """Load previously solved problems from database"""
    cache = create_monitored_cache()

    # Fetch recent solutions from database
    recent_solutions = await fetch_recent_solutions(limit=1000)

    for record in recent_solutions:
        problem = record['problem']
        solution = record['solution']
        await cache.set(problem, solution)

    print(f"Warmed cache with {len(recent_solutions)} historical solutions")
```

### Strategy 3: Predictive Warming

```python
async def predictive_cache_warming():
    """Pre-load problems likely to be requested"""
    cache = create_monitored_cache()

    # Get statistics on most frequently accessed problems
    stats = await cache.get_statistics()

    # Identify patterns (e.g., math operations 9-5 PM)
    # Pre-load predicted problems
    predicted_problems = predict_likely_problems(stats)

    for problem in predicted_problems:
        solution = await solve_problem(problem)
        await cache.set(problem, solution)
```

### Strategy 4: Scheduled Warming

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

async def scheduled_warm_job():
    """Run cache warming on schedule"""
    await warm_cache_from_history()
    await warm_cache_common_problems()

# Schedule warming every hour
scheduler.add_job(scheduled_warm_job, 'interval', hours=1)
scheduler.start()
```

---

## Troubleshooting

### Issue 1: Low Hit Rate

**Symptoms**: Hit rate below 30%

**Diagnosis**:
```python
stats = await cache.get_statistics()
print(f"Hit Rate: {stats['hit_rate']:.2%}")
```

**Solutions**:

1. **Increase TTL**: Solutions are expiring too quickly
   ```python
   # In CacheConfig
   ttl_seconds=7200  # 2 hours instead of 1 hour
   ```

2. **Increase Cache Size**: Cache is evicting too frequently
   ```python
   # In CacheConfig
   max_size=5000  # Increase from 1000
   ```

3. **Check Cache Key**: Problems may not be normalized correctly
   ```python
   from bubblelabs_nodes.solution_cache import ProblemHasher

   hasher = ProblemHasher()
   normalized = hasher.normalize_problem(problem)
   print(f"Normalized: {normalized}")
   ```

4. **Enable Cache Warming**: Pre-load common problems
   ```python
   await warm_cache_from_history()
   ```

### Issue 2: High Memory Usage

**Symptoms**: Memory usage increasing over time

**Diagnosis**:
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.0f} MB")
```

**Solutions**:

1. **Reduce Max Size**: Limit cache size
   ```python
   # In CacheConfig
   max_size=500  # Reduce from 1000
   ```

2. **Reduce TTL**: Solutions expire faster
   ```python
   # In CacheConfig
   ttl_seconds=1800  # 30 minutes instead of 1 hour
   ```

3. **Clear Cache**: Manually clear if needed
   ```python
   await cache.clear()
   ```

4. **Use LRU Eviction**: Ensure LRU is enabled (default)
   ```python
   # In InMemoryCache
   InMemoryCache(max_size=1000, ttl=3600)
   ```

### Issue 3: Cache Not Working

**Symptoms**: No cache hits, all requests miss

**Diagnosis**:
```python
from bubblelabs_nodes.gauntlet_config import CacheConfig

config = CacheConfig()
print(f"Enabled: {config.enabled}")
print(f"Type: {config.cache_type.value}")
```

**Solutions**:

1. **Check Configuration**: Ensure cache is enabled
   ```bash
   # Set environment variable
   export CACHE_ENABLED=true
   ```

2. **Check Integration**: Verify cache is integrated
   ```python
   # In solveAtomicProblem()
   # Should include cache lookup
   ```

3. **Test Cache Directly**:
   ```python
   cache = create_monitored_cache()

   test_problem = {'type': 'test', 'value': 42}
   test_solution = {'result': 'success'}

   await cache.set(test_problem, test_solution)
   retrieved, hit = await cache.get(test_problem)

   assert hit is True
   assert retrieved == test_solution
   ```

### Issue 4: Stale Solutions

**Symptoms**: Cache returns outdated solutions

**Diagnosis**:
```python
problem = {'type': 'test', 'value': 42}
solution, hit = await cache.get(problem)

if hit:
    # Verify solution is current
    is_valid = await verify_solution(solution)
    if not is_valid:
        await cache.invalidate(problem)
```

**Solutions**:

1. **Reduce TTL**: Solutions expire faster
   ```python
   # In CacheConfig
   ttl_seconds=600  # 10 minutes
   ```

2. **Implement Versioning**: Add version to cache key
   ```python
   problem = {'type': 'test', 'value': 42, 'version': 2}
   ```

3. **Manual Invalidation**: Invalidate when solutions change
   ```python
   await cache.invalidate(problem)
   ```

### Issue 5: Slow Cache Performance

**Symptoms**: Cache hits are slow

**Diagnosis**:
```python
import time

start = time.time()
solution, hit = await cache.get(problem)
duration = time.time() - start

print(f"Cache lookup took {duration * 1000:.0f}ms")
```

**Solutions**:

1. **Use In-Memory Cache**: Faster than Redis
   ```python
   # In CacheConfig
   cache_type=CacheType.MEMORY
   ```

2. **Optimize Hashing**: Ensure hashing is efficient
   ```python
   from bubblelabs_nodes.solution_cache import ProblemHasher

   hasher = ProblemHasher()
   hash_val = hasher.generate_hash(problem)
   ```

3. **Reduce Lock Contention**: Check for concurrent access
   ```python
   # Cache uses asyncio.Lock internally
   # Ensure operations are async
   ```

---

## Best Practices

### 1. Set Appropriate TTL

```python
# Short TTL for frequently changing data
CacheConfig(ttl_seconds=600)  # 10 minutes

# Long TTL for stable problems
CacheConfig(ttl_seconds=7200)  # 2 hours
```

### 2. Monitor Cache Health

```python
# Regular health checks
async def monitor_cache_health():
    cache = create_monitored_cache()
    health = await cache.monitor.get_health_status()

    if health['health'] == 'unhealthy':
        # Alert or take action
        send_alert(f"Cache unhealthy: hit rate {health['hit_rate']:.2%}")
```

### 3. Use Cache Warming

```python
# Warm cache on startup
async def startup_cache_warm():
    await warm_cache_from_history()
    await warm_cache_common_problems()
```

### 4. Implement Cache Invalidation Strategy

```python
# Invalidate when solutions change
async def on_solution_updated(problem, old_solution, new_solution):
    cache = create_monitored_cache()
    await cache.invalidate(problem)
    await cache.set(problem, new_solution)
```

### 5. Set Reasonable Cache Size

```python
# Balance memory and hit rate
CacheConfig(max_size=1000)  # Good starting point
# Adjust based on monitoring
```

### 6. Use Structured Logging

```python
# Cache operations are automatically logged
# Logs include problem hash, solution ID, timestamps
```

### 7. Test Cache Behavior

```python
# Verify cache in unit tests
async def test_cache_behavior():
    cache = create_monitored_cache()

    problem = {'type': 'test', 'value': 42}
    solution = {'result': 'success'}

    # First access should miss
    retrieved, hit = await cache.get(problem)
    assert hit is False

    # Set solution
    await cache.set(problem, solution)

    # Second access should hit
    retrieved, hit = await cache.get(problem)
    assert hit is True
    assert retrieved == solution
```

### 8. Handle Cache Failures Gracefully

```python
async def solve_with_cache(problem):
    try:
        cache = create_monitored_cache()
        solution, hit = await cache.get(problem)

        if hit:
            return solution

        # Fallback to solving
        solution = await solve_problem(problem)
        await cache.set(problem, solution)
        return solution

    except Exception as e:
        # Cache failure - solve without cache
        logger.error(f"Cache failure: {e}")
        return await solve_problem(problem)
```

---

## Performance Tuning

### Optimizing Hit Rate

1. **Analyze Access Patterns**: Identify frequently requested problems
2. **Pre-load Common Problems**: Warm cache with common solutions
3. **Increase TTL**: Keep solutions longer
4. **Increase Size**: Reduce evictions

### Optimizing Memory Usage

1. **Reduce Size**: Limit cache size
2. **Reduce TTL**: Solutions expire faster
3. **Monitor Evictions**: High eviction rate indicates size issues

### Optimizing Response Time

1. **Use In-Memory Cache**: Faster than Redis
2. **Optimize Hashing**: Ensure efficient hashing
3. **Reduce Lock Contention**: Use async operations

---

## Cache Lifecycle

### Cache Creation

```python
from bubblelabs_nodes import create_monitored_cache
from bubblelabs_nodes.gauntlet_config import CacheConfig

config = CacheConfig(
    enabled=True,
    cache_type=CacheType.MEMORY,
    ttl_seconds=3600,
    max_size=1000
)

cache = create_monitored_cache(config)
```

### Cache Usage

```python
# Get from cache
solution, hit = await cache.get(problem)

# Set to cache
await cache.set(problem, solution)

# Check if cached
is_cached = await cache.has(problem)

# Invalidate
await cache.invalidate(problem)

# Clear all
await cache.clear()
```

### Cache Monitoring

```python
# Get statistics
stats = await cache.get_statistics()

# Get health status
health = await cache.monitor.get_health_status()
```

---

## Summary

The solution cache provides:
- ✅ 100x speedup on cache hits
- ✅ Configurable TTL and size limits
- ✅ LRU eviction policy
- ✅ Comprehensive monitoring and metrics
- ✅ Structured logging for debugging
- ✅ Health status tracking
- ✅ Multiple warming strategies

For more information:
- `bubblelabs_nodes/solution_cache.py` - Implementation
- `bubblelabs_nodes/cache_monitoring.py` - Monitoring
- `CACHE_ARCHITECTURE.md` - Architecture documentation
- `test_solution_cache.py` - Test suite
