# Solution Caching Documentation

## Overview

The Gauntlet system includes a high-performance solution caching layer that dramatically speeds up repeated problem solving by storing previously computed solutions. When a problem is encountered multiple times, the cached solution is returned instantly instead of recomputing it.

## How Caching Works

### Cache Flow

```
┌─────────────────┐
│  solveProblem() │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Generate Problem Hash  │  (SHA256 of normalized problem)
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐      ┌──────────────┐
│  Check Cache            │──NO──│  Solve       │
│  (by hash)              │      │  Problem     │
└────────┬────────────────┘      └──────┬───────┘
         │                               │
        YES                              │
         │                               ▼
         │                      ┌──────────────┐
         │                      │  Store in    │
         │                      │  Cache       │
         │                      └──────┬───────┘
         │                               │
         ▼                               ▼
┌─────────────────────────────────────────────┐
│          Return Cached Solution             │
└─────────────────────────────────────────────┘
```

### Problem Hashing

The cache uses a sophisticated hashing algorithm to identify duplicate problems:

1. **Normalization**: Remove metadata fields (id, timestamp, cached)
2. **Sorting**: Sort all dictionary keys and list items
3. **Hashing**: Generate SHA256 hash of normalized problem
4. **Cache Key**: Hex digest of hash

This ensures that structurally identical problems generate the same cache key, regardless of cosmetic differences.

## Configuration Options

### Environment Variables

All cache configuration is done via environment variables:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CACHE_ENABLED` | boolean | `true` | Enable/disable caching globally |
| `CACHE_TYPE` | string | `memory` | Cache backend: `memory`, `redis`, `none` |
| `CACHE_TTL_SECONDS` | integer | `3600` | Time-to-live for cached solutions (seconds) |
| `CACHE_MAX_SIZE` | integer | `1000` | Maximum number of cached solutions |
| `CACHE_REDIS_URL` | string | `null` | Redis connection URL (required if CACHE_TYPE=redis) |

### Example Configurations

#### Development (Default)
```bash
CACHE_ENABLED=true
CACHE_TYPE=memory
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000
```

#### Production (Redis Backend)
```bash
CACHE_ENABLED=true
CACHE_TYPE=redis
CACHE_TTL_SECONDS=86400
CACHE_MAX_SIZE=100000
CACHE_REDIS_URL=redis://redis-cluster:6379/0
```

#### Disable Caching
```bash
CACHE_ENABLED=false
# or
CACHE_TYPE=none
```

## Usage Examples

### Basic Usage

```python
from bubblelabs_nodes import solveProblem

# First call - cache miss
problem = {
    'id': 'problem_1',
    'statement': 'What is 2 + 2?',
    'type': 'math'
}

result1 = await solveProblem(problem)
# Result computed and cached

# Second call - cache hit
result2 = await solveProblem(problem)
# Result returned from cache (instant)
```

### Cache Hit/Miss Detection

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

# Check if solution is cached
problem = {'statement': 'test problem'}

if await cache.has(problem):
    print("Solution in cache")
    cached = await cache.get(problem)
else:
    print("Not cached - need to solve")
```

### Manual Cache Invalidation

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

# Invalidate specific problem
problem = {'statement': 'test problem'}
await cache.invalidate(problem)

# Clear all cache
await cache.clear()
```

## Cache Statistics

### Accessing Statistics

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()
stats = cache.get_statistics()

print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
print(f"Evictions: {stats['evictions']}")
```

### Statistics Fields

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Whether caching is enabled |
| `type` | string | Cache backend type |
| `hits` | integer | Number of cache hits |
| `misses` | integer | Number of cache misses |
| `hit_rate` | float | Hit rate as percentage (0.0-1.0) |
| `size` | integer | Current number of cached entries |
| `max_size` | integer | Maximum cache size |
| `evictions` | integer | Number of evicted entries |

## Performance Characteristics

### Benchmark Results

Based on performance benchmarks (`test_cache_benchmarks.py`):

- **Cache Hit Speedup**: 2-100x faster (depending on solve complexity)
- **Memory Usage**: ~324 bytes per cached entry
- **Cache Hit Time**: ~0.03-0.05ms (consistent across cache sizes)
- **Cache Populate Time**: ~20ms for 1000 entries

### When to Use Caching

Caching is most effective when:

- ✅ Problems are solved multiple times
- ✅ Problem solving is expensive (>100ms)
- ✅ Problem structure is consistent
- ✅ Memory is available for cache

Caching is less effective when:

- ❌ Problems are unique (never repeated)
- ❌ Problem solving is very fast (<1ms)
- ❌ Memory is constrained
- ❌ Problems change frequently

## Cache Backends

### In-Memory Cache (Default)

**Pros:**
- Fastest performance
- No external dependencies
- Simple configuration

**Cons:**
- Limited to single process
- Lost on restart
- Limited by process memory

**Use Case:** Development, single-instance deployments

### Redis Cache (Production)

**Pros:**
- Shared across multiple processes
- Persists across restarts
- Scales horizontally
- Advanced features (pub/sub, transactions)

**Cons:**
- Network latency
- External dependency
- More complex setup

**Use Case:** Production, multi-instance deployments

### No Cache

**Use Case:** Testing, debugging, memory-constrained environments

## Monitoring & Observability

### Logging

The cache logs the following events:

```
INFO: Cache HIT for problem: What is 2 + 2?
INFO: Cache MISS for problem: What is 3 + 3?
WARNING: Failed to cache solution: ...
```

### Metrics

The following metrics are available for monitoring:

- `cache_hits`: Total cache hits
- `cache_misses`: Total cache misses
- `cache_hit_rate`: Hit rate percentage
- `cache_size`: Current cache size
- `cache_evictions`: Number of evicted entries

These can be exported to Prometheus or other monitoring systems.

## Troubleshooting

### Cache Not Working

**Problem:** Solutions not being cached

**Solutions:**
1. Check `CACHE_ENABLED` is set to `true`
2. Verify cache configuration is valid
3. Check logs for error messages
4. Validate cache statistics

### High Memory Usage

**Problem:** Cache consuming too much memory

**Solutions:**
1. Reduce `CACHE_MAX_SIZE`
2. Reduce `CACHE_TTL_SECONDS`
3. Monitor cache size and evictions
4. Consider Redis backend for shared memory

### Low Hit Rate

**Problem:** Cache hit rate below 10%

**Solutions:**
1. Analyze problem structure - are they truly identical?
2. Check problem normalization is working
3. Increase `CACHE_TTL_SECONDS`
4. Review cache miss patterns in logs

### Redis Connection Issues

**Problem:** Cannot connect to Redis

**Solutions:**
1. Verify `CACHE_REDIS_URL` is correct
2. Check Redis is running: `redis-cli ping`
3. Test connection: `redis-cli -u redis://localhost:6379`
4. Check network connectivity and firewall rules

## Best Practices

1. **Enable in Production**: Always use caching in production environments
2. **Use Redis for Scale**: Use Redis backend for multi-instance deployments
3. **Monitor Hit Rate**: Track cache hit rate to ensure effectiveness
4. **Tune TTL**: Adjust TTL based on how often problems repeat
5. **Set Reasonable Limits**: Use appropriate max_size to prevent memory issues
6. **Cache Warming**: Pre-populate cache with common problems
7. **Regular Monitoring**: Set up alerts for cache hit rate and memory usage

## API Reference

### create_solution_cache(config=None)

Create a solution cache instance.

**Parameters:**
- `config` (dict, optional): Cache configuration

**Returns:**
- `AtomicSolutionCache`: Cache instance

**Example:**
```python
cache = create_solution_cache(config={
    'max_size': 5000,
    'ttl': 7200
})
```

### AtomicSolutionCache.solve(problem, solve_func)

Solve problem with caching.

**Parameters:**
- `problem` (dict): Problem to solve
- `solve_func` (callable): Async solve function

**Returns:**
- Solution result

### AtomicSolutionCache.get_statistics()

Get cache statistics.

**Returns:**
- `dict`: Statistics dictionary

## Further Reading

- [Cache Administration Guide](./CACHE_ADMINISTRATION_GUIDE.md)
- [Performance Benchmarks](../../test_cache_benchmarks.py)
- [Configuration Reference](./gauntlet_config.py)
