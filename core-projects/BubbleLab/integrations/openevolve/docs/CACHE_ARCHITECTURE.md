# Cache Architecture Document

## Overview

The Gauntlet system employs a multi-layered caching strategy to optimize problem-solving performance by avoiding redundant computations.

## Design Decisions

### 1. Caching Strategy: LRU with TTL

**Choice:** LRU (Least Recently Used) cache with TTL (Time To Live)

**Rationale:**
- **LRU** ensures frequently-used problems stay in cache
- **TTL** prevents stale solutions from being used indefinitely
- **Size limits** prevent unbounded memory growth
- **Combination** provides optimal balance of hit rate and memory usage

**Configuration:**
```python
cache = AtomicSolutionCache(
    cache_type="memory",  # LRU implementation
    max_size=1000,        # Maximum entries
    ttl_seconds=3600      # 1 hour TTL
)
```

### 2. Cache Key Format

**Format:** SHA256 hash of normalized problem

**Implementation:**
```python
key = hashlib.sha256(
    json.dumps(normalized_problem, sort_keys=True).encode()
).hexdigest()
```

**Properties:**
- **Deterministic:** Same problem always produces same key
- **Collision-resistant:** SHA256 minimizes collision risk
- **Size-predictable:** Always 64 hex characters

### 3. Cache Storage Backends

#### In-Memory Cache (Default)

**Pros:**
- Fastest access (microsecond latency)
- No external dependencies
- Simple deployment

**Cons:**
- Limited to single process
- Lost on restart
- Memory bounded

**Use case:** Development, testing, single-process deployments

#### Redis Cache (Production)

**Pros:**
- Shared across processes
- Persists across restarts
- Distributed caching
- Advanced features (pub/sub, transactions)

**Cons:**
- Network latency (millisecond)
- External dependency
- Additional infrastructure

**Use case:** Production, multi-process deployments

### 4. Cache Invalidation Strategy

**Time-based (TTL):**
- Primary invalidation mechanism
- Configurable per cache instance
- Default: 1 hour

**Manual:**
- Explicit `invalidate()` method
- Clear all with `clear()`
- Per-key invalidation

**Automatic:**
- Size-based eviction (LRU)
- TTL expiration

**No automatic invalidation on:**
- Problem updates (use TTL)
- Solution changes (use manual invalidation)

### 5. Cache Warming Strategy

**Passive Warming (Current):**
- Cache populates on-demand
- First request is cache miss
- Subsequent requests are cache hits

**Future Active Warming:**
- Pre-load common problems on startup
- Background refresh of expiring entries
- Predictive pre-fetching

## Data Structures

### Cache Entry

```python
{
    'key': 'abc123...',          # SHA256 hash
    'value': {...},              # Solution
    'created_at': timestamp,     # Creation time
    'last_accessed': timestamp,  # Last access time
    'access_count': int,         # Number of accesses
    'ttl_seconds': int           # Time-to-live
}
```

### Cache Statistics

```python
{
    'hits': int,              # Cache hits
    'misses': int,            # Cache misses
    'hit_rate': float,        # hits / (hits + misses)
    'evictions': int,         # Entries evicted
    'size': int,              # Current size
    'max_size': int           # Maximum size
}
```

## Algorithms

### Normalization

```python
def normalize_problem(problem: Dict) -> Dict:
    """
    Normalize problem for consistent hashing.

    Steps:
    1. Remove metadata fields (timestamp, id metadata)
    2. Sort all dictionaries
    3. Convert lists to tuples (hashable)
    4. Remove whitespace
    5. Lowercase keys
    """
    # Remove non-content fields
    normalized = {
        k: v for k, v in problem.items()
        if k not in METADATA_FIELDS
    }

    # Sort dictionaries recursively
    normalized = sort_dict_recursive(normalized)

    return normalized
```

### LRU Eviction

```python
def evict_lru():
    """
    Evict least recently used entry.

    Algorithm:
    1. Find entry with lowest last_accessed timestamp
    2. Remove from cache
    3. Update statistics
    """
    if cache.size >= cache.max_size:
        lru_key = min(
            cache.entries.keys(),
            key=lambda k: cache.entries[k].last_accessed
        )
        del cache.entries[lru_key]
        cache.stats.evictions += 1
```

### TTL Expiration

```python
def is_expired(entry) -> bool:
    """
    Check if entry has expired.

    Returns:
        True if TTL has passed
    """
    now = time.time()
    age = now - entry.created_at
    return age > entry.ttl_seconds
```

## Performance Characteristics

### Time Complexity

| Operation | In-Memory | Redis |
|-----------|-----------|-------|
| get() | O(1) | O(1) |
| set() | O(1) | O(1) |
| has() | O(1) | O(1) |
| invalidate() | O(1) | O(1) |
| clear() | O(n) | O(1) |

### Space Complexity

| Backend | Per Entry | Overhead |
|---------|-----------|----------|
| In-Memory | Size of solution | ~100 bytes |
| Redis | Size of solution | ~50 bytes |

### Latency

| Backend | Get | Set |
|---------|-----|-----|
| In-Memory | ~1μs | ~2μs |
| Redis (local) | ~1ms | ~1ms |
| Redis (remote) | ~10ms | ~10ms |

## Configuration

### Environment Variables

```bash
# Enable/disable caching
CACHE_ENABLED=true

# Cache type
CACHE_TYPE=memory  # or redis

# In-memory cache settings
CACHE_MAX_SIZE=1000
CACHE_TTL_SECONDS=3600

# Redis cache settings
CACHE_REDIS_URL=redis://localhost:6379/0
```

### Code Configuration

```python
from bubblelabs_nodes import create_solution_cache

# In-memory cache
cache = create_solution_cache(
    cache_type="memory",
    max_size=1000,
    ttl_seconds=3600
)

# Redis cache
cache = create_solution_cache(
    cache_type="redis",
    redis_url="redis://localhost:6379",
    max_size=10000,
    ttl_seconds=7200
)
```

## Monitoring

### Key Metrics

- **Hit Rate:** Target > 30%
- **Eviction Rate:** Should be low (< 5%)
- **Size:** Should stay below max_size
- **Latency:** Get/set operations

### Example Monitoring

```python
stats = cache.get_statistics()

print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Size: {stats['size']}/{stats['max_size']}")
print(f"Evictions: {stats['evictions']}")

# Alert if hit rate too low
if stats['hit_rate'] < 0.3:
    logger.warning("Low cache hit rate - consider increasing cache size")
```

## Future Enhancements

1. **Active Warming:** Pre-load common problems
2. **Hierarchical Caching:** L1 (memory) + L2 (Redis)
3. **Compression:** Compress large solutions
4. **Sharding:** Distribute cache across Redis nodes
5. **Replication:** Multi-master Redis setup
6. **Analytics:** Track most-cached problems
