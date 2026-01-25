# Query Optimizer Quick Reference

## Fast Start

```python
from query_optimizer import get_query_optimizer

# Get optimizer instance
optimizer = get_query_optimizer()

# Execute query (auto-optimized and cached)
cursor = optimizer.execute("SELECT * FROM users WHERE id = ?", (1,))
rows = cursor.fetchall()
```

---

## Key Features

### 1. Query Rewriting
Automatically optimizes queries:
- `SELECT *` → Explicit column names
- Detects missing indexes
- Suggests JOIN optimizations
- Identifies `IN` vs `EXISTS` opportunities

```python
original = "SELECT * FROM users"
rewritten = optimizer.rewrite_query(original)
# Result: "SELECT id, username, email, created_at FROM users"
```

### 2. N+1 Query Detection
Three detection strategies:
- Pattern matching
- Temporal clustering
- Foreign key analysis

```python
queries = [...]  # Your query log
issues = optimizer.detect_n_plus_one(queries)

for issue in issues:
    print(f"Severity: {issue.severity}")
    print(f"Fix: {issue.suggested_fix}")
```

### 3. Smart Caching
- Configurable TTL (default: 60s)
- LRU eviction policy
- Memory limits (default: 100MB)
- Hit rate tracking

```python
# Custom cache settings
optimizer = QueryOptimizer(
    db_path="my.db",
    cache_ttl=300,        # 5 minutes
    cache_max_size=1000,  # Max entries
    cache_max_memory_mb=100  # Max memory
)
```

### 4. Query Analysis
```python
plan = optimizer.analyze_query("SELECT * FROM posts WHERE user_id = ?")
print(f"Cost: {plan.estimated_cost}")
print(f"Indexes: {plan.indexes_used}")
print(f"Optimizations: {plan.optimizations}")
```

### 5. Statistics
```python
stats = optimizer.get_statistics()
print(f"Hit Rate: {stats['cache_stats']['hit_rate']}")
print(f"Total Queries: {stats['total_queries']}")
print(f"Avg Time: {stats['avg_query_time']}s")
```

---

## Configuration

### Constructor Parameters
```python
QueryOptimizer(
    db_path: str,                      # Database path
    enable_cache: bool = True,         # Enable caching
    slow_query_threshold: float = 1.0, # Slow query threshold (seconds)
    cache_ttl: int = 60,              # Cache TTL (seconds)
    cache_max_size: int = 1000,        # Max cache entries
    cache_max_memory_mb: int = 100     # Max cache memory (MB)
)
```

### Connection Pool Settings
```python
ConnectionPool(
    db_path: str,
    pool_size: int = 5,        # Base pool size
    max_overflow: int = 10,    # Additional connections
    timeout: float = 30.0      # Connection timeout (seconds)
)
```

---

## API Methods

### Execution
| Method | Description |
|--------|-------------|
| `execute(query, params, auto_optimize)` | Execute query with optimization |
| `rewrite_query(query)` | Get optimized query string |

### Analysis
| Method | Description |
|--------|-------------|
| `analyze_query(query)` | Get query execution plan |
| `recommend_indexes(query)` | Get index recommendations |
| `detect_n_plus_one(queries)` | Detect N+1 patterns |

### Cache
| Method | Description |
|--------|-------------|
| `clear_cache(invalidate_schema)` | Clear query cache |
| `invalidate_schema_cache()` | Force schema reload |

### Maintenance
| Method | Description |
|--------|-------------|
| `get_statistics()` | Get comprehensive stats |
| `export_statistics(file)` | Export to JSON |
| `optimize_database()` | Run ANALYZE, VACUUM |

---

## N+1 Severity Levels

| Occurrences | Severity | Action |
|-------------|----------|--------|
| 100+ | Critical | Fix immediately |
| 50-99 | High | Fix urgently |
| 20-49 | Medium | Fix soon |
| 5-19 | Low | Monitor |

---

## Best Practices

### 1. Use Prepared Statements
```python
# ✅ Good
optimizer.execute("SELECT * FROM users WHERE id = ?", (user_id,))

# ❌ Bad
optimizer.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

### 2. Monitor Cache Performance
```python
stats = optimizer.get_statistics()
hit_rate = float(stats['cache_stats']['hit_rate'].rstrip('%'))

if hit_rate < 50:
    # Increase TTL or cache size
    optimizer.cache_ttl = 300
```

### 3. Regular Optimization
```python
# Run weekly
import schedule

schedule.every().week.do(optimizer.optimize_database)
```

### 4. N+1 Detection in Tests
```python
def test_no_n_plus_one():
    queries = capture_queries(test_function)
    issues = optimizer.detect_n_plus_one(queries)

    critical = [i for i in issues if i.severity == 'critical']
    assert len(critical) == 0
```

---

## Examples

### Example 1: Cache Performance
```python
import time

# First execution (cache miss)
start = time.time()
optimizer.execute("SELECT * FROM users WHERE id = 1")
time1 = time.time() - start

# Second execution (cache hit)
start = time.time()
optimizer.execute("SELECT * FROM users WHERE id = 1")
time2 = time.time() - start

print(f"Speedup: {time1/time2:.1f}x")
```

### Example 2: Index Recommendations
```python
query = "SELECT * FROM posts WHERE user_id = ? ORDER BY created_at"
recs = optimizer.recommend_indexes(query)

for rec in recs:
    print(f"Add index on: {rec['column']}")
    print(f"Reason: {rec['reason']}")
```

### Example 3: Export Statistics
```python
# Export for analysis
optimizer.export_statistics("stats.json")

# Load and analyze
import json
with open("stats.json") as f:
    stats = json.load(f)

print(f"Total queries: {stats['total_queries']}")
print(f"Cache hit rate: {stats['cache_stats']['hit_rate']}")
```

---

## Troubleshooting

### Low Cache Hit Rate
**Problem:** Hit rate < 50%
**Solutions:**
- Increase `cache_ttl`
- Increase `cache_max_size`
- Check query parameterization

### Memory Usage
**Problem:** Cache exceeds memory limit
**Solutions:**
- Reduce `cache_max_memory_mb`
- Reduce `cache_max_size`
- Clear cache periodically

### Slow Queries
**Problem:** Queries exceed threshold
**Solutions:**
- Check `slow_queries` list
- Use `analyze_query()` to check plan
- Apply `recommend_indexes()`
- Use `rewrite_query()` for optimization

### Connection Pool Exhaustion
**Problem:** Timeout waiting for connection
**Solutions:**
- Increase `pool_size`
- Increase `max_overflow`
- Check for connection leaks
- Increase `timeout`

---

## Performance Tuning

### High-Throughput Systems
```python
optimizer = QueryOptimizer(
    db_path="prod.db",
    cache_ttl=600,          # 10 minutes
    cache_max_size=10000,   # Large cache
    cache_max_memory_mb=500,
    slow_query_threshold=0.5  # Stricter threshold
)
```

### Low-Memory Systems
```python
optimizer = QueryOptimizer(
    db_path="prod.db",
    cache_ttl=30,           # Shorter TTL
    cache_max_size=100,     # Smaller cache
    cache_max_memory_mb=10,
    pool_size=2,            # Fewer connections
    max_overflow=5
)
```

### Development/Debugging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

optimizer = QueryOptimizer(
    db_path="dev.db",
    enable_cache=False,     # Disable cache
    slow_query_threshold=0.1  # Catch all slow queries
)
```

---

## Common Patterns

### Batch Inserts
```python
# Use transaction
conn = optimizer.pool.get_connection()
try:
    conn.execute("BEGIN TRANSACTION")
    for item in items:
        conn.execute("INSERT INTO table (col) VALUES (?)", (item,))
    conn.commit()
finally:
    optimizer.pool.return_connection(conn)
```

### Pagination
```python
def get_page(page_number, page_size=20):
    offset = (page_number - 1) * page_size
    query = "SELECT * FROM posts LIMIT ? OFFSET ?"
    return optimizer.execute(query, (page_size, offset))
```

### Count Queries
```python
# COUNT queries are automatically cached
count = optimizer.execute("SELECT COUNT(*) FROM users").fetchone()[0]
```

---

## File Location
```
C:\Users\mmeadow\Documents\OpenEvolve\Frontend\query_optimizer.py
```

## Documentation
- Full Report: `QUERY_OPTIMIZER_IMPLEMENTATION_REPORT.md`
- Examples: Run `python query_optimizer.py`

## Version
- Implemented: 2026-01-22
- Status: Production Ready
