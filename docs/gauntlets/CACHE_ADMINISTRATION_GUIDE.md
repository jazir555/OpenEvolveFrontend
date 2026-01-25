# Cache Administration Guide

This guide is for system administrators responsible for monitoring, maintaining, and troubleshooting the solution caching system.

## Table of Contents

1. [Monitoring Cache Performance](#monitoring-cache-performance)
2. [Manual Cache Management](#manual-cache-management)
3. [Cache Warming Strategies](#cache-warming-strategies)
4. [Troubleshooting Common Issues](#troubleshooting-common-issues)
5. [Operational Procedures](#operational-procedures)

---

## Monitoring Cache Performance

### Key Metrics to Track

#### 1. Cache Hit Rate

**What it measures:** Percentage of requests served from cache

**Formula:** `hit_rate = hits / (hits + misses)`

**Target:** >30% for effective caching

**How to monitor:**
```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()
stats = cache.get_statistics()

hit_rate = stats['hit_rate']
if hit_rate < 0.3:
    print("WARNING: Low cache hit rate!")
```

**Alerting threshold:**
- 🟢 Healthy: >30%
- 🟡 Warning: 10-30%
- 🔴 Critical: <10%

#### 2. Cache Size

**What it measures:** Current number of cached entries

**Maximum:** `CACHE_MAX_SIZE` configuration value

**How to monitor:**
```python
stats = cache.get_statistics()
size = stats['size']
max_size = stats['max_size']
usage_percent = (size / max_size) * 100

print(f"Cache usage: {size}/{max_size} ({usage_percent:.1f}%)")
```

**Alerting threshold:**
- 🟢 Healthy: <80%
- 🟡 Warning: 80-90%
- 🔴 Critical: >90%

#### 3. Eviction Count

**What it measures:** Number of entries evicted due to size limits

**What it indicates:** Cache is too small or TTL is too short

**How to monitor:**
```python
stats = cache.get_statistics()
evictions = stats['evictions']

if evictions > 100:
    print("High eviction count - consider increasing cache size")
```

### Monitoring Dashboards

#### Prometheus Integration

```python
from prometheus_client import Gauge, Counter

# Define metrics
cache_hits = Counter('cache_hits_total', 'Total cache hits')
cache_misses = Counter('cache_misses_total', 'Total cache misses')
cache_size = Gauge('cache_size', 'Current cache size')
cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate')

# Update metrics
stats = cache.get_statistics()
cache_hits.set(stats['hits'])
cache_misses.set(stats['misses'])
cache_size.set(stats['size'])
cache_hit_rate.set(stats['hit_rate'])
```

#### Grafana Dashboard Queries

```promql
# Cache hit rate over time
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Cache size trend
cache_size

# Eviction rate
rate(cache_evictions_total[5m])
```

### Log Monitoring

#### Key Log Patterns

```bash
# Cache hits
grep "Cache HIT" /var/log/gauntlet.log | wc -l

# Cache misses
grep "Cache MISS" /var/log/gauntlet.log | wc -l

# Cache errors
grep "Failed to cache" /var/log/gauntlet.log
```

#### Real-time Monitoring

```bash
# Watch cache activity in real-time
tail -f /var/log/gauntlet.log | grep --line-buffered "Cache"
```

---

## Manual Cache Management

### Viewing Cache Contents

#### In-Memory Cache

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

# Get statistics
stats = cache.get_statistics()
print(f"Entries: {stats['size']}")
print(f"Max size: {stats['max_size']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

#### Redis Cache

```bash
# Connect to Redis
redis-cli -u redis://localhost:6379

# View all cache keys
KEYS gauntlet:solution:*

# Count cache entries
DBSIZE

# View specific entry
GET gauntlet:solution:<hash>

# Check TTL for entry
TTL gauntlet:solution:<hash>
```

### Clearing Cache

#### Clear Specific Problem

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

problem = {
    'statement': 'What is 2 + 2?',
    'type': 'math'
}

# Remove from cache
await cache.invalidate(problem)
print("Problem removed from cache")
```

#### Clear All Cache

```python
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

# Clear entire cache
await cache.clear()
print("Cache cleared")
```

#### Redis - Clear All

```bash
# Clear all gauntlet cache entries
redis-cli --scan --pattern "gauntlet:solution:*" | xargs redis-cli DEL

# Or flush entire database (DANGER!)
redis-cli FLUSHDB
```

### Exporting/Importing Cache

#### Export to JSON

```python
import json
from bubblelabs_nodes import create_solution_cache

cache = create_solution_cache()

# Note: This requires extending the cache implementation
# to support iteration over cached entries

# For Redis:
import redis
r = redis.from_url('redis://localhost:6379')
keys = r.keys('gauntlet:solution:*')

cache_data = {}
for key in keys:
    value = r.get(key)
    cache_data[key.decode()] = json.loads(value)

with open('cache_export.json', 'w') as f:
    json.dump(cache_data, f, indent=2)
```

---

## Cache Warming Strategies

Cache warming is the process of pre-populating the cache with commonly accessed problems.

### Strategy 1: Replay Historical Problems

```python
import json
from bubblelabs_nodes import solveProblem

# Load historical problems
with open('historical_problems.json', 'r') as f:
    problems = json.load(f)

# Warm cache with common problems
for problem in problems[:100]:  # Top 100 most common
    await solveProblem(problem)

print(f"Warmed cache with {len(problems)} problems")
```

### Strategy 2: Scheduled Warming

```python
import asyncio
from datetime import datetime

async def warm_cache_job():
    """Run cache warming every hour"""
    while True:
        print(f"[{datetime.now()}] Starting cache warm...")

        # Load and solve common problems
        common_problems = load_common_problems()
        for problem in common_problems:
            await solveProblem(problem)

        print(f"[{datetime.now()}] Cache warm complete")

        # Wait 1 hour
        await asyncio.sleep(3600)

# Start warming job
asyncio.create_task(warm_cache_job())
```

### Strategy 3: Predictive Warming

```python
from collections import Counter

async def predictive_warming(access_log):
    """Predict and warm likely future problems"""

    # Analyze access patterns
    problem_patterns = Counter()
    with open(access_log, 'r') as f:
        for line in f:
            if 'Cache MISS' in line:
                # Extract problem type
                problem_type = extract_problem_type(line)
                problem_patterns[problem_type] += 1

    # Warm cache with predicted problems
    for problem_type, _ in problem_patterns.most_common(50):
        predicted_problem = generate_example_problem(problem_type)
        await solveProblem(predicted_problem)

    print("Predictive warming complete")
```

---

## Troubleshooting Common Issues

### Issue: Low Cache Hit Rate

**Symptoms:**
- Hit rate <10%
- High cache miss count
- No performance improvement from caching

**Diagnosis:**
```python
stats = cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Misses: {stats['misses']}")
```

**Possible Causes:**
1. Problems are unique (never repeated)
2. Problem structure varies
3. Cache TTL too short
4. Cache size too small

**Solutions:**
1. Analyze problem diversity
2. Increase `CACHE_TTL_SECONDS`
3. Increase `CACHE_MAX_SIZE`
4. Review problem normalization

### Issue: High Memory Usage

**Symptoms:**
- Process memory growing
- OOM errors
- System slowdown

**Diagnosis:**
```python
import psutil
process = psutil.Process()
mem_info = process.memory_info()

print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
```

**Possible Causes:**
1. Cache size too large
2. Memory leak in cache implementation
3. Large solution objects

**Solutions:**
1. Reduce `CACHE_MAX_SIZE`
2. Reduce `CACHE_TTL_SECONDS`
3. Monitor and restart process periodically
4. Use Redis backend for shared memory

### Issue: Redis Connection Failures

**Symptoms:**
- Cache errors in logs
- All cache misses
- Connection timeout errors

**Diagnosis:**
```bash
# Test Redis connection
redis-cli -u redis://localhost:6379 ping

# Should return: PONG
```

**Possible Causes:**
1. Redis not running
2. Wrong Redis URL
3. Network/firewall issues
4. Redis overloaded

**Solutions:**
1. Start Redis: `sudo systemctl start redis`
2. Verify `CACHE_REDIS_URL` configuration
3. Check network connectivity
4. Monitor Redis performance: `redis-cli INFO`

### Issue: Stale Cache Data

**Symptoms:**
- Returning outdated solutions
- Incorrect results
- Logic bugs

**Diagnosis:**
```python
# Check cache age
import time
problem = {'statement': 'test problem'}
key = cache.hasher.generate_hash(problem)

# For Redis, check TTL
ttl = r.ttl(f"gauntlet:solution:{key}")
print(f"Time until expiration: {ttl}s")
```

**Possible Causes:**
1. TTL too long
2. Problem solutions changed
3. Bug in cache invalidation

**Solutions:**
1. Reduce `CACHE_TTL_SECONDS`
2. Implement cache versioning
3. Clear cache manually
4. Add cache invalidation on updates

---

## Operational Procedures

### Daily Checks

```bash
#!/bin/bash
# daily_cache_check.sh

echo "=== Daily Cache Health Check ==="

# Check cache hit rate
hit_rate=$(python -c "from bubblelabs_nodes import create_solution_cache; c=create_solution_cache(); print(f\"{c.get_statistics()['hit_rate']:.2f}\")")
echo "Cache hit rate: ${hit_rate}"

# Check cache size
size=$(python -c "from bubblelabs_nodes import create_solution_cache; c=create_solution_cache(); print(c.get_statistics()['size'])")
echo "Cache size: ${size}"

# Check Redis connection
if redis-cli -u redis://localhost:6379 ping > /dev/null 2>&1; then
    echo "Redis: OK"
else
    echo "Redis: FAILED"
fi

# Check for errors
error_count=$(grep -c "Failed to cache" /var/log/gauntlet.log)
echo "Cache errors (24h): ${error_count}"

echo "=== Check Complete ==="
```

### Cache Rotation

```bash
#!/bin/bash
# rotate_cache.sh

# Clear old cache entries
redis-cli --scan --pattern "gauntlet:solution:*" | \
  while read key; do
    ttl=$(redis-cli TTL "$key")
    if [ "$ttl" -eq -1 ]; then
      # No expiry set - set one
      redis-cli EXPIRE "$key" 3600
    fi
  done

echo "Cache rotation complete"
```

### Emergency Cache Flush

```bash
#!/bin/bash
# emergency_cache_flush.sh

echo "WARNING: This will flush the entire cache!"
echo "Press Ctrl+C to cancel..."
sleep 5

# Flush cache
redis-cli --scan --pattern "gauntlet:solution:*" | \
  xargs redis-cli DEL

echo "Cache flushed"

# Notify team
send_alert "Cache has been emergency flushed"
```

### Backup Cache Data

```bash
#!/bin/bash
# backup_cache.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="/backups/cache_backup_${DATE}.json"

# Export cache
python3 - <<EOF
import redis
import json

r = redis.from_url('redis://localhost:6379')
keys = r.keys('gauntlet:solution:*')

cache_data = {}
for key in keys:
    ttl = r.ttl(key)
    value = r.get(key)
    cache_data[key.decode()] = {
        'value': json.loads(value),
        'ttl': ttl
    }

with open('${BACKUP_FILE}', 'w') as f:
    json.dump(cache_data, f, indent=2)

print(f"Backup complete: {len(cache_data)} entries")
EOF

echo "Cache backed up to: ${BACKUP_FILE}"
```

---

## Maintenance Schedule

### Daily
- Monitor cache hit rate
- Check error logs
- Verify Redis connectivity

### Weekly
- Review cache size trends
- Analyze eviction patterns
- Performance benchmarks

### Monthly
- Review and tune TTL settings
- Analyze problem patterns
- Cache warming optimization
- Backup cache data

### Quarterly
- Capacity planning
- Architecture review
- Performance optimization
- Documentation updates

---

## Contact & Support

For cache-related issues:
- **Documentation**: [Cache Documentation](./CACHE_DOCUMENTATION.md)
- **Issues**: GitHub Issues
- **Emergency**: on-call rotation
