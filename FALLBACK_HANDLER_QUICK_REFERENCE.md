# Fallback Handler - Quick Reference Guide

## Overview
Production-ready fallback system with circuit breaking, intelligent caching, timeout enforcement, and graceful degradation strategies for OpenEvolve service unavailability.

**File:** `fallback_handler.py`
**Lines:** 1,025
**Status:** Production-Ready

---

## Quick Start

### Basic Usage
```python
from fallback_handler import FallbackHandler

# Initialize
handler = FallbackHandler()

# Get fallback result
result = handler.get_fallback_result(
    operation_type="evolution",
    input_data={
        'content': 'def hello(): print("world")',
        'evolution_mode': 'standard'
    }
)

# Check result
if result.success:
    print(f"Success: {result.result}")
    print(f"Cached: {result.cached}")
    print(f"Response time: {result.response_time_ms}ms")
else:
    print(f"Error: {result.error}")
```

---

## Configuration Options

### Constructor Parameters
```python
handler = FallbackHandler(
    cache_max_size=100,           # Maximum cache entries
    cache_ttl_seconds=300,        # Cache entry lifetime (5 minutes)
    circuit_config=None,          # CircuitBreakerConfig (uses defaults)
    default_timeout_ms=5000       # Default operation timeout (5 seconds)
)
```

### Circuit Breaker Configuration
```python
from fallback_handler import CircuitBreakerConfig, FallbackHandler

config = CircuitBreakerConfig(
    failure_threshold=5,      # Failures before opening circuit
    success_threshold=2,      # Successes to close circuit
    timeout_ms=60000,         # Time to stay open (1 minute)
    half_open_max_calls=3     # Max calls in half-open state
)

handler = FallbackHandler(circuit_config=config)
```

---

## Operation Types

| Operation Type | Description | Fallback Strategy |
|----------------|-------------|-------------------|
| `evolution` | Code evolution operations | Basic code improvements |
| `blue_team_solution` | Blue Team fixes | Pass-through with metadata |
| `red_team_critique` | Red Team critique | Empty findings list |
| `evaluator_assessment` | Quality evaluation | Neutral score (50.0) |
| `content_analysis` | Content analysis | Basic metrics + keywords |
| `decomposition` | Problem decomposition | Single sub-problem |
| `leanaide_evolution` | Lean proof evolution | Error with suggestions |
| `mdap_optimization` | MDAP optimization | Empty policy |
| `mcts_search` | MCTS search | Empty search tree |

---

## Circuit Breaker States

### CLOSED (Normal)
- All requests flow through
- Failures increment counter
- Successes reset counter

### OPEN (Failing)
- All requests rejected immediately
- Raises `CircuitOpenError`
- Waits for timeout period

### HALF_OPEN (Testing)
- Limited requests allowed
- Tests if service recovered
- Successes → CLOSED
- Failures → OPEN

### Manual Control
```python
# Check circuit state
state = handler.get_circuit_breaker_state("evolution")
print(f"Circuit state: {state.value}")  # closed | open | half_open

# Reset circuit manually
handler.reset_circuit_breaker("evolution")  # Specific operation
handler.reset_circuit_breaker()              # All circuits
```

---

## Exception Handling

### Exception Types
```python
from fallback_handler import TimeoutError, CircuitOpenError

try:
    result = handler.get_fallback_result(
        operation_type="evolution",
        input_data={'content': 'code'},
        timeout_ms=3000
    )
except TimeoutError:
    print("Operation timed out")
except CircuitOpenError:
    print("Circuit breaker is open, service degraded")
```

### Result Error Checking
```python
result = handler.get_fallback_result(...)

if not result.success:
    # Check error type from result
    if "Dependency error" in result.error:
        # Import/attribute error
        print("Missing dependencies")
    elif "Data error" in result.error:
        # Validation error
        print("Invalid input data")
    else:
        # Unexpected error
        print(f"Unexpected: {result.error}")
```

---

## Cache Management

### Cache Statistics
```python
stats = handler.cache.get_statistics()
print(f"Hit rate: {stats['hit_rate']:.1f}%")
print(f"Size: {stats['size']}/{stats['max_size']}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
print(f"Evictions: {stats['evictions']}")
```

### Cache Operations
```python
# Clear cache
handler.clear_cache()

# Check if result was cached
result = handler.get_fallback_result(...)
if result.cached:
    print("Served from cache")
```

### TTL Behavior
- Complex operations (evolution, decomposition): 60 seconds
- Medium operations (analysis, evaluation): 120 seconds
- Simple operations: 300 seconds (default)

---

## Monitoring & Health

### Health Status
```python
health = handler.get_health_status()
print(f"Status: {health['status']}")  # healthy | degraded | unhealthy
print(f"Score: {health['health_score']}/100")
print(f"Open circuits: {health['open_circuits']}")
print(f"Cache hit rate: {health['cache_hit_rate']:.1f}%")
```

### Full Statistics
```python
stats = handler.get_fallback_stats()
print(json.dumps(stats, indent=2))

# Output:
{
    "total_fallbacks": 42,
    "fallbacks_by_type": {
        "evolution": 15,
        "content_analysis": 12,
        "decomposition": 10,
        "blue_team_solution": 5
    },
    "cache": {
        "size": 45,
        "max_size": 100,
        "hits": 28,
        "misses": 14,
        "evictions": 2,
        "hit_rate": 66.7
    },
    "circuit_breakers": {
        "evolution": {
            "state": "closed",
            "failure_count": 0,
            "success_count": 3
        },
        "decomposition": {
            "state": "open",
            "failure_count": 5,
            "success_count": 0
        }
    },
    "timestamp": "2026-01-22T10:30:15.123456"
}
```

---

## Degradation Levels

### LOW
- Minimal impact
- Most functionality preserved
- Full fallback response

### MEDIUM
- Reduced functionality
- Manual review suggested
- Basic transformations applied

### HIGH
- Critical features unavailable
- Workarounds provided
- Limited processing

### CRITICAL
- Import/dependency errors
- Fallback to dict structure
- Error messages included

---

## Timeout Handling

### Default Timeout
```python
handler = FallbackHandler(default_timeout_ms=5000)  # 5 seconds
```

### Per-Request Override
```python
result = handler.get_fallback_result(
    operation_type="evolution",
    input_data={'content': 'code'},
    timeout_ms=2000  # Override for this request only
)
```

### Timeout Behavior
- Operation cancelled after timeout
- `TimeoutError` raised
- Circuit breaker records failure
- Resources cleaned up properly

---

## Thread Safety

All operations are thread-safe:
- Cache operations use `RLock`
- Circuit breakers use `RLock`
- Safe for concurrent access
- No race conditions

---

## Logging Configuration

### Default Setup
```python
# Uses Python logging module
# Format: %(asctime)s - %(name)s - %(levelname)s - %(message)s
# Level: INFO

import logging
logging.basicConfig(level=logging.DEBUG)
```

### Custom Configuration
```python
handler = FallbackHandler()

# Access logger
logger = handler.logger

# Configure handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.DEBUG)
```

### Log Levels
- **DEBUG:** Cache operations, state changes
- **INFO:** Fallback invocation, successes
- **WARNING:** Unknown operations, circuit half-open
- **ERROR:** Circuit opens, failures, import errors
- **EXCEPTION:** Unexpected errors with stack traces

---

## Common Patterns

### Pattern 1: Retry with Circuit Breaker
```python
from fallback_handler import CircuitOpenError
import time

handler = FallbackHandler()
max_retries = 3

for attempt in range(max_retries):
    try:
        result = handler.get_fallback_result(...)
        break  # Success
    except CircuitOpenError:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise  # Final attempt failed
```

### Pattern 2: Monitor Health
```python
handler = FallbackHandler()

# Periodic health check
def monitor_fallback_health():
    health = handler.get_health_status()
    if health['status'] == 'unhealthy':
        print("ALERT: Fallback system unhealthy!")
        # Send alert, take action
    elif health['status'] == 'degraded':
        print("WARNING: Fallback system degraded")
        # Log warning, monitor closely

# Run every 60 seconds
import schedule
schedule.every(60).seconds.do(monitor_fallback_health)
```

### Pattern 3: Graceful Degradation
```python
def process_with_fallback(operation_type, data):
    handler = FallbackHandler()

    try:
        # Try primary service first
        result = primary_service.process(operation_type, data)
        return result, 'primary'
    except PrimaryServiceError:
        # Fall back to fallback handler
        result = handler.get_fallback_result(operation_type, data)

        if result.success:
            degradation = result.result.get('degradation_level', 'unknown')
            print(f"Using fallback (degradation: {degradation})")
            return result.result, 'fallback'
        else:
            # Both failed
            raise Exception("Primary and fallback both failed")
```

---

## Performance Tips

### 1. Tune Cache Size
```python
# High-traffic scenarios
handler = FallbackHandler(cache_max_size=1000)

# Memory-constrained
handler = FallbackHandler(cache_max_size=50)
```

### 2. Adjust TTL for Your Use Case
```python
# Fast-changing data
handler = FallbackHandler(cache_ttl_seconds=60)

# Stable data
handler = FallbackHandler(cache_ttl_seconds=600)
```

### 3. Circuit Breaker Tuning
```python
# Aggressive (fail fast)
config = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_ms=30000
)

# Lenient (more retries)
config = CircuitBreakerConfig(
    failure_threshold=10,
    timeout_ms=120000
)
```

### 4. Timeout Optimization
```python
# Fast operations
handler = FallbackHandler(default_timeout_ms=1000)

# Slow operations
handler = FallbackHandler(default_timeout_ms=10000)
```

---

## Troubleshooting

### Issue: Low Cache Hit Rate
**Solution:**
- Increase `cache_ttl_seconds`
- Check if input data varies too much
- Verify cache key generation logic

### Issue: Circuit Breaker Opening Frequently
**Solution:**
- Increase `failure_threshold`
- Check upstream service health
- Review timeout configuration

### Issue: High Memory Usage
**Solution:**
- Reduce `cache_max_size`
- Reduce `cache_ttl_seconds`
- Monitor eviction rate

### Issue: Slow Response Times
**Solution:**
- Reduce `default_timeout_ms`
- Profile fallback implementations
- Check for lock contention

---

## Best Practices

1. **Always check `result.success`** before using results
2. **Monitor circuit breaker states** proactively
3. **Tune timeouts** based on actual operation duration
4. **Review cache hit rates** regularly
5. **Handle specific exceptions** appropriately
6. **Log all fallback invocations** for analysis
7. **Set up alerts** for unhealthy status
8. **Test fallback paths** in staging
9. **Document degradation levels** for your operations
10. **Monitor health status** in dashboards

---

## API Reference

### Classes
- `FallbackHandler` - Main handler class
- `FallbackResult` - Result dataclass
- `FallbackCache` - Cache with statistics
- `CircuitBreaker` - Circuit breaker implementation
- `CircuitBreakerConfig` - Configuration dataclass
- `CircuitState` - State enum (CLOSED, OPEN, HALF_OPEN)
- `CacheStatistics` - Statistics tracking

### Key Methods
- `get_fallback_result()` - Get fallback with circuit breaking
- `get_fallback_stats()` - Get comprehensive statistics
- `get_health_status()` - Get health summary
- `get_circuit_breaker_state()` - Check circuit state
- `reset_circuit_breaker()` - Manual circuit reset
- `clear_cache()` - Clear all cache entries

---

## Support & Documentation

- Full Implementation: `fallback_handler.py`
- Enhancement Report: `FALLBACK_HANDLER_ENHANCEMENT_REPORT.md`
- Usage Examples: Bottom of `fallback_handler.py`

For issues or questions, refer to the comprehensive documentation in the enhancement report.
