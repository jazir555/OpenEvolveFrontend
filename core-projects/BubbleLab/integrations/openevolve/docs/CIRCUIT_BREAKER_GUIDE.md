# Circuit Breaker Guide for Gauntlet System

This guide explains how to use circuit breakers to provide fault isolation and prevent cascading failures in the Gauntlet system.

## Table of Contents

1. [Overview](#overview)
2. [Circuit Breaker Concepts](#circuit-breaker-concepts)
3. [Level Breakers](#level-breakers)
4. [Hierarchical Management](#hierarchical-management)
5. [Configuration](#configuration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Circuit breakers prevent cascading failures by:

- **Detecting failures** at each decomposition level
- **Opening circuits** to stop operations to failing components
- **Automatic recovery** after timeout
- **Hierarchical isolation** to prevent cross-level contamination

### Quick Start

```python
from bubblelabs_nodes import create_circuit_breaker_manager

manager =_create_circuit_breaker_manager(
    strategy='hierarchical',
    failure_threshold=5,
    recovery_timeout=60
)

async def operation():
    # Your operation logic
    return result

# Execute with circuit breaker protection
success, result, error = await manager.execute_at_level(
    level=0,
    operation=operation,
    context={}
)
```

---

## Circuit Breaker Concepts

### States

1. **CLOSED**: Normal operation, requests pass through
2. **OPEN**: Circuit is open, requests are blocked
3. **HALF_OPEN**: Testing if system has recovered

### State Transitions

```
CLOSED → (failures reach threshold) → OPEN
OPEN → (recovery timeout expires) → HALF_OPEN
HALF_OPEN → (successful call) → CLOSED
HALF_OPEN → (failed call) → OPEN
```

### Strategies

**Individual (INDIVIDUAL)**
- Per-problem circuit breakers
- Isolates at finest granularity

**Hierarchical (HIERARCHICAL)**
- Per-level circuit breakers (recommended)
- Balances isolation and performance

**Global (GLOBAL)**
- Single breaker for all operations
- Maximum isolation but reduced performance

---

## Level Breakers

### Using LevelCircuitBreaker

```python
from bubblelabs_nodes import LevelCircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout_seconds=60,
    half_open_max_calls=3
)

breaker = LevelCircuitBreaker(level=0, config=config)

async def risky_operation():
    return await some_api_call()

success, result, error = await breaker.execute(
    operation=risky_operation,
    context={}
)

if success:
    print("Operation succeeded")
else:
    print(f"Operation failed: {error}")
```

### Checking Breaker State

```python
print(f"Breaker state: {breaker.state}")

if breaker.state == CircuitBreakerState.OPEN:
    print("Circuit is OPEN - operations blocked")
elif breaker.state == CircuitBreakerState.HALF_OPEN:
    print("Circuit is HALF-OPEN - testing recovery")
else:
    print("Circuit is CLOSED - normal operation")
```

---

## Hierarchical Management

### HierarchicalCircuitBreakerManager

```python
from bubblelabs_nodes import HierarchicalCircuitBreakerManager

manager = HierarchicalCircuitBreakerManager()

# Get breaker for specific level
breaker_0 = manager.get_breaker(level=0)
breaker_1 = manager.get_breaker(level=1)
breaker_2 = manager.get_breaker(level=2)

# Each level has independent breaker
```

### Execution by Level

```python
# Execute at level 0 with circuit breaker protection
success, result, error = await manager.execute_at_level(
    level=0,
    operation=operation,
    context={}
)

# If level 0 circuit is open, level 1 can still work
```

### State Monitoring

```python
# Get all circuit states
states = manager.get_all_states()

for level, state in states.items():
    print(f"Level {level}: {state}")
```

---

## Configuration

### Circuit Breaker Config

```python
from bubblelabs_nodes import CircuitBreakerConfig

config = CircuitBreakerConfig(
    enabled=True,
    strategy=CircuitBreakerStrategy.HIERARCHICAL,
    failure_threshold=5,        # Open after 5 failures
    recovery_timeout_seconds=60,  # Wait 60s before recovery
    half_open_max_calls=3       # Try 3 calls in half-open
)
```

### Environment Variables

```bash
# Enable circuit breakers
CIRCUIT_BREAKER_ENABLED=true

# Strategy
CIRCUIT_BREAKER_STRATEGY=hierarchical

# Thresholds
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60
```

---

## Best Practices

### 1. Set Appropriate Thresholds

```python
# Too low: Too sensitive
config = CircuitBreakerConfig(failure_threshold=2)

# Good: Balanced
config = CircuitBreakerConfig(failure_threshold=5)

# Too high: Too tolerant
config = CircuitBreakerConfig(failure_threshold=100)
```

### 2. Tune Recovery Timeout

```python
# Fast recovery (for non-critical systems)
config = CircuitBreakerConfig(recovery_timeout_seconds=10)

# Standard recovery
config = CircuitBreakerConfig(recovery_timeout_seconds=60)

# Slow recovery (for critical systems)
config = CircuitBreakerConfig(recovery_timeout_seconds=300)
```

### 3. Monitor Circuit States

```python
from bubblelabs_nodes import get_metrics_collector

collector = get_metrics_collector()

# Track circuit breaker state
state = breaker.state
collector.set_gauge(
    f"circuit_breaker_level_{breaker.level}_state",
    0 if state == CircuitBreakerState.CLOSED else
    1 if state == CircuitBreakerState.HALF_OPEN else
    2  # OPEN
)
```

### 4. Handle Open Circuits

```python
success, result, error = await breaker.execute(operation, context)

if not success and 'circuit open' in error.lower():
    print("Circuit is open - using fallback")
    result = await fallback_operation()
```

### 5. Test Recovery

```python
# Manually test recovery after opening
breaker.state = CircuitBreakerState.OPEN

# Wait for recovery timeout
await asyncio.sleep(config.recovery_timeout_seconds + 5)

# Try recovery
async def recovery_test():
    return await quick_operation()

success, result, error = await breaker.execute(
    recovery_test,
    context={}
)

# Should transition to CLOSED if successful
```

---

## Troubleshooting

### Issue 1: Circuit Never Opens

**Symptoms:**
- Failures occurring but circuit stays CLOSED

**Diagnosis:**
```python
print(f"Failure count: {breaker.failure_count}")
print(f"Threshold: {breaker.config.failure_threshold}")
```

**Solution:**
- Check if threshold is too high
- Verify failures are being tracked
- Review circuit breaker configuration

### Issue 2: Circuit Never Closes

**Symptoms:**
- Circuit stays OPEN even after system recovers

**Diagnosis:**
```python
print(f"Last failure time: {breaker.last_failure_time}")
print(f"Recovery timeout: {breaker.config.recovery_timeout_seconds}")
```

**Solution:**
- Increase recovery timeout
- Verify recovery logic is working
- Check half-open max calls configuration

### Issue 3: Too Many False Positives

**Symptoms:**
- Circuit opens on transient errors

**Diagnosis:**
```python
# Check error types
for error in breaker.recent_errors:
    print(f"Error type: {error['type']}")
```

**Solution:**
- Implement retry logic for transient errors
- Increase failure threshold
- Use exponential backoff before counting as failure

### Issue 4: Performance Impact

**Symptoms:**
- Circuit breaker adding significant overhead

**Diagnosis:**
```python
# Measure overhead
import time

start = time.time()
# ... operations ...
elapsed = time.time() - start
print(f"Overhead: {elapsed * 1000:.2f}ms")
```

**Solution:**
- Reduce state checks
- Use asynchronous state updates
- Consider individual strategy if hierarchical not needed

---

## Summary

Circuit breakers in Gauntlet provide:
- ✅ Fault isolation at each level
- ✅ Prevention of cascading failures
- ✅ Automatic recovery mechanisms
- ✅ Hierarchical or individual strategies
- ✅ Comprehensive monitoring and alerting

For more information:
- `bubblelabs_nodes/circuit_breakers.py` - Circuit breaker implementation
- `CONFIGURATION_GUIDE.md` - Configuration options
- `METRICS_GUIDE.md` - Monitoring circuit breaker health
