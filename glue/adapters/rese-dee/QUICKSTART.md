# RESE Deep Exploration Engine - Quick Start Guide

This guide will help you get started with the RESE Deep Exploration Engine (DEE) in 5 minutes.

## Prerequisites

- Python 3.11+
- Bash shell (for probe scripts)
- Git (for cloning the repository)

## Installation

### 1. Clone and Navigate

```bash
cd glue/adapters/rese-dee
```

### 2. Set Environment Variables

```bash
export EXPLORATION_DEPTH=10
export MCTS_ITERATIONS=100
export EXPLORATION_TIMEOUT_MS=10000
export MAX_HYPOTHESES=50
```

### 3. Run Probe Scripts

Verify the installation:

```bash
cd probes
bash check_dee.sh
```

Expected output:
```
===================================
RESE DEE Probe Script
===================================

Test 1: Checking if DEE module exists...
✓ DEE module found

Test 2: Checking if RESE schemas exist...
✓ RESE schemas found

...

===================================
All tests passed!
===================================
```

## Usage

### Python API

```python
from glue.adapters.rese_dee.src.dee_adapter import DEEAdapter

# Initialize adapter
adapter = DEEAdapter()

# Perform exploration
result = adapter.explore({
    "problem_statement": "System performance degrades when user count exceeds 1000",
    "domain": "performance",
    "context": {
        "system": "web application",
        "database": "postgresql"
    }
})

# Print results
print(f"Best hypothesis: {result['best_hypothesis']['statement']}")
print(f"Confidence: {result['best_confidence']:.2f}")
print(f"Patterns found: {len(result['patterns'])}")
print(f"Iterations: {result['tree_statistics']['iterations']}")
```

### Command Line Interface

```bash
python src/dee_adapter.py \
    --problem "System is slow under load" \
    --domain "performance"
```

### Batch Exploration

```python
results = adapter.batch_explore({
    "problems": [
        {
            "problem_statement": "Problem 1",
            "domain": "performance"
        },
        {
            "problem_statement": "Problem 2",
            "domain": "security"
        }
    ]
})

print(f"Explored {results['successful_results']}/{results['total_problems']} problems")
```

## Configuration

All configuration via environment variables:

```bash
# Core exploration parameters
export EXPLORATION_DEPTH=10              # Maximum search depth
export MCTS_ITERATIONS=1000              # Maximum MCTS iterations
export MCTS_EXPLORATION_CONSTANT=1.414   # UCB exploration constant

# Timeouts and limits
export EXPLORATION_TIMEOUT_MS=10000      # Per-operation timeout (ms)
export MAX_HYPOTHESES=100                # Maximum hypotheses to generate
export PATTERN_RECOGNITION_THRESHOLD=0.7 # Minimum confidence for patterns

# Convergence
export CONVERGENCE_THRESHOLD=0.001       # Convergence threshold
```

## Testing

### Run Unit Tests

```bash
cd tests
pytest test_dee.py -v
```

### Run Integration Tests

```bash
pytest test_integration.py -v
```

### Run All Tests

```bash
pytest -v
```

## Docker

### Build Image

```bash
docker build -t rese-dee-adapter:latest .
```

### Run Container

```bash
docker run --rm \
  -e EXPLORATION_DEPTH=10 \
  -e MCTS_ITERATIONS=100 \
  rese-dee-adapter:latest \
  python src/dee_adapter.py --health
```

### Check Health

```bash
docker run --rm rese-dee-adapter:latest \
  python src/dee_adapter.py --health
```

## Example Workflows

### Example 1: Performance Problem Investigation

```python
adapter = DEEAdapter()

result = adapter.explore({
    "problem_statement": """
    Database queries are slow when concurrent users exceed 500.
    Average query time increases from 50ms to 2000ms.
    CPU usage spikes to 95% on database server.
    """,
    "domain": "performance",
    "context": {
        "database": "postgresql",
        "concurrent_users": 500,
        "query_time_normal": 50,
        "query_time_peak": 2000
    }
})

print("Top Hypothesis:")
print(f"  Statement: {result['best_hypothesis']['statement']}")
print(f"  Confidence: {result['best_hypothesis']['confidence']:.2f}")
print(f"  Evidence: {len(result['best_hypothesis']['evidence'])} items")
```

### Example 2: Security Analysis

```python
result = adapter.explore({
    "problem_statement": """
    Unauthorized access attempts detected from multiple IP addresses.
    Rate limiting not triggered for distributed attacks.
    Session tokens not invalidated after password change.
    """,
    "domain": "security",
    "context": {
        "attack_type": "distributed_brute_force",
        "affected_endpoints": ["/api/login", "/api/auth"],
        "severity": "high"
    }
})

print("Security Hypotheses:")
for i, pattern in enumerate(result['patterns'][:3], 1):
    print(f"{i}. {pattern['description']}")
    print(f"   Confidence: {pattern['confidence']:.2f}")
```

### Example 3: Architecture Decision

```python
result = adapter.explore({
    "problem_statement": """
    Need to choose between microservices and monolithic architecture.
    Team size: 10 developers.
    Expected traffic: 10K requests per second.
    Deployment frequency required: Multiple times per day.
    """,
    "domain": "architecture",
    "context": {
        "team_size": 10,
        "traffic_rps": 10000,
        "deployment_frequency": "daily",
        "scaling_requirements": "high"
    }
})

print("Architecture Analysis:")
print(f"  Best approach confidence: {result['best_confidence']:.2f}")
print(f"  Exploration iterations: {result['tree_statistics']['iterations']}")
print(f"  Convergence: {result['tree_statistics']['convergence_reached']}")
```

## Monitoring

### Health Check

```python
health = adapter.get_health()

print(f"Status: {health['status']}")
print(f"Circuit Breaker: {health['circuit_breaker_state']}")
print(f"DLQ Size: {health['dlq_size']}")
```

### View Dead Letter Queue

```python
dlq_items = adapter.get_dlq_contents()

for item in dlq_items:
    print(f"Error: {item['error']}")
    print(f"Type: {item['error_type']}")
    print(f"Timestamp: {item['timestamp']}")
    print()
```

### Clear DLQ

```python
adapter.clear_dlq()
```

## Troubleshooting

### Problem: "Configuration validation failed"

**Solution:** Set all required environment variables:
```bash
export EXPLORATION_DEPTH=10
export MCTS_ITERATIONS=1000
export EXPLORATION_TIMEOUT_MS=10000
# ... (see full list above)
```

### Problem: "Circuit breaker is OPEN"

**Solution:** Wait for recovery timeout (default 60s) or investigate pattern recognition failures:
```python
health = adapter.get_health()
print(f"Circuit breaker state: {health['circuit_breaker_state']}")

# Check DLQ for root cause
dlq_items = adapter.get_dlq_contents()
for item in dlq_items:
    if item['error_type'] == 'system':
        print(f"System error: {item['error']}")
```

### Problem: Low confidence results (< 0.5)

**Solutions:**
1. Improve problem statement clarity
2. Provide more context
3. Increase MCTS_ITERATIONS
4. Adjust exploration strategy

```python
# Try with more iterations
import os
os.environ['MCTS_ITERATIONS'] = '5000'
result = adapter.explore({...})
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory for detailed use cases
2. **Read Documentation**: See `README.md` for complete API reference
3. **Integration**: Learn how to integrate with RESE pipeline
4. **Contribute**: See CONTRIBUTING.md for development guidelines

## Support

- **Documentation**: `README.md`
- **Architecture**: `ADR.md`
- **Source Recovery**: `../rese-integration/SOURCE_RECOVERY_REPORT.md`
- **Issues**: Create issue in project repository

## Performance Tips

1. **Start Small**: Use lower MCTS_ITERATIONS (50-100) for testing
2. **Increase Gradually**: Ramp up to 1000+ iterations for production
3. **Set Timeouts**: Always set EXPLORATION_TIMEOUT_MS to prevent hangs
4. **Monitor Resources**: Check CPU/memory usage during exploration

```bash
# Quick test (10 seconds)
export MCTS_ITERATIONS=50
export EXPLORATION_TIMEOUT_MS=10000

# Production exploration (2-5 minutes)
export MCTS_ITERATIONS=5000
export EXPLORATION_TIMEOUT_MS=300000
```

Happy exploring!
