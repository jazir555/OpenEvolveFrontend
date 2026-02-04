# RESE Deep Exploration Engine (DEE) Adapter

## Overview

The RESE Deep Exploration Engine (DEE) Adapter provides advanced hypothesis generation, pattern recognition, and MCTS-based exploration for complex problem spaces. This adapter implements Phase III of the RESE pipeline: Monte Carlo Metacognitive Refinement.

## Features

- **Hypothesis Generation**: Automatic generation of testable hypotheses from problem statements
- **Pattern Recognition**: Cross-domain pattern matching (structural, functional, causal)
- **MCTS Exploration**: Monte Carlo Tree Search for intelligent hypothesis refinement
- **Circuit Breaker**: Automatic failure detection and graceful degradation
- **Idempotency**: UPSERT logic with deduplication by hypothesis_id
- **Structured Logging**: JSON Lines format with correlation_id
- **Timeout Protection**: All operations bounded by configurable timeouts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DEE Adapter                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │           Deep Exploration Engine                  │ │
│  │                                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐              │ │
│  │  │   Hypothesis │  │   Pattern    │              │ │
│  │  │  Generator   │  │  Recognizer  │              │ │
│  │  └──────────────┘  └──────────────┘              │ │
│  │                                                    │ │
│  │  ┌──────────────────────────────────────────┐    │ │
│  │  │         MCTS Explainer                   │    │ │
│  │  │  (Selection → Expansion → Simulation)    │    │ │
│  │  └──────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Circuit Breaker ← → Dead Letter Queue                  │
└─────────────────────────────────────────────────────────┘
```

## Configuration

All configuration via environment variables (Law of Configuration Explicitness):

```bash
# Exploration Parameters
export EXPLORATION_DEPTH=10              # Maximum search depth
export MCTS_ITERATIONS=1000              # Maximum MCTS iterations
export MCTS_EXPLORATION_CONSTANT=1.414   # UCB exploration constant (default: sqrt(2))
export CONVERGENCE_THRESHOLD=0.001       # Convergence threshold

# Timeouts and Limits
export EXPLORATION_TIMEOUT_MS=10000      # Per-operation timeout (ms)
export MAX_HYPOTHESES=100                # Maximum hypotheses to generate
export PATTERN_RECOGNITION_THRESHOLD=0.7 # Minimum confidence for patterns

# Alternative Strategies (if used)
export BEAM_WIDTH=10                     # Beam search width
export TEMPERATURE=1.0                   # Simulated annealing temperature
export POPULATION_SIZE=50                # Genetic algorithm population
export MUTATION_RATE=0.1                 # Genetic algorithm mutation rate

# Dead Letter Queue
export DLQ_MAX_SIZE=1000                 # Maximum DLQ entries

# Optional Tracing
export CORRELATION_ID=<uuid>             # For distributed tracing
```

## Usage

### As a Python Library

```python
from glue.adapters.rese_dee.src.dee_adapter import DEEAdapter

# Initialize adapter (reads config from environment)
adapter = DEEAdapter()

# Single exploration
result = adapter.explore({
    "problem_statement": "System exhibits unexpected behavior under load",
    "domain": "system_architecture",
    "context": {
        "load_level": "high",
        "component": "database"
    },
    "correlation_id": "my-trace-id"  # Optional
})

print(f"Best hypothesis: {result['best_hypothesis']['statement']}")
print(f"Confidence: {result['best_confidence']}")
print(f"Patterns found: {len(result['patterns'])}")
```

### Batch Exploration

```python
result = adapter.batch_explore({
    "problems": [
        {
            "problem_statement": "Problem 1",
            "domain": "domain_a"
        },
        {
            "problem_statement": "Problem 2",
            "domain": "domain_b"
        }
    ],
    "context": {"shared": "context"}
})

print(f"Successful: {result['successful_results']}/{result['total_problems']}")
```

### Command Line Interface

```bash
# Set environment variables
export EXPLORATION_DEPTH=5
export MCTS_ITERATIONS=100
export EXPLORATION_TIMEOUT_MS=5000

# Run exploration
python src/dee_adapter.py \
    --problem "System is slow" \
    --domain "performance"

# Check health
python src/dee_adapter.py --health

# View configuration
python src/dee_adapter.py --config

# View Dead Letter Queue
python src/dee_adapter.py --dlq
```

## API Contract

### Request Format

```json
{
  "problem_statement": "string (required)",
  "domain": "string (required)",
  "context": {
    "key": "value"
  },
  "correlation_id": "uuid (optional)"
}
```

### Response Format

```json
{
  "search_id": "uuid",
  "root_hypothesis": {
    "hypothesis_id": "uuid",
    "statement": "string",
    "confidence": 0.0-1.0,
    "status": "pending|testing|confirmed|refuted",
    "evidence": [],
    "counter_evidence": []
  },
  "best_hypothesis": { /* same as root_hypothesis */ },
  "best_confidence": 0.0-1.0,
  "tree_statistics": {
    "iterations": 1000,
    "convergence_reached": true,
    "convergence_iteration": 500,
    "total_nodes": 150,
    "max_depth": 8
  },
  "execution_time_ms": 1234.56,
  "strategy": "mcts",
  "patterns": [
    {
      "pattern_id": "uuid",
      "type": "structural|functional|causal",
      "description": "string",
      "confidence": 0.0-1.0,
      "domains": ["string"],
      "instances": []
    }
  ],
  "timestamp": "ISO-8601"
}
```

## Error Handling

The DEE Adapter implements three-tier error handling:

### 1. Transient Failures
- **Examples**: Network timeouts, temporary unavailability
- **Strategy**: Exponential backoff with jitter (max 3 retries)
- **No DLQ entry**: Request retried automatically

### 2. Logic Failures
- **Examples**: Invalid input, validation failures
- **Strategy**: Immediate failure, add to DLQ
- **DLQ entry**: Added with error type "logic"

### 3. System Failures
- **Examples**: Circuit breaker open, pattern recognition failures
- **Strategy**: Circuit breaker opens, stop requests
- **DLQ entry**: Added with error type "system"

### Dead Letter Queue Access

```python
# Get DLQ contents
dlq_items = adapter.get_dlq_contents()

# Clear DLQ
adapter.clear_dlq()
```

## Testing

### Run Probe Scripts

```bash
cd glue/adapters/rese-dee/probes
bash check_dee.sh
```

### Run Unit Tests

```bash
cd glue/adapters/rese-dee/tests
pytest test_dee.py -v
```

### Run Integration Tests

```bash
pytest test_integration.py -v
```

## Docker Deployment

### Build Image

```bash
cd glue/adapters/rese-dee
docker build -t rese-dee-adapter:latest .
```

### Run Container

```bash
docker run --rm \
  -e EXPLORATION_DEPTH=10 \
  -e MCTS_ITERATIONS=1000 \
  -e EXPLORATION_TIMEOUT_MS=10000 \
  rese-dee-adapter:latest \
  python src/dee_adapter.py --health
```

### Docker Compose

```yaml
version: '3.8'
services:
  rese-dee:
    build: ./glue/adapters/rese-dee
    environment:
      - EXPLORATION_DEPTH=10
      - MCTS_ITERATIONS=1000
      - EXPLORATION_TIMEOUT_MS=10000
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "python", "-c", "from src.dee_adapter import DEEAdapter; DEEAdapter().get_health()"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Monitoring

### Health Check

```python
health = adapter.get_health()
print(health)
# {
#   "status": "healthy" | "degraded",
#   "circuit_breaker_state": "CLOSED" | "OPEN" | "HALF_OPEN",
#   "dlq_size": 0,
#   "config": {...},
#   "timestamp": "ISO-8601"
# }
```

### Structured Logs

All logs are JSON Lines format:

```json
{"msg": "Exploration started", "level": "info", "correlation_id": "uuid", "source_service": "rese_dee", "timestamp": "ISO-8601", "domain": "performance"}
{"msg": "Hypothesis generation complete", "level": "info", "correlation_id": "uuid", "count": 15, "elapsed_ms": 123}
```

## CLAUDE.md Compliance

### Law of Configuration Explicitness ✓
- All config values via environment variables
- Crashes immediately if required vars missing
- No magic defaults

### Law of Idempotency ✓
- Hypothesis deduplication by hypothesis_id
- Evidence deduplication by evidence_id
- UPSERT logic for all state updates

### Circuit Breaker ✓
- Pattern recognition failures trigger circuit breaker
- Automatic recovery after timeout
- Graceful degradation

### Structured Logging ✓
- JSON Lines format
- correlation_id in all logs
- source_service: "rese_dee"

### Timeout ✓
- All operations have timeout (default 10000ms)
- Configurable via EXPLORATION_TIMEOUT_MS

### UTC ✓
- All timestamps in UTC timezone
- ISO-8601 format

## Performance Characteristics

- **Time Complexity**: O(n log n) for pattern recognition, O(n) for MCTS
- **Space Complexity**: O(n) where n = number of hypotheses
- **Typical Runtime**: 1-10 seconds for 1000 MCTS iterations
- **Scalability**: Linear scaling with iteration count

## Troubleshooting

### Circuit Breaker Open

**Symptom**: Requests fail with "Circuit breaker is OPEN"

**Solution**:
1. Check logs for root cause
2. Wait for recovery timeout (default 60s)
3. Investigate pattern recognition failures
4. Clear DLQ and retry

### Timeout Errors

**Symptom**: "MCTS timeout reached"

**Solution**:
1. Increase EXPLORATION_TIMEOUT_MS
2. Reduce MCTS_ITERATIONS
3. Reduce EXPLORATION_DEPTH
4. Check system resources

### Low Confidence Results

**Symptom**: Best hypothesis confidence < 0.5

**Solution**:
1. Improve problem statement clarity
2. Provide more context
3. Increase MCTS_ITERATIONS
4. Adjust exploration strategy

## References

- RESE Technical Manual: `The Recursive Epistemic Solvability Engine (RESE)_ A Technical Manual for Overcoming Intractable Problem Spaces.txt`
- Source Recovery Report: `glue/adapters/rese-integration/SOURCE_RECOVERY_REPORT.md`
- CLAUDE.md Constitution: Project root `CLAUDE.md`

## License

Part of the OpenEvolve Frontend Mega-Structure.
