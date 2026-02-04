# RESE Pipeline Orchestrator

## Overview

The RESE (Recursive Epistemic Solvability Engine) Pipeline Orchestrator is a comprehensive system that coordinates all four phases of the RESE methodology:

1. **Phase I: Epistemic Audit** - Systematic falsification of assumptions
2. **Phase II: Isomorphic Mapping** - Cross-domain pattern recognition
3. **Phase III: MCTS Search** - Hypothesis generation and validation
4. **Phase IV: Architecture Assembly** - Solution synthesis and validation

## Architecture

```
Input Problem
       ↓
┌──────────────────────────────────────┐
│     RESE Pipeline Orchestrator       │
│                                      │
│  ┌────────────────────────────────┐  │
│  │       Event Bus                │  │
│  │  - Publish/Subscribe           │  │
│  │  - Event deduplication         │  │
│  │  - Correlation tracking        │  │
│  └────────────────────────────────┘  │
│                                      │
│  ┌──────────┐  ┌──────────┐         │
│  │ Phase I  │→ │ Phase II │         │
│  └──────────┘  └──────────┘         │
│       ↓              ↓              │
│  ┌──────────┐  ┌──────────┐         │
│  │Phase III │→ │ Phase IV │         │
│  └──────────┘  └──────────┘         │
│                                      │
│  ┌────────────────────────────────┐  │
│  │  Failure Handler                │  │
│  │  - Circuit Breakers             │  │
│  │  - Retry with Backoff           │  │
│  │  - Dead Letter Queue            │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
       ↓
Synthesized Architecture
```

## Features

### CLAUDE.md Compliance

All components follow the 6 Immutable Laws:

1. **Law of the Air Gap**: No imports from core-projects
2. **Law of Runtime Truth**: Verify before using
3. **Law of the Untouchable DB**: SELECT privileges only
4. **Law of Idempotency**: Entire pipeline safe to replay
5. **Law of Configuration Explicitness**: All config via env vars
6. **Law of UTC**: All operations in UTC timezone

### Failure Management

The orchestrator implements comprehensive failure handling:

- **Transient Failures** (network blips, timeouts): Exponential backoff retry with jitter
- **Logic Failures** (bad data, validation): Dead Letter Queue (DLQ)
- **System Failures** (service down): Circuit breaker (stop calling)

### Event Bus

Inter-phase communication via publish/subscribe event bus:

- Event deduplication (idempotency)
- Correlation tracking across all phases
- Event persistence (optional)
- JSON Lines logging with correlation_id

### Correlation Tracking

Every pipeline execution has a unique correlation ID that:

- Traces across all phases
- Appears in all log entries
- Enables end-to-end debugging

## Installation

### Prerequisites

- Python 3.9+
- Docker (for deployment)
- Kubernetes (optional, for production)

### Local Development

```bash
# Install dependencies
cd glue/orchestration
pip install -r requirements.txt

# Set environment variables
export PIPELINE_TIMEOUT_MS=300000
export PHASE_I_TIMEOUT_MS=60000
export PHASE_II_TIMEOUT_MS=90000
export PHASE_III_TIMEOUT_MS=120000
export PHASE_IV_TIMEOUT_MS=60000
export MAX_RETRIES=3
export RETRY_INITIAL_DELAY_MS=1000
export RETRY_MAX_DELAY_MS=30000
export RETRY_BACKOFF_MULTIPLIER=2.0
# ... (see docker-compose.yml for full list)

# Run pipeline
python -m rese_pipeline --problem "Solve X"
```

### Docker Deployment

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f rese-pipeline

# Stop services
docker-compose down
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f infra/k8s-rese-deployment.yaml

# Check status
kubectl get pods -n rese-system

# View logs
kubectl logs -f deployment/rese-pipeline -n rese-system

# Scale deployment
kubectl scale deployment rese-pipeline --replicas=5 -n rese-system
```

## Configuration

All configuration is via environment variables (Law of Configuration Explicitness).

### Required Variables

```bash
# Pipeline timeouts (milliseconds)
PIPELINE_TIMEOUT_MS=300000
PHASE_I_TIMEOUT_MS=60000
PHASE_II_TIMEOUT_MS=90000
PHASE_III_TIMEOUT_MS=120000
PHASE_IV_TIMEOUT_MS=60000

# Retry configuration
MAX_RETRIES=3
RETRY_INITIAL_DELAY_MS=1000
RETRY_MAX_DELAY_MS=30000
```

### Optional Variables (with defaults)

```bash
# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=60000
CIRCUIT_BREAKER_HALF_OPEN_ATTEMPTS=3

# Dead Letter Queue
DLQ_MAX_SIZE=1000
DLQ_PERSIST_PATH=/data/dlq

# Event bus
EVENT_BUS_MAX_EVENTS=10000
EVENT_BUS_PERSIST_EVENTS=true
EVENT_BUS_PERSIST_PATH=/data/events

# Phase enablement
ENABLE_PHASE_I=true
ENABLE_PHASE_II=true
ENABLE_PHASE_III=true
ENABLE_PHASE_IV=true

# DEE configuration
DEE_EXPLORATION_DEPTH=10
DEE_MCTS_ITERATIONS=1000
DEE_CONVERGENCE_THRESHOLD=0.001

# LLTL configuration
LLTL_ENCODING_DIM=128
LLTL_TIMEOUT_MS=3000

# SCE configuration
SCE_CONTRADICTION_DETECTION=true
SCE_FORMAL_VERIFICATION=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Usage

### Python API

```python
from glue.orchestration.rese_pipeline import RESEPipeline

# Initialize pipeline (loads config from env vars)
pipeline = RESEPipeline()

# Execute pipeline
result = pipeline.execute(
    problem_statement="Design a quantum error correction system",
    context={"domain": "quantum_computing"}
)

# Check result
if result["status"] == "completed":
    print(f"Pipeline completed in {result['execution_time_ms']:.2f}ms")

    # Access phase results
    phase_i_result = result["results"]["phase_i"]
    phase_ii_result = result["results"]["phase_ii"]
    phase_iii_result = result["results"]["phase_iii"]
    phase_iv_result = result["results"]["phase_iv"]
else:
    print(f"Pipeline failed: {result['error']}")
```

### CLI

```bash
# Execute pipeline
python -m glue.orchestration.rese_pipeline --problem "Solve X"

# Show configuration
python -m glue.orchestration.rese_pipeline --config

# Show statistics
python -m glue.orchestration.rese_pipeline --stats

# Show DLQ contents
python -m glue.orchestration.rese_pipeline --dlq
```

## Monitoring

### Health Checks

```bash
# Check pipeline health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready
```

### Statistics

```python
# Get pipeline statistics
stats = pipeline.get_stats()
print(json.dumps(stats, indent=2))
```

## Development

### Adding a New Phase

1. Create phase executor class inheriting from `PhaseExecutor`
2. Implement `execute()` method
3. Add phase configuration variables
4. Register phase in `RESEPipeline`
5. Add event types to `EventBus`

## Performance

### Typical Execution Times

- Phase I: 10-30 seconds
- Phase II: 30-90 seconds
- Phase III: 60-120 seconds (MCTS intensive)
- Phase IV: 20-40 seconds
- **Total**: 2-5 minutes

### Resource Requirements

- **Minimum**: 2 CPU, 4GB RAM
- **Recommended**: 4 CPU, 8GB RAM
- **Production**: 8+ CPU, 16GB+ RAM with autoscaling

## License

MIT
