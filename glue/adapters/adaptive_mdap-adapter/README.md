# Adaptive MDAP/MAKER Adapter

**Federation Constitution Compliant Adapter** for Adaptive Multi-Dimensional Adaptive Processing (MDAP) and MAKER Engine integration.

## Overview

This adapter provides the **Anti-Corruption Layer (ACL)** between the OpenEvolve core projects (Adaptive MDAP, MAKER Engine) and the glue orchestration layer. It ensures:

- ✅ **Zero Trust Architecture**: All APIs verified at runtime
- ✅ **Anti-Corruption Layer**: Schema transformation to canonical format
- ✅ **Circuit Breaker Pattern**: Prevents cascading failures
- ✅ **Contract Testing**: Fail-fast on API violations
- ✅ **Idempotency**: All operations safe to retry
- ✅ **Observability**: Structured logging with correlation IDs

## Quick Start

### 1. Configure Environment Variables

```bash
# Required (service will not start without this)
export ADAPTIVE_MDAP_TIMEOUT_MS=5000

# Optional (with defaults shown)
export ADAPTIVE_MDAP_MAX_RETRIES=3
export ADAPTIVE_MDAP_RETRY_DELAY_MS=100
export ADAPTIVE_MDAP_CIRCUIT_BREAKER_THRESHOLD=5
export ADAPTIVE_MDAP_CIRCUIT_BREAKER_TIMEOUT_MS=60000
export ADAPTIVE_MDAP_LOG_LEVEL=INFO
```

### 2. Run Probes (Runtime Verification)

```bash
# Verify Adaptive MDAP APIs
./probes/check_adaptive_mdap_api.sh

# Verify MAKER Engine APIs
./probes/check_maker_api.sh

# Verify Integration
./probes/check_integration.sh
```

### 3. Use the Adapter

```python
from src import (
    get_adapter,
    CanonicalSubProblem,
    ProcessingDomain,
    TaskStatus
)

# Get adapter instance
adapter = get_adapter()

# Create subproblem
subproblem = CanonicalSubProblem(
    id="task-001",
    description="Implement secure authentication with OAuth2",
    domain="security",
    depth=3,
    dependencies=["user-model", "token-service"]
)

# Analyze complexity
response = adapter.analyze_complexity(
    subproblem=subproblem,
    correlation_id="my-request-123"
)

# Check result
if response.status == TaskStatus.COMPLETED:
    print(f"Complexity Score: {response.complexity_score.overall_score}")
    print(f"Strategy: {response.strategy.strategy}")
    print(f"Agents: {response.strategy.n_agents}")
else:
    print(f"Error: {response.error}")
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Core Projects                             │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Adaptive MDAP   │         │   MAKER Engine   │          │
│  │     Module       │         │                  │          │
│  └────────┬─────────┘         └────────┬─────────┘          │
└───────────┼──────────────────────────┼──────────────────────┘
            │                          │
            │     ┌────────────────────┴──────────────────┐    │
            │     │   THIS ADAPTER (Anti-Corruption)     │    │
            │     │   ┌──────────────────────────────┐   │    │
            │     │   │  ACL Transformation          │   │    │
            │     │   │  External → Canonical        │   │    │
            │     │   └──────────────────────────────┘   │    │
            │     │   ┌──────────────────────────────┐   │    │
            │     │   │  Circuit Breaker             │   │    │
            │     │   │  Retry Logic                 │   │    │
            │     │   │  Health Checks               │   │    │
            │     │   └──────────────────────────────┘   │    │
            │     └────────────────────┬──────────────────┘    │
            └──────────────────────────┼──────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Canonical Schema         │
                        │  (Single Source of Truth)   │
                        └──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Glue Orchestration        │
                        │   (Event Bus / Workflows)   │
                        └──────────────────────────────┘
```

## Canonical Schema

### SubProblem

```python
@dataclass
class CanonicalSubProblem:
    id: str                          # Required: Unique identifier
    description: str                 # Required: Problem description
    domain: str                      # Required: Domain (e.g., "security", "ml")
    depth: int = 1                   # Optional: Depth in decomposition tree
    dependencies: List[str] = []     # Optional: List of dependent subproblem IDs
    metadata: Dict[str, Any] = {}    # Optional: Additional metadata
```

### Complexity Score

```python
@dataclass
class CanonicalComplexityScore:
    overall_score: float             # Required: 0.0 to 1.0
    text_length_score: float = 0.0
    domain_rarity_score: float = 0.0
    depth_score: float = 0.0
    dependency_score: float = 0.0
    feature_weights: Dict[str, float] = {}
    timestamp: str                   # UTC ISO-8601
```

### Strategy

```python
@dataclass
class CanonicalStrategy:
    strategy: str                    # Strategy name (e.g., "MAKER_ULTRA")
    n_agents: int                    # Number of agents to use
    k_ahead: int                     # K-ahead voting parameter
    max_retries: int                 # Maximum retry attempts
    timeout_ms: int                  # Timeout in milliseconds
```

## API Reference

### AdaptiveMDAPAdapter

#### `analyze_complexity(subproblem, correlation_id=None)`

Analyze subproblem complexity.

**Parameters:**
- `subproblem` (CanonicalSubProblem | Any): Subproblem to analyze
- `correlation_id` (str, optional): Correlation ID for distributed tracing

**Returns:** `CanonicalResponse`

**Example:**
```python
response = adapter.analyze_complexity(
    subproblem=CanonicalSubProblem(
        id="task-001",
        description="Build ML pipeline",
        domain="ml"
    ),
    correlation_id="req-001"
)
```

#### `allocate_resources(complexity_score, correlation_id=None)`

Allocate resources based on complexity score.

**Parameters:**
- `complexity_score` (CanonicalComplexityScore | Any): Complexity score
- `correlation_id` (str, optional): Correlation ID

**Returns:** `CanonicalResponse`

#### `health_check()`

Perform health check and return status.

**Returns:** `Dict[str, Any]` with:
- `status`: "healthy" or "degraded"
- `circuit_breaker_state`: Current circuit state
- `mdap_available`: Whether MDAP is available
- `metrics`: Performance metrics

### MakerAdapter

#### `execute_maker_step(step, current_state, history, team, correlation_id=None)`

Execute a MAKER voting step.

**Parameters:**
- `step` (CanonicalMakerStep): Step to execute
- `current_state` (Any): Current execution state
- `history` (List[Dict]): Execution history
- `team` (Any): MAKER team configuration
- `correlation_id` (str, optional): Correlation ID

**Returns:** `CanonicalMakerResult`

#### `check_red_flags(raw_text, candidate, expected_schema=None, correlation_id=None)`

Check content for red flags.

**Returns:** `Tuple[bool, List[str]]` - (is_flagged, reasons)

## Docker Deployment

### Build

```bash
docker build -t adaptive-mdap-adapter:1.0.0 .
```

### Run

```bash
docker run --rm \
  -e ADAPTIVE_MDAP_TIMEOUT_MS=5000 \
  -e ADAPTIVE_MDAP_LOG_LEVEL=INFO \
  adaptive-mdap-adapter:1.0.0
```

### Health Check

The container includes a built-in health check:

```bash
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Testing

### Run Contract Tests

```bash
# From project root
pytest glue/adapters/adaptive_mdap-adapter/tests/contract.test.py -v

# From adapter directory
pytest tests/contract.test.py -v
```

### Run Probes

```bash
# All probes
cd glue/adapters/adaptive_mdap-adapter
./probes/check_adaptive_mdap_api.sh
./probes/check_maker_api.sh
./probes/check_integration.sh
```

## Federation Constitution Compliance

| Law | Requirement | Implementation |
|-----|-------------|----------------|
| 1. Air Gap | No imports from core-projects/ | ✅ Separate adapter, ACL pattern |
| 2. Runtime Truth | Probes verify APIs | ✅ 3 probe scripts verify behavior |
| 3. Untouchable DB | SELECT-only operations | ✅ Stateless adapter, no DB |
| 4. Idempotency | Safe to retry | ✅ All operations idempotent |
| 5. Config Explicitness | No magic defaults | ✅ Crash on missing required env vars |
| 6. UTC | UTC ISO-8601 timestamps | ✅ All timestamps in UTC |

## Error Handling

### Circuit Breaker States

1. **CLOSED**: Normal operation, requests allowed
2. **OPEN**: Failing, requests rejected
3. **HALF_OPEN**: Testing if service recovered

### Retry Logic

- Exponential backoff: delay doubles each retry
- Max retries configured via `ADAPTIVE_MDAP_MAX_RETRIES`
- Initial delay configured via `ADAPTIVE_MDAP_RETRY_DELAY_MS`

### Graceful Degradation

When components are unavailable:
- Returns error response with clear error code
- Logs structured error with correlation ID
- Circuit breaker prevents further failures

## Monitoring

### Metrics

The adapter tracks:
- `requests_total`: Total requests processed
- `requests_success`: Successful requests
- `requests_failed`: Failed requests
- `circuit_breaker_trips`: Circuit breaker openings

### Logging

Structured JSON logs include:
- `timestamp`: UTC ISO-8601
- `correlation_id`: Distributed tracing
- `source_service`: Adapter name
- `execution_time_ms`: Operation duration

## Examples

See the `examples/` directory for complete examples:

- `basic_complexity_analysis.py`: Basic complexity analysis
- `resource_allocation.py`: Resource allocation based on complexity
- `maker_voting.py`: MAKER voting execution
- `integration.py`: Full MDAP/MAKER integration

## Troubleshooting

### Service fails to start

**Error**: `ADAPTIVE_MDAP_TIMEOUT_MS is required`

**Solution**: Set the required environment variable:
```bash
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
```

### Circuit breaker is open

**Symptom**: All requests rejected with "CIRCUIT_BREAKER_OPEN"

**Solution**:
1. Check if underlying services are healthy
2. Wait for circuit breaker timeout (default 60s)
3. Check logs for failure reasons

### Import errors

**Error**: `No module named 'adaptive_mdap'`

**Solution**:
1. Run probes to verify installation
2. Check Python path includes `src/`
3. Install dependencies: `pip install -r requirements.txt`

## Contributing

When making changes:

1. **Add Contract Tests**: Any new API fields need contract tests
2. **Update Probes**: New functionality needs runtime verification
3. **Update ADR.md**: Document architectural decisions
4. **Test Compliance**: Verify all 6 Federation Constitution laws

## References

- [Architecture Decision Record](ADR.md)
- [Federation Constitution](../../../CLAUDE.md)
- [Canonical Schema](../../schemas/adaptive-mdap-canonical.ts)

---

**Version**: 1.0.0
**Last Updated**: 2025-02-17
**Status**: Production Ready ✅
