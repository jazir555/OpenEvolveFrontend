# Z3 Adapter

## Purpose

Anti-Corruption Layer (ACL) for Z3 Theorem Prover integration.

## Responsibilities

1. **Normalize Data**: Convert Z3's data format to Canonical Schema
2. **Failure Management**: Implement circuit breakers and retries
3. **API Verification**: Probe Z3's API to verify availability
4. **Contract Enforcement**: Ensure Z3 doesn't break our integration

## Directory Structure

```
z3-adapter/
├── src/              # Adapter implementation code
├── probes/           # API discovery scripts (run BEFORE implementing)
├── tests/            # Contract tests (run on startup)
└── Dockerfile        # Containerize the adapter
```

## Protocol

### Phase 1: Probe (Discovery)

Before writing any code:

1. Create `probes/check_api.sh`
2. Execute against running Z3 container
3. Verify API returns expected fields
4. **IF PROBE FAILS**: Do not write adapter code

### Phase 2: Implement

After successful probe:

1. Define Canonical Schema in `/glue/schemas/`
2. Write adapter with:
   - Circuit breaker (stop hammering if Z3 is down)
   - Exponential backoff retry (with jitter)
   - Request timeout (MANDATORY: 5000ms default)
   - Structured logging (correlation_id, source_service, target_service)

### Phase 3: Contract Test

Protect against upstream changes:

1. Create `tests/contract.test.ts`
2. Assert on all fields we depend on
3. Test runs on container startup
4. **IF CONTRACT FAILS**: Adapter refuses to start

## Integration Points

Z3 provides:
- SMT constraint solving
- Proof generation
- Model checking

We expose via Canonical Schema:
- Standardized proof request format
- Normalized proof status
- Unified error handling

## Failure Scenarios

| Scenario | Strategy |
|----------|----------|
| Network timeout | Exponential backoff retry (max 3 attempts) |
| Z3 container down | Circuit breaker opens, stop requests |
| Invalid query | Return to DLQ, don't block pipeline |
| API changed | Contract test fails, adapter refuses start |

## Configuration

Required environment variables:

```bash
Z3_API_URL=http://z3-core:8000
Z3_TIMEOUT_MS=5000
Z3_RETRY_MAX=3
Z3_CIRCUIT_BREAKER_THRESHOLD=5
```

## Logs

All logs must include:
```json
{
  "correlation_id": "uuid-here",
  "source_service": "z3-adapter",
  "target_service": "z3-core",
  "msg": "Proof request completed",
  "proof_status": "sat",
  "duration_ms": 1234
}
```

## See Also

- Z3 upstream documentation: [link]
- Canonical Schema: `/glue/schemas/z3-schema.ts`
- ADR: `/glue/adapters/z3-adapter/ADR.md`
