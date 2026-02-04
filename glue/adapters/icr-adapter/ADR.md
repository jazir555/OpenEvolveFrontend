# ADR: ICR (Iterative Contextual Refinements) Adapter Integration

**Status:** Accepted
**Date:** 2025-02-03
**Authors:** OpenEvolve Federation
**Compliance:** Federation Constitution v1.0

---

## Context

The ICR (Iterative Contextual Refinements) system is a powerful AI framework with 7 distinct operational modes. However, ICR operates in complete isolation within the `core-projects/` directory and lacks external integration capabilities.

### Key Challenges

1. **Multiple Modes**: ICR has 7 distinct modes (Refine, React, Deepthink, Adaptive Deepthink, Agentic, Contextual, Generative UI) with different APIs
2. **No Air Gap**: Direct imports from `core-projects/` violate the Federation Constitution
3. **API Uncertainty**: Documentation may not reflect actual API behavior
4. **Failures**: Network issues, API rate limits, and service outages must be handled gracefully
5. **Idempotency**: Duplicate requests must not cause side effects

---

## Decision

Create a complete ICR Adapter following Federation Constitution laws:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Glue Layer                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              ICR Adapter                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│  │
│  │  │   Canonical  │  │  Circuit     │  │    Retry     ││  │
│  │  │   Schemas    │  │  Breaker     │  │    Logic     ││  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘│  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              ICR Core (core-projects/ICR)                    │
│                    [READ ONLY]                               │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Canonical Schemas** (`icr-canonical.ts`)
   - Zod schemas for all 7 modes
   - Enforce data structure contracts
   - Anti-Corruption Layer (ACL)

2. **ICR Client** (`icr-client.ts`)
   - HTTP client with axios
   - Circuit breaker pattern
   - Exponential backoff retry with jitter
   - Structured JSON logging
   - UTC timestamps only

3. **ICR Adapter** (`adapter.ts`)
   - One method per mode (7 methods)
   - Request builders with correlation IDs
   - Type-safe interfaces
   - Health check support

4. **Probes** (`probes/`)
   - `check_api.sh` - Verify API connectivity
   - `check_modes.sh` - Verify all 7 modes accessible
   - `check_refinement.sh` - Test actual refinement operation

5. **Contract Tests** (`tests/contract.test.ts`)
   - Validate API structure
   - Prevent data corruption from API changes
   - Test error handling
   - Test circuit breaker and retry logic

---

## Rationale

### Why This Approach?

1. **Zero Trust (Federation Constitution)**
   - Probes verify API before writing code
   - No imports from `core-projects/`
   - Contract tests validate runtime behavior

2. **Circuit Breaker**
   - Prevents cascading failures
   - Fast fail when service is down
   - Automatic recovery detection

3. **Exponential Backoff with Jitter**
   - Handles transient failures
   - Prevents thundering herd
   - Configurable retry parameters

4. **Canonical Schemas**
   - Normalize different mode APIs
   - Type safety with Zod
   - Single source of truth

5. **Structured Logging**
   - JSON Lines format
   - Correlation ID tracking
   - Observability out of the box

6. **Configuration Explicitness**
   - No magic defaults
   - Crash on missing config
   - Fail fast principle

### Alternatives Considered

#### Alternative 1: Direct Imports from Core Projects
**Rejected**: Violates Law of Air Gap. Creates tight coupling. Makes ICR updates dangerous.

#### Alternative 2: Simple HTTP Client Without Circuit Breaker
**Rejected**: No failure handling. Cascading failures possible. Violates Federation Constitution.

#### Alternative 3: Use WebSocket Instead of HTTP
**Rejected**: Overkill for request/response pattern. HTTP is simpler and more reliable.

#### Alternative 4: Each Mode as Separate Adapter
**Rejected**: Code duplication. Harder to maintain. No benefit over single adapter.

---

## Consequences

### Positive

1. **Isolation**: Complete air gap between Glue Layer and Core
2. **Reliability**: Circuit breaker prevents cascading failures
3. **Observability**: Structured logging with correlation IDs
4. **Safety**: Contract tests catch API changes
5. **Flexibility**: Easy to add new modes or modify existing ones
6. **Type Safety**: TypeScript + Zod provide end-to-end type safety

### Negative

1. **Complexity**: More code than simple HTTP client
2. **Learning Curve**: Developers must understand circuit breaker and retry logic
3. **Maintenance**: Probes and tests must be kept in sync with API

### Risks

1. **API Changes**: ICR API may change, breaking contract
   - **Mitigation**: Contract tests catch changes early
   - **Mitigation**: Probes verify API before deployment

2. **Circuit Breaker False Positives**: Temporary network issues may open circuit
   - **Mitigation**: Configurable threshold and timeout
   - **Mitigation**: Manual reset capability

3. **Idempotency Violations**: ICR may not be truly idempotent
   - **Mitigation**: Correlation ID tracking for deduplication
   - **Mitigation**: Document non-idempotent operations

---

## Implementation Details

### Mode Mappings

| Mode | Purpose | Key Options |
|------|---------|-------------|
| Refine | Iterative refinements | evolution_mode, refinement_stages |
| React | React app development | worker_count, enable_preview |
| Deepthink | Strategic problem-solving | strategy_count, enable_red_team |
| Adaptive Deepthink | Agent with deepthink | conversation_id, enable_streaming |
| Agentic | Tool-based refinement | enable_diff_tools, enable_file_tools |
| Contextual | Multi-agent collaboration | enable_memory_agent, memory_compression_threshold |
| Generative UI | UI generation | enable_interaction_capture, quality_threshold |

### Error Handling Strategy

```
Error Occurs
    │
    ├─→ Is Retryable? (429, 5xx)
    │       ├─→ Yes: Retry with exponential backoff
    │       │       ├─→ Success: Record success, close circuit
    │       │       └─→ Max retries reached: Record failure, open circuit
    │       └─→ No: Immediate failure, open circuit if threshold exceeded
    │
    └─→ Circuit State
        ├─→ Closed: Allow request
        ├─→ Open: Reject request immediately
        └─→ Half-Open: Allow one test request
```

### Retry Logic

```typescript
delay = initial_delay * (backoff_factor ^ attempt) + jitter
```

Example with defaults:
- Attempt 0: 1000ms + jitter
- Attempt 1: 2000ms + jitter
- Attempt 2: 4000ms + jitter

Jitter is 0-50% of base delay (randomized).

### Circuit Breaker State Machine

```
CLOSED ─(failures >= threshold)──→ OPEN
                                    │
                                    │ (timeout elapsed)
                                    ▼
                               HALF_OPEN ─(success)──→ CLOSED
                                    │
                                    │ (failure)
                                    ▼
                                  OPEN
```

---

## Federation Constitution Compliance Checklist

- ✅ **Law of Air Gap**: No imports from `core-projects/`
- ✅ **Law of Runtime Truth**: Probes verify API before code
- ✅ **Law of Untouchable DB**: SELECT only (no DB writes)
- ✅ **Law of Idempotency**: All operations safe to retry
- ✅ **Law of Configuration Explicitness**: Required env vars crash if missing
- ✅ **Law of UTC**: All timestamps in UTC ISO-8601 format

---

## Testing Strategy

### 1. Probes (Before Implementation)

```bash
./probes/check_api.sh           # Verify API exists
./probes/check_modes.sh         # Verify all 7 modes
./probes/check_refinement.sh    # Verify actual operation
```

### 2. Contract Tests (On Every Deploy)

```bash
npm run test:contract
```

Tests validate:
- Request/response structure for all 7 modes
- Error handling (4xx, 5xx)
- Circuit breaker behavior
- Retry logic
- Idempotency
- UTC timestamps

### 3. Manual Testing

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

// Test each mode
const refine = await icrAdapter.createRefinementRequest('test');
const react = await icrAdapter.createReactRequest('test');
// ... etc

// Test health check
const health = await icrAdapter.healthCheck();

// Test circuit breaker
const state = icrAdapter.getCircuitBreakerState();
```

---

## Monitoring

### Key Metrics

1. **Circuit Breaker State**: CLOSED, OPEN, HALF_OPEN
2. **Failure Count**: Number of consecutive failures
3. **Success Rate**: Percentage of successful requests
4. **Retry Count**: Average retries per request
5. **Execution Time**: Average request duration
6. **Mode Usage**: Requests per mode

### Logging Format

```json
{
  "level": "info|warn|error",
  "msg": "Human-readable message",
  "correlation_id": "UUID for tracing",
  "timestamp_utc": "ISO-8601 timestamp",
  "source_service": "icr-adapter",
  "target_service": "icr-core",
  "mode": "refine|react|...",
  "attempt": 1,
  "delay_ms": 1000
}
```

---

## Rollout Plan

### Phase 1: Verification
1. Run probes to verify ICR API is accessible
2. Run contract tests to validate API structure
3. Document any deviations from expected schema

### Phase 2: Development
1. Implement canonical schemas
2. Implement ICR client with circuit breaker
3. Implement adapter with all 7 modes
4. Write contract tests

### Phase 3: Testing
1. Run all contract tests
2. Test error scenarios
3. Test circuit breaker behavior
4. Test retry logic

### Phase 4: Deployment
1. Deploy to staging environment
2. Monitor logs and metrics
3. Run probes in staging
4. Fix any issues

### Phase 5: Production
1. Deploy to production
2. Monitor closely for first 24 hours
3. Be ready to rollback if circuit breaker opens
4. Document production learnings

---

## Rollback Plan

If critical issues are discovered:

1. **Immediate**: Disable adapter in feature flags
2. **Short-term**: Deploy previous version without ICR integration
3. **Long-term**: Fix issues, re-run probes, re-deploy

### Rollback Triggers

- Circuit breaker remains OPEN for > 5 minutes
- Error rate exceeds 50%
- Data corruption detected
- Contract tests failing in production

---

## Future Improvements

1. **Streaming Support**: Add streaming responses for long-running operations
2. **Metrics Export**: Export Prometheus metrics for monitoring
3. **Request Queuing**: Queue requests when circuit is OPEN
4. **Cache Layer**: Cache mode responses for identical requests
5. **Batch Operations**: Support multiple mode requests in single call
6. **WebSocket Fallback**: Use WebSocket if HTTP is unavailable

---

## References

- Federation Constitution: `/CLAUDE.md`
- ICR Documentation: `/core-projects/Iterative-Contextual-Refinements/README.md`
- Circuit Breaker Pattern: https://martinfowler.com/bliki/CircuitBreaker.html
- Exponential Backoff: https://en.wikipedia.org/wiki/Exponential_backoff

---

**Remember:** You are building a skyscraper on top of moving tectonic plates. Flexibility is fatal. Rigidity in architecture is a necessity.
