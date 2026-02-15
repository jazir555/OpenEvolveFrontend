# ADR-001: Circuit Breaker Pattern Implementation

## Status
**Accepted**

## Context
The OpenEvolve Frontend integrates with 30+ external services (OpenEvolve API, RAGBits, Datapizza, Z3, LeanAide, etc.). These services can experience:
- Transient failures (network blips, timeouts)
- Cascading failures (one service failure affecting others)
- Partial outages (some endpoints working, others failing)

Without protection, a failing service can cause the entire application to hang, become unresponsive, or crash due to resource exhaustion.

## Problem
When an external service starts failing:
1. **Infinite hangs**: Requests timeout (30s+), blocking UI threads
2. **Resource exhaustion**: Too many open connections/requests waiting for timeout
3. **Cascading failures**: Slow responses cause downstream services to also fail
4. **Poor UX**: Application appears frozen, users don't know what's happening

## Decision
Implement the **Circuit Breaker Pattern** across all HTTP clients following these principles:

### Implementation
- **Threshold**: Trip after 5 consecutive failures
- **Timeout**: Stay open for 60 seconds before attempting recovery
- **States**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **Auto-reset**: Transition from OPEN → HALF_OPEN after timeout
- **Manual reset**: Allow manual reset via `reset()` method

### Per-Service Circuit Breakers
Each HTTP client (OpenEvolve, RAGBits, Datapizza, Z3, LeanAide) has its own circuit breaker instance.

### Code Location
- `glue/lib/circuit-breaker.ts` - Reusable circuit breaker implementation
- Integrated in all API clients: `openevolveApi.ts`, `ragbitsClient.ts`, `DatapizzaClient.ts`

### Example Usage
```typescript
const cb = new CircuitBreaker({
  threshold: 5,
  timeout_ms: 60000,
  onStateChange: (old, newState) => {
    logger.warn(`Circuit: ${old} -> ${newState}`);
  }
});

try {
  const result = await cb.execute(async () => {
    return await fetch(url);
  });
} catch (error) {
  if (cb.getState() === CircuitState.OPEN) {
    // Use fallback data or show cached results
  }
}
```

## Consequences

### Positive
- ✅ **Prevents cascading failures** - Failing services don't hang the app
- ✅ **Fast failure** - Requests rejected immediately when circuit is OPEN
- ✅ **Automatic recovery** - Service returns to normal when healthy
- ✅ **Observable** - State changes logged for monitoring
- ✅ **Manual control** - Can manually reset if needed

### Negative
- ⚠️ **Added complexity** - Need to handle OPEN state in application code
- ⚠️ **Fallback logic required** - Must have cached/default data when circuit is open
- ⚠️ **Tuning required** - Threshold/timeout values may need adjustment per service

### Mitigations
- Provide clear error messages when circuit is OPEN
- Use cached data as fallback where possible
- Monitor circuit breaker state changes
- Document threshold/timeout choices

## Alternatives Considered

### Alternative 1: Retry Only
**Description**: Retry requests with exponential backoff, no circuit breaker.

**Pros**: Simpler, handles transient failures

**Cons**: Doesn't prevent resource exhaustion, keeps hitting failing service

**Rejected**: Insufficient protection against sustained outages

### Alternative 2: Timeout Only
**Description**: Set request timeouts, let them fail naturally

**Pros**: Simple, no additional code

**Cons**: Still wastes resources waiting for timeout, doesn't prevent cascading failures

**Rejected**: Doesn't address the core problem of repeated failures

### Alternative 3: Service Mesh (Istio, Linkerd)
**Description**: Use service mesh for circuit breaking

**Pros**: Infrastructure-level solution, works for all services

**Cons**: Additional infrastructure complexity, heavy weight for frontend app

**Rejected**: Overkill for frontend application, adds infrastructure dependency

## Related Decisions
- [ADR-002: Structured Logging with Correlation IDs](./002-structured-logging.md)
- [ADR-003: Retry Logic with Exponential Backoff](./003-retry-logic.md)

## Implementation Date
2026-02-15

## Author
OpenEvolve Federation Team
