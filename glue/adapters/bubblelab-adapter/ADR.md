# Architecture Decision Record (ADR): BubbleLab Adapter

**Status:** Accepted
**Date:** 2026-02-03
**Context:** OpenEvolve Federation - BubbleLab Integration

---

## Context

BubbleLab is a workflow engine that allows users to create "bubbles" - modular workflow components that can be chained together to perform complex operations. The OpenEvolve Federation needs to integrate BubbleLab as a core project while maintaining strict architectural boundaries.

### Key Challenges

1. **Air Gap Compliance**: BubbleLab source code is in `core-projects/` and must remain immutable
2. **API Complexity**: BubbleLab has multiple endpoint types (flows, executions, validation, webhooks)
3. **Credential Management**: Multiple credential types (database, Slack, AI services)
4. **Event Streaming**: Support for SSE-based streaming execution
5. **Runtime Verification**: Need to verify API contracts before trusting integration

### BubbleLab Architecture Understanding

From analysis of BubbleLab codebase:
- **BubbleFlows**: TypeScript classes extending `BubbleFlow` base class
- **Bubbles**: Individual components (PostgreSQL, Slack, AI Agent, etc.)
- **Triggers**: Event types (webhook/http, schedule, manual)
- **Credentials**: Per-bubble credential requirements
- **Execution**: Can be synchronous or streaming (SSE)

---

## Decision

### 1. Anti-Corruption Layer (ACL) Implementation

**Decision**: Implement a full ACL with canonical schema mapping between BubbleLab and OpenEvolve.

**Rationale**:
- BubbleLab uses camelCase, OpenEvolve canonical uses snake_case
- Credential types need normalization
- Execution status needs standardization
- Future-proofing against BubbleLab API changes

**Implementation**:
```typescript
// Canonical schema
interface CanonicalBubbleFlow {
  id: string;
  name: string;
  event_type: EventType;  // Enum
  required_credentials: Record<string, CredentialType[]>;
  // ... other fields
}

// Mapping functions
mapToCanonicalBubbleFlow(apiResponse): CanonicalBubbleFlow
mapFromCanonicalBubbleFlow(canonical): ApiRequest
```

### 2. Circuit Breaker Pattern

**Decision**: Use circuit breaker for all BubbleLab API calls.

**Rationale**:
- BubbleLab workflows can be long-running (up to timeout)
- Cascading failures must be prevented
- Service degradation handling required

**Configuration**:
- Threshold: 5 failures before opening
- Timeout: 60 seconds in OPEN state
- Reset: Automatic transition to HALF_OPEN after timeout

### 3. Retry Logic with Exponential Backoff

**Decision**: Retry transient failures with jittered exponential backoff.

**Rationale**:
- Network blips common in distributed systems
- BubbleLab containers may be temporarily unavailable
- Jitter prevents thundering herd

**Configuration**:
- Max retries: 3
- Base delay: 1000ms
- Max delay: 10,000ms
- Jitter: 500ms

### 4. Idempotency Strategy

**Decision**: Implement idempotent operations where possible, document non-idempotent ones.

**Idempotent Operations**:
- GET requests (list flows, get flow, get history)
- DELETE with verification
- Upsert (check if exists before create)

**Non-Idempotent Operations** (documented):
- POST /bubble-flow (creates new flow)
- POST /bubble-flow/:id/execute (creates new execution)

**Strategy**:
```typescript
async upsertBubbleFlow(flow: CanonicalBubbleFlow) {
  if (flow.id) {
    try {
      return await this.updateBubbleFlow(flow.id, flow);
    } catch {
      // Flow doesn't exist, create it
    }
  }
  return await this.createBubbleFlow(flow);
}
```

### 5. Contract Testing

**Decision**: Implement contract tests that run on adapter startup.

**Rationale**:
- "Law of Runtime Truth" - verify before trusting
- API contracts can change without notice
- Prevent silent failures from breaking changes

**Implementation**:
```typescript
// Run on adapter startup
validateAllContracts(): boolean {
  // Test health endpoint
  // Test flow list endpoint
  // Test execution endpoint
  // Fail fast if contract violated
}
```

### 6. Timeout Enforcement

**Decision**: All API calls must have explicit timeouts from environment variables.

**Rationale**:
- "Law of Configuration Explicitness"
- Workflow executions can hang indefinitely
- No magic defaults allowed

**Implementation**:
```typescript
// Required env vars
BUBBLELAB_API_URL=http://bubblelab-core:3000
TIMEOUT_MS=5000  // Fails fast if not set

// Per-operation timeouts
executeBubbleFlow(timeout_ms = 30000)  // Longer for executions
healthCheck(timeout_ms = 2000)  // Shorter for health
```

### 7. Structured Logging (JSON Lines)

**Decision**: All logs must be structured JSON with correlation IDs.

**Format**:
```json
{
  "level": "info",
  "msg": "BubbleFlow executed successfully",
  "timestamp": "2026-02-03T10:30:00.000Z",
  "correlation_id": "a1b2c3d4-...",
  "source_service": "bubblelab-adapter",
  "target_service": "bubblelab-api",
  "flow_id": "123",
  "execution_id": "exec-456"
}
```

### 8. UTC Timestamps (Law of UTC)

**Decision**: All timestamps processed and stored in UTC ISO-8601 format.

**Implementation**:
```typescript
toUTCISOString(date: Date): string {
  return date.toISOString();  // Always UTC
}

fromUTCISOString(isoString: string): Date {
  return new Date(isoString);  // Preserves UTC
}
```

---

## Alternatives Considered

### Alternative 1: Direct API Usage (No Adapter)
**Rejected**: Violates "Air Gap" law, creates tight coupling

### Alternative 2: Shared Types Import
**Rejected**: Would require importing from `core-projects/`, violates isolation

### Alternative 3: GraphQL Middleware
**Rejected**: Adds unnecessary complexity, BubbleLab has REST API

### Alternative 4: Event-Driven Only (No Direct Calls)
**Rejected**: Still need synchronous operations for some use cases

---

## Consequences

### Positive

1. **Isolation**: Full compliance with Air Gap law
2. **Resilience**: Circuit breaker prevents cascading failures
3. **Observability**: Structured logs with correlation IDs
4. **Safety**: Contract tests catch breaking changes early
5. **Flexibility**: Canonical schema allows swapping BubbleLab implementation

### Negative

1. **Complexity**: Additional mapping layer adds maintenance overhead
2. **Performance**: Mapping adds small latency (~1-2ms per request)
3. **Testing**: More surface area to test (contracts + adapter logic)

### Risks

1. **Contract Drift**: BubbleLab API changes could break contracts
   - **Mitigation**: Contract tests run on startup, fail fast
2. **Credential Leaks**: Storing credentials in memory
   - **Mitigation**: Never log credentials, use encrypted storage
3. **Event Bus Bottleneck**: High-volume workflow executions
   - **Mitigation**: Async event emission with batching

---

## Implementation Details

### File Structure

```
bubblelab-adapter/
├── probes/
│   ├── check_api.sh           # API endpoint probes
│   ├── check_bubbles.sh       # Bubble operation probes
│   └── check_workflows.sh     # Workflow execution probes
├── src/
│   ├── adapter.ts             # Main adapter with circuit breaker
│   ├── bubble-client.ts       # Direct API client
│   ├── bubblelab-canonical.ts # Canonical schema & mapping
│   └── index.ts               # Exports
├── tests/
│   ├── contract.test.ts       # Contract validation tests
│   └── jest.config.js         # Jest configuration
├── ADR.md                     # This document
├── README.md                  # Usage documentation
└── package.json               # Dependencies
```

### Environment Variables

**Required**:
- `BUBBLELAB_API_URL`: Base URL of BubbleLab API
- `TIMEOUT_MS`: Default timeout for requests (ms)

**Optional**:
- `BUBBLELAB_AUTH_TOKEN`: Authentication token
- `CIRCUIT_BREAKER_THRESHOLD`: Failures before trip (default: 5)
- `CIRCUIT_BREAKER_TIMEOUT_MS`: Time in OPEN state (default: 60000)
- `RETRY_MAX_RETRIES`: Maximum retry attempts (default: 3)

### Usage Example

```typescript
import { createBubbleLabAdapter } from '@openevolve/bubblelab-adapter';

// Create adapter (validates env vars)
const adapter = createBubbleLabAdapter();

// Health check
const isHealthy = await adapter.healthCheck();

// List flows
const flows = await adapter.listBubbleFlows();

// Execute flow
const result = await adapter.executeBubbleFlow(
  'flow-id-123',
  { data: 'test' },
  { [CredentialType.DATABASE_CRED]: 456 },
  'correlation-abc-123'
);

// Get metrics
const metrics = adapter.getMetrics();
```

---

## Validation

### Compliance Checklist

- [x] Law of Air Gap: No imports from `core-projects/`
- [x] Law of Runtime Truth: Probes verify API before use
- [x] Law of Untouchable DB: No direct DB writes
- [x] Law of Idempotency: Documented idempotent operations
- [x] Law of Configuration Explicitness: All required env vars validated
- [x] Law of UTC: All timestamps in UTC ISO-8601

### Testing Strategy

1. **Unit Tests**: Canonical schema validation
2. **Contract Tests**: API response structure validation
3. **Integration Tests**: End-to-end workflow execution
4. **Probe Tests**: Shell scripts verify API availability

---

## References

- [Federation Constitution](../../../../CLAUDE.md)
- [BubbleLab API Documentation](../../../../core-projects/BubbleLab/docs/api.md)
- [Anti-Corruption Layer Pattern](https://martinfowler.com/bliki/AnticorruptionLayer.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

**Authors**: OpenEvolve Federation
**Last Updated**: 2026-02-03
**Version**: 1.0.0
