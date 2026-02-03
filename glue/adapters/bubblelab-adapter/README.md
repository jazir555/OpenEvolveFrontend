# BubbleLab Adapter

Integrates BubbleLab workflow engine with the OpenEvolve Federation orchestration layer.

## Overview

The BubbleLab adapter provides a resilient, contract-validated integration with BubbleLab's workflow API. It implements the full Federation Constitution compliance including circuit breakers, retry logic, canonical schema mapping, and structured logging.

## Features

- **Circuit Breaker**: Prevents cascading failures from BubbleLab outages
- **Exponential Backoff Retry**: Handles transient network failures gracefully
- **Canonical Schema Mapping**: Anti-corruption layer between BubbleLab and OpenEvolve
- **Contract Testing**: Validates API contracts on startup
- **Idempotent Operations**: Safe retry logic for critical operations
- **Structured Logging**: JSON Lines output with correlation IDs
- **UTC Timestamps**: All times in UTC ISO-8601 format (Law of UTC)

## Installation

```bash
cd glue/adapters/bubblelab-adapter
npm install
```

## Environment Variables

### Required

```bash
# Base URL of BubbleLab API (no default, must be set)
export BUBBLELAB_API_URL="http://bubblelab-core:3000"

# Request timeout in milliseconds (no default, must be set)
export TIMEOUT_MS="5000"
```

### Optional

```bash
# Authentication token (if BubbleLab requires auth)
export BUBBLELAB_AUTH_TOKEN="your-token-here"

# Circuit breaker configuration
export CIRCUIT_BREAKER_THRESHOLD="5"      # Failures before trip (default: 5)
export CIRCUIT_BREAKER_TIMEOUT_MS="60000" # Time in OPEN state (default: 60s)

# Retry configuration
export RETRY_MAX_RETRIES="3"              # Max retry attempts (default: 3)
export RETRY_BASE_DELAY_MS="1000"         # Base delay (default: 1000ms)
export RETRY_MAX_DELAY_MS="10000"         # Max delay (default: 10000ms)
```

## Usage

### Basic Example

```typescript
import { createBubbleLabAdapter } from '@openevolve/bubblelab-adapter';

// Create adapter (validates required env vars)
const adapter = createBubbleLabAdapter();

// Health check
const isHealthy = await adapter.healthCheck();
console.log('BubbleLab healthy:', isHealthy);

// List all workflows
const flows = await adapter.listBubbleFlows();
console.log('Found', flows.length, 'workflows');

// Get a specific workflow
const flow = await adapter.getBubbleFlow('flow-id-123');
console.log('Workflow:', flow.name);
```

### Creating a Workflow

```typescript
import { CanonicalBubbleFlow, EventType } from '@openevolve/bubblelab-adapter';

const newFlow: CanonicalBubbleFlow = {
  name: 'My Test Workflow',
  description: 'A test workflow for BubbleLab integration',
  event_type: EventType.WEBHOOK_HTTP,
  code: `
    import { BubbleFlow } from '@bubblelab/bubble-core';

    export class MyTestFlow extends BubbleFlow<'webhook/http'> {
      constructor() {
        super('my-test-flow', 'A test workflow');
      }

      async handle(payload: any) {
        return {
          message: 'Hello from BubbleLab!',
          received: payload,
          timestamp: new Date().toISOString()
        };
      }
    }
  `,
  bubbles: [],
  webhook_active: false,
};

const created = await adapter.createBubbleFlow(newFlow, 'correlation-id-123');
console.log('Created workflow:', created.id);
```

### Executing a Workflow

```typescript
import { CredentialType } from '@openevolve/bubblelab-adapter';

// Execute with payload and credentials
const result = await adapter.executeBubbleFlow(
  'flow-id-123',
  {
    data: 'test data',
    userId: 'user-456',
  },
  {
    [CredentialType.DATABASE_CRED]: 789,
    [CredentialType.SLACK_CRED]: 101,
  },
  'correlation-id-abc-123'
);

console.log('Execution status:', result.status);
console.log('Output:', result.output);
```

### Idempotent Upsert

```typescript
// Create or update if exists
const flow: CanonicalBubbleFlow = {
  id: 'flow-id-123',  // Include ID for upsert
  name: 'Updated Workflow',
  event_type: EventType.MANUAL,
  // ... other fields
};

const upserted = await adapter.upsertBubbleFlow(flow, 'correlation-id');
```

### Getting Execution History

```typescript
const history = await adapter.getExecutionHistory(
  'flow-id-123',
  50,  // limit
  0,   // offset
  'correlation-id'
);

console.log('Execution history:', history.length, 'runs');
```

### Monitoring Metrics

```typescript
const metrics = adapter.getMetrics();
console.log('Adapter metrics:', {
  total_requests: metrics.total_requests,
  success_rate: metrics.successful_requests / metrics.total_requests,
  circuit_state: metrics.circuit_breaker_state,
  circuit_failures: metrics.circuit_breaker_failure_count,
  avg_duration_ms: metrics.average_request_duration_ms,
});

// Reset circuit breaker if needed (e.g., after manual health check)
if (metrics.circuit_breaker_state === 'open') {
  adapter.resetCircuitBreaker();
}
```

## Probes

The adapter includes probe scripts to verify BubbleLab API functionality:

```bash
# Test API endpoints
cd probes
./check_api.sh

# Test bubble operations
./check_bubbles.sh

# Test workflow execution
./check_workflows.sh
```

All probes output JSON Lines for log aggregation:

```json
{"level":"info","msg":"Starting BubbleLab API probe","timestamp":"2026-02-03T10:30:00.000Z","probe":"check_api.sh"}
{"level":"info","msg":"Health check passed: ok","timestamp":"2026-02-03T10:30:01.000Z","probe":"check_api.sh"}
```

## Contract Testing

Run contract tests to verify API compatibility:

```bash
npm run test:contract
```

Contract tests validate:
- Health check response structure
- BubbleFlow list response structure
- BubbleFlow creation response structure
- Execution response structure
- Execution history response structure

If contracts are violated, the adapter will refuse to start (Law of Runtime Truth).

## Canonical Schema

The adapter maps BubbleLab's API responses to a canonical schema:

### CanonicalBubbleFlow

```typescript
interface CanonicalBubbleFlow {
  id?: string;
  name: string;
  description?: string;
  event_type: EventType;
  code?: string;
  bubbles?: CanonicalBubble[];
  required_credentials?: Record<string, CredentialType[]>;
  webhook_active: boolean;
  webhook_url?: string;
  created_at?: string;  // UTC ISO-8601
  updated_at?: string;  // UTC ISO-8601
}
```

### CanonicalExecutionResult

```typescript
interface CanonicalExecutionResult {
  execution_id?: string;
  flow_id: string;
  status: ExecutionStatus;
  output?: any;
  error?: string;
  started_at: string;      // UTC ISO-8601
  completed_at?: string;   // UTC ISO-8601
  duration_ms?: number;
  logs?: Array<{
    timestamp: string;    // UTC ISO-8601
    level: string;
    message: string;
  }>;
}
```

## Error Handling

The adapter distinguishes between error types:

### Transient Failures (Retried)
- Network timeouts
- Connection refused
- HTTP 5xx errors

### Logic Failures (Not Retried, Sent to DLQ)
- Validation errors (HTTP 400)
- Not found errors (HTTP 404)
- Authentication errors (HTTP 401/403)

### System Failures (Circuit Breaker)
- Circuit breaker OPEN
- Service unavailable
- Sustained failure threshold exceeded

## Development

### Running Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Watch mode
npm run test:watch
```

### Building

```bash
npm run build
```

### File Structure

```
bubblelab-adapter/
├── probes/           # Shell probe scripts
│   ├── check_api.sh
│   ├── check_bubbles.sh
│   └── check_workflows.sh
├── src/              # TypeScript source
│   ├── adapter.ts
│   ├── bubble-client.ts
│   ├── bubblelab-canonical.ts
│   └── index.ts
├── tests/            # Contract tests
│   ├── contract.test.ts
│   └── jest.config.js
├── ADR.md            # Architecture decision record
├── README.md         # This file
└── package.json
```

## Federation Constitution Compliance

✅ **Law of Air Gap**: No imports from `core-projects/`
✅ **Law of Runtime Truth**: Probes verify API before use
✅ **Law of Untouchable DB**: No direct database writes
✅ **Law of Idempotency**: Documented idempotent operations
✅ **Law of Configuration Explicitness**: All required env vars validated
✅ **Law of UTC**: All timestamps in UTC ISO-8601 format

## Troubleshooting

### Adapter fails to start

**Problem**: Missing required environment variables

**Solution**:
```bash
export BUBBLELAB_API_URL="http://bubblelab-core:3000"
export TIMEOUT_MS="5000"
```

### Circuit breaker is OPEN

**Problem**: Too many failures to BubbleLab API

**Solution**:
1. Check BubbleLab API health: `curl ${BUBBLELAB_API_URL}/health`
2. Verify network connectivity
3. Check adapter metrics: `adapter.getMetrics()`
4. Manual reset: `adapter.resetCircuitBreaker()`

### Contract tests failing

**Problem**: BubbleLab API response structure changed

**Solution**:
1. Update contract schemas in `tests/contract.test.ts`
2. Update mapping functions in `src/bubblelab-canonical.ts`
3. Re-run contract tests: `npm run test:contract`

## References

- [Federation Constitution](../../../../CLAUDE.md)
- [Architecture Decision Record](./ADR.md)
- [BubbleLab Documentation](../../../../core-projects/BubbleLab/README.md)

## License

MIT

---

**Maintainer**: OpenEvolve Federation
**Version**: 1.0.0
**Last Updated**: 2026-02-03
