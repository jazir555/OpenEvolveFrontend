# ICR Adapter - Iterative Contextual Refinements Integration

**Version:** 1.0.0
**License:** Apache-2.0
**Compliance:** Federation Constitution v1.0

---

## Overview

The ICR Adapter provides a zero-trust integration layer for the OpenEvolve Iterative Contextual Refinements (ICR) system. This adapter implements the Anti-Corruption Layer (ACL) pattern, normalizing all 7 ICR modes into canonical schemas.

### What is ICR?

Iterative Contextual Refinements (ICR) is a powerful AI system with 7 distinct operational modes:

1. **Refine Mode** - Traditional iterative refinements with automated feature suggestion
2. **React Mode** - React application development with orchestrator-coordination
3. **Deepthink Mode** - Complex problem-solving through strategic decomposition
4. **Adaptive Deepthink Mode** - Full deepthink mode access to an agent
5. **Agentic Mode** - General-purpose iterative refinement with tool-based manipulation
6. **Contextual Mode** - Iterative refinement through specialized agent collaboration
7. **Generative UI Mode** - Interactive UI development with user interaction capture

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP (Canonical Schemas)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ICR Adapter (Glue Layer)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Canonical  │  │    Circuit   │  │     Retry    │      │
│  │   Schemas    │  │   Breaker    │  │    Logic     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP (Internal API)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ICR Core (core-projects/ICR)                    │
│           [READ ONLY - IMMUTABLE]                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration (Required)

**Federation Constitution Compliance:** All environment variables are **REQUIRED**. The adapter will **crash immediately** if any are missing (Law of Configuration Explicitness).

### Required Environment Variables

```bash
# Base URL of the ICR API (REQUIRED - no default)
export OPENEVOLVE_ICR_API_URL="http://localhost:8080"

# Request timeout in milliseconds (REQUIRED - no default)
export TIMEOUT_MS="5000"

# Maximum retry attempts (optional, default: 3)
export MAX_RETRIES="3"

# Initial retry delay in milliseconds (optional, default: 1000)
export INITIAL_RETRY_DELAY_MS="1000"

# Backoff multiplier (optional, default: 2.0)
export BACKOFF_FACTOR="2.0"

# Circuit breaker failure threshold (optional, default: 5)
export CIRCUIT_BREAKER_THRESHOLD="5"

# Circuit breaker timeout in milliseconds (optional, default: 60000)
export CIRCUIT_BREAKER_TIMEOUT_MS="60000"

# Enable jitter in retry delays (optional, default: true)
export ENABLE_JITTER="true"

# Enable debug logging (optional, default: false)
export DEBUG="true"
```

---

## Installation

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run probes (MUST pass before using adapter)
npm run probe

# Run contract tests
npm run test:contract
```

---

## Usage

### Basic Example

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

// Refine Mode
const refineResponse = await icrAdapter.createRefinementRequest(
  'Create a REST API for user management',
  {
    temperature: 0.7,
    evolution_mode: 'quality',
    refinement_stages: 3
  }
);

console.log(refineResponse.result.content);
```

### All 7 Modes

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

// 1. Refine Mode
const refine = await icrAdapter.createRefinementRequest(
  'Create a todo list application',
  {
    temperature: 0.7,
    evolution_mode: 'novelty',
    refinement_stages: 5
  }
);

// 2. React Mode
const react = await icrAdapter.createReactRequest(
  'Build a React dashboard with charts',
  {
    worker_count: 5,
    enable_preview: true,
    model_name: 'claude-3-opus'
  }
);

// 3. Deepthink Mode
const deepthink = await icrAdapter.createDeepthinkRequest(
  'Design a scalable microservices architecture',
  {
    strategy_count: 3,
    sub_strategy_count: 5,
    hypothesis_count: 10,
    enable_iterative_corrections: true,
    enable_red_team: true,
    red_team_aggressiveness: 'medium'
  }
);

// 4. Adaptive Deepthink Mode
const adaptive = await icrAdapter.createAdaptiveDeepthinkRequest(
  'Help me solve this complex problem step by step',
  {
    enable_streaming: true,
    conversation_id: 'conv-123'
  }
);

// 5. Agentic Mode
const agentic = await icrAdapter.createAgenticRequest(
  'Refactor this codebase for better performance',
  {
    enable_diff_tools: true,
    enable_file_tools: true,
    enable_web_search: false
  }
);

// 6. Contextual Mode
const contextual = await icrAdapter.createContextualRequest(
  'Write a comprehensive guide on TypeScript',
  {
    enable_memory_agent: true,
    memory_compression_threshold: 8000
  }
);

// 7. Generative UI Mode
const gui = await icrAdapter.createGenerativeUIRequest(
  'Create a modern login form with validation',
  {
    enable_interaction_capture: true,
    quality_threshold: 0.8,
    max_iterations: 5
  }
);
```

### Health Check

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

const health = await icrAdapter.healthCheck();

console.log('Status:', health.status);
console.log('Available Modes:', health.available_modes);
console.log('Uptime:', health.uptime_seconds, 'seconds');
```

### Circuit Breaker Monitoring

```typescript
import { icrAdapter } from '@openevolve/icr-adapter';

// Get circuit breaker state
const state = icrAdapter.getCircuitBreakerState();

console.log('Circuit State:', state.state);
console.log('Failure Count:', state.failureCount);

if (state.state === 'open') {
  console.log('Circuit is OPEN - requests are being blocked');
  // Implement recovery logic
  icrAdapter.resetCircuitBreaker();
}
```

---

## Canonical Schemas

The adapter enforces strict canonical schemas for all requests and responses. This is the Anti-Corruption Layer in action.

### Request Structure

All mode requests follow this structure:

```typescript
{
  mode: 'refine' | 'react' | 'deepthink' | 'adaptive_deepthink' | 'agentic' | 'contextual' | 'generative_ui',
  prompt: string,
  options?: {
    temperature?: number,
    top_p?: number,
    max_iterations?: number,
    model_name?: string,
    provider?: 'google' | 'openai' | 'anthropic',
    // Mode-specific options...
  },
  metadata: {
    correlation_id: string (UUID),
    timestamp_utc: string (ISO-8601),
    source_service: string
  }
}
```

### Response Structure

All mode responses follow this structure:

```typescript
{
  mode: 'refine' | ...,
  request: { ... },
  result: {
    success: boolean,
    content: string,
    error?: string,
    execution_time_ms: number,
    iteration_count: number,
    // Mode-specific fields...
  },
  metadata: {
    correlation_id: string,
    timestamp_utc: string,
    source_service: string,
    completed_at_utc: string
  }
}
```

---

## Error Handling

The adapter implements sophisticated error handling:

### Retry Logic

- **Transient failures** (429, 5xx): Automatic retry with exponential backoff
- **Client errors** (4xx): No retry (immediate failure)
- **Maximum retries:** Configurable via `MAX_RETRIES`

### Circuit Breaker

- **Closes** on success
- **Opens** after `CIRCUIT_BREAKER_THRESHOLD` consecutive failures
- **Half-opens** after timeout to test recovery
- **Rejects** requests when open (fast fail)

### Example

```typescript
try {
  const response = await icrAdapter.createRefinementRequest(prompt);
  console.log('Success:', response.result.content);
} catch (error) {
  if (error.message.includes('Circuit breaker is OPEN')) {
    console.error('Service is down - circuit open');
    // Implement fallback logic
  } else if (error.response?.status === 429) {
    console.error('Rate limited - too many requests');
  } else if (error.response?.status === 500) {
    console.error('Internal server error - retries exhausted');
  } else {
    console.error('Unknown error:', error.message);
  }
}
```

---

## Federation Constitution Compliance

This adapter adheres to all Federation Constitution laws:

### ✅ Law of the Air Gap (Source Code Isolation)

- **No imports** from `core-projects/` directory
- All ICR interactions via HTTP API only
- Complete isolation between Glue Layer and Core

### ✅ Law of Runtime Truth (Anti-Hallucination)

- **Probe scripts** verify API before writing code
- **Contract tests** validate API structure at runtime
- Documentation is secondary to execution

### ✅ Law of Idempotency (The Replayability Pact)

- All operations safe to retry
- Correlation ID tracking for deduplication
- No side effects from duplicate requests

### ✅ Law of Configuration Explicitness

- **REQUIRED** environment variables (no magic defaults)
- Crashes immediately if configuration is missing
- Explicit validation at startup

### ✅ Law of UTC

- All timestamps in UTC ISO-8601 format
- No local timezone conversions
- Consistent time handling across all operations

---

## Probes (Runtime Truth Verification)

Before using the adapter, run the probe scripts to verify the ICR API:

```bash
# Test basic API connectivity
./probes/check_api.sh

# Verify all 7 modes are accessible
./probes/check_modes.sh

# Test a simple refinement operation
./probes/check_refinement.sh
```

**If any probe fails, do NOT proceed with implementation.** The API contract may have changed.

---

## Contract Tests

Contract tests validate the API structure and prevent data corruption from API changes:

```bash
# Run all contract tests
npm test

# Run only contract tests
npm run test:contract
```

Tests validate:
- ✅ All 7 modes return expected response structure
- ✅ Health check returns correct format
- ✅ Error responses are handled properly
- ✅ Circuit breaker opens after failures
- ✅ Retry logic works correctly
- ✅ Idempotency is maintained
- ✅ UTC timestamps are used throughout

---

## Monitoring and Observability

### Structured Logging

The adapter outputs JSON Lines (jsonl) format:

```json
{
  "level": "info",
  "msg": "Executing ICR mode request",
  "correlation_id": "abc-123",
  "mode": "refine",
  "timestamp_utc": "2025-01-01T00:00:00.000Z",
  "source_service": "icr-adapter",
  "target_service": "icr-core"
}
```

### Correlation IDs

All requests include a correlation ID for distributed tracing:

```typescript
const correlationId = 'my-trace-id-12345';

const response = await icrAdapter.createRefinementRequest(
  prompt,
  options,
  correlationId  // Trace this request across all services
);
```

---

## Troubleshooting

### "Missing required environment variable"

**Solution:** Set all required environment variables before starting the service:

```bash
export OPENEVOLVE_ICR_API_URL="http://localhost:8080"
export TIMEOUT_MS="5000"
```

### "Circuit breaker is OPEN"

**Solution:** The ICR API is down or failing. Check the API status and reset the circuit breaker when recovered:

```typescript
const state = icrAdapter.getCircuitBreakerState();
console.log('State:', state);
console.log('Failures:', state.failureCount);
console.log('Last failure:', new Date(state.lastFailureTime!));

// When API is recovered:
icrAdapter.resetCircuitBreaker();
```

### "Unexpected response structure"

**Solution:** The API contract may have changed. Run contract tests:

```bash
npm run test:contract
```

Update canonical schemas if contract has changed.

---

## API Reference

### ICRAdapter

#### Methods

- `createRefinementRequest(prompt, options?, correlationId?)` - Refine mode
- `createReactRequest(prompt, options?, correlationId?)` - React mode
- `createDeepthinkRequest(prompt, options?, correlationId?)` - Deepthink mode
- `createAdaptiveDeepthinkRequest(prompt, options?, correlationId?)` - Adaptive Deepthink mode
- `createAgenticRequest(prompt, options?, correlationId?)` - Agentic mode
- `createContextualRequest(prompt, options?, correlationId?)` - Contextual mode
- `createGenerativeUIRequest(prompt, options?, correlationId?)` - Generative UI mode
- `healthCheck(correlationId?)` - Health check
- `getCircuitBreakerState()` - Get circuit breaker state
- `resetCircuitBreaker()` - Reset circuit breaker

### ICRClient

#### Methods

- `executeMode(request, correlationId?)` - Execute a mode request
- `healthCheck(request?, correlationId?)` - Health check
- `getCircuitBreakerState()` - Get circuit breaker state
- `resetCircuitBreaker()` - Reset circuit breaker

---

## Contributing

When modifying this adapter:

1. **Run probes first** - Verify API behavior
2. **Update schemas** - If contract changed, update canonical schemas
3. **Update tests** - Add/modify contract tests
4. **Test locally** - Run all tests and probes
5. **Update docs** - Document any API changes

**Never skip the probes.** The Federation Constitution demands runtime truth verification.

---

## License

Apache-2.0

---

## Support

For issues or questions:

1. Check probe results
2. Review contract test output
3. Check circuit breaker state
4. Review structured logs with correlation ID

---

**Remember:** You are building a skyscraper on top of moving tectonic plates. Flexibility is fatal. Rigidity in architecture is a necessity.
