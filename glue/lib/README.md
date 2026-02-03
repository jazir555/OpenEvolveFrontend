# Glue Layer Shared Utilities

Federation Constitution compliant utilities for the OpenEvolve Glue Layer.

## Overview

This library provides core utilities that enforce the Federation Constitution's laws:

- **Law of UTC**: All timestamps in UTC ISO-8601 format
- **Law of Configuration Explicitness**: No magic defaults, crash on missing config
- **Failure Management**: Exponential backoff and circuit breaker patterns
- **Observability**: Structured JSON Lines logging with correlation tracking

## Modules

### 1. Logger (`logger.ts`)

Structured JSON Lines logger with UTC timestamps and automatic correlation IDs.

```typescript
import { logger } from './lib';

// Basic logging
logger.info('User Sync Started', {
  source_service: 'crm-adapter',
  target_service: 'user-service',
  user_id: '12345',
});

// Error logging
logger.error('User Sync Failed', error, {
  correlation_id: 'evt-abc-123',
  retry_count: 2,
});

// Output:
// {"level":"info","msg":"User Sync Started","timestamp":"2025-01-15T10:30:00.000Z","correlation_id":"a1b2c3d4-...","source_service":"crm-adapter","target_service":"user-service","user_id":"12345"}
```

**Features:**
- JSON Lines output (one JSON object per line)
- Auto-generated correlation IDs (UUID v4)
- UTC timestamps (ISO-8601)
- Structured context support
- Child logger with preset context

### 2. Retry (`retry.ts`)

Exponential backoff with jitter for transient failures.

```typescript
import { retryWithBackoff } from './lib';

const result = await retryWithBackoff(
  async () => {
    const response = await fetch('http://service:8000/api');
    if (!response.ok) throw new Error('HTTP error');
    return response.json();
  },
  {
    max_retries: 5,
    base_delay_ms: 1000,
    max_delay_ms: 30000,
    jitter_ms: 500,
    onRetry: (attempt, error) => {
      logger.warn('Retry attempt', { attempt, error_message: error.message });
    },
  }
);
```

**Features:**
- Exponential backoff: `base_delay * 2^attempt`
- Random jitter to prevent thundering herd
- Configurable max delay cap
- Retry callback for observability
- Throws last error after all retries exhausted

### 3. Circuit Breaker (`circuit-breaker.ts`)

Circuit breaker pattern to prevent cascading failures.

```typescript
import { CircuitBreaker, CircuitState } from './lib';

const cb = new CircuitBreaker({
  threshold: 5,           // Trip after 5 failures
  timeout_ms: 60000,      // Stay open for 1 minute
  onStateChange: (old, newState) => {
    logger.info('Circuit state changed', { old_state: old, new_state: newState });
  },
});

try {
  const result = await cb.execute(async () => {
    return await externalApiCall();
  });
} catch (error) {
  if (cb.getState() === CircuitState.OPEN) {
    // Use fallback or cached data
    logger.error('Service is down, circuit is open', error);
  }
}
```

**States:**
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Circuit tripped, requests fail immediately
- `HALF_OPEN`: Testing if service has recovered

**Features:**
- Configurable failure threshold
- Auto-reset after timeout
- State change callbacks
- Statistics tracking

### 4. Environment Validator (`env-validator.ts`)

Validate environment variables with type checking. Crashes immediately if required vars are missing.

```typescript
import { validateEnv, validateEnvWithTypes, getEnv } from './lib';

// Simple validation
validateEnv(['DATABASE_URL', 'API_KEY', 'SERVICE_PORT']);

// Validation with type checking
const config = validateEnvWithTypes([
  { name: 'DATABASE_URL', type: 'url', required: true },
  { name: 'SERVICE_PORT', type: 'port', required: true },
  { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'DEBUG_MODE', type: 'boolean', required: false, default: false },
]);

// Get single variable
const dbUrl = getEnv('DATABASE_URL', 'url');
const port = getEnv('SERVICE_PORT', 'port');
```

**Supported Types:**
- `string`: Plain string
- `number`: Numeric value
- `boolean`: true/false or 1/0
- `url`: Valid URL format
- `port`: Valid port (1-65535)

**Features:**
- Type validation with clear error messages
- Optional defaults for non-required vars
- Crashes immediately on validation failure
- No magic defaults (Law of Configuration Explicitness)

## Complete Example

Combining all utilities for a production adapter:

```typescript
import { logger, retryWithBackoff, CircuitBreaker, validateEnvWithTypes } from './lib';

// 1. Validate environment at startup (crashes if invalid)
const config = validateEnvWithTypes([
  { name: 'TARGET_API_URL', type: 'url', required: true },
  { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
  { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'CIRCUIT_THRESHOLD', type: 'number', required: false, default: 5 },
]);

// 2. Create circuit breaker
const circuitBreaker = new CircuitBreaker({
  threshold: config.CIRCUIT_THRESHOLD,
  timeout_ms: 60000,
  onStateChange: (old, newState) => {
    logger.warn('Circuit breaker state changed', {
      old_state: old,
      new_state: newState,
    });
  },
});

// 3. Make resilient API calls
async function callExternalService(data: any) {
  const correlationId = generateCorrelationId();

  return retryWithBackoff(
    async () => {
      logger.info('Calling external service', {
        correlation_id: correlationId,
        target_service: 'external-api',
      });

      return circuitBreaker.execute(async () => {
        const response = await fetch(config.TARGET_API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
          signal: AbortSignal.timeout(config.TIMEOUT_MS),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return response.json();
      });
    },
    {
      max_retries: config.MAX_RETRIES,
      base_delay_ms: 1000,
      max_delay_ms: 10000,
      jitter_ms: 500,
      onRetry: (attempt, error) => {
        logger.warn('Retry attempt', {
          correlation_id: correlationId,
          attempt,
          error_message: error.message,
        });
      },
    }
  );
}
```

## Installation

```bash
npm install
```

## Building

```bash
npm run build
```

## Testing

```bash
npm test
```

## Federation Constitution Compliance

This library enforces the following laws:

1. **Law of UTC**: All timestamps are UTC ISO-8601
2. **Law of Configuration Explicitness**: Crashes on missing env vars, no magic defaults
3. **Failure Management**:
   - Transient failures: Exponential backoff with jitter
   - System failures: Circuit breaker pattern
   - Logic failures: Should go to Dead Letter Queue (not handled here)
4. **Observability**: Structured logging with correlation tracking

## License

MIT
