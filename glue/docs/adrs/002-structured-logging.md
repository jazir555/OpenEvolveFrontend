# ADR-002: Structured Logging with Correlation IDs

## Status
**Accepted**

## Context
The OpenEvolve Frontend orchestrates 30+ services with complex request flows:
- User action → Frontend → OpenEvolve API → RAGBits → Datapizza
- Multiple async operations in parallel
- Errors can occur at any level
- Need to trace requests across services for debugging

## Problem
Traditional logging (`console.log()`, `console.error()`) has several issues:

1. **Unstructured**: No consistent format, hard to parse/analyze
2. **No tracing**: Can't follow a single request across multiple services
3. **Lost context**: Error logs don't include request details (URL, params, etc.)
4. **Not searchable**: Can't easily filter logs by service/user/operation
5. **Timezone issues**: Local timestamps make cross-timezone debugging hard

## Decision
Implement **structured logging with correlation IDs** following Federation Constitution laws:

### Implementation

#### Log Format
All logs are **JSON Lines** (one JSON object per line):
```json
{
  "level": "error",
  "msg": "API request failed",
  "timestamp": "2026-02-15T10:30:45.123Z",
  "correlation_id": "openevolve-1234567890-abc",
  "source_service": "openevolve-plugin",
  "target_service": "openevolve-api",
  "operation": "search_evolutions",
  "status": 500,
  "duration_ms": 1234
}
```

#### Correlation IDs
- Generated per client instance: `{service}-{timestamp}-{random}`
- Passed in all HTTP requests (via headers or query params)
- Included in all log statements
- Enables distributed tracing across services

#### Required Fields
Every log MUST include:
- `level`: log level (error, warn, info, debug)
- `msg`: human-readable message
- `timestamp`: UTC ISO-8601 timestamp
- `correlation_id`: request tracking ID
- `source_service`: which service/client made the log

#### Contextual Fields (when applicable)
- `target_service`: which external service was called
- `operation`: what operation was performed
- `status`: HTTP status code or error code
- `duration_ms`: operation duration
- `error`: error message/stack trace

### Code Location
- `glue/lib/structuredLogger.ts` - Logger implementation
- `glue/lib/logger.ts` - Simplified logger wrapper

### Example Usage
```typescript
import { logger, LogContext } from './structuredLogger';

const context: LogContext = {
  correlation_id: 'openevolve-1234567890-abc',
  source_service: 'openevolve-plugin',
  target_service: 'openevolve-api',
  operation: 'search_evolutions'
};

logger.info('Starting search', context);
logger.error('Search failed', error, context);
```

## Consequences

### Positive
- ✅ **Searchable**: Can filter logs by any field (correlation_id, service, etc.)
- ✅ **Traceable**: Follow requests across services using correlation_id
- ✅ **Timezone-safe**: All timestamps in UTC
- ✅ **Tool-friendly**: JSON Lines works with grep, jq, ELK, CloudWatch, etc.
- ✅ **Debuggable**: Full context in every log entry

### Negative
- ⚠️ **Verbose**: JSON logs are longer than plain text
- ⚠️ **Requires parsing**: Can't read as easily without JSON tools
- ⚠️ **Disk space**: More disk space for log files

### Mitigations
- Use log levels appropriately (don't log everything at DEBUG in production)
- Use log aggregation tools (ELK, CloudWatch, Datadog)
- Implement log rotation/compression

## Alternatives Considered

### Alternative 1: Plain Text Logging
**Description**: Continue using `console.log()` with string messages

**Pros**: Simple, human-readable

**Cons**: Unstructured, hard to parse, no correlation, not searchable

**Rejected**: Violates Federation Constitution observability requirements

### Alternative 2: External Logging Library (Winston, Pino)
**Description**: Use established logging library

**Pros**: Battle-tested, feature-rich

**Cons**: Additional dependency, heavier weight

**Rejected**: Our simple logger is sufficient for our needs

### Alternative 3: Cloud Logging Service (Sentry, LogRocket)
**Description**: Send logs directly to SaaS service

**Pros**: No local storage, built-in UI

**Cons**: Vendor lock-in, cost, requires network

**Rejected**: Want local logs first, can add remote logging later

## Related Decisions
- [ADR-001: Circuit Breaker Pattern](./001-circuit-breaker.md)
- [ADR-003: Retry Logic with Exponential Backoff](./003-retry-logic.md)

## Implementation Date
2026-02-15

## Author
OpenEvolve Federation Team
