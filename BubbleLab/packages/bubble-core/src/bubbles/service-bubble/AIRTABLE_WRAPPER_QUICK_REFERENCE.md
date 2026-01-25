# Airtable Wrapper Service - Quick Reference

## Overview

OpenEvolve-specific Airtable wrapper with comprehensive resilience patterns. This is a production-ready implementation wrapping the core Airtable functionality with enterprise-grade reliability features.

## Location

`BubbleLab/packages/bubble-core/src/bubbles/service-bubble/airtable-wrapper.ts`

## Features

### 12 Complete Operations

#### Table Operations (8)
1. **listRecords** - List records with pagination, filtering, sorting
2. **getRecord** - Get a specific record by ID
3. **createRecord** - Create a new record
4. **updateRecord** - Update an existing record
5. **deleteRecord** - Delete a record
6. **batchCreate** - Create up to 10 records
7. **batchUpdate** - Update up to 10 records
8. **batchDelete** - Delete up to 10 records

#### Query Operations (2)
9. **queryRecords** - Query with formula filters
10. **searchRecords** - Full-text search across fields

#### Metadata Operations (2)
11. **getSchema** - Get table schema and field definitions
12. **listTables** - List all tables in a base

### Resilience Patterns

#### Circuit Breaker
- **Threshold**: Opens after 5 consecutive failures
- **Timeout**: 60 seconds before attempting recovery
- **Half-Open State**: 3 successful attempts to close circuit
- **State Tracking**: CLOSED → OPEN → HALF_OPEN → CLOSED

#### Retry Logic
- **Strategy**: Exponential backoff with jitter
- **Delays**: 1s, 2s, 4s, 8s, 16s
- **Max Retries**: 3 attempts
- **Retry On**: 429, 500, 502, 503, 504
- **No Retry On**: 400, 401, 403, 404

#### Rate Limiting
- **Limit**: 5 requests per second per Airtable base
- **Implementation**: Token bucket algorithm
- **Enforcement**: Automatic with clear error messages
- **Headers**: Respects `Retry-After` header

#### Request Deduplication
- **In-Flight Requests**: Deduplicates concurrent identical requests
- **Result Caching**: 60-second TTL for successful responses
- **Cache Key**: Based on operation + base + timestamp
- **Memory Management**: Automatic cleanup of expired entries

#### Dead Letter Queue
- **Purpose**: Captures permanent failures for manual inspection
- **Max Size**: 1000 entries (FIFO eviction)
- **Metadata**: Operation, input, error, timestamp, retry count
- **Access**: Programmatic retrieval and clearing

### Security Features

#### Input Validation
- **Base ID**: Must start with `app` followed by alphanumeric characters
- **Record ID**: Must start with `rec` followed by alphanumeric characters
- **Field ID**: Must start with `fld` followed by alphanumeric characters
- **Table Names**: 1-255 characters, validated via Zod schemas

#### Authentication
- **Method**: Bearer token (Personal Access Token)
- **Validation**: Token format checked before API calls
- **Storage**: Credentials injected at runtime via CredentialType enum

#### Error Sanitization
- **Stack Traces**: Removed from all error messages
- **File Paths**: Replaced with `[internal]`
- **Secrets**: Redacted using pattern matching (password, token, key, secret)
- **User Messages**: Safe to display to end users

### Structured Logging

#### Format
```json
{
  "timestamp": "2024-01-01T00:00:00.000Z",
  "level": "info",
  "service": "airtable-client",
  "correlationId": "abc123...",
  "msg": "Airtable API request",
  "operation": "listRecords",
  "baseId": "app..."
}
```

#### Correlation IDs
- **Generation**: Cryptographically random (16 bytes → 32 hex chars)
- **Scope**: Per-operation for tracing
- **Propagation**: Through all log entries

## Usage Examples

### Basic Usage

```typescript
import { AirtableWrapperBubble } from './airtable-wrapper.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// List records
const bubble = new AirtableWrapperBubble({
  operation: 'listRecords',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  maxRecords: 50,
  fields: ['Name', 'Email', 'Status'],
  sort: [{ field: 'Name', direction: 'asc' }],
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await bubble.performAction();
if (result.result.success) {
  console.log('Records:', result.result.records);
  console.log('Count:', result.result.count);
  console.log('Offset:', result.result.offset);
} else {
  console.error('Error:', result.result.error);
}
```

### Create Record

```typescript
const createBubble = new AirtableWrapperBubble({
  operation: 'createRecord',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  fields: {
    Name: 'John Doe',
    Email: 'john@example.com',
    Status: 'Active'
  },
  typecast: true,  // Auto-convert field types
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await createBubble.performAction();
```

### Batch Update

```typescript
const batchBubble = new AirtableWrapperBubble({
  operation: 'batchUpdate',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  records: [
    { id: 'rec1', fields: { Status: 'Updated' } },
    { id: 'rec2', fields: { Status: 'Updated' } },
  ],
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await batchBubble.performAction();
console.log('Updated:', result.result.count);
```

### Query with Formula

```typescript
const queryBubble = new AirtableWrapperBubble({
  operation: 'queryRecords',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  filterByFormula: '{Status} = "Active" AND {Created} > TODAY() - 30',
  maxRecords: 100,
  sort: [{ field: 'Created', direction: 'desc' }],
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await queryBubble.performAction();
```

### Search Records

```typescript
const searchBubble = new AirtableWrapperBubble({
  operation: 'searchRecords',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  searchString: 'John',
  fields: ['Name', 'Email', 'Notes'],  // Search only these fields
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await searchBubble.performAction();
```

### Get Table Schema

```typescript
const schemaBubble = new AirtableWrapperBubble({
  operation: 'getSchema',
  baseId: 'appYourBaseId',
  tableId: 'tblYourTableId',
  credentials: {
    [CredentialType.AIRTABLE_CRED]: 'patYourApiKey...'
  }
});

const result = await schemaBubble.performAction();
if (result.result.success) {
  console.log('Table:', result.result.name);
  console.log('Fields:', result.result.fields);
  console.log('Primary Field:', result.result.primaryFieldId);
}
```

## Monitoring & Debugging

### Circuit Breaker Status

```typescript
const bubble = new AirtableWrapperBubble({ ... });

// Get current state
const state = bubble.getCircuitBreakerState();
console.log('Circuit state:', state);  // 'closed' | 'open' | 'half_open'

// Get detailed stats
const stats = bubble.getCircuitBreakerStats();
console.log('Failures:', stats.failureCount);
console.log('Successes:', stats.successCount);
console.log('Last failure:', new Date(stats.lastFailureTime));

// Reset manually if needed
await bubble.resetCircuitBreaker();
```

### Dead Letter Queue

```typescript
// Get failed operations
const dlqEntries = bubble.getDeadLetterEntries();
dlqEntries.forEach(entry => {
  console.error('Failed operation:', entry.operation);
  console.error('Error:', entry.error.message);
  console.error('Timestamp:', new Date(entry.timestamp));
  console.error('Retries:', entry.retryCount);
});

// Clear queue after processing
bubble.clearDeadLetterQueue();
```

### Deduplication Stats

```typescript
const stats = bubble.getDeduplicatorStats();
console.log('Pending requests:', stats.pendingRequests);
console.log('Cached results:', stats.completedRequests);
```

## Error Handling

### Common Errors

#### Rate Limit Exceeded (429)
```
RATE_LIMIT_EXCEEDED: Airtable rate limit exceeded.
Retry after: 5 seconds. Max 5 requests/sec per base.
```
**Solution**: Implement client-side throttling or wait for Retry-After duration.

#### Invalid Base ID
```
Invalid Airtable base ID format (must start with app)
```
**Solution**: Ensure base ID starts with `app` followed by alphanumeric characters.

#### Circuit Breaker Open
```
Circuit breaker is OPEN for operation: listRecords.
Last failure: 2024-01-01T00:00:00.000Z.
Retry after: 2024-01-01T00:01:00.000Z
```
**Solution**: Wait for timeout duration or manually reset circuit breaker.

#### Authentication Failed (401)
```
AIRTABLE_API_ERROR: 401 Unauthorized - Invalid authentication
```
**Solution**: Verify API key is valid and has required scopes.

#### Table Not Found (404)
```
AIRTABLE_API_ERROR: 404 Not Found - Table not found
```
**Solution**: Verify base ID and table ID are correct and accessible.

### Testing Credentials

```typescript
const bubble = new AirtableWrapperBubble({ ... });
const isValid = await bubble.testCredential();

if (!isValid) {
  console.error('Invalid Airtable API key');
}
```

## Best Practices

### 1. Pagination
Always handle pagination for large datasets:
```typescript
let allRecords = [];
let offset = null;

do {
  const result = await bubble.performAction({
    ...params,
    offset: offset || undefined
  });

  allRecords = allRecords.concat(result.result.records);
  offset = result.result.offset;
} while (offset);
```

### 2. Batch Operations
Use batch operations for efficiency:
```typescript
// GOOD: 10 records in 1 request
await batchCreate({ records: [...10 records] });

// BAD: 10 requests
for (const record of records) {
  await createRecord({ fields: record });
}
```

### 3. Error Recovery
Implement circuit breaker monitoring:
```typescript
const state = bubble.getCircuitBreakerState();
if (state === 'open') {
  // Use cached data or fallback
  return getCachedData();
}
```

### 4. Field Selection
Specify only required fields for better performance:
```typescript
// GOOD: Only fetch needed fields
fields: ['Name', 'Email']

// BAD: Fetch all fields
// Don't specify fields parameter
```

### 5. Type Safety
Leverage TypeScript discriminated unions:
```typescript
type Result = AirtableWrapperResult;
// TypeScript knows exact result type based on operation
```

## Performance Considerations

### Rate Limits
- **Per Base**: 5 requests/second
- **Per Workspace**: Varies by plan
- **Batch Operations**: Count as 1 request

### Timeouts
- **GET Requests**: 30 seconds
- **POST/PATCH**: 60 seconds
- **DELETE**: 30 seconds

### Recommendations
1. **Cache Frequently Accessed Data**: Use built-in deduplication
2. **Batch When Possible**: Reduces request count
3. **Filter Early**: Use Airtable formulas instead of client-side filtering
4. **Monitor Circuit Breaker**: Prevent cascading failures
5. **Handle DLQ**: Process failed operations asynchronously

## Testing

Run the test suite:
```bash
cd BubbleLab/packages/bubble-core
npm test -- airtable-wrapper.test.ts
```

Test coverage includes:
- All 12 operations
- Circuit breaker behavior
- Retry logic
- Rate limiting
- Input validation
- Error handling
- Credential testing

## Migration from Core Airtable

The wrapper is fully compatible with the core Airtable implementation but adds resilience patterns. To migrate:

```typescript
// Before (core)
import { AirtableBubble } from './airtable.js';

// After (wrapper)
import { AirtableWrapperBubble } from './airtable-wrapper.js';

// API is identical - just change the class name
```

## Support

For issues or questions:
1. Check circuit breaker status
2. Review dead letter queue
3. Verify API credentials
4. Check rate limits
5. Review structured logs with correlation ID

## Summary

The Airtable Wrapper provides enterprise-grade resilience patterns while maintaining a simple, clean API. All 12 operations are protected by circuit breakers, retry logic, rate limiting, and comprehensive error handling.

**Key Benefits:**
- ✅ Production-ready reliability
- ✅ Automatic fault tolerance
- ✅ Comprehensive logging
- ✅ Security best practices
- ✅ Type-safe operations
- ✅ Easy debugging
- ✅ Graceful degradation
