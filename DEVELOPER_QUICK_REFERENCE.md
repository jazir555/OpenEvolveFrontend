# Code Quality Utilities - Quick Reference Guide

## Overview
This guide provides quick reference for the new code quality utilities added to improve logging, error handling, and timestamp consistency across the BubbleLab application.

---

## 📝 Structured Logging

### Import
```typescript
import { logger, createLogger } from '../utils/logger';
```

### Usage Examples

**Basic logging**:
```typescript
logger.info({
  msg: 'User logged in',
  user_id: '12345',
  username: 'john_doe',
});

logger.error({
  msg: 'Database connection failed',
  error: err.message,
  host: 'db.example.com',
  port: 5432,
});
```

**With correlation ID**:
```typescript
const correlationId = generateCorrelationId();

logger.info({
  msg: 'Processing payment',
  correlation_id: correlationId,
  payment_id: 'pay_123',
  amount: 99.99,
});
```

**Create custom logger**:
```typescript
const apiLogger = createLogger('payment-service');

apiLogger.info({
  msg: 'Payment processed',
  transaction_id: 'txn_abc123',
});
```

**Log levels**:
```typescript
logger.debug({ msg: 'Debug info' });     // Development only
logger.info({ msg: 'Informational' });    // General info
logger.warn({ msg: 'Warning occurred' }); // Warnings
logger.error({ msg: 'Error occurred' });  // Errors
```

**Log output format**:
```json
{
  "level": "info",
  "timestamp": "2025-01-19T12:00:00.000Z",
  "source_service": "bubble-studio",
  "msg": "User logged in",
  "user_id": "12345",
  "username": "john_doe"
}
```

---

## ❌ Error Handling

### Import
```typescript
import {
  NetworkError,
  AuthenticationError,
  ValidationError,
  ServerError,
  NotFoundError,
  RateLimitError,
  getErrorCode,
  getErrorMessage,
  getErrorCorrelationId,
  isCustomError,
} from '../lib/errors';
```

### Creating Errors

**Basic error**:
```typescript
throw new NetworkError('Failed to connect to server', correlationId);
```

**Error with details**:
```typescript
throw new ValidationError('Invalid email format', correlationId, {
  field: 'email',
  value: 'not-an-email',
  expected_format: 'user@example.com',
});
```

**Server error with status code**:
```typescript
throw new ServerError('Database query failed', 500, correlationId, {
  query: 'SELECT * FROM users',
  database: 'postgres',
});
```

**Rate limit error**:
```typescript
throw new RateLimitError(
  'Too many requests',
  60,  // retry after 60 seconds
  correlationId
);
```

### Handling Errors

**Specific error handling**:
```typescript
try {
  await api.post('/payments', data);
} catch (error) {
  if (error instanceof NetworkError) {
    console.log('Network issue, correlation:', error.correlationId);
    // Show retry button
  } else if (error instanceof AuthenticationError) {
    // Redirect to login
    redirectToLogin();
  } else if (error instanceof ValidationError) {
    // Show form errors
    displayFieldErrors(error.details);
  } else if (error instanceof RateLimitError) {
    // Show countdown
    showRetryCountdown(error.retryAfter);
  }
}
```

**Generic error handling**:
```typescript
try {
  await api.get('/data');
} catch (error) {
  const code = getErrorCode(error);
  const message = getErrorMessage(error);
  const correlationId = getErrorCorrelationId(error);

  console.log(`Error ${code}: ${message}`);
  if (correlationId) {
    console.log(`Reference ID: ${correlationId}`);
  }
}
```

**Type guard**:
```typescript
if (isCustomError(error)) {
  // error is guaranteed to be BaseError or subclass
  console.log(error.code);
  console.log(error.correlationId);
}
```

### Error Codes
- `NETWORK_ERROR` - Network connectivity issues
- `AUTHENTICATION_ERROR` - Authentication failed (401)
- `AUTHORIZATION_ERROR` - Insufficient permissions (403)
- `VALIDATION_ERROR` - Input validation failed (400)
- `SERVER_ERROR` - Server-side error (5xx)
- `NOT_FOUND_ERROR` - Resource not found (404)
- `RATE_LIMIT_ERROR` - Rate limit exceeded (429)
- `GENERIC_ERROR` - Unhandled error
- `UNKNOWN_ERROR` - Unknown error type

---

## 🕐 Timestamp Utilities

### Import
```typescript
import {
  getCurrentTimestamp,
  toUtcISO,
  isValidUtcISO,
  getCurrentTimeMs,
  calculateDuration,
  addDuration,
} from '../utils/timestamp';
```

### Current Timestamp

**Get current UTC timestamp**:
```typescript
const now = getCurrentTimestamp();
// "2025-01-19T12:00:00.000Z"
```

**Use in database records**:
```typescript
const user = await db.insert({
  name: 'John Doe',
  created_at: getCurrentTimestamp(),
  updated_at: getCurrentTimestamp(),
});
```

### Timestamp Conversion

**Convert any date format to UTC ISO-8601**:
```typescript
const fromDate = toUtcISO(new Date());
// "2025-01-19T12:00:00.000Z"

const fromMs = toUtcISO(1705689600000);
// "2025-01-19T12:00:00.000Z"

const fromString = toUtcISO("2025-01-19T12:00:00.000Z");
// "2025-01-19T12:00:00.000Z"
```

### Validation

**Validate timestamp format**:
```typescript
if (isValidUtcISO(userInput)) {
  // Safe to use
  const timestamp = toUtcISO(userInput);
} else {
  throw new ValidationError('Invalid timestamp format');
}
```

### Duration Calculations

**Measure operation duration**:
```typescript
const start = getCurrentTimestamp();
// ... do work ...
const durationMs = calculateDuration(start);
console.log(`Operation took ${durationMs}ms`);
```

**With explicit end time**:
```typescript
const start = getCurrentTimestamp();
// ... work ...
const end = getCurrentTimestamp();
const duration = calculateDuration(start, end);
```

### Performance Measurements

**For performance timing (use Date.now() alternative)**:
```typescript
const start = getCurrentTimeMs(); // Returns milliseconds since epoch
// ... do work ...
const duration = getCurrentTimeMs() - start;
console.log(`Took ${duration}ms`);
```

**Note**: Use `getCurrentTimeMs()` only for performance measurements. Use `getCurrentTimestamp()` for timestamps that will be stored or transmitted.

### Date Arithmetic

**Add duration to timestamp**:
```typescript
const now = getCurrentTimestamp();
const inOneHour = addDuration(now, 60 * 60 * 1000); // Add 1 hour
const tomorrow = addDuration(now, 24 * 60 * 60 * 1000); // Add 1 day
```

---

## 🔗 API Client Usage

### Import
```typescript
import { api } from '../lib/api';
```

### Basic Requests

**GET request**:
```typescript
const data = await api.get('/api/users');
```

**POST request**:
```typescript
const newUser = await api.post('/api/users', {
  name: 'John Doe',
  email: 'john@example.com',
});
```

**PUT/PATCH**:
```typescript
const updated = await api.put('/api/users/123', { name: 'Jane Doe' });
const patched = await api.patch('/api/users/123', { email: 'jane@example.com' });
```

**DELETE**:
```typescript
await api.delete('/api/users/123');
```

### Error Handling

**All API requests include correlation IDs automatically**:
```typescript
try {
  const result = await api.post('/api/payments', paymentData);
  console.log('Payment successful');
} catch (error) {
  if (error instanceof NetworkError) {
    console.log('Network error:', error.message);
    console.log('Correlation ID:', error.correlationId);
    // Use correlation ID to trace the request in logs
  }
}
```

### Streaming Requests

**Server-Sent Events**:
```typescript
const response = await api.postStream('/api/stream', { query: 'test' });

const reader = response.body?.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader!.read();
  if (done) break;

  const text = decoder.decode(value);
  console.log('Received:', text);
}
```

---

## 🎯 Best Practices

### Logging

**DO**:
```typescript
logger.info({
  msg: 'Clear, descriptive message',
  correlation_id: correlationId,
  relevant_context: 'value',
});
```

**DON'T**:
```typescript
console.log('Making request to', endpoint);  // ❌ No structure
console.error('Error:', error);               // ❌ No context
```

### Error Handling

**DO**:
```typescript
throw new NetworkError(
  'Failed to connect to payment gateway',
  correlationId,
  { gateway: 'stripe', amount: 99.99 }
);
```

**DON'T**:
```typescript
throw new Error('Payment failed');  // ❌ Generic error
```

### Timestamps

**DO**:
```typescript
const timestamp = getCurrentTimestamp();  // ✅ UTC ISO-8601
user.created_at = timestamp;
```

**DON'T**:
```typescript
user.created_at = new Date().toString();      // ❌ Local time
user.created_at = Date.now();                 // ❌ Milliseconds
user.created_at = new Date().toISOString();   // ⚠️ Works, but use utility
```

### Performance Timing

**DO**:
```typescript
const start = getCurrentTimeMs();  // For performance only
// ... work ...
const duration = getCurrentTimeMs() - start;
```

**DON'T**:
```typescript
const start = Date.now();  // ⚠️ Works, but use utility for consistency
```

---

## 📊 Correlation ID Tracking

### Generate Correlation ID

```typescript
import { generateCorrelationId } from '../lib/api';

// Or generate manually
function generateCorrelationId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
```

### Use Throughout Request

```typescript
const correlationId = generateCorrelationId();

// Log with correlation ID
logger.info({
  msg: 'Starting operation',
  correlation_id: correlationId,
});

// Pass to API (automatically done by API client)
await api.post('/api/endpoint', data);

// Include in errors
throw new NetworkError('Operation failed', correlationId);
```

---

## 🔍 Debugging Tips

### Enable Debug Logging

```typescript
// Set NODE_ENV=development to see debug logs
logger.debug({
  msg: 'Detailed debug info',
  variable_state: { complex: 'object' },
});
```

### Trace Requests

1. Get correlation ID from error or logs
2. Search logs for that correlation ID
3. See entire request journey across services

**Example**:
```bash
# Search logs for correlation ID
grep "550e8400-e29b-41d4-a716-446655440000" logs/*.log
```

### Validate Timestamps

```typescript
// Before using user input
if (!isValidUtcISO(userTimestamp)) {
  throw new ValidationError('Invalid timestamp format');
}

// Convert to standard format
const standardTimestamp = toUtcISO(userTimestamp);
```

---

## 📚 Additional Resources

- **Full Documentation**: See `CODE_QUALITY_FIXES_SUMMARY.md`
- **CLAUDE.md**: Federation Constitution guidelines
- **Type Definitions**: See source files for complete type definitions

---

## 🚀 Quick Start Template

```typescript
import { logger } from '../utils/logger';
import { getCurrentTimestamp, calculateDuration } from '../utils/timestamp';
import { NetworkError, ValidationError } from '../lib/errors';

async function processPayment(paymentData: PaymentData) {
  const correlationId = generateCorrelationId();
  const startTime = getCurrentTimestamp();

  logger.info({
    msg: 'Processing payment',
    correlation_id: correlationId,
    amount: paymentData.amount,
  });

  try {
    // Validate input
    if (!paymentData.amount || paymentData.amount <= 0) {
      throw new ValidationError('Invalid payment amount', correlationId);
    }

    // Process payment
    const result = await paymentGateway.charge(paymentData);

    const duration = calculateDuration(startTime);

    logger.info({
      msg: 'Payment processed successfully',
      correlation_id: correlationId,
      payment_id: result.id,
      duration_ms: duration,
    });

    return result;

  } catch (error) {
    const duration = calculateDuration(startTime);

    logger.error({
      msg: 'Payment processing failed',
      correlation_id: correlationId,
      error: error instanceof Error ? error.message : String(error),
      duration_ms: duration,
    });

    throw error;
  }
}
```

---

**Remember**: All timestamps in UTC ISO-8601 format per CLAUDE.md LAW OF UTC! 🌍
