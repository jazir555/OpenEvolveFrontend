# HTTP Service Bubble - Quick Reference

## Import

```typescript
import { HttpBubble } from '@bubblelab/bubble-core';
```

## Basic Usage

### GET Request
```typescript
const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
});

const result = await httpBubble.performAction();
console.log(result.data); // Response data
```

### POST Request
```typescript
const httpBubble = new HttpBubble({
  operation: 'post',
  url: 'https://api.example.com/users',
  body: { name: 'John', email: 'john@example.com' },
});

const result = await httpBubble.performAction();
```

## Quick Configuration

### With Retry
```typescript
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  retryEnabled: true,
  maxRetries: 3,
  retryStrategy: 'exponential', // or 'linear'
})
```

### With Circuit Breaker
```typescript
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  circuitBreakerEnabled: true,
  circuitBreakerThreshold: 5,
  circuitBreakerTimeout: 60000,
})
```

### With Query Parameters
```typescript
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/search',
  queryParams: { q: 'search', page: 1, limit: 20 },
})
```

### With Authentication
```typescript
// Bearer token
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  authType: 'bearer',
  credentials: { [CredentialType.CUSTOM_AUTH_KEY]: 'your-token' },
})

// API Key
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  authType: 'api-key',
  credentials: { [CredentialType.CUSTOM_AUTH_KEY]: 'your-key' },
})
```

### With Timeout
```typescript
new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  timeout: 10000, // 10 seconds
})
```

## Response Structure

```typescript
{
  success: boolean,
  data: unknown,           // Parsed response data
  status: number,         // HTTP status code
  statusText: string,     // HTTP status text
  headers: Record<string, string>,
  body: string,           // Raw response body
  error?: string,         // Error message if failed
  metrics: {
    totalAttempts: number,
    responseTime: number,
    lastAttemptTime: number,
    retryCount: number,
  },
}
```

## Error Handling

```typescript
const result = await httpBubble.performAction();

if (result.success) {
  console.log('Success:', result.data);
} else if (result.metrics.circuitBreakerTripped) {
  console.log('Circuit breaker open - use fallback');
} else if (result.status >= 500) {
  console.log('Server error - retry later');
} else if (result.status >= 400) {
  console.log('Client error - check request');
} else {
  console.log('Network error:', result.error);
}
```

## All Operations

```typescript
// GET
new HttpBubble({ operation: 'get', url: '...' })

// POST
new HttpBubble({ operation: 'post', url: '...', body: {...} })

// PUT
new HttpBubble({ operation: 'put', url: '...', body: {...} })

// PATCH
new HttpBubble({ operation: 'patch', url: '...', body: {...} })

// DELETE
new HttpBubble({ operation: 'delete', url: '...' })

// HEAD
new HttpBubble({ operation: 'head', url: '...' })

// OPTIONS
new HttpBubble({ operation: 'options', url: '...' })
```

## Common Configurations

### Production API Client
```typescript
const apiClient = new HttpBubble({
  operation: 'post',
  url: 'https://api.production.com/data',
  authType: 'bearer',
  credentials: { [CredentialType.CUSTOM_AUTH_KEY]: process.env.API_TOKEN },
  retryEnabled: true,
  maxRetries: 3,
  retryStrategy: 'exponential',
  circuitBreakerEnabled: true,
  timeout: 30000,
  headers: {
    'X-Request-ID': generateRequestId(),
    'X-Service': 'my-service',
  },
});
```

### Fast Health Check
```typescript
const healthCheck = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/health',
  timeout: 5000, // 5 seconds
  retryEnabled: false, // No retry for health checks
});
```

### File Download
```typescript
const download = new HttpBubble({
  operation: 'get',
  url: 'https://example.com/file.pdf',
  responseType: 'blob',
  timeout: 60000, // 60 seconds for large files
});
```

## Testing

```typescript
import { HttpBubble } from '@bubblelab/bubble-core';

// Mock fetch
global.fetch = vi.fn();

const mockResponse = {
  ok: true,
  status: 200,
  text: vi.fn().mockResolvedValue('{"data": true}'),
  headers: new Headers(),
};

global.fetch.mockResolvedValue(mockResponse);

const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
});

const result = await httpBubble.performAction();
expect(result.success).toBe(true);
```

## Key Features Summary

✅ All HTTP methods
✅ Automatic retry with exponential backoff
✅ Circuit breaker pattern
✅ Query parameters
✅ Custom headers
✅ Multiple authentication types
✅ Timeout handling
✅ Response parsing (JSON, text, blob)
✅ Comprehensive metrics
✅ Error handling

## Full Documentation

See `HTTP_BUBBLE_README.md` for complete documentation.
