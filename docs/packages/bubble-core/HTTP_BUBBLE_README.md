# HTTP Service Bubble - Production-Ready HTTP Client

## Overview

The HTTP Service Bubble is a production-ready HTTP client with enterprise-grade features including automatic retry logic, circuit breaker pattern, comprehensive error handling, and detailed metrics.

## Features

### Core HTTP Operations
- ✅ All HTTP methods: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
- ✅ Query parameters support
- ✅ Custom headers
- ✅ Multiple body types (JSON, text, FormData, URLSearchParams)
- ✅ Response parsing (JSON, text, blob, arraybuffer)
- ✅ Redirect handling

### Advanced Features
- 🔄 **Automatic Retry Logic**
  - Configurable retry strategies (exponential, linear)
  - Configurable retry attempts
  - Retry on specific HTTP status codes
  - Retry on network errors

- ⚡ **Circuit Breaker Pattern**
  - Prevents cascading failures
  - Automatic circuit opening after threshold failures
  - Half-open state for testing service recovery
  - Configurable timeout and thresholds

- 🛡️ **Error Handling**
  - Comprehensive error messages
  - Error code classification
  - Graceful degradation
  - Detailed error metrics

- ⏱️ **Timeout Handling**
  - Configurable request timeouts
  - Automatic timeout detection
  - Timeout error messages

- 🔒 **Authentication**
  - Bearer token
  - Basic authentication
  - API key (multiple formats)
  - Custom header authentication

- 📊 **Metrics & Monitoring**
  - Request/response timing
  - Retry count tracking
  - Circuit breaker state
  - Response size tracking

## Installation

The HTTP bubble is included in the `@bubblelab/bubble-core` package.

## Usage

### Basic GET Request

```typescript
import { HttpBubble } from '@bubblelab/bubble-core';

const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
});

const result = await httpBubble.performAction();

if (result.success) {
  console.log('Data:', result.data);
  console.log('Status:', result.status);
  console.log('Response time:', result.metrics.responseTime);
} else {
  console.error('Error:', result.error);
}
```

### POST Request with JSON Body

```typescript
const httpBubble = new HttpBubble({
  operation: 'post',
  url: 'https://api.example.com/users',
  body: {
    name: 'John Doe',
    email: 'john@example.com',
  },
  headers: {
    'Content-Type': 'application/json',
  },
});

const result = await httpBubble.performAction();
```

### Request with Query Parameters

```typescript
const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/search',
  queryParams: {
    q: 'search term',
    page: 1,
    limit: 20,
    sort: 'relevance',
  },
});

const result = await httpBubble.performAction();
```

### Retry Configuration

```typescript
const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  retryEnabled: true,
  maxRetries: 5,
  retryStrategy: 'exponential',
  retryDelay: 1000,
  retryMultiplier: 2,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504],
});

const result = await httpBubble.performAction();
console.log('Retries:', result.metrics.retryCount);
```

### Circuit Breaker Configuration

```typescript
const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  circuitBreakerEnabled: true,
  circuitBreakerThreshold: 5,
  circuitBreakerTimeout: 60000,
  circuitBreakerHalfOpenAttempts: 1,
});

const result = await httpBubble.performAction();

if (result.metrics.circuitBreakerTripped) {
  console.log('Circuit breaker is open, using fallback logic');
}
```

### Authentication

```typescript
// Bearer token
const bearerBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/protected',
  authType: 'bearer',
  credentials: {
    [CredentialType.CUSTOM_AUTH_KEY]: 'your-token-here',
  },
});

// API Key
const apiKeyBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  authType: 'api-key',
  credentials: {
    [CredentialType.CUSTOM_AUTH_KEY]: 'your-api-key',
  },
});

// Custom header
const customBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  authType: 'custom',
  authHeader: 'X-Custom-Auth',
  credentials: {
    [CredentialType.CUSTOM_AUTH_KEY]: 'custom-auth-value',
  },
});
```

### Different Response Types

```typescript
// JSON response (default)
const jsonBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  responseType: 'json',
});

// Text response
const textBubble = new HttpBubble({
  operation: 'get',
  url: 'https://example.com',
  responseType: 'text',
});

// Blob response (for files, images)
const blobBubble = new HttpBubble({
  operation: 'get',
  url: 'https://example.com/image.png',
  responseType: 'blob',
});
```

### Custom Timeout and Redirects

```typescript
const httpBubble = new HttpBubble({
  operation: 'get',
  url: 'https://api.example.com/data',
  timeout: 10000, // 10 seconds
  followRedirects: true,
  maxRedirects: 5,
});
```

## Configuration Reference

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | enum | required | Operation type: `request`, `get`, `post`, `put`, `patch`, `delete`, `head`, `options` |
| `url` | string | required | The URL to make the HTTP request to |
| `method` | enum | auto | HTTP method (overrides operation default) |
| `headers` | object | `{}` | Custom HTTP headers |
| `body` | string\|object\|FormData\|URLSearchParams | - | Request body |
| `queryParams` | object | `{}` | Query parameters |
| `timeout` | number | `30000` | Request timeout in milliseconds (100-300000) |
| `followRedirects` | boolean | `true` | Whether to follow HTTP redirects |
| `maxRedirects` | number | `20` | Maximum number of redirects |

### Retry Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `retryEnabled` | boolean | `true` | Enable automatic retry |
| `maxRetries` | number | `3` | Maximum number of retry attempts (0-10) |
| `retryStrategy` | enum | `'exponential'` | Strategy: `exponential`, `linear`, `none` |
| `retryDelay` | number | `1000` | Initial retry delay in milliseconds |
| `retryMultiplier` | number | `2` | Multiplier for exponential backoff |
| `retryableStatusCodes` | array | `[408, 429, 500, 502, 503, 504]` | Status codes that trigger retry |
| `retryableErrors` | array | `['ECONNRESET', 'ETIMEDOUT', ...]` | Error codes that trigger retry |

### Circuit Breaker Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `circuitBreakerEnabled` | boolean | `false` | Enable circuit breaker |
| `circuitBreakerThreshold` | number | `5` | Failures before opening circuit |
| `circuitBreakerTimeout` | number | `60000` | Time to keep circuit open (ms) |
| `circuitBreakerHalfOpenAttempts` | number | `1` | Successful requests to close circuit |

### Authentication

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `authType` | enum | `'none'` | Type: `none`, `bearer`, `basic`, `api-key`, `api-key-header`, `custom` |
| `authHeader` | string | - | Custom header name (for authType: `custom`) |
| `credentials` | object | - | Credential object |

### Response Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `responseType` | enum | `'json'` | Expected response type: `json`, `text`, `blob`, `arraybuffer` |
| `validateStatus` | boolean | `true` | Validate HTTP status codes |
| `successStatusCodes` | array | `[200, 201, 202, 204]` | Status codes considered successful |

## Response Structure

```typescript
{
  success: boolean,
  data: unknown,
  status: number,
  statusText: string,
  headers: Record<string, string>,
  body: string,
  contentType?: string,
  contentLength?: number,
  error?: string,
  errorCode?: string,
  metrics: {
    totalAttempts: number,
    responseTime: number,
    lastAttemptTime: number,
    retryCount: number,
    fromCache?: boolean,
    circuitBreakerTripped?: boolean,
  },
  request: {
    url: string,
    method: string,
    headers?: Record<string, string>,
  }
}
```

## Retry Strategies

### Exponential Backoff (Default)

Retry delay increases exponentially with each attempt:
- Attempt 1: 1000ms
- Attempt 2: 2000ms
- Attempt 3: 4000ms
- Attempt 4: 8000ms

```typescript
retryStrategy: 'exponential',
retryDelay: 1000,
retryMultiplier: 2,
```

### Linear Backoff

Retry delay increases linearly:
- Attempt 1: 1000ms
- Attempt 2: 2000ms
- Attempt 3: 3000ms
- Attempt 4: 4000ms

```typescript
retryStrategy: 'linear',
retryDelay: 1000,
```

## Circuit Breaker States

The circuit breaker has three states:

1. **Closed** (Normal)
   - Requests pass through normally
   - Failures are counted
   - Circuit opens after threshold failures

2. **Open** (Failed)
   - Requests are blocked immediately
   - Returns 503 Service Unavailable
   - Waits for timeout before transitioning

3. **Half-Open** (Testing)
   - One request allowed through
   - Success closes the circuit
   - Failure reopens the circuit

## Error Handling

### HTTP Status Errors

```typescript
const result = await httpBubble.performAction();

if (!result.success && result.status > 0) {
  // HTTP error (4xx, 5xx)
  console.error(`HTTP ${result.status}: ${result.statusText}`);
  console.error('Response:', result.data);
}
```

### Network Errors

```typescript
if (!result.success && result.status === 0) {
  // Network error
  console.error('Network error:', result.error);
  console.error('Error code:', result.errorCode);
}
```

### Circuit Breaker Errors

```typescript
if (result.metrics.circuitBreakerTripped) {
  // Circuit breaker is open
  console.warn('Service unavailable:', result.error);
  // Use fallback logic or cached data
}
```

## Best Practices

### 1. Always Set Appropriate Timeouts

```typescript
// For fast APIs
timeout: 5000,  // 5 seconds

// For slow APIs
timeout: 60000, // 60 seconds
```

### 2. Use Circuit Breakers for Critical Services

```typescript
circuitBreakerEnabled: true,
circuitBreakerThreshold: 5,
circuitBreakerTimeout: 60000,
```

### 3. Configure Retry Based on Service Characteristics

```typescript
// For idempotent operations
retryEnabled: true,
maxRetries: 3,

// For non-idempotent operations
retryEnabled: false,
```

### 4. Monitor Metrics

```typescript
console.log('Response time:', result.metrics.responseTime);
console.log('Retry count:', result.metrics.retryCount);
console.log('Total attempts:', result.metrics.totalAttempts);
```

### 5. Handle All Error Cases

```typescript
if (result.success) {
  // Process successful response
} else if (result.metrics.circuitBreakerTripped) {
  // Circuit breaker open - use fallback
} else if (result.status >= 500) {
  // Server error - retry with backoff
} else if (result.status >= 400) {
  // Client error - don't retry
} else {
  // Network error - check connection
}
```

## Testing

```typescript
import { HttpBubble } from '@bubblelab/bubble-core';

// Mock fetch for testing
global.fetch = vi.fn();

describe('HttpBubble', () => {
  it('should make GET request', async () => {
    const mockResponse = {
      ok: true,
      status: 200,
      statusText: 'OK',
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
    expect(result.status).toBe(200);
  });
});
```

## Performance Considerations

1. **Timeouts**: Set appropriate timeouts to prevent hanging requests
2. **Retries**: Use exponential backoff to avoid overwhelming services
3. **Circuit Breakers**: Enable for critical services to prevent cascading failures
4. **Connection Pooling**: The HTTP client automatically handles connection pooling
5. **Response Size**: Monitor `contentLength` for large responses

## Troubleshooting

### High Retry Count

If you see high retry counts:
- Check service health
- Increase timeout values
- Review retryable status codes
- Consider enabling circuit breaker

### Circuit Breaker Opening

If circuit breaker opens frequently:
- Increase threshold
- Check service capacity
- Review timeout settings
- Consider load balancing

### Timeout Errors

If requests timeout frequently:
- Increase timeout value
- Check network connectivity
- Monitor service response times
- Consider using separate timeouts for different operations

## Examples

See the test file (`http-bubble.test.ts`) for comprehensive examples of all features.

## License

Part of the BubbleLab framework.
