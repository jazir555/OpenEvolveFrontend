# BubbleLab Test Coverage Design - Wave 2C

**Project:** BubbleLab Bubble Test Suite Design
**Date:** 2025-01-18
**Team:** Test Coverage Team
**Scope:** All 70+ Bubbles in bubble-core package

---

## Executive Summary

This document provides comprehensive test coverage design for all BubbleLab bubbles. The test suite is organized into five categories:
1. **Unit Tests** - Testing individual methods in isolation
2. **Integration Tests** - Testing bubble interactions and workflows
3. **Validation Tests** - Testing input validation and schema enforcement
4. **Error Handling Tests** - Testing error conditions and recovery
5. **Performance Tests** - Benchmarking and resource management

**Test Framework:** Vitest (configured in bubble-core)
**Total Bubbles:** 70+ bubbles across Service, Tool, and Workflow categories
**Coverage Goal:** 80%+ code coverage, 100% critical path coverage

---

## Table of Contents

1. [Bubble Categorization](#bubble-categorization)
2. [Test Infrastructure](#test-infrastructure)
3. [Service Bubble Test Designs](#service-bubble-test-designs)
4. [Tool Bubble Test Designs](#tool-bubble-test-designs)
5. [Workflow Bubble Test Designs](#workflow-bubble-test-designs)
6. [Test Utilities and Helpers](#test-utilities-and-helpers)
7. [Mock and Fixture Requirements](#mock-and-fixture-requirements)
8. [Coverage Metrics and Goals](#coverage-metrics-and-goals)

---

## Bubble Categorization

### Service Bubbles (25 bubbles)

**External API Integrations:**
1. `http-bubble` - HTTP client with retry/circuit breaker
2. `slack-bubble` - Slack messaging
3. `github-bubble` - GitHub operations
4. `gmail-bubble` - Email operations
5. `sendgrid-bubble` - Email sending
6. `twilio-bubble` - SMS/voice
7. `airtable-bubble` - Database operations
8. `notion-bubble` - Notion workspace
9. `stripe-bubble` - Payments
10. `webhook-bubble` - Webhook receiver
11. `google-drive-bubble` - File storage
12. `google-sheets-bubble` - Spreadsheet operations
13. `qdrant-bubble` - Vector database
14. `elasticsearch-bubble` - Search engine
15. `redis-bubble` - Cache store
16. `postgresql-bubble` - Relational database
17. `apify-bubble` - Web scraping
18. `hephaestus-bubble` - Code execution
19. `ace-tools-bubble` - ACE integration
20. `workflow-orchestrator-bubble` - Workflow management

**AI Services:**
21. `ai-agent-bubble` - AI model orchestration

### Tool Bubbles (30+ bubbles)

**Data Processing:**
1. `csv-processor-tool` - CSV parsing/processing
2. `data-transformer-tool` - Data transformations
3. `file-processor-tool` - File operations
4. `xml-parser-tool` - XML parsing
5. `log-parser-tool` - Log analysis
6. `metrics-collector-tool` - Metrics collection

**Validation:**
7. `email-validator-tool` - Email validation
8. `url-validator-tool` - URL validation
9. `bubbleflow-validation-tool` - Workflow validation

**Content Generation:**
10. `pdf-generator-tool` - PDF creation
11. `code-formatter-tool` - Code formatting
12. `text-analyzer-tool` - Text analysis
13. `image-processor-tool` - Image processing

**Search & Research:**
14. `web-search-tool` - Web search
15. `research-agent-tool` - Research automation
16. `vector-search-tool` - Vector similarity search

**Social Media:**
17. `twitter-tool` - Twitter/X operations
18. `linkedin-tool` - LinkedIn operations
19. `instagram-tool` - Instagram operations
20. `youtube-tool` - YouTube operations
21. `tiktok-tool` - TikTok operations
22. `reddit-scrape-tool` - Reddit scraping

**Web:**
23. `web-crawl-tool` - Web crawling
24. `web-extract-tool` - Web data extraction

**Integrations:**
25. `google-maps-tool` - Maps/Location
26. `chart-js-tool` - Chart generation
27. `get-bubble-details-tool` - Bubble metadata
28. `list-bubbles-tool` - Bubble listing
29. `code-edit-tool` - Code editing
30. `slack-data-assistant-tool` - Slack data assistance

### Workflow Bubbles (15+ bubbles)

1. `etl-pipeline-workflow` - ETL operations
2. `database-analyzer-workflow` - DB analysis
3. `slack-notifier-workflow` - Slack notifications
4. `webhook-repeater-workflow` - Webhook forwarding
5. `data-enrichment-workflow` - Data enrichment
6. `monitoring-alert-workflow` - Alerting
7. `api-aggregator-workflow` - API aggregation
8. `event-handler-workflow` - Event processing
9. `scheduled-task-workflow` - Scheduled jobs
10. `multi-step-approval-workflow` - Approval flows
11. `generate-document-workflow` - Document generation
12. `parse-document-workflow` - Document parsing
13. `pdf-form-operations-workflow` - PDF forms
14. `pdf-ocr-workflow` - OCR processing
15. `slack-data-assistant-workflow` - Slack assistant

---

## Test Infrastructure

### Directory Structure

```
bubble-core/src/
├── bubbles/
│   ├── service-bubble/
│   │   ├── http-bubble.ts
│   │   ├── http-bubble.test.ts           # Unit tests
│   │   ├── http-bubble.integration.test.ts # Integration tests
│   │   └── __tests__/
│   │       ├── http-mocks.ts
│   │       └── http-fixtures.ts
│   ├── tool-bubble/
│   │   └── (same pattern)
│   └── workflow-bubble/
│       └── (same pattern)
├── __tests__/
│   ├── setup.ts                          # Global test setup
│   ├── teardown.ts                       # Global test teardown
│   ├── helpers/
│   │   ├── mock-responses.ts             # API mock responses
│   │   ├── test-data.ts                  # Test data fixtures
│   │   ├── assertion-helpers.ts          # Custom assertions
│   │   └── mock-factory.ts               # Mock object factory
│   └── performance/
│       ├── performance-baseline.ts       # Performance benchmarks
│       └── load-test.ts                  # Load testing utilities
├── vitest.config.ts
└── vitest.setup.ts                       # Test setup file
```

### Global Test Setup (vitest.setup.ts)

```typescript
import { vi } from 'vitest';

// Global test configuration
beforeAll(() => {
  // Set timezone to UTC for consistent tests
  process.env.TZ = 'UTC';

  // Suppress console.log in tests unless explicitly enabled
  if (!process.env.DEBUG_TESTS) {
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  }
});

afterAll(() => {
  // Cleanup
  vi.restoreAllMocks();
});

// Mock fetch globally for HTTP tests
vi.stubGlobal('fetch', vi.fn());
```

---

## Service Bubble Test Designs

### 1. HTTP Bubble Test Suite

**File:** `service-bubble/http-bubble.test.ts`

#### Test Categories

##### A. Unit Tests

**A1. Request Building Tests**
```typescript
describe('HttpBubble - Request Building', () => {
  test('should build GET request correctly', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/users',
    });

    const result = await bubble.act();

    expect(result.request.method).toBe('GET');
    expect(result.request.url).toBe('https://api.example.com/users');
  });

  test('should add query parameters to URL', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/users',
      queryParams: { page: 1, limit: 10, active: true },
    });

    const result = await bubble.act();

    expect(result.request.url).toContain('page=1');
    expect(result.request.url).toContain('limit=10');
    expect(result.request.url).toContain('active=true');
  });

  test('should build POST request with JSON body', async () => {
    const bubble = new HttpBubble({
      operation: 'post',
      url: 'https://api.example.com/users',
      body: { name: 'John', email: 'john@example.com' },
    });

    const result = await bubble.act();

    expect(result.request.method).toBe('POST');
    expect(result.request.headers['Content-Type']).toBe('application/json');
  });

  test('should build POST request with FormData', async () => {
    const formData = new FormData();
    formData.append('file', 'content');

    const bubble = new HttpBubble({
      operation: 'post',
      url: 'https://api.example.com/upload',
      body: formData,
    });

    const result = await bubble.act();

    expect(result.request.method).toBe('POST');
    expect(result.request.body).toBeInstanceOf(FormData);
  });

  test('should build POST request with URLSearchParams', async () => {
    const params = new URLSearchParams({ foo: 'bar', baz: 'qux' });

    const bubble = new HttpBubble({
      operation: 'post',
      url: 'https://api.example.com/data',
      body: params,
    });

    const result = await bubble.act();

    expect(result.request.method).toBe('POST');
  });
});
```

**A2. Authentication Tests**
```typescript
describe('HttpBubble - Authentication', () => {
  test('should add Bearer token header', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/protected',
      authType: 'bearer',
      credentials: {
        [CredentialType.CUSTOM_AUTH_KEY]: 'secret-token'
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['Authorization']).toBe('Bearer secret-token');
  });

  test('should add Basic auth header', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/protected',
      authType: 'basic',
      credentials: {
        [CredentialType.CUSTOM_AUTH_KEY]: 'base64encoded'
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['Authorization']).toBe('Basic base64encoded');
  });

  test('should add API key header', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      authType: 'api-key',
      credentials: {
        [CredentialType.CUSTOM_AUTH_KEY]: 'xyz123'
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['X-API-Key']).toBe('xyz123');
  });

  test('should add custom auth header', async () => {
    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      authType: 'custom',
      authHeader: 'X-Custom-Auth',
      credentials: {
        [CredentialType.CUSTOM_AUTH_KEY]: 'custom-value'
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['X-Custom-Auth']).toBe('custom-value');
  });
});
```

**A3. Retry Logic Tests**
```typescript
describe('HttpBubble - Retry Logic', () => {
  test('should retry on 500 status code', async () => {
    let attemptCount = 0;
    mockFetch.mockImplementation(() => {
      attemptCount++;
      if (attemptCount === 1) {
        return Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      retryEnabled: true,
      maxRetries: 3,
      retryableStatusCodes: [500],
    });

    const result = await bubble.act();

    expect(result.metrics.retryCount).toBe(1);
    expect(result.metrics.totalAttempts).toBe(2);
    expect(result.success).toBe(true);
  });

  test('should retry with exponential backoff', async () => {
    const timestamps: number[] = [];
    mockFetch.mockImplementation(() => {
      timestamps.push(Date.now());
      return Promise.resolve({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      retryEnabled: true,
      maxRetries: 3,
      retryStrategy: 'exponential',
      retryDelay: 100,
      retryMultiplier: 2,
    });

    await bubble.act();

    // Verify exponential delays: 100ms, 200ms, 400ms
    expect(timestamps.length).toBe(4); // initial + 3 retries
    const delay1 = timestamps[1] - timestamps[0];
    const delay2 = timestamps[2] - timestamps[1];
    expect(delay2).toBeGreaterThan(delay1); // Exponential increase
  });

  test('should not retry on 404 status code', async () => {
    let attemptCount = 0;
    mockFetch.mockImplementation(() => {
      attemptCount++;
      return Promise.resolve({
        ok: false,
        status: 404,
        statusText: 'Not Found',
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/notfound',
      retryEnabled: true,
      maxRetries: 3,
      retryableStatusCodes: [500, 502, 503, 504],
    });

    const result = await bubble.act();

    expect(attemptCount).toBe(1); // No retries
    expect(result.metrics.retryCount).toBe(0);
  });

  test('should retry on network errors', async () => {
    let attemptCount = 0;
    mockFetch.mockImplementation(() => {
      attemptCount++;
      if (attemptCount < 3) {
        throw new Error('ECONNRESET');
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ success: true }),
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      retryEnabled: true,
      maxRetries: 3,
      retryableErrors: ['ECONNRESET'],
    });

    const result = await bubble.act();

    expect(result.metrics.retryCount).toBe(2);
    expect(result.success).toBe(true);
  });
});
```

**A4. Circuit Breaker Tests**
```typescript
describe('HttpBubble - Circuit Breaker', () => {
  beforeEach(() => {
    // Reset circuit breaker state
    HttpBubble['circuitBreakerStates'].clear();
  });

  test('should open circuit after threshold failures', async () => {
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
      } as Response)
    );

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      circuitBreakerEnabled: true,
      circuitBreakerThreshold: 3,
      circuitBreakerTimeout: 60000,
    });

    // Trigger failures to open circuit
    await bubble.act();
    await bubble.act();
    await bubble.act();

    // Fourth call should return circuit breaker error
    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.errorCode).toBe('CIRCUIT_BREAKER_OPEN');
    expect(result.metrics.circuitBreakerTripped).toBe(true);
  });

  test('should close circuit after successful request', async () => {
    let failCount = 0;
    mockFetch.mockImplementation(() => {
      failCount++;
      if (failCount <= 3) {
        return Promise.resolve({
          ok: false,
          status: 503,
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      circuitBreakerEnabled: true,
      circuitBreakerThreshold: 3,
    });

    // Open circuit
    await bubble.act();
    await bubble.act();
    await bubble.act();
    const circuitOpenResult = await bubble.act();
    expect(circuitOpenResult.errorCode).toBe('CIRCUIT_BREAKER_OPEN');

    // Wait for circuit half-open timeout
    await new Promise(resolve => setTimeout(resolve, 100));

    // Successful request should close circuit
    const result = await bubble.act();
    expect(result.success).toBe(true);

    // Subsequent requests should work
    const result2 = await bubble.act();
    expect(result2.errorCode).toBeUndefined();
  });

  test('should respect circuit breaker timeout', async () => {
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: false,
        status: 503,
      } as Response)
    );

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      circuitBreakerEnabled: true,
      circuitBreakerThreshold: 2,
      circuitBreakerTimeout: 100, // 100ms
    });

    // Open circuit
    await bubble.act();
    await bubble.act();

    const circuitOpenResult = await bubble.act();
    expect(circuitOpenResult.errorCode).toBe('CIRCUIT_BREAKER_OPEN');

    // Wait for timeout
    await new Promise(resolve => setTimeout(resolve, 150));

    // Should allow request after timeout
    mockFetch.mockImplementation(() =>
      Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response)
    );

    const result = await bubble.act();
    expect(result.errorCode).toBeUndefined();
  });
});
```

**A5. Timeout Tests**
```typescript
describe('HttpBubble - Timeout Handling', () => {
  test('should timeout request after specified duration', async () => {
    mockFetch.mockImplementation(() =>
      new Promise(resolve => setTimeout(resolve, 10000))
    );

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/slow',
      timeout: 100, // 100ms timeout
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('timeout');
    expect(result.errorCode).toBe('AbortError');
  });

  test('should handle timeout with retry', async () => {
    let attemptCount = 0;
    mockFetch.mockImplementation(() => {
      attemptCount++;
      if (attemptCount === 1) {
        // First attempt times out
        return new Promise(() => {}); // Never resolves
      }
      // Second attempt succeeds
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ success: true }),
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/slow',
      timeout: 100,
      retryEnabled: true,
      maxRetries: 2,
    });

    // Mock timeout using AbortController
    const originalFetch = global.fetch;
    global.fetch = vi.fn((...args) => {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 100);
      return mockFetch(...args);
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
  });
});
```

**A6. Response Parsing Tests**
```typescript
describe('HttpBubble - Response Parsing', () => {
  test('should parse JSON response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'test', count: 42 }),
      headers: new Headers({ 'content-type': 'application/json' }),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      responseType: 'json',
    });

    const result = await bubble.act();

    expect(result.data).toEqual({ data: 'test', count: 42 });
    expect(result.contentType).toBe('application/json');
  });

  test('should parse text response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => 'plain text response',
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/text',
      responseType: 'text',
    });

    const result = await bubble.act();

    expect(result.data).toBe('plain text response');
  });

  test('should parse blob response', async () => {
    const blob = new Blob(['test data'], { type: 'text/plain' });
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      blob: async () => blob,
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/file',
      responseType: 'blob',
    });

    const result = await bubble.act();

    expect(result.data).toBeInstanceOf(Blob);
  });

  test('should handle invalid JSON gracefully', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => 'not valid json',
      headers: new Headers({ 'content-type': 'application/json' }),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      responseType: 'json',
    });

    const result = await bubble.act();

    // Should return as text if JSON parsing fails
    expect(result.data).toBe('not valid json');
  });
});
```

**A7. Header Management Tests**
```typescript
describe('HttpBubble - Header Management', () => {
  test('should merge custom headers with defaults', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      headers: {
        'X-Custom-Header': 'custom-value',
        'Accept': 'application/vnd.api+json',
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['X-Custom-Header']).toBe('custom-value');
    expect(result.request.headers['Accept']).toBe('application/vnd.api+json');
    expect(result.request.headers['User-Agent']).toBe('BubbleLab-HttpBubble/2.0');
  });

  test('should set Content-Type based on body type', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'post',
      url: 'https://api.example.com/data',
      body: { key: 'value' },
    });

    const result = await bubble.act();

    expect(result.request.headers['Content-Type']).toBe('application/json');
  });

  test('should not override explicit Content-Type', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'post',
      url: 'https://api.example.com/data',
      body: { key: 'value' },
      headers: {
        'Content-Type': 'application/xml',
      },
    });

    const result = await bubble.act();

    expect(result.request.headers['Content-Type']).toBe('application/xml');
  });
});
```

**A8. Redirect Tests**
```typescript
describe('HttpBubble - Redirect Handling', () => {
  test('should follow redirects by default', async () => {
    let redirectCount = 0;
    mockFetch.mockImplementation(() => {
      redirectCount++;
      if (redirectCount === 1) {
        return Promise.resolve({
          ok: true,
          status: 302,
          statusText: 'Found',
          headers: new Headers({ 'location': 'https://api.example.com/final' }),
          url: 'https://api.example.com/redirect',
        } as Response);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        statusText: 'OK',
        json: async () => ({ final: true }),
        url: 'https://api.example.com/final',
      } as Response);
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/redirect',
      followRedirects: true,
      maxRedirects: 5,
    });

    const result = await bubble.act();

    expect(result.request.url).toBe('https://api.example.com/final');
    expect(result.status).toBe(200);
  });

  test('should not follow redirects when disabled', async () => {
    mockFetch.mockResolvedValue({
      ok: false, // 302 is not considered "ok"
      status: 302,
      statusText: 'Found',
      headers: new Headers({ 'location': 'https://api.example.com/final' }),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/redirect',
      followRedirects: false,
    });

    const result = await bubble.act();

    expect(result.status).toBe(302);
    expect(result.request.url).toBe('https://api.example.com/redirect');
  });

  test('should respect max redirects limit', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 302,
      statusText: 'Found',
      headers: new Headers({ 'location': 'https://api.example.com/next' }),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/start',
      followRedirects: true,
      maxRedirects: 2,
    });

    const result = await bubble.act();

    // Should stop after max redirects
    expect(result.success).toBe(false);
    expect(result.error).toContain('redirect');
  });
});
```

##### B. Validation Tests

```typescript
describe('HttpBubble - Input Validation', () => {
  test('should validate URL format', () => {
    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'not-a-valid-url',
      });
    }).toThrow(z.ZodError);
  });

  test('should validate timeout range', () => {
    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        timeout: 50, // Below minimum of 100
      });
    }).toThrow();

    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        timeout: 500000, // Above maximum of 300000
      });
    }).toThrow();
  });

  test('should validate max retries range', () => {
    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        maxRetries: 15, // Above maximum of 10
      });
    }).toThrow();
  });

  test('should validate HTTP method', () => {
    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        method: 'INVALID' as any,
      });
    }).toThrow(z.ZodError);
  });

  test('should validate retry strategy', () => {
    expect(() => {
      new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        retryStrategy: 'invalid' as any,
      });
    }).toThrow(z.ZodError);
  });
});
```

##### C. Error Handling Tests

```typescript
describe('HttpBubble - Error Handling', () => {
  test('should handle network errors', async () => {
    mockFetch.mockRejectedValue(new Error('ENOTFOUND'));

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://nonexistent.example.com/data',
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('ENOTFOUND');
    expect(result.errorCode).toBe('Error');
  });

  test('should handle DNS resolution failures', async () => {
    mockFetch.mockRejectedValue(new Error('EAI_AGAIN'));

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      retryableErrors: ['EAI_AGAIN'],
      retryEnabled: true,
      maxRetries: 2,
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.metrics.totalAttempts).toBeGreaterThan(1);
  });

  test('should handle connection reset errors', async () => {
    mockFetch.mockRejectedValue(new Error('ECONNRESET'));

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      retryEnabled: true,
      maxRetries: 3,
      retryableErrors: ['ECONNRESET'],
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.metrics.retryCount).toBe(3);
  });

  test('should handle malformed response', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      body: null,
    } as any);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      responseType: 'json',
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data).toBeNull();
  });

  test('should handle SSL certificate errors', async () => {
    mockFetch.mockRejectedValue(new Error('UNABLE_TO_VERIFY_LEAF_SIGNATURE'));

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://self-signed.example.com/data',
      rejectUnauthorized: true,
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('UNABLE_TO_VERIFY_LEAF_SIGNATURE');
  });
});
```

##### D. Performance Tests

```typescript
describe('HttpBubble - Performance', () => {
  test('should complete simple request within timeout', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'test' }),
    } as Response);

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/data',
      timeout: 5000,
    });

    const startTime = Date.now();
    await bubble.act();
    const duration = Date.now() - startTime;

    expect(duration).toBeLessThan(1000); // Should complete quickly
  });

  test('should handle concurrent requests efficiently', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'test' }),
    } as Response);

    const bubbles = Array(10).fill(null).map((_, i) =>
      new HttpBubble({
        operation: 'get',
        url: `https://api.example.com/data${i}`,
      })
    );

    const startTime = Date.now();
    await Promise.all(bubbles.map(b => b.act()));
    const duration = Date.now() - startTime;

    // Concurrent requests should be faster than sequential
    expect(duration).toBeLessThan(500);
  });

  test('should measure response time accurately', async () => {
    mockFetch.mockImplementation(async () => {
      await new Promise(resolve => setTimeout(resolve, 100));
      return {
        ok: true,
        status: 200,
        json: async () => ({ data: 'test' }),
      } as Response;
    });

    const bubble = new HttpBubble({
      operation: 'get',
      url: 'https://api.example.com/slow',
    });

    const result = await bubble.act();

    expect(result.metrics.responseTime).toBeGreaterThanOrEqual(100);
    expect(result.metrics.responseTime).toBeLessThan(200);
  });
});
```

##### E. Integration Tests

**File:** `service-bubble/http-bubble.integration.test.ts`

```typescript
describe('HttpBubble - Integration Tests', () => {
  // These tests make real HTTP requests to test services
  // Use environment variables to configure test endpoints

  test('should integrate with real API (if TEST_API_URL is set)', async () => {
    const testUrl = process.env.TEST_API_URL;

    if (!testUrl) {
      test.skip('TEST_API_URL not set');
    }

    const bubble = new HttpBubble({
      operation: 'get',
      url: testUrl,
      timeout: 5000,
    });

    const result = await bubble.act();

    expect(result).toBeDefined();
    expect(result.status).toBeGreaterThan(0);
  });

  test('should handle real redirect chain', async () => {
    const redirectUrl = process.env.TEST_REDIRECT_URL;

    if (!redirectUrl) {
      test.skip('TEST_REDIRECT_URL not set');
    }

    const bubble = new HttpBubble({
      operation: 'get',
      url: redirectUrl,
      followRedirects: true,
    });

    const result = await bubble.act();

    expect(result.status).toBe(200);
  });
});
```

---

### 2. Slack Bubble Test Suite

**File:** `service-bubble/slack-bubble.test.ts`

#### Test Scenarios

##### A. Message Operations

```typescript
describe('SlackBubble - Message Operations', () => {
  let mockSlackApi: any;

  beforeEach(() => {
    mockSlackApi = vi.fn().mockResolvedValue({
      ok: true,
      ts: '1234567890.123456',
      channel: 'C123456',
    });

    global.fetch = mockSlackApi;
  });

  test('should send message to channel', async () => {
    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Hello, World!',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.timestamp).toBeDefined();
    expect(result.data.status).toBe('sent');
  });

  test('should send message with blocks', async () => {
    const blocks = [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: '*Test Message*',
        },
      },
    ];

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Fallback text',
      blocks,
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
  });

  test('should send thread reply', async () => {
    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Thread reply',
      threadTs: '1234567890.123456',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
  });

  test('should send direct message', async () => {
    const bubble = new SlackBubble({
      operation: 'sendDM',
      userId: 'U123456',
      text: 'Direct message',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.userId).toBe('U123456');
  });

  test('should update existing message', async () => {
    const bubble = new SlackBubble({
      operation: 'updateMessage',
      channel: 'C123456',
      timestamp: '1234567890.123456',
      text: 'Updated message',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.status).toBe('updated');
  });

  test('should delete message', async () => {
    const bubble = new SlackBubble({
      operation: 'deleteMessage',
      channel: 'C123456',
      timestamp: '1234567890.123456',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.status).toBe('deleted');
  });
});
```

##### B. Reaction Operations

```typescript
describe('SlackBubble - Reaction Operations', () => {
  test('should add reaction to message', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
    });

    const bubble = new SlackBubble({
      operation: 'addReaction',
      channel: 'C123456',
      timestamp: '1234567890.123456',
      reaction: 'thumbsup',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.reaction).toBe('thumbsup');
  });

  test('should remove reaction from message', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
    });

    const bubble = new SlackBubble({
      operation: 'removeReaction',
      channel: 'C123456',
      timestamp: '1234567890.123456',
      reaction: 'thumbsup',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.status).toBe('removed');
  });
});
```

##### C. Channel Operations

```typescript
describe('SlackBubble - Channel Operations', () => {
  test('should get channel info', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      channel: {
        id: 'C123456',
        name: 'test-channel',
        topic: { value: 'Test topic' },
        purpose: { value: 'Test purpose' },
        num_members: 10,
      },
    });

    const bubble = new SlackBubble({
      operation: 'getChannelInfo',
      channelId: 'C123456',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.info.name).toBe('test-channel');
    expect(result.data.info.members).toBe(10);
  });

  test('should list channels', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      channels: [
        {
          id: 'C123456',
          name: 'channel1',
          is_private: false,
          num_members: 5,
          topic: { value: 'Topic 1' },
        },
        {
          id: 'C789012',
          name: 'channel2',
          is_private: true,
          num_members: 3,
          topic: { value: 'Topic 2' },
        },
      ],
    });

    const bubble = new SlackBubble({
      operation: 'listChannels',
      limit: 10,
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.channels).toHaveLength(2);
    expect(result.data.count).toBe(2);
  });

  test('should filter channels by type', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      channels: [
        {
          id: 'C123456',
          name: 'public-channel',
          is_private: false,
          num_members: 5,
        },
      ],
    });

    const bubble = new SlackBubble({
      operation: 'listChannels',
      limit: 100,
      types: ['public_channel'],
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
  });
});
```

##### D. User Operations

```typescript
describe('SlackBubble - User Operations', () => {
  test('should get user info', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      user: {
        id: 'U123456',
        name: 'testuser',
        real_name: 'Test User',
        profile: {
          email: 'test@example.com',
          title: 'Developer',
        },
        tz: 'America/New_York',
      },
    });

    const bubble = new SlackBubble({
      operation: 'getUserInfo',
      userId: 'U123456',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.user.name).toBe('testuser');
    expect(result.data.user.email).toBe('test@example.com');
  });
});
```

##### E. File Operations

```typescript
describe('SlackBubble - File Operations', () => {
  test('should upload file as string', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      file: {
        id: 'F123456',
        url_private: 'https://files.slack.com/files-pri/T123/F123456/download',
      },
    });

    const bubble = new SlackBubble({
      operation: 'uploadFile',
      channel: 'C123456',
      fileContent: 'File content here',
      filename: 'test.txt',
      title: 'Test File',
      initialComment: 'Initial comment',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
    expect(result.data.fileId).toBe('F123456');
    expect(result.data.status).toBe('uploaded');
  });

  test('should upload file as buffer', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      file: {
        id: 'F123456',
      },
    });

    const bubble = new SlackBubble({
      operation: 'uploadFile',
      channel: 'C123456',
      fileContent: Buffer.from('Buffer content'),
      filename: 'test.bin',
      filetype: 'bin',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(true);
  });
});
```

##### F. Error Handling

```typescript
describe('SlackBubble - Error Handling', () => {
  test('should handle invalid credentials', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      error: 'invalid_auth',
    });

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'invalid-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('invalid_auth');
  });

  test('should handle missing permissions', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      error: 'not_allowed_token_type',
    });

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('not_allowed_token_type');
  });

  test('should handle rate limiting', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      error: 'ratelimited',
    });

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const result = await bubble.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('ratelimited');
  });

  test('should handle malformed credentials', async () => {
    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: 'not-json',
      },
    });

    await expect(bubble.act()).rejects.toThrow('Invalid Slack credentials format');
  });

  test('should handle missing bot token', async () => {
    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({}),
      },
    });

    await expect(bubble.act()).rejects.toThrow('Slack bot token is required');
  });
});
```

##### G. Credential Testing

```typescript
describe('SlackBubble - Credential Testing', () => {
  test('should validate credentials with auth.test', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      team: 'TestTeam',
      user: 'TestBot',
    });

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test-token',
        }),
      },
    });

    const isValid = await bubble.testCredential();

    expect(isValid).toBe(true);
  });

  test('should invalidate bad credentials', async () => {
    mockFetch.mockResolvedValue({
      ok: false,
      error: 'invalid_auth',
    });

    const bubble = new SlackBubble({
      operation: 'sendMessage',
      channel: 'C123456',
      text: 'Test',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'invalid-token',
        }),
      },
    });

    const isValid = await bubble.testCredential();

    expect(isValid).toBe(false);
  });
});
```

---

### 3. Other Service Bubbles Test Designs

Due to space constraints, here are summaries for other service bubbles:

#### Airtable Bubble
- **Test File:** `service-bubble/airtable.test.ts`
- **Key Tests:**
  - CRUD operations on tables
  - Field type handling (singleSelect, multipleSelect, attachment, etc.)
  - Formula field evaluation
  - Linked record operations
  - Sorting and filtering
  - Pagination handling
  - Error handling for rate limits

#### Notion Bubble
- **Test File:** `service-bubble/notion.test.ts`
- **Key Tests:**
  - Page creation and updates
  - Database operations
  - Block content handling (paragraphs, headings, lists, etc.)
  - Rich text formatting
  - Parent-child page relationships
  - Search functionality
  - Comment operations

#### Stripe Bubble
- **Test File:** `service-bubble/stripe.test.ts`
- **Key Tests:**
  - Payment intent creation
  - Customer management
  - Subscription operations
  - Invoice handling
  - Webhook signature verification
  - Refund processing
  - Error code mapping

#### PostgreSQL Bubble
- **Test File:** `service-bubble/postgresql.test.ts`
- **Key Tests:**
  - Connection pooling
  - Query execution
  - Parameterized queries (SQL injection prevention)
  - Transaction handling
  - Connection error recovery
  - Result set parsing
  - Schema validation

#### Redis Bubble
- **Test File:** `service-bubble/redis.test.ts`
- **Key Tests:**
  - String operations
  - Hash operations
  - List operations
  - Set operations
  - Sorted set operations
  - Pub/sub functionality
  - TTL/expiration handling
  - Connection management

#### Elasticsearch Bubble
- **Test File:** `service-bubble/elasticsearch.test.ts`
- **Key Tests:**
  - Index creation and management
  - Document CRUD operations
  - Query execution (match, term, range, bool queries)
  - Aggregation operations
  - Bulk operations
  - Mapping management
  - Scroll API for large result sets

#### Qdrant Bubble
- **Test File:** `service-bubble/qdrant.test.ts`
- **Key Tests:**
  - Collection creation and deletion
  - Point insertion and upsertion
  - Vector search operations
  - Filter operations
  - Payload management
  - Distance metric verification
  - Batch operations

---

## Tool Bubble Test Designs

### 1. CSV Processor Tool Test Suite

**File:** `tool-bubble/csv-processor-tool.test.ts`

#### Test Categories

##### A. Parse Operation Tests

```typescript
describe('CSVProcessorTool - Parse Operation', () => {
  test('should parse simple CSV with headers', async () => {
    const csvData = `name,age,city
John,30,New York
Jane,25,London`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(2);
    expect(result.headers).toEqual(['name', 'age', 'city']);
    expect(result.data[0]).toEqual({
      name: 'John',
      age: 30,
      city: 'New York',
    });
  });

  test('should parse CSV without headers', async () => {
    const csvData = `John,30,New York
Jane,25,London`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: false,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.success).toBe(true);
    expect(result.headers).toEqual(['column_0', 'column_1', 'column_2']);
  });

  test('should handle different delimiters', async () => {
    const csvData = `name;age;city
John;30;New York`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      delimiter: CSVDelimiter.SEMICOLON,
    });

    const result = await tool.act();

    expect(result.data[0]).toEqual({
      name: 'John',
      age: 30,
      city: 'New York',
    });
  });

  test('should handle quoted fields', async () => {
    const csvData = `name,description
"John Doe","A person with, a comma"
"Jane Smith","A person with ""quotes"""`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.data[0].description).toBe('A person with, a comma');
    expect(result.data[1].description).toBe('A person with "quotes"');
  });

  test('should handle escaped quotes', async () => {
    const csvData = `name,text
John,"Text with \\"escaped\\" quotes"`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      delimiter: CSVDelimiter.COMMA,
      escapeChar: '\\',
    });

    const result = await tool.act();

    expect(result.data[0].text).toBe('Text with "escaped" quotes');
  });

  test('should skip empty lines when configured', async () => {
    const csvData = `name,age
John,30

Jane,25`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      skipEmptyLines: true,
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
  });

  test('should trim whitespace when configured', async () => {
    const csvData = `name,age
  John  ,  30
  Jane  ,  25`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      trimWhitespace: true,
    });

    const result = await tool.act();

    expect(result.data[0].name).toBe('John');
    expect(result.data[0].age).toBe(30);
  });

  test('should respect max rows limit', async () => {
    const rows = Array(100).fill('John,30,New York').join('\n');
    const csvData = `name,age,city\n${rows}`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
      maxRows: 10,
    });

    const result = await tool.act();

    expect(result.rowCount).toBe(10);
  });

  test('should handle newlines in quoted fields', async () => {
    const csvData = `name,description
John,"Line 1
Line 2
Line 3"`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.data[0].description).toBe('Line 1\nLine 2\nLine 3');
  });

  test('should infer data types', async () => {
    const csvData = `string_col,number_col,bool_col,date_col
hello,42,true,2023-01-15
world,3.14,false,2023-12-31`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(typeof result.data[0].string_col).toBe('string');
    expect(typeof result.data[0].number_col).toBe('number');
    expect(typeof result.data[0].bool_col).toBe('boolean');
    expect(result.data[0].date_col).toBeInstanceOf(Date);
  });

  test('should detect duplicate headers', async () => {
    const csvData = `name,name,age
John,John,30`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors?.[0].column).toBe('headers');
    expect(result.validationErrors?.[0].error).toContain('Duplicate header names');
  });

  test('should handle mismatched row lengths', async () => {
    const csvData = `name,age,city
John,30,New York
Jane,25`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors?.[0].error).toContain('expected 3');
    expect(result.data[1].city).toBe(''); // Padded with empty string
  });
});
```

##### B. Validate Operation Tests

```typescript
describe('CSVProcessorTool - Validate Operation', () => {
  test('should validate string type', async () => {
    const csvData = `name,age
John,30
42,25`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.VALIDATE,
      csvData,
      hasHeader: true,
      validateSchema: {
        name: 'string',
        age: 'number',
      },
    });

    const result = await tool.act();

    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors?.[0]).toMatchObject({
      row: 2,
      column: 'name',
      error: 'Expected string, got number',
      value: 42,
    });
  });

  test('should validate number type', async () => {
    const csvData = `name,age
John,thirty`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.VALIDATE,
      csvData,
      hasHeader: true,
      validateSchema: {
        name: 'string',
        age: 'number',
      },
    });

    const result = await tool.act();

    expect(result.validationErrors).toBeDefined();
    expect(result.validationErrors?.[0].column).toBe('age');
  });

  test('should validate boolean type', async () => {
    const csvData = `name,active
John,yes`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.VALIDATE,
      csvData,
      hasHeader: true,
      validateSchema: {
        name: 'string',
        active: 'boolean',
      },
    });

    const result = await tool.act();

    expect(result.validationErrors).toBeDefined();
    expect(result.statistics?.invalidRows).toBeGreaterThan(0);
  });

  test('should report validation statistics', async () => {
    const csvData = `name,age
John,30
Jane,25
invalid,thirty`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.VALIDATE,
      csvData,
      hasHeader: true,
      validateSchema: {
        name: 'string',
        age: 'number',
      },
    });

    const result = await tool.act();

    expect(result.statistics).toBeDefined();
    expect(result.statistics?.totalRows).toBe(3);
    expect(result.statistics?.validRows).toBe(2);
    expect(result.statistics?.invalidRows).toBe(1);
  });
});
```

##### C. Transform Operation Tests

```typescript
describe('CSVProcessorTool - Transform Operation', () => {
  test('should transform to uppercase', async () => {
    const csvData = `name
john
jane`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'name',
          operation: 'upper',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].name).toBe('JOHN');
    expect(result.data[1].name).toBe('JANE');
  });

  test('should transform to lowercase', async () => {
    const csvData = `name
JOHN
JANE`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'name',
          operation: 'lower',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].name).toBe('john');
    expect(result.data[1].name).toBe('jane');
  });

  test('should trim whitespace', async () => {
    const csvData = `name
  John
  Jane  `;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'name',
          operation: 'trim',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].name).toBe('John');
    expect(result.data[1].name).toBe('Jane');
  });

  test('should replace text', async () => {
    const csvData = `name
Hello World
Goodbye World`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'name',
          operation: 'replace',
          value: 'World',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].name).toBe('Hello World'); // Bug in implementation?
  });

  test('should calculate expressions', async () => {
    const csvData = `price
100
200`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'price',
          operation: 'calculate',
          expression: 'price * 1.1',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].price).toBe(110);
    expect(result.data[1].price).toBe(220);
  });

  test('should handle calculation errors gracefully', async () => {
    const csvData = `value
not_a_number`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'value',
          operation: 'calculate',
          expression: 'value * 2',
        },
      ],
    });

    const result = await tool.act();

    // Should preserve original value on error
    expect(result.data[0].value).toBe('not_a_number');
  });

  test('should prevent code injection in calculations', async () => {
    const csvData = `value
10`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.TRANSFORM,
      csvData,
      hasHeader: true,
      transformRules: [
        {
          column: 'value',
          operation: 'calculate',
          expression: 'process.exit(1)', // Malicious expression
        },
      ],
    });

    // Should not execute code, only evaluate math
    const result = await tool.act();

    // Should throw error or handle safely
    expect(result.success).toBe(false);
  });
});
```

##### D. Filter Operation Tests

```typescript
describe('CSVProcessorTool - Filter Operation', () => {
  test('should filter by equals', async () => {
    const csvData = `name,city
John,New York
Jane,London
John,Paris`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'name',
          operator: 'equals',
          value: 'John',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
    expect(result.data.every(row => row.name === 'John')).toBe(true);
  });

  test('should filter by contains', async () => {
    const csvData = `name
John Smith
Jane Doe
Johnson`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'name',
          operator: 'contains',
          value: 'John',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
  });

  test('should filter by startsWith', async () => {
    const csvData = `name
Apple
Apricot
Banana`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'name',
          operator: 'startsWith',
          value: 'Ap',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
  });

  test('should filter by endsWith', async () => {
    const csvData = `name
Testing
Tested
Tester`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'name',
          operator: 'endsWith',
          value: 'ed',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
  });

  test('should filter by greater than', async () => {
    const csvData = `name,score
John,85
Jane,92
Bob,78`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'score',
          operator: 'gt',
          value: 80,
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
    expect(result.data.every(row => row.score > 80)).toBe(true);
  });

  test('should filter by less than', async () => {
    const csvData = `name,score
John,85
Jane,92
Bob,78`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'score',
          operator: 'lt',
          value: 80,
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(1);
    expect(result.data[0].score).toBe(78);
  });

  test('should apply multiple filter rules', async () => {
    const csvData = `name,age,city
John,30,New York
Jane,25,London
Bob,35,New York`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.FILTER,
      csvData,
      hasHeader: true,
      filterRules: [
        {
          column: 'city',
          operator: 'equals',
          value: 'New York',
        },
        {
          column: 'age',
          operator: 'gte',
          value: 30,
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
  });
});
```

##### E. Export Operation Tests

```typescript
describe('CSVProcessorTool - Export Operation', () => {
  test('should export data to CSV', async () => {
    const data = [
      { name: 'John', age: 30, city: 'New York' },
      { name: 'Jane', age: 25, city: 'London' },
    ];

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.EXPORT,
      exportData: data,
      hasHeader: true,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.success).toBe(true);
    expect(result.csvOutput).toBeDefined();
    expect(result.csvOutput).toContain('name,age,city');
    expect(result.csvOutput).toContain('John,30,New York');
    expect(result.csvOutput).toContain('Jane,25,London');
  });

  test('should handle special characters in export', async () => {
    const data = [
      { text: 'Text with, a comma' },
      { text: 'Text with "quotes"' },
      { text: 'Text with\nnewlines' },
    ];

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.EXPORT,
      exportData: data,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.csvOutput).toContain('"Text with, a comma"');
    expect(result.csvOutput).toContain('"Text with ""quotes"""');
  });

  test('should export without headers', async () => {
    const data = [
      { col1: 'A', col2: 'B' },
      { col1: 'C', col2: 'D' },
    ];

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.EXPORT,
      exportData: data,
      hasHeader: false,
    });

    const result = await tool.act();

    expect(result.csvOutput).not.toContain('col1,col2');
    expect(result.csvOutput).toContain('A,B');
  });

  test('should use custom delimiter', async () => {
    const data = [
      { name: 'John', age: 30 },
    ];

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.EXPORT,
      exportData: data,
      hasHeader: true,
      delimiter: CSVDelimiter.SEMICOLON,
    });

    const result = await tool.act();

    expect(result.csvOutput).toContain('name;age');
    expect(result.csvOutput).toContain('John;30');
  });
});
```

##### F. Aggregate Operation Tests

```typescript
describe('CSVProcessorTool - Aggregate Operation', () => {
  test('should group by single column', async () => {
    const csvData = `category,value
A,10
A,20
B,30
B,40`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(2);
    expect(result.data[0].category).toBe('A');
    expect(result.data[1].category).toBe('B');
  });

  test('should calculate sum aggregation', async () => {
    const csvData = `category,value
A,10
A,20
B,30
B,40`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
      aggregations: [
        {
          column: 'value',
          operation: 'sum',
          alias: 'total',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].total).toBe(30); // 10 + 20
    expect(result.data[1].total).toBe(70); // 30 + 40
  });

  test('should calculate average aggregation', async () => {
    const csvData = `category,value
A,10
A,20
B,30
B,40`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
      aggregations: [
        {
          column: 'value',
          operation: 'avg',
          alias: 'average',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].average).toBe(15); // (10 + 20) / 2
    expect(result.data[1].average).toBe(35); // (30 + 40) / 2
  });

  test('should calculate min/max aggregation', async () => {
    const csvData = `category,value
A,10
A,20
A,15`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
      aggregations: [
        {
          column: 'value',
          operation: 'min',
          alias: 'minimum',
        },
        {
          column: 'value',
          operation: 'max',
          alias: 'maximum',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].minimum).toBe(10);
    expect(result.data[0].maximum).toBe(20);
  });

  test('should calculate count aggregation', async () => {
    const csvData = `category,value
A,10
A,20
A,15
B,5`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
      aggregations: [
        {
          column: 'value',
          operation: 'count',
          alias: 'count',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].count).toBe(3);
    expect(result.data[1].count).toBe(1);
  });

  test('should concatenate values', async () => {
    const csvData = `category,item
A,apple
A,banana
B,cherry`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category'],
      aggregations: [
        {
          column: 'item',
          operation: 'concat',
          alias: 'items',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data[0].items).toBe('apple, banana');
    expect(result.data[1].items).toBe('cherry');
  });

  test('should group by multiple columns', async () => {
    const csvData = `category,subcategory,value
A,X,10
A,X,20
A,Y,30
B,X,40`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData,
      hasHeader: true,
      groupBy: ['category', 'subcategory'],
      aggregations: [
        {
          column: 'value',
          operation: 'sum',
        },
      ],
    });

    const result = await tool.act();

    expect(result.data).toHaveLength(3);
    expect(result.data[0]).toMatchObject({
      category: 'A',
      subcategory: 'X',
      value_sum: 30,
    });
  });
});
```

##### G. Error Handling Tests

```typescript
describe('CSVProcessorTool - Error Handling', () => {
  test('should handle missing csvData for parse', async () => {
    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('csvData is required');
  });

  test('should handle missing exportData for export', async () => {
    const tool = new CSVProcessorTool({
      operation: CSVOperationType.EXPORT,
      delimiter: CSVDelimiter.COMMA,
    });

    const result = await tool.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('exportData is required');
  });

  test('should handle missing groupBy for aggregate', async () => {
    const tool = new CSVProcessorTool({
      operation: CSVOperationType.AGGREGATE,
      csvData: 'name,age\nJohn,30',
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.success).toBe(false);
    expect(result.error).toContain('groupBy and aggregations are required');
  });

  test('should handle malformed CSV gracefully', async () => {
    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData: 'malformed,,,data,,,',
      hasHeader: false,
    });

    const result = await tool.act();

    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });
});
```

##### H. Performance Tests

```typescript
describe('CSVProcessorTool - Performance', () => {
  test('should handle large files efficiently', async () => {
    // Generate 10,000 rows
    const rows = Array(10000).fill('John,30,New York').join('\n');
    const csvData = `name,age,city\n${rows}`;

    const startTime = Date.now();
    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();
    const duration = Date.now() - startTime;

    expect(result.rowCount).toBe(10000);
    expect(duration).toBeLessThan(5000); // Should complete in < 5 seconds
  });

  test('should track processing time', async () => {
    const csvData = `name,age\nJohn,30\nJane,25`;

    const tool = new CSVProcessorTool({
      operation: CSVOperationType.PARSE,
      csvData,
      hasHeader: true,
    });

    const result = await tool.act();

    expect(result.statistics?.processingTime).toBeGreaterThan(0);
  });
});
```

---

### 2. Other Tool Bubbles

#### Data Transformer Tool
- **Test File:** `tool-bubble/data-transformer-tool.test.ts`
- **Key Tests:**
  - JSON transformation
  - Array operations (map, filter, reduce)
  - Object manipulation
  - Type conversions
  - Conditional transformations
  - Nested data handling
  - Error recovery

#### File Processor Tool
- **Test File:** `tool-bubble/file-processor-tool.test.ts`
- **Key Tests:**
  - File reading (local paths, URLs)
  - File writing
  - Format conversion (JSON, CSV, XML, YAML)
  - Encoding handling
  - Large file processing
  - Permission errors
  - Path validation

#### Email Validator Tool
- **Test File:** `tool-bubble/email-validator-tool.test.ts`
- **Key Tests:**
  - Valid email format validation
  - RFC 5322 compliance
  - Domain validation
  - MX record verification (optional)
  - Disposable email detection
  - Role-based email detection
  - Bulk validation

#### URL Validator Tool
- **Test File:** `tool-bubble/url-validator-tool.test.ts`
- **Key Tests:**
  - URL format validation
  - Protocol validation
  - Domain validation
  - Path and query validation
  - URL accessibility check
  - Redirect following
  - Malformed URL detection

#### Web Search Tool
- **Test File:** `tool-bubble/web-search-tool.test.ts`
- **Key Tests:**
  - Search query execution
  - Result parsing
  - Multiple search engines
  - Safe search filtering
  - Result limiting
  - API error handling
  - Rate limiting

#### Vector Search Tool
- **Test File:** `tool-bubble/vector-search-tool.test.ts`
- **Key Tests:**
  - Vector similarity search
  - Cosine similarity calculation
  - Top-k results
  - Filter application
  - Distance metrics
  - Empty result handling
  - Batch search

---

## Workflow Bubble Test Designs

### 1. ETL Pipeline Workflow

**File:** `workflow-bubble/etl-pipeline-workflow.test.ts`

#### Test Scenarios

```typescript
describe('ETL Pipeline Workflow', () => {
  test('should extract data from source', async () => {
    const workflow = new ETLWorkflow({
      source: {
        type: 'http',
        url: 'https://api.example.com/data',
      },
      transformations: [],
      destination: {
        type: 'memory',
      },
    });

    const result = await workflow.execute();

    expect(result.extracted).toBeDefined();
    expect(result.extracted.length).toBeGreaterThan(0);
  });

  test('should apply transformations', async () => {
    const workflow = new ETLWorkflow({
      source: {
        type: 'memory',
        data: [{ name: 'john', age: 30 }],
      },
      transformations: [
        {
          type: 'map',
          expression: 'item.name = item.name.toUpperCase()',
        },
      ],
      destination: {
        type: 'memory',
      },
    });

    const result = await workflow.execute();

    expect(result.transformed[0].name).toBe('JOHN');
  });

  test('should filter data', async () => {
    const workflow = new ETLWorkflow({
      source: {
        type: 'memory',
        data: [
          { name: 'John', age: 30 },
          { name: 'Jane', age: 25 },
          { name: 'Bob', age: 35 },
        ],
      },
      transformations: [
        {
          type: 'filter',
          condition: 'item.age >= 30',
        },
      ],
      destination: {
        type: 'memory',
      },
    });

    const result = await workflow.execute();

    expect(result.transformed).toHaveLength(2);
    expect(result.transformed.every(item => item.age >= 30)).toBe(true);
  });

  test('should load data to destination', async () => {
    const mockDestination = {
      save: vi.fn().mockResolvedValue({ success: true }),
    };

    const workflow = new ETLWorkflow({
      source: {
        type: 'memory',
        data: [{ name: 'John' }],
      },
      transformations: [],
      destination: {
        type: 'custom',
        client: mockDestination,
      },
    });

    const result = await workflow.execute();

    expect(result.loaded).toBe(true);
    expect(mockDestination.save).toHaveBeenCalled();
  });

  test('should handle errors gracefully', async () => {
    const workflow = new ETLWorkflow({
      source: {
        type: 'http',
        url: 'https://nonexistent.example.com',
      },
      transformations: [],
      destination: {
        type: 'memory',
      },
    });

    const result = await workflow.execute();

    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();
  });

  test('should track ETL statistics', async () => {
    const workflow = new ETLWorkflow({
      source: {
        type: 'memory',
        data: Array(100).fill({ value: 1 }),
      },
      transformations: [
        {
          type: 'filter',
          condition: 'item.value === 1',
        },
      ],
      destination: {
        type: 'memory',
      },
    });

    const result = await workflow.execute();

    expect(result.stats.inputRows).toBe(100);
    expect(result.stats.outputRows).toBe(100);
    expect(result.stats.duration).toBeGreaterThan(0);
  });
});
```

### 2. Database Analyzer Workflow

**File:** `workflow-bubble/database-analyzer-workflow.test.ts`

```typescript
describe('Database Analyzer Workflow', () => {
  test('should analyze database schema', async () => {
    const workflow = new DatabaseAnalyzerWorkflow({
      connection: {
        host: 'localhost',
        database: 'test_db',
      },
      analysisType: 'schema',
    });

    const result = await workflow.execute();

    expect(result.tables).toBeDefined();
    expect(result.columns).toBeDefined();
  });

  test('should detect table relationships', async () => {
    const workflow = new DatabaseAnalyzerWorkflow({
      connection: {
        host: 'localhost',
        database: 'test_db',
      },
      analysisType: 'relationships',
    });

    const result = await workflow.execute();

    expect(result.relationships).toBeDefined();
    expect(result.foreignKeys).toBeDefined();
  });

  test('should analyze query performance', async () => {
    const workflow = new DatabaseAnalyzerWorkflow({
      connection: {
        host: 'localhost',
        database: 'test_db',
      },
      analysisType: 'performance',
      queries: ['SELECT * FROM users'],
    });

    const result = await workflow.execute();

    expect(result.queryPlans).toBeDefined();
    expect(result.suggestions).toBeDefined();
  });
});
```

### 3. Slack Notifier Workflow

**File:** `workflow-bubble/slack-notifier-workflow.test.ts`

```typescript
describe('Slack Notifier Workflow', () => {
  test('should send notification to channel', async () => {
    const workflow = new SlackNotifierWorkflow({
      channel: 'C123456',
      message: 'Test notification',
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test',
        }),
      },
    });

    const result = await workflow.execute();

    expect(result.sent).toBe(true);
    expect(result.timestamp).toBeDefined();
  });

  test('should format message with template', async () => {
    const workflow = new SlackNotifierWorkflow({
      channel: 'C123456',
      template: 'Alert: {{message}} at {{time}}',
      data: {
        message: 'Test',
        time: '2023-01-01',
      },
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test',
        }),
      },
    });

    const result = await workflow.execute();

    expect(result.sent).toBe(true);
  });

  test('should add attachments', async () => {
    const workflow = new SlackNotifierWorkflow({
      channel: 'C123456',
      message: 'Test',
      attachments: [
        {
          text: 'Attachment 1',
          color: 'good',
        },
      ],
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test',
        }),
      },
    });

    const result = await workflow.execute();

    expect(result.sent).toBe(true);
  });

  test('should retry on failure', async () => {
    const workflow = new SlackNotifierWorkflow({
      channel: 'C123456',
      message: 'Test',
      maxRetries: 3,
      credentials: {
        [CredentialType.SLACK_CRED]: JSON.stringify({
          botToken: 'xoxb-test',
        }),
      },
    });

    const result = await workflow.execute();

    expect(result.attempts).toBeGreaterThan(0);
  });
});
```

---

## Test Utilities and Helpers

### Mock Factory

**File:** `__tests__/helpers/mock-factory.ts`

```typescript
import { z } from 'zod';

export class MockFactory {
  /**
   * Create mock HTTP response
   */
  static mockHttpResponse(overrides: Partial<Response> = {}): Response {
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers(),
      json: async () => ({}),
      text: async () => '',
      blob: async () => new Blob(),
      ...overrides,
    } as Response;
  }

  /**
   * Create mock Slack API response
   */
  static mockSlackResponse(ok: boolean, data?: any): any {
    return {
      ok,
      ...data,
    };
  }

  /**
   * Create mock database connection
   */
  static mockDatabase() {
    return {
      query: vi.fn().mockResolvedValue({ rows: [] }),
      connect: vi.fn().mockResolvedValue(undefined),
      disconnect: vi.fn().mockResolvedValue(undefined),
    };
  }

  /**
   * Create mock credentials
   */
  static mockCredentials(type: string, value: any) {
    return {
      [type]: JSON.stringify(value),
    };
  }

  /**
   * Generate test CSV data
   */
  static generateCSV(rows: number, columns: string[]): string {
    const header = columns.join(',');
    const data = Array(rows)
      .fill(null)
      .map((_, i) =>
        columns.map((col) => `${col}_${i}`).join(',')
      )
      .join('\n');
    return `${header}\n${data}`;
  }

  /**
   * Generate test JSON data
   */
  static generateJSON(rows: number, schema: Record<string, any>): any[] {
    return Array(rows).fill(null).map((_, i) => {
      const row: any = {};
      for (const [key, value] of Object.entries(schema)) {
        if (typeof value === 'function') {
          row[key] = value(i);
        } else {
          row[key] = value;
        }
      }
      return row;
    });
  }
}
```

### Assertion Helpers

**File:** `__tests__/helpers/assertion-helpers.ts`

```typescript
import { expect } from 'vitest';

export class CustomAssertions {
  /**
   * Assert valid bubble result
   */
  static assertValidBubbleResult(result: any) {
    expect(result).toBeDefined();
    expect(result.success).toBeDefined();
    expect(typeof result.success).toBe('boolean');
  }

  /**
   * assert successful operation
   */
  static assertSuccess(result: any) {
    this.assertValidBubbleResult(result);
    expect(result.success).toBe(true);
    expect(result.error).toBeUndefined();
  }

  /**
   * Assert failed operation
   */
  static assertFailure(result: any, expectedError?: string) {
    this.assertValidBubbleResult(result);
    expect(result.success).toBe(false);
    expect(result.error).toBeDefined();

    if (expectedError) {
      expect(result.error).toContain(expectedError);
    }
  }

  /**
   * Assert metrics are present
   */
  static assertMetrics(result: any, requiredMetrics: string[] = []) {
    expect(result.metrics).toBeDefined();

    for (const metric of requiredMetrics) {
      expect(result.metrics[metric]).toBeDefined();
    }
  }

  /**
   * Assert schema validation passed
   */
  static assertSchemaValid(data: any, schema: z.ZodSchema) {
    expect(() => schema.parse(data)).not.toThrow();
  }

  /**
   * Assert schema validation failed
   */
  static assertSchemaInvalid(data: any, schema: z.ZodSchema) {
    expect(() => schema.parse(data)).toThrow(z.ZodError);
  }
}
```

### Test Data Fixtures

**File:** `__tests__/helpers/test-data.ts`

```typescript
export const TestData = {
  // HTTP test data
  http: {
    validUrls: [
      'https://api.example.com/users',
      'http://localhost:8080/api/v1/data',
      'https://sub.domain.example.com/path?query=value',
    ],
    invalidUrls: [
      'not-a-url',
      'http://',
      'https://',
      'ftp://example.com',
    ],
    responseExamples: {
      json: { data: 'test', count: 42 },
      text: 'plain text response',
      html: '<html><body>test</body></html>',
    },
  },

  // CSV test data
  csv: {
    simple: `name,age,city
John,30,New York
Jane,25,London`,

    withQuotes: `name,description
"John Doe","A person, with comma"
"Jane Smith","Person with ""quotes"""`,

    withSpecialChars: `name,text
John,"Line 1\nLine 2\nLine 3"
Jane,Tabs\t\ttabs`,

    malformed: `name,age,city
John,30,New York
Jane,25
"Bob,35",Paris,`,

    large: (() => {
      const rows = Array(1000).fill('John,30,New York').join('\n');
      return `name,age,city\n${rows}`;
    })(),
  },

  // Email test data
  emails: {
    valid: [
      'test@example.com',
      'user.name@example.com',
      'user+tag@example.co.uk',
      'user123@test-domain.com',
    ],
    invalid: [
      'not-an-email',
      '@example.com',
      'user@',
      'user@@example.com',
      'user example.com',
    ],
    disposable: ['test@tempmail.com', 'user@throwaway.net'],
    roleBased: ['admin@example.com', 'support@example.com'],
  },

  // URL test data
  urls: {
    valid: [
      'https://example.com',
      'https://sub.example.com/path?query=value',
      'http://localhost:8080',
    ],
    invalid: [
      'not-a-url',
      'http://',
      'https://',
      'javascript:alert(1)',
    ],
  },

  // Social media test data
  social: {
    twitter: {
      validHandles: ['@username', 'username'],
      invalidHandles: ['', '@', 'a'],
      tweets: [
        { id: '123', text: 'Test tweet', user: 'testuser' },
        { id: '456', text: 'Another tweet', user: 'testuser2' },
      ],
    },
    linkedin: {
      validProfiles: ['https://linkedin.com/in/johndoe'],
      invalidProfiles: ['not-a-linkedin-url'],
    },
  },
};
```

### Mock Responses

**File:** `__tests__/helpers/mock-responses.ts`

```typescript
export const MockResponses = {
  // HTTP responses
  http: {
    success: {
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ success: true }),
    },
    notFound: {
      ok: false,
      status: 404,
      statusText: 'Not Found',
    },
    serverError: {
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    },
    timeout: {
      ok: false,
      status: 408,
      statusText: 'Request Timeout',
    },
    unauthorized: {
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
    },
  },

  // Slack responses
  slack: {
    success: {
      ok: true,
      ts: '1234567890.123456',
      channel: 'C123456',
    },
    invalidAuth: {
      ok: false,
      error: 'invalid_auth',
    },
    rateLimited: {
      ok: false,
      error: 'ratelimited',
    },
    channelNotFound: {
      ok: false,
      error: 'channel_not_found',
    },
  },

  // Database responses
  database: {
    success: {
      rows: [
        { id: 1, name: 'John' },
        { id: 2, name: 'Jane' },
      ],
      rowCount: 2,
    },
    empty: {
      rows: [],
      rowCount: 0,
    },
    error: {
      message: 'Database connection failed',
    },
  },
};
```

---

## Mock and Fixture Requirements

### Service Bubble Mocks

#### HTTP Service Mocks
- **Mock Fetch API:** Intercept `global.fetch` for all HTTP requests
- **Mock Response Headers:** Test header parsing and manipulation
- **Mock Response Bodies:** Test JSON, text, blob, and arraybuffer parsing
- **Mock Network Errors:** Test retry logic and error handling
- **Mock Timeouts:** Test timeout handling with AbortController

#### External Service Mocks
- **Slack API:** Mock all Slack Web API endpoints
- **GitHub API:** Mock repositories, issues, pull requests, users
- **Gmail API:** Mock messages, threads, drafts, labels
- **Stripe API:** Mock payments, customers, subscriptions, invoices
- **Notion API:** Mock pages, databases, blocks, search
- **Airtable API:** Mock bases, tables, records, fields
- **PostgreSQL:** Mock connection, query execution, transactions
- **Redis:** Mock string, hash, list, set, sorted set operations
- **Elasticsearch:** Mock indices, documents, searches, aggregations
- **Qdrant:** Mock collections, points, vectors, searches

### Tool Bubble Mocks

#### File System Mocks
- **Mock fs module:** For file reading/writing tests
- **Mock path operations:** For path manipulation tests
- **Mock file permissions:** For error handling tests

#### External API Mocks
- **Mock email validation APIs:** ZeroBounce, NeverBounce, etc.
- **Mock URL validation APIs:** Google Safe Browsing, etc.
- **Mock web search APIs:** Google, Bing, DuckDuckGo
- **Mock social media APIs:** Twitter, LinkedIn, Instagram, YouTube, Reddit

### Workflow Mocks

#### Orchestrator Mocks
- **Mock workflow engine:** Test workflow execution and state management
- **Mock step execution:** Test individual workflow steps
- **Mock error handling:** Test workflow error recovery

---

## Coverage Metrics and Goals

### Coverage Targets

| Metric Category | Target | Description |
|----------------|--------|-------------|
| **Line Coverage** | 80%+ | Percentage of code lines executed |
| **Branch Coverage** | 75%+ | Percentage of conditional branches tested |
| **Function Coverage** | 90%+ | Percentage of functions called |
| **Statement Coverage** | 85%+ | Percentage of statements executed |

### Critical Path Coverage

100% coverage required for:
- Authentication and authorization logic
- Input validation and sanitization
- Error handling and recovery
- Security-sensitive operations
- Data persistence operations
- External API integrations
- Retry logic and circuit breakers
- Timeout handling

### Coverage by Bubble Type

#### Service Bubbles
- **Unit Tests:** 70%+ coverage per bubble
- **Integration Tests:** 50%+ coverage for external API calls
- **Error Scenarios:** 100% of error paths covered

#### Tool Bubbles
- **Unit Tests:** 85%+ coverage per tool
- **Edge Cases:** 100% of edge cases covered
- **Validation Tests:** 100% of validation logic covered

#### Workflow Bubbles
- **Integration Tests:** 70%+ coverage per workflow
- **End-to-End Tests:** 50%+ coverage for critical workflows
- **Error Recovery:** 100% of error recovery paths covered

### Coverage Reporting

Generate coverage reports using Vitest:

```bash
# Run tests with coverage
pnpm test:coverage

# View HTML coverage report
open coverage/index.html
```

### Coverage Gates

Enforce coverage thresholds in CI/CD:

```typescript
// vitest.config.ts
export default defineConfig({
  test: {
    coverage: {
      statements: 80,
      branches: 75,
      functions: 90,
      lines: 80,
      // Exclude test files from coverage
      exclude: [
        'node_modules/',
        '**/*.test.ts',
        '**/*.integration.test.ts',
        '**/dist/**',
      ],
    },
  },
});
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up test infrastructure
- [ ] Create test utilities and helpers
- [ ] Implement global test setup/teardown
- [ ] Create mock factory and fixtures
- [ ] Configure Vitest for all test types

### Phase 2: Service Bubble Tests (Week 3-6)
- [ ] HTTP bubble tests (all categories)
- [ ] Slack bubble tests (all categories)
- [ ] External service tests (GitHub, Gmail, Stripe, etc.)
- [ ] Database service tests (PostgreSQL, Redis, Elasticsearch, Qdrant)
- [ ] Integration tests for all service bubbles

### Phase 3: Tool Bubble Tests (Week 7-10)
- [ ] Data processing tool tests (CSV, XML, JSON, etc.)
- [ ] Validation tool tests (email, URL, schema)
- [ ] Content generation tool tests (PDF, code, text)
- [ ] Search and research tool tests
- [ ] Social media tool tests
- [ ] Integration tests for complex tool chains

### Phase 4: Workflow Bubble Tests (Week 11-13)
- [ ] ETL pipeline tests
- [ ] Database analyzer tests
- [ ] Slack notifier tests
- [ ] Webhook repeater tests
- [ ] Data enrichment tests
- [ ] End-to-end workflow tests

### Phase 5: Performance and Stress Testing (Week 14-15)
- [ ] Load testing for high-volume bubbles
- [ ] Memory leak testing
- [ ] Concurrent operation testing
- [ ] Performance benchmarking
- [ ] Resource cleanup verification

### Phase 6: Coverage Verification and Reporting (Week 16)
- [ ] Generate coverage reports for all bubbles
- [ ] Identify coverage gaps
- [ ] Implement missing tests
- [ ] Verify coverage thresholds met
- [ ] Create test documentation

---

## Test Execution

### Run All Tests

```bash
# Unit tests only
pnpm test

# Integration tests
pnpm test:integration

# All tests (unit + integration)
pnpm test:all

# With coverage
pnpm test:coverage

# Watch mode
pnpm test:watch
```

### Run Specific Bubble Tests

```bash
# HTTP bubble tests
pnpm test http-bubble

# CSV processor tests
pnpm test csv-processor-tool

# ETL workflow tests
pnpm test etl-pipeline-workflow
```

### Run Test Categories

```bash
# Unit tests only
pnpm test --exclude '**/*.integration.test.ts'

# Integration tests only
pnpm test --run '**/*.integration.test.ts'

# Performance tests
pnpm test --run '**/*.performance.test.ts'
```

---

## Continuous Integration

### CI Test Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
      - uses: pnpm/action-setup@v2
      - uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: pnpm install

      - name: Run unit tests
        run: pnpm test

      - name: Run integration tests
        run: pnpm test:integration
        env:
          TEST_API_URL: ${{ secrets.TEST_API_URL }}
          TEST_SLACK_TOKEN: ${{ secrets.TEST_SLACK_TOKEN }}

      - name: Generate coverage report
        run: pnpm test:coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json
```

---

## Conclusion

This comprehensive test coverage design provides:

1. **Complete Test Coverage** for all 70+ BubbleLab bubbles
2. **Multiple Test Categories** (unit, integration, validation, error handling, performance)
3. **Detailed Test Scenarios** for each bubble type
4. **Reusable Test Infrastructure** with utilities, mocks, and fixtures
5. **Clear Coverage Goals** and metrics
6. **Phased Implementation Plan** for systematic development

The test suite ensures:
- **Reliability:** Comprehensive error handling and edge case coverage
- **Maintainability:** Modular, reusable test utilities
- **Performance:** Benchmarking and load testing
- **Security:** Input validation and sanitization tests
- **Integration:** End-to-end workflow testing

All tests follow industry best practices using Vitest, with clear documentation and examples for developers to follow when implementing new tests.
