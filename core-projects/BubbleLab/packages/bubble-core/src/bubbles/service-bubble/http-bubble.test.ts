import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HttpBubble } from './http-bubble.js';
import { CredentialType } from '@bubblelab/shared-schemas';

// Mock fetch for testing
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('HttpBubble - Advanced Features', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    // Reset circuit breaker states
    (HttpBubble as any).circuitBreakerStates.clear();
  });

  describe('Static Properties', () => {
    it('should have correct static properties', () => {
      expect(HttpBubble.bubbleName).toBe('http');
      expect(HttpBubble.service).toBe('nodex-core');
      expect(HttpBubble.type).toBe('service');
      expect(HttpBubble.alias).toBe('http');
      expect(HttpBubble.shortDescription).toContain('Production-ready');
    });
  });

  describe('Parameter Validation', () => {
    it('should validate required parameters', () => {
      const validParams = {
        operation: 'get' as const,
        url: 'https://api.example.com/data',
        timeout: 5000,
      };

      const result = HttpBubble.schema.safeParse(validParams);
      expect(result.success).toBe(true);

      if (result.success) {
        expect(result.data.url).toBe('https://api.example.com/data');
        expect(result.data.timeout).toBe(5000);
      }
    });

    it('should reject invalid URL', () => {
      const invalidParams = {
        operation: 'get' as const,
        url: 'not-a-url',
      };

      const result = HttpBubble.schema.safeParse(invalidParams);
      expect(result.success).toBe(false);
    });

    it('should validate retry configuration', () => {
      const params = {
        operation: 'post' as const,
        url: 'https://api.example.com/data',
        retryEnabled: true,
        maxRetries: 5,
        retryStrategy: 'exponential' as const,
        retryDelay: 2000,
        retryMultiplier: 3,
      };

      const result = HttpBubble.schema.safeParse(params);
      expect(result.success).toBe(true);

      if (result.success) {
        expect(result.data.maxRetries).toBe(5);
        expect(result.data.retryStrategy).toBe('exponential');
        expect(result.data.retryMultiplier).toBe(3);
      }
    });

    it('should validate circuit breaker configuration', () => {
      const params = {
        operation: 'get' as const,
        url: 'https://api.example.com/data',
        circuitBreakerEnabled: true,
        circuitBreakerThreshold: 10,
        circuitBreakerTimeout: 120000,
      };

      const result = HttpBubble.schema.safeParse(params);
      expect(result.success).toBe(true);

      if (result.success) {
        expect(result.data.circuitBreakerEnabled).toBe(true);
        expect(result.data.circuitBreakerThreshold).toBe(10);
        expect(result.data.circuitBreakerTimeout).toBe(120000);
      }
    });
  });

  describe('Basic HTTP Operations', () => {
    it('should make successful GET request', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"message": "success"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.status).toBe(200);
      expect(result.statusText).toBe('OK');
      expect(result.data).toEqual({ message: 'success' });
      expect(result.metrics.totalAttempts).toBe(1);
      expect(result.metrics.retryCount).toBe(0);
    });

    it('should make POST request with JSON body', async () => {
      const mockResponse = {
        ok: true,
        status: 201,
        statusText: 'Created',
        text: vi.fn().mockResolvedValue('{"id": 123}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/create',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'post',
        url: 'https://api.example.com/create',
        body: { name: 'Test', value: 42 },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.status).toBe(201);
      expect(result.data).toEqual({ id: 123 });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('https://api.example.com/create'),
        expect.objectContaining({
          method: 'POST',
          body: '{"name":"Test","value":42}',
        })
      );
    });

    it('should handle PUT request', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"updated": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/update/123',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'put',
        url: 'https://api.example.com/update/123',
        body: { status: 'active' },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: 'PUT' })
      );
    });

    it('should handle PATCH request', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"patched": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/patch/123',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'patch',
        url: 'https://api.example.com/patch/123',
        body: { field: 'updated' },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: 'PATCH' })
      );
    });

    it('should handle DELETE request', async () => {
      const mockResponse = {
        ok: true,
        status: 204,
        statusText: 'No Content',
        text: vi.fn().mockResolvedValue(''),
        headers: new Headers(),
        url: 'https://api.example.com/delete/123',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'delete',
        url: 'https://api.example.com/delete/123',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(result.status).toBe(204);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: 'DELETE' })
      );
    });

    it('should handle HEAD request', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue(''),
        headers: new Headers([['content-length', '1234']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'head',
        url: 'https://api.example.com/data',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: 'HEAD' })
      );
    });

    it('should handle OPTIONS request', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue(''),
        headers: new Headers([['allow', 'GET, POST, PUT, DELETE']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'options',
        url: 'https://api.example.com/data',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ method: 'OPTIONS' })
      );
    });
  });

  describe('Query Parameters', () => {
    it('should append query parameters to URL', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('[]'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/search?page=1&limit=10&sort=name',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/search',
        queryParams: {
          page: 1,
          limit: 10,
          sort: 'name',
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/search?page=1&limit=10&sort=name',
        expect.any(Object)
      );
    });

    it('should handle query parameters with existing URL params', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('[]'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/search?existing=param&added=value',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/search?existing=param',
        queryParams: {
          added: 'value',
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.example.com/search?existing=param&added=value',
        expect.any(Object)
      );
    });
  });

  describe('Headers', () => {
    it('should include custom headers', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"data": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        headers: {
          'X-Custom-Header': 'custom-value',
          'X-Request-ID': '12345',
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      const fetchCall = mockFetch.mock.calls[0];
      expect(fetchCall[0]).toContain('https://api.example.com/data');
      expect(fetchCall[1]?.headers).toMatchObject({
        'X-Custom-Header': 'custom-value',
        'X-Request-ID': '12345',
      });
    });

    it('should add Bearer authentication', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"data": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        authType: 'bearer',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'my-token',
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      const fetchCall = mockFetch.mock.calls[mockFetch.mock.calls.length - 1];
      expect(fetchCall[1]?.headers).toMatchObject({
        'Authorization': 'Bearer my-token',
      });
    });

    it('should add API Key authentication', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"data": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        authType: 'api-key',
        credentials: {
          [CredentialType.CUSTOM_AUTH_KEY]: 'api-key-123',
        },
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(true);
      const fetchCall = mockFetch.mock.calls[mockFetch.mock.calls.length - 1];
      expect(fetchCall[1]?.headers).toMatchObject({
        'X-API-Key': 'api-key-123',
      });
    });
  });

  describe('Retry Logic', () => {
    it('should retry on retryable status codes', async () => {
      const mockResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: vi.fn().mockResolvedValue('{"error": "Service unavailable"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        retryEnabled: true,
        maxRetries: 2,
        retryDelay: 100,
      });

      const result = await httpBubble.performAction();

      expect(mockFetch).toHaveBeenCalledTimes(3); // 1 initial + 2 retries
      expect(result.metrics.totalAttempts).toBe(3);
      expect(result.metrics.retryCount).toBe(2);
    });

    it('should not retry when retry is disabled', async () => {
      const mockResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: vi.fn().mockResolvedValue('{"error": "Service unavailable"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        retryEnabled: false,
      });

      const result = await httpBubble.performAction();

      expect(mockFetch).toHaveBeenCalledTimes(1); // Only initial attempt
      expect(result.metrics.retryCount).toBe(0);
    });

    it('should stop retrying after successful attempt', async () => {
      const errorResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: vi.fn().mockResolvedValue('{"error": "Service unavailable"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      const successResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"data": "success"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValueOnce(errorResponse).mockResolvedValueOnce(successResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        retryEnabled: true,
        maxRetries: 3,
        retryDelay: 100,
      });

      const result = await httpBubble.performAction();

      expect(mockFetch).toHaveBeenCalledTimes(2); // 1 initial + 1 retry
      expect(result.success).toBe(true);
      expect(result.metrics.retryCount).toBe(1);
    });
  });

  describe('Circuit Breaker', () => {
    it('should open circuit after threshold failures', async () => {
      const mockResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: vi.fn().mockResolvedValue('{"error": "Service unavailable"}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        circuitBreakerEnabled: true,
        circuitBreakerThreshold: 3,
        retryEnabled: false,
      });

      // Trigger failures to open circuit
      await httpBubble.performAction();
      await httpBubble.performAction();
      await httpBubble.performAction();

      // Fourth attempt should be blocked by circuit breaker
      const result = await httpBubble.performAction();

      expect(result.error).toContain('Circuit breaker is open');
      expect(result.metrics.circuitBreakerTripped).toBe(true);
    });

    it('should allow request after circuit breaker timeout', async () => {
      // This test would need to manipulate time or use a short timeout
      // For now, we'll just verify the structure
      const mockResponse = {
        ok: false,
        status: 503,
        statusText: 'Service Unavailable',
        text: vi.fn().mockResolvedValue(''),
        headers: new Headers(),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        circuitBreakerEnabled: true,
        circuitBreakerThreshold: 2,
        circuitBreakerTimeout: 100, // Short timeout for testing
        retryEnabled: false,
      });

      // Trigger circuit breaker
      await httpBubble.performAction();
      await httpBubble.performAction();

      // Circuit should be open
      const blockedResult = await httpBubble.performAction();
      expect(blockedResult.error).toContain('Circuit breaker is open');

      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 150));

      // Should be allowed now (half-open state)
      const result = await httpBubble.performAction();
      // Result depends on fetch, but circuit should allow the attempt
      expect(mockFetch).toHaveBeenCalledTimes(4); // 2 initial + 1 blocked + 1 after timeout
    }, 10000);
  });

  describe('Timeout Handling', () => {
    it('should handle request timeout', async () => {
      const controller = new AbortController();
      const mockFetchWithTimeout = vi.fn(() =>
        new Promise((_, reject) => {
          setTimeout(() => {
            const error = new Error('Request timeout');
            error.name = 'AbortError';
            reject(error);
          }, 100);
        })
      );

      global.fetch = mockFetchWithTimeout;

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        timeout: 50,
        retryEnabled: false,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toContain('timeout');
      expect(result.errorCode).toBe('AbortError');
    });
  });

  describe('Response Types', () => {
    it('should parse JSON response', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"key": "value"}'),
        headers: {
          get: vi.fn((name: string) => name === 'content-type' ? 'application/json' : null),
          forEach: vi.fn((callback: Function) => {
            callback('application/json', 'content-type');
          }),
        },
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        responseType: 'json',
      });

      const result = await httpBubble.performAction();

      expect(result.data).toEqual({ key: 'value' });
    });

    it('should parse text response', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('plain text response'),
        headers: {
          get: vi.fn((name: string) => name === 'content-type' ? 'text/plain' : null),
          forEach: vi.fn((callback: Function) => {
            callback('text/plain', 'content-type');
          }),
        },
        url: 'https://api.example.com/data',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        responseType: 'text',
      });

      const result = await httpBubble.performAction();

      expect(result.data).toBe('plain text response');
    });
  });

  describe('Error Handling', () => {
    it('should handle network errors', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
        retryEnabled: false,
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.error).toBe('Network error');
      expect(result.status).toBe(0);
    });

    it('should include detailed error information', async () => {
      const mockResponse = {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: vi.fn().mockResolvedValue('{"error": "Resource not found"}'),
        headers: {
          get: vi.fn((name: string) => name === 'content-type' ? 'application/json' : null),
          forEach: vi.fn((callback: Function) => {
            callback('application/json', 'content-type');
          }),
        },
        url: 'https://api.example.com/notfound',
      };

      mockFetch.mockResolvedValue(mockResponse);

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/notfound',
      });

      const result = await httpBubble.performAction();

      expect(result.success).toBe(false);
      expect(result.status).toBe(404);
      expect(result.statusText).toBe('Not Found');
      expect(result.error).toBe('HTTP 404: Not Found');
      expect(result.data).toEqual({ error: 'Resource not found' });
    });
  });

  describe('Metrics', () => {
    it('should include accurate timing metrics', async () => {
      const mockResponse = {
        ok: true,
        status: 200,
        statusText: 'OK',
        text: vi.fn().mockResolvedValue('{"data": true}'),
        headers: new Headers([['content-type', 'application/json']]),
        url: 'https://api.example.com/data',
      };

      mockFetch.mockImplementation(async () => {
        await new Promise(resolve => setTimeout(resolve, 100));
        return mockResponse;
      });

      const httpBubble = new HttpBubble({
        operation: 'get',
        url: 'https://api.example.com/data',
      });

      const result = await httpBubble.performAction();

      expect(result.metrics.responseTime).toBeGreaterThanOrEqual(100);
      expect(result.metrics.lastAttemptTime).toBeGreaterThanOrEqual(100);
      expect(result.metrics.totalAttempts).toBe(1);
    });
  });
});
