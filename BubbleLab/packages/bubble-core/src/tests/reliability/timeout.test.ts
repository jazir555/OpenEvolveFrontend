/**
 * Timeout Tests (Bug #2)
 *
 * Tests for request timeout functionality:
 * - Request completes within configured timeout
 * - Request properly times out after configured duration
 * - Timeout error includes correlation ID
 * - Timeout doesn't prevent retries
 * - Timeout doesn't prevent circuit breaker from opening
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ApiClient, ApiClientConfig } from '../../BubbleLab/apps/bubble-studio/src/lib/api';

// Mock fetch to simulate timeout behavior
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Mock logger
vi.mock('../../BubbleLab/apps/bubble-studio/src/utils/logger', () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock token refresh
vi.mock('../../BubbleLab/apps/bubble-studio/src/lib/token-refresh', () => ({
  refreshToken: vi.fn(() => Promise.resolve('mock-token')),
}));

// Mock toast
vi.mock('react-toastify', () => ({
  toast: {
    error: vi.fn(),
  },
}));

describe('Timeout Tests (Bug #2)', () => {
  let client: ApiClient;
  let config: ApiClientConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    config = {
      baseURL: 'http://localhost:8000',
      timeout: 5000, // 5 second timeout for testing
      enableRetry: false, // Disable retry initially
      maxRetries: 3,
      retryDelay: 1000,
    };

    client = new ApiClient(config);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Request Timeout Behavior', () => {
    it('should complete successful request within timeout', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'success' }),
      });

      const promise = client.get('/api/test');
      await vi.advanceTimersByTimeAsync(1000); // Advance 1 second
      const result = await promise;

      expect(result).toEqual({ data: 'success' });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should timeout after configured duration', async () => {
      // Make fetch hang indefinitely
      mockFetch.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            // Never resolve
          })
      );

      const startTime = Date.now();
      const promise = client.get('/api/test');

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      try {
        await promise;
        expect.fail('Should have thrown timeout error');
      } catch (error) {
        const elapsed = Date.now() - startTime;
        expect(error).toBeInstanceOf(Error);
        expect(error.message).toMatch(/aborted|timeout/);
      }
    });

    it('should include correlation ID in timeout logs', async () => {
      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      mockFetch.mockImplementationOnce(
        () =>
          new Promise(() => {
            // Never resolve
          })
      );

      const promise = client.get('/api/test');
      await vi.advanceTimersByTimeAsync(6000);

      try {
        await promise;
      } catch (error) {
        // Expected
      }

      // Verify logger was called with correlation_id
      expect(logger.warn).toHaveBeenCalledWith(
        expect.objectContaining({
          msg: 'Request timeout',
          correlation_id: expect.any(String),
          timeout_ms: 5000,
        })
      );
    });
  });

  describe('Timeout Configuration', () => {
    it('should use 30 second default timeout', () => {
      const defaultClient = new ApiClient('http://localhost:8000');
      // Access private property for testing
      // @ts-ignore
      expect(defaultClient.timeout).toBe(30000);
    });

    it('should use custom timeout when provided', () => {
      const customClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 10000,
      });
      // @ts-ignore
      expect(customClient.timeout).toBe(10000);
    });

    it('should timeout faster with shorter configured duration', async () => {
      const fastClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 1000, // 1 second timeout
      });

      mockFetch.mockImplementationOnce(
        () =>
          new Promise(() => {
            // Never resolve
          })
      );

      const promise = fastClient.get('/api/test');
      await vi.advanceTimersByTimeAsync(2000);

      try {
        await promise;
        expect.fail('Should have timed out');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
      }
    });
  });

  describe('Timeout with Retry Logic', () => {
    it('should retry after timeout', async () => {
      const retryClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 1000,
        enableRetry: true,
        maxRetries: 2,
        retryDelay: 500,
      });

      // First two attempts timeout, third succeeds
      mockFetch
        .mockImplementationOnce(
          () =>
            new Promise(() => {
              // Never resolve - timeout
            })
        )
        .mockImplementationOnce(
          () =>
            new Promise(() => {
              // Never resolve - timeout
            })
        )
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: 'success' }),
        });

      const promise = retryClient.get('/api/test');

      // Advance through first timeout and retry delay
      await vi.advanceTimersByTimeAsync(1500);
      await vi.advanceTimersByTimeAsync(1500);
      await vi.advanceTimersByTimeAsync(500);

      try {
        await promise;
        expect.fail('Should have failed after all retries');
      } catch (error) {
        // Expected to fail after retries
        expect(mockFetch).toHaveBeenCalledTimes(3);
      }
    });
  });

  describe('Streaming Request Timeout', () => {
    it('should timeout streaming requests', async () => {
      mockFetch.mockImplementationOnce(
        () =>
          new Promise(() => {
            // Never resolve - simulating hanging stream
          })
      );

      const promise = client.post('/api/stream', { data: 'test' });
      await vi.advanceTimersByTimeAsync(6000);

      try {
        await promise;
        expect.fail('Should have timed out');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
      }
    });

    it('should handle partial stream data before timeout', async () => {
      const { ReadableStream } = require('stream/web');

      mockFetch.mockResolvedValueOnce({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            // Send some data then hang
            controller.enqueue(new TextEncoder().encode('partial'));
            // Never close - will timeout
          },
        }),
      });

      const promise = client.get('/api/stream');
      await vi.advanceTimersByTimeAsync(6000);

      // Should timeout waiting for stream to complete
      try {
        await promise;
        expect.fail('Should have timed out');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
      }
    });
  });

  describe('Timeout Error Handling', () => {
    it('should throw AbortError on timeout', async () => {
      mockFetch.mockImplementationOnce(
        () =>
          new Promise(() => {
            // Never resolve
          })
      );

      const promise = client.get('/api/test');
      await vi.advanceTimersByTimeAsync(6000);

      try {
        await promise;
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error.name).toBe('AbortError');
      }
    });

    it('should preserve timeout information in error', async () => {
      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      mockFetch.mockImplementationOnce(
        () =>
          new Promise(() => {
            // Never resolve
          })
      );

      const promise = client.get('/api/test');
      await vi.advanceTimersByTimeAsync(6000);

      try {
        await promise;
      } catch (error) {
        // Expected
      }

      expect(logger.warn).toHaveBeenCalledWith(
        expect.objectContaining({
          timeout_ms: 5000,
        })
      );
    });
  });

  describe('Timeout with Different HTTP Methods', () => {
    it('should timeout GET requests', async () => {
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const promise = client.get('/api/test');
      await vi.advanceTimersByTimeAsync(6000);

      await expect(promise).rejects.toThrow();
    });

    it('should timeout POST requests', async () => {
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const promise = client.post('/api/test', { data: 'test' });
      await vi.advanceTimersByTimeAsync(6000);

      await expect(promise).rejects.toThrow();
    });

    it('should timeout PUT requests', async () => {
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const promise = client.put('/api/test', { data: 'test' });
      await vi.advanceTimersByTimeAsync(6000);

      await expect(promise).rejects.toThrow();
    });

    it('should timeout DELETE requests', async () => {
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const promise = client.delete('/api/test');
      await vi.advanceTimersByTimeAsync(6000);

      await expect(promise).rejects.toThrow();
    });

    it('should timeout PATCH requests', async () => {
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const promise = client.patch('/api/test', { data: 'test' });
      await vi.advanceTimersByTimeAsync(6000);

      await expect(promise).rejects.toThrow();
    });
  });
});
