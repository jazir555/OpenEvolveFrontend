/**
 * Retry Logic Tests (Bug #3)
 *
 * Tests for retry logic with exponential backoff:
 * - Successful request doesn't retry
 * - Failed request retries configured number of times
 * - Retry delays follow exponential backoff (1s, 2s, 4s, 8s)
 * - Jitter is applied (0-30% random)
 * - Retries stop on success
 * - 429 (rate limit) errors trigger retry
 * - 5xx errors trigger retry
 * - Network errors trigger retry
 * - 4xx errors (except 429) don't retry
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ApiClient, ApiClientConfig, ApiHttpError } from '../../BubbleLab/apps/bubble-studio/src/lib/api';

// Mock fetch
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

describe('Retry Logic Tests (Bug #3)', () => {
  let client: ApiClient;
  let config: ApiClientConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    config = {
      baseURL: 'http://localhost:8000',
      timeout: 30000,
      enableRetry: true,
      maxRetries: 3,
      retryDelay: 1000,
    };

    client = new ApiClient(config);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Basic Retry Behavior', () => {
    it('should not retry on successful request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: 'success' }),
      });

      const result = await client.get('/api/test');

      expect(result).toEqual({ data: 'success' });
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('should retry on network error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Failed to fetch'));

      try {
        await client.get('/api/test');
        expect.fail('Should have thrown after retries');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect(mockFetch).toHaveBeenCalledTimes(4); // Initial + 3 retries
      }
    });

    it('should retry configured number of times', async () => {
      const customClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 30000,
        enableRetry: true,
        maxRetries: 5,
        retryDelay: 100,
      });

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await customClient.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(6); // Initial + 5 retries
    });

    it('should stop retrying on success', async () => {
      // First two fail, third succeeds
      mockFetch
        .mockRejectedValueOnce(new Error('Failed to fetch'))
        .mockRejectedValueOnce(new Error('Failed to fetch'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ data: 'success' }),
        });

      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      const promise = client.get('/api/test');

      // Advance through first attempt and retry
      await vi.advanceTimersByTimeAsync(100);
      await vi.advanceTimersByTimeAsync(1200); // 1s base + jitter
      await vi.advanceTimersByTimeAsync(2200); // 2s base + jitter
      await vi.advanceTimersByTimeAsync(100);

      const result = await promise;

      expect(result).toEqual({ data: 'success' });
      expect(mockFetch).toHaveBeenCalledTimes(3); // Stopped after success
    });
  });

  describe('Exponential Backoff', () => {
    it('should use exponential backoff delays', async () => {
      const delays: number[] = [];
      const originalSetTimeout = global.setTimeout;

      // Capture setTimeout calls to measure delays
      global.setTimeout = vi.fn((fn, delay) => {
        delays.push(delay as number);
        return originalSetTimeout(fn, delay);
      }) as any;

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      global.setTimeout = originalSetTimeout;

      // Check exponential progression: 1s, 2s, 4s (with jitter)
      // Allow for 0-30% jitter variation
      expect(delays[0]).toBeGreaterThanOrEqual(1000);
      expect(delays[0]).toBeLessThanOrEqual(1300);

      expect(delays[1]).toBeGreaterThanOrEqual(2000);
      expect(delays[1]).toBeLessThanOrEqual(2600);

      expect(delays[2]).toBeGreaterThanOrEqual(4000);
      expect(delays[2]).toBeLessThanOrEqual(5200);
    });

    it('should calculate delays correctly for each attempt', async () => {
      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      // Verify logger was called with delay information
      const logCalls = (logger.info as any).mock.calls;
      const retryLogs = logCalls.filter((call: any) => call[0]?.msg?.includes('Retry'));

      expect(retryLogs.length).toBeGreaterThan(0);

      // Check that delays increase exponentially
      const delays = retryLogs.map((call: any) => call[0]?.delay_ms);
      expect(delays[0]).toBeGreaterThan(0); // ~1000ms
      expect(delays[1]).toBeGreaterThan(delays[0]); // ~2000ms
      expect(delays[2]).toBeGreaterThan(delays[1]); // ~4000ms
    });

    it('should use custom retry delay as base', async () => {
      const customClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 30000,
        enableRetry: true,
        maxRetries: 3,
        retryDelay: 500, // 500ms base
      });

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await customClient.get('/api/test');
      } catch (error) {
        // Expected
      }

      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');
      const logCalls = (logger.info as any).mock.calls;
      const retryLogs = logCalls.filter((call: any) => call[0]?.msg?.includes('Retry'));

      // Check delays: 500ms, 1000ms, 2000ms (with jitter)
      const delays = retryLogs.map((call: any) => call[0]?.delay_ms);
      expect(delays[0]).toBeGreaterThanOrEqual(500);
      expect(delays[1]).toBeGreaterThanOrEqual(1000);
      expect(delays[2]).toBeGreaterThanOrEqual(2000);
    });
  });

  describe('Jitter Application', () => {
    it('should apply 0-30% jitter to delays', async () => {
      const delays: number[] = [];
      const originalSetTimeout = global.setTimeout;

      global.setTimeout = vi.fn((fn, delay) => {
        delays.push(delay as number);
        return originalSetTimeout(fn, delay);
      }) as any;

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      global.setTimeout = originalSetTimeout;

      // Verify jitter is applied (not exact multiples)
      const baseDelay = 1000;
      delays.forEach((delay, index) => {
        const expectedBase = baseDelay * Math.pow(2, index);
        const minJitter = expectedBase;
        const maxJitter = expectedBase * 1.3;

        expect(delay).toBeGreaterThanOrEqual(minJitter);
        expect(delay).toBeLessThanOrEqual(maxJitter);
      });
    });

    it('should have random jitter between retries', async () => {
      // Run multiple times to check randomness
      const delaySets: number[][] = [];

      for (let i = 0; i < 5; i++) {
        vi.clearAllMocks();
        const delays: number[] = [];
        const originalSetTimeout = global.setTimeout;

        global.setTimeout = vi.fn((fn, delay) => {
          delays.push(delay as number);
          return originalSetTimeout(fn, delay);
        }) as any;

        mockFetch.mockRejectedValue(new Error('Network error'));

        try {
          await client.get('/api/test');
        } catch (error) {
          // Expected
        }

        global.setTimeout = originalSetTimeout;
        delaySets.push([...delays]);
      }

      // At least some variation should exist across runs
      // (This is probabilistic, but very likely with 5 runs)
      const firstDelays = delaySets.map(set => set[0]);
      const hasVariation = new Set(firstDelays).size > 1;
      expect(hasVariation).toBe(true);
    });
  });

  describe('Retryable Error Types', () => {
    it('should retry on 429 rate limit errors', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 429,
        headers: {
          get: (name: string) => (name === 'Retry-After' ? '60' : null),
        },
        text: async () => 'Rate limit exceeded',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4); // Initial + 3 retries
    });

    it('should retry on 500 internal server error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        text: async () => 'Internal server error',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });

    it('should retry on 502 bad gateway error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 502,
        text: async () => 'Bad gateway',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });

    it('should retry on 503 service unavailable error', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 503,
        text: async () => 'Service unavailable',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });

    it('should retry on network timeout errors', async () => {
      mockFetch.mockRejectedValue(new Error('Request timeout'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });

    it('should retry on ECONNREFUSED errors', async () => {
      mockFetch.mockRejectedValue(new Error('ECONNREFUSED'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });

    it('should retry on ENOTFOUND errors', async () => {
      mockFetch.mockRejectedValue(new Error('ENOTFOUND'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(4);
    });
  });

  describe('Non-Retryable Error Types', () => {
    it('should not retry on 400 bad request', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        text: async () => 'Bad request',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });

    it('should not retry on 401 unauthorized', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });

    it('should not retry on 403 forbidden', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 403,
        text: async () => 'Forbidden',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });

    it('should not retry on 404 not found', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 404,
        text: async () => 'Not found',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });

    it('should not retry on validation errors (422)', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 422,
        text: async () => 'Validation error',
      });

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // No retries
    });
  });

  describe('Retry Logging', () => {
    it('should log retry attempts with correlation ID', async () => {
      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      const logCalls = (logger.info as any).mock.calls;
      const retryLogs = logCalls.filter((call: any) => call[0]?.msg?.includes('Retry'));

      expect(retryLogs.length).toBeGreaterThan(0);

      retryLogs.forEach((call: any) => {
        expect(call[0]).toMatchObject({
          msg: expect.stringContaining('Retry attempt'),
          correlation_id: expect.any(String),
          attempt: expect.any(Number),
          max_retries: 3,
          delay_ms: expect.any(Number),
          error: expect.any(String),
        });
      });
    });

    it('should include attempt number in logs', async () => {
      const { logger } = await import('../../BubbleLab/apps/bubble-studio/src/utils/logger');

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await client.get('/api/test');
      } catch (error) {
        // Expected
      }

      const logCalls = (logger.info as any).mock.calls;
      const retryLogs = logCalls.filter((call: any) => call[0]?.msg?.includes('Retry'));

      expect(retryLogs[0][0]?.attempt).toBe(1);
      expect(retryLogs[1][0]?.attempt).toBe(2);
      expect(retryLogs[2][0]?.attempt).toBe(3);
    });
  });

  describe('Retry Disabled', () => {
    it('should not retry when disabled', async () => {
      const noRetryClient = new ApiClient({
        baseURL: 'http://localhost:8000',
        timeout: 30000,
        enableRetry: false,
      });

      mockFetch.mockRejectedValue(new Error('Network error'));

      try {
        await noRetryClient.get('/api/test');
      } catch (error) {
        // Expected
      }

      expect(mockFetch).toHaveBeenCalledTimes(1); // Only initial attempt
    });
  });
});
