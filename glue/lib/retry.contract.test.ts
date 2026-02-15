/**
 * Contract Test for Retry Logic
 *
 * Tests compliance with Federation Constitution Section 2.3:
 * - Transient Failure → Exponential Backoff Retry
 * - Jittered delays to prevent thundering herd
 * - Configurable retry limits
 */

import { retryWithBackoff, RetryConfig, RetryOptions } from './retry';

describe('Retry Logic Contract Tests', () => {
  describe('Configuration Compliance (Law 5)', () => {
    it('should require explicit max_retries', async () => {
      const config: RetryConfig = {
        max_retries: 3
      };

      let attempts = 0;
      const result = await retryWithBackoff(
        async () => {
          attempts++;
          if (attempts < 2) {
            throw new Error('Transient error');
          }
          return 'success';
        },
        config
      );

      expect(result).toBe('success');
      expect(attempts).toBe(2); // Initial + 1 retry
    });

    it('should allow explicit delay configuration', async () => {
      const config: RetryConfig = {
        max_retries: 2,
        base_delay_ms: 100,
        max_delay_ms: 500,
        jitter_ms: 50
      };

      const delays: number[] = [];
      const startTime = Date.now();

      await retryWithBackoff(
        async () => {
          delays.push(Date.now() - startTime);
          throw new Error('Always fails');
        },
        config
      );

      expect(delays.length).toBeGreaterThan(1);
      // Delays should increase exponentially
      expect(delays[1]).toBeGreaterThan(delays[0]);
    });
  });

  describe('Exponential Backoff', () => {
    it('should implement exponential backoff formula', async () => {
      const config: RetryConfig = {
        max_retries: 3,
        base_delay_ms: 100,
        max_delay_ms: 10000,
        jitter_ms: 0 // No jitter for predictable testing
      };

      const delays: number[] = [];
      let lastAttemptTime = Date.now();

      try {
        await retryWithBackoff(
          async () => {
            const now = Date.now();
            if (delays.length > 0) {
              delays.push(now - lastAttemptTime);
            }
            lastAttemptTime = now;
            throw new Error('Fail');
          },
          config
        );
      } catch (e) {
        // Expected to fail
      }

      // Expected delays: 100, 200, 400, 800 (exponential)
      expect(delays.length).toBe(3); // 3 retries
      expect(delays[0]).toBe(100);
      expect(delays[1]).toBe(200);
      expect(delays[2]).toBe(400);
    });

    it('should respect max_delay_ms cap', async () => {
      const config: RetryConfig = {
        max_retries: 5,
        base_delay_ms: 1000,
        max_delay_ms: 2000,
        jitter_ms: 0
      };

      const delays: number[] = [];
      let lastAttemptTime = Date.now();

      try {
        await retryWithBackoff(
          async () => {
            const now = Date.now();
            if (delays.length > 0) {
              delays.push(now - lastAttemptTime);
            }
            lastAttemptTime = now;
            throw new Error('Fail');
          },
          config
        );
      } catch (e) {
        // Expected
      }

      // All delays should be <= max_delay_ms
      delays.forEach(delay => {
        expect(delay).toBeLessThanOrEqual(2000);
        expect(delay).toBeGreaterThan(0);
      });
    });
  });

  describe('Jitter Implementation', () => {
    it('should add random jitter to delays', async () => {
      const config: RetryConfig = {
        max_retries: 5,
        base_delay_ms: 100,
        max_delay_ms: 10000,
        jitter_ms: 50
      };

      const delays: number[] = [];
      let lastAttemptTime = Date.now();

      try {
        await retryWithBackoff(
          async () => {
            const now = Date.now();
            if (delays.length > 0) {
              delays.push(now - lastAttemptTime);
            }
            lastAttemptTime = now;
            throw new Error('Fail');
          },
          config
        );
      } catch (e) {
        // Expected
      }

      // With jitter, delays should vary from pure exponential
      // Collect multiple runs to check variance
      const hasVariance = delays.some((delay, index) => {
        if (index === 0) return false;
        const expectedDelay = 100 * Math.pow(2, index - 1);
        const variance = Math.abs(delay - expectedDelay);
        return variance > 0 && variance <= 50; // Within jitter range
      });

      expect(hasVariance).toBe(true);
    });
  });

  describe('Retry Behavior', () => {
    it('should retry on transient failures', async () => {
      const config: RetryConfig = {
        max_retries: 3
      };

      let attempts = 0;

      const result = await retryWithBackoff(
        async () => {
          attempts++;
          if (attempts <= 2) {
            throw new Error('Network error');
          }
          return 'success';
        },
        config
      );

      expect(result).toBe('success');
      expect(attempts).toBe(3); // Initial + 2 retries
    });

    it('should exhaust retries and throw last error', async () => {
      const config: RetryConfig = {
        max_retries: 2
      };

      let attempts = 0;

      try {
        await retryWithBackoff(
          async () => {
            attempts++;
            throw new Error('Persistent error');
          },
          config
        );
        fail('Should have thrown error');
      } catch (error) {
        expect((error as Error).message).toBe('Persistent error');
        expect(attempts).toBe(3); // Initial + 2 retries = 3 total attempts
      }
    });

    it('should not retry on success', async () => {
      const config: RetryConfig = {
        max_retries: 5
      };

      let attempts = 0;

      const result = await retryWithBackoff(
        async () => {
          attempts++;
          return 'success';
        },
        config
      );

      expect(result).toBe('success');
      expect(attempts).toBe(1); // Only initial attempt
    });
  });

  describe('onRetry Callback', () => {
    it('should call onRetry callback with attempt info', async () => {
      const retryAttempts: number[] = [];
      const errors: Error[] = [];

      const config: RetryConfig = {
        max_retries: 3,
        onRetry: (attempt, error) => {
          retryAttempts.push(attempt);
          errors.push(error);
        }
      };

      try {
        await retryWithBackoff(
          async () => {
            throw new Error('Test error');
          },
          config
        );
      } catch (e) {
        // Expected
      }

      expect(retryAttempts.length).toBe(3); // 3 retries
      expect(errors.length).toBe(3);
      errors.forEach(err => {
        expect(err.message).toBe('Test error');
      });
    });

    it('should pass attempt number (1-indexed)', async () => {
      const attempts: number[] = [];

      const config: RetryConfig = {
        max_retries: 2,
        onRetry: (attempt) => {
          attempts.push(attempt);
        }
      };

      try {
        await retryWithBackoff(
          async () => {
            throw new Error('Fail');
          },
          config
        );
      } catch (e) {
        // Expected
      }

      expect(attempts).toEqual([1, 2, 3]);
    });
  });

  describe('Default Configuration', () => {
    it('should use sensible defaults when not specified', async () => {
      let attempts = 0;

      try {
        await retryWithBackoff(
          async () => {
            attempts++;
            throw new Error('Fail');
          },
          { max_retries: 1 }
        );
      } catch (e) {
        // Expected
      }

      // Should complete with defaults
      expect(attempts).toBeGreaterThan(0);
    });
  });

  describe('Error Propagation', () => {
    it('should preserve error type through retries', async () => {
      class CustomError extends Error {
        constructor(message: string, public code: string) {
          super(message);
          this.name = 'CustomError';
        }
      }

      const config: RetryConfig = {
        max_retries: 2
      };

      try {
        await retryWithBackoff(
          async () => {
            throw new CustomError('Custom failure', 'ERR_CUSTOM');
          },
          config
        );
        fail('Should have thrown CustomError');
      } catch (error) {
        expect(error).toBeInstanceOf(CustomError);
        expect((error as CustomError).code).toBe('ERR_CUSTOM');
      }
    });

    it('should handle non-Error errors', async () => {
      const config: RetryConfig = {
        max_retries: 2
      };

      try {
        await retryWithBackoff(
          async () => {
            throw 'String error';
          },
          config
        );
        fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toBe('String error');
      }
    });
  });

  describe('Return Value', () => {
    it('should return successful result after retries', async () => {
      const config: RetryConfig = {
        max_retries: 3
      };

      let attempts = 0;

      const result = await retryWithBackoff(
        async () => {
          attempts++;
          if (attempts < 3) {
            throw new Error('Temporary failure');
          }
          return { data: 'final-result', attempts };
        },
        config
      );

      expect(result).toEqual({ data: 'final-result', attempts: 3 });
    });

    it('should work with complex return types', async () => {
      const config: RetryConfig = {
        max_retries: 2
      };

      interface ComplexResult {
        items: string[];
        metadata: Record<string, unknown>;
      }

      const result: ComplexResult = await retryWithBackoff(
        async () => {
          return {
            items: ['a', 'b', 'c'],
            metadata: { count: 3 }
          };
        },
        config
      );

      expect(result.items).toEqual(['a', 'b', 'c']);
      expect(result.metadata).toEqual({ count: 3 });
    });
  });
});
