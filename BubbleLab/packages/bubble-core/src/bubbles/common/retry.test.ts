/**
 * Comprehensive tests for common retry utilities
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  calculateDelay,
  sleep,
  retryWithBackoff,
  retryWithTimeout,
  withTimeout,
  CircuitBreaker,
  CircuitBreakerState,
  executeWithResilience,
  DEFAULT_RETRY_OPTIONS,
  defaultCircuitBreakerConfig,
  type RetryOptions,
  type CircuitBreakerOptions
} from './retry.js';
import { NetworkError, TimeoutError } from './error-handlers.js';

describe('retry utilities', () => {
  describe('calculateDelay', () => {
    it('should calculate exponential backoff without jitter', () => {
      const options = { ...DEFAULT_RETRY_OPTIONS, jitter: false };

      expect(calculateDelay(0, options)).toBe(1000); // 1000 * 2^0
      expect(calculateDelay(1, options)).toBe(2000); // 1000 * 2^1
      expect(calculateDelay(2, options)).toBe(4000); // 1000 * 2^2
      expect(calculateDelay(3, options)).toBe(8000); // 1000 * 2^3
    });

    it('should cap delay at maxDelayMs', () => {
      const options = { ...DEFAULT_RETRY_OPTIONS, jitter: false };

      // With baseDelayMs=1000, maxDelayMs=30000, backoffMultiplier=2
      // Attempt 10 would be 1000 * 2^10 = 1024000, but should be capped at 30000
      expect(calculateDelay(10, options)).toBe(30000);
    });

    it('should add jitter when enabled', () => {
      const options = { ...DEFAULT_RETRY_OPTIONS, jitter: true, jitterAmount: 0.1 };

      const delay = calculateDelay(1, options);
      const baseDelay = 2000; // 1000 * 2^1
      const jitterRange = baseDelay * 0.1; // 200

      // Delay should be within ±jitterRange of baseDelay
      expect(delay).toBeGreaterThanOrEqual(baseDelay - jitterRange);
      expect(delay).toBeLessThanOrEqual(baseDelay + jitterRange);
    });

    it('should use custom backoff multiplier', () => {
      const options = {
        ...DEFAULT_RETRY_OPTIONS,
        jitter: false,
        backoffMultiplier: 3
      };

      expect(calculateDelay(0, options)).toBe(1000); // 1000 * 3^0
      expect(calculateDelay(1, options)).toBe(3000); // 1000 * 3^1
      expect(calculateDelay(2, options)).toBe(9000); // 1000 * 3^2
    });
  });

  describe('sleep', () => {
    it('should sleep for specified duration', async () => {
      const start = Date.now();
      await sleep(100);
      const elapsed = Date.now() - start;

      // Allow some tolerance for timer accuracy
      expect(elapsed).toBeGreaterThanOrEqual(90);
      expect(elapsed).toBeLessThan(150);
    });

    it('should resolve immediately for zero duration', async () => {
      const start = Date.now();
      await sleep(0);
      const elapsed = Date.now() - start;

      expect(elapsed).toBeLessThan(10);
    });
  });

  describe('retryWithBackoff', () => {
    it('should return result on first success', async () => {
      const fn = vi.fn().mockResolvedValue('success');
      const result = await retryWithBackoff(fn);

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should retry on retryable errors', async () => {
      const fn = vi.fn()
        .mockRejectedValueOnce(new NetworkError('Temporary failure'))
        .mockRejectedValueOnce(new NetworkError('Another failure'))
        .mockResolvedValue('success');

      const result = await retryWithBackoff(fn, {
        maxAttempts: 3,
        baseDelayMs: 10
      });

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should not retry non-retryable errors', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('Permanent failure'));
      const isRetryable = vi.fn().mockReturnValue(false);

      await expect(retryWithBackoff(fn, {
        maxAttempts: 3,
        isRetryable
      })).rejects.toThrow('Permanent failure');

      expect(fn).toHaveBeenCalledTimes(1);
      expect(isRetryable).toHaveBeenCalledTimes(1);
    });

    it('should exhaust retries and throw last error', async () => {
      const fn = vi.fn().mockRejectedValue(new NetworkError('Persistent failure'));

      await expect(retryWithBackoff(fn, {
        maxAttempts: 3,
        baseDelayMs: 10
      })).rejects.toThrow('Persistent failure');

      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should call onRetry callback', async () => {
      const fn = vi.fn()
        .mockRejectedValueOnce(new NetworkError('Failure'))
        .mockResolvedValue('success');

      const onRetry = vi.fn();
      await retryWithBackoff(fn, {
        maxAttempts: 3,
        baseDelayMs: 10,
        onRetry
      });

      expect(onRetry).toHaveBeenCalledTimes(1);
      expect(onRetry).toHaveBeenCalledWith(1, expect.any(NetworkError));
    });

    it('should respect custom correlation ID and operation name', async () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
      const fn = vi.fn()
        .mockRejectedValueOnce(new NetworkError('Failure'))
        .mockResolvedValue('success');

      await retryWithBackoff(fn, {
        maxAttempts: 2,
        baseDelayMs: 10,
        correlationId: 'test-123',
        operation: 'TestOperation'
      });

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[test-123]'),
        expect.stringContaining('[TestOperation]')
      );

      consoleSpy.mockRestore();
    });

    it('should use default options when none provided', async () => {
      const fn = vi.fn().mockResolvedValue('success');
      const result = await retryWithBackoff(fn);

      expect(result).toBe('success');
    });
  });

  describe('retryWithTimeout', () => {
    it('should succeed within timeout', async () => {
      const fn = vi.fn().mockResolvedValue('success');
      const result = await retryWithTimeout(fn, 1000);

      expect(result).toBe('success');
    });

    it('should timeout and retry', async () => {
      const fn = vi.fn()
        .mockImplementationOnce(() => new Promise((_, reject) =>
          setTimeout(() => reject(new TimeoutError('Timeout', 100)), 200)
        ))
        .mockResolvedValue('success');

      const result = await retryWithTimeout(fn, 100, {
        maxAttempts: 3,
        baseDelayMs: 10
      });

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(2);
    });

    it('should throw TimeoutError after all retries timeout', async () => {
      const fn = vi.fn().mockImplementation(() =>
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 200)
        )
      );

      await expect(retryWithTimeout(fn, 100, {
        maxAttempts: 2,
        baseDelayMs: 10
      })).rejects.toThrow();

      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe('withTimeout', () => {
    it('should resolve before timeout', async () => {
      const fn = () => new Promise(resolve => setTimeout(() => resolve('success'), 50));
      const result = await withTimeout(fn(), 200);

      expect(result).toBe('success');
    });

    it('should timeout after specified duration', async () => {
      const fn = () => new Promise(resolve => setTimeout(() => resolve('success'), 200));

      await expect(withTimeout(fn(), 100)).rejects.toThrow(TimeoutError);
    });

    it('should propagate errors from function', async () => {
      const fn = () => Promise.reject(new Error('Function error'));

      await expect(withTimeout(fn(), 1000)).rejects.toThrow('Function error');
    });

    it('should use custom operation name in timeout error', async () => {
      const fn = () => new Promise(() => {}); // Never resolves

      await expect(withTimeout(fn(), 100, 'CustomOperation')).rejects.toThrow('CustomOperation timed out');
    });
  });

  describe('CircuitBreaker', () => {
    let circuitBreaker: CircuitBreaker;

    beforeEach(() => {
      circuitBreaker = new CircuitBreaker({
        failureThreshold: 3,
        successThreshold: 2,
        timeoutMs: 1000
      }, 'TestCircuit');
    });

    afterEach(() => {
      circuitBreaker.reset();
    });

    describe('initial state', () => {
      it('should start in CLOSED state', () => {
        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
      });

      it('should provide initial stats', () => {
        const stats = circuitBreaker.getStats();

        expect(stats.state).toBe(CircuitBreakerState.CLOSED);
        expect(stats.failureCount).toBe(0);
        expect(stats.successCount).toBe(0);
        expect(stats.openedAt).toBeUndefined();
      });
    });

    describe('successful execution', () => {
      it('should execute function successfully in CLOSED state', async () => {
        const fn = vi.fn().mockResolvedValue('success');
        const result = await circuitBreaker.execute(fn, 'testOperation');

        expect(result).toBe('success');
        expect(fn).toHaveBeenCalledTimes(1);
      });

      it('should increment success count', async () => {
        const fn = vi.fn().mockResolvedValue('success');

        await circuitBreaker.execute(fn);
        await circuitBreaker.execute(fn);

        const stats = circuitBreaker.getStats();
        expect(stats.successCount).toBe(2);
      });

      it('should reset failure count on success', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));
        const successFn = vi.fn().mockResolvedValue('success');

        // Fail twice
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        expect(circuitBreaker.getStats().failureCount).toBe(2);

        // Succeed once
        await circuitBreaker.execute(successFn);

        expect(circuitBreaker.getStats().failureCount).toBe(0);
      });
    });

    describe('failed execution', () => {
      it('should propagate errors', async () => {
        const fn = vi.fn().mockRejectedValue(new Error('Failure'));

        await expect(circuitBreaker.execute(fn)).rejects.toThrow('Failure');
      });

      it('should increment failure count', async () => {
        const fn = vi.fn().mockRejectedValue(new Error('Failure'));

        await expect(circuitBreaker.execute(fn)).rejects.toThrow();
        await expect(circuitBreaker.execute(fn)).rejects.toThrow();

        expect(circuitBreaker.getStats().failureCount).toBe(2);
      });

      it('should open circuit after failure threshold', async () => {
        const fn = vi.fn().mockRejectedValue(new Error('Failure'));

        // Fail 3 times (threshold)
        await expect(circuitBreaker.execute(fn)).rejects.toThrow();
        await expect(circuitBreaker.execute(fn)).rejects.toThrow();
        await expect(circuitBreaker.execute(fn)).rejects.toThrow();

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.OPEN);
      });

      it('should reset success count on failure', async () => {
        const successFn = vi.fn().mockResolvedValue('success');
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

        // Succeed twice
        await circuitBreaker.execute(successFn);
        await circuitBreaker.execute(successFn);

        expect(circuitBreaker.getStats().successCount).toBe(2);

        // Fail once
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        expect(circuitBreaker.getStats().successCount).toBe(0);
      });
    });

    describe('OPEN state', () => {
      it('should reject requests when circuit is OPEN', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

        // Open the circuit
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.OPEN);

        // Try to execute again - should be rejected immediately
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow(NetworkError);
      });

      it('should transition to HALF_OPEN after timeout', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

        // Open the circuit
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.OPEN);

        // Wait for timeout
        await new Promise(resolve => setTimeout(resolve, 1100));

        // Try to execute - should transition to HALF_OPEN
        const successFn = vi.fn().mockResolvedValue('success');
        await circuitBreaker.execute(successFn);

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.HALF_OPEN);
      });
    });

    describe('HALF_OPEN state', () => {
      it('should close circuit after success threshold', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));
        const successFn = vi.fn().mockResolvedValue('success');

        // Open the circuit
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        // Wait for timeout
        await new Promise(resolve => setTimeout(resolve, 1100));

        // Execute successful requests to close circuit
        await circuitBreaker.execute(successFn);
        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.HALF_OPEN);

        await circuitBreaker.execute(successFn);
        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
      });

      it('should reopen circuit on failure in HALF_OPEN', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));
        const successFn = vi.fn().mockResolvedValue('success');

        // Open the circuit
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        // Wait for timeout
        await new Promise(resolve => setTimeout(resolve, 1100));

        // Succeed once
        await circuitBreaker.execute(successFn);
        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.HALF_OPEN);

        // Fail in HALF_OPEN
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.OPEN);
      });
    });

    describe('reset', () => {
      it('should reset to CLOSED state', async () => {
        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

        // Open the circuit
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
        await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.OPEN);

        // Reset
        circuitBreaker.reset();

        expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
        expect(circuitBreaker.getStats().failureCount).toBe(0);
        expect(circuitBreaker.getStats().successCount).toBe(0);
        expect(circuitBreaker.getStats().openedAt).toBeUndefined();
      });
    });

    describe('callbacks', () => {
      it('should call onOpen callback', async () => {
        const onOpen = vi.fn();
        const cb = new CircuitBreaker({
          failureThreshold: 2,
          onOpen
        }, 'CallbackTest');

        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

        await expect(cb.execute(errorFn)).rejects.toThrow();
        await expect(cb.execute(errorFn)).rejects.toThrow();

        expect(onOpen).toHaveBeenCalledTimes(1);
      });

      it('should call onClose callback', async () => {
        const onClose = vi.fn();
        const cb = new CircuitBreaker({
          failureThreshold: 2,
          successThreshold: 2,
          timeoutMs: 500,
          onClose
        }, 'CallbackTest');

        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));
        const successFn = vi.fn().mockResolvedValue('success');

        // Open circuit
        await expect(cb.execute(errorFn)).rejects.toThrow();
        await expect(cb.execute(errorFn)).rejects.toThrow();

        // Wait for timeout and close circuit
        await new Promise(resolve => setTimeout(resolve, 600));
        await cb.execute(successFn);
        await cb.execute(successFn);

        expect(onClose).toHaveBeenCalledTimes(1);
      });

      it('should call onHalfOpen callback', async () => {
        const onHalfOpen = vi.fn();
        const cb = new CircuitBreaker({
          failureThreshold: 2,
          timeoutMs: 500,
          onHalfOpen
        }, 'CallbackTest');

        const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));
        const successFn = vi.fn().mockResolvedValue('success');

        // Open circuit
        await expect(cb.execute(errorFn)).rejects.toThrow();
        await expect(cb.execute(errorFn)).rejects.toThrow();

        // Wait for timeout
        await new Promise(resolve => setTimeout(resolve, 600));

        // Execute to trigger HALF_OPEN
        await cb.execute(successFn);

        expect(onHalfOpen).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('executeWithResilience', () => {
    it('should combine circuit breaker and retry', async () => {
      const circuitBreaker = new CircuitBreaker({
        failureThreshold: 5,
        successThreshold: 2
      }, 'ResilienceTest');

      const fn = vi.fn()
        .mockRejectedValueOnce(new NetworkError('Temporary failure'))
        .mockResolvedValue('success');

      const result = await executeWithResilience(fn, circuitBreaker, {
        maxAttempts: 3,
        baseDelayMs: 10
      });

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(2);
    });

    it('should fail when circuit breaker is open', async () => {
      const circuitBreaker = new CircuitBreaker({
        failureThreshold: 2
      }, 'ResilienceTest');

      const errorFn = vi.fn().mockRejectedValue(new Error('Failure'));

      // Open circuit
      await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();
      await expect(circuitBreaker.execute(errorFn)).rejects.toThrow();

      // Try with resilience pattern
      await expect(executeWithResilience(errorFn, circuitBreaker)).rejects.toThrow(NetworkError);
    });
  });

  describe('defaultCircuitBreakerConfig', () => {
    it('should have sensible default values', () => {
      expect(defaultCircuitBreakerConfig.failureThreshold).toBe(5);
      expect(defaultCircuitBreakerConfig.successThreshold).toBe(2);
      expect(defaultCircuitBreakerConfig.timeoutMs).toBe(60000);
      expect(defaultCircuitBreakerConfig.monitoringPeriodMs).toBe(60000);
      expect(defaultCircuitBreakerConfig.onOpen).toBeDefined();
      expect(defaultCircuitBreakerConfig.onClose).toBeDefined();
      expect(defaultCircuitBreakerConfig.onHalfOpen).toBeDefined();
    });
  });

  describe('DEFAULT_RETRY_OPTIONS', () => {
    it('should have sensible default values', () => {
      expect(DEFAULT_RETRY_OPTIONS.maxAttempts).toBe(3);
      expect(DEFAULT_RETRY_OPTIONS.baseDelayMs).toBe(1000);
      expect(DEFAULT_RETRY_OPTIONS.maxDelayMs).toBe(30000);
      expect(DEFAULT_RETRY_OPTIONS.backoffMultiplier).toBe(2);
      expect(DEFAULT_RETRY_OPTIONS.jitter).toBe(true);
      expect(DEFAULT_RETRY_OPTIONS.jitterAmount).toBe(0.1);
      expect(DEFAULT_RETRY_OPTIONS.correlationId).toBeDefined();
      expect(DEFAULT_RETRY_OPTIONS.operation).toBe('Retry Operation');
    });
  });
});
