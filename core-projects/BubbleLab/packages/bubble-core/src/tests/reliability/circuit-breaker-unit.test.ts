/**
 * Circuit Breaker Unit Tests (Bug #5, #7)
 *
 * Tests for circuit breaker functionality
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  CircuitBreaker,
  CircuitBreakerState,
  CircuitBreakerConfig,
} from '../../../security/circuit-breaker';

describe('Circuit Breaker Unit Tests', () => {
  let circuitBreaker: CircuitBreaker;
  let config: CircuitBreakerConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    config = {
      failureThreshold: 5,
      timeout: 60000,
      halfOpenAttempts: 3,
    };

    circuitBreaker = new CircuitBreaker('test-api', config);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Initial State', () => {
    it('should start in CLOSED state', () => {
      expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
    });

    it('should initialize with zero counts', () => {
      const metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(0);
      expect(metrics.successCount).toBe(0);
    });
  });

  describe('CLOSED to OPEN Transition', () => {
    it('should open circuit after threshold failures', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 3,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      for (let i = 0; i < 3; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.OPEN);
    });

    it('should reset failure count on success', async () => {
      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Failure 1');
        });
      } catch (error) {
        // Expected
      }

      let metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(1);

      await circuitBreaker.execute(async () => {
        return 'success';
      });

      metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(0);
    });
  });

  describe('OPEN State Behavior', () => {
    it('should block requests when OPEN', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      try {
        await strictCircuit.execute(async () => {
          return 'should not execute';
        });
        expect.fail('Should have thrown');
      } catch (error) {
        expect((error as Error).message).toContain('OPEN');
      }
    });
  });

  describe('OPEN to HALF_OPEN Transition', () => {
    it('should transition to HALF_OPEN after timeout', async () => {
      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 2,
      });

      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      await vi.advanceTimersByTimeAsync(6000);

      try {
        await shortTimeoutCircuit.execute(async () => {
          return 'test';
        });
      } catch (error) {
        // Expected
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.HALF_OPEN);
    });
  });

  describe('HALF_OPEN to CLOSED Transition', () => {
    it('should close after successful attempts', async () => {
      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 3,
      });

      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      await vi.advanceTimersByTimeAsync(6000);

      for (let i = 0; i < 3; i++) {
        await shortTimeoutCircuit.execute(async () => {
          return `success ${i}`;
        });
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.CLOSED);
    });
  });

  describe('Metrics Tracking', () => {
    it('should track failure count', async () => {
      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Failure');
        });
      } catch (error) {
        // Expected
      }

      const metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(1);
    });

    it('should track success count', async () => {
      await circuitBreaker.execute(async () => 'success');
      await circuitBreaker.execute(async () => 'success');

      const metrics = circuitBreaker.getMetrics();
      expect(metrics.successCount).toBe(2);
    });
  });

  describe('Manual Reset', () => {
    it('should reset to CLOSED', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      strictCircuit.reset();

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.CLOSED);
    });
  });
});
