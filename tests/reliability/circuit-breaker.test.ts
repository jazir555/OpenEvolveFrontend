/**
 * Circuit Breaker Tests (Bug #5, #7)
 *
 * Tests for circuit breaker functionality:
 * - Circuit breaker starts in CLOSED state
 * - Consecutive failures open the circuit
 * - OPEN state blocks requests immediately
 * - After timeout, moves to HALF_OPEN
 * - Consecutive successes close the circuit
 * - Failure in HALF_OPEN returns to OPEN
 * - State transitions are logged
 * - Metrics track correctly (failureCount, successCount, etc.)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  CircuitBreaker,
  CircuitBreakerState,
  createEvolutionApiCircuitBreaker,
  CircuitBreakerConfig,
} from '../../BubbleLab/apps/bubble-studio/src/lib/circuitBreaker';

// Mock console methods to verify logging
const mockConsoleInfo = vi.fn();
const mockConsoleWarn = vi.fn();
const mockConsoleError = vi.fn();

describe('Circuit Breaker Tests (Bug #5, #7)', () => {
  let circuitBreaker: CircuitBreaker;
  let config: CircuitBreakerConfig;

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock console methods
    global.console.info = mockConsoleInfo;
    global.console.warn = mockConsoleWarn;
    global.console.error = mockConsoleError;

    config = {
      failureThreshold: 5,
      timeout: 60000,
      halfOpenAttempts: 3,
    };

    circuitBreaker = new CircuitBreaker('test-api', config);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Initial State', () => {
    it('should start in CLOSED state', () => {
      expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
    });

    it('should initialize with zero failure and success counts', () => {
      const metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(0);
      expect(metrics.successCount).toBe(0);
    });

    it('should log initialization', () => {
      expect(mockConsoleInfo).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Initialized with config',
        config
      );
    });
  });

  describe('State Transitions - CLOSED to OPEN', () => {
    it('should open circuit after failure threshold is reached', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 3,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Trigger 3 failures
      for (let i = 0; i < 3; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Simulated failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.OPEN);
    });

    it('should not open circuit before failure threshold', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 5,
        timeout: 60000,
        halfOpenAttempts: 3,
      });

      // Trigger only 3 failures (below threshold of 5)
      for (let i = 0; i < 3; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Simulated failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.CLOSED);
    });

    it('should reset failure count on success', async () => {
      // First attempt fails
      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Failure 1');
        });
      } catch (error) {
        // Expected
      }

      let metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(1);

      // Second attempt succeeds
      await circuitBreaker.execute(async () => {
        return 'success';
      });

      metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(0);
      expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
    });

    it('should log each failure', async () => {
      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Test failure');
        });
      } catch (error) {
        // Expected
      }

      expect(mockConsoleWarn).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Failure recorded (1/5)'
      );
    });

    it('should log state transition to OPEN', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Trigger failures to open circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      expect(mockConsoleError).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Transitioned from closed to OPEN after 2 failures. ' +
          'Will attempt reset after 60000ms.'
      );
    });
  });

  describe('OPEN State Behavior', () => {
    it('should block requests when circuit is OPEN', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
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

      // Try to make request through open circuit
      try {
        await strictCircuit.execute(async () => {
          return 'should not execute';
        });
        expect.fail('Should have thrown circuit open error');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain('OPEN');
        expect((error as Error).message).toContain('Blocking request');
      }
    });

    it('should reject immediately without executing function', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      let executed = false;

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Try to execute through open circuit
      try {
        await strictCircuit.execute(async () => {
          executed = true;
          return 'should not execute';
        });
      } catch (error) {
        // Expected
      }

      expect(executed).toBe(false);
    });

    it('should log when blocking request in OPEN state', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Try to make request
      try {
        await strictCircuit.execute(async () => {
          return 'test';
        });
      } catch (error) {
        // Expected
      }

      expect(mockConsoleError).toHaveBeenCalledWith(
        expect.stringContaining('Circuit breaker [test-api] is OPEN')
      );
    });
  });

  describe('State Transitions - OPEN to HALF_OPEN', () => {
    it('should transition to HALF_OPEN after timeout', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000, // 5 seconds
        halfOpenAttempts: 2,
      });

      // Open the circuit
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

      // Advance time past timeout
      await vi.advanceTimersByTimeAsync(6000);

      // Make a request to trigger transition
      try {
        await shortTimeoutCircuit.execute(async () => {
          return 'test';
        });
      } catch (error) {
        // Expected
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.HALF_OPEN);

      vi.useRealTimers();
    });

    it('should log transition to HALF_OPEN', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      // Make request
      try {
        await shortTimeoutCircuit.execute(async () => {
          return 'test';
        });
      } catch (error) {
        // Expected
      }

      expect(mockConsoleInfo).toHaveBeenCalledWith(
        expect.stringContaining('Transitioned from OPEN to HALF_OPEN')
      );

      vi.useRealTimers();
    });

    it('should allow requests in HALF_OPEN state', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      // Make request - should execute
      const result = await shortTimeoutCircuit.execute(async () => {
        return 'half-open success';
      });

      expect(result).toBe('half-open success');
      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.HALF_OPEN);

      vi.useRealTimers();
    });
  });

  describe('State Transitions - HALF_OPEN to CLOSED', () => {
    it('should close circuit after halfOpenAttempts successes', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 3, // Need 3 successes
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      // Make successful requests to close circuit
      for (let i = 0; i < 3; i++) {
        await shortTimeoutCircuit.execute(async () => {
          return `success ${i}`;
        });
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.CLOSED);

      vi.useRealTimers();
    });

    it('should track half-open attempts', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 3,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      // Make successful requests
      await shortTimeoutCircuit.execute(async () => {
        return 'success 1';
      });

      expect(mockConsoleInfo).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Half-open success (1/3)'
      );

      await shortTimeoutCircuit.execute(async () => {
        return 'success 2';
      });

      expect(mockConsoleInfo).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Half-open success (2/3)'
      );

      vi.useRealTimers();
    });

    it('should log transition to CLOSED', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout
      await vi.advanceTimersByTimeAsync(6000);

      // Make successful requests to close
      await shortTimeoutCircuit.execute(async () => {
        return 'success 1';
      });
      await shortTimeoutCircuit.execute(async () => {
        return 'success 2';
      });

      expect(mockConsoleInfo).toHaveBeenCalledWith(
        expect.stringContaining('Transitioned from HALF_OPEN to CLOSED')
      );
      expect(mockConsoleInfo).toHaveBeenCalledWith(
        expect.stringContaining('Service has recovered')
      );

      vi.useRealTimers();
    });
  });

  describe('State Transitions - HALF_OPEN to OPEN', () => {
    it('should return to OPEN on failure in HALF_OPEN', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 5000,
        halfOpenAttempts: 3,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Advance past timeout to HALF_OPEN
      await vi.advanceTimersByTimeAsync(6000);

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      // Make one successful request
      await shortTimeoutCircuit.execute(async () => {
        return 'success';
      });

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.HALF_OPEN);

      // Then fail
      try {
        await shortTimeoutCircuit.execute(async () => {
          throw new Error('Half-open failure');
        });
      } catch (error) {
        // Expected
      }

      expect(shortTimeoutCircuit.getState()).toBe(CircuitBreakerState.OPEN);

      vi.useRealTimers();
    });
  });

  describe('Metrics Tracking', () => {
    it('should track failure count correctly', async () => {
      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Failure 1');
        });
      } catch (error) {
        // Expected
      }

      let metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(1);

      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Failure 2');
        });
      } catch (error) {
        // Expected
      }

      metrics = circuitBreaker.getMetrics();
      expect(metrics.failureCount).toBe(2);
    });

    it('should track success count correctly', async () => {
      await circuitBreaker.execute(async () => {
        return 'success 1';
      });

      await circuitBreaker.execute(async () => {
        return 'success 2';
      });

      await circuitBreaker.execute(async () => {
        return 'success 3';
      });

      const metrics = circuitBreaker.getMetrics();
      expect(metrics.successCount).toBe(3);
    });

    it('should track last failure time', async () => {
      const beforeFailure = Date.now();

      try {
        await circuitBreaker.execute(async () => {
          throw new Error('Test failure');
        });
      } catch (error) {
        // Expected
      }

      const metrics = circuitBreaker.getMetrics();
      expect(metrics.lastFailureTime).toBeGreaterThanOrEqual(beforeFailure);
      expect(metrics.lastFailureTime).toBeLessThanOrEqual(Date.now());
    });

    it('should calculate time until reset', async () => {
      vi.useFakeTimers();

      const shortTimeoutCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 10000, // 10 seconds
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await shortTimeoutCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      let metrics = shortTimeoutCircuit.getMetrics();
      expect(metrics.timeUntilReset).toBeGreaterThan(0);
      expect(metrics.timeUntilReset).toBeLessThanOrEqual(10000);

      // Advance 5 seconds
      await vi.advanceTimersByTimeAsync(5000);

      metrics = shortTimeoutCircuit.getMetrics();
      expect(metrics.timeUntilReset).toBeGreaterThan(0);
      expect(metrics.timeUntilReset).toBeLessThanOrEqual(5000);

      vi.useRealTimers();
    });

    it('should include circuit name in metrics', () => {
      const metrics = circuitBreaker.getMetrics();
      expect(metrics.name).toBe('test-api');
    });

    it('should include current state in metrics', () => {
      const metrics = circuitBreaker.getMetrics();
      expect(metrics.state).toBe(CircuitBreakerState.CLOSED);
    });
  });

  describe('Manual Reset', () => {
    it('should reset circuit to CLOSED state', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
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

      // Manual reset
      strictCircuit.reset();

      expect(strictCircuit.getState()).toBe(CircuitBreakerState.CLOSED);

      const metrics = strictCircuit.getMetrics();
      expect(metrics.failureCount).toBe(0);
      expect(metrics.successCount).toBe(0);
    });

    it('should log manual reset', () => {
      circuitBreaker.reset();

      expect(mockConsoleWarn).toHaveBeenCalledWith(
        '[CircuitBreaker:test-api] Manual reset triggered'
      );
    });

    it('should allow requests after reset', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Reset
      strictCircuit.reset();

      // Should be able to make requests
      const result = await strictCircuit.execute(async () => {
        return 'success after reset';
      });

      expect(result).toBe('success after reset');
    });
  });

  describe('Evolution API Circuit Breaker', () => {
    it('should create circuit breaker with Evolution API config', () => {
      const evolutionCircuit = createEvolutionApiCircuitBreaker();

      const metrics = evolutionCircuit.getMetrics();
      expect(metrics.name).toBe('evolution-api');
      expect(metrics.state).toBe(CircuitBreakerState.CLOSED);
    });

    it('should have correct Evolution API configuration', () => {
      const evolutionCircuit = createEvolutionApiCircuitBreaker();

      // Trigger failures to test threshold
      for (let i = 0; i < 4; i++) {
        try {
          evolutionCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Should still be CLOSED (threshold is 5)
      expect(evolutionCircuit.getState()).toBe(CircuitBreakerState.CLOSED);

      // One more failure should open it
      try {
        evolutionCircuit.execute(async () => {
          throw new Error('Failure');
        });
      } catch (error) {
        // Expected
      }

      expect(evolutionCircuit.getState()).toBe(CircuitBreakerState.OPEN);
    });
  });

  describe('Concurrent Request Handling', () => {
    it('should handle multiple concurrent requests in CLOSED state', async () => {
      const promises = [
        circuitBreaker.execute(async () => 'result1'),
        circuitBreaker.execute(async () => 'result2'),
        circuitBreaker.execute(async () => 'result3'),
      ];

      const results = await Promise.all(promises);

      expect(results).toEqual(['result1', 'result2', 'result3']);
      expect(circuitBreaker.getState()).toBe(CircuitBreakerState.CLOSED);
    });

    it('should block all concurrent requests when OPEN', async () => {
      const strictCircuit = new CircuitBreaker('test-api', {
        failureThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 2,
      });

      // Open the circuit
      for (let i = 0; i < 2; i++) {
        try {
          await strictCircuit.execute(async () => {
            throw new Error('Failure');
          });
        } catch (error) {
          // Expected
        }
      }

      // Try multiple concurrent requests
      const promises = [
        strictCircuit.execute(async () => 'result1'),
        strictCircuit.execute(async () => 'result2'),
        strictCircuit.execute(async () => 'result3'),
      ];

      const results = await Promise.allSettled(promises);

      results.forEach((result) => {
        expect(result.status).toBe('rejected');
        expect((result.reason as Error).message).toContain('OPEN');
      });
    });
  });
});
