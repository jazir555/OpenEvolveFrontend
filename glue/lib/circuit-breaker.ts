/**
 * Circuit Breaker Pattern
 *
 * Follows the Federation Constitution:
 * - Failure Management: System failures trigger circuit breaker
 * - Prevents cascading failures by stopping calls to dead services
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Circuit is tripped, requests fail immediately
 * - HALF_OPEN: Testing if service has recovered
 */

import { logger } from './logger';

export enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

export interface CircuitBreakerOptions {
  threshold: number;      // Failures before tripping
  timeout_ms: number;     // How long to stay OPEN before HALF_OPEN
  reset_timeout_ms?: number; // How long to wait in HALF_OPEN before closing
  onStateChange?: (oldState: CircuitState, newState: CircuitState) => void;
}

export interface CircuitBreakerStats {
  state: CircuitState;
  failure_count: number;
  success_count: number;
  last_failure_time?: Date;
  last_state_change?: Date;
}

const DEFAULT_OPTIONS: Required<Omit<CircuitBreakerOptions, 'onStateChange'>> = {
  threshold: 5,
  timeout_ms: 60000,     // 1 minute
  reset_timeout_ms: 10000, // 10 seconds
};

/**
 * Circuit Breaker implementation
 *
 * Prevents cascading failures by stopping calls to failing services
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failure_count: number = 0;
  private success_count: number = 0;
  private last_failure_time?: Date;
  private last_state_change: Date = new Date();
  private next_attempt_time?: Date;

  private readonly options: Required<Omit<CircuitBreakerOptions, 'onStateChange'>> & {
    onStateChange?: CircuitBreakerOptions['onStateChange'];
  };

  constructor(options: CircuitBreakerOptions) {
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
    };
  }

  /**
   * Execute function through circuit breaker
   *
   * @param fn - Async function to execute
   * @returns Result of function execution
   * @throws Error if circuit is OPEN or function fails
   */
  async execute<T>(fn: () => Promise<T>): Promise<T> {
    // Check if we should transition from OPEN to HALF_OPEN
    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.transitionTo(CircuitState.HALF_OPEN);
      } else {
        const waitTime = this.next_attempt_time
          ? Math.max(0, this.next_attempt_time.getTime() - Date.now())
          : this.options.timeout_ms;

        throw new Error(
          `Circuit breaker is OPEN. Rejecting request. Try again in ${Math.round(waitTime)}ms.`
        );
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  /**
   * Handle successful execution
   */
  private onSuccess(): void {
    this.success_count++;

    if (this.state === CircuitState.HALF_OPEN) {
      // Service recovered, close the circuit
      this.transitionTo(CircuitState.CLOSED);
      this.failure_count = 0;
    }
  }

  /**
   * Handle failed execution
   */
  private onFailure(): void {
    this.failure_count++;
    this.last_failure_time = new Date();

    if (this.state === CircuitState.HALF_OPEN) {
      // Service still failing, reopen circuit
      this.transitionTo(CircuitState.OPEN);
    } else if (this.failure_count >= this.options.threshold) {
      // Threshold reached, trip the circuit
      this.transitionTo(CircuitState.OPEN);
    }

    logger.warn('Circuit breaker failure recorded', {
      state: this.state,
      failure_count: this.failure_count,
      threshold: this.options.threshold,
    });
  }

  /**
   * Check if enough time has passed to attempt a reset
   */
  private shouldAttemptReset(): boolean {
    if (!this.next_attempt_time) {
      return false;
    }
    return Date.now() >= this.next_attempt_time.getTime();
  }

  /**
   * Transition to new state
   */
  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;
    this.last_state_change = new Date();

    // Set next attempt time when opening circuit
    if (newState === CircuitState.OPEN) {
      this.next_attempt_time = new Date(
        Date.now() + this.options.timeout_ms
      );
    } else if (newState === CircuitState.CLOSED) {
      this.next_attempt_time = undefined;
    }

    logger.info('Circuit breaker state changed', {
      old_state: oldState,
      new_state: newState,
      failure_count: this.failure_count,
    });

    // Call state change callback if provided
    if (this.options.onStateChange) {
      this.options.onStateChange(oldState, newState);
    }
  }

  /**
   * Get current circuit state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Get circuit breaker statistics
   */
  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failure_count: this.failure_count,
      success_count: this.success_count,
      last_failure_time: this.last_failure_time,
      last_state_change: this.last_state_change,
    };
  }

  /**
   * Manually reset circuit breaker to CLOSED state
   */
  reset(): void {
    this.transitionTo(CircuitState.CLOSED);
    this.failure_count = 0;
    this.success_count = 0;
    this.last_failure_time = undefined;
    this.next_attempt_time = undefined;
  }
}

/**
 * Example usage:
 *
 * ```typescript
 * import { CircuitBreaker } from './circuit-breaker';
 *
 * // Create circuit breaker
 * const cb = new CircuitBreaker({
 *   threshold: 5,           // Trip after 5 failures
 *   timeout_ms: 60000,      // Stay open for 1 minute
 *   onStateChange: (old, newState) => {
 *     console.log(`Circuit: ${old} -> ${newState}`);
 *   },
 * });
 *
 * // Use circuit breaker
 * try {
 *   const result = await cb.execute(async () => {
 *     const response = await fetch('http://service:8000/api');
 *     if (!response.ok) throw new Error('HTTP error');
 *     return response.json();
 *   });
 * } catch (error) {
 *   if (cb.getState() === CircuitState.OPEN) {
 *     logger.error('Service is down, circuit is open', error);
 *     // Use fallback or cached data
 *   } else {
 *     throw error;
 *   }
 * }
 *
 * // Check state
 * const stats = cb.getStats();
 * console.log(stats);
 * // { state: 'closed', failure_count: 2, success_count: 10, ... }
 *
 * // Manual reset (e.g., after health check passes)
 * cb.reset();
 * ```
 */
