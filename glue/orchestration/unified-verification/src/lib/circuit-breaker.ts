/**
 * Circuit Breaker Pattern
 *
 * Local copy used by the unified-verification package (mirrors glue/lib/circuit-breaker.ts)
 * so the package type-checks and runs self-contained.
 */

import { logger } from './logger';

export enum CircuitState {
  CLOSED = 'closed',
  OPEN = 'open',
  HALF_OPEN = 'half_open',
}

export interface CircuitBreakerOptions {
  threshold: number;
  timeout_ms: number;
  reset_timeout_ms?: number;
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
  timeout_ms: 60000,
  reset_timeout_ms: 10000,
};

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

  async execute<T>(fn: () => Promise<T>): Promise<T> {
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

  private onSuccess(): void {
    this.success_count++;

    if (this.state === CircuitState.HALF_OPEN) {
      this.transitionTo(CircuitState.CLOSED);
      this.failure_count = 0;
    }
  }

  private onFailure(): void {
    this.failure_count++;
    this.last_failure_time = new Date();

    if (this.state === CircuitState.HALF_OPEN) {
      this.transitionTo(CircuitState.OPEN);
    } else if (this.failure_count >= this.options.threshold) {
      this.transitionTo(CircuitState.OPEN);
    }

    logger.warn('Circuit breaker failure recorded', {
      state: this.state,
      failure_count: this.failure_count,
      threshold: this.options.threshold,
    });
  }

  private shouldAttemptReset(): boolean {
    if (!this.next_attempt_time) {
      return false;
    }
    return Date.now() >= this.next_attempt_time.getTime();
  }

  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    this.state = newState;
    this.last_state_change = new Date();

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

    if (this.options.onStateChange) {
      this.options.onStateChange(oldState, newState);
    }
  }

  getState(): CircuitState {
    return this.state;
  }

  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failure_count: this.failure_count,
      success_count: this.success_count,
      last_failure_time: this.last_failure_time,
      last_state_change: this.last_state_change,
    };
  }

  reset(): void {
    this.transitionTo(CircuitState.CLOSED);
    this.failure_count = 0;
    this.success_count = 0;
    this.last_failure_time = undefined;
    this.next_attempt_time = undefined;
  }
}
