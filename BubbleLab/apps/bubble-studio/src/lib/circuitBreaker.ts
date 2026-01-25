/**
 * Circuit Breaker Implementation
 *
 * Follows Federation Constitution Failure Management Strategy:
 * - Transient Failure → Exponential Backoff Retry (Jittered)
 * - System Failure → Circuit Breaker. Stop hammering the dead service.
 *
 * Based on AntiCorruptionLayer implementation but simplified for client-side use.
 */

/**
 * Circuit Breaker States
 */
export enum CircuitBreakerState {
  CLOSED = 'closed',      // Normal operation
  OPEN = 'open',          // Circuit is open, blocking requests
  HALF_OPEN = 'half_open' // Testing if service has recovered
}

/**
 * Circuit Breaker Configuration
 */
export interface CircuitBreakerConfig {
  failureThreshold: number;  // Number of failures before opening
  timeout: number;           // Milliseconds to wait before attempting reset
  halfOpenAttempts: number;  // Number of successful attempts needed to close
}

/**
 * Circuit Breaker Implementation
 */
export class CircuitBreaker {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenAttemptCount = 0;
  private name: string;

  constructor(
    name: string,
    private config: CircuitBreakerConfig
  ) {
    this.name = name;
    console.info(`[CircuitBreaker:${name}] Initialized with config`, config);
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    // Check if circuit is open
    if (this.state === CircuitBreakerState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.transitionToHalfOpen();
      } else {
        const error = new Error(
          `Circuit breaker [${this.name}] is OPEN. Blocking request to prevent cascading failure.`
        );
        console.error(`[CircuitBreaker:${this.name}] ${error.message}`);
        throw error;
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
    this.failureCount = 0;

    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.halfOpenAttemptCount++;

      console.info(
        `[CircuitBreaker:${this.name}] Half-open success (${this.halfOpenAttemptCount}/${this.config.halfOpenAttempts})`
      );

      if (this.halfOpenAttemptCount >= this.config.halfOpenAttempts) {
        this.transitionToClosed();
      }
    } else if (this.state === CircuitBreakerState.CLOSED) {
      this.successCount++;
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    console.warn(
      `[CircuitBreaker:${this.name}] Failure recorded (${this.failureCount}/${this.config.failureThreshold})`
    );

    if (this.failureCount >= this.config.failureThreshold) {
      this.transitionToOpen();
    }
  }

  private transitionToOpen(): void {
    const previousState = this.state;
    this.state = CircuitBreakerState.OPEN;

    console.error(
      `[CircuitBreaker:${this.name}] Transitioned from ${previousState} to OPEN after ${this.failureCount} failures. ` +
      `Will attempt reset after ${this.config.timeout}ms.`
    );
  }

  private transitionToHalfOpen(): void {
    const previousState = this.state;
    this.state = CircuitBreakerState.HALF_OPEN;
    this.halfOpenAttemptCount = 0;

    console.info(
      `[CircuitBreaker:${this.name}] Transitioned from ${previousState} to HALF_OPEN. ` +
      `Testing if service has recovered.`
    );
  }

  private transitionToClosed(): void {
    const previousState = this.state;
    this.state = CircuitBreakerState.CLOSED;

    console.info(
      `[CircuitBreaker:${this.name}] Transitioned from ${previousState} to CLOSED. ` +
      `Service has recovered.`
    );
  }

  private shouldAttemptReset(): boolean {
    const timeSinceLastFailure = Date.now() - this.lastFailureTime;
    const shouldReset = timeSinceLastFailure > this.config.timeout;

    if (shouldReset) {
      console.info(
        `[CircuitBreaker:${this.name}] Timeout elapsed (${timeSinceLastFailure}ms). ` +
        `Attempting reset...`
      );
    }

    return shouldReset;
  }

  getState(): CircuitBreakerState {
    return this.state;
  }

  getMetrics() {
    return {
      name: this.name,
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
      timeUntilReset: Math.max(0, this.config.timeout - (Date.now() - this.lastFailureTime)),
    };
  }

  reset(): void {
    console.warn(`[CircuitBreaker:${this.name}] Manual reset triggered`);
    this.state = CircuitBreakerState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
    this.halfOpenAttemptCount = 0;
  }
}

/**
 * Create a circuit breaker for Evolution API
 */
export function createEvolutionApiCircuitBreaker(): CircuitBreaker {
  return new CircuitBreaker('evolution-api', {
    failureThreshold: 5,      // Open after 5 consecutive failures
    timeout: 60000,           // Wait 60 seconds before attempting reset
    halfOpenAttempts: 3,      // Require 3 successful attempts to close
  });
}
