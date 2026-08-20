/**
 * Circuit Breaker Pattern Implementation
 * Per CLAUDE.md Section 2.3: System Failure → Circuit Breaker
 * Stop hammering the dead service. Wait for a health check to pass.
 */

import { apiLogger, LogContext } from './structuredLogger';

export enum CircuitState {
  CLOSED = 'closed',     // Normal operation
  OPEN = 'open',         // Circuit is open, rejecting requests
  HALF_OPEN = 'half-open' // Testing if service has recovered
}

export interface CircuitBreakerConfig {
  failureThreshold: number;  // Number of failures before opening
  successThreshold: number;  // Number of successes to close circuit
  timeoutMs: number;         // How long to wait before attempting recovery
  monitoringPeriodMs: number;// Period to monitor for consecutive failures
}

export interface CircuitBreakerStats {
  state: CircuitState;
  failureCount: number;
  successCount: number;
  lastFailureTime?: number;
  lastSuccessTime?: number;
  openedAt?: number;
  nextAttemptAt?: number;
}

export class CircuitBreaker {
  private state: CircuitState;
  private failureCount: number;
  private successCount: number;
  private lastFailureTime?: number;
  private lastSuccessTime?: number;
  private openedAt?: number;
  private nextAttemptAt?: number;
  private config: CircuitBreakerConfig;
  private serviceName: string;

  constructor(serviceName: string, config: CircuitBreakerConfig) {
    this.serviceName = serviceName;
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.config = {
      failureThreshold: config.failureThreshold || 5,
      successThreshold: config.successThreshold || 2,
      timeoutMs: config.timeoutMs || 60000, // 1 minute default
      monitoringPeriodMs: config.monitoringPeriodMs || 10000 // 10 seconds default
    };
  }

  /**
   * Execute operation with circuit breaker protection
   */
  async execute<T>(
    operation: () => Promise<T>,
    context?: LogContext
  ): Promise<T> {
    const correlationId = context?.correlation_id
      || `cb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const cbContext: LogContext = {
      ...context,
      correlation_id: correlationId,
      source_service: 'circuit-breaker',
      target_service: this.serviceName,
      circuit_state: this.state
    };

    // Check if circuit is open
    if (this.state === CircuitState.OPEN) {
      if (Date.now() < (this.nextAttemptAt || 0)) {
        // Circuit is still open, reject request
        apiLogger.error('Circuit breaker is open, rejecting request', undefined, cbContext);
        throw new CircuitBreakerOpenError(
          `Circuit breaker is open for ${this.serviceName}. `
          + `Will retry after ${new Date(this.nextAttemptAt || 0).toISOString()}`
        );
      }

      // Timeout has elapsed, attempt recovery
      apiLogger.info('Circuit breaker timeout elapsed, entering half-open state', cbContext);
      this.state = CircuitState.HALF_OPEN;
      this.successCount = 0;
    }

    // Execute operation
    try {
      const result = await operation();
      this.onSuccess(cbContext);
      return result;
    } catch (error) {
      this.onFailure(error as Error, cbContext);
      throw error;
    }
  }

  /**
   * Handle successful operation
   */
  private onSuccess(context: LogContext): void {
    this.lastSuccessTime = Date.now();

    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;
      apiLogger.info('Circuit breaker half-open success', {
        ...context,
        success_count: this.successCount,
        required: this.config.successThreshold
      });

      if (this.successCount >= this.config.successThreshold) {
        // Service has recovered, close circuit
        this.closeCircuit(context);
      }
    } else {
      // Reset failure count on success in closed state
      this.failureCount = Math.max(0, this.failureCount - 1);
    }
  }

  /**
   * Handle failed operation
   */
  private onFailure(error: Error, context: LogContext): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();

    apiLogger.warn('Circuit breaker operation failed', {
      ...context,
      failure_count: this.failureCount,
      threshold: this.config.failureThreshold,
      error: error.message
    });

    if (this.state === CircuitState.HALF_OPEN) {
      // Service still failing, reopen circuit
      this.openCircuit(context);
    } else if (this.failureCount >= this.config.failureThreshold) {
      // Too many failures, open circuit
      this.openCircuit(context);
    }
  }

  /**
   * Open circuit (stop accepting requests)
   */
  private openCircuit(context: LogContext): void {
    this.state = CircuitState.OPEN;
    this.openedAt = Date.now();
    this.nextAttemptAt = Date.now() + this.config.timeoutMs;

    apiLogger.error('Circuit breaker opened', undefined, {
      ...context,
      failure_count: this.failureCount,
      opened_at: new Date(this.openedAt).toISOString(),
      next_attempt_at: new Date(this.nextAttemptAt).toISOString(),
      service: this.serviceName
    });
  }

  /**
   * Close circuit (resume normal operation)
   */
  private closeCircuit(context: LogContext): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.openedAt = undefined;
    this.nextAttemptAt = undefined;

    apiLogger.info('Circuit breaker closed', {
      ...context,
      service: this.serviceName
    });
  }

  /**
   * Get current circuit breaker state
   */
  getState(): CircuitBreakerStats {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
      lastSuccessTime: this.lastSuccessTime,
      openedAt: this.openedAt,
      nextAttemptAt: this.nextAttemptAt
    };
  }

  /**
   * Reset circuit breaker to closed state
   */
  reset(): void {
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.openedAt = undefined;
    this.nextAttemptAt = undefined;

    apiLogger.info('Circuit breaker manually reset', {
      service: this.serviceName
    });
  }
}

/**
 * Custom error for circuit breaker open state
 */
export class CircuitBreakerOpenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CircuitBreakerOpenError';
  }
}

/**
 * Circuit breaker registry for managing multiple circuit breakers
 */
export class CircuitBreakerRegistry {
  private breakers: Map<string, CircuitBreaker> = new Map();

  /**
   * Get or create circuit breaker for a service
   */
  get(serviceName: string, config?: CircuitBreakerConfig): CircuitBreaker {
    if (!this.breakers.has(serviceName)) {
      const defaultConfig: CircuitBreakerConfig = {
        failureThreshold: 5,
        successThreshold: 2,
        timeoutMs: 60000,
        monitoringPeriodMs: 10000
      };

      this.breakers.set(serviceName, new CircuitBreaker(serviceName, config || defaultConfig));
    }

    return this.breakers.get(serviceName)!;
  }

  /**
   * Get stats for all circuit breakers
   */
  getAllStats(): Map<string, CircuitBreakerStats> {
    const stats = new Map<string, CircuitBreakerStats>();

    this.breakers.forEach((breaker, serviceName) => {
      stats.set(serviceName, breaker.getState());
    });

    return stats;
  }

  /**
   * Reset all circuit breakers
   */
  resetAll(): void {
    this.breakers.forEach((breaker) => breaker.reset());
  }

  /**
   * Remove circuit breaker for a service
   */
  remove(serviceName: string): void {
    this.breakers.delete(serviceName);
  }
}

// Global circuit breaker registry
export const circuitBreakerRegistry = new CircuitBreakerRegistry();
