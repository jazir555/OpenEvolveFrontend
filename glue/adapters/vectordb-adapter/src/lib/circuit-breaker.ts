/**
 * Circuit Breaker - prevents cascading failures against flaky vector DB backends.
 */

import { Logger } from './logger';

export type CircuitState = 'closed' | 'open' | 'half-open';

export interface CircuitBreakerOptions {
  threshold?: number;
  timeout?: number;
  logger?: Logger;
}

export class CircuitBreaker {
  private readonly threshold: number;
  private readonly timeout: number;
  private readonly logger?: Logger;
  private failures = 0;
  private state: CircuitState = 'closed';
  private openedAt = 0;

  constructor(options: CircuitBreakerOptions = {}) {
    this.threshold = options.threshold ?? 5;
    this.timeout = options.timeout ?? 60000;
    this.logger = options.logger;
  }

  getState(): CircuitState {
    return this.state;
  }

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.openedAt >= this.timeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit breaker is open');
      }
    }

    try {
      const result = await fn();
      this.failures = 0;
      this.state = 'closed';
      return result;
    } catch (error) {
      this.failures += 1;
      if (this.failures >= this.threshold) {
        this.state = 'open';
        this.openedAt = Date.now();
        this.logger?.error('Circuit breaker opened', error as Error);
      }
      throw error;
    }
  }
}
