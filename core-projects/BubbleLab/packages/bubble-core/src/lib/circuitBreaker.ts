export interface CircuitBreakerOptions {
  failureThreshold?: number;
  successThreshold?: number;
  timeout?: number;
  halfOpenAttempts?: number;
  monitoringPeriodMs?: number;
}

export class CircuitBreaker {
  private state: 'closed' | 'open' | 'half_open' = 'closed';
  private failureCount = 0;
  private successCount = 0;

  constructor(public name: string, private options: CircuitBreakerOptions = {}) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    try {
      const result = await fn();
      this.successCount += 1;
      return result;
    } catch (err) {
      this.failureCount += 1;
      throw err;
    }
  }
}
