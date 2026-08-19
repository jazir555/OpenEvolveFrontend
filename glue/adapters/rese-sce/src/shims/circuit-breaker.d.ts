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

export class CircuitBreaker {
  constructor(options: CircuitBreakerOptions);
  execute<T>(fn: () => Promise<T>): Promise<T>;
  getState(): CircuitState;
  getStats(): CircuitBreakerStats;
  reset(): void;
}
