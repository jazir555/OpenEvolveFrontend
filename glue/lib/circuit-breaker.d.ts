export declare enum CircuitState {
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"
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
export declare class CircuitBreaker {
    private state;
    private failure_count;
    private success_count;
    private last_failure_time?;
    private last_state_change;
    private next_attempt_time?;
    private readonly options;
    constructor(options: CircuitBreakerOptions);
    execute<T>(fn: () => Promise<T>): Promise<T>;
    private onSuccess;
    private onFailure;
    private shouldAttemptReset;
    private transitionTo;
    getState(): CircuitState;
    getStats(): CircuitBreakerStats;
    reset(): void;
}
//# sourceMappingURL=circuit-breaker.d.ts.map