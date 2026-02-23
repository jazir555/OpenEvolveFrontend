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
/**
 * Circuit Breaker implementation
 *
 * Prevents cascading failures by stopping calls to failing services
 */
export declare class CircuitBreaker {
    private state;
    private failure_count;
    private success_count;
    private last_failure_time?;
    private last_state_change;
    private next_attempt_time?;
    private readonly options;
    constructor(options: CircuitBreakerOptions);
    /**
     * Execute function through circuit breaker
     *
     * @param fn - Async function to execute
     * @returns Result of function execution
     * @throws Error if circuit is OPEN or function fails
     */
    execute<T>(fn: () => Promise<T>): Promise<T>;
    /**
     * Handle successful execution
     */
    private onSuccess;
    /**
     * Handle failed execution
     */
    private onFailure;
    /**
     * Check if enough time has passed to attempt a reset
     */
    private shouldAttemptReset;
    /**
     * Transition to new state
     */
    private transitionTo;
    /**
     * Get current circuit state
     */
    getState(): CircuitState;
    /**
     * Get circuit breaker statistics
     */
    getStats(): CircuitBreakerStats;
    /**
     * Manually reset circuit breaker to CLOSED state
     */
    reset(): void;
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
