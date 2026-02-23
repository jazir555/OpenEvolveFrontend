/**
 * Circuit Breaker Pattern Implementation
 * Per CLAUDE.md Section 2.3: System Failure → Circuit Breaker
 * Stop hammering the dead service. Wait for a health check to pass.
 */
import { LogContext } from './structuredLogger';
export declare enum CircuitState {
    CLOSED = "closed",// Normal operation
    OPEN = "open",// Circuit is open, rejecting requests
    HALF_OPEN = "half-open"
}
export interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeoutMs: number;
    monitoringPeriodMs: number;
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
export declare class CircuitBreaker {
    private state;
    private failureCount;
    private successCount;
    private lastFailureTime?;
    private lastSuccessTime?;
    private openedAt?;
    private nextAttemptAt?;
    private config;
    private serviceName;
    constructor(serviceName: string, config: CircuitBreakerConfig);
    /**
     * Execute operation with circuit breaker protection
     */
    execute<T>(operation: () => Promise<T>, context?: LogContext): Promise<T>;
    /**
     * Handle successful operation
     */
    private onSuccess;
    /**
     * Handle failed operation
     */
    private onFailure;
    /**
     * Open circuit (stop accepting requests)
     */
    private openCircuit;
    /**
     * Close circuit (resume normal operation)
     */
    private closeCircuit;
    /**
     * Get current circuit breaker state
     */
    getState(): CircuitBreakerStats;
    /**
     * Reset circuit breaker to closed state
     */
    reset(): void;
}
/**
 * Custom error for circuit breaker open state
 */
export declare class CircuitBreakerOpenError extends Error {
    constructor(message: string);
}
/**
 * Circuit breaker registry for managing multiple circuit breakers
 */
export declare class CircuitBreakerRegistry {
    private breakers;
    /**
     * Get or create circuit breaker for a service
     */
    get(serviceName: string, config?: CircuitBreakerConfig): CircuitBreaker;
    /**
     * Get stats for all circuit breakers
     */
    getAllStats(): Map<string, CircuitBreakerStats>;
    /**
     * Reset all circuit breakers
     */
    resetAll(): void;
    /**
     * Remove circuit breaker for a service
     */
    remove(serviceName: string): void;
}
export declare const circuitBreakerRegistry: CircuitBreakerRegistry;
//# sourceMappingURL=circuitBreaker.d.ts.map