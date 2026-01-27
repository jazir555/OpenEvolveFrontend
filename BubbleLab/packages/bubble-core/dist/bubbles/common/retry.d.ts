/**
 * Common retry utilities for Bubble implementations
 * Provides exponential backoff, jitter, and circuit breaker patterns
 */
/**
 * Retry configuration options
 */
export interface RetryOptions {
    /** Maximum number of retry attempts (default: 3) */
    maxAttempts?: number;
    /** Base delay in milliseconds (default: 1000) */
    baseDelayMs?: number;
    /** Maximum delay in milliseconds (default: 30000) */
    maxDelayMs?: number;
    /** Exponential backoff multiplier (default: 2) */
    backoffMultiplier?: number;
    /** Whether to add jitter to prevent thundering herd (default: true) */
    jitter?: boolean;
    /** Jitter amount as percentage (0-1, default: 0.1) */
    jitterAmount?: number;
    /** Function to determine if an error is retryable */
    isRetryable?: (error: unknown) => boolean;
    /** Callback called before each retry attempt */
    onRetry?: (attempt: number, error: unknown) => void;
    /** Correlation ID for logging */
    correlationId?: string;
    /** Operation name for logging */
    operation?: string;
}
/**
 * Default retry configuration
 */
export declare const DEFAULT_RETRY_OPTIONS: Required<RetryOptions>;
/**
 * Calculate delay with exponential backoff and jitter
 * @param attempt - Current attempt number (0-indexed)
 * @param options - Retry options
 * @returns Delay in milliseconds
 */
export declare function calculateDelay(attempt: number, options: Required<RetryOptions>): number;
/**
 * Sleep for a specified duration
 * @param ms - Milliseconds to sleep
 * @returns Promise that resolves after sleep duration
 */
export declare function sleep(ms: number): Promise<void>;
/**
 * Retry a function with exponential backoff
 * @param fn - Function to retry
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 * @throws Last error if all retries are exhausted
 */
export declare function retryWithBackoff<T>(fn: () => Promise<T>, options?: RetryOptions): Promise<T>;
/**
 * Retry with a timeout for each attempt
 * @param fn - Function to retry
 * @param timeoutMs - Timeout in milliseconds for each attempt
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 * @throws TimeoutError if timeout is exceeded
 */
export declare function retryWithTimeout<T>(fn: () => Promise<T>, timeoutMs: number, options?: RetryOptions): Promise<T>;
/**
 * Wrap a promise with a timeout
 * @param promise - Promise to wrap
 * @param timeoutMs - Timeout in milliseconds
 * @param operation - Operation name for error messages
 * @returns Promise that resolves or rejects with TimeoutError
 */
export declare function withTimeout<T>(promise: Promise<T>, timeoutMs: number, operation?: string): Promise<T>;
/**
 * Circuit breaker state
 */
export declare enum CircuitBreakerState {
    CLOSED = "closed",// Normal operation
    OPEN = "open",// Failing, reject requests
    HALF_OPEN = "half_open"
}
/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerOptions {
    /** Number of failures before opening circuit (default: 5) */
    failureThreshold?: number;
    /** Number of successes before closing circuit (default: 2) */
    successThreshold?: number;
    /** Timeout in milliseconds before attempting recovery (default: 60000) */
    timeoutMs?: number;
    /** Monitoring period for resetting failure count (default: 60000) */
    monitoringPeriodMs?: number;
    /** Callback when circuit opens */
    onOpen?: () => void;
    /** Callback when circuit closes */
    onClose?: () => void;
    /** Callback when circuit enters half-open state */
    onHalfOpen?: () => void;
}
/**
 * Circuit breaker for preventing cascading failures
 */
export declare class CircuitBreaker {
    private options;
    name: string;
    private state;
    private failureCount;
    private successCount;
    private lastFailureTime?;
    private openedAt?;
    constructor(options?: CircuitBreakerOptions, name?: string);
    /**
     * Execute a function with circuit breaker protection
     * @param fn - Function to execute
     * @returns Promise that resolves with the function result
     * @throws Error if circuit is open or function fails
     */
    execute<T>(fn: () => Promise<T>, operationName?: string): Promise<T>;
    /**
     * Handle successful execution
     */
    private onSuccess;
    /**
     * Handle failed execution
     */
    private onFailure;
    /**
     * Check if circuit should attempt reset
     */
    private shouldAttemptReset;
    /**
     * Transition to a new state
     */
    private transitionTo;
    /**
     * Get current circuit breaker state
     */
    getState(): CircuitBreakerState;
    /**
     * Get circuit breaker stats
     */
    getStats(): {
        state: CircuitBreakerState;
        failureCount: number;
        successCount: number;
        openedAt?: number;
    };
    /**
     * Reset circuit breaker to closed state
     */
    reset(): void;
}
/**
 * Default circuit breaker configuration for external services
 */
export declare const defaultCircuitBreakerConfig: CircuitBreakerOptions;
/**
 * Combine retry and circuit breaker patterns
 * @param fn - Function to execute
 * @param circuitBreaker - Circuit breaker instance
 * @param retryOptions - Retry options
 * @returns Promise that resolves with the function result
 */
export declare function executeWithResilience<T>(fn: () => Promise<T>, circuitBreaker: CircuitBreaker, retryOptions?: RetryOptions): Promise<T>;
//# sourceMappingURL=retry.d.ts.map