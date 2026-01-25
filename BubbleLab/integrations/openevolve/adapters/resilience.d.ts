/**
 * Resilience Patterns for OpenEvolve Integration
 *
 * Implements Federation Constitution requirements:
 * - Circuit Breaker pattern for fault tolerance
 * - Exponential Backoff Retry with jitter for transient failures
 * - Request Deduplication for idempotency
 * - Dead Letter Queue for permanent failures
 * - Rate Limiting for API protection
 * - Input Validation for security
 * - Structured Logging for observability
 */
/**
 * Circuit Breaker States
 */
export declare enum CircuitBreakerState {
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"
}
/**
 * Circuit Breaker Configuration
 */
export interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    halfOpenAttempts: number;
}
/**
 * Thread-safe Circuit Breaker Implementation
 * Prevents cascading failures by temporarily disabling failing services
 */
export declare class CircuitBreaker {
    private config;
    private state;
    private failureCount;
    private successCount;
    private lastFailureTime;
    private halfOpenAttemptCount;
    private stateLock;
    constructor(config: CircuitBreakerConfig);
    /**
     * Execute operation with circuit breaker protection
     */
    execute<T>(operation: string, fn: () => Promise<T>): Promise<T>;
    /**
     * Record successful operation
     */
    onSuccess(): Promise<void>;
    /**
     * Record failed operation
     */
    onFailure(): Promise<void>;
    /**
     * Get current circuit breaker state
     */
    getState(): CircuitBreakerState;
    /**
     * Get statistics
     */
    getStats(): {
        state: CircuitBreakerState;
        failureCount: number;
        successCount: number;
        lastFailureTime: number;
        halfOpenAttemptCount: number;
    };
    /**
     * Reset circuit breaker to CLOSED state
     */
    reset(): Promise<void>;
    private transitionTo;
    private shouldAttemptReset;
    private acquireStateLock;
    private releaseStateLock;
    private validateConfig;
}
/**
 * Retry Configuration
 */
export interface RetryConfig {
    maxRetries: number;
    baseDelay: number;
    maxDelay: number;
    jitterMultiplier: number;
}
/**
 * Error classification for retry logic
 */
export declare function isTransientError(error: unknown): boolean;
/**
 * Retry with exponential backoff and jitter
 * Implements Federation Constitution requirement for fault tolerance
 */
export declare function retryWithBackoff<T>(operation: string, fn: () => Promise<T>, config: RetryConfig): Promise<T>;
/**
 * Request deduplicator for idempotency
 * Ensures that identical in-flight requests are not duplicated
 */
export declare class RequestDeduplicator {
    private pendingRequests;
    private completedRequests;
    private readonly ttl;
    constructor(ttl?: number);
    /**
     * Execute operation with deduplication
     * If a request with the same key is in flight, return its promise
     */
    execute<T>(key: string, operation: () => Promise<T>, options?: {
        cacheResult?: boolean;
        skipCache?: boolean;
    }): Promise<T>;
    /**
     * Clear cached results
     */
    clearCache(): void;
    /**
     * Clear cache for specific key
     */
    clearKey(key: string): void;
    /**
     * Get statistics
     */
    getStats(): {
        pendingRequests: number;
        completedRequests: number;
    };
    /**
     * Start periodic cleanup of expired cache entries
     */
    private startCleanupTimer;
}
/**
 * Dead Letter Queue Entry
 */
export interface DeadLetterEntry {
    id: string;
    operation: string;
    input: unknown;
    error: Error;
    timestamp: number;
    retryCount: number;
}
/**
 * In-memory Dead Letter Queue
 * In production, this should persist to a database or message queue
 */
export declare class DeadLetterQueue {
    private queue;
    private readonly maxSize;
    constructor(maxSize?: number);
    /**
     * Add failed operation to DLQ
     */
    push(operation: string, input: unknown, error: Error, retryCount?: number): Promise<void>;
    /**
     * Consume entries from DLQ
     */
    consume(limit?: number): Promise<DeadLetterEntry[]>;
    /**
     * Get all entries
     */
    getAll(): DeadLetterEntry[];
    /**
     * Clear queue
     */
    clear(): void;
    /**
     * Get queue size
     */
    size(): number;
}
/**
 * Combined resilience configuration
 */
export interface ResilienceConfig {
    circuitBreaker?: CircuitBreakerConfig;
    retry?: RetryConfig;
    deduplication?: {
        enabled: boolean;
        ttl?: number;
        cacheResult?: boolean;
    };
    deadLetterQueue?: {
        enabled: boolean;
        maxSize?: number;
    };
}
/**
 * Resilience wrapper that combines all patterns
 */
export declare class ResilienceWrapper {
    private operationName;
    private circuitBreaker?;
    private deduplicator?;
    private dlq?;
    private retryConfig;
    constructor(operationName: string, config: ResilienceConfig);
    /**
     * Execute operation with full resilience protection
     */
    execute<T>(key: string, operation: () => Promise<T>, input?: unknown): Promise<T>;
    /**
     * Get circuit breaker state
     */
    getCircuitBreakerState(): CircuitBreakerState | undefined;
    /**
     * Get circuit breaker stats
     */
    getCircuitBreakerStats(): {
        state: CircuitBreakerState;
        failureCount: number;
        successCount: number;
        lastFailureTime: number;
        halfOpenAttemptCount: number;
    } | undefined;
    /**
     * Get deduplicator stats
     */
    getDeduplicatorStats(): {
        pendingRequests: number;
        completedRequests: number;
    } | undefined;
    /**
     * Get DLQ entries
     */
    getDeadLetterEntries(): DeadLetterEntry[];
    /**
     * Clear DLQ
     */
    clearDeadLetterQueue(): void;
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker(): Promise<void>;
}
export interface RateLimitConfig {
    readonly maxRequests: number;
    readonly windowMs: number;
}
export declare class RateLimiter {
    private config;
    private requests;
    constructor(config: RateLimitConfig);
    checkLimit(identifier: string): boolean;
    getRemainingRequests(identifier: string): number;
    cleanup(): void;
}
export declare class InputValidator {
    static sanitizeString(input: string, maxLength?: number): string;
    static validateApiKey(apiKey: string): string;
}
export type LogLevel = 'info' | 'warn' | 'error' | 'debug';
export interface LogContext {
    readonly correlationId: string;
    readonly [key: string]: unknown;
}
export declare class StructuredLogger {
    private serviceName;
    constructor(serviceName: string);
    private log;
    info(data: Record<string, unknown>, error?: unknown): void;
    warn(data: Record<string, unknown>, error?: unknown): void;
    error(data: Record<string, unknown>, error?: unknown): void;
    debug(data: Record<string, unknown>, error?: unknown): void;
    child(context: LogContext): StructuredLogger;
}
export declare function sanitizeError(error: unknown): string;
export declare function generateCorrelationId(): string;
export declare const DEFAULT_CIRCUIT_BREAKER_CONFIG: CircuitBreakerConfig;
export declare const DEFAULT_RETRY_CONFIG: RetryConfig;
export declare const DEFAULT_RESILIENCE_CONFIG: ResilienceConfig;
//# sourceMappingURL=resilience.d.ts.map