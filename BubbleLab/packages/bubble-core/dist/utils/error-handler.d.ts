/**
 * Comprehensive Error Handling Utilities for Bubbles
 *
 * Provides standardized error handling, timeouts, retry logic, and circuit breakers
 * for all service and tool bubbles in the BubbleLab ecosystem.
 */
/**
 * Error categories with HTTP status codes
 */
export declare enum ErrorCategory {
    VALIDATION = "ValidationError",
    AUTHENTICATION = "AuthenticationError",
    AUTHORIZATION = "AuthorizationError",
    NOT_FOUND = "NotFoundError",
    CONFLICT = "ConflictError",
    RATE_LIMIT = "RateLimitError",
    SERVER_ERROR = "ServerError",
    NETWORK_ERROR = "NetworkError",
    TIMEOUT_ERROR = "TimeoutError",
    UNKNOWN_ERROR = "UnknownError"
}
/**
 * Mapping of error categories to HTTP status codes
 */
export declare const ErrorStatusCodes: Record<ErrorCategory, number>;
/**
 * Standardized error response structure
 */
export interface ErrorResponse<T = unknown> {
    success: false;
    error: {
        code: string;
        message: string;
        details?: T;
        timestamp: string;
        correlationId: string;
        statusCode: number;
        retryable: boolean;
    };
}
/**
 * Success response structure
 */
export interface SuccessResponse<T = unknown> {
    success: true;
    data: T;
    correlationId: string;
    timestamp: string;
}
/**
 * Categorize an error into a standard error category
 */
export declare function categorizeError(error: unknown, statusCode?: number): ErrorCategory;
/**
 * Sanitize error message to remove sensitive information
 */
export declare function sanitizeErrorMessage(message: string): string;
/**
 * Create a standardized error response
 */
export declare function createErrorResponse<T = unknown>(error: unknown, correlationId: string, details?: T): ErrorResponse<T>;
/**
 * Create a success response
 */
export declare function createSuccessResponse<T>(data: T, correlationId: string): SuccessResponse<T>;
/**
 * Generate a correlation ID for tracking requests
 */
export declare function generateCorrelationId(): string;
/**
 * Timeout wrapper for async operations
 */
export declare function withTimeout<T>(promise: Promise<T>, timeoutMs: number, operation: string): Promise<T>;
/**
 * Retry logic with exponential backoff
 */
export declare function retryWithBackoff<T>(operation: () => Promise<T>, options?: {
    maxAttempts?: number;
    baseDelayMs?: number;
    maxDelayMs?: number;
    correlationId?: string;
    operationName?: string;
    operation?: string;
}): Promise<T>;
/**
 * Circuit breaker state
 */
export declare enum CircuitState {
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"
}
/**
 * Circuit breaker configuration
 */
export interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeoutMs: number;
    monitoringPeriodMs: number;
}
/**
 * Circuit breaker implementation
 */
export declare class CircuitBreaker {
    private config;
    private state;
    private failureCount;
    private successCount;
    private lastFailureTime?;
    private nextAttemptTime?;
    constructor(config: CircuitBreakerConfig);
    /**
     * Execute an operation with circuit breaker protection
     */
    execute<T>(operation: () => Promise<T>, operationName: string): Promise<T>;
    private onSuccess;
    private onFailure;
    /**
     * Get current circuit state
     */
    getState(): CircuitState;
    /**
     * Reset circuit breaker to CLOSED state
     */
    reset(): void;
    /**
     * Get circuit breaker metrics
     */
    getMetrics(): {
        state: CircuitState;
        failureCount: number;
        successCount: number;
        lastFailureTime: number | undefined;
        nextAttemptTime: number | undefined;
    };
}
/**
 * Default circuit breaker configuration
 */
export declare const defaultCircuitBreakerConfig: CircuitBreakerConfig;
//# sourceMappingURL=error-handler.d.ts.map