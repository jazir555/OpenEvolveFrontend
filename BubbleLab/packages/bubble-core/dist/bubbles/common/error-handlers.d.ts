/**
 * Common error handling utilities for Bubble implementations
 * Provides standardized error classes, error categorization, and error recovery strategies
 */
/**
 * Base error class for all Bubble-related errors
 */
export declare class BubbleError extends Error {
    code: string;
    statusCode: number;
    retryable: boolean;
    details?: Record<string, unknown> | undefined;
    constructor(message: string, code: string, statusCode?: number, retryable?: boolean, details?: Record<string, unknown> | undefined);
    toJSON(): {
        name: string;
        message: string;
        code: string;
        statusCode: number;
        retryable: boolean;
        details: Record<string, unknown> | undefined;
        stack: string | undefined;
    };
}
/**
 * Authentication/authorization error
 */
export declare class AuthenticationError extends BubbleError {
    constructor(message?: string, details?: Record<string, unknown>);
}
/**
 * Authorization/permission error
 */
export declare class AuthorizationError extends BubbleError {
    constructor(message?: string, details?: Record<string, unknown>);
}
/**
 * Validation error (invalid input)
 */
export declare class ValidationError extends BubbleError {
    field?: string | undefined;
    constructor(message: string, field?: string | undefined, details?: Record<string, unknown>);
}
/**
 * Not found error
 */
export declare class NotFoundError extends BubbleError {
    constructor(resource: string, identifier?: string);
}
/**
 * Rate limiting error
 */
export declare class RateLimitError extends BubbleError {
    retryAfter?: number | undefined;
    constructor(message?: string, retryAfter?: number | undefined, details?: Record<string, unknown>);
}
/**
 * Network/transient error (retryable)
 */
export declare class NetworkError extends BubbleError {
    constructor(message?: string, details?: Record<string, unknown>);
}
/**
 * Timeout error
 */
export declare class TimeoutError extends BubbleError {
    timeout: number;
    constructor(message: string | undefined, timeout: number, details?: Record<string, unknown>);
}
/**
 * Configuration error
 */
export declare class ConfigurationError extends BubbleError {
    configKey?: string | undefined;
    constructor(message: string, configKey?: string | undefined, details?: Record<string, unknown>);
}
/**
 * External service error
 */
export declare class ExternalServiceError extends BubbleError {
    serviceCode?: string | undefined;
    constructor(service: string, message: string, serviceCode?: string | undefined, details?: Record<string, unknown>);
}
/**
 * Error categories for error handling strategies
 */
export declare enum ErrorCategory {
    TRANSIENT = "transient",// Temporary failures (network, timeout)
    PERMANENT = "permanent",// Permanent failures (auth, validation)
    THROTTLED = "throttled",// Rate limiting
    UNKNOWN = "unknown"
}
/**
 * Categorize an error for error handling strategy
 * @param error - Error to categorize
 * @returns Error category
 */
export declare function categorizeError(error: unknown): ErrorCategory;
/**
 * Determine if an error is retryable
 * @param error - Error to check
 * @returns True if error is retryable
 */
export declare function isRetryable(error: unknown): boolean;
/**
 * Create a standardized error response object
 * @param error - Error to format
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted error response
 */
export declare function createErrorResponse(error: unknown, correlationId: string): {
    error: Error;
    message: string;
    code: string;
    statusCode: number;
    correlationId: string;
    retryable: boolean;
    details?: Record<string, unknown>;
};
/**
 * Wrap an error with additional context
 * @param error - Original error
 * @param context - Additional context
 * @returns Wrapped error
 */
export declare function wrapError(error: unknown, context: {
    message?: string;
    code?: string;
    details?: Record<string, unknown>;
}): BubbleError;
/**
 * Create a success response object
 * @param data - Response data
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted success response
 */
export declare function createSuccessResponse<T>(data: T, correlationId: string): {
    success: true;
    data: T;
    correlationId: string;
};
/**
 * Safe error parsing that never throws
 * @param error - Error to parse
 * @returns Parsed error object
 */
export declare function safeParseError(error: unknown): {
    message: string;
    stack?: string;
    code?: string;
    statusCode?: number;
};
/**
 * Log error with consistent format
 * @param error - Error to log
 * @param context - Additional context
 */
export declare function logError(error: unknown, context?: Record<string, unknown>): void;
/**
 * Assert a condition is true, throw ValidationError if not
 * @param condition - Condition to check
 * @param message - Error message if condition is false
 * @param field - Field name for validation error
 */
export declare function assert(condition: boolean, message: string, field?: string): asserts condition;
/**
 * Assert a value is not null/undefined, throw ValidationError if it is
 * @param value - Value to check
 * @param message - Error message if value is null/undefined
 * @param field - Field name for validation error
 */
export declare function assertNonNull<T>(value: T | null | undefined, message?: string, field?: string): asserts value is T;
//# sourceMappingURL=error-handlers.d.ts.map