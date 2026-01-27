/**
 * Common error handling utilities for Bubble implementations
 * Provides standardized error classes, error categorization, and error recovery strategies
 */
/**
 * Base error class for all Bubble-related errors
 */
export class BubbleError extends Error {
    code;
    statusCode;
    retryable;
    details;
    constructor(message, code, statusCode = 500, retryable = false, details) {
        super(message);
        this.code = code;
        this.statusCode = statusCode;
        this.retryable = retryable;
        this.details = details;
        this.name = this.constructor.name;
        Error.captureStackTrace?.(this, this.constructor);
    }
    toJSON() {
        return {
            name: this.name,
            message: this.message,
            code: this.code,
            statusCode: this.statusCode,
            retryable: this.retryable,
            details: this.details,
            stack: process.env.NODE_ENV === 'development' ? this.stack : undefined
        };
    }
}
/**
 * Authentication/authorization error
 */
export class AuthenticationError extends BubbleError {
    constructor(message = 'Authentication failed', details) {
        super(message, 'AUTH_ERROR', 401, false, details);
    }
}
/**
 * Authorization/permission error
 */
export class AuthorizationError extends BubbleError {
    constructor(message = 'Access denied', details) {
        super(message, 'AUTHZ_ERROR', 403, false, details);
    }
}
/**
 * Validation error (invalid input)
 */
export class ValidationError extends BubbleError {
    field;
    constructor(message, field, details) {
        super(message, 'VALIDATION_ERROR', 400, false, {
            ...details,
            field
        });
        this.field = field;
    }
}
/**
 * Not found error
 */
export class NotFoundError extends BubbleError {
    constructor(resource, identifier) {
        super(identifier ? `${resource} '${identifier}' not found` : `${resource} not found`, 'NOT_FOUND', 404, false, { resource, identifier });
    }
}
/**
 * Rate limiting error
 */
export class RateLimitError extends BubbleError {
    retryAfter;
    constructor(message = 'Rate limit exceeded', retryAfter, details) {
        super(message, 'RATE_LIMIT', 429, true, {
            ...details,
            retryAfter
        });
        this.retryAfter = retryAfter;
    }
}
/**
 * Network/transient error (retryable)
 */
export class NetworkError extends BubbleError {
    constructor(message = 'Network error', details) {
        super(message, 'NETWORK_ERROR', 503, true, details);
    }
}
/**
 * Timeout error
 */
export class TimeoutError extends BubbleError {
    timeout;
    constructor(message = 'Operation timed out', timeout, details) {
        super(message, 'TIMEOUT', 408, true, {
            ...details,
            timeout
        });
        this.timeout = timeout;
    }
}
/**
 * Configuration error
 */
export class ConfigurationError extends BubbleError {
    configKey;
    constructor(message, configKey, details) {
        super(message, 'CONFIG_ERROR', 500, false, {
            ...details,
            configKey
        });
        this.configKey = configKey;
    }
}
/**
 * External service error
 */
export class ExternalServiceError extends BubbleError {
    serviceCode;
    constructor(service, message, serviceCode, details) {
        super(`External service '${service}' error: ${message}`, 'EXTERNAL_SERVICE_ERROR', 502, true, {
            ...details,
            service,
            serviceCode
        });
        this.serviceCode = serviceCode;
    }
}
/**
 * Error categories for error handling strategies
 */
export var ErrorCategory;
(function (ErrorCategory) {
    ErrorCategory["TRANSIENT"] = "transient";
    ErrorCategory["PERMANENT"] = "permanent";
    ErrorCategory["THROTTLED"] = "throttled";
    ErrorCategory["UNKNOWN"] = "unknown";
})(ErrorCategory || (ErrorCategory = {}));
/**
 * Categorize an error for error handling strategy
 * @param error - Error to categorize
 * @returns Error category
 */
export function categorizeError(error) {
    if (error instanceof BubbleError) {
        if (error instanceof NetworkError || error instanceof TimeoutError) {
            return ErrorCategory.TRANSIENT;
        }
        if (error instanceof RateLimitError) {
            return ErrorCategory.THROTTLED;
        }
        if (error.retryable) {
            return ErrorCategory.TRANSIENT;
        }
        return ErrorCategory.PERMANENT;
    }
    if (error instanceof Error) {
        const errorMessage = error.message.toLowerCase();
        // Network-related errors
        if (errorMessage.includes('network') ||
            errorMessage.includes('econnrefused') ||
            errorMessage.includes('enotfound') ||
            errorMessage.includes('econnreset') ||
            errorMessage.includes('etimedout')) {
            return ErrorCategory.TRANSIENT;
        }
        // Timeout errors
        if (errorMessage.includes('timeout')) {
            return ErrorCategory.TRANSIENT;
        }
        // Rate limiting
        if (errorMessage.includes('rate limit') || errorMessage.includes('too many requests')) {
            return ErrorCategory.THROTTLED;
        }
    }
    return ErrorCategory.UNKNOWN;
}
/**
 * Determine if an error is retryable
 * @param error - Error to check
 * @returns True if error is retryable
 */
export function isRetryable(error) {
    const category = categorizeError(error);
    return category === ErrorCategory.TRANSIENT || category === ErrorCategory.THROTTLED;
}
/**
 * Create a standardized error response object
 * @param error - Error to format
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted error response
 */
export function createErrorResponse(error, correlationId) {
    if (error instanceof BubbleError) {
        return {
            error: error,
            message: error.message,
            code: error.code,
            statusCode: error.statusCode,
            correlationId,
            retryable: error.retryable,
            details: error.details
        };
    }
    if (error instanceof Error) {
        const category = categorizeError(error);
        return {
            error,
            message: error.message,
            code: category.toUpperCase(),
            statusCode: 500,
            correlationId,
            retryable: isRetryable(error),
            details: {
                originalMessage: error.message,
                stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
            }
        };
    }
    return {
        error: new Error(String(error)),
        message: String(error),
        code: 'UNKNOWN_ERROR',
        statusCode: 500,
        correlationId,
        retryable: false,
        details: { originalValue: error }
    };
}
/**
 * Wrap an error with additional context
 * @param error - Original error
 * @param context - Additional context
 * @returns Wrapped error
 */
export function wrapError(error, context) {
    if (error instanceof BubbleError) {
        // Add additional details to existing BubbleError
        return new BubbleError(context.message || error.message, context.code || error.code, error.statusCode, error.retryable, {
            ...error.details,
            ...context.details
        });
    }
    const message = error instanceof Error ? error.message : String(error);
    return new BubbleError(context.message || message, context.code || 'WRAPPED_ERROR', 500, false, {
        originalMessage: message,
        ...context.details
    });
}
/**
 * Create a success response object
 * @param data - Response data
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted success response
 */
export function createSuccessResponse(data, correlationId) {
    return {
        success: true,
        data,
        correlationId
    };
}
/**
 * Safe error parsing that never throws
 * @param error - Error to parse
 * @returns Parsed error object
 */
export function safeParseError(error) {
    if (error instanceof BubbleError) {
        return {
            message: error.message,
            stack: error.stack,
            code: error.code,
            statusCode: error.statusCode
        };
    }
    if (error instanceof Error) {
        return {
            message: error.message,
            stack: error.stack
        };
    }
    return {
        message: String(error)
    };
}
/**
 * Log error with consistent format
 * @param error - Error to log
 * @param context - Additional context
 */
export function logError(error, context) {
    const parsed = safeParseError(error);
    const logData = {
        ...parsed,
        ...context,
        timestamp: new Date().toISOString()
    };
    // Use console.error with structured data
    console.error(JSON.stringify(logData));
}
/**
 * Assert a condition is true, throw ValidationError if not
 * @param condition - Condition to check
 * @param message - Error message if condition is false
 * @param field - Field name for validation error
 */
export function assert(condition, message, field) {
    if (!condition) {
        throw new ValidationError(message, field);
    }
}
/**
 * Assert a value is not null/undefined, throw ValidationError if it is
 * @param value - Value to check
 * @param message - Error message if value is null/undefined
 * @param field - Field name for validation error
 */
export function assertNonNull(value, message = 'Value is required', field) {
    if (value === null || value === undefined) {
        throw new ValidationError(message, field);
    }
}
//# sourceMappingURL=error-handlers.js.map