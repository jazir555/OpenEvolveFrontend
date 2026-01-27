/**
 * Comprehensive Error Handling Utilities for Bubbles
 *
 * Provides standardized error handling, timeouts, retry logic, and circuit breakers
 * for all service and tool bubbles in the BubbleLab ecosystem.
 */
import { randomUUID } from 'crypto';
/**
 * Error categories with HTTP status codes
 */
export var ErrorCategory;
(function (ErrorCategory) {
    ErrorCategory["VALIDATION"] = "ValidationError";
    ErrorCategory["AUTHENTICATION"] = "AuthenticationError";
    ErrorCategory["AUTHORIZATION"] = "AuthorizationError";
    ErrorCategory["NOT_FOUND"] = "NotFoundError";
    ErrorCategory["CONFLICT"] = "ConflictError";
    ErrorCategory["RATE_LIMIT"] = "RateLimitError";
    ErrorCategory["SERVER_ERROR"] = "ServerError";
    ErrorCategory["NETWORK_ERROR"] = "NetworkError";
    ErrorCategory["TIMEOUT_ERROR"] = "TimeoutError";
    ErrorCategory["UNKNOWN_ERROR"] = "UnknownError";
})(ErrorCategory || (ErrorCategory = {}));
/**
 * Mapping of error categories to HTTP status codes
 */
export const ErrorStatusCodes = {
    [ErrorCategory.VALIDATION]: 400,
    [ErrorCategory.AUTHENTICATION]: 401,
    [ErrorCategory.AUTHORIZATION]: 403,
    [ErrorCategory.NOT_FOUND]: 404,
    [ErrorCategory.CONFLICT]: 409,
    [ErrorCategory.RATE_LIMIT]: 429,
    [ErrorCategory.SERVER_ERROR]: 500,
    [ErrorCategory.NETWORK_ERROR]: 503,
    [ErrorCategory.TIMEOUT_ERROR]: 504,
    [ErrorCategory.UNKNOWN_ERROR]: 500,
};
/**
 * Type guard to check if an error is retryable
 */
function isRetryableError(error, statusCode) {
    // Retry on network errors
    if (error.name === 'NetworkError' || error.name === 'FetchError') {
        return true;
    }
    // Retry on timeout errors
    if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
        return true;
    }
    // Retry on 5xx server errors
    if (statusCode && statusCode >= 500 && statusCode < 600) {
        return true;
    }
    // Retry on 429 rate limit errors
    if (statusCode === 429) {
        return true;
    }
    // Do not retry on 4xx client errors (except 429)
    if (statusCode && statusCode >= 400 && statusCode < 500 && statusCode !== 429) {
        return false;
    }
    // Do not retry on validation, authentication, or authorization errors
    if (error.name.includes('Validation') ||
        error.name.includes('Authentication') ||
        error.name.includes('Authorization')) {
        return false;
    }
    return false;
}
/**
 * Categorize an error into a standard error category
 */
export function categorizeError(error, statusCode) {
    if (error instanceof Error) {
        const errorMessage = error.message.toLowerCase();
        const errorName = error.name.toLowerCase();
        // Check for timeout errors
        if (errorName.includes('timeout') ||
            errorMessage.includes('timeout') ||
            errorMessage.includes('timed out')) {
            return ErrorCategory.TIMEOUT_ERROR;
        }
        // Check for network errors
        if (errorName.includes('network') ||
            errorName.includes('fetch') ||
            errorMessage.includes('econnrefused') ||
            errorMessage.includes('enotfound') ||
            errorMessage.includes('etimedout')) {
            return ErrorCategory.NETWORK_ERROR;
        }
        // Check for authentication errors
        if (errorName.includes('authentication') ||
            errorMessage.includes('unauthorized') ||
            errorMessage.includes('invalid token') ||
            errorMessage.includes('not authenticated')) {
            return ErrorCategory.AUTHENTICATION;
        }
        // Check for authorization errors
        if (errorName.includes('authorization') ||
            errorMessage.includes('forbidden') ||
            errorMessage.includes('not allowed') ||
            errorMessage.includes('access denied')) {
            return ErrorCategory.AUTHORIZATION;
        }
        // Check for validation errors
        if (errorName.includes('validation') ||
            errorMessage.includes('invalid') ||
            errorMessage.includes('malformed')) {
            return ErrorCategory.VALIDATION;
        }
        // Check for not found errors
        if (errorName.includes('notfound') ||
            errorMessage.includes('not found') ||
            errorMessage.includes('does not exist')) {
            return ErrorCategory.NOT_FOUND;
        }
        // Check for conflict errors
        if (errorName.includes('conflict') ||
            errorMessage.includes('conflict') ||
            errorMessage.includes('already exists')) {
            return ErrorCategory.CONFLICT;
        }
        // Check for rate limit errors
        if (errorName.includes('ratelimit') ||
            errorMessage.includes('rate limit') ||
            errorMessage.includes('too many requests')) {
            return ErrorCategory.RATE_LIMIT;
        }
    }
    // Use status code if available
    if (statusCode) {
        if (statusCode === 400)
            return ErrorCategory.VALIDATION;
        if (statusCode === 401)
            return ErrorCategory.AUTHENTICATION;
        if (statusCode === 403)
            return ErrorCategory.AUTHORIZATION;
        if (statusCode === 404)
            return ErrorCategory.NOT_FOUND;
        if (statusCode === 409)
            return ErrorCategory.CONFLICT;
        if (statusCode === 429)
            return ErrorCategory.RATE_LIMIT;
        if (statusCode >= 500)
            return ErrorCategory.SERVER_ERROR;
    }
    return ErrorCategory.UNKNOWN_ERROR;
}
/**
 * Sanitize error message to remove sensitive information
 */
export function sanitizeErrorMessage(message) {
    // Remove credentials
    let sanitized = message.replace(/(password|token|key|secret|authorization)["\s:=]+["']?[^"'\s,)}]+["']?/gi, '$1=[REDACTED]');
    // Remove file paths (Windows and Unix)
    sanitized = sanitized.replace(/[a-zA-Z]:\\[^'"`\s,)}]+/g, '[PATH]');
    sanitized = sanitized.replace(/\/[^'"`\s,)}]+/g, '[PATH]');
    // Remove IP addresses
    sanitized = sanitized.replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, '[IP]');
    // Remove stack traces in production
    const lines = sanitized.split('\n');
    const filteredLines = lines.filter((line) => {
        const trimmed = line.trim();
        // Keep error message, remove stack trace lines
        return (trimmed.length > 0 &&
            !trimmed.startsWith('at ') &&
            !trimmed.startsWith('    at ') &&
            !trimmed.includes('node_modules/'));
    });
    return filteredLines.join('\n');
}
/**
 * Create a standardized error response
 */
export function createErrorResponse(error, correlationId, details) {
    const category = categorizeError(error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    const sanitizedMessage = sanitizeErrorMessage(errorMessage);
    return {
        success: false,
        error: {
            code: category,
            message: sanitizedMessage,
            details,
            timestamp: new Date().toISOString(),
            correlationId,
            statusCode: ErrorStatusCodes[category],
            retryable: isRetryableError(error instanceof Error ? error : new Error(String(error))),
        },
    };
}
/**
 * Create a success response
 */
export function createSuccessResponse(data, correlationId) {
    return {
        success: true,
        data,
        correlationId,
        timestamp: new Date().toISOString(),
    };
}
/**
 * Generate a correlation ID for tracking requests
 */
export function generateCorrelationId() {
    return randomUUID();
}
/**
 * Timeout wrapper for async operations
 */
export async function withTimeout(promise, timeoutMs, operation) {
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
            reject(new Error(`Operation '${operation}' timed out after ${timeoutMs}ms`));
        }, timeoutMs);
    });
    return Promise.race([promise, timeoutPromise]);
}
/**
 * Retry logic with exponential backoff
 */
export async function retryWithBackoff(operation, options = {}) {
    const { maxAttempts = 3, baseDelayMs = 1000, maxDelayMs = 16000, correlationId = generateCorrelationId(), operationName = 'operation', } = options;
    let lastError;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await operation();
        }
        catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
            const category = categorizeError(lastError);
            const retryable = isRetryableError(lastError);
            if (!retryable || attempt === maxAttempts) {
                throw lastError;
            }
            // Calculate delay with exponential backoff and jitter
            const delay = Math.min(baseDelayMs * Math.pow(2, attempt - 1) + Math.random() * 1000, maxDelayMs);
            console.warn(`[${correlationId}] ${operationName} failed (attempt ${attempt}/${maxAttempts}): ${lastError.message}. Retrying in ${Math.round(delay)}ms...`);
            await new Promise((resolve) => setTimeout(resolve, delay));
        }
    }
    throw lastError;
}
/**
 * Circuit breaker state
 */
export var CircuitState;
(function (CircuitState) {
    CircuitState["CLOSED"] = "closed";
    CircuitState["OPEN"] = "open";
    CircuitState["HALF_OPEN"] = "half_open";
})(CircuitState || (CircuitState = {}));
/**
 * Circuit breaker implementation
 */
export class CircuitBreaker {
    config;
    state = CircuitState.CLOSED;
    failureCount = 0;
    successCount = 0;
    lastFailureTime;
    nextAttemptTime;
    constructor(config) {
        this.config = config;
    }
    /**
     * Execute an operation with circuit breaker protection
     */
    async execute(operation, operationName) {
        // Check if circuit should transition from OPEN to HALF_OPEN
        if (this.state === CircuitState.OPEN &&
            this.nextAttemptTime &&
            Date.now() >= this.nextAttemptTime) {
            this.state = CircuitState.HALF_OPEN;
            this.successCount = 0;
            console.warn(`[CircuitBreaker] ${operationName}: Transitioning to HALF_OPEN`);
        }
        // Reject if circuit is OPEN
        if (this.state === CircuitState.OPEN) {
            const retryAfter = Math.max(0, Math.ceil(((this.nextAttemptTime || 0) - Date.now()) / 1000));
            throw new Error(`Circuit breaker is OPEN for ${operationName}. Retry after ${retryAfter}s`);
        }
        try {
            const result = await operation();
            this.onSuccess(operationName);
            return result;
        }
        catch (error) {
            this.onFailure(operationName);
            throw error;
        }
    }
    onSuccess(operationName) {
        if (this.state === CircuitState.HALF_OPEN) {
            this.successCount++;
            if (this.successCount >= this.config.successThreshold) {
                this.state = CircuitState.CLOSED;
                this.failureCount = 0;
                console.log(`[CircuitBreaker] ${operationName}: Circuit CLOSED (recovered)`);
            }
        }
        else {
            // Reset failure count on success in CLOSED state
            this.failureCount = 0;
        }
    }
    onFailure(operationName) {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        if (this.state === CircuitState.HALF_OPEN) {
            // Immediate transition back to OPEN if failure in HALF_OPEN
            this.state = CircuitState.OPEN;
            this.nextAttemptTime = Date.now() + this.config.monitoringPeriodMs;
            console.warn(`[CircuitBreaker] ${operationName}: Circuit OPEN (failed in HALF_OPEN)`);
        }
        else if (this.failureCount >= this.config.failureThreshold) {
            // Open circuit after threshold reached
            this.state = CircuitState.OPEN;
            this.nextAttemptTime = Date.now() + this.config.monitoringPeriodMs;
            console.warn(`[CircuitBreaker] ${operationName}: Circuit OPEN (${this.failureCount} failures)`);
        }
    }
    /**
     * Get current circuit state
     */
    getState() {
        return this.state;
    }
    /**
     * Reset circuit breaker to CLOSED state
     */
    reset() {
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
        this.lastFailureTime = undefined;
        this.nextAttemptTime = undefined;
    }
    /**
     * Get circuit breaker metrics
     */
    getMetrics() {
        return {
            state: this.state,
            failureCount: this.failureCount,
            successCount: this.successCount,
            lastFailureTime: this.lastFailureTime,
            nextAttemptTime: this.nextAttemptTime,
        };
    }
}
/**
 * Default circuit breaker configuration
 */
export const defaultCircuitBreakerConfig = {
    failureThreshold: 5,
    successThreshold: 2,
    timeoutMs: 30000,
    monitoringPeriodMs: 60000,
};
//# sourceMappingURL=error-handler.js.map