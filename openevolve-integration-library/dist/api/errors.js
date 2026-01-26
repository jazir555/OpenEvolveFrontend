"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnknownError = exports.ServiceUnavailableError = exports.CircuitBreakerError = exports.RetryError = exports.ParseError = exports.CancellationError = exports.NetworkError = exports.ConfigurationError = exports.NotFoundError = exports.RateLimitError = exports.TimeoutError = exports.ExecutionError = exports.ValidationError = exports.AuthorizationError = exports.AuthenticationError = exports.ConnectionError = exports.IntegrationError = void 0;
exports.isIntegrationError = isIntegrationError;
exports.isRetryableError = isRetryableError;
exports.isCriticalError = isCriticalError;
exports.createIntegrationError = createIntegrationError;
class IntegrationError extends Error {
    constructor(integration, code, message, details) {
        super(message);
        this.integration = integration;
        this.code = code;
        this.details = details;
        this.name = 'IntegrationError';
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, IntegrationError);
        }
        this.timestamp = new Date().toISOString();
    }
    getRetryAfterMs() {
        if (this.details?.retryAfter) {
            return this.details.retryAfter * 1000;
        }
        return undefined;
    }
    toJSON() {
        return {
            name: this.name,
            integration: this.integration,
            code: this.code,
            message: this.message,
            details: this.details,
            timestamp: this.timestamp,
        };
    }
    static fromJSON(obj) {
        const error = new IntegrationError(obj.integration, obj.code, obj.message, obj.details);
        error.timestamp = obj.timestamp;
        return error;
    }
}
exports.IntegrationError = IntegrationError;
class ConnectionError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'CONNECTION_ERROR', message, {
            ...details,
            hint: 'Check if the integration service is running and accessible',
        });
        this.name = 'ConnectionError';
    }
}
exports.ConnectionError = ConnectionError;
class AuthenticationError extends IntegrationError {
    constructor(integration, message = 'Authentication failed', details) {
        super(integration, 'AUTHENTICATION_ERROR', message, {
            ...details,
            hint: 'Check API keys and authentication credentials',
        });
        this.name = 'AuthenticationError';
    }
}
exports.AuthenticationError = AuthenticationError;
class AuthorizationError extends IntegrationError {
    constructor(integration, message = 'Insufficient permissions', details) {
        super(integration, 'AUTHORIZATION_ERROR', message, {
            ...details,
            hint: 'Check if the user has required permissions',
        });
        this.name = 'AuthorizationError';
    }
}
exports.AuthorizationError = AuthorizationError;
class ValidationError extends IntegrationError {
    constructor(integration, errors, details) {
        const errorMessage = `Validation failed with ${errors.length} error(s)`;
        super(integration, 'VALIDATION_ERROR', errorMessage, {
            ...details,
            errors,
        });
        this.errors = errors;
        this.name = 'ValidationError';
    }
    getErrorMessages() {
        return this.errors.map((e) => `[${e.field}] ${e.message}`);
    }
    getErrorForField(field) {
        return this.errors.find((e) => e.field === field)?.message;
    }
}
exports.ValidationError = ValidationError;
class ExecutionError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'EXECUTION_ERROR', message, {
            ...details,
            hint: 'Check integration logs for more details',
        });
        this.name = 'ExecutionError';
    }
}
exports.ExecutionError = ExecutionError;
class TimeoutError extends IntegrationError {
    constructor(integration, timeout, details) {
        super(integration, 'TIMEOUT_ERROR', `Request timeout after ${timeout}ms`, {
            ...details,
            timeout,
            hint: 'Consider increasing timeout or optimizing the request',
        });
        this.name = 'TimeoutError';
    }
}
exports.TimeoutError = TimeoutError;
class RateLimitError extends IntegrationError {
    constructor(integration, retryAfter, details) {
        super(integration, 'RATE_LIMIT_ERROR', `Rate limit exceeded. Retry after ${retryAfter} seconds`, {
            ...details,
            retryAfter,
            hint: 'Implement exponential backoff or reduce request frequency',
        });
        this.name = 'RateLimitError';
    }
    getRetryAfterMs() {
        return super.getRetryAfterMs() || 5000;
    }
}
exports.RateLimitError = RateLimitError;
class NotFoundError extends IntegrationError {
    constructor(integration, resource, identifier, details) {
        super(integration, 'NOT_FOUND_ERROR', `${resource} '${identifier}' not found`, {
            ...details,
            resource,
            identifier,
            hint: 'Verify the resource exists and you have access to it',
        });
        this.name = 'NotFoundError';
    }
}
exports.NotFoundError = NotFoundError;
class ConfigurationError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'CONFIGURATION_ERROR', message, {
            ...details,
            hint: 'Check integration configuration',
        });
        this.name = 'ConfigurationError';
    }
}
exports.ConfigurationError = ConfigurationError;
class NetworkError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'NETWORK_ERROR', message, {
            ...details,
            hint: 'Check network connectivity and backend status',
        });
        this.name = 'NetworkError';
    }
}
exports.NetworkError = NetworkError;
class CancellationError extends IntegrationError {
    constructor(integration, executionId, details) {
        super(integration, 'CANCELLATION_ERROR', `Execution ${executionId} was cancelled`, {
            ...details,
            executionId,
            hint: 'The operation was cancelled by the user or system',
        });
        this.name = 'CancellationError';
    }
}
exports.CancellationError = CancellationError;
class ParseError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'PARSE_ERROR', message, {
            ...details,
            hint: 'Check response format and content-type',
        });
        this.name = 'ParseError';
    }
}
exports.ParseError = ParseError;
class RetryError extends IntegrationError {
    constructor(integration, originalError, attempts, details) {
        super(integration, 'RETRY_ERROR', `All ${attempts} retry attempts failed: ${originalError.message}`, {
            ...details,
            originalError: {
                name: originalError.name,
                message: originalError.message,
                stack: originalError.stack,
            },
            attempts,
            hint: 'Check backend status and network connectivity',
        });
        this.name = 'RetryError';
    }
}
exports.RetryError = RetryError;
class CircuitBreakerError extends IntegrationError {
    constructor(integration, details) {
        super(integration, 'CIRCUIT_OPEN_ERROR', `Circuit breaker is open for ${integration}. Request blocked.`, {
            ...details,
            hint: 'The integration has failed too many times recently and is currently in a cooldown period.',
        });
        this.name = 'CircuitBreakerError';
    }
}
exports.CircuitBreakerError = CircuitBreakerError;
class ServiceUnavailableError extends IntegrationError {
    constructor(integration, message = 'Service is currently unavailable', details) {
        super(integration, 'SERVICE_UNAVAILABLE_ERROR', message, {
            ...details,
            hint: 'The service is currently down or under heavy load. Please try again later.',
        });
        this.name = 'ServiceUnavailableError';
    }
}
exports.ServiceUnavailableError = ServiceUnavailableError;
class UnknownError extends IntegrationError {
    constructor(integration, message, details) {
        super(integration, 'UNKNOWN_ERROR', message, {
            ...details,
            hint: 'An unexpected error occurred. Please contact support if the issue persists.',
        });
        this.name = 'UnknownError';
    }
}
exports.UnknownError = UnknownError;
function isIntegrationError(error) {
    return error instanceof IntegrationError;
}
function isRetryableError(error) {
    if (!error)
        return false;
    if (error instanceof RetryError) {
        return false;
    }
    if (error instanceof TimeoutError || error instanceof NetworkError || error instanceof RateLimitError) {
        return true;
    }
    if (error instanceof IntegrationError) {
        if (['NETWORK_ERROR', 'TIMEOUT_ERROR', 'RATE_LIMIT_ERROR', 'CONNECTION_ERROR', 'SERVICE_UNAVAILABLE_ERROR'].includes(error.code)) {
            return true;
        }
        if (error.code === 'EXECUTION_ERROR' && (error.message.includes('Server error') || error.message.includes('502') || error.message.includes('503') || error.message.includes('504'))) {
            return true;
        }
        return false;
    }
    const err = error;
    const retryableCodes = ['ECONNABORTED', 'ECONNREFUSED', 'ETIMEDOUT', 'ERR_NETWORK', 'ENOTFOUND', 'EAI_AGAIN'];
    if (err.code && retryableCodes.includes(err.code)) {
        return true;
    }
    return false;
}
function isCriticalError(error) {
    if (!error)
        return false;
    return (error instanceof ValidationError ||
        error instanceof AuthenticationError ||
        error instanceof AuthorizationError ||
        error instanceof ConfigurationError ||
        error instanceof NotFoundError ||
        error instanceof CancellationError ||
        (error instanceof IntegrationError &&
            [
                'VALIDATION_ERROR',
                'AUTHENTICATION_ERROR',
                'AUTHORIZATION_ERROR',
                'CONFIGURATION_ERROR',
                'NOT_FOUND_ERROR',
                'CANCELLATION_ERROR',
            ].includes(error.code)));
}
function createIntegrationError(integration, error) {
    if (!error) {
        return new UnknownError(integration, 'An unknown error occurred with no error details');
    }
    if (error instanceof IntegrationError) {
        if (error.integration === 'backend' || error.integration === 'unknown' || error.integration === 'client_init') {
            error.integration = integration;
        }
        return error;
    }
    if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
        return new TimeoutError(integration, error.config?.timeout || 30000);
    }
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND' || error.code === 'EAI_AGAIN' || error.code === 'ERR_NETWORK') {
        return new ConnectionError(integration, `Failed to connect to backend: ${error.message || error.code}`, {
            code: error.code,
            originalError: error,
        });
    }
    if (error.name === 'AbortError') {
        return new TimeoutError(integration, 0, { hint: 'Operation was aborted or timed out' });
    }
    const response = error.response || (error.error && error.error.response);
    if (response) {
        const status = response.status;
        const data = response.data;
        const message = data?.message || data?.error || error.message || `HTTP error ${status}`;
        switch (status) {
            case 400:
                return new ValidationError(integration, data?.errors || [
                    { field: 'request', message: message, code: 'BAD_REQUEST' }
                ]);
            case 401:
                return new AuthenticationError(integration, message);
            case 403:
                return new AuthorizationError(integration, message);
            case 404:
                return new NotFoundError(integration, 'resource', error.config?.url || 'unknown');
            case 408:
                return new TimeoutError(integration, error.config?.timeout || 0, { status });
            case 429:
                return new RateLimitError(integration, data?.retryAfter || parseInt(response.headers?.['retry-after']) || 60);
            case 503:
                return new ServiceUnavailableError(integration, message);
            default:
                if (status >= 500) {
                    return new ExecutionError(integration, `Server error (${status}): ${message}`, { status, data });
                }
        }
    }
    const directStatus = error.status || (error.response && error.response.status) || (typeof error.code === 'number' ? error.code : undefined);
    if (directStatus === 503) {
        return new ServiceUnavailableError(integration, error.message || 'Service Unavailable');
    }
    if (error instanceof Error) {
        if (error.status === 503 || error.code === 503) {
            return new ServiceUnavailableError(integration, error.message);
        }
        return new UnknownError(integration, error.message, {
            name: error.name,
            stack: error.stack,
        });
    }
    if (typeof error === 'object' && error.message) {
        return new UnknownError(integration, String(error.message), { details: error });
    }
    return new UnknownError(integration, typeof error === 'string' ? error : 'Unknown error occurred', {
        originalError: error,
        errorType: typeof error
    });
}
//# sourceMappingURL=errors.js.map