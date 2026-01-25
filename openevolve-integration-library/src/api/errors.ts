/**
 * Custom Error Classes for OpenEvolve Integration Library
 *
 * Provides typed error handling for all integration operations
 */

/**
 * Base integration error
 */
export class IntegrationError extends Error {
  /**
   * Create a new integration error
   *
   * @param integration - Integration name
   * @param code - Error code
   * @param message - Error message
   * @param details - Additional error details
   */
  constructor(
    public integration: string,
    public code: string,
    message: string,
    public details?: any
  ) {
    super(message);
    this.name = 'IntegrationError';

    // Maintain proper stack trace
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, IntegrationError);
    }

    // Include additional properties
    this.timestamp = new Date().toISOString();
  }

  /** Timestamp when error occurred */
  readonly timestamp: string;

  /** Get retry after duration in milliseconds if available */
  getRetryAfterMs(): number | undefined {
    if (this.details?.retryAfter) {
      return this.details.retryAfter * 1000;
    }
    return undefined;
  }

  /** Convert error to plain object */
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

  /** Create error from plain object */
  static fromJSON(obj: any): IntegrationError {
    const error = new IntegrationError(
      obj.integration,
      obj.code,
      obj.message,
      obj.details
    );
    (error as any).timestamp = obj.timestamp;
    return error;
  }
}

/**
 * Connection error - failed to connect to integration
 */
export class ConnectionError extends IntegrationError {
  /**
   * Create a new connection error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'CONNECTION_ERROR', message, {
      ...details,
      hint: 'Check if the integration service is running and accessible',
    });
    this.name = 'ConnectionError';
  }
}

/**
 * Authentication error - invalid credentials
 */
export class AuthenticationError extends IntegrationError {
  /**
   * Create a new authentication error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string = 'Authentication failed', details?: any) {
    super(integration, 'AUTHENTICATION_ERROR', message, {
      ...details,
      hint: 'Check API keys and authentication credentials',
    });
    this.name = 'AuthenticationError';
  }
}

/**
 * Authorization error - insufficient permissions
 */
export class AuthorizationError extends IntegrationError {
  /**
   * Create a new authorization error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string = 'Insufficient permissions', details?: any) {
    super(integration, 'AUTHORIZATION_ERROR', message, {
      ...details,
      hint: 'Check if the user has required permissions',
    });
    this.name = 'AuthorizationError';
  }
}

/**
 * Validation error - invalid input data
 */
export class ValidationError extends IntegrationError {
  /**
   * Create a new validation error
   *
   * @param integration - Integration name
   * @param errors - Array of validation errors
   * @param details - Additional details
   */
  constructor(
    integration: string,
    public errors: Array<{
      field: string;
      message: string;
      code: string;
    }>,
    details?: any
  ) {
    const errorMessage = `Validation failed with ${errors.length} error(s)`;
    super(integration, 'VALIDATION_ERROR', errorMessage, {
      ...details,
      errors,
    });
    this.name = 'ValidationError';
  }

  /** Get all error messages */
  getErrorMessages(): string[] {
    return this.errors.map((e) => `[${e.field}] ${e.message}`);
  }

  /** Get error message for specific field */
  getErrorForField(field: string): string | undefined {
    return this.errors.find((e) => e.field === field)?.message;
  }
}

/**
 * Execution error - error during integration execution
 */
export class ExecutionError extends IntegrationError {
  /**
   * Create a new execution error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details including stage, executionId, etc.
   */
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'EXECUTION_ERROR', message, {
      ...details,
      hint: 'Check integration logs for more details',
    });
    this.name = 'ExecutionError';
  }
}

/**
 * Timeout error - request took too long
 */
export class TimeoutError extends IntegrationError {
  /**
   * Create a new timeout error
   *
   * @param integration - Integration name
   * @param timeout - Timeout duration in milliseconds
   * @param details - Additional details
   */
  constructor(integration: string, timeout: number, details?: any) {
    super(
      integration,
      'TIMEOUT_ERROR',
      `Request timeout after ${timeout}ms`,
      {
        ...details,
        timeout,
        hint: 'Consider increasing timeout or optimizing the request',
      }
    );
    this.name = 'TimeoutError';
  }
}

/**
 * Rate limit error - too many requests
 */
export class RateLimitError extends IntegrationError {
  /**
   * Create a new rate limit error
   *
   * @param integration - Integration name
   * @param retryAfter - Seconds to wait before retrying
   * @param details - Additional details
   */
  constructor(integration: string, retryAfter: number, details?: any) {
    super(
      integration,
      'RATE_LIMIT_ERROR',
      `Rate limit exceeded. Retry after ${retryAfter} seconds`,
      {
        ...details,
        retryAfter,
        hint: 'Implement exponential backoff or reduce request frequency',
      }
    );
    this.name = 'RateLimitError';
  }

  /** Get retry after duration in milliseconds */
  getRetryAfterMs(): number {
    return super.getRetryAfterMs() || 5000;
  }
}

/**
 * Not found error - resource not found
 */
export class NotFoundError extends IntegrationError {
  /**
   * Create a new not found error
   *
   * @param integration - Integration name
   * @param resource - Resource type
   * @param identifier - Resource identifier
   * @param details - Additional details
   */
  constructor(integration: string, resource: string, identifier: string, details?: any) {
    super(
      integration,
      'NOT_FOUND_ERROR',
      `${resource} '${identifier}' not found`,
      {
        ...details,
        resource,
        identifier,
        hint: 'Verify the resource exists and you have access to it',
      }
    );
    this.name = 'NotFoundError';
  }
}

/**
 * Configuration error - invalid configuration
 */
export class ConfigurationError extends IntegrationError {
  /**
   * Create a new configuration error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'CONFIGURATION_ERROR', message, {
      ...details,
      hint: 'Check integration configuration',
    });
    this.name = 'ConfigurationError';
  }
}

/**
 * Network error - network-related issues
 */
export class NetworkError extends IntegrationError {
  /**
   * Create a new network error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'NETWORK_ERROR', message, {
      ...details,
      hint: 'Check network connectivity and backend status',
    });
    this.name = 'NetworkError';
  }
}

/**
 * Cancellation error - operation was cancelled
 */
export class CancellationError extends IntegrationError {
  /**
   * Create a new cancellation error
   *
   * @param integration - Integration name
   * @param executionId - Execution ID that was cancelled
   * @param details - Additional details
   */
  constructor(integration: string, executionId: string, details?: any) {
    super(
      integration,
      'CANCELLATION_ERROR',
      `Execution ${executionId} was cancelled`,
      {
        ...details,
        executionId,
        hint: 'The operation was cancelled by the user or system',
      }
    );
    this.name = 'CancellationError';
  }
}

/**
 * Parse error - failed to parse response
 */
export class ParseError extends IntegrationError {
  /**
   * Create a new parse error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'PARSE_ERROR', message, {
      ...details,
      hint: 'Check response format and content-type',
    });
    this.name = 'ParseError';
  }
}

/**
 * Retry error - all retry attempts exhausted
 */
export class RetryError extends IntegrationError {
  /**
   * Create a new retry error
   *
   * @param integration - Integration name
   * @param originalError - Original error that caused retries
   * @param attempts - Number of retry attempts
   * @param details - Additional details
   */
  constructor(integration: string, originalError: Error, attempts: number, details?: any) {
    super(
      integration,
      'RETRY_ERROR',
      `All ${attempts} retry attempts failed: ${originalError.message}`,
      {
        ...details,
        originalError: {
          name: originalError.name,
          message: originalError.message,
          stack: originalError.stack,
        },
        attempts,
        hint: 'Check backend status and network connectivity',
      }
    );
    this.name = 'RetryError';
  }
}

/**
 * Circuit breaker error - circuit is currently open
 */
export class CircuitBreakerError extends IntegrationError {
  /**
   * Create a new circuit breaker error
   *
   * @param integration - Integration name
   * @param details - Additional details
   */
  constructor(integration: string, details?: any) {
    super(
      integration,
      'CIRCUIT_OPEN_ERROR',
      `Circuit breaker is open for ${integration}. Request blocked.`,
      {
        ...details,
        hint: 'The integration has failed too many times recently and is currently in a cooldown period.',
      }
    );
    this.name = 'CircuitBreakerError';
  }
}

/**
 * Service unavailable error - backend service is down or overloaded
 */
export class ServiceUnavailableError extends IntegrationError {
  /**
   * Create a new service unavailable error
   *
   * @param integration - Integration name
   * @param message - Error message
   * @param details - Additional details
   */
  constructor(integration: string, message: string = 'Service is currently unavailable', details?: any) {
    super(integration, 'SERVICE_UNAVAILABLE_ERROR', message, {
      ...details,
      hint: 'The service is currently down or under heavy load. Please try again later.',
    });
    this.name = 'ServiceUnavailableError';
  }
}

/**
 * Unknown error - generic fallback for unhandled errors
 */


export class UnknownError extends IntegrationError {
  constructor(integration: string, message: string, details?: any) {
    super(integration, 'UNKNOWN_ERROR', message, {
      ...details,
      hint: 'An unexpected error occurred. Please contact support if the issue persists.',
    });
    this.name = 'UnknownError';
  }
}

/**
 * Check if an object is an IntegrationError
 */
export function isIntegrationError(error: any): error is IntegrationError {
  return error instanceof IntegrationError;
}

/**
 * Determine if an error is a retryable error
 *
 * @param error - Error to check
 * @returns Whether the error is retryable
 */
export function isRetryableError(error: any): boolean {
  if (!error) return false;

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
    
    // Retry on server errors
    if (error.code === 'EXECUTION_ERROR' && (error.message.includes('Server error') || error.message.includes('502') || error.message.includes('503') || error.message.includes('504'))) {
      return true;
    }

    
    return false;
  }

  // Axios/Network error fallbacks
  const err = error as any;
  const retryableCodes = ['ECONNABORTED', 'ECONNREFUSED', 'ETIMEDOUT', 'ERR_NETWORK', 'ENOTFOUND', 'EAI_AGAIN'];
  if (err.code && retryableCodes.includes(err.code)) {
    return true;
  }

  return false;
}

/**
 * Determine if an error is a critical error that should not be retried
 *
 * @param error - Error to check
 * @returns Whether the error is critical
 */
export function isCriticalError(error: any): boolean {
  if (!error) return false;

  return (
    error instanceof ValidationError ||
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
      ].includes(error.code))
  );
}

/**
 * Create an integration error from a generic error
 *
 * @param integration - Integration name
 * @param error - Generic error or Error object
 * @returns Integration error
 */
export function createIntegrationError(integration: string, error: any): IntegrationError {
  if (!error) {
    return new UnknownError(integration, 'An unknown error occurred with no error details');
  }

  if (error instanceof IntegrationError) {
    // If it's a generic integration error, update it with more specific context
    if (error.integration === 'backend' || error.integration === 'unknown' || error.integration === 'client_init') {
      error.integration = integration;
    }
    return error;
  }

  // Handle Axios/Network Errors
  if (error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT') {
    return new TimeoutError(integration, error.config?.timeout || 30000);
  }

  if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND' || error.code === 'EAI_AGAIN' || error.code === 'ERR_NETWORK') {
    return new ConnectionError(integration, `Failed to connect to backend: ${error.message || error.code}`, {
      code: error.code,
      originalError: error,
    });
  }

  // Handle AbortError
  if (error.name === 'AbortError') {
    return new TimeoutError(integration, 0, { hint: 'Operation was aborted or timed out' });
  }

  // Handle case where we pass an object with response property (Axios style)
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

  // Handle object that might have status/message directly (e.g. from tests or other libs)
  const directStatus = error.status || (error.response && error.response.status) || (typeof error.code === 'number' ? error.code : undefined);
  if (directStatus === 503) {
    return new ServiceUnavailableError(integration, error.message || 'Service Unavailable');
  }


  // Handle standard Error object
  if (error instanceof Error) {
    if ((error as any).status === 503 || (error as any).code === 503) {
      return new ServiceUnavailableError(integration, error.message);
    }
    return new UnknownError(integration, error.message, {
      name: error.name,
      stack: error.stack,
    });
  }

  // Handle object that might have message
  if (typeof error === 'object' && error.message) {
    return new UnknownError(integration, String(error.message), { details: error });
  }

  // Default to generic unknown error
  return new UnknownError(integration, typeof error === 'string' ? error : 'Unknown error occurred', {
    originalError: error,
    errorType: typeof error
  });
}


