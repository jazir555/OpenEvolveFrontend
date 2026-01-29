/**
 * Common error handling utilities for Bubble implementations
 * Provides standardized error classes, error categorization, and error recovery strategies
 */

/**
 * Base error class for all Bubble-related errors
 */
export class BubbleError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500,
    public retryable: boolean = false,
    public details?: Record<string, unknown>
  ) {
    super(message);
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
  constructor(message: string = 'Authentication failed', details?: Record<string, unknown>) {
    super(message, 'AUTH_ERROR', 401, false, details);
  }
}

/**
 * Authorization/permission error
 */
export class AuthorizationError extends BubbleError {
  constructor(message: string = 'Access denied', details?: Record<string, unknown>) {
    super(message, 'AUTHZ_ERROR', 403, false, details);
  }
}

/**
 * Validation error (invalid input)
 */
export class ValidationError extends BubbleError {
  constructor(message: string, public field?: string, details?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', 400, false, {
      ...details,
      field
    });
  }
}

/**
 * Not found error
 */
export class NotFoundError extends BubbleError {
  constructor(resource: string, identifier?: string) {
    super(
      identifier ? `${resource} '${identifier}' not found` : `${resource} not found`,
      'NOT_FOUND',
      404,
      false,
      { resource, identifier }
    );
  }
}

/**
 * Rate limiting error
 */
export class RateLimitError extends BubbleError {
  constructor(
    message: string = 'Rate limit exceeded',
    public retryAfter?: number,
    details?: Record<string, unknown>
  ) {
    super(message, 'RATE_LIMIT', 429, true, {
      ...details,
      retryAfter
    });
  }
}

/**
 * Network/transient error (retryable)
 */
export class NetworkError extends BubbleError {
  constructor(message: string = 'Network error', details?: Record<string, unknown>) {
    super(message, 'NETWORK_ERROR', 503, true, details);
  }
}

/**
 * Timeout error
 */
export class TimeoutError extends BubbleError {
  constructor(message: string = 'Operation timed out', public timeout: number, details?: Record<string, unknown>) {
    super(message, 'TIMEOUT', 408, true, {
      ...details,
      timeout
    });
  }
}

/**
 * Configuration error
 */
export class ConfigurationError extends BubbleError {
  constructor(message: string, public configKey?: string, details?: Record<string, unknown>) {
    super(message, 'CONFIG_ERROR', 500, false, {
      ...details,
      configKey
    });
  }
}

/**
 * External service error
 */
export class ExternalServiceError extends BubbleError {
  constructor(
    service: string,
    message: string,
    public serviceCode?: string,
    details?: Record<string, unknown>
  ) {
    super(
      `External service '${service}' error: ${message}`,
      'EXTERNAL_SERVICE_ERROR',
      502,
      true,
      {
        ...details,
        service,
        serviceCode
      }
    );
  }
}

/**
 * Error categories for error handling strategies
 */
export enum ErrorCategory {
  TRANSIENT = 'transient', // Temporary failures (network, timeout)
  PERMANENT = 'permanent', // Permanent failures (auth, validation)
  THROTTLED = 'throttled', // Rate limiting
  UNKNOWN = 'unknown'
}

/**
 * Categorize an error for error handling strategy
 * @param error - Error to categorize
 * @returns Error category
 */
export function categorizeError(error: unknown): ErrorCategory {
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
    if (
      errorMessage.includes('network') ||
      errorMessage.includes('econnrefused') ||
      errorMessage.includes('enotfound') ||
      errorMessage.includes('econnreset') ||
      errorMessage.includes('etimedout')
    ) {
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
export function isRetryable(error: unknown): boolean {
  const category = categorizeError(error);
  return category === ErrorCategory.TRANSIENT || category === ErrorCategory.THROTTLED;
}

/**
 * Create a standardized error response object
 * @param error - Error to format
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted error response
 */
export function createErrorResponse(
  error: unknown,
  correlationId: string
): {
  error: Error;
  message: string;
  code: string;
  statusCode: number;
  correlationId: string;
  retryable: boolean;
  details?: Record<string, unknown>;
} {
  if (error instanceof BubbleError) {
    return {
      error: error as Error,
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
export function wrapError(error: unknown, context: {
  message?: string;
  code?: string;
  details?: Record<string, unknown>;
}): BubbleError {
  if (error instanceof BubbleError) {
    // Add additional details to existing BubbleError
    return new BubbleError(
      context.message || error.message,
      context.code || error.code,
      error.statusCode,
      error.retryable,
      {
        ...error.details,
        ...context.details
      }
    );
  }

  const message = error instanceof Error ? error.message : String(error);
  return new BubbleError(
    context.message || message,
    context.code || 'WRAPPED_ERROR',
    500,
    false,
    {
      originalMessage: message,
      ...context.details
    }
  );
}

/**
 * Create a success response object
 * @param data - Response data
 * @param correlationId - Correlation ID for tracking
 * @returns Formatted success response
 */
export function createSuccessResponse<T>(
  data: T,
  correlationId: string
): {
  success: true;
  data: T;
  correlationId: string;
} {
  return {
    success: true as const,
    data,
    correlationId
  };
}

/**
 * Safe error parsing that never throws
 * @param error - Error to parse
 * @returns Parsed error object
 */
export function safeParseError(error: unknown): {
  message: string;
  stack?: string;
  code?: string;
  statusCode?: number;
} {
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
export function logError(error: unknown, context?: Record<string, unknown>): void {
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
export function assert(condition: boolean, message: string, field?: string): asserts condition {
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
export function assertNonNull<T>(
  value: T | null | undefined,
  message: string = 'Value is required',
  field?: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new ValidationError(message, field);
  }
}
