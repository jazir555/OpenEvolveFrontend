/**
 * Custom Error Classes
 *
 * Provides specific error types for different error scenarios.
 * All errors include correlation IDs for tracing and error codes for programmatic handling.
 */

/**
 * Base error class with correlation ID support
 */
export class BaseError extends Error {
  constructor(
    message: string,
    public code: string,
    public correlationId?: string,
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
      correlation_id: this.correlationId,
      details: this.details,
    };
  }
}

/**
 * Network-related errors (connectivity, DNS, timeouts)
 */
export class NetworkError extends BaseError {
  constructor(
    message: string,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'NETWORK_ERROR', correlationId, details);
  }
}

/**
 * Authentication errors (401, invalid tokens, expired sessions)
 */
export class AuthenticationError extends BaseError {
  constructor(
    message: string,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'AUTHENTICATION_ERROR', correlationId, details);
  }
}

/**
 * Authorization errors (403, insufficient permissions)
 */
export class AuthorizationError extends BaseError {
  constructor(
    message: string,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'AUTHORIZATION_ERROR', correlationId, details);
  }
}

/**
 * Validation errors (400, malformed data, schema violations)
 */
export class ValidationError extends BaseError {
  constructor(
    message: string,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'VALIDATION_ERROR', correlationId, details);
  }
}

/**
 * Server errors (5xx, unexpected server behavior)
 */
export class ServerError extends BaseError {
  constructor(
    message: string,
    public statusCode: number,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'SERVER_ERROR', correlationId, { ...details, statusCode });
  }
}

/**
 * Not found errors (404, resource doesn't exist)
 */
export class NotFoundError extends BaseError {
  constructor(
    message: string,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(message, 'NOT_FOUND_ERROR', correlationId, details);
  }
}

/**
 * Rate limit errors (429, too many requests)
 */
export class RateLimitError extends BaseError {
  constructor(
    message: string,
    public retryAfter?: number,
    correlationId?: string,
    details?: Record<string, unknown>
  ) {
    super(
      message,
      'RATE_LIMIT_ERROR',
      correlationId,
      { ...details, retryAfter }
    );
  }
}

/**
 * Type guard to check if an error is our custom error type
 */
export function isCustomError(error: unknown): error is BaseError {
  return error instanceof BaseError;
}

/**
 * Extract error code from any error
 */
export function getErrorCode(error: unknown): string {
  if (isCustomError(error)) {
    return error.code;
  }
  if (error instanceof Error) {
    return 'GENERIC_ERROR';
  }
  return 'UNKNOWN_ERROR';
}

/**
 * Extract correlation ID from any error
 */
export function getErrorCorrelationId(error: unknown): string | undefined {
  if (isCustomError(error)) {
    return error.correlationId;
  }
  return undefined;
}

/**
 * Convert unknown error to string for display
 */
export function getErrorMessage(error: unknown): string {
  if (typeof error === 'string') {
    return error;
  }
  if (isCustomError(error)) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
}
