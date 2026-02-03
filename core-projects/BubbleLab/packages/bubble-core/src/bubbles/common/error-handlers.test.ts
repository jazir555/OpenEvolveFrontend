/**
 * Comprehensive tests for common error handling utilities
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  BubbleError,
  AuthenticationError,
  AuthorizationError,
  ValidationError,
  NotFoundError,
  RateLimitError,
  NetworkError,
  TimeoutError,
  ConfigurationError,
  ExternalServiceError,
  ErrorCategory,
  categorizeError,
  isRetryable,
  createErrorResponse,
  wrapError,
  createSuccessResponse,
  safeParseError,
  logError,
  assert,
  assertNonNull
} from './error-handlers.js';

describe('error-handlers', () => {
  describe('BubbleError', () => {
    it('should create base error with all properties', () => {
      const details = { userId: '123', action: 'test' };
      const error = new BubbleError('Test error', 'TEST_CODE', 400, true, details);

      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_CODE');
      expect(error.statusCode).toBe(400);
      expect(error.retryable).toBe(true);
      expect(error.details).toEqual(details);
      expect(error.name).toBe('BubbleError');
    });

    it('should serialize to JSON correctly', () => {
      const error = new BubbleError('Test error', 'TEST_CODE', 500, false, { key: 'value' });
      const json = error.toJSON();

      expect(json.name).toBe('BubbleError');
      expect(json.message).toBe('Test error');
      expect(json.code).toBe('TEST_CODE');
      expect(json.statusCode).toBe(500);
      expect(json.retryable).toBe(false);
      expect(json.details).toEqual({ key: 'value' });
    });

    it('should include stack trace in development mode', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      const error = new BubbleError('Test error', 'TEST_CODE');
      const json = error.toJSON();

      expect(json.stack).toBeDefined();
      expect(typeof json.stack).toBe('string');

      process.env.NODE_ENV = originalEnv;
    });

    it('should not include stack trace in production mode', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'production';

      const error = new BubbleError('Test error', 'TEST_CODE');
      const json = error.toJSON();

      expect(json.stack).toBeUndefined();

      process.env.NODE_ENV = originalEnv;
    });
  });

  describe('AuthenticationError', () => {
    it('should create authentication error with defaults', () => {
      const error = new AuthenticationError();

      expect(error.message).toBe('Authentication failed');
      expect(error.code).toBe('AUTH_ERROR');
      expect(error.statusCode).toBe(401);
      expect(error.retryable).toBe(false);
    });

    it('should create authentication error with custom message and details', () => {
      const details = { userId: '123' };
      const error = new AuthenticationError('Invalid token', details);

      expect(error.message).toBe('Invalid token');
      expect(error.details).toEqual(details);
    });
  });

  describe('AuthorizationError', () => {
    it('should create authorization error with defaults', () => {
      const error = new AuthorizationError();

      expect(error.message).toBe('Access denied');
      expect(error.code).toBe('AUTHZ_ERROR');
      expect(error.statusCode).toBe(403);
      expect(error.retryable).toBe(false);
    });
  });

  describe('ValidationError', () => {
    it('should create validation error with field', () => {
      const error = new ValidationError('Invalid email', 'email');

      expect(error.message).toBe('Invalid email');
      expect(error.code).toBe('VALIDATION_ERROR');
      expect(error.statusCode).toBe(400);
      expect(error.field).toBe('email');
      expect(error.details).toEqual({ field: 'email' });
    });

    it('should create validation error without field', () => {
      const error = new ValidationError('Invalid data');

      expect(error.field).toBeUndefined();
    });
  });

  describe('NotFoundError', () => {
    it('should create not found error with identifier', () => {
      const error = new NotFoundError('User', '123');

      expect(error.message).toBe("User '123' not found");
      expect(error.code).toBe('NOT_FOUND');
      expect(error.statusCode).toBe(404);
      expect(error.details).toEqual({ resource: 'User', identifier: '123' });
    });

    it('should create not found error without identifier', () => {
      const error = new NotFoundError('User');

      expect(error.message).toBe('User not found');
      expect(error.details).toEqual({ resource: 'User', identifier: undefined });
    });
  });

  describe('RateLimitError', () => {
    it('should create rate limit error with defaults', () => {
      const error = new RateLimitError();

      expect(error.message).toBe('Rate limit exceeded');
      expect(error.code).toBe('RATE_LIMIT');
      expect(error.statusCode).toBe(429);
      expect(error.retryable).toBe(true);
      expect(error.retryAfter).toBeUndefined();
    });

    it('should create rate limit error with retryAfter', () => {
      const error = new RateLimitError('Too many requests', 60);

      expect(error.message).toBe('Too many requests');
      expect(error.retryAfter).toBe(60);
      expect(error.details).toEqual({ retryAfter: 60 });
    });
  });

  describe('NetworkError', () => {
    it('should create network error with defaults', () => {
      const error = new NetworkError();

      expect(error.message).toBe('Network error');
      expect(error.code).toBe('NETWORK_ERROR');
      expect(error.statusCode).toBe(503);
      expect(error.retryable).toBe(true);
    });
  });

  describe('TimeoutError', () => {
    it('should create timeout error with timeout value', () => {
      const error = new TimeoutError('Request timed out', 5000);

      expect(error.message).toBe('Request timed out');
      expect(error.code).toBe('TIMEOUT');
      expect(error.statusCode).toBe(408);
      expect(error.retryable).toBe(true);
      expect(error.timeout).toBe(5000);
      expect(error.details).toEqual({ timeout: 5000 });
    });
  });

  describe('ConfigurationError', () => {
    it('should create configuration error with defaults', () => {
      const error = new ConfigurationError('Missing config key');

      expect(error.message).toBe('Missing config key');
      expect(error.code).toBe('CONFIG_ERROR');
      expect(error.statusCode).toBe(500);
      expect(error.retryable).toBe(false);
      expect(error.configKey).toBeUndefined();
    });

    it('should create configuration error with configKey', () => {
      const error = new ConfigurationError('Invalid config', 'API_KEY');

      expect(error.configKey).toBe('API_KEY');
      expect(error.details).toEqual({ configKey: 'API_KEY' });
    });
  });

  describe('ExternalServiceError', () => {
    it('should create external service error with all details', () => {
      const details = { endpoint: '/api/test' };
      const error = new ExternalServiceError('Stripe', 'Payment failed', 'PAYMENT_DECLINED', details);

      expect(error.message).toBe('External service \'Stripe\' error: Payment failed');
      expect(error.code).toBe('EXTERNAL_SERVICE_ERROR');
      expect(error.statusCode).toBe(502);
      expect(error.retryable).toBe(true);
      expect(error.serviceCode).toBe('PAYMENT_DECLINED');
      expect(error.details).toEqual({
        service: 'Stripe',
        serviceCode: 'PAYMENT_DECLINED',
        endpoint: '/api/test'
      });
    });
  });

  describe('categorizeError', () => {
    it('should categorize NetworkError as TRANSIENT', () => {
      const error = new NetworkError();
      expect(categorizeError(error)).toBe(ErrorCategory.TRANSIENT);
    });

    it('should categorize TimeoutError as TRANSIENT', () => {
      const error = new TimeoutError('Timeout', 5000);
      expect(categorizeError(error)).toBe(ErrorCategory.TRANSIENT);
    });

    it('should categorize RateLimitError as THROTTLED', () => {
      const error = new RateLimitError();
      expect(categorizeError(error)).toBe(ErrorCategory.THROTTLED);
    });

    it('should categorize non-retryable BubbleError as PERMANENT', () => {
      const error = new ValidationError('Invalid input');
      expect(categorizeError(error)).toBe(ErrorCategory.PERMANENT);
    });

    it('should categorize retryable BubbleError as TRANSIENT', () => {
      const error = new BubbleError('Temporary failure', 'TEMP_ERROR', 500, true);
      expect(categorizeError(error)).toBe(ErrorCategory.TRANSIENT);
    });

    it('should categorize generic Error with network message as TRANSIENT', () => {
      const error = new Error('ECONNREFUSED connection refused');
      expect(categorizeError(error)).toBe(ErrorCategory.TRANSIENT);
    });

    it('should categorize generic Error with timeout message as TRANSIENT', () => {
      // The error message needs to contain 'timeout' to be categorized as TRANSIENT
      const error = new Error('Request timeout');
      expect(categorizeError(error)).toBe(ErrorCategory.TRANSIENT);
    });

    it('should categorize generic Error with rate limit message as THROTTLED', () => {
      const error = new Error('Rate limit exceeded');
      expect(categorizeError(error)).toBe(ErrorCategory.THROTTLED);
    });

    it('should categorize unknown errors as UNKNOWN', () => {
      const error = new Error('Unknown error');
      expect(categorizeError(error)).toBe(ErrorCategory.UNKNOWN);
    });

    it('should categorize non-Error objects as UNKNOWN', () => {
      expect(categorizeError('string')).toBe(ErrorCategory.UNKNOWN);
      expect(categorizeError(null)).toBe(ErrorCategory.UNKNOWN);
      expect(categorizeError(undefined)).toBe(ErrorCategory.UNKNOWN);
    });
  });

  describe('isRetryable', () => {
    it('should return true for TRANSIENT errors', () => {
      const error = new NetworkError();
      expect(isRetryable(error)).toBe(true);
    });

    it('should return true for THROTTLED errors', () => {
      const error = new RateLimitError();
      expect(isRetryable(error)).toBe(true);
    });

    it('should return false for PERMANENT errors', () => {
      const error = new ValidationError('Invalid input');
      expect(isRetryable(error)).toBe(false);
    });

    it('should return false for UNKNOWN errors', () => {
      const error = new Error('Unknown error');
      expect(isRetryable(error)).toBe(false);
    });
  });

  describe('createErrorResponse', () => {
    it('should create response from BubbleError', () => {
      const error = new ValidationError('Invalid email', 'email');
      const correlationId = 'test-123';
      const response = createErrorResponse(error, correlationId);

      expect(response.error).toBe(error);
      expect(response.message).toBe('Invalid email');
      expect(response.code).toBe('VALIDATION_ERROR');
      expect(response.statusCode).toBe(400);
      expect(response.correlationId).toBe(correlationId);
      expect(response.retryable).toBe(false);
      expect(response.details).toEqual({ field: 'email' });
    });

    it('should create response from generic Error', () => {
      const error = new Error('Network error');
      const correlationId = 'test-123';
      const response = createErrorResponse(error, correlationId);

      expect(response.error).toBe(error);
      expect(response.message).toBe('Network error');
      expect(response.code).toBe('TRANSIENT'); // Categorized as transient
      expect(response.statusCode).toBe(500);
      expect(response.correlationId).toBe(correlationId);
      expect(response.retryable).toBe(true); // Network errors are retryable
      expect(response.details?.originalMessage).toBe('Network error');
    });

    it('should create response from non-Error object', () => {
      const error = 'string error';
      const correlationId = 'test-123';
      const response = createErrorResponse(error, correlationId);

      expect(response.message).toBe('string error');
      expect(response.code).toBe('UNKNOWN_ERROR');
      expect(response.statusCode).toBe(500);
      expect(response.correlationId).toBe(correlationId);
      expect(response.retryable).toBe(false);
      expect(response.details?.originalValue).toBe('string error');
    });

    it('should include stack trace in development mode', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      const error = new Error('Test error');
      const response = createErrorResponse(error, 'test-123');

      expect(response.details?.stack).toBeDefined();

      process.env.NODE_ENV = originalEnv;
    });
  });

  describe('wrapError', () => {
    it('should wrap BubbleError with additional context', () => {
      const originalError = new ValidationError('Invalid email', 'email');
      const wrapped = wrapError(originalError, {
        message: 'Failed to validate user input',
        code: 'USER_VALIDATION_FAILED',
        details: { context: 'user_registration' }
      });

      expect(wrapped.message).toBe('Failed to validate user input');
      expect(wrapped.code).toBe('USER_VALIDATION_FAILED');
      expect(wrapped.statusCode).toBe(400);
      expect(wrapped.retryable).toBe(false);
      expect(wrapped.details).toEqual({
        field: 'email',
        context: 'user_registration'
      });
    });

    it('should wrap generic Error', () => {
      const originalError = new Error('Database connection failed');
      const wrapped = wrapError(originalError, {
        message: 'Failed to connect to database',
        code: 'DB_CONNECTION_ERROR'
      });

      expect(wrapped.message).toBe('Failed to connect to database');
      expect(wrapped.code).toBe('DB_CONNECTION_ERROR');
      expect(wrapped.details?.originalMessage).toBe('Database connection failed');
    });

    it('should wrap non-Error objects', () => {
      const wrapped = wrapError('string error', {
        message: 'Operation failed'
      });

      expect(wrapped.message).toBe('Operation failed');
      expect(wrapped.code).toBe('WRAPPED_ERROR');
      expect(wrapped.details?.originalMessage).toBe('string error');
    });
  });

  describe('createSuccessResponse', () => {
    it('should create success response with data', () => {
      const data = { id: '123', name: 'Test' };
      const correlationId = 'test-123';
      const response = createSuccessResponse(data, correlationId);

      expect(response.success).toBe(true);
      expect(response.data).toEqual(data);
      expect(response.correlationId).toBe(correlationId);
    });

    it('should work with various data types', () => {
      expect(createSuccessResponse('string', 'cid').data).toBe('string');
      expect(createSuccessResponse(123, 'cid').data).toBe(123);
      expect(createSuccessResponse(null, 'cid').data).toBe(null);
      expect(createSuccessResponse(undefined, 'cid').data).toBe(undefined);
    });
  });

  describe('safeParseError', () => {
    it('should parse BubbleError', () => {
      const error = new ValidationError('Invalid input', 'field');
      const parsed = safeParseError(error);

      expect(parsed.message).toBe('Invalid input');
      expect(parsed.code).toBe('VALIDATION_ERROR');
      expect(parsed.statusCode).toBe(400);
      expect(parsed.stack).toBeDefined();
    });

    it('should parse generic Error', () => {
      const error = new Error('Test error');
      const parsed = safeParseError(error);

      expect(parsed.message).toBe('Test error');
      expect(parsed.stack).toBeDefined();
      expect(parsed.code).toBeUndefined();
      expect(parsed.statusCode).toBeUndefined();
    });

    it('should parse non-Error objects', () => {
      expect(safeParseError('string error').message).toBe('string error');
      expect(safeParseError(null).message).toBe('null');
      expect(safeParseError(123).message).toBe('123');
    });
  });

  describe('logError', () => {
    it('should log error with context', () => {
      const error = new ValidationError('Invalid input', 'field');
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      logError(error, { userId: '123', action: 'test' });

      expect(consoleSpy).toHaveBeenCalled();
      const loggedData = JSON.parse(consoleSpy.mock.calls[0][0] as string);
      expect(loggedData.message).toBe('Invalid input');
      expect(loggedData.userId).toBe('123');
      expect(loggedData.action).toBe('test');
      expect(loggedData.timestamp).toBeDefined();

      consoleSpy.mockRestore();
    });

    it('should log non-Error objects', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      logError('string error');

      expect(consoleSpy).toHaveBeenCalled();
      const loggedData = JSON.parse(consoleSpy.mock.calls[0][0] as string);
      expect(loggedData.message).toBe('string error');

      consoleSpy.mockRestore();
    });
  });

  describe('assert', () => {
    it('should not throw when condition is true', () => {
      expect(() => assert(true, 'Should not throw')).not.toThrow();
      expect(() => assert(1 === 1, 'Should not throw')).not.toThrow();
    });

    it('should throw ValidationError when condition is false', () => {
      expect(() => assert(false, 'Condition failed')).toThrow(ValidationError);
      expect(() => assert(false, 'Condition failed', 'testField')).toThrow(ValidationError);
    });

    it('should use field name in error', () => {
      try {
        assert(false, 'Validation failed', 'email');
      } catch (error) {
        expect((error as ValidationError).field).toBe('email');
      }
    });
  });

  describe('assertNonNull', () => {
    it('should not throw when value is not null/undefined', () => {
      expect(() => assertNonNull('value')).not.toThrow();
      expect(() => assertNonNull(0)).not.toThrow();
      expect(() => assertNonNull(false)).not.toThrow();
      expect(() => assertNonNull('')).not.toThrow();
    });

    it('should throw ValidationError when value is null', () => {
      expect(() => assertNonNull(null)).toThrow(ValidationError);
    });

    it('should throw ValidationError when value is undefined', () => {
      expect(() => assertNonNull(undefined)).toThrow(ValidationError);
    });

    it('should use custom message and field', () => {
      try {
        assertNonNull(null, 'Email is required', 'email');
      } catch (error) {
        expect((error as ValidationError).message).toBe('Email is required');
        expect((error as ValidationError).field).toBe('email');
      }
    });

    it('should narrow type correctly', () => {
      const value: string | null = 'test';
      assertNonNull(value);
      // After assertion, TypeScript should know value is string
      expect(value.toUpperCase()).toBe('TEST');
    });
  });
});
