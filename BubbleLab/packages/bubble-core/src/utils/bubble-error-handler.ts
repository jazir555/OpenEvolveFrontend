/**
 * Bubble Error Handler Mixin
 *
 * Provides reusable error handling methods for all bubble types
 * Integrates with the error-handler utilities for consistent error responses
 */

import {
  createErrorResponse,
  createSuccessResponse,
  generateCorrelationId,
  withTimeout,
  retryWithBackoff,
  CircuitBreaker,
  defaultCircuitBreakerConfig,
  type ErrorResponse,
  type SuccessResponse,
} from './error-handler.js';

/**
 * Configuration for bubble error handling
 */
export interface BubbleErrorHandlerConfig {
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  enableCircuitBreaker?: boolean;
  circuitBreakerThreshold?: number;
}

/**
 * Default configuration
 */
export const defaultBubbleErrorHandlerConfig: BubbleErrorHandlerConfig = {
  timeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000,
  enableCircuitBreaker: true,
  circuitBreakerThreshold: 5,
};

/**
 * Mixin class for adding error handling to bubbles
 */
export class BubbleErrorHandler {
  protected circuitBreaker?: CircuitBreaker;
  protected errorHandlerConfig: BubbleErrorHandlerConfig;

  constructor(config?: BubbleErrorHandlerConfig) {
    this.errorHandlerConfig = {
      ...defaultBubbleErrorHandlerConfig,
      ...config,
    };

    if (this.errorHandlerConfig.enableCircuitBreaker) {
      this.circuitBreaker = new CircuitBreaker({
        ...defaultCircuitBreakerConfig,
        failureThreshold: this.errorHandlerConfig.circuitBreakerThreshold || 5,
      });
    }
  }

  /**
   * Wrap an operation with comprehensive error handling
   */
  protected async executeWithErrorHandling<T>(
    operation: () => Promise<T>,
    operationName: string,
    context?: {
      timeout?: number;
      retry?: boolean;
      correlationId?: string;
    }
  ): Promise<SuccessResponse<T> | ErrorResponse> {
    const correlationId = context?.correlationId || generateCorrelationId();
    const startTime = Date.now();

    try {
      // Step 1: Input validation (if provided in context)
      // This should be done before executing the operation

      // Step 2: Circuit breaker check (if enabled)
      if (this.circuitBreaker) {
        const result = await this.circuitBreaker.execute(
          async () => {
            // Step 3: Timeout wrapper
            const timeout = context?.timeout || this.errorHandlerConfig.timeout || 30000;

            // Step 4: Retry logic (if enabled)
            if (context?.retry !== false && this.errorHandlerConfig.retryAttempts) {
              return await retryWithBackoff(
                async () => {
                  return await withTimeout(
                    operation(),
                    timeout,
                    operationName
                  );
                },
                {
                  maxAttempts: this.errorHandlerConfig.retryAttempts,
                  baseDelayMs: this.errorHandlerConfig.retryDelay,
                  correlationId,
                  operation: operationName,
                }
              );
            } else {
              return await withTimeout(operation(), timeout, operationName);
            }
          },
          operationName
        );

        return createSuccessResponse(result, correlationId);
      } else {
        // No circuit breaker, just timeout and retry
        const timeout = context?.timeout || this.errorHandlerConfig.timeout || 30000;

        if (context?.retry !== false && this.errorHandlerConfig.retryAttempts) {
          const result = await retryWithBackoff(
            async () => {
              return await withTimeout(
                operation(),
                timeout,
                operationName
              );
            },
            {
              maxAttempts: this.errorHandlerConfig.retryAttempts,
              baseDelayMs: this.errorHandlerConfig.retryDelay,
              correlationId,
              operation: operationName,
            }
          );

          return createSuccessResponse(result, correlationId);
        } else {
          const result = await withTimeout(operation(), timeout, operationName);
          return createSuccessResponse(result, correlationId);
        }
      }
    } catch (error) {
      const duration = Date.now() - startTime;
      console.error(
        `[${correlationId}] ${operationName} failed after ${duration}ms:`,
        error
      );

      return createErrorResponse(error, correlationId);
    }
  }

  /**
   * Reset the circuit breaker (useful for testing or manual recovery)
   */
  protected resetCircuitBreaker(): void {
    if (this.circuitBreaker) {
      this.circuitBreaker.reset();
    }
  }

  /**
   * Get circuit breaker metrics (useful for monitoring)
   */
  protected getCircuitBreakerMetrics() {
    if (!this.circuitBreaker) {
      return { enabled: false };
    }

    return {
      enabled: true,
      ...this.circuitBreaker.getMetrics(),
    };
  }
}

/**
 * Type guard to check if a response is an error response
 */
export function isErrorResponse<T>(
  response: SuccessResponse<T> | ErrorResponse
): response is ErrorResponse {
  return !response.success;
}

/**
 * Type guard to check if a response is a success response
 */
export function isSuccessResponse<T>(
  response: SuccessResponse<T> | ErrorResponse
): response is SuccessResponse<T> {
  return response.success;
}

/**
 * Extract data from a success response, throwing if it's an error
 */
export function getDataOrThrow<T>(
  response: SuccessResponse<T> | ErrorResponse
): T {
  if (isSuccessResponse(response)) {
    return response.data;
  }

  throw new Error(
    `Operation failed: ${response.error.message} (${response.error.code})`
  );
}

/**
 * Extract error from an error response, throwing if it's a success
 */
export function getErrorOrThrow<T>(
  response: SuccessResponse<T> | ErrorResponse
): ErrorResponse {
  if (isErrorResponse(response)) {
    return response;
  }

  throw new Error('Operation succeeded, cannot extract error');
}
