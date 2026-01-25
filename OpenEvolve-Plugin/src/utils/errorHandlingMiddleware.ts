/**
 * Error Handling Middleware
 * Provides centralized error handling middleware for API requests and other operations
 */

import { gracefulErrorHandler, ErrorHandlingOptions } from '@/utils/gracefulErrorHandler';
import { errorLogger } from '@/utils/errorLogging';
import { toast } from 'react-toastify';

/**
 * Middleware function type
 */
export type MiddlewareFunction<T = any, R = any> = (input: T) => Promise<R>;

/**
 * Error handling middleware configuration
 */
export interface ErrorHandlingMiddlewareConfig {
  strategy?: 'retry' | 'fallback' | 'cache' | 'graceful_degradation' | 'circuit_breaker';
  maxRetries?: number;
  retryDelay?: number;
  fallbackValue?: any;
  cacheKey?: string;
  timeout?: number;
  showUserNotification?: boolean;
  logError?: boolean;
  context?: {
    component?: string;
    function?: string;
    operation?: string;
    additionalData?: Record<string, any>;
  };
}

/**
 * Creates an error handling middleware
 */
export function createErrorHandlingMiddleware<T = any, R = any>(
  config: ErrorHandlingMiddlewareConfig = {}
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return (next: MiddlewareFunction<T, R>): MiddlewareFunction<T, R> => {
    return async (input: T): Promise<R> => {
      const options: ErrorHandlingOptions = {
        strategy: config.strategy || 'retry',
        maxRetries: config.maxRetries || 3,
        retryDelay: config.retryDelay || 1000,
        fallbackValue: config.fallbackValue,
        cacheKey: config.cacheKey,
        timeout: config.timeout || 30000,
        showUserNotification: config.showUserNotification !== false,
        logError: config.logError !== false,
        context: {
          component: config.context?.component,
          function: config.context?.function,
          operation: config.context?.operation,
          additionalData: config.context?.additionalData,
        }
      };

      const result = await gracefulErrorHandler.executeWithErrorHandling<R>(
        async () => next(input),
        options
      );

      if (!result.success) {
        throw result.error || new Error('Operation failed');
      }

      return result.data!;
    };
  };
}

/**
 * API request error handling middleware
 */
export function createApiErrorHandlingMiddleware<T = any, R = any>(
  operationName: string,
  component: string = 'ApiMiddleware'
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return createErrorHandlingMiddleware<T, R>({
    strategy: 'retry',
    maxRetries: 3,
    retryDelay: 1000,
    showUserNotification: true,
    logError: true,
    context: {
      component,
      function: 'apiRequest',
      operation: operationName,
    }
  });
}

/**
 * Async operation error handling middleware
 */
export function createAsyncOperationMiddleware<T = any, R = any>(
  operationName: string,
  component: string = 'AsyncOperationMiddleware'
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return createErrorHandlingMiddleware<T, R>({
    strategy: 'retry',
    maxRetries: 2,
    retryDelay: 500,
    showUserNotification: false,
    logError: true,
    context: {
      component,
      function: 'asyncOperation',
      operation: operationName,
    }
  });
}

/**
 * Error handling middleware for React components
 */
export function createComponentErrorHandlingMiddleware<T = any, R = any>(
  componentName: string,
  operationName: string
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return createErrorHandlingMiddleware<T, R>({
    strategy: 'retry',
    maxRetries: 2,
    retryDelay: 500,
    showUserNotification: true,
    logError: true,
    context: {
      component: componentName,
      function: 'componentOperation',
      operation: operationName,
    }
  });
}

/**
 * Error handling middleware with circuit breaker
 */
export function createCircuitBreakerMiddleware<T = any, R = any>(
  operationName: string,
  component: string = 'CircuitBreakerMiddleware',
  failureThreshold: number = 5
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return (next: MiddlewareFunction<T, R>): MiddlewareFunction<T, R> => {
    return async (input: T): Promise<R> => {
      // Check if circuit is open for this operation
      const circuitKey = `${component}_${operationName}`;
      const circuitState = gracefulErrorHandler.getCircuitBreakerStatus();
      
      // If circuit is open, fail fast
      if (circuitState.openCircuits.includes(circuitKey)) {
        const error = new Error(`Circuit breaker open for ${operationName}`);
        errorLogger.logError(error, 'warn', {
          component,
          function: 'circuitBreaker',
          operation: operationName,
          additionalData: { circuitKey, reason: 'circuit_open' }
        });
        
        if (false) { // Don't show notification for circuit breaker
          toast.warn(`Service temporarily unavailable: ${operationName}`);
        }
        
        throw error;
      }

      // Execute with error handling
      const result = await gracefulErrorHandler.executeWithErrorHandling<R>(
        async () => next(input),
        {
          strategy: 'circuit_breaker',
          maxRetries: 1, // Circuit breaker handles retries internally
          retryDelay: 500,
          showUserNotification: true,
          logError: true,
          context: {
            component,
            function: 'circuitBreakerOperation',
            operation: operationName,
            additionalData: { circuitKey }
          }
        }
      );

      if (!result.success) {
        throw result.error || new Error('Operation failed');
      }

      return result.data!;
    };
  };
}

/**
 * Error handling middleware with caching
 */
export function createCachingMiddleware<T = any, R = any>(
  cacheKey: string,
  ttl: number = 300000, // 5 minutes default
  operationName: string = 'cachedOperation',
  component: string = 'CachingMiddleware'
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return (next: MiddlewareFunction<T, R>): MiddlewareFunction<T, R> => {
    return async (input: T): Promise<R> => {
      // Check cache first
      const cachedResult = gracefulErrorHandler.cache.get(cacheKey);
      if (cachedResult && Date.now() - cachedResult.timestamp < cachedResult.ttl) {
        return cachedResult.data;
      }

      // Execute with error handling
      const result = await gracefulErrorHandler.executeWithErrorHandling<R>(
        async () => {
          const data = await next(input);
          // Cache the result
          gracefulErrorHandler.cache.set(cacheKey, {
            data,
            timestamp: Date.now(),
            ttl
          });
          return data;
        },
        {
          strategy: 'cache',
          maxRetries: 2,
          retryDelay: 500,
          showUserNotification: false,
          logError: true,
          cacheKey,
          context: {
            component,
            function: 'cachedOperation',
            operation: operationName,
            additionalData: { cacheKey }
          }
        }
      );

      if (!result.success) {
        throw result.error || new Error('Operation failed');
      }

      return result.data!;
    };
  };
}

/**
 * Compose multiple middleware functions
 */
export function composeMiddleware<T = any, R = any>(
  ...middlewares: Array<(next: MiddlewareFunction<any, any>) => MiddlewareFunction<any, any>>
): (next: MiddlewareFunction<T, R>) => MiddlewareFunction<T, R> {
  return (next: MiddlewareFunction<T, R>): MiddlewareFunction<T, R> => {
    return middlewares.reverse().reduce((acc, curr) => curr(acc), next as any);
  };
}

/**
 * Default error handling middleware that can be applied globally
 */
export const defaultErrorHandlingMiddleware = createErrorHandlingMiddleware();

/**
 * Example usage:
 * 
 * const apiCallWithMiddleware = composeMiddleware(
 *   createApiErrorHandlingMiddleware('getUserData', 'UserService'),
 *   createCachingMiddleware('user_data_cache', 60000, 'getUserData', 'UserService')
 * )(async (userId) => {
 *   // API call implementation
 *   return await fetch(`/api/users/${userId}`).then(r => r.json());
 * });
 * 
 * const result = await apiCallWithMiddleware('123');
 */