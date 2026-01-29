/**
 * Graceful Error Handler
 * Provides enhanced error handling with retry logic, fallback mechanisms, and user-friendly error messages
 */

import { errorLogger } from './errorLogging';
import { ComprehensiveErrorHandler } from './enhancedErrorHandling';
import { toast } from 'react-toastify';

// Define error handling strategies
export type ErrorHandlingStrategy = 'retry' | 'fallback' | 'cache' | 'graceful_degradation' | 'circuit_breaker';

// Define error context interface
export interface ErrorContext {
  component?: string;
  function?: string;
  operation?: string;
  userId?: string;
  sessionId?: string;
  url?: string;
  userAgent?: string;
  timestamp?: Date;
  additionalData?: Record<string, any>;
}

// Define error handling options
export interface ErrorHandlingOptions {
  strategy?: ErrorHandlingStrategy;
  maxRetries?: number;
  retryDelay?: number;
  fallbackValue?: any;
  cacheKey?: string;
  timeout?: number;
  showUserNotification?: boolean;
  logError?: boolean;
  context?: ErrorContext;
}

// Define error result
export interface ErrorHandlingResult<T = any> {
  success: boolean;
  data?: T;
  error?: Error;
  retries?: number;
  strategyUsed?: ErrorHandlingStrategy;
}

/**
 * Graceful Error Handler Class
 * Provides comprehensive error handling with multiple strategies
 */
export class GracefulErrorHandler {
  private comprehensiveErrorHandler: ComprehensiveErrorHandler;
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private circuitBreakers: Map<string, { failures: number; lastFailure: number; isOpen: boolean }> = new Map();

  constructor() {
    this.comprehensiveErrorHandler = new ComprehensiveErrorHandler();
  }

  /**
   * Execute an operation with graceful error handling
   */
  async executeWithErrorHandling<T>(
    operation: () => Promise<T>,
    options: ErrorHandlingOptions = {}
  ): Promise<ErrorHandlingResult<T>> {
    const {
      strategy = 'retry',
      maxRetries = 3,
      retryDelay = 1000,
      fallbackValue = null,
      cacheKey,
      timeout = 30000,
      showUserNotification = true,
      logError = true,
      context = {}
    } = options;

    let retries = 0;
    let lastError: Error | undefined;

    // Check cache first if cacheKey is provided
    if (cacheKey && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      if (Date.now() - cached.timestamp < cached.ttl) {
        return { success: true, data: cached.data };
      } else {
        this.cache.delete(cacheKey); // Remove expired cache
      }
    }

    while (retries <= maxRetries) {
      try {
        // Create a timeout promise
        const timeoutPromise = new Promise<never>((_, reject) => {
          setTimeout(() => reject(new Error('Operation timeout')), timeout);
        });

        // Execute the operation with timeout
        const result = await Promise.race([
          operation(),
          timeoutPromise
        ]);

        // Cache the result if cacheKey is provided
        if (cacheKey) {
          this.cache.set(cacheKey, {
            data: result,
            timestamp: Date.now(),
            ttl: 300000 // 5 minutes default TTL
          });
        }

        // Reset circuit breaker on success
        if (context.operation) {
          this.circuitBreakers.delete(context.operation);
        }

        return { success: true, data: result, retries };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        retries++;

        if (logError) {
          await this.comprehensiveErrorHandler.handleError(lastError, context, {
            autoRecover: true,
            autoReport: true,
            metadata: { retries, strategy }
          });
        }

        if (showUserNotification) {
          this.showUserNotification(lastError, strategy, retries, maxRetries);
        }

        // Apply error handling strategy
        const shouldContinue = await this.applyStrategy(
          strategy,
          lastError,
          context,
          retries,
          maxRetries,
          retryDelay
        );

        if (!shouldContinue) {
          break;
        }
      }
    }

    // If all retries failed, try fallback strategy
    if (strategy !== 'fallback' && fallbackValue !== undefined) {
      try {
        return { success: true, data: fallbackValue, error: lastError, strategyUsed: 'fallback' };
      } catch (fallbackError) {
        console.error('Fallback strategy failed:', fallbackError);
      }
    }

    return { success: false, error: lastError, retries };
  }

  /**
   * Apply error handling strategy
   */
  private async applyStrategy(
    strategy: ErrorHandlingStrategy,
    error: Error,
    context: ErrorContext,
    retries: number,
    maxRetries: number,
    retryDelay: number
  ): Promise<boolean> {
    switch (strategy) {
      case 'retry':
        if (retries <= maxRetries) {
          if (retryDelay > 0) {
            await this.delay(retryDelay * Math.pow(2, retries - 1)); // Exponential backoff
          }
          return true;
        }
        return false;

      case 'circuit_breaker':
        return this.handleCircuitBreaker(context.operation || 'unknown', error, maxRetries);

      case 'cache':
        // Already handled in the main function
        return false;

      case 'graceful_degradation':
        // Implement graceful degradation logic
        return false;

      case 'fallback':
        // Fallback is handled after all retries
        return false;

      default:
        return false;
    }
  }

  /**
   * Handle circuit breaker logic
   */
  private handleCircuitBreaker(operation: string, error: Error, failureThreshold: number = 5): boolean {
    const circuitState = this.circuitBreakers.get(operation) || { failures: 0, lastFailure: 0, isOpen: false };
    const now = Date.now();

    // Check if circuit is open and should remain open
    if (circuitState.isOpen) {
      const resetTimeout = 30000; // 30 seconds
      if (now - circuitState.lastFailure < resetTimeout) {
        // Circuit is still open, don't retry
        return false;
      } else {
        // Reset circuit after timeout
        circuitState.failures = 0;
        circuitState.isOpen = false;
        circuitState.lastFailure = 0;
        this.circuitBreakers.set(operation, circuitState);
        return true; // Allow retry after reset
      }
    }

    // Increment failure count
    circuitState.failures++;
    circuitState.lastFailure = now;

    if (circuitState.failures >= failureThreshold) {
      circuitState.isOpen = true;
      this.circuitBreakers.set(operation, circuitState);
      
      toast.warn(`Circuit breaker opened for ${operation} due to repeated failures. Will reset in 30 seconds.`);
      return false;
    }

    this.circuitBreakers.set(operation, circuitState);
    return true;
  }

  /**
   * Show user notification for error
   */
  private showUserNotification(error: Error, strategy: ErrorHandlingStrategy, retries: number, maxRetries: number): void {
    let message = `An error occurred: ${error.message}`;

    if (retries <= maxRetries) {
      message += ` (Attempt ${retries}/${maxRetries})`;
      toast.info(message);
    } else {
      message = `Operation failed after ${maxRetries} attempts: ${error.message}`;
      toast.error(message);
    }
  }

  /**
   * Delay helper function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Set cache TTL for a specific key
   */
  setCacheTTL(cacheKey: string, ttl: number): void {
    if (this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      cached.ttl = ttl;
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Clear circuit breakers
   */
  clearCircuitBreakers(): void {
    this.circuitBreakers.clear();
  }

  /**
   * Get cache status
   */
  getCacheStatus(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys())
    };
  }

  /**
   * Get circuit breaker status
   */
  getCircuitBreakerStatus(): { openCircuits: string[]; total: number } {
    const openCircuits = Array.from(this.circuitBreakers.entries())
      .filter(([_, state]) => state.isOpen)
      .map(([key, _]) => key);

    return {
      openCircuits,
      total: this.circuitBreakers.size
    };
  }
}

// Create a singleton instance
export const gracefulErrorHandler = new GracefulErrorHandler();

/**
 * Higher-order function to wrap async operations with graceful error handling
 */
export function withGracefulErrorHandling<T>(
  operation: () => Promise<T>,
  options: ErrorHandlingOptions = {}
): () => Promise<ErrorHandlingResult<T>> {
  return async (): Promise<ErrorHandlingResult<T>> => {
    return gracefulErrorHandler.executeWithErrorHandling(operation, options);
  }

}

/**
 * Hook-like function for React components to handle errors gracefully
 */
export function useGracefulErrorHandler() {
  return {
    execute: <T>(operation: () => Promise<T>, options: ErrorHandlingOptions = {}) => 
      gracefulErrorHandler.executeWithErrorHandling<T>(operation, options),
    cache: gracefulErrorHandler.cache,
    setCacheTTL: (cacheKey: string, ttl: number) => gracefulErrorHandler.setCacheTTL(cacheKey, ttl),
    clearCache: () => gracefulErrorHandler.clearCache(),
    circuitBreakers: gracefulErrorHandler.getCircuitBreakerStatus(),
    clearCircuitBreakers: () => gracefulErrorHandler.clearCircuitBreakers()
  };
}