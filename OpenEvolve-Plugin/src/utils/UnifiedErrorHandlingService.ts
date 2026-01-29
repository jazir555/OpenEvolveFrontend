/**
 * Unified Error Handling Service
 * Provides a comprehensive, centralized error handling solution that integrates
 * all error handling components into a cohesive system
 */

import { errorLogger, ErrorLogger, ErrorSeverity, ErrorContext } from './errorLogging';
import { 
  gracefulErrorHandler, 
  ErrorHandlingOptions, 
  ErrorHandlingResult, 
  ErrorHandlingStrategy,
  ErrorContext as GrErrorContext
} from './gracefulErrorHandler';
import { ComprehensiveErrorHandler } from './enhancedErrorHandling';
import { toast } from 'react-toastify';

// Define unified error handling options
export interface UnifiedErrorHandlingOptions {
  strategy?: ErrorHandlingStrategy;
  maxRetries?: number;
  retryDelay?: number;
  fallbackValue?: any;
  cacheKey?: string;
  timeout?: number;
  showUserNotification?: boolean;
  logError?: boolean;
  logLevel?: ErrorSeverity;
  context?: ErrorContext & GrErrorContext;
  enableRecovery?: boolean;
  enableReporting?: boolean;
  enableCaching?: boolean;
  enableCircuitBreaker?: boolean;
  recoveryStrategies?: string[];
  reportingDestinations?: string[];
  metadata?: Record<string, any>;
}

// Define unified error result
export interface UnifiedErrorResult<T = any> {
  success: boolean;
  data?: T;
  error?: Error;
  retries?: number;
  strategyUsed?: ErrorHandlingStrategy;
  recoveryAttempts?: number;
  reportingStatus?: any;
  errorId?: string;
  classification?: any;
}

/**
 * Unified Error Handling Service
 * Integrates all error handling components into a single cohesive system
 */
export class UnifiedErrorHandlingService {
  private errorLogger: ErrorLogger;
  private comprehensiveErrorHandler: ComprehensiveErrorHandler;
  private fallbackStrategies: Map<string, (error: any, context: any) => any> = new Map();
  
  constructor() {
    this.errorLogger = errorLogger;
    this.comprehensiveErrorHandler = new ComprehensiveErrorHandler();
    
    // Register default fallback strategies
    this.registerFallbackStrategy('default', (error: any, context: any) => {
      console.warn(`Using default fallback for error: ${error.message}`);
      return null;
    });
    
    this.registerFallbackStrategy('empty_array', (error: any, context: any) => {
      return [];
    });
    
    this.registerFallbackStrategy('empty_object', (error: any, context: any) => {
      return {};
    });
    
    this.registerFallbackStrategy('zero_value', (error: any, context: any) => {
      return 0;
    });
    
    this.registerFallbackStrategy('empty_string', (error: any, context: any) => {
      return '';
    });
  }

  /**
   * Execute an operation with comprehensive error handling
   */
  async executeWithErrorHandling<T>(
    operation: () => Promise<T>,
    options: UnifiedErrorHandlingOptions = {}
  ): Promise<UnifiedErrorResult<T>> {
    const {
      strategy = 'retry',
      maxRetries = 3,
      retryDelay = 1000,
      fallbackValue = null,
      cacheKey,
      timeout = 30000,
      showUserNotification = true,
      logError = true,
      logLevel = 'error',
      context = {},
      enableRecovery = true,
      enableReporting = true,
      enableCaching = true,
      enableCircuitBreaker = true,
      recoveryStrategies = [],
      reportingDestinations = [],
      metadata = {}
    } = options;

    // Prepare context with additional metadata
    const enrichedContext: ErrorContext & GrErrorContext = {
      ...context,
      additionalData: {
        ...context.additionalData,
        ...metadata,
        timestamp: new Date(),
        userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'unknown',
        url: typeof window !== 'undefined' ? window.location.href : 'unknown',
      }
    };

    let retries = 0;
    let recoveryAttempts = 0;
    let lastError: Error | undefined;

    // Check cache first if caching is enabled and cacheKey is provided
    if (enableCaching && cacheKey) {
      try {
        const cachedResult = await this.checkCache(cacheKey);
        if (cachedResult !== undefined) {
          return { success: true, data: cachedResult as T };
        }
      } catch (cacheError) {
        console.warn('Cache check failed:', cacheError);
      }
    }

    // Circuit breaker check if enabled
    if (enableCircuitBreaker && context.function) {
      const circuitOpen = await this.isCircuitOpen(context.function);
      if (circuitOpen) {
        const circuitError = new Error(`Circuit breaker is open for ${context.function}`);
        if (logError) {
          this.errorLogger.logError(circuitError, logLevel, enrichedContext);
        }
        
        // Try fallback if available
        const fallbackResult = await this.executeFallback(fallbackValue, context.function || 'unknown');
        if (fallbackResult !== undefined) {
          return { success: true, data: fallbackResult as T, error: circuitError };
        }
        
        return { success: false, error: circuitError };
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

        // Store in cache if caching is enabled and cacheKey is provided
        if (enableCaching && cacheKey) {
          try {
            await this.storeInCache(cacheKey, result);
          } catch (cacheError) {
            console.warn('Cache store failed:', cacheError);
          }
        }

        // Close circuit if it was open
        if (enableCircuitBreaker && context.function) {
          await this.closeCircuit(context.function);
        }

        return { 
          success: true, 
          data: result, 
          retries,
          recoveryAttempts
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        retries++;

        // Log error if enabled
        if (logError) {
          this.errorLogger.logError(lastError, logLevel, enrichedContext);
        }

        // Show user notification if enabled
        if (showUserNotification) {
          this.showUserNotification(lastError, strategy, retries, maxRetries);
        }

        // Attempt recovery if enabled
        if (enableRecovery) {
          const recoveryResult = await this.attemptRecovery(
            lastError,
            enrichedContext,
            recoveryStrategies
          );
          
          if (recoveryResult.success) {
            recoveryAttempts++;
            
            // If recovery was successful, try the operation again
            if (recoveryResult.shouldRetry) {
              continue;
            }
            
            // If recovery provided a fallback value, return it
            if (recoveryResult.fallbackValue !== undefined) {
              return { 
                success: true, 
                data: recoveryResult.fallbackValue as T, 
                error: lastError,
                retries,
                recoveryAttempts
              };
            }
          }
        }

        // Apply error handling strategy
        const shouldContinue = await this.applyStrategy(
          strategy,
          lastError,
          enrichedContext,
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
    if (fallbackValue !== undefined) {
      try {
        return { 
          success: true, 
          data: fallbackValue as T, 
          error: lastError, 
          retries,
          recoveryAttempts,
          strategyUsed: 'fallback' 
        };
      } catch (fallbackError) {
        console.error('Fallback strategy failed:', fallbackError);
      }
    }

    // Try registered fallback strategy
    if (context.function) {
      const fallbackResult = await this.executeFallback(fallbackValue, context.function);
      if (fallbackResult !== undefined) {
        return { 
          success: true, 
          data: fallbackResult as T, 
          error: lastError, 
          retries,
          recoveryAttempts,
          strategyUsed: 'fallback' 
        };
      }
    }

    // Final error reporting if enabled
    let reportingStatus = null;
    let errorId = undefined;
    let classification = undefined;

    if (enableReporting && lastError) {
      try {
        const comprehensiveResult = await this.comprehensiveErrorHandler.handleError(
          lastError,
          enrichedContext,
          {
            autoRecover: false, // We already tried recovery above
            autoReport: true,
            metadata: { ...metadata, retries, recoveryAttempts }
          }
        );
        
        errorId = comprehensiveResult.errorId;
        classification = comprehensiveResult.classification;
        reportingStatus = comprehensiveResult.reportingResult;
      } catch (reportingError) {
        console.error('Comprehensive error handling failed:', reportingError);
      }
    }

    return { 
      success: false, 
      error: lastError, 
      retries,
      recoveryAttempts,
      reportingStatus,
      errorId,
      classification
    };
  }

  /**
   * Apply error handling strategy
   */
  private async applyStrategy(
    strategy: ErrorHandlingStrategy,
    error: Error,
    context: ErrorContext & GrErrorContext,
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
        return this.handleCircuitBreaker(context.function || 'unknown', error, maxRetries);

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
    // This is a simplified circuit breaker implementation
    // In a real implementation, you'd want to use a more robust solution
    const circuitStateKey = `circuit_breaker_${operation}`;
    
    if (typeof window !== 'undefined' && window.sessionStorage) {
      try {
        let circuitState = JSON.parse(sessionStorage.getItem(circuitStateKey) || '{"failures": 0, "isOpen": false, "lastFailure": 0}');
        
        const now = Date.now();
        const resetTimeout = 30000; // 30 seconds
        
        // Check if circuit is open and should remain open
        if (circuitState.isOpen) {
          if (now - circuitState.lastFailure < resetTimeout) {
            // Circuit is still open, don't retry
            return false;
          } else {
            // Reset circuit after timeout
            circuitState = { failures: 0, isOpen: false, lastFailure: 0 };
            sessionStorage.setItem(circuitStateKey, JSON.stringify(circuitState));
            return true; // Allow retry after reset
          }
        }

        // Increment failure count
        circuitState.failures++;
        circuitState.lastFailure = now;

        if (circuitState.failures >= failureThreshold) {
          circuitState.isOpen = true;
          sessionStorage.setItem(circuitStateKey, JSON.stringify(circuitState));
          
          if (typeof toast !== 'undefined') {
            toast.warn(`Circuit breaker opened for ${operation} due to repeated failures. Will reset in 30 seconds.`);
          }
          return false;
        }

        sessionStorage.setItem(circuitStateKey, JSON.stringify(circuitState));
        return true;
      } catch (storageError) {
        console.error('Circuit breaker storage error:', storageError);
        return true; // Allow operation to continue if storage fails
      }
    }
    
    return true; // Allow operation if not in browser environment
  }

  /**
   * Check if circuit is open
   */
  private async isCircuitOpen(operation: string): Promise<boolean> {
    const circuitStateKey = `circuit_breaker_${operation}`;
    
    if (typeof window !== 'undefined' && window.sessionStorage) {
      try {
        const circuitStateStr = sessionStorage.getItem(circuitStateKey);
        if (!circuitStateStr) return false;
        
        const circuitState = JSON.parse(circuitStateStr);
        const now = Date.now();
        const resetTimeout = 30000; // 30 seconds
        
        if (circuitState.isOpen && now - circuitState.lastFailure < resetTimeout) {
          return true;
        }
        
        // If timeout has passed, reset the circuit
        if (circuitState.isOpen) {
          sessionStorage.removeItem(circuitStateKey);
        }
        
        return false;
      } catch (storageError) {
        console.error('Circuit breaker check error:', storageError);
        return false; // Assume circuit is closed if check fails
      }
    }
    
    return false; // Circuit is closed if not in browser environment
  }

  /**
   * Close circuit
   */
  private async closeCircuit(operation: string): Promise<void> {
    const circuitStateKey = `circuit_breaker_${operation}`;
    
    if (typeof window !== 'undefined' && window.sessionStorage) {
      try {
        sessionStorage.removeItem(circuitStateKey);
      } catch (storageError) {
        console.error('Circuit breaker close error:', storageError);
      }
    }
  }

  /**
   * Check cache for a value
   */
  private async checkCache(cacheKey: string): Promise<any> {
    if (typeof window !== 'undefined' && window.sessionStorage) {
      try {
        const cachedItemStr = sessionStorage.getItem(`cache_${cacheKey}`);
        if (!cachedItemStr) return undefined;
        
        const cachedItem = JSON.parse(cachedItemStr);
        const now = Date.now();
        
        // Check if cache is expired (default TTL is 5 minutes)
        if (now - cachedItem.timestamp > (cachedItem.ttl || 300000)) {
          sessionStorage.removeItem(`cache_${cacheKey}`);
          return undefined;
        }
        
        return cachedItem.data;
      } catch (cacheError) {
        console.error('Cache check error:', cacheError);
        return undefined;
      }
    }
    
    return undefined;
  }

  /**
   * Store value in cache
   */
  private async storeInCache(cacheKey: string, data: any): Promise<void> {
    if (typeof window !== 'undefined' && window.sessionStorage) {
      try {
        const cacheItem = {
          data,
          timestamp: Date.now(),
          ttl: 300000 // 5 minutes default TTL
        };
        
        sessionStorage.setItem(`cache_${cacheKey}`, JSON.stringify(cacheItem));
      } catch (cacheError) {
        console.error('Cache store error:', cacheError);
      }
    }
  }

  /**
   * Attempt error recovery using registered strategies
   */
  private async attemptRecovery(
    error: Error,
    context: ErrorContext & GrErrorContext,
    strategies: string[]
  ): Promise<{
    success: boolean;
    shouldRetry?: boolean;
    fallbackValue?: any;
  }> {
    // For now, we'll use the comprehensive error handler's recovery
    // In a more advanced implementation, we could use specific recovery strategies
    
    try {
      // Classify the error first
      const classification = this.comprehensiveErrorHandler.getComponents().classifier.classifyError(error);
      
      // Attempt recovery using the comprehensive handler
      const recoveryResult = await this.comprehensiveErrorHandler.getComponents().recovery.attemptRecovery(
        error,
        classification,
        { ...context, error }
      );
      
      if (recoveryResult.success) {
        return { success: true, shouldRetry: true };
      }
      
      return { success: false };
    } catch (recoveryError) {
      console.error('Recovery attempt failed:', recoveryError);
      return { success: false };
    }
  }

  /**
   * Execute fallback strategy
   */
  private async executeFallback(fallbackValue: any, operationName: string): Promise<any> {
    if (fallbackValue !== undefined) {
      return fallbackValue;
    }
    
    // Try to find a registered fallback strategy based on operation name
    const strategyKey = operationName.toLowerCase().includes('list') || operationName.toLowerCase().includes('array') 
      ? 'empty_array' 
      : operationName.toLowerCase().includes('object') || operationName.toLowerCase().includes('data')
        ? 'empty_object'
        : operationName.toLowerCase().includes('count') || operationName.toLowerCase().includes('number')
          ? 'zero_value'
          : 'default';
    
    const fallbackFn = this.fallbackStrategies.get(strategyKey);
    if (fallbackFn) {
      try {
        return fallbackFn(null, { operation: operationName });
      } catch (fallbackError) {
        console.error('Fallback execution failed:', fallbackError);
        return this.fallbackStrategies.get('default')!(null, { operation: operationName });
      }
    }
    
    return undefined;
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
   * Register a fallback strategy
   */
  registerFallbackStrategy(name: string, strategy: (error: any, context: any) => any): void {
    this.fallbackStrategies.set(name, strategy);
  }

  /**
   * Unregister a fallback strategy
   */
  unregisterFallbackStrategy(name: string): boolean {
    return this.fallbackStrategies.delete(name);
  }

  /**
   * Get all registered fallback strategies
   */
  getFallbackStrategies(): string[] {
    return Array.from(this.fallbackStrategies.keys());
  }

  /**
   * Handle a caught error with comprehensive processing
   */
  async handleError(
    error: any,
    context: ErrorContext = {},
    options: {
      logError?: boolean;
      showNotification?: boolean;
      attemptRecovery?: boolean;
      reportError?: boolean;
    } = {}
  ): Promise<UnifiedErrorResult> {
    const { 
      logError = true, 
      showNotification = true, 
      attemptRecovery = true, 
      reportError = true 
    } = options;

    // Log the error
    if (logError) {
      this.errorLogger.logError(error, 'error', context);
    }

    // Show notification
    if (showNotification) {
      toast.error(`Error: ${error.message || 'An error occurred'}`);
    }

    // Attempt recovery
    let recoveryResult = null;
    if (attemptRecovery) {
      try {
        const classification = this.comprehensiveErrorHandler.getComponents().classifier.classifyError(error);
        recoveryResult = await this.comprehensiveErrorHandler.getComponents().recovery.attemptRecovery(
          error,
          classification,
          context
        );
      } catch (recoveryError) {
        console.error('Recovery attempt failed:', recoveryError);
      }
    }

    // Report error
    let reportingResult = null;
    if (reportError) {
      try {
        reportingResult = await this.comprehensiveErrorHandler.handleError(
          error,
          context,
          { autoRecover: false, autoReport: true }
        );
      } catch (reportingError) {
        console.error('Error reporting failed:', reportingError);
      }
    }

    return {
      success: false,
      error: error instanceof Error ? error : new Error(String(error)),
      recoveryAttempts: recoveryResult?.success ? 1 : 0,
      reportingStatus: reportingResult
    };
  }

  /**
   * Wrap an async function with error handling
   */
  wrapAsyncFunction<T>(
    fn: (...args: any[]) => Promise<T>,
    options: UnifiedErrorHandlingOptions = {}
  ): (...args: any[]) => Promise<UnifiedErrorResult<T>> {
    return async (...args: any[]): Promise<UnifiedErrorResult<T>> => {
      return this.executeWithErrorHandling(
        () => fn(...args),
        options
      );
    };
  }

  /**
   * Wrap a sync function with error handling (converts to async)
   */
  wrapSyncFunction<T>(
    fn: (...args: any[]) => T,
    options: UnifiedErrorHandlingOptions = {}
  ): (...args: any[]) => Promise<UnifiedErrorResult<T>> {
    return async (...args: any[]): Promise<UnifiedErrorResult<T>> => {
      return this.executeWithErrorHandling(
        () => Promise.resolve(fn(...args)),
        options
      );
    };
  }

  /**
   * Get error statistics
   */
  getErrorStatistics() {
    return this.comprehensiveErrorHandler.getErrorStatistics();
  }

  /**
   * Get error history
   */
  getErrorHistory(limit: number = 50) {
    return this.comprehensiveErrorHandler.getErrorHistory(limit);
  }

  /**
   * Get queue status
   */
  getQueueStatus() {
    return this.comprehensiveErrorHandler.getQueueStatus();
  }
}

// Create a singleton instance
export const unifiedErrorHandlingService = new UnifiedErrorHandlingService();

/**
 * Convenience function to execute operations with unified error handling
 */
export async function withUnifiedErrorHandling<T>(
  operation: () => Promise<T>,
  options: UnifiedErrorHandlingOptions = {}
): Promise<UnifiedErrorResult<T>> {
  return unifiedErrorHandlingService.executeWithErrorHandling(operation, options);
}

/**
 * Hook-like function for React components to handle errors with the unified service
 */
export function useUnifiedErrorHandler() {
  return {
    execute: <T>(operation: () => Promise<T>, options: UnifiedErrorHandlingOptions = {}) =>
      unifiedErrorHandlingService.executeWithErrorHandling<T>(operation, options),
    handleError: (error: any, context: ErrorContext = {}, options: any = {}) =>
      unifiedErrorHandlingService.handleError(error, context, options),
    registerFallbackStrategy: (name: string, strategy: (error: any, context: any) => any) =>
      unifiedErrorHandlingService.registerFallbackStrategy(name, strategy),
    getErrorStatistics: () => unifiedErrorHandlingService.getErrorStatistics(),
    getErrorHistory: (limit: number = 50) => unifiedErrorHandlingService.getErrorHistory(limit),
    getQueueStatus: () => unifiedErrorHandlingService.getQueueStatus(),
    wrapAsync: <T>(fn: (...args: any[]) => Promise<T>, options: UnifiedErrorHandlingOptions = {}) =>
      unifiedErrorHandlingService.wrapAsyncFunction(fn, options),
    wrapSync: <T>(fn: (...args: any[]) => T, options: UnifiedErrorHandlingOptions = {}) =>
      unifiedErrorHandlingService.wrapSyncFunction(fn, options)
  };
}