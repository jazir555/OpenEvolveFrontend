/**
 * Error Handling Decorators
 * Provides decorators for consistent error handling across methods and classes
 */

import { errorLogger } from '@/utils/errorLogging';
import { gracefulErrorHandler } from '@/utils/gracefulErrorHandler';
import { toast } from 'react-toastify';

// Define decorator options
interface ErrorHandlingDecoratorOptions {
  retries?: number;
  retryDelay?: number;
  fallbackValue?: any;
  logError?: boolean;
  notifyUser?: boolean;
  errorContext?: string;
  suppressError?: boolean; // Whether to prevent error from propagating
}

/**
 * Method decorator for automatic error handling
 */
export function HandleError(options: ErrorHandlingDecoratorOptions = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const {
        retries = 0,
        retryDelay = 1000,
        fallbackValue = undefined,
        logError = true,
        notifyUser = true,
        errorContext = `${target.constructor.name}.${propertyKey}`,
        suppressError = false
      } = options;

      let attempt = 0;
      let lastError: any;

      while (attempt <= retries) {
        try {
          return await method.apply(this, args);
        } catch (error) {
          lastError = error;
          attempt++;

          if (logError) {
            errorLogger.logError(error, 'error', {
              component: errorContext,
              function: propertyKey,
              additionalData: {
                attempt,
                maxRetries: retries,
                arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
              }
            });
          }

          if (notifyUser) {
            toast.error(`Error in ${errorContext}: ${error.message}`);
          }

          if (attempt <= retries) {
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, retryDelay));
          }
        }
      }

      // If all retries failed
      if (suppressError) {
        return fallbackValue;
      } else {
        throw lastError;
      }
    };

    return descriptor;
  };
}

/**
 * Class decorator for automatic error handling on all methods
 */
export function HandleClassErrors(options: ErrorHandlingDecoratorOptions = {}) {
  return function (constructor: Function) {
    const proto = constructor.prototype;

    // Get all method names from the prototype
    const methodNames = Object.getOwnPropertyNames(proto).filter(name => {
      const descriptor = Object.getOwnPropertyDescriptor(proto, name);
      return descriptor && typeof descriptor.value === 'function' && name !== 'constructor';
    });

    // Apply the HandleError decorator to each method
    methodNames.forEach(methodName => {
      const descriptor = Object.getOwnPropertyDescriptor(proto, methodName);
      if (descriptor) {
        const method = descriptor.value;
        descriptor.value = async function (...args: any[]) {
          const {
            retries = 0,
            retryDelay = 1000,
            fallbackValue = undefined,
            logError = true,
            notifyUser = true,
            errorContext = `${constructor.name}.${methodName}`,
            suppressError = false
          } = options;

          let attempt = 0;
          let lastError: any;

          while (attempt <= retries) {
            try {
              return await method.apply(this, args);
            } catch (error) {
              lastError = error;
              attempt++;

              if (logError) {
                errorLogger.logError(error, 'error', {
                  component: errorContext,
                  function: methodName,
                  additionalData: {
                    attempt,
                    maxRetries: retries,
                    arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
                  }
                });
              }

              if (notifyUser) {
                toast.error(`Error in ${errorContext}: ${error.message}`);
              }

              if (attempt <= retries) {
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, retryDelay));
              }
            }
          }

          // If all retries failed
          if (suppressError) {
            return fallbackValue;
          } else {
            throw lastError;
          }
        };
        Object.defineProperty(proto, methodName, descriptor);
      }
    });
  };
}

/**
 * Async operation decorator with comprehensive error handling
 */
export function SafeAsyncOperation(options: ErrorHandlingDecoratorOptions = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const {
        retries = 0,
        retryDelay = 1000,
        fallbackValue = undefined,
        logError = true,
        notifyUser = true,
        errorContext = `${target.constructor.name}.${propertyKey}`,
        suppressError = false
      } = options;

      try {
        // Use the graceful error handler for more sophisticated error handling
        const result = await gracefulErrorHandler.executeWithErrorHandling(
          async () => {
            return await method.apply(this, args);
          },
          {
            strategy: retries > 0 ? 'retry' : 'fallback',
            maxRetries: retries,
            retryDelay,
            fallbackValue,
            showUserNotification: notifyUser,
            logError,
            context: {
              component: errorContext,
              function: propertyKey,
              operation: `ASYNC_OP_${propertyKey.toUpperCase()}`,
              additionalData: {
                arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
              }
            }
          }
        );

        if (result.success) {
          return result.data;
        } else {
          if (suppressError) {
            return fallbackValue;
          } else {
            throw result.error || new Error(`Operation failed after ${retries} retries`);
          }
        }
      } catch (error) {
        if (logError) {
          errorLogger.logError(error, 'error', {
            component: errorContext,
            function: propertyKey,
            additionalData: {
              arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
            }
          });
        }

        if (notifyUser) {
          toast.error(`Operation failed: ${error.message}`);
        }

        if (suppressError) {
          return fallbackValue;
        } else {
          throw error;
        }
      }
    };

    return descriptor;
  };
}

/**
 * Network operation decorator with specific network error handling
 */
export function HandleNetworkOperation(options: ErrorHandlingDecoratorOptions = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const {
        retries = 3,
        retryDelay = 1000,
        fallbackValue = undefined,
        logError = true,
        notifyUser = true,
        errorContext = `${target.constructor.name}.${propertyKey}`,
        suppressError = false
      } = { retries: 3, ...options }; // Default to 3 retries for network operations

      let attempt = 0;
      let lastError: any;

      while (attempt <= retries) {
        try {
          const result = await method.apply(this, args);
          
          // If successful, clear any network error notifications
          if (attempt > 0) {
            toast.dismiss(); // Clear any previous error toasts
            toast.success('Connection restored!');
          }
          
          return result;
        } catch (error) {
          lastError = error;
          attempt++;

          // Check if it's a network error
          const isNetworkError = 
            error.message.toLowerCase().includes('network') ||
            error.message.toLowerCase().includes('fetch') ||
            error.message.toLowerCase().includes('http') ||
            error.message.includes('502') ||
            error.message.includes('503') ||
            error.message.includes('504') ||
            error.message.includes('408') ||
            error.name === 'TypeError' ||
            error.message.includes('Failed to fetch') ||
            error.message.includes('Network Error');

          if (logError) {
            errorLogger.logError(error, isNetworkError ? 'error' : 'warn', {
              component: errorContext,
              function: propertyKey,
              additionalData: {
                attempt,
                maxRetries: retries,
                isNetworkError,
                arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
              }
            });
          }

          if (notifyUser) {
            if (isNetworkError) {
              if (attempt <= retries) {
                toast.warn(`Network issue (${attempt}/${retries}). Retrying...`);
              } else {
                toast.error(`Network error: ${error.message}`);
              }
            } else {
              toast.error(`Error: ${error.message}`);
            }
          }

          if (attempt <= retries) {
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, retryDelay * attempt)); // Exponential backoff
          }
        }
      }

      // If all retries failed
      if (suppressError) {
        return fallbackValue;
      } else {
        throw lastError;
      }
    };

    return descriptor;
  };
}

/**
 * Validation decorator to handle validation errors specifically
 */
export function HandleValidation(options: ErrorHandlingDecoratorOptions = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const {
        fallbackValue = undefined,
        logError = true,
        notifyUser = true,
        errorContext = `${target.constructor.name}.${propertyKey}`,
        suppressError = false
      } = options;

      try {
        return await method.apply(this, args);
      } catch (error) {
        // Check if it's a validation error
        const isValidationError = 
          error.message.toLowerCase().includes('validation') ||
          error.message.toLowerCase().includes('invalid') ||
          error.message.toLowerCase().includes('required') ||
          error.message.toLowerCase().includes('format') ||
          error.message.includes('422');

        if (isValidationError) {
          if (logError) {
            errorLogger.logError(error, 'warn', {
              component: errorContext,
              function: propertyKey,
              additionalData: {
                isValidation: true,
                arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
              }
            });
          }

          if (notifyUser) {
            toast.error(`Validation error: ${error.message}`);
          }
        } else {
          // Re-throw non-validation errors
          if (logError) {
            errorLogger.logError(error, 'error', {
              component: errorContext,
              function: propertyKey,
              additionalData: {
                isValidation: false,
                arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg)
              }
            });
          }

          if (notifyUser) {
            toast.error(`Error: ${error.message}`);
          }
        }

        if (suppressError) {
          return fallbackValue;
        } else {
          throw error;
        }
      }
    };

    return descriptor;
  };
}

/**
 * Decorator factory for creating custom error handling decorators
 */
export function createErrorHandlingDecorator(customHandler: (error: any, context: any) => any) {
  return function (options: ErrorHandlingDecoratorOptions = {}) {
    return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
      const method = descriptor.value;

      descriptor.value = async function (...args: any[]) {
        try {
          return await method.apply(this, args);
        } catch (error) {
          const context = {
            component: `${target.constructor.name}.${propertyKey}`,
            function: propertyKey,
            error,
            arguments: args,
            options
          };

          return customHandler(error, context);
        }
      };

      return descriptor;
    };
  };
}

/**
 * Memoized error handler decorator to prevent duplicate error handling
 */
export function MemoizeError(options: ErrorHandlingDecoratorOptions = {}) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const method = descriptor.value;
    const errorCache = new Map<string, { error: any; timestamp: number; ttl: number }>();

    descriptor.value = async function (...args: any[]) {
      const {
        fallbackValue = undefined,
        logError = true,
        notifyUser = true,
        errorContext = `${target.constructor.name}.${propertyKey}`,
        suppressError = false,
        ...restOptions
      } = options;

      // Create a cache key based on the method and arguments
      const cacheKey = `${propertyKey}_${JSON.stringify(args)}`;
      const now = Date.now();
      const ttl = restOptions.retryDelay ? restOptions.retryDelay * 2 : 30000; // Default 30 second TTL

      // Check if we have a cached error for this operation
      if (errorCache.has(cacheKey)) {
        const cached = errorCache.get(cacheKey)!;
        if (now - cached.timestamp < cached.ttl) {
          // Still within TTL, return fallback or re-throw
          if (suppressError) {
            return fallbackValue;
          } else {
            throw cached.error;
          }
        } else {
          // Expired, remove from cache
          errorCache.delete(cacheKey);
        }
      }

      try {
        const result = await method.apply(this, args);
        // If successful, clear any cached error
        errorCache.delete(cacheKey);
        return result;
      } catch (error) {
        // Cache the error
        errorCache.set(cacheKey, { error, timestamp: now, ttl });

        if (logError) {
          errorLogger.logError(error, 'error', {
            component: errorContext,
            function: propertyKey,
            additionalData: {
              arguments: args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : arg),
              cached: true
            }
          });
        }

        if (notifyUser) {
          toast.error(`Error: ${error.message}`);
        }

        if (suppressError) {
          return fallbackValue;
        } else {
          throw error;
        }
      }
    };

    return descriptor;
  };
}