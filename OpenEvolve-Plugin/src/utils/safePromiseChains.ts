/**
 * Safe Promise Chain Utilities
 * 
 * Provides utilities for safe promise handling with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe promise with error handling
 */
export function safePromise<T>(
  executor: (resolve: (value: T) => void, reject: (reason: any) => void) => void,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: T;
  }
): Promise<T> {
  const { onError, fallbackValue } = options || {};
  
  return new Promise<T>((resolve, reject) => {
    try {
      executor(
        (value) => {
          try {
            resolve(value);
          } catch (resolveError) {
            const typedError = resolveError instanceof Error ? resolveError : new Error(String(resolveError));
            errorLogger.logError(
              typedError,
              'error',
              { 
                component: 'SafePromise', 
                function: 'safePromise.executor.resolve', 
                additionalData: { hasOnError: !!onError } 
              }
            );
            onError?.(typedError);
            resolve(fallbackValue as T);
          }
        },
        (reason) => {
          try {
            const error = reason instanceof Error ? reason : new Error(String(reason));
            errorLogger.logError(
              error,
              'error',
              { 
                component: 'SafePromise', 
                function: 'safePromise.executor.reject', 
                additionalData: { hasOnError: !!onError } 
              }
            );
            onError?.(error);
            reject(reason);
          } catch (rejectError) {
            const typedError = rejectError instanceof Error ? rejectError : new Error(String(rejectError));
            errorLogger.logError(
              typedError,
              'error',
              { 
                component: 'SafePromise', 
                function: 'safePromise.executor.reject.error', 
                additionalData: { hasOnError: !!onError } 
              }
            );
            onError?.(typedError);
            reject(reason);
          }
        }
      );
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safePromise.executor', 
          additionalData: { hasOnError: !!onError } 
        }
      );
      onError?.(typedError);
      resolve(fallbackValue as T);
    }
  });
}

/**
 * Safe promise resolution with error handling
 */
export function safeResolve<T>(
  value: T | PromiseLike<T>,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: T;
  }
): Promise<T> {
  const { onError, fallbackValue } = options || {};
  
  return Promise.resolve(value).catch(error => {
    const typedError = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafePromise', 
        function: 'safeResolve', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    onError?.(typedError);
    return fallbackValue as T;
  });
}

/**
 * Safe promise rejection with error handling
 */
export function safeReject<T = never>(
  reason: any,
  options?: {
    onError?: (error: Error) => void;
  }
): Promise<T> {
  const { onError } = options || {};
  
  try {
    const error = reason instanceof Error ? reason : new Error(String(reason));
    errorLogger.logError(
      error,
      'error',
      { 
        component: 'SafePromise', 
        function: 'safeReject', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    onError?.(error);
    return Promise.reject(reason);
  } catch (loggingError) {
    const typedError = loggingError instanceof Error ? loggingError : new Error(String(loggingError));
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafePromise', 
        function: 'safeReject.logging', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    onError?.(typedError);
    return Promise.reject(reason);
  }
}

/**
 * Safe promise chain with error handling
 */
export function safeChain<T, U>(
  promise: Promise<T>,
  onFulfilled: (value: T) => U | PromiseLike<U>,
  onRejected?: (reason: any) => U | PromiseLike<U>,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: U;
  }
): Promise<U> {
  const { onError, fallbackValue } = options || {};
  
  return promise
    .then(value => {
      try {
        return onFulfilled(value);
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafePromise', 
            function: 'safeChain.onFulfilled', 
            additionalData: { hasOnError: !!onError } 
          }
        );
        onError?.(typedError);
        return fallbackValue as U;
      }
    })
    .catch(error => {
      if (onRejected) {
        try {
          return onRejected(error);
        } catch (rejectionError) {
          const typedError = rejectionError instanceof Error ? rejectionError : new Error(String(rejectionError));
          errorLogger.logError(
            typedError,
            'error',
            { 
              component: 'SafePromise', 
              function: 'safeChain.onRejected', 
              additionalData: { hasOnError: !!onError } 
            }
          );
          onError?.(typedError);
          return fallbackValue as U;
        }
      } else {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafePromise', 
            function: 'safeChain.catch', 
            additionalData: { hasOnError: !!onError } 
          }
        );
        onError?.(typedError);
        return fallbackValue as U;
      }
    });
}

/**
 * Safe promise all with error handling
 */
export function safePromiseAll<T>(
  promises: Iterable<Promise<T>>,
  options?: {
    onError?: (error: Error) => void;
    continueOnError?: boolean;
  }
): Promise<T[]> {
  const { onError, continueOnError = false } = options || {};
  
  if (continueOnError) {
    // If continueOnError is true, we'll handle individual promise rejections
    const promiseArray = Array.from(promises);
    const wrappedPromises = promiseArray.map((promise, index) => 
      promise.catch(error => {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafePromise', 
            function: 'safePromiseAll.continueOnError', 
            additionalData: { promiseIndex: index, hasOnError: !!onError } 
          }
        );
        onError?.(typedError);
        // Return a special value to indicate failure, or re-throw to fail the whole operation
        throw error;
      })
    );
    
    return Promise.all(wrappedPromises).catch(error => {
      const typedError = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safePromiseAll.continueOnError.all', 
          additionalData: { hasOnError: !!onError } 
        }
      );
      onError?.(typedError);
      return [];
    });
  } else {
    // If continueOnError is false, fail fast on first error
    return Promise.all(promises).catch(error => {
      const typedError = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safePromiseAll', 
          additionalData: { hasOnError: !!onError } 
        }
      );
      onError?.(typedError);
      return [];
    });
  }
}

/**
 * Safe promise all settled with error handling
 */
export function safePromiseAllSettled<T>(
  promises: Iterable<Promise<T>>,
  options?: {
    onError?: (error: Error) => void;
  }
): Promise<PromiseSettledResult<T>[]> {
  const { onError } = options || {};
  
  return Promise.allSettled(promises).then(results => {
    // Log any rejected promises
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        const typedError = result.reason instanceof Error ? result.reason : new Error(String(result.reason));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafePromise', 
            function: 'safePromiseAllSettled', 
            additionalData: { promiseIndex: index, hasOnError: !!onError } 
          }
        );
        onError?.(typedError);
      }
    });
    return results;
  }).catch(error => {
    const typedError = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafePromise', 
        function: 'safePromiseAllSettled.catch', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    onError?.(typedError);
    return [];
  });
}

/**
 * Safe promise race with error handling
 */
export function safePromiseRace<T>(
  promises: Iterable<Promise<T>>,
  options?: {
    onError?: (error: Error) => void;
  }
): Promise<T> {
  const { onError } = options || {};
  
  return Promise.race(promises).catch(error => {
    const typedError = error instanceof Error ? error : new Error(String(error));
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafePromise', 
        function: 'safePromiseRace', 
        additionalData: { hasOnError: !!onError } 
      }
    );
    onError?.(typedError);
    
    // Since race rejects with the first rejection, we need to return a fallback
    // In a real implementation, you might want to handle this differently
    throw error;
  });
}

/**
 * Safe async function wrapper with error handling
 */
export function safeAsync<T>(
  asyncFn: () => Promise<T>,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: T;
    retries?: number;
    retryDelay?: number;
  }
): Promise<T> {
  const { onError, fallbackValue, retries = 0, retryDelay = 1000 } = options || {};
  
  let attempt = 0;
  
  const attemptAsync = (): Promise<T> => {
    return asyncFn().catch(error => {
      const typedError = error instanceof Error ? error : new Error(String(error));
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safeAsync', 
          additionalData: { attempt, retries, hasOnError: !!onError } 
        }
      );
      
      if (attempt < retries) {
        attempt++;
        onError?.(typedError);
        return new Promise(resolve => setTimeout(() => {
          resolve(attemptAsync());
        }, retryDelay));
      } else {
        onError?.(typedError);
        return fallbackValue as T;
      }
    });
  };
  
  return attemptAsync();
}

/**
 * Safe promise with timeout
 */
export function safePromiseWithTimeout<T>(
  promise: Promise<T>,
  timeout: number,
  options?: {
    onError?: (error: Error) => void;
    timeoutMessage?: string;
  }
): Promise<T> {
  const { onError, timeoutMessage = 'Promise timed out' } = options || {};
  
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => {
      const timeoutError = new Error(timeoutMessage);
      errorLogger.logError(
        timeoutError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safePromiseWithTimeout', 
          additionalData: { timeout, hasOnError: !!onError } 
        }
      );
      onError?.(timeoutError);
      reject(timeoutError);
    }, timeout);
  });
  
  return Promise.race([promise, timeoutPromise]).catch(error => {
    const typedError = error instanceof Error ? error : new Error(String(error));
    if (typedError.message !== timeoutMessage) {
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safePromiseWithTimeout.race', 
          additionalData: { timeout, hasOnError: !!onError } 
        }
      );
      onError?.(typedError);
    }
    throw error;
  });
}

/**
 * Safe promise retry mechanism
 */
export function safeRetry<T>(
  asyncFn: () => Promise<T>,
  options?: {
    retries?: number;
    retryDelay?: number;
    onError?: (error: Error) => void;
    fallbackValue?: T;
    shouldRetry?: (error: Error, attempt: number) => boolean;
  }
): Promise<T> {
  const { 
    retries = 3, 
    retryDelay = 1000, 
    onError, 
    fallbackValue,
    shouldRetry = () => true 
  } = options || {};
  
  const attempt = (remainingRetries: number): Promise<T> => {
    return asyncFn().catch(error => {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      // Log the error
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafePromise', 
          function: 'safeRetry', 
          additionalData: { remainingRetries, hasOnError: !!onError } 
        }
      );
      
      onError?.(typedError);
      
      // Check if we should retry
      if (remainingRetries > 0 && shouldRetry(typedError, retries - remainingRetries)) {
        return new Promise(resolve => {
          setTimeout(() => {
            resolve(attempt(remainingRetries - 1));
          }, retryDelay);
        });
      } else {
        // No more retries or shouldn't retry
        return fallbackValue as T;
      }
    });
  };
  
  return attempt(retries);
}

/**
 * Safe promise waterfall (sequential execution)
 */
export function safeWaterfall<T>(
  functions: Array<() => Promise<T>>,
  options?: {
    onError?: (error: Error) => void;
    continueOnError?: boolean;
  }
): Promise<T[]> {
  const { onError, continueOnError = false } = options || {};
  
  const results: T[] = [];
  
  return functions.reduce((promise, fn) => {
    return promise.then(async (acc) => {
      try {
        const result = await fn();
        acc.push(result);
        return acc;
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafePromise', 
            function: 'safeWaterfall', 
            additionalData: { hasOnError: !!onError } 
          }
        );
        onError?.(typedError);
        
        if (continueOnError) {
          // Create a structured error result instead of undefined
          const errorResult = {
            success: false,
            error: typedError,
            functionName: fn.name || 'anonymous',
            timestamp: new Date().toISOString()
          };
          acc.push(errorResult as any);
          return acc;
        } else {
          throw error;
        }
      }
    });
  }, Promise.resolve([] as T[]));
}