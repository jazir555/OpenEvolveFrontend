/**
 * Safe Wrapper Functions
 * Provides safe wrapper functions for various operations to handle errors gracefully
 */

import { errorLogger } from './errorLogging';
import { gracefulErrorHandler } from './gracefulErrorHandler';
import { toast } from 'react-toastify';

// Define options for safe operations
interface SafeOperationOptions {
  fallbackValue?: any;
  logError?: boolean;
  notifyUser?: boolean;
  errorContext?: string;
  suppressError?: boolean;
  retries?: number;
  retryDelay?: number;
}

/**
 * Safely execute an async operation with error handling
 */
export async function safeAsyncOperation<T>(
  operation: () => Promise<T>,
  options: SafeOperationOptions = {}
): Promise<{ success: boolean; data?: T; error?: any }> {
  const {
    fallbackValue = undefined,
    logError = true,
    notifyUser = true,
    errorContext = 'safeAsyncOperation',
    suppressError = false,
    retries = 0,
    retryDelay = 1000,
  } = options;

  let attempt = 0;
  let lastError: any;

  while (attempt <= retries) {
    try {
      const result = await operation();
      return { success: true, data: result };
    } catch (error) {
      lastError = error;
      attempt++;

      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'safeAsyncOperation',
          additionalData: {
            attempt,
            maxRetries: retries,
          },
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
    return { success: true, data: fallbackValue };
  } else {
    return { success: false, error: lastError };
  }
}

/**
 * Safely execute a synchronous operation with error handling
 */
export function safeSyncOperation<T>(
  operation: () => T,
  options: SafeOperationOptions = {}
): { success: boolean; data?: T; error?: any } {
  const {
    fallbackValue = undefined,
    logError = true,
    notifyUser = true,
    errorContext = 'safeSyncOperation',
    suppressError = false,
  } = options;

  try {
    const result = operation();
    return { success: true, data: result };
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'error', {
        component: errorContext,
        function: 'safeSyncOperation',
        additionalData: {},
      });
    }

    if (notifyUser) {
      toast.error(`Error in ${errorContext}: ${error.message}`);
    }

    if (suppressError) {
      return { success: true, data: fallbackValue };
    } else {
      return { success: false, error };
    }
  }
}

/**
 * Safely access object properties with fallback
 */
export function safeGet<T>(
  obj: any,
  path: string | string[],
  fallbackValue?: T
): T | undefined {
  try {
    const keys = Array.isArray(path) ? path : path.split('.');
    let result = obj;

    for (const key of keys) {
      if (result == null) {
        return fallbackValue;
      }
      result = result[key];
    }

    return result as T;
  } catch (error) {
    errorLogger.logError(error, 'warn', {
      component: 'safeGet',
      function: 'safeGet',
      additionalData: { path, objType: typeof obj },
    });
    return fallbackValue;
  }
}

/**
 * Safely parse JSON with fallback
 */
export function safeJsonParse<T = any>(
  jsonString: string,
  fallbackValue?: T,
  options: { logError?: boolean; errorContext?: string } = {}
): T | undefined {
  const { logError = true, errorContext = 'safeJsonParse' } = options;

  try {
    return JSON.parse(jsonString) as T;
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'warn', {
        component: errorContext,
        function: 'safeJsonParse',
        additionalData: { jsonStringLength: jsonString.length },
      });
    }
    return fallbackValue;
  }
}

/**
 * Safely stringify JSON with fallback
 */
export function safeJsonStringify(
  obj: any,
  fallbackValue: string = '',
  options: { logError?: boolean; errorContext?: string } = {}
): string {
  const { logError = true, errorContext = 'safeJsonStringify' } = options;

  try {
    return JSON.stringify(obj);
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'warn', {
        component: errorContext,
        function: 'safeJsonStringify',
        additionalData: { objType: typeof obj },
      });
    }
    return fallbackValue;
  }
}

/**
 * Safe timeout function
 */
export function safeTimeout(
  callback: () => void,
  delay: number,
  options: { logError?: boolean; errorContext?: string } = {}
): NodeJS.Timeout {
  const { logError = true, errorContext = 'safeTimeout' } = options;

  return setTimeout(() => {
    try {
      callback();
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'timeoutCallback',
          additionalData: { delay },
        });
      }
      toast.error(`Error in timeout callback: ${error.message}`);
    }
  }, delay);
}

/**
 * Safe interval function
 */
export function safeInterval(
  callback: () => void,
  interval: number,
  options: { logError?: boolean; errorContext?: string } = {}
): NodeJS.Timeout {
  const { logError = true, errorContext = 'safeInterval' } = options;

  return setInterval(() => {
    try {
      callback();
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'intervalCallback',
          additionalData: { interval },
        });
      }
      toast.error(`Error in interval callback: ${error.message}`);
    }
  }, interval);
}

/**
 * Safe promise chain with error handling
 */
export async function safePromiseChain<T>(
  operations: Array<() => Promise<T>>,
  options: SafeOperationOptions = {}
): Promise<{ success: boolean; data?: T; error?: any; operationIndex?: number }> {
  const {
    fallbackValue = undefined,
    logError = true,
    notifyUser = true,
    errorContext = 'safePromiseChain',
    suppressError = false,
  } = options;

  for (let i = 0; i < operations.length; i++) {
    try {
      const result = await operations[i]();
      return { success: true, data: result, operationIndex: i };
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'safePromiseChain',
          additionalData: { operationIndex: i, totalOperations: operations.length },
        });
      }

      if (notifyUser) {
        toast.error(`Error in operation ${i + 1}/${operations.length}: ${error.message}`);
      }

      // If not suppressing error, return immediately with the error
      if (!suppressError) {
        return { success: false, error, operationIndex: i };
      }
    }
  }

  // If all operations failed but errors were suppressed, return fallback
  if (suppressError) {
    return { success: true, data: fallbackValue };
  } else {
    return { success: false, error: new Error('All operations in chain failed') };
  }
}

/**
 * Safe event handler wrapper
 */
export function safeEventHandler<T extends Event>(
  handler: (event: T) => void,
  options: { logError?: boolean; errorContext?: string; preventDefaultOnError?: boolean } = {}
): (event: T) => void {
  const {
    logError = true,
    errorContext = 'safeEventHandler',
    preventDefaultOnError = true,
  } = options;

  return (event: T) => {
    try {
      handler(event);
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'safeEventHandler',
          additionalData: { eventType: event.type },
        });
      }

      if (preventDefaultOnError && 'preventDefault' in event) {
        (event as any).preventDefault();
      }

      toast.error(`Event handler error: ${error.message}`);
    }
  };
}

/**
 * Safe effect hook wrapper for React
 */
export function safeEffect(
  effect: () => void | (() => void),
  deps?: React.DependencyList,
  options: { logError?: boolean; errorContext?: string } = {}
): () => void | (() => void) {
  const { logError = true, errorContext = 'safeEffect' } = options;

  return () => {
    try {
      return effect();
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'safeEffect',
          additionalData: { deps: deps ? deps.length : 0 },
        });
      }

      toast.error(`Effect error: ${error.message}`);
      return () => {}; // Return empty cleanup function
    }
  };
}

/**
 * Safe reducer function wrapper
 */
export function safeReducer<S, A>(
  reducer: (state: S, action: A) => S,
  initialState: S,
  options: { logError?: boolean; errorContext?: string } = {}
): [S, (action: A) => void] {
  const { logError = true, errorContext = 'safeReducer' } = options;
  let currentState = initialState;

  const dispatch = (action: A) => {
    try {
      currentState = reducer(currentState, action);
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'error', {
          component: errorContext,
          function: 'safeReducer',
          additionalData: { actionType: (action as any)?.type },
        });
      }

      toast.error(`Reducer error: ${error.message}`);
    }
  };

  return [currentState, dispatch];
}

/**
 * Safe context consumer wrapper
 */
export function safeContextConsumer<T>(
  context: React.Context<T>,
  fallbackValue?: T,
  options: { logError?: boolean; errorContext?: string } = {}
): T {
  const { logError = true, errorContext = 'safeContextConsumer' } = options;

  try {
    return React.useContext(context);
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'warn', {
        component: errorContext,
        function: 'safeContextConsumer',
        additionalData: {},
      });
    }

    return fallbackValue as T;
  }
}

/**
 * Safe storage access with error handling
 */
export function safeStorage<T>(
  storage: Storage,
  key: string,
  fallbackValue?: T,
  options: { logError?: boolean; errorContext?: string } = {}
): T | null {
  const { logError = true, errorContext = 'safeStorage' } = options;

  try {
    const item = storage.getItem(key);
    if (item === null) return fallbackValue as T;
    return safeJsonParse(item, fallbackValue, { logError: false });
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'warn', {
        component: errorContext,
        function: 'safeStorage',
        additionalData: { key, storageType: storage === localStorage ? 'localStorage' : 'sessionStorage' },
      });
    }
    return fallbackValue as T;
  }
}

/**
 * Safe storage setter with error handling
 */
export function safeSetStorage(
  storage: Storage,
  key: string,
  value: any,
  options: { logError?: boolean; errorContext?: string } = {}
): boolean {
  const { logError = true, errorContext = 'safeSetStorage' } = options;

  try {
    const serializedValue = typeof value === 'string' ? value : safeJsonStringify(value);
    storage.setItem(key, serializedValue);
    return true;
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'warn', {
        component: errorContext,
        function: 'safeSetStorage',
        additionalData: { key, storageType: storage === localStorage ? 'localStorage' : 'sessionStorage' },
      });
    }
    return false;
  }
}

/**
 * Safe fetch wrapper with error handling
 */
export async function safeFetch(
  input: RequestInfo,
  init?: RequestInit,
  options: SafeOperationOptions & { timeout?: number } = {}
): Promise<{ success: boolean; data?: Response; error?: any }> {
  const {
    fallbackValue = undefined,
    logError = true,
    notifyUser = true,
    errorContext = 'safeFetch',
    suppressError = false,
    timeout = 10000, // 10 seconds default
  } = options;

  try {
    // Create abort controller for timeout
    const abortController = new AbortController();
    const timeoutId = setTimeout(() => abortController.abort(), timeout);

    const response = await fetch(input, {
      ...init,
      signal: abortController.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    return { success: true, data: response };
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'error', {
        component: errorContext,
        function: 'safeFetch',
        additionalData: { input, timeout },
      });
    }

    if (notifyUser) {
      toast.error(`Network request failed: ${error.message}`);
    }

    if (suppressError) {
      return { success: true, data: fallbackValue };
    } else {
      return { success: false, error };
    }
  }
}

/**
 * Safe array operation with error handling
 */
export function safeArrayOperation<T, R>(
  array: T[],
  operation: (arr: T[]) => R,
  options: { logError?: boolean; errorContext?: string; fallbackValue?: R } = {}
): R | undefined {
  const { logError = true, errorContext = 'safeArrayOperation', fallbackValue = undefined } = options;

  try {
    return operation(array);
  } catch (error) {
    if (logError) {
      errorLogger.logError(error, 'error', {
        component: errorContext,
        function: 'safeArrayOperation',
        additionalData: { arrayLength: array.length },
      });
    }

    return fallbackValue;
  }
}

/**
 * Safe map operation with error handling for individual elements
 */
export function safeMap<T, R>(
  array: T[],
  mapper: (item: T, index: number) => R,
  options: { logError?: boolean; errorContext?: string; skipErrors?: boolean } = {}
): R[] {
  const { logError = true, errorContext = 'safeMap', skipErrors = true } = options;
  const result: R[] = [];

  for (let i = 0; i < array.length; i++) {
    try {
      result.push(mapper(array[i], i));
    } catch (error) {
      if (logError) {
        errorLogger.logError(error, 'warn', {
          component: errorContext,
          function: 'safeMap',
          additionalData: { index: i, arrayLength: array.length },
        });
      }

      if (!skipErrors) {
        throw error; // Re-throw if not skipping errors
      }
      // If skipping errors, continue to next element
    }
  }

  return result;
}

/**
 * Safe async map operation with error handling
 */
export async function safeAsyncMap<T, R>(
  array: T[],
  asyncMapper: (item: T, index: number) => Promise<R>,
  options: { logError?: boolean; errorContext?: string; skipErrors?: boolean; concurrency?: number } = {}
): Promise<R[]> {
  const { logError = true, errorContext = 'safeAsyncMap', skipErrors = true, concurrency = Infinity } = options;
  
  // Limit concurrency if specified
  if (concurrency < Infinity && concurrency > 0) {
    const results: R[] = [];
    for (let i = 0; i < array.length; i += concurrency) {
      const chunk = array.slice(i, i + concurrency);
      const chunkPromises = chunk.map((item, idx) => 
        safeAsyncOperation(() => asyncMapper(item, i + idx), { suppressError: skipErrors })
      );
      const chunkResults = await Promise.all(chunkPromises);
      
      for (const chunkResult of chunkResults) {
        if (chunkResult.success) {
          results.push(chunkResult.data as R);
        } else if (!skipErrors) {
          throw chunkResult.error;
        }
      }
    }
    return results;
  } else {
    // Process all items in parallel
    const promises = array.map((item, index) => 
      safeAsyncOperation(() => asyncMapper(item, index), { suppressError: skipErrors })
    );
    
    const results = await Promise.all(promises);
    return results
      .filter(result => result.success)
      .map(result => result.data as R);
  }
}