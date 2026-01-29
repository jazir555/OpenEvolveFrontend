/**
 * Safe Timer Utilities
 * 
 * Provides utilities for safe timer operations with comprehensive error handling
 */

import { errorLogger } from './errorLogging';

/**
 * Safe setTimeout with error handling
 */
export function safeSetTimeout(
  callback: () => void,
  delay: number,
  options?: {
    onError?: (error: Error) => void;
  }
): number {
  const { onError } = options || {};
  
  try {
    return window.setTimeout(() => {
      try {
        callback();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'safeSetTimeout.callback', 
            additionalData: { delay } 
          }
        );
        
        onError?.(typedError);
      }
    }, delay);
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeTimer', 
        function: 'safeSetTimeout', 
        additionalData: { delay } 
      }
    );
    
    onError?.(typedError);
    
    // Return a no-op timeout ID
    return -1;
  }
}

/**
 * Safe clearTimeout with error handling
 */
export function safeClearTimeout(timeoutId: number): void {
  try {
    window.clearTimeout(timeoutId);
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'SafeTimer', 
        function: 'safeClearTimeout', 
        additionalData: { timeoutId } 
      }
    );
  }
}

/**
 * Safe setInterval with error handling
 */
export function safeSetInterval(
  callback: () => void,
  interval: number,
  options?: {
    onError?: (error: Error) => void;
  }
): number {
  const { onError } = options || {};
  
  try {
    return window.setInterval(() => {
      try {
        callback();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'safeSetInterval.callback', 
            additionalData: { interval } 
          }
        );
        
        onError?.(typedError);
      }
    }, interval);
  } catch (error) {
    const typedError = error instanceof Error ? error : new Error(String(error));
    
    errorLogger.logError(
      typedError,
      'error',
      { 
        component: 'SafeTimer', 
        function: 'safeSetInterval', 
        additionalData: { interval } 
      }
    );
    
    onError?.(typedError);
    
    // Return a no-op interval ID
    return -1;
  }
}

/**
 * Safe clearInterval with error handling
 */
export function safeClearInterval(intervalId: number): void {
  try {
    window.clearInterval(intervalId);
  } catch (error) {
    errorLogger.logError(
      error instanceof Error ? error : new Error(String(error)),
      'error',
      { 
        component: 'SafeTimer', 
        function: 'safeClearInterval', 
        additionalData: { intervalId } 
      }
    );
  }
}

/**
 * Safe timer manager with error handling
 */
export class SafeTimerManager {
  private timeouts: Set<number> = new Set();
  private intervals: Set<number> = new Set();
  private onError?: (error: Error) => void;

  constructor(options?: { onError?: (error: Error) => void }) {
    this.onError = options?.onError;
  }

  /**
   * Set a safe timeout
   */
  setTimeout(
    callback: () => void,
    delay: number
  ): number {
    try {
      const timeoutId = safeSetTimeout(callback, delay, { onError: this.onError });
      
      if (timeoutId !== -1) {
        this.timeouts.add(timeoutId);
      }
      
      return timeoutId;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.setTimeout', 
          additionalData: { delay } 
        }
      );
      return -1;
    }
  }

  /**
   * Set a safe interval
   */
  setInterval(
    callback: () => void,
    interval: number
  ): number {
    try {
      const intervalId = safeSetInterval(callback, interval, { onError: this.onError });
      
      if (intervalId !== -1) {
        this.intervals.add(intervalId);
      }
      
      return intervalId;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.setInterval', 
          additionalData: { interval } 
        }
      );
      return -1;
    }
  }

  /**
   * Clear a timeout
   */
  clearTimeout(timeoutId: number): void {
    try {
      safeClearTimeout(timeoutId);
      this.timeouts.delete(timeoutId);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.clearTimeout', 
          additionalData: { timeoutId } 
        }
      );
    }
  }

  /**
   * Clear an interval
   */
  clearInterval(intervalId: number): void {
    try {
      safeClearInterval(intervalId);
      this.intervals.delete(intervalId);
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.clearInterval', 
          additionalData: { intervalId } 
        }
      );
    }
  }

  /**
   * Clear all timeouts
   */
  clearAllTimeouts(): void {
    try {
      this.timeouts.forEach(timeoutId => {
        try {
          safeClearTimeout(timeoutId);
        } catch (clearError) {
          errorLogger.logError(
            clearError instanceof Error ? clearError : new Error(String(clearError)),
            'error',
            { 
              component: 'SafeTimer', 
              function: 'SafeTimerManager.clearAllTimeouts.timeout', 
              additionalData: { timeoutId } 
            }
          );
        }
      });
      this.timeouts.clear();
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.clearAllTimeouts' 
        }
      );
    }
  }

  /**
   * Clear all intervals
   */
  clearAllIntervals(): void {
    try {
      this.intervals.forEach(intervalId => {
        try {
          safeClearInterval(intervalId);
        } catch (clearError) {
          errorLogger.logError(
            clearError instanceof Error ? clearError : new Error(String(clearError)),
            'error',
            { 
              component: 'SafeTimer', 
              function: 'SafeTimerManager.clearAllIntervals.interval', 
              additionalData: { intervalId } 
            }
          );
        }
      });
      this.intervals.clear();
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.clearAllIntervals' 
        }
      );
    }
  }

  /**
   * Clear all timers
   */
  clearAll(): void {
    this.clearAllTimeouts();
    this.clearAllIntervals();
  }

  /**
   * Get the number of active timeouts
   */
  getTimeoutCount(): number {
    try {
      return this.timeouts.size;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.getTimeoutCount' 
        }
      );
      return 0;
    }
  }

  /**
   * Get the number of active intervals
   */
  getIntervalCount(): number {
    try {
      return this.intervals.size;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.getIntervalCount' 
        }
      );
      return 0;
    }
  }

  /**
   * Get the total number of active timers
   */
  getTimerCount(): number {
    try {
      return this.timeouts.size + this.intervals.size;
    } catch (error) {
      errorLogger.logError(
        error instanceof Error ? error : new Error(String(error)),
        'error',
        { 
          component: 'SafeTimer', 
          function: 'SafeTimerManager.getTimerCount' 
        }
      );
      return 0;
    }
  }
}

/**
 * Safe timer with retry mechanism
 */
export function safeSetTimeoutWithRetry(
  callback: () => void,
  delay: number,
  options?: {
    onError?: (error: Error) => void;
    maxRetries?: number;
    retryDelay?: number;
  }
): number {
  const { onError, maxRetries = 3, retryDelay = 1000 } = options || {};
  let retryCount = 0;
  
  const safeCallback = () => {
    try {
      callback();
    } catch (error) {
      const typedError = error instanceof Error ? error : new Error(String(error));
      
      errorLogger.logError(
        typedError,
        'error',
        { 
          component: 'SafeTimer', 
          function: 'safeSetTimeoutWithRetry.callback', 
          additionalData: { delay, retryCount } 
        }
      );
      
      onError?.(typedError);
      
      // If we've reached max retries, don't try again
      if (retryCount >= maxRetries) {
        return;
      }
      
      // Retry after delay
      retryCount++;
      safeSetTimeoutWithRetry(callback, retryDelay, { 
        onError, 
        maxRetries, 
        retryDelay 
      });
    }
  };
  
  return safeSetTimeout(safeCallback, delay, { onError });
}

/**
 * Safe delayed promise with error handling
 */
export function safeDelay(
  ms: number,
  options?: {
    onError?: (error: Error) => void;
  }
): Promise<void> {
  const { onError } = options || {};
  
  return new Promise((resolve) => {
    safeSetTimeout(() => {
      try {
        resolve();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'safeDelay', 
            additionalData: { ms } 
          }
        );
        
        onError?.(typedError);
        resolve(); // Still resolve to avoid hanging promises
      }
    }, ms);
  });
}

/**
 * Safe timer with cancellation support
 */
export function createCancellableTimer(
  callback: () => void,
  delay: number,
  options?: {
    onError?: (error: Error) => void;
  }
): { id: number; cancel: () => void } {
  const { onError } = options || {};
  let cancelled = false;
  
  const safeCallback = () => {
    if (!cancelled) {
      try {
        callback();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'createCancellableTimer.callback', 
            additionalData: { delay } 
          }
        );
        
        onError?.(typedError);
      }
    }
  };
  
  const id = safeSetTimeout(safeCallback, delay, { onError });
  
  return {
    id,
    cancel: () => {
      if (!cancelled) {
        cancelled = true;
        safeClearTimeout(id);
      }
    }
  };
}

/**
 * Safe interval with cancellation support
 */
export function createCancellableInterval(
  callback: () => void,
  interval: number,
  options?: {
    onError?: (error: Error) => void;
  }
): { id: number; cancel: () => void } {
  const { onError } = options || {};
  let cancelled = false;
  
  const safeCallback = () => {
    if (!cancelled) {
      try {
        callback();
      } catch (error) {
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'createCancellableInterval.callback', 
            additionalData: { interval } 
          }
        );
        
        onError?.(typedError);
      }
    }
  };
  
  const id = safeSetInterval(safeCallback, interval, { onError });
  
  return {
    id,
    cancel: () => {
      if (!cancelled) {
        cancelled = true;
        safeClearInterval(id);
      }
    }
  };
}

/**
 * Safe timer that executes callback with error handling
 */
export function safeExecuteWithTimeout<T>(
  fn: () => T,
  timeout: number,
  options?: {
    onError?: (error: Error) => void;
    fallbackValue?: T;
  }
): Promise<T> {
  const { onError, fallbackValue } = options || {};
  
  return new Promise((resolve, reject) => {
    let completed = false;
    
    // Set up the timeout
    const timeoutId = safeSetTimeout(() => {
      if (!completed) {
        completed = true;
        const timeoutError = new Error(`Function timed out after ${timeout}ms`);
        errorLogger.logError(timeoutError, 'error', {
          component: 'SafeTimer',
          function: 'safeExecuteWithTimeout',
          additionalData: { timeout }
        });
        reject(timeoutError);
      }
    }, timeout);
    
    // Execute the function
    try {
      const result = fn();
      if (!completed) {
        completed = true;
        safeClearTimeout(timeoutId);
        resolve(result);
      }
    } catch (error) {
      if (!completed) {
        completed = true;
        safeClearTimeout(timeoutId);
        const typedError = error instanceof Error ? error : new Error(String(error));
        
        errorLogger.logError(
          typedError,
          'error',
          { 
            component: 'SafeTimer', 
            function: 'safeExecuteWithTimeout.fn', 
            additionalData: { timeout } 
          }
        );
        
        onError?.(typedError);
        resolve(fallbackValue as T); // Resolve with fallback to avoid hanging
      }
    }
  });
}