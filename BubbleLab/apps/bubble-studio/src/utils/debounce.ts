/**
 * Debounce and Throttle Utilities
 * Control the rate of function execution
 */

/**
 * Debounce function execution
 */
export function debounce<T extends (...args: never[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(later, wait);
  };
}

/**
 * Throttle function execution
 */
export function throttle<T extends (...args: never[]) => unknown>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Debounce promise (for async functions)
 */
export function debouncePromise<T extends (...args: never[]) => Promise<unknown>>(
  func: T,
  wait: number
): (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>>> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let pendingResolve: ((value: Awaited<ReturnType<T>>) => void) | null = null;
  let pendingArgs: Parameters<T> | null = null;

  return function executedFunction(...args: Parameters<T>): Promise<Awaited<ReturnType<T>>> {
    return new Promise((resolve) => {
      pendingArgs = args;
      pendingResolve = resolve as (value: Awaited<ReturnType<T>>) => void;

      if (timeout) {
        clearTimeout(timeout);
      }

      timeout = setTimeout(async () => {
        if (pendingArgs && pendingResolve) {
          const result = await func(...pendingArgs);
          pendingResolve(result);
          pendingResolve = null;
          pendingArgs = null;
        }
      }, wait);
    });
  };
}

/**
 * Create a debounced function that can be called immediately
 */
export function debounceWithImmediate<T extends (...args: never[]) => unknown>(
  func: T,
  wait: number
): {
  fn: (...args: Parameters<T>) => void;
  flush: () => void;
  cancel: () => void;
} {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  const debounced = (...args: Parameters<T>) => {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) {
      clearTimeout(timeout);
    }

    timeout = setTimeout(later, wait);
  };

  return {
    fn: debounced,
    flush: () => {
      if (timeout) {
        clearTimeout(timeout);
        func();
      }
    },
    cancel: () => {
      if (timeout) {
        clearTimeout(timeout);
      }
    },
  };
}
