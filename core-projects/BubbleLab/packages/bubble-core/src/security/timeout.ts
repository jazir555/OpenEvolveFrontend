/**
 * Timeout Utility
 * Provides timeout handling for async operations
 */

export class TimeoutError extends Error {
  constructor(message: string, public readonly timeoutMs: number) {
    super(message);
    this.name = 'TimeoutError';
  }
}

/**
 * Wraps a promise with a timeout
 * @param promise The promise to wrap
 * @param timeoutMs Timeout in milliseconds (default: 30000)
 * @returns Promise that rejects with TimeoutError if timeout is exceeded
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number = 30000
): Promise<T> {
  let timeoutHandle: NodeJS.Timeout;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutHandle = setTimeout(() => {
      reject(
        new TimeoutError(
          `Operation timed out after ${timeoutMs}ms`,
          timeoutMs
        )
      );
    }, timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutHandle!);
  }
}

/**
 * Creates a timeout wrapper function
 * @param timeoutMs Timeout in milliseconds
 * @returns Function that wraps promises with timeout
 */
export function createTimeoutWrapper(timeoutMs: number = 30000) {
  return <T>(promise: Promise<T>): Promise<T> => withTimeout(promise, timeoutMs);
}
