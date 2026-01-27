/**
 * Timeout Utility
 * Provides timeout handling for async operations
 */
export class TimeoutError extends Error {
    timeoutMs;
    constructor(message, timeoutMs) {
        super(message);
        this.timeoutMs = timeoutMs;
        this.name = 'TimeoutError';
    }
}
/**
 * Wraps a promise with a timeout
 * @param promise The promise to wrap
 * @param timeoutMs Timeout in milliseconds (default: 30000)
 * @returns Promise that rejects with TimeoutError if timeout is exceeded
 */
export async function withTimeout(promise, timeoutMs = 30000) {
    let timeoutHandle;
    const timeoutPromise = new Promise((_, reject) => {
        timeoutHandle = setTimeout(() => {
            reject(new TimeoutError(`Operation timed out after ${timeoutMs}ms`, timeoutMs));
        }, timeoutMs);
    });
    try {
        return await Promise.race([promise, timeoutPromise]);
    }
    finally {
        clearTimeout(timeoutHandle);
    }
}
/**
 * Creates a timeout wrapper function
 * @param timeoutMs Timeout in milliseconds
 * @returns Function that wraps promises with timeout
 */
export function createTimeoutWrapper(timeoutMs = 30000) {
    return (promise) => withTimeout(promise, timeoutMs);
}
//# sourceMappingURL=timeout.js.map