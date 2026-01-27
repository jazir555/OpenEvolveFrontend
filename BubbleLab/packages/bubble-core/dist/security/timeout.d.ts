/**
 * Timeout Utility
 * Provides timeout handling for async operations
 */
export declare class TimeoutError extends Error {
    readonly timeoutMs: number;
    constructor(message: string, timeoutMs: number);
}
/**
 * Wraps a promise with a timeout
 * @param promise The promise to wrap
 * @param timeoutMs Timeout in milliseconds (default: 30000)
 * @returns Promise that rejects with TimeoutError if timeout is exceeded
 */
export declare function withTimeout<T>(promise: Promise<T>, timeoutMs?: number): Promise<T>;
/**
 * Creates a timeout wrapper function
 * @param timeoutMs Timeout in milliseconds
 * @returns Function that wraps promises with timeout
 */
export declare function createTimeoutWrapper(timeoutMs?: number): <T>(promise: Promise<T>) => Promise<T>;
//# sourceMappingURL=timeout.d.ts.map