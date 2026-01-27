/**
 * Result Type for Error Handling
 *
 * Provides a type-safe way to handle operations that can fail
 * without throwing exceptions. Inspired by Rust's Result<T, E> and
 * functional programming patterns.
 */
/**
 * Result type - either success with data or failure with error
 */
export type Result<T, E = Error> = {
    success: true;
    data: T;
} | {
    success: false;
    error: E;
};
/**
 * Check if result is successful
 */
export declare function isSuccess<T, E>(result: Result<T, E>): result is {
    success: true;
    data: T;
};
/**
 * Check if result is failure
 */
export declare function isFailure<T, E>(result: Result<T, E>): result is {
    success: false;
    error: E;
};
/**
 * Wrap an async operation in a Result
 */
export declare function wrapAsync<T>(operation: () => Promise<T>): Promise<Result<T>>;
/**
 * Wrap a synchronous operation in a Result
 */
export declare function wrapSync<T>(operation: () => T): Result<T>;
/**
 * Map over the success value of a Result
 */
export declare function mapResult<T, U, E>(result: Result<T, E>, mapper: (data: T) => U): Result<U, E>;
/**
 * Map over the error value of a Result
 */
export declare function mapError<T, E, F>(result: Result<T, E>, mapper: (error: E) => F): Result<T, F>;
/**
 * Chain operations that return Results
 */
export declare function chainResult<T, U, E>(result: Result<T, E>, fn: (data: T) => Promise<Result<U, E>>): Promise<Result<U, E>>;
export declare function chainResult<T, U, E>(result: Result<T, E>, fn: (data: T) => Result<U, E>): Result<U, E>;
/**
 * Get data from Result or throw if failure
 */
export declare function unwrapOrThrow<T, E>(result: Result<T, E>): T;
/**
 * Get data from Result or return default value
 */
export declare function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T;
/**
 * Get data from Result or compute default value
 */
export declare function unwrapOrElse<T, E>(result: Result<T, E>, fn: (error: E) => T): T;
/**
 * Execute async operations in parallel and collect all Results
 */
export declare function all<T, E>(operations: Array<() => Promise<Result<T, E>>>): Promise<Array<Result<T, E>>>;
/**
 * Execute operations in parallel and return Result with array of data
 * Fails fast on first error
 */
export declare function allSuccess<T, E>(operations: Array<() => Promise<Result<T, E>>>): Promise<Result<T[], E>>;
/**
 * Retry an operation with exponential backoff
 */
export declare function retry<T>(operation: () => Promise<Result<T>>, options?: {
    maxAttempts?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    backoffMultiplier?: number;
    shouldRetry?: (error: Error) => boolean;
}): Promise<Result<T>>;
/**
 * Create a successful Result
 */
export declare function ok<T>(data: T): Result<T>;
/**
 * Create a failed Result
 */
export declare function err<E>(error: E): Result<never, E>;
//# sourceMappingURL=result.d.ts.map