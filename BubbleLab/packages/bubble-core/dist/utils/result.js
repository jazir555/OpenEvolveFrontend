/**
 * Result Type for Error Handling
 *
 * Provides a type-safe way to handle operations that can fail
 * without throwing exceptions. Inspired by Rust's Result<T, E> and
 * functional programming patterns.
 */
/**
 * Check if result is successful
 */
export function isSuccess(result) {
    return result.success === true;
}
/**
 * Check if result is failure
 */
export function isFailure(result) {
    return result.success === false;
}
/**
 * Wrap an async operation in a Result
 */
export async function wrapAsync(operation) {
    try {
        const data = await operation();
        return { success: true, data };
    }
    catch (error) {
        return {
            success: false,
            error: error instanceof Error ? error : new Error(String(error)),
        };
    }
}
/**
 * Wrap a synchronous operation in a Result
 */
export function wrapSync(operation) {
    try {
        const data = operation();
        return { success: true, data };
    }
    catch (error) {
        return {
            success: false,
            error: error instanceof Error ? error : new Error(String(error)),
        };
    }
}
/**
 * Map over the success value of a Result
 */
export function mapResult(result, mapper) {
    if (result.success) {
        return { success: true, data: mapper(result.data) };
    }
    return result;
}
/**
 * Map over the error value of a Result
 */
export function mapError(result, mapper) {
    if (!result.success) {
        return { success: false, error: mapper(result.error) };
    }
    return result;
}
export function chainResult(result, fn) {
    if (result.success) {
        return fn(result.data);
    }
    return result;
}
/**
 * Get data from Result or throw if failure
 */
export function unwrapOrThrow(result) {
    if (result.success) {
        return result.data;
    }
    throw result.error;
}
/**
 * Get data from Result or return default value
 */
export function unwrapOr(result, defaultValue) {
    return result.success ? result.data : defaultValue;
}
/**
 * Get data from Result or compute default value
 */
export function unwrapOrElse(result, fn) {
    return result.success ? result.data : fn(result.error);
}
/**
 * Execute async operations in parallel and collect all Results
 */
export async function all(operations) {
    return Promise.all(operations.map(async (op) => {
        try {
            return await op();
        }
        catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error : new Error(String(error))
            };
        }
    }));
}
/**
 * Execute operations in parallel and return Result with array of data
 * Fails fast on first error
 */
export async function allSuccess(operations) {
    const results = await all(operations);
    for (const result of results) {
        if (!result.success) {
            return result;
        }
    }
    return {
        success: true,
        data: results.map((r) => r.data),
    };
}
/**
 * Retry an operation with exponential backoff
 */
export async function retry(operation, options = {}) {
    const { maxAttempts = 3, initialDelayMs = 1000, maxDelayMs = 10000, backoffMultiplier = 2, shouldRetry = () => true, } = options;
    let lastError;
    let delay = initialDelayMs;
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const result = await wrapAsync(operation);
        if (result.success) {
            return result.data;
        }
        lastError = result.error;
        // Don't retry after last attempt
        if (attempt >= maxAttempts || !shouldRetry(lastError)) {
            break;
        }
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, delay));
        delay = Math.min(delay * backoffMultiplier, maxDelayMs);
    }
    return { success: false, error: lastError || new Error('Operation failed') };
}
/**
 * Create a successful Result
 */
export function ok(data) {
    return { success: true, data };
}
/**
 * Create a failed Result
 */
export function err(error) {
    return { success: false, error };
}
//# sourceMappingURL=result.js.map