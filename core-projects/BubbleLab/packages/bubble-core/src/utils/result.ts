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
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

/**
 * Check if result is successful
 */
export function isSuccess<T, E>(result: Result<T, E>): result is { success: true; data: T } {
  return result.success === true;
}

/**
 * Check if result is failure
 */
export function isFailure<T, E>(result: Result<T, E>): result is { success: false; error: E } {
  return result.success === false;
}

/**
 * Wrap an async operation in a Result
 */
export async function wrapAsync<T>(
  operation: () => Promise<T>
): Promise<Result<T>> {
  try {
    const data = await operation();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}

/**
 * Wrap a synchronous operation in a Result
 */
export function wrapSync<T>(
  operation: () => T
): Result<T> {
  try {
    const data = operation();
    return { success: true, data };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error : new Error(String(error)),
    };
  }
}

/**
 * Map over the success value of a Result
 */
export function mapResult<T, U, E>(
  result: Result<T, E>,
  mapper: (data: T) => U
): Result<U, E> {
  if (result.success) {
    return { success: true, data: mapper(result.data) };
  }
  return result;
}

/**
 * Map over the error value of a Result
 */
export function mapError<T, E, F>(
  result: Result<T, E>,
  mapper: (error: E) => F
): Result<T, F> {
  if (!result.success) {
    return { success: false, error: mapper(result.error) };
  }
  return result;
}

/**
 * Chain operations that return Results
 */
export async function chainResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => Promise<Result<U, E>>
): Promise<Result<U, E>>;
export function chainResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => Result<U, E>
): Result<U, E>;
export function chainResult<T, U, E>(
  result: Result<T, E>,
  fn: (data: T) => Result<U, E> | Promise<Result<U, E>>
): Result<U, E> | Promise<Result<U, E>> {
  if (result.success) {
    return fn(result.data);
  }
  return result;
}

/**
 * Get data from Result or throw if failure
 */
export function unwrapOrThrow<T, E>(result: Result<T, E>): T {
  if (result.success) {
    return result.data;
  }
  throw result.error;
}

/**
 * Get data from Result or return default value
 */
export function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  return result.success ? result.data : defaultValue;
}

/**
 * Get data from Result or compute default value
 */
export function unwrapOrElse<T, E>(result: Result<T, E>, fn: (error: E) => T): T {
  return result.success ? result.data : fn(result.error);
}

/**
 * Execute async operations in parallel and collect all Results
 */
export async function all<T, E>(
  operations: Array<() => Promise<Result<T, E>>>
): Promise<Array<Result<T, E>>> {
  return Promise.all(operations.map(async (op) => {
    try {
      return await op();
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error as E : new Error(String(error)) as E
      } as Result<T, E>;
    }
  }));
}

/**
 * Execute operations in parallel and return Result with array of data
 * Fails fast on first error
 */
export async function allSuccess<T, E>(
  operations: Array<() => Promise<Result<T, E>>>
): Promise<Result<T[], E>> {
  const results = await all(operations);

  for (const result of results) {
    if (!result.success) {
      return result;
    }
  }

  return {
    success: true,
    data: results.map((r) => (r as { success: true; data: T }).data),
  };
}

/**
 * Retry an operation with exponential backoff
 */
export async function retry<T>(
  operation: () => Promise<Result<T>>,
  options: {
    maxAttempts?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    backoffMultiplier?: number;
    shouldRetry?: (error: Error) => boolean;
  } = {}
): Promise<Result<T>> {
  const {
    maxAttempts = 3,
    initialDelayMs = 1000,
    maxDelayMs = 10000,
    backoffMultiplier = 2,
    shouldRetry = () => true,
  } = options;

  let lastError: Error | undefined;
  let delay = initialDelayMs;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = await wrapAsync(operation);

    if (result.success) {
      return result.data as unknown as Result<T, Error>;
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
export function ok<T>(data: T): Result<T> {
  return { success: true, data };
}

/**
 * Create a failed Result
 */
export function err<E>(error: E): Result<never, E> {
  return { success: false, error };
}
