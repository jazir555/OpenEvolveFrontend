/**
 * Exponential Backoff with Jitter
 *
 * Follows the Federation Constitution:
 * - Failure Management: Transient failures get exponential backoff with jitter
 * - Law of Configuration Explicitness: All timeouts configurable
 */

import { logger } from './logger';

export interface RetryOptions {
  max_retries: number;
  base_delay_ms: number;
  max_delay_ms: number;
  jitter_ms: number;
  onRetry?: (attempt: number, error: Error) => void;
}

export interface RetryConfig extends Partial<RetryOptions> {
  max_retries: number;
}

const DEFAULT_RETRY_OPTIONS: Omit<Required<RetryOptions>, 'onRetry'> & {
  onRetry?: (attempt: number, error: Error) => void;
} = {
  max_retries: 3,
  base_delay_ms: 1000,
  max_delay_ms: 30000,
  jitter_ms: 500,
  onRetry: undefined,
};

/**
 * Calculate delay with exponential backoff and jitter
 *
 * Formula: min(base_delay * 2^attempt + random_jitter, max_delay)
 */
function calculateDelay(
  attempt: number,
  base_delay_ms: number,
  max_delay_ms: number,
  jitter_ms: number
): number {
  const exponentialDelay = base_delay_ms * Math.pow(2, attempt);
  const jitter = Math.random() * jitter_ms;
  const delay = exponentialDelay + jitter;

  return Math.min(delay, max_delay_ms);
}

/**
 * Retry a function with exponential backoff and jitter
 *
 * Use this for transient failures (network blips, temporary timeouts)
 * Do not use for logic failures (bad data should go to DLQ)
 *
 * @param fn - Async function to retry
 * @param options - Retry configuration
 * @returns Result of successful function execution
 * @throws Last error if all retries exhausted
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: RetryConfig
): Promise<T> {
  const config: Omit<Required<RetryOptions>, 'onRetry'> & {
    onRetry?: (attempt: number, error: Error) => void;
  } = {
    ...DEFAULT_RETRY_OPTIONS,
    ...options,
  };

  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= config.max_retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // If this was the last attempt, don't delay
      if (attempt === config.max_retries) {
        break;
      }

      const delay = calculateDelay(
        attempt,
        config.base_delay_ms,
        config.max_delay_ms,
        config.jitter_ms
      );

      logger.warn('Retrying after error', {
        attempt: attempt + 1,
        max_retries: config.max_retries + 1,
        delay_ms: Math.round(delay),
        error_name: lastError.name,
        error_message: lastError.message,
      });

      // Call onRetry callback if provided
      if (config.onRetry) {
        config.onRetry(attempt + 1, lastError);
      }

      // Wait before retrying
      await sleep(delay);
    }
  }

  // All retries exhausted
  logger.error('All retries exhausted', lastError, {
    max_retries: config.max_retries + 1,
  });

  throw lastError;
}

/**
 * Sleep for specified milliseconds
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Example usage:
 *
 * ```typescript
 * import { retryWithBackoff } from './retry';
 *
 * // Simple retry with defaults
 * const result = await retryWithBackoff(
 *   async () => {
 *     const response = await fetch('http://service:8000/api');
 *     if (!response.ok) throw new Error('HTTP error');
 *     return response.json();
 *   },
 *   { max_retries: 3 }
 * );
 *
 * // Custom retry configuration
 * const data = await retryWithBackoff(
 *   async () => {
 *     return await apiClient.getData();
 *   },
 *   {
 *     max_retries: 5,
 *     base_delay_ms: 500,
 *     max_delay_ms: 10000,
 *     jitter_ms: 200,
 *     onRetry: (attempt, error) => {
 *       console.log(`Attempt ${attempt} failed: ${error.message}`);
 *     },
 *   }
 * );
 *
 * // Retry with correlation ID
 * const result = await retryWithBackoff(
 *   async () => {
 *     logger.info('Calling external API', {
 *       correlation_id: ctx.id,
 *       target_service: 'external-api',
 *     });
 *     return await externalApiCall();
 *   },
 *   { max_retries: 3 }
 * );
 * ```
 */
