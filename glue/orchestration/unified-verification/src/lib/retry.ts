/**
 * Exponential Backoff with Jitter
 *
 * Local copy used by the unified-verification package (mirrors glue/lib/retry.ts)
 * so the package type-checks and runs self-contained.
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

      if (config.onRetry) {
        config.onRetry(attempt + 1, lastError);
      }

      await sleep(delay);
    }
  }

  logger.error('All retries exhausted', lastError, {
    max_retries: config.max_retries + 1,
  });

  throw lastError;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
