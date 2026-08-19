export interface RetryOptions {
  max_retries: number;
  base_delay_ms: number;
  max_delay_ms: number;
  jitter_ms?: number;
  onRetry?: (attempt: number, error: Error) => void;
}

export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: RetryOptions
): Promise<T>;
