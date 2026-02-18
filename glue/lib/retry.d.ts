/**
 * Exponential Backoff with Jitter
 *
 * Follows the Federation Constitution:
 * - Failure Management: Transient failures get exponential backoff with jitter
 * - Law of Configuration Explicitness: All timeouts configurable
 */
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
export declare function retryWithBackoff<T>(fn: () => Promise<T>, options: RetryConfig): Promise<T>;
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
//# sourceMappingURL=retry.d.ts.map