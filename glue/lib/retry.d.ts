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
export declare function retryWithBackoff<T>(fn: () => Promise<T>, options: RetryConfig): Promise<T>;
//# sourceMappingURL=retry.d.ts.map