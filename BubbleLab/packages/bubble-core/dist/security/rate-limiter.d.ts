/**
 * Rate Limiter
 * Provides rate limiting for API calls and operations
 */
interface RateLimitConfig {
    maxRequests: number;
    windowMs: number;
}
export declare class RateLimitExceededError extends Error {
    readonly retryAfter: number;
    constructor(message: string, retryAfter: number);
}
export declare class RateLimiter {
    private config;
    private requests;
    constructor(config: RateLimitConfig);
    /**
     * Check if the request is within rate limits
     * @param identifier Unique identifier for the request source (e.g., API key, user ID)
     * @throws RateLimitExceededError if rate limit is exceeded
     */
    checkLimit(identifier?: string): Promise<void>;
    /**
     * Reset rate limit for a specific identifier
     * @param identifier Unique identifier for the request source
     */
    reset(identifier?: string): void;
    /**
     * Clear all rate limit records
     */
    clearAll(): void;
    /**
     * Get remaining requests for an identifier
     * @param identifier Unique identifier for the request source
     * @returns Number of remaining requests
     */
    getRemainingRequests(identifier?: string): number;
}
/**
 * Default rate limiter: 60 requests per minute
 */
export declare const defaultRateLimiter: RateLimiter;
/**
 * Aggressive rate limiter: 10 requests per second
 */
export declare const aggressiveRateLimiter: RateLimiter;
export {};
//# sourceMappingURL=rate-limiter.d.ts.map