/**
 * Rate Limiter
 * Provides rate limiting for API calls and operations
 */

interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
}

interface RequestRecord {
  count: number;
  resetTime: number;
}

export class RateLimitExceededError extends Error {
  constructor(
    message: string,
    public readonly retryAfter: number
  ) {
    super(message);
    this.name = 'RateLimitExceededError';
  }
}

export class RateLimiter {
  private requests: Map<string, RequestRecord> = new Map();

  constructor(private config: RateLimitConfig) {
    this.config = config;
  }

  /**
   * Check if the request is within rate limits
   * @param identifier Unique identifier for the request source (e.g., API key, user ID)
   * @throws RateLimitExceededError if rate limit is exceeded
   */
  async checkLimit(identifier: string = 'default'): Promise<void> {
    const now = Date.now();
    const record = this.requests.get(identifier);

    if (!record || now > record.resetTime) {
      // Create new window
      this.requests.set(identifier, {
        count: 1,
        resetTime: now + this.config.windowMs,
      });
      return;
    }

    if (record.count >= this.config.maxRequests) {
      const retryAfter = Math.ceil((record.resetTime - now) / 1000);
      throw new RateLimitExceededError(
        `Rate limit exceeded. Max ${this.config.maxRequests} requests per ${this.config.windowMs}ms`,
        retryAfter
      );
    }

    record.count++;
  }

  /**
   * Reset rate limit for a specific identifier
   * @param identifier Unique identifier for the request source
   */
  reset(identifier: string = 'default'): void {
    this.requests.delete(identifier);
  }

  /**
   * Clear all rate limit records
   */
  clearAll(): void {
    this.requests.clear();
  }

  /**
   * Get remaining requests for an identifier
   * @param identifier Unique identifier for the request source
   * @returns Number of remaining requests
   */
  getRemainingRequests(identifier: string = 'default'): number {
    const record = this.requests.get(identifier);
    const now = Date.now();

    if (!record || now > record.resetTime) {
      return this.config.maxRequests;
    }

    return Math.max(0, this.config.maxRequests - record.count);
  }
}

/**
 * Default rate limiter: 60 requests per minute
 */
export const defaultRateLimiter = new RateLimiter({
  maxRequests: 60,
  windowMs: 60000,
});

/**
 * Aggressive rate limiter: 10 requests per second
 */
export const aggressiveRateLimiter = new RateLimiter({
  maxRequests: 10,
  windowMs: 1000,
});
