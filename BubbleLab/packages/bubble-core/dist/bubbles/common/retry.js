/**
 * Common retry utilities for Bubble implementations
 * Provides exponential backoff, jitter, and circuit breaker patterns
 */
import { generateCorrelationId } from '../../utils/error-handler.js';
import { TimeoutError, NetworkError } from './error-handlers.js';
/**
 * Default retry configuration
 */
export const DEFAULT_RETRY_OPTIONS = {
    maxAttempts: 3,
    baseDelayMs: 1000,
    maxDelayMs: 30000,
    backoffMultiplier: 2,
    jitter: true,
    jitterAmount: 0.1,
    isRetryable: () => true,
    onRetry: () => { },
    correlationId: generateCorrelationId(),
    operation: 'Retry Operation'
};
/**
 * Calculate delay with exponential backoff and jitter
 * @param attempt - Current attempt number (0-indexed)
 * @param options - Retry options
 * @returns Delay in milliseconds
 */
export function calculateDelay(attempt, options) {
    // Calculate base delay with exponential backoff
    const exponentialDelay = Math.min(options.baseDelayMs * Math.pow(options.backoffMultiplier, attempt), options.maxDelayMs);
    // Add jitter if enabled
    if (options.jitter) {
        const jitterRange = exponentialDelay * options.jitterAmount;
        const jitterValue = (Math.random() - 0.5) * 2 * jitterRange;
        return Math.max(0, exponentialDelay + jitterValue);
    }
    return exponentialDelay;
}
/**
 * Sleep for a specified duration
 * @param ms - Milliseconds to sleep
 * @returns Promise that resolves after sleep duration
 */
export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
/**
 * Retry a function with exponential backoff
 * @param fn - Function to retry
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 * @throws Last error if all retries are exhausted
 */
export async function retryWithBackoff(fn, options = {}) {
    const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
    let lastError;
    for (let attempt = 0; attempt < opts.maxAttempts; attempt++) {
        try {
            return await fn();
        }
        catch (error) {
            lastError = error;
            // Check if error is retryable
            if (!opts.isRetryable(error)) {
                throw error;
            }
            // Don't retry after last attempt
            if (attempt === opts.maxAttempts - 1) {
                break;
            }
            // Calculate delay
            const delay = calculateDelay(attempt, opts);
            // Log retry
            console.log(`[${opts.correlationId}] [${opts.operation}] Attempt ${attempt + 1}/${opts.maxAttempts} failed. Retrying in ${delay.toFixed(0)}ms...`);
            // Call onRetry callback
            opts.onRetry(attempt + 1, error);
            // Sleep before retry
            await sleep(delay);
        }
    }
    throw lastError;
}
/**
 * Retry with a timeout for each attempt
 * @param fn - Function to retry
 * @param timeoutMs - Timeout in milliseconds for each attempt
 * @param options - Retry options
 * @returns Promise that resolves with the function result
 * @throws TimeoutError if timeout is exceeded
 */
export async function retryWithTimeout(fn, timeoutMs, options = {}) {
    return retryWithBackoff(async () => {
        return await withTimeout(fn(), timeoutMs, options.operation || 'Timeout Operation');
    }, options);
}
/**
 * Wrap a promise with a timeout
 * @param promise - Promise to wrap
 * @param timeoutMs - Timeout in milliseconds
 * @param operation - Operation name for error messages
 * @returns Promise that resolves or rejects with TimeoutError
 */
export function withTimeout(promise, timeoutMs, operation = 'Operation') {
    return Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new TimeoutError(`${operation} timed out after ${timeoutMs}ms`, timeoutMs)), timeoutMs))
    ]);
}
/**
 * Circuit breaker state
 */
export var CircuitBreakerState;
(function (CircuitBreakerState) {
    CircuitBreakerState["CLOSED"] = "closed";
    CircuitBreakerState["OPEN"] = "open";
    CircuitBreakerState["HALF_OPEN"] = "half_open"; // Testing if service has recovered
})(CircuitBreakerState || (CircuitBreakerState = {}));
/**
 * Circuit breaker for preventing cascading failures
 */
export class CircuitBreaker {
    options;
    name;
    state = CircuitBreakerState.CLOSED;
    failureCount = 0;
    successCount = 0;
    lastFailureTime;
    openedAt;
    constructor(options = {}, name = 'CircuitBreaker') {
        this.options = options;
        this.name = name;
        this.options = {
            failureThreshold: 5,
            successThreshold: 2,
            timeoutMs: 60000,
            monitoringPeriodMs: 60000,
            ...options
        };
    }
    /**
     * Execute a function with circuit breaker protection
     * @param fn - Function to execute
     * @returns Promise that resolves with the function result
     * @throws Error if circuit is open or function fails
     */
    async execute(fn, operationName = 'Operation') {
        // Check if circuit should transition from OPEN to HALF_OPEN
        if (this.state === CircuitBreakerState.OPEN) {
            if (this.shouldAttemptReset()) {
                this.transitionTo(CircuitBreakerState.HALF_OPEN);
            }
            else {
                throw new NetworkError(`Circuit breaker '${this.name}' is OPEN. Rejecting request for '${operationName}'`);
            }
        }
        try {
            const result = await fn();
            this.onSuccess();
            return result;
        }
        catch (error) {
            this.onFailure();
            throw error;
        }
    }
    /**
     * Handle successful execution
     */
    onSuccess() {
        this.failureCount = 0;
        if (this.state === CircuitBreakerState.HALF_OPEN) {
            this.successCount++;
            if (this.successCount >= this.options.successThreshold) {
                this.transitionTo(CircuitBreakerState.CLOSED);
            }
        }
        else if (this.state === CircuitBreakerState.CLOSED) {
            this.successCount++;
        }
    }
    /**
     * Handle failed execution
     */
    onFailure() {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        this.successCount = 0;
        if (this.state === CircuitBreakerState.HALF_OPEN ||
            this.failureCount >= this.options.failureThreshold) {
            this.transitionTo(CircuitBreakerState.OPEN);
        }
    }
    /**
     * Check if circuit should attempt reset
     */
    shouldAttemptReset() {
        if (!this.openedAt)
            return false;
        return Date.now() - this.openedAt >= this.options.timeoutMs;
    }
    /**
     * Transition to a new state
     */
    transitionTo(newState) {
        const oldState = this.state;
        this.state = newState;
        if (newState === CircuitBreakerState.OPEN) {
            this.openedAt = Date.now();
            this.options.onOpen?.();
            console.warn(`[CircuitBreaker:${this.name}] Transitioned from ${oldState} to OPEN`);
        }
        else if (newState === CircuitBreakerState.HALF_OPEN) {
            this.options.onHalfOpen?.();
            console.log(`[CircuitBreaker:${this.name}] Transitioned from ${oldState} to HALF_OPEN`);
        }
        else if (newState === CircuitBreakerState.CLOSED) {
            this.openedAt = undefined;
            this.options.onClose?.();
            console.log(`[CircuitBreaker:${this.name}] Transitioned from ${oldState} to CLOSED`);
        }
    }
    /**
     * Get current circuit breaker state
     */
    getState() {
        return this.state;
    }
    /**
     * Get circuit breaker stats
     */
    getStats() {
        return {
            state: this.state,
            failureCount: this.failureCount,
            successCount: this.successCount,
            openedAt: this.openedAt
        };
    }
    /**
     * Reset circuit breaker to closed state
     */
    reset() {
        this.state = CircuitBreakerState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
        this.openedAt = undefined;
        console.log(`[CircuitBreaker:${this.name}] Manually reset to CLOSED`);
    }
}
/**
 * Default circuit breaker configuration for external services
 */
export const defaultCircuitBreakerConfig = {
    failureThreshold: 5,
    successThreshold: 2,
    timeoutMs: 60000,
    monitoringPeriodMs: 60000,
    onOpen: () => console.warn('[CircuitBreaker] Circuit opened due to failures'),
    onClose: () => console.log('[CircuitBreaker] Circuit closed - service recovered'),
    onHalfOpen: () => console.log('[CircuitBreaker] Attempting recovery - half-open state')
};
/**
 * Combine retry and circuit breaker patterns
 * @param fn - Function to execute
 * @param circuitBreaker - Circuit breaker instance
 * @param retryOptions - Retry options
 * @returns Promise that resolves with the function result
 */
export async function executeWithResilience(fn, circuitBreaker, retryOptions) {
    return circuitBreaker.execute(() => retryWithBackoff(fn, retryOptions), retryOptions?.operation || 'Resilient Operation');
}
//# sourceMappingURL=retry.js.map