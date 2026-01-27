/**
 * Mock Resilience Module for Testing
 *
 * Provides mock implementations of resilience patterns to avoid
 * dependency on external integrations/openevolve module during testing
 */
// ============================================================================
// CIRCUIT BREAKER
// ============================================================================
export var CircuitBreakerState;
(function (CircuitBreakerState) {
    CircuitBreakerState["CLOSED"] = "closed";
    CircuitBreakerState["OPEN"] = "open";
    CircuitBreakerState["HALF_OPEN"] = "half_open";
})(CircuitBreakerState || (CircuitBreakerState = {}));
export class CircuitBreaker {
    constructor(_config) {
        this.validateConfig();
    }
    validateConfig() {
        // Mock validation
    }
    async execute(_operation, fn) {
        return await fn();
    }
    async onSuccess() {
        // Mock success handler
    }
    async onFailure() {
        // Mock failure handler
    }
    getState() {
        return CircuitBreakerState.CLOSED;
    }
    getStats() {
        return {
            state: CircuitBreakerState.CLOSED,
            failureCount: 0,
            successCount: 0,
        };
    }
    reset() {
        // Mock reset
    }
}
export const DEFAULT_RESILIENCE_CONFIG = {
    maxRetries: 3,
    initialDelayMs: 1000,
    maxDelayMs: 10000,
    timeoutMs: 30000,
    circuitBreaker: {
        failureThreshold: 5,
        successThreshold: 2,
        timeout: 60000,
        halfOpenAttempts: 3,
    },
};
export class ResilienceWrapper {
    config;
    circuitBreaker;
    deduplicatorStats = new Map();
    deadLetterQueue = [];
    constructor(config = {}) {
        this.config = { ...DEFAULT_RESILIENCE_CONFIG, ...config };
        this.circuitBreaker = new CircuitBreaker(this.config.circuitBreaker || DEFAULT_RESILIENCE_CONFIG.circuitBreaker);
    }
    async execute(operation, fn, _options) {
        return await this.circuitBreaker.execute(operation, fn);
    }
    getCircuitBreaker() {
        return this.circuitBreaker;
    }
    getCircuitBreakerState() {
        return this.circuitBreaker.getState();
    }
    getCircuitBreakerStats() {
        return this.circuitBreaker.getStats();
    }
    async resetCircuitBreaker() {
        this.circuitBreaker.reset();
    }
    getDeduplicatorStats() {
        return {
            totalProcessed: Array.from(this.deduplicatorStats.values()).reduce((a, b) => a + b, 0),
            duplicates: this.deduplicatorStats.get('duplicates') || 0,
            byKey: Object.fromEntries(this.deduplicatorStats),
        };
    }
    getDeadLetterEntries() {
        return [...this.deadLetterQueue];
    }
    clearDeadLetterQueue() {
        this.deadLetterQueue = [];
    }
}
export class RateLimiter {
    config;
    requestCount = 0;
    lastReset = Date.now();
    constructor(config) {
        this.config = config;
        // Mock rate limiter
    }
    checkLimit(_key) {
        const now = Date.now();
        const elapsed = now - this.lastReset;
        // Reset counter every second or window
        const windowMs = this.config.windowMs || 1000;
        if (elapsed >= windowMs) {
            this.requestCount = 0;
            this.lastReset = now;
        }
        const maxRequests = this.config.maxRequests || this.config.requestsPerSecond || 5;
        if (this.requestCount < maxRequests) {
            this.requestCount++;
            return true;
        }
        return false;
    }
    async acquire() {
        return true;
    }
    async release() {
        // Mock release
    }
    getStats() {
        const maxRequests = this.config.maxRequests || this.config.requestsPerSecond || 5;
        return {
            available: maxRequests,
            used: 0,
        };
    }
}
// ============================================================================
// INPUT VALIDATOR
// ============================================================================
export class InputValidator {
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    static validateURL(url) {
        try {
            new URL(url);
            return true;
        }
        catch {
            return false;
        }
    }
    static sanitizeString(input, maxLength = 1000) {
        if (!input)
            return '';
        return input.substring(0, maxLength).trim();
    }
    static validateFilePath(filePath) {
        // Prevent path traversal
        const normalized = filePath.replace(/\\/g, '/');
        return !normalized.includes('..') && !normalized.includes('~');
    }
}
// ============================================================================
// STRUCTURED LOGGER
// ============================================================================
export var LogLevel;
(function (LogLevel) {
    LogLevel["DEBUG"] = "debug";
    LogLevel["INFO"] = "info";
    LogLevel["WARN"] = "warn";
    LogLevel["ERROR"] = "error";
})(LogLevel || (LogLevel = {}));
export class StructuredLogger {
    context;
    constructor(context = {}) {
        this.context = context;
    }
    debug(_message, _meta) {
        // Mock debug log
    }
    info(_message, _meta) {
        // Mock info log
    }
    warn(_message, _meta) {
        // Mock warn log
    }
    error(_message, _error, _meta) {
        // Mock error log
    }
    withContext(context) {
        return new StructuredLogger({ ...this.context, ...context });
    }
    static create(context) {
        return new StructuredLogger(context);
    }
}
// ============================================================================
// ERROR SANITIZATION
// ============================================================================
export function sanitizeError(error) {
    if (error instanceof Error) {
        return error;
    }
    if (typeof error === 'string') {
        return new Error(error);
    }
    if (error && typeof error === 'object') {
        return new Error(error.message || JSON.stringify(error));
    }
    return new Error('Unknown error');
}
// ============================================================================
// CORRELATION ID GENERATOR
// ============================================================================
export function generateCorrelationId() {
    return `test-${Date.now()}-${Math.random().toString(36).substring(7)}`;
}
// ============================================================================
// ERROR CLASSIFICATION
// ============================================================================
export function isTransientError(error) {
    if (error instanceof Error) {
        const errorMessage = error.message.toLowerCase();
        // Network errors
        if (errorMessage.includes('ECONNREFUSED') ||
            errorMessage.includes('ECONNRESET') ||
            errorMessage.includes('ETIMEDOUT') ||
            errorMessage.includes('ENOTFOUND') ||
            errorMessage.includes('EAI_AGAIN')) {
            return true;
        }
        // HTTP errors that might be transient
        if (errorMessage.includes('503') || // Service Unavailable
            errorMessage.includes('502') || // Bad Gateway
            errorMessage.includes('504') || // Gateway Timeout
            errorMessage.includes('429') // Too Many Requests
        ) {
            return true;
        }
        // Specific error messages
        if (errorMessage.includes('timeout') ||
            errorMessage.includes('temporary') ||
            errorMessage.includes('transient') ||
            errorMessage.includes('rate limit')) {
            return true;
        }
    }
    return false;
}
// ============================================================================
// MOCK FACTORY
// ============================================================================
export function createMockResilienceWrapper(config) {
    return new ResilienceWrapper(config);
}
export function createMockCircuitBreaker(config) {
    return new CircuitBreaker(config || DEFAULT_RESILIENCE_CONFIG.circuitBreaker);
}
export function createMockRateLimiter(config) {
    return new RateLimiter(config || { requestsPerSecond: 10 });
}
export function createMockLogger(context) {
    return StructuredLogger.create(context);
}
//# sourceMappingURL=resilience.js.map