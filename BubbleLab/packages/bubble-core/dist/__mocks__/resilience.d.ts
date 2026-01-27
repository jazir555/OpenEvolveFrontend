/**
 * Mock Resilience Module for Testing
 *
 * Provides mock implementations of resilience patterns to avoid
 * dependency on external integrations/openevolve module during testing
 */
export declare enum CircuitBreakerState {
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"
}
export interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    halfOpenAttempts: number;
}
export declare class CircuitBreaker {
    constructor(_config: CircuitBreakerConfig);
    private validateConfig;
    execute<T>(_operation: string, fn: () => Promise<T>): Promise<T>;
    onSuccess(): Promise<void>;
    onFailure(): Promise<void>;
    getState(): CircuitBreakerState;
    getStats(): {
        state: CircuitBreakerState;
        failureCount: number;
        successCount: number;
    };
    reset(): void;
}
export interface ResilienceConfig {
    retry?: {
        maxRetries?: number;
        baseDelay?: number;
        maxDelay?: number;
        jitterMultiplier?: number;
    };
    maxRetries?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    timeoutMs?: number;
    circuitBreaker?: CircuitBreakerConfig;
    deduplication?: {
        enabled?: boolean;
        ttl?: number;
        cacheResult?: boolean;
    };
    deadLetterQueue?: {
        enabled?: boolean;
        maxSize?: number;
    };
}
export declare const DEFAULT_RESILIENCE_CONFIG: ResilienceConfig;
export declare class ResilienceWrapper {
    private config;
    private circuitBreaker;
    private deduplicatorStats;
    private deadLetterQueue;
    constructor(config?: ResilienceConfig);
    execute<T>(operation: string, fn: () => Promise<T>, _options?: {
        timeout?: number;
        retries?: number;
    }): Promise<T>;
    getCircuitBreaker(): CircuitBreaker;
    getCircuitBreakerState(): CircuitBreakerState;
    getCircuitBreakerStats(): {
        state: CircuitBreakerState;
        failureCount: number;
        successCount: number;
    };
    resetCircuitBreaker(): Promise<void>;
    getDeduplicatorStats(): {
        totalProcessed: number;
        duplicates: number;
        byKey: {
            [k: string]: number;
        };
    };
    getDeadLetterEntries(): {
        error: string;
        timestamp: number;
        data: any;
    }[];
    clearDeadLetterQueue(): void;
}
export interface RateLimiterConfig {
    requestsPerSecond?: number;
    maxRequests?: number;
    windowMs?: number;
    burstCapacity?: number;
}
export declare class RateLimiter {
    private config;
    private requestCount;
    private lastReset;
    constructor(config: RateLimiterConfig);
    checkLimit(_key: string): boolean;
    acquire(): Promise<boolean>;
    release(): Promise<void>;
    getStats(): {
        available: number;
        used: number;
    };
}
export declare class InputValidator {
    static validateEmail(email: string): boolean;
    static validateURL(url: string): boolean;
    static sanitizeString(input: string, maxLength?: number): string;
    static validateFilePath(filePath: string): boolean;
}
export declare enum LogLevel {
    DEBUG = "debug",
    INFO = "info",
    WARN = "warn",
    ERROR = "error"
}
export interface LogContext {
    correlationId?: string;
    [key: string]: any;
}
export declare class StructuredLogger {
    private context;
    constructor(context?: LogContext);
    debug(_message: string, _meta?: any): void;
    info(_message: string, _meta?: any): void;
    warn(_message: string, _meta?: any): void;
    error(_message: string, _error?: Error | any, _meta?: any): void;
    withContext(context: LogContext): StructuredLogger;
    static create(context?: LogContext): StructuredLogger;
}
export declare function sanitizeError(error: any): Error;
export declare function generateCorrelationId(): string;
export declare function isTransientError(error: unknown): boolean;
export declare function createMockResilienceWrapper(config?: ResilienceConfig): ResilienceWrapper;
export declare function createMockCircuitBreaker(config?: CircuitBreakerConfig): CircuitBreaker;
export declare function createMockRateLimiter(config?: RateLimiterConfig): RateLimiter;
export declare function createMockLogger(context?: LogContext): StructuredLogger;
//# sourceMappingURL=resilience.d.ts.map