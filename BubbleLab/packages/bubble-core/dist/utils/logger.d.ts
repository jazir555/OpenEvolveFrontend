/**
 * Structured Logging Utility for BubbleLab Bubbles
 *
 * Provides consistent, structured logging across all bubbles with:
 * - Log levels (DEBUG, INFO, WARN, ERROR)
 * - Structured JSON output
 * - Contextual metadata
 * - Correlation tracking
 */
export declare enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
}
export interface LogContext {
    correlation_id?: string;
    bubble_id?: string;
    bubble_type?: string;
    operation?: string;
    user_id?: string;
    request_id?: string;
    duration_ms?: number;
    [key: string]: unknown;
}
export interface LogEntry {
    timestamp: string;
    level: string;
    context: string;
    message: string;
    error?: {
        name: string;
        message: string;
        stack?: string;
    };
    [key: string]: unknown;
}
/**
 * Logger class for structured logging
 */
export declare class Logger {
    private context;
    private minLevel;
    constructor(context: string, minLevel?: LogLevel);
    /**
     * Log debug message (only in development)
     */
    debug(message: string, meta?: LogContext): void;
    /**
     * Log informational message
     */
    info(message: string, meta?: LogContext): void;
    /**
     * Log warning message
     */
    warn(message: string, meta?: LogContext): void;
    /**
     * Log error with optional error object
     */
    error(message: string, error?: Error | unknown, meta?: LogContext): void;
    /**
     * Create child logger with additional context
     */
    child(additionalContext: string): Logger;
    /**
     * Log with timing
     */
    time<T>(operation: string, fn: () => Promise<T>, meta?: LogContext): Promise<T>;
    /**
     * Internal log method
     */
    private log;
    /**
     * Get log level from environment variable
     */
    private static getLogLevelFromEnv;
}
/**
 * Create a logger instance
 */
export declare function createLogger(context: string): Logger;
/**
 * Global logger for usage without context
 */
export declare const globalLogger: Logger;
/**
 * Utility to generate correlation IDs
 */
export declare function generateCorrelationId(): string;
//# sourceMappingURL=logger.d.ts.map