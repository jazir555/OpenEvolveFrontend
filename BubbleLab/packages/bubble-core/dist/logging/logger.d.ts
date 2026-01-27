/**
 * Structured Logging Infrastructure for BubbleLab
 *
 * Provides centralized, structured logging with correlation ID tracking,
 * log levels, and multiple transport options
 */
export interface LogContext {
    correlation_id?: string;
    bubble?: string;
    operation?: string;
    user_id?: string;
    duration_ms?: number;
    status?: string;
    error_code?: string;
    [key: string]: unknown;
}
export interface LogMetadata {
    timestamp: string;
    level: string;
    message: string;
    context: LogContext;
}
import { Request, Response, NextFunction } from 'express';
/**
 * Express middleware to inject correlation ID into requests
 */
export declare function correlationIdMiddleware(req: Request, res: Response, next: NextFunction): void;
declare class BubbleLogger {
    private logger;
    private correlationId;
    constructor(options?: {
        level?: string;
        environment?: 'development' | 'production';
        elasticsearchUrl?: string;
        elasticsearchIndex?: string;
    });
    /**
     * Set correlation ID for logger instance
     */
    setCorrelationId(correlationId: string): void;
    /**
     * Get correlation ID
     */
    getCorrelationId(): string | undefined;
    /**
     * Log info message
     */
    info(message: string, context?: LogContext): void;
    /**
     * Log warning message
     */
    warn(message: string, context?: LogContext): void;
    /**
     * Log error message
     */
    error(message: string, error?: Error | unknown, context?: LogContext): void;
    /**
     * Log debug message
     */
    debug(message: string, context?: LogContext): void;
    /**
     * Log operation with duration
     */
    logOperation(bubble: string, operation: string, durationMs: number, status: 'success' | 'error', context?: LogContext): void;
    /**
     * Create a child logger with additional default context
     */
    child(defaultContext: LogContext): BubbleLogger;
}
/**
 * Get or create a logger instance
 */
export declare function getLogger(name?: string, options?: {
    level?: string;
    environment?: 'development' | 'production';
    elasticsearchUrl?: string;
    elasticsearchIndex?: string;
}): BubbleLogger;
/**
 * Get logger with bubble context
 */
export declare function getBubbleLogger(bubbleName: string): BubbleLogger;
/**
 * Create logger for a specific request
 */
export declare function createRequestLogger(correlationId: string, context?: LogContext): BubbleLogger;
/**
 * Express middleware to log HTTP requests
 */
export declare function requestLoggingMiddleware(logger?: BubbleLogger): (req: Request, res: Response, next: NextFunction) => void;
export { BubbleLogger };
export default getLogger;
//# sourceMappingURL=logger.d.ts.map