/**
 * Structured JSON Lines Logger
 *
 * Follows the Federation Constitution:
 * - Law of UTC: All timestamps in UTC
 * - Observability: JSON Lines format with correlation_id, source_service, target_service
 */
export interface LoggerContext {
    correlation_id?: string;
    source_service?: string;
    target_service?: string;
    [key: string]: any;
}
export declare enum LogLevel {
    DEBUG = "debug",
    INFO = "info",
    WARN = "warn",
    ERROR = "error"
}
export interface LogEntry {
    level: LogLevel;
    msg: string;
    timestamp: string;
    correlation_id: string;
    [key: string]: any;
}
/**
 * Structured Logger with JSON Lines output
 *
 * All timestamps are UTC (Law of UTC)
 * All entries include correlation_id (auto-generated if not provided)
 * Output is single-line JSON for log aggregation
 */
export declare class Logger {
    private serviceName;
    constructor(serviceName?: string);
    /**
     * Log debug message
     */
    debug(msg: string, context?: LoggerContext): void;
    /**
     * Log info message
     */
    info(msg: string, context?: LoggerContext): void;
    /**
     * Log warning message
     */
    warn(msg: string, context?: LoggerContext): void;
    /**
     * Log error message with optional Error object
     */
    error(msg: string, error?: Error, context?: LoggerContext): void;
    /**
     * Write log entry to stdout as JSON line
     */
    private writeLog;
    /**
     * Create a child logger with preset context
     */
    child(context: LoggerContext): Logger;
}
/**
 * Default logger instance
 */
export declare const logger: Logger;
/**
 * Example usage:
 *
 * ```typescript
 * import { logger } from './logger';
 *
 * // Basic usage
 * logger.info('User Sync Started', {
 *   source_service: 'crm-adapter',
 *   target_service: 'user-service',
 *   user_id: '12345',
 * });
 *
 * // With correlation ID
 * logger.info('Processing event', {
 *   correlation_id: 'evt-abc-123',
 *   event_type: 'user.created',
 * });
 *
 * // Error logging
 * logger.error('User Sync Failed', error, {
 *   correlation_id: 'evt-abc-123',
 *   retry_count: 2,
 * });
 *
 * // Output:
 * {"level":"info","msg":"User Sync Started","timestamp":"2025-01-15T10:30:00.000Z","correlation_id":"a1b2c3d4-...","source_service":"crm-adapter","target_service":"user-service","user_id":"12345"}
 * ```
 */
