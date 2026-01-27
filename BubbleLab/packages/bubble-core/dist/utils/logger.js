/**
 * Structured Logging Utility for BubbleLab Bubbles
 *
 * Provides consistent, structured logging across all bubbles with:
 * - Log levels (DEBUG, INFO, WARN, ERROR)
 * - Structured JSON output
 * - Contextual metadata
 * - Correlation tracking
 */
export var LogLevel;
(function (LogLevel) {
    LogLevel[LogLevel["DEBUG"] = 0] = "DEBUG";
    LogLevel[LogLevel["INFO"] = 1] = "INFO";
    LogLevel[LogLevel["WARN"] = 2] = "WARN";
    LogLevel[LogLevel["ERROR"] = 3] = "ERROR";
})(LogLevel || (LogLevel = {}));
/**
 * Logger class for structured logging
 */
export class Logger {
    context;
    minLevel;
    constructor(context, minLevel = Logger.getLogLevelFromEnv()) {
        this.context = context;
        this.minLevel = minLevel;
    }
    /**
     * Log debug message (only in development)
     */
    debug(message, meta) {
        if (process.env.NODE_ENV === 'development') {
            this.log(LogLevel.DEBUG, message, meta);
        }
    }
    /**
     * Log informational message
     */
    info(message, meta) {
        this.log(LogLevel.INFO, message, meta);
    }
    /**
     * Log warning message
     */
    warn(message, meta) {
        this.log(LogLevel.WARN, message, meta);
    }
    /**
     * Log error with optional error object
     */
    error(message, error, meta) {
        const errorMeta = {
            ...meta,
            error: error instanceof Error
                ? {
                    name: error.name,
                    message: error.message,
                    stack: error.stack,
                    ...error.code ? { code: error.code } : {},
                }
                : error,
        };
        this.log(LogLevel.ERROR, message, errorMeta);
    }
    /**
     * Create child logger with additional context
     */
    child(additionalContext) {
        const newContext = `${this.context}:${additionalContext}`;
        return new Logger(newContext, this.minLevel);
    }
    /**
     * Log with timing
     */
    async time(operation, fn, meta) {
        const startTime = Date.now();
        this.info(`Starting: ${operation}`, meta);
        try {
            const result = await fn();
            const duration = Date.now() - startTime;
            this.info(`Completed: ${operation}`, { ...meta, duration_ms: duration });
            return result;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            this.error(`Failed: ${operation}`, error, { ...meta, duration_ms: duration });
            throw error;
        }
    }
    /**
     * Internal log method
     */
    log(level, message, meta) {
        if (level < this.minLevel) {
            return;
        }
        const logEntry = {
            timestamp: new Date().toISOString(),
            level: LogLevel[level],
            context: this.context,
            message,
            ...meta,
        };
        const output = JSON.stringify(logEntry);
        // Output to appropriate stream
        switch (level) {
            case LogLevel.ERROR:
                console.error(output);
                break;
            case LogLevel.WARN:
                console.warn(output);
                break;
            case LogLevel.DEBUG:
                console.debug(output);
                break;
            default:
                console.log(output);
        }
    }
    /**
     * Get log level from environment variable
     */
    static getLogLevelFromEnv() {
        const envLevel = process.env.LOG_LEVEL?.toUpperCase();
        switch (envLevel) {
            case 'DEBUG':
                return LogLevel.DEBUG;
            case 'INFO':
                return LogLevel.INFO;
            case 'WARN':
                return LogLevel.WARN;
            case 'ERROR':
                return LogLevel.ERROR;
            default:
                return process.env.NODE_ENV === 'production' ? LogLevel.INFO : LogLevel.DEBUG;
        }
    }
}
/**
 * Create a logger instance
 */
export function createLogger(context) {
    return new Logger(context);
}
/**
 * Global logger for usage without context
 */
export const globalLogger = new Logger('Global');
/**
 * Utility to generate correlation IDs
 */
export function generateCorrelationId() {
    return `corr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
//# sourceMappingURL=logger.js.map