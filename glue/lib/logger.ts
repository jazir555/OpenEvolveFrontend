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

export enum LogLevel {
  DEBUG = 'debug',
  INFO = 'info',
  WARN = 'warn',
  ERROR = 'error',
}

export interface LogEntry {
  level: LogLevel;
  msg: string;
  timestamp: string; // ISO-8601 UTC
  correlation_id: string;
  [key: string]: any;
}

/**
 * Auto-generate a UUID v4 compliant correlation ID
 */
function generateCorrelationId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Structured Logger with JSON Lines output
 *
 * All timestamps are UTC (Law of UTC)
 * All entries include correlation_id (auto-generated if not provided)
 * Output is single-line JSON for log aggregation
 */
export class Logger {
  private serviceName: string;

  constructor(serviceName: string = 'unknown') {
    this.serviceName = serviceName;
  }

  /**
   * Log debug message
   */
  debug(msg: string, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.DEBUG, msg, context);
  }

  /**
   * Log info message
   */
  info(msg: string, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.INFO, msg, context);
  }

  /**
   * Log warning message
   */
  warn(msg: string, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.WARN, msg, context);
  }

  /**
   * Log error message with optional Error object
   */
  error(msg: string, error?: Error, context: LoggerContext = {}): void {
    const errorContext: LoggerContext = {
      ...context,
      ...(error && {
        error_name: error.name,
        error_message: error.message,
        error_stack: error.stack,
      }),
    };
    this.writeLog(LogLevel.ERROR, msg, errorContext);
  }

  /**
   * Write log entry to stdout as JSON line
   */
  private writeLog(level: LogLevel, msg: string, context: LoggerContext): void {
    const entry: LogEntry = {
      level,
      msg,
      timestamp: new Date().toISOString(), // UTC ISO-8601
      correlation_id: context.correlation_id || generateCorrelationId(),
      source_service: context.source_service || this.serviceName,
      ...context,
    };

    // Remove correlation_id from root to avoid duplication
    delete (entry as any).context;

    // Output as single-line JSON
    console.log(JSON.stringify(entry));
  }

  /**
   * Create a child logger with preset context
   */
  child(context: LoggerContext): Logger {
    const childLogger = new Logger(this.serviceName);
    const originalWriteLog = childLogger.writeLog.bind(childLogger);

    (childLogger as any).writeLog = (level: LogLevel, msg: string, ctx: LoggerContext) => {
      originalWriteLog(level, msg, { ...context, ...ctx });
    };

    return childLogger;
  }
}

/**
 * Default logger instance
 */
export const logger = new Logger();

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
