/**
 * Structured Logging Utility for BubbleLab Bubbles
 *
 * Provides consistent, structured logging across all bubbles with:
 * - Log levels (DEBUG, INFO, WARN, ERROR)
 * - Structured JSON output
 * - Contextual metadata
 * - Correlation tracking
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
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
export class Logger {
  constructor(
    private context: string,
    private minLevel: LogLevel = Logger.getLogLevelFromEnv()
  ) {}

  /**
   * Log debug message (only in development)
   */
  debug(message: string, meta?: LogContext): void {
    if (process.env.NODE_ENV === 'development') {
      this.log(LogLevel.DEBUG, message, meta);
    }
  }

  /**
   * Log informational message
   */
  info(message: string, meta?: LogContext): void {
    this.log(LogLevel.INFO, message, meta);
  }

  /**
   * Log warning message
   */
  warn(message: string, meta?: LogContext): void {
    this.log(LogLevel.WARN, message, meta);
  }

  /**
   * Log error with optional error object
   */
  error(message: string, error?: Error | unknown, meta?: LogContext): void {
    const errorMeta = {
      ...meta,
      error: error instanceof Error
        ? {
            name: error.name,
            message: error.message,
            stack: error.stack,
            ...(error as any).code ? { code: (error as any).code } : {},
          }
        : error,
    };
    this.log(LogLevel.ERROR, message, errorMeta);
  }

  /**
   * Create child logger with additional context
   */
  child(additionalContext: string): Logger {
    const newContext = `${this.context}:${additionalContext}`;
    return new Logger(newContext, this.minLevel);
  }

  /**
   * Log with timing
   */
  async time<T>(
    operation: string,
    fn: () => Promise<T>,
    meta?: LogContext
  ): Promise<T> {
    const startTime = Date.now();
    this.info(`Starting: ${operation}`, meta);

    try {
      const result = await fn();
      const duration = Date.now() - startTime;
      this.info(`Completed: ${operation}`, { ...meta, duration_ms: duration });
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.error(`Failed: ${operation}`, error, { ...meta, duration_ms: duration });
      throw error;
    }
  }

  /**
   * Internal log method
   */
  private log(level: LogLevel, message: string, meta?: LogContext): void {
    if (level < this.minLevel) {
      return;
    }

    const logEntry: LogEntry = {
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
  private static getLogLevelFromEnv(): LogLevel {
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
export function createLogger(context: string): Logger {
  return new Logger(context);
}

/**
 * Global logger for usage without context
 */
export const globalLogger = new Logger('Global');

/**
 * Utility to generate correlation IDs
 */
export function generateCorrelationId(): string {
  return `corr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}
