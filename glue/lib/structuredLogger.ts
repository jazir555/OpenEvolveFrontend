/**
 * Structured JSON Logger - Compliance with CLAUDE.md Section 3.3
 * Implements JSON Lines logging with correlation_id, source_service, target_service
 */

export interface LogContext {
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  user_id?: string;
  workflow_id?: string;
  [key: string]: any;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string; // UTC ISO-8601
  level: LogLevel;
  msg: string;
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  error?: {
    message: string;
    stack?: string;
    code?: string;
  };
  [key: string]: any;
}

class StructuredLogger {
  private serviceName: string;
  private minLevel: LogLevel;
  private correlationIdGenerator: () => string;

  constructor(serviceName: string, minLevel: LogLevel = 'info') {
    this.serviceName = serviceName;
    this.minLevel = minLevel;
    this.correlationIdGenerator = () => {
      return `cid-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    };
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: Record<LogLevel, number> = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3
    };
    return levels[level] >= levels[this.minLevel];
  }

  private formatTimestamp(): string {
    // Law of UTC: All timestamps in UTC ISO-8601
    return new Date().toISOString();
  }

  private createLogEntry(
    level: LogLevel,
    message: string,
    context?: LogContext,
    error?: Error
  ): LogEntry {
    const entry: LogEntry = {
      timestamp: this.formatTimestamp(),
      level,
      msg: message,
      source_service: this.serviceName
    };

    // Add context if provided
    if (context) {
      if (context.correlation_id) {
        entry.correlation_id = context.correlation_id;
      } else {
        entry.correlation_id = this.correlationIdGenerator();
      }
      if (context.source_service) {
        entry.source_service = context.source_service;
      }
      if (context.target_service) {
        entry.target_service = context.target_service;
      }
      // Merge other context properties
      Object.keys(context).forEach(key => {
        if (!['correlation_id', 'source_service', 'target_service'].includes(key)) {
          entry[key] = context[key];
        }
      });
    }

    // Add error details if provided
    if (error) {
      entry.error = {
        message: error.message,
        stack: error.stack,
        code: (error as any).code
      };
    }

    return entry;
  }

  private log(entry: LogEntry): void {
    if (!this.shouldLog(entry.level)) {
      return;
    }

    // Output as JSON line (JSONL format)
    const jsonLine = JSON.stringify(entry);

    // Map levels to console methods for backwards compatibility
    const consoleMethod = {
      debug: console.debug,
      info: console.info,
      warn: console.warn,
      error: console.error
    }[entry.level] || console.log;

    consoleMethod(jsonLine);
  }

  debug(message: string, context?: LogContext): void {
    this.log(this.createLogEntry('debug', message, context));
  }

  info(message: string, context?: LogContext): void {
    this.log(this.createLogEntry('info', message, context));
  }

  warn(message: string, context?: LogContext): void {
    this.log(this.createLogEntry('warn', message, context));
  }

  error(message: string, error?: Error, context?: LogContext): void {
    this.log(this.createLogEntry('error', message, context, error));
  }

  // Create child logger with inherited context
  child(additionalContext: LogContext): StructuredLogger {
    const childLogger = new StructuredLogger(this.serviceName, this.minLevel);
    childLogger.correlationIdGenerator = () => {
      return additionalContext.correlation_id || this.correlationIdGenerator();
    };
    return childLogger;
  }

  setMinLevel(level: LogLevel): void {
    this.minLevel = level;
  }
}

// Export default instances for common services
export const logger = new StructuredLogger('frontend-service');
export const apiLogger = new StructuredLogger('frontend-api');
export const ragbitsLogger = new StructuredLogger('ragbits-plugin');
export const mitosisLogger = new StructuredLogger('mitosis-plugin');
export const leanaideLogger = new StructuredLogger('leanaide-plugin');

export { StructuredLogger };
