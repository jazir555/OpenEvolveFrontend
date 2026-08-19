/**
 * Structured JSON Lines Logger
 *
 * Local copy used by the unified-verification package so the package type-checks
 * and runs self-contained (its tsconfig rootDir is ./src). The implementation
 * mirrors glue/lib/logger.ts; message parameters are typed loosely so both the
 * glue-style calls (info(msg, context)) and the object-style calls used by some
 * components (info({ msg })) type-check.
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

function generateCorrelationId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

export class Logger {
  private serviceName: string;

  constructor(serviceName: string = 'unknown') {
    this.serviceName = serviceName;
  }

  debug(msg: any, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.DEBUG, msg, context);
  }

  info(msg: any, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.INFO, msg, context);
  }

  warn(msg: any, context: LoggerContext = {}): void {
    this.writeLog(LogLevel.WARN, msg, context);
  }

  error(msg: any, error?: any, context: LoggerContext = {}): void {
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

  private writeLog(level: LogLevel, msg: any, context: LoggerContext): void {
    const entry: LogEntry = {
      level,
      msg: typeof msg === 'string' ? msg : JSON.stringify(msg),
      timestamp: new Date().toISOString(),
      correlation_id: context.correlation_id || generateCorrelationId(),
      source_service: context.source_service || this.serviceName,
      ...context,
    };

    delete (entry as any).context;

    console.log(JSON.stringify(entry));
  }

  child(context: LoggerContext): Logger {
    const childLogger = new Logger(this.serviceName);
    const originalWriteLog = childLogger.writeLog.bind(childLogger);

    (childLogger as any).writeLog = (level: LogLevel, msg: any, ctx: LoggerContext) => {
      originalWriteLog(level, msg, { ...context, ...ctx });
    };

    return childLogger;
  }
}

export const logger = new Logger();
