export interface LogContext {
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  operation?: string;
  [key: string]: unknown;
}

type LogRecord = Record<string, unknown>;

function toRecord(context?: LogContext): LogRecord {
  return context ? { ...context } : {};
}

export const logger = {
  info(message: string, context?: LogContext): void {
    console.info(message, toRecord(context));
  },
  warn(message: string, context?: LogContext): void {
    console.warn(message, toRecord(context));
  },
  error(message: string, error?: Error, context?: LogContext): void {
    const payload: LogRecord = toRecord(context);
    if (error) {
      payload.error = error.message;
      payload.stack = error.stack;
    }
    console.error(message, payload);
  },
};
