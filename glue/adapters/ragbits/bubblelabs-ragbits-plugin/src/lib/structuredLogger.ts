export interface LogContext {
  correlation_id?: string;
  source_service?: string;
  target_service?: string;
  operation?: string;
  [key: string]: unknown;
}

function asPayload(context?: LogContext, error?: Error): Record<string, unknown> {
  const payload: Record<string, unknown> = context ? { ...context } : {};
  if (error) {
    payload.error = error.message;
    payload.stack = error.stack;
  }
  return payload;
}

export const ragbitsLogger = {
  info(message: string, context?: LogContext): void {
    console.info(message, asPayload(context));
  },
  warn(message: string, context?: LogContext): void {
    console.warn(message, asPayload(context));
  },
  error(message: string, error?: Error, context?: LogContext): void {
    console.error(message, asPayload(context, error));
  },
};
