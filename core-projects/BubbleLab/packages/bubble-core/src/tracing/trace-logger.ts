/**
 * Trace Logger - Internal logging for the tracing system
 *
 * This module provides logging capabilities for the tracing system itself,
 * separate from application logging.
 */

export class TraceLogger {
  private debugMode: boolean;

  constructor(debugMode: boolean = false) {
    this.debugMode = debugMode || process.env.OTEL_DEBUG === 'true';
  }

  debug(message: string, meta?: Record<string, unknown>): void {
    if (this.debugMode) {
      console.debug(
        JSON.stringify({
          level: 'debug',
          message: `[OpenTelemetry] ${message}`,
          ...meta,
          timestamp: new Date().toISOString(),
        })
      );
    }
  }

  info(message: string, meta?: Record<string, unknown>): void {
    console.info(
      JSON.stringify({
        level: 'info',
        message: `[OpenTelemetry] ${message}`,
        ...meta,
        timestamp: new Date().toISOString(),
      })
    );
  }

  warn(message: string, meta?: Record<string, unknown>): void {
    console.warn(
      JSON.stringify({
        level: 'warn',
        message: `[OpenTelemetry] ${message}`,
        ...meta,
        timestamp: new Date().toISOString(),
      })
    );
  }

  error(message: string, error?: Error | unknown, meta?: Record<string, unknown>): void {
    console.error(
      JSON.stringify({
        level: 'error',
        message: `[OpenTelemetry] ${message}`,
        error: error instanceof Error ? {
          message: error.message,
          stack: error.stack,
          name: error.name,
        } : error,
        ...meta,
        timestamp: new Date().toISOString(),
      })
    );
  }
}
