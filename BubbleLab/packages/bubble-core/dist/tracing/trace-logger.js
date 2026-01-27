/**
 * Trace Logger - Internal logging for the tracing system
 *
 * This module provides logging capabilities for the tracing system itself,
 * separate from application logging.
 */
export class TraceLogger {
    debugMode;
    constructor(debugMode = false) {
        this.debugMode = debugMode || process.env.OTEL_DEBUG === 'true';
    }
    debug(message, meta) {
        if (this.debugMode) {
            console.debug(JSON.stringify({
                level: 'debug',
                message: `[OpenTelemetry] ${message}`,
                ...meta,
                timestamp: new Date().toISOString(),
            }));
        }
    }
    info(message, meta) {
        console.info(JSON.stringify({
            level: 'info',
            message: `[OpenTelemetry] ${message}`,
            ...meta,
            timestamp: new Date().toISOString(),
        }));
    }
    warn(message, meta) {
        console.warn(JSON.stringify({
            level: 'warn',
            message: `[OpenTelemetry] ${message}`,
            ...meta,
            timestamp: new Date().toISOString(),
        }));
    }
    error(message, error, meta) {
        console.error(JSON.stringify({
            level: 'error',
            message: `[OpenTelemetry] ${message}`,
            error: error instanceof Error ? {
                message: error.message,
                stack: error.stack,
                name: error.name,
            } : error,
            ...meta,
            timestamp: new Date().toISOString(),
        }));
    }
}
//# sourceMappingURL=trace-logger.js.map