/**
 * Trace Logger - Internal logging for the tracing system
 *
 * This module provides logging capabilities for the tracing system itself,
 * separate from application logging.
 */
export declare class TraceLogger {
    private debugMode;
    constructor(debugMode?: boolean);
    debug(message: string, meta?: Record<string, unknown>): void;
    info(message: string, meta?: Record<string, unknown>): void;
    warn(message: string, meta?: Record<string, unknown>): void;
    error(message: string, error?: Error | unknown, meta?: Record<string, unknown>): void;
}
//# sourceMappingURL=trace-logger.d.ts.map