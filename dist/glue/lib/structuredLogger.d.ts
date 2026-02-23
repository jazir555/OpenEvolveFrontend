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
    timestamp: string;
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
declare class StructuredLogger {
    private serviceName;
    private minLevel;
    private correlationIdGenerator;
    constructor(serviceName: string, minLevel?: LogLevel);
    private shouldLog;
    private formatTimestamp;
    private createLogEntry;
    private log;
    debug(message: string, context?: LogContext): void;
    info(message: string, context?: LogContext): void;
    warn(message: string, context?: LogContext): void;
    error(message: string, error?: Error, context?: LogContext): void;
    child(additionalContext: LogContext): StructuredLogger;
    setMinLevel(level: LogLevel): void;
}
export declare const logger: StructuredLogger;
export declare const apiLogger: StructuredLogger;
export declare const ragbitsLogger: StructuredLogger;
export declare const mitosisLogger: StructuredLogger;
export declare const leanaideLogger: StructuredLogger;
export { StructuredLogger };
//# sourceMappingURL=structuredLogger.d.ts.map