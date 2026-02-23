export interface LogContext {
    correlation_id?: string;
    source_service?: string;
    target_service?: string;
    operation?: string;
    [key: string]: unknown;
}
export declare const logger: {
    info(message: string, context?: LogContext): void;
    warn(message: string, context?: LogContext): void;
    error(message: string, error?: Error, context?: LogContext): void;
};
//# sourceMappingURL=structuredLogger.d.ts.map