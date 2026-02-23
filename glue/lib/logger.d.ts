export interface LoggerContext {
    correlation_id?: string;
    source_service?: string;
    target_service?: string;
    [key: string]: any;
}
export declare enum LogLevel {
    DEBUG = "debug",
    INFO = "info",
    WARN = "warn",
    ERROR = "error"
}
export interface LogEntry {
    level: LogLevel;
    msg: string;
    timestamp: string;
    correlation_id: string;
    [key: string]: any;
}
export declare class Logger {
    private serviceName;
    constructor(serviceName?: string);
    debug(msg: string, context?: LoggerContext): void;
    info(msg: string, context?: LoggerContext): void;
    warn(msg: string, context?: LoggerContext): void;
    error(msg: string, error?: Error, context?: LoggerContext): void;
    private writeLog;
    child(context: LoggerContext): Logger;
}
export declare const logger: Logger;
//# sourceMappingURL=logger.d.ts.map