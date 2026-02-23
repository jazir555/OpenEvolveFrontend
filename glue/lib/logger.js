"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.logger = exports.Logger = exports.LogLevel = void 0;
var LogLevel;
(function (LogLevel) {
    LogLevel["DEBUG"] = "debug";
    LogLevel["INFO"] = "info";
    LogLevel["WARN"] = "warn";
    LogLevel["ERROR"] = "error";
})(LogLevel || (exports.LogLevel = LogLevel = {}));
function generateCorrelationId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
class Logger {
    constructor(serviceName = 'unknown') {
        this.serviceName = serviceName;
    }
    debug(msg, context = {}) {
        this.writeLog(LogLevel.DEBUG, msg, context);
    }
    info(msg, context = {}) {
        this.writeLog(LogLevel.INFO, msg, context);
    }
    warn(msg, context = {}) {
        this.writeLog(LogLevel.WARN, msg, context);
    }
    error(msg, error, context = {}) {
        const errorContext = {
            ...context,
            ...(error && {
                error_name: error.name,
                error_message: error.message,
                error_stack: error.stack,
            }),
        };
        this.writeLog(LogLevel.ERROR, msg, errorContext);
    }
    writeLog(level, msg, context) {
        const entry = {
            level,
            msg,
            timestamp: new Date().toISOString(),
            correlation_id: context.correlation_id || generateCorrelationId(),
            source_service: context.source_service || this.serviceName,
            ...context,
        };
        delete entry.context;
        console.log(JSON.stringify(entry));
    }
    child(context) {
        const childLogger = new Logger(this.serviceName);
        const originalWriteLog = childLogger.writeLog.bind(childLogger);
        childLogger.writeLog = (level, msg, ctx) => {
            originalWriteLog(level, msg, { ...context, ...ctx });
        };
        return childLogger;
    }
}
exports.Logger = Logger;
exports.logger = new Logger();
//# sourceMappingURL=logger.js.map