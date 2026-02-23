"use strict";
/**
 * Structured JSON Lines Logger
 *
 * Follows the Federation Constitution:
 * - Law of UTC: All timestamps in UTC
 * - Observability: JSON Lines format with correlation_id, source_service, target_service
 */
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.logger = exports.Logger = exports.LogLevel = void 0;
var LogLevel;
(function (LogLevel) {
    LogLevel["DEBUG"] = "debug";
    LogLevel["INFO"] = "info";
    LogLevel["WARN"] = "warn";
    LogLevel["ERROR"] = "error";
})(LogLevel || (exports.LogLevel = LogLevel = {}));
/**
 * Auto-generate a UUID v4 compliant correlation ID
 */
function generateCorrelationId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0;
        var v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
/**
 * Structured Logger with JSON Lines output
 *
 * All timestamps are UTC (Law of UTC)
 * All entries include correlation_id (auto-generated if not provided)
 * Output is single-line JSON for log aggregation
 */
var Logger = /** @class */ (function () {
    function Logger(serviceName) {
        if (serviceName === void 0) { serviceName = 'unknown'; }
        this.serviceName = serviceName;
    }
    /**
     * Log debug message
     */
    Logger.prototype.debug = function (msg, context) {
        if (context === void 0) { context = {}; }
        this.writeLog(LogLevel.DEBUG, msg, context);
    };
    /**
     * Log info message
     */
    Logger.prototype.info = function (msg, context) {
        if (context === void 0) { context = {}; }
        this.writeLog(LogLevel.INFO, msg, context);
    };
    /**
     * Log warning message
     */
    Logger.prototype.warn = function (msg, context) {
        if (context === void 0) { context = {}; }
        this.writeLog(LogLevel.WARN, msg, context);
    };
    /**
     * Log error message with optional Error object
     */
    Logger.prototype.error = function (msg, error, context) {
        if (context === void 0) { context = {}; }
        var errorContext = __assign(__assign({}, context), (error && {
            error_name: error.name,
            error_message: error.message,
            error_stack: error.stack,
        }));
        this.writeLog(LogLevel.ERROR, msg, errorContext);
    };
    /**
     * Write log entry to stdout as JSON line
     */
    Logger.prototype.writeLog = function (level, msg, context) {
        var entry = __assign({ level: level, msg: msg, timestamp: new Date().toISOString(), correlation_id: context.correlation_id || generateCorrelationId(), source_service: context.source_service || this.serviceName }, context);
        // Remove correlation_id from root to avoid duplication
        delete entry.context;
        // Output as single-line JSON
        console.log(JSON.stringify(entry));
    };
    /**
     * Create a child logger with preset context
     */
    Logger.prototype.child = function (context) {
        var childLogger = new Logger(this.serviceName);
        var originalWriteLog = childLogger.writeLog.bind(childLogger);
        childLogger.writeLog = function (level, msg, ctx) {
            originalWriteLog(level, msg, __assign(__assign({}, context), ctx));
        };
        return childLogger;
    };
    return Logger;
}());
exports.Logger = Logger;
/**
 * Default logger instance
 */
exports.logger = new Logger();
/**
 * Example usage:
 *
 * ```typescript
 * import { logger } from './logger';
 *
 * // Basic usage
 * logger.info('User Sync Started', {
 *   source_service: 'crm-adapter',
 *   target_service: 'user-service',
 *   user_id: '12345',
 * });
 *
 * // With correlation ID
 * logger.info('Processing event', {
 *   correlation_id: 'evt-abc-123',
 *   event_type: 'user.created',
 * });
 *
 * // Error logging
 * logger.error('User Sync Failed', error, {
 *   correlation_id: 'evt-abc-123',
 *   retry_count: 2,
 * });
 *
 * // Output:
 * {"level":"info","msg":"User Sync Started","timestamp":"2025-01-15T10:30:00.000Z","correlation_id":"a1b2c3d4-...","source_service":"crm-adapter","target_service":"user-service","user_id":"12345"}
 * ```
 */
