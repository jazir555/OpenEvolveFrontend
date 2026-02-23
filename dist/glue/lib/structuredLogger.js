"use strict";
/**
 * Structured JSON Logger - Compliance with CLAUDE.md Section 3.3
 * Implements JSON Lines logging with correlation_id, source_service, target_service
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.StructuredLogger = exports.leanaideLogger = exports.mitosisLogger = exports.ragbitsLogger = exports.apiLogger = exports.logger = void 0;
class StructuredLogger {
    constructor(serviceName, minLevel = 'info') {
        this.serviceName = serviceName;
        this.minLevel = minLevel;
        this.correlationIdGenerator = () => {
            return `cid-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
        };
    }
    shouldLog(level) {
        const levels = {
            debug: 0,
            info: 1,
            warn: 2,
            error: 3
        };
        return levels[level] >= levels[this.minLevel];
    }
    formatTimestamp() {
        // Law of UTC: All timestamps in UTC ISO-8601
        return new Date().toISOString();
    }
    createLogEntry(level, message, context, error) {
        const entry = {
            timestamp: this.formatTimestamp(),
            level,
            msg: message,
            source_service: this.serviceName
        };
        // Add context if provided
        if (context) {
            if (context.correlation_id) {
                entry.correlation_id = context.correlation_id;
            }
            else {
                entry.correlation_id = this.correlationIdGenerator();
            }
            if (context.source_service) {
                entry.source_service = context.source_service;
            }
            if (context.target_service) {
                entry.target_service = context.target_service;
            }
            // Merge other context properties
            Object.keys(context).forEach(key => {
                if (!['correlation_id', 'source_service', 'target_service'].includes(key)) {
                    entry[key] = context[key];
                }
            });
        }
        // Add error details if provided
        if (error) {
            entry.error = {
                message: error.message,
                stack: error.stack,
                code: error.code
            };
        }
        return entry;
    }
    log(entry) {
        if (!this.shouldLog(entry.level)) {
            return;
        }
        // Output as JSON line (JSONL format)
        const jsonLine = JSON.stringify(entry);
        // Map levels to console methods for backwards compatibility
        const consoleMethod = {
            debug: console.debug,
            info: console.info,
            warn: console.warn,
            error: console.error
        }[entry.level] || console.log;
        consoleMethod(jsonLine);
    }
    debug(message, context) {
        this.log(this.createLogEntry('debug', message, context));
    }
    info(message, context) {
        this.log(this.createLogEntry('info', message, context));
    }
    warn(message, context) {
        this.log(this.createLogEntry('warn', message, context));
    }
    error(message, error, context) {
        this.log(this.createLogEntry('error', message, context, error));
    }
    // Create child logger with inherited context
    child(additionalContext) {
        const childLogger = new StructuredLogger(this.serviceName, this.minLevel);
        childLogger.correlationIdGenerator = () => {
            return additionalContext.correlation_id || this.correlationIdGenerator();
        };
        return childLogger;
    }
    setMinLevel(level) {
        this.minLevel = level;
    }
}
exports.StructuredLogger = StructuredLogger;
// Export default instances for common services
exports.logger = new StructuredLogger('frontend-service');
exports.apiLogger = new StructuredLogger('frontend-api');
exports.ragbitsLogger = new StructuredLogger('ragbits-plugin');
exports.mitosisLogger = new StructuredLogger('mitosis-plugin');
exports.leanaideLogger = new StructuredLogger('leanaide-plugin');
//# sourceMappingURL=structuredLogger.js.map