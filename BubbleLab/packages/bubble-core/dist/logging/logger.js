/**
 * Structured Logging Infrastructure for BubbleLab
 *
 * Provides centralized, structured logging with correlation ID tracking,
 * log levels, and multiple transport options
 */
import winston from 'winston';
import { ElasticsearchTransport } from 'winston-elasticsearch';
// ============================================================================
// LOG FORMATS
// ============================================================================
/**
 * JSON log format for production
 */
const jsonFormat = winston.format.combine(winston.format.timestamp({ format: 'ISO-8601' }), winston.format.errors({ stack: true }), winston.format.metadata({ key: 'context' }), winston.format.json());
/**
 * Pretty print format for development
 */
const prettyFormat = winston.format.combine(winston.format.colorize(), winston.format.timestamp({ format: 'HH:mm:ss' }), winston.format.printf(({ timestamp, level, message, context, ...metadata }) => {
    let msg = `${timestamp} [${level}]: ${message}`;
    if (context && Object.keys(context).length > 0) {
        msg += ` ${JSON.stringify(context)}`;
    }
    if (Object.keys(metadata).length > 0) {
        msg += ` ${JSON.stringify(metadata)}`;
    }
    return msg;
}));
/**
 * Express middleware to inject correlation ID into requests
 */
export function correlationIdMiddleware(req, res, next) {
    // Check for existing correlation ID in headers
    const correlationId = req.headers['x-correlation-id'] ||
        req.headers['x-request-id'] ||
        generateCorrelationId();
    // Attach to request
    req.headers['x-correlation-id'] = correlationId;
    req.headers['x-request-id'] = correlationId;
    // Also attach to response
    res.setHeader('x-correlation-id', correlationId);
    next();
}
/**
 * Generate a unique correlation ID
 */
function generateCorrelationId() {
    return `cid_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;
}
// ============================================================================
// LOGGER CLASS
// ============================================================================
class BubbleLogger {
    logger;
    correlationId;
    constructor(options) {
        const { level = process.env.LOG_LEVEL || 'info', environment = process.env.NODE_ENV || 'development', elasticsearchUrl, elasticsearchIndex = 'bubblelab-logs' } = options || {};
        const transports = [];
        // Console transport
        transports.push(new winston.transports.Console({
            format: environment === 'production' ? jsonFormat : prettyFormat
        }));
        // File transport (all logs)
        transports.push(new winston.transports.File({
            filename: 'logs/combined.log',
            format: jsonFormat,
            level: 'info'
        }));
        // File transport (errors only)
        transports.push(new winston.transports.File({
            filename: 'logs/error.log',
            format: jsonFormat,
            level: 'error'
        }));
        // Elasticsearch transport (optional)
        if (elasticsearchUrl) {
            transports.push(new ElasticsearchTransport({
                level: 'info',
                clientOpts: {
                    node: elasticsearchUrl
                },
                index: elasticsearchIndex,
                dataStream: true
            }));
        }
        this.logger = winston.createLogger({
            level,
            transports,
            format: jsonFormat,
            exitOnError: false
        });
    }
    /**
     * Set correlation ID for logger instance
     */
    setCorrelationId(correlationId) {
        this.correlationId = correlationId;
    }
    /**
     * Get correlation ID
     */
    getCorrelationId() {
        return this.correlationId;
    }
    /**
     * Log info message
     */
    info(message, context) {
        this.logger.info(message, {
            ...context,
            correlation_id: this.correlationId || context?.correlation_id
        });
    }
    /**
     * Log warning message
     */
    warn(message, context) {
        this.logger.warn(message, {
            ...context,
            correlation_id: this.correlationId || context?.correlation_id
        });
    }
    /**
     * Log error message
     */
    error(message, error, context) {
        const errorContext = {
            ...context,
            correlation_id: this.correlationId || context?.correlation_id
        };
        if (error instanceof Error) {
            this.logger.error(message, {
                ...errorContext,
                error: {
                    name: error.name,
                    message: error.message,
                    stack: error.stack
                }
            });
        }
        else {
            this.logger.error(message, {
                ...errorContext,
                error: String(error)
            });
        }
    }
    /**
     * Log debug message
     */
    debug(message, context) {
        this.logger.debug(message, {
            ...context,
            correlation_id: this.correlationId || context?.correlation_id
        });
    }
    /**
     * Log operation with duration
     */
    logOperation(bubble, operation, durationMs, status, context) {
        this.info(`Operation completed: ${operation}`, {
            bubble,
            operation,
            duration_ms: durationMs,
            status,
            ...context
        });
    }
    /**
     * Create a child logger with additional default context
     */
    child(defaultContext) {
        const childLogger = new BubbleLogger();
        childLogger.logger = this.logger.child(defaultContext);
        childLogger.correlationId = this.correlationId;
        return childLogger;
    }
}
// ============================================================================
// LOGGER FACTORY
// ============================================================================
const loggers = new Map();
/**
 * Get or create a logger instance
 */
export function getLogger(name, options) {
    const loggerName = name || 'default';
    if (!loggers.has(loggerName)) {
        const logger = new BubbleLogger(options);
        loggers.set(loggerName, logger);
    }
    return loggers.get(loggerName);
}
/**
 * Get logger with bubble context
 */
export function getBubbleLogger(bubbleName) {
    return getLogger().child({ bubble: bubbleName });
}
/**
 * Create logger for a specific request
 */
export function createRequestLogger(correlationId, context) {
    const logger = new BubbleLogger();
    logger.setCorrelationId(correlationId);
    if (context) {
        return logger.child(context);
    }
    return logger;
}
// ============================================================================
// REQUEST LOGGING MIDDLEWARE
// ============================================================================
/**
 * Express middleware to log HTTP requests
 */
export function requestLoggingMiddleware(logger = getLogger()) {
    return (req, res, next) => {
        const start = Date.now();
        const correlationId = req.headers['x-correlation-id'];
        // Create request-specific logger
        const requestLogger = createRequestLogger(correlationId, {
            bubble: 'http-server',
            operation: `${req.method} ${req.path}`,
            user_id: req.headers['x-user-id']
        });
        // Attach logger to request
        req.logger = requestLogger;
        // Log request
        requestLogger.info('Incoming request', {
            method: req.method,
            path: req.path,
            query: req.query,
            ip: req.ip,
            user_agent: req.get('user-agent')
        });
        // Log response
        res.on('finish', () => {
            const duration = Date.now() - start;
            const status = res.statusCode < 400 ? 'success' : 'error';
            requestLogger.logOperation('http-server', `${req.method} ${req.path}`, duration, status, {
                status_code: res.statusCode,
                method: req.method,
                path: req.path
            });
        });
        next();
    };
}
// ============================================================================
// EXPORTS
// ============================================================================
export { BubbleLogger };
export default getLogger;
//# sourceMappingURL=logger.js.map