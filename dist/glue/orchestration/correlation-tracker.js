"use strict";
/**
 * Correlation Tracker - Distributed Tracing & Request Lineage
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Observability: Correlation ID propagation across all services
 * - UTC timestamps only
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.correlationTracker = exports.CorrelationTracker = void 0;
exports.createCorrelationMiddleware = createCorrelationMiddleware;
const uuid_1 = require("uuid");
const logger_1 = require("../lib/logger");
/**
 * Correlation Tracker for distributed tracing
 *
 * Generates UUID v4 correlation IDs and tracks request lineage across services
 */
class CorrelationTracker {
    constructor() {
        this.traces = new Map();
        this.logger = new logger_1.Logger('correlation-tracker');
    }
    /**
     * Generate a new UUID v4 correlation ID
     */
    generateCorrelationId() {
        return (0, uuid_1.v4)();
    }
    /**
     * Generate a new trace ID for distributed tracing
     */
    generateTraceId() {
        return (0, uuid_1.v4)();
    }
    /**
     * Create a new correlation context
     */
    createContext(metadata = {}) {
        const correlationId = this.generateCorrelationId();
        const traceId = this.generateTraceId();
        return {
            correlation_id: correlationId,
            trace_id: traceId,
            service_path: [],
            start_time: new Date().toISOString(),
            metadata
        };
    }
    /**
     * Create a child context from existing correlation ID
     */
    createChildContext(parentContext, serviceName) {
        return {
            correlation_id: parentContext.correlation_id,
            trace_id: parentContext.trace_id,
            parent_id: parentContext.parent_id,
            service_path: [...parentContext.service_path],
            start_time: parentContext.start_time,
            metadata: { ...parentContext.metadata }
        };
    }
    /**
     * Record a service call in the correlation context
     */
    recordServiceCall(context, serviceName, operation) {
        const call = {
            service: serviceName,
            timestamp: new Date().toISOString(),
            operation
        };
        context.service_path.push(call);
        this.logger.debug('Service call recorded', {
            correlation_id: context.correlation_id,
            service: serviceName,
            operation
        });
    }
    /**
     * Create a distributed tracing span
     */
    createSpan(traceId, parentSpanId, serviceName, operationName, tags = {}) {
        const span = {
            span_id: (0, uuid_1.v4)(),
            parent_span_id: parentSpanId,
            trace_id: traceId,
            service_name: serviceName,
            operation_name: operationName,
            start_time: Date.now(),
            tags,
            status: 'ok'
        };
        // Store span in trace
        if (!this.traces.has(traceId)) {
            this.traces.set(traceId, []);
        }
        this.traces.get(traceId).push(span);
        return span;
    }
    /**
     * Complete a distributed tracing span
     */
    completeSpan(span, status = 'ok') {
        span.end_time = Date.now();
        span.status = status;
        if (status === 'error') {
            this.logger.error('Span completed with error', undefined, {
                trace_id: span.trace_id,
                span_id: span.span_id,
                service: span.service_name,
                operation: span.operation_name
            });
        }
    }
    /**
     * Get trace by ID
     */
    getTrace(traceId) {
        return this.traces.get(traceId);
    }
    /**
     * Calculate total duration of a correlation context
     */
    calculateDuration(context) {
        const startTime = new Date(context.start_time).getTime();
        const endTime = Date.now();
        return endTime - startTime;
    }
    /**
     * Format correlation context for logging
     */
    formatForLogging(context) {
        return {
            correlation_id: context.correlation_id,
            trace_id: context.trace_id,
            parent_id: context.parent_id,
            service_path_count: context.service_path.length,
            duration_ms: this.calculateDuration(context)
        };
    }
    /**
     * Extract correlation ID from headers or generate new one
     */
    extractOrGenerate(headers) {
        // Check various header names for correlation ID
        const correlationId = headers['x-correlation-id'] ||
            headers['x-request-id'] ||
            headers['correlation-id'] ||
            undefined;
        return correlationId || this.generateCorrelationId();
    }
    /**
     * Inject correlation ID into headers
     */
    injectIntoHeaders(context) {
        return {
            'x-correlation-id': context.correlation_id,
            'x-trace-id': context.trace_id || '',
            'x-parent-id': context.parent_id || ''
        };
    }
    /**
     * Cleanup old traces to prevent memory leaks
     */
    cleanup(maxAgeMs = 3600000) {
        const now = Date.now();
        const cleaned = [];
        for (const [traceId, spans] of this.traces.entries()) {
            const oldestSpan = spans[0];
            if (now - oldestSpan.start_time > maxAgeMs) {
                this.traces.delete(traceId);
                cleaned.push(traceId);
            }
        }
        if (cleaned.length > 0) {
            this.logger.info('Cleaned up old traces', {
                traces_cleaned: cleaned.length
            });
        }
    }
}
exports.CorrelationTracker = CorrelationTracker;
/**
 * Singleton instance
 */
exports.correlationTracker = new CorrelationTracker();
/**
 * Middleware helper for Express/fastify style request handlers
 */
function createCorrelationMiddleware(tracker) {
    return (req, res, next) => {
        const correlationId = tracker.extractOrGenerate(req.headers);
        const context = tracker.createContext();
        // Attach to request
        req.correlation = context;
        // Inject into response headers
        res.setHeader('x-correlation-id', context.correlation_id);
        next();
    };
}
/**
 * Example usage:
 *
 * ```typescript
 * import { correlationTracker } from './correlation-tracker';
 *
 * // Generate new correlation ID
 * const context = correlationTracker.createContext({
 *   user_id: '12345',
 *   workflow: 'rag-pipeline'
 * });
 *
 * // Record service calls
 * correlationTracker.recordServiceCall(context, 'ragbits-adapter', 'extract-chunks');
 * correlationTracker.recordServiceCall(context, 'vector-db-adapter', 'index-embeddings');
 *
 * // Create distributed trace span
 * const span = correlationTracker.createSpan(
 *   context.trace_id!,
 *   undefined,
 *   'orchestration',
 *   'process-document',
 *   { document_id: 'doc-123' }
 * );
 *
 * // Do work...
 *
 * // Complete span
 * correlationTracker.completeSpan(span, 'ok');
 *
 * // Log with correlation
 * logger.info('Document processed', {
 *   correlation_id: context.correlation_id,
 *   duration_ms: correlationTracker.calculateDuration(context)
 * });
 * ```
 */
//# sourceMappingURL=correlation-tracker.js.map