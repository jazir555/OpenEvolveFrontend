"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tracer = void 0;
exports.getTracer = getTracer;
exports.resetTracer = resetTracer;
exports.traceMethod = traceMethod;
exports.generateServiceMap = generateServiceMap;
const api_1 = require("@opentelemetry/api");
const logger_1 = require("../logger");
class Tracer {
    constructor(serviceName) {
        this.serviceName = serviceName;
        this.logger = new logger_1.Logger(serviceName);
        this.tracer = api_1.trace.getTracer(serviceName, '1.0.0');
        this.logger.info('Tracer initialized', {
            service: serviceName,
        });
    }
    startSpan(options) {
        const span = this.tracer.startSpan(options.name, {
            kind: options.kind || api_1.SpanKind.INTERNAL,
            attributes: {
                ...options.attributes,
                'service.name': this.serviceName,
                'correlation.id': options.correlationId,
            },
        });
        return span;
    }
    async traceAsync(options, fn) {
        const span = this.startSpan(options);
        try {
            const result = await api_1.context.with(api_1.trace.setSpan(api_1.context.active(), span), async () => {
                return await fn(span);
            });
            span.setStatus({
                code: api_1.SpanStatusCode.OK,
            });
            return result;
        }
        catch (error) {
            span.recordException(error);
            span.setStatus({
                code: api_1.SpanStatusCode.ERROR,
                message: error instanceof Error ? error.message : 'Unknown error',
            });
            this.logger.error('Span execution failed', error, {
                span_name: options.name,
                correlation_id: options.correlationId,
            });
            throw error;
        }
        finally {
            span.end();
        }
    }
    trace(options, fn) {
        const span = this.startSpan(options);
        try {
            const result = api_1.context.with(api_1.trace.setSpan(api_1.context.active(), span), () => {
                return fn(span);
            });
            span.setStatus({
                code: api_1.SpanStatusCode.OK,
            });
            return result;
        }
        catch (error) {
            span.recordException(error);
            span.setStatus({
                code: api_1.SpanStatusCode.ERROR,
                message: error instanceof Error ? error.message : 'Unknown error',
            });
            this.logger.error('Span execution failed', error, {
                span_name: options.name,
                correlation_id: options.correlationId,
            });
            throw error;
        }
        finally {
            span.end();
        }
    }
    getCurrentSpan() {
        return api_1.trace.getSpan(api_1.context.active());
    }
    addEvent(name, attributes) {
        const span = this.getCurrentSpan();
        if (span) {
            span.addEvent(name, attributes);
        }
    }
    setAttributes(attributes) {
        const span = this.getCurrentSpan();
        if (span) {
            span.setAttributes(attributes);
        }
    }
    getCorrelationId() {
        const span = this.getCurrentSpan();
        if (span) {
            return span.attributes['correlation.id'];
        }
        return undefined;
    }
    injectContext(carrier) {
        const span = this.getCurrentSpan();
        if (span) {
            carrier['traceparent'] = span.spanContext().traceId;
        }
    }
    extractContext(carrier) {
        return api_1.context.active();
    }
    createChildSpan(parentSpan, name, attributes) {
        const childSpan = this.tracer.startSpan(name, {
            kind: api_1.SpanKind.INTERNAL,
            attributes: {
                ...attributes,
                'service.name': this.serviceName,
                'parent.id': parentSpan.spanContext().spanId,
            },
        });
        return childSpan;
    }
    async traceHttpRequest(options) {
        const { method, url, headers, correlationId, fn } = options;
        return this.traceAsync({
            name: `HTTP ${method}`,
            kind: api_1.SpanKind.CLIENT,
            attributes: {
                'http.method': method,
                'http.url': url,
                'http.headers': JSON.stringify(headers || {}),
            },
            correlationId,
        }, async (span) => {
            const start = Date.now();
            try {
                const result = await fn();
                const duration = Date.now() - start;
                span.setAttributes({
                    'http.status_code': 200,
                    'http.response_time_ms': duration,
                });
                return result;
            }
            catch (error) {
                const duration = Date.now() - start;
                span.setAttributes({
                    'http.status_code': 500,
                    'http.response_time_ms': duration,
                    'error.message': error instanceof Error ? error.message : 'Unknown error',
                });
                throw error;
            }
        });
    }
    async traceDatabaseOperation(options) {
        const { operation, table, query, correlationId, fn } = options;
        return this.traceAsync({
            name: `DB ${operation}`,
            kind: api_1.SpanKind.CLIENT,
            attributes: {
                'db.operation': operation,
                'db.table': table,
                'db.query': query,
            },
            correlationId,
        }, async (span) => {
            const start = Date.now();
            try {
                const result = await fn();
                const duration = Date.now() - start;
                span.setAttributes({
                    'db.duration_ms': duration,
                    'db.status': 'success',
                });
                return result;
            }
            catch (error) {
                const duration = Date.now() - start;
                span.setAttributes({
                    'db.duration_ms': duration,
                    'db.status': 'error',
                    'error.message': error instanceof Error ? error.message : 'Unknown error',
                });
                throw error;
            }
        });
    }
    async traceKnowledgeExtraction(options) {
        const { source, method, entityCount, relationCount, correlationId, fn } = options;
        return this.traceAsync({
            name: `Knowledge Extraction: ${method}`,
            kind: api_1.SpanKind.INTERNAL,
            attributes: {
                'extraction.source': source,
                'extraction.method': method,
            },
            correlationId,
        }, async (span) => {
            const start = Date.now();
            try {
                const result = await fn();
                const duration = Date.now() - start;
                span.setAttributes({
                    'extraction.duration_ms': duration,
                    'extraction.entity_count': entityCount || 0,
                    'extraction.relation_count': relationCount || 0,
                    'extraction.status': 'success',
                });
                return result;
            }
            catch (error) {
                const duration = Date.now() - start;
                span.setAttributes({
                    'extraction.duration_ms': duration,
                    'extraction.status': 'error',
                    'error.message': error instanceof Error ? error.message : 'Unknown error',
                });
                throw error;
            }
        });
    }
}
exports.Tracer = Tracer;
let globalTracer = null;
function getTracer(serviceName) {
    if (!globalTracer) {
        const name = serviceName || process.env.SERVICE_NAME || 'unknown-service';
        globalTracer = new Tracer(name);
    }
    return globalTracer;
}
function resetTracer() {
    globalTracer = null;
}
function traceMethod(options) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function (...args) {
            const tracer = getTracer();
            const name = options.name || `${target.constructor.name}.${propertyKey}`;
            return tracer.traceAsync({
                name,
                kind: options.kind,
                attributes: {
                    ...options.attributes,
                    method: propertyKey,
                    class: target.constructor.name,
                },
            }, async (span) => {
                return originalMethod.apply(this, args);
            });
        };
        return descriptor;
    };
}
function generateServiceMap(spans) {
    const nodes = new Map();
    for (const span of spans) {
        const serviceName = span.attributes['service.name'] || 'unknown';
        const operation = span.name || 'unknown';
        if (!nodes.has(serviceName)) {
            nodes.set(serviceName, {
                service: serviceName,
                operations: [],
                dependencies: [],
                span_count: 0,
                error_count: 0,
                avg_duration_ms: 0,
            });
        }
        const node = nodes.get(serviceName);
        node.operations.push(operation);
        node.span_count++;
        if (span.status.code === api_1.SpanStatusCode.ERROR) {
            node.error_count++;
        }
        const duration = span.duration / 1000000;
        node.avg_duration_ms = (node.avg_duration_ms * (node.span_count - 1) + duration) / node.span_count;
    }
    const rootService = Array.from(nodes.keys())[0];
    return nodes.get(rootService) || {
        service: 'unknown',
        operations: [],
        dependencies: [],
        span_count: 0,
        error_count: 0,
        avg_duration_ms: 0,
    };
}
//# sourceMappingURL=tracer.js.map