/**
 * BubbleTracer - Specialized tracing for Bubble operations
 *
 * This module provides bubble-specific tracing utilities that automatically
 * create appropriate spans with bubble-specific attributes.
 */
import { trace, context, SpanStatusCode } from '@opentelemetry/api';
import { TracingManager } from './tracing-manager.js';
/**
 * Standardized span names for bubble operations
 */
export var BubbleSpanName;
(function (BubbleSpanName) {
    // Lifecycle
    BubbleSpanName["BUBBLE_INSTANTIATION"] = "bubble.instantiation";
    BubbleSpanName["BUBBLE_VALIDATION"] = "bubble.validation";
    BubbleSpanName["BUBBLE_EXECUTION"] = "bubble.execution";
    // External operations
    BubbleSpanName["HTTP_REQUEST"] = "bubble.http.request";
    BubbleSpanName["DATABASE_QUERY"] = "bubble.database.query";
    BubbleSpanName["API_CALL"] = "bubble.api.call";
    // Workflow operations
    BubbleSpanName["WORKFLOW_START"] = "workflow.start";
    BubbleSpanName["WORKFLOW_END"] = "workflow.end";
    BubbleSpanName["WORKFLOW_STEP"] = "workflow.step";
    // Tool operations
    BubbleSpanName["TOOL_EXECUTION"] = "tool.execution";
    BubbleSpanName["TOOL_VALIDATION"] = "tool.validation";
})(BubbleSpanName || (BubbleSpanName = {}));
/**
 * Create a standardized bubble span
 */
export class BubbleTracer {
    manager;
    serviceName;
    constructor(serviceName = 'bubble-lab') {
        this.manager = TracingManager.getInstance();
        this.serviceName = serviceName;
    }
    /**
     * Create a span for bubble instantiation
     */
    createInstantiationSpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.BUBBLE_INSTANTIATION, {
            attributes: {
                'bubble.name': attributes.bubbleName,
                'bubble.type': attributes.bubbleType,
                'bubble.variable_name': attributes.variableName,
                'bubble.class_name': attributes.className,
                'operation.phase': 'instantiation',
            },
        });
        return span;
    }
    /**
     * Create a span for bubble validation
     */
    createValidationSpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.BUBBLE_VALIDATION, {
            attributes: {
                'bubble.name': attributes.bubbleName,
                'bubble.type': attributes.bubbleType,
                'validation.phase': attributes.validationPhase,
                'operation.phase': 'validation',
            },
        });
        return span;
    }
    /**
     * Create a span for bubble execution
     */
    createExecutionSpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.BUBBLE_EXECUTION, {
            attributes: {
                'bubble.name': attributes.bubbleName,
                'bubble.type': attributes.bubbleType,
                'bubble.operation': attributes.operation,
                'correlation.id': attributes.correlationId,
                'execution.id': attributes.executionId,
                'operation.phase': 'execution',
            },
        });
        return span;
    }
    /**
     * Create a span for HTTP requests
     */
    createHTTPRequestSpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.HTTP_REQUEST, {
            attributes: {
                'http.method': attributes.method,
                'http.url': attributes.url,
                'bubble.name': attributes.bubbleName,
            },
        });
        return span;
    }
    /**
     * Create a span for database queries
     */
    createDatabaseQuerySpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.DATABASE_QUERY, {
            attributes: {
                'db.system': attributes.dbSystem,
                'db.name': attributes.dbName,
                'db.operation': attributes.operation,
                'bubble.name': attributes.bubbleName,
            },
        });
        return span;
    }
    /**
     * Create a span for API calls
     */
    createAPICallSpan(attributes) {
        if (!this.manager.isEnabled()) {
            return null;
        }
        const tracer = this.manager.getTracer(this.serviceName);
        const span = tracer.startSpan(BubbleSpanName.API_CALL, {
            attributes: {
                'api.name': attributes.apiName,
                'api.operation': attributes.operation,
                'bubble.name': attributes.bubbleName,
            },
        });
        return span;
    }
    /**
     * Wrap a bubble's action method with automatic tracing
     */
    async traceBubbleAction(attributes, fn) {
        const span = this.createExecutionSpan(attributes);
        if (!span) {
            return fn(null);
        }
        const startTime = Date.now();
        try {
            const result = await context.with(trace.setSpan(context.active(), span), async () => {
                return await fn(span);
            });
            span.setStatus({ code: SpanStatusCode.OK });
            span.setAttribute('success', true);
            return result;
        }
        catch (error) {
            if (error instanceof Error) {
                span.recordException(error);
                span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: error.message,
                });
                span.setAttributes({
                    'error.type': error.constructor.name,
                    'error.message': error.message,
                    'error.stack': error.stack,
                });
            }
            span.setAttribute('success', false);
            throw error;
        }
        finally {
            const duration = Date.now() - startTime;
            span.setAttribute('duration.ms', duration);
            span.end();
        }
    }
}
/**
 * Convenience function to create a bubble span
 */
export function createBubbleSpan(name, attributes) {
    const manager = TracingManager.getInstance();
    if (!manager.isEnabled()) {
        return null;
    }
    const tracer = manager.getTracer('bubble-lab');
    return tracer.startSpan(name, {
        attributes,
    });
}
/**
 * Wrap any async function with bubble tracing
 */
export async function withBubbleTracing(bubbleName, bubbleType, operation, fn) {
    const tracer = new BubbleTracer();
    return tracer.traceBubbleAction({
        bubbleName,
        bubbleType,
        operation,
    }, fn);
}
//# sourceMappingURL=bubble-tracer.js.map