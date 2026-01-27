/**
 * BubbleTracer - Specialized tracing for Bubble operations
 *
 * This module provides bubble-specific tracing utilities that automatically
 * create appropriate spans with bubble-specific attributes.
 */
import { Span } from '@opentelemetry/api';
import type { SpanAttributes } from './types.js';
/**
 * Standardized span names for bubble operations
 */
export declare enum BubbleSpanName {
    BUBBLE_INSTANTIATION = "bubble.instantiation",
    BUBBLE_VALIDATION = "bubble.validation",
    BUBBLE_EXECUTION = "bubble.execution",
    HTTP_REQUEST = "bubble.http.request",
    DATABASE_QUERY = "bubble.database.query",
    API_CALL = "bubble.api.call",
    WORKFLOW_START = "workflow.start",
    WORKFLOW_END = "workflow.end",
    WORKFLOW_STEP = "workflow.step",
    TOOL_EXECUTION = "tool.execution",
    TOOL_VALIDATION = "tool.validation"
}
/**
 * Create a standardized bubble span
 */
export declare class BubbleTracer {
    private manager;
    private serviceName;
    constructor(serviceName?: string);
    /**
     * Create a span for bubble instantiation
     */
    createInstantiationSpan(attributes: {
        bubbleName: string;
        bubbleType: 'service' | 'tool' | 'workflow';
        variableName?: string;
        className?: string;
    }): Span | null;
    /**
     * Create a span for bubble validation
     */
    createValidationSpan(attributes: {
        bubbleName: string;
        bubbleType: 'service' | 'tool' | 'workflow';
        validationPhase: 'input' | 'output' | 'credential';
    }): Span | null;
    /**
     * Create a span for bubble execution
     */
    createExecutionSpan(attributes: {
        bubbleName: string;
        bubbleType: 'service' | 'tool' | 'workflow';
        operation: string;
        correlationId?: string;
        executionId?: string;
    }): Span | null;
    /**
     * Create a span for HTTP requests
     */
    createHTTPRequestSpan(attributes: {
        url: string;
        method: string;
        bubbleName?: string;
    }): Span | null;
    /**
     * Create a span for database queries
     */
    createDatabaseQuerySpan(attributes: {
        dbSystem: string;
        dbName?: string;
        operation?: string;
        bubbleName?: string;
    }): Span | null;
    /**
     * Create a span for API calls
     */
    createAPICallSpan(attributes: {
        apiName: string;
        operation: string;
        bubbleName?: string;
    }): Span | null;
    /**
     * Wrap a bubble's action method with automatic tracing
     */
    traceBubbleAction<T>(attributes: {
        bubbleName: string;
        bubbleType: 'service' | 'tool' | 'workflow';
        operation: string;
        correlationId?: string;
        executionId?: string;
    }, fn: (span: Span | null) => Promise<T>): Promise<T>;
}
/**
 * Convenience function to create a bubble span
 */
export declare function createBubbleSpan(name: string, attributes: SpanAttributes): Span | null;
/**
 * Wrap any async function with bubble tracing
 */
export declare function withBubbleTracing<T>(bubbleName: string, bubbleType: 'service' | 'tool' | 'workflow', operation: string, fn: (span: Span | null) => Promise<T>): Promise<T>;
//# sourceMappingURL=bubble-tracer.d.ts.map