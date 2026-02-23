import { Context, Span, SpanKind } from '@opentelemetry/api';
export interface TraceOptions {
    name: string;
    kind?: SpanKind;
    attributes?: Record<string, any>;
    correlationId?: string;
}
export interface SpanMetadata {
    service: string;
    operation: string;
    correlation_id?: string;
    parent_span_id?: string;
    [key: string]: any;
}
export declare class Tracer {
    private tracer;
    private serviceName;
    private logger;
    constructor(serviceName: string);
    startSpan(options: TraceOptions): Span;
    traceAsync<T>(options: TraceOptions, fn: (span: Span) => Promise<T>): Promise<T>;
    trace<T>(options: TraceOptions, fn: (span: Span) => T): T;
    getCurrentSpan(): Span | undefined;
    addEvent(name: string, attributes?: Record<string, any>): void;
    setAttributes(attributes: Record<string, any>): void;
    getCorrelationId(): string | undefined;
    injectContext(carrier: Record<string, string>): void;
    extractContext(carrier: Record<string, string>): Context;
    createChildSpan(parentSpan: Span, name: string, attributes?: Record<string, any>): Span;
    traceHttpRequest<T>(options: {
        method: string;
        url: string;
        headers?: Record<string, string>;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
    traceDatabaseOperation<T>(options: {
        operation: string;
        table?: string;
        query?: string;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
    traceKnowledgeExtraction<T>(options: {
        source: string;
        method: string;
        entityCount?: number;
        relationCount?: number;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
}
export declare function getTracer(serviceName?: string): Tracer;
export declare function resetTracer(): void;
export declare function traceMethod(options: {
    name?: string;
    kind?: SpanKind;
    attributes?: Record<string, any>;
}): (target: any, propertyKey: string, descriptor: PropertyDescriptor) => PropertyDescriptor;
export interface ServiceMapNode {
    service: string;
    operations: string[];
    dependencies: ServiceMapNode[];
    span_count: number;
    error_count: number;
    avg_duration_ms: number;
}
export declare function generateServiceMap(spans: Span[]): ServiceMapNode;
//# sourceMappingURL=tracer.d.ts.map