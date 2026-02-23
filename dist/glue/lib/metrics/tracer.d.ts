/**
 * OpenTelemetry Tracer
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: OTEL endpoint via environment variables
 * - Observability: Distributed tracing for request lineage
 * - Correlation ID binding for log aggregation
 *
 * Features:
 * - OpenTelemetry integration
 * - Span creation and propagation
 * - Correlation ID binding
 * - Request lineage tracking
 * - Service map generation
 */
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
/**
 * Tracer class
 *
 * Manages distributed tracing with OpenTelemetry
 */
export declare class Tracer {
    private tracer;
    private serviceName;
    private logger;
    constructor(serviceName: string);
    /**
     * Start a new span
     */
    startSpan(options: TraceOptions): Span;
    /**
     * Execute function within span context
     */
    traceAsync<T>(options: TraceOptions, fn: (span: Span) => Promise<T>): Promise<T>;
    /**
     * Execute synchronous function within span context
     */
    trace<T>(options: TraceOptions, fn: (span: Span) => T): T;
    /**
     * Get current active span
     */
    getCurrentSpan(): Span | undefined;
    /**
     * Add event to current span
     */
    addEvent(name: string, attributes?: Record<string, any>): void;
    /**
     * Set attributes on current span
     */
    setAttributes(attributes: Record<string, any>): void;
    /**
     * Get correlation ID from current span context
     */
    getCorrelationId(): string | undefined;
    /**
     * Inject context into carrier for propagation
     */
    injectContext(carrier: Record<string, string>): void;
    /**
     * Extract context from carrier
     */
    extractContext(carrier: Record<string, string>): Context;
    /**
     * Create child span with correlation ID
     */
    createChildSpan(parentSpan: Span, name: string, attributes?: Record<string, any>): Span;
    /**
     * Trace HTTP request
     */
    traceHttpRequest<T>(options: {
        method: string;
        url: string;
        headers?: Record<string, string>;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
    /**
     * Trace database operation
     */
    traceDatabaseOperation<T>(options: {
        operation: string;
        table?: string;
        query?: string;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
    /**
     * Trace knowledge extraction operation
     */
    traceKnowledgeExtraction<T>(options: {
        source: string;
        method: string;
        entityCount?: number;
        relationCount?: number;
        correlationId?: string;
        fn: () => Promise<T>;
    }): Promise<T>;
}
/**
 * Get or create global tracer
 */
export declare function getTracer(serviceName?: string): Tracer;
/**
 * Reset global tracer (useful for testing)
 */
export declare function resetTracer(): void;
/**
 * Decorator for tracing async methods
 */
export declare function traceMethod(options: {
    name?: string;
    kind?: SpanKind;
    attributes?: Record<string, any>;
}): (target: any, propertyKey: string, descriptor: PropertyDescriptor) => PropertyDescriptor;
/**
 * Service map node
 */
export interface ServiceMapNode {
    service: string;
    operations: string[];
    dependencies: ServiceMapNode[];
    span_count: number;
    error_count: number;
    avg_duration_ms: number;
}
/**
 * Generate service map from spans
 */
export declare function generateServiceMap(spans: Span[]): ServiceMapNode;
/**
 * Example usage:
 *
 * ```typescript
 * import { getTracer, traceMethod } from './tracer';
 *
 * const tracer = getTracer('crm-adapter');
 *
 * // Trace async operation
 * await tracer.traceAsync({
 *   name: 'fetch-users',
 *   kind: SpanKind.CLIENT,
 *   correlationId: 'abc-123',
 * }, async (span) => {
 *   span.setAttributes({ user_id: '12345' });
 *   const users = await fetchUsers();
 *   return users;
 * });
 *
 * // Trace HTTP request
 * await tracer.traceHttpRequest({
 *   method: 'GET',
 *   url: 'http://api:8000/users',
 *   correlationId: 'abc-123',
 *   fn: async () => {
 *     return await fetch('http://api:8000/users').then(r => r.json());
 *   }
 * });
 *
 * // Trace database operation
 * await tracer.traceDatabaseOperation({
 *   operation: 'SELECT',
 *   table: 'users',
 *   correlationId: 'abc-123',
 *   fn: async () => {
 *     return await db.query('SELECT * FROM users');
 *   }
 * });
 *
 * // Use decorator
 * class UserService {
 *   @traceMethod({ name: 'User Service: Get User' })
 *   async getUser(id: string) {
 *     return await db.findUser(id);
 *   }
 * }
 * ```
 */
//# sourceMappingURL=tracer.d.ts.map