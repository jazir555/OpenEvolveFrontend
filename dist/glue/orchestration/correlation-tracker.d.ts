/**
 * Correlation Tracker - Distributed Tracing & Request Lineage
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Observability: Correlation ID propagation across all services
 * - UTC timestamps only
 */
export interface CorrelationContext {
    correlation_id: string;
    parent_id?: string;
    trace_id?: string;
    service_path: ServiceCall[];
    start_time: string;
    metadata: Record<string, any>;
}
export interface ServiceCall {
    service: string;
    timestamp: string;
    operation?: string;
    duration_ms?: number;
}
export interface DistributedTraceSpan {
    span_id: string;
    parent_span_id?: string;
    trace_id: string;
    service_name: string;
    operation_name: string;
    start_time: number;
    end_time?: number;
    tags: Record<string, string>;
    status: 'ok' | 'error';
}
/**
 * Correlation Tracker for distributed tracing
 *
 * Generates UUID v4 correlation IDs and tracks request lineage across services
 */
export declare class CorrelationTracker {
    private logger;
    private traces;
    constructor();
    /**
     * Generate a new UUID v4 correlation ID
     */
    generateCorrelationId(): string;
    /**
     * Generate a new trace ID for distributed tracing
     */
    generateTraceId(): string;
    /**
     * Create a new correlation context
     */
    createContext(metadata?: Record<string, any>): CorrelationContext;
    /**
     * Create a child context from existing correlation ID
     */
    createChildContext(parentContext: CorrelationContext, serviceName: string): CorrelationContext;
    /**
     * Record a service call in the correlation context
     */
    recordServiceCall(context: CorrelationContext, serviceName: string, operation?: string): void;
    /**
     * Create a distributed tracing span
     */
    createSpan(traceId: string, parentSpanId: string | undefined, serviceName: string, operationName: string, tags?: Record<string, string>): DistributedTraceSpan;
    /**
     * Complete a distributed tracing span
     */
    completeSpan(span: DistributedTraceSpan, status?: 'ok' | 'error'): void;
    /**
     * Get trace by ID
     */
    getTrace(traceId: string): DistributedTraceSpan[] | undefined;
    /**
     * Calculate total duration of a correlation context
     */
    calculateDuration(context: CorrelationContext): number;
    /**
     * Format correlation context for logging
     */
    formatForLogging(context: CorrelationContext): Record<string, any>;
    /**
     * Extract correlation ID from headers or generate new one
     */
    extractOrGenerate(headers: Record<string, string>): string;
    /**
     * Inject correlation ID into headers
     */
    injectIntoHeaders(context: CorrelationContext): Record<string, string>;
    /**
     * Cleanup old traces to prevent memory leaks
     */
    cleanup(maxAgeMs?: number): void;
}
/**
 * Singleton instance
 */
export declare const correlationTracker: CorrelationTracker;
/**
 * Middleware helper for Express/fastify style request handlers
 */
export declare function createCorrelationMiddleware(tracker: CorrelationTracker): (req: any, res: any, next: any) => void;
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
//# sourceMappingURL=correlation-tracker.d.ts.map