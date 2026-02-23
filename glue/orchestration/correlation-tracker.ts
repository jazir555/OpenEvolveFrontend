/**
 * Correlation Tracker - Distributed Tracing & Request Lineage
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Observability: Correlation ID propagation across all services
 * - UTC timestamps only
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../lib/logger';

export interface CorrelationContext {
  correlation_id: string;
  parent_id?: string;
  trace_id?: string;
  service_path: ServiceCall[];
  start_time: string; // ISO-8601 UTC
  metadata: Record<string, any>;
}

export interface ServiceCall {
  service: string;
  timestamp: string; // ISO-8601 UTC
  operation?: string;
  duration_ms?: number;
}

export interface DistributedTraceSpan {
  span_id: string;
  parent_span_id?: string;
  trace_id: string;
  service_name: string;
  operation_name: string;
  start_time: number; // Unix timestamp
  end_time?: number;
  tags: Record<string, string>;
  status: 'ok' | 'error';
}

/**
 * Correlation Tracker for distributed tracing
 *
 * Generates UUID v4 correlation IDs and tracks request lineage across services
 */
export class CorrelationTracker {
  private logger: Logger;
  private traces: Map<string, DistributedTraceSpan[]> = new Map();

  constructor() {
    this.logger = new Logger('correlation-tracker');
  }

  /**
   * Generate a new UUID v4 correlation ID
   */
  generateCorrelationId(): string {
    return uuidv4();
  }

  /**
   * Generate a new trace ID for distributed tracing
   */
  generateTraceId(): string {
    return uuidv4();
  }

  /**
   * Create a new correlation context
   */
  createContext(metadata: Record<string, any> = {}): CorrelationContext {
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
  createChildContext(parentContext: CorrelationContext, serviceName: string): CorrelationContext {
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
  recordServiceCall(
    context: CorrelationContext,
    serviceName: string,
    operation?: string
  ): void {
    const call: ServiceCall = {
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
  createSpan(
    traceId: string,
    parentSpanId: string | undefined,
    serviceName: string,
    operationName: string,
    tags: Record<string, string> = {}
  ): DistributedTraceSpan {
    const span: DistributedTraceSpan = {
      span_id: uuidv4(),
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
    this.traces.get(traceId)!.push(span);

    return span;
  }

  /**
   * Complete a distributed tracing span
   */
  completeSpan(span: DistributedTraceSpan, status: 'ok' | 'error' = 'ok'): void {
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
  getTrace(traceId: string): DistributedTraceSpan[] | undefined {
    return this.traces.get(traceId);
  }

  /**
   * Calculate total duration of a correlation context
   */
  calculateDuration(context: CorrelationContext): number {
    const startTime = new Date(context.start_time).getTime();
    const endTime = Date.now();
    return endTime - startTime;
  }

  /**
   * Format correlation context for logging
   */
  formatForLogging(context: CorrelationContext): Record<string, any> {
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
  extractOrGenerate(headers: Record<string, string>): string {
    // Check various header names for correlation ID
    const correlationId =      headers['x-correlation-id']
      || headers['x-request-id']
      || headers['correlation-id']
      || undefined;

    return correlationId || this.generateCorrelationId();
  }

  /**
   * Inject correlation ID into headers
   */
  injectIntoHeaders(context: CorrelationContext): Record<string, string> {
    return {
      'x-correlation-id': context.correlation_id,
      'x-trace-id': context.trace_id || '',
      'x-parent-id': context.parent_id || ''
    };
  }

  /**
   * Cleanup old traces to prevent memory leaks
   */
  cleanup(maxAgeMs: number = 3600000): void {
    const now = Date.now();
    const cleaned: string[] = [];

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

/**
 * Singleton instance
 */
export const correlationTracker = new CorrelationTracker();

/**
 * Middleware helper for Express/fastify style request handlers
 */
export function createCorrelationMiddleware(tracker: CorrelationTracker) {
  return (req: any, res: any, next: any) => {
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
