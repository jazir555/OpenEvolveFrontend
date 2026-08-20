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

import { trace, context, Context, Span, SpanStatusCode, SpanKind } from '@opentelemetry/api';
import type { Logger as LoggerType } from './glue-modules';
const { Logger } = require('../logger');

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
export class Tracer {
  private tracer: any;
  private serviceName: string;
  private logger: LoggerType;

  constructor(serviceName: string) {
    this.serviceName = serviceName;
    this.logger = new Logger(serviceName);

    // Get tracer from OpenTelemetry API
    this.tracer = trace.getTracer(serviceName, '1.0.0');

    this.logger.info('Tracer initialized', {
      service: serviceName,
    });
  }

  /**
   * Start a new span
   */
  startSpan(options: TraceOptions): Span {
    const span = this.tracer.startSpan(options.name, {
      kind: options.kind || SpanKind.INTERNAL,
      attributes: {
        ...options.attributes,
        'service.name': this.serviceName,
        'correlation.id': options.correlationId,
      },
    });

    return span;
  }

  /**
   * Execute function within span context
   */
  async traceAsync<T>(
    options: TraceOptions,
    fn: (span: Span) => Promise<T>
  ): Promise<T> {
    const span = this.startSpan(options);

    try {
      const result = await context.with(trace.setSpan(context.active(), span), async () => {
        return await fn(span);
      });

      span.setStatus({
        code: SpanStatusCode.OK,
      });

      return result;
    } catch (error) {
      span.recordException(error as Error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : 'Unknown error',
      });

      this.logger.error('Span execution failed', error as Error, {
        span_name: options.name,
        correlation_id: options.correlationId,
      });

      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Execute synchronous function within span context
   */
  trace<T>(options: TraceOptions, fn: (span: Span) => T): T {
    const span = this.startSpan(options);

    try {
      const result = context.with(trace.setSpan(context.active(), span), () => {
        return fn(span);
      });

      span.setStatus({
        code: SpanStatusCode.OK,
      });

      return result;
    } catch (error) {
      span.recordException(error as Error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : 'Unknown error',
      });

      this.logger.error('Span execution failed', error as Error, {
        span_name: options.name,
        correlation_id: options.correlationId,
      });

      throw error;
    } finally {
      span.end();
    }
  }

  /**
   * Get current active span
   */
  getCurrentSpan(): Span | undefined {
    return trace.getSpan(context.active());
  }

  /**
   * Add event to current span
   */
  addEvent(name: string, attributes?: Record<string, any>): void {
    const span = this.getCurrentSpan();
    if (span) {
      span.addEvent(name, attributes);
    }
  }

  /**
   * Set attributes on current span
   */
  setAttributes(attributes: Record<string, any>): void {
    const span = this.getCurrentSpan();
    if (span) {
      span.setAttributes(attributes);
    }
  }

  /**
   * Get correlation ID from current span context
   */
  getCorrelationId(): string | undefined {
    const span = this.getCurrentSpan();
    if (span) {
      return (span as any).attributes['correlation.id'] as string;
    }
    return undefined;
  }

  /**
   * Inject context into carrier for propagation
   */
  injectContext(carrier: Record<string, string>): void {
    const span = this.getCurrentSpan();
    if (span) {
      // This would use the propagator from OpenTelemetry
      // Simplified version here
      carrier['traceparent'] = span.spanContext().traceId;
    }
  }

  /**
   * Extract context from carrier
   */
  extractContext(_carrier: Record<string, string>): Context {
    // This would use the propagator from OpenTelemetry
    // Simplified version here
    return context.active();
  }

  /**
   * Create child span with correlation ID
   */
  createChildSpan(
    parentSpan: Span,
    name: string,
    attributes?: Record<string, any>
  ): Span {
    const childSpan = this.tracer.startSpan(name, {
      kind: SpanKind.INTERNAL,
      attributes: {
        ...attributes,
        'service.name': this.serviceName,
        'parent.id': parentSpan.spanContext().spanId,
      },
    });

    return childSpan;
  }

  /**
   * Trace HTTP request
   */
  async traceHttpRequest<T>(options: {
    method: string;
    url: string;
    headers?: Record<string, string>;
    correlationId?: string;
    fn: () => Promise<T>;
  }): Promise<T> {
    const { method, url, headers, correlationId, fn } = options;

    return this.traceAsync(
      {
        name: `HTTP ${method}`,
        kind: SpanKind.CLIENT,
        attributes: {
          'http.method': method,
          'http.url': url,
          'http.headers': JSON.stringify(headers || {}),
        },
        correlationId,
      },
      async (span) => {
        const start = Date.now();

        try {
          const result = await fn();
          const duration = Date.now() - start;

          span.setAttributes({
            'http.status_code': 200,
            'http.response_time_ms': duration,
          });

          return result;
        } catch (error) {
          const duration = Date.now() - start;

          span.setAttributes({
            'http.status_code': 500,
            'http.response_time_ms': duration,
            'error.message': error instanceof Error ? error.message : 'Unknown error',
          });

          throw error;
        }
      }
    );
  }

  /**
   * Trace database operation
   */
  async traceDatabaseOperation<T>(options: {
    operation: string;
    table?: string;
    query?: string;
    correlationId?: string;
    fn: () => Promise<T>;
  }): Promise<T> {
    const { operation, table, query, correlationId, fn } = options;

    return this.traceAsync(
      {
        name: `DB ${operation}`,
        kind: SpanKind.CLIENT,
        attributes: {
          'db.operation': operation,
          'db.table': table,
          'db.query': query,
        },
        correlationId,
      },
      async (span) => {
        const start = Date.now();

        try {
          const result = await fn();
          const duration = Date.now() - start;

          span.setAttributes({
            'db.duration_ms': duration,
            'db.status': 'success',
          });

          return result;
        } catch (error) {
          const duration = Date.now() - start;

          span.setAttributes({
            'db.duration_ms': duration,
            'db.status': 'error',
            'error.message': error instanceof Error ? error.message : 'Unknown error',
          });

          throw error;
        }
      }
    );
  }

  /**
   * Trace knowledge extraction operation
   */
  async traceKnowledgeExtraction<T>(options: {
    source: string;
    method: string;
    entityCount?: number;
    relationCount?: number;
    correlationId?: string;
    fn: () => Promise<T>;
  }): Promise<T> {
    const { source, method, entityCount, relationCount, correlationId, fn } = options;

    return this.traceAsync(
      {
        name: `Knowledge Extraction: ${method}`,
        kind: SpanKind.INTERNAL,
        attributes: {
          'extraction.source': source,
          'extraction.method': method,
        },
        correlationId,
      },
      async (span) => {
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
        } catch (error) {
          const duration = Date.now() - start;

          span.setAttributes({
            'extraction.duration_ms': duration,
            'extraction.status': 'error',
            'error.message': error instanceof Error ? error.message : 'Unknown error',
          });

          throw error;
        }
      }
    );
  }
}

/**
 * Global tracer instance
 */
let globalTracer: Tracer | null = null;

/**
 * Get or create global tracer
 */
export function getTracer(serviceName?: string): Tracer {
  if (!globalTracer) {
    const name = serviceName || process.env.SERVICE_NAME || 'unknown-service';
    globalTracer = new Tracer(name);
  }
  return globalTracer;
}

/**
 * Reset global tracer (useful for testing)
 */
export function resetTracer(): void {
  globalTracer = null;
}

/**
 * Decorator for tracing async methods
 */
export function traceMethod(options: {
  name?: string;
  kind?: SpanKind;
  attributes?: Record<string, any>;
}) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;

    descriptor.value = async function (...args: any[]) {
      const tracer = getTracer();
      const name = options.name || `${target.constructor.name}.${propertyKey}`;

      return tracer.traceAsync(
        {
          name,
          kind: options.kind,
          attributes: {
            ...options.attributes,
            method: propertyKey,
            class: target.constructor.name,
          },
        },
        async (_span) => {
          return originalMethod.apply(this, args);
        }
      );
    };

    return descriptor;
  };
}

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
export function generateServiceMap(spans: Span[]): ServiceMapNode {
  const nodes = new Map<string, ServiceMapNode>();

  // Process spans to build map
  for (const span of spans) {
    const serviceName = (span as any).attributes['service.name'] || 'unknown';
    const operation = (span as any).name || 'unknown';

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

    const node = nodes.get(serviceName)!;
    node.operations.push(operation);
    node.span_count++;

    // Track errors
    if ((span as any).status.code === SpanStatusCode.ERROR) {
      node.error_count++;
    }

    // Track duration if available
    const duration = (span as any).duration / 1000000; // Convert to ms
    node.avg_duration_ms = (node.avg_duration_ms * (node.span_count - 1) + duration) / node.span_count;
  }

  // Return root node (first service)
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
