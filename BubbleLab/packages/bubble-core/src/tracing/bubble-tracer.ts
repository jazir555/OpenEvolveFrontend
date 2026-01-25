/**
 * BubbleTracer - Specialized tracing for Bubble operations
 *
 * This module provides bubble-specific tracing utilities that automatically
 * create appropriate spans with bubble-specific attributes.
 */

import { trace, context, Span, SpanStatusCode } from '@opentelemetry/api';
import type { SpanAttributes } from './types.js';
import { TracingManager } from './tracing-manager.js';

/**
 * Standardized span names for bubble operations
 */
export enum BubbleSpanName {
  // Lifecycle
  BUBBLE_INSTANTIATION = 'bubble.instantiation',
  BUBBLE_VALIDATION = 'bubble.validation',
  BUBBLE_EXECUTION = 'bubble.execution',

  // External operations
  HTTP_REQUEST = 'bubble.http.request',
  DATABASE_QUERY = 'bubble.database.query',
  API_CALL = 'bubble.api.call',

  // Workflow operations
  WORKFLOW_START = 'workflow.start',
  WORKFLOW_END = 'workflow.end',
  WORKFLOW_STEP = 'workflow.step',

  // Tool operations
  TOOL_EXECUTION = 'tool.execution',
  TOOL_VALIDATION = 'tool.validation',
}

/**
 * Create a standardized bubble span
 */
export class BubbleTracer {
  private manager: TracingManager;
  private serviceName: string;

  constructor(serviceName: string = 'bubble-lab') {
    this.manager = TracingManager.getInstance();
    this.serviceName = serviceName;
  }

  /**
   * Create a span for bubble instantiation
   */
  createInstantiationSpan(attributes: {
    bubbleName: string;
    bubbleType: 'service' | 'tool' | 'workflow';
    variableName?: string;
    className?: string;
  }): Span | null {
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
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Create a span for bubble validation
   */
  createValidationSpan(attributes: {
    bubbleName: string;
    bubbleType: 'service' | 'tool' | 'workflow';
    validationPhase: 'input' | 'output' | 'credential';
  }): Span | null {
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
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Create a span for bubble execution
   */
  createExecutionSpan(attributes: {
    bubbleName: string;
    bubbleType: 'service' | 'tool' | 'workflow';
    operation: string;
    correlationId?: string;
    executionId?: string;
  }): Span | null {
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
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Create a span for HTTP requests
   */
  createHTTPRequestSpan(attributes: {
    url: string;
    method: string;
    bubbleName?: string;
  }): Span | null {
    if (!this.manager.isEnabled()) {
      return null;
    }

    const tracer = this.manager.getTracer(this.serviceName);
    const span = tracer.startSpan(BubbleSpanName.HTTP_REQUEST, {
      attributes: {
        'http.method': attributes.method,
        'http.url': attributes.url,
        'bubble.name': attributes.bubbleName,
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Create a span for database queries
   */
  createDatabaseQuerySpan(attributes: {
    dbSystem: string;
    dbName?: string;
    operation?: string;
    bubbleName?: string;
  }): Span | null {
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
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Create a span for API calls
   */
  createAPICallSpan(attributes: {
    apiName: string;
    operation: string;
    bubbleName?: string;
  }): Span | null {
    if (!this.manager.isEnabled()) {
      return null;
    }

    const tracer = this.manager.getTracer(this.serviceName);
    const span = tracer.startSpan(BubbleSpanName.API_CALL, {
      attributes: {
        'api.name': attributes.apiName,
        'api.operation': attributes.operation,
        'bubble.name': attributes.bubbleName,
      } as SpanAttributes,
    });

    return span;
  }

  /**
   * Wrap a bubble's action method with automatic tracing
   */
  async traceBubbleAction<T>(
    attributes: {
      bubbleName: string;
      bubbleType: 'service' | 'tool' | 'workflow';
      operation: string;
      correlationId?: string;
      executionId?: string;
    },
    fn: (span: Span | null) => Promise<T>
  ): Promise<T> {
    const span = this.createExecutionSpan(attributes);

    if (!span) {
      return fn(null);
    }

    const startTime = Date.now();

    try {
      const result = await context.with(
        trace.setSpan(context.active(), span),
        async () => {
          return await fn(span);
        }
      );

      span.setStatus({ code: SpanStatusCode.OK });
      span.setAttribute('success', true);

      return result;
    } catch (error) {
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
        } as SpanAttributes);
      }

      span.setAttribute('success', false);
      throw error;
    } finally {
      const duration = Date.now() - startTime;
      span.setAttribute('duration.ms', duration);
      span.end();
    }
  }
}

/**
 * Convenience function to create a bubble span
 */
export function createBubbleSpan(
  name: string,
  attributes: SpanAttributes
): Span | null {
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
export async function withBubbleTracing<T>(
  bubbleName: string,
  bubbleType: 'service' | 'tool' | 'workflow',
  operation: string,
  fn: (span: Span | null) => Promise<T>
): Promise<T> {
  const tracer = new BubbleTracer();
  return tracer.traceBubbleAction(
    {
      bubbleName,
      bubbleType,
      operation,
    },
    fn
  );
}
