/**
 * Tracer utilities for creating and managing spans
 *
 * This module provides convenient functions for creating spans and wrapping
 * functions with automatic tracing.
 */

import { trace, context, Span, SpanStatusCode, Context } from '@opentelemetry/api';
import type { TraceConfig, SpanAttributes } from './types.js';
import { TracingManager } from './tracing-manager.js';

/**
 * Get or create a tracer for a specific component
 */
export function createTracer(name: string, version: string = '1.0.0') {
  const manager = TracingManager.getInstance();
  return manager.getTracer(name, version);
}

/**
 * Create a span with the given configuration
 */
export function createSpan(config: TraceConfig): Span | null {
  const manager = TracingManager.getInstance();

  if (!manager.isEnabled()) {
    return null;
  }

  const tracer = createTracer('bubble-lab');

  return tracer.startSpan(config.name, {
    kind: config.kind,
    attributes: config.attributes,
    startTime: config.recordEvents ? Date.now() : undefined,
  });
}

/**
 * Wrap a function with automatic tracing
 */
export function wrapWithTracing<T extends (...args: unknown[]) => ReturnType<T>>(
  config: TraceConfig,
  fn: T
): T {
  return (async (...args: unknown[]) => {
    const span = createSpan(config);

    if (!span) {
      return fn(...args);
    }

    const ctx = trace.setSpan(context.active(), span);

    try {
      const result = await context.with(ctx, () => fn(...args));

      span.setStatus({
        code: SpanStatusCode.OK,
      });

      return result;
    } catch (error) {
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error.message,
        });

        // Add error attributes
        span.setAttributes({
          'error.type': error.constructor.name,
          'error.message': error.message,
          'error.stack': error.stack,
        } as SpanAttributes);
      }

      throw error;
    } finally {
      span.end();
    }
  }) as T;
}

/**
 * Wrap an async function with automatic tracing
 */
export async function traceAsync<T>(
  config: TraceConfig,
  fn: (span: Span | null) => Promise<T>
): Promise<T> {
  const manager = TracingManager.getInstance();

  if (!manager.isEnabled()) {
    return fn(null);
  }

  const span = createSpan(config);

  if (!span) {
    return fn(null);
  }

  const ctx = trace.setSpan(context.active(), span);

  try {
    const result = await context.with(ctx, () => fn(span));

    span.setStatus({
      code: SpanStatusCode.OK,
    });

    return result;
  } catch (error) {
    if (error instanceof Error) {
      span.recordException(error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message,
      });

      // Add error attributes
      span.setAttributes({
        'error.type': error.constructor.name,
        'error.message': error.message,
        'error.stack': error.stack,
      } as SpanAttributes);
    }

    throw error;
  } finally {
    span.end();
  }
}

/**
 * Execute a function within an existing span context
 */
export async function runInSpanContext<T>(
  spanContext: Context,
  fn: () => Promise<T>
): Promise<T> {
  return context.with(spanContext, fn);
}

/**
 * Get the current active span
 */
export function getCurrentSpan(): Span | null {
  return trace.getSpan(context.active()) ?? null;
}

/**
 * Add attributes to the current span
 */
export function addSpanAttributes(attributes: SpanAttributes): void {
  const span = getCurrentSpan();
  if (span) {
    span.setAttributes(attributes);
  }
}

/**
 * Add an event to the current span
 */
export function addSpanEvent(name: string, attributes?: SpanAttributes): void {
  const span = getCurrentSpan();
  if (span) {
    span.addEvent(name, attributes);
  }
}

/**
 * Record an exception in the current span
 */
export function recordException(error: Error | string): void {
  const span = getCurrentSpan();
  if (span) {
    const exception = typeof error === 'string' ? new Error(error) : error;
    span.recordException(exception);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: exception.message,
    });
  }
}

/**
 * Set the status of the current span
 */
export function setSpanStatus(
  code: SpanStatusCode,
  message?: string
): void {
  const span = getCurrentSpan();
  if (span) {
    span.setStatus({ code, message });
  }
}

/**
 * Make a span the active span in the current context
 */
export function makeSpanActive(span: Span): Context {
  return trace.setSpan(context.active(), span);
}

/**
 * Create a root span (no parent)
 */
export function createRootSpan(name: string, attributes?: SpanAttributes): Span | null {
  const manager = TracingManager.getInstance();

  if (!manager.isEnabled()) {
    return null;
  }

  const tracer = createTracer('bubble-lab');
  return tracer.startSpan(name, {
    root: true,
    attributes,
  });
}

/**
 * Create a child span from the current context
 */
export function createChildSpan(
  name: string,
  attributes?: SpanAttributes
): Span | null {
  const manager = TracingManager.getInstance();

  if (!manager.isEnabled()) {
    return null;
  }

  const tracer = createTracer('bubble-lab');
  return tracer.startSpan(name, {
    attributes,
  });
}
