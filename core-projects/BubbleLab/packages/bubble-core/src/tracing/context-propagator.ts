/**
 * Context Propagation - Trace context propagation across service boundaries
 *
 * This module handles extraction and injection of trace context to enable
 * distributed tracing across multiple services and components.
 */

import {
  propagation,
  context,
  trace,
  Context,
  TextMapGetter,
  TextMapSetter,
  Span,
} from '@opentelemetry/api';
import type { SpanContext } from './types.js';
import { TraceLogger } from './trace-logger.js';

/**
 * Trace context propagator
 */
export class TracePropagator {
  private logger = new TraceLogger();

  /**
   * Extract trace context from incoming request headers
   */
  extractFromHeaders(headers: Record<string, string>): Context | null {
    try {
      const getter: TextMapGetter<Record<string, string>> = {
        get: (carrier, key) => carrier[key],
        keys: (carrier) => Object.keys(carrier),
      };

      const extractedContext = propagation.extract(context.active(), headers, getter);

      this.logger.debug('Extracted trace context from headers', {
        hasContext: extractedContext !== context.active(),
      });

      return extractedContext;
    } catch (error) {
      this.logger.error('Failed to extract trace context', error);
      return null;
    }
  }

  /**
   * Inject trace context into outgoing request headers
   */
  injectIntoHeaders(headers: Record<string, string>): Record<string, string> {
    try {
      const setter: TextMapSetter<Record<string, string>> = {
        set: (carrier, key, value) => {
          carrier[key] = value;
        },
      };

      propagation.inject(context.active(), headers, setter);

      this.logger.debug('Injected trace context into headers', {
        headersCount: Object.keys(headers).length,
      });

      return headers;
    } catch (error) {
      this.logger.error('Failed to inject trace context', error);
      return headers;
    }
  }

  /**
   * Extract trace context from a carrier object
   */
  extract(carrier: unknown, getter?: TextMapGetter<unknown>): Context {
    const defaultGetter: TextMapGetter<unknown> = {
      get: (carrier, key) => {
        if (typeof carrier === 'object' && carrier !== null) {
          const value = (carrier as Record<string, unknown>)[key];
          return typeof value === 'string' ? value : undefined;
        }
        return undefined;
      },
      keys: (carrier) => {
        if (typeof carrier === 'object' && carrier !== null) {
          return Object.keys(carrier);
        }
        return [];
      },
    };

    return propagation.extract(
      context.active(),
      carrier,
      getter || defaultGetter
    );
  }

  /**
   * Inject trace context into a carrier object
   */
  inject(carrier: unknown, setter?: TextMapSetter<unknown>): void {
    const defaultSetter: TextMapSetter<unknown> = {
      set: (carrier, key, value) => {
        if (typeof carrier === 'object' && carrier !== null) {
          (carrier as Record<string, unknown>)[key] = value;
        }
      },
    };

    propagation.inject(context.active(), carrier, setter || defaultSetter);
  }

  /**
   * Get the current span context
   */
  getCurrentContext(): SpanContext | null {
    const currentSpan = trace.getSpan(context.active());
    if (!currentSpan) {
      return null;
    }

    const spanContext = currentSpan.spanContext();
    return {
      traceId: spanContext.traceId,
      spanId: spanContext.spanId,
      traceFlags: spanContext.traceFlags,
      // parentSpanId is not directly available on Span object
      // parentSpanId: currentSpan.parentSpanId,
    };
  }

  /**
   * Create a new context with a specific span
   */
  createContextWithSpan(span: Span): Context {
    return trace.setSpan(context.active(), span);
  }

  /**
   * Execute a function within a specific context
   */
  async executeInContext<T>(
    ctx: Context,
    fn: () => Promise<T>
  ): Promise<T> {
    return context.with(ctx, fn);
  }

  /**
   * Propagate context to an async operation
   */
  async propagateToAsync<T>(
    fn: () => Promise<T>
  ): Promise<T> {
    const currentContext = context.active();
    return context.with(currentContext, fn);
  }
}

/**
 * Global trace propagator instance
 */
const globalPropagator = new TracePropagator();

/**
 * Propagate trace context from incoming headers
 */
export function propagateContext(headers: Record<string, string>): Context | null {
  return globalPropagator.extractFromHeaders(headers);
}

/**
 * Extract trace context from a carrier
 */
export function extractContext(
  carrier: unknown,
  getter?: TextMapGetter<unknown>
): Context {
  return globalPropagator.extract(carrier, getter);
}

/**
 * Inject trace context into a carrier
 */
export function injectContext(
  carrier: unknown,
  setter?: TextMapSetter<unknown>
): void {
  globalPropagator.inject(carrier, setter);
}

/**
 * Get the current trace context
 */
export function getCurrentTraceContext(): SpanContext | null {
  return globalPropagator.getCurrentContext();
}

/**
 * Execute a function with trace context propagation
 */
export async function withTracePropagation<T>(
  fn: () => Promise<T>
): Promise<T> {
  return globalPropagator.propagateToAsync(fn);
}

/**
 * Utility to inject context into HTTP headers for outgoing requests
 */
export function injectHeadersIntoRequest(
  headers: Record<string, string>
): Record<string, string> {
  return globalPropagator.injectIntoHeaders(headers);
}

/**
 * Utility to extract context from HTTP headers for incoming requests
 */
export function extractContextFromRequest(
  headers: Record<string, string>
): Context {
  return globalPropagator.extractFromHeaders(headers) || context.active();
}

/**
 * Wrap axios/fetch headers with trace context
 */
export function withTraceHeaders(
  existingHeaders: Record<string, string> = {}
): Record<string, string> {
  return injectHeadersIntoRequest(existingHeaders);
}
