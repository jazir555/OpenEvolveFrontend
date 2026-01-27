/**
 * Tracer utilities for creating and managing spans
 *
 * This module provides convenient functions for creating spans and wrapping
 * functions with automatic tracing.
 */
import { Span, SpanStatusCode, Context } from '@opentelemetry/api';
import type { TraceConfig, SpanAttributes } from './types.js';
/**
 * Get or create a tracer for a specific component
 */
export declare function createTracer(name: string, version?: string): import("@opentelemetry/api").Tracer;
/**
 * Create a span with the given configuration
 */
export declare function createSpan(config: TraceConfig): Span | null;
/**
 * Wrap a function with automatic tracing
 */
export declare function wrapWithTracing<T extends (...args: unknown[]) => ReturnType<T>>(config: TraceConfig, fn: T): T;
/**
 * Wrap an async function with automatic tracing
 */
export declare function traceAsync<T>(config: TraceConfig, fn: (span: Span | null) => Promise<T>): Promise<T>;
/**
 * Execute a function within an existing span context
 */
export declare function runInSpanContext<T>(spanContext: Context, fn: () => Promise<T>): Promise<T>;
/**
 * Get the current active span
 */
export declare function getCurrentSpan(): Span | null;
/**
 * Add attributes to the current span
 */
export declare function addSpanAttributes(attributes: SpanAttributes): void;
/**
 * Add an event to the current span
 */
export declare function addSpanEvent(name: string, attributes?: SpanAttributes): void;
/**
 * Record an exception in the current span
 */
export declare function recordException(error: Error | string): void;
/**
 * Set the status of the current span
 */
export declare function setSpanStatus(code: SpanStatusCode, message?: string): void;
/**
 * Make a span the active span in the current context
 */
export declare function makeSpanActive(span: Span): Context;
/**
 * Create a root span (no parent)
 */
export declare function createRootSpan(name: string, attributes?: SpanAttributes): Span | null;
/**
 * Create a child span from the current context
 */
export declare function createChildSpan(name: string, attributes?: SpanAttributes): Span | null;
//# sourceMappingURL=tracer.d.ts.map