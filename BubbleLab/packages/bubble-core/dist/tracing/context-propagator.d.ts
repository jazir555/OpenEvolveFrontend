/**
 * Context Propagation - Trace context propagation across service boundaries
 *
 * This module handles extraction and injection of trace context to enable
 * distributed tracing across multiple services and components.
 */
import { Context, TextMapGetter, TextMapSetter, Span } from '@opentelemetry/api';
import type { SpanContext } from './types.js';
/**
 * Trace context propagator
 */
export declare class TracePropagator {
    private logger;
    /**
     * Extract trace context from incoming request headers
     */
    extractFromHeaders(headers: Record<string, string>): Context | null;
    /**
     * Inject trace context into outgoing request headers
     */
    injectIntoHeaders(headers: Record<string, string>): Record<string, string>;
    /**
     * Extract trace context from a carrier object
     */
    extract(carrier: unknown, getter?: TextMapGetter<unknown>): Context;
    /**
     * Inject trace context into a carrier object
     */
    inject(carrier: unknown, setter?: TextMapSetter<unknown>): void;
    /**
     * Get the current span context
     */
    getCurrentContext(): SpanContext | null;
    /**
     * Create a new context with a specific span
     */
    createContextWithSpan(span: Span): Context;
    /**
     * Execute a function within a specific context
     */
    executeInContext<T>(ctx: Context, fn: () => Promise<T>): Promise<T>;
    /**
     * Propagate context to an async operation
     */
    propagateToAsync<T>(fn: () => Promise<T>): Promise<T>;
}
/**
 * Propagate trace context from incoming headers
 */
export declare function propagateContext(headers: Record<string, string>): Context | null;
/**
 * Extract trace context from a carrier
 */
export declare function extractContext(carrier: unknown, getter?: TextMapGetter<unknown>): Context;
/**
 * Inject trace context into a carrier
 */
export declare function injectContext(carrier: unknown, setter?: TextMapSetter<unknown>): void;
/**
 * Get the current trace context
 */
export declare function getCurrentTraceContext(): SpanContext | null;
/**
 * Execute a function with trace context propagation
 */
export declare function withTracePropagation<T>(fn: () => Promise<T>): Promise<T>;
/**
 * Utility to inject context into HTTP headers for outgoing requests
 */
export declare function injectHeadersIntoRequest(headers: Record<string, string>): Record<string, string>;
/**
 * Utility to extract context from HTTP headers for incoming requests
 */
export declare function extractContextFromRequest(headers: Record<string, string>): Context;
/**
 * Wrap axios/fetch headers with trace context
 */
export declare function withTraceHeaders(existingHeaders?: Record<string, string>): Record<string, string>;
//# sourceMappingURL=context-propagator.d.ts.map