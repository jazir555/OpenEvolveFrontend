/**
 * OpenTelemetry Distributed Tracing for BubbleLab
 *
 * This module provides comprehensive distributed tracing capabilities for all bubble operations,
 * enabling end-to-end request tracking across the entire BubbleLab ecosystem.
 *
 * Features:
 * - Automatic span creation for all bubble operations
 * - Context propagation across service boundaries
 * - Performance monitoring and analysis
 * - Integration with Jaeger for visualization
 * - Support for multiple propagation formats (W3C, B3, Jaeger)
 */
export { TracingManager } from './tracing-manager.js';
export { createTracer, wrapWithTracing, traceAsync } from './tracer.js';
export { TracePropagator, propagateContext, extractContext, injectContext } from './context-propagator.js';
export { BubbleTracer, createBubbleSpan } from './bubble-tracer.js';
export { TraceExporter } from './trace-exporter.js';
export type { ExporterType } from './trace-exporter.js';
export { TraceMetricsAnalyzer, analyzePerformance } from './trace-metrics.js';
export { TraceAlertManager } from './trace-alerts.js';
export type { AlertRule } from './trace-alerts.js';
export type { TracingConfig, TraceConfig, SpanAttributes, SpanContext, TraceMetrics, PerformanceAnalysis, } from './types.js';
//# sourceMappingURL=index.d.ts.map