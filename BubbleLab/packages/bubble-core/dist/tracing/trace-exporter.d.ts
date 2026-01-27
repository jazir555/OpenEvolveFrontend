/**
 * Trace Exporter - Export traces to various backends
 *
 * This module provides support for exporting traces to different backends
 * including Jaeger, OpenTelemetry Collector, and console output.
 */
import { SpanExporter } from '@opentelemetry/sdk-trace-base';
import { ExporterType, type OtlpExporterOptions } from './types.js';
export { type ExporterType } from './types.js';
/**
 * Trace exporter with support for multiple backends
 */
export declare class TraceExporter {
    private exporter;
    private type;
    private logger;
    constructor(config: {
        type: ExporterType;
        options: unknown;
    });
    /**
     * Create the appropriate exporter based on configuration
     */
    private createExporter;
    /**
     * Create OTLP exporter
     */
    private createOtlpExporter;
    /**
     * Create console exporter (for debugging)
     */
    private createConsoleExporter;
    /**
     * Create collector exporter
     */
    private createCollectorExporter;
    /**
     * Get the underlying exporter
     */
    getExporter(): SpanExporter;
    /**
     * Get the exporter type
     */
    getExporterType(): ExporterType;
    /**
     * Shutdown the exporter
     */
    shutdown(): Promise<void>;
    /**
     * Force flush any pending spans
     */
    forceFlush(): Promise<void>;
}
/**
 * OTLP configuration helper
 */
export declare class OtlpConfigHelper {
    /**
     * Create OTLP exporter configuration for Honeycomb
     */
    static forHoneycomb(apiKey: string, dataset?: string): {
        type: ExporterType;
        options: OtlpExporterOptions;
    };
    /**
     * Create OTLP exporter configuration for New Relic
     */
    static forNewRelic(apiKey: string): {
        type: ExporterType;
        options: OtlpExporterOptions;
    };
    /**
     * Create OTLP exporter configuration for custom collector
     */
    static forCollector(endpoint: string, headers?: Record<string, string>): {
        type: ExporterType;
        options: OtlpExporterOptions;
    };
}
//# sourceMappingURL=trace-exporter.d.ts.map