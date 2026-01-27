/**
 * TracingManager - Central management of OpenTelemetry distributed tracing
 *
 * This module provides a singleton manager for configuring and controlling
 * all aspects of distributed tracing in BubbleLab.
 */
import type { TracingConfig, SpanAttributes, SpanContext } from './types.js';
import { ExporterType } from './trace-exporter.js';
export declare class TracingManager {
    private static instance;
    private provider;
    private config;
    private exporter;
    private processor;
    private contextManager;
    private logger;
    private initialized;
    private constructor();
    /**
     * Get the singleton instance
     */
    static getInstance(): TracingManager;
    /**
     * Initialize the tracing system with the provided configuration
     */
    initialize(config: TracingConfig): Promise<void>;
    /**
     * Validate the tracing configuration
     */
    private validateConfig;
    /**
     * Build the OpenTelemetry Resource with attributes
     */
    private buildResource;
    /**
     * Check if tracing is enabled
     */
    isEnabled(): boolean;
    /**
     * Get the current configuration
     */
    getConfig(): TracingConfig | null;
    /**
     * Get a tracer for a specific component
     */
    getTracer(name: string, version?: string): import("@opentelemetry/api").Tracer;
    /**
     * Get the current span context
     */
    getCurrentSpanContext(): SpanContext | null;
    /**
     * Add attributes to the current span
     */
    addAttributes(attributes: SpanAttributes): void;
    /**
     * Record an exception in the current span
     */
    recordException(error: Error | string): void;
    /**
     * Force flush all pending spans
     */
    flush(): Promise<void>;
    /**
     * Shutdown the tracing system
     */
    shutdown(): Promise<void>;
    /**
     * Get trace statistics
     */
    getStats(): {
        initialized: boolean;
        enabled: boolean;
        serviceName: string | undefined;
        exporter: ExporterType | undefined;
        sampleRate: number | undefined;
    };
}
//# sourceMappingURL=tracing-manager.d.ts.map