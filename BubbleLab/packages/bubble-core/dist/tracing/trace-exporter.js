/**
 * Trace Exporter - Export traces to various backends
 *
 * This module provides support for exporting traces to different backends
 * including Jaeger, OpenTelemetry Collector, and console output.
 */
import { ConsoleSpanExporter, } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';
import { OTLPTraceExporter as OTLPTraceExporterHTTP } from '@opentelemetry/exporter-trace-otlp-http';
import { ExporterType, } from './types.js';
import { TraceLogger } from './trace-logger.js';
/**
 * Trace exporter with support for multiple backends
 */
export class TraceExporter {
    exporter = null;
    type;
    logger = new TraceLogger();
    constructor(config) {
        this.type = config.type;
        this.exporter = this.createExporter(config);
    }
    /**
     * Create the appropriate exporter based on configuration
     */
    createExporter(config) {
        switch (config.type) {
            case ExporterType.OTLP:
                return this.createOtlpExporter(config.options);
            case ExporterType.CONSOLE:
                return this.createConsoleExporter(config.options);
            case ExporterType.COLLECTOR:
                return this.createCollectorExporter(config.options);
            default:
                this.logger.warn(`Unknown exporter type: ${config.type}, falling back to console`);
                return new ConsoleSpanExporter();
        }
    }
    /**
     * Create OTLP exporter
     */
    createOtlpExporter(options) {
        // Determine if HTTP or gRPC based on URL
        const useHTTP = options.url.startsWith('http://') || options.url.startsWith('https://');
        let exporter;
        if (useHTTP) {
            exporter = new OTLPTraceExporterHTTP({
                url: options.url,
                headers: options.headers,
            });
            this.logger.info('Created OTLP HTTP exporter', {
                url: options.url,
            });
        }
        else {
            exporter = new OTLPTraceExporter({
                url: options.url,
                headers: options.headers,
            });
            this.logger.info('Created OTLP gRPC exporter', {
                url: options.url,
            });
        }
        return exporter;
    }
    /**
     * Create console exporter (for debugging)
     */
    createConsoleExporter(options) {
        this.logger.info('Created console exporter', options);
        return new ConsoleSpanExporter();
    }
    /**
     * Create collector exporter
     */
    createCollectorExporter(options) {
        const exporter = new OTLPTraceExporterHTTP({
            url: options.endpoint,
            headers: options.headers,
        });
        this.logger.info('Created collector exporter', {
            endpoint: options.endpoint,
        });
        return exporter;
    }
    /**
     * Get the underlying exporter
     */
    getExporter() {
        if (!this.exporter) {
            throw new Error('Exporter not initialized');
        }
        return this.exporter;
    }
    /**
     * Get the exporter type
     */
    getExporterType() {
        return this.type;
    }
    /**
     * Shutdown the exporter
     */
    async shutdown() {
        if (this.exporter) {
            await this.exporter.shutdown();
            this.exporter = null;
            this.logger.info('Trace exporter shut down');
        }
    }
    /**
     * Force flush any pending spans
     */
    async forceFlush() {
        if (this.exporter && 'forceFlush' in this.exporter) {
            await this.exporter.forceFlush();
        }
    }
}
/**
 * OTLP configuration helper
 */
export class OtlpConfigHelper {
    /**
     * Create OTLP exporter configuration for Honeycomb
     */
    static forHoneycomb(apiKey, dataset) {
        return {
            type: ExporterType.OTLP,
            options: {
                url: 'https://api.honeycomb.io:443/v1/traces',
                headers: {
                    'x-honeycomb-team': apiKey,
                    ...(dataset ? { 'x-honeycomb-dataset': dataset } : {}),
                },
            },
        };
    }
    /**
     * Create OTLP exporter configuration for New Relic
     */
    static forNewRelic(apiKey) {
        return {
            type: ExporterType.OTLP,
            options: {
                url: 'https://otlp.nr-data.net:4317/v1/traces',
                headers: {
                    'api-key': apiKey,
                },
            },
        };
    }
    /**
     * Create OTLP exporter configuration for custom collector
     */
    static forCollector(endpoint, headers) {
        return {
            type: ExporterType.COLLECTOR,
            options: {
                url: endpoint,
                headers,
            },
        };
    }
}
//# sourceMappingURL=trace-exporter.js.map