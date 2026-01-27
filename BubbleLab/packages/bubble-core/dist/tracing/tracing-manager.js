/**
 * TracingManager - Central management of OpenTelemetry distributed tracing
 *
 * This module provides a singleton manager for configuring and controlling
 * all aspects of distributed tracing in BubbleLab.
 */
import { trace, context, propagation, } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { BatchSpanProcessor, SimpleSpanProcessor, } from '@opentelemetry/sdk-trace-base';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes, } from '@opentelemetry/semantic-conventions';
import { AsyncLocalStorageContextManager } from '@opentelemetry/context-async-hooks';
import { W3CTraceContextPropagator } from '@opentelemetry/core';
import { TraceExporter } from './trace-exporter.js';
import { TraceLogger } from './trace-logger.js';
export class TracingManager {
    static instance = null;
    provider = null;
    config = null;
    exporter = null;
    processor = null;
    contextManager = null;
    logger = new TraceLogger();
    initialized = false;
    constructor() {
        // Private constructor for singleton
    }
    /**
     * Get the singleton instance
     */
    static getInstance() {
        if (!TracingManager.instance) {
            TracingManager.instance = new TracingManager();
        }
        return TracingManager.instance;
    }
    /**
     * Initialize the tracing system with the provided configuration
     */
    async initialize(config) {
        if (this.initialized) {
            this.logger.warn('Tracing already initialized, skipping re-initialization');
            return;
        }
        try {
            this.logger.info('Initializing OpenTelemetry tracing...', { config });
            this.config = config;
            // Validate configuration
            this.validateConfig(config);
            // Set up resource with attributes
            const resource = this.buildResource(config);
            // Create tracer provider
            this.provider = new NodeTracerProvider({
                resource,
            });
            // Set up context manager for async context propagation
            this.contextManager = new AsyncLocalStorageContextManager();
            this.contextManager.enable();
            // Set up propagator (W3C Trace Context)
            propagation.setGlobalPropagator(new W3CTraceContextPropagator());
            // Create and configure exporter
            this.exporter = new TraceExporter(config.exporter);
            // Set up span processor
            if (config.batchExport) {
                this.processor = new BatchSpanProcessor(this.exporter.getExporter(), {
                    maxQueueSize: config.batchExport.maxQueueSize || 2048,
                    maxExportBatchSize: config.batchExport.maxExportBatchSize || 512,
                });
            }
            else {
                this.processor = new SimpleSpanProcessor(this.exporter.getExporter());
            }
            this.provider.addSpanProcessor(this.processor);
            // Register the provider globally
            this.provider.register({
                contextManager: this.contextManager,
            });
            this.initialized = true;
            this.logger.info('OpenTelemetry tracing initialized successfully', {
                serviceName: config.serviceName,
                exporter: config.exporter.type,
                sampleRate: config.sampleRate,
            });
        }
        catch (error) {
            this.logger.error('Failed to initialize OpenTelemetry tracing', error);
            throw error;
        }
    }
    /**
     * Validate the tracing configuration
     */
    validateConfig(config) {
        if (!config.serviceName || config.serviceName.trim().length === 0) {
            throw new Error('serviceName is required in tracing configuration');
        }
        if (config.sampleRate < 0 || config.sampleRate > 1) {
            throw new Error('sampleRate must be between 0 and 1');
        }
        if (!config.exporter || !config.exporter.type) {
            throw new Error('exporter configuration is required');
        }
    }
    /**
     * Build the OpenTelemetry Resource with attributes
     */
    buildResource(config) {
        const attributes = {
            [SemanticResourceAttributes.SERVICE_NAME]: config.serviceName,
            [SemanticResourceAttributes.SERVICE_VERSION]: process.env.npm_package_version || '1.0.0',
            ...config.resourceAttributes,
        };
        // Add environment information
        if (process.env.NODE_ENV) {
            attributes[SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT] = process.env.NODE_ENV;
        }
        return new Resource(attributes);
    }
    /**
     * Check if tracing is enabled
     */
    isEnabled() {
        return this.config?.enabled ?? false;
    }
    /**
     * Get the current configuration
     */
    getConfig() {
        return this.config;
    }
    /**
     * Get a tracer for a specific component
     */
    getTracer(name, version = '1.0.0') {
        if (!this.initialized) {
            this.logger.warn('Tracing not initialized, returning no-op tracer');
            return trace.getTracer(name, version);
        }
        return trace.getTracer(name, version);
    }
    /**
     * Get the current span context
     */
    getCurrentSpanContext() {
        const currentSpan = trace.getSpan(context.active());
        if (!currentSpan) {
            return null;
        }
        const spanContext = currentSpan.spanContext();
        return {
            traceId: spanContext.traceId,
            spanId: spanContext.spanId,
            traceFlags: spanContext.traceFlags,
            parentSpanId: currentSpan.parentSpanId,
        };
    }
    /**
     * Add attributes to the current span
     */
    addAttributes(attributes) {
        const currentSpan = trace.getSpan(context.active());
        if (currentSpan) {
            currentSpan.setAttributes(attributes);
        }
    }
    /**
     * Record an exception in the current span
     */
    recordException(error) {
        const currentSpan = trace.getSpan(context.active());
        if (currentSpan) {
            const exception = typeof error === 'string' ? new Error(error) : error;
            currentSpan.recordException(exception);
            currentSpan.setStatus({
                code: 2, // ERROR
                message: exception.message,
            });
        }
    }
    /**
     * Force flush all pending spans
     */
    async flush() {
        if (this.provider) {
            this.logger.info('Flushing traces...');
            await this.provider.forceFlush();
            this.logger.info('Traces flushed successfully');
        }
    }
    /**
     * Shutdown the tracing system
     */
    async shutdown() {
        if (!this.initialized) {
            return;
        }
        this.logger.info('Shutting down OpenTelemetry tracing...');
        if (this.provider) {
            await this.provider.shutdown();
            this.provider = null;
        }
        this.processor = null;
        this.exporter = null;
        this.contextManager = null;
        this.config = null;
        this.initialized = false;
        this.logger.info('OpenTelemetry tracing shut down successfully');
    }
    /**
     * Get trace statistics
     */
    getStats() {
        return {
            initialized: this.initialized,
            enabled: this.isEnabled(),
            serviceName: this.config?.serviceName,
            exporter: this.config?.exporter.type,
            sampleRate: this.config?.sampleRate,
        };
    }
}
//# sourceMappingURL=tracing-manager.js.map