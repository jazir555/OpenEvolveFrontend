"use strict";
/**
 * Prometheus Metrics Collector
 *
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Observability: Structured metrics for monitoring
 * - Integration with Circuit Breakers
 *
 * Metrics collected:
 * - Request latency (histogram)
 * - Error rates (counter)
 * - Circuit breaker states (gauge)
 * - Adapter health (gauge)
 * - Knowledge extraction metrics (counter, gauge, histogram)
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MetricsCollector = void 0;
exports.getMetricsCollector = getMetricsCollector;
exports.resetMetricsCollector = resetMetricsCollector;
const prom_client_1 = require("prom-client");
const circuit_breaker_1 = require("../circuit-breaker");
const logger_1 = require("../logger");
/**
 * Metrics Collector class
 *
 * Central registry for all Prometheus metrics
 */
class MetricsCollector {
    constructor(prefix = 'openevolve_') {
        this.prefix = prefix;
        this.registry = new prom_client_1.Registry();
        this.initializeMetrics();
        this.registerDefaultMetrics();
    }
    /**
     * Initialize all Prometheus metrics
     */
    initializeMetrics() {
        // HTTP Request Metrics
        this.httpRequestDuration = new prom_client_1.Histogram({
            name: `${this.prefix}http_request_duration_seconds`,
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['service', 'operation', 'status'],
            registers: [this.registry],
            buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
        });
        this.httpRequestsTotal = new prom_client_1.Counter({
            name: `${this.prefix}http_requests_total`,
            help: 'Total number of HTTP requests',
            labelNames: ['service', 'operation', 'status'],
            registers: [this.registry],
        });
        this.httpRequestsInProgress = new prom_client_1.Gauge({
            name: `${this.prefix}http_requests_in_progress`,
            help: 'Number of HTTP requests in progress',
            labelNames: ['service'],
            registers: [this.registry],
        });
        // Error Metrics
        this.errorsTotal = new prom_client_1.Counter({
            name: `${this.prefix}errors_total`,
            help: 'Total number of errors',
            labelNames: ['service', 'error_type'],
            registers: [this.registry],
        });
        this.errorsByType = new prom_client_1.Counter({
            name: `${this.prefix}errors_by_type_total`,
            help: 'Errors categorized by type',
            labelNames: ['service', 'operation', 'error_type'],
            registers: [this.registry],
        });
        // Circuit Breaker Metrics
        this.circuitBreakerState = new prom_client_1.Gauge({
            name: `${this.prefix}circuit_breaker_state`,
            help: 'Current state of circuit breaker (0=closed, 1=half_open, 2=open)',
            labelNames: ['service', 'circuit'],
            registers: [this.registry],
        });
        this.circuitBreakerFailures = new prom_client_1.Counter({
            name: `${this.prefix}circuit_breaker_failures_total`,
            help: 'Total circuit breaker failures',
            labelNames: ['service', 'circuit'],
            registers: [this.registry],
        });
        this.circuitBreakerSuccesses = new prom_client_1.Counter({
            name: `${this.prefix}circuit_breaker_successes_total`,
            help: 'Total circuit breaker successes',
            labelNames: ['service', 'circuit'],
            registers: [this.registry],
        });
        this.circuitBreakerRejects = new prom_client_1.Counter({
            name: `${this.prefix}circuit_breaker_rejects_total`,
            help: 'Total requests rejected by circuit breaker',
            labelNames: ['service', 'circuit'],
            registers: [this.registry],
        });
        // Adapter Health Metrics
        this.adapterHealth = new prom_client_1.Gauge({
            name: `${this.prefix}adapter_health`,
            help: 'Adapter health status (0=unhealthy, 1=degraded, 2=healthy)',
            labelNames: ['adapter'],
            registers: [this.registry],
        });
        this.adapterLastSuccess = new prom_client_1.Gauge({
            name: `${this.prefix}adapter_last_success_timestamp`,
            help: 'Unix timestamp of last successful operation',
            labelNames: ['adapter'],
            registers: [this.registry],
        });
        this.adapterLastFailure = new prom_client_1.Gauge({
            name: `${this.prefix}adapter_last_failure_timestamp`,
            help: 'Unix timestamp of last failed operation',
            labelNames: ['adapter'],
            registers: [this.registry],
        });
        // Knowledge Extraction Metrics
        this.knowledgeExtractionTotal = new prom_client_1.Counter({
            name: `${this.prefix}knowledge_extraction_total`,
            help: 'Total knowledge extraction operations',
            labelNames: ['source', 'method', 'entity_type', 'success'],
            registers: [this.registry],
        });
        this.knowledgeExtractionDuration = new prom_client_1.Histogram({
            name: `${this.prefix}knowledge_extraction_duration_seconds`,
            help: 'Duration of knowledge extraction operations',
            labelNames: ['source', 'method'],
            registers: [this.registry],
            buckets: [0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
        });
        this.knowledgeExtractionEntitiesExtracted = new prom_client_1.Gauge({
            name: `${this.prefix}knowledge_extraction_entities`,
            help: 'Number of entities extracted',
            labelNames: ['source', 'entity_type'],
            registers: [this.registry],
        });
        this.knowledgeExtractionRelationsExtracted = new prom_client_1.Gauge({
            name: `${this.prefix}knowledge_extraction_relations`,
            help: 'Number of relations extracted',
            labelNames: ['source'],
            registers: [this.registry],
        });
        // Event Bus Metrics
        this.eventsProcessed = new prom_client_1.Counter({
            name: `${this.prefix}events_processed_total`,
            help: 'Total events processed',
            labelNames: ['event_type', 'status'],
            registers: [this.registry],
        });
        this.eventProcessingDuration = new prom_client_1.Histogram({
            name: `${this.prefix}event_processing_duration_seconds`,
            help: 'Duration of event processing',
            labelNames: ['event_type'],
            registers: [this.registry],
            buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
        });
        this.eventsInQueue = new prom_client_1.Gauge({
            name: `${this.prefix}events_in_queue`,
            help: 'Number of events currently in queue',
            labelNames: ['queue_name'],
            registers: [this.registry],
        });
        // Retry Metrics
        this.retryAttempts = new prom_client_1.Counter({
            name: `${this.prefix}retry_attempts_total`,
            help: 'Total retry attempts',
            labelNames: ['service', 'operation'],
            registers: [this.registry],
        });
        this.retrySuccess = new prom_client_1.Counter({
            name: `${this.prefix}retry_success_total`,
            help: 'Total successful retries',
            labelNames: ['service', 'operation'],
            registers: [this.registry],
        });
        this.retryFailure = new prom_client_1.Counter({
            name: `${this.prefix}retry_failure_total`,
            help: 'Total failed retries after all attempts',
            labelNames: ['service', 'operation'],
            registers: [this.registry],
        });
    }
    /**
     * Register default Prometheus metrics (CPU, memory, etc.)
     */
    registerDefaultMetrics() {
        (0, prom_client_1.collectDefaultMetrics)({
            register: this.registry,
            prefix: this.prefix,
        });
    }
    /**
     * Record HTTP request duration
     */
    recordHttpRequestDuration(labels, duration) {
        this.httpRequestDuration.observe({
            service: labels.service,
            operation: labels.operation || 'unknown',
            status: labels.status || 'unknown',
        }, duration);
    }
    /**
     * Increment HTTP request counter
     */
    incrementHttpRequests(labels) {
        this.httpRequestsTotal.inc({
            service: labels.service,
            operation: labels.operation || 'unknown',
            status: labels.status || 'unknown',
        });
    }
    /**
     * Increment/decrement HTTP requests in progress
     */
    setHttpRequestsInProgress(service, delta) {
        if (delta > 0) {
            this.httpRequestsInProgress.inc({ service });
        }
        else {
            this.httpRequestsInProgress.dec({ service });
        }
    }
    /**
     * Record an error
     */
    recordError(labels) {
        this.errorsTotal.inc({
            service: labels.service,
            error_type: labels.error_type || 'unknown',
        });
        if (labels.operation) {
            this.errorsByType.inc({
                service: labels.service,
                operation: labels.operation,
                error_type: labels.error_type || 'unknown',
            });
        }
    }
    /**
     * Update circuit breaker state metric
     */
    setCircuitBreakerState(service, circuit, state) {
        const stateValue = state === circuit_breaker_1.CircuitState.CLOSED ? 0 : state === circuit_breaker_1.CircuitState.HALF_OPEN ? 1 : 2;
        this.circuitBreakerState.set({ service, circuit }, stateValue);
    }
    /**
     * Record circuit breaker failure
     */
    recordCircuitBreakerFailure(service, circuit) {
        this.circuitBreakerFailures.inc({ service, circuit });
    }
    /**
     * Record circuit breaker success
     */
    recordCircuitBreakerSuccess(service, circuit) {
        this.circuitBreakerSuccesses.inc({ service, circuit });
    }
    /**
     * Record circuit breaker reject (request rejected due to open circuit)
     */
    recordCircuitBreakerReject(service, circuit) {
        this.circuitBreakerRejects.inc({ service, circuit });
    }
    /**
     * Set adapter health status
     */
    setAdapterHealth(adapter, health) {
        const healthValue = health === 'healthy' ? 2 : health === 'degraded' ? 1 : 0;
        this.adapterHealth.set({ adapter }, healthValue);
    }
    /**
     * Update adapter last success timestamp
     */
    setAdapterLastSuccess(adapter) {
        this.adapterLastSuccess.set({ adapter }, Date.now() / 1000);
    }
    /**
     * Update adapter last failure timestamp
     */
    setAdapterLastFailure(adapter) {
        this.adapterLastFailure.set({ adapter }, Date.now() / 1000);
    }
    /**
     * Record knowledge extraction operation
     */
    recordKnowledgeExtraction(labels) {
        this.knowledgeExtractionTotal.inc({
            source: labels.source,
            method: labels.method,
            entity_type: labels.entity_type || 'all',
            success: labels.success,
        });
    }
    /**
     * Record knowledge extraction duration
     */
    recordKnowledgeExtractionDuration(source, method, duration) {
        this.knowledgeExtractionDuration.observe({ source, method }, duration);
    }
    /**
     * Set number of entities extracted
     */
    setEntitiesExtracted(source, entityType, count) {
        this.knowledgeExtractionEntitiesExtracted.set({ source, entity_type: entityType }, count);
    }
    /**
     * Set number of relations extracted
     */
    setRelationsExtracted(source, count) {
        this.knowledgeExtractionRelationsExtracted.set({ source }, count);
    }
    /**
     * Record event processed
     */
    recordEventProcessed(eventType, status) {
        this.eventsProcessed.inc({ event_type: eventType, status });
    }
    /**
     * Record event processing duration
     */
    recordEventProcessingDuration(eventType, duration) {
        this.eventProcessingDuration.observe({ event_type: eventType }, duration);
    }
    /**
     * Set events in queue
     */
    setEventsInQueue(queueName, count) {
        this.eventsInQueue.set({ queue_name: queueName }, count);
    }
    /**
     * Record retry attempt
     */
    recordRetryAttempt(service, operation) {
        this.retryAttempts.inc({ service, operation });
    }
    /**
     * Record successful retry
     */
    recordRetrySuccess(service, operation) {
        this.retrySuccess.inc({ service, operation });
    }
    /**
     * Record failed retry
     */
    recordRetryFailure(service, operation) {
        this.retryFailure.inc({ service, operation });
    }
    /**
     * Get metrics registry
     */
    getRegistry() {
        return this.registry;
    }
    /**
     * Get metrics as Prometheus text format
     */
    async getMetrics() {
        return await this.registry.metrics();
    }
    /**
     * Clear all metrics
     */
    clearMetrics() {
        this.registry.clear();
    }
    /**
     * Reset a specific metric
     */
    resetMetric(metricName) {
        this.registry.getSingleMetric(`${this.prefix}${metricName}`)?.reset();
    }
}
exports.MetricsCollector = MetricsCollector;
/**
 * Global metrics collector instance
 */
let globalMetricsCollector = null;
/**
 * Get or create global metrics collector
 */
function getMetricsCollector() {
    if (!globalMetricsCollector) {
        const prefix = process.env.METRICS_PREFIX || 'openevolve_';
        globalMetricsCollector = new MetricsCollector(prefix);
        logger_1.logger.info('Metrics collector initialized', {
            prefix,
        });
    }
    return globalMetricsCollector;
}
/**
 * Reset global metrics collector (useful for testing)
 */
function resetMetricsCollector() {
    globalMetricsCollector = null;
}
/**
 * Example usage:
 *
 * ```typescript
 * import { getMetricsCollector } from './metrics-collector';
 *
 * const metrics = getMetricsCollector();
 *
 * // Record HTTP request
 * const start = Date.now();
 * try {
 *   await makeRequest();
 *   const duration = (Date.now() - start) / 1000;
 *   metrics.recordHttpRequestDuration({
 *     service: 'crm-adapter',
 *     operation: 'fetch-users',
 *     status: 'success',
 *   }, duration);
 *   metrics.incrementHttpRequests({
 *     service: 'crm-adapter',
 *     operation: 'fetch-users',
 *     status: '2xx',
 *   });
 * } catch (error) {
 *   metrics.recordError({
 *     service: 'crm-adapter',
 *     operation: 'fetch-users',
 *     error_type: error.name,
 *   });
 * }
 *
 * // Update circuit breaker state
 * metrics.setCircuitBreakerState('crm-adapter', 'api', CircuitState.OPEN);
 *
 * // Record knowledge extraction
 * metrics.recordKnowledgeExtraction({
 *   source: 'github',
 *   method: 'code-analysis',
 *   entity_type: 'class',
 *   success: 'true',
 * });
 * metrics.setEntitiesExtracted('github', 'class', 42);
 *
 * // Get metrics for Prometheus
 * const metricsText = await metrics.getMetrics();
 * ```
 */
//# sourceMappingURL=metrics-collector.js.map