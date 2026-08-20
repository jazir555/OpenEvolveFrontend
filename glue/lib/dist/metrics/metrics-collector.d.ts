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
import { Registry } from 'prom-client';
import type { CircuitState as CircuitStateType } from './glue-modules';
export interface MetricsLabels {
    service: string;
    operation?: string;
    status?: string;
    error_type?: string;
    adapter?: string;
    [key: string]: string | number | undefined;
}
export interface KnowledgeExtractionLabels {
    source: string;
    method: string;
    entity_type?: string;
    success: string;
}
/**
 * Metrics Collector class
 *
 * Central registry for all Prometheus metrics
 */
export declare class MetricsCollector {
    private registry;
    private prefix;
    private httpRequestDuration;
    private httpRequestsTotal;
    private httpRequestsInProgress;
    private errorsTotal;
    private errorsByType;
    private circuitBreakerState;
    private circuitBreakerFailures;
    private circuitBreakerSuccesses;
    private circuitBreakerRejects;
    private adapterHealth;
    private adapterLastSuccess;
    private adapterLastFailure;
    private knowledgeExtractionTotal;
    private knowledgeExtractionDuration;
    private knowledgeExtractionEntitiesExtracted;
    private knowledgeExtractionRelationsExtracted;
    private eventsProcessed;
    private eventProcessingDuration;
    private eventsInQueue;
    private retryAttempts;
    private retrySuccess;
    private retryFailure;
    constructor(prefix?: string);
    /**
     * Initialize all Prometheus metrics
     */
    private initializeMetrics;
    /**
     * Register default Prometheus metrics (CPU, memory, etc.)
     */
    private registerDefaultMetrics;
    /**
     * Record HTTP request duration
     */
    recordHttpRequestDuration(labels: MetricsLabels, duration: number): void;
    /**
     * Increment HTTP request counter
     */
    incrementHttpRequests(labels: MetricsLabels): void;
    /**
     * Increment/decrement HTTP requests in progress
     */
    setHttpRequestsInProgress(service: string, delta: number): void;
    /**
     * Record an error
     */
    recordError(labels: MetricsLabels): void;
    /**
     * Update circuit breaker state metric
     */
    setCircuitBreakerState(service: string, circuit: string, state: CircuitStateType): void;
    /**
     * Record circuit breaker failure
     */
    recordCircuitBreakerFailure(service: string, circuit: string): void;
    /**
     * Record circuit breaker success
     */
    recordCircuitBreakerSuccess(service: string, circuit: string): void;
    /**
     * Record circuit breaker reject (request rejected due to open circuit)
     */
    recordCircuitBreakerReject(service: string, circuit: string): void;
    /**
     * Set adapter health status
     */
    setAdapterHealth(adapter: string, health: 'unhealthy' | 'degraded' | 'healthy'): void;
    /**
     * Update adapter last success timestamp
     */
    setAdapterLastSuccess(adapter: string): void;
    /**
     * Update adapter last failure timestamp
     */
    setAdapterLastFailure(adapter: string): void;
    /**
     * Record knowledge extraction operation
     */
    recordKnowledgeExtraction(labels: KnowledgeExtractionLabels): void;
    /**
     * Record knowledge extraction duration
     */
    recordKnowledgeExtractionDuration(source: string, method: string, duration: number): void;
    /**
     * Set number of entities extracted
     */
    setEntitiesExtracted(source: string, entityType: string, count: number): void;
    /**
     * Set number of relations extracted
     */
    setRelationsExtracted(source: string, count: number): void;
    /**
     * Record event processed
     */
    recordEventProcessed(eventType: string, status: 'success' | 'failure'): void;
    /**
     * Record event processing duration
     */
    recordEventProcessingDuration(eventType: string, duration: number): void;
    /**
     * Set events in queue
     */
    setEventsInQueue(queueName: string, count: number): void;
    /**
     * Record retry attempt
     */
    recordRetryAttempt(service: string, operation: string): void;
    /**
     * Record successful retry
     */
    recordRetrySuccess(service: string, operation: string): void;
    /**
     * Record failed retry
     */
    recordRetryFailure(service: string, operation: string): void;
    /**
     * Get metrics registry
     */
    getRegistry(): Registry;
    /**
     * Get metrics as Prometheus text format
     */
    getMetrics(): Promise<string>;
    /**
     * Clear all metrics
     */
    clearMetrics(): void;
    /**
     * Reset a specific metric
     */
    resetMetric(metricName: string): void;
}
/**
 * Get or create global metrics collector
 */
export declare function getMetricsCollector(): MetricsCollector;
/**
 * Reset global metrics collector (useful for testing)
 */
export declare function resetMetricsCollector(): void;
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
//# sourceMappingURL=metrics-collector.d.ts.map