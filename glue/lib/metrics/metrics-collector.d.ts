import { Registry } from 'prom-client';
import { CircuitState } from '../circuit-breaker';
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
    private initializeMetrics;
    private registerDefaultMetrics;
    recordHttpRequestDuration(labels: MetricsLabels, duration: number): void;
    incrementHttpRequests(labels: MetricsLabels): void;
    setHttpRequestsInProgress(service: string, delta: number): void;
    recordError(labels: MetricsLabels): void;
    setCircuitBreakerState(service: string, circuit: string, state: CircuitState): void;
    recordCircuitBreakerFailure(service: string, circuit: string): void;
    recordCircuitBreakerSuccess(service: string, circuit: string): void;
    recordCircuitBreakerReject(service: string, circuit: string): void;
    setAdapterHealth(adapter: string, health: 'unhealthy' | 'degraded' | 'healthy'): void;
    setAdapterLastSuccess(adapter: string): void;
    setAdapterLastFailure(adapter: string): void;
    recordKnowledgeExtraction(labels: KnowledgeExtractionLabels): void;
    recordKnowledgeExtractionDuration(source: string, method: string, duration: number): void;
    setEntitiesExtracted(source: string, entityType: string, count: number): void;
    setRelationsExtracted(source: string, count: number): void;
    recordEventProcessed(eventType: string, status: 'success' | 'failure'): void;
    recordEventProcessingDuration(eventType: string, duration: number): void;
    setEventsInQueue(queueName: string, count: number): void;
    recordRetryAttempt(service: string, operation: string): void;
    recordRetrySuccess(service: string, operation: string): void;
    recordRetryFailure(service: string, operation: string): void;
    getRegistry(): Registry;
    getMetrics(): Promise<string>;
    clearMetrics(): void;
    resetMetric(metricName: string): void;
}
export declare function getMetricsCollector(): MetricsCollector;
export declare function resetMetricsCollector(): void;
//# sourceMappingURL=metrics-collector.d.ts.map