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

import { Registry, Counter, Histogram, Gauge, collectDefaultMetrics } from 'prom-client';
import { CircuitState } from '../circuit-breaker';
import { logger, LoggerContext } from '../logger';

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
export class MetricsCollector {
  private registry: Registry;
  private prefix: string;

  // HTTP Request Metrics
  private httpRequestDuration: Histogram<string>;
  private httpRequestsTotal: Counter<string>;
  private httpRequestsInProgress: Gauge<string>;

  // Error Metrics
  private errorsTotal: Counter<string>;
  private errorsByType: Counter<string>;

  // Circuit Breaker Metrics
  private circuitBreakerState: Gauge<string>;
  private circuitBreakerFailures: Counter<string>;
  private circuitBreakerSuccesses: Counter<string>;
  private circuitBreakerRejects: Counter<string>;

  // Adapter Health Metrics
  private adapterHealth: Gauge<string>;
  private adapterLastSuccess: Gauge<string>;
  private adapterLastFailure: Gauge<string>;

  // Knowledge Extraction Metrics
  private knowledgeExtractionTotal: Counter<string>;
  private knowledgeExtractionDuration: Histogram<string>;
  private knowledgeExtractionEntitiesExtracted: Gauge<string>;
  private knowledgeExtractionRelationsExtracted: Gauge<string>;

  // Event Bus Metrics
  private eventsProcessed: Counter<string>;
  private eventProcessingDuration: Histogram<string>;
  private eventsInQueue: Gauge<string>;

  // Retry Metrics
  private retryAttempts: Counter<string>;
  private retrySuccess: Counter<string>;
  private retryFailure: Counter<string>;

  constructor(prefix: string = 'openevolve_') {
    this.prefix = prefix;
    this.registry = new Registry();

    this.initializeMetrics();
    this.registerDefaultMetrics();
  }

  /**
   * Initialize all Prometheus metrics
   */
  private initializeMetrics(): void {
    // HTTP Request Metrics
    this.httpRequestDuration = new Histogram<string>({
      name: `${this.prefix}http_request_duration_seconds`,
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['service', 'operation', 'status'],
      registers: [this.registry],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    });

    this.httpRequestsTotal = new Counter<string>({
      name: `${this.prefix}http_requests_total`,
      help: 'Total number of HTTP requests',
      labelNames: ['service', 'operation', 'status'],
      registers: [this.registry],
    });

    this.httpRequestsInProgress = new Gauge<string>({
      name: `${this.prefix}http_requests_in_progress`,
      help: 'Number of HTTP requests in progress',
      labelNames: ['service'],
      registers: [this.registry],
    });

    // Error Metrics
    this.errorsTotal = new Counter<string>({
      name: `${this.prefix}errors_total`,
      help: 'Total number of errors',
      labelNames: ['service', 'error_type'],
      registers: [this.registry],
    });

    this.errorsByType = new Counter<string>({
      name: `${this.prefix}errors_by_type_total`,
      help: 'Errors categorized by type',
      labelNames: ['service', 'operation', 'error_type'],
      registers: [this.registry],
    });

    // Circuit Breaker Metrics
    this.circuitBreakerState = new Gauge<string>({
      name: `${this.prefix}circuit_breaker_state`,
      help: 'Current state of circuit breaker (0=closed, 1=half_open, 2=open)',
      labelNames: ['service', 'circuit'],
      registers: [this.registry],
    });

    this.circuitBreakerFailures = new Counter<string>({
      name: `${this.prefix}circuit_breaker_failures_total`,
      help: 'Total circuit breaker failures',
      labelNames: ['service', 'circuit'],
      registers: [this.registry],
    });

    this.circuitBreakerSuccesses = new Counter<string>({
      name: `${this.prefix}circuit_breaker_successes_total`,
      help: 'Total circuit breaker successes',
      labelNames: ['service', 'circuit'],
      registers: [this.registry],
    });

    this.circuitBreakerRejects = new Counter<string>({
      name: `${this.prefix}circuit_breaker_rejects_total`,
      help: 'Total requests rejected by circuit breaker',
      labelNames: ['service', 'circuit'],
      registers: [this.registry],
    });

    // Adapter Health Metrics
    this.adapterHealth = new Gauge<string>({
      name: `${this.prefix}adapter_health`,
      help: 'Adapter health status (0=unhealthy, 1=degraded, 2=healthy)',
      labelNames: ['adapter'],
      registers: [this.registry],
    });

    this.adapterLastSuccess = new Gauge<string>({
      name: `${this.prefix}adapter_last_success_timestamp`,
      help: 'Unix timestamp of last successful operation',
      labelNames: ['adapter'],
      registers: [this.registry],
    });

    this.adapterLastFailure = new Gauge<string>({
      name: `${this.prefix}adapter_last_failure_timestamp`,
      help: 'Unix timestamp of last failed operation',
      labelNames: ['adapter'],
      registers: [this.registry],
    });

    // Knowledge Extraction Metrics
    this.knowledgeExtractionTotal = new Counter<string>({
      name: `${this.prefix}knowledge_extraction_total`,
      help: 'Total knowledge extraction operations',
      labelNames: ['source', 'method', 'entity_type', 'success'],
      registers: [this.registry],
    });

    this.knowledgeExtractionDuration = new Histogram<string>({
      name: `${this.prefix}knowledge_extraction_duration_seconds`,
      help: 'Duration of knowledge extraction operations',
      labelNames: ['source', 'method'],
      registers: [this.registry],
      buckets: [0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
    });

    this.knowledgeExtractionEntitiesExtracted = new Gauge<string>({
      name: `${this.prefix}knowledge_extraction_entities`,
      help: 'Number of entities extracted',
      labelNames: ['source', 'entity_type'],
      registers: [this.registry],
    });

    this.knowledgeExtractionRelationsExtracted = new Gauge<string>({
      name: `${this.prefix}knowledge_extraction_relations`,
      help: 'Number of relations extracted',
      labelNames: ['source'],
      registers: [this.registry],
    });

    // Event Bus Metrics
    this.eventsProcessed = new Counter<string>({
      name: `${this.prefix}events_processed_total`,
      help: 'Total events processed',
      labelNames: ['event_type', 'status'],
      registers: [this.registry],
    });

    this.eventProcessingDuration = new Histogram<string>({
      name: `${this.prefix}event_processing_duration_seconds`,
      help: 'Duration of event processing',
      labelNames: ['event_type'],
      registers: [this.registry],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
    });

    this.eventsInQueue = new Gauge<string>({
      name: `${this.prefix}events_in_queue`,
      help: 'Number of events currently in queue',
      labelNames: ['queue_name'],
      registers: [this.registry],
    });

    // Retry Metrics
    this.retryAttempts = new Counter<string>({
      name: `${this.prefix}retry_attempts_total`,
      help: 'Total retry attempts',
      labelNames: ['service', 'operation'],
      registers: [this.registry],
    });

    this.retrySuccess = new Counter<string>({
      name: `${this.prefix}retry_success_total`,
      help: 'Total successful retries',
      labelNames: ['service', 'operation'],
      registers: [this.registry],
    });

    this.retryFailure = new Counter<string>({
      name: `${this.prefix}retry_failure_total`,
      help: 'Total failed retries after all attempts',
      labelNames: ['service', 'operation'],
      registers: [this.registry],
    });
  }

  /**
   * Register default Prometheus metrics (CPU, memory, etc.)
   */
  private registerDefaultMetrics(): void {
    collectDefaultMetrics({
      register: this.registry,
      prefix: this.prefix,
    });
  }

  /**
   * Record HTTP request duration
   */
  recordHttpRequestDuration(labels: MetricsLabels, duration: number): void {
    this.httpRequestDuration.observe(
      {
        service: labels.service,
        operation: labels.operation || 'unknown',
        status: labels.status || 'unknown',
      },
      duration
    );
  }

  /**
   * Increment HTTP request counter
   */
  incrementHttpRequests(labels: MetricsLabels): void {
    this.httpRequestsTotal.inc({
      service: labels.service,
      operation: labels.operation || 'unknown',
      status: labels.status || 'unknown',
    });
  }

  /**
   * Increment/decrement HTTP requests in progress
   */
  setHttpRequestsInProgress(service: string, delta: number): void {
    if (delta > 0) {
      this.httpRequestsInProgress.inc({ service });
    } else {
      this.httpRequestsInProgress.dec({ service });
    }
  }

  /**
   * Record an error
   */
  recordError(labels: MetricsLabels): void {
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
  setCircuitBreakerState(service: string, circuit: string, state: CircuitState): void {
    const stateValue = state === CircuitState.CLOSED ? 0 : state === CircuitState.HALF_OPEN ? 1 : 2;
    this.circuitBreakerState.set({ service, circuit }, stateValue);
  }

  /**
   * Record circuit breaker failure
   */
  recordCircuitBreakerFailure(service: string, circuit: string): void {
    this.circuitBreakerFailures.inc({ service, circuit });
  }

  /**
   * Record circuit breaker success
   */
  recordCircuitBreakerSuccess(service: string, circuit: string): void {
    this.circuitBreakerSuccesses.inc({ service, circuit });
  }

  /**
   * Record circuit breaker reject (request rejected due to open circuit)
   */
  recordCircuitBreakerReject(service: string, circuit: string): void {
    this.circuitBreakerRejects.inc({ service, circuit });
  }

  /**
   * Set adapter health status
   */
  setAdapterHealth(adapter: string, health: 'unhealthy' | 'degraded' | 'healthy'): void {
    const healthValue = health === 'healthy' ? 2 : health === 'degraded' ? 1 : 0;
    this.adapterHealth.set({ adapter }, healthValue);
  }

  /**
   * Update adapter last success timestamp
   */
  setAdapterLastSuccess(adapter: string): void {
    this.adapterLastSuccess.set({ adapter }, Date.now() / 1000);
  }

  /**
   * Update adapter last failure timestamp
   */
  setAdapterLastFailure(adapter: string): void {
    this.adapterLastFailure.set({ adapter }, Date.now() / 1000);
  }

  /**
   * Record knowledge extraction operation
   */
  recordKnowledgeExtraction(labels: KnowledgeExtractionLabels): void {
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
  recordKnowledgeExtractionDuration(source: string, method: string, duration: number): void {
    this.knowledgeExtractionDuration.observe({ source, method }, duration);
  }

  /**
   * Set number of entities extracted
   */
  setEntitiesExtracted(source: string, entityType: string, count: number): void {
    this.knowledgeExtractionEntitiesExtracted.set({ source, entity_type: entityType }, count);
  }

  /**
   * Set number of relations extracted
   */
  setRelationsExtracted(source: string, count: number): void {
    this.knowledgeExtractionRelationsExtracted.set({ source }, count);
  }

  /**
   * Record event processed
   */
  recordEventProcessed(eventType: string, status: 'success' | 'failure'): void {
    this.eventsProcessed.inc({ event_type: eventType, status });
  }

  /**
   * Record event processing duration
   */
  recordEventProcessingDuration(eventType: string, duration: number): void {
    this.eventProcessingDuration.observe({ event_type: eventType }, duration);
  }

  /**
   * Set events in queue
   */
  setEventsInQueue(queueName: string, count: number): void {
    this.eventsInQueue.set({ queue_name: queueName }, count);
  }

  /**
   * Record retry attempt
   */
  recordRetryAttempt(service: string, operation: string): void {
    this.retryAttempts.inc({ service, operation });
  }

  /**
   * Record successful retry
   */
  recordRetrySuccess(service: string, operation: string): void {
    this.retrySuccess.inc({ service, operation });
  }

  /**
   * Record failed retry
   */
  recordRetryFailure(service: string, operation: string): void {
    this.retryFailure.inc({ service, operation });
  }

  /**
   * Get metrics registry
   */
  getRegistry(): Registry {
    return this.registry;
  }

  /**
   * Get metrics as Prometheus text format
   */
  async getMetrics(): Promise<string> {
    return await this.registry.metrics();
  }

  /**
   * Clear all metrics
   */
  clearMetrics(): void {
    this.registry.clear();
  }

  /**
   * Reset a specific metric
   */
  resetMetric(metricName: string): void {
    this.registry.getSingleMetric(`${this.prefix}${metricName}`)?.reset();
  }
}

/**
 * Global metrics collector instance
 */
let globalMetricsCollector: MetricsCollector | null = null;

/**
 * Get or create global metrics collector
 */
export function getMetricsCollector(): MetricsCollector {
  if (!globalMetricsCollector) {
    const prefix = process.env.METRICS_PREFIX || 'openevolve_';
    globalMetricsCollector = new MetricsCollector(prefix);

    logger.info('Metrics collector initialized', {
      prefix,
    });
  }
  return globalMetricsCollector;
}

/**
 * Reset global metrics collector (useful for testing)
 */
export function resetMetricsCollector(): void {
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
