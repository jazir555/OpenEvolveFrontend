/**
 * Prometheus Metrics Collection for BubbleLab
 *
 * Provides comprehensive metrics collection for all bubble operations including:
 * - Operation duration and counts
 * - Circuit breaker state
 * - Rate limiting
 * - Error tracking
 * - Performance metrics
 * - Business metrics
 */
import { Registry, Counter, Histogram, Gauge } from 'prom-client';
declare const register: Registry<"text/plain; version=0.0.4; charset=utf-8">;
/**
 * Tracks the duration of bubble operations
 * Labels: bubble, operation, status
 */
export declare const bubbleOperationDuration: Histogram<"status" | "operation" | "bubble">;
/**
 * Tracks the total number of operations
 * Labels: bubble, operation, status
 */
export declare const bubbleOperationTotal: Counter<"status" | "operation" | "bubble">;
/**
 * Tracks the number of operation retries
 * Labels: bubble, operation
 */
export declare const bubbleOperationRetryTotal: Counter<"operation" | "bubble">;
/**
 * Tracks the current state of circuit breakers
 * Labels: bubble, state (0=closed, 1=open, 2=half_open)
 */
export declare const circuitBreakerState: Gauge<"state" | "bubble">;
/**
 * Tracks the number of circuit breaker failures
 * Labels: bubble
 */
export declare const circuitBreakerFailureTotal: Counter<"bubble">;
/**
 * Tracks the number of circuit breaker successes
 * Labels: bubble
 */
export declare const circuitBreakerSuccessTotal: Counter<"bubble">;
/**
 * Tracks the number of rate limit violations
 * Labels: bubble
 */
export declare const rateLimitExceededTotal: Counter<"bubble">;
/**
 * Tracks the remaining rate limit quota
 * Labels: bubble
 */
export declare const rateLimitRemaining: Gauge<"bubble">;
/**
 * Tracks the total number of errors
 * Labels: bubble, error_type, operation
 */
export declare const bubbleErrorTotal: Counter<"operation" | "bubble" | "error_type">;
/**
 * Tracks validation errors
 * Labels: bubble
 */
export declare const bubbleValidationErrorTotal: Counter<"bubble" | "validation_error_type">;
/**
 * Tracks authentication errors
 * Labels: bubble
 */
export declare const bubbleAuthenticationErrorTotal: Counter<"bubble">;
/**
 * Tracks blocked SQL injection attempts
 * Labels: bubble
 */
export declare const sqlInjectionBlockedTotal: Counter<"bubble">;
/**
 * Tracks blocked XSS attempts
 * Labels: bubble
 */
export declare const xssBlockedTotal: Counter<"bubble">;
/**
 * Tracks unauthorized access attempts
 * Labels: bubble
 */
export declare const unauthorizedAccessTotal: Counter<"bubble">;
/**
 * Tracks request size in bytes
 * Labels: bubble, operation
 */
export declare const bubbleRequestSizeBytes: Histogram<"operation" | "bubble">;
/**
 * Tracks response size in bytes
 * Labels: bubble, operation
 */
export declare const bubbleResponseSizeBytes: Histogram<"operation" | "bubble">;
/**
 * Tracks memory usage
 * Labels: bubble
 */
export declare const bubbleMemoryUsageBytes: Gauge<"bubble">;
/**
 * Tracks database connection pool usage
 * Labels: bubble
 */
export declare const dbConnectionPoolUsage: Gauge<"bubble" | "pool_type">;
/**
 * Tracks the number of currently active operations
 * Labels: bubble
 */
export declare const bubbleActiveOperations: Gauge<"bubble">;
/**
 * Tracks throughput (operations per second)
 * Labels: bubble
 */
export declare const bubbleThroughputPerSecond: Gauge<"bubble">;
/**
 * Tracks active workflows
 * Labels: bubble
 */
export declare const activeWorkflows: Gauge<"bubble">;
/**
 * Get metrics in Prometheus format
 */
export declare function getMetrics(): Promise<string>;
/**
 * Get metrics as JSON for custom processing
 */
export declare function getMetricsAsJson(): Promise<Record<string, unknown>>;
/**
 * Clear all metrics (useful for testing)
 */
export declare function clearMetrics(): void;
/**
 * Reset a specific metric
 */
export declare function resetMetric(metric: Counter | Histogram | Gauge): void;
/**
 * Decorator to automatically track operation metrics
 */
export declare function trackOperation(bubbleName: string, operationName: string): (target: unknown, propertyKey: string, descriptor: PropertyDescriptor) => PropertyDescriptor;
/**
 * Record an error metric
 */
export declare function recordError(bubbleName: string, errorType: string, operation: string): void;
/**
 * Record a validation error
 */
export declare function recordValidationError(bubbleName: string, errorType: string): void;
/**
 * Update circuit breaker state
 */
export declare function updateCircuitBreakerState(bubbleName: string, state: 'closed' | 'open' | 'half_open'): void;
/**
 * Record a circuit breaker failure
 */
export declare function recordCircuitBreakerFailure(bubbleName: string): void;
/**
 * Record a circuit breaker success
 */
export declare function recordCircuitBreakerSuccess(bubbleName: string): void;
export { register };
export default register;
//# sourceMappingURL=prometheus.d.ts.map