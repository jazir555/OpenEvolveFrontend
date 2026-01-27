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
import { Registry, Counter, Histogram, Gauge, collectDefaultMetrics } from 'prom-client';
// Create a custom registry for BubbleLab metrics
const register = new Registry();
// Collect default metrics (CPU, memory, etc.)
collectDefaultMetrics({ register });
// ============================================================================
// OPERATION METRICS
// ============================================================================
/**
 * Tracks the duration of bubble operations
 * Labels: bubble, operation, status
 */
export const bubbleOperationDuration = new Histogram({
    name: 'bubble_operation_duration_seconds',
    help: 'Duration of bubble operations in seconds',
    labelNames: ['bubble', 'operation', 'status'],
    buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 300],
    registers: [register]
});
/**
 * Tracks the total number of operations
 * Labels: bubble, operation, status
 */
export const bubbleOperationTotal = new Counter({
    name: 'bubble_operation_total',
    help: 'Total number of bubble operations',
    labelNames: ['bubble', 'operation', 'status'],
    registers: [register]
});
/**
 * Tracks the number of operation retries
 * Labels: bubble, operation
 */
export const bubbleOperationRetryTotal = new Counter({
    name: 'bubble_operation_retry_total',
    help: 'Total number of operation retries',
    labelNames: ['bubble', 'operation'],
    registers: [register]
});
// ============================================================================
// CIRCUIT BREAKER METRICS
// ============================================================================
/**
 * Tracks the current state of circuit breakers
 * Labels: bubble, state (0=closed, 1=open, 2=half_open)
 */
export const circuitBreakerState = new Gauge({
    name: 'circuit_breaker_state',
    help: 'Current state of circuit breakers (0=closed, 1=open, 2=half_open)',
    labelNames: ['bubble', 'state'],
    registers: [register]
});
/**
 * Tracks the number of circuit breaker failures
 * Labels: bubble
 */
export const circuitBreakerFailureTotal = new Counter({
    name: 'circuit_breaker_failure_total',
    help: 'Total number of circuit breaker failures',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks the number of circuit breaker successes
 * Labels: bubble
 */
export const circuitBreakerSuccessTotal = new Counter({
    name: 'circuit_breaker_success_total',
    help: 'Total number of circuit breaker successes',
    labelNames: ['bubble'],
    registers: [register]
});
// ============================================================================
// RATE LIMITING METRICS
// ============================================================================
/**
 * Tracks the number of rate limit violations
 * Labels: bubble
 */
export const rateLimitExceededTotal = new Counter({
    name: 'rate_limit_exceeded_total',
    help: 'Total number of rate limit violations',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks the remaining rate limit quota
 * Labels: bubble
 */
export const rateLimitRemaining = new Gauge({
    name: 'rate_limit_remaining',
    help: 'Remaining rate limit quota',
    labelNames: ['bubble'],
    registers: [register]
});
// ============================================================================
// ERROR METRICS
// ============================================================================
/**
 * Tracks the total number of errors
 * Labels: bubble, error_type, operation
 */
export const bubbleErrorTotal = new Counter({
    name: 'bubble_error_total',
    help: 'Total number of bubble errors',
    labelNames: ['bubble', 'error_type', 'operation'],
    registers: [register]
});
/**
 * Tracks validation errors
 * Labels: bubble
 */
export const bubbleValidationErrorTotal = new Counter({
    name: 'bubble_validation_error_total',
    help: 'Total number of validation errors',
    labelNames: ['bubble', 'validation_error_type'],
    registers: [register]
});
/**
 * Tracks authentication errors
 * Labels: bubble
 */
export const bubbleAuthenticationErrorTotal = new Counter({
    name: 'bubble_authentication_error_total',
    help: 'Total number of authentication errors',
    labelNames: ['bubble'],
    registers: [register]
});
// ============================================================================
// SECURITY METRICS
// ============================================================================
/**
 * Tracks blocked SQL injection attempts
 * Labels: bubble
 */
export const sqlInjectionBlockedTotal = new Counter({
    name: 'sql_injection_blocked_total',
    help: 'Total number of blocked SQL injection attempts',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks blocked XSS attempts
 * Labels: bubble
 */
export const xssBlockedTotal = new Counter({
    name: 'xss_blocked_total',
    help: 'Total number of blocked XSS attempts',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks unauthorized access attempts
 * Labels: bubble
 */
export const unauthorizedAccessTotal = new Counter({
    name: 'unauthorized_access_total',
    help: 'Total number of unauthorized access attempts',
    labelNames: ['bubble'],
    registers: [register]
});
// ============================================================================
// PERFORMANCE METRICS
// ============================================================================
/**
 * Tracks request size in bytes
 * Labels: bubble, operation
 */
export const bubbleRequestSizeBytes = new Histogram({
    name: 'bubble_request_size_bytes',
    help: 'Size of requests in bytes',
    labelNames: ['bubble', 'operation'],
    buckets: [100, 1000, 10000, 100000, 1000000, 10000000],
    registers: [register]
});
/**
 * Tracks response size in bytes
 * Labels: bubble, operation
 */
export const bubbleResponseSizeBytes = new Histogram({
    name: 'bubble_response_size_bytes',
    help: 'Size of responses in bytes',
    labelNames: ['bubble', 'operation'],
    buckets: [100, 1000, 10000, 100000, 1000000, 10000000],
    registers: [register]
});
/**
 * Tracks memory usage
 * Labels: bubble
 */
export const bubbleMemoryUsageBytes = new Gauge({
    name: 'bubble_memory_usage_bytes',
    help: 'Memory usage in bytes',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks database connection pool usage
 * Labels: bubble
 */
export const dbConnectionPoolUsage = new Gauge({
    name: 'db_connection_pool_usage',
    help: 'Database connection pool usage',
    labelNames: ['bubble', 'pool_type'],
    registers: [register]
});
// ============================================================================
// BUSINESS METRICS
// ============================================================================
/**
 * Tracks the number of currently active operations
 * Labels: bubble
 */
export const bubbleActiveOperations = new Gauge({
    name: 'bubble_active_operations',
    help: 'Number of currently active operations',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks throughput (operations per second)
 * Labels: bubble
 */
export const bubbleThroughputPerSecond = new Gauge({
    name: 'bubble_throughput_per_second',
    help: 'Operations per second',
    labelNames: ['bubble'],
    registers: [register]
});
/**
 * Tracks active workflows
 * Labels: bubble
 */
export const activeWorkflows = new Gauge({
    name: 'active_workflows',
    help: 'Number of active workflows',
    labelNames: ['bubble'],
    registers: [register]
});
// ============================================================================
// METRICS COLLECTION HELPERS
// ============================================================================
/**
 * Get metrics in Prometheus format
 */
export async function getMetrics() {
    return await register.metrics();
}
/**
 * Get metrics as JSON for custom processing
 */
export async function getMetricsAsJson() {
    const metrics = await register.getMetricsAsJSON();
    return metrics;
}
/**
 * Clear all metrics (useful for testing)
 */
export function clearMetrics() {
    register.clear();
}
/**
 * Reset a specific metric
 */
export function resetMetric(metric) {
    metric.reset();
}
// ============================================================================
// OPERATION TRACKING DECORATOR
// ============================================================================
/**
 * Decorator to automatically track operation metrics
 */
export function trackOperation(bubbleName, operationName) {
    return function (target, propertyKey, descriptor) {
        const originalMethod = descriptor.value;
        descriptor.value = async function (...args) {
            const start = Date.now();
            let status = 'success';
            // Increment active operations
            bubbleActiveOperations.inc({ [bubbleName]: 1 });
            try {
                const result = await originalMethod.apply(this, args);
                return result;
            }
            catch (error) {
                status = 'error';
                throw error;
            }
            finally {
                const duration = (Date.now() - start) / 1000;
                // Record metrics
                bubbleOperationDuration.observe({ bubble: bubbleName, operation: operationName, status }, duration);
                bubbleOperationTotal.inc({
                    bubble: bubbleName,
                    operation: operationName,
                    status
                });
                // Decrement active operations
                bubbleActiveOperations.dec({ [bubbleName]: 1 });
            }
        };
        return descriptor;
    };
}
// ============================================================================
// ERROR TRACKING HELPER
// ============================================================================
/**
 * Record an error metric
 */
export function recordError(bubbleName, errorType, operation) {
    bubbleErrorTotal.inc({
        bubble: bubbleName,
        error_type: errorType,
        operation
    });
}
/**
 * Record a validation error
 */
export function recordValidationError(bubbleName, errorType) {
    bubbleValidationErrorTotal.inc({
        bubble: bubbleName,
        validation_error_type: errorType
    });
}
// ============================================================================
// CIRCUIT BREAKER TRACKING
// ============================================================================
/**
 * Update circuit breaker state
 */
export function updateCircuitBreakerState(bubbleName, state) {
    // Reset all states for this bubble
    circuitBreakerState.reset();
    // Set the current state
    const stateValue = state === 'closed' ? 0 : state === 'open' ? 1 : 2;
    circuitBreakerState.set({ bubble: bubbleName, state }, stateValue);
}
/**
 * Record a circuit breaker failure
 */
export function recordCircuitBreakerFailure(bubbleName) {
    circuitBreakerFailureTotal.inc({ bubble: bubbleName });
}
/**
 * Record a circuit breaker success
 */
export function recordCircuitBreakerSuccess(bubbleName) {
    circuitBreakerSuccessTotal.inc({ bubble: bubbleName });
}
// ============================================================================
// EXPORT REGISTRY FOR EXTERNAL USE
// ============================================================================
export { register };
export default register;
//# sourceMappingURL=prometheus.js.map