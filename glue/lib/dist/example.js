"use strict";
/**
 * Complete Example: Using All Glue Lib Utilities
 *
 * This example demonstrates how to use all utilities together
 * in a production adapter following the Federation Constitution.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.callExternalService = callExternalService;
exports.syncUser = syncUser;
exports.healthCheck = healthCheck;
exports.shutdown = shutdown;
const index_1 = require("./index");
// ============================================================================
// 1. ENVIRONMENT VALIDATION (Startup)
// ============================================================================
// Law of Configuration Explicitness: Crash immediately if missing config
const config = (0, index_1.validateEnvWithTypes)([
    { name: 'TARGET_API_URL', type: 'url', required: true },
    { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
    { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
    { name: 'CIRCUIT_THRESHOLD', type: 'number', required: false, default: 5 },
    { name: 'SERVICE_NAME', type: 'string', required: false, default: 'adapter' },
]);
index_1.logger.info('Configuration validated', {
    service_name: config.SERVICE_NAME,
    target_api: config.TARGET_API_URL,
});
// ============================================================================
// 2. CIRCUIT BREAKER SETUP
// ============================================================================
// System Failure Management: Stop hammering dead services
const circuitBreaker = new index_1.CircuitBreaker({
    threshold: config.CIRCUIT_THRESHOLD,
    timeout_ms: 60000, // 1 minute
    onStateChange: (oldState, newState) => {
        index_1.logger.warn('Circuit breaker state changed', {
            old_state: oldState,
            new_state: newState,
            service: 'target-api',
        });
        // If circuit opens, alert ops team
        if (newState === index_1.CircuitState.OPEN) {
            index_1.logger.error('Circuit opened - service is down', undefined, {
                service: 'target-api',
                action: 'alert_ops',
            });
        }
    },
});
async function callExternalService(req) {
    const correlationId = req.correlation_id || generateCorrelationId();
    const context = {
        correlation_id: correlationId,
        source_service: config.SERVICE_NAME,
        target_service: 'external-api',
    };
    index_1.logger.info('Starting external API call', context);
    try {
        const result = await (0, index_1.retryWithBackoff)(async () => {
            index_1.logger.info('Attempting API call', context);
            return circuitBreaker.execute(async () => {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), config.TIMEOUT_MS);
                try {
                    const response = await fetch(config.TARGET_API_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Correlation-ID': correlationId,
                        },
                        body: JSON.stringify(req.data),
                        signal: controller.signal,
                    });
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    const data = await response.json();
                    index_1.logger.info('API call successful', {
                        ...context,
                        response_status: response.status,
                    });
                    return data;
                }
                finally {
                    clearTimeout(timeoutId);
                }
            });
        }, {
            max_retries: config.MAX_RETRIES,
            base_delay_ms: 1000,
            max_delay_ms: 10000,
            jitter_ms: 500,
            onRetry: (attempt, error) => {
                index_1.logger.warn('Retry attempt scheduled', {
                    ...context,
                    attempt,
                    error_message: error.message,
                    next_retry_in_ms: 1000 * Math.pow(2, attempt - 1),
                });
            },
        });
        return result;
    }
    catch (error) {
        // All retries exhausted or circuit is open
        const isCircuitOpen = circuitBreaker.getState() === index_1.CircuitState.OPEN;
        index_1.logger.error('API call failed after all retries', error, {
            ...context,
            circuit_state: circuitBreaker.getState(),
            is_circuit_open: isCircuitOpen,
            failure_type: isCircuitOpen ? 'system' : 'transient',
        });
        // If circuit is open, this is a system failure
        // If retries exhausted, this might be a logic failure (bad data)
        throw error;
    }
}
// ============================================================================
// 4. BUSINESS LOGIC
// ============================================================================
async function syncUser(userId) {
    const correlationId = `sync-user-${userId}-${Date.now()}`;
    index_1.logger.info('Starting user sync', {
        correlation_id: correlationId,
        user_id: userId,
    });
    try {
        const result = await callExternalService({
            data: { user_id: userId, action: 'sync' },
            correlation_id: correlationId,
        });
        index_1.logger.info('User sync completed', {
            correlation_id: correlationId,
            user_id: userId,
            result: result,
        });
    }
    catch (error) {
        // Circuit is open - system is down, don't retry immediately
        if (circuitBreaker.getState() === index_1.CircuitState.OPEN) {
            index_1.logger.error('System is down, scheduling retry', error, {
                correlation_id: correlationId,
                user_id: userId,
                retry_strategy: 'exponential_backoff',
                retry_after_ms: 60000,
            });
            // In production, send to Dead Letter Queue or schedule retry
            return;
        }
        // Logic failure - bad data, don't retry
        index_1.logger.error('User sync failed - bad data?', error, {
            correlation_id: correlationId,
            user_id: userId,
            action: 'send_to_dlq',
        });
        // Send to Dead Letter Queue for manual inspection
    }
}
// ============================================================================
// 5. HEALTH CHECK
// ============================================================================
async function healthCheck() {
    const stats = circuitBreaker.getStats();
    index_1.logger.info('Health check performed', {
        circuit_state: stats.state,
        failure_count: stats.failure_count,
        success_count: stats.success_count,
    });
    return {
        healthy: stats.state !== index_1.CircuitState.OPEN,
        details: stats,
    };
}
// ============================================================================
// 6. GRACEFUL SHUTDOWN
// ============================================================================
async function shutdown() {
    index_1.logger.info('Shutting down gracefully', {
        circuit_state: circuitBreaker.getState(),
    });
    // Wait for in-flight requests
    // Close connections
    // Clear resources
    index_1.logger.info('Shutdown complete');
}
// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
function generateCorrelationId() {
    return `evt-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
//# sourceMappingURL=example.js.map