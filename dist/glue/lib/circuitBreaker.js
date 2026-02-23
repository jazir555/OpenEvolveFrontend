"use strict";
/**
 * Circuit Breaker Pattern Implementation
 * Per CLAUDE.md Section 2.3: System Failure → Circuit Breaker
 * Stop hammering the dead service. Wait for a health check to pass.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.circuitBreakerRegistry = exports.CircuitBreakerRegistry = exports.CircuitBreakerOpenError = exports.CircuitBreaker = exports.CircuitState = void 0;
const structuredLogger_1 = require("./structuredLogger");
var CircuitState;
(function (CircuitState) {
    CircuitState["CLOSED"] = "closed";
    CircuitState["OPEN"] = "open";
    CircuitState["HALF_OPEN"] = "half-open"; // Testing if service has recovered
})(CircuitState || (exports.CircuitState = CircuitState = {}));
class CircuitBreaker {
    constructor(serviceName, config) {
        this.serviceName = serviceName;
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
        this.config = {
            failureThreshold: config.failureThreshold || 5,
            successThreshold: config.successThreshold || 2,
            timeoutMs: config.timeoutMs || 60000, // 1 minute default
            monitoringPeriodMs: config.monitoringPeriodMs || 10000 // 10 seconds default
        };
    }
    /**
     * Execute operation with circuit breaker protection
     */
    async execute(operation, context) {
        const correlationId = context?.correlation_id ||
            `cb-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const cbContext = {
            ...context,
            correlation_id: correlationId,
            source_service: 'circuit-breaker',
            target_service: this.serviceName,
            circuit_state: this.state
        };
        // Check if circuit is open
        if (this.state === CircuitState.OPEN) {
            if (Date.now() < (this.nextAttemptAt || 0)) {
                // Circuit is still open, reject request
                structuredLogger_1.apiLogger.error('Circuit breaker is open, rejecting request', undefined, cbContext);
                throw new CircuitBreakerOpenError(`Circuit breaker is open for ${this.serviceName}. ` +
                    `Will retry after ${new Date(this.nextAttemptAt || 0).toISOString()}`);
            }
            // Timeout has elapsed, attempt recovery
            structuredLogger_1.apiLogger.info('Circuit breaker timeout elapsed, entering half-open state', cbContext);
            this.state = CircuitState.HALF_OPEN;
            this.successCount = 0;
        }
        // Execute operation
        try {
            const result = await operation();
            this.onSuccess(cbContext);
            return result;
        }
        catch (error) {
            this.onFailure(error, cbContext);
            throw error;
        }
    }
    /**
     * Handle successful operation
     */
    onSuccess(context) {
        this.lastSuccessTime = Date.now();
        if (this.state === CircuitState.HALF_OPEN) {
            this.successCount++;
            structuredLogger_1.apiLogger.info('Circuit breaker half-open success', {
                ...context,
                success_count: this.successCount,
                required: this.config.successThreshold
            });
            if (this.successCount >= this.config.successThreshold) {
                // Service has recovered, close circuit
                this.closeCircuit(context);
            }
        }
        else {
            // Reset failure count on success in closed state
            this.failureCount = Math.max(0, this.failureCount - 1);
        }
    }
    /**
     * Handle failed operation
     */
    onFailure(error, context) {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        structuredLogger_1.apiLogger.warn('Circuit breaker operation failed', {
            ...context,
            failure_count: this.failureCount,
            threshold: this.config.failureThreshold,
            error: error.message
        });
        if (this.state === CircuitState.HALF_OPEN) {
            // Service still failing, reopen circuit
            this.openCircuit(context);
        }
        else if (this.failureCount >= this.config.failureThreshold) {
            // Too many failures, open circuit
            this.openCircuit(context);
        }
    }
    /**
     * Open circuit (stop accepting requests)
     */
    openCircuit(context) {
        this.state = CircuitState.OPEN;
        this.openedAt = Date.now();
        this.nextAttemptAt = Date.now() + this.config.timeoutMs;
        structuredLogger_1.apiLogger.error('Circuit breaker opened', {
            ...context,
            failure_count: this.failureCount,
            opened_at: new Date(this.openedAt).toISOString(),
            next_attempt_at: new Date(this.nextAttemptAt).toISOString(),
            service: this.serviceName
        });
    }
    /**
     * Close circuit (resume normal operation)
     */
    closeCircuit(context) {
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
        this.openedAt = undefined;
        this.nextAttemptAt = undefined;
        structuredLogger_1.apiLogger.info('Circuit breaker closed', {
            ...context,
            service: this.serviceName
        });
    }
    /**
     * Get current circuit breaker state
     */
    getState() {
        return {
            state: this.state,
            failureCount: this.failureCount,
            successCount: this.successCount,
            lastFailureTime: this.lastFailureTime,
            lastSuccessTime: this.lastSuccessTime,
            openedAt: this.openedAt,
            nextAttemptAt: this.nextAttemptAt
        };
    }
    /**
     * Reset circuit breaker to closed state
     */
    reset() {
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
        this.openedAt = undefined;
        this.nextAttemptAt = undefined;
        structuredLogger_1.apiLogger.info('Circuit breaker manually reset', {
            service: this.serviceName
        });
    }
}
exports.CircuitBreaker = CircuitBreaker;
/**
 * Custom error for circuit breaker open state
 */
class CircuitBreakerOpenError extends Error {
    constructor(message) {
        super(message);
        this.name = 'CircuitBreakerOpenError';
    }
}
exports.CircuitBreakerOpenError = CircuitBreakerOpenError;
/**
 * Circuit breaker registry for managing multiple circuit breakers
 */
class CircuitBreakerRegistry {
    constructor() {
        this.breakers = new Map();
    }
    /**
     * Get or create circuit breaker for a service
     */
    get(serviceName, config) {
        if (!this.breakers.has(serviceName)) {
            const defaultConfig = {
                failureThreshold: 5,
                successThreshold: 2,
                timeoutMs: 60000,
                monitoringPeriodMs: 10000
            };
            this.breakers.set(serviceName, new CircuitBreaker(serviceName, config || defaultConfig));
        }
        return this.breakers.get(serviceName);
    }
    /**
     * Get stats for all circuit breakers
     */
    getAllStats() {
        const stats = new Map();
        this.breakers.forEach((breaker, serviceName) => {
            stats.set(serviceName, breaker.getState());
        });
        return stats;
    }
    /**
     * Reset all circuit breakers
     */
    resetAll() {
        this.breakers.forEach((breaker) => breaker.reset());
    }
    /**
     * Remove circuit breaker for a service
     */
    remove(serviceName) {
        this.breakers.delete(serviceName);
    }
}
exports.CircuitBreakerRegistry = CircuitBreakerRegistry;
// Global circuit breaker registry
exports.circuitBreakerRegistry = new CircuitBreakerRegistry();
//# sourceMappingURL=circuitBreaker.js.map