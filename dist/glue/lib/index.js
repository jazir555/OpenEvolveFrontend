"use strict";
/**
 * Shared Utilities Library
 *
 * Federation Constitution compliant utilities for the Glue Layer.
 *
 * Modules:
 * - logger: Structured JSON Lines logging with UTC timestamps
 * - retry: Exponential backoff with jitter for transient failures
 * - circuit-breaker: Circuit breaker pattern for system failures
 * - env-validator: Environment variable validation with type checking
 * - metrics: Prometheus metrics collection and monitoring
 *
 * Usage:
 * ```typescript
 * import { logger, retryWithBackoff, CircuitBreaker, validateEnv, initializeMonitoring } from './lib';
 * ```
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.createRequestTrackingMiddleware = exports.createHealthMiddleware = exports.createMetricsMiddleware = exports.getAlertManager = exports.AlertManager = exports.getTracer = exports.Tracer = exports.HealthEndpointHandler = exports.HealthChecker = exports.getMetricsCollector = exports.MetricsCollector = exports.initializeMonitoring = exports.getEnv = exports.validateEnvWithTypes = exports.validateEnv = exports.CircuitState = exports.CircuitBreaker = exports.retryWithBackoff = exports.LogLevel = exports.logger = exports.Logger = void 0;
// Logger
var logger_1 = require("./logger");
Object.defineProperty(exports, "Logger", { enumerable: true, get: function () { return logger_1.Logger; } });
Object.defineProperty(exports, "logger", { enumerable: true, get: function () { return logger_1.logger; } });
Object.defineProperty(exports, "LogLevel", { enumerable: true, get: function () { return logger_1.LogLevel; } });
// Retry
var retry_1 = require("./retry");
Object.defineProperty(exports, "retryWithBackoff", { enumerable: true, get: function () { return retry_1.retryWithBackoff; } });
// Circuit Breaker
var circuit_breaker_1 = require("./circuit-breaker");
Object.defineProperty(exports, "CircuitBreaker", { enumerable: true, get: function () { return circuit_breaker_1.CircuitBreaker; } });
Object.defineProperty(exports, "CircuitState", { enumerable: true, get: function () { return circuit_breaker_1.CircuitState; } });
// Environment Validator
var env_validator_1 = require("./env-validator");
Object.defineProperty(exports, "validateEnv", { enumerable: true, get: function () { return env_validator_1.validateEnv; } });
Object.defineProperty(exports, "validateEnvWithTypes", { enumerable: true, get: function () { return env_validator_1.validateEnvWithTypes; } });
Object.defineProperty(exports, "getEnv", { enumerable: true, get: function () { return env_validator_1.getEnv; } });
// Metrics and Monitoring
var metrics_1 = require("./metrics");
Object.defineProperty(exports, "initializeMonitoring", { enumerable: true, get: function () { return metrics_1.initializeMonitoring; } });
Object.defineProperty(exports, "MetricsCollector", { enumerable: true, get: function () { return metrics_1.MetricsCollector; } });
Object.defineProperty(exports, "getMetricsCollector", { enumerable: true, get: function () { return metrics_1.getMetricsCollector; } });
Object.defineProperty(exports, "HealthChecker", { enumerable: true, get: function () { return metrics_1.HealthChecker; } });
Object.defineProperty(exports, "HealthEndpointHandler", { enumerable: true, get: function () { return metrics_1.HealthEndpointHandler; } });
Object.defineProperty(exports, "Tracer", { enumerable: true, get: function () { return metrics_1.Tracer; } });
Object.defineProperty(exports, "getTracer", { enumerable: true, get: function () { return metrics_1.getTracer; } });
Object.defineProperty(exports, "AlertManager", { enumerable: true, get: function () { return metrics_1.AlertManager; } });
Object.defineProperty(exports, "getAlertManager", { enumerable: true, get: function () { return metrics_1.getAlertManager; } });
Object.defineProperty(exports, "createMetricsMiddleware", { enumerable: true, get: function () { return metrics_1.createMetricsMiddleware; } });
Object.defineProperty(exports, "createHealthMiddleware", { enumerable: true, get: function () { return metrics_1.createHealthMiddleware; } });
Object.defineProperty(exports, "createRequestTrackingMiddleware", { enumerable: true, get: function () { return metrics_1.createRequestTrackingMiddleware; } });
/**
 * Complete example combining all utilities:
 *
 * ```typescript
 * import { logger, retryWithBackoff, CircuitBreaker, validateEnvWithTypes } from './lib';
 *
 * // 1. Validate environment at startup (crashes if invalid)
 * const config = validateEnvWithTypes([
 *   { name: 'TARGET_API_URL', type: 'url', required: true },
 *   { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
 *   { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
 * ]);
 *
 * // 2. Create circuit breaker for external service
 * const circuitBreaker = new CircuitBreaker({
 *   threshold: 5,
 *   timeout_ms: 60000,
 *   onStateChange: (old, newState) => {
 *     logger.warn('Circuit breaker state changed', {
 *       old_state: old,
 *       new_state: newState,
 *     });
 *   },
 * });
 *
 * // 3. Use all utilities together
 * async function callExternalService(data: any) {
 *   const correlationId = generateCorrelationId();
 *
 *   return retryWithBackoff(
 *     async () => {
 *       logger.info('Calling external service', {
 *         correlation_id: correlationId,
 *         target_service: 'external-api',
 *       });
 *
 *       return circuitBreaker.execute(async () => {
 *         const response = await fetch(config.TARGET_API_URL, {
 *           method: 'POST',
 *           headers: { 'Content-Type': 'application/json' },
 *           body: JSON.stringify(data),
 *           signal: AbortSignal.timeout(config.TIMEOUT_MS),
 *         });
 *
 *         if (!response.ok) {
 *           throw new Error(`HTTP ${response.status}: ${response.statusText}`);
 *         }
 *
 *         return response.json();
 *       });
 *     },
 *     {
 *       max_retries: config.MAX_RETRIES,
 *       base_delay_ms: 1000,
 *       max_delay_ms: 10000,
 *       jitter_ms: 500,
 *       onRetry: (attempt, error) => {
 *         logger.warn('Retry attempt', {
 *           correlation_id: correlationId,
 *           attempt,
 *           error_message: error.message,
 *         });
 *       },
 *     }
 *   );
 * }
 * ```
 */
//# sourceMappingURL=index.js.map