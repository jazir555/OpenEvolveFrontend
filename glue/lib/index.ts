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

// Logger
export {
  Logger,
  logger,
  LogLevel,
  type LoggerContext,
  type LogEntry,
} from './logger';

// Retry
export {
  retryWithBackoff,
  type RetryOptions,
  type RetryConfig,
} from './retry';

// Circuit Breaker
export {
  CircuitBreaker,
  CircuitState,
  type CircuitBreakerOptions,
  type CircuitBreakerStats,
} from './circuit-breaker';

// Environment Validator
export {
  validateEnv,
  validateEnvWithTypes,
  getEnv,
  type EnvType,
  type EnvVar,
  type ValidationResult,
} from './env-validator';

// Metrics and Monitoring
export {
  initializeMonitoring,
  MetricsCollector,
  getMetricsCollector,
  HealthChecker,
  HealthEndpointHandler,
  Tracer,
  getTracer,
  AlertManager,
  getAlertManager,
  createMetricsMiddleware,
  createHealthMiddleware,
  createRequestTrackingMiddleware,
  // Types
  type MetricsLabels,
  type KnowledgeExtractionLabels,
  type HealthStatus,
  type HealthCheckResult,
  type HealthCheckOptions,
  type HealthCheckFunction,
  type TraceOptions,
  type SpanMetadata,
  type AlertSeverity,
  type AlertRule,
  type AlertCondition,
  type NotificationChannel,
  type Alert,
  type AlertHistory,
  type MonitoringConfig,
} from './metrics';

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
