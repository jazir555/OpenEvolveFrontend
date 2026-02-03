/**
 * Complete Example: Using All Glue Lib Utilities
 *
 * This example demonstrates how to use all utilities together
 * in a production adapter following the Federation Constitution.
 */

import {
  logger,
  retryWithBackoff,
  CircuitBreaker,
  CircuitState,
  validateEnvWithTypes,
  type LoggerContext,
} from './index';

// ============================================================================
// 1. ENVIRONMENT VALIDATION (Startup)
// ============================================================================
// Law of Configuration Explicitness: Crash immediately if missing config

const config = validateEnvWithTypes([
  { name: 'TARGET_API_URL', type: 'url', required: true },
  { name: 'TIMEOUT_MS', type: 'number', required: false, default: 5000 },
  { name: 'MAX_RETRIES', type: 'number', required: false, default: 3 },
  { name: 'CIRCUIT_THRESHOLD', type: 'number', required: false, default: 5 },
  { name: 'SERVICE_NAME', type: 'string', required: false, default: 'adapter' },
]);

logger.info('Configuration validated', {
  service_name: config.SERVICE_NAME,
  target_api: config.TARGET_API_URL,
});

// ============================================================================
// 2. CIRCUIT BREAKER SETUP
// ============================================================================
// System Failure Management: Stop hammering dead services

const circuitBreaker = new CircuitBreaker({
  threshold: config.CIRCUIT_THRESHOLD,
  timeout_ms: 60000, // 1 minute
  onStateChange: (oldState, newState) => {
    logger.warn('Circuit breaker state changed', {
      old_state: oldState,
      new_state: newState,
      service: 'target-api',
    });

    // If circuit opens, alert ops team
    if (newState === CircuitState.OPEN) {
      logger.error('Circuit opened - service is down', undefined, {
        service: 'target-api',
        action: 'alert_ops',
      });
    }
  },
});

// ============================================================================
// 3. MAKE RESILIENT API CALLS
// ============================================================================
// Transient Failure Management: Exponential backoff with jitter

interface ApiRequest {
  data: any;
  correlation_id?: string;
}

async function callExternalService(req: ApiRequest): Promise<any> {
  const correlationId = req.correlation_id || generateCorrelationId();

  const context: LoggerContext = {
    correlation_id: correlationId,
    source_service: config.SERVICE_NAME,
    target_service: 'external-api',
  };

  logger.info('Starting external API call', context);

  try {
    const result = await retryWithBackoff(
      async () => {
        logger.info('Attempting API call', context);

        return circuitBreaker.execute(async () => {
          const controller = new AbortController();
          const timeoutId = setTimeout(
            () => controller.abort(),
            config.TIMEOUT_MS
          );

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
              throw new Error(
                `HTTP ${response.status}: ${response.statusText}`
              );
            }

            const data = await response.json();

            logger.info('API call successful', {
              ...context,
              response_status: response.status,
            });

            return data;
          } finally {
            clearTimeout(timeoutId);
          }
        });
      },
      {
        max_retries: config.MAX_RETRIES,
        base_delay_ms: 1000,
        max_delay_ms: 10000,
        jitter_ms: 500,
        onRetry: (attempt, error) => {
          logger.warn('Retry attempt scheduled', {
            ...context,
            attempt,
            error_message: error.message,
            next_retry_in_ms: 1000 * Math.pow(2, attempt - 1),
          });
        },
      }
    );

    return result;
  } catch (error) {
    // All retries exhausted or circuit is open
    const isCircuitOpen = circuitBreaker.getState() === CircuitState.OPEN;

    logger.error('API call failed after all retries', error as Error, {
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

async function syncUser(userId: string): Promise<void> {
  const correlationId = `sync-user-${userId}-${Date.now()}`;

  logger.info('Starting user sync', {
    correlation_id: correlationId,
    user_id: userId,
  });

  try {
    const result = await callExternalService({
      data: { user_id: userId, action: 'sync' },
      correlation_id: correlationId,
    });

    logger.info('User sync completed', {
      correlation_id: correlationId,
      user_id: userId,
      result: result,
    });
  } catch (error) {
    // Circuit is open - system is down, don't retry immediately
    if (circuitBreaker.getState() === CircuitState.OPEN) {
      logger.error('System is down, scheduling retry', error as Error, {
        correlation_id: correlationId,
        user_id: userId,
        retry_strategy: 'exponential_backoff',
        retry_after_ms: 60000,
      });

      // In production, send to Dead Letter Queue or schedule retry
      return;
    }

    // Logic failure - bad data, don't retry
    logger.error('User sync failed - bad data?', error as Error, {
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

async function healthCheck(): Promise<{ healthy: boolean; details: any }> {
  const stats = circuitBreaker.getStats();

  logger.info('Health check performed', {
    circuit_state: stats.state,
    failure_count: stats.failure_count,
    success_count: stats.success_count,
  });

  return {
    healthy: stats.state !== CircuitState.OPEN,
    details: stats,
  };
}

// ============================================================================
// 6. GRACEFUL SHUTDOWN
// ============================================================================

async function shutdown(): Promise<void> {
  logger.info('Shutting down gracefully', {
    circuit_state: circuitBreaker.getState(),
  });

  // Wait for in-flight requests
  // Close connections
  // Clear resources

  logger.info('Shutdown complete');
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function generateCorrelationId(): string {
  return `evt-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// ============================================================================
// EXAMPLE OUTPUT
// ============================================================================

/*
{"level":"info","msg":"Configuration validated","timestamp":"2025-01-15T10:30:00.000Z","correlation_id":"a1b2c3d4-...","source_service":"adapter","service_name":"my-adapter","target_api":"http://api:8000"}

{"level":"info","msg":"Starting user sync","timestamp":"2025-01-15T10:30:01.000Z","correlation_id":"sync-user-123-1736945401000","source_service":"adapter","user_id":"123"}

{"level":"info","msg":"Starting external API call","timestamp":"2025-01-15T10:30:01.100Z","correlation_id":"sync-user-123-1736945401000","source_service":"adapter","target_service":"external-api"}

{"level":"info","msg":"Attempting API call","timestamp":"2025-01-15T10:30:01.200Z","correlation_id":"sync-user-123-1736945401000","source_service":"adapter","target_service":"external-api"}

{"level":"warn","msg":"Retry attempt scheduled","timestamp":"2025-01-15T10:30:02.500Z","correlation_id":"sync-user-123-1736945401000","attempt":1,"error_message":"Connection timeout","next_retry_in_ms":1000}

{"level":"warn","msg":"Circuit breaker state changed","timestamp":"2025-01-15T10:30:05.000Z","old_state":"closed","new_state":"open","service":"target-api"}

{"level":"error","msg":"Circuit opened - service is down","timestamp":"2025-01-15T10:30:05.100Z","correlation_id":"x1y2z3a4-...","service":"target-api","action":"alert_ops"}

{"level":"error","msg":"API call failed after all retries","timestamp":"2025-01-15T10:30:05.200Z","correlation_id":"sync-user-123-1736945401000","source_service":"adapter","target_service":"external-api","circuit_state":"open","is_circuit_open":true,"failure_type":"system","error_name":"Error","error_message":"Circuit breaker is OPEN"}
*/

export { callExternalService, syncUser, healthCheck, shutdown };
