import { ApiClient, ApiClientConfig } from '@/lib/api';
import { EVOLUTION_API_BASE_URL } from '@/env';
import { createEvolutionApiCircuitBreaker } from '@/lib/circuitBreaker';
import type {
  EvolutionStartPayload,
  EvolutionStartResponse,
} from '@/types/evolution';

// Create ApiClient with retry and timeout configuration
const evolutionClientConfig: ApiClientConfig = {
  baseURL: EVOLUTION_API_BASE_URL,
  timeout: 30000,      // 30 seconds
  enableRetry: true,   // Enable retry logic
  maxRetries: 3,       // Maximum 3 retries
  retryDelay: 1000,    // Base delay 1 second
};

const evolutionApiClient = new ApiClient(evolutionClientConfig);

// Create circuit breaker for Evolution API
const evolutionCircuitBreaker = createEvolutionApiCircuitBreaker();

export interface EvolutionControlResponse {
  evolution_id: string;
  status: string;
  paused_at?: string;
  resumed_at?: string;
}

/**
 * Evolution API with Circuit Breaker Protection
 *
 * All API calls are wrapped with circuit breaker to prevent cascading failures
 * following Federation Constitution: System Failure → Circuit Breaker
 *
 * BUG #13 FIX: Added idempotency key support to prevent duplicate operations
 */
export const evolutionApi = {
  startEvolution: async (
    payload: EvolutionStartPayload & { idempotencyKey?: string }
  ): Promise<EvolutionStartResponse> => {
    console.info('[EvolutionAPI] Starting evolution with circuit breaker protection');

    return evolutionCircuitBreaker.execute(async () => {
      return evolutionApiClient.post<EvolutionStartResponse>(
        '/api/v1/evolution/start',
        payload
      );
    });
  },

  pauseEvolution: async (
    evolutionId: string,
    options?: { idempotencyKey?: string }
  ): Promise<EvolutionControlResponse> => {
    console.info('[EvolutionAPI] Pausing evolution with circuit breaker protection');

    return evolutionCircuitBreaker.execute(async () => {
      return evolutionApiClient.post<EvolutionControlResponse>(
        `/api/v1/evolution/${evolutionId}/pause`,
        options?.idempotencyKey ? { idempotencyKey: options.idempotencyKey } : {}
      );
    });
  },

  resumeEvolution: async (
    evolutionId: string,
    options?: { idempotencyKey?: string }
  ): Promise<EvolutionControlResponse> => {
    console.info('[EvolutionAPI] Resuming evolution with circuit breaker protection');

    return evolutionCircuitBreaker.execute(async () => {
      return evolutionApiClient.post<EvolutionControlResponse>(
        `/api/v1/evolution/${evolutionId}/resume`,
        options?.idempotencyKey ? { idempotencyKey: options.idempotencyKey } : {}
      );
    });
  },
};

/**
 * Get circuit breaker metrics for monitoring
 */
export function getEvolutionApiCircuitBreakerMetrics() {
  return evolutionCircuitBreaker.getMetrics();
}

/**
 * Manually reset circuit breaker (for testing/recovery)
 */
export function resetEvolutionApiCircuitBreaker() {
  evolutionCircuitBreaker.reset();
}
