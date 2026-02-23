/**
 * ROMA Canonical Adapter
 *
 * Provides the canonical adapter layer for ROMA integration following the
 * "Law of the Air Gap" from the Federation Constitution.
 *
 * This adapter:
 * 1. Transforms between ROMA API format and canonical format
 * 2. Integrates with the event bus
 * 3. Implements circuit breaker protection
 * 4. Provides retry logic with jitter
 * 5. Ensures idempotency
 * 6. Handles DLQ routing
 */

import axios, { AxiosInstance } from 'axios';
import { EventEmitter } from 'events';
import {
  RomaExecutionRequest,
  RomaExecutionResponse,
  RomaCheckpoint,
  RomaExecutionStatus,
  transformRomaResponseToCanonical,
  transformCanonicalToRomaRequest,
  transformRomaCheckpointToCanonical,
  validateRomaExecutionRequest,
  validateRomaExecutionResponse,
  validateRomaCheckpoint,
  type ValidationResult,
} from '../../../schemas/roma-canonical';

// ============================================================================
// TYPES
// ============================================================================

export interface RomaAdapterConfig {
  serverUrl: string;
  apiKey?: string;
  timeout: number;
  maxRetries: number;
  enableCircuitBreaker: boolean;
  circuitBreakerThreshold: number;
  circuitBreakerTimeout: number;
  enableIdempotency: boolean;
}

export interface AdapterExecutionContext {
  correlationId: string;
  timestamp: string;
  sourceService: string;
}

// ============================================================================
// CIRCUIT BREAKER
// ============================================================================

type CircuitBreakerState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

class CircuitBreaker {
  private state: CircuitBreakerState = 'CLOSED';
  private failureCount = 0;
  private lastFailureTime?: Date;
  private successCount = 0;

  constructor(
    private threshold: number,
    private timeout: number
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (this.shouldAttemptReset()) {
        this.state = 'HALF_OPEN';
      } else {
        throw new Error('Circuit breaker is OPEN - rejecting request');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess() {
    this.successCount++;
    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
      this.failureCount = 0;
    }
  }

  private onFailure() {
    this.failureCount++;
    this.lastFailureTime = new Date();

    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN';
    }
  }

  private shouldAttemptReset(): boolean {
    if (!this.lastFailureTime) return false;
    const elapsed = Date.now() - this.lastFailureTime.getTime();
    return elapsed >= this.timeout;
  }

  getState(): CircuitBreakerState {
    return this.state;
  }

  getStats() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      lastFailureTime: this.lastFailureTime,
    };
  }
}

// ============================================================================
// IDEMPOTENCY CACHE
// ============================================================================

class IdempotencyCache {
  private cache = new Map<string, RomaExecutionResponse>();

  generateKey(request: RomaExecutionRequest): string {
    return `roma:${request.goal}:${JSON.stringify(request.metadata || {})}`;
  }

  get(request: RomaExecutionRequest): RomaExecutionResponse | undefined {
    return this.cache.get(this.generateKey(request));
  }

  set(request: RomaExecutionRequest, response: RomaExecutionResponse): void {
    this.cache.set(this.generateKey(request), response);
  }

  clear(): void {
    this.cache.clear();
  }

  getSize(): number {
    return this.cache.size;
  }
}

// ============================================================================
// ROMA CANONICAL ADAPTER
// ============================================================================

export class RomaCanonicalAdapter extends EventEmitter {
  private client: AxiosInstance;
  private circuitBreaker: CircuitBreaker;
  private idempotencyCache: IdempotencyCache;
  private deadLetterQueue: Array<{ request: RomaExecutionRequest; error: Error; timestamp: string }> = [];

  constructor(private config: RomaAdapterConfig) {
    super();

    // Initialize HTTP client
    this.client = axios.create({
      baseURL: config.serverUrl,
      timeout: config.timeout,
      headers: config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {},
    });

    // Initialize circuit breaker
    this.circuitBreaker = new CircuitBreaker(
      config.circuitBreakerThreshold,
      config.circuitBreakerTimeout
    );

    // Initialize idempotency cache
    this.idempotencyCache = new IdempotencyCache();
  }

  // ========================================================================
  // PUBLIC METHODS
  // ========================================================================

  /**
   * Execute a ROMA task with canonical transformation
   */
  async executeTask(
    request: RomaExecutionRequest,
    context: AdapterExecutionContext
  ): Promise<RomaExecutionResponse> {
    // Validate request
    const validation = validateRomaExecutionRequest(request);
    if (!validation.isValid) {
      throw new Error(`Invalid request: ${validation.errors.join(', ')}`);
    }

    // Check idempotency cache
    if (this.config.enableIdempotency) {
      const cached = this.idempotencyCache.get(request);
      if (cached) {
        this.emit('idempotency_hit', { request, context, cachedResponse: cached });
        return cached;
      }
    }

    // Emit execution started event
    this.emit('execution_started', {
      correlationId: context.correlationId,
      goal: request.goal,
      timestamp: context.timestamp,
    });

    try {
      // Execute with circuit breaker
      const response = await this.circuitBreaker.execute(async () => {
        return await this.executeTaskInternal(request, context);
      });

      // Cache result
      if (this.config.enableIdempotency) {
        this.idempotencyCache.set(request, response);
      }

      // Emit success event
      this.emit('execution_completed', {
        correlationId: context.correlationId,
        executionId: response.execution_id,
        timestamp: context.timestamp,
      });

      return response;
    } catch (error) {
      // Route to DLQ
      this.deadLetterQueue.push({
        request,
        error: error as Error,
        timestamp: new Date().toISOString(),
      });

      // Emit failure event
      this.emit('execution_failed', {
        correlationId: context.correlationId,
        error: (error as Error).message,
        timestamp: context.timestamp,
      });

      throw error;
    }
  }

  /**
   * Get execution details
   */
  async getExecution(
    executionId: string,
    context: AdapterExecutionContext
  ): Promise<RomaExecutionResponse> {
    try {
      const response = await this.circuitBreaker.execute(async () => {
        const apiResponse = await this.client.get(`/api/v1/executions/${executionId}`);
        return transformRomaResponseToCanonical(apiResponse.data);
      });

      return response;
    } catch (error) {
      this.emit('execution_retrieval_failed', {
        correlationId: context.correlationId,
        executionId,
        error: (error as Error).message,
        timestamp: context.timestamp,
      });
      throw error;
    }
  }

  /**
   * Get execution checkpoint
   */
  async getCheckpoint(
    executionId: string,
    context: AdapterExecutionContext
  ): Promise<RomaCheckpoint> {
    try {
      const response = await this.circuitBreaker.execute(async () => {
        const apiResponse = await this.client.get(`/api/v1/executions/${executionId}/checkpoint`);
        return transformRomaCheckpointToCanonical(apiResponse.data);
      });

      this.emit('checkpoint_retrieved', {
        correlationId: context.correlationId,
        executionId,
        checkpointId: response.checkpoint_id,
        timestamp: context.timestamp,
      });

      return response;
    } catch (error) {
      this.emit('checkpoint_retrieval_failed', {
        correlationId: context.correlationId,
        executionId,
        error: (error as Error).message,
        timestamp: context.timestamp,
      });
      throw error;
    }
  }

  /**
   * Cancel execution
   */
  async cancelExecution(
    executionId: string,
    context: AdapterExecutionContext
  ): Promise<void> {
    try {
      await this.circuitBreaker.execute(async () => {
        await this.client.post(`/api/v1/executions/${executionId}/cancel`);
      });

      this.emit('execution_cancelled', {
        correlationId: context.correlationId,
        executionId,
        timestamp: context.timestamp,
      });
    } catch (error) {
      this.emit('execution_cancellation_failed', {
        correlationId: context.correlationId,
        executionId,
        error: (error as Error).message,
        timestamp: context.timestamp,
      });
      throw error;
    }
  }

  // ========================================================================
  // INTERNAL METHODS
  // ========================================================================

  private async executeTaskInternal(
    request: RomaExecutionRequest,
    context: AdapterExecutionContext
  ): Promise<RomaExecutionResponse> {
    // Transform to API format
    const apiRequest = transformCanonicalToRomaRequest(request);

    // Add correlation ID
    apiRequest.metadata = {
      ...(apiRequest.metadata || {}),
      correlation_id: context.correlationId,
      source_service: context.sourceService,
    };

    // Execute with retry
    let lastError: Error | undefined;
    for (let attempt = 1; attempt <= this.config.maxRetries; attempt++) {
      try {
        const apiResponse = await this.client.post('/api/v1/executions', apiRequest);
        const response = transformRomaResponseToCanonical(apiResponse.data);

        // Validate response
        const validation = validateRomaExecutionResponse(response);
        if (!validation.isValid) {
          throw new Error(`Invalid response: ${validation.errors.join(', ')}`);
        }

        return response;
      } catch (error) {
        lastError = error as Error;

        // Don't retry on validation errors
        if ((error as any).response?.status === 400) {
          throw error;
        }

        // Add jitter to retry delay
        if (attempt < this.config.maxRetries) {
          const baseDelay = Math.pow(2, attempt) * 1000;
          const jitter = Math.random() * 500;
          await this.delay(baseDelay + jitter);
        }
      }
    }

    throw lastError || new Error('Max retries exceeded');
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  async healthCheck(): Promise<{ healthy: boolean; details: any }> {
    try {
      // Check circuit breaker state
      const breakerStats = this.circuitBreaker.getStats();

      // Check ROMA server health
      const response = await this.client.get('/health').timeout(5000);

      return {
        healthy: response.data.status === 'healthy',
        details: {
          roma_server: response.data,
          circuit_breaker: breakerStats,
          idempotency_cache_size: this.idempotencyCache.getSize(),
          dead_letter_queue_size: this.deadLetterQueue.length,
        },
      };
    } catch (error) {
      return {
        healthy: false,
        details: {
          error: (error as Error).message,
          circuit_breaker: this.circuitBreaker.getStats(),
        },
      };
    }
  }

  // ========================================================================
  // DEAD LETTER QUEUE
  // ========================================================================

  getDeadLetterQueue() {
    return [...this.deadLetterQueue];
  }

  clearDeadLetterQueue() {
    this.deadLetterQueue = [];
  }

  retryDeadLetterQueue(context: AdapterExecutionContext): Promise<RomaExecutionResponse>[] {
    const promises = this.deadLetterQueue.map(({ request, error }) => {
      console.log(`Retrying failed request: ${request.goal} (error: ${error.message})`);
      return this.executeTask(request, context);
    });

    this.clearDeadLetterQueue();
    return promises;
  }
}

// ============================================================================
// FACTORY
// ============================================================================

export function createRomaAdapter(config: Partial<RomaAdapterConfig> = {}): RomaCanonicalAdapter {
  const defaultConfig: RomaAdapterConfig = {
    serverUrl: process.env.ROMA_SERVER_URL || 'http://localhost:8000',
    apiKey: process.env.ROMA_API_KEY,
    timeout: parseInt(process.env.ROMA_TIMEOUT || '30000'),
    maxRetries: parseInt(process.env.ROMA_MAX_RETRIES || '3'),
    enableCircuitBreaker: true,
    circuitBreakerThreshold: parseInt(process.env.ROMA_CIRCUIT_BREAKER_THRESHOLD || '5'),
    circuitBreakerTimeout: parseInt(process.env.ROMA_CIRCUIT_BREAKER_TIMEOUT || '60000'),
    enableIdempotency: true,
  };

  return new RomaCanonicalAdapter({ ...defaultConfig, ...config });
}
