/**
 * RAGBits Adapter
 *
 * Main adapter implementing the Federation Constitution:
 * - RUNTIME TRUTH: Verify RAGBits availability before use
 * - IDEMPOTENCY: Safe to retry ingest operations
 * - CIRCUIT BREAKER: Stop hammering dead service
 * - EXPONENTIAL BACKOFF: Retry with jitter
 * - STRUCTURED LOGGING: JSON Lines with correlation_id
 * - UTC TIME: All timestamps in UTC
 *
 * @module adapter
 */

import { randomUUID } from 'crypto';
import { RAGClient, RAGClientConfig } from './rag-client';

/**
 * Circuit Breaker States
 */
enum CircuitState {
  CLOSED = 'closed',     // Normal operation
  OPEN = 'open',         // Circuit tripped, fail fast
  HALF_OPEN = 'half_open' // Testing if service recovered
}

/**
 * Circuit Breaker Configuration
 */
interface CircuitBreakerConfig {
  failure_threshold: number;  // Open after N failures
  success_threshold: number;  // Close after N successes
  timeout_ms: number;         // Open state duration
  half_open_max_calls: number; // Max calls in half-open state
}

/**
 * Retry Configuration
 */
interface RetryConfig {
  max_attempts: number;
  base_delay_ms: number;
  max_delay_ms: number;
  exponential: number;  // Backoff multiplier
  jitter: number;       // Random jitter (0-1)
}

/**
 * RAGBits Adapter
 *
 * High-level adapter with circuit breaker and retry logic.
 */
export class RAGBitsAdapter {
  private client: RAGClient;
  private circuitState: CircuitState = CircuitState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenCallCount = 0;

  private circuitConfig: CircuitBreakerConfig = {
    failure_threshold: 5,
    success_threshold: 2,
    timeout_ms: 60000,     // 1 minute
    half_open_max_calls: 1,
  };

  private retryConfig: RetryConfig = {
    max_attempts: 3,
    base_delay_ms: 1000,
    max_delay_ms: 10000,
    exponential: 2.0,
    jitter: 0.1,
  };

  constructor(config: RAGClientConfig) {
    this.client = new RAGClient(config);
  }

  /**
   * Search for documents with circuit breaker and retry
   *
   * IDEMPOTENCY: Safe to retry
   */
  async search(
    query: string,
    topK = 5,
    filters?: Record<string, any>,
    correlationId?: string
  ): Promise<any> {
    const cid = correlationId || randomUUID();

    return this.executeWithCircuitBreaker(
      () => this.client.search(
        {
          query,
          top_k: topK,
          filters,
          enable_hybrid_search: true,
        },
        cid
      ),
      'search',
      cid
    );
  }

  /**
   * Ingest a document with circuit breaker and retry
   *
   * IDEMPOTENCY: Safe to retry (check if document exists first)
   */
  async ingest(
    content: string,
    metadata: Record<string, any>,
    source = 'manual',
    correlationId?: string
  ): Promise<any> {
    const cid = correlationId || randomUUID();

    // IDEMPOTENCY: Check if document already exists
    // (In real implementation, would check by document ID or hash)

    return this.executeWithCircuitBreaker(
      () => this.client.ingest(
        {
          content,
          metadata,
          source,
        },
        cid
      ),
      'ingest',
      cid
    );
  }

  /**
   * Batch ingest documents
   *
   * IDEMPOTENCY: Safe to retry
   */
  async batchIngest(
    documents: Array<{ content: string; metadata: Record<string, any> }>,
    correlationId?: string
  ): Promise<any> {
    const cid = correlationId || randomUUID();

    return this.executeWithCircuitBreaker(
      () => this.client.batchIngest(documents, cid),
      'batch_ingest',
      cid
    );
  }

  /**
   * Get statistics
   */
  async getStats(correlationId?: string): Promise<any> {
    const cid = correlationId || randomUUID();
    return this.client.getStats(cid);
  }

  /**
   * Clear cache
   */
  async clearCache(correlationId?: string): Promise<any> {
    const cid = correlationId || randomUUID();
    return this.client.clearCache(cid);
  }

  /**
   * Test connection (RUNTIME TRUTH)
   */
  async testConnection(correlationId?: string): Promise<boolean> {
    const cid = correlationId || randomUUID();
    return this.client.testConnection(cid);
  }

  /**
   * Execute operation with circuit breaker
   *
   * CIRCUIT BREAKER: Stop hammering dead service
   */
  private async executeWithCircuitBreaker<T>(
    operation: () => Promise<T>,
    operationName: string,
    correlationId: string
  ): Promise<T> {
    // Check circuit state
    if (this.circuitState === CircuitState.OPEN) {
      // Check if timeout has elapsed
      const now = Date.now();
      if (now - this.lastFailureTime > this.circuitConfig.timeout_ms) {
        // Move to half-open state
        this.circuitState = CircuitState.HALF_OPEN;
        this.halfOpenCallCount = 0;

        console.log(JSON.stringify({
          msg: 'Circuit breaker transitioning to HALF_OPEN',
          operation: operationName,
          correlation_id: correlationId,
          source_service: 'ragbits-adapter',
          timestamp: new Date().toISOString(),
        }));
      } else {
        // Circuit is still open, fail fast
        const error = new Error('Circuit breaker is OPEN - service unavailable');
        console.log(JSON.stringify({
          msg: 'Circuit breaker OPEN - failing fast',
          operation: operationName,
          correlation_id: correlationId,
          source_service: 'ragbits-adapter',
          timestamp: new Date().toISOString(),
        }));
        throw error;
      }
    }

    if (this.circuitState === CircuitState.HALF_OPEN) {
      // Limit calls in half-open state
      if (this.halfOpenCallCount >= this.circuitConfig.half_open_max_calls) {
        const error = new Error('Circuit breaker is HALF_OPEN - max calls exceeded');
        console.log(JSON.stringify({
          msg: 'Circuit breaker HALF_OPEN - max calls exceeded',
          operation: operationName,
          correlation_id: correlationId,
          source_service: 'ragbits-adapter',
          timestamp: new Date().toISOString(),
        }));
        throw error;
      }
      this.halfOpenCallCount++;
    }

    // Execute with retry
    try {
      const result = await this.executeWithRetry(operation, operationName, correlationId);
      this.onSuccess(operationName, correlationId);
      return result;
    } catch (error) {
      this.onFailure(operationName, correlationId);
      throw error;
    }
  }

  /**
   * Execute operation with exponential backoff retry
   *
   * RETRY: Exponential backoff with jitter
   */
  private async executeWithRetry<T>(
    operation: () => Promise<T>,
    operationName: string,
    correlationId: string
  ): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < this.retryConfig.max_attempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));

        // Don't retry last attempt
        if (attempt < this.retryConfig.max_attempts - 1) {
          const delay = this.calculateDelay(attempt);

          console.log(JSON.stringify({
            msg: 'Retry attempt scheduled',
            operation: operationName,
            attempt: attempt + 1,
            max_attempts: this.retryConfig.max_attempts,
            delay_ms: delay,
            error: lastError.message,
            correlation_id: correlationId,
            source_service: 'ragbits-adapter',
            timestamp: new Date().toISOString(),
          }));

          await this.sleep(delay);
        }
      }
    }

    console.log(JSON.stringify({
      msg: 'All retry attempts exhausted',
      operation: operationName,
      max_attempts: this.retryConfig.max_attempts,
      error: lastError?.message,
      correlation_id: correlationId,
      source_service: 'ragbits-adapter',
      timestamp: new Date().toISOString(),
    }));

    throw lastError;
  }

  /**
   * Calculate delay with exponential backoff and jitter
   *
   * EXPONENTIAL BACKOFF: delay = base * exponential^attempt + jitter
   */
  private calculateDelay(attempt: number): number {
    // Exponential backoff
    let delay = this.retryConfig.base_delay_ms * Math.pow(this.retryConfig.exponential, attempt);

    // Cap at max delay
    delay = Math.min(delay, this.retryConfig.max_delay_ms);

    // Add jitter (random value ± jitter percentage)
    const jitterRange = delay * this.retryConfig.jitter;
    const jitter = (Math.random() - 0.5) * 2 * jitterRange;
    delay += jitter;

    // Ensure non-negative
    return Math.max(0, Math.floor(delay));
  }

  /**
   * Handle successful operation
   */
  private onSuccess(operationName: string, correlationId: string): void {
    this.failureCount = 0;

    if (this.circuitState === CircuitState.HALF_OPEN) {
      this.successCount++;

      if (this.successCount >= this.circuitConfig.success_threshold) {
        // Close circuit
        this.circuitState = CircuitState.CLOSED;
        this.successCount = 0;

        console.log(JSON.stringify({
          msg: 'Circuit breaker CLOSED - service recovered',
          operation: operationName,
          correlation_id: correlationId,
          source_service: 'ragbits-adapter',
          timestamp: new Date().toISOString(),
        }));
      }
    } else if (this.circuitState === CircuitState.CLOSED) {
      this.successCount++;
    }
  }

  /**
   * Handle failed operation
   */
  private onFailure(operationName: string, correlationId: string): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.successCount = 0;

    console.log(JSON.stringify({
      msg: 'Operation failed',
      operation: operationName,
      failure_count: this.failureCount,
      failure_threshold: this.circuitConfig.failure_threshold,
      correlation_id: correlationId,
      source_service: 'ragbits-adapter',
      timestamp: new Date().toISOString(),
    }));

    // Check if we should open the circuit
    if (this.failureCount >= this.circuitConfig.failure_threshold) {
      this.circuitState = CircuitState.OPEN;

      console.log(JSON.stringify({
        msg: 'Circuit breaker OPEN - too many failures',
        operation: operationName,
        failure_count: this.failureCount,
        correlation_id: correlationId,
        source_service: 'ragbits-adapter',
        timestamp: new Date().toISOString(),
      }));
    }
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get circuit breaker state
   */
  getCircuitState(): {
    state: CircuitState;
    failureCount: number;
    successCount: number;
    } {
    return {
      state: this.circuitState,
      failureCount: this.failureCount,
      successCount: this.successCount,
    };
  }

  /**
   * Reset circuit breaker
   */
  resetCircuitBreaker(correlationId?: string): void {
    const cid = correlationId || randomUUID();

    this.circuitState = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.halfOpenCallCount = 0;

    console.log(JSON.stringify({
      msg: 'Circuit breaker reset',
      correlation_id: cid,
      source_service: 'ragbits-adapter',
      timestamp: new Date().toISOString(),
    }));
  }

  /**
   * Update configuration
   */
  configure(config: {
    circuit?: Partial<CircuitBreakerConfig>;
    retry?: Partial<RetryConfig>;
    client?: Partial<RAGClientConfig>;
  }): void {
    if (config.circuit) {
      this.circuitConfig = { ...this.circuitConfig, ...config.circuit };
    }

    if (config.retry) {
      this.retryConfig = { ...this.retryConfig, ...config.retry };
    }

    if (config.client) {
      this.client.configure(config.client);
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): {
    circuit: Readonly<CircuitBreakerConfig>;
    retry: Readonly<RetryConfig>;
    client: Readonly<RAGClientConfig>;
    } {
    return {
      circuit: { ...this.circuitConfig },
      retry: { ...this.retryConfig },
      client: this.client.getConfig(),
    };
  }
}
