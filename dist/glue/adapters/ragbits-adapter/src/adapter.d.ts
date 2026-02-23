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
import { RAGClientConfig } from './rag-client';
/**
 * Circuit Breaker States
 */
declare enum CircuitState {
    CLOSED = "closed",// Normal operation
    OPEN = "open",// Circuit tripped, fail fast
    HALF_OPEN = "half_open"
}
/**
 * Circuit Breaker Configuration
 */
interface CircuitBreakerConfig {
    failure_threshold: number;
    success_threshold: number;
    timeout_ms: number;
    half_open_max_calls: number;
}
/**
 * Retry Configuration
 */
interface RetryConfig {
    max_attempts: number;
    base_delay_ms: number;
    max_delay_ms: number;
    exponential: number;
    jitter: number;
}
/**
 * RAGBits Adapter
 *
 * High-level adapter with circuit breaker and retry logic.
 */
export declare class RAGBitsAdapter {
    private client;
    private circuitState;
    private failureCount;
    private successCount;
    private lastFailureTime;
    private halfOpenCallCount;
    private circuitConfig;
    private retryConfig;
    constructor(config: RAGClientConfig);
    /**
     * Search for documents with circuit breaker and retry
     *
     * IDEMPOTENCY: Safe to retry
     */
    search(query: string, topK?: number, filters?: Record<string, any>, correlationId?: string): Promise<any>;
    /**
     * Ingest a document with circuit breaker and retry
     *
     * IDEMPOTENCY: Safe to retry (check if document exists first)
     */
    ingest(content: string, metadata: Record<string, any>, source?: string, correlationId?: string): Promise<any>;
    /**
     * Batch ingest documents
     *
     * IDEMPOTENCY: Safe to retry
     */
    batchIngest(documents: Array<{
        content: string;
        metadata: Record<string, any>;
    }>, correlationId?: string): Promise<any>;
    /**
     * Get statistics
     */
    getStats(correlationId?: string): Promise<any>;
    /**
     * Clear cache
     */
    clearCache(correlationId?: string): Promise<any>;
    /**
     * Test connection (RUNTIME TRUTH)
     */
    testConnection(correlationId?: string): Promise<boolean>;
    /**
     * Execute operation with circuit breaker
     *
     * CIRCUIT BREAKER: Stop hammering dead service
     */
    private executeWithCircuitBreaker;
    /**
     * Execute operation with exponential backoff retry
     *
     * RETRY: Exponential backoff with jitter
     */
    private executeWithRetry;
    /**
     * Calculate delay with exponential backoff and jitter
     *
     * EXPONENTIAL BACKOFF: delay = base * exponential^attempt + jitter
     */
    private calculateDelay;
    /**
     * Handle successful operation
     */
    private onSuccess;
    /**
     * Handle failed operation
     */
    private onFailure;
    /**
     * Sleep for specified milliseconds
     */
    private sleep;
    /**
     * Get circuit breaker state
     */
    getCircuitState(): {
        state: CircuitState;
        failureCount: number;
        successCount: number;
    };
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker(correlationId?: string): void;
    /**
     * Update configuration
     */
    configure(config: {
        circuit?: Partial<CircuitBreakerConfig>;
        retry?: Partial<RetryConfig>;
        client?: Partial<RAGClientConfig>;
    }): void;
    /**
     * Get current configuration
     */
    getConfig(): {
        circuit: Readonly<CircuitBreakerConfig>;
        retry: Readonly<RetryConfig>;
        client: Readonly<RAGClientConfig>;
    };
}
export {};
//# sourceMappingURL=adapter.d.ts.map