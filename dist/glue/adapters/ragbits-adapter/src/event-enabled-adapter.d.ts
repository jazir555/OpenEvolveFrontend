/**
 * RAGBits Event-Enabled Adapter
 *
 * Following Federation Constitution:
 * - Law of Runtime Truth: Verify RAGBits availability
 * - Law of Idempotency: Safe to retry ingest operations
 * - Failure Management: Transient → Retry, Logic → DLQ, System → Circuit Breaker
 * - Observability: JSON Lines logging with correlation_id
 * - Law of UTC: All timestamps in UTC ISO-8601
 *
 * Integrates RAGBits adapter with central event bus for pub/sub messaging
 */
import { EventEnabledAdapter, AdapterOperationResult } from '../../lib/event-enabled-adapter';
import { EventBus } from '../../orchestration/event-bus';
import { RAGClientConfig } from './rag-client';
export interface RAGBitsAdapterConfig extends RAGClientConfig {
    eventBus: EventBus;
    publishEvents?: boolean;
    subscribeToEvents?: boolean;
    dlqEnabled?: boolean;
    circuitBreakerEnabled?: boolean;
    retryMaxRetries?: number;
    retryBaseDelayMs?: number;
}
/**
 * RAGBits Adapter with Event Bus Integration
 *
 * Publishes events:
 * - RAGRetrieved: When search completes successfully
 * - KnowledgeExtracted: When documents are ingested
 *
 * Subscribes to events:
 * - KnowledgeExtracted: From other adapters to index in RAGBits
 * - VectorIndexed: To update RAGBits index when vectors are indexed
 */
export declare class RAGBitsEventAdapter extends EventEnabledAdapter {
    private client;
    constructor(config: RAGBitsAdapterConfig);
    /**
     * Setup event subscriptions
     */
    private setupEventSubscriptions;
    /**
     * Handle KnowledgeExtracted event
     *
     * When knowledge is extracted (e.g., from documents), ingest it into RAGBits
     */
    private handleKnowledgeExtracted;
    /**
     * Handle VectorIndexed event
     *
     * When vectors are indexed in the vector DB, update RAGBits metadata
     */
    private handleVectorIndexed;
    /**
     * Search for documents with event publishing
     *
     * IDEMPOTENCY: Safe to retry
     */
    search(query: string, topK?: number, filters?: Record<string, any>, correlationId?: string): Promise<AdapterOperationResult>;
    /**
     * Ingest a document with event publishing
     *
     * IDEMPOTENCY: Safe to retry (check if document exists first)
     */
    ingest(content: string, metadata: Record<string, any>, source?: string, correlationId?: string): Promise<AdapterOperationResult>;
    /**
     * Batch ingest documents with event publishing
     *
     * IDEMPOTENCY: Safe to retry
     */
    batchIngest(documents: Array<{
        content: string;
        metadata: Record<string, any>;
    }>, correlationId?: string): Promise<AdapterOperationResult>;
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
}
/**
 * Helper function to create RAGBits adapter with default configuration
 */
export declare function createRAGBitsEventAdapter(config: RAGBitsAdapterConfig): RAGBitsEventAdapter;
/**
 * Example usage:
 *
 * ```typescript
 * import { createRAGBitsEventAdapter } from './event-enabled-adapter';
 * import { eventBus } from '../../orchestration/event-bus';
 *
 * const adapter = createRAGBitsEventAdapter({
 *   api_url: process.env.RAGBITS_API_URL!,
 *   timeout_ms: parseInt(process.env.TIMEOUT_MS || '5000'),
 *   eventBus,
 *   publishEvents: true,
 *   subscribeToEvents: true,
 *   dlqEnabled: true,
 *   circuitBreakerEnabled: true,
 *   retryMaxRetries: 3,
 *   retryBaseDelayMs: 1000,
 * });
 *
 * // Search with automatic event publishing
 * const result = await adapter.search('What is machine learning?', 5);
 * if (result.success) {
 *   console.log('Search results:', result.data);
 *   console.log('Event published:', result.event_published);
 * }
 *
 * // Ingest with automatic event publishing
 * const ingestResult = await adapter.ingest(
 *   'Machine learning is a subset of AI...',
 *   { document_id: 'doc-123', category: 'ml' }
 * );
 *
 * // Get stats
 * const stats = await adapter.getStats();
 * console.log('RAGBits stats:', stats);
 *
 * // Get circuit breaker state
 * const cbState = adapter.getCircuitBreakerState();
 * console.log('Circuit breaker state:', cbState);
 *
 * // Get DLQ stats
 * const dlqStats = adapter.getDLQStats();
 * console.log('DLQ stats:', dlqStats);
 * ```
 */
//# sourceMappingURL=event-enabled-adapter.d.ts.map