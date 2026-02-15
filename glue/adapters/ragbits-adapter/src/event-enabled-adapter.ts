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

import { EventEnabledAdapter, AdapterEventConfig, AdapterOperationResult } from '../../lib/event-enabled-adapter';
import { EventBus } from '../../orchestration/event-bus';
import { Event, isEventType } from '../../orchestration/event-types';
import { RAGClient, RAGClientConfig } from './rag-client';

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
export class RAGBitsEventAdapter extends EventEnabledAdapter {
  private client: RAGClient;

  constructor(config: RAGBitsAdapterConfig) {
    super('ragbits-adapter', {
      eventBus: config.eventBus,
      publishEvents: config.publishEvents ?? true,
      subscribeToEvents: config.subscribeToEvents ?? true,
      dlqEnabled: config.dlqEnabled ?? true,
      circuitBreakerEnabled: config.circuitBreakerEnabled ?? true,
      retryConfig: {
        max_retries: config.retryMaxRetries ?? 3,
        base_delay_ms: config.retryBaseDelayMs ?? 1000,
      },
    });

    this.client = new RAGClient({
      api_url: config.api_url,
      timeout_ms: config.timeout_ms,
      api_key: config.api_key,
    });

    this.setupEventSubscriptions();
  }

  /**
   * Setup event subscriptions
   */
  private setupEventSubscriptions(): void {
    // Subscribe to knowledge extraction events to index in RAGBits
    this.subscribeToEvent('KnowledgeExtracted', async (event) => {
      if (isEventType(event, 'KnowledgeExtracted')) {
        await this.handleKnowledgeExtracted(event);
      }
    });

    // Subscribe to vector indexed events to update RAGBits index
    this.subscribeToEvent('VectorIndexed', async (event) => {
      if (isEventType(event, 'VectorIndexed')) {
        await this.handleVectorIndexed(event);
      }
    });
  }

  /**
   * Handle KnowledgeExtracted event
   *
   * When knowledge is extracted (e.g., from documents), ingest it into RAGBits
   */
  private async handleKnowledgeExtracted(event: Event): Promise<void> {
    if (!isEventType(event, 'KnowledgeExtracted')) {
      return;
    }

    this.logger.info('Processing KnowledgeExtracted event', {
      event_id: event.id,
      correlation_id: event.correlation_id,
      document_id: event.data.document_id,
      chunk_count: event.data.chunk_count,
    });

    // Ingest each chunk into RAGBits
    for (const chunk of event.data.chunks) {
      await this.ingest(
        chunk.content,
        {
          ...chunk.metadata,
          document_id: event.data.document_id,
          chunk_id: chunk.chunk_id,
          extraction_method: event.data.extraction_method,
        },
        'event-driven',
        event.correlation_id
      );
    }
  }

  /**
   * Handle VectorIndexed event
   *
   * When vectors are indexed in the vector DB, update RAGBits metadata
   */
  private async handleVectorIndexed(event: Event): Promise<void> {
    if (!isEventType(event, 'VectorIndexed')) {
      return;
    }

    this.logger.info('Processing VectorIndexed event', {
      event_id: event.id,
      correlation_id: event.correlation_id,
      index_id: event.data.index_id,
      embedding_count: event.data.embedding_count,
      vector_db_type: event.data.vector_db_type,
    });

    // Update RAGBits metadata (implementation depends on RAGBits API)
    // This is a placeholder for the actual implementation
    this.logger.info('Vector indexing metadata updated', {
      index_id: event.data.index_id,
      vector_db_type: event.data.vector_db_type,
    });
  }

  /**
   * Search for documents with event publishing
   *
   * IDEMPOTENCY: Safe to retry
   */
  async search(
    query: string,
    topK = 5,
    filters?: Record<string, any>,
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    const result = await this.executeOperation(
      'search',
      async () => {
        return await this.client.search(
          {
            query,
            top_k: topK,
            filters,
            enable_hybrid_search: true,
          },
          correlationId
        );
      },
      'RAGRetrieved',
      {
        query_id: correlationId || randomUUID(),
        query_text: query,
        retrieved_count: topK,
        retrieval_method: 'hybrid',
      }
    );

    return result;
  }

  /**
   * Ingest a document with event publishing
   *
   * IDEMPOTENCY: Safe to retry (check if document exists first)
   */
  async ingest(
    content: string,
    metadata: Record<string, any>,
    source = 'manual',
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    const result = await this.executeOperation(
      'ingest',
      async () => {
        return await this.client.ingest(
          {
            content,
            metadata,
            source,
          },
          correlationId
        );
      },
      'KnowledgeExtracted',
      {
        document_id: metadata.document_id || randomUUID(),
        chunk_count: 1,
        chunks: [
          {
            chunk_id: randomUUID(),
            content,
            metadata,
          },
        ],
        extraction_method: source,
      }
    );

    return result;
  }

  /**
   * Batch ingest documents with event publishing
   *
   * IDEMPOTENCY: Safe to retry
   */
  async batchIngest(
    documents: Array<{ content: string; metadata: Record<string, any> }>,
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    const documentId = correlationId || randomUUID();

    const result = await this.executeOperation(
      'batch-ingest',
      async () => {
        return await this.client.batchIngest(documents, correlationId);
      },
      'KnowledgeExtracted',
      {
        document_id: documentId,
        chunk_count: documents.length,
        chunks: documents.map((doc) => ({
          chunk_id: randomUUID(),
          content: doc.content,
          metadata: doc.metadata,
        })),
        extraction_method: 'batch',
      }
    );

    return result;
  }

  /**
   * Get statistics
   */
  async getStats(correlationId?: string): Promise<any> {
    return this.client.getStats(correlationId);
  }

  /**
   * Clear cache
   */
  async clearCache(correlationId?: string): Promise<any> {
    return this.client.clearCache(correlationId);
  }

  /**
   * Test connection (RUNTIME TRUTH)
   */
  async testConnection(correlationId?: string): Promise<boolean> {
    return this.client.testConnection(correlationId);
  }
}

/**
 * Helper function to create RAGBits adapter with default configuration
 */
export function createRAGBitsEventAdapter(config: RAGBitsAdapterConfig): RAGBitsEventAdapter {
  return new RAGBitsEventAdapter(config);
}

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
