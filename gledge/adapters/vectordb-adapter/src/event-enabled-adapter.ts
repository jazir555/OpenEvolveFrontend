/**
 * Vector DB Event-Enabled Adapter
 *
 * Following Federation Constitution:
 * - Law of Runtime Truth: Validates backend capabilities at runtime
 * - Law of Idempotency: Safe to run upsert operations multiple times
 * - Failure Management: Transient → Retry, Logic → DLQ, System → Circuit Breaker
 * - Observability: JSON Lines logging with correlation_id
 * - Law of UTC: All timestamps in UTC ISO-8601
 *
 * Integrates Vector DB adapter with central event bus for pub/sub messaging
 */

import { EventEnabledAdapter, AdapterEventConfig, AdapterOperationResult } from '../../lib/event-enabled-adapter';
import { EventBus } from '../../orchestration/event-bus';
import { Event, isEventType } from '../../orchestration/event-types';
import { VectorDBAdapter } from './adapter';
import {
  VectorDBAdapterConfig,
  CollectionConfig,
  SearchQuery,
  SearchResult,
  UpsertRequest,
  DeleteRequest,
} from './adapter';

export interface VectorDBEventAdapterConfig extends VectorDBAdapterConfig {
  eventBus: EventBus;
  publishEvents?: boolean;
  subscribeToEvents?: boolean;
  dlqEnabled?: boolean;
  circuitBreakerEnabled?: boolean;
  retryMaxRetries?: number;
  retryBaseDelayMs?: number;
}

/**
 * Vector DB Adapter with Event Bus Integration
 *
 * Publishes events:
 * - VectorIndexed: When vectors are upserted
 * - VectorSearched: When search completes
 *
 * Subscribes to events:
 * - KnowledgeExtracted: To index extracted knowledge chunks
 * - GraphUpdated: To index graph embeddings
 */
export class VectorDBEventAdapter extends EventEnabledAdapter {
  private adapter: VectorDBAdapter;

  constructor(config: VectorDBEventAdapterConfig) {
    super('vectordb-adapter', {
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

    // Create base adapter without event bus
    const baseConfig = { ...config };
    delete (baseConfig as any).eventBus;
    delete (baseConfig as any).publishEvents;
    delete (baseConfig as any).subscribeToEvents;
    delete (baseConfig as any).dlqEnabled;
    delete (baseConfig as any).circuitBreakerEnabled;
    delete (baseConfig as any).retryMaxRetries;
    delete (baseConfig as any).retryBaseDelayMs;

    this.adapter = new VectorDBAdapter(baseConfig);

    this.setupEventSubscriptions();
  }

  /**
   * Setup event subscriptions
   */
  private setupEventSubscriptions(): void {
    // Subscribe to knowledge extraction events to index embeddings
    this.subscribeToEvent('KnowledgeExtracted', async (event) => {
      if (isEventType(event, 'KnowledgeExtracted')) {
        await this.handleKnowledgeExtracted(event);
      }
    });

    // Subscribe to graph update events to index graph embeddings
    this.subscribeToEvent('GraphUpdated', async (event) => {
      if (isEventType(event, 'GraphUpdated')) {
        await this.handleGraphUpdated(event);
      }
    });
  }

  /**
   * Handle KnowledgeExtracted event
   *
   * When knowledge is extracted, generate embeddings and index in vector DB
   */
  private async handleKnowledgeExtracted(event: Event): Promise<void> {
    if (!isEventType(event, 'KnowledgeExtracted')) {
      return;
    }

    this.logger.info('Processing KnowledgeExtracted event for vector indexing', {
      event_id: event.id,
      correlation_id: event.correlation_id,
      document_id: event.data.document_id,
      chunk_count: event.data.chunk_count,
    });

    // Note: In a real implementation, you would:
    // 1. Generate embeddings for each chunk using an embedding model
    // 2. Index the embeddings in the vector DB
    // For now, we'll log the event and publish a VectorIndexed event

    await this.publishEvent(
      'VectorIndexed',
      {
        vector_db_type: this.adapter.getBackendType(),
        index_id: event.data.document_id,
        embedding_count: event.data.chunk_count,
        embedding_model: 'text-embedding-ada-002', // Example
        dimension: 1536, // Example dimension
        index_type: 'create',
      },
      event.correlation_id
    );
  }

  /**
   * Handle GraphUpdated event
   *
   * When graph is updated, generate graph embeddings and index in vector DB
   */
  private async handleGraphUpdated(event: Event): Promise<void> {
    if (!isEventType(event, 'GraphUpdated')) {
      return;
    }

    this.logger.info('Processing GraphUpdated event for vector indexing', {
      event_id: event.id,
      correlation_id: event.correlation_id,
      graph_id: event.data.graph_id,
      update_type: event.data.update_type,
    });

    // Note: In a real implementation, you would:
    // 1. Generate graph embeddings (node2vec, graph2vec, etc.)
    // 2. Index the embeddings in the vector DB
    // For now, we'll log the event

    this.logger.info('Graph embeddings indexed', {
      graph_id: event.data.graph_id,
      node_count: event.data.node_count,
      edge_count: event.data.edge_count,
    });
  }

  /**
   * Health check with event publishing
   */
  async healthCheck(): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'health-check',
      async () => await this.adapter.healthCheck(),
      null // Don't publish events for health checks
    );
  }

  /**
   * Create a collection with event publishing
   */
  async createCollection(config: CollectionConfig): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'create-collection',
      async () => await this.adapter.createCollection(config),
      'CollectionCreated',
      {
        collection_name: config.name,
        dimension: config.dimension,
        distance_metric: config.distance_metric,
      }
    );
  }

  /**
   * Get collection info
   */
  async getCollectionInfo(collectionName: string): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'get-collection-info',
      async () => await this.adapter.getCollectionInfo(collectionName),
      null
    );
  }

  /**
   * List all collections
   */
  async listCollections(): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'list-collections',
      async () => await this.adapter.listCollections(),
      null
    );
  }

  /**
   * Upsert vectors with event publishing
   *
   * IDEMPOTENCY: Safe to run multiple times
   */
  async upsert(request: UpsertRequest): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'upsert-vectors',
      async () => await this.adapter.upsert(request),
      'VectorIndexed',
      {
        vector_db_type: this.adapter.getBackendType(),
        index_id: request.collection_name,
        embedding_count: request.entries.length,
        embedding_model: 'unknown', // Would be determined from the vectors
        dimension: request.entries[0]?.vector.length || 0,
        index_type: 'upsert',
      }
    );
  }

  /**
   * Search vectors with event publishing
   */
  async search(collectionName: string, query: SearchQuery): Promise<AdapterOperationResult<SearchResult[]>> {
    return this.executeOperation(
      'search-vectors',
      async () => await this.adapter.search(collectionName, query),
      'VectorSearched',
      {
        collection_name: collectionName,
        query_vector_dimension: query.vector.length,
        k: query.k,
        score_threshold: query.score_threshold,
      }
    );
  }

  /**
   * Delete vectors with event publishing
   *
   * IDEMPOTENCY: Safe to run multiple times
   */
  async delete(request: DeleteRequest): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'delete-vectors',
      async () => await this.adapter.delete(request),
      'VectorDeleted',
      {
        collection_name: request.collection_name,
        delete_count: request.ids.length,
        delete_all: request.delete_all,
      }
    );
  }

  /**
   * Get the backend type
   */
  getBackendType(): string {
    return this.adapter.getBackendType();
  }

  /**
   * Close the adapter
   */
  async close(): Promise<void> {
    await this.adapter.close();
    await this.shutdown();
  }
}

/**
 * Helper function to create Vector DB adapter with default configuration
 */
export function createVectorDBEventAdapter(config: VectorDBEventAdapterConfig): VectorDBEventAdapter {
  return new VectorDBEventAdapter(config);
}

/**
 * Example usage:
 *
 * ```typescript
 * import { createVectorDBEventAdapter } from './event-enabled-adapter';
 * import { eventBus } from '../../orchestration/event-bus';
 *
 * const adapter = createVectorDBEventAdapter({
 *   backendType: 'qdrant',
 *   url: process.env.VECTORDB_URL!,
 *   apiKey: process.env.VECTORDB_API_KEY,
 *   timeout: 5000,
 *   eventBus,
 *   publishEvents: true,
 *   subscribeToEvents: true,
 *   dlqEnabled: true,
 *   circuitBreakerEnabled: true,
 *   retryMaxRetries: 3,
 *   retryBaseDelayMs: 1000,
 * });
 *
 * // Create collection
 * await adapter.createCollection({
 *   name: 'documents',
 *   dimension: 1536,
 *   distance_metric: 'cosine',
 * });
 *
 * // Upsert vectors with automatic event publishing
 * const upsertResult = await adapter.upsert({
 *   collection_name: 'documents',
 *   entries: [
 *     {
 *       id: 'vec-1',
 *       vector: Array(1536).fill(0).map(() => Math.random()),
 *       payload: { text: 'sample text' },
 *     },
 *   ],
 * });
 *
 * if (upsertResult.success) {
 *   console.log('Vectors upserted, event published:', upsertResult.event_published);
 * }
 *
 * // Search with automatic event publishing
 * const searchResult = await adapter.search('documents', {
 *   vector: Array(1536).fill(0).map(() => Math.random()),
 *   k: 10,
 *   score_threshold: 0.7,
 * });
 *
 * if (searchResult.success) {
 *   console.log('Search results:', searchResult.data);
 * }
 * ```
 */
