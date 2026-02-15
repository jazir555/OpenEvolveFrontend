/**
 * Knowledge Processing Workflow Example
 *
 * Demonstrates complete event-driven integration:
 * 1. RAGBits extracts knowledge from documents
 * 2. Vector DB indexes embeddings
 * 3. Graphiti builds knowledge graph
 * 4. All adapters communicate via event bus
 *
 * Following Federation Constitution:
 * - Failure Management: Transient → Retry, Logic → DLQ, System → Circuit Breaker
 * - Observability: JSON Lines logging with correlation IDs
 * - Law of Idempotency: Safe to replay events
 */

import { EventBus } from '../event-bus';
import { DeadLetterQueue } from '../dead-letter-queue';
import { createBaseEvent } from '../event-types';
import { randomUUID } from 'crypto';
import { RAGBitsEventAdapter } from '../../adapters/ragbits-adapter/src/event-enabled-adapter';
import { VectorDBEventAdapter } from '../../adapters/vectordb-adapter/src/event-enabled-adapter';
import { Logger } from '../../lib/logger';

const logger = new Logger('knowledge-workflow');

/**
 * Knowledge Processing Workflow
 *
 * Orchestrates the flow of knowledge through multiple adapters
 */
export class KnowledgeProcessingWorkflow {
  private eventBus: EventBus;
  private ragbitsAdapter: RAGBitsEventAdapter;
  private vectorDBAdapter: VectorDBEventAdapter;
  private dlq: DeadLetterQueue;

  constructor() {
    // Initialize event bus
    this.eventBus = new EventBus({
      type: 'memory', // Use 'redis' for production
      persistence_enabled: true,
      circuit_breaker_enabled: true,
      dlq_enabled: true,
    });

    this.dlq = this.eventBus.getDLQ()!;

    // Initialize adapters
    this.ragbitsAdapter = new RAGBitsEventAdapter({
      api_url: process.env.RAGBITS_API_URL || 'http://localhost:8000',
      timeout_ms: parseInt(process.env.TIMEOUT_MS || '5000'),
      eventBus: this.eventBus,
      publishEvents: true,
      subscribeToEvents: true,
      dlqEnabled: true,
      circuitBreakerEnabled: true,
      retryMaxRetries: 3,
      retryBaseDelayMs: 1000,
    });

    this.vectorDBAdapter = new VectorDBEventAdapter({
      backendType: 'qdrant',
      url: process.env.VECTORDB_URL || 'http://localhost:6333',
      apiKey: process.env.VECTORDB_API_KEY,
      timeout: 5000,
      eventBus: this.eventBus,
      publishEvents: true,
      subscribeToEvents: true,
      dlqEnabled: true,
      circuitBreakerEnabled: true,
      retryMaxRetries: 3,
      retryBaseDelayMs: 1000,
    });

    this.setupWorkflowMonitoring();
  }

  /**
   * Setup workflow monitoring and metrics
   */
  private setupWorkflowMonitoring(): void {
    // Monitor event bus stats
    setInterval(() => {
      const stats = this.eventBus.getStats();
      logger.info('Event bus stats', {
        events_published: stats.events_published,
        events_received: stats.events_received,
        events_failed: stats.events_failed,
        subscriptions: stats.subscriptions,
        uptime_seconds: stats.uptime_seconds,
      });
    }, 60000); // Every minute

    // Monitor DLQ stats
    setInterval(() => {
      const stats = this.dlq.getStats();
      if (stats.total_entries > 0) {
        logger.warn('DLQ stats', {
          total_entries: stats.total_entries,
          pending_entries: stats.pending_entries,
          processed_entries: stats.processed_entries,
          failed_permanently: stats.failed_permanently,
          by_event_type: stats.by_event_type,
        });
      }
    }, 60000); // Every minute

    // Process DLQ retries periodically
    setInterval(async () => {
      const processed = await this.dlq.processRetry(async (event) => {
        logger.info('Retrying DLQ event', {
          event_id: event.id,
          event_type: event.type,
        });
        // Events will be reprocessed by subscribers
      });
      if (processed > 0) {
        logger.info('DLQ retry processed', { count: processed });
      }
    }, 30000); // Every 30 seconds
  }

  /**
   * Process a document through the complete workflow
   *
   * Flow:
   * 1. Document → RAGBits (extract knowledge chunks)
   * 2. KnowledgeExtracted event → Vector DB (index embeddings)
   * 3. KnowledgeExtracted event → Graphiti (build knowledge graph)
   * 4. VectorIndexed event → RAGBits (update metadata)
   *
   * @param document - Document content and metadata
   * @param correlationId - Optional correlation ID for tracing
   */
  async processDocument(
    document: {
      content: string;
      metadata: {
        title: string;
        author?: string;
        category?: string;
        tags?: string[];
      };
    },
    correlationId?: string
  ): Promise<{
    success: boolean;
    document_id: string;
    correlation_id: string;
    steps_completed: string[];
    errors: string[];
  }> {
    const cid = correlationId || randomUUID();
    const documentId = `doc-${randomUUID()}`;
    const stepsCompleted: string[] = [];
    const errors: string[] = [];

    logger.info('Starting document processing workflow', {
      document_id: documentId,
      correlation_id: cid,
      title: document.metadata.title,
    });

    try {
      // Step 1: Ingest document into RAGBits
      logger.info('Step 1: Ingesting document into RAGBits', {
        document_id: documentId,
        correlation_id: cid,
      });

      const ragbitsResult = await this.ragbitsAdapter.ingest(
        document.content,
        {
          ...document.metadata,
          document_id: documentId,
        },
        'workflow',
        cid
      );

      if (ragbitsResult.success) {
        stepsCompleted.push('ragbits-ingest');
        logger.info('Step 1 completed: Document ingested into RAGBits', {
          document_id: documentId,
          correlation_id: cid,
          duration_ms: ragbitsResult.duration_ms,
          event_published: ragbitsResult.event_published,
        });
      } else {
        errors.push(`RAGBits ingest failed: ${ragbitsResult.error?.message}`);
        logger.error('Step 1 failed: RAGBits ingest failed', ragbitsResult.error, {
          document_id: documentId,
          correlation_id: cid,
        });
      }

      // Note: Subsequent steps (Vector DB indexing, Graph building) are triggered
      // automatically by event subscriptions when KnowledgeExtracted event is published

      // Wait for events to propagate (in production, use proper async coordination)
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Step 2: Verify Vector DB received the event
      logger.info('Step 2: Verifying Vector DB indexing', {
        document_id: documentId,
        correlation_id: cid,
      });

      // Check if VectorIndexed event was published
      const vectorIndexEvents = this.eventBus.getHistory({
        correlation_id: cid,
      }).filter((e) => e.type === 'VectorIndexed');

      if (vectorIndexEvents.length > 0) {
        stepsCompleted.push('vector-index');
        logger.info('Step 2 completed: Vectors indexed in Vector DB', {
          document_id: documentId,
          correlation_id: cid,
          index_count: vectorIndexEvents.length,
        });
      } else {
        logger.warn('Step 2 pending: Vector DB indexing not yet completed', {
          document_id: documentId,
          correlation_id: cid,
        });
      }

      logger.info('Document processing workflow completed', {
        document_id: documentId,
        correlation_id: cid,
        steps_completed: stepsCompleted,
        errors: errors,
      });

      return {
        success: errors.length === 0,
        document_id: documentId,
        correlation_id: cid,
        steps_completed: stepsCompleted,
        errors: errors,
      };
    } catch (error) {
      const err = error as Error;
      logger.error('Document processing workflow failed', err, {
        document_id: documentId,
        correlation_id: cid,
        steps_completed: stepsCompleted,
      });

      return {
        success: false,
        document_id: documentId,
        correlation_id: cid,
        steps_completed: stepsCompleted,
        errors: [...errors, err.message],
      };
    }
  }

  /**
   * Search across all knowledge stores
   *
   * Demonstrates coordinated search using RAGBits
   */
  async searchKnowledge(
    query: string,
    options: {
      top_k?: number;
      filters?: Record<string, any>;
      include_graph?: boolean;
    } = {}
  ): Promise<{
    success: boolean;
    correlation_id: string;
    results?: any;
    rag_duration_ms?: number;
    vector_duration_ms?: number;
    graph_duration_ms?: number;
  }> {
    const cid = randomUUID();

    logger.info('Starting knowledge search', {
      correlation_id: cid,
      query,
      top_k: options.top_k || 5,
    });

    try {
      // Search in RAGBits (which internally uses Vector DB)
      const ragbitsResult = await this.ragbitsAdapter.search(
        query,
        options.top_k || 5,
        options.filters,
        cid
      );

      if (ragbitsResult.success) {
        logger.info('Knowledge search completed', {
          correlation_id: cid,
          query,
          duration_ms: ragbitsResult.duration_ms,
          result_count: ragbitsResult.data?.results?.length || 0,
        });

        return {
          success: true,
          correlation_id: cid,
          results: ragbitsResult.data,
          rag_duration_ms: ragbitsResult.duration_ms,
        };
      } else {
        logger.error('Knowledge search failed', ragbitsResult.error, {
          correlation_id: cid,
          query,
        });

        return {
          success: false,
          correlation_id: cid,
        };
      }
    } catch (error) {
      logger.error('Knowledge search failed with error', error as Error, {
        correlation_id: cid,
        query,
      });

      return {
        success: false,
        correlation_id: cid,
      };
    }
  }

  /**
   * Get workflow statistics
   */
  getStats(): {
    event_bus: any;
    dlq: any;
    ragbits_circuit_breaker: any;
    vectordb_circuit_breaker: any;
  } {
    return {
      event_bus: this.eventBus.getStats(),
      dlq: this.dlq.getStats(),
      ragbits_circuit_breaker: this.ragbitsAdapter.getCircuitBreakerState(),
      vectordb_circuit_breaker: this.vectorDBAdapter.getCircuitBreakerState(),
    };
  }

  /**
   * Shutdown workflow gracefully
   */
  async shutdown(): Promise<void> {
    logger.info('Shutting down knowledge processing workflow');

    await this.ragbitsAdapter.shutdown();
    await this.vectorDBAdapter.close();
    await this.eventBus.shutdown();

    logger.info('Workflow shutdown complete');
  }
}

/**
 * Example usage
 *
 * ```typescript
 * import { KnowledgeProcessingWorkflow } from './knowledge-processing-workflow';
 *
 * const workflow = new KnowledgeProcessingWorkflow();
 *
 * // Process a document
 * const result = await workflow.processDocument({
 *   content: 'Machine learning is a subset of artificial intelligence...',
 *   metadata: {
 *     title: 'Introduction to ML',
 *     author: 'John Doe',
 *     category: 'AI',
 *     tags: ['ml', 'ai', 'tutorial'],
 *   },
 * });
 *
 * console.log('Processing result:', result);
 * // {
 * //   success: true,
 * //   document_id: 'doc-123',
 * //   correlation_id: 'corr-456',
 * //   steps_completed: ['ragbits-ingest', 'vector-index'],
 * //   errors: []
 * // }
 *
 * // Search knowledge
 * const searchResult = await workflow.searchKnowledge('What is machine learning?', {
 *   top_k: 5,
 * });
 *
 * console.log('Search results:', searchResult.results);
 *
 * // Get statistics
 * const stats = workflow.getStats();
 * console.log('Workflow stats:', stats);
 *
 * // Shutdown
 * await workflow.shutdown();
 * ```
 */

/**
 * Event Flow Diagram
 *
 * ```
 * User Request
 *     ↓
 * processDocument()
 *     ↓
 * RAGBits.ingest()
 *     ↓
 * Publish: KnowledgeExtracted
 *     ↓
 *     ├─→ Vector DB subscribes → Index embeddings → Publish: VectorIndexed
 *     │                                            ↓
 *     │                                        RAGBits updates metadata
 *     │
 *     └─→ Graphiti subscribes → Build graph → Publish: GraphUpdated
 * ```
 *
 * Failure Scenarios:
 *
 * 1. Transient Failure (RAGBits timeout)
 *    - RAGBits adapter retries 3 times with exponential backoff
 *    - Success: Continue workflow
 *    - Failure: Send to DLQ, publish failure event
 *
 * 2. Logic Failure (Invalid document format)
 *    - RAGBits adapter detects validation error
 *    - Send to DLQ immediately (no retry)
 *    - Publish failure event
 *    - Manual intervention required
 *
 * 3. System Failure (Vector DB down)
 *    - Vector DB adapter circuit breaker opens
 *    - KnowledgeExtracted event still published
 *    - VectorIndexed event not published
 *    - Workflow continues (partial completion)
 *    - Circuit breaker closes when Vector DB recovers
 *    - Replay events from event history
 */

export default KnowledgeProcessingWorkflow;
