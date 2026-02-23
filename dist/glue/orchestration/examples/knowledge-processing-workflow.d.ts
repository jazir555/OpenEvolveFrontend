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
/**
 * Knowledge Processing Workflow
 *
 * Orchestrates the flow of knowledge through multiple adapters
 */
export declare class KnowledgeProcessingWorkflow {
    private eventBus;
    private ragbitsAdapter;
    private vectorDBAdapter;
    private dlq;
    constructor();
    /**
     * Setup workflow monitoring and metrics
     */
    private setupWorkflowMonitoring;
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
    processDocument(document: {
        content: string;
        metadata: {
            title: string;
            author?: string;
            category?: string;
            tags?: string[];
        };
    }, correlationId?: string): Promise<{
        success: boolean;
        document_id: string;
        correlation_id: string;
        steps_completed: string[];
        errors: string[];
    }>;
    /**
     * Search across all knowledge stores
     *
     * Demonstrates coordinated search using RAGBits
     */
    searchKnowledge(query: string, options?: {
        top_k?: number;
        filters?: Record<string, any>;
        include_graph?: boolean;
    }): Promise<{
        success: boolean;
        correlation_id: string;
        results?: any;
        rag_duration_ms?: number;
        vector_duration_ms?: number;
        graph_duration_ms?: number;
    }>;
    /**
     * Get workflow statistics
     */
    getStats(): {
        event_bus: any;
        dlq: any;
        ragbits_circuit_breaker: any;
        vectordb_circuit_breaker: any;
    };
    /**
     * Shutdown workflow gracefully
     */
    shutdown(): Promise<void>;
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
//# sourceMappingURL=knowledge-processing-workflow.d.ts.map