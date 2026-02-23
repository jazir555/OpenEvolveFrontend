/**
 * Event Type Definitions
 *
 * Follows the Federation Constitution:
 * - Law of Idempotency: All events have unique IDs for deduplication
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - Canonical schema for all orchestration events
 */
/**
 * Base event interface
 */
export interface BaseEvent {
    id: string;
    type: string;
    timestamp: string;
    correlation_id: string;
    source_service: string;
    data: any;
    metadata?: Record<string, any>;
}
/**
 * Event: Knowledge Extracted
 * Fired when RAGBits extracts knowledge chunks from a document
 */
export interface KnowledgeExtractedEvent extends BaseEvent {
    type: 'KnowledgeExtracted';
    data: {
        document_id: string;
        chunk_count: number;
        chunks: Array<{
            chunk_id: string;
            content: string;
            metadata: Record<string, any>;
        }>;
        extraction_method: string;
    };
}
/**
 * Event: Proof Verified
 * Fired when Z3 or LeanAide verifies a formal proof
 */
export interface ProofVerifiedEvent extends BaseEvent {
    type: 'ProofVerified';
    data: {
        proof_id: string;
        theorem_name: string;
        verification_system: 'z3' | 'lean-aide' | 'both';
        status: 'valid' | 'invalid' | 'unknown';
        verification_time_ms: number;
        proof_steps?: number;
        cross_validated?: boolean;
    };
}
/**
 * Event: Graph Updated
 * Fired when Graphiti or KarateClub updates the knowledge graph
 */
export interface GraphUpdatedEvent extends BaseEvent {
    type: 'GraphUpdated';
    data: {
        graph_id: string;
        update_type: 'node_added' | 'edge_added' | 'node_updated' | 'graph_merged';
        node_count?: number;
        edge_count?: number;
        graph_system: 'graphiti' | 'karate-club' | 'both';
        changes: Array<{
            type: 'node' | 'edge';
            action: 'added' | 'updated' | 'deleted';
            id: string;
        }>;
    };
}
/**
 * Event: Vector Indexed
 * Fired when Vector DB indexes embeddings
 */
export interface VectorIndexedEvent extends BaseEvent {
    type: 'VectorIndexed';
    data: {
        vector_db_type: 'chroma' | 'pinecone' | 'weaviate' | 'qdrant';
        index_id: string;
        embedding_count: number;
        embedding_model: string;
        dimension: number;
        index_type: 'create' | 'update' | 'delete';
    };
}
/**
 * Event: RAG Retrieved
 * Fired when RAG retrieves relevant chunks
 */
export interface RAGRetrievedEvent extends BaseEvent {
    type: 'RAGRetrieved';
    data: {
        query_id: string;
        query_text: string;
        retrieved_count: number;
        chunks: Array<{
            chunk_id: string;
            score: number;
            content_preview: string;
        }>;
        retrieval_method: 'semantic' | 'hybrid' | 'graph-guided';
    };
}
/**
 * Event: Workflow Started
 * Fired when a workflow execution begins
 */
export interface WorkflowStartedEvent extends BaseEvent {
    type: 'WorkflowStarted';
    data: {
        workflow_id: string;
        workflow_name: string;
        input_data: any;
        steps: Array<{
            step_id: string;
            step_name: string;
            service: string;
        }>;
    };
}
/**
 * Event: Workflow Completed
 * Fired when a workflow completes successfully
 */
export interface WorkflowCompletedEvent extends BaseEvent {
    type: 'WorkflowCompleted';
    data: {
        workflow_id: string;
        workflow_name: string;
        duration_ms: number;
        output_data: any;
        steps_completed: number;
        steps_failed: number;
    };
}
/**
 * Event: Workflow Failed
 * Fired when a workflow fails
 */
export interface WorkflowFailedEvent extends BaseEvent {
    type: 'WorkflowFailed';
    data: {
        workflow_id: string;
        workflow_name: string;
        failure_reason: string;
        failed_step: string;
        error_details: any;
        duration_ms: number;
    };
}
/**
 * Union type of all events
 */
export type Event = KnowledgeExtractedEvent | ProofVerifiedEvent | GraphUpdatedEvent | VectorIndexedEvent | RAGRetrievedEvent | WorkflowStartedEvent | WorkflowCompletedEvent | WorkflowFailedEvent;
/**
 * Event handler function type
 */
export type EventHandler = (event: Event) => Promise<void> | void;
/**
 * Event subscription
 */
export interface EventSubscription {
    eventType: string;
    handler: EventHandler;
    subscriptionId: string;
}
/**
 * Create a base event with common fields
 */
export declare function createBaseEvent<T extends Event['type']>(type: T, sourceService: string, correlationId: string, data: any): Event;
/**
 * Type guard to check if event is of specific type
 */
export declare function isEventType<T extends Event['type']>(event: Event, type: T): event is Extract<Event, {
    type: T;
}>;
/**
 * Event validation result
 */
export interface EventValidationResult {
    valid: boolean;
    errors: string[];
}
/**
 * Validate event structure
 */
export declare function validateEvent(event: any): EventValidationResult;
/**
 * Example usage:
 *
 * ```typescript
 * import { createBaseEvent, isEventType } from './event-types';
 *
 * // Create a KnowledgeExtracted event
 * const event = createBaseEvent(
 *   'KnowledgeExtracted',
 *   'ragbits-adapter',
 *   'corr-123',
 *   {
 *     document_id: 'doc-456',
 *     chunk_count: 10,
 *     chunks: [...],
 *     extraction_method: 'recursive'
 *   }
 * );
 *
 * // Check event type
 * if (isEventType(event, 'KnowledgeExtracted')) {
 *   console.log(`Extracted ${event.data.chunk_count} chunks`);
 * }
 *
 * // Validate event
 * const validation = validateEvent(event);
 * if (!validation.valid) {
 *   console.error('Invalid event:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=event-types.d.ts.map