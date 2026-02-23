/**
 * RAGBits to Graphiti Sync
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify API calls work
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 *
 * Synchronizes data from RAGBits (RAG system) to Graphiti (Temporal Knowledge Graph)
 */
import { SyncResult } from './canonical';
/**
 * Document Chunk from RAGBits
 */
export interface DocumentChunkData {
    id: string;
    content: string;
    source: string;
    chunk_index: number;
    metadata?: Record<string, any>;
    embedding?: number[];
    timestamp: string;
}
/**
 * Graph Episode for Graphiti
 */
export interface GraphEpisode {
    name: string;
    content: string;
    source_description?: string;
    episode_type: 'text' | 'document' | 'message' | 'code' | 'event';
    valid_at: string;
    group_id?: string;
    metadata?: Record<string, any>;
}
/**
 * Entity extracted from document chunk
 */
export interface Entity {
    name: string;
    labels: string[];
    summary?: string;
    attributes: Record<string, any>;
}
/**
 * Relationship between entities
 */
export interface Relationship {
    subject: string;
    predicate: string;
    object: string;
    fact: string;
    attributes: Record<string, any>;
}
/**
 * Configuration for RAGBits to Graphiti sync
 */
export interface RAGBitsToGraphitiConfig {
    ragbits_api_url: string;
    graphiti_api_url: string;
    timeout_ms: number;
    max_retries: number;
    retry_delay_ms: number;
    batch_size: number;
    extract_entities: boolean;
    extract_relationships: boolean;
    entity_extraction_threshold: number;
}
/**
 * RAGBits to Graphiti Synchronization
 *
 * Handles one-way sync from RAGBits to Graphiti knowledge graph
 */
export declare class RAGBitsToGraphitiSync {
    private readonly config;
    private readonly logger;
    private readonly circuitBreaker;
    private readonly ragbitsServiceName;
    private readonly graphitiServiceName;
    constructor(config: RAGBitsToGraphitiConfig);
    /**
     * Sync a single document chunk to Graphiti
     *
     * @param chunk - Document chunk from RAGBits
     * @param correlationId - Correlation ID for tracing
     * @returns Sync result
     */
    syncDocument(chunk: DocumentChunkData, correlationId: string): Promise<SyncResult>;
    /**
     * Sync a batch of document chunks to Graphiti
     *
     * @param chunks - Array of document chunks from RAGBits
     * @param correlationId - Correlation ID for tracing
     * @returns Sync result
     */
    syncBatch(chunks: DocumentChunkData[], correlationId: string): Promise<SyncResult>;
    /**
     * Convert document chunk to graph episode
     *
     * @param chunk - Document chunk
     * @returns Graph episode
     */
    private convertChunkToEpisode;
    /**
     * Extract entities from document chunk
     *
     * @param chunk - Document chunk
     * @param correlationId - Correlation ID for tracing
     * @returns Extracted entities and relationships
     */
    private extractEntitiesAndRelationships;
    /**
     * Extract entities from document chunk (standalone method)
     *
     * @param chunk - Document chunk
     * @returns Array of extracted entities
     */
    extractEntities(chunk: DocumentChunkData): Promise<Entity[]>;
    /**
     * Extract relationships from document chunk
     *
     * @param chunk - Document chunk
     * @returns Array of extracted relationships
     */
    extractRelationships(chunk: DocumentChunkData): Promise<Relationship[]>;
    /**
     * Add temporal metadata to episode
     *
     * @param episode - Graph episode
     * @returns Enhanced episode with temporal metadata
     */
    private addTemporalMetadata;
    /**
     * Add episode to Graphiti
     *
     * @param episode - Graph episode to add
     * @param entities - Entities to add
     * @param relationships - Relationships to add
     * @param correlationId - Correlation ID for tracing
     * @returns Promise that resolves when added
     */
    private addToGraphiti;
    /**
     * Simulate API call (placeholder)
     *
     * @param correlationId - Correlation ID for tracing
     * @returns Promise that resolves after a delay
     */
    private simulateApiCall;
    /**
     * Get circuit breaker stats
     *
     * @returns Circuit breaker statistics
     */
    getCircuitBreakerStats(): any;
    /**
     * Reset circuit breaker
     */
    resetCircuitBreaker(): void;
}
export default RAGBitsToGraphitiSync;
//# sourceMappingURL=ragbits-to-graphiti.d.ts.map