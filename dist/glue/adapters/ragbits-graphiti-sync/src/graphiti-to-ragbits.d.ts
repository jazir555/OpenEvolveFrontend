/**
 * Graphiti to RAGBits Sync
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify API calls work
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 *
 * Synchronizes data from Graphiti (Temporal Knowledge Graph) to RAGBits (RAG system)
 * Enhances retrieval with knowledge graph entities
 */
import { SyncResult, EnhancedQuery, BoostFactor } from './canonical';
/**
 * Graphiti Entity
 */
export interface GraphitiEntity {
    id: string;
    name: string;
    labels: string[];
    summary?: string;
    attributes: Record<string, any>;
    created_at: string;
    updated_at?: string;
}
/**
 * Graphiti Episode
 */
export interface GraphitiEpisode {
    id: string;
    name: string;
    content: string;
    episode_type: string;
    valid_at: string;
    created_at: string;
    entity_edges: string[];
}
/**
 * Entity with temporal context
 */
export interface TemporalEntity {
    entity: GraphitiEntity;
    valid_at: string;
    episodes: string[];
    relationships: Relationship[];
}
/**
 * Relationship between entities
 */
export interface Relationship {
    id: string;
    source_entity_id: string;
    target_entity_id: string;
    relation_type: string;
    fact: string;
    valid_at: string;
}
/**
 * Configuration for Graphiti to RAGBits sync
 */
export interface GraphitiToRAGBitsConfig {
    graphiti_api_url: string;
    ragbits_api_url: string;
    timeout_ms: number;
    max_retries: number;
    retry_delay_ms: number;
    batch_size: number;
    enable_enhancement: boolean;
    boost_threshold: number;
    min_confidence: number;
}
/**
 * Graphiti to RAGBits Synchronization
 *
 * Handles one-way sync from Graphiti knowledge graph to RAGBits
 * Enhances retrieval with knowledge graph context
 */
export declare class GraphitiToRAGBitsSync {
    private readonly config;
    private readonly logger;
    private readonly circuitBreaker;
    private readonly graphitiServiceName;
    private readonly ragbitsServiceName;
    constructor(config: GraphitiToRAGBitsConfig);
    /**
     * Sync entity metadata to RAGBits
     *
     * @param entity - Graphiti entity to sync
     * @param correlationId - Correlation ID for tracing
     * @returns Sync result
     */
    syncEntity(entity: GraphitiEntity, correlationId: string): Promise<SyncResult>;
    /**
     * Enhance retrieval query with knowledge graph entities
     *
     * @param query - Original user query
     * @param correlationId - Correlation ID for tracing
     * @returns Enhanced query with entity context
     */
    enhanceRetrieval(query: string, correlationId: string): Promise<EnhancedQuery>;
    /**
     * Extract keywords from entity for retrieval enhancement
     *
     * @param entity - Graphiti entity
     * @returns Array of keywords
     */
    extractKeywords(entity: GraphitiEntity): string[];
    /**
     * Create boost factor for entity
     *
     * @param entity - Graphiti entity
     * @returns Boost factor
     */
    createEntityBoost(entity: GraphitiEntity): BoostFactor;
    /**
     * Update RAGBits with entity metadata
     *
     * @param entity - Graphiti entity
     * @param keywords - Keywords for the entity
     * @param boostFactor - Boost factor for retrieval
     * @param correlationId - Correlation ID for tracing
     * @returns Promise that resolves when updated
     */
    private updateRAGBitsWithEntity;
    /**
     * Search for entities relevant to query
     *
     * @param query - User query
     * @param correlationId - Correlation ID for tracing
     * @returns Array of relevant entities
     */
    private searchEntities;
    /**
     * Build enhanced query with entity context
     *
     * @param originalQuery - Original user query
     * @param entities - Relevant entities
     * @returns Enhanced query
     */
    private buildEnhancedQuery;
    /**
     * Calculate confidence score for entity
     *
     * @param entity - Graphiti entity
     * @returns Confidence score (0-1)
     */
    private calculateEntityConfidence;
    /**
     * Update retrieval strategy based on entities
     *
     * @param query - Original query
     * @param entities - Relevant entities
     * @returns Updated query strategy
     */
    updateRetrievalStrategy(query: string, entities: GraphitiEntity[]): string;
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
export default GraphitiToRAGBitsSync;
//# sourceMappingURL=graphiti-to-ragbits.d.ts.map