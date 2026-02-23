/**
 * Graphiti Adapter
 *
 * Main adapter implementation for Graphiti temporal knowledge graph integration.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: Environment variables for all config
 * - Law of UTC: All timestamps in UTC
 * - Law of Idempotency: All operations safe to run multiple times
 * - Runtime Truth: Verify connection before marking as initialized
 *
 * Architecture:
 * - Graphiti Client -> Canonical Schema -> Event Bus -> Other Adapters
 */
import { Logger } from '../../lib/logger';
import { CircuitState } from '../../lib/circuit-breaker';
import { CanonicalEntity, CanonicalSearchQuery, CanonicalSearchResult, AddEpisodeOperation, AddEpisodeResult, AddTripletOperation, AddTripletResult, GraphStatistics } from '../../schemas/graphiti-canonical';
export interface GraphitiAdapterConfig {
    graphiti_api_url: string;
    neo4j_uri: string;
    neo4j_user: string;
    neo4j_password: string;
    timeout_ms?: number;
    max_retries?: number;
    retry_delay_ms?: number;
    circuit_breaker_threshold?: number;
    circuit_breaker_timeout_ms?: number;
    openai_api_key?: string;
    anthropic_api_key?: string;
    logger?: Logger;
}
export declare class GraphitiAdapter {
    private readonly config;
    private readonly client;
    private readonly temporalOps;
    private readonly circuitBreaker;
    private readonly log;
    private initialized;
    constructor(config: GraphitiAdapterConfig);
    /**
     * Initialize adapter and verify Graphiti connection
     * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
     */
    initialize(): Promise<void>;
    /**
     * Add an episode to the knowledge graph
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     */
    addEpisode(operation: AddEpisodeOperation, correlationId?: string): Promise<AddEpisodeResult>;
    /**
     * Add multiple episodes in bulk
     */
    addEpisodesBulk(operations: AddEpisodeOperation[], correlationId?: string): Promise<AddEpisodeResult[]>;
    /**
     * Add a triplet (subject -> predicate -> object) to the graph
     * Following CLAUDE.md: Law of Idempotency - duplicate triplets are merged
     */
    addTriplet(operation: AddTripletOperation, correlationId?: string): Promise<AddTripletResult>;
    /**
     * Search the knowledge graph
     */
    search(query: CanonicalSearchQuery, correlationId?: string): Promise<CanonicalSearchResult>;
    /**
     * Query knowledge at a specific point in time
     */
    queryAtPointInTime(query: string, timestamp: string, maxResults?: number, correlationId?: string): Promise<CanonicalSearchResult>;
    /**
     * Get entity timeline
     */
    getEntityTimeline(entityName: string, startTime: string, endTime: string, correlationId?: string): Promise<any[]>;
    /**
     * Get an entity by UUID
     */
    getEntity(uuid: string, correlationId?: string): Promise<CanonicalEntity | null>;
    /**
     * Get graph statistics
     */
    getStatistics(correlationId?: string): Promise<GraphStatistics>;
    /**
     * Check adapter health
     */
    healthCheck(): Promise<{
        healthy: boolean;
        circuit_state: CircuitState;
        initialized: boolean;
        graphiti_connected: boolean;
    }>;
    /**
     * Close adapter and cleanup resources
     */
    close(): Promise<void>;
}
export * from './graph-client';
export * from './temporal-ops';
//# sourceMappingURL=adapter.d.ts.map