/**
 * Graphiti Client
 *
 * HTTP client for Graphiti REST API.
 * Follows the Federation Constitution:
 * - Law of Configuration Explicitness: All config via parameters
 * - Law of Runtime Truth: Verify API endpoints actually work
 * - Circuit Breaker: Prevent cascading failures
 * - Retry Logic: Handle transient failures with exponential backoff
 *
 * Note: This client wraps Graphiti's Python-based API server.
 * The actual graph operations are performed by Graphiti core running
 * in a separate container/service.
 */
import { Logger } from '../../lib/logger';
import { CanonicalEntity, CanonicalSearchQuery, CanonicalSearchResult, AddEpisodeOperation, AddEpisodeResult, AddTripletOperation, AddTripletResult, GraphStatistics } from '../../schemas/graphiti-canonical';
export interface GraphitiClientConfig {
    neo4jUri: string;
    neo4jUser: string;
    neo4jPassword: string;
    openaiApiKey?: string;
    anthropicApiKey?: string;
    timeoutMs: number;
    logger: Logger;
}
export declare class GraphitiClient {
    private readonly config;
    private readonly log;
    private initialized;
    private graphitiInstance;
    constructor(config: GraphitiClientConfig);
    /**
     * Initialize Graphiti client and verify connection
     * Following CLAUDE.md: RUNTIME TRUTH - verify before marking ready
     */
    initialize(): Promise<void>;
    /**
     * Test connection to Graphiti/Neo4j
     */
    private testConnection;
    /**
     * Build indices and constraints in Neo4j
     */
    buildIndices(): Promise<void>;
    /**
     * Add an episode to Graphiti
     */
    addEpisode(operation: AddEpisodeOperation, correlationId: string): Promise<AddEpisodeResult>;
    /**
     * Add a triplet (subject -> predicate -> object) to Graphiti
     */
    addTriplet(operation: AddTripletOperation, correlationId: string): Promise<AddTripletResult>;
    /**
     * Search the knowledge graph
     */
    search(query: CanonicalSearchQuery, correlationId: string): Promise<CanonicalSearchResult>;
    /**
     * Get an entity by UUID
     */
    getEntity(uuid: string, correlationId: string): Promise<CanonicalEntity | null>;
    /**
     * Get graph statistics
     */
    getStatistics(): Promise<GraphStatistics>;
    /**
     * Close the client connection
     */
    close(): Promise<void>;
}
//# sourceMappingURL=graph-client.d.ts.map