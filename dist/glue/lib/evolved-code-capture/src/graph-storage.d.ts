/**
 * Graph Storage Integration for Evolved Code
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify Graphiti connection before use
 * - Law of Idempotency: Safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breaker for transient failures
 *
 * Integrates with Graphiti adapter to store evolved code as temporal episodes
 * for lineage tracking and knowledge graph-based retrieval.
 */
import { Logger } from '../../logger';
import { EvolvedCode, EvolutionLineage } from './canonical';
export interface GraphStorageConfig {
    graphiti_adapter_url: string;
    episode_type_base: string;
    timeout_ms?: number;
    max_retries?: number;
    circuit_breaker_threshold?: number;
    circuit_breaker_timeout_ms?: number;
    logger?: Logger;
}
/**
 * Graph Storage for Evolved Code
 *
 * Integrates with Graphiti adapter to store evolved code as temporal episodes
 */
export declare class GraphStorage {
    private readonly config;
    private readonly logger;
    private readonly circuitBreaker;
    private initialized;
    private readonly httpClient;
    constructor(config: GraphStorageConfig);
    /**
     * Initialize graph storage
     * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
     */
    initialize(): Promise<void>;
    /**
     * Store evolved code as a Graphiti episode
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     */
    storeAsEpisode(evolvedCode: EvolvedCode, correlationId?: string): Promise<{
        episode_id: string;
        success: boolean;
    }>;
    /**
     * Create episode description from evolved code
     */
    private createEpisodeDescription;
    /**
     * Link problem to solution in the knowledge graph
     * Creates a relationship between problem and solution entities
     */
    linkProblemToSolution(problemId: string, solutionId: string, correlationId?: string): Promise<{
        success: boolean;
        edge_id: string;
    }>;
    /**
     * Track evolution lineage for a code solution
     * Builds the evolution tree from initial to final solution
     */
    trackEvolutionLineage(codeId: string, correlationId?: string): Promise<EvolutionLineage>;
    /**
     * Build lineage from search results
     */
    private buildLineageFromResults;
    /**
     * Calculate depth of a node
     */
    private calculateNodeDepth;
    /**
     * Get evolution history for a problem type
     * Returns all evolved code solutions for a given problem type
     */
    getEvolutionHistory(problemType: string, correlationId?: string): Promise<EvolvedCode[]>;
    /**
     * Check graph storage health
     */
    healthCheck(): Promise<{
        healthy: boolean;
        initialized: boolean;
        circuit_state: string;
        graphiti_connected: boolean;
    }>;
    /**
     * Close graph storage and cleanup resources
     */
    close(): Promise<void>;
}
export type { GraphStorageConfig };
//# sourceMappingURL=graph-storage.d.ts.map