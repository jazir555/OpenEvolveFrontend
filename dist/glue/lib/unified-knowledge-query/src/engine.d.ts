/**
 * Unified Knowledge Query Engine
 *
 * Main engine for querying multiple knowledge systems.
 *
 * Federation Constitution Compliance:
 * - Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breakers, retries, fallbacks
 * - Observability: Structured logging with correlation IDs
 * - Law of UTC: All timestamps in UTC
 */
import { UnifiedQueryResult, QueryOptions, SystemHealth, EngineMetrics } from './canonical';
/**
 * Engine Constructor Options
 */
interface EngineOptions {
    ragbitsUrl?: string;
    graphitiUrl?: string;
    vectordbUrl?: string;
    timeout?: number;
    enableFallback?: boolean;
    maxResults?: number;
}
/**
 * Unified Knowledge Query Engine
 *
 * Main entry point for querying multiple knowledge systems
 */
export declare class UnifiedKnowledgeQueryEngine {
    private logger;
    private router;
    private fusion;
    private fallback;
    private systems;
    private metrics;
    private startTime;
    constructor(options?: EngineOptions);
    /**
     * Execute unified knowledge query
     *
     * @param query - Query text
     * @param options - Query options
     * @returns Unified query results
     */
    query(query: string, options?: QueryOptions): Promise<UnifiedQueryResult>;
    /**
     * Semantic search query
     */
    semanticSearch(query: string, options?: QueryOptions): Promise<UnifiedQueryResult>;
    /**
     * Temporal query with time filters
     */
    temporalQuery(query: string, startDate: string, endDate: string, options?: QueryOptions): Promise<UnifiedQueryResult>;
    /**
     * Graph traversal query
     */
    graphTraversal(query: string, options?: QueryOptions): Promise<UnifiedQueryResult>;
    /**
     * Hybrid query across all systems
     */
    hybridQuery(query: string, options?: QueryOptions): Promise<UnifiedQueryResult>;
    /**
     * Health check for all systems
     */
    healthCheck(): Promise<SystemHealth[]>;
    /**
     * Get engine metrics
     */
    getMetrics(): Promise<EngineMetrics>;
    /**
     * Reset engine metrics
     */
    resetMetrics(): void;
    /**
     * Execute queries according to plan
     */
    private executeQueries;
    /**
     * Execute query against a single system
     */
    private executeSystemQuery;
    /**
     * Create client for system
     */
    private createClient;
    /**
     * Apply filters to results
     */
    private applyFilters;
    /**
     * Check health of a single system
     */
    private checkSystemHealth;
    /**
     * Initialize systems from config
     */
    private initializeSystems;
    /**
     * Update average query time
     */
    private updateAverageQueryTime;
    /**
     * Generate correlation ID (UUID v4)
     */
    private generateCorrelationId;
}
/**
 * Default engine instance
 */
export declare const defaultEngine: UnifiedKnowledgeQueryEngine;
export {};
//# sourceMappingURL=engine.d.ts.map