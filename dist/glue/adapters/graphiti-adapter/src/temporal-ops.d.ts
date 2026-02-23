/**
 * Graphiti Temporal Operations
 *
 * Temporal knowledge graph operations for Graphiti.
 * Follows the Federation Constitution:
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Idempotency: Operations safe to run multiple times
 *
 * Features:
 * - Point-in-time queries
 * - Time-range filtering
 * - Entity timeline retrieval
 * - Temporal contradiction detection
 */
import { Logger } from '../../lib/logger';
import { GraphitiClient } from './graph-client';
import { CanonicalSearchResult } from '../../schemas/graphiti-canonical';
export declare class GraphitiTemporalOps {
    private readonly client;
    private readonly log;
    constructor(client: GraphitiClient, logger: Logger);
    /**
     * Query knowledge at a specific point in time
     * Following CLAUDE.md: Law of UTC - all timestamps in UTC
     */
    queryAtPointInTime(query: string, timestamp: string, maxResults: number | undefined, correlationId: string): Promise<CanonicalSearchResult>;
    /**
     * Search within a time range
     */
    searchTimeRange(query: string, startTime: string, endTime: string, maxResults: number | undefined, correlationId: string): Promise<CanonicalSearchResult>;
    /**
     * Get entity timeline
     * Returns all episodes/edges related to an entity within a time range
     */
    getEntityTimeline(entityName: string, startTime: string, endTime: string, correlationId: string): Promise<any[]>;
    /**
     * Detect temporal contradictions
     * Finds facts that were true at one point but became false later
     */
    detectContradictions(entityName: string, correlationId: string): Promise<any[]>;
    /**
     * Get knowledge evolution over time
     * Shows how knowledge about an entity changed
     */
    getKnowledgeEvolution(entityName: string, startTime: string, endTime: string, correlationId: string): Promise<any[]>;
}
//# sourceMappingURL=temporal-ops.d.ts.map