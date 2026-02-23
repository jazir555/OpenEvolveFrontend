/**
 * KarateClub Adapter
 *
 * Main adapter for KarateClub graph ML operations.
 * Follows CLAUDE.md principles:
 * - Law of Air Gap: No imports from core-projects
 * - Runtime Truth: Validate against actual KarateClub
 * - Configuration Explicitness: All config via environment variables
 * - UTC Timestamps: All times in UTC
 * - Idempotent Operations: Safe to retry
 *
 * Architecture:
 * [Core OpenEvolve] --> [KarateClub Adapter (Canonical Layer)] --> [KarateClub Python Engine]
 */
import { NodeEmbeddingRequest, NodeEmbeddingResponse, CommunityDetectionRequest, CommunityDetectionResponse, GraphAnalysisRequest, GraphAnalysisResponse } from '../../schemas/karateclub-canonical';
import { KarateClubClientConfig } from './ml-client';
export interface AdapterConfig extends KarateClubClientConfig {
    enableMetrics?: boolean;
    logLevel?: 'debug' | 'info' | 'warn' | 'error';
}
export declare class KarateClubAdapter {
    private client;
    private config;
    private metrics;
    constructor(config?: AdapterConfig);
    /**
     * Validate environment variables (Law of Configuration Explicitness)
     */
    private validateEnvironment;
    /**
     * Structured logging (JSON Lines)
     */
    private log;
    /**
     * Check if message should be logged based on log level
     */
    private shouldLog;
    /**
     * Update metrics
     */
    private updateMetrics;
    /**
     * Get current metrics
     */
    getMetrics(): {
        averageExecutionTimeMs: number;
        successRate: number;
        totalRequests: number;
        successfulRequests: number;
        failedRequests: number;
        totalExecutionTimeMs: number;
    };
    /**
     * Generate node embeddings
     */
    generateNodeEmbeddings(request: NodeEmbeddingRequest): Promise<NodeEmbeddingResponse>;
    /**
     * Detect communities
     */
    detectCommunities(request: CommunityDetectionRequest): Promise<CommunityDetectionResponse>;
    /**
     * Perform comprehensive graph analysis
     */
    analyzeGraph(request: GraphAnalysisRequest): Promise<GraphAnalysisResponse>;
    /**
     * Calculate basic graph statistics
     */
    private calculateGraphStatistics;
    /**
     * Health check
     */
    healthCheck(): Promise<{
        healthy: boolean;
        version?: string;
        error?: string;
    }>;
    /**
     * Reset metrics
     */
    resetMetrics(): void;
}
export declare function getDefaultAdapter(config?: AdapterConfig): KarateClubAdapter;
export declare function createAdapter(config?: AdapterConfig): KarateClubAdapter;
//# sourceMappingURL=adapter.d.ts.map