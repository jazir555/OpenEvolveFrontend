/**
 * KarateClub ML Client
 *
 * Python-based client for executing KarateClub algorithms.
 * Follows CLAUDE.md principles:
 * - Runtime Truth: Execute against live KarateClub
 * - Configuration Explicitness: All parameters via environment/config
 * - UTC Timestamps: All times in UTC
 * - Circuit Breaker: Handle failures gracefully
 * - Retry Logic: Fewer retries for long ML operations
 *
 * This client spawns Python subprocesses to run KarateClub operations.
 */
import { NodeEmbeddingRequest, NodeEmbeddingResponse, CommunityDetectionRequest, CommunityDetectionResponse } from '../../schemas/karateclub-canonical';
interface CircuitBreakerConfig {
    failureThreshold: number;
    successThreshold: number;
    timeout: number;
    halfOpenMaxCalls: number;
}
export interface KarateClubClientConfig {
    apiUrl?: string;
    pythonPath?: string;
    timeoutMs?: number;
    maxRetries?: number;
    tempDir?: string;
    circuitBreaker?: Partial<CircuitBreakerConfig>;
}
export declare class KarateClubMLClient {
    private config;
    private circuitBreaker;
    private circuitBreakerConfig;
    constructor(config?: KarateClubClientConfig);
    /**
     * Check circuit breaker before making request
     */
    private checkCircuitBreaker;
    /**
     * Record success in circuit breaker
     */
    private recordSuccess;
    /**
     * Record failure in circuit breaker
     */
    private recordFailure;
    /**
     * Structured logging (JSON Lines)
     */
    private log;
    /**
     * Execute Python script with timeout
     */
    private executePython;
    /**
     * Convert graph structure to temporary JSON file
     */
    private writeGraphFile;
    /**
     * Generate Python script for node embedding
     */
    private generateNodeEmbeddingScript;
    /**
     * Generate Python script for community detection
     */
    private generateCommunityScript;
    /**
     * Execute with retry logic
     */
    private executeWithRetry;
    /**
     * Generate node embeddings
     */
    generateNodeEmbeddings(request: NodeEmbeddingRequest): Promise<NodeEmbeddingResponse>;
    /**
     * Detect communities
     */
    detectCommunities(request: CommunityDetectionRequest): Promise<CommunityDetectionResponse>;
    /**
     * Health check
     */
    healthCheck(): Promise<{
        healthy: boolean;
        version?: string;
        error?: string;
    }>;
}
export {};
//# sourceMappingURL=ml-client.d.ts.map