/**
 * Evolved Code Capturer
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify connections before use
 * - Law of Idempotency: All operations safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Failure Management: Circuit breakers and proper error handling
 * - Observability: Structured logging with correlation tracking
 *
 * Main orchestrator for capturing evolved code from OpenEvolve and storing
 * it in knowledge systems (Vector DB + Graphiti).
 */
import { Logger } from '../logger';
import { VectorStorageConfig } from './vector-storage';
import { GraphStorageConfig } from './graph-storage';
import { EvolvedCode, Problem, EvolutionMetrics, EvolutionLineage, SimilarSolution, CaptureResult, CaptureMetrics } from './canonical';
export interface EvolvedCodeCapturerConfig {
    vector_storage: VectorStorageConfig;
    graph_storage: GraphStorageConfig;
    enable_vector_storage: boolean;
    enable_graph_storage: boolean;
    timeout_ms?: number;
    max_retries?: number;
    track_metrics: boolean;
    metrics_retention_days: number;
    logger?: Logger;
}
/**
 * Main capturer class for evolved code
 *
 * Orchestrates the capture and storage of evolved code from OpenEvolve
 * into both Vector DB (for semantic search) and Graphiti (for lineage tracking).
 */
export declare class EvolvedCodeCapturer {
    private readonly config;
    private readonly logger;
    private readonly vectorStorage;
    private readonly graphStorage;
    private readonly metricsTracker;
    private initialized;
    constructor(config: EvolvedCodeCapturerConfig);
    /**
     * Initialize capturer and verify connections
     * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
     */
    initialize(): Promise<void>;
    /**
     * Capture evolution result
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     *
     * Stores evolved code in both Vector DB (for semantic search) and Graphiti (for lineage)
     */
    captureEvolution(problem: Problem, solution: EvolvedCode, metrics: EvolutionMetrics, correlationId?: string): Promise<CaptureResult>;
    /**
     * Search for similar problems
     * Returns previously solved problems that are semantically similar
     */
    searchSimilarProblems(problem: Problem, maxResults?: number, correlationId?: string): Promise<SimilarSolution[]>;
    /**
     * Get evolution lineage for a code solution
     * Returns the full evolution tree from initial to final solution
     */
    getEvolutionLineage(codeId: string, correlationId?: string): Promise<EvolutionLineage>;
    /**
     * Get capture metrics
     * Returns aggregated statistics about captured code
     */
    getMetrics(correlationId?: string): Promise<CaptureMetrics>;
    /**
     * Reset metrics
     */
    resetMetrics(correlationId?: string): Promise<void>;
    /**
     * Check capturer health
     */
    healthCheck(): Promise<{
        healthy: boolean;
        initialized: boolean;
        vector_storage: {
            enabled: boolean;
            healthy: boolean;
        };
        graph_storage: {
            enabled: boolean;
            healthy: boolean;
        };
    }>;
    /**
     * Close capturer and cleanup resources
     */
    close(): Promise<void>;
    /**
     * Create capture result
     */
    private createCaptureResult;
}
/**
 * Create capturer from environment variables
 * Following CLAUDE.md: Law of Configuration Explicitness
 */
export declare function createCapturerFromEnv(logger?: Logger): EvolvedCodeCapturer;
export type { EvolvedCodeCapturerConfig };
//# sourceMappingURL=capturer.d.ts.map