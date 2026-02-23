/**
 * Vector Storage Integration for Evolved Code
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify vector DB connection before use
 * - Law of Idempotency: Safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 * - Failure Management: Circuit breaker for transient failures
 *
 * Integrates with Vector DB adapter to store evolved code embeddings
 * for semantic search and similarity matching.
 */
import { Logger } from '../../logger';
import { EvolvedCode, Problem, SimilarSolution, StoreWithEmbeddingRequest, SearchSimilarRequest } from './canonical';
export interface VectorStorageConfig {
    vectordb_adapter_url: string;
    collection_name: string;
    embedding_dimension: number;
    embedding_model?: string;
    embedding_api_key?: string;
    timeout_ms?: number;
    max_retries?: number;
    circuit_breaker_threshold?: number;
    circuit_breaker_timeout_ms?: number;
    logger?: Logger;
}
/**
 * Generate embedding from text
 * This is a placeholder - actual implementation depends on embedding service
 */
export interface EmbeddingGenerator {
    generateEmbedding(text: string): Promise<number[]>;
}
/**
 * Simple embedding generator using character-based hashing
 * This is a fallback for demonstration - production should use proper embeddings
 */
export declare class SimpleEmbeddingGenerator implements EmbeddingGenerator {
    private readonly dimension;
    constructor(dimension?: number);
    /**
     * Generate a simple hash-based embedding
     * Note: This is NOT semantically meaningful. Use real embeddings in production.
     */
    generateEmbedding(text: string): Promise<number[]>;
}
/**
 * OpenAI embedding generator (for production use)
 */
export declare class OpenAIEmbeddingGenerator implements EmbeddingGenerator {
    private readonly apiKey;
    private readonly model;
    private readonly logger;
    constructor(apiKey: string, model?: string, logger?: Logger);
    generateEmbedding(text: string): Promise<number[]>;
}
/**
 * Vector Storage for Evolved Code
 *
 * Integrates with Vector DB adapter to store and search evolved code
 */
export declare class VectorStorage {
    private readonly config;
    private readonly logger;
    private readonly circuitBreaker;
    private readonly embeddingGenerator;
    private initialized;
    private readonly httpClient;
    constructor(config: VectorStorageConfig);
    /**
     * Initialize vector storage
     * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
     */
    initialize(): Promise<void>;
    /**
     * Generate embedding for evolved code
     * Combines problem description and code for better semantic representation
     */
    generateEmbedding(evolvedCode: EvolvedCode): Promise<number[]>;
    /**
     * Generate embedding for problem search
     */
    generateProblemEmbedding(problem: Problem): Promise<number[]>;
    /**
     * Create text representation for embedding
     * Combines problem description, code, and metadata
     */
    private createEmbeddingText;
    /**
     * Create text representation for problem embedding
     */
    private createProblemEmbeddingText;
    /**
     * Store evolved code with embedding
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     */
    storeWithEmbedding(request: StoreWithEmbeddingRequest, correlationId?: string): Promise<void>;
    /**
     * Search for similar problems
     * Returns evolved code that solved similar problems
     */
    searchSimilar(request: SearchSimilarRequest, correlationId?: string): Promise<SimilarSolution[]>;
    /**
     * Delete stale code older than timestamp
     * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
     */
    deleteStaleCode(timestamp_utc: string, correlationId?: string): Promise<number>;
    /**
     * Check vector storage health
     */
    healthCheck(): Promise<{
        healthy: boolean;
        initialized: boolean;
        circuit_state: string;
        collection_exists: boolean;
    }>;
    /**
     * Close vector storage and cleanup resources
     */
    close(): Promise<void>;
}
export { SimpleEmbeddingGenerator, OpenAIEmbeddingGenerator };
export type { EmbeddingGenerator };
//# sourceMappingURL=vector-storage.d.ts.map