/**
 * Proof Vector Index
 *
 * Provides semantic search capabilities for formal proofs using vector embeddings.
 *
 * Federation Constitution Compliance:
 * - Law of Configuration Explicitness: All URLs and timeouts via env vars
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Prevents cascading failures
 * - Retry Logic: Handles transient failures
 */
import { FormalProof, Theorem, SimilarProof, IndexResult } from './canonical';
/**
 * Vector storage interface
 *
 * Abstracts the underlying vector database (could be Pinecone, Weaviate, pgvector, etc.)
 */
interface VectorStore {
    insert(embeddingId: string, embedding: number[], metadata: Record<string, any>): Promise<void>;
    search(embedding: number[], k: number, filter?: Record<string, any>): Promise<Array<{
        id: string;
        score: number;
        metadata: Record<string, any>;
    }>>;
    delete(embeddingId: string): Promise<void>;
    update(embeddingId: string, embedding: number[], metadata: Record<string, any>): Promise<void>;
}
/**
 * Proof Vector Index
 *
 * Manages vector embeddings for formal proofs and provides semantic search
 */
export declare class ProofVectorIndex {
    private vectorStore;
    private embeddingDimension;
    constructor(vectorStore?: VectorStore, embeddingDimension?: number);
    /**
     * Index a proof for semantic search
     *
     * @param proof - The proof to index
     * @param correlationId - Optional correlation ID for tracing
     * @returns Index result
     */
    indexProof(proof: FormalProof, correlationId?: string): Promise<IndexResult>;
    /**
     * Search for similar theorems
     *
     * @param theorem - The theorem to find similar proofs for
     * @param k - Number of results to return
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    searchSimilarTheorems(theorem: Theorem, k?: number, correlationId?: string): Promise<SimilarProof[]>;
    /**
     * Search by content (natural language query)
     *
     * @param content - Natural language query
     * @param k - Number of results to return
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    searchByContent(content: string, k?: number, correlationId?: string): Promise<SimilarProof[]>;
    /**
     * Delete a proof from the vector index
     *
     * @param proofId - ID of the proof to delete
     * @param correlationId - Optional correlation ID for tracing
     */
    deleteProof(proofId: string, correlationId?: string): Promise<void>;
    /**
     * Update a proof in the vector index
     *
     * @param proof - Updated proof
     * @param correlationId - Optional correlation ID for tracing
     */
    updateProof(proof: FormalProof, correlationId?: string): Promise<void>;
    /**
     * Generate embedding for a theorem
     *
     * In production, this would call an embedding model (e.g., OpenAI, Sentence Transformers)
     * For now, we generate a simple hash-based embedding for demonstration
     *
     * @param theorem - The theorem to embed
     * @returns Embedding vector
     */
    private generateTheoremEmbedding;
    /**
     * Generate embedding for a proof
     *
     * @param proof - The proof to embed
     * @returns Embedding vector
     */
    private generateProofEmbedding;
    /**
     * Generate embedding from text
     *
     * In production, this would use a proper embedding model.
     * For demonstration, we generate a simple hash-based embedding.
     *
     * TODO: Replace with actual embedding model (OpenAI, Cohere, or local Sentence Transformers)
     *
     * @param text - Text to embed
     * @returns Embedding vector
     */
    private generateTextEmbedding;
    /**
     * Simple hash function for demonstration
     *
     * TODO: Replace with proper embedding model
     */
    private simpleHash;
    /**
     * Filter search results by proof system
     *
     * @param results - Results to filter
     * @param system - System to filter by
     * @returns Filtered results
     */
    filterBySystem(results: SimilarProof[], system: string): SimilarProof[];
    /**
     * Filter search results by status
     *
     * @param results - Results to filter
     * @param status - Status to filter by
     * @returns Filtered results
     */
    filterByStatus(results: SimilarProof[], status: string): SimilarProof[];
    /**
     * Filter search results by minimum similarity score
     *
     * @param results - Results to filter
     * @param minScore - Minimum similarity score
     * @returns Filtered results
     */
    filterByMinScore(results: SimilarProof[], minScore: number): SimilarProof[];
}
export {};
/**
 * Example usage:
 *
 * ```typescript
 * import { ProofVectorIndex } from './vector-index';
 * import { FormalProof, Theorem } from './canonical';
 *
 * // Create vector index
 * const vectorIndex = new ProofVectorIndex();
 *
 * // Index a proof
 * const proof: FormalProof = { ... };
 * await vectorIndex.indexProof(proof, 'correlation-123');
 *
 * // Search for similar theorems
 * const theorem: Theorem = { ... };
 * const similarProofs = await vectorIndex.searchSimilarTheorems(theorem, 10, 'correlation-123');
 *
 * // Search by content
 * const results = await vectorIndex.searchByContent('prove that addition is commutative', 10);
 *
 * // Filter results
 * const validLeanProofs = vectorIndex.filterBySystem(
 *   vectorIndex.filterByStatus(similarProofs, 'valid'),
 *   'leanaide'
 * );
 * ```
 */
//# sourceMappingURL=vector-index.d.ts.map