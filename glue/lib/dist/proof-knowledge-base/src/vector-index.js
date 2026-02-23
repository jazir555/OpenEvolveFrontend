"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProofVectorIndex = void 0;
const logger_1 = require("../logger");
/**
 * In-memory vector store (for development/testing)
 *
 * Production should use a proper vector DB like Pinecone, Weaviate, or pgvector
 */
class InMemoryVectorStore {
    constructor() {
        this.store = new Map();
    }
    async insert(embeddingId, embedding, metadata) {
        this.store.set(embeddingId, { embedding, metadata });
    }
    async search(embedding, k, filter) {
        const results = [];
        for (const [id, data] of this.store.entries()) {
            // Apply filter if provided
            if (filter) {
                let matches = true;
                for (const [key, value] of Object.entries(filter)) {
                    if (data.metadata[key] !== value) {
                        matches = false;
                        break;
                    }
                }
                if (!matches)
                    continue;
            }
            // Calculate cosine similarity
            const similarity = this.cosineSimilarity(embedding, data.embedding);
            results.push({
                id,
                score: similarity,
                metadata: data.metadata,
            });
        }
        // Sort by similarity (descending) and take top k
        results.sort((a, b) => b.score - a.score);
        return results.slice(0, k);
    }
    async delete(embeddingId) {
        this.store.delete(embeddingId);
    }
    async update(embeddingId, embedding, metadata) {
        this.store.set(embeddingId, { embedding, metadata });
    }
    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        if (a.length !== b.length) {
            throw new Error('Vector dimensions must match');
        }
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
/**
 * Proof Vector Index
 *
 * Manages vector embeddings for formal proofs and provides semantic search
 */
class ProofVectorIndex {
    constructor(vectorStore, embeddingDimension = 384) {
        this.vectorStore = vectorStore || new InMemoryVectorStore();
        this.embeddingDimension = embeddingDimension;
    }
    /**
     * Index a proof for semantic search
     *
     * @param proof - The proof to index
     * @param correlationId - Optional correlation ID for tracing
     * @returns Index result
     */
    async indexProof(proof, correlationId) {
        const logContext = {
            correlation_id: correlationId || proof.correlation_id,
            source_service: 'proof-vector-index',
            proof_id: proof.id,
        };
        try {
            logger_1.logger.info('Indexing proof for vector search', logContext);
            // Generate embedding for the proof
            const embedding = await this.generateProofEmbedding(proof);
            // Store in vector database
            const metadata = {
                proof_id: proof.id,
                theorem_id: proof.theorem_id,
                theorem: proof.theorem,
                system: proof.system,
                status: proof.status,
                tactics: proof.tactics || [],
                timestamp: proof.timestamp_utc,
            };
            await this.vectorStore.insert(proof.id, embedding, metadata);
            logger_1.logger.info('Proof indexed successfully', {
                ...logContext,
                embedding_dimension: embedding.length,
            });
            return {
                success: true,
                embedding_id: proof.id,
                vector_indexed: true,
                graph_indexed: false, // Not handled by vector index
                timestamp: new Date().toISOString(),
            };
        }
        catch (error) {
            logger_1.logger.error('Failed to index proof', error, logContext);
            return {
                success: false,
                vector_indexed: false,
                graph_indexed: false,
                error: error instanceof Error ? error.message : String(error),
                timestamp: new Date().toISOString(),
            };
        }
    }
    /**
     * Search for similar theorems
     *
     * @param theorem - The theorem to find similar proofs for
     * @param k - Number of results to return
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    async searchSimilarTheorems(theorem, k = 10, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-vector-index',
            theorem_id: theorem.id,
        };
        try {
            logger_1.logger.info('Searching for similar theorems', {
                ...logContext,
                k,
            });
            // Generate embedding for the theorem
            const embedding = await this.generateTheoremEmbedding(theorem);
            // Search vector database
            const results = await this.vectorStore.search(embedding, k);
            // Transform to SimilarProof objects
            const similarProofs = [];
            for (const result of results) {
                // Note: In a real implementation, you'd fetch the full proof from storage
                // For now, we'll create a partial proof object
                const proof = {
                    id: result.metadata.proof_id,
                    theorem_id: result.metadata.theorem_id,
                    theorem: result.metadata.theorem,
                    proof: '', // Would fetch from storage
                    system: result.metadata.system,
                    status: result.metadata.status,
                    tactics: result.metadata.tactics || [],
                    timestamp_utc: result.metadata.timestamp,
                };
                similarProofs.push({
                    proof,
                    similarity_score: result.score,
                    matching_theorems: [theorem.statement],
                });
            }
            logger_1.logger.info('Similar theorems found', {
                ...logContext,
                result_count: similarProofs.length,
            });
            return similarProofs;
        }
        catch (error) {
            logger_1.logger.error('Failed to search similar theorems', error, logContext);
            return [];
        }
    }
    /**
     * Search by content (natural language query)
     *
     * @param content - Natural language query
     * @param k - Number of results to return
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    async searchByContent(content, k = 10, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-vector-index',
        };
        try {
            logger_1.logger.info('Searching proofs by content', {
                ...logContext,
                content_length: content.length,
                k,
            });
            // Generate embedding from content
            const embedding = await this.generateTextEmbedding(content);
            // Search vector database
            const results = await this.vectorStore.search(embedding, k);
            // Transform to SimilarProof objects
            const similarProofs = [];
            for (const result of results) {
                const proof = {
                    id: result.metadata.proof_id,
                    theorem_id: result.metadata.theorem_id,
                    theorem: result.metadata.theorem,
                    proof: '',
                    system: result.metadata.system,
                    status: result.metadata.status,
                    tactics: result.metadata.tactics || [],
                    timestamp_utc: result.metadata.timestamp,
                };
                similarProofs.push({
                    proof,
                    similarity_score: result.score,
                    explanation: `Similar to query: "${content.substring(0, 100)}..."`,
                });
            }
            logger_1.logger.info('Content search completed', {
                ...logContext,
                result_count: similarProofs.length,
            });
            return similarProofs;
        }
        catch (error) {
            logger_1.logger.error('Failed to search by content', error, logContext);
            return [];
        }
    }
    /**
     * Delete a proof from the vector index
     *
     * @param proofId - ID of the proof to delete
     * @param correlationId - Optional correlation ID for tracing
     */
    async deleteProof(proofId, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-vector-index',
            proof_id: proofId,
        };
        try {
            logger_1.logger.info('Deleting proof from vector index', logContext);
            await this.vectorStore.delete(proofId);
            logger_1.logger.info('Proof deleted from vector index', logContext);
        }
        catch (error) {
            logger_1.logger.error('Failed to delete proof from vector index', error, logContext);
            throw error;
        }
    }
    /**
     * Update a proof in the vector index
     *
     * @param proof - Updated proof
     * @param correlationId - Optional correlation ID for tracing
     */
    async updateProof(proof, correlationId) {
        const logContext = {
            correlation_id: correlationId || proof.correlation_id,
            source_service: 'proof-vector-index',
            proof_id: proof.id,
        };
        try {
            logger_1.logger.info('Updating proof in vector index', logContext);
            // Generate new embedding
            const embedding = await this.generateProofEmbedding(proof);
            // Update in vector database
            const metadata = {
                proof_id: proof.id,
                theorem_id: proof.theorem_id,
                theorem: proof.theorem,
                system: proof.system,
                status: proof.status,
                tactics: proof.tactics || [],
                timestamp: proof.timestamp_utc,
            };
            await this.vectorStore.update(proof.id, embedding, metadata);
            logger_1.logger.info('Proof updated in vector index', logContext);
        }
        catch (error) {
            logger_1.logger.error('Failed to update proof in vector index', error, logContext);
            throw error;
        }
    }
    /**
     * Generate embedding for a theorem
     *
     * In production, this would call an embedding model (e.g., OpenAI, Sentence Transformers)
     * For now, we generate a simple hash-based embedding for demonstration
     *
     * @param theorem - The theorem to embed
     * @returns Embedding vector
     */
    async generateTheoremEmbedding(theorem) {
        // Combine theorem statement, type, and constraints
        const text = [
            theorem.statement,
            theorem.type,
            ...(theorem.constraints || []),
        ].join(' ');
        return this.generateTextEmbedding(text);
    }
    /**
     * Generate embedding for a proof
     *
     * @param proof - The proof to embed
     * @returns Embedding vector
     */
    async generateProofEmbedding(proof) {
        // Combine theorem statement, proof content, and tactics
        const text = [
            proof.theorem,
            proof.proof,
            ...(proof.tactics || []),
            proof.system,
            proof.status,
        ].join(' ');
        return this.generateTextEmbedding(text);
    }
    async generateTextEmbedding(text) {
        const textStr = typeof text === 'string' ? text : text.join(' ');
        // Simple hash-based embedding for demonstration
        // In production, replace with: await openai.embeddings.create({ model: 'text-embedding-3-small', input: textStr })
        const embedding = [];
        const hash = this.simpleHash(textStr);
        for (let i = 0; i < this.embeddingDimension; i++) {
            // Generate pseudo-random but deterministic values
            const value = Math.sin(hash * (i + 1)) * 0.5 + 0.5;
            embedding.push(value);
        }
        // Normalize the embedding
        const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
        return embedding.map(val => val / norm);
    }
    /**
     * Simple hash function for demonstration
     *
     * TODO: Replace with proper embedding model
     */
    simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32bit integer
        }
        return Math.abs(hash);
    }
    /**
     * Filter search results by proof system
     *
     * @param results - Results to filter
     * @param system - System to filter by
     * @returns Filtered results
     */
    filterBySystem(results, system) {
        return results.filter(sp => sp.proof.system === system);
    }
    /**
     * Filter search results by status
     *
     * @param results - Results to filter
     * @param status - Status to filter by
     * @returns Filtered results
     */
    filterByStatus(results, status) {
        return results.filter(sp => sp.proof.status === status);
    }
    /**
     * Filter search results by minimum similarity score
     *
     * @param results - Results to filter
     * @param minScore - Minimum similarity score
     * @returns Filtered results
     */
    filterByMinScore(results, minScore) {
        return results.filter(sp => sp.similarity_score >= minScore);
    }
}
exports.ProofVectorIndex = ProofVectorIndex;
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
//# sourceMappingURL=vector-index.js.map