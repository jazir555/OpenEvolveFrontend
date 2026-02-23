/**
 * Proof Knowledge Base Repository
 *
 * Main interface for storing, searching, and managing formal proofs.
 * Unifies vector index (semantic search) and graph index (lineage tracking).
 *
 * Federation Constitution Compliance:
 * - Law of Configuration Explicitness: All env vars validated at startup
 * - Law of Idempotency: All operations are safe to retry
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Prevents cascading failures
 * - Retry Logic: Handles transient failures
 */
import { FormalProof, Theorem, SimilarProof, ProofLineage, ProofMetrics, StorageResult, UpdateResult } from './canonical';
/**
 * Proof Knowledge Base Configuration
 */
interface ProofKnowledgeBaseConfig {
    vectorIndexEnabled: boolean;
    graphIndexEnabled: boolean;
    validationEnabled: boolean;
    autoValidateOnStore: boolean;
    z3ApiUrl?: string;
    leanaideApiUrl?: string;
}
/**
 * Proof Knowledge Base
 *
 * Main repository for storing and searching formal proofs
 */
export declare class ProofKnowledgeBase {
    private vectorIndex;
    private graphIndex;
    private validator;
    private config;
    private proofStorage;
    private theoremStorage;
    constructor(config?: Partial<ProofKnowledgeBaseConfig>);
    /**
     * Store a proof in the knowledge base
     *
     * Idempotent operation: Can be called multiple times safely
     *
     * @param proof - The proof to store
     * @param correlationId - Optional correlation ID for tracing
     * @returns Storage result
     */
    storeProof(proof: FormalProof, correlationId?: string): Promise<StorageResult>;
    /**
     * Store a theorem in the knowledge base
     *
     * @param theorem - The theorem to store
     * @param correlationId - Optional correlation ID for tracing
     * @returns Storage result
     */
    storeTheorem(theorem: Theorem, correlationId?: string): Promise<StorageResult>;
    /**
     * Search for similar proofs
     *
     * @param theorem - The theorem to find similar proofs for
     * @param maxResults - Maximum number of results to return
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    searchSimilar(theorem: Theorem, maxResults?: number, correlationId?: string): Promise<SimilarProof[]>;
    /**
     * Search proofs by content (natural language query)
     *
     * @param query - Natural language query
     * @param maxResults - Maximum number of results
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of similar proofs
     */
    searchByContent(query: string, maxResults?: number, correlationId?: string): Promise<SimilarProof[]>;
    /**
     * Validate proof dependencies
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Whether dependencies are valid
     */
    validateDependencies(proofId: string, correlationId?: string): Promise<boolean>;
    /**
     * Get proof lineage
     *
     * @param proofId - ID of the proof
     * @param depth - Depth of lineage to traverse
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof lineage
     */
    getProofLineage(proofId: string, depth?: number, correlationId?: string): Promise<ProofLineage | null>;
    /**
     * Update a proof
     *
     * Idempotent operation
     *
     * @param proofId - ID of the proof to update
     * @param newProof - Updated proof data
     * @param correlationId - Optional correlation ID for tracing
     * @returns Update result
     */
    updateProof(proofId: string, newProof: FormalProof, correlationId?: string): Promise<UpdateResult>;
    /**
     * Get a proof by ID
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof or null if not found
     */
    getProof(proofId: string, correlationId?: string): Promise<FormalProof | null>;
    /**
     * Get a theorem by ID
     *
     * @param theoremId - ID of the theorem
     * @param correlationId - Optional correlation ID for tracing
     * @returns Theorem or null if not found
     */
    getTheorem(theoremId: string, correlationId?: string): Promise<Theorem | null>;
    /**
     * Get knowledge base metrics
     *
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof metrics
     */
    getMetrics(correlationId?: string): Promise<ProofMetrics>;
    /**
     * Validate environment variables (Law of Configuration Explicitness)
     *
     * Crashes immediately if required configuration is missing
     */
    private validateEnvironment;
    /**
     * Generate a UUID v4
     */
    private generateId;
    /**
     * Delete a proof from the knowledge base
     *
     * @param proofId - ID of the proof to delete
     * @param correlationId - Optional correlation ID for tracing
     */
    deleteProof(proofId: string, correlationId?: string): Promise<void>;
}
export {};
/**
 * Example usage:
 *
 * ```typescript
 * import { ProofKnowledgeBase } from './repository';
 * import { FormalProof, Theorem } from './canonical';
 *
 * // Create knowledge base
 * const kb = new ProofKnowledgeBase({
 *   vectorIndexEnabled: true,
 *   graphIndexEnabled: true,
 *   validationEnabled: true,
 *   autoValidateOnStore: true,
 *   z3ApiUrl: 'http://z3-core:8000',
 *   leanaideApiUrl: 'http://leanaide-core:8000',
 * });
 *
 * // Store a proof
 * const proof: FormalProof = { ... };
 * await kb.storeProof(proof, 'correlation-123');
 *
 * // Store a theorem
 * const theorem: Theorem = { ... };
 * await kb.storeTheorem(theorem, 'correlation-123');
 *
 * // Search for similar proofs
 * const similar = await kb.searchSimilar(theorem, 10, 'correlation-123');
 *
 * // Get proof lineage
 * const lineage = await kb.getProofLineage(proof.id, 3, 'correlation-123');
 *
 * // Get metrics
 * const metrics = await kb.getMetrics('correlation-123');
 * ```
 */
//# sourceMappingURL=repository.d.ts.map