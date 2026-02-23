/**
 * Proof Graph Index
 *
 * Manages the graph structure of proofs and their dependencies using Graphiti.
 * Tracks lineage, dependencies, and relationships between proofs.
 *
 * Federation Constitution Compliance:
 * - Law of Configuration Explicitness: All URLs via env vars
 * - Law of UTC: All timestamps in UTC
 * - Idempotency: Safe to replay operations
 */
import { FormalProof, ProofDependency, ProofLineage, ProofHistory, StorageResult } from './canonical';
/**
 * Graph Entity Interface
 *
 * Represents an entity in the knowledge graph
 */
interface GraphEntity {
    name: string;
    entityType: string;
    observations: Array<{
        name: string;
        createdAt: string;
    }>;
    metadata?: Record<string, any>;
}
/**
 * Graph Relationship Interface
 *
 * Represents a relationship between entities
 */
interface GraphRelationship {
    from: string;
    to: string;
    relationshipName: string;
    createdAt: string;
    metadata?: Record<string, any>;
}
/**
 * Graph Episode Interface
 *
 * Represents an episode (event) in the knowledge graph
 */
interface GraphEpisode {
    name: string;
    episodeType: string;
    entities: GraphEntity[];
    relationships: GraphRelationship[];
    createdAt: string;
    metadata?: Record<string, any>;
}
/**
 * Graph Client Interface
 *
 * Abstracts the graph database (Graphiti, Neo4j, etc.)
 */
interface GraphClient {
    addEpisode(episode: GraphEpisode): Promise<void>;
    searchEntities(query: string, entityType?: string): Promise<GraphEntity[]>;
    getEntity(name: string): Promise<GraphEntity | null>;
    searchRelationships(fromEntity: string, relationshipName?: string): Promise<GraphRelationship[]>;
    getEntityRelationships(entityName: string): Promise<GraphRelationship[]>;
}
/**
 * Proof Graph Index
 *
 * Manages the graph structure of proofs and their dependencies
 */
export declare class ProofGraphIndex {
    private graphClient;
    constructor(graphClient?: GraphClient);
    /**
     * Store a proof in the graph
     *
     * Creates entities for the proof and theorem, and relationships between them
     *
     * @param proof - The proof to store
     * @param correlationId - Optional correlation ID for tracing
     * @returns Storage result
     */
    storeProof(proof: FormalProof, correlationId?: string): Promise<StorageResult>;
    /**
     * Link a theorem to a proof
     *
     * @param theoremId - ID of the theorem
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     */
    linkTheoremToProof(theoremId: string, proofId: string, correlationId?: string): Promise<void>;
    /**
     * Get the lineage of a proof (ancestors and descendants)
     *
     * @param proofId - ID of the proof
     * @param depth - Depth of lineage to traverse
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof lineage
     */
    getProofLineage(proofId: string, depth?: number, correlationId?: string): Promise<ProofLineage>;
    /**
     * Get dependencies for a proof
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of proof dependencies
     */
    getProofDependencies(proofId: string, correlationId?: string): Promise<ProofDependency[]>;
    /**
     * Trace the history of a proof
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof history
     */
    traceProofHistory(proofId: string, correlationId?: string): Promise<ProofHistory>;
    /**
     * Search for proofs by theorem
     *
     * @param theoremStatement - The theorem statement
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of proof IDs
     */
    searchByTheorem(theoremStatement: string, correlationId?: string): Promise<string[]>;
    /**
     * Create a graph episode from a proof
     *
     * @param proof - The proof
     * @returns Graph episode
     */
    private createProofEpisode;
    /**
     * Create a graph entity from a theorem
     *
     * @param theorem - The theorem
     * @returns Graph entity
     */
    private createTheoremEntity;
    /**
     * Create dependency relationships from a proof
     *
     * @param proof - The proof
     * @returns Array of relationships
     */
    private createProofDependencies;
    /**
     * Get ancestor proofs
     *
     * @param proofId - Proof ID
     * @param depth - Traversal depth
     * @param correlationId - Optional correlation ID
     * @returns Array of ancestors
     */
    private getAncestors;
    /**
     * Get descendant proofs
     *
     * @param proofId - Proof ID
     * @param depth - Traversal depth
     * @param correlationId - Optional correlation ID
     * @returns Array of descendants
     */
    private getDescendants;
}
export {};
/**
 * Example usage:
 *
 * ```typescript
 * import { ProofGraphIndex } from './graph-index';
 * import { FormalProof } from './canonical';
 *
 * // Create graph index
 * const graphIndex = new ProofGraphIndex();
 *
 * // Store a proof
 * const proof: FormalProof = { ... };
 * await graphIndex.storeProof(proof, 'correlation-123');
 *
 * // Get lineage
 * const lineage = await graphIndex.getProofLineage(proof.id, 3, 'correlation-123');
 *
 * // Get dependencies
 * const deps = await graphIndex.getProofDependencies(proof.id);
 *
 * // Search by theorem
 * const proofIds = await graphIndex.searchByTheorem('commutative property');
 * ```
 */
//# sourceMappingURL=graph-index.d.ts.map