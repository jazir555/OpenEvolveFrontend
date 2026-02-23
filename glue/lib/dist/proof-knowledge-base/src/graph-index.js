"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ProofGraphIndex = void 0;
const logger_1 = require("../logger");
/**
 * In-memory graph client (for development/testing)
 *
 * Production should use Graphiti SDK
 */
class InMemoryGraphClient {
    constructor() {
        this.entities = new Map();
        this.relationships = [];
        this.episodes = [];
    }
    async addEpisode(episode) {
        this.episodes.push(episode);
        // Add entities
        for (const entity of episode.entities) {
            if (!this.entities.has(entity.name)) {
                this.entities.set(entity.name, entity);
            }
            else {
                // Merge observations
                const existing = this.entities.get(entity.name);
                existing.observations.push(...entity.observations);
            }
        }
        // Add relationships
        this.relationships.push(...episode.relationships);
    }
    async searchEntities(query, entityType) {
        const results = [];
        for (const entity of this.entities.values()) {
            if (entityType && entity.entityType !== entityType) {
                continue;
            }
            if (entity.name.toLowerCase().includes(query.toLowerCase())) {
                results.push(entity);
            }
        }
        return results;
    }
    async getEntity(name) {
        return this.entities.get(name) || null;
    }
    async searchRelationships(fromEntity, relationshipName) {
        return this.relationships.filter(r => {
            if (r.from !== fromEntity) {
                return false;
            }
            if (relationshipName && r.relationshipName !== relationshipName) {
                return false;
            }
            return true;
        });
    }
    async getEntityRelationships(entityName) {
        return this.relationships.filter(r => r.from === entityName || r.to === entityName);
    }
}
/**
 * Proof Graph Index
 *
 * Manages the graph structure of proofs and their dependencies
 */
class ProofGraphIndex {
    constructor(graphClient) {
        this.graphClient = graphClient || new InMemoryGraphClient();
    }
    /**
     * Store a proof in the graph
     *
     * Creates entities for the proof and theorem, and relationships between them
     *
     * @param proof - The proof to store
     * @param correlationId - Optional correlation ID for tracing
     * @returns Storage result
     */
    async storeProof(proof, correlationId) {
        const logContext = {
            correlation_id: correlationId || proof.correlation_id,
            source_service: 'proof-graph-index',
            proof_id: proof.id,
        };
        try {
            logger_1.logger.info('Storing proof in graph index', logContext);
            // Create graph episode
            const episode = this.createProofEpisode(proof);
            // Add to graph
            await this.graphClient.addEpisode(episode);
            logger_1.logger.info('Proof stored in graph index', {
                ...logContext,
                entity_count: episode.entities.length,
                relationship_count: episode.relationships.length,
            });
            return {
                success: true,
                proof_id: proof.id,
                timestamp: new Date().toISOString(),
            };
        }
        catch (error) {
            logger_1.logger.error('Failed to store proof in graph index', error, logContext);
            return {
                success: false,
                error: error instanceof Error ? error.message : String(error),
                timestamp: new Date().toISOString(),
            };
        }
    }
    /**
     * Link a theorem to a proof
     *
     * @param theoremId - ID of the theorem
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     */
    async linkTheoremToProof(theoremId, proofId, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-graph-index',
            theorem_id: theoremId,
            proof_id: proofId,
        };
        try {
            logger_1.logger.info('Linking theorem to proof', logContext);
            const episode = {
                name: `link-${theoremId}-${proofId}`,
                episodeType: 'theorem_proof_link',
                entities: [],
                relationships: [
                    {
                        from: `theorem:${theoremId}`,
                        to: `proof:${proofId}`,
                        relationshipName: 'PROVED_BY',
                        createdAt: new Date().toISOString(),
                    },
                ],
                createdAt: new Date().toISOString(),
            };
            await this.graphClient.addEpisode(episode);
            logger_1.logger.info('Theorem linked to proof', logContext);
        }
        catch (error) {
            logger_1.logger.error('Failed to link theorem to proof', error, logContext);
            throw error;
        }
    }
    /**
     * Get the lineage of a proof (ancestors and descendants)
     *
     * @param proofId - ID of the proof
     * @param depth - Depth of lineage to traverse
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof lineage
     */
    async getProofLineage(proofId, depth = 3, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-graph-index',
            proof_id: proofId,
        };
        try {
            logger_1.logger.info('Getting proof lineage', {
                ...logContext,
                depth,
            });
            // Get ancestors (proofs this depends on)
            const ancestors = await this.getAncestors(proofId, depth, correlationId);
            // Get descendants (proofs that depend on this)
            const descendants = await this.getDescendants(proofId, depth, correlationId);
            const lineage = {
                proof_id: proofId,
                ancestors,
                descendants,
                full_tree: {
                    ancestors,
                    descendants,
                    depth,
                },
                computed_at: new Date().toISOString(),
            };
            logger_1.logger.info('Proof lineage retrieved', {
                ...logContext,
                ancestor_count: ancestors.length,
                descendant_count: descendants.length,
            });
            return lineage;
        }
        catch (error) {
            logger_1.logger.error('Failed to get proof lineage', error, logContext);
            throw error;
        }
    }
    /**
     * Get dependencies for a proof
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of proof dependencies
     */
    async getProofDependencies(proofId, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-graph-index',
            proof_id: proofId,
        };
        try {
            logger_1.logger.info('Getting proof dependencies', logContext);
            const relationships = await this.graphClient.searchRelationships(`proof:${proofId}`, 'DEPENDS_ON');
            const dependencies = relationships.map((rel, idx) => ({
                id: `dep-${proofId}-${idx}`,
                proof_id: proofId,
                depends_on_proof_id: rel.to.replace('proof:', ''),
                type: 'direct_dependency',
                validity: true,
                created_at: rel.createdAt,
            }));
            logger_1.logger.info('Proof dependencies retrieved', {
                ...logContext,
                dependency_count: dependencies.length,
            });
            return dependencies;
        }
        catch (error) {
            logger_1.logger.error('Failed to get proof dependencies', error, logContext);
            return [];
        }
    }
    /**
     * Trace the history of a proof
     *
     * @param proofId - ID of the proof
     * @param correlationId - Optional correlation ID for tracing
     * @returns Proof history
     */
    async traceProofHistory(proofId, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-graph-index',
            proof_id: proofId,
        };
        try {
            logger_1.logger.info('Tracing proof history', logContext);
            // Get the proof entity
            const entity = await this.graphClient.getEntity(`proof:${proofId}`);
            if (!entity) {
                logger_1.logger.warn('Proof entity not found', logContext);
                return {
                    proof_id: proofId,
                    versions: [],
                    version_count: 0,
                };
            }
            // Extract version information from observations
            const versions = entity.observations
                .filter(obs => obs.name.startsWith('version:'))
                .map(obs => ({
                version_id: obs.name.replace('version:', ''),
                timestamp: obs.createdAt,
                changes: [],
            }));
            const history = {
                proof_id: proofId,
                versions,
                version_count: versions.length,
                current_version_id: versions.length > 0 ? versions[versions.length - 1].version_id : undefined,
            };
            logger_1.logger.info('Proof history traced', {
                ...logContext,
                version_count: history.version_count,
            });
            return history;
        }
        catch (error) {
            logger_1.logger.error('Failed to trace proof history', error, logContext);
            return {
                proof_id: proofId,
                versions: [],
                version_count: 0,
            };
        }
    }
    /**
     * Search for proofs by theorem
     *
     * @param theoremStatement - The theorem statement
     * @param correlationId - Optional correlation ID for tracing
     * @returns Array of proof IDs
     */
    async searchByTheorem(theoremStatement, correlationId) {
        const logContext = {
            correlation_id: correlationId,
            source_service: 'proof-graph-index',
        };
        try {
            logger_1.logger.info('Searching proofs by theorem', {
                ...logContext,
                theorem_length: theoremStatement.length,
            });
            const entities = await this.graphClient.searchEntities(theoremStatement, 'proof');
            const proofIds = entities.map(e => e.name.replace('proof:', ''));
            logger_1.logger.info('Proof search completed', {
                ...logContext,
                result_count: proofIds.length,
            });
            return proofIds;
        }
        catch (error) {
            logger_1.logger.error('Failed to search proofs by theorem', error, logContext);
            return [];
        }
    }
    /**
     * Create a graph episode from a proof
     *
     * @param proof - The proof
     * @returns Graph episode
     */
    createProofEpisode(proof) {
        const entities = [];
        const relationships = [];
        // Create proof entity
        entities.push({
            name: `proof:${proof.id}`,
            entityType: 'proof',
            observations: [
                {
                    name: `system:${proof.system}`,
                    createdAt: proof.timestamp_utc,
                },
                {
                    name: `status:${proof.status}`,
                    createdAt: proof.timestamp_utc,
                },
                {
                    name: `theorem:${proof.theorem.substring(0, 100)}`,
                    createdAt: proof.timestamp_utc,
                },
            ],
            metadata: {
                theorem_id: proof.theorem_id,
                confidence: proof.confidence,
                tactics: proof.tactics,
            },
        });
        // Create theorem entity
        entities.push({
            name: `theorem:${proof.theorem_id}`,
            entityType: 'theorem',
            observations: [
                {
                    name: `statement:${proof.theorem.substring(0, 100)}`,
                    createdAt: proof.timestamp_utc,
                },
            ],
        });
        // Create relationship: theorem -> PROVED_BY -> proof
        relationships.push({
            from: `theorem:${proof.theorem_id}`,
            to: `proof:${proof.id}`,
            relationshipName: 'PROVED_BY',
            createdAt: proof.timestamp_utc,
        });
        // Create dependency relationships
        if (proof.dependencies && proof.dependencies.length > 0) {
            for (const depId of proof.dependencies) {
                relationships.push({
                    from: `proof:${proof.id}`,
                    to: `proof:${depId}`,
                    relationshipName: 'DEPENDS_ON',
                    createdAt: proof.timestamp_utc,
                });
            }
        }
        return {
            name: `proof-episode-${proof.id}`,
            episodeType: 'proof_creation',
            entities,
            relationships,
            createdAt: proof.timestamp_utc,
            metadata: {
                proof_id: proof.id,
                correlation_id: proof.correlation_id,
            },
        };
    }
    /**
     * Create a graph entity from a theorem
     *
     * @param theorem - The theorem
     * @returns Graph entity
     */
    createTheoremEntity(theorem) {
        return {
            name: `theorem:${theorem.id}`,
            entityType: 'theorem',
            observations: [
                {
                    name: `statement:${theorem.statement.substring(0, 100)}`,
                    createdAt: theorem.created_at,
                },
                {
                    name: `type:${theorem.type}`,
                    createdAt: theorem.created_at,
                },
            ],
            metadata: {
                constraints: theorem.constraints,
                dependencies: theorem.dependencies,
            },
        };
    }
    /**
     * Create dependency relationships from a proof
     *
     * @param proof - The proof
     * @returns Array of relationships
     */
    createProofDependencies(proof) {
        const relationships = [];
        if (proof.dependencies && proof.dependencies.length > 0) {
            for (const depId of proof.dependencies) {
                relationships.push({
                    from: `proof:${proof.id}`,
                    to: `proof:${depId}`,
                    relationshipName: 'DEPENDS_ON',
                    createdAt: proof.timestamp_utc,
                });
            }
        }
        return relationships;
    }
    /**
     * Get ancestor proofs
     *
     * @param proofId - Proof ID
     * @param depth - Traversal depth
     * @param correlationId - Optional correlation ID
     * @returns Array of ancestors
     */
    async getAncestors(proofId, depth, correlationId) {
        const ancestors = [];
        const visited = new Set();
        const queue = [
            { proof_id: proofId, depth: 0 },
        ];
        while (queue.length > 0) {
            const current = queue.shift();
            if (current.depth >= depth || visited.has(current.proof_id)) {
                continue;
            }
            visited.add(current.proof_id);
            const relationships = await this.graphClient.searchRelationships(`proof:${current.proof_id}`, 'DEPENDS_ON');
            for (const rel of relationships) {
                const depId = rel.to.replace('proof:', '');
                if (!visited.has(depId)) {
                    ancestors.push({
                        proof_id: depId,
                        depth: current.depth + 1,
                        relationship: 'dependency',
                    });
                    queue.push({ proof_id: depId, depth: current.depth + 1 });
                }
            }
        }
        return ancestors;
    }
    /**
     * Get descendant proofs
     *
     * @param proofId - Proof ID
     * @param depth - Traversal depth
     * @param correlationId - Optional correlation ID
     * @returns Array of descendants
     */
    async getDescendants(proofId, depth, correlationId) {
        const descendants = [];
        const visited = new Set();
        // Get all relationships and find ones that point to this proof
        const allRelationships = await this.graphClient.getEntityRelationships(`proof:${proofId}`);
        const dependents = allRelationships.filter(r => r.to === `proof:${proofId}`);
        const queue = dependents.map(r => ({
            proof_id: r.from.replace('proof:', ''),
            depth: 1,
        }));
        while (queue.length > 0) {
            const current = queue.shift();
            if (current.depth >= depth || visited.has(current.proof_id)) {
                continue;
            }
            visited.add(current.proof_id);
            descendants.push({
                proof_id: current.proof_id,
                depth: current.depth,
                relationship: 'dependent',
            });
            // Recursively find dependents
            const childRelationships = await this.graphClient.getEntityRelationships(`proof:${current.proof_id}`);
            const childDependents = childRelationships.filter(r => r.to === `proof:${current.proof_id}`);
            for (const rel of childDependents) {
                const depId = rel.from.replace('proof:', '');
                if (!visited.has(depId)) {
                    queue.push({ proof_id: depId, depth: current.depth + 1 });
                }
            }
        }
        return descendants;
    }
}
exports.ProofGraphIndex = ProofGraphIndex;
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
//# sourceMappingURL=graph-index.js.map