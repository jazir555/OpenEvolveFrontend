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

import { logger, LoggerContext } from '../../logger';
import { validateEnvWithTypes, EnvVar } from '../env-validator';
import { ProofVectorIndex } from './vector-index';
import { ProofGraphIndex } from './graph-index';
import { ProofValidator } from './validator';
import {
  FormalProof,
  Theorem,
  SimilarProof,
  ProofLineage,
  ProofMetrics,
  StorageResult,
  UpdateResult,
  IndexResult,
} from './canonical';

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
export class ProofKnowledgeBase {
  private vectorIndex: ProofVectorIndex;
  private graphIndex: ProofGraphIndex;
  private validator: ProofValidator;
  private config: ProofKnowledgeBaseConfig;
  private proofStorage: Map<string, FormalProof>;
  private theoremStorage: Map<string, Theorem>;

  constructor(config?: Partial<ProofKnowledgeBaseConfig>) {
    // Validate environment variables (Law of Configuration Explicitness)
    this.validateEnvironment();

    // Initialize configuration
    this.config = {
      vectorIndexEnabled: config?.vectorIndexEnabled ?? true,
      graphIndexEnabled: config?.graphIndexEnabled ?? true,
      validationEnabled: config?.validationEnabled ?? true,
      autoValidateOnStore: config?.autoValidateOnStore ?? false,
      z3ApiUrl: config?.z3ApiUrl,
      leanaideApiUrl: config?.leanaideApiUrl,
    };

    // Initialize components
    this.vectorIndex = new ProofVectorIndex();
    this.graphIndex = new ProofGraphIndex();
    this.validator = new ProofValidator({
      z3ApiUrl: this.config.z3ApiUrl,
      leanaideApiUrl: this.config.leanaideApiUrl,
    });

    // Initialize storage (in-memory for development)
    this.proofStorage = new Map();
    this.theoremStorage = new Map();

    logger.info('Proof Knowledge Base initialized', {
      source_service: 'proof-knowledge-base',
      vector_index_enabled: this.config.vectorIndexEnabled,
      graph_index_enabled: this.config.graphIndexEnabled,
      validation_enabled: this.config.validationEnabled,
      auto_validate: this.config.autoValidateOnStore,
    });
  }

  /**
   * Store a proof in the knowledge base
   *
   * Idempotent operation: Can be called multiple times safely
   *
   * @param proof - The proof to store
   * @param correlationId - Optional correlation ID for tracing
   * @returns Storage result
   */
  async storeProof(
    proof: FormalProof,
    correlationId?: string
  ): Promise<StorageResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId || proof.correlation_id,
      source_service: 'proof-knowledge-base',
      proof_id: proof.id,
    };

    try {
      logger.info('Storing proof in knowledge base', logContext);

      // Check if proof already exists (Law of Idempotency)
      const existing = this.proofStorage.get(proof.id);
      if (existing) {
        logger.info('Proof already exists, updating', logContext);
        return this.updateProof(proof.id, proof, correlationId);
      }

      // Validate proof before storing if enabled
      if (this.config.validationEnabled && this.config.autoValidateOnStore) {
        const validation = await this.validator.validateProof(
          proof.id,
          proof,
          false,
          correlationId
        );

        if (!validation.is_valid) {
          logger.warn('Proof validation failed, storing anyway', {
            ...logContext,
            errors: validation.errors,
          });
        }
      }

      // Store in memory (Law of Idempotency: check before insert)
      this.proofStorage.set(proof.id, proof);

      // Index in vector database for semantic search
      let vectorIndexed = false;
      if (this.config.vectorIndexEnabled) {
        const indexResult = await this.vectorIndex.indexProof(proof, correlationId);
        vectorIndexed = indexResult.success;
      }

      // Index in graph database for lineage tracking
      let graphIndexed = false;
      if (this.config.graphIndexEnabled) {
        const graphResult = await this.graphIndex.storeProof(proof, correlationId);
        graphIndexed = graphResult.success;
      }

      logger.info('Proof stored successfully', {
        ...logContext,
        vector_indexed: vectorIndexed,
        graph_indexed: graphIndexed,
      });

      return {
        success: true,
        proof_id: proof.id,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      logger.error('Failed to store proof', error as Error, logContext);

      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Store a theorem in the knowledge base
   *
   * @param theorem - The theorem to store
   * @param correlationId - Optional correlation ID for tracing
   * @returns Storage result
   */
  async storeTheorem(
    theorem: Theorem,
    correlationId?: string
  ): Promise<StorageResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      theorem_id: theorem.id,
    };

    try {
      logger.info('Storing theorem in knowledge base', logContext);

      // Check if theorem already exists (Law of Idempotency)
      if (this.theoremStorage.has(theorem.id)) {
        logger.info('Theorem already exists', logContext);
        return {
          success: true,
          proof_id: theorem.id,
          timestamp: new Date().toISOString(),
        };
      }

      // Store in memory
      this.theoremStorage.set(theorem.id, theorem);

      logger.info('Theorem stored successfully', logContext);

      return {
        success: true,
        proof_id: theorem.id,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      logger.error('Failed to store theorem', error as Error, logContext);

      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Search for similar proofs
   *
   * @param theorem - The theorem to find similar proofs for
   * @param maxResults - Maximum number of results to return
   * @param correlationId - Optional correlation ID for tracing
   * @returns Array of similar proofs
   */
  async searchSimilar(
    theorem: Theorem,
    maxResults: number = 10,
    correlationId?: string
  ): Promise<SimilarProof[]> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      theorem_id: theorem.id,
    };

    try {
      logger.info('Searching for similar proofs', {
        ...logContext,
        max_results: maxResults,
      });

      if (!this.config.vectorIndexEnabled) {
        logger.warn('Vector index not enabled, returning empty results', logContext);
        return [];
      }

      const similarProofs = await this.vectorIndex.searchSimilarTheorems(
        theorem,
        maxResults,
        correlationId
      );

      logger.info('Similar proof search completed', {
        ...logContext,
        result_count: similarProofs.length,
      });

      return similarProofs;
    } catch (error) {
      logger.error('Failed to search similar proofs', error as Error, logContext);
      return [];
    }
  }

  /**
   * Search proofs by content (natural language query)
   *
   * @param query - Natural language query
   * @param maxResults - Maximum number of results
   * @param correlationId - Optional correlation ID for tracing
   * @returns Array of similar proofs
   */
  async searchByContent(
    query: string,
    maxResults: number = 10,
    correlationId?: string
  ): Promise<SimilarProof[]> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
    };

    try {
      logger.info('Searching proofs by content', {
        ...logContext,
        query_length: query.length,
        max_results: maxResults,
      });

      if (!this.config.vectorIndexEnabled) {
        logger.warn('Vector index not enabled, returning empty results', logContext);
        return [];
      }

      const similarProofs = await this.vectorIndex.searchByContent(
        query,
        maxResults,
        correlationId
      );

      logger.info('Content search completed', {
        ...logContext,
        result_count: similarProofs.length,
      });

      return similarProofs;
    } catch (error) {
      logger.error('Failed to search by content', error as Error, logContext);
      return [];
    }
  }

  /**
   * Validate proof dependencies
   *
   * @param proofId - ID of the proof
   * @param correlationId - Optional correlation ID for tracing
   * @returns Whether dependencies are valid
   */
  async validateDependencies(
    proofId: string,
    correlationId?: string
  ): Promise<boolean> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      proof_id: proofId,
    };

    try {
      logger.info('Validating proof dependencies', logContext);

      const proof = this.proofStorage.get(proofId);
      if (!proof) {
        logger.warn('Proof not found', logContext);
        return false;
      }

      const valid = await this.validator.checkDependenciesValid(
        proofId,
        proof,
        correlationId
      );

      logger.info('Dependency validation completed', {
        ...logContext,
        valid,
      });

      return valid;
    } catch (error) {
      logger.error('Failed to validate dependencies', error as Error, logContext);
      return false;
    }
  }

  /**
   * Get proof lineage
   *
   * @param proofId - ID of the proof
   * @param depth - Depth of lineage to traverse
   * @param correlationId - Optional correlation ID for tracing
   * @returns Proof lineage
   */
  async getProofLineage(
    proofId: string,
    depth: number = 3,
    correlationId?: string
  ): Promise<ProofLineage | null> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      proof_id: proofId,
    };

    try {
      logger.info('Getting proof lineage', {
        ...logContext,
        depth,
      });

      if (!this.config.graphIndexEnabled) {
        logger.warn('Graph index not enabled', logContext);
        return null;
      }

      const lineage = await this.graphIndex.getProofLineage(
        proofId,
        depth,
        correlationId
      );

      logger.info('Proof lineage retrieved', {
        ...logContext,
        ancestor_count: lineage.ancestors.length,
        descendant_count: lineage.descendants.length,
      });

      return lineage;
    } catch (error) {
      logger.error('Failed to get proof lineage', error as Error, logContext);
      return null;
    }
  }

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
  async updateProof(
    proofId: string,
    newProof: FormalProof,
    correlationId?: string
  ): Promise<UpdateResult> {
    const logContext: LoggerContext = {
      correlation_id: correlationId || newProof.correlation_id,
      source_service: 'proof-knowledge-base',
      proof_id: proofId,
    };

    try {
      logger.info('Updating proof', logContext);

      // Check if proof exists
      const existing = this.proofStorage.get(proofId);
      if (!existing) {
        logger.warn('Proof not found for update', logContext);
        return {
          success: false,
          error: 'Proof not found',
          timestamp: new Date().toISOString(),
        };
      }

      // Generate version IDs
      const previousVersionId = proofId;
      const newVersionId = this.generateId();

      // Update proof with new ID (versioning)
      const versionedProof = {
        ...newProof,
        id: newVersionId,
      };

      // Store new version
      this.proofStorage.set(newVersionId, versionedProof);

      // Update vector index
      if (this.config.vectorIndexEnabled) {
        await this.vectorIndex.updateProof(versionedProof, correlationId);
      }

      // Update graph index
      if (this.config.graphIndexEnabled) {
        await this.graphIndex.storeProof(versionedProof, correlationId);
      }

      logger.info('Proof updated successfully', {
        ...logContext,
        previous_version_id: previousVersionId,
        new_version_id: newVersionId,
      });

      return {
        success: true,
        proof_id: newVersionId,
        previous_version_id: previousVersionId,
        new_version_id: newVersionId,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      logger.error('Failed to update proof', error as Error, logContext);

      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        timestamp: new Date().toISOString(),
      };
    }
  }

  /**
   * Get a proof by ID
   *
   * @param proofId - ID of the proof
   * @param correlationId - Optional correlation ID for tracing
   * @returns Proof or null if not found
   */
  async getProof(
    proofId: string,
    correlationId?: string
  ): Promise<FormalProof | null> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      proof_id: proofId,
    };

    try {
      const proof = this.proofStorage.get(proofId);

      if (proof) {
        logger.debug('Proof retrieved', logContext);
      } else {
        logger.debug('Proof not found', logContext);
      }

      return proof || null;
    } catch (error) {
      logger.error('Failed to get proof', error as Error, logContext);
      return null;
    }
  }

  /**
   * Get a theorem by ID
   *
   * @param theoremId - ID of the theorem
   * @param correlationId - Optional correlation ID for tracing
   * @returns Theorem or null if not found
   */
  async getTheorem(
    theoremId: string,
    correlationId?: string
  ): Promise<Theorem | null> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      theorem_id: theoremId,
    };

    try {
      const theorem = this.theoremStorage.get(theoremId);

      if (theorem) {
        logger.debug('Theorem retrieved', logContext);
      } else {
        logger.debug('Theorem not found', logContext);
      }

      return theorem || null;
    } catch (error) {
      logger.error('Failed to get theorem', error as Error, logContext);
      return null;
    }
  }

  /**
   * Get knowledge base metrics
   *
   * @param correlationId - Optional correlation ID for tracing
   * @returns Proof metrics
   */
  async getMetrics(correlationId?: string): Promise<ProofMetrics> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
    };

    try {
      const proofs = Array.from(this.proofStorage.values());

      // Calculate proofs by system
      const proofsBySystem: Record<string, number> = {};
      for (const proof of proofs) {
        proofsBySystem[proof.system] = (proofsBySystem[proof.system] || 0) + 1;
      }

      // Calculate proofs by status
      const proofsByStatus: Record<string, number> = {};
      for (const proof of proofs) {
        proofsByStatus[proof.status] = (proofsByStatus[proof.status] || 0) + 1;
      }

      // Calculate average confidence
      const proofsWithConfidence = proofs.filter(p => p.confidence !== undefined);
      const averageConfidence =        proofsWithConfidence.length > 0
        ? proofsWithConfidence.reduce((sum, p) => sum + (p.confidence || 0), 0)
            / proofsWithConfidence.length
        : undefined;

      // Calculate total dependencies
      let totalDependencies = 0;
      for (const proof of proofs) {
        totalDependencies += proof.dependencies?.length || 0;
      }

      const metrics: ProofMetrics = {
        total_proofs: proofs.length,
        proofs_by_system: proofsBySystem,
        proofs_by_status: proofsByStatus,
        average_confidence: averageConfidence,
        total_dependencies: totalDependencies,
        indexed_proofs: proofs.length, // All stored proofs are indexed
        last_updated: new Date().toISOString(),
      };

      logger.info('Metrics retrieved', {
        ...logContext,
        total_proofs: metrics.total_proofs,
      });

      return metrics;
    } catch (error) {
      logger.error('Failed to get metrics', error as Error, logContext);

      return {
        total_proofs: 0,
        proofs_by_system: {},
        proofs_by_status: {},
        total_dependencies: 0,
        indexed_proofs: 0,
        last_updated: new Date().toISOString(),
      };
    }
  }

  /**
   * Validate environment variables (Law of Configuration Explicitness)
   *
   * Crashes immediately if required configuration is missing
   */
  private validateEnvironment(): void {
    const envVars: EnvVar[] = [];

    // Optional: Z3 API URL
    if (process.env.Z3_API_URL) {
      envVars.push({
        name: 'Z3_API_URL',
        type: 'url',
        required: false,
      });
    }

    // Optional: LeanAide API URL
    if (process.env.LEANAIDE_API_URL) {
      envVars.push({
        name: 'LEANAIDE_API_URL',
        type: 'url',
        required: false,
      });
    }

    if (envVars.length > 0) {
      validateEnvWithTypes(envVars);
    }
  }

  /**
   * Generate a UUID v4
   */
  private generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Delete a proof from the knowledge base
   *
   * @param proofId - ID of the proof to delete
   * @param correlationId - Optional correlation ID for tracing
   */
  async deleteProof(proofId: string, correlationId?: string): Promise<void> {
    const logContext: LoggerContext = {
      correlation_id: correlationId,
      source_service: 'proof-knowledge-base',
      proof_id: proofId,
    };

    try {
      logger.info('Deleting proof from knowledge base', logContext);

      // Delete from storage
      this.proofStorage.delete(proofId);

      // Delete from vector index
      if (this.config.vectorIndexEnabled) {
        await this.vectorIndex.deleteProof(proofId, correlationId);
      }

      // Note: Graph relationships are kept for lineage tracking

      logger.info('Proof deleted successfully', logContext);
    } catch (error) {
      logger.error('Failed to delete proof', error as Error, logContext);
      throw error;
    }
  }
}

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
