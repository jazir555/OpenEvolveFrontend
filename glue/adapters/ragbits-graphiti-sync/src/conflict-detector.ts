/**
 * Conflict Detector
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Detect conflicts from actual data
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Graceful handling of detection errors
 *
 * Detects conflicts between RAGBits and Graphiti data during synchronization
 */

import { v4 as uuidv4 } from 'uuid';
import { logger, Logger } from '../../lib/logger';
import {
  Conflict,
  ConflictReport,
  ConflictType,
  ConflictSeverity,
  ConflictResolution,
  SyncOperation,
  SyncDirection,
} from './canonical';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * RAGBits data for conflict detection
 */
export interface RAGBitsData {
  chunks?: DocumentChunkData[];
  entities?: Map<string, any>;
  metadata?: Record<string, any>;
}

/**
 * Document chunk data
 */
export interface DocumentChunkData {
  id: string;
  content: string;
  source: string;
  chunk_index: number;
  metadata?: Record<string, any>;
  timestamp: string;
}

/**
 * Graphiti data for conflict detection
 */
export interface GraphitiData {
  episodes?: GraphitiEpisodeData[];
  entities?: Map<string, GraphitiEntityData>;
  relationships?: Map<string, GraphitiRelationshipData>;
  metadata?: Record<string, any>;
}

/**
 * Graphiti episode data
 */
export interface GraphitiEpisodeData {
  id: string;
  name: string;
  content: string;
  valid_at: string;
  created_at: string;
  entity_edges: string[];
  metadata?: Record<string, any>;
}

/**
 * Graphiti entity data
 */
export interface GraphitiEntityData {
  id: string;
  name: string;
  labels: string[];
  summary?: string;
  attributes: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

/**
 * Graphiti relationship data
 */
export interface GraphitiRelationshipData {
  id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  fact: string;
  valid_at: string;
  created_at: string;
}

/**
 * Conflict detection configuration
 */
export interface ConflictDetectorConfig {
  enable_entity_detection: boolean;
  enable_temporal_detection: boolean;
  enable_semantic_detection: boolean;
  semantic_similarity_threshold: number;
  temporal_drift_threshold_ms: number;
  auto_resolve_minor_conflicts: boolean;
}

// ============================================================================
// MAIN CONFLICT DETECTOR CLASS
// ============================================================================

/**
 * Conflict Detector
 *
 * Detects conflicts between RAGBits and Graphiti data
 */
export class ConflictDetector {
  private readonly config: ConflictDetectorConfig;
  private readonly logger: Logger;
  private readonly serviceName = 'conflict-detector';

  constructor(config: ConflictDetectorConfig) {
    this.config = config;
    this.logger = new Logger(this.serviceName);
  }

  /**
   * Detect conflicts between RAGBits and Graphiti data
   *
   * @param ragbitsData - Data from RAGBits system
   * @param graphitiData - Data from Graphiti system
   * @param syncOperation - Sync operation context
   * @returns Conflict report
   */
  async detectConflicts(
    ragbitsData: RAGBitsData,
    graphitiData: GraphitiData,
    syncOperation: SyncOperation
  ): Promise<ConflictReport> {
    this.logger.info('Starting conflict detection', {
      correlation_id: syncOperation.correlation_id,
      sync_operation_id: syncOperation.id,
      direction: syncOperation.direction,
    });

    const conflicts: Conflict[] = [];
    const startTime = Date.now();

    // Detect entity conflicts
    if (this.config.enable_entity_detection) {
      const entityConflicts = await this.detectEntityConflicts(
        ragbitsData,
        graphitiData,
        syncOperation.correlation_id
      );
      conflicts.push(...entityConflicts);
    }

    // Detect temporal conflicts
    if (this.config.enable_temporal_detection) {
      const temporalConflicts = await this.detectTemporalConflicts(
        ragbitsData,
        graphitiData,
        syncOperation.correlation_id
      );
      conflicts.push(...temporalConflicts);
    }

    // Detect semantic conflicts
    if (this.config.enable_semantic_detection) {
      const semanticConflicts = await this.detectSemanticConflicts(
        ragbitsData,
        graphitiData,
        syncOperation.correlation_id
      );
      conflicts.push(...semanticConflicts);
    }

    // Create conflict report
    const report: ConflictReport = {
      sync_operation_id: syncOperation.id,
      conflicts,
      resolutions: [],
      unresolved: conflicts.filter((c) => !c.resolved).map((c) => c.id),
      total_conflicts: conflicts.length,
      resolved_count: 0,
      unresolved_count: conflicts.length,
      timestamp_utc: new Date().toISOString(),
      correlation_id: syncOperation.correlation_id,
    };

    const durationMs = Date.now() - startTime;

    this.logger.info('Conflict detection completed', {
      correlation_id: syncOperation.correlation_id,
      total_conflicts: conflicts.length,
      duration_ms: durationMs,
      high_severity: conflicts.filter((c) => c.severity === 'high' || c.severity === 'critical')
        .length,
    });

    return report;
  }

  /**
   * Detect entity conflicts between systems
   *
   * @param ragbitsData - RAGBits data
   * @param graphitiData - Graphiti data
   * @param correlationId - Correlation ID for tracing
   * @returns Array of entity conflicts
   */
  private async detectEntityConflicts(
    ragbitsData: RAGBitsData,
    graphitiData: GraphitiData,
    correlationId: string
  ): Promise<Conflict[]> {
    this.logger.debug('Detecting entity conflicts', {
      correlation_id: correlationId,
    });

    const conflicts: Conflict[] = [];

    if (!ragbitsData.chunks || !graphitiData.episodes) {
      return conflicts;
    }

    // Check for entity mismatches
    const ragbitsEntities = this.extractEntitiesFromChunks(ragbitsData.chunks);
    const graphitiEntities = new Map(Object.entries(graphitiData.entities || {}));

    for (const [entityName, ragbitsEntity] of ragbitsEntities) {
      const graphitiEntity = graphitiEntities.get(entityName);

      if (graphitiEntity) {
        // Check for attribute mismatches
        const attributeConflicts = this.detectAttributeMismatches(
          ragbitsEntity,
          graphitiEntity,
          correlationId
        );
        conflicts.push(...attributeConflicts);
      }
    }

    this.logger.debug('Entity conflict detection completed', {
      correlation_id: correlationId,
      conflicts_count: conflicts.length,
    });

    return conflicts;
  }

  /**
   * Detect temporal conflicts between systems
   *
   * @param ragbitsData - RAGBits data
   * @param graphitiData - Graphiti data
   * @param correlationId - Correlation ID for tracing
   * @returns Array of temporal conflicts
   */
  private async detectTemporalConflicts(
    ragbitsData: RAGBitsData,
    graphitiData: GraphitiData,
    correlationId: string
  ): Promise<Conflict[]> {
    this.logger.debug('Detecting temporal conflicts', {
      correlation_id: correlationId,
    });

    const conflicts: Conflict[] = [];

    if (!ragbitsData.chunks || !graphitiData.episodes) {
      return conflicts;
    }

    // Check for temporal drifts
    for (const chunk of ragbitsData.chunks) {
      const episode = graphitiData.episodes.find(
        (ep) => ep.metadata?.ragbits_chunk_id === chunk.id
      );

      if (episode) {
        const chunkTime = new Date(chunk.timestamp).getTime();
        const episodeTime = new Date(episode.valid_at).getTime();
        const drift = Math.abs(chunkTime - episodeTime);

        if (drift > this.config.temporal_drift_threshold_ms) {
          conflicts.push({
            id: uuidv4(),
            type: ConflictType.temporal_inconsistency,
            severity: this.calculateTemporalSeverity(drift),
            ragbits_data: { timestamp: chunk.timestamp },
            graphiti_data: { valid_at: episode.valid_at },
            chunk_id: chunk.id,
            episode_id: episode.id,
            description: `Temporal drift detected: ${drift}ms difference between RAGBits chunk and Graphiti episode`,
            detected_at_utc: new Date().toISOString(),
            resolved: false,
            correlation_id: correlationId,
          });
        }
      }
    }

    this.logger.debug('Temporal conflict detection completed', {
      correlation_id: correlationId,
      conflicts_count: conflicts.length,
    });

    return conflicts;
  }

  /**
   * Detect semantic conflicts between systems
   *
   * @param ragbitsData - RAGBits data
   * @param graphitiData - Graphiti data
   * @param correlationId - Correlation ID for tracing
   * @returns Array of semantic conflicts
   */
  private async detectSemanticConflicts(
    ragbitsData: RAGBitsData,
    graphitiData: GraphitiData,
    correlationId: string
  ): Promise<Conflict[]> {
    this.logger.debug('Detecting semantic conflicts', {
      correlation_id: correlationId,
    });

    const conflicts: Conflict[] = [];

    if (!ragbitsData.chunks || !graphitiData.episodes) {
      return conflicts;
    }

    // Check for semantic differences
    for (const chunk of ragbitsData.chunks) {
      const episode = graphitiData.episodes.find(
        (ep) => ep.metadata?.ragbits_chunk_id === chunk.id
      );

      if (episode) {
        // Calculate semantic similarity
        const similarity = this.calculateSemanticSimilarity(chunk.content, episode.content);

        if (similarity < this.config.semantic_similarity_threshold) {
          conflicts.push({
            id: uuidv4(),
            type: ConflictType.semantic_conflict,
            severity: ConflictSeverity.medium,
            ragbits_data: { content: chunk.content },
            graphiti_data: { content: episode.content },
            chunk_id: chunk.id,
            episode_id: episode.id,
            description: `Semantic conflict detected: similarity score ${similarity.toFixed(2)} below threshold ${this.config.semantic_similarity_threshold}`,
            suggested_resolution: 'merge',
            detected_at_utc: new Date().toISOString(),
            resolved: false,
            correlation_id: correlationId,
          });
        }
      }
    }

    this.logger.debug('Semantic conflict detection completed', {
      correlation_id: correlationId,
      conflicts_count: conflicts.length,
    });

    return conflicts;
  }

  /**
   * Detect attribute mismatches between entities
   *
   * @param ragbitsEntity - Entity from RAGBits
   * @param graphitiEntity - Entity from Graphiti
   * @param correlationId - Correlation ID for tracing
   * @returns Array of conflicts
   */
  private detectAttributeMismatches(
    ragbitsEntity: any,
    graphitiEntity: GraphitiEntityData,
    correlationId: string
  ): Conflict[] {
    const conflicts: Conflict[] = [];

    // Check for label mismatches
    const ragbitsLabels = ragbitsEntity.labels || [];
    const graphitiLabels = graphitiEntity.labels || [];

    const missingInGraphiti = ragbitsLabels.filter((l: string) => !graphitiLabels.includes(l));
    const missingInRagbits = graphitiLabels.filter((l) => !ragbitsLabels.includes(l));

    if (missingInGraphiti.length > 0 || missingInRagBits.length > 0) {
      conflicts.push({
        id: uuidv4(),
        type: ConflictType.entity_mismatch,
        severity: ConflictSeverity.low,
        ragbits_data: { labels: ragbitsLabels },
        graphiti_data: { labels: graphitiLabels },
        entity_id: graphitiEntity.id,
        description: `Label mismatch: RAGBits has [${ragbitsLabels.join(', ')}], Graphiti has [${graphitiLabels.join(', ')}]`,
        suggested_resolution: 'merge',
        detected_at_utc: new Date().toISOString(),
        resolved: false,
        correlation_id: correlationId,
      });
    }

    return conflicts;
  }

  /**
   * Extract entities from document chunks
   *
   * @param chunks - Document chunks
   * @returns Map of entity name to entity data
   */
  private extractEntitiesFromChunks(chunks: DocumentChunkData[]): Map<string, any> {
    const entities = new Map<string, any>();

    for (const chunk of chunks) {
      // Extract entities from metadata
      if (chunk.metadata?.entities) {
        for (const entity of chunk.metadata.entities) {
          if (!entities.has(entity.name)) {
            entities.set(entity.name, entity);
          }
        }
      }
    }

    return entities;
  }

  /**
   * Calculate semantic similarity between two texts
   *
   * @param text1 - First text
   * @param text2 - Second text
   * @returns Similarity score (0-1)
   */
  private calculateSemanticSimilarity(text1: string, text2: string): number {
    // Simple word overlap similarity (placeholder)
    // In production, this would use actual embeddings
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));

    const intersection = new Set([...words1].filter((x) => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    if (union.size === 0) {
      return 1.0;
    }

    return intersection.size / union.size;
  }

  /**
   * Calculate severity of temporal conflict
   *
   * @param driftMs - Temporal drift in milliseconds
   * @returns Severity level
   */
  private calculateTemporalSeverity(driftMs: number): ConflictSeverity {
    if (driftMs > 86400000) {
      // More than 1 day
      return ConflictSeverity.critical;
    } if (driftMs > 3600000) {
      // More than 1 hour
      return ConflictSeverity.high;
    } if (driftMs > 60000) {
      // More than 1 minute
      return ConflictSeverity.medium;
    }
    return ConflictSeverity.low;
  }

  /**
   * Auto-resolve conflicts if enabled
   *
   * @param conflicts - Array of conflicts
   * @param resolutionStrategy - Resolution strategy to apply
   * @returns Array of resolved conflict IDs
   */
  autoResolveConflicts(conflicts: Conflict[], resolutionStrategy: ConflictResolution): string[] {
    if (!this.config.auto_resolve_minor_conflicts) {
      return [];
    }

    const resolved: string[] = [];

    for (const conflict of conflicts) {
      // Only auto-resolve low severity conflicts
      if (conflict.severity === ConflictSeverity.low && !conflict.resolved) {
        conflict.resolved = true;
        conflict.resolution_strategy = resolutionStrategy;
        conflict.resolution_notes = 'Auto-resolved by conflict detector';
        resolved.push(conflict.id);

        this.logger.info('Auto-resolved conflict', {
          conflict_id: conflict.id,
          conflict_type: conflict.type,
          strategy: resolutionStrategy,
        });
      }
    }

    return resolved;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default ConflictDetector;
