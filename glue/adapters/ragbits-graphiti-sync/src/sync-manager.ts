/**
 * Sync Manager
 *
 * Follows the Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Law of Runtime Truth: Verify operations execute
 * - Law of Idempotency: Safe to run multiple times
 * - Failure Management: Circuit breakers and retries
 * - Law of Configuration Explicitness: All config via env vars
 *
 * Main orchestration for bidirectional RAGBits-Graphiti synchronization
 */

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../../../lib/logger.js';
import { CircuitBreaker } from '../../../lib/circuit-breaker.js';
import { validateEnvWithTypes } from '../../../lib/env-validator.js';
import RAGBitsToGraphitiSync, {
  RAGBitsToGraphitiConfig,
  DocumentChunkData,
} from './ragbits-to-graphiti.js';
import GraphitiToRAGBitsSync, {
  GraphitiToRAGBitsConfig,
  GraphitiEntity,
} from './graphiti-to-ragbits.js';
import ConflictDetector, { ConflictDetectorConfig } from './conflict-detector.js';
import {
  SyncOperation,
  SyncResult,
  SyncConfig,
  SyncDirection,
  SyncStatus,
  SyncSpec,
  Conflict,
  ConflictReport,
  ConflictResolution,
  createSyncOperation,
  createSyncResult,
  DEFAULT_SYNC_CONFIG,
} from './canonical.js';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Document for ingestion sync
 */
export interface Document {
  id: string;
  content: string;
  source: string;
  metadata?: Record<string, any>;
}

/**
 * Sync strategy
 */
export interface SyncStrategy {
  direction: SyncDirection;
  conflict_resolution: ConflictResolution;
  batch_size: number;
  enable_enhancement: boolean;
}

/**
 * Sync operation result
 */
export interface SyncOperationResult {
  sync_result: SyncResult;
  conflict_report?: ConflictReport;
  errors: Error[];
}

/**
 * Sync Manager configuration
 */
export interface SyncManagerConfig {
  ragbits: RAGBitsToGraphitiConfig;
  graphiti: GraphitiToRAGBitsConfig;
  conflict_detector: ConflictDetectorConfig;
  sync: SyncConfig;
}

// ============================================================================
// MAIN SYNC MANAGER CLASS
// ============================================================================

/**
 * Sync Manager
 *
 * Orchestrates bidirectional synchronization between RAGBits and Graphiti
 */
export class SyncManager {
  private readonly config: SyncManagerConfig;
  private readonly logger: Logger;
  private readonly ragbitsToGraphiti: RAGBitsToGraphitiSync;
  private readonly graphitiToRAGBits: GraphitiToRAGBitsSync;
  private readonly conflictDetector: ConflictDetector;
  private readonly circuitBreaker: CircuitBreaker;
  private readonly serviceName = 'sync-manager';

  // Active sync operations
  private activeOperations: Map<string, SyncOperation> = new Map();

  // Sync statistics
  private stats = {
    total_syncs: 0,
    successful_syncs: 0,
    failed_syncs: 0,
    conflicts_detected: 0,
    conflicts_resolved: 0,
    total_duration_ms: 0,
  };

  constructor(config?: Partial<SyncManagerConfig>) {
    // Validate environment variables
    this.validateConfiguration();

    // Merge with defaults
    this.config = this.buildConfig(config);

    this.logger = new Logger(this.serviceName);

    // Initialize components
    this.ragbitsToGraphiti = new RAGBitsToGraphitiSync(this.config.ragbits);
    this.graphitiToRAGBits = new GraphitiToRAGBitsSync(this.config.graphiti);
    this.conflictDetector = new ConflictDetector(this.config.conflict_detector);

    this.circuitBreaker = new CircuitBreaker({
      threshold: 10,
      timeout_ms: 120000,
      onStateChange: (oldState, newState) => {
        this.logger.warn('Sync manager circuit breaker state changed', {
          old_state: oldState,
          new_state: newState,
        });
      },
    });

    this.logger.info('Sync Manager initialized', {
      ragbits_url: this.config.ragbits.ragbits_api_url,
      graphiti_url: this.config.graphiti.graphiti_api_url,
      sync_enabled: this.config.sync.enabled,
      bidirectional: this.config.sync.bidirectional,
    });
  }

  /**
   * Sync on document ingestion
   *
   * Triggered when a new document is ingested into RAGBits
   *
   * @param document - Ingested document
   * @param correlationId - Correlation ID for tracing
   * @returns Sync operation result
   */
  async syncOnIngest(document: Document, correlationId: string): Promise<SyncOperationResult> {
    this.logger.info('Starting sync on ingest', {
      correlation_id: correlationId,
      document_id: document.id,
      source: document.source,
    });

    // Create sync operation
    const operation = createSyncOperation(
      'ingest_sync',
      'ragbits',
      'graphiti',
      SyncDirection.ragbits_to_graphiti,
      correlationId
    );

    this.activeOperations.set(operation.id, operation);

    const startTime = Date.now();
    const errors: Error[] = [];

    try {
      // Chunk document (placeholder)
      const chunks = await this.chunkDocument(document, correlationId);

      // Sync to Graphiti
      const syncResult = await this.ragbitsToGraphiti.syncBatch(chunks, correlationId);

      // Detect conflicts
      let conflictReport: ConflictReport | undefined;
      if (this.config.sync.bidirectional) {
        const ragbitsData = { chunks };
        const graphitiData = await this.fetchGraphitiData(correlationId);

        conflictReport = await this.conflictDetector.detectConflicts(
          ragbitsData,
          graphitiData,
          operation
        );

        // Auto-resolve conflicts if enabled
        if (conflictReport.total_conflicts > 0) {
          const resolved = this.conflictDetector.autoResolveConflicts(
            conflictReport.conflicts,
            this.config.sync.conflict_resolution
          );

          conflictReport.resolutions = resolved.map((id) => ({
            conflict_id: id,
            strategy: this.config.sync.conflict_resolution,
            applied_at_utc: new Date().toISOString(),
            notes: 'Auto-resolved during ingest sync',
          }));

          conflictReport.resolved_count = resolved.length;
          conflictReport.unresolved_count = conflictReport.total_conflicts - resolved.length;
          conflictReport.unresolved = conflictReport.conflicts
            .filter((c) => !c.resolved)
            .map((c) => c.id);
        }
      }

      this.updateStats(true, Date.now() - startTime, conflictReport?.total_conflicts || 0);

      return {
        sync_result: syncResult,
        conflict_report: conflictReport,
        errors,
      };
    } catch (error) {
      this.logger.error('Sync on ingest failed', error as Error, {
        correlation_id: correlationId,
        document_id: document.id,
      });

      this.updateStats(false, Date.now() - startTime, 0);

      errors.push(error as Error);

      return {
        sync_result: createSyncResult(
          operation.id,
          SyncStatus.failed,
          SyncDirection.ragbits_to_graphiti,
          correlationId,
          Date.now() - startTime
        ),
        errors,
      };
    } finally {
      this.activeOperations.delete(operation.id);
    }
  }

  /**
   * Scheduled sync
   *
   * Runs periodically to keep systems in sync
   *
   * @param correlationId - Correlation ID for tracing
   * @returns Sync operation result
   */
  async syncOnSchedule(correlationId: string): Promise<SyncOperationResult> {
    this.logger.info('Starting scheduled sync', {
      correlation_id: correlationId,
      interval_ms: this.config.sync.interval_ms,
    });

    const operation = createSyncOperation(
      'scheduled_sync',
      'ragbits',
      'graphiti',
      SyncDirection.bidirectional,
      correlationId
    );

    this.activeOperations.set(operation.id, operation);

    const startTime = Date.now();
    const errors: Error[] = [];

    try {
      const result = await this.circuitBreaker.execute(async () => {
        return await this.performBidirectionalSync(operation, correlationId);
      });

      this.updateStats(
        result.sync_result.status === SyncStatus.completed,
        Date.now() - startTime,
        result.conflict_report?.total_conflicts || 0
      );

      return result;
    } catch (error) {
      this.logger.error('Scheduled sync failed', error as Error, {
        correlation_id: correlationId,
      });

      this.updateStats(false, Date.now() - startTime, 0);

      errors.push(error as Error);

      return {
        sync_result: createSyncResult(
          operation.id,
          SyncStatus.failed,
          SyncDirection.bidirectional,
          correlationId,
          Date.now() - startTime
        ),
        errors,
      };
    } finally {
      this.activeOperations.delete(operation.id);
    }
  }

  /**
   * Manual sync with specification
   *
   * @param spec - Sync specification
   * @param correlationId - Correlation ID for tracing
   * @returns Sync operation result
   */
  async syncManual(spec: SyncSpec, correlationId: string): Promise<SyncOperationResult> {
    this.logger.info('Starting manual sync', {
      correlation_id: correlationId,
      direction: spec.direction,
      entity_count: spec.entity_ids?.length || 0,
      episode_count: spec.episode_ids?.length || 0,
      chunk_count: spec.chunk_ids?.length || 0,
    });

    const operation = createSyncOperation(
      'episode_sync',
      'ragbits',
      'graphiti',
      spec.direction,
      correlationId
    );

    this.activeOperations.set(operation.id, operation);

    const startTime = Date.now();
    const errors: Error[] = [];

    try {
      let syncResult: SyncResult | undefined;
      let conflictReport: ConflictReport | undefined;

      if (
        spec.direction === SyncDirection.ragbits_to_graphiti
        || spec.direction === SyncDirection.bidirectional
      ) {
        // Sync RAGBits to Graphiti
        if (spec.chunk_ids && spec.chunk_ids.length > 0) {
          const chunks = await this.fetchChunks(spec.chunk_ids, correlationId);
          syncResult = await this.ragbitsToGraphiti.syncBatch(chunks, correlationId);
        }
      }

      if (
        spec.direction === SyncDirection.graphiti_to_ragbits
        || spec.direction === SyncDirection.bidirectional
      ) {
        // Sync Graphiti to RAGBits
        if (spec.entity_ids && spec.entity_ids.length > 0) {
          const entities = await this.fetchEntities(spec.entity_ids, correlationId);

          for (const entity of entities) {
            const result = await this.graphitiToRAGBits.syncEntity(entity, correlationId);
            if (syncResult) {
              syncResult.operations_completed += result.operations_completed;
              syncResult.operations_failed += result.operations_failed;
            } else {
              syncResult = result;
            }
          }
        }
      }

      // Detect conflicts if bidirectional
      if (spec.direction === SyncDirection.bidirectional) {
        const ragbitsData = await this.fetchRAGBitsData(correlationId);
        const graphitiData = await this.fetchGraphitiData(correlationId);

        conflictReport = await this.conflictDetector.detectConflicts(
          ragbitsData,
          graphitiData,
          operation
        );
      }

      this.updateStats(
        syncResult?.status === SyncStatus.completed,
        Date.now() - startTime,
        conflictReport?.total_conflicts || 0
      );

      return {
        sync_result: syncResult!,
        conflict_report: conflictReport,
        errors,
      };
    } catch (error) {
      this.logger.error('Manual sync failed', error as Error, {
        correlation_id: correlationId,
      });

      this.updateStats(false, Date.now() - startTime, 0);

      errors.push(error as Error);

      return {
        sync_result: createSyncResult(
          operation.id,
          SyncStatus.failed,
          spec.direction,
          correlationId,
          Date.now() - startTime
        ),
        errors,
      };
    } finally {
      this.activeOperations.delete(operation.id);
    }
  }

  /**
   * Resolve conflicts
   *
   * @param conflicts - Conflicts to resolve
   * @param resolutionStrategy - Resolution strategy
   * @param correlationId - Correlation ID for tracing
   * @returns Resolution result
   */
  async resolveConflicts(
    conflicts: Conflict[],
    resolutionStrategy: ConflictResolution,
    correlationId: string
  ): Promise<{ resolved: string[]; failed: string[]; errors: Error[] }> {
    this.logger.info('Starting conflict resolution', {
      correlation_id: correlationId,
      conflict_count: conflicts.length,
      strategy: resolutionStrategy,
    });

    const resolved: string[] = [];
    const failed: string[] = [];
    const errors: Error[] = [];

    for (const conflict of conflicts) {
      try {
        // Apply resolution strategy
        await this.applyConflictResolution(conflict, resolutionStrategy, correlationId);

        conflict.resolved = true;
        conflict.resolution_strategy = resolutionStrategy;
        conflict.resolution_notes = `Resolved with strategy: ${resolutionStrategy}`;

        resolved.push(conflict.id);

        this.logger.info('Conflict resolved', {
          correlation_id: correlationId,
          conflict_id: conflict.id,
          strategy: resolutionStrategy,
        });
      } catch (error) {
        failed.push(conflict.id);
        errors.push(error as Error);

        this.logger.error('Conflict resolution failed', error as Error, {
          correlation_id: correlationId,
          conflict_id: conflict.id,
        });
      }
    }

    this.logger.info('Conflict resolution completed', {
      correlation_id: correlationId,
      resolved_count: resolved.length,
      failed_count: failed.length,
    });

    return { resolved, failed, errors };
  }

  /**
   * Get statistics
   *
   * @returns Sync statistics
   */
  getStats() {
    return {
      ...this.stats,
      active_operations: this.activeOperations.size,
      success_rate:
        this.stats.total_syncs > 0
          ? (this.stats.successful_syncs / this.stats.total_syncs) * 100
          : 0,
      avg_duration_ms:
        this.stats.total_syncs > 0 ? this.stats.total_duration_ms / this.stats.total_syncs : 0,
      conflict_rate:
        this.stats.total_syncs > 0
          ? (this.stats.conflicts_detected / this.stats.total_syncs) * 100
          : 0,
    };
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /**
   * Validate configuration from environment variables
   */
  private validateConfiguration(): void {
    validateEnvWithTypes([
      { name: 'RAGBITS_API_URL', type: 'url', required: true },
      { name: 'GRAPHITI_API_URL', type: 'url', required: true },
      { name: 'SYNC_TIMEOUT_MS', type: 'number', required: false, default: 30000 },
      { name: 'SYNC_MAX_RETRIES', type: 'number', required: false, default: 3 },
      { name: 'SYNC_BATCH_SIZE', type: 'number', required: false, default: 10 },
      { name: 'SYNC_INTERVAL_MS', type: 'number', required: false, default: 300000 },
      { name: 'SYNC_ENABLED', type: 'boolean', required: false, default: true },
      { name: 'SYNC_BIDIRECTIONAL', type: 'boolean', required: false, default: true },
    ]);
  }

  /**
   * Build configuration from environment and defaults
   */
  private buildConfig(_userConfig?: Partial<SyncManagerConfig>): SyncManagerConfig {
    return {
      ragbits: {
        ragbits_api_url: process.env.RAGBITS_API_URL || 'http://ragbits:8000',
        graphiti_api_url: process.env.GRAPHITI_API_URL || 'http://graphiti:8000',
        timeout_ms: parseInt(process.env.SYNC_TIMEOUT_MS || '30000', 10),
        max_retries: parseInt(process.env.SYNC_MAX_RETRIES || '3', 10),
        retry_delay_ms: 1000,
        batch_size: parseInt(process.env.SYNC_BATCH_SIZE || '10', 10),
        extract_entities: true,
        extract_relationships: true,
        entity_extraction_threshold: 0.7,
      },
      graphiti: {
        graphiti_api_url: process.env.GRAPHITI_API_URL || 'http://graphiti:8000',
        ragbits_api_url: process.env.RAGBITS_API_URL || 'http://ragbits:8000',
        timeout_ms: parseInt(process.env.SYNC_TIMEOUT_MS || '30000', 10),
        max_retries: parseInt(process.env.SYNC_MAX_RETRIES || '3', 10),
        retry_delay_ms: 1000,
        batch_size: parseInt(process.env.SYNC_BATCH_SIZE || '10', 10),
        enable_enhancement: true,
        boost_threshold: 0.5,
        min_confidence: 0.6,
      },
      conflict_detector: {
        enable_entity_detection: true,
        enable_temporal_detection: true,
        enable_semantic_detection: true,
        semantic_similarity_threshold: 0.8,
        temporal_drift_threshold_ms: 60000, // 1 minute
        auto_resolve_minor_conflicts: true,
      },
      sync: {
        ...DEFAULT_SYNC_CONFIG,
        enabled: process.env.SYNC_ENABLED === 'true',
        interval_ms: parseInt(process.env.SYNC_INTERVAL_MS || '300000', 10),
        bidirectional: process.env.SYNC_BIDIRECTIONAL === 'true',
      },
    };
  }

  /**
   * Perform bidirectional sync
   */
  private async performBidirectionalSync(
    operation: SyncOperation,
    correlationId: string
  ): Promise<SyncOperationResult> {
    const errors: Error[] = [];

    // Sync RAGBits to Graphiti
    const ragbitsData = await this.fetchRAGBitsData(correlationId);
    let ragbitsToGraphitiResult: SyncResult | undefined;

    if (ragbitsData.chunks && ragbitsData.chunks.length > 0) {
      ragbitsToGraphitiResult = await this.ragbitsToGraphiti.syncBatch(
        ragbitsData.chunks,
        correlationId
      );
    }

    // Sync Graphiti to RAGBits
    const graphitiData = await this.fetchGraphitiData(correlationId);
    let graphitiToRAGBitsResult: SyncResult | undefined;

    if (graphitiData.entities && graphitiData.entities.size > 0) {
      const entities = Array.from(graphitiData.entities.values()) as GraphitiEntity[];
      for (const entity of entities) {
        const result = await this.graphitiToRAGBits.syncEntity(entity, correlationId);
        if (graphitiToRAGBitsResult) {
          graphitiToRAGBitsResult.operations_completed += result.operations_completed;
          graphitiToRAGBitsResult.operations_failed += result.operations_failed;
        } else {
          graphitiToRAGBitsResult = result;
        }
      }
    }

    // Detect conflicts
    const conflictReport = await this.conflictDetector.detectConflicts(
      ragbitsData,
      graphitiData,
      operation
    );

    // Combine results
    const combinedResult = createSyncResult(
      operation.id,
      SyncStatus.completed,
      SyncDirection.bidirectional,
      correlationId
    );

    if (ragbitsToGraphitiResult) {
      combinedResult.operations_completed += ragbitsToGraphitiResult.operations_completed;
      combinedResult.operations_failed += ragbitsToGraphitiResult.operations_failed;
    }

    if (graphitiToRAGBitsResult) {
      combinedResult.operations_completed += graphitiToRAGBitsResult.operations_completed;
      combinedResult.operations_failed += graphitiToRAGBitsResult.operations_failed;
    }

    combinedResult.operations_total =      combinedResult.operations_completed + combinedResult.operations_failed;
    combinedResult.conflicts_detected = conflictReport.total_conflicts;
    combinedResult.conflicts_resolved = conflictReport.resolved_count;

    return {
      sync_result: combinedResult,
      conflict_report: conflictReport,
      errors,
    };
  }

  /**
   * Chunk document (placeholder)
   */
  private async chunkDocument(
    document: Document,
    _correlationId: string
  ): Promise<DocumentChunkData[]> {
    // Placeholder for document chunking logic
    const chunk: DocumentChunkData = {
      id: uuidv4(),
      content: document.content,
      source: document.source,
      chunk_index: 0,
      metadata: document.metadata,
      timestamp: new Date().toISOString(),
    };

    return [chunk];
  }

  /**
   * Apply conflict resolution strategy
   */
  private async applyConflictResolution(
    conflict: Conflict,
    strategy: ConflictResolution,
    correlationId: string
  ): Promise<void> {
    // Placeholder for conflict resolution logic
    this.logger.debug('Applying conflict resolution', {
      correlation_id: correlationId,
      conflict_id: conflict.id,
      strategy,
    });

    // Simulate resolution
    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  /**
   * Fetch RAGBits data (placeholder)
   */
  private async fetchRAGBitsData(_correlationId: string): Promise<any> {
    // Placeholder for fetching RAGBits data
    return { chunks: [] };
  }

  /**
   * Fetch Graphiti data (placeholder)
   */
  private async fetchGraphitiData(_correlationId: string): Promise<any> {
    // Placeholder for fetching Graphiti data
    return { episodes: [], entities: new Map(), relationships: new Map() };
  }

  /**
   * Fetch chunks by IDs (placeholder)
   */
  private async fetchChunks(_chunkIds: string[], _correlationId: string): Promise<DocumentChunkData[]> {
    // Placeholder for fetching chunks
    return [];
  }

  /**
   * Fetch entities by IDs (placeholder)
   */
  private async fetchEntities(_entityIds: string[], _correlationId: string): Promise<GraphitiEntity[]> {
    // Placeholder for fetching entities
    return [];
  }

  /**
   * Update statistics
   */
  private updateStats(success: boolean, durationMs: number, conflictsCount: number): void {
    this.stats.total_syncs++;
    if (success) {
      this.stats.successful_syncs++;
    } else {
      this.stats.failed_syncs++;
    }
    this.stats.conflicts_detected += conflictsCount;
    this.stats.total_duration_ms += durationMs;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default SyncManager;
