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

import { v4 as uuidv4 } from 'uuid';
import { Logger } from '../../logger';
import { validateEnvWithTypes } from '../../env-validator';
import { VectorStorage, VectorStorageConfig } from './vector-storage';
import { GraphStorage, GraphStorageConfig } from './graph-storage';
import {
  EvolvedCode,
  Problem,
  EvolutionMetrics,
  EvolutionLineage,
  SimilarSolution,
  CaptureResult,
  CaptureMetrics,
  StoreWithEmbeddingRequest,
  SearchSimilarRequest,
  GetLineageRequest,
  validateEvolvedCode,
  validateProblem,
  validateEvolutionMetrics,
  validateCaptureResult,
} from './canonical';

// ============================================================================
// CONFIGURATION
// ============================================================================

export interface EvolvedCodeCapturerConfig {
  // Vector storage configuration
  vector_storage: VectorStorageConfig;

  // Graph storage configuration
  graph_storage: GraphStorageConfig;

  // Capture configuration
  enable_vector_storage: boolean;
  enable_graph_storage: boolean;

  // Timeout and retry configuration
  timeout_ms?: number;
  max_retries?: number;

  // Metrics configuration
  track_metrics: boolean;
  metrics_retention_days: number;

  // Logging
  logger?: Logger;
}

const DEFAULT_CONFIG = {
  timeout_ms: 60000, // 1 minute
  max_retries: 3,
  track_metrics: true,
  metrics_retention_days: 30,
  enable_vector_storage: true,
  enable_graph_storage: true,
};

// ============================================================================
// METRICS TRACKING
// ============================================================================

/**
 * Internal metrics tracking
 */
export class MetricsTracker {
  private total_captures: number = 0;
  private successful_captures: number = 0;
  private failed_captures: number = 0;
  private total_processing_time_ms: number = 0;
  private last_capture_timestamp?: string;
  private problem_type_distribution: Record<string, number> = {};
  private language_distribution: Record<string, number> = {};

  recordCapture(success: boolean, processingTimeMs: number, problemType?: string, language?: string): void {
    this.total_captures++;
    this.total_processing_time_ms += processingTimeMs;
    this.last_capture_timestamp = new Date().toISOString();

    if (success) {
      this.successful_captures++;
    } else {
      this.failed_captures++;
    }

    if (problemType) {
      this.problem_type_distribution[problemType] = (this.problem_type_distribution[problemType] || 0) + 1;
    }

    if (language) {
      this.language_distribution[language] = (this.language_distribution[language] || 0) + 1;
    }
  }

  getMetrics(): CaptureMetrics {
    return {
      total_captures: this.total_captures,
      successful_captures: this.successful_captures,
      failed_captures: this.failed_captures,
      average_processing_time_ms: this.total_captures > 0
        ? this.total_processing_time_ms / this.total_captures
        : 0,
      last_capture_timestamp: this.last_capture_timestamp,
      problem_type_distribution: this.problem_type_distribution,
      language_distribution: this.language_distribution,
    };
  }

  reset(): void {
    this.total_captures = 0;
    this.successful_captures = 0;
    this.failed_captures = 0;
    this.total_processing_time_ms = 0;
    this.last_capture_timestamp = undefined;
    this.problem_type_distribution = {};
    this.language_distribution = {};
  }
}

// ============================================================================
// EVOLVED CODE CAPTURER
// ============================================================================

/**
 * Main capturer class for evolved code
 *
 * Orchestrates the capture and storage of evolved code from OpenEvolve
 * into both Vector DB (for semantic search) and Graphiti (for lineage tracking).
 */
export class EvolvedCodeCapturer {
  private readonly config: Required<Omit<EvolvedCodeCapturerConfig, 'logger'>> & {
    logger?: Logger;
  };

  private readonly logger: Logger;
  private readonly vectorStorage: VectorStorage;
  private readonly graphStorage: GraphStorage;
  private readonly metricsTracker: MetricsTracker;
  private initialized: boolean = false;

  constructor(config: EvolvedCodeCapturerConfig) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.logger = this.config.logger || new Logger('evolved-code-capturer');

    // Initialize storage backends
    this.vectorStorage = new VectorStorage(this.config.vector_storage);
    this.graphStorage = new GraphStorage(this.config.graph_storage);

    // Initialize metrics tracker
    this.metricsTracker = new MetricsTracker();

    this.logger.info('EvolvedCodeCapturer initialized', {
      correlation_id: 'capturer-init',
      vector_storage_enabled: this.config.enable_vector_storage,
      graph_storage_enabled: this.config.enable_graph_storage,
      track_metrics: this.config.track_metrics,
    });
  }

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  /**
   * Initialize capturer and verify connections
   * Following CLAUDE.md: RUNTIME TRUTH - verify before marking as ready
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn('EvolvedCodeCapturer already initialized', {
        correlation_id: 'capturer-init',
      });
      return;
    }

    const correlationId = uuidv4();

    this.logger.info('Initializing EvolvedCodeCapturer', {
      correlation_id: correlationId,
    });

    try {
      // Initialize vector storage if enabled
      if (this.config.enable_vector_storage) {
        await this.vectorStorage.initialize();
        this.logger.info('Vector storage initialized', {
          correlation_id: correlationId,
        });
      }

      // Initialize graph storage if enabled
      if (this.config.enable_graph_storage) {
        await this.graphStorage.initialize();
        this.logger.info('Graph storage initialized', {
          correlation_id: correlationId,
        });
      }

      this.initialized = true;
      this.logger.info('EvolvedCodeCapturer initialized successfully', {
        correlation_id: correlationId,
      });
    } catch (error) {
      this.logger.error('Failed to initialize EvolvedCodeCapturer', error as Error, {
        correlation_id: correlationId,
      });
      throw new Error(
        `EvolvedCodeCapturer initialization failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // ========================================================================
  // CAPTURE OPERATIONS
  // ========================================================================

  /**
   * Capture evolution result
   * Following CLAUDE.md: Law of Idempotency - safe to run multiple times
   *
   * Stores evolved code in both Vector DB (for semantic search) and Graphiti (for lineage)
   */
  async captureEvolution(
    problem: Problem,
    solution: EvolvedCode,
    metrics: EvolutionMetrics,
    correlationId?: string
  ): Promise<CaptureResult> {
    const cid = correlationId || uuidv4();
    const startTime = Date.now();

    this.logger.info('Capturing evolution', {
      correlation_id: cid,
      problem_type: problem.type,
      language: solution.language,
      fitness_score: metrics.fitness_score,
    });

    // Validate inputs
    const problemValidation = validateProblem(problem);
    if (!problemValidation.success) {
      return this.createCaptureResult({
        success: false,
        code_id: solution.id,
        timestamp_utc: new Date().toISOString(),
        processing_time_ms: Date.now() - startTime,
        correlation_id: cid,
        error: `Invalid problem: ${problemValidation.errors.join(', ')}`,
      });
    }

    const solutionValidation = validateEvolvedCode(solution);
    if (!solutionValidation.success) {
      return this.createCaptureResult({
        success: false,
        code_id: solution.id,
        timestamp_utc: new Date().toISOString(),
        processing_time_ms: Date.now() - startTime,
        correlation_id: cid,
        error: `Invalid evolved code: ${solutionValidation.errors.join(', ')}`,
      });
    }

    const metricsValidation = validateEvolutionMetrics(metrics);
    if (!metricsValidation.success) {
      return this.createCaptureResult({
        success: false,
        code_id: solution.id,
        timestamp_utc: new Date().toISOString(),
        processing_time_ms: Date.now() - startTime,
        correlation_id: cid,
        error: `Invalid metrics: ${metricsValidation.errors.join(', ')}`,
      });
    }

    try {
      // Store in vector database
      let vectorStorageId: string | undefined;
      if (this.config.enable_vector_storage) {
        const storeRequest: StoreWithEmbeddingRequest = {
          evolved_code: solution,
          correlation_id: cid,
        };

        await this.vectorStorage.storeWithEmbedding(storeRequest, cid);
        vectorStorageId = solution.id; // Use code_id as vector storage ID

        this.logger.info('Stored in vector database', {
          correlation_id: cid,
          code_id: solution.id,
        });
      }

      // Store in graph database
      let graphEpisodeId: string | undefined;
      if (this.config.enable_graph_storage) {
        const episodeResult = await this.graphStorage.storeAsEpisode(solution, cid);
        graphEpisodeId = episodeResult.episode_id;

        this.logger.info('Stored in graph database', {
          correlation_id: cid,
          code_id: solution.id,
          episode_id: graphEpisodeId,
        });
      }

      const processingTimeMs = Date.now() - startTime;

      // Record metrics
      if (this.config.track_metrics) {
        this.metricsTracker.recordCapture(
          true,
          processingTimeMs,
          problem.type,
          solution.language
        );
      }

      const result: CaptureResult = {
        success: true,
        code_id: solution.id,
        vector_storage_id: vectorStorageId,
        graph_episode_id: graphEpisodeId,
        timestamp_utc: new Date().toISOString(),
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
      };

      this.logger.info('Evolution captured successfully', {
        correlation_id: cid,
        code_id: solution.id,
        processing_time_ms: processingTimeMs,
      });

      return result;
    } catch (error) {
      const processingTimeMs = Date.now() - startTime;

      // Record metrics
      if (this.config.track_metrics) {
        this.metricsTracker.recordCapture(
          false,
          processingTimeMs,
          problem.type,
          solution.language
        );
      }

      const result = this.createCaptureResult({
        success: false,
        code_id: solution.id,
        timestamp_utc: new Date().toISOString(),
        processing_time_ms: processingTimeMs,
        correlation_id: cid,
        error: error instanceof Error ? error.message : String(error),
      });

      this.logger.error('Failed to capture evolution', error as Error, {
        correlation_id: cid,
        code_id: solution.id,
      });

      return result;
    }
  }

  // ========================================================================
  // SEARCH OPERATIONS
  // ========================================================================

  /**
   * Search for similar problems
   * Returns previously solved problems that are semantically similar
   */
  async searchSimilarProblems(
    problem: Problem,
    maxResults: number = 10,
    correlationId?: string
  ): Promise<SimilarSolution[]> {
    const cid = correlationId || uuidv4();

    this.logger.info('Searching for similar problems', {
      correlation_id: cid,
      problem_type: problem.type,
      max_results: maxResults,
    });

    if (!this.config.enable_vector_storage) {
      this.logger.warn('Vector storage is disabled, cannot search similar problems', {
        correlation_id: cid,
      });
      return [];
    }

    // Validate problem
    const validation = validateProblem(problem);
    if (!validation.success) {
      throw new Error(`Invalid problem: ${validation.errors.join(', ')}`);
    }

    try {
      const searchRequest: SearchSimilarRequest = {
        problem,
        max_results: maxResults,
        similarity_threshold: 0.5,
        correlation_id: cid,
      };

      const results = await this.vectorStorage.searchSimilar(searchRequest, cid);

      this.logger.info('Similar problems search completed', {
        correlation_id: cid,
        results_count: results.length,
      });

      return results;
    } catch (error) {
      this.logger.error('Failed to search similar problems', error as Error, {
        correlation_id: cid,
      });
      throw error;
    }
  }

  // ========================================================================
  // LINEAGE OPERATIONS
  // ========================================================================

  /**
   * Get evolution lineage for a code solution
   * Returns the full evolution tree from initial to final solution
   */
  async getEvolutionLineage(
    codeId: string,
    correlationId?: string
  ): Promise<EvolutionLineage> {
    const cid = correlationId || uuidv4();

    this.logger.info('Getting evolution lineage', {
      correlation_id: cid,
      code_id: codeId,
    });

    if (!this.config.enable_graph_storage) {
      this.logger.warn('Graph storage is disabled, cannot get lineage', {
        correlation_id: cid,
      });
      throw new Error('Graph storage is disabled');
    }

    try {
      const lineage = await this.graphStorage.trackEvolutionLineage(codeId, cid);

      this.logger.info('Evolution lineage retrieved successfully', {
        correlation_id: cid,
        code_id: codeId,
        total_nodes: lineage.total_nodes,
        depth: lineage.depth,
      });

      return lineage;
    } catch (error) {
      this.logger.error('Failed to get evolution lineage', error as Error, {
        correlation_id: cid,
        code_id: codeId,
      });
      throw error;
    }
  }

  // ========================================================================
  // METRICS OPERATIONS
  // ========================================================================

  /**
   * Get capture metrics
   * Returns aggregated statistics about captured code
   */
  async getMetrics(correlationId?: string): Promise<CaptureMetrics> {
    const cid = correlationId || uuidv4();

    this.logger.info('Getting capture metrics', {
      correlation_id: cid,
    });

    const metrics = this.metricsTracker.getMetrics();

    this.logger.info('Capture metrics retrieved', {
      correlation_id: cid,
      total_captures: metrics.total_captures,
      successful_captures: metrics.successful_captures,
      failed_captures: metrics.failed_captures,
    });

    return metrics;
  }

  /**
   * Reset metrics
   */
  async resetMetrics(correlationId?: string): Promise<void> {
    const cid = correlationId || uuidv4();

    this.logger.info('Resetting capture metrics', {
      correlation_id: cid,
    });

    this.metricsTracker.reset();

    this.logger.info('Capture metrics reset', {
      correlation_id: cid,
    });
  }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================

  /**
   * Check capturer health
   */
  async healthCheck(): Promise<{
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
  }> {
    const vectorHealth = this.config.enable_vector_storage
      ? await this.vectorStorage.healthCheck()
      : { enabled: false, healthy: false };

    const graphHealth = this.config.enable_graph_storage
      ? await this.graphStorage.healthCheck()
      : { enabled: false, healthy: false };

    const healthy = this.initialized
      && (!this.config.enable_vector_storage || vectorHealth.healthy)
      && (!this.config.enable_graph_storage || graphHealth.healthy);

    return {
      healthy,
      initialized: this.initialized,
      vector_storage: {
        enabled: this.config.enable_vector_storage,
        healthy: vectorHealth.healthy,
      },
      graph_storage: {
        enabled: this.config.enable_graph_storage,
        healthy: graphHealth.healthy,
      },
    };
  }

  // ========================================================================
  // CLEANUP
  // ========================================================================

  /**
   * Close capturer and cleanup resources
   */
  async close(): Promise<void> {
    this.logger.info('Closing EvolvedCodeCapturer', {
      correlation_id: 'capturer-close',
    });

    try {
      await this.vectorStorage.close();
      await this.graphStorage.close();
      this.initialized = false;

      this.logger.info('EvolvedCodeCapturer closed successfully', {
        correlation_id: 'capturer-close',
      });
    } catch (error) {
      this.logger.error('Error closing EvolvedCodeCapturer', error as Error, {
        correlation_id: 'capturer-close',
      });
    }
  }

  // ========================================================================
  // PRIVATE METHODS
  // ========================================================================

  /**
   * Create capture result
   */
  private createCaptureResult(partial: Partial<CaptureResult>): CaptureResult {
    return {
      success: partial.success ?? false,
      code_id: partial.code_id || uuidv4(),
      vector_storage_id: partial.vector_storage_id,
      graph_episode_id: partial.graph_episode_id,
      timestamp_utc: partial.timestamp_utc || new Date().toISOString(),
      processing_time_ms: partial.processing_time_ms || 0,
      correlation_id: partial.correlation_id,
      error: partial.error,
      warnings: partial.warnings,
    };
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create capturer from environment variables
 * Following CLAUDE.md: Law of Configuration Explicitness
 */
export function createCapturerFromEnv(logger?: Logger): EvolvedCodeCapturer {
  // Validate environment variables
  const config = validateEnvWithTypes([
    { name: 'VECTORDB_ADAPTER_URL', type: 'url', required: true },
    { name: 'GRAPHITI_ADAPTER_URL', type: 'url', required: true },
    { name: 'EVOLVED_CODE_COLLECTION', type: 'string', required: true },
    { name: 'EMBEDDING_DIMENSION', type: 'number', required: false, default: 1536 },
    { name: 'OPENAI_API_KEY', type: 'string', required: false },
    { name: 'ENABLE_VECTOR_STORAGE', type: 'boolean', required: false, default: true },
    { name: 'ENABLE_GRAPH_STORAGE', type: 'boolean', required: false, default: true },
    { name: 'TRACK_METRICS', type: 'boolean', required: false, default: true },
  ]);

  return new EvolvedCodeCapturer({
    vector_storage: {
      vectordb_adapter_url: config.VECTORDB_ADAPTER_URL as string,
      collection_name: config.EVOLVED_CODE_COLLECTION as string,
      embedding_dimension: config.EMBEDDING_DIMENSION as number,
      embedding_api_key: config.OPENAI_API_KEY as string | undefined,
      logger,
    },
    graph_storage: {
      graphiti_adapter_url: config.GRAPHITI_ADAPTER_URL as string,
      episode_type_base: 'evolved_code',
      logger,
    },
    enable_vector_storage: config.ENABLE_VECTOR_STORAGE as boolean,
    enable_graph_storage: config.ENABLE_GRAPH_STORAGE as boolean,
    track_metrics: config.TRACK_METRICS as boolean,
    metrics_retention_days: 30,
  });
}

// ============================================================================
// EXPORTS
// ============================================================================


