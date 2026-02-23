"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.EvolvedCodeCapturer = void 0;
exports.createCapturerFromEnv = createCapturerFromEnv;
const uuid_1 = require("uuid");
const logger_1 = require("../logger");
const env_validator_1 = require("../env-validator");
const vector_storage_1 = require("./vector-storage");
const graph_storage_1 = require("./graph-storage");
const canonical_1 = require("./canonical");
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
class MetricsTracker {
    constructor() {
        this.total_captures = 0;
        this.successful_captures = 0;
        this.failed_captures = 0;
        this.total_processing_time_ms = 0;
        this.problem_type_distribution = {};
        this.language_distribution = {};
    }
    recordCapture(success, processingTimeMs, problemType, language) {
        this.total_captures++;
        this.total_processing_time_ms += processingTimeMs;
        this.last_capture_timestamp = new Date().toISOString();
        if (success) {
            this.successful_captures++;
        }
        else {
            this.failed_captures++;
        }
        if (problemType) {
            this.problem_type_distribution[problemType] = (this.problem_type_distribution[problemType] || 0) + 1;
        }
        if (language) {
            this.language_distribution[language] = (this.language_distribution[language] || 0) + 1;
        }
    }
    getMetrics() {
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
    reset() {
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
class EvolvedCodeCapturer {
    constructor(config) {
        this.initialized = false;
        this.config = {
            ...DEFAULT_CONFIG,
            ...config,
        };
        this.logger = this.config.logger || new logger_1.Logger('evolved-code-capturer');
        // Initialize storage backends
        this.vectorStorage = new vector_storage_1.VectorStorage(this.config.vector_storage);
        this.graphStorage = new graph_storage_1.GraphStorage(this.config.graph_storage);
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
    async initialize() {
        if (this.initialized) {
            this.logger.warn('EvolvedCodeCapturer already initialized', {
                correlation_id: 'capturer-init',
            });
            return;
        }
        const correlationId = (0, uuid_1.v4)();
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
        }
        catch (error) {
            this.logger.error('Failed to initialize EvolvedCodeCapturer', error, {
                correlation_id: correlationId,
            });
            throw new Error(`EvolvedCodeCapturer initialization failed: ${error instanceof Error ? error.message : String(error)}`);
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
    async captureEvolution(problem, solution, metrics, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const startTime = Date.now();
        this.logger.info('Capturing evolution', {
            correlation_id: cid,
            problem_type: problem.type,
            language: solution.language,
            fitness_score: metrics.fitness_score,
        });
        // Validate inputs
        const problemValidation = (0, canonical_1.validateProblem)(problem);
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
        const solutionValidation = (0, canonical_1.validateEvolvedCode)(solution);
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
        const metricsValidation = (0, canonical_1.validateEvolutionMetrics)(metrics);
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
            let vectorStorageId;
            if (this.config.enable_vector_storage) {
                const storeRequest = {
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
            let graphEpisodeId;
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
                this.metricsTracker.recordCapture(true, processingTimeMs, problem.type, solution.language);
            }
            const result = {
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
        }
        catch (error) {
            const processingTimeMs = Date.now() - startTime;
            // Record metrics
            if (this.config.track_metrics) {
                this.metricsTracker.recordCapture(false, processingTimeMs, problem.type, solution.language);
            }
            const result = this.createCaptureResult({
                success: false,
                code_id: solution.id,
                timestamp_utc: new Date().toISOString(),
                processing_time_ms: processingTimeMs,
                correlation_id: cid,
                error: error instanceof Error ? error.message : String(error),
            });
            this.logger.error('Failed to capture evolution', error, {
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
    async searchSimilarProblems(problem, maxResults = 10, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
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
        const validation = (0, canonical_1.validateProblem)(problem);
        if (!validation.success) {
            throw new Error(`Invalid problem: ${validation.errors.join(', ')}`);
        }
        try {
            const searchRequest = {
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
        }
        catch (error) {
            this.logger.error('Failed to search similar problems', error, {
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
    async getEvolutionLineage(codeId, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
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
        }
        catch (error) {
            this.logger.error('Failed to get evolution lineage', error, {
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
    async getMetrics(correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
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
    async resetMetrics(correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
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
    async healthCheck() {
        const vectorHealth = this.config.enable_vector_storage
            ? await this.vectorStorage.healthCheck()
            : { enabled: false, healthy: false };
        const graphHealth = this.config.enable_graph_storage
            ? await this.graphStorage.healthCheck()
            : { enabled: false, healthy: false };
        const healthy = this.initialized &&
            (!this.config.enable_vector_storage || vectorHealth.healthy) &&
            (!this.config.enable_graph_storage || graphHealth.healthy);
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
    async close() {
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
        }
        catch (error) {
            this.logger.error('Error closing EvolvedCodeCapturer', error, {
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
    createCaptureResult(partial) {
        return {
            success: partial.success ?? false,
            code_id: partial.code_id || (0, uuid_1.v4)(),
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
exports.EvolvedCodeCapturer = EvolvedCodeCapturer;
// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================
/**
 * Create capturer from environment variables
 * Following CLAUDE.md: Law of Configuration Explicitness
 */
function createCapturerFromEnv(logger) {
    // Validate environment variables
    const config = (0, env_validator_1.validateEnvWithTypes)([
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
            vectordb_adapter_url: config.VECTORDB_ADAPTER_URL,
            collection_name: config.EVOLVED_CODE_COLLECTION,
            embedding_dimension: config.EMBEDDING_DIMENSION,
            embedding_api_key: config.OPENAI_API_KEY,
            logger,
        },
        graph_storage: {
            graphiti_adapter_url: config.GRAPHITI_ADAPTER_URL,
            episode_type_base: 'evolved_code',
            logger,
        },
        enable_vector_storage: config.ENABLE_VECTOR_STORAGE,
        enable_graph_storage: config.ENABLE_GRAPH_STORAGE,
        track_metrics: config.TRACK_METRICS,
    });
}
//# sourceMappingURL=capturer.js.map