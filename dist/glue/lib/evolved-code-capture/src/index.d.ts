/**
 * Evolved Code Capture - Main Entry Point
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: No imports from core-projects
 * - Anti-Corruption Layer: Canonical schemas for all data
 * - Law of Idempotency: All operations safe to run multiple times
 * - Law of Configuration Explicitness: All config via environment variables
 *
 * Exports all public APIs for the evolved code capture system.
 */
export { ProblemSchema, EvolutionMetricsSchema, EvolvedCodeSchema, SimilarSolutionSchema, EvolutionLineageSchema, CaptureResultSchema, CaptureMetricsSchema, StoreWithEmbeddingRequestSchema, SearchSimilarRequestSchema, GetLineageRequestSchema, type Problem, type ProblemType, type Constraints, type EvolutionMetrics, type EvolvedCode, type Language, type SimilarSolution, type EvolutionNode, type EvolutionLineage, type CaptureResult, type CaptureMetrics, type StoreWithEmbeddingRequest, type SearchSimilarRequest, type GetLineageRequest, validateEvolvedCode, validateProblem, validateEvolutionMetrics, validateCaptureResult, } from './canonical';
export { VectorStorage, SimpleEmbeddingGenerator, OpenAIEmbeddingGenerator, type VectorStorageConfig, type EmbeddingGenerator, } from './vector-storage';
export { GraphStorage, type GraphStorageConfig, } from './graph-storage';
export { EvolvedCodeCapturer, MetricsTracker, createCapturerFromEnv, type EvolvedCodeCapturerConfig, } from './capturer';
export { OpenEvolveClient, OpenEvolveIntegration, type OpenEvolveIntegrationConfig, } from './openevolve-integration';
/**
 * Version of the evolved code capture system
 */
export declare const VERSION = "1.0.0";
/**
 * Create a fully configured capturer with default settings
 *
 * This is the recommended way to create a capturer instance.
 *
 * @example
 * ```typescript
 * import { createCapturer } from '@openevolve/evolved-code-capture';
 *
 * const capturer = createCapturer({
 *   vector_storage: {
 *     vectordb_adapter_url: 'http://vectordb-adapter:8000',
 *     collection_name: 'evolved_code',
 *   },
 *   graph_storage: {
 *     graphiti_adapter_url: 'http://graphiti-adapter:8000',
 *   },
 * });
 *
 * await capturer.initialize();
 * ```
 */
export declare function createCapturer(config: import('./capturer').EvolvedCodeCapturerConfig): import('./capturer').EvolvedCodeCapturer;
//# sourceMappingURL=index.d.ts.map