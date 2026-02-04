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

// ============================================================================
// CANONICAL SCHEMAS
// ============================================================================

export {
  // Schemas
  ProblemSchema,
  EvolutionMetricsSchema,
  EvolvedCodeSchema,
  SimilarSolutionSchema,
  EvolutionLineageSchema,
  CaptureResultSchema,
  CaptureMetricsSchema,
  StoreWithEmbeddingRequestSchema,
  SearchSimilarRequestSchema,
  GetLineageRequestSchema,

  // Types
  type Problem,
  type ProblemType,
  type Constraints,
  type EvolutionMetrics,
  type EvolvedCode,
  type Language,
  type SimilarSolution,
  type EvolutionNode,
  type EvolutionLineage,
  type CaptureResult,
  type CaptureMetrics,
  type StoreWithEmbeddingRequest,
  type SearchSimilarRequest,
  type GetLineageRequest,

  // Validation functions
  validateEvolvedCode,
  validateProblem,
  validateEvolutionMetrics,
  validateCaptureResult,
} from './canonical';

// ============================================================================
// VECTOR STORAGE
// ============================================================================

export {
  VectorStorage,
  SimpleEmbeddingGenerator,
  OpenAIEmbeddingGenerator,
  type VectorStorageConfig,
  type EmbeddingGenerator,
} from './vector-storage';

// ============================================================================
// GRAPH STORAGE
// ============================================================================

export {
  GraphStorage,
  type GraphStorageConfig,
} from './graph-storage';

// ============================================================================
// CAPTURER
// ============================================================================

export {
  EvolvedCodeCapturer,
  MetricsTracker,
  createCapturerFromEnv,
  type EvolvedCodeCapturerConfig,
} from './capturer';

// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================

/**
 * Version of the evolved code capture system
 */
export const VERSION = '1.0.0';

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
export function createCapturer(
  config: import('./capturer').EvolvedCodeCapturerConfig
): import('./capturer').EvolvedCodeCapturer {
  const { EvolvedCodeCapturer } = require('./capturer');
  return new EvolvedCodeCapturer(config);
}
