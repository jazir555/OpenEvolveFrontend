"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.VERSION = exports.OpenEvolveIntegration = exports.OpenEvolveClient = exports.createCapturerFromEnv = exports.MetricsTracker = exports.EvolvedCodeCapturer = exports.GraphStorage = exports.OpenAIEmbeddingGenerator = exports.SimpleEmbeddingGenerator = exports.VectorStorage = exports.validateCaptureResult = exports.validateEvolutionMetrics = exports.validateProblem = exports.validateEvolvedCode = exports.GetLineageRequestSchema = exports.SearchSimilarRequestSchema = exports.StoreWithEmbeddingRequestSchema = exports.CaptureMetricsSchema = exports.CaptureResultSchema = exports.EvolutionLineageSchema = exports.SimilarSolutionSchema = exports.EvolvedCodeSchema = exports.EvolutionMetricsSchema = exports.ProblemSchema = void 0;
exports.createCapturer = createCapturer;
// ============================================================================
// CANONICAL SCHEMAS
// ============================================================================
var canonical_1 = require("./canonical");
// Schemas
Object.defineProperty(exports, "ProblemSchema", { enumerable: true, get: function () { return canonical_1.ProblemSchema; } });
Object.defineProperty(exports, "EvolutionMetricsSchema", { enumerable: true, get: function () { return canonical_1.EvolutionMetricsSchema; } });
Object.defineProperty(exports, "EvolvedCodeSchema", { enumerable: true, get: function () { return canonical_1.EvolvedCodeSchema; } });
Object.defineProperty(exports, "SimilarSolutionSchema", { enumerable: true, get: function () { return canonical_1.SimilarSolutionSchema; } });
Object.defineProperty(exports, "EvolutionLineageSchema", { enumerable: true, get: function () { return canonical_1.EvolutionLineageSchema; } });
Object.defineProperty(exports, "CaptureResultSchema", { enumerable: true, get: function () { return canonical_1.CaptureResultSchema; } });
Object.defineProperty(exports, "CaptureMetricsSchema", { enumerable: true, get: function () { return canonical_1.CaptureMetricsSchema; } });
Object.defineProperty(exports, "StoreWithEmbeddingRequestSchema", { enumerable: true, get: function () { return canonical_1.StoreWithEmbeddingRequestSchema; } });
Object.defineProperty(exports, "SearchSimilarRequestSchema", { enumerable: true, get: function () { return canonical_1.SearchSimilarRequestSchema; } });
Object.defineProperty(exports, "GetLineageRequestSchema", { enumerable: true, get: function () { return canonical_1.GetLineageRequestSchema; } });
// Validation functions
Object.defineProperty(exports, "validateEvolvedCode", { enumerable: true, get: function () { return canonical_1.validateEvolvedCode; } });
Object.defineProperty(exports, "validateProblem", { enumerable: true, get: function () { return canonical_1.validateProblem; } });
Object.defineProperty(exports, "validateEvolutionMetrics", { enumerable: true, get: function () { return canonical_1.validateEvolutionMetrics; } });
Object.defineProperty(exports, "validateCaptureResult", { enumerable: true, get: function () { return canonical_1.validateCaptureResult; } });
// ============================================================================
// VECTOR STORAGE
// ============================================================================
var vector_storage_1 = require("./vector-storage");
Object.defineProperty(exports, "VectorStorage", { enumerable: true, get: function () { return vector_storage_1.VectorStorage; } });
Object.defineProperty(exports, "SimpleEmbeddingGenerator", { enumerable: true, get: function () { return vector_storage_1.SimpleEmbeddingGenerator; } });
Object.defineProperty(exports, "OpenAIEmbeddingGenerator", { enumerable: true, get: function () { return vector_storage_1.OpenAIEmbeddingGenerator; } });
// ============================================================================
// GRAPH STORAGE
// ============================================================================
var graph_storage_1 = require("./graph-storage");
Object.defineProperty(exports, "GraphStorage", { enumerable: true, get: function () { return graph_storage_1.GraphStorage; } });
// ============================================================================
// CAPTURER
// ============================================================================
var capturer_1 = require("./capturer");
Object.defineProperty(exports, "EvolvedCodeCapturer", { enumerable: true, get: function () { return capturer_1.EvolvedCodeCapturer; } });
Object.defineProperty(exports, "MetricsTracker", { enumerable: true, get: function () { return capturer_1.MetricsTracker; } });
Object.defineProperty(exports, "createCapturerFromEnv", { enumerable: true, get: function () { return capturer_1.createCapturerFromEnv; } });
// ============================================================================
// OPENEVOLVE INTEGRATION
// ============================================================================
var openevolve_integration_1 = require("./openevolve-integration");
Object.defineProperty(exports, "OpenEvolveClient", { enumerable: true, get: function () { return openevolve_integration_1.OpenEvolveClient; } });
Object.defineProperty(exports, "OpenEvolveIntegration", { enumerable: true, get: function () { return openevolve_integration_1.OpenEvolveIntegration; } });
// ============================================================================
// CONVENIENCE EXPORTS
// ============================================================================
/**
 * Version of the evolved code capture system
 */
exports.VERSION = '1.0.0';
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
function createCapturer(config) {
    const { EvolvedCodeCapturer } = require('./capturer');
    return new EvolvedCodeCapturer(config);
}
//# sourceMappingURL=index.js.map