/**
 * Unified Verification Orchestrator - Public API
 *
 * Exports all components for formal verification orchestration:
 * - Main orchestrator for verification requests
 * - Strategy selector for intelligent system selection
 * - Cross validator for multi-system verification
 * - Confidence aggregator for score combination
 * - Canonical schemas for data validation
 */

// Main orchestrator
export {
  UnifiedVerificationOrchestrator
} from './orchestrator';

// Components
export {
  VerificationStrategySelector
} from './strategy-selector';

export {
  CrossValidator
} from './cross-validator';

export {
  ConfidenceAggregator
} from './confidence-aggregator';

// Canonical schemas and types
export {
  CanonicalSchemas,
  // Types
  type Problem,
  type Constraints,
  type VerificationRequest,
  type VerificationResult,
  type CrossValidationResult,
  type ConfidenceScore,
  type VerificationOptions,
  type StrategyEffectiveness,
  type ComparisonReport,
  type Disagreement,
  type VerificationStrategy,
  // Schema objects
  ProblemSchema,
  ConstraintsSchema,
  VerificationRequestSchema,
  VerificationResultSchema,
  CrossValidationResultSchema,
  ConfidenceScoreSchema,
  VerificationOptionsSchema,
  StrategyEffectivenessSchema,
  ComparisonReportSchema,
  DisagreementSchema
} from './canonical';

// Utility types
export type {
  SystemConfig,
  StrategySelection
} from './strategy-selector';

export type {
  SystemEndpoint
} from './cross-validator';

export type {
  NormalizedScore,
  WeightMap,
  CombinedScore,
  Evidence
} from './confidence-aggregator';
