/**
 * Evolved Code Capture - Canonical Schema Definitions
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of the Air Gap: These are canonical schemas, no core project imports
 * - Law of UTC: All timestamps in UTC ISO-8601 format
 * - Law of Configuration Explicitness: No magic defaults
 * - Anti-Corruption Layer: Normalize all evolved code to this schema
 *
 * This schema defines the canonical data models for capturing evolved code
 * from OpenEvolve and storing it in knowledge systems (Vector DB + Graphiti).
 */

import { z } from 'zod';

// ============================================================================
// PROBLEM SCHEMA
// ============================================================================

/**
 * Problem Type Classification
 */
export const ProblemTypeEnum = z.enum([
  'algorithm_optimization',
  'code_refactoring',
  'performance_tuning',
  'bug_fix',
  'feature_implementation',
  'code_migration',
  'parallelization',
  'memory_optimization',
  'numerical_computation',
  'data_structure_design',
  'other',
]);

export type ProblemType = z.infer<typeof ProblemTypeEnum>;

/**
 * Constraints definition
 */
export const ConstraintsSchema = z.object({
  max_memory_mb: z.number().optional(),
  max_runtime_ms: z.number().optional(),
  required_libraries: z.array(z.string()).optional(),
  forbidden_patterns: z.array(z.string()).optional(),
  language_version: z.string().optional(),
  target_platform: z.string().optional(),
  custom_constraints: z.record(z.string(), z.any()).optional(),
});

export type Constraints = z.infer<typeof ConstraintsSchema>;

/**
 * Problem definition
 */
export const ProblemSchema = z.object({
  description: z.string().min(1, 'Problem description is required'),
  type: ProblemTypeEnum,
  constraints: ConstraintsSchema.optional(),
  input_spec: z.string().optional(), // Input format specification
  output_spec: z.string().optional(), // Output format specification
  test_cases: z.array(z.object({
    input: z.any(),
    expected_output: z.any(),
    description: z.string().optional(),
  })).optional(),
  difficulty: z.enum(['easy', 'medium', 'hard', 'expert']).optional(),
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export type Problem = z.infer<typeof ProblemSchema>;

// ============================================================================
// EVOLUTION METRICS SCHEMA
// ============================================================================

/**
 * Metrics from the evolution process
 */
export const EvolutionMetricsSchema = z.object({
  iterations: z.number().int().nonnegative(),
  fitness_score: z.number(), // Can be any float value
  fitness_improvement: z.number(), // Improvement from initial to final
  duration_ms: z.number().int().positive(),
  generations: z.number().int().nonnegative().optional(),
  population_size: z.number().int().positive().optional(),
  mutation_rate: z.number().min(0).max(1).optional(),
  crossover_rate: z.number().min(0).max(1).optional(),
  convergence_generation: z.number().int().nonnegative().optional(), // Generation where solution converged
  total_evaluations: z.number().int().nonnegative().optional(),
  success_rate: z.number().min(0).max(1).optional(), // Percentage of successful test cases
  benchmark_score: z.number().optional(), // Comparison to baseline
  additional_metrics: z.record(z.string(), z.any()).optional(),
});

export type EvolutionMetrics = z.infer<typeof EvolutionMetricsSchema>;

// ============================================================================
// EVOLVED CODE SCHEMA
// ============================================================================

/**
 * Programming language
 */
export const LanguageEnum = z.enum([
  'python',
  'javascript',
  'typescript',
  'java',
  'c',
  'cpp',
  'csharp',
  'go',
  'rust',
  'julia',
  'matlab',
  'r',
  'other',
]);

export type Language = z.infer<typeof LanguageEnum>;

/**
 * The evolved solution code
 */
export const EvolvedCodeSchema = z.object({
  id: z.string().uuid('Invalid UUID format'),
  problem: ProblemSchema,

  // Code content
  language: LanguageEnum,
  code: z.string().min(1, 'Code content is required'),
  function_name: z.string().optional(),
  class_name: z.string().optional(),

  // Evolution metadata
  metrics: EvolutionMetricsSchema,
  timestamp_utc: z.string().datetime('Must be UTC ISO-8601 timestamp'),

  // Solution metadata
  is_valid: z.boolean(),
  validation_errors: z.array(z.string()).optional(),
  execution_time_ms: z.number().int().nonnegative().optional(),
  memory_used_mb: z.number().int().nonnegative().optional(),

  // Parent tracking for lineage
  parent_code_id: z.string().uuid().optional(),
  generation_number: z.number().int().nonnegative().optional(),

  // Additional metadata
  tags: z.array(z.string()).optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export type EvolvedCode = z.infer<typeof EvolvedCodeSchema>;

// ============================================================================
// SIMILAR SOLUTION SCHEMA
// ============================================================================

/**
 * A similar solution found in the knowledge base
 */
export const SimilarSolutionSchema = z.object({
  evolved_code: EvolvedCodeSchema,
  similarity_score: z.number().min(0).max(1), // 0 to 1, where 1 is exact match
  similarity_method: z.enum(['semantic', 'structural', 'hybrid']),
  problem_similarity: z.number().min(0).max(1).optional(),
  code_similarity: z.number().min(0).max(1).optional(),
  matched_features: z.array(z.string()).optional(),
  distance: z.number().optional(), // For vector-based similarity
});

export type SimilarSolution = z.infer<typeof SimilarSolutionSchema>;

// ============================================================================
// EVOLUTION LINEAGE SCHEMA
// ============================================================================

/**
 * A node in the evolution tree
 */
export const EvolutionNodeSchema = z.object({
  code_id: z.string().uuid(),
  code: z.string().optional(),
  fitness_score: z.number(),
  timestamp_utc: z.string().datetime(),
  generation: z.number().int().nonnegative(),
  parent_id: z.string().uuid().optional(),
  children_ids: z.array(z.string().uuid()),
});

export type EvolutionNode = z.infer<typeof EvolutionNodeSchema>;

/**
 * Full lineage tracking from initial solution to final evolved solution
 */
export const EvolutionLineageSchema = z.object({
  root_code_id: z.string().uuid(),
  final_code_id: z.string().uuid(),
  total_nodes: z.number().int().positive(),
  depth: z.number().int().positive(),
  nodes: z.array(EvolutionNodeSchema),
  branches: z.number().int().nonnegative().optional(),
  metadata: z.record(z.string(), z.any()).optional(),
});

export type EvolutionLineage = z.infer<typeof EvolutionLineageSchema>;

// ============================================================================
// CAPTURE RESULT SCHEMA
// ============================================================================

/**
 * Result of capturing an evolved solution
 */
export const CaptureResultSchema = z.object({
  success: z.boolean(),
  code_id: z.string().uuid(),
  vector_storage_id: z.string().optional(),
  graph_episode_id: z.string().optional(),
  timestamp_utc: z.string().datetime(),
  processing_time_ms: z.number().int().nonnegative(),
  correlation_id: z.string().uuid().optional(),
  error: z.string().optional(),
  warnings: z.array(z.string()).optional(),
});

export type CaptureResult = z.infer<typeof CaptureResultSchema>;

// ============================================================================
// CAPTURE METRICS SCHEMA
// ============================================================================

/**
 * Aggregated metrics for the capture system
 */
export const CaptureMetricsSchema = z.object({
  total_captures: z.number().int().nonnegative(),
  successful_captures: z.number().int().nonnegative(),
  failed_captures: z.number().int().nonnegative(),
  average_processing_time_ms: z.number().nonnegative(),
  total_storage_size_bytes: z.number().int().nonnegative().optional(),
  last_capture_timestamp: z.string().datetime().optional(),
  problem_type_distribution: z.record(z.string(), z.number().int().nonnegative()).optional(),
  language_distribution: z.record(z.string(), z.number().int().nonnegative()).optional(),
});

export type CaptureMetrics = z.infer<typeof CaptureMetricsSchema>;

// ============================================================================
// STORAGE REQUEST SCHEMAS
// ============================================================================

/**
 * Request to store code with embedding
 */
export const StoreWithEmbeddingRequestSchema = z.object({
  evolved_code: EvolvedCodeSchema,
  embedding: z.array(z.number()).optional(), // If not provided, will generate
  correlation_id: z.string().uuid().optional(),
});

export type StoreWithEmbeddingRequest = z.infer<typeof StoreWithEmbeddingRequestSchema>;

/**
 * Request to search for similar solutions
 */
export const SearchSimilarRequestSchema = z.object({
  problem: ProblemSchema,
  max_results: z.number().int().positive().default(10),
  similarity_threshold: z.number().min(0).max(1).default(0.5),
  filters: z.object({
    language: LanguageEnum.optional(),
    problem_type: ProblemTypeEnum.optional(),
    min_fitness: z.number().optional(),
    date_range: z.object({
      start: z.string().datetime(),
      end: z.string().datetime(),
    }).optional(),
  }).optional(),
  correlation_id: z.string().uuid().optional(),
});

export type SearchSimilarRequest = z.infer<typeof SearchSimilarRequestSchema>;

/**
 * Request to get evolution lineage
 */
export const GetLineageRequestSchema = z.object({
  code_id: z.string().uuid(),
  include_code_content: z.boolean().default(false),
  max_depth: z.number().int().positive().optional(),
  correlation_id: z.string().uuid().optional(),
});

export type GetLineageRequest = z.infer<typeof GetLineageRequestSchema>;

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validate evolved code against canonical schema
 */
export function validateEvolvedCode(data: unknown): {
  success: boolean;
  data?: EvolvedCode;
  errors: string[];
} {
  const result = EvolvedCodeSchema.safeParse(data);

  if (result.success) {
    return {
      success: true,
      data: result.data,
      errors: [],
    };
  }

  return {
    success: false,
    errors: result.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`),
  };
}

/**
 * Validate problem against canonical schema
 */
export function validateProblem(data: unknown): {
  success: boolean;
  data?: Problem;
  errors: string[];
} {
  const result = ProblemSchema.safeParse(data);

  if (result.success) {
    return {
      success: true,
      data: result.data,
      errors: [],
    };
  }

  return {
    success: false,
    errors: result.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`),
  };
}

/**
 * Validate evolution metrics against canonical schema
 */
export function validateEvolutionMetrics(data: unknown): {
  success: boolean;
  data?: EvolutionMetrics;
  errors: string[];
} {
  const result = EvolutionMetricsSchema.safeParse(data);

  if (result.success) {
    return {
      success: true,
      data: result.data,
      errors: [],
    };
  }

  return {
    success: false,
    errors: result.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`),
  };
}

/**
 * Validate capture result against canonical schema
 */
export function validateCaptureResult(data: unknown): {
  success: boolean;
  data?: CaptureResult;
  errors: string[];
} {
  const result = CaptureResultSchema.safeParse(data);

  if (result.success) {
    return {
      success: true,
      data: result.data,
      errors: [],
    };
  }

  return {
    success: false,
    errors: result.error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`),
  };
}
