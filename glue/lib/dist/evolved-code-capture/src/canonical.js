"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.GetLineageRequestSchema = exports.SearchSimilarRequestSchema = exports.StoreWithEmbeddingRequestSchema = exports.CaptureMetricsSchema = exports.CaptureResultSchema = exports.EvolutionLineageSchema = exports.EvolutionNodeSchema = exports.SimilarSolutionSchema = exports.EvolvedCodeSchema = exports.LanguageEnum = exports.EvolutionMetricsSchema = exports.ProblemSchema = exports.ConstraintsSchema = exports.ProblemTypeEnum = void 0;
exports.validateEvolvedCode = validateEvolvedCode;
exports.validateProblem = validateProblem;
exports.validateEvolutionMetrics = validateEvolutionMetrics;
exports.validateCaptureResult = validateCaptureResult;
const zod_1 = require("zod");
// ============================================================================
// PROBLEM SCHEMA
// ============================================================================
/**
 * Problem Type Classification
 */
exports.ProblemTypeEnum = zod_1.z.enum([
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
/**
 * Constraints definition
 */
exports.ConstraintsSchema = zod_1.z.object({
    max_memory_mb: zod_1.z.number().optional(),
    max_runtime_ms: zod_1.z.number().optional(),
    required_libraries: zod_1.z.array(zod_1.z.string()).optional(),
    forbidden_patterns: zod_1.z.array(zod_1.z.string()).optional(),
    language_version: zod_1.z.string().optional(),
    target_platform: zod_1.z.string().optional(),
    custom_constraints: zod_1.z.record(zod_1.z.string(), zod_1.z.any()).optional(),
});
/**
 * Problem definition
 */
exports.ProblemSchema = zod_1.z.object({
    description: zod_1.z.string().min(1, 'Problem description is required'),
    type: exports.ProblemTypeEnum,
    constraints: exports.ConstraintsSchema.optional(),
    input_spec: zod_1.z.string().optional(), // Input format specification
    output_spec: zod_1.z.string().optional(), // Output format specification
    test_cases: zod_1.z.array(zod_1.z.object({
        input: zod_1.z.any(),
        expected_output: zod_1.z.any(),
        description: zod_1.z.string().optional(),
    })).optional(),
    difficulty: zod_1.z.enum(['easy', 'medium', 'hard', 'expert']).optional(),
    tags: zod_1.z.array(zod_1.z.string()).optional(),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.any()).optional(),
});
// ============================================================================
// EVOLUTION METRICS SCHEMA
// ============================================================================
/**
 * Metrics from the evolution process
 */
exports.EvolutionMetricsSchema = zod_1.z.object({
    iterations: zod_1.z.number().int().nonnegative(),
    fitness_score: zod_1.z.number(), // Can be any float value
    fitness_improvement: zod_1.z.number(), // Improvement from initial to final
    duration_ms: zod_1.z.number().int().positive(),
    generations: zod_1.z.number().int().nonnegative().optional(),
    population_size: zod_1.z.number().int().positive().optional(),
    mutation_rate: zod_1.z.number().min(0).max(1).optional(),
    crossover_rate: zod_1.z.number().min(0).max(1).optional(),
    convergence_generation: zod_1.z.number().int().nonnegative().optional(), // Generation where solution converged
    total_evaluations: zod_1.z.number().int().nonnegative().optional(),
    success_rate: zod_1.z.number().min(0).max(1).optional(), // Percentage of successful test cases
    benchmark_score: zod_1.z.number().optional(), // Comparison to baseline
    additional_metrics: zod_1.z.record(zod_1.z.string(), zod_1.z.any()).optional(),
});
// ============================================================================
// EVOLVED CODE SCHEMA
// ============================================================================
/**
 * Programming language
 */
exports.LanguageEnum = zod_1.z.enum([
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
/**
 * The evolved solution code
 */
exports.EvolvedCodeSchema = zod_1.z.object({
    id: zod_1.z.string().uuid('Invalid UUID format'),
    problem: exports.ProblemSchema,
    // Code content
    language: exports.LanguageEnum,
    code: zod_1.z.string().min(1, 'Code content is required'),
    function_name: zod_1.z.string().optional(),
    class_name: zod_1.z.string().optional(),
    // Evolution metadata
    metrics: exports.EvolutionMetricsSchema,
    timestamp_utc: zod_1.z.string().datetime('Must be UTC ISO-8601 timestamp'),
    // Solution metadata
    is_valid: zod_1.z.boolean(),
    validation_errors: zod_1.z.array(zod_1.z.string()).optional(),
    execution_time_ms: zod_1.z.number().int().nonnegative().optional(),
    memory_used_mb: zod_1.z.number().int().nonnegative().optional(),
    // Parent tracking for lineage
    parent_code_id: zod_1.z.string().uuid().optional(),
    generation_number: zod_1.z.number().int().nonnegative().optional(),
    // Additional metadata
    tags: zod_1.z.array(zod_1.z.string()).optional(),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.any()).optional(),
});
// ============================================================================
// SIMILAR SOLUTION SCHEMA
// ============================================================================
/**
 * A similar solution found in the knowledge base
 */
exports.SimilarSolutionSchema = zod_1.z.object({
    evolved_code: exports.EvolvedCodeSchema,
    similarity_score: zod_1.z.number().min(0).max(1), // 0 to 1, where 1 is exact match
    similarity_method: zod_1.z.enum(['semantic', 'structural', 'hybrid']),
    problem_similarity: zod_1.z.number().min(0).max(1).optional(),
    code_similarity: zod_1.z.number().min(0).max(1).optional(),
    matched_features: zod_1.z.array(zod_1.z.string()).optional(),
    distance: zod_1.z.number().optional(), // For vector-based similarity
});
// ============================================================================
// EVOLUTION LINEAGE SCHEMA
// ============================================================================
/**
 * A node in the evolution tree
 */
exports.EvolutionNodeSchema = zod_1.z.object({
    code_id: zod_1.z.string().uuid(),
    code: zod_1.z.string().optional(),
    fitness_score: zod_1.z.number(),
    timestamp_utc: zod_1.z.string().datetime(),
    generation: zod_1.z.number().int().nonnegative(),
    parent_id: zod_1.z.string().uuid().optional(),
    children_ids: zod_1.z.array(zod_1.z.string().uuid()),
});
/**
 * Full lineage tracking from initial solution to final evolved solution
 */
exports.EvolutionLineageSchema = zod_1.z.object({
    root_code_id: zod_1.z.string().uuid(),
    final_code_id: zod_1.z.string().uuid(),
    total_nodes: zod_1.z.number().int().positive(),
    depth: zod_1.z.number().int().positive(),
    nodes: zod_1.z.array(exports.EvolutionNodeSchema),
    branches: zod_1.z.number().int().nonnegative().optional(),
    metadata: zod_1.z.record(zod_1.z.string(), zod_1.z.any()).optional(),
});
// ============================================================================
// CAPTURE RESULT SCHEMA
// ============================================================================
/**
 * Result of capturing an evolved solution
 */
exports.CaptureResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    code_id: zod_1.z.string().uuid(),
    vector_storage_id: zod_1.z.string().optional(),
    graph_episode_id: zod_1.z.string().optional(),
    timestamp_utc: zod_1.z.string().datetime(),
    processing_time_ms: zod_1.z.number().int().nonnegative(),
    correlation_id: zod_1.z.string().uuid().optional(),
    error: zod_1.z.string().optional(),
    warnings: zod_1.z.array(zod_1.z.string()).optional(),
});
// ============================================================================
// CAPTURE METRICS SCHEMA
// ============================================================================
/**
 * Aggregated metrics for the capture system
 */
exports.CaptureMetricsSchema = zod_1.z.object({
    total_captures: zod_1.z.number().int().nonnegative(),
    successful_captures: zod_1.z.number().int().nonnegative(),
    failed_captures: zod_1.z.number().int().nonnegative(),
    average_processing_time_ms: zod_1.z.number().nonnegative(),
    total_storage_size_bytes: zod_1.z.number().int().nonnegative().optional(),
    last_capture_timestamp: zod_1.z.string().datetime().optional(),
    problem_type_distribution: zod_1.z.record(zod_1.z.string(), zod_1.z.number().int().nonnegative()).optional(),
    language_distribution: zod_1.z.record(zod_1.z.string(), zod_1.z.number().int().nonnegative()).optional(),
});
// ============================================================================
// STORAGE REQUEST SCHEMAS
// ============================================================================
/**
 * Request to store code with embedding
 */
exports.StoreWithEmbeddingRequestSchema = zod_1.z.object({
    evolved_code: exports.EvolvedCodeSchema,
    embedding: zod_1.z.array(zod_1.z.number()).optional(), // If not provided, will generate
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Request to search for similar solutions
 */
exports.SearchSimilarRequestSchema = zod_1.z.object({
    problem: exports.ProblemSchema,
    max_results: zod_1.z.number().int().positive().default(10),
    similarity_threshold: zod_1.z.number().min(0).max(1).default(0.5),
    filters: zod_1.z.object({
        language: exports.LanguageEnum.optional(),
        problem_type: exports.ProblemTypeEnum.optional(),
        min_fitness: zod_1.z.number().optional(),
        date_range: zod_1.z.object({
            start: zod_1.z.string().datetime(),
            end: zod_1.z.string().datetime(),
        }).optional(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Request to get evolution lineage
 */
exports.GetLineageRequestSchema = zod_1.z.object({
    code_id: zod_1.z.string().uuid(),
    include_code_content: zod_1.z.boolean().default(false),
    max_depth: zod_1.z.number().int().positive().optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================
/**
 * Validate evolved code against canonical schema
 */
function validateEvolvedCode(data) {
    const result = exports.EvolvedCodeSchema.safeParse(data);
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
function validateProblem(data) {
    const result = exports.ProblemSchema.safeParse(data);
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
function validateEvolutionMetrics(data) {
    const result = exports.EvolutionMetricsSchema.safeParse(data);
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
function validateCaptureResult(data) {
    const result = exports.CaptureResultSchema.safeParse(data);
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
//# sourceMappingURL=canonical.js.map