"use strict";
/**
 * Hybrid PES-Evolution Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for hybrid workflows that combine
 * Plan-Execute-Summarize (PES) with Evolutionary Optimization paradigms.
 *
 * The hybrid approach enables:
 * - PES_OPTIMIZE: Use PES to plan, then evolutionary algorithms to optimize
 * - EVOLVE_SOLVE: Use evolution to explore solution space, then PES to refine
 * - ADAPTIVE_EXECUTE: Dynamically switch between paradigms based on performance
 *
 * This schema bridges LoongFlow (PES + Evolution) with OpenEvolve (pure evolution)
 * to enable powerful hybrid problem-solving strategies.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for hybrid workflow
 * data in the glue layer. Do not pass raw API responses between services.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.KnowledgeTransfer = exports.AdaptiveTrigger = exports.HybridExecutionResult = exports.IntegrationMetrics = exports.EvolutionResult = exports.PopulationIndividual = exports.EvolutionaryKnowledge = exports.HybridTask = exports.EvolutionConfig = exports.KnowledgeType = exports.KnowledgeSourceType = exports.AdaptiveAction = exports.AdaptiveTriggerCondition = exports.IntegrationStrategy = exports.HybridTaskType = void 0;
exports.transformLoongFlowSolutionToKnowledge = transformLoongFlowSolutionToKnowledge;
exports.transformHybridResultToSummary = transformHybridResultToSummary;
exports.validateHybridTask = validateHybridTask;
exports.validateEvolutionaryKnowledge = validateEvolutionaryKnowledge;
exports.validateHybridExecutionResult = validateHybridExecutionResult;
exports.validateAdaptiveTrigger = validateAdaptiveTrigger;
exports.isHybridTask = isHybridTask;
exports.isHybridExecutionResult = isHybridExecutionResult;
exports.isEvolutionaryKnowledge = isEvolutionaryKnowledge;
exports.isAdaptiveTrigger = isAdaptiveTrigger;
const zod_1 = require("zod");
const pes_canonical_1 = require("./pes-canonical");
const loongflow_canonical_1 = require("./loongflow-canonical");
/**
 * Hybrid Task Type Enum
 *
 * Defines the type of hybrid task.
 */
exports.HybridTaskType = zod_1.z.enum([
    'pes_optimize', // PES planning, then evolutionary optimization
    'evolve_solve', // Evolutionary exploration, then PES refinement
    'adaptive_execute', // Dynamic switching based on performance
    'parallel_hybrid', // Run both in parallel, merge results
    'interleaved', // Alternate between PES and evolution
]);
/**
 * Integration Strategy Enum
 *
 * Defines how PES and Evolution are integrated.
 */
exports.IntegrationStrategy = zod_1.z.enum([
    'sequential', // Run one after the other
    'parallel', // Run both simultaneously
    'interleaved', // Alternate steps
    'adaptive', // Dynamic switching based on conditions
]);
/**
 * Adaptive Trigger Condition Enum
 *
 * Conditions that trigger adaptive behavior changes.
 */
exports.AdaptiveTriggerCondition = zod_1.z.enum([
    'low_confidence', // Solution confidence below threshold
    'high_complexity', // Problem complexity exceeds threshold
    'stagnation', // No improvement for N iterations
    'timeout_risk', // Risk of exceeding timeout
    'convergence_detected', // Population converged
    'divergence_detected', // Population diverging
]);
/**
 * Adaptive Action Enum
 *
 * Actions to take when trigger condition is met.
 */
exports.AdaptiveAction = zod_1.z.enum([
    'switch_to_evolution', // Switch from PES to evolution
    'switch_to_pes', // Switch from evolution to PES
    'increase_iterations', // Increase iteration budget
    'call_for_help', // Invoke additional solver
    'merge_populations', // Merge solution populations
    'reset_and_restart', // Reset with new parameters
]);
/**
 * Knowledge Source Type Enum
 *
 * Types of evolutionary knowledge sources.
 */
exports.KnowledgeSourceType = zod_1.z.enum([
    'loongflow_solution', // Solution from LoongFlow
    'openevolve_optimization', // Optimization from OpenEvolve
    'hybrid_result', // Result from hybrid workflow
    'external_solution', // Solution from external source
]);
/**
 * Knowledge Type Enum
 *
 * Types of knowledge that can be extracted.
 */
exports.KnowledgeType = zod_1.z.enum([
    'solution_pattern', // Reusable solution pattern
    'planning_strategy', // Planning approach
    'execution_approach', // Execution method
    'parameter_setting', // Effective parameter values
]);
/**
 * ============================================================================
 * HYBRID TASK SCHEMAS
 * ============================================================================
 */
/**
 * Evolution Configuration Schema
 *
 * Generic evolutionary algorithm configuration (for OpenEvolve).
 */
exports.EvolutionConfig = zod_1.z.object({
    generations: zod_1.z.number()
        .int("Generations must be an integer")
        .positive("Generations must be positive")
        .describe("Number of evolutionary generations"),
    population_size: zod_1.z.number()
        .int("Population size must be an integer")
        .positive("Population size must be positive")
        .describe("Size of population"),
    mutation_rate: zod_1.z.number()
        .min(0, "Mutation rate must be >= 0")
        .max(1, "Mutation rate must be <= 1")
        .describe("Mutation probability (0-1)"),
    crossover_rate: zod_1.z.number()
        .min(0, "Crossover rate must be >= 0")
        .max(1, "Crossover rate must be <= 1")
        .describe("Crossover probability (0-1)"),
    // Optional selection method
    selection_method: zod_1.z.string().optional()
        .describe("Selection method (e.g., 'tournament', 'roulette')"),
    // Optional elitism
    elitism_count: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of elite individuals to preserve"),
    // Optional mutation parameters
    mutation_strength: zod_1.z.number().positive().optional()
        .describe("Mutation strength"),
    // Optional representation
    representation_type: zod_1.z.string().optional()
        .describe("Solution representation type"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional evolution metadata"),
});
/**
 * Hybrid Task Schema
 *
 * Represents a task that combines PES and Evolution.
 */
exports.HybridTask = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique task identifier (UUID)"),
    // Task type
    type: exports.HybridTaskType.describe("Type of hybrid task"),
    // Problem (canonical PES Problem)
    problem: pes_canonical_1.Problem.describe("Problem to solve"),
    // PES configuration (LoongFlow)
    pes_config: loongflow_canonical_1.LoongFlowConfig.optional()
        .describe("PES configuration (LoongFlow)"),
    // Evolution configuration
    evolution_config: exports.EvolutionConfig.optional()
        .describe("Evolutionary algorithm configuration"),
    // Integration strategy
    integration_strategy: exports.IntegrationStrategy
        .describe("How to integrate PES and Evolution"),
    // Optional adaptive triggers
    adaptive_triggers: zod_1.z.array(zod_1.z.object({
        condition: exports.AdaptiveTriggerCondition.describe("Trigger condition"),
        threshold: zod_1.z.number().describe("Threshold value"),
        action: exports.AdaptiveAction.describe("Action to take"),
        metadata: zod_1.z.record(zod_1.z.any()).optional().describe("Additional trigger metadata"),
    })).optional().describe("Adaptive behavior triggers"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional task metadata"),
    // Timestamp (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when task was created (ISO-8601)"),
    // Optional correlation ID
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // Optional priority
    priority: zod_1.z.number().int().min(1).max(10).optional()
        .describe("Priority level (1-10)"),
    // Optional timeout
    timeout_ms: zod_1.z.number().int().positive().optional()
        .describe("Task timeout in milliseconds"),
});
/**
 * Evolutionary Knowledge Schema
 *
 * Represents knowledge extracted from evolutionary processes.
 */
exports.EvolutionaryKnowledge = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique knowledge identifier (UUID)"),
    // Source
    source_type: exports.KnowledgeSourceType.describe("Source of this knowledge"),
    // Problem reference
    problem_id: zod_1.z.string().uuid().describe("ID of related problem"),
    // Knowledge type
    knowledge_type: exports.KnowledgeType.describe("Type of knowledge"),
    // Content
    content: zod_1.z.object({
        pattern: zod_1.z.string().describe("Knowledge pattern or strategy"),
        success_rate: zod_1.z.number().min(0).max(1).describe("Observed success rate"),
        avg_score: zod_1.z.number().min(0).max(1).describe("Average performance score"),
        usage_count: zod_1.z.number().int().nonnegative().describe("Number of times used"),
        context: zod_1.z.record(zod_1.z.any()).optional().describe("Context where effective"),
    }).describe("Knowledge content"),
    // Optional source reference
    source_id: zod_1.z.string().optional()
        .describe("ID of the source (solution ID, optimization ID, etc.)"),
    // Optional validity
    valid_until: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp until which this knowledge is valid"),
    // Timestamp (Law of UTC)
    extracted_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when knowledge was extracted (ISO-8601)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional knowledge metadata"),
});
/**
 * Population Individual Schema
 *
 * Represents an individual in an evolutionary population.
 */
exports.PopulationIndividual = zod_1.z.object({
    id: zod_1.z.string().describe("Individual identifier"),
    genes: zod_1.z.any().describe("Individual's genetic encoding"),
    fitness: zod_1.z.number()
        .min(0, "Fitness must be >= 0")
        .describe("Fitness score"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional individual metadata"),
    // Optional lineage
    parent_ids: zod_1.z.array(zod_1.z.string()).optional()
        .describe("IDs of parent individuals"),
    generation: zod_1.z.number().int().nonnegative().optional()
        .describe("Generation when this individual was created"),
});
/**
 * Evolution Result Schema
 *
 * Represents results from an evolutionary optimization run.
 */
exports.EvolutionResult = zod_1.z.object({
    best_fitness: zod_1.z.number()
        .min(0, "Best fitness must be >= 0")
        .describe("Best fitness achieved"),
    generations: zod_1.z.number()
        .int("Generations must be an integer")
        .nonnegative("Generations must be non-negative")
        .describe("Number of generations completed"),
    population: zod_1.z.array(exports.PopulationIndividual)
        .describe("Final population"),
    // Optional convergence info
    converged: zod_1.z.boolean().optional()
        .describe("Whether the population converged"),
    convergence_generation: zod_1.z.number().int().nonnegative().optional()
        .describe("Generation at which convergence occurred"),
    // Optional diversity metrics
    diversity_score: zod_1.z.number().min(0).max(1).optional()
        .describe("Population diversity score (0-1)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional evolution metadata"),
});
/**
 * Integration Metrics Schema
 *
 * Metrics for hybrid workflow integration.
 */
exports.IntegrationMetrics = zod_1.z.object({
    pes_iterations: zod_1.z.number()
        .int("PES iterations must be an integer")
        .nonnegative("PES iterations must be non-negative")
        .describe("Number of PES iterations performed"),
    evolution_generations: zod_1.z.number()
        .int("Evolution generations must be an integer")
        .nonnegative("Evolution generations must be non-negative")
        .describe("Number of evolutionary generations performed"),
    total_duration_ms: zod_1.z.number()
        .int("Total duration must be an integer")
        .nonnegative("Total duration must be non-negative")
        .describe("Total execution duration in milliseconds"),
    synergy_score: zod_1.z.number()
        .min(0, "Synergy score must be >= 0")
        .max(1, "Synergy score must be <= 1")
        .describe("Synergy between PES and Evolution (0-1)"),
    // Optional timing breakdown
    pes_time_ms: zod_1.z.number().int().nonnegative().optional()
        .describe("Time spent in PES phases"),
    evolution_time_ms: zod_1.z.number().int().nonnegative().optional()
        .describe("Time spent in evolution phases"),
    // Optional switch count
    paradigm_switches: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of paradigm switches (for adaptive strategies)"),
    // Optional knowledge transfer
    knowledge_transferred: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of knowledge items transferred between paradigms"),
});
/**
 * Hybrid Execution Result Schema
 *
 * Represents the result of a hybrid workflow execution.
 */
exports.HybridExecutionResult = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique result identifier (UUID)"),
    // Task reference
    task_id: zod_1.z.string().uuid().describe("ID of the hybrid task"),
    // PES result (if PES was used)
    pes_result: pes_canonical_1.ExecutionResult.optional()
        .describe("Result from PES phase"),
    // Evolution result (if evolution was used)
    evolution_result: exports.EvolutionResult.optional()
        .describe("Result from evolution phase"),
    // Integration metrics
    integration_metrics: exports.IntegrationMetrics
        .describe("Metrics about the integration of PES and Evolution"),
    // Best overall solution
    best_solution: zod_1.z.union([
        loongflow_canonical_1.LoongFlowSolution,
        exports.PopulationIndividual,
    ]).optional().describe("Best solution found"),
    // Summary of execution
    summary: zod_1.z.string().optional()
        .describe("Summary of the hybrid execution"),
    // Knowledge extracted
    knowledge_extracted: zod_1.z.array(exports.EvolutionaryKnowledge).optional()
        .describe("Knowledge extracted during execution"),
    // Error (if failed)
    error: zod_1.z.object({
        code: zod_1.z.string().describe("Error code"),
        message: zod_1.z.string().describe("Error message"),
        details: zod_1.z.record(zod_1.z.any()).optional().describe("Error details"),
    }).optional().describe("Error details (present if failed)"),
    // Timestamps (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when execution started (ISO-8601)"),
    completed_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when execution completed (ISO-8601)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional result metadata"),
    // Optional correlation ID
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Adaptive Trigger Schema
 *
 * Represents a trigger for adaptive behavior changes.
 */
exports.AdaptiveTrigger = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique trigger identifier (UUID)"),
    // Condition
    condition: exports.AdaptiveTriggerCondition.describe("Trigger condition"),
    // Threshold
    threshold: zod_1.z.number().describe("Threshold value for trigger"),
    // Action to take
    action: exports.AdaptiveAction.describe("Action to take when triggered"),
    // Optional cooldown
    cooldown_ms: zod_1.z.number().int().positive().optional()
        .describe("Cooldown period before re-triggering (milliseconds)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional trigger metadata"),
    // Timestamp (Law of UTC)
    created_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when trigger was created (ISO-8601)"),
    // Optional last triggered
    last_triggered_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when trigger was last fired (ISO-8601)"),
});
/**
 * Knowledge Transfer Schema
 *
 * Represents knowledge transferred between paradigms.
 */
exports.KnowledgeTransfer = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique transfer identifier (UUID)"),
    // Source and destination
    source_paradigm: zod_1.z.enum(['pes', 'evolution', 'hybrid'])
        .describe("Source paradigm"),
    destination_paradigm: zod_1.z.enum(['pes', 'evolution', 'hybrid'])
        .describe("Destination paradigm"),
    // Knowledge being transferred
    knowledge: exports.EvolutionaryKnowledge.describe("Knowledge to transfer"),
    // Transfer status
    status: zod_1.z.enum(['pending', 'in_progress', 'completed', 'failed'])
        .describe("Transfer status"),
    // Optional result
    transfer_result: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Result of the transfer"),
    // Timestamps (Law of UTC)
    initiated_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when transfer was initiated (ISO-8601)"),
    completed_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when transfer was completed (ISO-8601)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional transfer metadata"),
});
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform LoongFlow solution to evolutionary knowledge
 */
function transformLoongFlowSolutionToKnowledge(solution, problemId, knowledgeType) {
    return {
        id: crypto.randomUUID(),
        source_type: 'loongflow_solution',
        problem_id: problemId,
        knowledge_type: knowledgeType,
        content: {
            pattern: solution.generate_plan,
            success_rate: solution.score,
            avg_score: solution.score,
            usage_count: 1,
            context: {
                island_id: solution.island_id,
                iteration: solution.iteration,
            },
        },
        source_id: solution.solution_id,
        extracted_at: new Date().toISOString(),
        metadata: solution.metadata,
    };
}
/**
 * Transform hybrid execution result to summary
 */
function transformHybridResultToSummary(result) {
    const { Summary: SummarySchema } = require('./pes-canonical');
    return {
        id: crypto.randomUUID(),
        result_id: result.id,
        evaluation: `Hybrid execution ${result.error ? 'failed' : 'completed'} with synergy score of ${result.integration_metrics.synergy_score}`,
        insights: [
            `PES iterations: ${result.integration_metrics.pes_iterations}`,
            `Evolution generations: ${result.integration_metrics.evolution_generations}`,
            `Total duration: ${result.integration_metrics.total_duration_ms}ms`,
            `Paradigm switches: ${result.integration_metrics.paradigm_switches || 0}`,
        ],
        performance_assessment: {
            success_criteria_met: !result.error,
            quality_score: result.best_solution
                ? ('fitness' in result.best_solution ? result.best_solution.fitness : result.best_solution.score)
                : 0,
            efficiency_score: result.integration_metrics.synergy_score,
        },
        recommendations: result.knowledge_extracted?.map(k => k.content.pattern) || [],
        created_at: result.completed_at || new Date().toISOString(),
    };
}
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate a HybridTask object
 */
function validateHybridTask(data) {
    const result = exports.HybridTask.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an EvolutionaryKnowledge object
 */
function validateEvolutionaryKnowledge(data) {
    const result = exports.EvolutionaryKnowledge.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a HybridExecutionResult object
 */
function validateHybridExecutionResult(data) {
    const result = exports.HybridExecutionResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an AdaptiveTrigger object
 */
function validateAdaptiveTrigger(data) {
    const result = exports.AdaptiveTrigger.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */
/**
 * Check if data is a valid HybridTask
 */
function isHybridTask(data) {
    return exports.HybridTask.safeParse(data).success;
}
/**
 * Check if data is a valid HybridExecutionResult
 */
function isHybridExecutionResult(data) {
    return exports.HybridExecutionResult.safeParse(data).success;
}
/**
 * Check if data is a valid EvolutionaryKnowledge
 */
function isEvolutionaryKnowledge(data) {
    return exports.EvolutionaryKnowledge.safeParse(data).success;
}
/**
 * Check if data is a valid AdaptiveTrigger
 */
function isAdaptiveTrigger(data) {
    return exports.AdaptiveTrigger.safeParse(data).success;
}
/**
 * ============================================================================
 * EXAMPLES
 * ============================================================================

/**
 * Example usage:
 * @example
 * ```typescript
 * import {
 *   validateHybridTask,
 *   createPESUTCTimestamp,
 *   createPESCorrelationId
 * } from './hybrid-pes-evolution-canonical';
 *
 * const hybridTask = {
 *   id: createPESCorrelationId(),
 *   type: 'adaptive_execute',
 *   problem: {
 *     id: createPESCorrelationId(),
 *     type: 'optimization',
 *     description: 'Optimize neural network hyperparameters',
 *     context: { dataset: 'MNIST' },
 *     constraints: ['max_layers <= 10'],
 *     success_criteria: ['accuracy > 0.95'],
 *     created_at: createPESUTCTimestamp(),
 *   },
 *   integration_strategy: 'adaptive',
 *   adaptive_triggers: [
 *     {
 *       condition: 'stagnation',
 *       threshold: 5,
 *       action: 'switch_to_evolution',
 *     },
 *     {
 *       condition: 'low_confidence',
 *       threshold: 0.6,
 *       action: 'increase_iterations',
 *     },
 *   ],
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateHybridTask(hybridTask);
 * if (!validation.success) {
 *   console.error('Invalid hybrid task:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=hybrid-pes-evolution-canonical.js.map