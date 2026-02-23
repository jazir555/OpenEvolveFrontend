"use strict";
/**
 * PES Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Plan-Execute-Summarize (PES)
 * pattern interactions. PES is a fundamental paradigm in AI problem-solving where:
 * 1. PLAN: Generate a strategy for solving the problem
 * 2. EXECUTE: Carry out the plan to produce a solution
 * 3. SUMMARIZE: Reflect on the result and extract insights
 *
 * This is the generic PES schema that can be used by any system implementing
 * the PES pattern, including LoongFlow and custom hybrid workflows.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for PES data in the glue layer.
 * Do not pass raw PES API responses between services.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Summary = exports.PerformanceAssessment = exports.ExecutionResult = exports.ExecutionMetrics = exports.Artifact = exports.LogEntry = exports.ExecutionPlan = exports.ExecutionStep = exports.Problem = exports.ExecutionState = exports.ExecutionStepType = exports.ProblemType = void 0;
exports.transformProblemToCanonical = transformProblemToCanonical;
exports.transformCanonicalToProblem = transformCanonicalToProblem;
exports.transformExecutionResultToCanonical = transformExecutionResultToCanonical;
exports.transformCanonicalToSummary = transformCanonicalToSummary;
exports.validateProblem = validateProblem;
exports.validateExecutionPlan = validateExecutionPlan;
exports.validateExecutionResult = validateExecutionResult;
exports.validateSummary = validateSummary;
exports.createPESUTCTimestamp = createPESUTCTimestamp;
exports.createPESCorrelationId = createPESCorrelationId;
exports.isProblem = isProblem;
exports.isExecutionResult = isExecutionResult;
exports.isSummary = isSummary;
const zod_1 = require("zod");
/**
 * Problem Type Enum
 *
 * Categorizes the type of problem being solved.
 */
exports.ProblemType = zod_1.z.enum([
    'optimization', // Finding optimal parameters/solutions
    'reasoning', // Logical deduction and inference
    'generation', // Creating new content/code
    'validation', // Verifying correctness/properties
    'classification', // Categorizing data
    'prediction', // Forecasting future values
    'synthesis', // Combining multiple sources
]);
/**
 * Execution Step Type Enum
 *
 * Defines the type of each step in an execution plan.
 */
exports.ExecutionStepType = zod_1.z.enum([
    'plan', // Planning step
    'execute', // Execution step
    'summarize', // Summarization/evaluation step
    'validate', // Validation step
    'transform', // Data transformation step
    'aggregate', // Aggregation step
]);
/**
 * Execution State Enum
 *
 * Tracks the state of an execution workflow.
 */
exports.ExecutionState = zod_1.z.enum([
    'pending', // Not yet started
    'running', // Currently executing
    'completed', // Successfully completed
    'failed', // Failed with error
    'cancelled', // Cancelled by user
    'timeout', // Exceeded timeout
    'paused', // Paused (can be resumed)
]);
/**
 * ============================================================================
 * CORE PES SCHEMAS
 * ============================================================================
 */
/**
 * Problem Schema
 *
 * Represents a problem to be solved using the PES paradigm.
 * This is the input to any PES-based workflow.
 */
exports.Problem = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique problem identifier (UUID)"),
    // Problem classification
    type: exports.ProblemType.describe("Type of problem to solve"),
    // Problem description
    description: zod_1.z.string()
        .min(1, "Problem description cannot be empty")
        .max(100000, "Problem description too long (max 100KB)")
        .describe("Human-readable problem description"),
    // Additional context
    context: zod_1.z.record(zod_1.z.any())
        .describe("Additional problem context and parameters"),
    // Constraints
    constraints: zod_1.z.array(zod_1.z.string())
        .describe("List of constraints/constraints that must be satisfied"),
    // Success criteria
    success_criteria: zod_1.z.array(zod_1.z.string())
        .describe("List of criteria that define successful solution"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for extensibility"),
    // Timestamp (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when problem was created (ISO-8601)"),
    // Optional priority
    priority: zod_1.z.number().int().min(1).max(10).optional()
        .describe("Priority level (1-10, higher = more important)"),
    // Optional tags
    tags: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Tags for categorization and filtering"),
});
/**
 * Execution Step Schema
 *
 * Represents a single step in an execution plan.
 */
exports.ExecutionStep = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique step identifier (UUID)"),
    // Step type
    type: exports.ExecutionStepType.describe("Type of execution step"),
    // Step description
    description: zod_1.z.string()
        .min(1, "Step description cannot be empty")
        .describe("Human-readable description of this step"),
    // Worker type
    worker_type: zod_1.z.string()
        .min(1, "Worker type cannot be empty")
        .describe("Type of worker that will execute this step"),
    // Step parameters
    parameters: zod_1.z.record(zod_1.z.any())
        .describe("Parameters for this step"),
    // Timeout
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(3600000, "Timeout cannot exceed 1 hour")
        .describe("Step timeout in milliseconds"),
    // Optional retry configuration
    retry_config: zod_1.z.object({
        max_retries: zod_1.z.number().int().nonnegative().optional(),
        backoff_ms: zod_1.z.number().int().positive().optional(),
        max_backoff_ms: zod_1.z.number().int().positive().optional(),
    }).optional().describe("Retry configuration"),
    // Optional dependencies
    depends_on: zod_1.z.array(zod_1.z.string().uuid()).optional()
        .describe("IDs of steps this step depends on"),
});
/**
 * Execution Plan Schema
 *
 * Represents a complete plan for solving a problem.
 */
exports.ExecutionPlan = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique plan identifier (UUID)"),
    // Problem reference
    problem_id: zod_1.z.string().uuid().describe("ID of the problem this plan solves"),
    // Plan steps
    steps: zod_1.z.array(exports.ExecutionStep)
        .min(1, "Plan must have at least one step")
        .describe("Ordered list of execution steps"),
    // Dependencies (explicit)
    dependencies: zod_1.z.array(zod_1.z.object({
        step_id: zod_1.z.string().uuid().describe("Step ID"),
        depends_on: zod_1.z.array(zod_1.z.string().uuid()).describe("IDs of prerequisite steps"),
    })).optional().describe("Explicit dependency graph"),
    // Resource requirements
    resource_requirements: zod_1.z.object({
        estimated_duration_ms: zod_1.z.number().int().positive().optional()
            .describe("Estimated total duration in milliseconds"),
        required_workers: zod_1.z.array(zod_1.z.string()).optional()
            .describe("Types of workers needed"),
        memory_mb: zod_1.z.number().int().positive().optional()
            .describe("Estimated memory requirement in MB"),
        cpu_cores: zod_1.z.number().int().positive().optional()
            .describe("Estimated CPU core requirement"),
    }).optional().describe("Resource requirements for this plan"),
    // Plan metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional plan metadata"),
    // Timestamp (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when plan was created (ISO-8601)"),
    // Optional planner info
    planner_type: zod_1.z.string().optional()
        .describe("Type of planner that created this plan"),
    // Optional confidence
    confidence_score: zod_1.z.number().min(0).max(1).optional()
        .describe("Planner's confidence in this plan (0-1)"),
});
/**
 * Log Entry Schema
 *
 * Represents a single log entry during execution.
 */
exports.LogEntry = zod_1.z.object({
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the log entry (ISO-8601)"),
    level: zod_1.z.enum(['debug', 'info', 'warn', 'error', 'fatal'])
        .describe("Log level"),
    message: zod_1.z.string()
        .describe("Log message"),
    // Optional structured data
    context: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional context data"),
    // Optional correlation
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    // Optional step reference
    step_id: zod_1.z.string().uuid().optional()
        .describe("ID of the step that generated this log"),
});
/**
 * Artifact Schema
 *
 * Represents an artifact produced during execution.
 */
exports.Artifact = zod_1.z.object({
    type: zod_1.z.string()
        .describe("Type of artifact (e.g., 'code', 'data', 'model', 'plot')"),
    uri: zod_1.z.string()
        .describe("URI or path to the artifact"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional artifact metadata"),
    // Optional size
    size_bytes: zod_1.z.number().int().nonnegative().optional()
        .describe("Size of artifact in bytes"),
    // Optional checksum
    checksum: zod_1.z.string().optional()
        .describe("Checksum for integrity verification"),
});
/**
 * Execution Metrics Schema
 *
 * Represents metrics collected during execution.
 */
exports.ExecutionMetrics = zod_1.z.object({
    duration_ms: zod_1.z.number()
        .int("Duration must be an integer")
        .nonnegative("Duration must be non-negative")
        .describe("Execution duration in milliseconds"),
    success: zod_1.z.boolean()
        .describe("Whether execution was successful"),
    score: zod_1.z.number().min(0).max(1).optional()
        .describe("Performance score (0-1)"),
    error_count: zod_1.z.number()
        .int("Error count must be an integer")
        .nonnegative("Error count must be non-negative")
        .describe("Number of errors encountered"),
    // Optional detailed metrics
    steps_completed: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of steps completed"),
    steps_failed: zod_1.z.number().int().nonnegative().optional()
        .describe("Number of steps failed"),
    memory_peak_mb: zod_1.z.number().int().nonnegative().optional()
        .describe("Peak memory usage in MB"),
    cpu_time_ms: zod_1.z.number().int().nonnegative().optional()
        .describe("Total CPU time in milliseconds"),
    // Optional iteration info
    iterations: zod_1.z.number().int().positive().optional()
        .describe("Number of iterations performed"),
});
/**
 * Execution Result Schema
 *
 * Represents the result of executing a plan.
 */
exports.ExecutionResult = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique result identifier (UUID)"),
    // References
    problem_id: zod_1.z.string().uuid().describe("ID of the problem being solved"),
    plan_id: zod_1.z.string().uuid().describe("ID of the plan that was executed"),
    // State
    state: exports.ExecutionState.describe("Current execution state"),
    // Outputs
    outputs: zod_1.z.record(zod_1.z.any())
        .describe("Execution outputs (results, data, etc.)"),
    // Metrics
    metrics: exports.ExecutionMetrics.describe("Execution metrics"),
    // Artifacts produced
    artifacts: zod_1.z.array(exports.Artifact).optional()
        .describe("Artifacts produced during execution"),
    // Execution logs
    logs: zod_1.z.array(exports.LogEntry).optional()
        .describe("Execution logs"),
    // Error details (if failed)
    error: zod_1.z.object({
        code: zod_1.z.string().describe("Error code"),
        message: zod_1.z.string().describe("Error message"),
        details: zod_1.z.record(zod_1.z.any()).optional().describe("Error details"),
        stack_trace: zod_1.z.string().optional().describe("Stack trace"),
    }).optional().describe("Error details (present if state is 'failed')"),
    // Timestamps (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when execution started (ISO-8601)"),
    completed_at: zod_1.z.string().datetime().optional()
        .describe("UTC timestamp when execution completed (ISO-8601)"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata"),
    // Optional correlation ID
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Performance Assessment Schema
 *
 * Represents an assessment of execution performance.
 */
exports.PerformanceAssessment = zod_1.z.object({
    success_criteria_met: zod_1.z.boolean()
        .describe("Whether all success criteria were met"),
    quality_score: zod_1.z.number()
        .min(0, "Quality score must be between 0 and 1")
        .max(1, "Quality score must be between 0 and 1")
        .describe("Overall quality score (0-1)"),
    efficiency_score: zod_1.z.number()
        .min(0, "Efficiency score must be between 0 and 1")
        .max(1, "Efficiency score must be between 0 and 1")
        .describe("Efficiency score (0-1)"),
    // Optional detailed scores
    criteria_scores: zod_1.z.record(zod_1.z.number().min(0).max(1)).optional()
        .describe("Per-criterion scores"),
    // Optional improvements
    improvement_suggestions: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Suggestions for improvement"),
});
/**
 * Summary Schema
 *
 * Represents a summary/evaluation of an execution result.
 * This is the "Summarize" phase of the PES paradigm.
 */
exports.Summary = zod_1.z.object({
    // Unique identifier
    id: zod_1.z.string().uuid().describe("Unique summary identifier (UUID)"),
    // Result reference
    result_id: zod_1.z.string().uuid().describe("ID of the result being summarized"),
    // Evaluation
    evaluation: zod_1.z.string()
        .min(1, "Evaluation cannot be empty")
        .describe("Summary evaluation of the execution"),
    // Insights
    insights: zod_1.z.array(zod_1.z.string())
        .describe("Key insights extracted from the execution"),
    // Performance assessment
    performance_assessment: exports.PerformanceAssessment
        .describe("Detailed performance assessment"),
    // Recommendations
    recommendations: zod_1.z.array(zod_1.z.string())
        .describe("Recommendations for future executions"),
    // Optional metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata"),
    // Timestamp (Law of UTC)
    created_at: zod_1.z.string().datetime()
        .describe("UTC timestamp when summary was created (ISO-8601)"),
    // Optional summarizer info
    summarizer_type: zod_1.z.string().optional()
        .describe("Type of summarizer that created this summary"),
    // Optional confidence
    confidence_score: zod_1.z.number().min(0).max(1).optional()
        .describe("Summarizer's confidence in this summary (0-1)"),
});
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform raw PES data to canonical Problem format
 */
function transformProblemToCanonical(rawProblem, generatedId) {
    const id = generatedId || rawProblem.id || crypto.randomUUID();
    return {
        id,
        type: rawProblem.type || 'reasoning',
        description: rawProblem.description || rawProblem.problem || '',
        context: rawProblem.context || {},
        constraints: rawProblem.constraints || [],
        success_criteria: rawProblem.success_criteria || rawProblem.objectives || [],
        metadata: rawProblem.metadata || {},
        created_at: rawProblem.created_at || new Date().toISOString(),
        priority: rawProblem.priority,
        tags: rawProblem.tags,
    };
}
/**
 * Transform canonical Problem to external format
 */
function transformCanonicalToProblem(canonical) {
    return {
        id: canonical.id,
        type: canonical.type,
        description: canonical.description,
        context: canonical.context,
        constraints: canonical.constraints,
        success_criteria: canonical.success_criteria,
        metadata: canonical.metadata,
        created_at: canonical.created_at,
        priority: canonical.priority,
        tags: canonical.tags,
    };
}
/**
 * Transform raw execution result to canonical ExecutionResult format
 */
function transformExecutionResultToCanonical(rawResult, generatedId) {
    const id = generatedId || rawResult.id || crypto.randomUUID();
    return {
        id,
        problem_id: rawResult.problem_id || rawResult.task_id || '',
        plan_id: rawResult.plan_id || rawResult.workflow_id || '',
        state: rawResult.state || rawResult.status || 'pending',
        outputs: rawResult.outputs || rawResult.result || {},
        metrics: rawResult.metrics || {
            duration_ms: rawResult.duration_ms || 0,
            success: rawResult.success !== undefined ? rawResult.success : true,
            error_count: rawResult.error_count || 0,
        },
        artifacts: rawResult.artifacts || [],
        logs: (rawResult.logs || []).map((log) => ({
            timestamp: log.timestamp || new Date().toISOString(),
            level: log.level || 'info',
            message: log.message || '',
            context: log.context,
            correlation_id: log.correlation_id,
            step_id: log.step_id,
        })),
        error: rawResult.error,
        created_at: rawResult.created_at || rawResult.started_at || new Date().toISOString(),
        completed_at: rawResult.completed_at || rawResult.finished_at,
        metadata: rawResult.metadata,
        correlation_id: rawResult.correlation_id,
    };
}
/**
 * Transform canonical Summary to external format
 */
function transformCanonicalToSummary(canonical) {
    return {
        id: canonical.id,
        result_id: canonical.result_id,
        evaluation: canonical.evaluation,
        insights: canonical.insights,
        performance_assessment: canonical.performance_assessment,
        recommendations: canonical.recommendations,
        metadata: canonical.metadata,
        created_at: canonical.created_at,
        summarizer_type: canonical.summarizer_type,
        confidence_score: canonical.confidence_score,
    };
}
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate a Problem object
 */
function validateProblem(data) {
    const result = exports.Problem.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an ExecutionPlan object
 */
function validateExecutionPlan(data) {
    const result = exports.ExecutionPlan.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate an ExecutionResult object
 */
function validateExecutionResult(data) {
    const result = exports.ExecutionResult.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Validate a Summary object
 */
function validateSummary(data) {
    const result = exports.Summary.safeParse(data);
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
 * Create a UTC timestamp (Law of UTC)
 */
function createPESUTCTimestamp() {
    return new Date().toISOString();
}
/**
 * Create a correlation ID
 */
function createPESCorrelationId() {
    return crypto.randomUUID();
}
/**
 * Check if data is a valid Problem
 */
function isProblem(data) {
    return exports.Problem.safeParse(data).success;
}
/**
 * Check if data is a valid ExecutionResult
 */
function isExecutionResult(data) {
    return exports.ExecutionResult.safeParse(data).success;
}
/**
 * Check if data is a valid Summary
 */
function isSummary(data) {
    return exports.Summary.safeParse(data).success;
}
/**
 * ============================================================================
 * EXAMPLES
 * ============================================================================
 */
/**
 * Example usage:
 * @example
 * ```typescript
 * import { validateProblem, createPESUTCTimestamp } from './pes-canonical';
 *
 * const problemData = {
 *   type: 'optimization',
 *   description: 'Find optimal hyperparameters for ML model',
 *   context: { model: 'neural_network', dataset: 'imagenet' },
 *   constraints: ['max_epochs <= 100', 'batch_size in [32, 64, 128]'],
 *   success_criteria: ['accuracy > 0.9', 'training_time < 3600s'],
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateProblem(problemData);
 * if (!validation.success) {
 *   console.error('Invalid problem:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=pes-canonical.js.map