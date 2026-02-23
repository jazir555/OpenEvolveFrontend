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

import { z } from 'zod';

/**
 * Problem Type Enum
 *
 * Categorizes the type of problem being solved.
 */
export const ProblemType = z.enum([
  'optimization',      // Finding optimal parameters/solutions
  'reasoning',         // Logical deduction and inference
  'generation',        // Creating new content/code
  'validation',        // Verifying correctness/properties
  'classification',    // Categorizing data
  'prediction',        // Forecasting future values
  'synthesis',         // Combining multiple sources
]);

export type ProblemType = z.infer<typeof ProblemType>;

/**
 * Execution Step Type Enum
 *
 * Defines the type of each step in an execution plan.
 */
export const ExecutionStepType = z.enum([
  'plan',              // Planning step
  'execute',           // Execution step
  'summarize',         // Summarization/evaluation step
  'validate',          // Validation step
  'transform',         // Data transformation step
  'aggregate',         // Aggregation step
]);

export type ExecutionStepType = z.infer<typeof ExecutionStepType>;

/**
 * Execution State Enum
 *
 * Tracks the state of an execution workflow.
 */
export const ExecutionState = z.enum([
  'pending',           // Not yet started
  'running',           // Currently executing
  'completed',         // Successfully completed
  'failed',            // Failed with error
  'cancelled',         // Cancelled by user
  'timeout',           // Exceeded timeout
  'paused',            // Paused (can be resumed)
]);

export type ExecutionState = z.infer<typeof ExecutionState>;

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
export const Problem = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique problem identifier (UUID)"),

  // Problem classification
  type: ProblemType.describe("Type of problem to solve"),

  // Problem description
  description: z.string()
    .min(1, "Problem description cannot be empty")
    .max(100000, "Problem description too long (max 100KB)")
    .describe("Human-readable problem description"),

  // Additional context
  context: z.record(z.any())
    .describe("Additional problem context and parameters"),

  // Constraints
  constraints: z.array(z.string())
    .describe("List of constraints/constraints that must be satisfied"),

  // Success criteria
  success_criteria: z.array(z.string())
    .describe("List of criteria that define successful solution"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for extensibility"),

  // Timestamp (Law of UTC)
  created_at: z.string().datetime()
    .describe("UTC timestamp when problem was created (ISO-8601)"),

  // Optional priority
  priority: z.number().int().min(1).max(10)
    .optional()
    .describe("Priority level (1-10, higher = more important)"),

  // Optional tags
  tags: z.array(z.string()).optional()
    .describe("Tags for categorization and filtering"),
});

export type Problem = z.infer<typeof Problem>;

/**
 * Execution Step Schema
 *
 * Represents a single step in an execution plan.
 */
export const ExecutionStep = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique step identifier (UUID)"),

  // Step type
  type: ExecutionStepType.describe("Type of execution step"),

  // Step description
  description: z.string()
    .min(1, "Step description cannot be empty")
    .describe("Human-readable description of this step"),

  // Worker type
  worker_type: z.string()
    .min(1, "Worker type cannot be empty")
    .describe("Type of worker that will execute this step"),

  // Step parameters
  parameters: z.record(z.any())
    .describe("Parameters for this step"),

  // Timeout
  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(3600000, "Timeout cannot exceed 1 hour")
    .describe("Step timeout in milliseconds"),

  // Optional retry configuration
  retry_config: z.object({
    max_retries: z.number().int().nonnegative().optional(),
    backoff_ms: z.number().int().positive().optional(),
    max_backoff_ms: z.number().int().positive().optional(),
  }).optional().describe("Retry configuration"),

  // Optional dependencies
  depends_on: z.array(z.string().uuid()).optional()
    .describe("IDs of steps this step depends on"),
});

export type ExecutionStep = z.infer<typeof ExecutionStep>;

/**
 * Execution Plan Schema
 *
 * Represents a complete plan for solving a problem.
 */
export const ExecutionPlan = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique plan identifier (UUID)"),

  // Problem reference
  problem_id: z.string().uuid().describe("ID of the problem this plan solves"),

  // Plan steps
  steps: z.array(ExecutionStep)
    .min(1, "Plan must have at least one step")
    .describe("Ordered list of execution steps"),

  // Dependencies (explicit)
  dependencies: z.array(z.object({
    step_id: z.string().uuid().describe("Step ID"),
    depends_on: z.array(z.string().uuid()).describe("IDs of prerequisite steps"),
  })).optional().describe("Explicit dependency graph"),

  // Resource requirements
  resource_requirements: z.object({
    estimated_duration_ms: z.number().int().positive().optional()
      .describe("Estimated total duration in milliseconds"),
    required_workers: z.array(z.string()).optional()
      .describe("Types of workers needed"),
    memory_mb: z.number().int().positive().optional()
      .describe("Estimated memory requirement in MB"),
    cpu_cores: z.number().int().positive().optional()
      .describe("Estimated CPU core requirement"),
  }).optional().describe("Resource requirements for this plan"),

  // Plan metadata
  metadata: z.record(z.any()).optional()
    .describe("Optional plan metadata"),

  // Timestamp (Law of UTC)
  created_at: z.string().datetime()
    .describe("UTC timestamp when plan was created (ISO-8601)"),

  // Optional planner info
  planner_type: z.string().optional()
    .describe("Type of planner that created this plan"),

  // Optional confidence
  confidence_score: z.number().min(0).max(1).optional()
    .describe("Planner's confidence in this plan (0-1)"),
});

export type ExecutionPlan = z.infer<typeof ExecutionPlan>;

/**
 * Log Entry Schema
 *
 * Represents a single log entry during execution.
 */
export const LogEntry = z.object({
  timestamp: z.string().datetime()
    .describe("UTC timestamp of the log entry (ISO-8601)"),

  level: z.enum(['debug', 'info', 'warn', 'error', 'fatal'])
    .describe("Log level"),

  message: z.string()
    .describe("Log message"),

  // Optional structured data
  context: z.record(z.any()).optional()
    .describe("Additional context data"),

  // Optional correlation
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  // Optional step reference
  step_id: z.string().uuid().optional()
    .describe("ID of the step that generated this log"),
});

export type LogEntry = z.infer<typeof LogEntry>;

/**
 * Artifact Schema
 *
 * Represents an artifact produced during execution.
 */
export const Artifact = z.object({
  type: z.string()
    .describe("Type of artifact (e.g., 'code', 'data', 'model', 'plot')"),

  uri: z.string()
    .describe("URI or path to the artifact"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Additional artifact metadata"),

  // Optional size
  size_bytes: z.number().int().nonnegative().optional()
    .describe("Size of artifact in bytes"),

  // Optional checksum
  checksum: z.string().optional()
    .describe("Checksum for integrity verification"),
});

export type Artifact = z.infer<typeof Artifact>;

/**
 * Execution Metrics Schema
 *
 * Represents metrics collected during execution.
 */
export const ExecutionMetrics = z.object({
  duration_ms: z.number()
    .int("Duration must be an integer")
    .nonnegative("Duration must be non-negative")
    .describe("Execution duration in milliseconds"),

  success: z.boolean()
    .describe("Whether execution was successful"),

  score: z.number().min(0).max(1).optional()
    .describe("Performance score (0-1)"),

  error_count: z.number()
    .int("Error count must be an integer")
    .nonnegative("Error count must be non-negative")
    .describe("Number of errors encountered"),

  // Optional detailed metrics
  steps_completed: z.number().int().nonnegative().optional()
    .describe("Number of steps completed"),

  steps_failed: z.number().int().nonnegative().optional()
    .describe("Number of steps failed"),

  memory_peak_mb: z.number().int().nonnegative().optional()
    .describe("Peak memory usage in MB"),

  cpu_time_ms: z.number().int().nonnegative().optional()
    .describe("Total CPU time in milliseconds"),

  // Optional iteration info
  iterations: z.number().int().positive().optional()
    .describe("Number of iterations performed"),
});

export type ExecutionMetrics = z.infer<typeof ExecutionMetrics>;

/**
 * Execution Result Schema
 *
 * Represents the result of executing a plan.
 */
export const ExecutionResult = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique result identifier (UUID)"),

  // References
  problem_id: z.string().uuid().describe("ID of the problem being solved"),
  plan_id: z.string().uuid().describe("ID of the plan that was executed"),

  // State
  state: ExecutionState.describe("Current execution state"),

  // Outputs
  outputs: z.record(z.any())
    .describe("Execution outputs (results, data, etc.)"),

  // Metrics
  metrics: ExecutionMetrics.describe("Execution metrics"),

  // Artifacts produced
  artifacts: z.array(Artifact).optional()
    .describe("Artifacts produced during execution"),

  // Execution logs
  logs: z.array(LogEntry).optional()
    .describe("Execution logs"),

  // Error details (if failed)
  error: z.object({
    code: z.string().describe("Error code"),
    message: z.string().describe("Error message"),
    details: z.record(z.any()).optional().describe("Error details"),
    stack_trace: z.string().optional().describe("Stack trace"),
  }).optional().describe("Error details (present if state is 'failed')"),

  // Timestamps (Law of UTC)
  created_at: z.string().datetime()
    .describe("UTC timestamp when execution started (ISO-8601)"),

  completed_at: z.string().datetime().optional()
    .describe("UTC timestamp when execution completed (ISO-8601)"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Optional metadata"),

  // Optional correlation ID
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type ExecutionResult = z.infer<typeof ExecutionResult>;

/**
 * Performance Assessment Schema
 *
 * Represents an assessment of execution performance.
 */
export const PerformanceAssessment = z.object({
  success_criteria_met: z.boolean()
    .describe("Whether all success criteria were met"),

  quality_score: z.number()
    .min(0, "Quality score must be between 0 and 1")
    .max(1, "Quality score must be between 0 and 1")
    .describe("Overall quality score (0-1)"),

  efficiency_score: z.number()
    .min(0, "Efficiency score must be between 0 and 1")
    .max(1, "Efficiency score must be between 0 and 1")
    .describe("Efficiency score (0-1)"),

  // Optional detailed scores
  criteria_scores: z.record(z.number().min(0).max(1)).optional()
    .describe("Per-criterion scores"),

  // Optional improvements
  improvement_suggestions: z.array(z.string()).optional()
    .describe("Suggestions for improvement"),
});

export type PerformanceAssessment = z.infer<typeof PerformanceAssessment>;

/**
 * Summary Schema
 *
 * Represents a summary/evaluation of an execution result.
 * This is the "Summarize" phase of the PES paradigm.
 */
export const Summary = z.object({
  // Unique identifier
  id: z.string().uuid().describe("Unique summary identifier (UUID)"),

  // Result reference
  result_id: z.string().uuid().describe("ID of the result being summarized"),

  // Evaluation
  evaluation: z.string()
    .min(1, "Evaluation cannot be empty")
    .describe("Summary evaluation of the execution"),

  // Insights
  insights: z.array(z.string())
    .describe("Key insights extracted from the execution"),

  // Performance assessment
  performance_assessment: PerformanceAssessment
    .describe("Detailed performance assessment"),

  // Recommendations
  recommendations: z.array(z.string())
    .describe("Recommendations for future executions"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Optional metadata"),

  // Timestamp (Law of UTC)
  created_at: z.string().datetime()
    .describe("UTC timestamp when summary was created (ISO-8601)"),

  // Optional summarizer info
  summarizer_type: z.string().optional()
    .describe("Type of summarizer that created this summary"),

  // Optional confidence
  confidence_score: z.number().min(0).max(1).optional()
    .describe("Summarizer's confidence in this summary (0-1)"),
});

export type Summary = z.infer<typeof Summary>;

/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */

/**
 * Transform raw PES data to canonical Problem format
 */
export function transformProblemToCanonical(
  rawProblem: any,
  generatedId?: string
): Problem {
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
export function transformCanonicalToProblem(
  canonical: Problem
): Record<string, any> {
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
export function transformExecutionResultToCanonical(
  rawResult: any,
  generatedId?: string
): ExecutionResult {
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
    logs: (rawResult.logs || []).map((log: any) => ({
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
export function transformCanonicalToSummary(
  canonical: Summary
): Record<string, any> {
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
export function validateProblem(data: unknown): {
  success: boolean;
  data?: Problem;
  errors?: string[];
} {
  const result = Problem.safeParse(data);

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
export function validateExecutionPlan(data: unknown): {
  success: boolean;
  data?: ExecutionPlan;
  errors?: string[];
} {
  const result = ExecutionPlan.safeParse(data);

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
export function validateExecutionResult(data: unknown): {
  success: boolean;
  data?: ExecutionResult;
  errors?: string[];
} {
  const result = ExecutionResult.safeParse(data);

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
export function validateSummary(data: unknown): {
  success: boolean;
  data?: Summary;
  errors?: string[];
} {
  const result = Summary.safeParse(data);

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
export function createPESUTCTimestamp(): string {
  return new Date().toISOString();
}

/**
 * Create a correlation ID
 */
export function createPESCorrelationId(): string {
  return crypto.randomUUID();
}

/**
 * Check if data is a valid Problem
 */
export function isProblem(data: unknown): data is Problem {
  return Problem.safeParse(data).success;
}

/**
 * Check if data is a valid ExecutionResult
 */
export function isExecutionResult(data: unknown): data is ExecutionResult {
  return ExecutionResult.safeParse(data).success;
}

/**
 * Check if data is a valid Summary
 */
export function isSummary(data: unknown): data is Summary {
  return Summary.safeParse(data).success;
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
