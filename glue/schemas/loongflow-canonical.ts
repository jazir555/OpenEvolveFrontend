/**
 * LoongFlow Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for LoongFlow interactions.
 * LoongFlow implements the Plan-Execute-Summarize (PES) paradigm with evolutionary
 * optimization using MAP-Elites and island models.
 *
 * Key LoongFlow Concepts:
 * - PES Paradigm: Plan → Execute → Summarize
 * - MAP-Elites: Quality-diversity optimization for maintaining diverse solution archive
 * - Island Model: Parallel evolution across multiple populations
 * - Boltzmann Sampling: Temperature-based selection for exploration/exploitation
 * - Workers: Planner, Executor, Summarizer agents
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for LoongFlow data in
 * the glue layer. Do not pass raw LoongFlow API responses between services.
 *
 * Reference: Task #1 scan results showing LoongFlow's Solution dataclass and
 * async worker interface.
 */

import { z } from 'zod';
import {
  Problem,
  ExecutionPlan,
  ExecutionResult,
  Summary,
  type Problem as ProblemType,
  type ExecutionPlan as ExecutionPlanType,
  type ExecutionResult as ExecutionResultType,
  type Summary as SummaryType,
} from './pes-canonical';

/**
 * LoongFlow State Enum
 *
 * Tracks the state of a LoongFlow execution.
 */
export const LoongFlowState = z.enum([
  'idle',              // Not started
  'planning',          // Planning phase active
  'executing',         // Execution phase active
  'summarizing',       // Summarization phase active
  'evolving',          // Evolutionary optimization active
  'completed',         // All phases complete
  'failed',            // Failed with error
  'cancelled',         // Cancelled by user
]);

export type LoongFlowState = z.infer<typeof LoongFlowState>;

/**
 * LoongFlow Worker Type Enum
 *
 * Types of workers in the LoongFlow system.
 */
export const LoongFlowWorkerType = z.enum([
  'planner',           // Plans the solution approach
  'executor',          // Executes the plan
  'summarizer',        // Summarizes and evaluates
  'evolver',           // Evolutionary optimizer
]);

export type LoongFlowWorkerType = z.infer<typeof LoongFlowWorkerType>;

/**
 * ============================================================================
 * LOONGFLOW-SPECIFIC SCHEMAS
 * ============================================================================
 */

/**
 * LoongFlow Solution Schema
 *
 * Direct mapping to LoongFlow's Solution dataclass from core-projects.
 * This represents a single solution in the evolutionary population.
 *
 * From LoongFlow source (src/loongflow/agentsdk/memory/evolution/base_memory.py):
 * ```python
 * @dataclass
 * class Solution:
 *     # Solution identification
 *     solution: str = ""
 *     solution_id: str = ""
 *
 *     # Evolution information
 *     generate_plan: str = ""
 *     parent_id: Optional[str] = ""
 *     island_id: Optional[int] = 0
 *     iteration: Optional[int] = 0
 *     timestamp: float = field(default_factory=time.time)
 *     generation: int = 0
 *     sample_cnt: int = 0
 *     sample_weight: float = 0.0
 *
 *     # Performance metrics
 *     score: Optional[float] = 0.0
 *     evaluation: Optional[str] = ""
 *     summary: str = ""
 *
 *     # Metadata
 *     metadata: Dict[str, Any] = field(default_factory=dict)
 * ```
 */
export const LoongFlowSolution = z.object({
  // Solution identification
  solution: z.string()
    .describe("Generated solution content (code, text, etc.)"),

  solution_id: z.string()
    .min(1, "Solution ID cannot be empty")
    .describe("Unique identifier for this solution"),

  // Evolution information
  generate_plan: z.string()
    .describe("Planning rationale for this solution"),

  parent_id: z.string()
    .describe("ID of parent solution (empty string for initial population)"),

  island_id: z.number()
    .int("Island ID must be an integer")
    .nonnegative("Island ID must be non-negative")
    .describe("Island ID in island model"),

  iteration: z.number()
    .int("Iteration must be an integer")
    .nonnegative("Iteration must be non-negative")
    .describe("Iteration number when this solution was created"),

  timestamp: z.number()
    .nonnegative("Timestamp must be non-negative")
    .describe("Unix timestamp when solution was created"),

  generation: z.number()
    .int("Generation must be an integer")
    .nonnegative("Generation must be non-negative")
    .describe("Generation number in evolution"),

  sample_cnt: z.number()
    .int("Sample count must be an integer")
    .nonnegative("Sample count must be non-negative")
    .describe("Number of times this solution was sampled"),

  sample_weight: z.number()
    .describe("Weight for Boltzmann sampling"),

  // Performance metrics
  score: z.number()
    .min(0, "Score must be between 0 and 1")
    .max(1, "Score must be between 0 and 1")
    .describe("Performance score (0-1)"),

  evaluation: z.string()
    .describe("Evaluation result of the solution"),

  summary: z.string()
    .describe("Reflection summary from summarization"),

  // Metadata
  metadata: z.record(z.any())
    .describe("Additional solution metadata"),
});

export type LoongFlowSolution = z.infer<typeof LoongFlowSolution>;

/**
 * LLM Configuration Schema
 *
 * Configuration for LLM providers used by LoongFlow.
 * Direct mapping to Python LLMConfig from src/loongflow/framework/pes/context/config.py
 */
export const LLMConfig = z.object({
  model: z.string()
    .min(1, "Model name cannot be empty")
    .describe("Model identifier (e.g., 'gpt-4o')"),

  url: z.string().optional()
    .describe("API endpoint URL for the language model"),

  api_key: z.string().optional()
    .describe("API key for authentication"),

  model_provider: z.string().optional()
    .describe("Provider of the model (e.g., 'openai', 'azure')"),

  temperature: z.number()
    .min(0, "Temperature must be >= 0")
    .max(2, "Temperature must be <= 2")
    .optional()
    .describe("Controls randomness. Lower is more deterministic"),

  context_length: z.number()
    .int("Context length must be an integer")
    .positive("Context length must be positive")
    .default(65536)
    .describe("Maximum context length in tokens for the model"),

  max_tokens: z.number()
    .int("Max tokens must be an integer")
    .positive("Max tokens must be positive")
    .default(16384)
    .describe("Maximum number of tokens to generate"),

  top_p: z.number()
    .min(0, "Top-p must be >= 0")
    .max(1, "Top-p must be <= 1")
    .optional()
    .describe("Controls nucleus sampling"),

  timeout: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .default(600)
    .describe("Timeout in seconds for a single LLM generation"),

  completion_token_price: z.number()
    .default(0.0)
    .describe("Price per token for completion requests"),

  prompt_token_price: z.number()
    .default(0.0)
    .describe("Price per token for prompt requests"),
});

export type LLMConfig = z.infer<typeof LLMConfig>;

/**
 * Worker Configuration Schema
 *
 * Configuration for LoongFlow workers.
 */
export const WorkerConfig = z.object({
  planner: z.string()
    .min(1, "Planner worker cannot be empty")
    .describe("Planner worker identifier/class"),

  executor: z.string()
    .min(1, "Executor worker cannot be empty")
    .describe("Executor worker identifier/class"),

  summarizer: z.string()
    .min(1, "Summarizer worker cannot be empty")
    .describe("Summarizer worker identifier/class"),

  // Optional worker configurations
  planner_config: z.record(z.any()).optional()
    .describe("Additional planner configuration"),

  executor_config: z.record(z.any()).optional()
    .describe("Additional executor configuration"),

  summarizer_config: z.record(z.any()).optional()
    .describe("Additional summarizer configuration"),
});

export type WorkerConfig = z.infer<typeof WorkerConfig>;

/**
 * Evolution Configuration Schema
 *
 * Configuration for evolutionary optimization in LoongFlow.
 */
export const EvolutionConfig = z.object({
  iterations: z.number()
    .int("Iterations must be an integer")
    .positive("Iterations must be positive")
    .describe("Number of evolutionary iterations"),

  islands: z.number()
    .int("Islands must be an integer")
    .positive("Islands must be positive")
    .describe("Number of islands in island model"),

  sample_size: z.number()
    .int("Sample size must be an integer")
    .positive("Sample size must be positive")
    .describe("Sample size for Boltzmann sampling"),

  // Optional MAP-Elites configuration
  map_elites_enabled: z.boolean().optional()
    .describe("Whether MAP-Elites is enabled"),

  map_dimensions: z.array(z.object({
    name: z.string().describe("Dimension name"),
    bins: z.number().int().positive().describe("Number of bins"),
  })).optional().describe("MAP-Elites fitness map dimensions"),

  // Optional Boltzmann sampling configuration
  temperature_initial: z.number().positive().optional()
    .describe("Initial temperature for Boltzmann sampling"),

  temperature_decay: z.number().min(0).max(1).optional()
    .describe("Temperature decay rate"),

  // Optional migration configuration (island model)
  migration_interval: z.number().int().positive().optional()
    .describe("Iterations between island migrations"),

  migration_size: z.number().int().nonnegative().optional()
    .describe("Number of solutions to migrate"),

  // Optional selection pressure
  tournament_size: z.number().int().positive().optional()
    .describe("Tournament size for selection"),

  elite_size: z.number().int().nonnegative().optional()
    .describe("Number of elite solutions to preserve"),
});

export type EvolutionConfig = z.infer<typeof EvolutionConfig>;

/**
 * LoongFlow Configuration Schema
 *
 * Complete configuration for a LoongFlow execution.
 */
export const LoongFlowConfig = z.object({
  // Task identification
  task_name: z.string()
    .min(1, "Task name cannot be empty")
    .describe("Name of the task to solve"),

  // LLM configuration
  llm_config: LLMConfig
    .describe("LLM provider configuration"),

  // Worker configuration
  workers: WorkerConfig
    .describe("Worker configuration"),

  // Evolution configuration
  evolution: EvolutionConfig
    .describe("Evolutionary optimization configuration"),

  // Optional timeout
  timeout_ms: z.number().int().positive().optional()
    .describe("Overall timeout for LoongFlow execution"),

  // Optional checkpointing
  checkpoint_interval: z.number().int().positive().optional()
    .describe("Iterations between checkpoints"),

  checkpoint_dir: z.string().optional()
    .describe("Directory for checkpoint storage"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Additional configuration metadata"),
});

export type LoongFlowConfig = z.infer<typeof LoongFlowConfig>;

/**
 * LoongFlow Request Schema
 *
 * Represents a request to execute LoongFlow on a problem.
 */
export const LoongFlowRequest = z.object({
  // Problem reference (canonical PES Problem)
  problem_id: z.string().uuid()
    .describe("ID of the problem to solve"),

  // LoongFlow configuration
  task_config: LoongFlowConfig
    .describe("LoongFlow configuration for this task"),

  // Initial context
  initial_context: z.record(z.any())
    .describe("Initial context for the execution"),

  // Optional seed solutions
  seed_solutions: z.array(LoongFlowSolution.partial()).optional()
    .describe("Optional seed solutions to initialize population"),

  // Optional correlation ID
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  // Optional timestamp
  created_at: z.string().datetime().optional()
    .describe("UTC timestamp when request was created"),
});

export type LoongFlowRequest = z.infer<typeof LoongFlowRequest>;

/**
 * LoongFlow Response Schema
 *
 * Represents a response from LoongFlow execution.
 */
export const LoongFlowResponse = z.object({
  // Request reference
  request_id: z.string().uuid()
    .describe("ID of the original request"),

  // Current state
  state: LoongFlowState
    .describe("Current LoongFlow state"),

  // Current solution (if any)
  current_solution: LoongFlowSolution.optional()
    .describe("Current best solution"),

  // Best solutions from all islands
  best_solutions: z.array(LoongFlowSolution)
    .describe("Best solutions from each island"),

  // Iteration info
  iteration: z.number()
    .int("Iteration must be an integer")
    .nonnegative("Iteration must be non-negative")
    .describe("Current iteration number"),

  // Checkpoints
  checkpoints: z.array(z.string())
    .describe("List of checkpoint identifiers"),

  // Execution metrics
  metrics: z.object({
    total_duration_ms: z.number().int().nonnegative().optional()
      .describe("Total execution duration in milliseconds"),

    planning_time_ms: z.number().int().nonnegative().optional()
      .describe("Time spent in planning phase"),

    execution_time_ms: z.number().int().nonnegative().optional()
      .describe("Time spent in execution phase"),

    summarization_time_ms: z.number().int().nonnegative().optional()
      .describe("Time spent in summarization phase"),

    solutions_generated: z.number().int().nonnegative().optional()
      .describe("Total solutions generated"),

    islands_completed: z.number().int().nonnegative().optional()
      .describe("Number of islands completed"),
  }).optional().describe("Execution metrics"),

  // Error (if failed)
  error: z.object({
    code: z.string().describe("Error code"),
    message: z.string().describe("Error message"),
    details: z.record(z.any()).optional().describe("Error details"),
  }).optional().describe("Error details (present if state is 'failed')"),

  // Optional timestamp
  timestamp: z.string().datetime().optional()
    .describe("UTC timestamp of the response"),

  // Optional correlation ID
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Additional metadata"),
});

export type LoongFlowResponse = z.infer<typeof LoongFlowResponse>;

/**
 * LoongFlow Checkpoint Schema
 *
 * Represents a checkpoint in LoongFlow execution.
 */
export const LoongFlowCheckpoint = z.object({
  checkpoint_id: z.string()
    .describe("Checkpoint identifier"),

  iteration: z.number().int().nonnegative()
    .describe("Iteration number at checkpoint"),

  state: LoongFlowState
    .describe("State at checkpoint"),

  // Population snapshots
  populations: z.array(z.object({
    island_id: z.number().int().nonnegative(),
    solutions: z.array(LoongFlowSolution),
  })).describe("Population snapshots for each island"),

  // Fitness maps (if MAP-Elites enabled)
  fitness_maps: z.array(z.object({
    island_id: z.number().int().nonnegative(),
    map: z.record(z.array(LoongFlowSolution)),
  })).optional().describe("MAP-Elites fitness maps"),

  // Timestamp (Law of UTC)
  timestamp: z.string().datetime()
    .describe("UTC timestamp of checkpoint (ISO-8601)"),

  // Optional metadata
  metadata: z.record(z.any()).optional()
    .describe("Additional checkpoint metadata"),
});

export type LoongFlowCheckpoint = z.infer<typeof LoongFlowCheckpoint>;

/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */

/**
 * Transform raw LoongFlow solution to canonical format
 */
export function transformLoongFlowSolutionToCanonical(
  rawSolution: any
): LoongFlowSolution {
  return {
    solution: rawSolution.solution || '',
    solution_id: rawSolution.solution_id || '',
    generate_plan: rawSolution.generate_plan || '',
    parent_id: rawSolution.parent_id || '',
    island_id: rawSolution.island_id ?? 0,
    iteration: rawSolution.iteration ?? 0,
    timestamp: rawSolution.timestamp || Date.now() / 1000,
    generation: rawSolution.generation ?? 0,
    sample_cnt: rawSolution.sample_cnt ?? 0,
    sample_weight: rawSolution.sample_weight ?? 0.0,
    score: rawSolution.score ?? 0.0,
    evaluation: rawSolution.evaluation || '',
    summary: rawSolution.summary || '',
    metadata: rawSolution.metadata || {},
  };
}

/**
 * Transform canonical LoongFlow solution to raw format
 */
export function transformCanonicalToLoongFlowSolution(
  canonical: LoongFlowSolution
): Record<string, any> {
  return {
    solution: canonical.solution,
    solution_id: canonical.solution_id,
    generate_plan: canonical.generate_plan,
    parent_id: canonical.parent_id,
    island_id: canonical.island_id,
    iteration: canonical.iteration,
    timestamp: canonical.timestamp,
    generation: canonical.generation,
    sample_cnt: canonical.sample_cnt,
    sample_weight: canonical.sample_weight,
    score: canonical.score,
    evaluation: canonical.evaluation,
    summary: canonical.summary,
    metadata: canonical.metadata,
  };
}

/**
 * Transform LoongFlow response to canonical ExecutionResult
 */
export function transformLoongFlowResponseToExecutionResult(
  lfResponse: LoongFlowResponse,
  problemId: string,
  planId: string
): ExecutionResultType {
  const { import_pessimistic: z } = require('zod');

  return {
    id: crypto.randomUUID(),
    problem_id: problemId,
    plan_id: planId,
    state: lfResponse.state === 'completed' ? 'completed'
      : lfResponse.state === 'failed' ? 'failed'
        : lfResponse.state === 'cancelled' ? 'cancelled'
          : 'running',
    outputs: {
      best_solutions: lfResponse.best_solutions,
      current_solution: lfResponse.current_solution,
      iteration: lfResponse.iteration,
      checkpoints: lfResponse.checkpoints,
    },
    metrics: {
      duration_ms: lfResponse.metrics?.total_duration_ms || 0,
      success: lfResponse.state === 'completed',
      score: lfResponse.current_solution?.score,
      error_count: lfResponse.error ? 1 : 0,
    },
    artifacts: [],
    logs: [],
    error: lfResponse.error,
    created_at: lfResponse.timestamp || new Date().toISOString(),
    completed_at: ['completed', 'failed', 'cancelled'].includes(lfResponse.state)
      ? lfResponse.timestamp || new Date().toISOString()
      : undefined,
    metadata: lfResponse.metadata,
    correlation_id: lfResponse.correlation_id,
  };
}

/**
 * Transform canonical Problem to LoongFlow request
 */
export function transformCanonicalProblemToLoongFlowRequest(
  problem: ProblemType,
  config: LoongFlowConfig,
  initialContext: Record<string, any> = {}
): LoongFlowRequest {
  return {
    problem_id: problem.id,
    task_config: config,
    initial_context: {
      ...initialContext,
      problem_description: problem.description,
      constraints: problem.constraints,
      success_criteria: problem.success_criteria,
    },
    correlation_id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
  };
}

/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */

/**
 * Validate a LoongFlowSolution object
 */
export function validateLoongFlowSolution(data: unknown): {
  success: boolean;
  data?: LoongFlowSolution;
  errors?: string[];
} {
  const result = LoongFlowSolution.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a LoongFlowConfig object
 */
export function validateLoongFlowConfig(data: unknown): {
  success: boolean;
  data?: LoongFlowConfig;
  errors?: string[];
} {
  const result = LoongFlowConfig.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a LoongFlowRequest object
 */
export function validateLoongFlowRequest(data: unknown): {
  success: boolean;
  data?: LoongFlowRequest;
  errors?: string[];
} {
  const result = LoongFlowRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a LoongFlowResponse object
 */
export function validateLoongFlowResponse(data: unknown): {
  success: boolean;
  data?: LoongFlowResponse;
  errors?: string[];
} {
  const result = LoongFlowResponse.safeParse(data);

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
 * Check if data is a valid LoongFlowSolution
 */
export function isLoongFlowSolution(data: unknown): data is LoongFlowSolution {
  return LoongFlowSolution.safeParse(data).success;
}

/**
 * Check if data is a valid LoongFlowRequest
 */
export function isLoongFlowRequest(data: unknown): data is LoongFlowRequest {
  return LoongFlowRequest.safeParse(data).success;
}

/**
 * Check if data is a valid LoongFlowResponse
 */
export function isLoongFlowResponse(data: unknown): data is LoongFlowResponse {
  return LoongFlowResponse.safeParse(data).success;
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
 * import {
 *   validateLoongFlowRequest,
 *   createPESUTCTimestamp,
 *   createPESCorrelationId
 * } from './loongflow-canonical';
 *
 * const request = {
 *   problem_id: createPESCorrelationId(),
 *   task_config: {
 *     task_name: 'Optimize neural network architecture',
 *     llm_config: {
 *       provider: 'openai',
 *       model: 'gpt-4',
 *       temperature: 0.7,
 *       max_tokens: 2000,
 *     },
 *     workers: {
 *       planner: 'LLMPlanner',
 *       executor: 'CodeExecutor',
 *       summarizer: 'QualitySummarizer',
 *     },
 *     evolution: {
 *       iterations: 10,
 *       islands: 4,
 *       sample_size: 100,
 *     },
 *   },
 *   initial_context: {
 *     dataset: 'MNIST',
 *     max_layers: 10,
 *   },
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateLoongFlowRequest(request);
 * if (!validation.success) {
 *   console.error('Invalid request:', validation.errors);
 * }
 * ```
 */
