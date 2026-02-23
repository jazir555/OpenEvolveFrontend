/**
 * ROMA Canonical Schema
 *
 * Defines the canonical data model for ROMA integration following the
 * Anti-Corruption Layer (ACL) pattern from the Federation Constitution.
 *
 * This schema provides:
 * 1. Canonical types for ROMA entities
 * 2. Zod validation schemas
 * 3. Transformation functions to/from canonical format
 * 4. Validation functions with detailed error reporting
 *
 * "Law of the Air Gap" - ROMA API format is isolated behind this canonical layer.
 */

import { z } from 'zod';

// ============================================================================
// ENUMS
// ============================================================================

/**
 * ROMA execution status enum
 */
export enum RomaExecutionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  TIMEOUT = 'timeout',
}

/**
 * ROMA module type enum
 */
export enum RomaModuleType {
  ATOMIZER = 'atomizer',
  PLANNER = 'planner',
  EXECUTOR = 'executor',
  AGGREGATOR = 'aggregator',
  VERIFIER = 'verifier',
}

/**
 * ROMA task type enum (MECE framework)
 */
export enum RomaTaskType {
  RETRIEVE = 'retrieve',
  WRITE = 'write',
  THINK = 'think',
  CODE_INTERPRET = 'code_interpret',
  IMAGE_GENERATION = 'image_generation',
}

/**
 * ROMA prediction strategy enum
 */
export enum RomaPredictionStrategy {
  PREDICT = 'predict',
  CHAIN_OF_THOUGHT = 'chain_of_thought',
  REACT = 'react',
  CODE_ACT = 'code_act',
  BEST_OF_N = 'best_of_n',
  REFINE = 'refine',
  PARALLEL = 'parallel',
  MAJORITY = 'majority',
}

/**
 * ROMA execution method enum
 */
export enum RomaExecutionMethod {
  TRADITIONAL = 'traditional',
  CLAUDIOMIRO = 'claudiomiro',
  DATAPIZZA = 'datapizza',
  ROMA = 'roma',
  HYBRID = 'hybrid',
  ROMA_MDAP_MAKER = 'roma_mdap_maker',
  AUTO = 'auto',
}

// ============================================================================
// CANONICAL TYPES
// ============================================================================

/**
 * Canonical ROMA execution request
 */
export interface RomaExecutionRequest {
  goal: string;
  max_depth?: number;
  config_profile?: string;
  execution_method?: RomaExecutionMethod;
  timeout_ms?: number;
  correlation_id?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Canonical ROMA execution response
 */
export interface RomaExecutionResponse {
  execution_id: string;
  status: RomaExecutionStatus;
  initial_goal: string;
  result?: unknown;
  statistics: RomaExecutionStatistics;
  timestamp: string; // UTC ISO-8601
  error?: string;
}

/**
 * Canonical ROMA execution statistics
 */
export interface RomaExecutionStatistics {
  total_tasks: number;
  completed_tasks: number;
  execution_time_ms: number;
  average_time_per_task_ms: number;
  tool_usage: Record<string, number>;
  module_usage: Record<string, number>;
}

/**
 * Canonical ROMA task node
 */
export interface RomaTaskNode {
  task_id: string;
  goal: string;
  status: RomaExecutionStatus;
  task_type: RomaTaskType;
  depth: number;
  parent_id?: string;
  children_ids: string[];
  result?: unknown;
  timestamp: string;
}

/**
 * Canonical ROMA checkpoint
 */
export interface RomaCheckpoint {
  checkpoint_id: string;
  execution_id: string;
  timestamp: string;
  nodes: RomaTaskNode[];
  dependencies: RomaDependency[];
  metadata: Record<string, unknown>;
}

/**
 * Canonical ROMA dependency
 */
export interface RomaDependency {
  from_task_id: string;
  to_task_id: string;
  dependency_type: 'hard' | 'soft';
}

/**
 * Canonical ROMA profile configuration
 */
export interface RomaProfileConfig {
  profile: string;
  max_depth: number;
  timeout: number;
  enable_verification: boolean;
  enable_checkpointing: boolean;
  modules: Partial<Record<RomaModuleType, RomaModuleConfig>>;
}

/**
 * Canonical ROMA module configuration
 */
export interface RomaModuleConfig {
  enabled: boolean;
  prediction_strategy?: RomaPredictionStrategy;
  max_retries: number;
  timeout: number;
}

// ============================================================================
// ZOD VALIDATION SCHEMAS
// ============================================================================

/**
 * Zod schema for RomaExecutionRequest
 */
export const RomaExecutionRequestSchema = z.object({
  goal: z.string().min(1, 'Goal cannot be empty'),
  max_depth: z.number().int().positive().max(10).optional(),
  config_profile: z.string().optional(),
  execution_method: z.nativeEnum(RomaExecutionMethod).optional(),
  timeout_ms: z.number().int().positive().max(600000).optional(),
  correlation_id: z.string().uuid().optional(),
  metadata: z.record(z.unknown()).optional(),
});

/**
 * Zod schema for RomaExecutionStatistics
 */
export const RomaExecutionStatisticsSchema = z.object({
  total_tasks: z.number().int().nonnegative(),
  completed_tasks: z.number().int().nonnegative(),
  execution_time_ms: z.number().int().nonnegative(),
  average_time_per_task_ms: z.number().int().nonnegative(),
  tool_usage: z.record(z.number().int().nonnegative()),
  module_usage: z.record(z.number().int().nonnegative()),
});

/**
 * Zod schema for RomaExecutionResponse
 */
export const RomaExecutionResponseSchema = z.object({
  execution_id: z.string().min(1),
  status: z.nativeEnum(RomaExecutionStatus),
  initial_goal: z.string().min(1),
  result: z.unknown().optional(),
  statistics: RomaExecutionStatisticsSchema,
  timestamp: z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/, 'Must be UTC ISO-8601 format'),
  error: z.string().optional(),
});

/**
 * Zod schema for RomaTaskNode
 */
export const RomaTaskNodeSchema = z.object({
  task_id: z.string().min(1),
  goal: z.string().min(1),
  status: z.nativeEnum(RomaExecutionStatus),
  task_type: z.nativeEnum(RomaTaskType),
  depth: z.number().int().nonnegative(),
  parent_id: z.string().optional(),
  children_ids: z.array(z.string()),
  result: z.unknown().optional(),
  timestamp: z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/),
});

/**
 * Zod schema for RomaCheckpoint
 */
export const RomaCheckpointSchema = z.object({
  checkpoint_id: z.string().min(1),
  execution_id: z.string().min(1),
  timestamp: z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/),
  nodes: z.array(RomaTaskNodeSchema),
  dependencies: z.array(z.object({
    from_task_id: z.string().min(1),
    to_task_id: z.string().min(1),
    dependency_type: z.enum(['hard', 'soft']),
  })),
  metadata: z.record(z.unknown()),
});

// ============================================================================
// TRANSFORMATION FUNCTIONS
// ============================================================================

/**
 * Transform ROMA API response to canonical format
 *
 * @param apiResponse - Raw response from ROMA API
 * @returns Canonical RomaExecutionResponse
 */
export function transformRomaResponseToCanonical(
  apiResponse: Record<string, unknown>
): RomaExecutionResponse {
  return {
    execution_id: String(apiResponse.execution_id || apiResponse.executionId || ''),
    status: apiResponse.status as RomaExecutionStatus,
    initial_goal: String(apiResponse.goal || apiResponse.initial_goal || ''),
    result: apiResponse.result,
    statistics: {
      total_tasks: Number(apiResponse.total_tasks ?? apiResponse.totalTasks ?? 0),
      completed_tasks: Number(apiResponse.completed_tasks ?? apiResponse.completedTasks ?? 0),
      execution_time_ms: Number(apiResponse.execution_time ?? apiResponse.executionTime ?? 0),
      average_time_per_task_ms: Number(apiResponse.average_time_per_task ?? 0),
      tool_usage: (apiResponse.tool_usage ?? apiResponse.toolUsage ?? {}) as Record<string, number>,
      module_usage: (apiResponse.module_usage ?? apiResponse.moduleUsage ?? {}) as Record<string, number>,
    },
    timestamp: normalizeTimestamp(apiResponse.timestamp ?? apiResponse.created_at),
    error: apiResponse.error ? String(apiResponse.error) : undefined,
  };
}

/**
 * Transform canonical request to ROMA API format
 *
 * @param canonical - Canonical RomaExecutionRequest
 * @returns ROMA API request body
 */
export function transformCanonicalToRomaRequest(
  canonical: RomaExecutionRequest
): Record<string, unknown> {
  return {
    goal: canonical.goal,
    max_depth: canonical.max_depth,
    config_profile: canonical.config_profile,
    execution_method: canonical.execution_method,
    timeout: canonical.timeout_ms ? Math.floor(canonical.timeout_ms / 1000) : undefined,
    metadata: {
      ...canonical.metadata,
      correlation_id: canonical.correlation_id,
    },
  };
}

/**
 * Transform ROMA checkpoint to canonical format
 *
 * @param apiResponse - Raw checkpoint response from ROMA API
 * @returns Canonical RomaCheckpoint
 */
export function transformRomaCheckpointToCanonical(
  apiResponse: Record<string, unknown>
): RomaCheckpoint {
  const nodes = (apiResponse.nodes || apiResponse.sub_tasks || []) as unknown[];

  return {
    checkpoint_id: String(apiResponse.checkpoint_id || apiResponse.checkpointId || ''),
    execution_id: String(apiResponse.execution_id || apiResponse.executionId || ''),
    timestamp: normalizeTimestamp(apiResponse.timestamp || apiResponse.created_at),
    nodes: (Array.isArray(nodes) ? nodes : []).map((node: unknown) => ({
      task_id: String((node as any).task_id || (node as any).taskId || ''),
      goal: String((node as any).goal || ''),
      status: (node as any).status as RomaExecutionStatus,
      task_type: (node as any).task_type as RomaTaskType,
      depth: Number((node as any).depth || 0),
      parent_id: (node as any).parent_id ? String((node as any).parent_id) : undefined,
      children_ids: ((node as any).children_ids || []).map(String),
      result: (node as any).result,
      timestamp: normalizeTimestamp((node as any).timestamp),
    })),
    dependencies: transformDependencies(apiResponse.dependencies),
    metadata: (apiResponse.metadata || {}) as Record<string, unknown>,
  };
}

/**
 * Transform dependencies to canonical format
 */
function transformDependencies(
  dependencies: unknown
): RomaDependency[] {
  if (!Array.isArray(dependencies)) return [];

  return dependencies.map((dep: unknown) => ({
    from_task_id: String((dep as any).from_task_id || (dep as any).fromTaskId || ''),
    to_task_id: String((dep as any).to_task_id || (dep as any).toTaskId || ''),
    dependency_type: ((dep as any).dependency_type || (dep as any).hard ? 'hard' : 'soft') as 'hard' | 'soft',
  }));
}

/**
 * Normalize timestamp to UTC ISO-8601 format
 */
function normalizeTimestamp(timestamp: unknown): string {
  if (typeof timestamp === 'string') {
    // Already a string - ensure it ends with Z for UTC
    return timestamp.endsWith('Z') ? timestamp : `${timestamp}Z`;
  }

  if (timestamp instanceof Date) {
    return timestamp.toISOString();
  }

  // Fallback to current time in UTC
  return new Date().toISOString();
}

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/**
 * Validation result type
 */
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

/**
 * Validate ROMA execution request
 *
 * @param request - Request to validate
 * @returns Validation result with errors if invalid
 */
export function validateRomaExecutionRequest(
  request: unknown
): ValidationResult {
  const result = RomaExecutionRequestSchema.safeParse(request);

  if (result.success) {
    return { isValid: true, errors: [] };
  }

  const errors = result.error.errors.map(
    (err) => `${err.path.join('.')}: ${err.message}`
  );

  return { isValid: false, errors };
}

/**
 * Validate ROMA execution response
 *
 * @param response - Response to validate
 * @returns Validation result with errors if invalid
 */
export function validateRomaExecutionResponse(
  response: unknown
): ValidationResult {
  const result = RomaExecutionResponseSchema.safeParse(response);

  if (result.success) {
    return { isValid: true, errors: [] };
  }

  const errors = result.error.errors.map(
    (err) => `${err.path.join('.')}: ${err.message}`
  );

  return { isValid: false, errors };
}

/**
 * Validate ROMA checkpoint
 *
 * @param checkpoint - Checkpoint to validate
 * @returns Validation result with errors if invalid
 */
export function validateRomaCheckpoint(
  checkpoint: unknown
): ValidationResult {
  const result = RomaCheckpointSchema.safeParse(checkpoint);

  if (result.success) {
    return { isValid: true, errors: [] };
  }

  const errors = result.error.errors.map(
    (err) => `${err.path.join('.')}: ${err.message}`
  );

  return { isValid: false, errors };
}

// ============================================================================
// EXAMPLES (for documentation and testing)
// ============================================================================

export const RomaExamples = {
  executionRequest: {
    goal: 'Design a RESTful API for user management',
    max_depth: 3,
    config_profile: 'balanced',
    execution_method: RomaExecutionMethod.ROMA,
    timeout_ms: 30000,
  } as RomaExecutionRequest,

  executionResponse: {
    execution_id: 'roma-1706900000-abc123',
    status: RomaExecutionStatus.COMPLETED,
    initial_goal: 'Design a RESTful API for user management',
    result: {
      summary: 'API design completed with CRUD endpoints',
      reasoning: 'Following REST principles...',
    },
    statistics: {
      total_tasks: 5,
      completed_tasks: 5,
      execution_time_ms: 12500,
      average_time_per_task_ms: 2500,
      tool_usage: {
        'search': 3,
        'code_generator': 2,
      },
      module_usage: {
        'atomizer': 1,
        'planner': 1,
        'executor': 5,
        'aggregator': 1,
        'verifier': 1,
      },
    },
    timestamp: '2026-02-22T12:34:56Z',
  } as RomaExecutionResponse,

  checkpoint: {
    checkpoint_id: 'ckpt-1706900000-xyz789',
    execution_id: 'roma-1706900000-abc123',
    timestamp: '2026-02-22T12:34:50Z',
    nodes: [
      {
        task_id: 'task-1',
        goal: 'Define API endpoints',
        status: RomaExecutionStatus.COMPLETED,
        task_type: RomaTaskType.WRITE,
        depth: 1,
        children_ids: ['task-2', 'task-3'],
        timestamp: '2026-02-22T12:34:51Z',
      },
    ],
    dependencies: [],
    metadata: {},
  } as RomaCheckpoint,
};
