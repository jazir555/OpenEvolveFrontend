/**
 * MAKER Canonical Schema - Anti-Corruption Layer (ACL)
 *
 * Federation Constitution Compliant Canonical Schema for MAKER Engine integration.
 * All external data must be transformed to/from this format.
 *
 * Law 6: All timestamps are UTC ISO-8601 strings
 */

import { z } from 'zod';

/**
 * Voting Mode Enum
 */
export const VotingModeSchema = z.enum([
  'simple',
  'k_ahead',
  'weighted',
  'consensus',
]);

export type VotingMode = z.infer<typeof VotingModeSchema>;

/**
 * Red Flag Severity Enum
 */
export const RedFlagSeveritySchema = z.enum([
  'low',
  'medium',
  'high',
  'critical',
]);

export type RedFlagSeverity = z.infer<typeof RedFlagSeveritySchema>;

/**
 * MAKER Configuration Schema
 */
export const MakerConfigSchema = z.object({
  k_min: z.number().int().min(1).default(2),
  k_max: z.number().int().min(1).default(8),
  max_votes_per_step: z.number().int().min(1).default(60),
  max_steps: z.number().int().min(1).default(1000),
  timeout_seconds: z.number().int().min(1).default(90),
  checkpoint_interval: z.number().int().min(1).default(25),
});

export type MakerConfig = z.infer<typeof MakerConfigSchema>;

/**
 * MAKER Step Schema
 */
export const MakerStepSchema = z.object({
  step_id: z.string().min(1, 'Step ID cannot be empty'),
  prompt_template: z.string().min(1, 'Prompt template cannot be empty'),
  task_type: z.string().default('general'),
  priority: z.number().int().min(0).default(0),
  system_prompt: z.string().optional(),
  expected_schema: z.record(z.any()).optional(),
  stop_sequences: z.array(z.string()).optional(),
  metadata: z.record(z.any()).optional(),
});

export type MakerStep = z.infer<typeof MakerStepSchema>;

/**
 * Agent Vote Schema
 */
export const AgentVoteSchema = z.object({
  agent_id: z.string(),
  vote: z.any(),
  raw_text: z.string(),
  timestamp: z.string().datetime('UTC timestamp required'),
  red_flags: z.array(z.string()).optional(),
});

export type AgentVote = z.infer<typeof AgentVoteSchema>;

/**
 * MAKER Run Result Schema
 */
export const MakerRunResultSchema = z.object({
  success: z.boolean(),
  steps_completed: z.number().int().min(0),
  votes_cast: z.number().int().min(0),
  red_flags_detected: z.number().int().min(0),
  final_action: z.any().optional(),
  agent_votes: z.array(AgentVoteSchema).optional(),
  red_flags: z.array(z.string()).optional(),
  metrics: z.record(z.any()).optional(),
  terminated_reason: z.string(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime('UTC timestamp required'),
  execution_time_ms: z.number().int().min(0).optional(),
});

export type MakerRunResult = z.infer<typeof MakerRunResultSchema>;

/**
 * Validation Functions
 */
export function validateMakerConfig(data: unknown): {
  success: boolean;
  data?: MakerConfig;
  errors?: string[];
} {
  const result = MakerConfigSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateMakerStep(data: unknown): {
  success: boolean;
  data?: MakerStep;
  errors?: string[];
} {
  const result = MakerStepSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateMakerRunResult(data: unknown): {
  success: boolean;
  data?: MakerRunResult;
  errors?: string[];
} {
  const result = MakerRunResultSchema.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Type Guards
 */
export function isMakerConfig(data: unknown): data is MakerConfig {
  return typeof data === 'object' && data !== null
    && 'k_min' in data && 'k_max' in data;
}

export function isMakerStep(data: unknown): data is MakerStep {
  return typeof data === 'object' && data !== null
    && 'step_id' in data && 'prompt_template' in data;
}

/**
 * Example Usage
 */
export const MakerExamples = {
  config: {
    k_min: 2,
    k_max: 7,
    max_votes_per_step: 30,
    max_steps: 100,
    timeout_seconds: 60,
    checkpoint_interval: 20,
  } as MakerConfig,

  step: {
    step_id: 'maker-step-001',
    prompt_template: 'Analyze this: {state}',
    task_type: 'analysis',
    priority: 1,
    system_prompt: 'You are a helpful assistant',
    expected_schema: { type: 'object' },
    metadata: { domain: 'general' },
  } as MakerStep,

  result: {
    success: true,
    steps_completed: 5,
    votes_cast: 15,
    red_flags_detected: 0,
    final_action: { action: 'continue', reason: 'Consensus reached' },
    agent_votes: [],
    red_flags: [],
    metrics: { total_votes: 15, unique_agents: 3 },
    terminated_reason: 'completed',
    timestamp: '2025-02-17T12:30:45.000Z',
    execution_time_ms: 1500,
  } as MakerRunResult,
};
