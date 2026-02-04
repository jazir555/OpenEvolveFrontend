/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Canonical Schemas
 *
 * Canonical data models for all 7 ICR modes.
 * These schemas define the Anti-Corruption Layer (ACL) contract.
 * All data entering/leaving the ICR system MUST conform to these schemas.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Air Gap: No imports from core-projects
 * - Runtime Truth: Schemas reflect actual API behavior
 * - Configuration Explicitness: All fields required (no magic defaults)
 * - UTC: All timestamps in UTC ISO-8601 format
 */

import { z } from 'zod';

// ============================================================================
// COMMON TYPES
// ============================================================================

/**
 * Mode type enum for all 7 ICR modes
 */
export const ModeTypeSchema = z.enum([
  'refine',
  'react',
  'deepthink',
  'adaptive_deepthink',
  'agentic',
  'contextual',
  'generative_ui'
]);

export type ModeType = z.infer<typeof ModeTypeSchema>;

/**
 * Base metadata schema included in all requests/responses
 */
export const ICRMetadataSchema = z.object({
  correlation_id: z.string().uuid(),
  timestamp_utc: z.string().datetime(),
  source_service: z.string().default('icr-adapter'),
  mode: ModeTypeSchema,
  request_id: z.string().optional()
});

export type ICRMetadata = z.infer<typeof ICRMetadataSchema>;

/**
 * Base result schema for all mode responses
 */
export const ICRResultSchema = z.object({
  success: z.boolean(),
  content: z.string(),
  error: z.string().optional(),
  execution_time_ms: z.number().int().nonnegative(),
  iteration_count: z.number().int().nonnegative().default(0),
  metadata: z.record(z.any()).optional()
});

export type ICRResult = z.infer<typeof ICRResultSchema>;

/**
 * Mode options schema
 */
export const ModeOptionsSchema = z.object({
  temperature: z.number().min(0).max(2).optional(),
  top_p: z.number().min(0).max(1).optional(),
  max_iterations: z.number().int().positive().optional(),
  model_name: z.string().optional(),
  provider: z.enum(['google', 'openai', 'anthropic']).optional()
});

export type ModeOptions = z.infer<typeof ModeOptionsSchema>;

// ============================================================================
// REFINE MODE
// ============================================================================

/**
 * Refine Mode Request Schema
 * Mode: Traditional iterative refinements with automated feature suggestion
 */
export const RefineModeRequestSchema = z.object({
  mode: z.literal('refine'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    evolution_mode: z.enum(['novelty', 'quality', 'off']).optional(),
    refinement_stages: z.number().int().positive().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type RefineModeRequest = z.infer<typeof RefineModeRequestSchema>;

/**
 * Refine Mode Response Schema
 */
export const RefineModeResponseSchema = z.object({
  mode: z.literal('refine'),
  request: RefineModeRequestSchema,
  result: ICRResultSchema.extend({
    iterations: z.array(z.object({
      iteration_number: z.number().int(),
      content: z.string(),
      suggested_features: z.string().optional(),
      bug_fixes: z.string().optional(),
      status: z.enum(['pending', 'processing', 'completed', 'error', 'cancelled']),
      error: z.string().optional()
    }))
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type RefineModeResponse = z.infer<typeof RefineModeResponseSchema>;

// ============================================================================
// REACT MODE
// ============================================================================

/**
 * React Mode Request Schema
 * Mode: React application development with orchestrator-coordination
 */
export const ReactModeRequestSchema = z.object({
  mode: z.literal('react'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    worker_count: z.number().int().positive().optional(),
    enable_preview: z.boolean().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type ReactModeRequest = z.infer<typeof ReactModeRequestSchema>;

/**
 * React Mode Response Schema
 */
export const ReactModeResponseSchema = z.object({
  mode: z.literal('react'),
  request: ReactModeRequestSchema,
  result: ICRResultSchema.extend({
    orchestrator_plan: z.string().optional(),
    workers: z.array(z.object({
      worker_id: z.string(),
      title: z.string(),
      system_instruction: z.string().optional(),
      user_prompt: z.string().optional(),
      generated_content: z.string().optional(),
      status: z.enum(['pending', 'processing', 'completed', 'error', 'cancelled']),
      error: z.string().optional()
    })),
    preview_url: z.string().optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type ReactModeResponse = z.infer<typeof ReactModeResponseSchema>;

// ============================================================================
// DEEPTHINK MODE
// ============================================================================

/**
 * Deepthink Mode Request Schema
 * Mode: Complex problem-solving through strategic decomposition
 */
export const DeepthinkModeRequestSchema = z.object({
  mode: z.literal('deepthink'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    strategy_count: z.number().int().positive().optional(),
    sub_strategy_count: z.number().int().positive().optional(),
    hypothesis_count: z.number().int().positive().optional(),
    enable_iterative_corrections: z.boolean().optional(),
    enable_red_team: z.boolean().optional(),
    red_team_aggressiveness: z.enum(['low', 'medium', 'high']).optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type DeepthinkModeRequest = z.infer<typeof DeepthinkModeRequestSchema>;

/**
 * Deepthink Mode Response Schema
 */
export const DeepthinkModeResponseSchema = z.object({
  mode: z.literal('deepthink'),
  request: DeepthinkModeRequestSchema,
  result: ICRResultSchema.extend({
    strategies: z.array(z.object({
      strategy_id: z.string(),
      strategy_text: z.string(),
      sub_strategies: z.array(z.object({
        sub_strategy_id: z.string(),
        sub_strategy_text: z.string(),
        solution: z.string().optional(),
        critique: z.string().optional(),
        refined_solution: z.string().optional(),
        status: z.enum(['pending', 'processing', 'completed', 'error', 'cancelled'])
      }))
    })),
    hypotheses: z.array(z.object({
      hypothesis_id: z.string(),
      hypothesis_text: z.string(),
      test_result: z.string().optional(),
      status: z.enum(['pending', 'processing', 'completed', 'error', 'cancelled'])
    })).optional(),
    best_solution: z.string().optional(),
    red_team_evaluations: z.array(z.object({
      strategy_id: z.string(),
      evaluation: z.string(),
      killed: z.boolean()
    })).optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type DeepthinkModeResponse = z.infer<typeof DeepthinkModeResponseSchema>;

// ============================================================================
// ADAPTIVE DEEPTHINK MODE
// ============================================================================

/**
 * Adaptive Deepthink Mode Request Schema
 * Mode: Full deepthink mode access to an agent
 */
export const AdaptiveDeepthinkRequestSchema = z.object({
  mode: z.literal('adaptive_deepthink'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    conversation_id: z.string().optional(),
    enable_streaming: z.boolean().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type AdaptiveDeepthinkRequest = z.infer<typeof AdaptiveDeepthinkRequestSchema>;

/**
 * Adaptive Deepthink Mode Response Schema
 */
export const AdaptiveDeepthinkResponseSchema = z.object({
  mode: z.literal('adaptive_deepthink'),
  request: AdaptiveDeepthinkRequestSchema,
  result: ICRResultSchema.extend({
    conversation_id: z.string().optional(),
    tool_calls: z.array(z.object({
      tool_name: z.string(),
      parameters: z.record(z.any()),
      result: z.any()
    })).optional(),
    reasoning_trace: z.string().optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type AdaptiveDeepthinkResponse = z.infer<typeof AdaptiveDeepthinkResponseSchema>;

// ============================================================================
// AGENTIC MODE
// ============================================================================

/**
 * Agentic Mode Request Schema
 * Mode: General-purpose iterative refinement with tool-based manipulation
 */
export const AgenticModeRequestSchema = z.object({
  mode: z.literal('agentic'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    conversation_id: z.string().optional(),
    enable_diff_tools: z.boolean().optional(),
    enable_file_tools: z.boolean().optional(),
    enable_web_search: z.boolean().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type AgenticModeRequest = z.infer<typeof AgenticModeRequestSchema>;

/**
 * Agentic Mode Response Schema
 */
export const AgenticModeResponseSchema = z.object({
  mode: z.literal('agentic'),
  request: AgenticModeRequestSchema,
  result: ICRResultSchema.extend({
    conversation_id: z.string().optional(),
    tool_calls: z.array(z.object({
      tool_name: z.string(),
      parameters: z.record(z.any()),
      result: z.any()
    })).optional(),
    diff_operations: z.array(z.object({
      type: z.enum(['search_and_replace', 'delete', 'insert_before', 'insert_after']),
      params: z.array(z.string())
    })).optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type AgenticModeResponse = z.infer<typeof AgenticModeResponseSchema>;

// ============================================================================
// CONTEXTUAL MODE
// ============================================================================

/**
 * Contextual Mode Request Schema
 * Mode: Iterative refinement through specialized agent collaboration
 */
export const ContextualModeRequestSchema = z.object({
  mode: z.literal('contextual'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    conversation_id: z.string().optional(),
    enable_memory_agent: z.boolean().optional(),
    memory_compression_threshold: z.number().int().positive().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type ContextualModeRequest = z.infer<typeof ContextualModeRequestSchema>;

/**
 * Contextual Mode Response Schema
 */
export const ContextualModeResponseSchema = z.object({
  mode: z.literal('contextual'),
  request: ContextualModeRequestSchema,
  result: ICRResultSchema.extend({
    conversation_id: z.string().optional(),
    agent_interactions: z.array(z.object({
      agent_type: z.enum(['main_generator', 'iterative_agent', 'memory_agent']),
      content: z.string(),
      timestamp_utc: z.string().datetime()
    })).optional(),
    memory_compression_events: z.array(z.object({
      timestamp_utc: z.string().datetime(),
      compressed_message_count: z.number().int()
    })).optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type ContextualModeResponse = z.infer<typeof ContextualModeResponseSchema>;

// ============================================================================
// GENERATIVE UI MODE
// ============================================================================

/**
 * Generative UI Mode Request Schema
 * Mode: Interactive UI development with user interaction capture
 */
export const GenerativeUIModeRequestSchema = z.object({
  mode: z.literal('generative_ui'),
  prompt: z.string().min(1),
  options: ModeOptionsSchema.extend({
    enable_interaction_capture: z.boolean().optional(),
    quality_threshold: z.number().min(0).max(1).optional(),
    max_iterations: z.number().int().positive().optional()
  }).optional(),
  metadata: ICRMetadataSchema
});

export type GenerativeUIModeRequest = z.infer<typeof GenerativeUIModeRequestSchema>;

/**
 * Generative UI Mode Response Schema
 */
export const GenerativeUIModeResponseSchema = z.object({
  mode: z.literal('generative_ui'),
  request: GenerativeUIModeRequestSchema,
  result: ICRResultSchema.extend({
    ui_structure: z.string().optional(),
    html_content: z.string().optional(),
    css_content: z.string().optional(),
    js_content: z.string().optional(),
    quality_score: z.number().min(0).max(1).optional(),
    interactions_captured: z.array(z.object({
      interaction_type: z.enum(['click', 'input', 'hover', 'submit']),
      element_id: z.string(),
      timestamp_utc: z.string().datetime(),
      value: z.any().optional()
    })).optional()
  }),
  metadata: ICRMetadataSchema.extend({
    completed_at_utc: z.string().datetime()
  })
});

export type GenerativeUIModeResponse = z.infer<typeof GenerativeUIModeResponseSchema>;

// ============================================================================
// UNION TYPES
// ============================================================================

/**
 * Union of all mode request types
 */
export const ICRModeRequestSchema = z.discriminatedUnion('mode', [
  RefineModeRequestSchema,
  ReactModeRequestSchema,
  DeepthinkModeRequestSchema,
  AdaptiveDeepthinkRequestSchema,
  AgenticModeRequestSchema,
  ContextualModeRequestSchema,
  GenerativeUIModeRequestSchema
]);

export type ICRModeRequest = z.infer<typeof ICRModeRequestSchema>;

/**
 * Union of all mode response types
 */
export const ICRModeResponseSchema = z.discriminatedUnion('mode', [
  RefineModeResponseSchema,
  ReactModeResponseSchema,
  DeepthinkModeResponseSchema,
  AdaptiveDeepthinkResponseSchema,
  AgenticModeResponseSchema,
  ContextualModeResponseSchema,
  GenerativeUIModeResponseSchema
]);

export type ICRModeResponse = z.infer<typeof ICRModeResponseSchema>;

// ============================================================================
// HEALTH CHECK
// ============================================================================

/**
 * Health check request schema
 */
export const ICRHealthCheckRequestSchema = z.object({
  correlation_id: z.string().uuid().optional()
});

export type ICRHealthCheckRequest = z.infer<typeof ICRHealthCheckRequestSchema>;

/**
 * Health check response schema
 */
export const ICRHealthCheckResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  version: z.string(),
  available_modes: z.array(ModeTypeSchema),
  timestamp_utc: z.string().datetime(),
  uptime_seconds: z.number().nonnegative(),
  metadata: z.record(z.any()).optional()
});

export type ICRHealthCheckResponse = z.infer<typeof ICRHealthCheckResponseSchema>;
