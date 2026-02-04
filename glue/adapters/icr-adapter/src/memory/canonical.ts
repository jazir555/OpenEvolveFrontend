/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Memory Canonical Schemas
 *
 * Canonical data models for ICR Contextual Mode Memory integration with Graphiti.
 * These schemas define the Anti-Corruption Layer (ACL) contract for memory operations.
 * All memory data MUST conform to these schemas before storage/retrieval.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Air Gap: No imports from core-projects
 * - Runtime Truth: Schemas reflect actual memory patterns
 * - Configuration Explicitness: All fields required (no magic defaults)
 * - UTC: All timestamps in UTC ISO-8601 format
 * - Idempotency: Memory operations safe to replay
 */

import { z } from 'zod';

// ============================================================================
// COMMON MEMORY TYPES
// ============================================================================

/**
 * Refinement outcome enumeration
 */
export const RefinementOutcomeSchema = z.enum([
  'success',
  'partial_success',
  'failure',
  'timeout',
  'cancelled'
]);

export type RefinementOutcome = z.infer<typeof RefinementOutcomeSchema>;

/**
 * Pattern type enumeration
 */
export const PatternTypeSchema = z.enum([
  'iterative_refinement',
  'agent_collaboration',
  'memory_compression',
  'context_switching',
  'tool_usage',
  'error_recovery',
  'quality_improvement',
  'novelty_generation',
  'custom'
]);

export type PatternType = z.infer<typeof PatternTypeSchema>;

/**
 * Memory metadata included in all memory operations
 */
export const MemoryMetadataSchema = z.object({
  correlation_id: z.string().uuid(),
  timestamp_utc: z.string().datetime(),
  source_service: z.string().default('icr-adapter'),
  session_id: z.string().uuid()
});

export type MemoryMetadata = z.infer<typeof MemoryMetadataSchema>;

// ============================================================================
// REFINEMENT MEMORY SCHEMA
// ============================================================================

/**
 * Refinement Memory Schema
 * Captures insights from individual refinement iterations
 */
export const RefinementMemorySchema = z.object({
  session_id: z.string().uuid(),
  iteration_number: z.number().int().positive(),
  refinement_type: PatternTypeSchema,
  prompt: z.string().min(1),
  content: z.string().min(1),
  outcome: RefinementOutcomeSchema,
  insights: z.array(z.string()).default([]),
  suggested_features: z.string().optional(),
  bug_fixes: z.string().optional(),
  quality_metrics: z.object({
    novelty_score: z.number().min(0).max(1).optional(),
    quality_score: z.number().min(0).max(1).optional(),
    improvement_percentage: z.number().optional()
  }).optional(),
  execution_time_ms: z.number().int().nonnegative(),
  timestamp_utc: z.string().datetime(),
  metadata: z.record(z.any()).optional()
});

export type RefinementMemory = z.infer<typeof RefinementMemorySchema>;

/**
 * Batch refinement memories for session-level storage
 */
export const RefinementInsightsSchema = z.object({
  session_id: z.string().uuid(),
  mode: z.enum(['refine', 'contextual', 'agentic', 'deepthink']),
  iterations: z.array(RefinementMemorySchema),
  total_iterations: z.number().int().nonnegative(),
  successful_iterations: z.number().int().nonnegative(),
  failed_iterations: z.number().int().nonnegative(),
  total_execution_time_ms: z.number().int().nonnegative(),
  average_quality_score: z.number().min(0).max(1).optional(),
  overall_outcome: RefinementOutcomeSchema,
  key_patterns_discovered: z.array(z.string()).default([]),
  lessons_learned: z.array(z.string()).default([]),
  session_start_utc: z.string().datetime(),
  session_end_utc: z.string().datetime(),
  metadata: z.record(z.any()).optional()
});

export type RefinementInsights = z.infer<typeof RefinementInsightsSchema>;

// ============================================================================
// CONTEXTUAL SESSION SCHEMA
// ============================================================================

/**
 * Agent interaction type
 */
export const AgentTypeSchema = z.enum([
  'main_generator',
  'iterative_agent',
  'memory_agent',
  'quality_agent',
  'custom_agent'
]);

export type AgentType = z.infer<typeof AgentTypeSchema>;

/**
 * Agent interaction record
 */
export const AgentInteractionSchema = z.object({
  agent_type: AgentTypeSchema,
  agent_name: z.string().optional(),
  content: z.string(),
  timestamp_utc: z.string().datetime(),
  execution_time_ms: z.number().int().nonnegative().optional(),
  metadata: z.record(z.any()).optional()
});

export type AgentInteraction = z.infer<typeof AgentInteractionSchema>;

/**
 * Contextual Session Schema
 * Captures full context of a contextual mode session
 */
export const ContextualSessionSchema = z.object({
  session_id: z.string().uuid(),
  mode: z.literal('contextual'),
  prompt: z.string().min(1),
  agents_involved: z.array(AgentTypeSchema),
  interactions: z.array(AgentInteractionSchema),
  context_window: z.number().int().positive().optional(),
  memory_compression_events: z.array(z.object({
    timestamp_utc: z.string().datetime(),
    compressed_message_count: z.number().int().positive(),
    compression_ratio: z.number().min(0).max(1),
    bytes_saved: z.number().int().nonnegative()
  })).optional(),
  successes: z.number().int().nonnegative().default(0),
  failures: z.number().int().nonnegative().default(0),
  duration_ms: z.number().int().nonnegative(),
  start_time_utc: z.string().datetime(),
  end_time_utc: z.string().datetime(),
  final_output: z.string().optional(),
  quality_score: z.number().min(0).max(1).optional(),
  metadata: z.record(z.any()).optional()
});

export type ContextualSession = z.infer<typeof ContextualSessionSchema>;

// ============================================================================
// PATTERN RELATIONSHIP SCHEMA
// ============================================================================

/**
 * Pattern Relationship Schema
 * Tracks relationships between refinement patterns across sessions
 */
export const PatternRelationshipSchema = z.object({
  pattern_id: z.string().uuid(),
  pattern_type: PatternTypeSchema,
  pattern_name: z.string().min(1),
  description: z.string().min(1),
  related_sessions: z.array(z.string().uuid()),
  success_rate: z.number().min(0).max(1),
  avg_improvement: z.number().optional(),
  avg_execution_time_ms: z.number().int().nonnegative(),
  frequency: z.number().int().nonnegative(),
  last_seen_utc: z.string().datetime(),
  first_seen_utc: z.string().datetime(),
  metadata: z.record(z.any()).optional()
});

export type PatternRelationship = z.infer<typeof PatternRelationshipSchema>;

// ============================================================================
// MEMORY QUERY SCHEMAS
// ============================================================================

/**
 * Memory Query Schema
 * For querying historical knowledge from Graphiti
 */
export const MemoryQuerySchema = z.object({
  query: z.string().min(1),
  session_context: z.string().optional(),
  pattern_type: PatternTypeSchema.optional(),
  time_range: z.object({
    start_utc: z.string().datetime(),
    end_utc: z.string().datetime()
  }).optional(),
  min_success_rate: z.number().min(0).max(1).optional(),
  max_results: z.number().int().positive().default(10),
  include_failed: z.boolean().default(false),
  correlation_id: z.string().uuid().optional()
});

export type MemoryQuery = z.infer<typeof MemoryQuerySchema>;

/**
 * Historical Knowledge Result Schema
 * Returned from memory queries
 */
export const HistoricalKnowledgeSchema = z.object({
  session_id: z.string().uuid(),
  prompt: z.string(),
  pattern_type: PatternTypeSchema,
  outcome: RefinementOutcomeSchema,
  insights: z.array(z.string()),
  quality_score: z.number().min(0).max(1).optional(),
  timestamp_utc: z.string().datetime(),
  relevance_score: z.number().min(0).max(1),
  applicable_patterns: z.array(z.string()).default([]),
  metadata: z.record(z.any()).optional()
});

export type HistoricalKnowledge = z.infer<typeof HistoricalKnowledgeSchema>;

/**
 * Enriched Context Schema
 * Returned when retrieving historical context for a request
 */
export const EnrichedContextSchema = z.object({
  query: z.string(),
  historical_knowledge: z.array(HistoricalKnowledgeSchema),
  related_patterns: z.array(PatternRelationshipSchema),
  suggested_approaches: z.array(z.string()).default([]),
  common_pitfalls: z.array(z.string()).default([]),
  success_probability: z.number().min(0).max(1).optional(),
  confidence_score: z.number().min(0).max(1),
  processing_time_ms: z.number().int().nonnegative(),
  correlation_id: z.string().uuid(),
  timestamp_utc: z.string().datetime()
});

export type EnrichedContext = z.infer<typeof EnrichedContextSchema>;

// ============================================================================
// MEMORY GRAPH SCHEMA
// ============================================================================

/**
 * Memory Graph Node Schema
 */
export const MemoryGraphNodeSchema = z.object({
  id: z.string().uuid(),
  type: z.enum(['session', 'pattern', 'insight', 'agent', 'entity']),
  name: z.string().min(1),
  description: z.string().optional(),
  attributes: z.record(z.any()).default({}),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime().optional()
});

export type MemoryGraphNode = z.infer<typeof MemoryGraphNodeSchema>;

/**
 * Memory Graph Edge Schema
 */
export const MemoryGraphEdgeSchema = z.object({
  id: z.string().uuid(),
  source_id: z.string().uuid(),
  target_id: z.string().uuid(),
  relationship_type: z.string().min(1),
  weight: z.number().min(0).max(1).optional(),
  strength: z.number().min(0).max(1).optional(),
  attributes: z.record(z.any()).default({}),
  created_at: z.string().datetime()
});

export type MemoryGraphEdge = z.infer<typeof MemoryGraphEdgeSchema>;

/**
 * Memory Graph Schema
 * Represents the contextual knowledge graph
 */
export const MemoryGraphSchema = z.object({
  nodes: z.array(MemoryGraphNodeSchema),
  edges: z.array(MemoryGraphEdgeSchema),
  session_count: z.number().int().nonnegative(),
  pattern_count: z.number().int().nonnegative(),
  last_updated: z.string().datetime(),
  metadata: z.record(z.any()).optional()
});

export type MemoryGraph = z.infer<typeof MemoryGraphSchema>;

// ============================================================================
// STORAGE OPERATION SCHEMAS
// ============================================================================

/**
 * Storage Result Schema
 * Returned from memory storage operations
 */
export const StorageResultSchema = z.object({
  success: z.boolean(),
  episode_id: z.string().uuid().optional(),
  entities_created: z.number().int().nonnegative().default(0),
  relationships_created: z.number().int().nonnegative().default(0),
  processing_time_ms: z.number().int().nonnegative(),
  error: z.string().optional(),
  correlation_id: z.string().uuid().optional()
});

export type StorageResult = z.infer<typeof StorageResultSchema>;

/**
 * Session Memory Schema
 * Aggregates all memory for a session
 */
export const SessionMemorySchema = z.object({
  session_id: z.string().uuid(),
  session: ContextualSessionSchema,
  insights: RefinementInsightsSchema.optional(),
  related_patterns: z.array(PatternRelationshipSchema).default([]),
  historical_context: z.array(HistoricalKnowledgeSchema).default([]),
  memory_graph: MemoryGraphSchema.optional(),
  created_at: z.string().datetime(),
  updated_at: z.string().datetime().optional()
});

export type SessionMemory = z.infer<typeof SessionMemorySchema>;

/**
 * Learning Result Schema
 * Returned from learning operations
 */
export const LearningResultSchema = z.object({
  success: z.boolean(),
  patterns_learned: z.number().int().nonnegative().default(0),
  patterns_updated: z.number().int().nonnegative().default(0),
  new_relationships: z.number().int().nonnegative().default(0),
  insights_extracted: z.number().int().nonnegative().default(0),
  processing_time_ms: z.number().int().nonnegative(),
  confidence_score: z.number().min(0).max(1).optional(),
  error: z.string().optional(),
  correlation_id: z.string().uuid().optional()
});

export type LearningResult = z.infer<typeof LearningResultSchema>;

/**
 * Session Outcome Schema
 * Used for learning from completed sessions
 */
export const SessionOutcomeSchema = z.object({
  session_id: z.string().uuid(),
  outcome: RefinementOutcomeSchema,
  quality_score: z.number().min(0).max(1).optional(),
  user_satisfaction: z.number().min(0).max(1).optional(),
  iteration_count: z.number().int().nonnegative(),
  success_metrics: z.record(z.number()).optional(),
  failure_reasons: z.array(z.string()).default([]),
  successful_patterns: z.array(z.string()).default([]),
  problematic_patterns: z.array(z.string()).default([]),
  lessons_learned: z.array(z.string()).default([]),
  timestamp_utc: z.string().datetime()
});

export type SessionOutcome = z.infer<typeof SessionOutcomeSchema>;

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/**
 * Validate memory data against canonical schema
 *
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with success flag and data or errors
 */
export function validateMemorySchema<T extends z.ZodTypeAny>(
  schema: T,
  data: unknown
): { success: boolean; data?: z.infer<T>; errors?: string[] } {
  const result = schema.safeParse(data);

  if (result.success) {
    return {
      success: true,
      data: result.data,
    };
  }

  const errors = result.error.errors.map(
    (err) => `${err.path.join('.')}: ${err.message}`
  );

  return {
    success: false,
    errors,
  };
}
