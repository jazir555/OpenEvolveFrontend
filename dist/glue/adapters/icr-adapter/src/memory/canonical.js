"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.SessionOutcomeSchema = exports.LearningResultSchema = exports.SessionMemorySchema = exports.StorageResultSchema = exports.MemoryGraphSchema = exports.MemoryGraphEdgeSchema = exports.MemoryGraphNodeSchema = exports.EnrichedContextSchema = exports.HistoricalKnowledgeSchema = exports.MemoryQuerySchema = exports.PatternRelationshipSchema = exports.ContextualSessionSchema = exports.AgentInteractionSchema = exports.AgentTypeSchema = exports.RefinementInsightsSchema = exports.RefinementMemorySchema = exports.MemoryMetadataSchema = exports.PatternTypeSchema = exports.RefinementOutcomeSchema = void 0;
exports.validateMemorySchema = validateMemorySchema;
const zod_1 = require("zod");
// ============================================================================
// COMMON MEMORY TYPES
// ============================================================================
/**
 * Refinement outcome enumeration
 */
exports.RefinementOutcomeSchema = zod_1.z.enum([
    'success',
    'partial_success',
    'failure',
    'timeout',
    'cancelled'
]);
/**
 * Pattern type enumeration
 */
exports.PatternTypeSchema = zod_1.z.enum([
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
/**
 * Memory metadata included in all memory operations
 */
exports.MemoryMetadataSchema = zod_1.z.object({
    correlation_id: zod_1.z.string().uuid(),
    timestamp_utc: zod_1.z.string().datetime(),
    source_service: zod_1.z.string().default('icr-adapter'),
    session_id: zod_1.z.string().uuid()
});
// ============================================================================
// REFINEMENT MEMORY SCHEMA
// ============================================================================
/**
 * Refinement Memory Schema
 * Captures insights from individual refinement iterations
 */
exports.RefinementMemorySchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    iteration_number: zod_1.z.number().int().positive(),
    refinement_type: exports.PatternTypeSchema,
    prompt: zod_1.z.string().min(1),
    content: zod_1.z.string().min(1),
    outcome: exports.RefinementOutcomeSchema,
    insights: zod_1.z.array(zod_1.z.string()).default([]),
    suggested_features: zod_1.z.string().optional(),
    bug_fixes: zod_1.z.string().optional(),
    quality_metrics: zod_1.z.object({
        novelty_score: zod_1.z.number().min(0).max(1).optional(),
        quality_score: zod_1.z.number().min(0).max(1).optional(),
        improvement_percentage: zod_1.z.number().optional()
    }).optional(),
    execution_time_ms: zod_1.z.number().int().nonnegative(),
    timestamp_utc: zod_1.z.string().datetime(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
/**
 * Batch refinement memories for session-level storage
 */
exports.RefinementInsightsSchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    mode: zod_1.z.enum(['refine', 'contextual', 'agentic', 'deepthink']),
    iterations: zod_1.z.array(exports.RefinementMemorySchema),
    total_iterations: zod_1.z.number().int().nonnegative(),
    successful_iterations: zod_1.z.number().int().nonnegative(),
    failed_iterations: zod_1.z.number().int().nonnegative(),
    total_execution_time_ms: zod_1.z.number().int().nonnegative(),
    average_quality_score: zod_1.z.number().min(0).max(1).optional(),
    overall_outcome: exports.RefinementOutcomeSchema,
    key_patterns_discovered: zod_1.z.array(zod_1.z.string()).default([]),
    lessons_learned: zod_1.z.array(zod_1.z.string()).default([]),
    session_start_utc: zod_1.z.string().datetime(),
    session_end_utc: zod_1.z.string().datetime(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
// ============================================================================
// CONTEXTUAL SESSION SCHEMA
// ============================================================================
/**
 * Agent interaction type
 */
exports.AgentTypeSchema = zod_1.z.enum([
    'main_generator',
    'iterative_agent',
    'memory_agent',
    'quality_agent',
    'custom_agent'
]);
/**
 * Agent interaction record
 */
exports.AgentInteractionSchema = zod_1.z.object({
    agent_type: exports.AgentTypeSchema,
    agent_name: zod_1.z.string().optional(),
    content: zod_1.z.string(),
    timestamp_utc: zod_1.z.string().datetime(),
    execution_time_ms: zod_1.z.number().int().nonnegative().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
/**
 * Contextual Session Schema
 * Captures full context of a contextual mode session
 */
exports.ContextualSessionSchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    mode: zod_1.z.literal('contextual'),
    prompt: zod_1.z.string().min(1),
    agents_involved: zod_1.z.array(exports.AgentTypeSchema),
    interactions: zod_1.z.array(exports.AgentInteractionSchema),
    context_window: zod_1.z.number().int().positive().optional(),
    memory_compression_events: zod_1.z.array(zod_1.z.object({
        timestamp_utc: zod_1.z.string().datetime(),
        compressed_message_count: zod_1.z.number().int().positive(),
        compression_ratio: zod_1.z.number().min(0).max(1),
        bytes_saved: zod_1.z.number().int().nonnegative()
    })).optional(),
    successes: zod_1.z.number().int().nonnegative().default(0),
    failures: zod_1.z.number().int().nonnegative().default(0),
    duration_ms: zod_1.z.number().int().nonnegative(),
    start_time_utc: zod_1.z.string().datetime(),
    end_time_utc: zod_1.z.string().datetime(),
    final_output: zod_1.z.string().optional(),
    quality_score: zod_1.z.number().min(0).max(1).optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
// ============================================================================
// PATTERN RELATIONSHIP SCHEMA
// ============================================================================
/**
 * Pattern Relationship Schema
 * Tracks relationships between refinement patterns across sessions
 */
exports.PatternRelationshipSchema = zod_1.z.object({
    pattern_id: zod_1.z.string().uuid(),
    pattern_type: exports.PatternTypeSchema,
    pattern_name: zod_1.z.string().min(1),
    description: zod_1.z.string().min(1),
    related_sessions: zod_1.z.array(zod_1.z.string().uuid()),
    success_rate: zod_1.z.number().min(0).max(1),
    avg_improvement: zod_1.z.number().optional(),
    avg_execution_time_ms: zod_1.z.number().int().nonnegative(),
    frequency: zod_1.z.number().int().nonnegative(),
    last_seen_utc: zod_1.z.string().datetime(),
    first_seen_utc: zod_1.z.string().datetime(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
// ============================================================================
// MEMORY QUERY SCHEMAS
// ============================================================================
/**
 * Memory Query Schema
 * For querying historical knowledge from Graphiti
 */
exports.MemoryQuerySchema = zod_1.z.object({
    query: zod_1.z.string().min(1),
    session_context: zod_1.z.string().optional(),
    pattern_type: exports.PatternTypeSchema.optional(),
    time_range: zod_1.z.object({
        start_utc: zod_1.z.string().datetime(),
        end_utc: zod_1.z.string().datetime()
    }).optional(),
    min_success_rate: zod_1.z.number().min(0).max(1).optional(),
    max_results: zod_1.z.number().int().positive().default(10),
    include_failed: zod_1.z.boolean().default(false),
    correlation_id: zod_1.z.string().uuid().optional()
});
/**
 * Historical Knowledge Result Schema
 * Returned from memory queries
 */
exports.HistoricalKnowledgeSchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    prompt: zod_1.z.string(),
    pattern_type: exports.PatternTypeSchema,
    outcome: exports.RefinementOutcomeSchema,
    insights: zod_1.z.array(zod_1.z.string()),
    quality_score: zod_1.z.number().min(0).max(1).optional(),
    timestamp_utc: zod_1.z.string().datetime(),
    relevance_score: zod_1.z.number().min(0).max(1),
    applicable_patterns: zod_1.z.array(zod_1.z.string()).default([]),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
/**
 * Enriched Context Schema
 * Returned when retrieving historical context for a request
 */
exports.EnrichedContextSchema = zod_1.z.object({
    query: zod_1.z.string(),
    historical_knowledge: zod_1.z.array(exports.HistoricalKnowledgeSchema),
    related_patterns: zod_1.z.array(exports.PatternRelationshipSchema),
    suggested_approaches: zod_1.z.array(zod_1.z.string()).default([]),
    common_pitfalls: zod_1.z.array(zod_1.z.string()).default([]),
    success_probability: zod_1.z.number().min(0).max(1).optional(),
    confidence_score: zod_1.z.number().min(0).max(1),
    processing_time_ms: zod_1.z.number().int().nonnegative(),
    correlation_id: zod_1.z.string().uuid(),
    timestamp_utc: zod_1.z.string().datetime()
});
// ============================================================================
// MEMORY GRAPH SCHEMA
// ============================================================================
/**
 * Memory Graph Node Schema
 */
exports.MemoryGraphNodeSchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    type: zod_1.z.enum(['session', 'pattern', 'insight', 'agent', 'entity']),
    name: zod_1.z.string().min(1),
    description: zod_1.z.string().optional(),
    attributes: zod_1.z.record(zod_1.z.any()).default({}),
    created_at: zod_1.z.string().datetime(),
    updated_at: zod_1.z.string().datetime().optional()
});
/**
 * Memory Graph Edge Schema
 */
exports.MemoryGraphEdgeSchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    source_id: zod_1.z.string().uuid(),
    target_id: zod_1.z.string().uuid(),
    relationship_type: zod_1.z.string().min(1),
    weight: zod_1.z.number().min(0).max(1).optional(),
    strength: zod_1.z.number().min(0).max(1).optional(),
    attributes: zod_1.z.record(zod_1.z.any()).default({}),
    created_at: zod_1.z.string().datetime()
});
/**
 * Memory Graph Schema
 * Represents the contextual knowledge graph
 */
exports.MemoryGraphSchema = zod_1.z.object({
    nodes: zod_1.z.array(exports.MemoryGraphNodeSchema),
    edges: zod_1.z.array(exports.MemoryGraphEdgeSchema),
    session_count: zod_1.z.number().int().nonnegative(),
    pattern_count: zod_1.z.number().int().nonnegative(),
    last_updated: zod_1.z.string().datetime(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
// ============================================================================
// STORAGE OPERATION SCHEMAS
// ============================================================================
/**
 * Storage Result Schema
 * Returned from memory storage operations
 */
exports.StorageResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    episode_id: zod_1.z.string().uuid().optional(),
    entities_created: zod_1.z.number().int().nonnegative().default(0),
    relationships_created: zod_1.z.number().int().nonnegative().default(0),
    processing_time_ms: zod_1.z.number().int().nonnegative(),
    error: zod_1.z.string().optional(),
    correlation_id: zod_1.z.string().uuid().optional()
});
/**
 * Session Memory Schema
 * Aggregates all memory for a session
 */
exports.SessionMemorySchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    session: exports.ContextualSessionSchema,
    insights: exports.RefinementInsightsSchema.optional(),
    related_patterns: zod_1.z.array(exports.PatternRelationshipSchema).default([]),
    historical_context: zod_1.z.array(exports.HistoricalKnowledgeSchema).default([]),
    memory_graph: exports.MemoryGraphSchema.optional(),
    created_at: zod_1.z.string().datetime(),
    updated_at: zod_1.z.string().datetime().optional()
});
/**
 * Learning Result Schema
 * Returned from learning operations
 */
exports.LearningResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    patterns_learned: zod_1.z.number().int().nonnegative().default(0),
    patterns_updated: zod_1.z.number().int().nonnegative().default(0),
    new_relationships: zod_1.z.number().int().nonnegative().default(0),
    insights_extracted: zod_1.z.number().int().nonnegative().default(0),
    processing_time_ms: zod_1.z.number().int().nonnegative(),
    confidence_score: zod_1.z.number().min(0).max(1).optional(),
    error: zod_1.z.string().optional(),
    correlation_id: zod_1.z.string().uuid().optional()
});
/**
 * Session Outcome Schema
 * Used for learning from completed sessions
 */
exports.SessionOutcomeSchema = zod_1.z.object({
    session_id: zod_1.z.string().uuid(),
    outcome: exports.RefinementOutcomeSchema,
    quality_score: zod_1.z.number().min(0).max(1).optional(),
    user_satisfaction: zod_1.z.number().min(0).max(1).optional(),
    iteration_count: zod_1.z.number().int().nonnegative(),
    success_metrics: zod_1.z.record(zod_1.z.number()).optional(),
    failure_reasons: zod_1.z.array(zod_1.z.string()).default([]),
    successful_patterns: zod_1.z.array(zod_1.z.string()).default([]),
    problematic_patterns: zod_1.z.array(zod_1.z.string()).default([]),
    lessons_learned: zod_1.z.array(zod_1.z.string()).default([]),
    timestamp_utc: zod_1.z.string().datetime()
});
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
function validateMemorySchema(schema, data) {
    const result = schema.safeParse(data);
    if (result.success) {
        return {
            success: true,
            data: result.data,
        };
    }
    const errors = result.error.errors.map((err) => `${err.path.join('.')}: ${err.message}`);
    return {
        success: false,
        errors,
    };
}
//# sourceMappingURL=canonical.js.map