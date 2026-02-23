"use strict";
/**
 * Graphiti Canonical Schema
 *
 * Canonical data models for Graphiti temporal knowledge graph integration.
 * Follows the Federation Constitution - Anti-Corruption Layer pattern.
 *
 * This schema defines the contract between the Graphiti adapter and the
 * rest of the OpenEvolve system, normalizing Graphiti's native format
 * into our canonical representation.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.GraphStatisticsSchema = exports.AddTripletResultSchema = exports.AddTripletOperationSchema = exports.AddEpisodeResultSchema = exports.AddEpisodeOperationSchema = exports.CanonicalSearchResultSchema = exports.CanonicalSearchQuerySchema = exports.TemporalFilterEnum = exports.CanonicalCommunitySchema = exports.CanonicalEpisodeSchema = exports.EpisodeTypeEnum = exports.CanonicalEntityEdgeSchema = exports.CanonicalEntitySchema = void 0;
exports.validateCanonical = validateCanonical;
const zod_1 = require("zod");
// ============================================================================
// ENTITY SCHEMAS
// ============================================================================
/**
 * Canonical Entity Node
 * Represents an entity in the knowledge graph (person, organization, concept, etc.)
 */
exports.CanonicalEntitySchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    name: zod_1.z.string().min(1),
    labels: zod_1.z.array(zod_1.z.string()).default([]),
    summary: zod_1.z.string().optional(),
    attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    created_at: zod_1.z.string().datetime(), // UTC ISO-8601
    updated_at: zod_1.z.string().datetime().optional(), // UTC ISO-8601
    metadata: zod_1.z.record(zod_1.z.unknown()).optional(),
});
/**
 * Canonical Entity Edge (Relationship)
 * Represents a relationship between two entities
 */
exports.CanonicalEntityEdgeSchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    source_entity_id: zod_1.z.string().uuid(),
    target_entity_id: zod_1.z.string().uuid(),
    relation_type: zod_1.z.string().min(1),
    fact: zod_1.z.string().min(1),
    summary: zod_1.z.string().optional(),
    attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    created_at: zod_1.z.string().datetime(), // UTC ISO-8601
    updated_at: zod_1.z.string().datetime().optional(), // UTC ISO-8601
    episodes: zod_1.z.array(zod_1.z.string().uuid()).default([]),
    metadata: zod_1.z.record(zod_1.z.unknown()).optional(),
});
// ============================================================================
// EPISODE SCHEMAS
// ============================================================================
/**
 * Episode Type Enumeration
 * Types of episodic data that can be added to the graph
 */
exports.EpisodeTypeEnum = zod_1.z.enum([
    'text',
    'message',
    'document',
    'code',
    'transaction',
    'event',
    'observation',
    'custom',
]);
/**
 * Canonical Episode
 * Represents an episode/event in the temporal knowledge graph
 */
exports.CanonicalEpisodeSchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    name: zod_1.z.string().min(1),
    content: zod_1.z.string().min(1),
    source_description: zod_1.z.string().optional(),
    episode_type: exports.EpisodeTypeEnum,
    valid_at: zod_1.z.string().datetime(), // When the event occurred (UTC)
    created_at: zod_1.z.string().datetime(), // When added to graph (UTC)
    group_id: zod_1.z.string().optional(),
    entity_edges: zod_1.z.array(zod_1.z.string().uuid()).default([]),
    metadata: zod_1.z.record(zod_1.z.unknown()).optional(),
});
// ============================================================================
// COMMUNITY SCHEMAS
// ============================================================================
/**
 * Canonical Community
 * Represents a cluster of related entities
 */
exports.CanonicalCommunitySchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    summary: zod_1.z.string().min(1),
    member_count: zod_1.z.number().int().nonnegative(),
    member_ids: zod_1.z.array(zod_1.z.string().uuid()),
    attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    created_at: zod_1.z.string().datetime(), // UTC ISO-8601
    updated_at: zod_1.z.string().datetime().optional(), // UTC ISO-8601
    metadata: zod_1.z.record(zod_1.z.unknown()).optional(),
});
// ============================================================================
// SEARCH SCHEMAS
// ============================================================================
/**
 * Temporal Filter Type
 */
exports.TemporalFilterEnum = zod_1.z.enum([
    'current',
    'time_range',
    'point_in_time',
    'all',
]);
/**
 * Canonical Search Query
 */
exports.CanonicalSearchQuerySchema = zod_1.z.object({
    query: zod_1.z.string().min(1),
    temporal_filter: exports.TemporalFilterEnum.default('current'),
    start_time: zod_1.z.string().datetime().optional(),
    end_time: zod_1.z.string().datetime().optional(),
    max_results: zod_1.z.number().int().positive().default(10),
    group_ids: zod_1.z.array(zod_1.z.string()).optional(),
    center_node_uuid: zod_1.z.string().uuid().optional(),
});
/**
 * Canonical Search Result
 */
exports.CanonicalSearchResultSchema = zod_1.z.object({
    edges: zod_1.z.array(exports.CanonicalEntityEdgeSchema),
    nodes: zod_1.z.array(exports.CanonicalEntitySchema),
    total_count: zod_1.z.number().int().nonnegative(),
    query_time_ms: zod_1.z.number().nonnegative(),
    metadata: zod_1.z.record(zod_1.z.unknown()).optional(),
});
// ============================================================================
// OPERATION SCHEMAS
// ============================================================================
/**
 * Add Episode Operation
 */
exports.AddEpisodeOperationSchema = zod_1.z.object({
    name: zod_1.z.string().min(1),
    content: zod_1.z.string().min(1),
    source_description: zod_1.z.string().optional(),
    episode_type: exports.EpisodeTypeEnum.default('text'),
    valid_at: zod_1.z.string().datetime(),
    group_id: zod_1.z.string().optional(),
    uuid: zod_1.z.string().uuid().optional(),
    entity_types: zod_1.z.record(zod_1.z.string()).optional(),
    excluded_entity_types: zod_1.z.array(zod_1.z.string()).optional(),
    update_communities: zod_1.z.boolean().default(false),
});
/**
 * Add Episode Result
 */
exports.AddEpisodeResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    episode_id: zod_1.z.string().uuid(),
    entities_extracted: zod_1.z.number().int().nonnegative(),
    relationships_extracted: zod_1.z.number().int().nonnegative(),
    communities_updated: zod_1.z.number().int().nonnegative().default(0),
    processing_time_ms: zod_1.z.number().nonnegative(),
    correlation_id: zod_1.z.string().optional(),
    error: zod_1.z.string().optional(),
});
/**
 * Add Triplet Operation (Subject -> Predicate -> Object)
 */
exports.AddTripletOperationSchema = zod_1.z.object({
    subject: zod_1.z.object({
        name: zod_1.z.string().min(1),
        labels: zod_1.z.array(zod_1.z.string()).default([]),
        summary: zod_1.z.string().optional(),
        attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    }),
    predicate: zod_1.z.object({
        relation_type: zod_1.z.string().min(1),
        fact: zod_1.z.string().min(1),
        attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    }),
    object: zod_1.z.object({
        name: zod_1.z.string().min(1),
        labels: zod_1.z.array(zod_1.z.string()).default([]),
        summary: zod_1.z.string().optional(),
        attributes: zod_1.z.record(zod_1.z.unknown()).default({}),
    }),
    group_id: zod_1.z.string().optional(),
    valid_at: zod_1.z.string().datetime().optional(),
});
/**
 * Add Triplet Result
 */
exports.AddTripletResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    subject_uuid: zod_1.z.string().uuid().optional(),
    object_uuid: zod_1.z.string().uuid().optional(),
    edge_uuid: zod_1.z.string().uuid().optional(),
    processing_time_ms: zod_1.z.number().nonnegative(),
    correlation_id: zod_1.z.string().optional(),
    error: zod_1.z.string().optional(),
});
// ============================================================================
// GRAPH STATISTICS SCHEMA
// ============================================================================
/**
 * Graph Statistics
 */
exports.GraphStatisticsSchema = zod_1.z.object({
    entities_count: zod_1.z.number().int().nonnegative(),
    relationships_count: zod_1.z.number().int().nonnegative(),
    episodes_count: zod_1.z.number().int().nonnegative(),
    communities_count: zod_1.z.number().int().nonnegative().default(0),
    initialized: zod_1.z.boolean(),
    connection_status: zod_1.z.enum(['connected', 'disconnected', 'error']),
    last_update: zod_1.z.string().datetime().optional(),
});
// ============================================================================
// VALIDATION HELPER
// ============================================================================
/**
 * Validate data against canonical schema
 *
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with success flag and data or errors
 */
function validateCanonical(schema, data) {
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
/**
 * Example usage:
 *
 * ```typescript
 * import { validateCanonical, CanonicalEntitySchema } from './graphiti-canonical';
 *
 * const entityData = {
 *   id: '550e8400-e29b-41d4-a716-446655440000',
 *   name: 'John Doe',
 *   labels: ['Person', 'Employee'],
 *   summary: 'Software engineer at TechCorp',
 *   attributes: { department: 'Engineering' },
 *   created_at: '2024-01-15T10:30:00.000Z',
 * };
 *
 * const result = validateCanonical(CanonicalEntitySchema, entityData);
 * if (result.success) {
 *   console.log('Valid entity:', result.data);
 * } else {
 *   console.error('Validation errors:', result.errors);
 * }
 * ```
 */
//# sourceMappingURL=graphiti-canonical.js.map