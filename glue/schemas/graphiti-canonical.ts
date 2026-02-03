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

import { z } from 'zod';

// ============================================================================
// ENTITY SCHEMAS
// ============================================================================

/**
 * Canonical Entity Node
 * Represents an entity in the knowledge graph (person, organization, concept, etc.)
 */
export const CanonicalEntitySchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  labels: z.array(z.string()).default([]),
  summary: z.string().optional(),
  attributes: z.record(z.unknown()).default({}),
  created_at: z.string().datetime(), // UTC ISO-8601
  updated_at: z.string().datetime().optional(), // UTC ISO-8601
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalEntity = z.infer<typeof CanonicalEntitySchema>;

/**
 * Canonical Entity Edge (Relationship)
 * Represents a relationship between two entities
 */
export const CanonicalEntityEdgeSchema = z.object({
  id: z.string().uuid(),
  source_entity_id: z.string().uuid(),
  target_entity_id: z.string().uuid(),
  relation_type: z.string().min(1),
  fact: z.string().min(1),
  summary: z.string().optional(),
  attributes: z.record(z.unknown()).default({}),
  created_at: z.string().datetime(), // UTC ISO-8601
  updated_at: z.string().datetime().optional(), // UTC ISO-8601
  episodes: z.array(z.string().uuid()).default([]),
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalEntityEdge = z.infer<typeof CanonicalEntityEdgeSchema>;

// ============================================================================
// EPISODE SCHEMAS
// ============================================================================

/**
 * Episode Type Enumeration
 * Types of episodic data that can be added to the graph
 */
export const EpisodeTypeEnum = z.enum([
  'text',
  'message',
  'document',
  'code',
  'transaction',
  'event',
  'observation',
  'custom',
]);

export type EpisodeType = z.infer<typeof EpisodeTypeEnum>;

/**
 * Canonical Episode
 * Represents an episode/event in the temporal knowledge graph
 */
export const CanonicalEpisodeSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  content: z.string().min(1),
  source_description: z.string().optional(),
  episode_type: EpisodeTypeEnum,
  valid_at: z.string().datetime(), // When the event occurred (UTC)
  created_at: z.string().datetime(), // When added to graph (UTC)
  group_id: z.string().optional(),
  entity_edges: z.array(z.string().uuid()).default([]),
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalEpisode = z.infer<typeof CanonicalEpisodeSchema>;

// ============================================================================
// COMMUNITY SCHEMAS
// ============================================================================

/**
 * Canonical Community
 * Represents a cluster of related entities
 */
export const CanonicalCommunitySchema = z.object({
  id: z.string().uuid(),
  summary: z.string().min(1),
  member_count: z.number().int().nonnegative(),
  member_ids: z.array(z.string().uuid()),
  attributes: z.record(z.unknown()).default({}),
  created_at: z.string().datetime(), // UTC ISO-8601
  updated_at: z.string().datetime().optional(), // UTC ISO-8601
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalCommunity = z.infer<typeof CanonicalCommunitySchema>;

// ============================================================================
// SEARCH SCHEMAS
// ============================================================================

/**
 * Temporal Filter Type
 */
export const TemporalFilterEnum = z.enum([
  'current',
  'time_range',
  'point_in_time',
  'all',
]);

export type TemporalFilter = z.infer<typeof TemporalFilterEnum>;

/**
 * Canonical Search Query
 */
export const CanonicalSearchQuerySchema = z.object({
  query: z.string().min(1),
  temporal_filter: TemporalFilterEnum.default('current'),
  start_time: z.string().datetime().optional(),
  end_time: z.string().datetime().optional(),
  max_results: z.number().int().positive().default(10),
  group_ids: z.array(z.string()).optional(),
  center_node_uuid: z.string().uuid().optional(),
});

export type CanonicalSearchQuery = z.infer<typeof CanonicalSearchQuerySchema>;

/**
 * Canonical Search Result
 */
export const CanonicalSearchResultSchema = z.object({
  edges: z.array(CanonicalEntityEdgeSchema),
  nodes: z.array(CanonicalEntitySchema),
  total_count: z.number().int().nonnegative(),
  query_time_ms: z.number().nonnegative(),
  metadata: z.record(z.unknown()).optional(),
});

export type CanonicalSearchResult = z.infer<typeof CanonicalSearchResultSchema>;

// ============================================================================
// OPERATION SCHEMAS
// ============================================================================

/**
 * Add Episode Operation
 */
export const AddEpisodeOperationSchema = z.object({
  name: z.string().min(1),
  content: z.string().min(1),
  source_description: z.string().optional(),
  episode_type: EpisodeTypeEnum.default('text'),
  valid_at: z.string().datetime(),
  group_id: z.string().optional(),
  uuid: z.string().uuid().optional(),
  entity_types: z.record(z.string()).optional(),
  excluded_entity_types: z.array(z.string()).optional(),
  update_communities: z.boolean().default(false),
});

export type AddEpisodeOperation = z.infer<typeof AddEpisodeOperationSchema>;

/**
 * Add Episode Result
 */
export const AddEpisodeResultSchema = z.object({
  success: z.boolean(),
  episode_id: z.string().uuid(),
  entities_extracted: z.number().int().nonnegative(),
  relationships_extracted: z.number().int().nonnegative(),
  communities_updated: z.number().int().nonnegative().default(0),
  processing_time_ms: z.number().nonnegative(),
  correlation_id: z.string().optional(),
  error: z.string().optional(),
});

export type AddEpisodeResult = z.infer<typeof AddEpisodeResultSchema>;

/**
 * Add Triplet Operation (Subject -> Predicate -> Object)
 */
export const AddTripletOperationSchema = z.object({
  subject: z.object({
    name: z.string().min(1),
    labels: z.array(z.string()).default([]),
    summary: z.string().optional(),
    attributes: z.record(z.unknown()).default({}),
  }),
  predicate: z.object({
    relation_type: z.string().min(1),
    fact: z.string().min(1),
    attributes: z.record(z.unknown()).default({}),
  }),
  object: z.object({
    name: z.string().min(1),
    labels: z.array(z.string()).default([]),
    summary: z.string().optional(),
    attributes: z.record(z.unknown()).default({}),
  }),
  group_id: z.string().optional(),
  valid_at: z.string().datetime().optional(),
});

export type AddTripletOperation = z.infer<typeof AddTripletOperationSchema>;

/**
 * Add Triplet Result
 */
export const AddTripletResultSchema = z.object({
  success: z.boolean(),
  subject_uuid: z.string().uuid().optional(),
  object_uuid: z.string().uuid().optional(),
  edge_uuid: z.string().uuid().optional(),
  processing_time_ms: z.number().nonnegative(),
  correlation_id: z.string().optional(),
  error: z.string().optional(),
});

export type AddTripletResult = z.infer<typeof AddTripletResultSchema>;

// ============================================================================
// GRAPH STATISTICS SCHEMA
// ============================================================================

/**
 * Graph Statistics
 */
export const GraphStatisticsSchema = z.object({
  entities_count: z.number().int().nonnegative(),
  relationships_count: z.number().int().nonnegative(),
  episodes_count: z.number().int().nonnegative(),
  communities_count: z.number().int().nonnegative().default(0),
  initialized: z.boolean(),
  connection_status: z.enum(['connected', 'disconnected', 'error']),
  last_update: z.string().datetime().optional(),
});

export type GraphStatistics = z.infer<typeof GraphStatisticsSchema>;

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
export function validateCanonical<T extends z.ZodTypeAny>(
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
