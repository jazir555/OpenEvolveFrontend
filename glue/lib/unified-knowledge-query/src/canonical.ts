/**
 * Canonical Schemas for Unified Knowledge Query Interface
 *
 * Federation Constitution Compliance:
 * - Anti-Corruption Layer: All data normalized to canonical format
 * - Type Safety: Zod schemas for runtime validation
 * - UTC Timestamps: All temporal data in ISO-8601 UTC
 */

import { z } from 'zod';

/**
 * Knowledge Domain Selection
 * Determines which systems to query
 */
export const KnowledgeDomainSchema = z.enum([
  'ragbits',
  'graphiti',
  'vectordb',
  'all'
]);
export type KnowledgeDomain = z.infer<typeof KnowledgeDomainSchema>;

/**
 * Knowledge Type Filter
 * Filters results by knowledge representation type
 */
export const KnowledgeTypeSchema = z.enum([
  'document',
  'entity',
  'proof',
  'code',
  'relationship',
  'all'
]);
export type KnowledgeType = z.infer<typeof KnowledgeTypeSchema>;

/**
 * Temporal Filter for time-based queries
 * All timestamps in UTC ISO-8601 format (Law of UTC)
 */
export const TemporalFilterSchema = z.object({
  startDate: z.string().datetime().optional(),
  endDate: z.string().datetime().optional(),
  pointInTime: z.string().datetime().optional(),
});
export type TemporalFilter = z.infer<typeof TemporalFilterSchema>;

/**
 * Query Type Strategy
 */
export const QueryTypeSchema = z.enum([
  'semantic-search',
  'temporal-query',
  'graph-traversal',
  'hybrid',
  'fallback'
]);
export type QueryType = z.infer<typeof QueryTypeSchema>;

/**
 * Main Unified Knowledge Query Input Schema
 */
export const UnifiedKnowledgeQuerySchema = z.object({
  // Core query text
  query: z.string().min(1),

  // Which systems to query
  domains: z.array(KnowledgeDomainSchema).min(1),

  // Query type strategy
  queryType: QueryTypeSchema.optional().default('hybrid'),

  // Temporal constraints
  temporalFilter: TemporalFilterSchema.optional(),

  // Knowledge type filters
  knowledgeTypes: z.array(KnowledgeTypeSchema).optional().default(['all']),

  // Result limits
  maxResults: z.number().int().positive().max(1000).optional().default(50),

  // Confidence threshold (0.0 to 1.0)
  minConfidence: z.number().min(0).max(1).optional().default(0.0),

  // Graph traversal depth
  maxDepth: z.number().int().positive().max(10).optional().default(2),

  // Include metadata
  includeMetadata: z.boolean().optional().default(true),

  // Correlation ID for tracing
  correlationId: z.string().uuid().optional(),
});
export type UnifiedKnowledgeQuery = z.infer<typeof UnifiedKnowledgeQuerySchema>;

/**
 * System Source Identification
 */
export const SystemSourceSchema = z.enum([
  'ragbits',
  'graphiti',
  'vectordb',
  'fused'
]);
export type SystemSource = z.infer<typeof SystemSourceSchema>;

/**
 * Individual Knowledge Result Item
 */
export const KnowledgeItemSchema = z.object({
  // Content
  content: z.string(),

  // Source system
  source: SystemSourceSchema,

  // Unique identifier
  id: z.string(),

  // Knowledge type
  type: KnowledgeTypeSchema,

  // Confidence score (0.0 to 1.0)
  confidence: z.number().min(0).max(1),

  // Relevance score (0.0 to 1.0)
  relevance: z.number().min(0).max(1),

  // Timestamp (UTC ISO-8601)
  timestamp: z.string().datetime(),

  // Metadata
  metadata: z.record(z.any()).optional(),
});
export type KnowledgeItem = z.infer<typeof KnowledgeItemSchema>;

/**
 * Entity from Graphiti
 */
export const EntitySchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.string().optional(),
  description: z.string().optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type Entity = z.infer<typeof EntitySchema>;

/**
 * Relationship from Graphiti
 */
export const RelationshipSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  relation: z.string(),
  weight: z.number().optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
export type Relationship = z.infer<typeof RelationshipSchema>;

/**
 * Source System Metadata
 */
export const SourceMetadataSchema = z.object({
  system: SystemSourceSchema,
  queryTimeMs: z.number(),
  resultCount: z.number(),
  success: z.boolean(),
  error: z.string().optional(),
});
export type SourceMetadata = z.infer<typeof SourceMetadataSchema>;

/**
 * Conflict Detection Report
 */
export const ConflictReportSchema = z.object({
  hasConflicts: z.boolean(),
  conflicts: z.array(z.object({
    field: z.string(),
    sources: z.array(SystemSourceSchema),
    values: z.array(z.any()),
    resolution: z.string().optional(),
  })),
});
export type ConflictReport = z.infer<typeof ConflictReportSchema>;

/**
 * Unified Query Result Schema
 */
export const UnifiedQueryResultSchema = z.object({
  // Query that was executed
  query: z.string(),

  // Merged results from all systems
  results: z.array(KnowledgeItemSchema),

  // Sources queried
  sources: z.array(SourceMetadataSchema),

  // Overall confidence score
  confidence: z.number().min(0).max(1),

  // Total execution time
  executionTimeMs: z.number(),

  // Conflict report
  conflicts: ConflictReportSchema.optional(),

  // Metadata
  metadata: z.record(z.any()).optional(),

  // Correlation ID
  correlationId: z.string().uuid(),
});
export type UnifiedQueryResult = z.infer<typeof UnifiedQueryResultSchema>;

/**
 * Query Plan for Execution
 */
export const QueryPlanSchema = z.object({
  query: UnifiedKnowledgeQuerySchema,
  strategy: QueryTypeSchema,
  systems: z.array(SystemSourceSchema),
  estimatedCost: z.number(),
  parallelizable: z.boolean(),
});
export type QueryPlan = z.infer<typeof QueryPlanSchema>;

/**
 * Cost Estimation
 */
export const CostEstimateSchema = z.object({
  timeMs: z.number(),
  complexity: z.enum(['low', 'medium', 'high']),
  resources: z.array(z.string()),
});
export type CostEstimate = z.infer<typeof CostEstimateSchema>;

/**
 * System Configuration
 */
export const SystemConfigSchema = z.object({
  name: SystemSourceSchema,
  enabled: z.boolean(),
  url: z.string().url(),
  timeout: z.number().int().positive(),
  priority: z.number().int().min(1).max(10),
});
export type SystemConfig = z.infer<typeof SystemConfigSchema>;

/**
 * Health Status
 */
export const HealthStatusSchema = z.enum([
  'healthy',
  'degraded',
  'unhealthy',
  'unknown'
]);
export type HealthStatus = z.infer<typeof HealthStatusSchema>;

/**
 * System Health Check Result
 */
export const SystemHealthSchema = z.object({
  system: SystemSourceSchema,
  status: HealthStatusSchema,
  responseTimeMs: z.number().optional(),
  lastCheck: z.string().datetime(),
  error: z.string().optional(),
});
export type SystemHealth = z.infer<typeof SystemHealthSchema>;

/**
 * Engine Metrics
 */
export const EngineMetricsSchema = z.object({
  totalQueries: z.number(),
  successfulQueries: z.number(),
  failedQueries: z.number(),
  averageQueryTime: z.number(),
  systemHealth: z.array(SystemHealthSchema),
  uptime: z.number(),
});
export type EngineMetrics = z.infer<typeof EngineMetricsSchema>;

/**
 * Query Options (simplified version for external API)
 */
export const QueryOptionsSchema = z.object({
  domains: z.array(KnowledgeDomainSchema).optional().default(['all']),
  knowledgeTypes: z.array(KnowledgeTypeSchema).optional().default(['all']),
  maxResults: z.number().int().positive().max(1000).optional().default(50),
  minConfidence: z.number().min(0).max(1).optional().default(0.0),
  temporalFilter: TemporalFilterSchema.optional(),
  queryType: QueryTypeSchema.optional().default('hybrid'),
  maxDepth: z.number().int().positive().max(10).optional().default(2),
});
export type QueryOptions = z.infer<typeof QueryOptionsSchema>;

/**
 * Validation helpers
 */
export const validateQuery = (query: unknown): UnifiedKnowledgeQuery => {
  return UnifiedKnowledgeQuerySchema.parse(query);
};

export const validateResult = (result: unknown): UnifiedQueryResult => {
  return UnifiedQueryResultSchema.parse(result);
};

/**
 * Type guards
 */
export const isValidQuery = (query: unknown): query is UnifiedKnowledgeQuery => {
  return UnifiedKnowledgeQuerySchema.safeParse(query).success;
};

export const isValidResult = (result: unknown): result is UnifiedQueryResult => {
  return UnifiedQueryResultSchema.safeParse(result).success;
};
