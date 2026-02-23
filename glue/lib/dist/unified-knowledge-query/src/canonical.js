"use strict";
/**
 * Canonical Schemas for Unified Knowledge Query Interface
 *
 * Federation Constitution Compliance:
 * - Anti-Corruption Layer: All data normalized to canonical format
 * - Type Safety: Zod schemas for runtime validation
 * - UTC Timestamps: All temporal data in ISO-8601 UTC
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.isValidResult = exports.isValidQuery = exports.validateResult = exports.validateQuery = exports.QueryOptionsSchema = exports.EngineMetricsSchema = exports.SystemHealthSchema = exports.HealthStatusSchema = exports.SystemConfigSchema = exports.CostEstimateSchema = exports.QueryPlanSchema = exports.UnifiedQueryResultSchema = exports.ConflictReportSchema = exports.SourceMetadataSchema = exports.RelationshipSchema = exports.EntitySchema = exports.KnowledgeItemSchema = exports.SystemSourceSchema = exports.UnifiedKnowledgeQuerySchema = exports.QueryTypeSchema = exports.TemporalFilterSchema = exports.KnowledgeTypeSchema = exports.KnowledgeDomainSchema = void 0;
const zod_1 = require("zod");
/**
 * Knowledge Domain Selection
 * Determines which systems to query
 */
exports.KnowledgeDomainSchema = zod_1.z.enum([
    'ragbits',
    'graphiti',
    'vectordb',
    'all'
]);
/**
 * Knowledge Type Filter
 * Filters results by knowledge representation type
 */
exports.KnowledgeTypeSchema = zod_1.z.enum([
    'document',
    'entity',
    'proof',
    'code',
    'relationship',
    'all'
]);
/**
 * Temporal Filter for time-based queries
 * All timestamps in UTC ISO-8601 format (Law of UTC)
 */
exports.TemporalFilterSchema = zod_1.z.object({
    startDate: zod_1.z.string().datetime().optional(),
    endDate: zod_1.z.string().datetime().optional(),
    pointInTime: zod_1.z.string().datetime().optional(),
});
/**
 * Query Type Strategy
 */
exports.QueryTypeSchema = zod_1.z.enum([
    'semantic-search',
    'temporal-query',
    'graph-traversal',
    'hybrid',
    'fallback'
]);
/**
 * Main Unified Knowledge Query Input Schema
 */
exports.UnifiedKnowledgeQuerySchema = zod_1.z.object({
    // Core query text
    query: zod_1.z.string().min(1),
    // Which systems to query
    domains: zod_1.z.array(exports.KnowledgeDomainSchema).min(1),
    // Query type strategy
    queryType: exports.QueryTypeSchema.optional().default('hybrid'),
    // Temporal constraints
    temporalFilter: exports.TemporalFilterSchema.optional(),
    // Knowledge type filters
    knowledgeTypes: zod_1.z.array(exports.KnowledgeTypeSchema).optional().default(['all']),
    // Result limits
    maxResults: zod_1.z.number().int().positive().max(1000).optional().default(50),
    // Confidence threshold (0.0 to 1.0)
    minConfidence: zod_1.z.number().min(0).max(1).optional().default(0.0),
    // Graph traversal depth
    maxDepth: zod_1.z.number().int().positive().max(10).optional().default(2),
    // Include metadata
    includeMetadata: zod_1.z.boolean().optional().default(true),
    // Correlation ID for tracing
    correlationId: zod_1.z.string().uuid().optional(),
});
/**
 * System Source Identification
 */
exports.SystemSourceSchema = zod_1.z.enum([
    'ragbits',
    'graphiti',
    'vectordb',
    'fused'
]);
/**
 * Individual Knowledge Result Item
 */
exports.KnowledgeItemSchema = zod_1.z.object({
    // Content
    content: zod_1.z.string(),
    // Source system
    source: exports.SystemSourceSchema,
    // Unique identifier
    id: zod_1.z.string(),
    // Knowledge type
    type: exports.KnowledgeTypeSchema,
    // Confidence score (0.0 to 1.0)
    confidence: zod_1.z.number().min(0).max(1),
    // Relevance score (0.0 to 1.0)
    relevance: zod_1.z.number().min(0).max(1),
    // Timestamp (UTC ISO-8601)
    timestamp: zod_1.z.string().datetime(),
    // Metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Entity from Graphiti
 */
exports.EntitySchema = zod_1.z.object({
    id: zod_1.z.string(),
    name: zod_1.z.string(),
    type: zod_1.z.string().optional(),
    description: zod_1.z.string().optional(),
    createdAt: zod_1.z.string().datetime(),
    updatedAt: zod_1.z.string().datetime(),
});
/**
 * Relationship from Graphiti
 */
exports.RelationshipSchema = zod_1.z.object({
    id: zod_1.z.string(),
    source: zod_1.z.string(),
    target: zod_1.z.string(),
    relation: zod_1.z.string(),
    weight: zod_1.z.number().optional(),
    createdAt: zod_1.z.string().datetime(),
    updatedAt: zod_1.z.string().datetime(),
});
/**
 * Source System Metadata
 */
exports.SourceMetadataSchema = zod_1.z.object({
    system: exports.SystemSourceSchema,
    queryTimeMs: zod_1.z.number(),
    resultCount: zod_1.z.number(),
    success: zod_1.z.boolean(),
    error: zod_1.z.string().optional(),
});
/**
 * Conflict Detection Report
 */
exports.ConflictReportSchema = zod_1.z.object({
    hasConflicts: zod_1.z.boolean(),
    conflicts: zod_1.z.array(zod_1.z.object({
        field: zod_1.z.string(),
        sources: zod_1.z.array(exports.SystemSourceSchema),
        values: zod_1.z.array(zod_1.z.any()),
        resolution: zod_1.z.string().optional(),
    })),
});
/**
 * Unified Query Result Schema
 */
exports.UnifiedQueryResultSchema = zod_1.z.object({
    // Query that was executed
    query: zod_1.z.string(),
    // Merged results from all systems
    results: zod_1.z.array(exports.KnowledgeItemSchema),
    // Sources queried
    sources: zod_1.z.array(exports.SourceMetadataSchema),
    // Overall confidence score
    confidence: zod_1.z.number().min(0).max(1),
    // Total execution time
    executionTimeMs: zod_1.z.number(),
    // Conflict report
    conflicts: exports.ConflictReportSchema.optional(),
    // Metadata
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
    // Correlation ID
    correlationId: zod_1.z.string().uuid(),
});
/**
 * Query Plan for Execution
 */
exports.QueryPlanSchema = zod_1.z.object({
    query: exports.UnifiedKnowledgeQuerySchema,
    strategy: exports.QueryTypeSchema,
    systems: zod_1.z.array(exports.SystemSourceSchema),
    estimatedCost: zod_1.z.number(),
    parallelizable: zod_1.z.boolean(),
});
/**
 * Cost Estimation
 */
exports.CostEstimateSchema = zod_1.z.object({
    timeMs: zod_1.z.number(),
    complexity: zod_1.z.enum(['low', 'medium', 'high']),
    resources: zod_1.z.array(zod_1.z.string()),
});
/**
 * System Configuration
 */
exports.SystemConfigSchema = zod_1.z.object({
    name: exports.SystemSourceSchema,
    enabled: zod_1.z.boolean(),
    url: zod_1.z.string().url(),
    timeout: zod_1.z.number().int().positive(),
    priority: zod_1.z.number().int().min(1).max(10),
});
/**
 * Health Status
 */
exports.HealthStatusSchema = zod_1.z.enum([
    'healthy',
    'degraded',
    'unhealthy',
    'unknown'
]);
/**
 * System Health Check Result
 */
exports.SystemHealthSchema = zod_1.z.object({
    system: exports.SystemSourceSchema,
    status: exports.HealthStatusSchema,
    responseTimeMs: zod_1.z.number().optional(),
    lastCheck: zod_1.z.string().datetime(),
    error: zod_1.z.string().optional(),
});
/**
 * Engine Metrics
 */
exports.EngineMetricsSchema = zod_1.z.object({
    totalQueries: zod_1.z.number(),
    successfulQueries: zod_1.z.number(),
    failedQueries: zod_1.z.number(),
    averageQueryTime: zod_1.z.number(),
    systemHealth: zod_1.z.array(exports.SystemHealthSchema),
    uptime: zod_1.z.number(),
});
/**
 * Query Options (simplified version for external API)
 */
exports.QueryOptionsSchema = zod_1.z.object({
    domains: zod_1.z.array(exports.KnowledgeDomainSchema).optional().default(['all']),
    knowledgeTypes: zod_1.z.array(exports.KnowledgeTypeSchema).optional().default(['all']),
    maxResults: zod_1.z.number().int().positive().max(1000).optional().default(50),
    minConfidence: zod_1.z.number().min(0).max(1).optional().default(0.0),
    temporalFilter: exports.TemporalFilterSchema.optional(),
    queryType: exports.QueryTypeSchema.optional().default('hybrid'),
    maxDepth: zod_1.z.number().int().positive().max(10).optional().default(2),
});
/**
 * Validation helpers
 */
const validateQuery = (query) => {
    return exports.UnifiedKnowledgeQuerySchema.parse(query);
};
exports.validateQuery = validateQuery;
const validateResult = (result) => {
    return exports.UnifiedQueryResultSchema.parse(result);
};
exports.validateResult = validateResult;
/**
 * Type guards
 */
const isValidQuery = (query) => {
    return exports.UnifiedKnowledgeQuerySchema.safeParse(query).success;
};
exports.isValidQuery = isValidQuery;
const isValidResult = (result) => {
    return exports.UnifiedQueryResultSchema.safeParse(result).success;
};
exports.isValidResult = isValidResult;
//# sourceMappingURL=canonical.js.map