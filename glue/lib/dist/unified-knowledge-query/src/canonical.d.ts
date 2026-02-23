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
export declare const KnowledgeDomainSchema: z.ZodEnum<["ragbits", "graphiti", "vectordb", "all"]>;
export type KnowledgeDomain = z.infer<typeof KnowledgeDomainSchema>;
/**
 * Knowledge Type Filter
 * Filters results by knowledge representation type
 */
export declare const KnowledgeTypeSchema: z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>;
export type KnowledgeType = z.infer<typeof KnowledgeTypeSchema>;
/**
 * Temporal Filter for time-based queries
 * All timestamps in UTC ISO-8601 format (Law of UTC)
 */
export declare const TemporalFilterSchema: z.ZodObject<{
    startDate: z.ZodOptional<z.ZodString>;
    endDate: z.ZodOptional<z.ZodString>;
    pointInTime: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    startDate?: string | undefined;
    endDate?: string | undefined;
    pointInTime?: string | undefined;
}, {
    startDate?: string | undefined;
    endDate?: string | undefined;
    pointInTime?: string | undefined;
}>;
export type TemporalFilter = z.infer<typeof TemporalFilterSchema>;
/**
 * Query Type Strategy
 */
export declare const QueryTypeSchema: z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>;
export type QueryType = z.infer<typeof QueryTypeSchema>;
/**
 * Main Unified Knowledge Query Input Schema
 */
export declare const UnifiedKnowledgeQuerySchema: z.ZodObject<{
    query: z.ZodString;
    domains: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "all"]>, "many">;
    queryType: z.ZodDefault<z.ZodOptional<z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>>>;
    temporalFilter: z.ZodOptional<z.ZodObject<{
        startDate: z.ZodOptional<z.ZodString>;
        endDate: z.ZodOptional<z.ZodString>;
        pointInTime: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    }, {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    }>>;
    knowledgeTypes: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>, "many">>>;
    maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    minConfidence: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    maxDepth: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    includeMetadata: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    correlationId: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    query: string;
    domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
    queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    knowledgeTypes: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[];
    maxResults: number;
    minConfidence: number;
    maxDepth: number;
    includeMetadata: boolean;
    correlationId?: string | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
}, {
    query: string;
    domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
    correlationId?: string | undefined;
    queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
    knowledgeTypes?: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[] | undefined;
    maxResults?: number | undefined;
    minConfidence?: number | undefined;
    maxDepth?: number | undefined;
    includeMetadata?: boolean | undefined;
}>;
export type UnifiedKnowledgeQuery = z.infer<typeof UnifiedKnowledgeQuerySchema>;
/**
 * System Source Identification
 */
export declare const SystemSourceSchema: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
export type SystemSource = z.infer<typeof SystemSourceSchema>;
/**
 * Individual Knowledge Result Item
 */
export declare const KnowledgeItemSchema: z.ZodObject<{
    content: z.ZodString;
    source: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
    id: z.ZodString;
    type: z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>;
    confidence: z.ZodNumber;
    relevance: z.ZodNumber;
    timestamp: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    source: "graphiti" | "vectordb" | "ragbits" | "fused";
    id: string;
    type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
    confidence: number;
    content: string;
    relevance: number;
    metadata?: Record<string, any> | undefined;
}, {
    timestamp: string;
    source: "graphiti" | "vectordb" | "ragbits" | "fused";
    id: string;
    type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
    confidence: number;
    content: string;
    relevance: number;
    metadata?: Record<string, any> | undefined;
}>;
export type KnowledgeItem = z.infer<typeof KnowledgeItemSchema>;
/**
 * Entity from Graphiti
 */
export declare const EntitySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    type: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: string;
    createdAt: string;
    updatedAt: string;
    description?: string | undefined;
    type?: string | undefined;
}, {
    name: string;
    id: string;
    createdAt: string;
    updatedAt: string;
    description?: string | undefined;
    type?: string | undefined;
}>;
export type Entity = z.infer<typeof EntitySchema>;
/**
 * Relationship from Graphiti
 */
export declare const RelationshipSchema: z.ZodObject<{
    id: z.ZodString;
    source: z.ZodString;
    target: z.ZodString;
    relation: z.ZodString;
    weight: z.ZodOptional<z.ZodNumber>;
    createdAt: z.ZodString;
    updatedAt: z.ZodString;
}, "strip", z.ZodTypeAny, {
    source: string;
    id: string;
    target: string;
    createdAt: string;
    updatedAt: string;
    relation: string;
    weight?: number | undefined;
}, {
    source: string;
    id: string;
    target: string;
    createdAt: string;
    updatedAt: string;
    relation: string;
    weight?: number | undefined;
}>;
export type Relationship = z.infer<typeof RelationshipSchema>;
/**
 * Source System Metadata
 */
export declare const SourceMetadataSchema: z.ZodObject<{
    system: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
    queryTimeMs: z.ZodNumber;
    resultCount: z.ZodNumber;
    success: z.ZodBoolean;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    system: "graphiti" | "vectordb" | "ragbits" | "fused";
    queryTimeMs: number;
    resultCount: number;
    error?: string | undefined;
}, {
    success: boolean;
    system: "graphiti" | "vectordb" | "ragbits" | "fused";
    queryTimeMs: number;
    resultCount: number;
    error?: string | undefined;
}>;
export type SourceMetadata = z.infer<typeof SourceMetadataSchema>;
/**
 * Conflict Detection Report
 */
export declare const ConflictReportSchema: z.ZodObject<{
    hasConflicts: z.ZodBoolean;
    conflicts: z.ZodArray<z.ZodObject<{
        field: z.ZodString;
        sources: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>, "many">;
        values: z.ZodArray<z.ZodAny, "many">;
        resolution: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        values: any[];
        field: string;
        sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
        resolution?: string | undefined;
    }, {
        values: any[];
        field: string;
        sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
        resolution?: string | undefined;
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    hasConflicts: boolean;
    conflicts: {
        values: any[];
        field: string;
        sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
        resolution?: string | undefined;
    }[];
}, {
    hasConflicts: boolean;
    conflicts: {
        values: any[];
        field: string;
        sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
        resolution?: string | undefined;
    }[];
}>;
export type ConflictReport = z.infer<typeof ConflictReportSchema>;
/**
 * Unified Query Result Schema
 */
export declare const UnifiedQueryResultSchema: z.ZodObject<{
    query: z.ZodString;
    results: z.ZodArray<z.ZodObject<{
        content: z.ZodString;
        source: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
        id: z.ZodString;
        type: z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>;
        confidence: z.ZodNumber;
        relevance: z.ZodNumber;
        timestamp: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        source: "graphiti" | "vectordb" | "ragbits" | "fused";
        id: string;
        type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
        confidence: number;
        content: string;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }, {
        timestamp: string;
        source: "graphiti" | "vectordb" | "ragbits" | "fused";
        id: string;
        type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
        confidence: number;
        content: string;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }>, "many">;
    sources: z.ZodArray<z.ZodObject<{
        system: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
        queryTimeMs: z.ZodNumber;
        resultCount: z.ZodNumber;
        success: z.ZodBoolean;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }, {
        success: boolean;
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }>, "many">;
    confidence: z.ZodNumber;
    executionTimeMs: z.ZodNumber;
    conflicts: z.ZodOptional<z.ZodObject<{
        hasConflicts: z.ZodBoolean;
        conflicts: z.ZodArray<z.ZodObject<{
            field: z.ZodString;
            sources: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>, "many">;
            values: z.ZodArray<z.ZodAny, "many">;
            resolution: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }, {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        hasConflicts: boolean;
        conflicts: {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }[];
    }, {
        hasConflicts: boolean;
        conflicts: {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }[];
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlationId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    correlationId: string;
    query: string;
    confidence: number;
    sources: {
        success: boolean;
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }[];
    results: {
        timestamp: string;
        source: "graphiti" | "vectordb" | "ragbits" | "fused";
        id: string;
        type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
        confidence: number;
        content: string;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }[];
    executionTimeMs: number;
    metadata?: Record<string, any> | undefined;
    conflicts?: {
        hasConflicts: boolean;
        conflicts: {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }[];
    } | undefined;
}, {
    correlationId: string;
    query: string;
    confidence: number;
    sources: {
        success: boolean;
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }[];
    results: {
        timestamp: string;
        source: "graphiti" | "vectordb" | "ragbits" | "fused";
        id: string;
        type: "all" | "code" | "proof" | "relationship" | "document" | "entity";
        confidence: number;
        content: string;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }[];
    executionTimeMs: number;
    metadata?: Record<string, any> | undefined;
    conflicts?: {
        hasConflicts: boolean;
        conflicts: {
            values: any[];
            field: string;
            sources: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
            resolution?: string | undefined;
        }[];
    } | undefined;
}>;
export type UnifiedQueryResult = z.infer<typeof UnifiedQueryResultSchema>;
/**
 * Query Plan for Execution
 */
export declare const QueryPlanSchema: z.ZodObject<{
    query: z.ZodObject<{
        query: z.ZodString;
        domains: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "all"]>, "many">;
        queryType: z.ZodDefault<z.ZodOptional<z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>>>;
        temporalFilter: z.ZodOptional<z.ZodObject<{
            startDate: z.ZodOptional<z.ZodString>;
            endDate: z.ZodOptional<z.ZodString>;
            pointInTime: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        }, {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        }>>;
        knowledgeTypes: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>, "many">>>;
        maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        minConfidence: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        maxDepth: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
        includeMetadata: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
        correlationId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        query: string;
        domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
        queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
        knowledgeTypes: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[];
        maxResults: number;
        minConfidence: number;
        maxDepth: number;
        includeMetadata: boolean;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
    }, {
        query: string;
        domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
        correlationId?: string | undefined;
        queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
        knowledgeTypes?: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[] | undefined;
        maxResults?: number | undefined;
        minConfidence?: number | undefined;
        maxDepth?: number | undefined;
        includeMetadata?: boolean | undefined;
    }>;
    strategy: z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>;
    systems: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>, "many">;
    estimatedCost: z.ZodNumber;
    parallelizable: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    query: {
        query: string;
        domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
        queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
        knowledgeTypes: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[];
        maxResults: number;
        minConfidence: number;
        maxDepth: number;
        includeMetadata: boolean;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
    };
    strategy: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    systems: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
    estimatedCost: number;
    parallelizable: boolean;
}, {
    query: {
        query: string;
        domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
        correlationId?: string | undefined;
        queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
        knowledgeTypes?: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[] | undefined;
        maxResults?: number | undefined;
        minConfidence?: number | undefined;
        maxDepth?: number | undefined;
        includeMetadata?: boolean | undefined;
    };
    strategy: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    systems: ("graphiti" | "vectordb" | "ragbits" | "fused")[];
    estimatedCost: number;
    parallelizable: boolean;
}>;
export type QueryPlan = z.infer<typeof QueryPlanSchema>;
/**
 * Cost Estimation
 */
export declare const CostEstimateSchema: z.ZodObject<{
    timeMs: z.ZodNumber;
    complexity: z.ZodEnum<["low", "medium", "high"]>;
    resources: z.ZodArray<z.ZodString, "many">;
}, "strip", z.ZodTypeAny, {
    timeMs: number;
    complexity: "medium" | "low" | "high";
    resources: string[];
}, {
    timeMs: number;
    complexity: "medium" | "low" | "high";
    resources: string[];
}>;
export type CostEstimate = z.infer<typeof CostEstimateSchema>;
/**
 * System Configuration
 */
export declare const SystemConfigSchema: z.ZodObject<{
    name: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
    enabled: z.ZodBoolean;
    url: z.ZodString;
    timeout: z.ZodNumber;
    priority: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    name: "graphiti" | "vectordb" | "ragbits" | "fused";
    url: string;
    enabled: boolean;
    timeout: number;
    priority: number;
}, {
    name: "graphiti" | "vectordb" | "ragbits" | "fused";
    url: string;
    enabled: boolean;
    timeout: number;
    priority: number;
}>;
export type SystemConfig = z.infer<typeof SystemConfigSchema>;
/**
 * Health Status
 */
export declare const HealthStatusSchema: z.ZodEnum<["healthy", "degraded", "unhealthy", "unknown"]>;
export type HealthStatus = z.infer<typeof HealthStatusSchema>;
/**
 * System Health Check Result
 */
export declare const SystemHealthSchema: z.ZodObject<{
    system: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
    status: z.ZodEnum<["healthy", "degraded", "unhealthy", "unknown"]>;
    responseTimeMs: z.ZodOptional<z.ZodNumber>;
    lastCheck: z.ZodString;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "unknown" | "unhealthy" | "degraded" | "healthy";
    system: "graphiti" | "vectordb" | "ragbits" | "fused";
    lastCheck: string;
    error?: string | undefined;
    responseTimeMs?: number | undefined;
}, {
    status: "unknown" | "unhealthy" | "degraded" | "healthy";
    system: "graphiti" | "vectordb" | "ragbits" | "fused";
    lastCheck: string;
    error?: string | undefined;
    responseTimeMs?: number | undefined;
}>;
export type SystemHealth = z.infer<typeof SystemHealthSchema>;
/**
 * Engine Metrics
 */
export declare const EngineMetricsSchema: z.ZodObject<{
    totalQueries: z.ZodNumber;
    successfulQueries: z.ZodNumber;
    failedQueries: z.ZodNumber;
    averageQueryTime: z.ZodNumber;
    systemHealth: z.ZodArray<z.ZodObject<{
        system: z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>;
        status: z.ZodEnum<["healthy", "degraded", "unhealthy", "unknown"]>;
        responseTimeMs: z.ZodOptional<z.ZodNumber>;
        lastCheck: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        status: "unknown" | "unhealthy" | "degraded" | "healthy";
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        lastCheck: string;
        error?: string | undefined;
        responseTimeMs?: number | undefined;
    }, {
        status: "unknown" | "unhealthy" | "degraded" | "healthy";
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        lastCheck: string;
        error?: string | undefined;
        responseTimeMs?: number | undefined;
    }>, "many">;
    uptime: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    uptime: number;
    totalQueries: number;
    successfulQueries: number;
    failedQueries: number;
    averageQueryTime: number;
    systemHealth: {
        status: "unknown" | "unhealthy" | "degraded" | "healthy";
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        lastCheck: string;
        error?: string | undefined;
        responseTimeMs?: number | undefined;
    }[];
}, {
    uptime: number;
    totalQueries: number;
    successfulQueries: number;
    failedQueries: number;
    averageQueryTime: number;
    systemHealth: {
        status: "unknown" | "unhealthy" | "degraded" | "healthy";
        system: "graphiti" | "vectordb" | "ragbits" | "fused";
        lastCheck: string;
        error?: string | undefined;
        responseTimeMs?: number | undefined;
    }[];
}>;
export type EngineMetrics = z.infer<typeof EngineMetricsSchema>;
/**
 * Query Options (simplified version for external API)
 */
export declare const QueryOptionsSchema: z.ZodObject<{
    domains: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "all"]>, "many">>>;
    knowledgeTypes: z.ZodDefault<z.ZodOptional<z.ZodArray<z.ZodEnum<["document", "entity", "proof", "code", "relationship", "all"]>, "many">>>;
    maxResults: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    minConfidence: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    temporalFilter: z.ZodOptional<z.ZodObject<{
        startDate: z.ZodOptional<z.ZodString>;
        endDate: z.ZodOptional<z.ZodString>;
        pointInTime: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    }, {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    }>>;
    queryType: z.ZodDefault<z.ZodOptional<z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>>>;
    maxDepth: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
}, "strip", z.ZodTypeAny, {
    domains: ("graphiti" | "vectordb" | "all" | "ragbits")[];
    queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    knowledgeTypes: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[];
    maxResults: number;
    minConfidence: number;
    maxDepth: number;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
}, {
    domains?: ("graphiti" | "vectordb" | "all" | "ragbits")[] | undefined;
    queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
    knowledgeTypes?: ("all" | "code" | "proof" | "relationship" | "document" | "entity")[] | undefined;
    maxResults?: number | undefined;
    minConfidence?: number | undefined;
    maxDepth?: number | undefined;
}>;
export type QueryOptions = z.infer<typeof QueryOptionsSchema>;
/**
 * Validation helpers
 */
export declare const validateQuery: (query: unknown) => UnifiedKnowledgeQuery;
export declare const validateResult: (result: unknown) => UnifiedQueryResult;
/**
 * Type guards
 */
export declare const isValidQuery: (query: unknown) => query is UnifiedKnowledgeQuery;
export declare const isValidResult: (result: unknown) => result is UnifiedQueryResult;
//# sourceMappingURL=canonical.d.ts.map