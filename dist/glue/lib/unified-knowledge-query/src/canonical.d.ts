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
    queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    maxDepth: number;
    domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
    knowledgeTypes: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[];
    maxResults: number;
    minConfidence: number;
    includeMetadata: boolean;
    correlationId?: string | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
}, {
    query: string;
    domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
    queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
    maxDepth?: number | undefined;
    correlationId?: string | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
    knowledgeTypes?: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[] | undefined;
    maxResults?: number | undefined;
    minConfidence?: number | undefined;
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
    type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
    id: string;
    content: string;
    source: "ragbits" | "graphiti" | "vectordb" | "fused";
    confidence: number;
    relevance: number;
    metadata?: Record<string, any> | undefined;
}, {
    timestamp: string;
    type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
    id: string;
    content: string;
    source: "ragbits" | "graphiti" | "vectordb" | "fused";
    confidence: number;
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
    type?: string | undefined;
    description?: string | undefined;
}, {
    name: string;
    id: string;
    createdAt: string;
    updatedAt: string;
    type?: string | undefined;
    description?: string | undefined;
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
    id: string;
    source: string;
    createdAt: string;
    updatedAt: string;
    target: string;
    relation: string;
    weight?: number | undefined;
}, {
    id: string;
    source: string;
    createdAt: string;
    updatedAt: string;
    target: string;
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
    system: "ragbits" | "graphiti" | "vectordb" | "fused";
    queryTimeMs: number;
    resultCount: number;
    error?: string | undefined;
}, {
    success: boolean;
    system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
        sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
        resolution?: string | undefined;
    }, {
        values: any[];
        field: string;
        sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
        resolution?: string | undefined;
    }>, "many">;
}, "strip", z.ZodTypeAny, {
    conflicts: {
        values: any[];
        field: string;
        sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
        resolution?: string | undefined;
    }[];
    hasConflicts: boolean;
}, {
    conflicts: {
        values: any[];
        field: string;
        sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
        resolution?: string | undefined;
    }[];
    hasConflicts: boolean;
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
        type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
        id: string;
        content: string;
        source: "ragbits" | "graphiti" | "vectordb" | "fused";
        confidence: number;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }, {
        timestamp: string;
        type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
        id: string;
        content: string;
        source: "ragbits" | "graphiti" | "vectordb" | "fused";
        confidence: number;
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
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }, {
        success: boolean;
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }, {
            values: any[];
            field: string;
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        conflicts: {
            values: any[];
            field: string;
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }[];
        hasConflicts: boolean;
    }, {
        conflicts: {
            values: any[];
            field: string;
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }[];
        hasConflicts: boolean;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlationId: z.ZodString;
}, "strip", z.ZodTypeAny, {
    query: string;
    results: {
        timestamp: string;
        type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
        id: string;
        content: string;
        source: "ragbits" | "graphiti" | "vectordb" | "fused";
        confidence: number;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }[];
    confidence: number;
    sources: {
        success: boolean;
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }[];
    correlationId: string;
    executionTimeMs: number;
    metadata?: Record<string, any> | undefined;
    conflicts?: {
        conflicts: {
            values: any[];
            field: string;
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }[];
        hasConflicts: boolean;
    } | undefined;
}, {
    query: string;
    results: {
        timestamp: string;
        type: "proof" | "all" | "code" | "document" | "entity" | "relationship";
        id: string;
        content: string;
        source: "ragbits" | "graphiti" | "vectordb" | "fused";
        confidence: number;
        relevance: number;
        metadata?: Record<string, any> | undefined;
    }[];
    confidence: number;
    sources: {
        success: boolean;
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
        queryTimeMs: number;
        resultCount: number;
        error?: string | undefined;
    }[];
    correlationId: string;
    executionTimeMs: number;
    metadata?: Record<string, any> | undefined;
    conflicts?: {
        conflicts: {
            values: any[];
            field: string;
            sources: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
            resolution?: string | undefined;
        }[];
        hasConflicts: boolean;
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
        queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
        maxDepth: number;
        domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
        knowledgeTypes: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[];
        maxResults: number;
        minConfidence: number;
        includeMetadata: boolean;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
    }, {
        query: string;
        domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
        queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
        maxDepth?: number | undefined;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
        knowledgeTypes?: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[] | undefined;
        maxResults?: number | undefined;
        minConfidence?: number | undefined;
        includeMetadata?: boolean | undefined;
    }>;
    strategy: z.ZodEnum<["semantic-search", "temporal-query", "graph-traversal", "hybrid", "fallback"]>;
    systems: z.ZodArray<z.ZodEnum<["ragbits", "graphiti", "vectordb", "fused"]>, "many">;
    estimatedCost: z.ZodNumber;
    parallelizable: z.ZodBoolean;
}, "strip", z.ZodTypeAny, {
    query: {
        query: string;
        queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
        maxDepth: number;
        domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
        knowledgeTypes: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[];
        maxResults: number;
        minConfidence: number;
        includeMetadata: boolean;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
    };
    strategy: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    systems: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
    estimatedCost: number;
    parallelizable: boolean;
}, {
    query: {
        query: string;
        domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
        queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
        maxDepth?: number | undefined;
        correlationId?: string | undefined;
        temporalFilter?: {
            startDate?: string | undefined;
            endDate?: string | undefined;
            pointInTime?: string | undefined;
        } | undefined;
        knowledgeTypes?: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[] | undefined;
        maxResults?: number | undefined;
        minConfidence?: number | undefined;
        includeMetadata?: boolean | undefined;
    };
    strategy: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    systems: ("ragbits" | "graphiti" | "vectordb" | "fused")[];
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
    complexity: "high" | "medium" | "low";
    timeMs: number;
    resources: string[];
}, {
    complexity: "high" | "medium" | "low";
    timeMs: number;
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
    enabled: boolean;
    name: "ragbits" | "graphiti" | "vectordb" | "fused";
    url: string;
    timeout: number;
    priority: number;
}, {
    enabled: boolean;
    name: "ragbits" | "graphiti" | "vectordb" | "fused";
    url: string;
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
    status: "unknown" | "healthy" | "unhealthy" | "degraded";
    system: "ragbits" | "graphiti" | "vectordb" | "fused";
    lastCheck: string;
    error?: string | undefined;
    responseTimeMs?: number | undefined;
}, {
    status: "unknown" | "healthy" | "unhealthy" | "degraded";
    system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
        status: "unknown" | "healthy" | "unhealthy" | "degraded";
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
        lastCheck: string;
        error?: string | undefined;
        responseTimeMs?: number | undefined;
    }, {
        status: "unknown" | "healthy" | "unhealthy" | "degraded";
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
        status: "unknown" | "healthy" | "unhealthy" | "degraded";
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
        status: "unknown" | "healthy" | "unhealthy" | "degraded";
        system: "ragbits" | "graphiti" | "vectordb" | "fused";
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
    queryType: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback";
    maxDepth: number;
    domains: ("ragbits" | "graphiti" | "all" | "vectordb")[];
    knowledgeTypes: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[];
    maxResults: number;
    minConfidence: number;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
}, {
    queryType?: "hybrid" | "semantic-search" | "temporal-query" | "graph-traversal" | "fallback" | undefined;
    maxDepth?: number | undefined;
    domains?: ("ragbits" | "graphiti" | "all" | "vectordb")[] | undefined;
    temporalFilter?: {
        startDate?: string | undefined;
        endDate?: string | undefined;
        pointInTime?: string | undefined;
    } | undefined;
    knowledgeTypes?: ("proof" | "all" | "code" | "document" | "entity" | "relationship")[] | undefined;
    maxResults?: number | undefined;
    minConfidence?: number | undefined;
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