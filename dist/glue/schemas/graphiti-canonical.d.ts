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
/**
 * Canonical Entity Node
 * Represents an entity in the knowledge graph (person, organization, concept, etc.)
 */
export declare const CanonicalEntitySchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    labels: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    summary: z.ZodOptional<z.ZodString>;
    attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: string;
    created_at: string;
    labels: string[];
    attributes: Record<string, unknown>;
    metadata?: Record<string, unknown> | undefined;
    summary?: string | undefined;
    updated_at?: string | undefined;
}, {
    name: string;
    id: string;
    created_at: string;
    metadata?: Record<string, unknown> | undefined;
    summary?: string | undefined;
    updated_at?: string | undefined;
    labels?: string[] | undefined;
    attributes?: Record<string, unknown> | undefined;
}>;
export type CanonicalEntity = z.infer<typeof CanonicalEntitySchema>;
/**
 * Canonical Entity Edge (Relationship)
 * Represents a relationship between two entities
 */
export declare const CanonicalEntityEdgeSchema: z.ZodObject<{
    id: z.ZodString;
    source_entity_id: z.ZodString;
    target_entity_id: z.ZodString;
    relation_type: z.ZodString;
    fact: z.ZodString;
    summary: z.ZodOptional<z.ZodString>;
    attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
    episodes: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    id: string;
    created_at: string;
    attributes: Record<string, unknown>;
    relation_type: string;
    source_entity_id: string;
    target_entity_id: string;
    fact: string;
    episodes: string[];
    metadata?: Record<string, unknown> | undefined;
    summary?: string | undefined;
    updated_at?: string | undefined;
}, {
    id: string;
    created_at: string;
    relation_type: string;
    source_entity_id: string;
    target_entity_id: string;
    fact: string;
    metadata?: Record<string, unknown> | undefined;
    summary?: string | undefined;
    updated_at?: string | undefined;
    attributes?: Record<string, unknown> | undefined;
    episodes?: string[] | undefined;
}>;
export type CanonicalEntityEdge = z.infer<typeof CanonicalEntityEdgeSchema>;
/**
 * Episode Type Enumeration
 * Types of episodic data that can be added to the graph
 */
export declare const EpisodeTypeEnum: z.ZodEnum<["text", "message", "document", "code", "transaction", "event", "observation", "custom"]>;
export type EpisodeType = z.infer<typeof EpisodeTypeEnum>;
/**
 * Canonical Episode
 * Represents an episode/event in the temporal knowledge graph
 */
export declare const CanonicalEpisodeSchema: z.ZodObject<{
    id: z.ZodString;
    name: z.ZodString;
    content: z.ZodString;
    source_description: z.ZodOptional<z.ZodString>;
    episode_type: z.ZodEnum<["text", "message", "document", "code", "transaction", "event", "observation", "custom"]>;
    valid_at: z.ZodString;
    created_at: z.ZodString;
    group_id: z.ZodOptional<z.ZodString>;
    entity_edges: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    id: string;
    content: string;
    created_at: string;
    valid_at: string;
    episode_type: "custom" | "text" | "message" | "event" | "code" | "document" | "transaction" | "observation";
    entity_edges: string[];
    metadata?: Record<string, unknown> | undefined;
    source_description?: string | undefined;
    group_id?: string | undefined;
}, {
    name: string;
    id: string;
    content: string;
    created_at: string;
    valid_at: string;
    episode_type: "custom" | "text" | "message" | "event" | "code" | "document" | "transaction" | "observation";
    metadata?: Record<string, unknown> | undefined;
    source_description?: string | undefined;
    group_id?: string | undefined;
    entity_edges?: string[] | undefined;
}>;
export type CanonicalEpisode = z.infer<typeof CanonicalEpisodeSchema>;
/**
 * Canonical Community
 * Represents a cluster of related entities
 */
export declare const CanonicalCommunitySchema: z.ZodObject<{
    id: z.ZodString;
    summary: z.ZodString;
    member_count: z.ZodNumber;
    member_ids: z.ZodArray<z.ZodString, "many">;
    attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    created_at: z.ZodString;
    updated_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    id: string;
    summary: string;
    created_at: string;
    attributes: Record<string, unknown>;
    member_count: number;
    member_ids: string[];
    metadata?: Record<string, unknown> | undefined;
    updated_at?: string | undefined;
}, {
    id: string;
    summary: string;
    created_at: string;
    member_count: number;
    member_ids: string[];
    metadata?: Record<string, unknown> | undefined;
    updated_at?: string | undefined;
    attributes?: Record<string, unknown> | undefined;
}>;
export type CanonicalCommunity = z.infer<typeof CanonicalCommunitySchema>;
/**
 * Temporal Filter Type
 */
export declare const TemporalFilterEnum: z.ZodEnum<["current", "time_range", "point_in_time", "all"]>;
export type TemporalFilter = z.infer<typeof TemporalFilterEnum>;
/**
 * Canonical Search Query
 */
export declare const CanonicalSearchQuerySchema: z.ZodObject<{
    query: z.ZodString;
    temporal_filter: z.ZodDefault<z.ZodEnum<["current", "time_range", "point_in_time", "all"]>>;
    start_time: z.ZodOptional<z.ZodString>;
    end_time: z.ZodOptional<z.ZodString>;
    max_results: z.ZodDefault<z.ZodNumber>;
    group_ids: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    center_node_uuid: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    query: string;
    max_results: number;
    temporal_filter: "all" | "current" | "point_in_time" | "time_range";
    start_time?: string | undefined;
    end_time?: string | undefined;
    group_ids?: string[] | undefined;
    center_node_uuid?: string | undefined;
}, {
    query: string;
    start_time?: string | undefined;
    end_time?: string | undefined;
    max_results?: number | undefined;
    temporal_filter?: "all" | "current" | "point_in_time" | "time_range" | undefined;
    group_ids?: string[] | undefined;
    center_node_uuid?: string | undefined;
}>;
export type CanonicalSearchQuery = z.infer<typeof CanonicalSearchQuerySchema>;
/**
 * Canonical Search Result
 */
export declare const CanonicalSearchResultSchema: z.ZodObject<{
    edges: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source_entity_id: z.ZodString;
        target_entity_id: z.ZodString;
        relation_type: z.ZodString;
        fact: z.ZodString;
        summary: z.ZodOptional<z.ZodString>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        created_at: z.ZodString;
        updated_at: z.ZodOptional<z.ZodString>;
        episodes: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        created_at: string;
        attributes: Record<string, unknown>;
        relation_type: string;
        source_entity_id: string;
        target_entity_id: string;
        fact: string;
        episodes: string[];
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
    }, {
        id: string;
        created_at: string;
        relation_type: string;
        source_entity_id: string;
        target_entity_id: string;
        fact: string;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
        attributes?: Record<string, unknown> | undefined;
        episodes?: string[] | undefined;
    }>, "many">;
    nodes: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        name: z.ZodString;
        labels: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        summary: z.ZodOptional<z.ZodString>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        created_at: z.ZodString;
        updated_at: z.ZodOptional<z.ZodString>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        id: string;
        created_at: string;
        labels: string[];
        attributes: Record<string, unknown>;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
    }, {
        name: string;
        id: string;
        created_at: string;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    }>, "many">;
    total_count: z.ZodNumber;
    query_time_ms: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    edges: {
        id: string;
        created_at: string;
        attributes: Record<string, unknown>;
        relation_type: string;
        source_entity_id: string;
        target_entity_id: string;
        fact: string;
        episodes: string[];
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
    }[];
    total_count: number;
    nodes: {
        name: string;
        id: string;
        created_at: string;
        labels: string[];
        attributes: Record<string, unknown>;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
    }[];
    query_time_ms: number;
    metadata?: Record<string, unknown> | undefined;
}, {
    edges: {
        id: string;
        created_at: string;
        relation_type: string;
        source_entity_id: string;
        target_entity_id: string;
        fact: string;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
        attributes?: Record<string, unknown> | undefined;
        episodes?: string[] | undefined;
    }[];
    total_count: number;
    nodes: {
        name: string;
        id: string;
        created_at: string;
        metadata?: Record<string, unknown> | undefined;
        summary?: string | undefined;
        updated_at?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    }[];
    query_time_ms: number;
    metadata?: Record<string, unknown> | undefined;
}>;
export type CanonicalSearchResult = z.infer<typeof CanonicalSearchResultSchema>;
/**
 * Add Episode Operation
 */
export declare const AddEpisodeOperationSchema: z.ZodObject<{
    name: z.ZodString;
    content: z.ZodString;
    source_description: z.ZodOptional<z.ZodString>;
    episode_type: z.ZodDefault<z.ZodEnum<["text", "message", "document", "code", "transaction", "event", "observation", "custom"]>>;
    valid_at: z.ZodString;
    group_id: z.ZodOptional<z.ZodString>;
    uuid: z.ZodOptional<z.ZodString>;
    entity_types: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodString>>;
    excluded_entity_types: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    update_communities: z.ZodDefault<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    name: string;
    content: string;
    update_communities: boolean;
    valid_at: string;
    episode_type: "custom" | "text" | "message" | "event" | "code" | "document" | "transaction" | "observation";
    uuid?: string | undefined;
    source_description?: string | undefined;
    group_id?: string | undefined;
    entity_types?: Record<string, string> | undefined;
    excluded_entity_types?: string[] | undefined;
}, {
    name: string;
    content: string;
    valid_at: string;
    update_communities?: boolean | undefined;
    uuid?: string | undefined;
    source_description?: string | undefined;
    episode_type?: "custom" | "text" | "message" | "event" | "code" | "document" | "transaction" | "observation" | undefined;
    group_id?: string | undefined;
    entity_types?: Record<string, string> | undefined;
    excluded_entity_types?: string[] | undefined;
}>;
export type AddEpisodeOperation = z.infer<typeof AddEpisodeOperationSchema>;
/**
 * Add Episode Result
 */
export declare const AddEpisodeResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    episode_id: z.ZodString;
    entities_extracted: z.ZodNumber;
    relationships_extracted: z.ZodNumber;
    communities_updated: z.ZodDefault<z.ZodNumber>;
    processing_time_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    processing_time_ms: number;
    episode_id: string;
    entities_extracted: number;
    relationships_extracted: number;
    communities_updated: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
}, {
    success: boolean;
    processing_time_ms: number;
    episode_id: string;
    entities_extracted: number;
    relationships_extracted: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    communities_updated?: number | undefined;
}>;
export type AddEpisodeResult = z.infer<typeof AddEpisodeResultSchema>;
/**
 * Add Triplet Operation (Subject -> Predicate -> Object)
 */
export declare const AddTripletOperationSchema: z.ZodObject<{
    subject: z.ZodObject<{
        name: z.ZodString;
        labels: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        summary: z.ZodOptional<z.ZodString>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        labels: string[];
        attributes: Record<string, unknown>;
        summary?: string | undefined;
    }, {
        name: string;
        summary?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    }>;
    predicate: z.ZodObject<{
        relation_type: z.ZodString;
        fact: z.ZodString;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        attributes: Record<string, unknown>;
        relation_type: string;
        fact: string;
    }, {
        relation_type: string;
        fact: string;
        attributes?: Record<string, unknown> | undefined;
    }>;
    object: z.ZodObject<{
        name: z.ZodString;
        labels: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
        summary: z.ZodOptional<z.ZodString>;
        attributes: z.ZodDefault<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        labels: string[];
        attributes: Record<string, unknown>;
        summary?: string | undefined;
    }, {
        name: string;
        summary?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    }>;
    group_id: z.ZodOptional<z.ZodString>;
    valid_at: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    object: {
        name: string;
        labels: string[];
        attributes: Record<string, unknown>;
        summary?: string | undefined;
    };
    subject: {
        name: string;
        labels: string[];
        attributes: Record<string, unknown>;
        summary?: string | undefined;
    };
    predicate: {
        attributes: Record<string, unknown>;
        relation_type: string;
        fact: string;
    };
    valid_at?: string | undefined;
    group_id?: string | undefined;
}, {
    object: {
        name: string;
        summary?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    };
    subject: {
        name: string;
        summary?: string | undefined;
        labels?: string[] | undefined;
        attributes?: Record<string, unknown> | undefined;
    };
    predicate: {
        relation_type: string;
        fact: string;
        attributes?: Record<string, unknown> | undefined;
    };
    valid_at?: string | undefined;
    group_id?: string | undefined;
}>;
export type AddTripletOperation = z.infer<typeof AddTripletOperationSchema>;
/**
 * Add Triplet Result
 */
export declare const AddTripletResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    subject_uuid: z.ZodOptional<z.ZodString>;
    object_uuid: z.ZodOptional<z.ZodString>;
    edge_uuid: z.ZodOptional<z.ZodString>;
    processing_time_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    processing_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    subject_uuid?: string | undefined;
    object_uuid?: string | undefined;
    edge_uuid?: string | undefined;
}, {
    success: boolean;
    processing_time_ms: number;
    correlation_id?: string | undefined;
    error?: string | undefined;
    subject_uuid?: string | undefined;
    object_uuid?: string | undefined;
    edge_uuid?: string | undefined;
}>;
export type AddTripletResult = z.infer<typeof AddTripletResultSchema>;
/**
 * Graph Statistics
 */
export declare const GraphStatisticsSchema: z.ZodObject<{
    entities_count: z.ZodNumber;
    relationships_count: z.ZodNumber;
    episodes_count: z.ZodNumber;
    communities_count: z.ZodDefault<z.ZodNumber>;
    initialized: z.ZodBoolean;
    connection_status: z.ZodEnum<["connected", "disconnected", "error"]>;
    last_update: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    initialized: boolean;
    entities_count: number;
    relationships_count: number;
    episodes_count: number;
    connection_status: "error" | "disconnected" | "connected";
    communities_count: number;
    last_update?: string | undefined;
}, {
    initialized: boolean;
    entities_count: number;
    relationships_count: number;
    episodes_count: number;
    connection_status: "error" | "disconnected" | "connected";
    communities_count?: number | undefined;
    last_update?: string | undefined;
}>;
export type GraphStatistics = z.infer<typeof GraphStatisticsSchema>;
/**
 * Validate data against canonical schema
 *
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with success flag and data or errors
 */
export declare function validateCanonical<T extends z.ZodTypeAny>(schema: T, data: unknown): {
    success: boolean;
    data?: z.infer<T>;
    errors?: string[];
};
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
//# sourceMappingURL=graphiti-canonical.d.ts.map