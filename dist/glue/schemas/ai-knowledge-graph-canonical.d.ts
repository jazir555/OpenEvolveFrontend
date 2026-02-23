/**
 * AI Knowledge Graph Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AI Knowledge Graph
 * (knowledge representation and reasoning) interactions.
 */
import { z } from 'zod';
/**
 * Node Type Enum
 */
export declare const NodeType: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
export type NodeType = z.infer<typeof NodeType>;
/**
 * Edge Type Enum
 */
export declare const EdgeType: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
export type EdgeType = z.infer<typeof EdgeType>;
/**
 * Graph Node Schema
 */
export declare const GraphNode: z.ZodObject<{
    node_id: z.ZodString;
    node_type: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
    label: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    embeddings: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        confidence: z.ZodOptional<z.ZodNumber>;
        source: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
        updated_at?: string | undefined;
    }, {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
        updated_at?: string | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    label: string;
    node_id: string;
    node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
    metadata?: {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
        updated_at?: string | undefined;
    } | undefined;
    description?: string | undefined;
    properties?: Record<string, any> | undefined;
    embeddings?: number[] | undefined;
}, {
    label: string;
    node_id: string;
    node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
    metadata?: {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
        updated_at?: string | undefined;
    } | undefined;
    description?: string | undefined;
    properties?: Record<string, any> | undefined;
    embeddings?: number[] | undefined;
}>;
export type GraphNode = z.infer<typeof GraphNode>;
/**
 * Graph Edge Schema
 */
export declare const GraphEdge: z.ZodObject<{
    edge_id: z.ZodOptional<z.ZodString>;
    edge_type: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
    source_id: z.ZodString;
    target_id: z.ZodString;
    weight: z.ZodOptional<z.ZodNumber>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        confidence: z.ZodOptional<z.ZodNumber>;
        source: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
    }, {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
    }>>;
}, "strip", z.ZodTypeAny, {
    source_id: string;
    target_id: string;
    edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
    metadata?: {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
    } | undefined;
    weight?: number | undefined;
    properties?: Record<string, any> | undefined;
    edge_id?: string | undefined;
}, {
    source_id: string;
    target_id: string;
    edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
    metadata?: {
        created_at?: string | undefined;
        source?: string | undefined;
        confidence?: number | undefined;
    } | undefined;
    weight?: number | undefined;
    properties?: Record<string, any> | undefined;
    edge_id?: string | undefined;
}>;
export type GraphEdge = z.infer<typeof GraphEdge>;
/**
 * Graph Query Schema
 */
export declare const GraphQuery: z.ZodObject<{
    query_type: z.ZodEnum<["node_lookup", "neighborhood", "path", "subgraph", "pattern_match", "similarity_search"]>;
    criteria: z.ZodObject<{
        node_id: z.ZodOptional<z.ZodString>;
        node_type: z.ZodOptional<z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>>;
        edge_type: z.ZodOptional<z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        max_depth: z.ZodOptional<z.ZodNumber>;
        limit: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        limit?: number | undefined;
        max_depth?: number | undefined;
        labels?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        node_id?: string | undefined;
        node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
        edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
    }, {
        limit?: number | undefined;
        max_depth?: number | undefined;
        labels?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        node_id?: string | undefined;
        node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
        edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
    }>;
    similarity: z.ZodOptional<z.ZodObject<{
        vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        threshold: z.ZodOptional<z.ZodNumber>;
        top_k: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        threshold?: number | undefined;
        top_k?: number | undefined;
        vector?: number[] | undefined;
    }, {
        threshold?: number | undefined;
        top_k?: number | undefined;
        vector?: number[] | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
    criteria: {
        limit?: number | undefined;
        max_depth?: number | undefined;
        labels?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        node_id?: string | undefined;
        node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
        edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
    };
    metadata?: Record<string, any> | undefined;
    similarity?: {
        threshold?: number | undefined;
        top_k?: number | undefined;
        vector?: number[] | undefined;
    } | undefined;
}, {
    query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
    criteria: {
        limit?: number | undefined;
        max_depth?: number | undefined;
        labels?: string[] | undefined;
        properties?: Record<string, any> | undefined;
        node_id?: string | undefined;
        node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
        edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
    };
    metadata?: Record<string, any> | undefined;
    similarity?: {
        threshold?: number | undefined;
        top_k?: number | undefined;
        vector?: number[] | undefined;
    } | undefined;
}>;
export type GraphQuery = z.infer<typeof GraphQuery>;
/**
 * Knowledge Graph Request Schema
 */
export declare const AiKnowledgeGraphRequest: z.ZodObject<{
    graph_id: z.ZodOptional<z.ZodString>;
    action: z.ZodEnum<["create_graph", "add_node", "add_edge", "add_nodes", "add_edges", "query", "update_node", "update_edge", "delete_node", "delete_edge", "export"]>;
    graph_config: z.ZodOptional<z.ZodObject<{
        name: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        schema_version: z.ZodOptional<z.ZodString>;
        storage_backend: z.ZodOptional<z.ZodEnum<["memory", "neo4j", "postgres"]>>;
    }, "strip", z.ZodTypeAny, {
        name?: string | undefined;
        description?: string | undefined;
        schema_version?: string | undefined;
        storage_backend?: "memory" | "neo4j" | "postgres" | undefined;
    }, {
        name?: string | undefined;
        description?: string | undefined;
        schema_version?: string | undefined;
        storage_backend?: "memory" | "neo4j" | "postgres" | undefined;
    }>>;
    node: z.ZodOptional<z.ZodObject<{
        node_id: z.ZodString;
        node_type: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
        label: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        embeddings: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        metadata: z.ZodOptional<z.ZodObject<{
            created_at: z.ZodOptional<z.ZodString>;
            updated_at: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
            source: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        }, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }, {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }>>;
    nodes: z.ZodOptional<z.ZodArray<z.ZodObject<{
        node_id: z.ZodString;
        node_type: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
        label: z.ZodString;
        description: z.ZodOptional<z.ZodString>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        embeddings: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
        metadata: z.ZodOptional<z.ZodObject<{
            created_at: z.ZodOptional<z.ZodString>;
            updated_at: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
            source: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        }, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }, {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }>, "many">>;
    edge: z.ZodOptional<z.ZodObject<{
        edge_id: z.ZodOptional<z.ZodString>;
        edge_type: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
        source_id: z.ZodString;
        target_id: z.ZodString;
        weight: z.ZodOptional<z.ZodNumber>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        metadata: z.ZodOptional<z.ZodObject<{
            created_at: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
            source: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        }, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }, {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }>>;
    edges: z.ZodOptional<z.ZodArray<z.ZodObject<{
        edge_id: z.ZodOptional<z.ZodString>;
        edge_type: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
        source_id: z.ZodString;
        target_id: z.ZodString;
        weight: z.ZodOptional<z.ZodNumber>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        metadata: z.ZodOptional<z.ZodObject<{
            created_at: z.ZodOptional<z.ZodString>;
            confidence: z.ZodOptional<z.ZodNumber>;
            source: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        }, {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }, {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }>, "many">>;
    query: z.ZodOptional<z.ZodObject<{
        query_type: z.ZodEnum<["node_lookup", "neighborhood", "path", "subgraph", "pattern_match", "similarity_search"]>;
        criteria: z.ZodObject<{
            node_id: z.ZodOptional<z.ZodString>;
            node_type: z.ZodOptional<z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>>;
            edge_type: z.ZodOptional<z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            labels: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
            max_depth: z.ZodOptional<z.ZodNumber>;
            limit: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        }, {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        }>;
        similarity: z.ZodOptional<z.ZodObject<{
            vector: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            threshold: z.ZodOptional<z.ZodNumber>;
            top_k: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        }, {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        }>>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
        criteria: {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        };
        metadata?: Record<string, any> | undefined;
        similarity?: {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        } | undefined;
    }, {
        query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
        criteria: {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        };
        metadata?: Record<string, any> | undefined;
        similarity?: {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        } | undefined;
    }>>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    action: "query" | "export" | "create_graph" | "add_node" | "add_edge" | "add_nodes" | "add_edges" | "update_node" | "update_edge" | "delete_node" | "delete_edge";
    node?: {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    } | undefined;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    edge?: {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    } | undefined;
    query?: {
        query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
        criteria: {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        };
        metadata?: Record<string, any> | undefined;
        similarity?: {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        } | undefined;
    } | undefined;
    graph_id?: string | undefined;
    edges?: {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }[] | undefined;
    nodes?: {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }[] | undefined;
    graph_config?: {
        name?: string | undefined;
        description?: string | undefined;
        schema_version?: string | undefined;
        storage_backend?: "memory" | "neo4j" | "postgres" | undefined;
    } | undefined;
}, {
    timeout_ms: number;
    action: "query" | "export" | "create_graph" | "add_node" | "add_edge" | "add_nodes" | "add_edges" | "update_node" | "update_edge" | "delete_node" | "delete_edge";
    node?: {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    } | undefined;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    edge?: {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    } | undefined;
    query?: {
        query_type: "path" | "node_lookup" | "neighborhood" | "subgraph" | "pattern_match" | "similarity_search";
        criteria: {
            limit?: number | undefined;
            max_depth?: number | undefined;
            labels?: string[] | undefined;
            properties?: Record<string, any> | undefined;
            node_id?: string | undefined;
            node_type?: "event" | "document" | "entity" | "relation" | "concept" | "attribute" | undefined;
            edge_type?: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property" | undefined;
        };
        metadata?: Record<string, any> | undefined;
        similarity?: {
            threshold?: number | undefined;
            top_k?: number | undefined;
            vector?: number[] | undefined;
        } | undefined;
    } | undefined;
    graph_id?: string | undefined;
    edges?: {
        source_id: string;
        target_id: string;
        edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
        } | undefined;
        weight?: number | undefined;
        properties?: Record<string, any> | undefined;
        edge_id?: string | undefined;
    }[] | undefined;
    nodes?: {
        label: string;
        node_id: string;
        node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
        metadata?: {
            created_at?: string | undefined;
            source?: string | undefined;
            confidence?: number | undefined;
            updated_at?: string | undefined;
        } | undefined;
        description?: string | undefined;
        properties?: Record<string, any> | undefined;
        embeddings?: number[] | undefined;
    }[] | undefined;
    graph_config?: {
        name?: string | undefined;
        description?: string | undefined;
        schema_version?: string | undefined;
        storage_backend?: "memory" | "neo4j" | "postgres" | undefined;
    } | undefined;
}>;
export type AiKnowledgeGraphRequest = z.infer<typeof AiKnowledgeGraphRequest>;
/**
 * Knowledge Graph Response Schema
 */
export declare const AiKnowledgeGraphResponse: z.ZodObject<{
    graph_id: z.ZodString;
    action: z.ZodEnum<["create_graph", "add_node", "add_edge", "add_nodes", "add_edges", "query", "update_node", "update_edge", "delete_node", "delete_edge", "export"]>;
    status: z.ZodEnum<["success", "failed", "timeout", "partial"]>;
    result: z.ZodOptional<z.ZodObject<{
        node_id: z.ZodOptional<z.ZodString>;
        edge_id: z.ZodOptional<z.ZodString>;
        nodes: z.ZodOptional<z.ZodArray<z.ZodObject<{
            node_id: z.ZodString;
            node_type: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
            label: z.ZodString;
            description: z.ZodOptional<z.ZodString>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            embeddings: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
            metadata: z.ZodOptional<z.ZodObject<{
                created_at: z.ZodOptional<z.ZodString>;
                updated_at: z.ZodOptional<z.ZodString>;
                confidence: z.ZodOptional<z.ZodNumber>;
                source: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            }, {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }, {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }>, "many">>;
        edges: z.ZodOptional<z.ZodArray<z.ZodObject<{
            edge_id: z.ZodOptional<z.ZodString>;
            edge_type: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
            source_id: z.ZodString;
            target_id: z.ZodString;
            weight: z.ZodOptional<z.ZodNumber>;
            properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
            metadata: z.ZodOptional<z.ZodObject<{
                created_at: z.ZodOptional<z.ZodString>;
                confidence: z.ZodOptional<z.ZodNumber>;
                source: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            }, {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            }>>;
        }, "strip", z.ZodTypeAny, {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }, {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }>, "many">>;
        paths: z.ZodOptional<z.ZodArray<z.ZodObject<{
            nodes: z.ZodArray<z.ZodString, "many">;
            edges: z.ZodArray<z.ZodString, "many">;
            length: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            length: number;
            edges: string[];
            nodes: string[];
        }, {
            length: number;
            edges: string[];
            nodes: string[];
        }>, "many">>;
        subgraph: z.ZodOptional<z.ZodObject<{
            nodes: z.ZodArray<z.ZodObject<{
                node_id: z.ZodString;
                node_type: z.ZodEnum<["entity", "concept", "relation", "attribute", "event", "document"]>;
                label: z.ZodString;
                description: z.ZodOptional<z.ZodString>;
                properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
                embeddings: z.ZodOptional<z.ZodArray<z.ZodNumber, "many">>;
                metadata: z.ZodOptional<z.ZodObject<{
                    created_at: z.ZodOptional<z.ZodString>;
                    updated_at: z.ZodOptional<z.ZodString>;
                    confidence: z.ZodOptional<z.ZodNumber>;
                    source: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                }, {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }, {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }>, "many">;
            edges: z.ZodArray<z.ZodObject<{
                edge_id: z.ZodOptional<z.ZodString>;
                edge_type: z.ZodEnum<["related_to", "part_of", "instance_of", "causes", "enables", "precedes", "located_at", "has_property", "semantic_similar"]>;
                source_id: z.ZodString;
                target_id: z.ZodString;
                weight: z.ZodOptional<z.ZodNumber>;
                properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
                metadata: z.ZodOptional<z.ZodObject<{
                    created_at: z.ZodOptional<z.ZodString>;
                    confidence: z.ZodOptional<z.ZodNumber>;
                    source: z.ZodOptional<z.ZodString>;
                }, "strip", z.ZodTypeAny, {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                }, {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                }>>;
            }, "strip", z.ZodTypeAny, {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }, {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }>, "many">;
        }, "strip", z.ZodTypeAny, {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        }, {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        }>>;
        export_data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        stats: z.ZodOptional<z.ZodObject<{
            node_count: z.ZodOptional<z.ZodNumber>;
            edge_count: z.ZodOptional<z.ZodNumber>;
            query_time_ms: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        }, {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        stats?: {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        } | undefined;
        edges?: {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }[] | undefined;
        nodes?: {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }[] | undefined;
        edge_id?: string | undefined;
        node_id?: string | undefined;
        subgraph?: {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        } | undefined;
        paths?: {
            length: number;
            edges: string[];
            nodes: string[];
        }[] | undefined;
        export_data?: Record<string, any> | undefined;
    }, {
        stats?: {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        } | undefined;
        edges?: {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }[] | undefined;
        nodes?: {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }[] | undefined;
        edge_id?: string | undefined;
        node_id?: string | undefined;
        subgraph?: {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        } | undefined;
        paths?: {
            length: number;
            edges: string[];
            nodes: string[];
        }[] | undefined;
        export_data?: Record<string, any> | undefined;
    }>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        updated_at: z.ZodOptional<z.ZodString>;
        processing_time_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }, {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "success" | "failed" | "partial" | "timeout";
    action: "query" | "export" | "create_graph" | "add_node" | "add_edge" | "add_nodes" | "add_edges" | "update_node" | "update_edge" | "delete_node" | "delete_edge";
    graph_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        stats?: {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        } | undefined;
        edges?: {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }[] | undefined;
        nodes?: {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }[] | undefined;
        edge_id?: string | undefined;
        node_id?: string | undefined;
        subgraph?: {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        } | undefined;
        paths?: {
            length: number;
            edges: string[];
            nodes: string[];
        }[] | undefined;
        export_data?: Record<string, any> | undefined;
    } | undefined;
}, {
    timestamp: string;
    status: "success" | "failed" | "partial" | "timeout";
    action: "query" | "export" | "create_graph" | "add_node" | "add_edge" | "add_nodes" | "add_edges" | "update_node" | "update_edge" | "delete_node" | "delete_edge";
    graph_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        created_at?: string | undefined;
        updated_at?: string | undefined;
        processing_time_ms?: number | undefined;
    } | undefined;
    result?: {
        stats?: {
            node_count?: number | undefined;
            edge_count?: number | undefined;
            query_time_ms?: number | undefined;
        } | undefined;
        edges?: {
            source_id: string;
            target_id: string;
            edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
            } | undefined;
            weight?: number | undefined;
            properties?: Record<string, any> | undefined;
            edge_id?: string | undefined;
        }[] | undefined;
        nodes?: {
            label: string;
            node_id: string;
            node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
            metadata?: {
                created_at?: string | undefined;
                source?: string | undefined;
                confidence?: number | undefined;
                updated_at?: string | undefined;
            } | undefined;
            description?: string | undefined;
            properties?: Record<string, any> | undefined;
            embeddings?: number[] | undefined;
        }[] | undefined;
        edge_id?: string | undefined;
        node_id?: string | undefined;
        subgraph?: {
            edges: {
                source_id: string;
                target_id: string;
                edge_type: "semantic_similar" | "enables" | "related_to" | "part_of" | "instance_of" | "causes" | "precedes" | "located_at" | "has_property";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                } | undefined;
                weight?: number | undefined;
                properties?: Record<string, any> | undefined;
                edge_id?: string | undefined;
            }[];
            nodes: {
                label: string;
                node_id: string;
                node_type: "event" | "document" | "entity" | "relation" | "concept" | "attribute";
                metadata?: {
                    created_at?: string | undefined;
                    source?: string | undefined;
                    confidence?: number | undefined;
                    updated_at?: string | undefined;
                } | undefined;
                description?: string | undefined;
                properties?: Record<string, any> | undefined;
                embeddings?: number[] | undefined;
            }[];
        } | undefined;
        paths?: {
            length: number;
            edges: string[];
            nodes: string[];
        }[] | undefined;
        export_data?: Record<string, any> | undefined;
    } | undefined;
}>;
export type AiKnowledgeGraphResponse = z.infer<typeof AiKnowledgeGraphResponse>;
/**
 * Error Model
 */
export declare const AiKnowledgeGraphError: z.ZodObject<{
    code: z.ZodEnum<["GRAPH_NOT_FOUND", "NODE_NOT_FOUND", "EDGE_NOT_FOUND", "INVALID_NODE_TYPE", "INVALID_EDGE_TYPE", "DUPLICATE_NODE", "DUPLICATE_EDGE", "QUERY_SYNTAX_ERROR", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "NODE_NOT_FOUND" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "GRAPH_NOT_FOUND" | "EDGE_NOT_FOUND" | "INVALID_NODE_TYPE" | "INVALID_EDGE_TYPE" | "DUPLICATE_NODE" | "DUPLICATE_EDGE" | "QUERY_SYNTAX_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "NODE_NOT_FOUND" | "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "GRAPH_NOT_FOUND" | "EDGE_NOT_FOUND" | "INVALID_NODE_TYPE" | "INVALID_EDGE_TYPE" | "DUPLICATE_NODE" | "DUPLICATE_EDGE" | "QUERY_SYNTAX_ERROR";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type AiKnowledgeGraphError = z.infer<typeof AiKnowledgeGraphError>;
/**
 * Validation Functions
 */
export declare function validateAiKnowledgeGraphRequest(data: unknown): {
    success: boolean;
    data?: AiKnowledgeGraphRequest;
    errors?: string[];
};
export declare function isAiKnowledgeGraphRequest(data: unknown): data is AiKnowledgeGraphRequest;
/**
 * Examples
 */
export declare const AiKnowledgeGraphExamples: {
    validAddNode: AiKnowledgeGraphRequest;
    validQuery: AiKnowledgeGraphRequest;
};
//# sourceMappingURL=ai-knowledge-graph-canonical.d.ts.map