"use strict";
/**
 * AI Knowledge Graph Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AI Knowledge Graph
 * (knowledge representation and reasoning) interactions.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AiKnowledgeGraphExamples = exports.AiKnowledgeGraphError = exports.AiKnowledgeGraphResponse = exports.AiKnowledgeGraphRequest = exports.GraphQuery = exports.GraphEdge = exports.GraphNode = exports.EdgeType = exports.NodeType = void 0;
exports.validateAiKnowledgeGraphRequest = validateAiKnowledgeGraphRequest;
exports.isAiKnowledgeGraphRequest = isAiKnowledgeGraphRequest;
const zod_1 = require("zod");
/**
 * Node Type Enum
 */
exports.NodeType = zod_1.z.enum([
    'entity',
    'concept',
    'relation',
    'attribute',
    'event',
    'document',
]);
/**
 * Edge Type Enum
 */
exports.EdgeType = zod_1.z.enum([
    'related_to',
    'part_of',
    'instance_of',
    'causes',
    'enables',
    'precedes',
    'located_at',
    'has_property',
    'semantic_similar',
]);
/**
 * Graph Node Schema
 */
exports.GraphNode = zod_1.z.object({
    node_id: zod_1.z.string().describe("Unique node identifier"),
    node_type: exports.NodeType.describe("Type of node"),
    label: zod_1.z.string().describe("Node label/name"),
    description: zod_1.z.string().optional().describe("Node description"),
    properties: zod_1.z.record(zod_1.z.any()).optional().describe("Node properties"),
    embeddings: zod_1.z.array(zod_1.z.number()).optional().describe("Vector embeddings"),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        updated_at: zod_1.z.string().datetime().optional(),
        confidence: zod_1.z.number().min(0).max(1).optional(),
        source: zod_1.z.string().optional().describe("Data source"),
    }).optional().describe("Node metadata"),
});
/**
 * Graph Edge Schema
 */
exports.GraphEdge = zod_1.z.object({
    edge_id: zod_1.z.string().optional().describe("Unique edge identifier"),
    edge_type: exports.EdgeType.describe("Type of edge/relation"),
    source_id: zod_1.z.string().describe("Source node ID"),
    target_id: zod_1.z.string().describe("Target node ID"),
    weight: zod_1.z.number().min(0).max(1).optional().describe("Edge weight"),
    properties: zod_1.z.record(zod_1.z.any()).optional().describe("Edge properties"),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        confidence: zod_1.z.number().min(0).max(1).optional(),
        source: zod_1.z.string().optional(),
    }).optional(),
});
/**
 * Graph Query Schema
 */
exports.GraphQuery = zod_1.z.object({
    query_type: zod_1.z.enum([
        'node_lookup',
        'neighborhood',
        'path',
        'subgraph',
        'pattern_match',
        'similarity_search',
    ]).describe("Type of query"),
    criteria: zod_1.z.object({
        node_id: zod_1.z.string().optional().describe("Specific node ID"),
        node_type: exports.NodeType.optional().describe("Filter by node type"),
        edge_type: exports.EdgeType.optional().describe("Filter by edge type"),
        properties: zod_1.z.record(zod_1.z.any()).optional().describe("Property filters"),
        labels: zod_1.z.array(zod_1.z.string()).optional().describe("Label filters"),
        max_depth: zod_1.z.number().int().min(1).optional().describe("Max traversal depth"),
        limit: zod_1.z.number().int().positive().optional().describe("Result limit"),
    }).describe("Query criteria"),
    similarity: zod_1.z.object({
        vector: zod_1.z.array(zod_1.z.number()).optional().describe("Query vector"),
        threshold: zod_1.z.number().min(0).max(1).optional().describe("Similarity threshold"),
        top_k: zod_1.z.number().int().positive().optional().describe("Top K results"),
    }).optional().describe("Similarity search parameters"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Knowledge Graph Request Schema
 */
exports.AiKnowledgeGraphRequest = zod_1.z.object({
    graph_id: zod_1.z.string().optional().describe("Graph identifier"),
    action: zod_1.z.enum([
        'create_graph',
        'add_node',
        'add_edge',
        'add_nodes',
        'add_edges',
        'query',
        'update_node',
        'update_edge',
        'delete_node',
        'delete_edge',
        'export',
    ]).describe("Action to perform"),
    graph_config: zod_1.z.object({
        name: zod_1.z.string().optional(),
        description: zod_1.z.string().optional(),
        schema_version: zod_1.z.string().optional(),
        storage_backend: zod_1.z.enum(['memory', 'neo4j', 'postgres']).optional(),
    }).optional().describe("Graph configuration"),
    node: exports.GraphNode.optional().describe("Node to add/update"),
    nodes: zod_1.z.array(exports.GraphNode).optional().describe("Multiple nodes"),
    edge: exports.GraphEdge.optional().describe("Edge to add/update"),
    edges: zod_1.z.array(exports.GraphEdge).optional().describe("Multiple edges"),
    query: exports.GraphQuery.optional().describe("Query to execute"),
    timeout_ms: zod_1.z.number()
        .int().positive().max(60000)
        .describe("Request timeout (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Knowledge Graph Response Schema
 */
exports.AiKnowledgeGraphResponse = zod_1.z.object({
    graph_id: zod_1.z.string().describe("Graph identifier"),
    action: zod_1.z.enum([
        'create_graph',
        'add_node',
        'add_edge',
        'add_nodes',
        'add_edges',
        'query',
        'update_node',
        'update_edge',
        'delete_node',
        'delete_edge',
        'export',
    ]).describe("Action performed"),
    status: zod_1.z.enum([
        'success',
        'failed',
        'timeout',
        'partial',
    ]).describe("Action status"),
    result: zod_1.z.object({
        node_id: zod_1.z.string().optional().describe("Created/updated node ID"),
        edge_id: zod_1.z.string().optional().describe("Created/updated edge ID"),
        nodes: zod_1.z.array(exports.GraphNode).optional().describe("Query result nodes"),
        edges: zod_1.z.array(exports.GraphEdge).optional().describe("Query result edges"),
        paths: zod_1.z.array(zod_1.z.object({
            nodes: zod_1.z.array(zod_1.z.string()),
            edges: zod_1.z.array(zod_1.z.string()),
            length: zod_1.z.number(),
        })).optional().describe("Path query results"),
        subgraph: zod_1.z.object({
            nodes: zod_1.z.array(exports.GraphNode),
            edges: zod_1.z.array(exports.GraphEdge),
        }).optional().describe("Subgraph query results"),
        export_data: zod_1.z.record(zod_1.z.any()).optional().describe("Exported graph data"),
        stats: zod_1.z.object({
            node_count: zod_1.z.number().optional(),
            edge_count: zod_1.z.number().optional(),
            query_time_ms: zod_1.z.number().optional(),
        }).optional().describe("Statistics"),
    }).optional().describe("Action result"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional(),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional(),
        updated_at: zod_1.z.string().datetime().optional(),
        processing_time_ms: zod_1.z.number().optional(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Error Model
 */
exports.AiKnowledgeGraphError = zod_1.z.object({
    code: zod_1.z.enum([
        'GRAPH_NOT_FOUND',
        'NODE_NOT_FOUND',
        'EDGE_NOT_FOUND',
        'INVALID_NODE_TYPE',
        'INVALID_EDGE_TYPE',
        'DUPLICATE_NODE',
        'DUPLICATE_EDGE',
        'QUERY_SYNTAX_ERROR',
        'VALIDATION_ERROR',
        'UNKNOWN_ERROR',
    ]),
    message: zod_1.z.string(),
    details: zod_1.z.record(zod_1.z.any()).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
function validateAiKnowledgeGraphRequest(data) {
    const result = exports.AiKnowledgeGraphRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function isAiKnowledgeGraphRequest(data) {
    return typeof data === 'object' && data !== null &&
        'action' in data;
}
/**
 * Examples
 */
exports.AiKnowledgeGraphExamples = {
    validAddNode: {
        graph_id: "kg_001",
        action: "add_node",
        node: {
            node_id: "node_001",
            node_type: "entity",
            label: "Albert Einstein",
            description: "Physicist who developed the theory of relativity",
            properties: {
                birth_year: 1879,
                nationality: "German-Swiss-American",
            },
            embeddings: [0.1, 0.2, 0.3, 0.4],
        },
        timeout_ms: 5000,
    },
    validQuery: {
        graph_id: "kg_001",
        action: "query",
        query: {
            query_type: "neighborhood",
            criteria: {
                node_id: "node_001",
                max_depth: 2,
                limit: 50,
            },
        },
        timeout_ms: 5000,
    },
};
//# sourceMappingURL=ai-knowledge-graph-canonical.js.map