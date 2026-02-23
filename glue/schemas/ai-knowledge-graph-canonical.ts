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
export const NodeType = z.enum([
  'entity',
  'concept',
  'relation',
  'attribute',
  'event',
  'document',
]);

export type NodeType = z.infer<typeof NodeType>;

/**
 * Edge Type Enum
 */
export const EdgeType = z.enum([
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

export type EdgeType = z.infer<typeof EdgeType>;

/**
 * Graph Node Schema
 */
export const GraphNode = z.object({
  node_id: z.string().describe("Unique node identifier"),

  node_type: NodeType.describe("Type of node"),

  label: z.string().describe("Node label/name"),

  description: z.string().optional().describe("Node description"),

  properties: z.record(z.any()).optional().describe("Node properties"),

  embeddings: z.array(z.number()).optional().describe("Vector embeddings"),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    updated_at: z.string().datetime().optional(),
    confidence: z.number().min(0).max(1).optional(),
    source: z.string().optional().describe("Data source"),
  }).optional().describe("Node metadata"),
});

export type GraphNode = z.infer<typeof GraphNode>;

/**
 * Graph Edge Schema
 */
export const GraphEdge = z.object({
  edge_id: z.string().optional().describe("Unique edge identifier"),

  edge_type: EdgeType.describe("Type of edge/relation"),

  source_id: z.string().describe("Source node ID"),

  target_id: z.string().describe("Target node ID"),

  weight: z.number().min(0).max(1).optional()
    .describe("Edge weight"),

  properties: z.record(z.any()).optional().describe("Edge properties"),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    confidence: z.number().min(0).max(1).optional(),
    source: z.string().optional(),
  }).optional(),
});

export type GraphEdge = z.infer<typeof GraphEdge>;

/**
 * Graph Query Schema
 */
export const GraphQuery = z.object({
  query_type: z.enum([
    'node_lookup',
    'neighborhood',
    'path',
    'subgraph',
    'pattern_match',
    'similarity_search',
  ]).describe("Type of query"),

  criteria: z.object({
    node_id: z.string().optional().describe("Specific node ID"),
    node_type: NodeType.optional().describe("Filter by node type"),
    edge_type: EdgeType.optional().describe("Filter by edge type"),
    properties: z.record(z.any()).optional().describe("Property filters"),
    labels: z.array(z.string()).optional().describe("Label filters"),
    max_depth: z.number().int().min(1).optional()
      .describe("Max traversal depth"),
    limit: z.number().int().positive().optional()
      .describe("Result limit"),
  }).describe("Query criteria"),

  similarity: z.object({
    vector: z.array(z.number()).optional().describe("Query vector"),
    threshold: z.number().min(0).max(1).optional()
      .describe("Similarity threshold"),
    top_k: z.number().int().positive().optional()
      .describe("Top K results"),
  }).optional().describe("Similarity search parameters"),

  metadata: z.record(z.any()).optional(),
});

export type GraphQuery = z.infer<typeof GraphQuery>;

/**
 * Knowledge Graph Request Schema
 */
export const AiKnowledgeGraphRequest = z.object({
  graph_id: z.string().optional().describe("Graph identifier"),

  action: z.enum([
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

  graph_config: z.object({
    name: z.string().optional(),
    description: z.string().optional(),
    schema_version: z.string().optional(),
    storage_backend: z.enum(['memory', 'neo4j', 'postgres']).optional(),
  }).optional().describe("Graph configuration"),

  node: GraphNode.optional().describe("Node to add/update"),

  nodes: z.array(GraphNode).optional().describe("Multiple nodes"),

  edge: GraphEdge.optional().describe("Edge to add/update"),

  edges: z.array(GraphEdge).optional().describe("Multiple edges"),

  query: GraphQuery.optional().describe("Query to execute"),

  timeout_ms: z.number()
    .int().positive().max(60000)
    .describe("Request timeout (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),

  metadata: z.record(z.any()).optional(),
});

export type AiKnowledgeGraphRequest = z.infer<typeof AiKnowledgeGraphRequest>;

/**
 * Knowledge Graph Response Schema
 */
export const AiKnowledgeGraphResponse = z.object({
  graph_id: z.string().describe("Graph identifier"),

  action: z.enum([
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

  status: z.enum([
    'success',
    'failed',
    'timeout',
    'partial',
  ]).describe("Action status"),

  result: z.object({
    node_id: z.string().optional().describe("Created/updated node ID"),
    edge_id: z.string().optional().describe("Created/updated edge ID"),
    nodes: z.array(GraphNode).optional().describe("Query result nodes"),
    edges: z.array(GraphEdge).optional().describe("Query result edges"),
    paths: z.array(z.object({
      nodes: z.array(z.string()),
      edges: z.array(z.string()),
      length: z.number(),
    })).optional().describe("Path query results"),
    subgraph: z.object({
      nodes: z.array(GraphNode),
      edges: z.array(GraphEdge),
    }).optional().describe("Subgraph query results"),
    export_data: z.record(z.any()).optional().describe("Exported graph data"),
    stats: z.object({
      node_count: z.number().optional(),
      edge_count: z.number().optional(),
      query_time_ms: z.number().optional(),
    }).optional().describe("Statistics"),
  }).optional().describe("Action result"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional(),

  metadata: z.object({
    created_at: z.string().datetime().optional(),
    updated_at: z.string().datetime().optional(),
    processing_time_ms: z.number().optional(),
  }).optional(),

  correlation_id: z.string().uuid().optional(),

  timestamp: z.string().datetime(),
});

export type AiKnowledgeGraphResponse = z.infer<typeof AiKnowledgeGraphResponse>;

/**
 * Error Model
 */
export const AiKnowledgeGraphError = z.object({
  code: z.enum([
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
  message: z.string(),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AiKnowledgeGraphError = z.infer<typeof AiKnowledgeGraphError>;

/**
 * Validation Functions
 */
export function validateAiKnowledgeGraphRequest(data: unknown): {
  success: boolean;
  data?: AiKnowledgeGraphRequest;
  errors?: string[];
} {
  const result = AiKnowledgeGraphRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function isAiKnowledgeGraphRequest(data: unknown): data is AiKnowledgeGraphRequest {
  return typeof data === 'object' && data !== null
    && 'action' in data;
}

/**
 * Examples
 */
export const AiKnowledgeGraphExamples = {
  validAddNode: {
    graph_id: "kg_001",
    action: "add_node" as const,
    node: {
      node_id: "node_001",
      node_type: "entity" as const,
      label: "Albert Einstein",
      description: "Physicist who developed the theory of relativity",
      properties: {
        birth_year: 1879,
        nationality: "German-Swiss-American",
      },
      embeddings: [0.1, 0.2, 0.3, 0.4],
    },
    timeout_ms: 5000,
  } as AiKnowledgeGraphRequest,

  validQuery: {
    graph_id: "kg_001",
    action: "query" as const,
    query: {
      query_type: "neighborhood" as const,
      criteria: {
        node_id: "node_001",
        max_depth: 2,
        limit: 50,
      },
    },
    timeout_ms: 5000,
  } as AiKnowledgeGraphRequest,
};
