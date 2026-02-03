/**
 * KarateClub Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for KarateClub graph ML operations.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for KarateClub data in the glue layer.
 * Do not pass raw KarateClub API responses between services.
 *
 * KarateClub provides 51 algorithms across 3 categories:
 * - Community Detection (10 algorithms)
 * - Node Embedding (32 algorithms)
 * - Graph Embedding (10 algorithms)
 */

import { z } from 'zod';

/**
 * Algorithm Categories
 */
export const AlgorithmCategory = z.enum([
  'community',
  'node_embedding',
  'graph_embedding',
]);

export type AlgorithmCategory = z.infer<typeof AlgorithmCategory>;

/**
 * Supported Node Embedding Algorithms (32)
 */
export const NodeEmbeddingAlgorithm = z.enum([
  // Neighbourhood-based (17)
  'deepwalk',
  'node2vec',
  'walklets',
  'grarep',
  'hope',
  'netmf',
  'boostne',
  'randne',
  'nodesketch',
  'diff2vec',
  'sociodim',
  'glee',
  'laplacian_eigenmaps',
  'nmf_admm',
  'line',

  // Structural (3)
  'graphwave',
  'role2vec',
  'sinr',

  // Attributed (9)
  'feather_n',
  'tadw',
  'musae',
  'ae',
  'fscnmf',
  'sine',
  'bane',
  'tene',
  'asne',

  // Meta (1)
  'neu',
]);

export type NodeEmbeddingAlgorithm = z.infer<typeof NodeEmbeddingAlgorithm>;

/**
 * Supported Community Detection Algorithms (10)
 */
export const CommunityAlgorithm = z.enum([
  // Overlapping (6)
  'danmf',
  'm_nmf',
  'ego_splitting',
  'nnsed',
  'bigclam',
  'symmnmf',

  // Non-overlapping (4)
  'gemsec',
  'edmot',
  'scd',
  'label_propagation',
]);

export type CommunityAlgorithm = z.infer<typeof CommunityAlgorithm>;

/**
 * Supported Graph Embedding Algorithms (10)
 */
export const GraphEmbeddingAlgorithm = z.enum([
  'graph2vec',
  'feather_g',
  'netlsd',
  'geoscattering',
  'wavelet_characteristic',
  'ige',
  'ldp',
  'gl2vec',
  'sf',
  'fgsd',
]);

export type GraphEmbeddingAlgorithm = z.infer<typeof GraphEmbeddingAlgorithm>;

/**
 * Graph Structure Schema
 */
export const GraphStructure = z.object({
  // Nodes in the graph
  nodes: z.array(z.object({
    id: z.string(),
    features: z.array(z.number()).optional().describe("Node feature vector"),
    label: z.string().optional().describe("Optional node label/class"),
    metadata: z.record(z.any()).optional().describe("Additional node metadata"),
  })),

  // Edges in the graph
  edges: z.array(z.object({
    source: z.string(),
    target: z.string(),
    weight: z.number().optional().describe("Optional edge weight"),
    attributes: z.record(z.any()).optional().describe("Additional edge attributes"),
  })),

  // Graph metadata
  directed: z.boolean().default(false).describe("Whether graph is directed"),
  weighted: z.boolean().default(false).describe("Whether graph is weighted"),
});

export type GraphStructure = z.infer<typeof GraphStructure>;

/**
 * Node Embedding Request Schema
 */
export const NodeEmbeddingRequest = z.object({
  // Algorithm to use
  algorithm: NodeEmbeddingAlgorithm,

  // Graph structure
  graph: GraphStructure,

  // Algorithm parameters
  parameters: z.object({
    dimensions: z.number().int().positive().default(128).describe("Embedding dimensionality"),
    walk_length: z.number().int().positive().optional().describe("Random walk length"),
    walk_number: z.number().int().positive().optional().describe("Number of random walks"),
    window_size: z.number().int().positive().optional().describe("Context window size"),
    p: z.number().positive().optional().describe("Node2Vec return parameter"),
    q: z.number().positive().optional().describe("Node2Vec in-out parameter"),
    epochs: z.number().int().positive().optional().describe("Training epochs"),
    seed: z.number().int().optional().describe("Random seed"),
  }).optional().describe("Algorithm-specific parameters"),

  // Timeout in milliseconds (MANDATORY)
  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(600000, "Timeout cannot exceed 10 minutes")
    .describe("Operation timeout in milliseconds"),

  // Optional correlation ID
  correlation_id: z.string().uuid().optional(),
});

export type NodeEmbeddingRequest = z.infer<typeof NodeEmbeddingRequest>;

/**
 * Node Embedding Response Schema
 */
export const NodeEmbeddingResponse = z.object({
  // Success indicator
  success: z.boolean(),

  // Embeddings keyed by node ID
  embeddings: z.record(z.array(z.number())).optional().describe("Node ID -> embedding vector"),

  // Dimensionality of embeddings
  dimensions: z.number().int().positive().describe("Embedding dimensionality"),

  // Algorithm used
  algorithm: NodeEmbeddingAlgorithm,

  // Execution metadata
  metadata: z.object({
    num_nodes: z.number().int().positive(),
    training_time_ms: z.number().positive(),
    convergence: z.boolean().optional(),
    epochs_completed: z.number().int().optional(),
  }),

  // Error details (if failed)
  error: z.string().optional(),

  // UTC timestamp
  timestamp: z.string().datetime(),

  correlation_id: z.string().uuid().optional(),
});

export type NodeEmbeddingResponse = z.infer<typeof NodeEmbeddingResponse>;

/**
 * Community Detection Request Schema
 */
export const CommunityDetectionRequest = z.object({
  // Algorithm to use
  algorithm: CommunityAlgorithm,

  // Graph structure
  graph: GraphStructure,

  // Algorithm parameters
  parameters: z.object({
    resolution: z.number().positive().optional().describe("Resolution parameter"),
    iterations: z.number().int().positive().optional().describe("Number of iterations"),
    seed: z.number().int().optional().describe("Random seed"),
  }).optional(),

  // Timeout
  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(600000, "Timeout cannot exceed 10 minutes"),

  correlation_id: z.string().uuid().optional(),
});

export type CommunityDetectionRequest = z.infer<typeof CommunityDetectionRequest>;

/**
 * Community Detection Response Schema
 */
export const CommunityDetectionResponse = z.object({
  // Success indicator
  success: z.boolean(),

  // Community assignments: node_id -> community_id
  memberships: z.record(z.number().int()).optional().describe("Node ID to community ID mapping"),

  // Alternative: overlapping communities (node_id -> [community_ids])
  overlapping_memberships: z.record(z.array(z.number().int())).optional(),

  // Number of communities detected
  num_communities: z.number().int().positive().optional(),

  // Community sizes
  community_sizes: z.record(z.number().int()).optional().describe("community_id -> size"),

  // Algorithm used
  algorithm: CommunityAlgorithm,

  // Execution metadata
  metadata: z.object({
    detection_time_ms: z.number().positive(),
    modularity: z.number().optional().describe("Graph modularity score"),
    coverage: z.number().optional().describe("Community coverage"),
  }),

  error: z.string().optional(),

  timestamp: z.string().datetime(),

  correlation_id: z.string().uuid().optional(),
});

export type CommunityDetectionResponse = z.infer<typeof CommunityDetectionResponse>;

/**
 * Graph Embedding Request Schema
 */
export const GraphEmbeddingRequest = z.object({
  // Algorithm to use
  algorithm: GraphEmbeddingAlgorithm,

  // Graph structures (note: graph embedders work on multiple graphs)
  graphs: z.array(GraphStructure).min(1),

  // Algorithm parameters
  parameters: z.object({
    dimensions: z.number().int().positive().default(128),
    wl_iterations: z.number().int().positive().optional().describe("Weisfeiler-Lehman iterations"),
    epochs: z.number().int().positive().optional(),
    learning_rate: z.number().positive().optional(),
    seed: z.number().int().optional(),
    scales: z.number().int().positive().optional().describe("Number of scales for wavelet methods"),
  }).optional(),

  // Timeout (longer for graph embeddings)
  timeout_ms: z.number()
    .int()
    .positive()
    .max(900000, "Timeout cannot exceed 15 minutes"),

  correlation_id: z.string().uuid().optional(),
});

export type GraphEmbeddingRequest = z.infer<typeof GraphEmbeddingRequest>;

/**
 * Graph Embedding Response Schema
 */
export const GraphEmbeddingResponse = z.object({
  success: z.boolean(),

  // Embeddings: graph_index -> embedding vector
  embeddings: z.record(z.array(z.number())).optional(),

  dimensions: z.number().int().positive(),

  algorithm: GraphEmbeddingAlgorithm,

  metadata: z.object({
    num_graphs: z.number().int().positive(),
    training_time_ms: z.number().positive(),
    epochs_completed: z.number().int().optional(),
  }),

  error: z.string().optional(),

  timestamp: z.string().datetime(),

  correlation_id: z.string().uuid().optional(),
});

export type GraphEmbeddingResponse = z.infer<typeof GraphEmbeddingResponse>;

/**
 * Combined Graph Analysis Request
 */
export const GraphAnalysisRequest = z.object({
  graph: GraphStructure,

  // Analysis types to perform
  analyses: z.array(z.enum([
    'node_embeddings',
    'community_detection',
    'graph_statistics',
    'centrality',
  ])).min(1),

  // Algorithm selections (optional, uses defaults if not specified)
  node_embedding_algorithm: NodeEmbeddingAlgorithm.optional(),
  community_algorithm: CommunityAlgorithm.optional(),

  // Parameters
  parameters: z.object({
    embedding_dimensions: z.number().int().positive().default(128),
    top_k_nodes: z.number().int().positive().default(10).describe("For centrality rankings"),
  }).optional(),

  timeout_ms: z.number()
    .int()
    .positive()
    .max(900000),

  correlation_id: z.string().uuid().optional(),
});

export type GraphAnalysisRequest = z.infer<typeof GraphAnalysisRequest>;

/**
 * Combined Graph Analysis Response
 */
export const GraphAnalysisResponse = z.object({
  success: z.boolean(),

  results: z.object({
    node_embeddings: z.record(z.array(z.number())).optional(),
    communities: z.record(z.number().int()).optional(),
    graph_statistics: z.object({
      num_nodes: z.number().int(),
      num_edges: z.number().int(),
      density: z.number(),
      is_connected: z.boolean(),
      avg_degree: z.number().optional(),
      avg_clustering: z.number().optional(),
    }).optional(),
    centrality: z.object({
      top_degree: z.array(z.object({
        node_id: z.string(),
        score: z.number(),
      })).optional(),
      top_betweenness: z.array(z.object({
        node_id: z.string(),
        score: z.number(),
      })).optional(),
      top_pagerank: z.array(z.object({
        node_id: z.string(),
        score: z.number(),
      })).optional(),
    }).optional(),
  }).optional(),

  algorithms_used: z.object({
    node_embedding: NodeEmbeddingAlgorithm.optional(),
    community_detection: CommunityAlgorithm.optional(),
  }).optional(),

  execution_time_ms: z.number().positive(),

  error: z.string().optional(),

  timestamp: z.string().datetime(),

  correlation_id: z.string().uuid().optional(),
});

export type GraphAnalysisResponse = z.infer<typeof GraphAnalysisResponse>;

/**
 * Validation helper functions
 */
export function validateNodeEmbeddingRequest(data: unknown) {
  return NodeEmbeddingRequest.safeParse(data);
}

export function validateCommunityDetectionRequest(data: unknown) {
  return CommunityDetectionRequest.safeParse(data);
}

export function validateGraphEmbeddingRequest(data: unknown) {
  return GraphEmbeddingRequest.safeParse(data);
}

export function validateGraphAnalysisRequest(data: unknown) {
  return GraphAnalysisRequest.safeParse(data);
}
