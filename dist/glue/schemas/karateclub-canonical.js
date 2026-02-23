"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.GraphAnalysisResponse = exports.GraphAnalysisRequest = exports.GraphEmbeddingResponse = exports.GraphEmbeddingRequest = exports.CommunityDetectionResponse = exports.CommunityDetectionRequest = exports.NodeEmbeddingResponse = exports.NodeEmbeddingRequest = exports.GraphStructure = exports.GraphEmbeddingAlgorithm = exports.CommunityAlgorithm = exports.NodeEmbeddingAlgorithm = exports.AlgorithmCategory = void 0;
exports.validateNodeEmbeddingRequest = validateNodeEmbeddingRequest;
exports.validateCommunityDetectionRequest = validateCommunityDetectionRequest;
exports.validateGraphEmbeddingRequest = validateGraphEmbeddingRequest;
exports.validateGraphAnalysisRequest = validateGraphAnalysisRequest;
const zod_1 = require("zod");
/**
 * Algorithm Categories
 */
exports.AlgorithmCategory = zod_1.z.enum([
    'community',
    'node_embedding',
    'graph_embedding',
]);
/**
 * Supported Node Embedding Algorithms (32)
 */
exports.NodeEmbeddingAlgorithm = zod_1.z.enum([
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
/**
 * Supported Community Detection Algorithms (10)
 */
exports.CommunityAlgorithm = zod_1.z.enum([
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
/**
 * Supported Graph Embedding Algorithms (10)
 */
exports.GraphEmbeddingAlgorithm = zod_1.z.enum([
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
/**
 * Graph Structure Schema
 */
exports.GraphStructure = zod_1.z.object({
    // Nodes in the graph
    nodes: zod_1.z.array(zod_1.z.object({
        id: zod_1.z.string(),
        features: zod_1.z.array(zod_1.z.number()).optional().describe("Node feature vector"),
        label: zod_1.z.string().optional().describe("Optional node label/class"),
        metadata: zod_1.z.record(zod_1.z.any()).optional().describe("Additional node metadata"),
    })),
    // Edges in the graph
    edges: zod_1.z.array(zod_1.z.object({
        source: zod_1.z.string(),
        target: zod_1.z.string(),
        weight: zod_1.z.number().optional().describe("Optional edge weight"),
        attributes: zod_1.z.record(zod_1.z.any()).optional().describe("Additional edge attributes"),
    })),
    // Graph metadata
    directed: zod_1.z.boolean().default(false).describe("Whether graph is directed"),
    weighted: zod_1.z.boolean().default(false).describe("Whether graph is weighted"),
});
/**
 * Node Embedding Request Schema
 */
exports.NodeEmbeddingRequest = zod_1.z.object({
    // Algorithm to use
    algorithm: exports.NodeEmbeddingAlgorithm,
    // Graph structure
    graph: exports.GraphStructure,
    // Algorithm parameters
    parameters: zod_1.z.object({
        dimensions: zod_1.z.number().int().positive().default(128).describe("Embedding dimensionality"),
        walk_length: zod_1.z.number().int().positive().optional().describe("Random walk length"),
        walk_number: zod_1.z.number().int().positive().optional().describe("Number of random walks"),
        window_size: zod_1.z.number().int().positive().optional().describe("Context window size"),
        p: zod_1.z.number().positive().optional().describe("Node2Vec return parameter"),
        q: zod_1.z.number().positive().optional().describe("Node2Vec in-out parameter"),
        epochs: zod_1.z.number().int().positive().optional().describe("Training epochs"),
        seed: zod_1.z.number().int().optional().describe("Random seed"),
    }).optional().describe("Algorithm-specific parameters"),
    // Timeout in milliseconds (MANDATORY)
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(600000, "Timeout cannot exceed 10 minutes")
        .describe("Operation timeout in milliseconds"),
    // Optional correlation ID
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Node Embedding Response Schema
 */
exports.NodeEmbeddingResponse = zod_1.z.object({
    // Success indicator
    success: zod_1.z.boolean(),
    // Embeddings keyed by node ID
    embeddings: zod_1.z.record(zod_1.z.array(zod_1.z.number())).optional().describe("Node ID -> embedding vector"),
    // Dimensionality of embeddings
    dimensions: zod_1.z.number().int().positive().describe("Embedding dimensionality"),
    // Algorithm used
    algorithm: exports.NodeEmbeddingAlgorithm,
    // Execution metadata
    metadata: zod_1.z.object({
        num_nodes: zod_1.z.number().int().positive(),
        training_time_ms: zod_1.z.number().positive(),
        convergence: zod_1.z.boolean().optional(),
        epochs_completed: zod_1.z.number().int().optional(),
    }),
    // Error details (if failed)
    error: zod_1.z.string().optional(),
    // UTC timestamp
    timestamp: zod_1.z.string().datetime(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Community Detection Request Schema
 */
exports.CommunityDetectionRequest = zod_1.z.object({
    // Algorithm to use
    algorithm: exports.CommunityAlgorithm,
    // Graph structure
    graph: exports.GraphStructure,
    // Algorithm parameters
    parameters: zod_1.z.object({
        resolution: zod_1.z.number().positive().optional().describe("Resolution parameter"),
        iterations: zod_1.z.number().int().positive().optional().describe("Number of iterations"),
        seed: zod_1.z.number().int().optional().describe("Random seed"),
    }).optional(),
    // Timeout
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(600000, "Timeout cannot exceed 10 minutes"),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Community Detection Response Schema
 */
exports.CommunityDetectionResponse = zod_1.z.object({
    // Success indicator
    success: zod_1.z.boolean(),
    // Community assignments: node_id -> community_id
    memberships: zod_1.z.record(zod_1.z.number().int()).optional().describe("Node ID to community ID mapping"),
    // Alternative: overlapping communities (node_id -> [community_ids])
    overlapping_memberships: zod_1.z.record(zod_1.z.array(zod_1.z.number().int())).optional(),
    // Number of communities detected
    num_communities: zod_1.z.number().int().positive().optional(),
    // Community sizes
    community_sizes: zod_1.z.record(zod_1.z.number().int()).optional().describe("community_id -> size"),
    // Algorithm used
    algorithm: exports.CommunityAlgorithm,
    // Execution metadata
    metadata: zod_1.z.object({
        detection_time_ms: zod_1.z.number().positive(),
        modularity: zod_1.z.number().optional().describe("Graph modularity score"),
        coverage: zod_1.z.number().optional().describe("Community coverage"),
    }),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Graph Embedding Request Schema
 */
exports.GraphEmbeddingRequest = zod_1.z.object({
    // Algorithm to use
    algorithm: exports.GraphEmbeddingAlgorithm,
    // Graph structures (note: graph embedders work on multiple graphs)
    graphs: zod_1.z.array(exports.GraphStructure).min(1),
    // Algorithm parameters
    parameters: zod_1.z.object({
        dimensions: zod_1.z.number().int().positive().default(128),
        wl_iterations: zod_1.z.number().int().positive().optional().describe("Weisfeiler-Lehman iterations"),
        epochs: zod_1.z.number().int().positive().optional(),
        learning_rate: zod_1.z.number().positive().optional(),
        seed: zod_1.z.number().int().optional(),
        scales: zod_1.z.number().int().positive().optional().describe("Number of scales for wavelet methods"),
    }).optional(),
    // Timeout (longer for graph embeddings)
    timeout_ms: zod_1.z.number()
        .int()
        .positive()
        .max(900000, "Timeout cannot exceed 15 minutes"),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Graph Embedding Response Schema
 */
exports.GraphEmbeddingResponse = zod_1.z.object({
    success: zod_1.z.boolean(),
    // Embeddings: graph_index -> embedding vector
    embeddings: zod_1.z.record(zod_1.z.array(zod_1.z.number())).optional(),
    dimensions: zod_1.z.number().int().positive(),
    algorithm: exports.GraphEmbeddingAlgorithm,
    metadata: zod_1.z.object({
        num_graphs: zod_1.z.number().int().positive(),
        training_time_ms: zod_1.z.number().positive(),
        epochs_completed: zod_1.z.number().int().optional(),
    }),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Combined Graph Analysis Request
 */
exports.GraphAnalysisRequest = zod_1.z.object({
    graph: exports.GraphStructure,
    // Analysis types to perform
    analyses: zod_1.z.array(zod_1.z.enum([
        'node_embeddings',
        'community_detection',
        'graph_statistics',
        'centrality',
    ])).min(1),
    // Algorithm selections (optional, uses defaults if not specified)
    node_embedding_algorithm: exports.NodeEmbeddingAlgorithm.optional(),
    community_algorithm: exports.CommunityAlgorithm.optional(),
    // Parameters
    parameters: zod_1.z.object({
        embedding_dimensions: zod_1.z.number().int().positive().default(128),
        top_k_nodes: zod_1.z.number().int().positive().default(10).describe("For centrality rankings"),
    }).optional(),
    timeout_ms: zod_1.z.number()
        .int()
        .positive()
        .max(900000),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Combined Graph Analysis Response
 */
exports.GraphAnalysisResponse = zod_1.z.object({
    success: zod_1.z.boolean(),
    results: zod_1.z.object({
        node_embeddings: zod_1.z.record(zod_1.z.array(zod_1.z.number())).optional(),
        communities: zod_1.z.record(zod_1.z.number().int()).optional(),
        graph_statistics: zod_1.z.object({
            num_nodes: zod_1.z.number().int(),
            num_edges: zod_1.z.number().int(),
            density: zod_1.z.number(),
            is_connected: zod_1.z.boolean(),
            avg_degree: zod_1.z.number().optional(),
            avg_clustering: zod_1.z.number().optional(),
        }).optional(),
        centrality: zod_1.z.object({
            top_degree: zod_1.z.array(zod_1.z.object({
                node_id: zod_1.z.string(),
                score: zod_1.z.number(),
            })).optional(),
            top_betweenness: zod_1.z.array(zod_1.z.object({
                node_id: zod_1.z.string(),
                score: zod_1.z.number(),
            })).optional(),
            top_pagerank: zod_1.z.array(zod_1.z.object({
                node_id: zod_1.z.string(),
                score: zod_1.z.number(),
            })).optional(),
        }).optional(),
    }).optional(),
    algorithms_used: zod_1.z.object({
        node_embedding: exports.NodeEmbeddingAlgorithm.optional(),
        community_detection: exports.CommunityAlgorithm.optional(),
    }).optional(),
    execution_time_ms: zod_1.z.number().positive(),
    error: zod_1.z.string().optional(),
    timestamp: zod_1.z.string().datetime(),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Validation helper functions
 */
function validateNodeEmbeddingRequest(data) {
    return exports.NodeEmbeddingRequest.safeParse(data);
}
function validateCommunityDetectionRequest(data) {
    return exports.CommunityDetectionRequest.safeParse(data);
}
function validateGraphEmbeddingRequest(data) {
    return exports.GraphEmbeddingRequest.safeParse(data);
}
function validateGraphAnalysisRequest(data) {
    return exports.GraphAnalysisRequest.safeParse(data);
}
//# sourceMappingURL=karateclub-canonical.js.map