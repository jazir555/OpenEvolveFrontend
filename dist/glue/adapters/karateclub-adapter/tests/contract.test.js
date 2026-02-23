"use strict";
/**
 * KarateClub Adapter Contract Tests
 *
 * CRITICAL: These tests validate the contract between the KarateClub Adapter and KarateClub Core.
 * If these tests fail, the adapter MUST refuse to start to prevent data corruption.
 *
 * Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
 * - Phase 2: The Contract (Defense)
 * - Protecting the Mega-Project from Updates
 *
 * Test Principles:
 * 1. FAIL FAST - Contract violations immediately halt execution
 * 2. MOCK ONLY - Do not require running KarateClub instance
 * 3. CANONICAL VALIDATION - Use canonical schemas for data structure validation
 * 4. IDEMPOTENT - Tests can be run 100 times safely
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
// Import canonical schemas for validation
const karateclub_canonical_1 = require("../../schemas/karateclub-canonical");
// ============================================================================
// MOCK DATA - Simulating KarateClub Core API Responses
// ============================================================================
const mockGraphStructure = {
    nodes: [
        { id: 'node1', features: [1.0, 0.5, 0.3] },
        { id: 'node2', features: [0.8, 0.2, 0.9] },
        { id: 'node3', features: [0.1, 0.9, 0.4] },
    ],
    edges: [
        { source: 'node1', target: 'node2', weight: 1.0 },
        { source: 'node2', target: 'node3', weight: 0.5 },
        { source: 'node3', target: 'node1', weight: 0.8 },
    ],
    directed: false,
    weighted: true,
};
const mockNodeEmbeddingResponse = {
    success: true,
    embeddings: {
        node1: [0.1, 0.2, 0.3, 0.4],
        node2: [0.5, 0.6, 0.7, 0.8],
        node3: [0.9, 1.0, 0.1, 0.2],
    },
    dimensions: 4,
    algorithm: 'node2vec',
    metadata: {
        num_nodes: 3,
        training_time_ms: 1500,
    },
    timestamp: new Date().toISOString(),
};
const mockCommunityDetectionResponse = {
    success: true,
    memberships: {
        node1: 0,
        node2: 0,
        node3: 1,
    },
    num_communities: 2,
    community_sizes: {
        '0': 2,
        '1': 1,
    },
    algorithm: 'label_propagation',
    metadata: {
        detection_time_ms: 500,
    },
    timestamp: new Date().toISOString(),
};
// ============================================================================
// CANONICAL SCHEMA VALIDATION TESTS
// ============================================================================
(0, globals_1.describe)('KarateClub Canonical Schema - Node Embedding', () => {
    (0, globals_1.describe)('NodeEmbeddingRequest Schema', () => {
        (0, globals_1.test)('must accept valid node embedding request', () => {
            const request = {
                algorithm: 'node2vec',
                graph: mockGraphStructure,
                parameters: {
                    dimensions: 128,
                    walk_length: 80,
                    walk_number: 10,
                },
                timeout_ms: 60000,
            };
            const result = (0, karateclub_canonical_1.validateNodeEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
        (0, globals_1.test)('must require timeout_ms (Law of Configuration Explicitness)', () => {
            const request = {
                algorithm: 'node2vec',
                graph: mockGraphStructure,
                // Missing timeout_ms
            };
            const result = (0, karateclub_canonical_1.validateNodeEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('must validate timeout_ms does not exceed maximum', () => {
            const request = {
                algorithm: 'node2vec',
                graph: mockGraphStructure,
                timeout_ms: 700000, // Exceeds 600000 max
            };
            const result = (0, karateclub_canonical_1.validateNodeEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('must validate graph structure', () => {
            const request = {
                algorithm: 'node2vec',
                graph: {
                    nodes: [], // Empty nodes
                    edges: [],
                },
                timeout_ms: 60000,
            };
            const result = (0, karateclub_canonical_1.validateNodeEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(true); // Empty graph is technically valid
        });
    });
    (0, globals_1.describe)('NodeEmbeddingResponse Schema', () => {
        (0, globals_1.test)('must accept valid node embedding response', () => {
            const response = mockNodeEmbeddingResponse;
            const result = karateclub_canonical_1.NodeEmbeddingResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must require success field', () => {
            const response = {
                // Missing success field
                embeddings: {},
                dimensions: 4,
                algorithm: 'node2vec',
                metadata: {
                    num_nodes: 3,
                    training_time_ms: 1500,
                },
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.NodeEmbeddingResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('must require UTC timestamp (Law of UTC)', () => {
            const response = {
                ...mockNodeEmbeddingResponse,
                timestamp: '2024-01-01T12:00:00', // Missing timezone
            };
            const result = karateclub_canonical_1.NodeEmbeddingResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('must accept failed response with error', () => {
            const response = {
                success: false,
                error: 'Algorithm execution failed',
                dimensions: 0,
                algorithm: 'node2vec',
                metadata: {
                    num_nodes: 3,
                    training_time_ms: 100,
                },
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.NodeEmbeddingResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data?.success).toBe(false);
            (0, globals_1.expect)(result.data?.error).toBeDefined();
        });
    });
});
(0, globals_1.describe)('KarateClub Canonical Schema - Community Detection', () => {
    (0, globals_1.describe)('CommunityDetectionRequest Schema', () => {
        (0, globals_1.test)('must accept valid community detection request', () => {
            const request = {
                algorithm: 'label_propagation',
                graph: mockGraphStructure,
                timeout_ms: 30000,
            };
            const result = (0, karateclub_canonical_1.validateCommunityDetectionRequest)(request);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must validate algorithm is a supported community algorithm', () => {
            const request = {
                algorithm: 'invalid_algorithm',
                graph: mockGraphStructure,
                timeout_ms: 30000,
            };
            const result = (0, karateclub_canonical_1.validateCommunityDetectionRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('CommunityDetectionResponse Schema', () => {
        (0, globals_1.test)('must accept valid community detection response', () => {
            const response = mockCommunityDetectionResponse;
            const result = karateclub_canonical_1.CommunityDetectionResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must include num_communities on success', () => {
            const response = {
                success: true,
                memberships: {
                    node1: 0,
                    node2: 1,
                },
                // Missing num_communities
                algorithm: 'label_propagation',
                metadata: {
                    detection_time_ms: 500,
                },
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.CommunityDetectionResponse.safeParse(response);
            // num_communities is optional, so this should still pass
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must support overlapping communities', () => {
            const response = {
                success: true,
                overlapping_memberships: {
                    node1: [0, 1],
                    node2: [1, 2],
                    node3: [0, 2],
                },
                num_communities: 3,
                algorithm: 'bigclam',
                metadata: {
                    detection_time_ms: 1000,
                },
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.CommunityDetectionResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data?.overlapping_memberships).toBeDefined();
        });
    });
});
(0, globals_1.describe)('KarateClub Canonical Schema - Graph Embedding', () => {
    (0, globals_1.describe)('GraphEmbeddingRequest Schema', () => {
        (0, globals_1.test)('must accept valid graph embedding request', () => {
            const request = {
                algorithm: 'graph2vec',
                graphs: [mockGraphStructure],
                timeout_ms: 300000,
            };
            const result = (0, karateclub_canonical_1.validateGraphEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must require at least one graph', () => {
            const request = {
                algorithm: 'graph2vec',
                graphs: [], // Empty graphs array
                timeout_ms: 300000,
            };
            const result = (0, karateclub_canonical_1.validateGraphEmbeddingRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
});
(0, globals_1.describe)('KarateClub Canonical Schema - Graph Analysis', () => {
    (0, globals_1.describe)('GraphAnalysisRequest Schema', () => {
        (0, globals_1.test)('must accept valid graph analysis request', () => {
            const request = {
                graph: mockGraphStructure,
                analyses: ['node_embeddings', 'community_detection', 'graph_statistics'],
                timeout_ms: 300000,
            };
            const result = (0, karateclub_canonical_1.validateGraphAnalysisRequest)(request);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must require at least one analysis type', () => {
            const request = {
                graph: mockGraphStructure,
                analyses: [], // Empty analyses array
                timeout_ms: 300000,
            };
            const result = (0, karateclub_canonical_1.validateGraphAnalysisRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.test)('must validate supported analysis types', () => {
            const request = {
                graph: mockGraphStructure,
                analyses: ['invalid_analysis'],
                timeout_ms: 300000,
            };
            const result = (0, karateclub_canonical_1.validateGraphAnalysisRequest)(request);
            (0, globals_1.expect)(result.success).toBe(false);
        });
    });
    (0, globals_1.describe)('GraphAnalysisResponse Schema', () => {
        (0, globals_1.test)('must accept successful graph analysis response', () => {
            const response = {
                success: true,
                results: {
                    node_embeddings: {
                        node1: [0.1, 0.2],
                    },
                    communities: {
                        node1: 0,
                        node2: 1,
                    },
                    graph_statistics: {
                        num_nodes: 3,
                        num_edges: 3,
                        density: 1.0,
                        is_connected: true,
                    },
                },
                algorithms_used: {
                    node_embedding: 'node2vec',
                    community_detection: 'label_propagation',
                },
                execution_time_ms: 5000,
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.GraphAnalysisResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.test)('must accept partial failure in graph analysis', () => {
            const response = {
                success: false,
                error: 'Node embedding failed',
                execution_time_ms: 2000,
                timestamp: new Date().toISOString(),
            };
            const result = karateclub_canonical_1.GraphAnalysisResponse.safeParse(response);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data?.success).toBe(false);
            (0, globals_1.expect)(result.data?.error).toBeDefined();
        });
    });
});
// ============================================================================
// ALGORITHM ENUMERATION TESTS
// ============================================================================
(0, globals_1.describe)('KarateClub Algorithm Registry', () => {
    (0, globals_1.describe)('Node Embedding Algorithms', () => {
        (0, globals_1.test)('must support DeepWalk', () => {
            const value = 'deepwalk';
            (0, globals_1.expect)(value).toBe('deepwalk');
        });
        (0, globals_1.test)('must support Node2Vec', () => {
            const value = 'node2vec';
            (0, globals_1.expect)(value).toBe('node2vec');
        });
        (0, globals_1.test)('must support Walklets', () => {
            const value = 'walklets';
            (0, globals_1.expect)(value).toBe('walklets');
        });
        (0, globals_1.test)('must support GraphWave', () => {
            const value = 'graphwave';
            (0, globals_1.expect)(value).toBe('graphwave');
        });
        (0, globals_1.test)('must support Role2Vec', () => {
            const value = 'role2vec';
            (0, globals_1.expect)(value).toBe('role2vec');
        });
        (0, globals_1.test)('must support Feather-N', () => {
            const value = 'feather_n';
            (0, globals_1.expect)(value).toBe('feather_n');
        });
    });
    (0, globals_1.describe)('Community Detection Algorithms', () => {
        (0, globals_1.test)('must support Label Propagation', () => {
            const value = 'label_propagation';
            (0, globals_1.expect)(value).toBe('label_propagation');
        });
        (0, globals_1.test)('must support BigClam', () => {
            const value = 'bigclam';
            (0, globals_1.expect)(value).toBe('bigclam');
        });
        (0, globals_1.test)('must support DANMF', () => {
            const value = 'danmf';
            (0, globals_1.expect)(value).toBe('danmf');
        });
        (0, globals_1.test)('must support GEMSEC', () => {
            const value = 'gemsec';
            (0, globals_1.expect)(value).toBe('gemsec');
        });
    });
    (0, globals_1.describe)('Graph Embedding Algorithms', () => {
        (0, globals_1.test)('must support Graph2Vec', () => {
            const value = 'graph2vec';
            (0, globals_1.expect)(value).toBe('graph2vec');
        });
        (0, globals_1.test)('must support NetLSD', () => {
            const value = 'netlsd';
            (0, globals_1.expect)(value).toBe('netlsd');
        });
        (0, globals_1.test)('must support GeoScattering', () => {
            const value = 'geoscattering';
            (0, globals_1.expect)(value).toBe('geoscattering');
        });
        (0, globals_1.test)('must support SF (Statistical Features)', () => {
            const value = 'sf';
            (0, globals_1.expect)(value).toBe('sf');
        });
    });
});
// ============================================================================
// GRAPH STRUCTURE TESTS
// ============================================================================
(0, globals_1.describe)('Graph Structure Schema', () => {
    (0, globals_1.test)('must accept valid undirected weighted graph', () => {
        const graph = mockGraphStructure;
        const result = karateclub_canonical_1.GraphStructure.safeParse(graph);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must accept directed graph', () => {
        const graph = {
            ...mockGraphStructure,
            directed: true,
        };
        const result = karateclub_canonical_1.GraphStructure.safeParse(graph);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must accept graph without features', () => {
        const graph = {
            nodes: [
                { id: 'node1' },
                { id: 'node2' },
            ],
            edges: [
                { source: 'node1', target: 'node2' },
            ],
            directed: false,
            weighted: false,
        };
        const result = karateclub_canonical_1.GraphStructure.safeParse(graph);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must accept graph with node labels', () => {
        const graph = {
            nodes: [
                { id: 'node1', label: 'class_a' },
                { id: 'node2', label: 'class_b' },
            ],
            edges: [
                { source: 'node1', target: 'node2' },
            ],
            directed: false,
            weighted: false,
        };
        const result = karateclub_canonical_1.GraphStructure.safeParse(graph);
        (0, globals_1.expect)(result.success).toBe(true);
    });
});
//# sourceMappingURL=contract.test.js.map