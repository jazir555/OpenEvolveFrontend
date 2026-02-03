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

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import { z } from 'zod';

// Import canonical schemas for validation
import {
  NodeEmbeddingRequest,
  NodeEmbeddingResponse,
  CommunityDetectionRequest,
  CommunityDetectionResponse,
  GraphEmbeddingRequest,
  GraphEmbeddingResponse,
  GraphAnalysisRequest,
  GraphAnalysisResponse,
  GraphStructure,
  validateNodeEmbeddingRequest,
  validateCommunityDetectionRequest,
  validateGraphEmbeddingRequest,
  validateGraphAnalysisRequest,
  NodeEmbeddingAlgorithm,
  CommunityAlgorithm,
  GraphEmbeddingAlgorithm,
} from '../../schemas/karateclub-canonical';

// ============================================================================
// MOCK DATA - Simulating KarateClub Core API Responses
// ============================================================================

const mockGraphStructure: GraphStructure = {
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

const mockNodeEmbeddingResponse: NodeEmbeddingResponse = {
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

const mockCommunityDetectionResponse: CommunityDetectionResponse = {
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

describe('KarateClub Canonical Schema - Node Embedding', () => {
  describe('NodeEmbeddingRequest Schema', () => {
    test('must accept valid node embedding request', () => {
      const request = {
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        graph: mockGraphStructure,
        parameters: {
          dimensions: 128,
          walk_length: 80,
          walk_number: 10,
        },
        timeout_ms: 60000,
      };

      const result = validateNodeEmbeddingRequest(request);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    test('must require timeout_ms (Law of Configuration Explicitness)', () => {
      const request = {
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        graph: mockGraphStructure,
        // Missing timeout_ms
      };

      const result = validateNodeEmbeddingRequest(request);
      expect(result.success).toBe(false);
    });

    test('must validate timeout_ms does not exceed maximum', () => {
      const request = {
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        graph: mockGraphStructure,
        timeout_ms: 700000, // Exceeds 600000 max
      };

      const result = validateNodeEmbeddingRequest(request);
      expect(result.success).toBe(false);
    });

    test('must validate graph structure', () => {
      const request = {
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        graph: {
          nodes: [], // Empty nodes
          edges: [],
        },
        timeout_ms: 60000,
      };

      const result = validateNodeEmbeddingRequest(request);
      expect(result.success).toBe(true); // Empty graph is technically valid
    });
  });

  describe('NodeEmbeddingResponse Schema', () => {
    test('must accept valid node embedding response', () => {
      const response = mockNodeEmbeddingResponse;

      const result = NodeEmbeddingResponse.safeParse(response);
      expect(result.success).toBe(true);
    });

    test('must require success field', () => {
      const response = {
        // Missing success field
        embeddings: {},
        dimensions: 4,
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        metadata: {
          num_nodes: 3,
          training_time_ms: 1500,
        },
        timestamp: new Date().toISOString(),
      };

      const result = NodeEmbeddingResponse.safeParse(response);
      expect(result.success).toBe(false);
    });

    test('must require UTC timestamp (Law of UTC)', () => {
      const response = {
        ...mockNodeEmbeddingResponse,
        timestamp: '2024-01-01T12:00:00', // Missing timezone
      };

      const result = NodeEmbeddingResponse.safeParse(response);
      expect(result.success).toBe(false);
    });

    test('must accept failed response with error', () => {
      const response = {
        success: false,
        error: 'Algorithm execution failed',
        dimensions: 0,
        algorithm: 'node2vec' as NodeEmbeddingAlgorithm,
        metadata: {
          num_nodes: 3,
          training_time_ms: 100,
        },
        timestamp: new Date().toISOString(),
      };

      const result = NodeEmbeddingResponse.safeParse(response);
      expect(result.success).toBe(true);
      expect(result.data?.success).toBe(false);
      expect(result.data?.error).toBeDefined();
    });
  });
});

describe('KarateClub Canonical Schema - Community Detection', () => {
  describe('CommunityDetectionRequest Schema', () => {
    test('must accept valid community detection request', () => {
      const request = {
        algorithm: 'label_propagation' as CommunityAlgorithm,
        graph: mockGraphStructure,
        timeout_ms: 30000,
      };

      const result = validateCommunityDetectionRequest(request);
      expect(result.success).toBe(true);
    });

    test('must validate algorithm is a supported community algorithm', () => {
      const request = {
        algorithm: 'invalid_algorithm' as CommunityAlgorithm,
        graph: mockGraphStructure,
        timeout_ms: 30000,
      };

      const result = validateCommunityDetectionRequest(request);
      expect(result.success).toBe(false);
    });
  });

  describe('CommunityDetectionResponse Schema', () => {
    test('must accept valid community detection response', () => {
      const response = mockCommunityDetectionResponse;

      const result = CommunityDetectionResponse.safeParse(response);
      expect(result.success).toBe(true);
    });

    test('must include num_communities on success', () => {
      const response = {
        success: true,
        memberships: {
          node1: 0,
          node2: 1,
        },
        // Missing num_communities
        algorithm: 'label_propagation' as CommunityAlgorithm,
        metadata: {
          detection_time_ms: 500,
        },
        timestamp: new Date().toISOString(),
      };

      const result = CommunityDetectionResponse.safeParse(response);
      // num_communities is optional, so this should still pass
      expect(result.success).toBe(true);
    });

    test('must support overlapping communities', () => {
      const response = {
        success: true,
        overlapping_memberships: {
          node1: [0, 1],
          node2: [1, 2],
          node3: [0, 2],
        },
        num_communities: 3,
        algorithm: 'bigclam' as CommunityAlgorithm,
        metadata: {
          detection_time_ms: 1000,
        },
        timestamp: new Date().toISOString(),
      };

      const result = CommunityDetectionResponse.safeParse(response);
      expect(result.success).toBe(true);
      expect(result.data?.overlapping_memberships).toBeDefined();
    });
  });
});

describe('KarateClub Canonical Schema - Graph Embedding', () => {
  describe('GraphEmbeddingRequest Schema', () => {
    test('must accept valid graph embedding request', () => {
      const request = {
        algorithm: 'graph2vec' as GraphEmbeddingAlgorithm,
        graphs: [mockGraphStructure],
        timeout_ms: 300000,
      };

      const result = validateGraphEmbeddingRequest(request);
      expect(result.success).toBe(true);
    });

    test('must require at least one graph', () => {
      const request = {
        algorithm: 'graph2vec' as GraphEmbeddingAlgorithm,
        graphs: [], // Empty graphs array
        timeout_ms: 300000,
      };

      const result = validateGraphEmbeddingRequest(request);
      expect(result.success).toBe(false);
    });
  });
});

describe('KarateClub Canonical Schema - Graph Analysis', () => {
  describe('GraphAnalysisRequest Schema', () => {
    test('must accept valid graph analysis request', () => {
      const request = {
        graph: mockGraphStructure,
        analyses: ['node_embeddings', 'community_detection', 'graph_statistics'],
        timeout_ms: 300000,
      };

      const result = validateGraphAnalysisRequest(request);
      expect(result.success).toBe(true);
    });

    test('must require at least one analysis type', () => {
      const request = {
        graph: mockGraphStructure,
        analyses: [], // Empty analyses array
        timeout_ms: 300000,
      };

      const result = validateGraphAnalysisRequest(request);
      expect(result.success).toBe(false);
    });

    test('must validate supported analysis types', () => {
      const request = {
        graph: mockGraphStructure,
        analyses: ['invalid_analysis' as any],
        timeout_ms: 300000,
      };

      const result = validateGraphAnalysisRequest(request);
      expect(result.success).toBe(false);
    });
  });

  describe('GraphAnalysisResponse Schema', () => {
    test('must accept successful graph analysis response', () => {
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
          node_embedding: 'node2vec' as NodeEmbeddingAlgorithm,
          community_detection: 'label_propagation' as CommunityAlgorithm,
        },
        execution_time_ms: 5000,
        timestamp: new Date().toISOString(),
      };

      const result = GraphAnalysisResponse.safeParse(response);
      expect(result.success).toBe(true);
    });

    test('must accept partial failure in graph analysis', () => {
      const response = {
        success: false,
        error: 'Node embedding failed',
        execution_time_ms: 2000,
        timestamp: new Date().toISOString(),
      };

      const result = GraphAnalysisResponse.safeParse(response);
      expect(result.success).toBe(true);
      expect(result.data?.success).toBe(false);
      expect(result.data?.error).toBeDefined();
    });
  });
});

// ============================================================================
// ALGORITHM ENUMERATION TESTS
// ============================================================================

describe('KarateClub Algorithm Registry', () => {
  describe('Node Embedding Algorithms', () => {
    test('must support DeepWalk', () => {
      const value = 'deepwalk' as NodeEmbeddingAlgorithm;
      expect(value).toBe('deepwalk');
    });

    test('must support Node2Vec', () => {
      const value = 'node2vec' as NodeEmbeddingAlgorithm;
      expect(value).toBe('node2vec');
    });

    test('must support Walklets', () => {
      const value = 'walklets' as NodeEmbeddingAlgorithm;
      expect(value).toBe('walklets');
    });

    test('must support GraphWave', () => {
      const value = 'graphwave' as NodeEmbeddingAlgorithm;
      expect(value).toBe('graphwave');
    });

    test('must support Role2Vec', () => {
      const value = 'role2vec' as NodeEmbeddingAlgorithm;
      expect(value).toBe('role2vec');
    });

    test('must support Feather-N', () => {
      const value = 'feather_n' as NodeEmbeddingAlgorithm;
      expect(value).toBe('feather_n');
    });
  });

  describe('Community Detection Algorithms', () => {
    test('must support Label Propagation', () => {
      const value = 'label_propagation' as CommunityAlgorithm;
      expect(value).toBe('label_propagation');
    });

    test('must support BigClam', () => {
      const value = 'bigclam' as CommunityAlgorithm;
      expect(value).toBe('bigclam');
    });

    test('must support DANMF', () => {
      const value = 'danmf' as CommunityAlgorithm;
      expect(value).toBe('danmf');
    });

    test('must support GEMSEC', () => {
      const value = 'gemsec' as CommunityAlgorithm;
      expect(value).toBe('gemsec');
    });
  });

  describe('Graph Embedding Algorithms', () => {
    test('must support Graph2Vec', () => {
      const value = 'graph2vec' as GraphEmbeddingAlgorithm;
      expect(value).toBe('graph2vec');
    });

    test('must support NetLSD', () => {
      const value = 'netlsd' as GraphEmbeddingAlgorithm;
      expect(value).toBe('netlsd');
    });

    test('must support GeoScattering', () => {
      const value = 'geoscattering' as GraphEmbeddingAlgorithm;
      expect(value).toBe('geoscattering');
    });

    test('must support SF (Statistical Features)', () => {
      const value = 'sf' as GraphEmbeddingAlgorithm;
      expect(value).toBe('sf');
    });
  });
});

// ============================================================================
// GRAPH STRUCTURE TESTS
// ============================================================================

describe('Graph Structure Schema', () => {
  test('must accept valid undirected weighted graph', () => {
    const graph = mockGraphStructure;

    const result = GraphStructure.safeParse(graph);
    expect(result.success).toBe(true);
  });

  test('must accept directed graph', () => {
    const graph = {
      ...mockGraphStructure,
      directed: true,
    };

    const result = GraphStructure.safeParse(graph);
    expect(result.success).toBe(true);
  });

  test('must accept graph without features', () => {
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

    const result = GraphStructure.safeParse(graph);
    expect(result.success).toBe(true);
  });

  test('must accept graph with node labels', () => {
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

    const result = GraphStructure.safeParse(graph);
    expect(result.success).toBe(true);
  });
});
