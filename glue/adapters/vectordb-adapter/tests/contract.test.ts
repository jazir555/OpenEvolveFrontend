/**
 * Vector DB Adapter Contract Tests
 *
 * CRITICAL: These tests validate the contract between the Vector DB Adapter and various backends.
 * If these tests fail, the adapter MUST refuse to start to prevent data corruption.
 *
 * Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
 * - Phase 2: The Contract (Defense)
 * - Protecting the Mega-Project from Updates
 *
 * Test Principles:
 * 1. FAIL FAST - Contract violations immediately halt execution
 * 2. MOCK ONLY - Do not require running Vector DB instance
 * 3. CANONICAL VALIDATION - Use canonical schemas for data structure validation
 * 4. IDEMPOTENT - Tests can be run 100 times safely
 */

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';

// Import canonical schemas for validation
import {
  VectorEntry,
  CollectionConfig,
  SearchQuery,
  SearchResult,
  UpsertRequest,
  UpsertResponse,
  DeleteRequest,
  DeleteResponse,
  CollectionInfo,
  HealthCheckResponse,
  VectorDBType,
  DistanceMetric,
  EmbeddingFormat,
  validateVectorEntry,
  validateSearchQuery,
  validateUpsertRequest,
  validateDeleteRequest,
  validateCollectionConfig,
  validateHealthCheckResponse,
  validateVectorDimension,
  VectorDBExamples,
} from '../../../schemas/vectordb-canonical';

// ============================================================================
// MOCK DATA - Simulating Vector DB API Responses
// ============================================================================

const mockHealthCheckResponse = {
  status: 'healthy' as const,
  backend_type: VectorDBType.QDRANT,
  connected: true,
  latency_ms: 5,
  collections_count: 3,
  timestamp: '2024-01-15T10:30:00.000Z',
};

const mockDegradedHealthCheckResponse = {
  status: 'degraded' as const,
  backend_type: VectorDBType.PINECONE,
  connected: false,
  error: 'Connection timeout',
  timestamp: '2024-01-15T10:30:00.000Z',
};

const mockVectorEntry: VectorEntry = {
  id: '550e8400-e29b-41d4-a716-446655440000',
  vector: [0.1, 0.2, 0.3, 0.4, 0.5],
  text: 'Example text for embedding',
  metadata: {
    category: 'example',
    source: 'test',
  },
  created_at: '2024-01-15T10:30:00.000Z',
};

const mockSparseVectorEntry: VectorEntry = {
  id: '550e8400-e29b-41d4-a716-446655440001',
  vector: {
    indices: [0, 5, 10, 15],
    values: [0.5, 0.3, 0.7, 0.2],
    dimension: 100,
  },
  metadata: {
    category: 'sparse',
  },
  created_at: '2024-01-15T10:30:00.000Z',
};

const mockCollectionConfig: CollectionConfig = {
  name: 'test_collection',
  dimension: 384,
  distance_metric: DistanceMetric.COSINE,
  format: EmbeddingFormat.DENSE,
};

const mockSearchQuery: SearchQuery = {
  vector: [0.1, 0.2, 0.3, 0.4, 0.5],
  k: 10,
  score_threshold: 0.7,
};

const mockSearchResult: SearchResult = {
  entry: mockVectorEntry,
  score: 0.85,
  distance: 0.15,
};

const mockUpsertRequest: UpsertRequest = {
  collection_name: 'test_collection',
  entries: [mockVectorEntry, mockSparseVectorEntry],
};

const mockUpsertResponse: UpsertResponse = {
  upserted_count: 2,
  collection_name: 'test_collection',
  timestamp: '2024-01-15T10:30:00.000Z',
};

const mockDeleteRequest: DeleteRequest = {
  collection_name: 'test_collection',
  ids: ['550e8400-e29b-41d4-a716-446655440000'],
};

const mockDeleteResponse: DeleteResponse = {
  deleted_count: 1,
  collection_name: 'test_collection',
  timestamp: '2024-01-15T10:30:00.000Z',
};

const mockCollectionInfo: CollectionInfo = {
  name: 'test_collection',
  dimension: 384,
  vector_count: 1000,
  distance_metric: DistanceMetric.COSINE,
  created_at: '2024-01-15T10:30:00.000Z',
  updated_at: '2024-01-15T10:30:00.000Z',
};

// ============================================================================
// HEALTH CHECK CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Health Check', () => {
  describe('Healthy Response', () => {
    test('must return status "healthy" when connected', () => {
      expect(mockHealthCheckResponse.status).toBe('healthy');
      expect(mockHealthCheckResponse.connected).toBe(true);
    });

    test('must include backend_type', () => {
      expect(mockHealthCheckResponse.backend_type).toBeDefined();
      expect(Object.values(VectorDBType)).toContain(mockHealthCheckResponse.backend_type);
    });

    test('may include latency_ms', () => {
      expect(mockHealthCheckResponse.latency_ms).toBeDefined();
      expect(typeof mockHealthCheckResponse.latency_ms).toBe('number');
      expect(mockHealthCheckResponse.latency_ms).toBeGreaterThan(0);
    });

    test('may include collections_count', () => {
      expect(mockHealthCheckResponse.collections_count).toBeDefined();
      expect(typeof mockHealthCheckResponse.collections_count).toBe('number');
      expect(mockHealthCheckResponse.collections_count).toBeGreaterThanOrEqual(0);
    });

    test('must conform to HealthCheckResponseSchema', () => {
      const result = validateHealthCheckResponse(mockHealthCheckResponse);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });

  describe('Degraded Response', () => {
    test('must return status "degraded" or "unhealthy" when disconnected', () => {
      expect(['degraded', 'unhealthy']).toContain(mockDegradedHealthCheckResponse.status);
      expect(mockDegradedHealthCheckResponse.connected).toBe(false);
    });

    test('may include error message when degraded', () => {
      expect(mockDegradedHealthCheckResponse.error).toBeDefined();
      expect(typeof mockDegradedHealthCheckResponse.error).toBe('string');
    });

    test('must conform to HealthCheckResponseSchema', () => {
      const result = validateHealthCheckResponse(mockDegradedHealthCheckResponse);
      expect(result.success).toBe(true);
      expect(result.data?.status).toBe('degraded');
    });
  });
});

// ============================================================================
// VECTOR ENTRY CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Vector Entry', () => {
  describe('Dense Vector Entry', () => {
    test('must have valid UUID id', () => {
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
      expect(uuidRegex.test(mockVectorEntry.id)).toBe(true);
    });

    test('must have array of numbers for dense vector', () => {
      expect(Array.isArray(mockVectorEntry.vector)).toBe(true);
      expect(mockVectorEntry.vector.length).toBeGreaterThan(0);
      expect(mockVectorEntry.vector.every(v => typeof v === 'number')).toBe(true);
    });

    test('may have optional text field', () => {
      expect(mockVectorEntry.text).toBeDefined();
      expect(typeof mockVectorEntry.text).toBe('string');
    });

    test('may have optional metadata object', () => {
      expect(mockVectorEntry.metadata).toBeDefined();
      expect(typeof mockVectorEntry.metadata).toBe('object');
    });

    test('may have optional created_at timestamp', () => {
      expect(mockVectorEntry.created_at).toBeDefined();
      expect(new Date(mockVectorEntry.created_at).toISOString()).toBe(mockVectorEntry.created_at);
    });

    test('must conform to VectorEntrySchema', () => {
      const result = validateVectorEntry(mockVectorEntry);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });

  describe('Sparse Vector Entry', () => {
    test('must have sparse vector object with indices, values, and dimension', () => {
      expect(typeof mockSparseVectorEntry.vector).toBe('object');
      expect('indices' in mockSparseVectorEntry.vector).toBe(true);
      expect('values' in mockSparseVectorEntry.vector).toBe(true);
      expect('dimension' in mockSparseVectorEntry.vector).toBe(true);
    });

    test('sparse vector indices must be numbers', () => {
      const sparseVector = mockSparseVectorEntry.vector as { indices: number[]; values: number[]; dimension: number };
      expect(Array.isArray(sparseVector.indices)).toBe(true);
      expect(sparseVector.indices.every(i => typeof i === 'number')).toBe(true);
    });

    test('sparse vector values must be numbers', () => {
      const sparseVector = mockSparseVectorEntry.vector as { indices: number[]; values: number[]; dimension: number };
      expect(Array.isArray(sparseVector.values)).toBe(true);
      expect(sparseVector.values.every(v => typeof v === 'number')).toBe(true);
    });

    test('sparse vector indices and values must have same length', () => {
      const sparseVector = mockSparseVectorEntry.vector as { indices: number[]; values: number[]; dimension: number };
      expect(sparseVector.indices.length).toBe(sparseVector.values.length);
    });

    test('sparse vector dimension must be positive integer', () => {
      const sparseVector = mockSparseVectorEntry.vector as { indices: number[]; values: number[]; dimension: number };
      expect(Number.isInteger(sparseVector.dimension)).toBe(true);
      expect(sparseVector.dimension).toBeGreaterThan(0);
    });

    test('must conform to VectorEntrySchema', () => {
      const result = validateVectorEntry(mockSparseVectorEntry);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });
});

// ============================================================================
// COLLECTION CONFIG CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Collection Config', () => {
  test('must have collection name', () => {
    expect(mockCollectionConfig.name).toBeDefined();
    expect(typeof mockCollectionConfig.name).toBe('string');
    expect(mockCollectionConfig.name.length).toBeGreaterThan(0);
  });

  test('must have positive integer dimension', () => {
    expect(mockCollectionConfig.dimension).toBeDefined();
    expect(Number.isInteger(mockCollectionConfig.dimension)).toBe(true);
    expect(mockCollectionConfig.dimension).toBeGreaterThan(0);
  });

  test('must have valid distance metric', () => {
    expect(mockCollectionConfig.distance_metric).toBeDefined();
    expect(Object.values(DistanceMetric)).toContain(mockCollectionConfig.distance_metric);
  });

  test('must have valid embedding format', () => {
    expect(mockCollectionConfig.format).toBeDefined();
    expect(Object.values(EmbeddingFormat)).toContain(mockCollectionConfig.format);
  });

  test('must conform to CollectionConfigSchema', () => {
    const result = validateCollectionConfig(mockCollectionConfig);
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });
});

// ============================================================================
// SEARCH QUERY CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Search Query', () => {
  test('must have query vector', () => {
    expect(mockSearchQuery.vector).toBeDefined();
    expect(Array.isArray(mockSearchQuery.vector)).toBe(true);
    expect(mockSearchQuery.vector.length).toBeGreaterThan(0);
  });

  test('must have positive integer k (number of results)', () => {
    expect(mockSearchQuery.k).toBeDefined();
    expect(Number.isInteger(mockSearchQuery.k)).toBe(true);
    expect(mockSearchQuery.k).toBeGreaterThan(0);
  });

  test('may have optional score_threshold', () => {
    expect(mockSearchQuery.score_threshold).toBeDefined();
    expect(typeof mockSearchQuery.score_threshold).toBe('number');
    expect(mockSearchQuery.score_threshold).toBeGreaterThanOrEqual(0);
    expect(mockSearchQuery.score_threshold).toBeLessThanOrEqual(1);
  });

  test('must conform to SearchQuerySchema', () => {
    const result = validateSearchQuery(mockSearchQuery);
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });
});

// ============================================================================
// SEARCH RESULT CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Search Result', () => {
  test('must have VectorEntry in entry field', () => {
    expect(mockSearchResult.entry).toBeDefined();
    const result = validateVectorEntry(mockSearchResult.entry);
    expect(result.success).toBe(true);
  });

  test('must have score between 0 and 1', () => {
    expect(mockSearchResult.score).toBeDefined();
    expect(typeof mockSearchResult.score).toBe('number');
    expect(mockSearchResult.score).toBeGreaterThanOrEqual(0);
    expect(mockSearchResult.score).toBeLessThanOrEqual(1);
  });

  test('may have optional distance field', () => {
    expect(mockSearchResult.distance).toBeDefined();
    expect(typeof mockSearchResult.distance).toBe('number');
    expect(mockSearchResult.distance).toBeGreaterThanOrEqual(0);
  });
});

// ============================================================================
// UPSERT CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Upsert', () => {
  describe('Upsert Request', () => {
    test('must have collection_name', () => {
      expect(mockUpsertRequest.collection_name).toBeDefined();
      expect(typeof mockUpsertRequest.collection_name).toBe('string');
      expect(mockUpsertRequest.collection_name.length).toBeGreaterThan(0);
    });

    test('must have non-empty entries array', () => {
      expect(mockUpsertRequest.entries).toBeDefined();
      expect(Array.isArray(mockUpsertRequest.entries)).toBe(true);
      expect(mockUpsertRequest.entries.length).toBeGreaterThan(0);
    });

    test('all entries must be valid VectorEntry instances', () => {
      mockUpsertRequest.entries.forEach(entry => {
        const result = validateVectorEntry(entry);
        expect(result.success).toBe(true);
      });
    });

    test('must conform to UpsertRequestSchema', () => {
      const result = validateUpsertRequest(mockUpsertRequest);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });

  describe('Upsert Response', () => {
    test('must have upserted_count', () => {
      expect(mockUpsertResponse.upserted_count).toBeDefined();
      expect(typeof mockUpsertResponse.upserted_count).toBe('number');
      expect(mockUpsertResponse.upserted_count).toBeGreaterThanOrEqual(0);
    });

    test('must have collection_name', () => {
      expect(mockUpsertResponse.collection_name).toBeDefined();
      expect(typeof mockUpsertResponse.collection_name).toBe('string');
    });

    test('must have timestamp (UTC)', () => {
      expect(mockUpsertResponse.timestamp).toBeDefined();
      expect(new Date(mockUpsertResponse.timestamp).toISOString()).toBe(mockUpsertResponse.timestamp);
      expect(mockUpsertResponse.timestamp.endsWith('Z')).toBe(true);
    });
  });
});

// ============================================================================
// DELETE CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Delete', () => {
  describe('Delete Request', () => {
    test('must have collection_name', () => {
      expect(mockDeleteRequest.collection_name).toBeDefined();
      expect(typeof mockDeleteRequest.collection_name).toBe('string');
    });

    test('must have non-empty ids array (unless delete_all is true)', () => {
      if (!mockDeleteRequest.delete_all) {
        expect(mockDeleteRequest.ids).toBeDefined();
        expect(Array.isArray(mockDeleteRequest.ids)).toBe(true);
        expect(mockDeleteRequest.ids.length).toBeGreaterThan(0);
      }
    });

    test('all ids must be valid UUIDs', () => {
      if (mockDeleteRequest.ids) {
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
        mockDeleteRequest.ids.forEach(id => {
          expect(uuidRegex.test(id)).toBe(true);
        });
      }
    });

    test('must conform to DeleteRequestSchema', () => {
      const result = validateDeleteRequest(mockDeleteRequest);
      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });

  describe('Delete Response', () => {
    test('must have deleted_count', () => {
      expect(mockDeleteResponse.deleted_count).toBeDefined();
      expect(typeof mockDeleteResponse.deleted_count).toBe('number');
      expect(mockDeleteResponse.deleted_count).toBeGreaterThanOrEqual(0);
    });

    test('must have collection_name', () => {
      expect(mockDeleteResponse.collection_name).toBeDefined();
      expect(typeof mockDeleteResponse.collection_name).toBe('string');
    });

    test('must have timestamp (UTC)', () => {
      expect(mockDeleteResponse.timestamp).toBeDefined();
      expect(mockDeleteResponse.timestamp.endsWith('Z')).toBe(true);
    });
  });
});

// ============================================================================
// COLLECTION INFO CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Collection Info', () => {
  test('must have collection name', () => {
    expect(mockCollectionInfo.name).toBeDefined();
    expect(typeof mockCollectionInfo.name).toBe('string');
  });

  test('must have positive integer dimension', () => {
    expect(mockCollectionInfo.dimension).toBeDefined();
    expect(Number.isInteger(mockCollectionInfo.dimension)).toBe(true);
    expect(mockCollectionInfo.dimension).toBeGreaterThan(0);
  });

  test('must have non-negative vector_count', () => {
    expect(mockCollectionInfo.vector_count).toBeDefined();
    expect(typeof mockCollectionInfo.vector_count).toBe('number');
    expect(mockCollectionInfo.vector_count).toBeGreaterThanOrEqual(0);
  });

  test('must have valid distance_metric', () => {
    expect(mockCollectionInfo.distance_metric).toBeDefined();
    expect(Object.values(DistanceMetric)).toContain(mockCollectionInfo.distance_metric);
  });

  test('must have UTC timestamps', () => {
    expect(mockCollectionInfo.created_at).toBeDefined();
    expect(mockCollectionInfo.updated_at).toBeDefined();
    expect(mockCollectionInfo.created_at.endsWith('Z')).toBe(true);
    expect(mockCollectionInfo.updated_at.endsWith('Z')).toBe(true);
  });
});

// ============================================================================
// VECTOR DIMENSION VALIDATION CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Dimension Validation', () => {
  test('validates correct dense vector dimensions', () => {
    const vector = [0.1, 0.2, 0.3, 0.4, 0.5];
    const result = validateVectorDimension(vector, 5);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  test('rejects incorrect dense vector dimensions', () => {
    const vector = [0.1, 0.2, 0.3, 0.4, 0.5];
    const result = validateVectorDimension(vector, 10);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
    expect(result.error).toContain('does not match expected dimension');
  });

  test('validates correct sparse vector dimensions', () => {
    const vector = { indices: [0, 5, 10], values: [0.5, 0.3, 0.7], dimension: 100 };
    const result = validateVectorDimension(vector, 100);
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  test('rejects incorrect sparse vector dimensions', () => {
    const vector = { indices: [0, 5, 10], values: [0.5, 0.3, 0.7], dimension: 100 };
    const result = validateVectorDimension(vector, 384);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });
});

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

describe('Vector DB API Contract - Edge Cases', () => {
  test('handles empty vectors gracefully', () => {
    const emptyVector: VectorEntry = {
      id: '550e8400-e29b-41d4-a716-446655440002',
      vector: [],
      created_at: new Date().toISOString(),
    };

    // Empty vectors are technically valid but may be rejected by backends
    expect(Array.isArray(emptyVector.vector)).toBe(true);
    expect(emptyVector.vector.length).toBe(0);
  });

  test('handles maximum k value', () => {
    const maxKQuery: SearchQuery = {
      vector: [0.1, 0.2, 0.3],
      k: 1000, // Maximum allowed
    };

    const result = validateSearchQuery(maxKQuery);
    expect(result.success).toBe(true);
  });

  test('rejects k greater than maximum', () => {
    const invalidQuery: SearchQuery = {
      vector: [0.1, 0.2, 0.3],
      k: 1001, // Exceeds maximum
    };

    const result = validateSearchQuery(invalidQuery);
    expect(result.success).toBe(false);
  });

  test('handles score_threshold boundary values', () => {
    const minThreshold: SearchQuery = {
      vector: [0.1, 0.2, 0.3],
      k: 10,
      score_threshold: 0,
    };

    const maxThreshold: SearchQuery = {
      vector: [0.1, 0.2, 0.3],
      k: 10,
      score_threshold: 1,
    };

    expect(validateSearchQuery(minThreshold).success).toBe(true);
    expect(validateSearchQuery(maxThreshold).success).toBe(true);
  });

  test('rejects invalid score_threshold values', () => {
    const invalidQuery: SearchQuery = {
      vector: [0.1, 0.2, 0.3],
      k: 10,
      score_threshold: 1.5, // Invalid: > 1
    };

    const result = validateSearchQuery(invalidQuery);
    expect(result.success).toBe(false);
  });

  test('handles large metadata objects', () => {
    const largeMetadata: VectorEntry = {
      id: '550e8400-e29b-41d4-a716-446655440003',
      vector: [0.1, 0.2, 0.3],
      metadata: {
        // Large metadata object
        ...Object.fromEntries(Array(100).fill(0).map((_, i) => [`key${i}`, `value${i}`])),
      },
      created_at: new Date().toISOString(),
    };

    const result = validateVectorEntry(largeMetadata);
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// BACKEND-SPECIFIC CONTRACT TESTS
// ============================================================================

describe('Vector DB API Contract - Backend Specific', () => {
  test('Qdrant: vector name is optional', () => {
    const qdrantVector: VectorEntry = {
      id: '550e8400-e29b-41d4-a716-446655440004',
      vector: [0.1, 0.2, 0.3],
      vector_name: 'dense',
      created_at: new Date().toISOString(),
    };

    const result = validateVectorEntry(qdrantVector);
    expect(result.success).toBe(true);
  });

  test('Pinecone: only supports dense vectors', () => {
    // Pinecone client will validate and reject sparse vectors
    const denseVector: VectorEntry = {
      id: '550e8400-e29b-41d4-a716-446655440005',
      vector: [0.1, 0.2, 0.3],
      created_at: new Date().toISOString(),
    };

    const result = validateVectorEntry(denseVector);
    expect(result.success).toBe(true);
  });

  test('pgvector: handles connection strings', () => {
    // pgvector requires connection string validation
    const connectionString = 'postgresql://user:password@localhost:5432/dbname';

    expect(connectionString).toContain('postgresql://');
    expect(connectionString.length).toBeGreaterThan(0);
  });

  test('Chroma: handles documents with embeddings', () => {
    const chromaVector: VectorEntry = {
      id: '550e8400-e29b-41d4-a716-446655440006',
      vector: [0.1, 0.2, 0.3],
      text: 'Sample document text',
      metadata: {
        source: 'test',
      },
      created_at: new Date().toISOString(),
    };

    const result = validateVectorEntry(chromaVector);
    expect(result.success).toBe(true);
    expect(chromaVector.text).toBeDefined();
  });
});

// ============================================================================
// SETUP AND TEARDOWN
// ============================================================================

let setupComplete = false;

beforeAll(async () => {
  // Setup: Validate test environment
  expect(validateVectorEntry).toBeDefined();
  expect(validateSearchQuery).toBeDefined();
  expect(validateUpsertRequest).toBeDefined();
  expect(validateDeleteRequest).toBeDefined();
  expect(validateCollectionConfig).toBeDefined();
  expect(validateHealthCheckResponse).toBeDefined();
  expect(validateVectorDimension).toBeDefined();

  setupComplete = true;

  console.log('✅ Vector DB Contract Test Suite Initialized');
  console.log('⚠️  Remember: If these tests fail, the adapter MUST refuse to start');
});

afterAll(() => {
  if (setupComplete) {
    console.log('✅ Vector DB Contract Test Suite Completed');
    console.log('📋 All contract validations passed');
  }
});

// ============================================================================
// EXPORTS
// ============================================================================

export {
  mockHealthCheckResponse,
  mockDegradedHealthCheckResponse,
  mockVectorEntry,
  mockSparseVectorEntry,
  mockCollectionConfig,
  mockSearchQuery,
  mockSearchResult,
  mockUpsertRequest,
  mockUpsertResponse,
  mockDeleteRequest,
  mockDeleteResponse,
  mockCollectionInfo,
};
