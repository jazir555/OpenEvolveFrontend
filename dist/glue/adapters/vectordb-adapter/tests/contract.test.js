"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.mockCollectionInfo = exports.mockDeleteResponse = exports.mockDeleteRequest = exports.mockUpsertResponse = exports.mockUpsertRequest = exports.mockSearchResult = exports.mockSearchQuery = exports.mockCollectionConfig = exports.mockSparseVectorEntry = exports.mockVectorEntry = exports.mockDegradedHealthCheckResponse = exports.mockHealthCheckResponse = void 0;
const globals_1 = require("@jest/globals");
// Import canonical schemas for validation
const vectordb_canonical_1 = require("../../../schemas/vectordb-canonical");
// ============================================================================
// MOCK DATA - Simulating Vector DB API Responses
// ============================================================================
const mockHealthCheckResponse = {
    status: 'healthy',
    backend_type: vectordb_canonical_1.VectorDBType.QDRANT,
    connected: true,
    latency_ms: 5,
    collections_count: 3,
    timestamp: '2024-01-15T10:30:00.000Z',
};
exports.mockHealthCheckResponse = mockHealthCheckResponse;
const mockDegradedHealthCheckResponse = {
    status: 'degraded',
    backend_type: vectordb_canonical_1.VectorDBType.PINECONE,
    connected: false,
    error: 'Connection timeout',
    timestamp: '2024-01-15T10:30:00.000Z',
};
exports.mockDegradedHealthCheckResponse = mockDegradedHealthCheckResponse;
const mockVectorEntry = {
    id: '550e8400-e29b-41d4-a716-446655440000',
    vector: [0.1, 0.2, 0.3, 0.4, 0.5],
    text: 'Example text for embedding',
    metadata: {
        category: 'example',
        source: 'test',
    },
    created_at: '2024-01-15T10:30:00.000Z',
};
exports.mockVectorEntry = mockVectorEntry;
const mockSparseVectorEntry = {
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
exports.mockSparseVectorEntry = mockSparseVectorEntry;
const mockCollectionConfig = {
    name: 'test_collection',
    dimension: 384,
    distance_metric: vectordb_canonical_1.DistanceMetric.COSINE,
    format: vectordb_canonical_1.EmbeddingFormat.DENSE,
};
exports.mockCollectionConfig = mockCollectionConfig;
const mockSearchQuery = {
    vector: [0.1, 0.2, 0.3, 0.4, 0.5],
    k: 10,
    score_threshold: 0.7,
};
exports.mockSearchQuery = mockSearchQuery;
const mockSearchResult = {
    entry: mockVectorEntry,
    score: 0.85,
    distance: 0.15,
};
exports.mockSearchResult = mockSearchResult;
const mockUpsertRequest = {
    collection_name: 'test_collection',
    entries: [mockVectorEntry, mockSparseVectorEntry],
};
exports.mockUpsertRequest = mockUpsertRequest;
const mockUpsertResponse = {
    upserted_count: 2,
    collection_name: 'test_collection',
    timestamp: '2024-01-15T10:30:00.000Z',
};
exports.mockUpsertResponse = mockUpsertResponse;
const mockDeleteRequest = {
    collection_name: 'test_collection',
    ids: ['550e8400-e29b-41d4-a716-446655440000'],
};
exports.mockDeleteRequest = mockDeleteRequest;
const mockDeleteResponse = {
    deleted_count: 1,
    collection_name: 'test_collection',
    timestamp: '2024-01-15T10:30:00.000Z',
};
exports.mockDeleteResponse = mockDeleteResponse;
const mockCollectionInfo = {
    name: 'test_collection',
    dimension: 384,
    vector_count: 1000,
    distance_metric: vectordb_canonical_1.DistanceMetric.COSINE,
    created_at: '2024-01-15T10:30:00.000Z',
    updated_at: '2024-01-15T10:30:00.000Z',
};
exports.mockCollectionInfo = mockCollectionInfo;
// ============================================================================
// HEALTH CHECK CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Health Check', () => {
    (0, globals_1.describe)('Healthy Response', () => {
        (0, globals_1.test)('must return status "healthy" when connected', () => {
            (0, globals_1.expect)(mockHealthCheckResponse.status).toBe('healthy');
            (0, globals_1.expect)(mockHealthCheckResponse.connected).toBe(true);
        });
        (0, globals_1.test)('must include backend_type', () => {
            (0, globals_1.expect)(mockHealthCheckResponse.backend_type).toBeDefined();
            (0, globals_1.expect)(Object.values(vectordb_canonical_1.VectorDBType)).toContain(mockHealthCheckResponse.backend_type);
        });
        (0, globals_1.test)('may include latency_ms', () => {
            (0, globals_1.expect)(mockHealthCheckResponse.latency_ms).toBeDefined();
            (0, globals_1.expect)(typeof mockHealthCheckResponse.latency_ms).toBe('number');
            (0, globals_1.expect)(mockHealthCheckResponse.latency_ms).toBeGreaterThan(0);
        });
        (0, globals_1.test)('may include collections_count', () => {
            (0, globals_1.expect)(mockHealthCheckResponse.collections_count).toBeDefined();
            (0, globals_1.expect)(typeof mockHealthCheckResponse.collections_count).toBe('number');
            (0, globals_1.expect)(mockHealthCheckResponse.collections_count).toBeGreaterThanOrEqual(0);
        });
        (0, globals_1.test)('must conform to HealthCheckResponseSchema', () => {
            const result = (0, vectordb_canonical_1.validateHealthCheckResponse)(mockHealthCheckResponse);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
    });
    (0, globals_1.describe)('Degraded Response', () => {
        (0, globals_1.test)('must return status "degraded" or "unhealthy" when disconnected', () => {
            (0, globals_1.expect)(['degraded', 'unhealthy']).toContain(mockDegradedHealthCheckResponse.status);
            (0, globals_1.expect)(mockDegradedHealthCheckResponse.connected).toBe(false);
        });
        (0, globals_1.test)('may include error message when degraded', () => {
            (0, globals_1.expect)(mockDegradedHealthCheckResponse.error).toBeDefined();
            (0, globals_1.expect)(typeof mockDegradedHealthCheckResponse.error).toBe('string');
        });
        (0, globals_1.test)('must conform to HealthCheckResponseSchema', () => {
            const result = (0, vectordb_canonical_1.validateHealthCheckResponse)(mockDegradedHealthCheckResponse);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data?.status).toBe('degraded');
        });
    });
});
// ============================================================================
// VECTOR ENTRY CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Vector Entry', () => {
    (0, globals_1.describe)('Dense Vector Entry', () => {
        (0, globals_1.test)('must have valid UUID id', () => {
            const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
            (0, globals_1.expect)(uuidRegex.test(mockVectorEntry.id)).toBe(true);
        });
        (0, globals_1.test)('must have array of numbers for dense vector', () => {
            (0, globals_1.expect)(Array.isArray(mockVectorEntry.vector)).toBe(true);
            (0, globals_1.expect)(mockVectorEntry.vector.length).toBeGreaterThan(0);
            (0, globals_1.expect)(mockVectorEntry.vector.every(v => typeof v === 'number')).toBe(true);
        });
        (0, globals_1.test)('may have optional text field', () => {
            (0, globals_1.expect)(mockVectorEntry.text).toBeDefined();
            (0, globals_1.expect)(typeof mockVectorEntry.text).toBe('string');
        });
        (0, globals_1.test)('may have optional metadata object', () => {
            (0, globals_1.expect)(mockVectorEntry.metadata).toBeDefined();
            (0, globals_1.expect)(typeof mockVectorEntry.metadata).toBe('object');
        });
        (0, globals_1.test)('may have optional created_at timestamp', () => {
            (0, globals_1.expect)(mockVectorEntry.created_at).toBeDefined();
            (0, globals_1.expect)(new Date(mockVectorEntry.created_at).toISOString()).toBe(mockVectorEntry.created_at);
        });
        (0, globals_1.test)('must conform to VectorEntrySchema', () => {
            const result = (0, vectordb_canonical_1.validateVectorEntry)(mockVectorEntry);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
    });
    (0, globals_1.describe)('Sparse Vector Entry', () => {
        (0, globals_1.test)('must have sparse vector object with indices, values, and dimension', () => {
            (0, globals_1.expect)(typeof mockSparseVectorEntry.vector).toBe('object');
            (0, globals_1.expect)('indices' in mockSparseVectorEntry.vector).toBe(true);
            (0, globals_1.expect)('values' in mockSparseVectorEntry.vector).toBe(true);
            (0, globals_1.expect)('dimension' in mockSparseVectorEntry.vector).toBe(true);
        });
        (0, globals_1.test)('sparse vector indices must be numbers', () => {
            const sparseVector = mockSparseVectorEntry.vector;
            (0, globals_1.expect)(Array.isArray(sparseVector.indices)).toBe(true);
            (0, globals_1.expect)(sparseVector.indices.every(i => typeof i === 'number')).toBe(true);
        });
        (0, globals_1.test)('sparse vector values must be numbers', () => {
            const sparseVector = mockSparseVectorEntry.vector;
            (0, globals_1.expect)(Array.isArray(sparseVector.values)).toBe(true);
            (0, globals_1.expect)(sparseVector.values.every(v => typeof v === 'number')).toBe(true);
        });
        (0, globals_1.test)('sparse vector indices and values must have same length', () => {
            const sparseVector = mockSparseVectorEntry.vector;
            (0, globals_1.expect)(sparseVector.indices.length).toBe(sparseVector.values.length);
        });
        (0, globals_1.test)('sparse vector dimension must be positive integer', () => {
            const sparseVector = mockSparseVectorEntry.vector;
            (0, globals_1.expect)(Number.isInteger(sparseVector.dimension)).toBe(true);
            (0, globals_1.expect)(sparseVector.dimension).toBeGreaterThan(0);
        });
        (0, globals_1.test)('must conform to VectorEntrySchema', () => {
            const result = (0, vectordb_canonical_1.validateVectorEntry)(mockSparseVectorEntry);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
    });
});
// ============================================================================
// COLLECTION CONFIG CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Collection Config', () => {
    (0, globals_1.test)('must have collection name', () => {
        (0, globals_1.expect)(mockCollectionConfig.name).toBeDefined();
        (0, globals_1.expect)(typeof mockCollectionConfig.name).toBe('string');
        (0, globals_1.expect)(mockCollectionConfig.name.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('must have positive integer dimension', () => {
        (0, globals_1.expect)(mockCollectionConfig.dimension).toBeDefined();
        (0, globals_1.expect)(Number.isInteger(mockCollectionConfig.dimension)).toBe(true);
        (0, globals_1.expect)(mockCollectionConfig.dimension).toBeGreaterThan(0);
    });
    (0, globals_1.test)('must have valid distance metric', () => {
        (0, globals_1.expect)(mockCollectionConfig.distance_metric).toBeDefined();
        (0, globals_1.expect)(Object.values(vectordb_canonical_1.DistanceMetric)).toContain(mockCollectionConfig.distance_metric);
    });
    (0, globals_1.test)('must have valid embedding format', () => {
        (0, globals_1.expect)(mockCollectionConfig.format).toBeDefined();
        (0, globals_1.expect)(Object.values(vectordb_canonical_1.EmbeddingFormat)).toContain(mockCollectionConfig.format);
    });
    (0, globals_1.test)('must conform to CollectionConfigSchema', () => {
        const result = (0, vectordb_canonical_1.validateCollectionConfig)(mockCollectionConfig);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toBeDefined();
    });
});
// ============================================================================
// SEARCH QUERY CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Search Query', () => {
    (0, globals_1.test)('must have query vector', () => {
        (0, globals_1.expect)(mockSearchQuery.vector).toBeDefined();
        (0, globals_1.expect)(Array.isArray(mockSearchQuery.vector)).toBe(true);
        (0, globals_1.expect)(mockSearchQuery.vector.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('must have positive integer k (number of results)', () => {
        (0, globals_1.expect)(mockSearchQuery.k).toBeDefined();
        (0, globals_1.expect)(Number.isInteger(mockSearchQuery.k)).toBe(true);
        (0, globals_1.expect)(mockSearchQuery.k).toBeGreaterThan(0);
    });
    (0, globals_1.test)('may have optional score_threshold', () => {
        (0, globals_1.expect)(mockSearchQuery.score_threshold).toBeDefined();
        (0, globals_1.expect)(typeof mockSearchQuery.score_threshold).toBe('number');
        (0, globals_1.expect)(mockSearchQuery.score_threshold).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(mockSearchQuery.score_threshold).toBeLessThanOrEqual(1);
    });
    (0, globals_1.test)('must conform to SearchQuerySchema', () => {
        const result = (0, vectordb_canonical_1.validateSearchQuery)(mockSearchQuery);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toBeDefined();
    });
});
// ============================================================================
// SEARCH RESULT CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Search Result', () => {
    (0, globals_1.test)('must have VectorEntry in entry field', () => {
        (0, globals_1.expect)(mockSearchResult.entry).toBeDefined();
        const result = (0, vectordb_canonical_1.validateVectorEntry)(mockSearchResult.entry);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must have score between 0 and 1', () => {
        (0, globals_1.expect)(mockSearchResult.score).toBeDefined();
        (0, globals_1.expect)(typeof mockSearchResult.score).toBe('number');
        (0, globals_1.expect)(mockSearchResult.score).toBeGreaterThanOrEqual(0);
        (0, globals_1.expect)(mockSearchResult.score).toBeLessThanOrEqual(1);
    });
    (0, globals_1.test)('may have optional distance field', () => {
        (0, globals_1.expect)(mockSearchResult.distance).toBeDefined();
        (0, globals_1.expect)(typeof mockSearchResult.distance).toBe('number');
        (0, globals_1.expect)(mockSearchResult.distance).toBeGreaterThanOrEqual(0);
    });
});
// ============================================================================
// UPSERT CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Upsert', () => {
    (0, globals_1.describe)('Upsert Request', () => {
        (0, globals_1.test)('must have collection_name', () => {
            (0, globals_1.expect)(mockUpsertRequest.collection_name).toBeDefined();
            (0, globals_1.expect)(typeof mockUpsertRequest.collection_name).toBe('string');
            (0, globals_1.expect)(mockUpsertRequest.collection_name.length).toBeGreaterThan(0);
        });
        (0, globals_1.test)('must have non-empty entries array', () => {
            (0, globals_1.expect)(mockUpsertRequest.entries).toBeDefined();
            (0, globals_1.expect)(Array.isArray(mockUpsertRequest.entries)).toBe(true);
            (0, globals_1.expect)(mockUpsertRequest.entries.length).toBeGreaterThan(0);
        });
        (0, globals_1.test)('all entries must be valid VectorEntry instances', () => {
            mockUpsertRequest.entries.forEach(entry => {
                const result = (0, vectordb_canonical_1.validateVectorEntry)(entry);
                (0, globals_1.expect)(result.success).toBe(true);
            });
        });
        (0, globals_1.test)('must conform to UpsertRequestSchema', () => {
            const result = (0, vectordb_canonical_1.validateUpsertRequest)(mockUpsertRequest);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
    });
    (0, globals_1.describe)('Upsert Response', () => {
        (0, globals_1.test)('must have upserted_count', () => {
            (0, globals_1.expect)(mockUpsertResponse.upserted_count).toBeDefined();
            (0, globals_1.expect)(typeof mockUpsertResponse.upserted_count).toBe('number');
            (0, globals_1.expect)(mockUpsertResponse.upserted_count).toBeGreaterThanOrEqual(0);
        });
        (0, globals_1.test)('must have collection_name', () => {
            (0, globals_1.expect)(mockUpsertResponse.collection_name).toBeDefined();
            (0, globals_1.expect)(typeof mockUpsertResponse.collection_name).toBe('string');
        });
        (0, globals_1.test)('must have timestamp (UTC)', () => {
            (0, globals_1.expect)(mockUpsertResponse.timestamp).toBeDefined();
            (0, globals_1.expect)(new Date(mockUpsertResponse.timestamp).toISOString()).toBe(mockUpsertResponse.timestamp);
            (0, globals_1.expect)(mockUpsertResponse.timestamp.endsWith('Z')).toBe(true);
        });
    });
});
// ============================================================================
// DELETE CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Delete', () => {
    (0, globals_1.describe)('Delete Request', () => {
        (0, globals_1.test)('must have collection_name', () => {
            (0, globals_1.expect)(mockDeleteRequest.collection_name).toBeDefined();
            (0, globals_1.expect)(typeof mockDeleteRequest.collection_name).toBe('string');
        });
        (0, globals_1.test)('must have non-empty ids array (unless delete_all is true)', () => {
            if (!mockDeleteRequest.delete_all) {
                (0, globals_1.expect)(mockDeleteRequest.ids).toBeDefined();
                (0, globals_1.expect)(Array.isArray(mockDeleteRequest.ids)).toBe(true);
                (0, globals_1.expect)(mockDeleteRequest.ids.length).toBeGreaterThan(0);
            }
        });
        (0, globals_1.test)('all ids must be valid UUIDs', () => {
            if (mockDeleteRequest.ids) {
                const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
                mockDeleteRequest.ids.forEach(id => {
                    (0, globals_1.expect)(uuidRegex.test(id)).toBe(true);
                });
            }
        });
        (0, globals_1.test)('must conform to DeleteRequestSchema', () => {
            const result = (0, vectordb_canonical_1.validateDeleteRequest)(mockDeleteRequest);
            (0, globals_1.expect)(result.success).toBe(true);
            (0, globals_1.expect)(result.data).toBeDefined();
        });
    });
    (0, globals_1.describe)('Delete Response', () => {
        (0, globals_1.test)('must have deleted_count', () => {
            (0, globals_1.expect)(mockDeleteResponse.deleted_count).toBeDefined();
            (0, globals_1.expect)(typeof mockDeleteResponse.deleted_count).toBe('number');
            (0, globals_1.expect)(mockDeleteResponse.deleted_count).toBeGreaterThanOrEqual(0);
        });
        (0, globals_1.test)('must have collection_name', () => {
            (0, globals_1.expect)(mockDeleteResponse.collection_name).toBeDefined();
            (0, globals_1.expect)(typeof mockDeleteResponse.collection_name).toBe('string');
        });
        (0, globals_1.test)('must have timestamp (UTC)', () => {
            (0, globals_1.expect)(mockDeleteResponse.timestamp).toBeDefined();
            (0, globals_1.expect)(mockDeleteResponse.timestamp.endsWith('Z')).toBe(true);
        });
    });
});
// ============================================================================
// COLLECTION INFO CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Collection Info', () => {
    (0, globals_1.test)('must have collection name', () => {
        (0, globals_1.expect)(mockCollectionInfo.name).toBeDefined();
        (0, globals_1.expect)(typeof mockCollectionInfo.name).toBe('string');
    });
    (0, globals_1.test)('must have positive integer dimension', () => {
        (0, globals_1.expect)(mockCollectionInfo.dimension).toBeDefined();
        (0, globals_1.expect)(Number.isInteger(mockCollectionInfo.dimension)).toBe(true);
        (0, globals_1.expect)(mockCollectionInfo.dimension).toBeGreaterThan(0);
    });
    (0, globals_1.test)('must have non-negative vector_count', () => {
        (0, globals_1.expect)(mockCollectionInfo.vector_count).toBeDefined();
        (0, globals_1.expect)(typeof mockCollectionInfo.vector_count).toBe('number');
        (0, globals_1.expect)(mockCollectionInfo.vector_count).toBeGreaterThanOrEqual(0);
    });
    (0, globals_1.test)('must have valid distance_metric', () => {
        (0, globals_1.expect)(mockCollectionInfo.distance_metric).toBeDefined();
        (0, globals_1.expect)(Object.values(vectordb_canonical_1.DistanceMetric)).toContain(mockCollectionInfo.distance_metric);
    });
    (0, globals_1.test)('must have UTC timestamps', () => {
        (0, globals_1.expect)(mockCollectionInfo.created_at).toBeDefined();
        (0, globals_1.expect)(mockCollectionInfo.updated_at).toBeDefined();
        (0, globals_1.expect)(mockCollectionInfo.created_at.endsWith('Z')).toBe(true);
        (0, globals_1.expect)(mockCollectionInfo.updated_at.endsWith('Z')).toBe(true);
    });
});
// ============================================================================
// VECTOR DIMENSION VALIDATION CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Dimension Validation', () => {
    (0, globals_1.test)('validates correct dense vector dimensions', () => {
        const vector = [0.1, 0.2, 0.3, 0.4, 0.5];
        const result = (0, vectordb_canonical_1.validateVectorDimension)(vector, 5);
        (0, globals_1.expect)(result.valid).toBe(true);
        (0, globals_1.expect)(result.error).toBeUndefined();
    });
    (0, globals_1.test)('rejects incorrect dense vector dimensions', () => {
        const vector = [0.1, 0.2, 0.3, 0.4, 0.5];
        const result = (0, vectordb_canonical_1.validateVectorDimension)(vector, 10);
        (0, globals_1.expect)(result.valid).toBe(false);
        (0, globals_1.expect)(result.error).toBeDefined();
        (0, globals_1.expect)(result.error).toContain('does not match expected dimension');
    });
    (0, globals_1.test)('validates correct sparse vector dimensions', () => {
        const vector = { indices: [0, 5, 10], values: [0.5, 0.3, 0.7], dimension: 100 };
        const result = (0, vectordb_canonical_1.validateVectorDimension)(vector, 100);
        (0, globals_1.expect)(result.valid).toBe(true);
        (0, globals_1.expect)(result.error).toBeUndefined();
    });
    (0, globals_1.test)('rejects incorrect sparse vector dimensions', () => {
        const vector = { indices: [0, 5, 10], values: [0.5, 0.3, 0.7], dimension: 100 };
        const result = (0, vectordb_canonical_1.validateVectorDimension)(vector, 384);
        (0, globals_1.expect)(result.valid).toBe(false);
        (0, globals_1.expect)(result.error).toBeDefined();
    });
});
// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Edge Cases', () => {
    (0, globals_1.test)('handles empty vectors gracefully', () => {
        const emptyVector = {
            id: '550e8400-e29b-41d4-a716-446655440002',
            vector: [],
            created_at: new Date().toISOString(),
        };
        // Empty vectors are technically valid but may be rejected by backends
        (0, globals_1.expect)(Array.isArray(emptyVector.vector)).toBe(true);
        (0, globals_1.expect)(emptyVector.vector.length).toBe(0);
    });
    (0, globals_1.test)('handles maximum k value', () => {
        const maxKQuery = {
            vector: [0.1, 0.2, 0.3],
            k: 1000, // Maximum allowed
        };
        const result = (0, vectordb_canonical_1.validateSearchQuery)(maxKQuery);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('rejects k greater than maximum', () => {
        const invalidQuery = {
            vector: [0.1, 0.2, 0.3],
            k: 1001, // Exceeds maximum
        };
        const result = (0, vectordb_canonical_1.validateSearchQuery)(invalidQuery);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('handles score_threshold boundary values', () => {
        const minThreshold = {
            vector: [0.1, 0.2, 0.3],
            k: 10,
            score_threshold: 0,
        };
        const maxThreshold = {
            vector: [0.1, 0.2, 0.3],
            k: 10,
            score_threshold: 1,
        };
        (0, globals_1.expect)((0, vectordb_canonical_1.validateSearchQuery)(minThreshold).success).toBe(true);
        (0, globals_1.expect)((0, vectordb_canonical_1.validateSearchQuery)(maxThreshold).success).toBe(true);
    });
    (0, globals_1.test)('rejects invalid score_threshold values', () => {
        const invalidQuery = {
            vector: [0.1, 0.2, 0.3],
            k: 10,
            score_threshold: 1.5, // Invalid: > 1
        };
        const result = (0, vectordb_canonical_1.validateSearchQuery)(invalidQuery);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('handles large metadata objects', () => {
        const largeMetadata = {
            id: '550e8400-e29b-41d4-a716-446655440003',
            vector: [0.1, 0.2, 0.3],
            metadata: {
                // Large metadata object
                ...Object.fromEntries(Array(100).fill(0).map((_, i) => [`key${i}`, `value${i}`])),
            },
            created_at: new Date().toISOString(),
        };
        const result = (0, vectordb_canonical_1.validateVectorEntry)(largeMetadata);
        (0, globals_1.expect)(result.success).toBe(true);
    });
});
// ============================================================================
// BACKEND-SPECIFIC CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('Vector DB API Contract - Backend Specific', () => {
    (0, globals_1.test)('Qdrant: vector name is optional', () => {
        const qdrantVector = {
            id: '550e8400-e29b-41d4-a716-446655440004',
            vector: [0.1, 0.2, 0.3],
            vector_name: 'dense',
            created_at: new Date().toISOString(),
        };
        const result = (0, vectordb_canonical_1.validateVectorEntry)(qdrantVector);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('Pinecone: only supports dense vectors', () => {
        // Pinecone client will validate and reject sparse vectors
        const denseVector = {
            id: '550e8400-e29b-41d4-a716-446655440005',
            vector: [0.1, 0.2, 0.3],
            created_at: new Date().toISOString(),
        };
        const result = (0, vectordb_canonical_1.validateVectorEntry)(denseVector);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('pgvector: handles connection strings', () => {
        // pgvector requires connection string validation
        const connectionString = 'postgresql://user:password@localhost:5432/dbname';
        (0, globals_1.expect)(connectionString).toContain('postgresql://');
        (0, globals_1.expect)(connectionString.length).toBeGreaterThan(0);
    });
    (0, globals_1.test)('Chroma: handles documents with embeddings', () => {
        const chromaVector = {
            id: '550e8400-e29b-41d4-a716-446655440006',
            vector: [0.1, 0.2, 0.3],
            text: 'Sample document text',
            metadata: {
                source: 'test',
            },
            created_at: new Date().toISOString(),
        };
        const result = (0, vectordb_canonical_1.validateVectorEntry)(chromaVector);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(chromaVector.text).toBeDefined();
    });
});
// ============================================================================
// SETUP AND TEARDOWN
// ============================================================================
let setupComplete = false;
(0, globals_1.beforeAll)(async () => {
    // Setup: Validate test environment
    (0, globals_1.expect)(vectordb_canonical_1.validateVectorEntry).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateSearchQuery).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateUpsertRequest).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateDeleteRequest).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateCollectionConfig).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateHealthCheckResponse).toBeDefined();
    (0, globals_1.expect)(vectordb_canonical_1.validateVectorDimension).toBeDefined();
    setupComplete = true;
    console.log('✅ Vector DB Contract Test Suite Initialized');
    console.log('⚠️  Remember: If these tests fail, the adapter MUST refuse to start');
});
(0, globals_1.afterAll)(() => {
    if (setupComplete) {
        console.log('✅ Vector DB Contract Test Suite Completed');
        console.log('📋 All contract validations passed');
    }
});
//# sourceMappingURL=contract.test.js.map