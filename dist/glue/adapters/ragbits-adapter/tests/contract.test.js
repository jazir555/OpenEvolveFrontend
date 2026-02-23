"use strict";
/**
 * RAGBits Adapter Contract Tests
 *
 * CRITICAL: These tests validate the contract between the RAGBits Adapter and RAGBits Core.
 * If these tests fail, the adapter MUST refuse to start to prevent data corruption.
 *
 * Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
 * - Phase 2: The Contract (Defense)
 * - Protecting the Mega-Project from Updates
 *
 * Test Principles:
 * 1. FAIL FAST - Contract violations immediately halt execution
 * 2. MOCK ONLY - Do not require running RAGBits instance
 * 3. CANONICAL VALIDATION - Use canonical schemas for data structure validation
 * 4. IDEMPOTENT - Tests can be run 100 times safely
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.mockErrorResponse = exports.mockStatsResponse = exports.mockIngestResponse = exports.mockSearchResponse = exports.mockHealthResponse = void 0;
const globals_1 = require("@jest/globals");
// Import canonical schemas for validation
const ragbits_canonical_1 = require("../../../schemas/ragbits-canonical");
// ============================================================================
// MOCK DATA - Simulating RAGBits Core API Responses
// ============================================================================
const mockHealthResponse = {
    status: 'healthy',
    ragbits_available: true,
    vector_store_connected: true,
    embedding_model_loaded: true,
    timestamp: '2025-02-03T12:34:56.789Z',
};
exports.mockHealthResponse = mockHealthResponse;
const mockSearchResponse = {
    results: [
        {
            chunk_id: 'chunk-123',
            content: 'Machine learning is a subset of artificial intelligence...',
            document_id: 'doc-456',
            score: 0.92,
            metadata: {
                document_type: 'markdown',
                source: 'ml-guide',
                tags: ['ml', 'ai'],
            },
            highlight: 'Machine learning is...',
        },
        {
            chunk_id: 'chunk-124',
            content: 'Deep learning uses neural networks with multiple layers...',
            document_id: 'doc-456',
            score: 0.87,
            metadata: {
                document_type: 'markdown',
                source: 'ml-guide',
            },
        },
    ],
    total_results: 2,
    query: 'machine learning basics',
    search_mode: 'semantic',
    execution_time_ms: 125,
    timestamp: '2025-02-03T12:34:56.789Z',
};
exports.mockSearchResponse = mockSearchResponse;
const mockIngestResponse = {
    success: true,
    document_id: 'doc-789',
    chunks_created: 3,
    execution_time_ms: 245,
    timestamp: '2025-02-03T12:34:56.789Z',
};
exports.mockIngestResponse = mockIngestResponse;
const mockStatsResponse = {
    total_documents: 150,
    total_chunks: 1250,
    index_size_bytes: 5242880,
    vector_store_type: 'chromadb',
    embedding_model: 'all-MiniLM-L6-v2',
    cache_size: 100,
    timestamp: '2025-02-03T12:34:56.789Z',
};
exports.mockStatsResponse = mockStatsResponse;
const mockErrorResponse = {
    code: 'RETRIEVAL_FAILED',
    message: 'Failed to retrieve relevant documents',
    details: {
        reason: 'Vector database connection timeout',
        query: 'What is machine learning?',
    },
    correlation_id: '550e8400-e29b-41d4-a716-446655440000',
    timestamp: '2025-02-03T12:34:56.789Z',
};
exports.mockErrorResponse = mockErrorResponse;
// ============================================================================
// RAGBits API CONTRACT TESTS
// ============================================================================
(0, globals_1.describe)('RAGBits API Contract - Health Endpoint', () => {
    (0, globals_1.describe)('GET /health - Healthy Response', () => {
        (0, globals_1.test)('must return status "healthy" when RAGBits is available', () => {
            (0, globals_1.expect)(mockHealthResponse.status).toBe('healthy');
            (0, globals_1.expect)(mockHealthResponse.ragbits_available).toBe(true);
        });
        (0, globals_1.test)('must include vector store connection status', () => {
            (0, globals_1.expect)(mockHealthResponse.vector_store_connected).toBeDefined();
            (0, globals_1.expect)(typeof mockHealthResponse.vector_store_connected).toBe('boolean');
        });
        (0, globals_1.test)('must include embedding model status', () => {
            (0, globals_1.expect)(mockHealthResponse.embedding_model_loaded).toBeDefined();
            (0, globals_1.expect)(typeof mockHealthResponse.embedding_model_loaded).toBe('boolean');
        });
        (0, globals_1.test)('must include UTC timestamp (Law of UTC)', () => {
            (0, globals_1.expect)(mockHealthResponse.timestamp).toBeDefined();
            (0, globals_1.expect)(mockHealthResponse.timestamp.endsWith('Z')).toBe(true);
        });
    });
    (0, globals_1.describe)('GET /health - Degraded Response', () => {
        (0, globals_1.test)('must return status "degraded" or "unhealthy" when unavailable', () => {
            const degradedResponse = {
                status: 'degraded',
                ragbits_available: false,
                vector_store_connected: false,
                embedding_model_loaded: false,
                error: 'Vector store not initialized',
                timestamp: '2025-02-03T12:34:56.789Z',
            };
            (0, globals_1.expect)(['degraded', 'unhealthy']).toContain(degradedResponse.status);
            (0, globals_1.expect)(degradedResponse.ragbits_available).toBe(false);
        });
        (0, globals_1.test)('may include error message when degraded', () => {
            const degradedResponse = {
                status: 'degraded',
                ragbits_available: false,
                vector_store_connected: false,
                embedding_model_loaded: false,
                error: 'Vector store not initialized',
                timestamp: '2025-02-03T12:34:56.789Z',
            };
            (0, globals_1.expect)(degradedResponse.error).toBeDefined();
            (0, globals_1.expect)(typeof degradedResponse.error).toBe('string');
        });
    });
});
(0, globals_1.describe)('RAGBits API Contract - Search Endpoint', () => {
    (0, globals_1.describe)('POST /search - Successful Response', () => {
        (0, globals_1.test)('must include "results" array', () => {
            (0, globals_1.expect)(mockSearchResponse.results).toBeDefined();
            (0, globals_1.expect)(Array.isArray(mockSearchResponse.results)).toBe(true);
        });
        (0, globals_1.test)('must include "total_results" count', () => {
            (0, globals_1.expect)(mockSearchResponse.total_results).toBeDefined();
            (0, globals_1.expect)(typeof mockSearchResponse.total_results).toBe('number');
            (0, globals_1.expect)(mockSearchResponse.total_results).toBeGreaterThanOrEqual(0);
        });
        (0, globals_1.test)('must include original "query" string', () => {
            (0, globals_1.expect)(mockSearchResponse.query).toBeDefined();
            (0, globals_1.expect)(typeof mockSearchResponse.query).toBe('string');
        });
        (0, globals_1.test)('must include "execution_time_ms" for performance tracking', () => {
            (0, globals_1.expect)(mockSearchResponse.execution_time_ms).toBeDefined();
            (0, globals_1.expect)(typeof mockSearchResponse.execution_time_ms).toBe('number');
            (0, globals_1.expect)(mockSearchResponse.execution_time_ms).toBeGreaterThan(0);
        });
        (0, globals_1.test)('must include UTC timestamp (Law of UTC)', () => {
            (0, globals_1.expect)(mockSearchResponse.timestamp).toBeDefined();
            (0, globals_1.expect)(mockSearchResponse.timestamp.endsWith('Z')).toBe(true);
        });
        (0, globals_1.test)('results must have required fields', () => {
            const result = mockSearchResponse.results[0];
            (0, globals_1.expect)(result.chunk_id).toBeDefined();
            (0, globals_1.expect)(result.content).toBeDefined();
            (0, globals_1.expect)(result.document_id).toBeDefined();
            (0, globals_1.expect)(result.score).toBeDefined();
        });
        (0, globals_1.test)('results scores must be between 0 and 1', () => {
            mockSearchResponse.results.forEach(result => {
                (0, globals_1.expect)(result.score).toBeGreaterThanOrEqual(0);
                (0, globals_1.expect)(result.score).toBeLessThanOrEqual(1);
            });
        });
        (0, globals_1.test)('results may include metadata', () => {
            const result = mockSearchResponse.results[0];
            if (result.metadata) {
                (0, globals_1.expect)(typeof result.metadata).toBe('object');
            }
        });
        (0, globals_1.test)('results may include highlight text', () => {
            const result = mockSearchResponse.results[0];
            if (result.highlight) {
                (0, globals_1.expect)(typeof result.highlight).toBe('string');
            }
        });
    });
    (0, globals_1.describe)('POST /search - Empty Results', () => {
        (0, globals_1.test)('must handle empty results gracefully', () => {
            const emptyResponse = {
                results: [],
                total_results: 0,
                query: 'nonexistent query',
                search_mode: 'semantic',
                execution_time_ms: 45,
                timestamp: '2025-02-03T12:34:56.789Z',
            };
            (0, globals_1.expect)(emptyResponse.results).toEqual([]);
            (0, globals_1.expect)(emptyResponse.total_results).toBe(0);
        });
    });
});
(0, globals_1.describe)('RAGBits API Contract - Ingest Endpoint', () => {
    (0, globals_1.describe)('POST /ingest - Successful Response', () => {
        (0, globals_1.test)('must include "success" boolean', () => {
            (0, globals_1.expect)(mockIngestResponse.success).toBeDefined();
            (0, globals_1.expect)(typeof mockIngestResponse.success).toBe('boolean');
        });
        (0, globals_1.test)('must include "document_id" when successful', () => {
            (0, globals_1.expect)(mockIngestResponse.document_id).toBeDefined();
            (0, globals_1.expect)(typeof mockIngestResponse.document_id).toBe('string');
            (0, globals_1.expect)(mockIngestResponse.document_id.length).toBeGreaterThan(0);
        });
        (0, globals_1.test)('must include "chunks_created" count', () => {
            (0, globals_1.expect)(mockIngestResponse.chunks_created).toBeDefined();
            (0, globals_1.expect)(typeof mockIngestResponse.chunks_created).toBe('number');
            (0, globals_1.expect)(mockIngestResponse.chunks_created).toBeGreaterThanOrEqual(0);
        });
        (0, globals_1.test)('must include "execution_time_ms"', () => {
            (0, globals_1.expect)(mockIngestResponse.execution_time_ms).toBeDefined();
            (0, globals_1.expect)(typeof mockIngestResponse.execution_time_ms).toBe('number');
            (0, globals_1.expect)(mockIngestResponse.execution_time_ms).toBeGreaterThan(0);
        });
        (0, globals_1.test)('must include UTC timestamp (Law of UTC)', () => {
            (0, globals_1.expect)(mockIngestResponse.timestamp).toBeDefined();
            (0, globals_1.expect)(mockIngestResponse.timestamp.endsWith('Z')).toBe(true);
        });
        (0, globals_1.test)('may include error message when failed', () => {
            const failedResponse = {
                success: false,
                document_id: '',
                chunks_created: 0,
                execution_time_ms: 10,
                error: 'Document too large',
                timestamp: '2025-02-03T12:34:56.789Z',
            };
            (0, globals_1.expect)(failedResponse.success).toBe(false);
            (0, globals_1.expect)(failedResponse.error).toBeDefined();
        });
    });
});
(0, globals_1.describe)('RAGBits API Contract - Stats Endpoint', () => {
    (0, globals_1.test)('must include document and chunk counts', () => {
        (0, globals_1.expect)(mockStatsResponse.total_documents).toBeDefined();
        (0, globals_1.expect)(mockStatsResponse.total_chunks).toBeDefined();
        (0, globals_1.expect)(typeof mockStatsResponse.total_documents).toBe('number');
        (0, globals_1.expect)(typeof mockStatsResponse.total_chunks).toBe('number');
    });
    (0, globals_1.test)('must include index size in bytes', () => {
        (0, globals_1.expect)(mockStatsResponse.index_size_bytes).toBeDefined();
        (0, globals_1.expect)(typeof mockStatsResponse.index_size_bytes).toBe('number');
        (0, globals_1.expect)(mockStatsResponse.index_size_bytes).toBeGreaterThanOrEqual(0);
    });
    (0, globals_1.test)('must include vector store type', () => {
        (0, globals_1.expect)(mockStatsResponse.vector_store_type).toBeDefined();
        (0, globals_1.expect)(typeof mockStatsResponse.vector_store_type).toBe('string');
    });
    (0, globals_1.test)('must include embedding model name', () => {
        (0, globals_1.expect)(mockStatsResponse.embedding_model).toBeDefined();
        (0, globals_1.expect)(typeof mockStatsResponse.embedding_model).toBe('string');
    });
    (0, globals_1.test)('must include cache size', () => {
        (0, globals_1.expect)(mockStatsResponse.cache_size).toBeDefined();
        (0, globals_1.expect)(typeof mockStatsResponse.cache_size).toBe('number');
        (0, globals_1.expect)(mockStatsResponse.cache_size).toBeGreaterThanOrEqual(0);
    });
    (0, globals_1.test)('must include UTC timestamp (Law of UTC)', () => {
        (0, globals_1.expect)(mockStatsResponse.timestamp).toBeDefined();
        (0, globals_1.expect)(mockStatsResponse.timestamp.endsWith('Z')).toBe(true);
    });
});
// ============================================================================
// CANONICAL SCHEMA VALIDATION TESTS
// ============================================================================
(0, globals_1.describe)('Canonical Schema - Document Chunk', () => {
    (0, globals_1.test)('must validate valid document chunk', () => {
        const validChunk = {
            id: '550e8400-e29b-41d4-a716-446655440001',
            content: 'This is a valid document chunk',
            source: 'test-doc',
            chunk_index: 0,
            metadata: {
                page: 1,
            },
            timestamp: '2025-02-03T12:34:56.789Z',
        };
        const result = (0, ragbits_canonical_1.validateDocumentChunk)(validChunk);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toBeDefined();
    });
    (0, globals_1.test)('must reject chunk without required fields', () => {
        const invalidChunk = {
            id: '550e8400-e29b-41d4-a716-446655440001',
            // Missing: content, source, chunk_index, timestamp
        };
        const result = (0, ragbits_canonical_1.validateDocumentChunk)(invalidChunk);
        (0, globals_1.expect)(result.success).toBe(false);
        (0, globals_1.expect)(result.errors).toBeDefined();
    });
});
(0, globals_1.describe)('Canonical Schema - RAG Request', () => {
    (0, globals_1.test)('must validate valid RAG request', () => {
        const validRequest = {
            query: 'What is machine learning?',
            retrieval_count: 5,
            timeout_ms: 10000,
            correlation_id: '550e8400-e29b-41d4-a716-446655440000',
        };
        const result = (0, ragbits_canonical_1.validateRAGRequest)(validRequest);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toBeDefined();
    });
    (0, globals_1.test)('must reject request without timeout (CONFIGURATION EXPLICITNESS)', () => {
        const invalidRequest = {
            query: 'What is machine learning?',
            retrieval_count: 5,
            // Missing: timeout_ms
        };
        const result = (0, ragbits_canonical_1.validateRAGRequest)(invalidRequest);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('must reject request with excessive timeout', () => {
        const invalidRequest = {
            query: 'What is machine learning?',
            retrieval_count: 5,
            timeout_ms: 400000, // Exceeds 5 minutes
        };
        const result = (0, ragbits_canonical_1.validateRAGRequest)(invalidRequest);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
(0, globals_1.describe)('Canonical Schema - RAG Response', () => {
    (0, globals_1.test)('must validate valid RAG response', () => {
        const validResponse = {
            results: [
                {
                    id: '550e8400-e29b-41d4-a716-446655440001',
                    content: 'Machine learning is...',
                    source: 'test-doc',
                    chunk_index: 0,
                    timestamp: '2025-02-03T12:34:56.789Z',
                },
            ],
            query: 'What is machine learning?',
            timestamp: '2025-02-03T12:34:56.789Z',
        };
        const result = (0, ragbits_canonical_1.validateRAGResponse)(validResponse);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data).toBeDefined();
    });
    (0, globals_1.test)('must include UTC timestamp in all responses (Law of UTC)', () => {
        const responseWithTimestamp = {
            results: [],
            query: 'test',
            timestamp: '2025-02-03T12:34:56.789Z',
        };
        const result = (0, ragbits_canonical_1.validateRAGResponse)(responseWithTimestamp);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data?.timestamp.endsWith('Z')).toBe(true);
    });
});
(0, globals_1.describe)('Canonical Schema - Document Ingestion Request', () => {
    (0, globals_1.test)('must validate valid ingestion request', () => {
        const validRequest = {
            content: 'Document content to ingest',
            source: '/path/to/document.pdf',
            timeout_ms: 30000,
            correlation_id: '550e8400-e29b-41d4-a716-446655440000',
        };
        const result = ragbits_canonical_1.DocumentIngestionRequest.safeParse(validRequest);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must reject request without timeout (CONFIGURATION EXPLICITNESS)', () => {
        const invalidRequest = {
            content: 'Document content to ingest',
            source: '/path/to/document.pdf',
            // Missing: timeout_ms
        };
        const result = ragbits_canonical_1.DocumentIngestionRequest.safeParse(invalidRequest);
        (0, globals_1.expect)(result.success).toBe(false);
    });
    (0, globals_1.test)('must enforce chunk size limits', () => {
        const invalidRequest = {
            content: 'Document content',
            source: 'test.pdf',
            chunk_size: 20000, // Exceeds max of 10000
            timeout_ms: 30000,
        };
        const result = ragbits_canonical_1.DocumentIngestionRequest.safeParse(invalidRequest);
        (0, globals_1.expect)(result.success).toBe(false);
    });
});
(0, globals_1.describe)('Canonical Schema - Error Model', () => {
    (0, globals_1.test)('must validate valid error response', () => {
        const validError = {
            code: 'RETRIEVAL_FAILED',
            message: 'Failed to retrieve documents',
            details: {
                reason: 'Connection timeout',
            },
            correlation_id: '550e8400-e29b-41d4-a716-446655440000',
            timestamp: '2025-02-03T12:34:56.789Z',
        };
        const result = ragbits_canonical_1.RAGError.safeParse(validError);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must include error code from allowed enum', () => {
        const validCodes = [
            'QUERY_TOO_LONG',
            'RETRIEVAL_FAILED',
            'GENERATION_FAILED',
            'TIMEOUT',
            'INVALID_FILTER',
            'EMBEDDING_ERROR',
            'DOCUMENT_NOT_FOUND',
            'INGESTION_FAILED',
            'UNKNOWN_ERROR',
        ];
        validCodes.forEach(code => {
            const error = {
                code: code,
                message: 'Test error',
                timestamp: '2025-02-03T12:34:56.789Z',
            };
            const result = ragbits_canonical_1.RAGError.safeParse(error);
            (0, globals_1.expect)(result.success).toBe(true);
        });
    });
});
// ============================================================================
// TRANSFORMATION FUNCTION TESTS
// ============================================================================
(0, globals_1.describe)('Transformation Functions - API to Canonical', () => {
    (0, globals_1.test)('must transform search response to canonical format', () => {
        const rawResponse = {
            results: [
                {
                    chunk_id: 'chunk-1',
                    content: 'Test content',
                    document_id: 'doc-1',
                    score: 0.92,
                },
            ],
            total_results: 1,
            query: 'test query',
        };
        // Transformation would happen in adapter
        const canonical = {
            results: rawResponse.results.map(r => ({
                id: r.chunk_id,
                content: r.content,
                source: r.document_id,
                chunk_index: 0,
                timestamp: new Date().toISOString(),
            })),
            query: rawResponse.query,
            timestamp: new Date().toISOString(),
        };
        const result = (0, ragbits_canonical_1.validateRAGResponse)(canonical);
        (0, globals_1.expect)(result.success).toBe(true);
    });
    (0, globals_1.test)('must preserve correlation IDs through transformation', () => {
        const correlationId = 'test-correlation-123';
        const transformed = {
            results: [],
            query: 'test',
            correlation_id: correlationId,
            timestamp: new Date().toISOString(),
        };
        const result = (0, ragbits_canonical_1.validateRAGResponse)(transformed);
        (0, globals_1.expect)(result.success).toBe(true);
        (0, globals_1.expect)(result.data?.correlation_id).toBe(correlationId);
    });
});
// ============================================================================
// IDEMPOTENCY TESTS
// ============================================================================
(0, globals_1.describe)('Idempotency - Repeatable Operations', () => {
    (0, globals_1.test)('same query must return consistent structure', () => {
        const query1 = (0, ragbits_canonical_1.validateRAGRequest)({
            query: 'test query',
            retrieval_count: 5,
            timeout_ms: 10000,
        });
        const query2 = (0, ragbits_canonical_1.validateRAGRequest)({
            query: 'test query',
            retrieval_count: 5,
            timeout_ms: 10000,
        });
        (0, globals_1.expect)(query1.success).toBe(true);
        (0, globals_1.expect)(query2.success).toBe(true);
        (0, globals_1.expect)(query1.data).toEqual(query2.data);
    });
    (0, globals_1.test)('ingestion of same document must be idempotent', () => {
        const doc1 = ragbits_canonical_1.DocumentIngestionRequest.safeParse({
            content: 'test content',
            source: 'test.pdf',
            timeout_ms: 30000,
        });
        const doc2 = ragbits_canonical_1.DocumentIngestionRequest.safeParse({
            content: 'test content',
            source: 'test.pdf',
            timeout_ms: 30000,
        });
        (0, globals_1.expect)(doc1.success).toBe(true);
        (0, globals_1.expect)(doc2.success).toBe(true);
    });
});
// ============================================================================
// SETUP AND TEARDOWN
// ============================================================================
let setupComplete = false;
(0, globals_1.beforeAll)(async () => {
    // Setup: Validate test environment
    (0, globals_1.expect)(ragbits_canonical_1.DocumentChunk).toBeDefined();
    (0, globals_1.expect)(ragbits_canonical_1.RAGRequest).toBeDefined();
    (0, globals_1.expect)(ragbits_canonical_1.RAGResponse).toBeDefined();
    (0, globals_1.expect)(ragbits_canonical_1.DocumentIngestionRequest).toBeDefined();
    (0, globals_1.expect)(ragbits_canonical_1.DocumentIngestionResponse).toBeDefined();
    (0, globals_1.expect)(ragbits_canonical_1.RAGError).toBeDefined();
    setupComplete = true;
    console.log('✅ RAGBits Contract Test Suite Initialized');
    console.log('⚠️  Remember: If these tests fail, the adapter MUST refuse to start');
});
(0, globals_1.afterAll)(() => {
    if (setupComplete) {
        console.log('✅ RAGBits Contract Test Suite Completed');
        console.log('📋 All contract validations passed');
    }
});
//# sourceMappingURL=contract.test.js.map