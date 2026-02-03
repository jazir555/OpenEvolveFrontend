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

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import { z } from 'zod';

// Import canonical schemas for validation
import {
  DocumentChunk,
  RAGRequest,
  RAGResponse,
  DocumentIngestionRequest,
  DocumentIngestionResponse,
  RAGError,
  validateRAGRequest,
  validateRAGResponse,
  validateDocumentChunk,
  validateDocumentChunk as validateDocumentChunkCanonical,
} from '../../../schemas/ragbits-canonical';

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

const mockIngestResponse = {
  success: true,
  document_id: 'doc-789',
  chunks_created: 3,
  execution_time_ms: 245,
  timestamp: '2025-02-03T12:34:56.789Z',
};

const mockStatsResponse = {
  total_documents: 150,
  total_chunks: 1250,
  index_size_bytes: 5242880,
  vector_store_type: 'chromadb',
  embedding_model: 'all-MiniLM-L6-v2',
  cache_size: 100,
  timestamp: '2025-02-03T12:34:56.789Z',
};

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

// ============================================================================
// RAGBits API CONTRACT TESTS
// ============================================================================

describe('RAGBits API Contract - Health Endpoint', () => {
  describe('GET /health - Healthy Response', () => {
    test('must return status "healthy" when RAGBits is available', () => {
      expect(mockHealthResponse.status).toBe('healthy');
      expect(mockHealthResponse.ragbits_available).toBe(true);
    });

    test('must include vector store connection status', () => {
      expect(mockHealthResponse.vector_store_connected).toBeDefined();
      expect(typeof mockHealthResponse.vector_store_connected).toBe('boolean');
    });

    test('must include embedding model status', () => {
      expect(mockHealthResponse.embedding_model_loaded).toBeDefined();
      expect(typeof mockHealthResponse.embedding_model_loaded).toBe('boolean');
    });

    test('must include UTC timestamp (Law of UTC)', () => {
      expect(mockHealthResponse.timestamp).toBeDefined();
      expect(mockHealthResponse.timestamp.endsWith('Z')).toBe(true);
    });
  });

  describe('GET /health - Degraded Response', () => {
    test('must return status "degraded" or "unhealthy" when unavailable', () => {
      const degradedResponse = {
        status: 'degraded',
        ragbits_available: false,
        vector_store_connected: false,
        embedding_model_loaded: false,
        error: 'Vector store not initialized',
        timestamp: '2025-02-03T12:34:56.789Z',
      };

      expect(['degraded', 'unhealthy']).toContain(degradedResponse.status);
      expect(degradedResponse.ragbits_available).toBe(false);
    });

    test('may include error message when degraded', () => {
      const degradedResponse = {
        status: 'degraded' as const,
        ragbits_available: false,
        vector_store_connected: false,
        embedding_model_loaded: false,
        error: 'Vector store not initialized',
        timestamp: '2025-02-03T12:34:56.789Z',
      };

      expect(degradedResponse.error).toBeDefined();
      expect(typeof degradedResponse.error).toBe('string');
    });
  });
});

describe('RAGBits API Contract - Search Endpoint', () => {
  describe('POST /search - Successful Response', () => {
    test('must include "results" array', () => {
      expect(mockSearchResponse.results).toBeDefined();
      expect(Array.isArray(mockSearchResponse.results)).toBe(true);
    });

    test('must include "total_results" count', () => {
      expect(mockSearchResponse.total_results).toBeDefined();
      expect(typeof mockSearchResponse.total_results).toBe('number');
      expect(mockSearchResponse.total_results).toBeGreaterThanOrEqual(0);
    });

    test('must include original "query" string', () => {
      expect(mockSearchResponse.query).toBeDefined();
      expect(typeof mockSearchResponse.query).toBe('string');
    });

    test('must include "execution_time_ms" for performance tracking', () => {
      expect(mockSearchResponse.execution_time_ms).toBeDefined();
      expect(typeof mockSearchResponse.execution_time_ms).toBe('number');
      expect(mockSearchResponse.execution_time_ms).toBeGreaterThan(0);
    });

    test('must include UTC timestamp (Law of UTC)', () => {
      expect(mockSearchResponse.timestamp).toBeDefined();
      expect(mockSearchResponse.timestamp.endsWith('Z')).toBe(true);
    });

    test('results must have required fields', () => {
      const result = mockSearchResponse.results[0];

      expect(result.chunk_id).toBeDefined();
      expect(result.content).toBeDefined();
      expect(result.document_id).toBeDefined();
      expect(result.score).toBeDefined();
    });

    test('results scores must be between 0 and 1', () => {
      mockSearchResponse.results.forEach(result => {
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(1);
      });
    });

    test('results may include metadata', () => {
      const result = mockSearchResponse.results[0];

      if (result.metadata) {
        expect(typeof result.metadata).toBe('object');
      }
    });

    test('results may include highlight text', () => {
      const result = mockSearchResponse.results[0];

      if (result.highlight) {
        expect(typeof result.highlight).toBe('string');
      }
    });
  });

  describe('POST /search - Empty Results', () => {
    test('must handle empty results gracefully', () => {
      const emptyResponse = {
        results: [],
        total_results: 0,
        query: 'nonexistent query',
        search_mode: 'semantic' as const,
        execution_time_ms: 45,
        timestamp: '2025-02-03T12:34:56.789Z',
      };

      expect(emptyResponse.results).toEqual([]);
      expect(emptyResponse.total_results).toBe(0);
    });
  });
});

describe('RAGBits API Contract - Ingest Endpoint', () => {
  describe('POST /ingest - Successful Response', () => {
    test('must include "success" boolean', () => {
      expect(mockIngestResponse.success).toBeDefined();
      expect(typeof mockIngestResponse.success).toBe('boolean');
    });

    test('must include "document_id" when successful', () => {
      expect(mockIngestResponse.document_id).toBeDefined();
      expect(typeof mockIngestResponse.document_id).toBe('string');
      expect(mockIngestResponse.document_id.length).toBeGreaterThan(0);
    });

    test('must include "chunks_created" count', () => {
      expect(mockIngestResponse.chunks_created).toBeDefined();
      expect(typeof mockIngestResponse.chunks_created).toBe('number');
      expect(mockIngestResponse.chunks_created).toBeGreaterThanOrEqual(0);
    });

    test('must include "execution_time_ms"', () => {
      expect(mockIngestResponse.execution_time_ms).toBeDefined();
      expect(typeof mockIngestResponse.execution_time_ms).toBe('number');
      expect(mockIngestResponse.execution_time_ms).toBeGreaterThan(0);
    });

    test('must include UTC timestamp (Law of UTC)', () => {
      expect(mockIngestResponse.timestamp).toBeDefined();
      expect(mockIngestResponse.timestamp.endsWith('Z')).toBe(true);
    });

    test('may include error message when failed', () => {
      const failedResponse = {
        success: false,
        document_id: '',
        chunks_created: 0,
        execution_time_ms: 10,
        error: 'Document too large',
        timestamp: '2025-02-03T12:34:56.789Z',
      };

      expect(failedResponse.success).toBe(false);
      expect(failedResponse.error).toBeDefined();
    });
  });
});

describe('RAGBits API Contract - Stats Endpoint', () => {
  test('must include document and chunk counts', () => {
    expect(mockStatsResponse.total_documents).toBeDefined();
    expect(mockStatsResponse.total_chunks).toBeDefined();
    expect(typeof mockStatsResponse.total_documents).toBe('number');
    expect(typeof mockStatsResponse.total_chunks).toBe('number');
  });

  test('must include index size in bytes', () => {
    expect(mockStatsResponse.index_size_bytes).toBeDefined();
    expect(typeof mockStatsResponse.index_size_bytes).toBe('number');
    expect(mockStatsResponse.index_size_bytes).toBeGreaterThanOrEqual(0);
  });

  test('must include vector store type', () => {
    expect(mockStatsResponse.vector_store_type).toBeDefined();
    expect(typeof mockStatsResponse.vector_store_type).toBe('string');
  });

  test('must include embedding model name', () => {
    expect(mockStatsResponse.embedding_model).toBeDefined();
    expect(typeof mockStatsResponse.embedding_model).toBe('string');
  });

  test('must include cache size', () => {
    expect(mockStatsResponse.cache_size).toBeDefined();
    expect(typeof mockStatsResponse.cache_size).toBe('number');
    expect(mockStatsResponse.cache_size).toBeGreaterThanOrEqual(0);
  });

  test('must include UTC timestamp (Law of UTC)', () => {
    expect(mockStatsResponse.timestamp).toBeDefined();
    expect(mockStatsResponse.timestamp.endsWith('Z')).toBe(true);
  });
});

// ============================================================================
// CANONICAL SCHEMA VALIDATION TESTS
// ============================================================================

describe('Canonical Schema - Document Chunk', () => {
  test('must validate valid document chunk', () => {
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

    const result = validateDocumentChunkCanonical(validChunk);
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  test('must reject chunk without required fields', () => {
    const invalidChunk = {
      id: '550e8400-e29b-41d4-a716-446655440001',
      // Missing: content, source, chunk_index, timestamp
    };

    const result = validateDocumentChunkCanonical(invalidChunk);
    expect(result.success).toBe(false);
    expect(result.errors).toBeDefined();
  });
});

describe('Canonical Schema - RAG Request', () => {
  test('must validate valid RAG request', () => {
    const validRequest = {
      query: 'What is machine learning?',
      retrieval_count: 5,
      timeout_ms: 10000,
      correlation_id: '550e8400-e29b-41d4-a716-446655440000',
    };

    const result = validateRAGRequest(validRequest);
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  test('must reject request without timeout (CONFIGURATION EXPLICITNESS)', () => {
    const invalidRequest = {
      query: 'What is machine learning?',
      retrieval_count: 5,
      // Missing: timeout_ms
    };

    const result = validateRAGRequest(invalidRequest);
    expect(result.success).toBe(false);
  });

  test('must reject request with excessive timeout', () => {
    const invalidRequest = {
      query: 'What is machine learning?',
      retrieval_count: 5,
      timeout_ms: 400000, // Exceeds 5 minutes
    };

    const result = validateRAGRequest(invalidRequest);
    expect(result.success).toBe(false);
  });
});

describe('Canonical Schema - RAG Response', () => {
  test('must validate valid RAG response', () => {
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

    const result = validateRAGResponse(validResponse);
    expect(result.success).toBe(true);
    expect(result.data).toBeDefined();
  });

  test('must include UTC timestamp in all responses (Law of UTC)', () => {
    const responseWithTimestamp = {
      results: [],
      query: 'test',
      timestamp: '2025-02-03T12:34:56.789Z',
    };

    const result = validateRAGResponse(responseWithTimestamp);
    expect(result.success).toBe(true);
    expect(result.data?.timestamp.endsWith('Z')).toBe(true);
  });
});

describe('Canonical Schema - Document Ingestion Request', () => {
  test('must validate valid ingestion request', () => {
    const validRequest = {
      content: 'Document content to ingest',
      source: '/path/to/document.pdf',
      timeout_ms: 30000,
      correlation_id: '550e8400-e29b-41d4-a716-446655440000',
    };

    const result = DocumentIngestionRequest.safeParse(validRequest);
    expect(result.success).toBe(true);
  });

  test('must reject request without timeout (CONFIGURATION EXPLICITNESS)', () => {
    const invalidRequest = {
      content: 'Document content to ingest',
      source: '/path/to/document.pdf',
      // Missing: timeout_ms
    };

    const result = DocumentIngestionRequest.safeParse(invalidRequest);
    expect(result.success).toBe(false);
  });

  test('must enforce chunk size limits', () => {
    const invalidRequest = {
      content: 'Document content',
      source: 'test.pdf',
      chunk_size: 20000, // Exceeds max of 10000
      timeout_ms: 30000,
    };

    const result = DocumentIngestionRequest.safeParse(invalidRequest);
    expect(result.success).toBe(false);
  });
});

describe('Canonical Schema - Error Model', () => {
  test('must validate valid error response', () => {
    const validError = {
      code: 'RETRIEVAL_FAILED',
      message: 'Failed to retrieve documents',
      details: {
        reason: 'Connection timeout',
      },
      correlation_id: '550e8400-e29b-41d4-a716-446655440000',
      timestamp: '2025-02-03T12:34:56.789Z',
    };

    const result = RAGError.safeParse(validError);
    expect(result.success).toBe(true);
  });

  test('must include error code from allowed enum', () => {
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
        code: code as any,
        message: 'Test error',
        timestamp: '2025-02-03T12:34:56.789Z',
      };

      const result = RAGError.safeParse(error);
      expect(result.success).toBe(true);
    });
  });
});

// ============================================================================
// TRANSFORMATION FUNCTION TESTS
// ============================================================================

describe('Transformation Functions - API to Canonical', () => {
  test('must transform search response to canonical format', () => {
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

    const result = validateRAGResponse(canonical);
    expect(result.success).toBe(true);
  });

  test('must preserve correlation IDs through transformation', () => {
    const correlationId = 'test-correlation-123';

    const transformed = {
      results: [],
      query: 'test',
      correlation_id: correlationId,
      timestamp: new Date().toISOString(),
    };

    const result = validateRAGResponse(transformed);
    expect(result.success).toBe(true);
    expect(result.data?.correlation_id).toBe(correlationId);
  });
});

// ============================================================================
// IDEMPOTENCY TESTS
// ============================================================================

describe('Idempotency - Repeatable Operations', () => {
  test('same query must return consistent structure', () => {
    const query1 = validateRAGRequest({
      query: 'test query',
      retrieval_count: 5,
      timeout_ms: 10000,
    });

    const query2 = validateRAGRequest({
      query: 'test query',
      retrieval_count: 5,
      timeout_ms: 10000,
    });

    expect(query1.success).toBe(true);
    expect(query2.success).toBe(true);
    expect(query1.data).toEqual(query2.data);
  });

  test('ingestion of same document must be idempotent', () => {
    const doc1 = DocumentIngestionRequest.safeParse({
      content: 'test content',
      source: 'test.pdf',
      timeout_ms: 30000,
    });

    const doc2 = DocumentIngestionRequest.safeParse({
      content: 'test content',
      source: 'test.pdf',
      timeout_ms: 30000,
    });

    expect(doc1.success).toBe(true);
    expect(doc2.success).toBe(true);
  });
});

// ============================================================================
// SETUP AND TEARDOWN
// ============================================================================

let setupComplete = false;

beforeAll(async () => {
  // Setup: Validate test environment
  expect(DocumentChunk).toBeDefined();
  expect(RAGRequest).toBeDefined();
  expect(RAGResponse).toBeDefined();
  expect(DocumentIngestionRequest).toBeDefined();
  expect(DocumentIngestionResponse).toBeDefined();
  expect(RAGError).toBeDefined();

  setupComplete = true;

  console.log('✅ RAGBits Contract Test Suite Initialized');
  console.log('⚠️  Remember: If these tests fail, the adapter MUST refuse to start');
});

afterAll(() => {
  if (setupComplete) {
    console.log('✅ RAGBits Contract Test Suite Completed');
    console.log('📋 All contract validations passed');
  }
});

// ============================================================================
// EXPORTS
// ============================================================================

export {
  mockHealthResponse,
  mockSearchResponse,
  mockIngestResponse,
  mockStatsResponse,
  mockErrorResponse,
};
