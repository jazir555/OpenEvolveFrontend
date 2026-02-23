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
declare const mockHealthResponse: {
    status: string;
    ragbits_available: boolean;
    vector_store_connected: boolean;
    embedding_model_loaded: boolean;
    timestamp: string;
};
declare const mockSearchResponse: {
    results: ({
        chunk_id: string;
        content: string;
        document_id: string;
        score: number;
        metadata: {
            document_type: string;
            source: string;
            tags: string[];
        };
        highlight: string;
    } | {
        chunk_id: string;
        content: string;
        document_id: string;
        score: number;
        metadata: {
            document_type: string;
            source: string;
            tags?: undefined;
        };
        highlight?: undefined;
    })[];
    total_results: number;
    query: string;
    search_mode: string;
    execution_time_ms: number;
    timestamp: string;
};
declare const mockIngestResponse: {
    success: boolean;
    document_id: string;
    chunks_created: number;
    execution_time_ms: number;
    timestamp: string;
};
declare const mockStatsResponse: {
    total_documents: number;
    total_chunks: number;
    index_size_bytes: number;
    vector_store_type: string;
    embedding_model: string;
    cache_size: number;
    timestamp: string;
};
declare const mockErrorResponse: {
    code: string;
    message: string;
    details: {
        reason: string;
        query: string;
    };
    correlation_id: string;
    timestamp: string;
};
export { mockHealthResponse, mockSearchResponse, mockIngestResponse, mockStatsResponse, mockErrorResponse, };
//# sourceMappingURL=contract.test.d.ts.map