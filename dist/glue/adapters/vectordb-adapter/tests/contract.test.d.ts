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
import { VectorEntry, CollectionConfig, SearchQuery, SearchResult, UpsertRequest, UpsertResponse, DeleteRequest, DeleteResponse, CollectionInfo } from '../../../schemas/vectordb-canonical';
declare const mockHealthCheckResponse: {
    status: "healthy";
    backend_type: any;
    connected: boolean;
    latency_ms: number;
    collections_count: number;
    timestamp: string;
};
declare const mockDegradedHealthCheckResponse: {
    status: "degraded";
    backend_type: any;
    connected: boolean;
    error: string;
    timestamp: string;
};
declare const mockVectorEntry: VectorEntry;
declare const mockSparseVectorEntry: VectorEntry;
declare const mockCollectionConfig: CollectionConfig;
declare const mockSearchQuery: SearchQuery;
declare const mockSearchResult: SearchResult;
declare const mockUpsertRequest: UpsertRequest;
declare const mockUpsertResponse: UpsertResponse;
declare const mockDeleteRequest: DeleteRequest;
declare const mockDeleteResponse: DeleteResponse;
declare const mockCollectionInfo: CollectionInfo;
export { mockHealthCheckResponse, mockDegradedHealthCheckResponse, mockVectorEntry, mockSparseVectorEntry, mockCollectionConfig, mockSearchQuery, mockSearchResult, mockUpsertRequest, mockUpsertResponse, mockDeleteRequest, mockDeleteResponse, mockCollectionInfo, };
//# sourceMappingURL=contract.test.d.ts.map