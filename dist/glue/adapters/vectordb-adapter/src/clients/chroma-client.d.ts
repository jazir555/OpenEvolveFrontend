/**
 * Chroma Client Implementation
 *
 * Chroma-specific vector database client with circuit breaker and retry logic.
 */
import { CollectionConfig, SearchQuery, SearchResult, UpsertRequest, UpsertResponse, DeleteRequest, DeleteResponse, CollectionInfo, HealthCheckResponse } from '../../../schemas/vectordb-canonical';
export interface ChromaClientConfig {
    url: string;
    timeout?: number;
    maxRetries?: number;
}
export declare class ChromaClient {
    private logger;
    private circuitBreaker;
    private config;
    private baseUrl;
    private headers;
    constructor(config: ChromaClientConfig);
    /**
     * Health check
     */
    healthCheck(): Promise<HealthCheckResponse>;
    /**
     * Create collection
     */
    createCollection(config: CollectionConfig): Promise<void>;
    /**
     * Get collection info
     */
    getCollectionInfo(collectionName: string): Promise<CollectionInfo>;
    /**
     * Upsert vectors
     */
    upsert(request: UpsertRequest): Promise<UpsertResponse>;
    /**
     * Search vectors
     */
    search(collectionName: string, query: SearchQuery): Promise<SearchResult[]>;
    /**
     * Delete vectors
     */
    delete(request: DeleteRequest): Promise<DeleteResponse>;
    /**
     * List collections
     */
    listCollections(): Promise<string[]>;
}
//# sourceMappingURL=chroma-client.d.ts.map