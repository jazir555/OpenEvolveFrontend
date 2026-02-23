/**
 * Pinecone Client Implementation
 *
 * Pinecone-specific vector database client with circuit breaker and retry logic.
 */
import { CollectionConfig, SearchQuery, SearchResult, UpsertRequest, UpsertResponse, DeleteRequest, DeleteResponse, CollectionInfo, HealthCheckResponse } from '../../../schemas/vectordb-canonical';
export interface PineconeClientConfig {
    apiKey: string;
    environment?: string;
    timeout?: number;
    maxRetries?: number;
}
export declare class PineconeClient {
    private logger;
    private circuitBreaker;
    private config;
    private baseUrl;
    private headers;
    constructor(config: PineconeClientConfig);
    /**
     * Health check
     */
    healthCheck(): Promise<HealthCheckResponse>;
    /**
     * Create index (collection in Pinecone)
     */
    createCollection(config: CollectionConfig): Promise<void>;
    /**
     * Get collection info
     */
    getCollectionInfo(collectionName: string): Promise<CollectionInfo>;
    /**
     * Get index URL for vector operations
     */
    private getIndexUrl;
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
     * List collections (indexes)
     */
    listCollections(): Promise<string[]>;
}
//# sourceMappingURL=pinecone-client.d.ts.map