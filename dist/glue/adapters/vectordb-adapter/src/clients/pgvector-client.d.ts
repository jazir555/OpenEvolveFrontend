/**
 * pgvector Client Implementation
 *
 * pgvector-specific vector database client with circuit breaker and retry logic.
 * Uses PostgreSQL with the pgvector extension for vector similarity search.
 */
import { CollectionConfig, SearchQuery, SearchResult, UpsertRequest, UpsertResponse, DeleteRequest, DeleteResponse, CollectionInfo, HealthCheckResponse } from '../../../schemas/vectordb-canonical';
export interface PgvectorClientConfig {
    connectionString: string;
    tableName?: string;
    timeout?: number;
    maxRetries?: number;
}
export declare class PgvectorClient {
    private logger;
    private circuitBreaker;
    private config;
    private tableName;
    private pool;
    constructor(config: PgvectorClientConfig);
    /**
     * Initialize connection pool
     */
    private initializePool;
    /**
     * Execute SQL query
     */
    private query;
    /**
     * Health check
     */
    healthCheck(): Promise<HealthCheckResponse>;
    /**
     * Create collection (table with vector column)
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
     * List collections (tables with vector columns)
     */
    listCollections(): Promise<string[]>;
    /**
     * Close connection pool
     */
    close(): Promise<void>;
}
//# sourceMappingURL=pgvector-client.d.ts.map