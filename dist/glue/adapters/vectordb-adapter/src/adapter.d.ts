/**
 * Vector DB Adapter - Main Entry Point
 *
 * Multi-backend vector database adapter supporting:
 * - Qdrant
 * - Pinecone
 * - Chroma
 * - pgvector
 *
 * Following CLAUDE.md Federation Constitution:
 * - Law of Configuration Explicitness: All config via environment variables
 * - Law of Runtime Truth: Validates backend capabilities at runtime
 * - Law of UTC: All timestamps in UTC ISO-8601
 * - JSON Lines logging with correlation_id
 */
import { VectorDBType, CollectionConfig, SearchQuery, SearchResult, UpsertRequest, UpsertResponse, DeleteRequest, DeleteResponse, CollectionInfo, HealthCheckResponse } from '../../schemas/vectordb-canonical';
export interface VectorDBAdapterConfig {
    backendType: VectorDBType;
    url?: string;
    apiKey?: string;
    connectionString?: string;
    timeout?: number;
    maxRetries?: number;
    environment?: string;
}
/**
 * Vector DB Adapter Class
 *
 * Main adapter that routes operations to the appropriate backend client.
 */
export declare class VectorDBAdapter {
    private logger;
    private config;
    private client;
    constructor(config?: VectorDBAdapterConfig);
    /**
     * Load configuration from environment variables
     * Following Law of Configuration Explicitness
     */
    private loadConfigFromEnv;
    /**
     * Initialize the appropriate backend client
     */
    private initializeClient;
    /**
     * Health check
     * Verifies the backend is accessible and responsive
     */
    healthCheck(): Promise<HealthCheckResponse>;
    /**
     * Create a collection (table/index) in the vector database
     */
    createCollection(config: CollectionConfig): Promise<void>;
    /**
     * Get information about a collection
     */
    getCollectionInfo(collectionName: string): Promise<CollectionInfo>;
    /**
     * List all collections
     */
    listCollections(): Promise<string[]>;
    /**
     * Upsert vectors to a collection
     * Following Law of Idempotency: Safe to run multiple times
     */
    upsert(request: UpsertRequest): Promise<UpsertResponse>;
    /**
     * Search for similar vectors
     */
    search(collectionName: string, query: SearchQuery): Promise<SearchResult[]>;
    /**
     * Delete vectors from a collection
     * Following Law of Idempotency: Safe to run multiple times
     */
    delete(request: DeleteRequest): Promise<DeleteResponse>;
    /**
     * Get the backend type
     */
    getBackendType(): VectorDBType;
    /**
     * Close the adapter and release resources
     */
    close(): Promise<void>;
}
/**
 * Create a VectorDB adapter instance from environment variables
 */
export declare function createVectorDBAdapter(): VectorDBAdapter;
/**
 * Create a VectorDB adapter instance with explicit configuration
 */
export declare function createVectorDBAdapterWithConfig(config: VectorDBAdapterConfig): VectorDBAdapter;
//# sourceMappingURL=adapter.d.ts.map