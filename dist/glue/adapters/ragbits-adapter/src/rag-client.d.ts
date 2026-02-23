/**
 * RAGBits HTTP Client
 *
 * Handles HTTP communication with the RAGBits server.
 * Following Federation Constitution:
 * - CONFIGURATION EXPLICITNESS: API URL via env, no defaults
 * - TIMEOUTS: MANDATORY on all requests
 * - STRUCTURED LOGGING: JSON Lines with correlation_id
 *
 * @module rag-client
 */
export interface RAGClientConfig {
    api_url: string;
    timeout_ms: number;
    api_key?: string;
}
export interface RAGSearchRequest {
    query: string;
    top_k?: number;
    filters?: Record<string, any>;
    score_threshold?: number;
    enable_hybrid_search?: boolean;
    enable_reranking?: boolean;
}
export interface RAGIngestRequest {
    content: string;
    metadata: Record<string, any>;
    source?: string;
}
/**
 * RAGBits HTTP Client
 *
 * Provides typed methods for interacting with the RAGBits REST API.
 */
export declare class RAGClient {
    private config;
    constructor(config: RAGClientConfig);
    /**
     * Test connection to RAGBits server
     */
    testConnection(correlationId?: string): Promise<boolean>;
    /**
     * Search for documents
     */
    search(request: RAGSearchRequest, correlationId?: string): Promise<any>;
    /**
     * Ingest a document
     */
    ingest(request: RAGIngestRequest, correlationId?: string): Promise<any>;
    /**
     * Batch ingest documents
     */
    batchIngest(requests: RAGIngestRequest[], correlationId?: string): Promise<any>;
    /**
     * Get index statistics
     */
    getStats(correlationId?: string): Promise<any>;
    /**
     * Clear cache
     */
    clearCache(correlationId?: string): Promise<any>;
    /**
     * Perform HTTP request with timeout and error handling
     *
     * TIMEOUT: MANDATORY - All requests must have timeout
     */
    private fetch;
    /**
     * Update client configuration
     */
    configure(config: Partial<RAGClientConfig>): void;
    /**
     * Get current configuration
     */
    getConfig(): Readonly<RAGClientConfig>;
}
//# sourceMappingURL=rag-client.d.ts.map