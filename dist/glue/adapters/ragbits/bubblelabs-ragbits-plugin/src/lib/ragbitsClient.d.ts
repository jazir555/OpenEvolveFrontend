export interface RagbitsClientConfig {
    serverUrl: string;
    apiKey?: string;
    timeout?: number;
}
export declare class RagbitsClient {
    private config;
    private correlationId;
    constructor(config: RagbitsClientConfig);
    /**
     * Update client configuration
     */
    configure(config: Partial<RagbitsClientConfig>): void;
    /**
     * Test connection to RAGBits server
     */
    testConnection(): Promise<boolean>;
    /**
     * Search for documents
     */
    search(request: {
        query: string;
        topK?: number;
        scoreThreshold?: number;
        filter?: Record<string, any>;
        enableHybridSearch?: boolean;
        enableReranking?: boolean;
    }): Promise<any>;
    /**
     * Ingest a document
     */
    ingest(request: {
        content: string;
        metadata: Record<string, any>;
    }): Promise<any>;
    /**
     * Batch ingest documents
     */
    batchIngest(requests: Array<{
        content: string;
        metadata: Record<string, any>;
    }>): Promise<any[]>;
    /**
     * Get index statistics
     */
    getIndexStats(): Promise<any>;
    /**
     * Clear cache
     */
    clearCache(): Promise<any>;
    /**
     * Perform HTTP request with timeout and error handling
     */
    private fetch;
}
//# sourceMappingURL=ragbitsClient.d.ts.map