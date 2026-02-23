/**
 * Datapizza Client - Production Implementation
 *
 * Law of Configuration Explicitness:
 * - baseUrl is REQUIRED (no magic defaults)
 * - timeout is REQUIRED (crashes loudly if not provided)
 *
 * Follows Federation Constitution:
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Handles failures gracefully
 * - Retry Logic: Exponential backoff for transient failures
 */
export interface DatapizzaClientConfig {
    baseUrl: string;
    apiKey?: string;
    timeout: number;
}
export interface PipelineRunRequest {
    dataSource: string;
    pipelineType: string;
    parameters?: Record<string, unknown>;
}
export interface PipelineRunResponse {
    success: boolean;
    pipelineId: string;
    dataSource: string;
    pipelineType: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    startedAt?: string;
    completedAt?: string;
    error?: string;
}
export interface DataProcessingRequest {
    data: unknown;
    processingType?: string;
    options?: Record<string, unknown>;
}
export interface DataProcessingResponse {
    success: boolean;
    dataId: string;
    processedData: unknown;
    processingType: string;
    metadata?: Record<string, unknown>;
}
export interface DataQueryRequest {
    query: string;
    dataSource?: string;
    limit?: number;
    offset?: number;
}
export interface DataQueryResponse {
    success: boolean;
    query: string;
    results: Array<{
        id: string;
        score: number;
        data: {
            content: string;
            source: string;
            metadata?: Record<string, unknown>;
        };
    }>;
    totalCount: number;
}
export interface PipelineRecommendationResponse {
    recommendedPipeline: string;
    confidence: number;
    alternatives: string[];
    reasoning: string;
}
export interface DataDomainResponse {
    domain: 'structured' | 'unstructured' | 'semi-structured' | 'general';
    confidence: number;
    detectedSchema?: Record<string, unknown>;
}
/**
 * Datapizza API Client
 *
 * Implements actual HTTP calls to Datapizza API with:
 * - Structured logging with correlation IDs
 * - Timeout enforcement (MANDATORY per Law 3.2)
 * - Error classification (transient vs permanent)
 * - Retry logic for transient failures
 */
export declare class DatapizzaClient {
    private config;
    private correlationId;
    constructor(config: DatapizzaClientConfig);
    /**
     * Update client configuration
     */
    configure(config: Partial<DatapizzaClientConfig>): void;
    /**
     * Test connection to Datapizza server
     *
     * @returns true if connection successful
     * @throws Error if connection fails after retries
     */
    testConnection(): Promise<boolean>;
    /**
     * Run a data pipeline
     *
     * @param request - Pipeline run parameters
     * @returns Pipeline execution result
     */
    runPipeline(request: PipelineRunRequest): Promise<PipelineRunResponse>;
    /**
     * Process data
     *
     * @param request - Data processing parameters
     * @returns Processing result
     */
    processData(request: DataProcessingRequest): Promise<DataProcessingResponse>;
    /**
     * Query data
     *
     * @param request - Query parameters
     * @returns Query results
     */
    queryData(request: DataQueryRequest): Promise<DataQueryResponse>;
    /**
     * Get pipeline recommendation
     *
     * @param dataSource - Data source identifier
     * @param context - Additional context
     * @returns Recommended pipeline
     */
    getPipelineRecommendation(dataSource: string, context?: string): Promise<PipelineRecommendationResponse>;
    /**
     * Detect data domain
     *
     * @param data - Data to analyze
     * @returns Detected domain with confidence
     */
    detectDataDomain(data: unknown): Promise<DataDomainResponse>;
    /**
     * Check if data is processable
     *
     * @param data - Data to check
     * @returns true if processable
     */
    isProcessableData(data: unknown): Promise<boolean>;
    /**
     * Clear cache
     *
     * @throws Error if cache clear fails
     */
    clearCache(): Promise<void>;
    /**
     * Perform HTTP request with timeout enforcement (Law 3.2)
     *
     * @param path - API endpoint path
     * @param options - Fetch options
     * @returns Response
     * @throws Error if request fails or times out
     */
    private fetchWithTimeout;
}
//# sourceMappingURL=DatapizzaClient.d.ts.map