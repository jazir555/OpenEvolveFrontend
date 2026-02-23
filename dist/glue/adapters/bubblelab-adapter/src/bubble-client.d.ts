/**
 * BubbleLab API Client
 *
 * Purpose: Direct API client for BubbleLab endpoints
 * Compliance: Law of Runtime Truth - wraps actual API calls
 *
 * Features:
 * - Timeout enforcement (Law of Configuration Explicitness)
 * - Retry logic for transient failures
 * - Structured error responses
 * - Idempotent operations where possible
 */
export interface BubbleLabClientConfig {
    api_url: string;
    timeout_ms: number;
    auth_token?: string;
    max_retries?: number;
}
export interface BubbleLabResponse<T = any> {
    success: boolean;
    data?: T;
    error?: string;
    status?: number;
}
export interface BubbleFlowListResponse {
    flows: Array<{
        id: string | number;
        name: string;
        description?: string;
        eventType: string;
        webhookActive: boolean;
        createdAt?: string;
        updatedAt?: string;
    }>;
}
export interface BubbleFlowCreateRequest {
    name: string;
    description?: string;
    code: string;
    eventType: string;
    webhookActive?: boolean;
}
export interface BubbleFlowCreateResponse {
    id: string | number;
    name: string;
    requiredCredentials?: Record<string, string[]>;
    webhookUrl?: string;
    createdAt?: string;
}
export interface BubbleFlowExecuteRequest {
    payload?: any;
    credentials?: Record<string, number>;
}
export interface BubbleFlowExecuteResponse {
    execution_id?: string;
    output?: any;
    error?: string;
    status?: string;
}
export declare class BubbleLabClient {
    private readonly config;
    constructor(config: BubbleLabClientConfig);
    /**
     * Check BubbleLab API health
     */
    healthCheck(): Promise<BubbleLabResponse<{
        status: string;
        version?: string;
    }>>;
    /**
     * List all BubbleFlows
     * Idempotent: GET operation, safe to retry
     */
    listBubbleFlows(): Promise<BubbleLabResponse<BubbleFlowListResponse>>;
    /**
     * Get a specific BubbleFlow by ID
     * Idempotent: GET operation, safe to retry
     */
    getBubbleFlow(flowId: string): Promise<BubbleLabResponse<any>>;
    /**
     * Create a new BubbleFlow
     * NOT idempotent: will create multiple flows on retry
     * Caller should implement deduplication logic
     */
    createBubbleFlow(request: BubbleFlowCreateRequest): Promise<BubbleLabResponse<BubbleFlowCreateResponse>>;
    /**
     * Update a BubbleFlow
     * NOT idempotent: multiple updates with same data are OK but not guaranteed
     */
    updateBubbleFlow(flowId: string, updates: Partial<BubbleFlowCreateRequest>): Promise<BubbleLabResponse<any>>;
    /**
     * Delete a BubbleFlow
     * Idempotent with check: verify flow doesn't exist after deletion
     */
    deleteBubbleFlow(flowId: string): Promise<BubbleLabResponse<void>>;
    /**
     * Execute a BubbleFlow
     * NOT idempotent: each execution creates a new run
     */
    executeBubbleFlow(flowId: string, request: BubbleFlowExecuteRequest): Promise<BubbleLabResponse<BubbleFlowExecuteResponse>>;
    /**
     * Get execution history for a BubbleFlow
     * Idempotent: GET operation, safe to retry
     */
    getExecutionHistory(flowId: string, limit?: number, offset?: number): Promise<BubbleLabResponse<any>>;
    /**
     * Validate BubbleFlow code without creating
     * Idempotent: Validation operation, no side effects
     */
    validateCode(code: string): Promise<BubbleLabResponse<any>>;
    /**
     * Make HTTP request to BubbleLab API with retry logic
     */
    private makeRequest;
    /**
     * Handle errors and return structured error response
     */
    private handleError;
}
/**
 * Create a BubbleLab client from environment variables
 */
export declare function createBubbleLabClient(): BubbleLabClient;
//# sourceMappingURL=bubble-client.d.ts.map