export interface RagbitsConfig {
    serverUrl: string;
    apiKey?: string;
}
export interface RagbitsSearchRequest {
    query: string;
    filters?: Record<string, unknown>;
    topK?: number;
}
export interface RagbitsSearchResult {
    content: string;
    metadata?: Record<string, unknown>;
    score?: number;
}
export interface RagbitsSearchResponse {
    success: boolean;
    results: RagbitsSearchResult[];
    error?: string;
}
export interface RagbitsIngestRequest {
    content: string;
    metadata?: Record<string, unknown>;
}
export interface RagbitsIngestResponse {
    success: boolean;
    artifactId?: string;
    error?: string;
}
export declare class RagbitsClient {
    private serverUrl;
    private apiKey?;
    constructor(config: RagbitsConfig);
    search(request: RagbitsSearchRequest): Promise<RagbitsSearchResponse>;
    ingest(request: RagbitsIngestRequest): Promise<RagbitsIngestResponse>;
    private request;
}
