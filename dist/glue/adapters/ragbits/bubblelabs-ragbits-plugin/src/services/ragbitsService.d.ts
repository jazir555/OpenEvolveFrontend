import { RagbitsClient } from '../lib/ragbitsClient';
import type { RAGBitsSearchRequest, RAGBitsSearchResponse, RAGBitsIngestRequest, RAGBitsIngestResponse, RAGBitsIndexStats } from '../types/plugin-types';
export declare class RagbitsService {
    private client;
    private cache;
    private cacheTTL;
    constructor(client: RagbitsClient, cacheTTL?: number);
    /**
     * Search for documents with caching
     */
    search(request: RAGBitsSearchRequest): Promise<RAGBitsSearchResponse>;
    /**
     * Ingest a document
     */
    ingest(request: RAGBitsIngestRequest): Promise<RAGBitsIngestResponse>;
    /**
     * Batch ingest documents
     */
    batchIngest(requests: RAGBitsIngestRequest[]): Promise<RAGBitsIngestResponse[]>;
    /**
     * Get index statistics
     */
    getIndexStats(): Promise<RAGBitsIndexStats>;
    /**
     * Clear cache
     */
    clearCache(): Promise<void>;
    /**
     * Get cache key for request
     */
    private getCacheKey;
    /**
     * Get from cache
     */
    private getFromCache;
    /**
     * Set cache entry
     */
    private setCache;
    /**
     * Invalidate cache entries by type
     */
    private invalidateCache;
    /**
     * Generate unique ID
     */
    private generateId;
}
//# sourceMappingURL=ragbitsService.d.ts.map