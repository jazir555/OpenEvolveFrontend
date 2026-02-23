"use strict";
// RAGBits Service
// High-level service wrapper for RAGBits client
Object.defineProperty(exports, "__esModule", { value: true });
exports.RagbitsService = void 0;
class RagbitsService {
    constructor(client, cacheTTL = 3600) {
        this.client = client;
        this.cache = new Map();
        this.cacheTTL = cacheTTL;
    }
    /**
     * Search for documents with caching
     */
    async search(request) {
        const startTime = Date.now();
        // Check cache
        const cacheKey = this.getCacheKey('search', request);
        const cached = this.getFromCache(cacheKey);
        if (cached) {
            return {
                ...cached,
                executionTime: Date.now() - startTime,
                metadata: {
                    ...cached.metadata,
                    cacheHit: true
                }
            };
        }
        // Perform search
        const response = await this.client.search({
            query: request.query,
            topK: request.topK || 10,
            scoreThreshold: request.scoreThreshold || 0.7,
            filter: request.filter,
            enableHybridSearch: request.enableHybridSearch,
            enableReranking: request.enableReranking
        });
        const result = {
            success: true,
            query: request.query,
            results: response.results || [],
            totalResults: response.results?.length || 0,
            executionTime: Date.now() - startTime,
            metadata: {
                searchType: response.searchType || 'semantic',
                vectorStoreUsed: response.vectorStoreUsed || 'unknown',
                rerankingApplied: response.enableReranking || false,
                cacheHit: false
            },
            errors: response.errors || [],
            warnings: response.warnings || [],
            timestamp: new Date()
        };
        // Cache results
        this.setCache(cacheKey, result);
        return result;
    }
    /**
     * Ingest a document
     */
    async ingest(request) {
        const startTime = Date.now();
        const response = await this.client.ingest({
            content: request.content,
            metadata: request.metadata
        });
        // Invalidate relevant cache entries
        this.invalidateCache('search');
        return {
            success: response.success || true,
            documentId: response.documentId || this.generateId(),
            message: response.message || 'Document ingested successfully',
            executionTime: Date.now() - startTime,
            errors: response.errors || [],
            warnings: response.warnings || [],
            timestamp: new Date()
        };
    }
    /**
     * Batch ingest documents
     */
    async batchIngest(requests) {
        const startTime = Date.now();
        const response = await this.client.batchIngest(requests.map(req => ({
            content: req.content,
            metadata: req.metadata
        })));
        // Invalidate relevant cache entries
        this.invalidateCache('search');
        return response.map((res, index) => ({
            success: res.success || true,
            documentId: res.documentId || this.generateId(),
            message: res.message || `Document ${index + 1} ingested successfully`,
            executionTime: (Date.now() - startTime) / requests.length,
            errors: res.errors || [],
            warnings: res.warnings || [],
            timestamp: new Date()
        }));
    }
    /**
     * Get index statistics
     */
    async getIndexStats() {
        const response = await this.client.getIndexStats();
        return {
            totalDocuments: response.totalDocuments || 0,
            documentsByType: response.documentsByType || {},
            documentsByStage: response.documentsByStage || {},
            documentsByTeam: response.documentsByTeam || {},
            indexSize: response.indexSize || 0,
            lastUpdated: response.lastUpdated ? new Date(response.lastUpdated) : new Date()
        };
    }
    /**
     * Clear cache
     */
    async clearCache() {
        this.cache.clear();
        await this.client.clearCache();
    }
    /**
     * Get cache key for request
     */
    getCacheKey(type, request) {
        return `${type}:${JSON.stringify(request)}`;
    }
    /**
     * Get from cache
     */
    getFromCache(key) {
        const entry = this.cache.get(key);
        if (!entry) {
            return null;
        }
        const age = (Date.now() - entry.timestamp) / 1000;
        if (age > this.cacheTTL) {
            this.cache.delete(key);
            return null;
        }
        return entry.data;
    }
    /**
     * Set cache entry
     */
    setCache(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }
    /**
     * Invalidate cache entries by type
     */
    invalidateCache(type) {
        const prefix = `${type}:`;
        for (const key of this.cache.keys()) {
            if (key.startsWith(prefix)) {
                this.cache.delete(key);
            }
        }
    }
    /**
     * Generate unique ID
     */
    generateId() {
        return `${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    }
}
exports.RagbitsService = RagbitsService;
//# sourceMappingURL=ragbitsService.js.map