"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.RAGClient = void 0;
const crypto_1 = require("crypto");
/**
 * RAGBits HTTP Client
 *
 * Provides typed methods for interacting with the RAGBits REST API.
 */
class RAGClient {
    constructor(config) {
        // CONFIGURATION EXPLICITNESS: Crash if API URL missing
        if (!config.api_url) {
            throw new Error('RAGBITS_API_URL environment variable is required');
        }
        // TIMEOUT: Crash if timeout missing
        if (!config.timeout_ms || config.timeout_ms <= 0) {
            throw new Error('TIMEOUT_MS environment variable is required and must be positive');
        }
        this.config = config;
    }
    /**
     * Test connection to RAGBits server
     */
    async testConnection(correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/health', {
                method: 'GET',
            }, cid);
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits connection test',
                success: response.ok,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return response.ok;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits connection test failed',
                error: error instanceof Error ? error.message : String(error),
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return false;
        }
    }
    /**
     * Search for documents
     */
    async search(request, correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            }, cid);
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText} (${response.status})`);
            }
            const data = await response.json();
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits search completed',
                query_length: request.query.length,
                top_k: request.top_k,
                results_count: data.results?.length || 0,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return data;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits search failed',
                error: error instanceof Error ? error.message : String(error),
                query_length: request.query.length,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            throw error;
        }
    }
    /**
     * Ingest a document
     */
    async ingest(request, correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/ingest', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            }, cid);
            if (!response.ok) {
                throw new Error(`Ingest failed: ${response.statusText} (${response.status})`);
            }
            const data = await response.json();
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits ingest completed',
                content_length: request.content.length,
                success: data.success,
                chunks_created: data.chunks_ingested,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return data;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits ingest failed',
                error: error instanceof Error ? error.message : String(error),
                content_length: request.content.length,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            throw error;
        }
    }
    /**
     * Batch ingest documents
     */
    async batchIngest(requests, correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/ingest/batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ documents: requests }),
            }, cid);
            if (!response.ok) {
                throw new Error(`Batch ingest failed: ${response.statusText} (${response.status})`);
            }
            const data = await response.json();
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits batch ingest completed',
                document_count: requests.length,
                success_count: data.success_count,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return data;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits batch ingest failed',
                error: error instanceof Error ? error.message : String(error),
                document_count: requests.length,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            throw error;
        }
    }
    /**
     * Get index statistics
     */
    async getStats(correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/stats', {
                method: 'GET',
            }, cid);
            if (!response.ok) {
                throw new Error(`Failed to get stats: ${response.statusText} (${response.status})`);
            }
            const data = await response.json();
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits stats retrieved',
                total_documents: data.ingested_documents,
                vector_store_type: data.vector_store_type,
                embedding_model: data.embedding_model,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return data;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits stats retrieval failed',
                error: error instanceof Error ? error.message : String(error),
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            throw error;
        }
    }
    /**
     * Clear cache
     */
    async clearCache(correlationId) {
        const startTime = Date.now();
        const cid = correlationId || (0, crypto_1.randomUUID)();
        try {
            const response = await this.fetch('/clear-cache', {
                method: 'POST',
            }, cid);
            if (!response.ok) {
                throw new Error(`Failed to clear cache: ${response.statusText} (${response.status})`);
            }
            const data = await response.json();
            const duration = Date.now() - startTime;
            // STRUCTURED LOGGING
            console.log(JSON.stringify({
                msg: 'RAGBits cache cleared',
                success: data.success,
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            return data;
        }
        catch (error) {
            const duration = Date.now() - startTime;
            console.log(JSON.stringify({
                msg: 'RAGBits cache clearing failed',
                error: error instanceof Error ? error.message : String(error),
                duration_ms: duration,
                correlation_id: cid,
                source_service: 'ragbits-adapter',
                target_service: 'ragbits-core',
                timestamp: new Date().toISOString(),
            }));
            throw error;
        }
    }
    /**
     * Perform HTTP request with timeout and error handling
     *
     * TIMEOUT: MANDATORY - All requests must have timeout
     */
    async fetch(path, options = {}, correlationId) {
        const url = `${this.config.api_url}${path}`;
        const timeout = this.config.timeout_ms;
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        try {
            // Add API key header if provided
            const headers = {
                ...options.headers,
                'X-Correlation-ID': correlationId,
            };
            if (this.config.api_key) {
                headers['Authorization'] = `Bearer ${this.config.api_key}`;
            }
            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal,
            });
            clearTimeout(timeoutId);
            return response;
        }
        catch (error) {
            clearTimeout(timeoutId);
            if (error instanceof Error) {
                if (error.name === 'AbortError') {
                    throw new Error(`Request timeout after ${timeout}ms`);
                }
                throw error;
            }
            throw new Error('Unknown error occurred');
        }
    }
    /**
     * Update client configuration
     */
    configure(config) {
        if (config.api_url !== undefined) {
            if (!config.api_url) {
                throw new Error('RAGBITS_API_URL cannot be empty');
            }
            this.config.api_url = config.api_url;
        }
        if (config.timeout_ms !== undefined) {
            if (!config.timeout_ms || config.timeout_ms <= 0) {
                throw new Error('TIMEOUT_MS must be positive');
            }
            this.config.timeout_ms = config.timeout_ms;
        }
        if (config.api_key !== undefined) {
            this.config.api_key = config.api_key;
        }
    }
    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
}
exports.RAGClient = RAGClient;
//# sourceMappingURL=rag-client.js.map