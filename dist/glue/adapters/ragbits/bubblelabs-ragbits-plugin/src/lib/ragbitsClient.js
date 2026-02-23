"use strict";
// RAGBits Client
// Handles HTTP communication with the RAGBits server
// Updated with structured logging, validation, and proper error handling
Object.defineProperty(exports, "__esModule", { value: true });
exports.RagbitsClient = void 0;
const structuredLogger_1 = require("./structuredLogger");
class RagbitsClient {
    constructor(config) {
        this.config = config;
        this.correlationId = `ragbits-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    }
    /**
     * Update client configuration
     */
    configure(config) {
        this.config = { ...this.config, ...config };
        structuredLogger_1.ragbitsLogger.info('RAGBits client configuration updated', {
            correlation_id: this.correlationId,
            target_service: 'ragbits',
            has_api_key: !!this.config.apiKey
        });
    }
    /**
     * Test connection to RAGBits server
     */
    async testConnection() {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'ragbits-plugin',
            target_service: 'ragbits-server',
            operation: 'test_connection'
        };
        try {
            structuredLogger_1.ragbitsLogger.info('Testing RAGBits connection', context);
            const response = await this.fetch('/health', {
                method: 'GET',
            });
            if (response.ok) {
                structuredLogger_1.ragbitsLogger.info('RAGBits connection successful', context);
            }
            else {
                structuredLogger_1.ragbitsLogger.warn('RAGBits connection failed', {
                    ...context,
                    status: response.status,
                    status_text: response.statusText
                });
            }
            return response.ok;
        }
        catch (error) {
            structuredLogger_1.ragbitsLogger.error('RAGBits connection test failed', error, context);
            return false;
        }
    }
    /**
     * Search for documents
     */
    async search(request) {
        const context = {
            correlation_id: this.correlationId,
            source_service: 'ragbits-plugin',
            target_service: 'ragbits-server',
            operation: 'search',
            query_length: request.query?.length
        };
        try {
            structuredLogger_1.ragbitsLogger.info('Executing RAGBits search', context);
            const response = await this.fetch('/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });
            if (!response.ok) {
                const errorText = await response.text();
                structuredLogger_1.ragbitsLogger.error('RAGBits search failed', new Error(errorText), {
                    ...context,
                    status: response.status,
                    status_text: response.statusText
                });
                throw new Error(`Search failed: ${response.statusText}`);
            }
            const result = await response.json();
            structuredLogger_1.ragbitsLogger.info('RAGBits search successful', {
                ...context,
                result_count: result.results?.length || 0
            });
            return result;
        }
        catch (error) {
            structuredLogger_1.ragbitsLogger.error('RAGBits search error', error, context);
            throw error;
        }
    }
    /**
     * Ingest a document
     */
    async ingest(request) {
        const response = await this.fetch('/ingest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });
        if (!response.ok) {
            throw new Error(`Ingest failed: ${response.statusText}`);
        }
        return response.json();
    }
    /**
     * Batch ingest documents
     */
    async batchIngest(requests) {
        const response = await this.fetch('/ingest/batch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ documents: requests }),
        });
        if (!response.ok) {
            throw new Error(`Batch ingest failed: ${response.statusText}`);
        }
        return response.json();
    }
    /**
     * Get index statistics
     */
    async getIndexStats() {
        const response = await this.fetch('/index/stats', {
            method: 'GET',
        });
        if (!response.ok) {
            throw new Error(`Failed to get index stats: ${response.statusText}`);
        }
        return response.json();
    }
    /**
     * Clear cache
     */
    async clearCache() {
        const response = await this.fetch('/cache/clear', {
            method: 'POST',
        });
        if (!response.ok) {
            throw new Error(`Failed to clear cache: ${response.statusText}`);
        }
        return response.json();
    }
    /**
     * Perform HTTP request with timeout and error handling
     */
    async fetch(path, options = {}) {
        const url = `${this.config.serverUrl}${path}`;
        const timeout = this.config.timeout || 30000;
        // Create abort controller for timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        try {
            // Add API key header if provided
            const headers = new Headers(options.headers);
            if (this.config.apiKey) {
                headers.set('Authorization', `Bearer ${this.config.apiKey}`);
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
}
exports.RagbitsClient = RagbitsClient;
//# sourceMappingURL=ragbitsClient.js.map