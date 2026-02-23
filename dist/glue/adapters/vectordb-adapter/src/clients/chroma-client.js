"use strict";
/**
 * Chroma Client Implementation
 *
 * Chroma-specific vector database client with circuit breaker and retry logic.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChromaClient = void 0;
const logger_1 = require("../../../lib/logger");
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
const retry_1 = require("../../../lib/retry");
const vectordb_canonical_1 = require("../../../schemas/vectordb-canonical");
class ChromaClient {
    constructor(config) {
        this.logger = new logger_1.Logger('vectordb-adapter:chroma-client');
        this.config = {
            timeout: 5000,
            maxRetries: 3,
            ...config,
        };
        this.baseUrl = this.config.url.replace(/\/$/, '');
        this.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        };
        // Circuit breaker configuration
        this.circuitBreaker = new circuit_breaker_1.CircuitBreaker({
            threshold: 5,
            timeout: 60000,
            logger: this.logger,
        });
    }
    /**
     * Health check
     */
    async healthCheck() {
        const startTime = Date.now();
        return this.circuitBreaker.execute(async () => {
            try {
                const response = await fetch(`${this.baseUrl}/api/v1/heartbeat`, {
                    method: 'GET',
                    headers: this.headers,
                    signal: AbortSignal.timeout(this.config.timeout),
                });
                if (!response.ok) {
                    throw new Error(`Chroma health check failed: ${response.statusText}`);
                }
                const latency = Date.now() - startTime;
                const result = {
                    status: 'healthy',
                    backend_type: vectordb_canonical_1.VectorDBType.CHROMA,
                    connected: true,
                    latency_ms: latency,
                    timestamp: new Date().toISOString(),
                };
                this.logger.info('Chroma health check successful', {
                    latency_ms: latency,
                });
                return result;
            }
            catch (error) {
                this.logger.error('Chroma health check failed', error);
                return {
                    status: 'unhealthy',
                    backend_type: vectordb_canonical_1.VectorDBType.CHROMA,
                    connected: false,
                    error: error.message,
                    timestamp: new Date().toISOString(),
                };
            }
        });
    }
    /**
     * Create collection
     */
    async createCollection(config) {
        return this.circuitBreaker.execute(async () => {
            const body = {
                name: config.name,
                metadata: {
                    dimension: config.dimension,
                    distance_metric: config.distance_metric,
                },
            };
            await (0, retry_1.retryWithJitter)(async () => {
                const response = await fetch(`${this.baseUrl}/api/v1/collections`, {
                    method: 'POST',
                    headers: this.headers,
                    body: JSON.stringify(body),
                    signal: AbortSignal.timeout(this.config.timeout),
                });
                if (!response.ok && response.status !== 409) {
                    throw new Error(`Failed to create collection: ${response.statusText}`);
                }
            }, this.config.maxRetries, this.logger);
            this.logger.info('Chroma collection created', {
                collection: config.name,
                dimension: config.dimension,
                distance_metric: config.distance_metric,
            });
        });
    }
    /**
     * Get collection info
     */
    async getCollectionInfo(collectionName) {
        return this.circuitBreaker.execute(async () => {
            const response = await fetch(`${this.baseUrl}/api/v1/collections/${collectionName}`, {
                method: 'GET',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Failed to get collection info: ${response.statusText}`);
            }
            const data = await response.json();
            return {
                name: data.name,
                dimension: data.metadata?.dimension || 0,
                vector_count: data.count || 0,
                distance_metric: data.metadata?.distance_metric || 'cosine',
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
            };
        });
    }
    /**
     * Upsert vectors
     */
    async upsert(request) {
        return this.circuitBreaker.execute(async () => {
            // Chroma processes entries in batches
            const batchSize = 100;
            let upsertedCount = 0;
            for (let i = 0; i < request.entries.length; i += batchSize) {
                const batch = request.entries.slice(i, i + batchSize);
                const chromaData = batch.map(vectordb_canonical_1.transformCanonicalToChroma);
                await (0, retry_1.retryWithJitter)(async () => {
                    const response = await fetch(`${this.baseUrl}/api/v1/collections/${request.collection_name}/upsert`, {
                        method: 'POST',
                        headers: this.headers,
                        body: JSON.stringify(chromaData),
                        signal: AbortSignal.timeout(this.config.timeout),
                    });
                    if (!response.ok) {
                        throw new Error(`Upsert failed: ${response.statusText}`);
                    }
                }, this.config.maxRetries, this.logger);
                upsertedCount += batch.length;
            }
            this.logger.info('Chroma upsert successful', {
                collection: request.collection_name,
                count: request.entries.length,
            });
            return {
                upserted_count: upsertedCount,
                collection_name: request.collection_name,
                timestamp: new Date().toISOString(),
            };
        });
    }
    /**
     * Search vectors
     */
    async search(collectionName, query) {
        return this.circuitBreaker.execute(async () => {
            if (!Array.isArray(query.vector)) {
                throw new Error('Chroma only supports dense vectors');
            }
            const requestBody = {
                query_embeddings: [query.vector],
                n_results: query.k,
                include: ['metadatas', 'documents', 'distances'],
            };
            if (query.filter) {
                requestBody.where = query.filter;
            }
            const response = await fetch(`${this.baseUrl}/api/v1/collections/${collectionName}/query`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(requestBody),
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            const data = await response.json();
            const results = [];
            // Chroma returns batch results, extract first batch
            if (data.ids && data.ids[0]) {
                for (let i = 0; i < data.ids[0].length; i++) {
                    const embedding = {
                        id: data.ids[0][i],
                        embedding: query.vector, // Chroma doesn't return vectors by default
                        document: data.documents?.[0]?.[i],
                        metadatas: data.metadatas?.[0]?.[i] || {},
                    };
                    results.push({
                        entry: (0, vectordb_canonical_1.transformChromaToCanonical)(embedding),
                        score: 1 - (data.distances?.[0]?.[i] || 0),
                        distance: data.distances?.[0]?.[i] || 0,
                    });
                    // Apply score threshold
                    if (query.score_threshold && results[i].score < query.score_threshold) {
                        results.splice(i);
                        break;
                    }
                }
            }
            this.logger.info('Chroma search successful', {
                collection: collectionName,
                result_count: results.length,
            });
            return results;
        });
    }
    /**
     * Delete vectors
     */
    async delete(request) {
        return this.circuitBreaker.execute(async () => {
            if (request.delete_all) {
                // Delete entire collection
                await (0, retry_1.retryWithJitter)(async () => {
                    const response = await fetch(`${this.baseUrl}/api/v1/collections/${request.collection_name}`, {
                        method: 'DELETE',
                        headers: this.headers,
                        signal: AbortSignal.timeout(this.config.timeout),
                    });
                    if (!response.ok) {
                        throw new Error(`Delete collection failed: ${response.statusText}`);
                    }
                }, this.config.maxRetries, this.logger);
            }
            else {
                // Delete specific IDs
                await (0, retry_1.retryWithJitter)(async () => {
                    const response = await fetch(`${this.baseUrl}/api/v1/collections/${request.collection_name}/delete`, {
                        method: 'POST',
                        headers: this.headers,
                        body: JSON.stringify({
                            ids: request.ids,
                        }),
                        signal: AbortSignal.timeout(this.config.timeout),
                    });
                    if (!response.ok) {
                        throw new Error(`Delete failed: ${response.statusText}`);
                    }
                }, this.config.maxRetries, this.logger);
            }
            this.logger.info('Chroma delete successful', {
                collection: request.collection_name,
                count: request.delete_all ? 'all' : request.ids.length,
            });
            return {
                deleted_count: request.delete_all ? -1 : request.ids.length,
                collection_name: request.collection_name,
                timestamp: new Date().toISOString(),
            };
        });
    }
    /**
     * List collections
     */
    async listCollections() {
        return this.circuitBreaker.execute(async () => {
            const response = await fetch(`${this.baseUrl}/api/v1/collections`, {
                method: 'GET',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Failed to list collections: ${response.statusText}`);
            }
            const data = await response.json();
            return data.map((c) => c.name);
        });
    }
}
exports.ChromaClient = ChromaClient;
//# sourceMappingURL=chroma-client.js.map