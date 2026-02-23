"use strict";
/**
 * Pinecone Client Implementation
 *
 * Pinecone-specific vector database client with circuit breaker and retry logic.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PineconeClient = void 0;
const logger_1 = require("../../../lib/logger");
const circuit_breaker_1 = require("../../../lib/circuit-breaker");
const retry_1 = require("../../../lib/retry");
const vectordb_canonical_1 = require("../../../schemas/vectordb-canonical");
class PineconeClient {
    constructor(config) {
        this.logger = new logger_1.Logger('vectordb-adapter:pinecone-client');
        this.config = {
            timeout: 5000,
            maxRetries: 3,
            ...config,
        };
        // Pinecone API URL format
        const environment = this.config.environment || 'us-east1-aws';
        this.baseUrl = `https://controller.${environment}.pinecone.io`;
        this.headers = {
            'Content-Type': 'application/json',
            'Api-Key': this.config.apiKey,
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
                const response = await fetch(`${this.baseUrl}/databases`, {
                    method: 'GET',
                    headers: this.headers,
                    signal: AbortSignal.timeout(this.config.timeout),
                });
                if (!response.ok) {
                    throw new Error(`Pinecone health check failed: ${response.statusText}`);
                }
                const latency = Date.now() - startTime;
                const result = {
                    status: 'healthy',
                    backend_type: vectordb_canonical_1.VectorDBType.PINECONE,
                    connected: true,
                    latency_ms: latency,
                    timestamp: new Date().toISOString(),
                };
                this.logger.info('Pinecone health check successful', {
                    latency_ms: latency,
                });
                return result;
            }
            catch (error) {
                this.logger.error('Pinecone health check failed', error);
                return {
                    status: 'unhealthy',
                    backend_type: vectordb_canonical_1.VectorDBType.PINECONE,
                    connected: false,
                    error: error.message,
                    timestamp: new Date().toISOString(),
                };
            }
        });
    }
    /**
     * Create index (collection in Pinecone)
     */
    async createCollection(config) {
        return this.circuitBreaker.execute(async () => {
            const body = {
                name: config.name,
                dimension: config.dimension,
                metric: config.distance_metric === 'cosine' ? 'cosine' :
                    config.distance_metric === 'euclidean' ? 'euclidean' :
                        'dotproduct',
                pods: 1,
                replicas: 1,
                pod_type: 'p1.x1',
            };
            await (0, retry_1.retryWithJitter)(async () => {
                const response = await fetch(`${this.baseUrl}/databases`, {
                    method: 'POST',
                    headers: this.headers,
                    body: JSON.stringify(body),
                    signal: AbortSignal.timeout(this.config.timeout),
                });
                if (!response.ok && response.status !== 409) {
                    throw new Error(`Failed to create index: ${response.statusText}`);
                }
            }, this.config.maxRetries, this.logger);
            this.logger.info('Pinecone index created', {
                index: config.name,
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
            const response = await fetch(`${this.baseUrl}/databases/${collectionName}`, {
                method: 'GET',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Failed to get index info: ${response.statusText}`);
            }
            const data = await response.json();
            return {
                name: data.database.name,
                dimension: data.database.dimension,
                vector_count: data.database.total_vector_count,
                distance_metric: data.database.metric.toLowerCase(),
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
            };
        });
    }
    /**
     * Get index URL for vector operations
     */
    getIndexUrl(collectionName) {
        const environment = this.config.environment || 'us-east1-aws';
        return `https://${collectionName}-${environment}.pinecone.io`;
    }
    /**
     * Upsert vectors
     */
    async upsert(request) {
        return this.circuitBreaker.execute(async () => {
            const vectors = request.entries.map(vectordb_canonical_1.transformCanonicalToPinecone);
            const indexUrl = this.getIndexUrl(request.collection_name);
            const namespace = request.namespace || '';
            await (0, retry_1.retryWithJitter)(async () => {
                const response = await fetch(`${indexUrl}/vectors/upsert?namespace=${namespace}`, {
                    method: 'POST',
                    headers: this.headers,
                    body: JSON.stringify({ vectors }),
                    signal: AbortSignal.timeout(this.config.timeout),
                });
                if (!response.ok) {
                    throw new Error(`Upsert failed: ${response.statusText}`);
                }
            }, this.config.maxRetries, this.logger);
            this.logger.info('Pinecone upsert successful', {
                collection: request.collection_name,
                count: request.entries.length,
                namespace,
            });
            return {
                upserted_count: request.entries.length,
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
            const indexUrl = this.getIndexUrl(collectionName);
            const namespace = query.filter?.namespace || '';
            const requestBody = {
                vector: Array.isArray(query.vector) ? query.vector : null,
                topK: query.k,
                includeMetadata: true,
                includeValues: true,
                namespace,
            };
            if (query.filter) {
                requestBody.filter = query.filter;
            }
            const response = await fetch(`${indexUrl}/query`, {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify(requestBody),
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            const data = await response.json();
            const results = data.matches.map((match) => ({
                entry: (0, vectordb_canonical_1.transformPineconeToCanonical)(match),
                score: match.score,
                distance: 1 - match.score,
            }));
            this.logger.info('Pinecone search successful', {
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
            const indexUrl = this.getIndexUrl(request.collection_name);
            const namespace = request.namespace || '';
            if (request.delete_all) {
                // Delete all vectors in namespace
                await (0, retry_1.retryWithJitter)(async () => {
                    const response = await fetch(`${indexUrl}/vectors/delete?deleteAll=true&namespace=${namespace}`, {
                        method: 'POST',
                        headers: this.headers,
                        signal: AbortSignal.timeout(this.config.timeout),
                    });
                    if (!response.ok) {
                        throw new Error(`Delete all failed: ${response.statusText}`);
                    }
                }, this.config.maxRetries, this.logger);
            }
            else {
                // Delete specific IDs
                await (0, retry_1.retryWithJitter)(async () => {
                    const response = await fetch(`${indexUrl}/vectors/delete`, {
                        method: 'POST',
                        headers: this.headers,
                        body: JSON.stringify({
                            ids: request.ids,
                            namespace,
                        }),
                        signal: AbortSignal.timeout(this.config.timeout),
                    });
                    if (!response.ok) {
                        throw new Error(`Delete failed: ${response.statusText}`);
                    }
                }, this.config.maxRetries, this.logger);
            }
            this.logger.info('Pinecone delete successful', {
                collection: request.collection_name,
                count: request.delete_all ? 'all' : request.ids.length,
                namespace,
            });
            return {
                deleted_count: request.delete_all ? -1 : request.ids.length,
                collection_name: request.collection_name,
                timestamp: new Date().toISOString(),
            };
        });
    }
    /**
     * List collections (indexes)
     */
    async listCollections() {
        return this.circuitBreaker.execute(async () => {
            const response = await fetch(`${this.baseUrl}/databases`, {
                method: 'GET',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout),
            });
            if (!response.ok) {
                throw new Error(`Failed to list indexes: ${response.statusText}`);
            }
            const data = await response.json();
            return data.databases.map((db) => db.name);
        });
    }
}
exports.PineconeClient = PineconeClient;
//# sourceMappingURL=pinecone-client.js.map