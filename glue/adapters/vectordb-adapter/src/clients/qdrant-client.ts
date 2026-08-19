/**
 * Qdrant Client Implementation
 *
 * Qdrant-specific vector database client with circuit breaker and retry logic.
 */

import { Logger } from '../lib/logger';
import { CircuitBreaker } from '../lib/circuit-breaker';
import { retryWithJitter } from '../lib/retry';
import {
  VectorEntry,
  CollectionConfig,
  SearchQuery,
  SearchResult,
  UpsertRequest,
  UpsertResponse,
  DeleteRequest,
  DeleteResponse,
  CollectionInfo,
  HealthCheckResponse,
  VectorDBType,
  transformQdrantToCanonical,
  transformCanonicalToQdrant,
} from '../schemas/vectordb-canonical';

export interface QdrantClientConfig {
  url: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
}

export class QdrantClient {
  private logger: Logger;
  private circuitBreaker: CircuitBreaker;
  private config: QdrantClientConfig;
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(config: QdrantClientConfig) {
    this.logger = new Logger('vectordb-adapter:qdrant-client');
    this.config = {
      timeout: 5000,
      maxRetries: 3,
      ...config,
    };

    this.baseUrl = this.config.url.replace(/\/$/, '');
    this.headers = {
      'Content-Type': 'application/json',
      ...(this.config.apiKey && { 'api-key': this.config.apiKey }),
    };

    // Circuit breaker configuration
    this.circuitBreaker = new CircuitBreaker({
      threshold: 5, // Open after 5 failures
      timeout: 60000, // Reset after 60 seconds
      logger: this.logger,
    });
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const startTime = Date.now();

    return this.circuitBreaker.execute(async () => {
      try {
        const response = await fetch(`${this.baseUrl}/`, {
          method: 'GET',
          headers: this.headers,
          signal: AbortSignal.timeout(this.config.timeout!),
        });

        if (!response.ok) {
          throw new Error(`Qdrant health check failed: ${response.statusText}`);
        }

        const data = await response.json() as any;
        const latency = Date.now() - startTime;

        const result: HealthCheckResponse = {
          status: 'healthy',
          backend_type: VectorDBType.QDRANT,
          connected: true,
          latency_ms: latency,
          timestamp: new Date().toISOString(),
        };

        this.logger.info('Qdrant health check successful', {
          latency_ms: latency,
          version: data.version,
        });

        return result;
      } catch (error) {
        this.logger.error('Qdrant health check failed', error as Error);
        return {
          status: 'unhealthy',
          backend_type: VectorDBType.QDRANT,
          connected: false,
          error: (error as Error).message,
          timestamp: new Date().toISOString(),
        };
      }
    });
  }

  /**
   * Create collection
   */
  async createCollection(config: CollectionConfig): Promise<void> {
    return this.circuitBreaker.execute(async () => {
      const body = {
        vectors: {
          size: config.dimension,
          distance: config.distance_metric.toUpperCase(), // Cosine, Euclidean, Dot
        },
      };

      await retryWithJitter(
        async () => {
          const response = await fetch(`${this.baseUrl}/collections/${config.name}`, {
            method: 'PUT',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(this.config.timeout!),
          });

          if (!response.ok && response.status !== 409) {
            // 409 = already exists, which is fine
            throw new Error(`Failed to create collection: ${response.statusText}`);
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('Qdrant collection created', {
        collection: config.name,
        dimension: config.dimension,
        distance_metric: config.distance_metric,
      });
    });
  }

  /**
   * Get collection info
   */
  async getCollectionInfo(collectionName: string): Promise<CollectionInfo> {
    return this.circuitBreaker.execute(async () => {
      const response = await fetch(
        `${this.baseUrl}/collections/${collectionName}`,
        {
          method: 'GET',
          headers: this.headers,
          signal: AbortSignal.timeout(this.config.timeout!),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to get collection info: ${response.statusText}`);
      }

      const data = await response.json() as any;
      const { result } = data;

      return {
        name: result.config.params.vectors.size,
        dimension: result.config.params.vectors.size,
        vector_count: result.points_count,
        distance_metric: result.config.params.vectors.distance.toLowerCase(),
        created_at: result.config.created_at || new Date().toISOString(),
        updated_at: result.config.updated_at || new Date().toISOString(),
      };
    });
  }

  /**
   * Upsert vectors
   */
  async upsert(request: UpsertRequest): Promise<UpsertResponse> {
    return this.circuitBreaker.execute(async () => {
      const points = request.entries.map(transformCanonicalToQdrant);

      await retryWithJitter(
        async () => {
          const response = await fetch(
            `${this.baseUrl}/collections/${request.collection_name}/points`,
            {
              method: 'PUT',
              headers: this.headers,
              body: JSON.stringify({ points }),
              signal: AbortSignal.timeout(this.config.timeout!),
            }
          );

          if (!response.ok) {
            throw new Error(`Upsert failed: ${response.statusText}`);
          }
        },
        this.config.maxRetries!,
        this.logger
      );

      this.logger.info('Qdrant upsert successful', {
        collection: request.collection_name,
        count: request.entries.length,
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
  async search(collectionName: string, query: SearchQuery): Promise<SearchResult[]> {
    return this.circuitBreaker.execute(async () => {
      const requestBody: any = {
        limit: query.k,
        vector: Array.isArray(query.vector) ? query.vector : undefined,
        with_payload: true,
        with_vector: true,
      };

      if (query.score_threshold !== undefined) {
        requestBody.score_threshold = query.score_threshold;
      }

      if (query.filter) {
        requestBody.filter = query.filter;
      }

      const response = await fetch(
        `${this.baseUrl}/collections/${collectionName}/points/query`,
        {
          method: 'POST',
          headers: this.headers,
          body: JSON.stringify(requestBody),
          signal: AbortSignal.timeout(this.config.timeout!),
        }
      );

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }

      const data = await response.json() as any;
      const results = data.result.map((point: any) => ({
        entry: transformQdrantToCanonical(point),
        score: point.score || 0,
        distance: 1 - point.score, // Convert score to distance
      }));

      this.logger.info('Qdrant search successful', {
        collection: collectionName,
        result_count: results.length,
      });

      return results;
    });
  }

  /**
   * Delete vectors
   */
  async delete(request: DeleteRequest): Promise<DeleteResponse> {
    return this.circuitBreaker.execute(async () => {
      if (request.delete_all) {
        // Delete all vectors in collection
        await retryWithJitter(
          async () => {
            const response = await fetch(
              `${this.baseUrl}/collections/${request.collection_name}`,
              {
                method: 'DELETE',
                headers: this.headers,
                signal: AbortSignal.timeout(this.config.timeout!),
              }
            );

            if (!response.ok) {
              throw new Error(`Delete collection failed: ${response.statusText}`);
            }
          },
          this.config.maxRetries!,
          this.logger
        );
      } else {
        // Delete specific IDs
        await retryWithJitter(
          async () => {
            const response = await fetch(
              `${this.baseUrl}/collections/${request.collection_name}/points`,
              {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({
                  points: request.ids,
                }),
                signal: AbortSignal.timeout(this.config.timeout!),
              }
            );

            if (!response.ok) {
              throw new Error(`Delete failed: ${response.statusText}`);
            }
          },
          this.config.maxRetries!,
          this.logger
        );
      }

      this.logger.info('Qdrant delete successful', {
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
  async listCollections(): Promise<string[]> {
    return this.circuitBreaker.execute(async () => {
      const response = await fetch(`${this.baseUrl}/collections`, {
        method: 'GET',
        headers: this.headers,
        signal: AbortSignal.timeout(this.config.timeout!),
      });

      if (!response.ok) {
        throw new Error(`Failed to list collections: ${response.statusText}`);
      }

      const data = await response.json() as any;
      return data.result.collections.map((c: any) => c.name);
    });
  }
}
